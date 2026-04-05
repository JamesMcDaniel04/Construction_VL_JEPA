"""Maintenance triage orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import numpy as np
import torch

from maintenance_triage_copilot.config import AppConfig
from maintenance_triage_copilot.domain.models import (
    AssetType,
    Citation,
    CorpusDocument,
    IncidentRecord,
    IssueCandidate,
    MediaAssetRecord,
    MediaAssetView,
    MediaType,
    NextStep,
    ReferenceState,
    SimilarIncident,
    StateAssessment,
    TriageAuditDetail,
    TriageAuditRecord,
    TriageRequest,
    TriageResponse,
    VisualObservation,
)
from maintenance_triage_copilot.encoding.text import MaintenanceTextEncoder
from maintenance_triage_copilot.models.adapter import VisualTextProjector
from maintenance_triage_copilot.models.backbones import IJEPAImageAdapter, VJEPAVideoAdapter
from maintenance_triage_copilot.models.policy import CalibratedTriagePolicy, IssueEvidence
from maintenance_triage_copilot.retrieval.chunking import chunk_document
from maintenance_triage_copilot.retrieval.index import SearchHit, VectorIndex
from maintenance_triage_copilot.storage.object_store import ObjectStore
from maintenance_triage_copilot.storage.protocol import MetadataStore
from maintenance_triage_copilot.telemetry import trace_operation


@dataclass
class AppState:
    config: AppConfig
    text_encoder: MaintenanceTextEncoder
    image_backbone: IJEPAImageAdapter
    video_backbone: VJEPAVideoAdapter
    projector: VisualTextProjector
    triage_policy: CalibratedTriagePolicy
    vector_index: VectorIndex
    metadata_store: MetadataStore
    object_store: ObjectStore
    asset_status: dict[str, dict[str, str | bool]]
    auth_mode: str = "none"
    telemetry_mode: str = "disabled"
    projector_checkpoint_path: str | None = None
    projector_checkpoint_loaded: bool = False
    policy_checkpoint_path: str | None = None
    policy_checkpoint_loaded: bool = False


class TriageService:
    def __init__(self, state: AppState):
        self.state = state

    def add_document(self, document: CorpusDocument) -> dict[str, Any]:
        self.state.metadata_store.add_document(document)
        chunks = chunk_document(
            document,
            chunk_size=self.state.config.retrieval.chunk_size,
            chunk_overlap=self.state.config.retrieval.chunk_overlap,
        )
        for chunk in chunks:
            embedding = self.state.text_encoder.encode_one(chunk["text"])
            self.state.vector_index.upsert(
                "documents",
                chunk["chunk_id"],
                embedding,
                chunk,
            )
        return {"status": "indexed", "document_id": document.document_id, "chunks": len(chunks)}

    def add_incident(self, incident: IncidentRecord) -> dict[str, Any]:
        self.state.metadata_store.add_incident(incident)
        payload = incident.model_dump()
        payload["text"] = incident.searchable_text()
        embedding = self.state.text_encoder.encode_one(payload["text"])
        self.state.vector_index.upsert("incidents", incident.incident_id, embedding, payload)
        return {"status": "indexed", "incident_id": incident.incident_id}

    def add_reference_state(self, reference_state: ReferenceState) -> dict[str, Any]:
        self.state.metadata_store.add_reference_state(reference_state)
        embedding = self._encode_visual(reference_state)
        payload = reference_state.model_dump()
        self.state.vector_index.upsert("states", reference_state.state_id, embedding, payload)
        return {"status": "indexed", "state_id": reference_state.state_id}

    def analyze(self, request: TriageRequest) -> TriageResponse:
        observation = request.observation
        state_hits = self._search_states(observation)
        state_assessment = self._build_state_assessment(state_hits, request.expected_state_label)
        retrieval_query = self._build_retrieval_query(request, state_assessment)
        document_hits = self.state.vector_index.search(
            "documents",
            retrieval_query,
            limit=self.state.config.retrieval.top_k_documents,
            equipment_family=observation.equipment_family,
        )
        incident_hits = self.state.vector_index.search(
            "incidents",
            retrieval_query,
            limit=self.state.config.retrieval.top_k_incidents,
            equipment_family=observation.equipment_family,
        )

        issue_evidence = self._build_issue_evidence(
            request,
            state_assessment,
            incident_hits,
            document_hits,
        )
        issue_candidates = self._build_issue_candidates(issue_evidence)
        next_steps = self._build_next_steps(document_hits)
        similar_incidents = self._build_similar_incidents(incident_hits)
        citations = self._collect_citations(document_hits, next_steps)
        escalation = self._build_escalation(
            request,
            issue_candidates,
            state_assessment,
            incident_hits,
            document_hits,
        )

        response = TriageResponse(
            issue_candidates=issue_candidates,
            state_assessment=state_assessment,
            next_steps=next_steps,
            similar_incidents=similar_incidents,
            escalation_recommendation=escalation,
            evidence_citations=citations,
        )
        self.state.metadata_store.record_triage(observation.observation_id, response)
        return response

    def persist_uploaded_asset(
        self,
        *,
        asset_type: AssetType,
        filename: str,
        content_type: str,
        data: bytes,
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> MediaAssetRecord:
        with trace_operation("object_store.put_bytes"):
            stored = self.state.object_store.put_bytes(
                asset_type=asset_type.value,
                filename=filename,
                data=data,
                content_type=content_type,
            )
        asset = MediaAssetRecord(
            asset_id=f"{asset_type.value}-{stored.sha256[:24]}",
            asset_type=asset_type,
            filename=filename,
            content_type=content_type,
            byte_size=stored.byte_size,
            sha256=stored.sha256,
            object_uri=stored.object_uri,
            created_at=stored.created_at,
            metadata=metadata or {},
        )
        self.state.metadata_store.add_media_asset(asset)
        return asset

    def record_triage_audit(
        self,
        *,
        request_id: str,
        principal: str,
        trace_id: str | None,
        request_payload: TriageRequest,
        response_payload: TriageResponse,
        linked_asset_ids: list[str],
        metadata: dict[str, str | int | float | bool] | None = None,
    ) -> TriageAuditRecord:
        audit = TriageAuditRecord(
            audit_id=f"audit-{uuid4().hex}",
            request_id=request_id,
            observation_id=request_payload.observation.observation_id,
            principal=principal,
            trace_id=trace_id,
            request_payload=request_payload.model_dump(mode="json"),
            response_payload=response_payload.model_dump(mode="json"),
            linked_asset_ids=linked_asset_ids,
            outcome_status="completed",
            created_at=datetime.now(tz=UTC),
            metadata=metadata or {},
        )
        self.state.metadata_store.record_triage_audit(audit)
        return audit

    def list_triage_audits(self, *, limit: int, offset: int) -> list[TriageAuditRecord]:
        return self.state.metadata_store.list_triage_audits(limit, offset)

    def get_triage_audit_detail(self, audit_id: str) -> TriageAuditDetail | None:
        audit = self.state.metadata_store.get_triage_audit(audit_id)
        if audit is None:
            return None
        assets: list[MediaAssetView] = []
        for asset_id in audit.linked_asset_ids:
            asset = self.state.metadata_store.get_media_asset(asset_id)
            if asset is None:
                continue
            assets.append(
                MediaAssetView(
                    **asset.model_dump(),
                    presigned_url=self.state.object_store.presigned_url(asset.object_uri),
                )
            )
        return TriageAuditDetail(audit=audit, linked_assets=assets)

    def health(self) -> dict[str, Any]:
        projector_status = (
            "checkpoint_loaded"
            if self.state.projector_checkpoint_loaded
            else "random_init"
        )
        policy_status = (
            "checkpoint_loaded"
            if self.state.policy_checkpoint_loaded
            else "bootstrap"
        )
        return {
            "status": "healthy",
            "components": {
                "api": "healthy",
                "metadata_store": self.state.metadata_store.status(),
                "vector_index": self.state.vector_index.status(),
                "object_store": self.state.object_store.status(),
                "image_backbone": self.state.asset_status.get(
                    "image_backbone", {"status": "ready"}
                ),
                "video_backbone": self.state.asset_status.get(
                    "video_backbone", {"status": "ready"}
                ),
                "text_encoder": self.state.asset_status.get("text_encoder", {"status": "ready"}),
                "manifest": self.state.asset_status.get("manifest", {"status": "missing"}),
                "auth": {"mode": self.state.auth_mode},
                "telemetry": {"mode": self.state.telemetry_mode},
                "projector": {
                    "status": projector_status,
                    "checkpoint_path": self.state.projector_checkpoint_path,
                    "checkpoint_loaded": self.state.projector_checkpoint_loaded,
                },
                "triage_policy": {
                    "status": policy_status,
                    "checkpoint_path": self.state.policy_checkpoint_path,
                    "checkpoint_loaded": self.state.policy_checkpoint_loaded,
                },
            },
        }

    def _encode_visual(self, media: ReferenceState | VisualObservation) -> np.ndarray:
        with trace_operation("visual.encode"):
            if isinstance(media, VisualObservation) and media.has_precomputed_embedding():
                return self._normalize_embedding(media.load_embedding())
            if media.media_type == MediaType.image:
                return self._normalize_embedding(
                    self.state.image_backbone.encode_observation(media)
                )
            return self._normalize_embedding(self.state.video_backbone.encode_observation(media))

    def _search_states(self, observation: VisualObservation) -> list[SearchHit]:
        embedding = self._encode_visual(observation)
        return self.state.vector_index.search(
            "states",
            embedding,
            limit=self.state.config.retrieval.top_k_states,
            equipment_family=observation.equipment_family,
        )

    def _build_state_assessment(
        self,
        hits: list[SearchHit],
        expected_state_label: str | None,
    ) -> StateAssessment:
        if not hits:
            return StateAssessment(
                confidence=0.0,
                matches_expected=None,
                summary="No reference states are indexed for this equipment family.",
            )

        best = hits[0]
        matched_state_label = best.payload["state_label"]
        summary = (
            f"Closest reference state is '{matched_state_label}' with similarity {best.score:.2f}. "
            f"{best.payload['description']}"
        )
        matches_expected = None
        if expected_state_label is not None:
            matches_expected = (
                matched_state_label == expected_state_label
                and best.score >= self.state.config.triage.state_match_threshold
            )
        return StateAssessment(
            matched_state_id=best.item_id,
            matched_state_label=matched_state_label,
            confidence=best.score,
            matches_expected=matches_expected,
            summary=summary,
        )

    def _build_retrieval_query(
        self,
        request: TriageRequest,
        state_assessment: StateAssessment,
    ) -> np.ndarray:
        pieces = []
        if request.question:
            pieces.append(request.question)
        if request.operator_context:
            pieces.append(request.operator_context)
        context = request.observation.context_text()
        if context:
            pieces.append(context)
        if state_assessment.matched_state_label:
            pieces.append(state_assessment.matched_state_label)
            matched = self.state.metadata_store.get_reference_state(
                state_assessment.matched_state_id or ""
            )
            if matched:
                pieces.append(matched.caption)
                pieces.append(matched.description)
        if not pieces:
            pieces.append(request.observation.equipment_family)
        text_embedding = self.state.text_encoder.encode_one("\n".join(pieces))

        visual_embedding = self._encode_visual(request.observation)
        projector_input = torch.from_numpy(visual_embedding).unsqueeze(0)
        with trace_operation("projector.inference"):
            with torch.no_grad():
                projected = self.state.projector(projector_input).squeeze(0).cpu().numpy()
        combined = text_embedding + (0.25 * projected.astype(np.float32))
        return self._normalize_embedding(combined)

    def _build_issue_evidence(
        self,
        request: TriageRequest,
        state_assessment: StateAssessment,
        incident_hits: list[SearchHit],
        document_hits: list[SearchHit],
    ) -> list[IssueEvidence]:
        feature_map: dict[str, dict[str, float]] = {}
        rationales: dict[str, str] = {}

        def ensure(issue_class: str) -> dict[str, float]:
            return feature_map.setdefault(
                issue_class,
                {
                    "incident_sum": 0.0,
                    "incident_max": 0.0,
                    "incident_count": 0.0,
                    "document_sum": 0.0,
                    "document_max": 0.0,
                    "document_count": 0.0,
                    "state_confidence": state_assessment.confidence,
                    "expected_true": 1.0 if state_assessment.matches_expected is True else 0.0,
                    "expected_false": 1.0 if state_assessment.matches_expected is False else 0.0,
                    "question_present": 1.0 if request.question else 0.0,
                    "operator_context_present": 1.0 if request.operator_context else 0.0,
                },
            )

        for hit in incident_hits:
            issue_class = str(hit.payload["issue_class"])
            features = ensure(issue_class)
            features["incident_sum"] += hit.score
            features["incident_max"] = max(features["incident_max"], hit.score)
            features["incident_count"] += 1.0
            rationales.setdefault(
                issue_class,
                f"Matched incident '{hit.payload['title']}' with similarity {hit.score:.2f}.",
            )

        for hit in document_hits:
            for tag in hit.payload.get("tags", []):
                issue_class = str(tag)
                features = ensure(issue_class)
                features["document_sum"] += hit.score
                features["document_max"] = max(features["document_max"], hit.score)
                features["document_count"] += 1.0
                rationales.setdefault(
                    issue_class,
                    f"Retrieved supporting document chunk '{hit.payload['title']}'.",
                )

        return [
            IssueEvidence(
                issue_class=issue_class,
                features=features,
                rationale=rationales.get(
                    issue_class,
                    "Calibrated policy used weak retrieval evidence for this issue class.",
                ),
            )
            for issue_class, features in feature_map.items()
        ]

    def _build_issue_candidates(self, evidence: list[IssueEvidence]) -> list[IssueCandidate]:
        ranked = self.state.triage_policy.rank_issues(
            evidence,
            top_k=self.state.config.policy.top_k_issues,
        )
        return [
            IssueCandidate(
                issue_class=issue_class,
                confidence=confidence,
                rationale=rationale,
            )
            for issue_class, confidence, rationale in ranked
        ]

    def _build_next_steps(self, document_hits: list[SearchHit]) -> list[NextStep]:
        steps: list[NextStep] = []
        for hit in document_hits:
            text = str(hit.payload["text"]).strip().replace("\n", " ")
            steps.append(
                NextStep(
                    step=text[:180],
                    confidence=hit.score,
                    citations=[
                        Citation(
                            document_id=hit.payload["document_id"],
                            title=hit.payload["title"],
                            source_type=hit.payload["source_type"],
                            chunk_id=hit.payload["chunk_id"],
                            snippet=text[:120],
                        )
                    ],
                )
            )
            if len(steps) >= self.state.config.triage.top_k_steps:
                break
        return steps

    def _build_similar_incidents(self, incident_hits: list[SearchHit]) -> list[SimilarIncident]:
        results: list[SimilarIncident] = []
        for hit in incident_hits:
            results.append(
                SimilarIncident(
                    incident_id=hit.item_id,
                    title=str(hit.payload["title"]),
                    issue_class=str(hit.payload["issue_class"]),
                    fix_summary=str(hit.payload["fix_summary"]),
                    similarity=hit.score,
                )
            )
        return results

    def _collect_citations(
        self,
        document_hits: list[SearchHit],
        next_steps: list[NextStep],
    ) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[tuple[str, str | None]] = set()
        for step in next_steps:
            for citation in step.citations:
                key = (citation.document_id, citation.chunk_id)
                if key not in seen:
                    citations.append(citation)
                    seen.add(key)
        for hit in document_hits:
            key = (str(hit.payload["document_id"]), str(hit.payload["chunk_id"]))
            if key not in seen:
                citations.append(
                    Citation(
                        document_id=str(hit.payload["document_id"]),
                        title=str(hit.payload["title"]),
                        source_type=hit.payload["source_type"],
                        chunk_id=str(hit.payload["chunk_id"]),
                        snippet=str(hit.payload["text"])[:120],
                    )
                )
                seen.add(key)
        return citations

    def _build_escalation(
        self,
        request: TriageRequest,
        issue_candidates: list[IssueCandidate],
        state_assessment: StateAssessment,
        incident_hits: list[SearchHit],
        document_hits: list[SearchHit],
    ) -> str:
        with trace_operation("policy.escalation"):
            action, risk = self.state.triage_policy.choose_escalation(
                {
                    "top_issue_confidence": (
                        issue_candidates[0].confidence if issue_candidates else 0.0
                    ),
                    "state_confidence": state_assessment.confidence,
                    "state_mismatch": 1.0 if state_assessment.matches_expected is False else 0.0,
                    "top_incident_score": incident_hits[0].score if incident_hits else 0.0,
                    "top_document_score": document_hits[0].score if document_hits else 0.0,
                    "no_issue_match": 1.0 if not issue_candidates else 0.0,
                    "expected_known": 1.0 if state_assessment.matches_expected is not None else 0.0,
                    "question_present": 1.0 if request.question else 0.0,
                    "operator_context_present": 1.0 if request.operator_context else 0.0,
                }
            )
        if (
            action == "proceed_with_guided_inspection"
            and risk < self.state.config.triage.escalation_threshold
        ):
            return "escalate_for_visual_review"
        return action

    @staticmethod
    def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(array)
        if norm < 1e-8:
            return cast(np.ndarray, array)
        return cast(np.ndarray, array / norm)
