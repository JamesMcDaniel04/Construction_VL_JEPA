"""Maintenance triage orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch

from maintenance_triage_copilot.config import AppConfig
from maintenance_triage_copilot.domain.models import (
    Citation,
    CorpusDocument,
    IncidentRecord,
    IssueCandidate,
    MediaType,
    NextStep,
    ReferenceState,
    SimilarIncident,
    StateAssessment,
    TriageRequest,
    TriageResponse,
    VisualObservation,
)
from maintenance_triage_copilot.encoding.text import MaintenanceTextEncoder
from maintenance_triage_copilot.models.adapter import VisualTextProjector
from maintenance_triage_copilot.models.backbones import IJEPAImageAdapter, VJEPAVideoAdapter
from maintenance_triage_copilot.retrieval.chunking import chunk_document
from maintenance_triage_copilot.retrieval.index import SearchHit, VectorIndex
from maintenance_triage_copilot.storage.protocol import MetadataStore


@dataclass
class AppState:
    config: AppConfig
    text_encoder: MaintenanceTextEncoder
    image_backbone: IJEPAImageAdapter
    video_backbone: VJEPAVideoAdapter
    projector: VisualTextProjector
    vector_index: VectorIndex
    metadata_store: MetadataStore


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

        issue_candidates = self._build_issue_candidates(incident_hits, document_hits)
        next_steps = self._build_next_steps(document_hits)
        similar_incidents = self._build_similar_incidents(incident_hits)
        citations = self._collect_citations(document_hits, next_steps)
        escalation = self._build_escalation(issue_candidates, state_assessment)

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

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "components": {
                "api": "healthy",
                "metadata_store": self.state.metadata_store.status(),
                "vector_index": self.state.vector_index.status(),
                "image_backbone": "ready",
                "video_backbone": "ready",
                "text_encoder": "ready",
            },
        }

    def _encode_visual(self, media: ReferenceState | VisualObservation) -> np.ndarray:
        if media.media_type == MediaType.image:
            return self.state.image_backbone.encode_observation(media)
        return self.state.video_backbone.encode_observation(media)

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
        with torch.no_grad():
            projected = self.state.projector(projector_input).squeeze(0).cpu().numpy()
        combined = text_embedding + (0.25 * projected.astype(np.float32))
        norm = np.linalg.norm(combined)
        if norm < 1e-8:
            return cast(np.ndarray, combined)
        return cast(np.ndarray, combined / norm)

    def _build_issue_candidates(
        self,
        incident_hits: list[SearchHit],
        document_hits: list[SearchHit],
    ) -> list[IssueCandidate]:
        aggregates: dict[str, float] = {}
        rationales: dict[str, str] = {}
        for hit in incident_hits:
            issue_class = str(hit.payload["issue_class"])
            aggregates[issue_class] = aggregates.get(issue_class, 0.0) + hit.score
            rationales.setdefault(
                issue_class,
                f"Matched incident '{hit.payload['title']}' with similarity {hit.score:.2f}.",
            )

        if not aggregates:
            for hit in document_hits:
                for tag in hit.payload.get("tags", [])[:1]:
                    aggregates[tag] = aggregates.get(tag, 0.0) + hit.score * 0.5
                    rationales.setdefault(
                        tag,
                        f"Retrieved supporting document chunk '{hit.payload['title']}'.",
                    )

        if not aggregates:
            return [
                IssueCandidate(
                    issue_class="needs_manual_review",
                    confidence=0.35,
                    rationale="No indexed incidents or tagged documents matched strongly enough.",
                )
            ]

        total = max(sum(aggregates.values()), 1e-6)
        ranked = sorted(aggregates.items(), key=lambda item: item[1], reverse=True)
        return [
            IssueCandidate(
                issue_class=issue_class,
                confidence=min(1.0, score / total),
                rationale=rationales[issue_class],
            )
            for issue_class, score in ranked[:3]
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
        issue_candidates: list[IssueCandidate],
        state_assessment: StateAssessment,
    ) -> str:
        top_confidence = issue_candidates[0].confidence if issue_candidates else 0.0
        if state_assessment.matches_expected is False:
            return "escalate_to_senior_technician"
        if top_confidence < self.state.config.triage.escalation_threshold:
            return "escalate_for_visual_review"
        return "proceed_with_guided_inspection"
