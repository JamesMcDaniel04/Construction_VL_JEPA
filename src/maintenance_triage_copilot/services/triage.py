"""Maintenance triage orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

import numpy as np
import torch

from maintenance_triage_copilot.auth.supabase import SupabaseAuthProvider
from maintenance_triage_copilot.config import AppConfig
from maintenance_triage_copilot.domain.models import (
    AdminDashboardMetrics,
    AssetCreateRequest,
    AssetPatchRequest,
    AssetRecord,
    AssetType,
    Citation,
    CorpusDocument,
    CorpusSourceType,
    FeedbackLabel,
    IncidentRecord,
    IssueCandidate,
    MediaAssetRecord,
    MediaAssetView,
    MediaType,
    NextStep,
    PilotUser,
    PilotUserInviteRequest,
    PilotUserInviteResponse,
    PilotUserSeed,
    PilotUserView,
    ReferenceState,
    SimilarIncident,
    SiteCreateRequest,
    SitePatchRequest,
    SiteRecord,
    StateAssessment,
    TriageAuditDetail,
    TriageAuditRecord,
    TriageCase,
    TriageCaseCreateRequest,
    TriageCaseFeedback,
    TriageCaseStatus,
    TriageRequest,
    TriageResponse,
    UserRole,
    VisualEvidenceStatus,
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

_STEP_PREFIX_RE = re.compile(
    r"^\s*(?:[-*•]+|\d+[.)]|step\s+\d+[:.)]?|[a-z][.)])\s*",
    re.IGNORECASE,
)
_STEP_CONDITIONAL_RE = re.compile(
    r"^(?:if|when|once|after|before)\b[^,]{0,120},\s*(.+)$",
    re.IGNORECASE,
)
_STEP_CLAUSE_SPLIT_RE = re.compile(
    r"\s*(?:\n+|(?<=[.!?;])\s+|,\s+(?=(?:then|after|before|if|verify|inspect|check|measure|tighten|reset|replace|confirm|record|test|ensure|isolate|review|escalate)\b))",
    re.IGNORECASE,
)
_ACTIONABLE_PREFIXES = (
    "inspect",
    "check",
    "verify",
    "measure",
    "reset",
    "replace",
    "tighten",
    "confirm",
    "record",
    "test",
    "ensure",
    "isolate",
    "review",
    "observe",
    "monitor",
    "align",
    "reseat",
    "re-seat",
    "restore",
    "de-energize",
    "energize",
    "re-energize",
    "lock out",
    "tag out",
    "photograph",
    "scan",
    "compare",
    "open",
    "close",
    "trace",
    "note",
    "escalate",
)
_REFERENCE_STATE_ISSUE_SUPPORT = 0.75
_INCIDENT_LINKED_STEP_SUPPORT = 0.08


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
    supabase_auth: SupabaseAuthProvider | None = None
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

    def list_documents(self, *, limit: int, offset: int) -> list[CorpusDocument]:
        return self.state.metadata_store.list_documents(limit, offset)

    def add_incident(self, incident: IncidentRecord) -> dict[str, Any]:
        self.state.metadata_store.add_incident(incident)
        payload = incident.model_dump()
        payload["text"] = incident.searchable_text()
        embedding = self.state.text_encoder.encode_one(payload["text"])
        self.state.vector_index.upsert("incidents", incident.incident_id, embedding, payload)
        return {"status": "indexed", "incident_id": incident.incident_id}

    def list_incidents(self, *, limit: int, offset: int) -> list[IncidentRecord]:
        return self.state.metadata_store.list_incidents(limit, offset)

    def add_reference_state(self, reference_state: ReferenceState) -> dict[str, Any]:
        self.state.metadata_store.add_reference_state(reference_state)
        embedding = self._encode_visual(reference_state)
        payload = reference_state.model_dump()
        self.state.vector_index.upsert("states", reference_state.state_id, embedding, payload)
        return {"status": "indexed", "state_id": reference_state.state_id}

    def list_reference_states(self, *, limit: int, offset: int) -> list[ReferenceState]:
        return self.state.metadata_store.list_reference_states(limit, offset)

    def seed_pilot_user(self, seed: PilotUserSeed) -> PilotUser:
        existing = self.state.metadata_store.get_pilot_user(seed.user_id)
        if existing is not None:
            return existing
        existing_by_email = self.state.metadata_store.get_pilot_user_by_email(seed.email)
        if existing_by_email is not None:
            return existing_by_email
        pilot_user = PilotUser(
            user_id=seed.user_id,
            organization_id=seed.organization_id,
            role=seed.role,
            display_name=seed.display_name,
            email=seed.email.strip().lower(),
        )
        self.state.metadata_store.add_pilot_user(pilot_user)
        return pilot_user

    def invite_pilot_user(
        self,
        request: PilotUserInviteRequest,
        *,
        invited_by_user_id: str,
    ) -> PilotUserInviteResponse:
        auth = self.state.supabase_auth
        if auth is None or not auth.configured_for_invites():
            raise RuntimeError("Supabase invite flow is not configured")
        invite_result = auth.invite_user(
            email=request.email.strip().lower(),
            display_name=request.display_name,
            redirect_to=self.state.config.supabase.web_redirect_url,
        )
        pilot_user = PilotUser(
            user_id=invite_result.user_id,
            organization_id=request.organization_id,
            role=request.role,
            display_name=request.display_name,
            email=invite_result.email,
            invited_by_user_id=invited_by_user_id,
        )
        self.state.metadata_store.add_pilot_user(pilot_user)
        return PilotUserInviteResponse(
            user=self.pilot_user_view(pilot_user),
            invite_status=invite_result.invite_status,
        )

    def list_pilot_users(self, *, limit: int, offset: int) -> list[PilotUser]:
        return self.state.metadata_store.list_pilot_users(limit, offset)

    def add_site(
        self,
        request: SiteCreateRequest,
        *,
        organization_id: str,
    ) -> SiteRecord:
        site = SiteRecord(
            site_id=request.site_id,
            organization_id=organization_id,
            name=request.name,
            code=request.code,
            active=request.active,
            metadata=request.metadata,
        )
        self.state.metadata_store.add_site(site)
        return site

    def update_site(
        self,
        site_id: str,
        patch: SitePatchRequest,
        *,
        organization_id: str | None,
        role: UserRole,
    ) -> SiteRecord | None:
        site = self.state.metadata_store.get_site(site_id)
        if site is None:
            return None
        if role != UserRole.service and site.organization_id != organization_id:
            raise PermissionError("Not allowed to update this site")
        updated = site.model_copy(
            update={
                "name": patch.name if patch.name is not None else site.name,
                "code": patch.code if patch.code is not None else site.code,
                "active": patch.active if patch.active is not None else site.active,
                "metadata": patch.metadata if patch.metadata is not None else site.metadata,
            }
        )
        self.state.metadata_store.add_site(updated)
        return updated

    def list_sites(
        self,
        *,
        limit: int,
        offset: int,
        organization_id: str | None,
        role: UserRole,
        query: str | None = None,
        active_only: bool = False,
    ) -> list[SiteRecord]:
        scoped_organization_id = None if role == UserRole.service else organization_id
        return self.state.metadata_store.list_sites(
            limit,
            offset,
            organization_id=scoped_organization_id,
            query=query,
            active_only=active_only,
        )

    def add_asset_catalog_record(
        self,
        request: AssetCreateRequest,
        *,
        organization_id: str,
    ) -> AssetRecord:
        site = self.state.metadata_store.get_site(request.site_id)
        if site is None:
            raise LookupError("Site not found")
        if site.organization_id != organization_id:
            raise PermissionError("Site does not belong to this organization")
        asset = AssetRecord(
            asset_id=request.asset_id,
            organization_id=organization_id,
            site_id=request.site_id,
            display_name=request.display_name,
            panel_family=request.panel_family,
            equipment_family=request.equipment_family,
            panel_id=request.panel_id,
            active=request.active,
            metadata=request.metadata,
        )
        self.state.metadata_store.add_asset_catalog_record(asset)
        return asset

    def update_asset_catalog_record(
        self,
        asset_id: str,
        patch: AssetPatchRequest,
        *,
        organization_id: str | None,
        role: UserRole,
    ) -> AssetRecord | None:
        asset = self.state.metadata_store.get_asset_catalog_record(asset_id)
        if asset is None:
            return None
        if role != UserRole.service and asset.organization_id != organization_id:
            raise PermissionError("Not allowed to update this asset")
        next_site_id = patch.site_id if patch.site_id is not None else asset.site_id
        site = self.state.metadata_store.get_site(next_site_id)
        if site is None:
            raise LookupError("Site not found")
        if site.organization_id != asset.organization_id:
            raise PermissionError("Asset site must remain within the same organization")
        updated = asset.model_copy(
            update={
                "site_id": next_site_id,
                "display_name": (
                    patch.display_name if patch.display_name is not None else asset.display_name
                ),
                "panel_family": (
                    patch.panel_family if patch.panel_family is not None else asset.panel_family
                ),
                "equipment_family": (
                    patch.equipment_family
                    if patch.equipment_family is not None
                    else asset.equipment_family
                ),
                "panel_id": patch.panel_id if patch.panel_id is not None else asset.panel_id,
                "active": patch.active if patch.active is not None else asset.active,
                "metadata": patch.metadata if patch.metadata is not None else asset.metadata,
            }
        )
        self.state.metadata_store.add_asset_catalog_record(updated)
        return updated

    def list_asset_catalog_records(
        self,
        *,
        limit: int,
        offset: int,
        organization_id: str | None,
        role: UserRole,
        site_id: str | None = None,
        query: str | None = None,
        active_only: bool = False,
    ) -> list[AssetRecord]:
        scoped_organization_id = None if role == UserRole.service else organization_id
        return self.state.metadata_store.list_asset_catalog_records(
            limit,
            offset,
            organization_id=scoped_organization_id,
            site_id=site_id,
            query=query,
            active_only=active_only,
        )

    @staticmethod
    def pilot_user_view(pilot_user: PilotUser) -> PilotUserView:
        return PilotUserView(
            user_id=pilot_user.user_id,
            organization_id=pilot_user.organization_id,
            role=pilot_user.role,
            display_name=pilot_user.display_name,
            email=pilot_user.email,
            invited_by_user_id=pilot_user.invited_by_user_id,
            created_at=pilot_user.created_at,
            active=pilot_user.active,
        )

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
        next_steps = self._build_next_steps(document_hits, incident_hits=incident_hits)
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
            visual_evidence_status=self._build_visual_evidence_status(
                issue_candidates,
                state_assessment,
                similar_incidents,
            ),
            uncertainty_summary=self._build_uncertainty_summary(
                issue_candidates,
                state_assessment,
                similar_incidents,
            ),
            safety_notices=self._build_safety_notices(),
        )
        self.state.metadata_store.record_triage(observation.observation_id, response)
        return response

    def create_case(
        self,
        request: TriageCaseCreateRequest,
        *,
        organization_id: str,
        user_id: str,
        display_name: str,
        role: UserRole,
    ) -> TriageCase:
        site = self.state.metadata_store.get_site(request.site_id)
        if site is None:
            raise LookupError("Site not found")
        asset = self.state.metadata_store.get_asset_catalog_record(request.asset_id)
        if asset is None:
            raise LookupError("Asset not found")
        if asset.site_id != site.site_id:
            raise ValueError("Asset does not belong to the selected site")
        if asset.organization_id != site.organization_id:
            raise ValueError("Asset and site organization mismatch")
        if role != UserRole.service and asset.organization_id != organization_id:
            raise PermissionError("Asset does not belong to the current organization")
        triage_case = TriageCase(
            case_id=f"case-{uuid4().hex[:16]}",
            organization_id=asset.organization_id,
            created_by_user_id=user_id,
            created_by_display_name=display_name,
            role=role,
            site_id=request.site_id,
            asset_id=request.asset_id,
            panel_family=asset.panel_family,
            equipment_family=asset.equipment_family,
            panel_id=asset.panel_id,
            question=request.question,
            operator_context=request.operator_context,
            expected_state_label=request.expected_state_label,
            status=TriageCaseStatus.draft,
            metadata=request.metadata,
        )
        self.state.metadata_store.save_case(triage_case)
        return triage_case

    def get_case(self, case_id: str) -> TriageCase | None:
        return self.state.metadata_store.get_case(case_id)

    def list_cases(
        self,
        *,
        limit: int,
        offset: int,
        organization_id: str | None,
        user_id: str | None,
        role: UserRole,
    ) -> list[TriageCase]:
        cases = self.state.metadata_store.list_cases(max(limit + offset, 200), 0)
        filtered = [
            item
            for item in cases
            if organization_id is None or item.organization_id == organization_id
        ]
        if role == UserRole.technician and user_id is not None:
            filtered = [item for item in filtered if item.created_by_user_id == user_id]
        return filtered[offset : offset + limit]

    @staticmethod
    def case_summary(triage_case: TriageCase) -> dict[str, Any]:
        helpful = None
        if triage_case.feedback is not None:
            helpful = FeedbackLabel.helpful in triage_case.feedback.labels
        top_issue_class = None
        escalation = None
        if triage_case.analysis is not None:
            top_issue_class = (
                triage_case.analysis.issue_candidates[0].issue_class
                if triage_case.analysis.issue_candidates
                else None
            )
            escalation = triage_case.analysis.escalation_recommendation
        return {
            "case_id": triage_case.case_id,
            "site_id": triage_case.site_id,
            "asset_id": triage_case.asset_id,
            "panel_family": triage_case.panel_family,
            "panel_id": triage_case.panel_id,
            "status": triage_case.status,
            "created_at": triage_case.created_at,
            "updated_at": triage_case.updated_at,
            "top_issue_class": top_issue_class,
            "escalation_recommendation": escalation,
            "helpful": helpful,
        }

    def analyze_case(
        self,
        *,
        triage_case: TriageCase,
        filename: str,
        content_type: str,
        data: bytes,
        request_id: str,
        principal: str,
        trace_id: str | None,
    ) -> tuple[TriageCase, TriageAuditRecord]:
        pending_case = triage_case.model_copy(
            update={
                "status": TriageCaseStatus.pending_analysis,
                "updated_at": datetime.now(tz=UTC),
            }
        )
        self.state.metadata_store.save_case(pending_case)
        media_type = self._media_type_from_upload(filename=filename, content_type=content_type)
        started = datetime.now(tz=UTC)
        stored_asset = self.persist_uploaded_asset(
            asset_type=AssetType.triage_upload,
            filename=filename,
            content_type=content_type,
            data=data,
            metadata={
                "case_id": triage_case.case_id,
                "equipment_family": triage_case.equipment_family,
                "panel_id": triage_case.panel_id or "",
            },
        )
        if media_type == MediaType.image:
            embedding = self.state.image_backbone.encode_raw_image(data)
        else:
            embedding = self.state.video_backbone.encode_raw_video(data)
        observation = VisualObservation(
            observation_id=f"obs-{uuid4().hex[:16]}",
            equipment_family=triage_case.equipment_family,
            panel_id=triage_case.panel_id,
            media_type=media_type,
            embedding_values=embedding.tolist(),
            metadata={
                "site_id": triage_case.site_id,
                "asset_id": triage_case.asset_id,
                "panel_family": triage_case.panel_family,
            },
        )
        triage_request = TriageRequest(
            observation=observation,
            question=triage_case.question,
            operator_context=triage_case.operator_context,
            expected_state_label=triage_case.expected_state_label,
        )
        triage_response = self.analyze(triage_request)
        audit = self.record_triage_audit(
            request_id=request_id,
            principal=principal,
            trace_id=trace_id,
            request_payload=triage_request,
            response_payload=triage_response,
            linked_asset_ids=[stored_asset.asset_id],
            metadata={"route": "/cases/analyze", "case_id": triage_case.case_id},
        )
        finished = datetime.now(tz=UTC)
        updated_case = pending_case.model_copy(
            update={
                "status": (
                    TriageCaseStatus.escalated
                    if triage_response.escalation_recommendation != "proceed_with_guided_inspection"
                    else TriageCaseStatus.analyzed
                ),
                "media_asset_id": stored_asset.asset_id,
                "analysis": triage_response,
                "latest_audit_id": audit.audit_id,
                "response_time_ms": (finished - started).total_seconds() * 1000.0,
                "updated_at": finished,
            }
        )
        self.state.metadata_store.save_case(updated_case)
        return updated_case, audit

    def submit_case_feedback(
        self,
        case_id: str,
        *,
        labels: list[FeedbackLabel],
        comment: str | None,
    ) -> TriageCase | None:
        existing = self.state.metadata_store.get_case(case_id)
        if existing is None:
            return None
        updated = existing.model_copy(
            update={
                "feedback": TriageCaseFeedback(labels=labels, comment=comment),
                "updated_at": datetime.now(tz=UTC),
            }
        )
        self.state.metadata_store.save_case(updated)
        return updated

    def admin_dashboard(self, *, organization_id: str | None = None) -> AdminDashboardMetrics:
        cases = self.state.metadata_store.list_cases(5000, 0)
        if organization_id is not None:
            cases = [item for item in cases if item.organization_id == organization_id]
        analyzed = [item for item in cases if item.analysis is not None]
        escalated = [
            item for item in analyzed if item.status == TriageCaseStatus.escalated
        ]
        helpful_cases = [
            item
            for item in cases
            if item.feedback is not None and FeedbackLabel.helpful in item.feedback.labels
        ]
        unresolved = [
            item
            for item in cases
            if item.status in {TriageCaseStatus.draft, TriageCaseStatus.pending_analysis}
        ]
        issue_counts: dict[str, int] = {}
        for item in analyzed:
            if not item.analysis or not item.analysis.issue_candidates:
                continue
            issue = item.analysis.issue_candidates[0].issue_class
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        top_issue_classes: list[dict[str, str | int]] = [
            {"issue_class": key, "count": value}
            for key, value in sorted(
                issue_counts.items(),
                key=lambda pair: pair[1],
                reverse=True,
            )[:5]
        ]
        helpful_rate = None
        feedback_cases = [item for item in cases if item.feedback is not None]
        if feedback_cases:
            helpful_rate = len(helpful_cases) / len(feedback_cases)
        return AdminDashboardMetrics(
            total_cases=len(cases),
            analyzed_cases=len(analyzed),
            escalated_cases=len(escalated),
            helpful_feedback_rate=helpful_rate,
            unresolved_cases=len(unresolved),
            top_issue_classes=top_issue_classes,
        )

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
            if media.has_precomputed_embedding():
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

        matched_state = self.state.metadata_store.get_reference_state(
            state_assessment.matched_state_id or ""
        )
        if matched_state is not None:
            issue_class = str(matched_state.metadata.get("issue_class", "")).strip()
            if issue_class:
                features = ensure(issue_class)
                support = state_assessment.confidence * _REFERENCE_STATE_ISSUE_SUPPORT
                features["document_sum"] += support
                features["document_max"] = max(features["document_max"], support)
                features["document_count"] += 1.0
                rationales.setdefault(
                    issue_class,
                    (
                        f"Matched curated reference state '{matched_state.state_label}' "
                        "associated with this issue class."
                    ),
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

    def _build_next_steps(
        self,
        document_hits: list[SearchHit],
        *,
        incident_hits: list[SearchHit] | None = None,
    ) -> list[NextStep]:
        steps: list[NextStep] = []
        seen: set[str] = set()
        document_lookup = {
            document.document_id: document
            for document in self.list_documents(limit=10_000, offset=0)
        }
        for document_id, title, source_type, source_text, chunk_id, base_score in (
            self._next_step_document_candidates(
                document_lookup,
                document_hits,
                incident_hits or [],
            )
        ):
            extracted_steps = self._extract_next_step_candidates(
                source_text,
                source_type=source_type,
            )
            if not extracted_steps:
                fallback = self._fallback_next_step(title, source_type)
                if fallback:
                    extracted_steps = [fallback]

            for step_index, step_text in enumerate(extracted_steps):
                normalized = self._normalize_step_key(step_text)
                if not normalized or normalized in seen:
                    continue
                seen.add(normalized)
                confidence = max(
                    0.05,
                    min(
                        1.0,
                        base_score
                        + self._source_type_step_bonus(source_type)
                        - (step_index * 0.04),
                    ),
                )
                steps.append(
                    NextStep(
                        step=step_text,
                        confidence=round(confidence, 6),
                        citations=[
                            Citation(
                                document_id=document_id,
                                title=title,
                                source_type=source_type,
                                chunk_id=chunk_id,
                                snippet=self._citation_snippet(step_text, source_text),
                            )
                        ],
                    )
                )
                if len(steps) >= self.state.config.triage.top_k_steps:
                    return steps
        return steps

    def _next_step_document_candidates(
        self,
        document_lookup: dict[str, CorpusDocument],
        document_hits: list[SearchHit],
        incident_hits: list[SearchHit],
    ) -> list[tuple[str, str, CorpusSourceType, str, str | None, float]]:
        candidates: list[tuple[str, str, CorpusSourceType, str, str | None, float]] = []
        seen: set[str] = set()

        for hit in incident_hits:
            for linked_document_id in hit.payload.get("linked_document_ids", []):
                document_id = str(linked_document_id)
                document = document_lookup.get(document_id)
                if document is None or document_id in seen:
                    continue
                seen.add(document_id)
                candidates.append(
                    (
                        document_id,
                        document.title,
                        document.source_type,
                        document.body,
                        None,
                        min(1.0, hit.score + _INCIDENT_LINKED_STEP_SUPPORT),
                    )
                )

        for hit in document_hits:
            document_id = str(hit.payload["document_id"])
            if document_id in seen:
                continue
            seen.add(document_id)
            document = document_lookup.get(document_id)
            source_type = (
                document.source_type
                if document is not None
                else CorpusSourceType(str(hit.payload["source_type"]))
            )
            source_text = document.body if document is not None else str(hit.payload["text"])
            title = document.title if document is not None else str(hit.payload["title"])
            candidates.append(
                (
                    document_id,
                    title,
                    source_type,
                    source_text,
                    str(hit.payload["chunk_id"]),
                    hit.score,
                )
            )

        return candidates

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
            and risk >= self.state.config.triage.escalation_threshold
        ):
            return "escalate_for_visual_review"
        return action

    def _build_visual_evidence_status(
        self,
        issue_candidates: list[IssueCandidate],
        state_assessment: StateAssessment,
        similar_incidents: list[SimilarIncident],
    ) -> VisualEvidenceStatus:
        top_issue_confidence = issue_candidates[0].confidence if issue_candidates else 0.0
        if state_assessment.confidence < 0.25 and top_issue_confidence < 0.45:
            return VisualEvidenceStatus.insufficient
        if (
            state_assessment.confidence < 0.55
            or top_issue_confidence < 0.65
            or not similar_incidents
        ):
            return VisualEvidenceStatus.limited
        return VisualEvidenceStatus.sufficient

    def _build_uncertainty_summary(
        self,
        issue_candidates: list[IssueCandidate],
        state_assessment: StateAssessment,
        similar_incidents: list[SimilarIncident],
    ) -> str:
        evidence_status = self._build_visual_evidence_status(
            issue_candidates,
            state_assessment,
            similar_incidents,
        )
        if evidence_status == VisualEvidenceStatus.sufficient:
            return (
                "Visual evidence supports a grounded shortlist, but technicians should still "
                "verify with measurement and standard safety procedure."
            )
        if evidence_status == VisualEvidenceStatus.insufficient:
            return (
                "Visual evidence is insufficient for a reliable panel-only triage result. "
                "Use measurement, manual inspection, or escalation before acting."
            )
        return (
            "Visual evidence is directionally useful but incomplete. Hidden electrical faults "
            "may not be visible in a photo or short clip."
        )

    @staticmethod
    def _build_safety_notices() -> list[str]:
        return [
            (
                "This tool suggests likely issue candidates and next inspection steps, "
                "not a definitive diagnosis."
            ),
            (
                "Follow lockout/tagout, measurement procedure, and site safety requirements "
                "before acting."
            ),
            (
                "Escalate when the visual evidence is limited or the observed condition "
                "could involve invisible electrical faults."
            ),
        ]

    def _extract_next_step_candidates(
        self,
        text: str,
        *,
        source_type: CorpusSourceType,
    ) -> list[str]:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not normalized:
            return []

        candidates: list[str] = []
        for fragment in _STEP_CLAUSE_SPLIT_RE.split(normalized):
            cleaned = self._clean_step_fragment(fragment)
            if not cleaned:
                continue
            if self._looks_actionable_step(cleaned, source_type=source_type):
                candidates.append(cleaned)

        if candidates:
            return candidates

        if source_type == CorpusSourceType.sop:
            fallback_sentences = [
                self._clean_step_fragment(sentence)
                for sentence in re.split(r"(?<=[.!?;])\s+", normalized)
            ]
            return [sentence for sentence in fallback_sentences if sentence]
        return []

    def _clean_step_fragment(self, text: str) -> str:
        cleaned = _STEP_PREFIX_RE.sub("", text).strip()
        cleaned = re.sub(r"\s+", " ", cleaned)
        conditional_match = _STEP_CONDITIONAL_RE.match(cleaned)
        if conditional_match:
            candidate = conditional_match.group(1).strip()
            if self._looks_actionable_step(candidate, source_type=CorpusSourceType.manual):
                cleaned = candidate
        cleaned = cleaned.rstrip(" ;,.")
        if not cleaned:
            return ""
        if len(cleaned) > 180:
            cut = cleaned[:180]
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            cleaned = cut.rstrip(" ;,.")
        return cleaned[:1].upper() + cleaned[1:]

    def _looks_actionable_step(
        self,
        text: str,
        *,
        source_type: CorpusSourceType,
    ) -> bool:
        lowered = text.strip().lower()
        if not lowered:
            return False
        if (
            source_type in {CorpusSourceType.ticket, CorpusSourceType.repair_note}
            and len(lowered) < 18
        ):
            return False
        for prefix in _ACTIONABLE_PREFIXES:
            if lowered.startswith(prefix + " ") or lowered == prefix:
                return True
        return lowered.startswith("do not ")

    @staticmethod
    def _source_type_step_bonus(source_type: CorpusSourceType) -> float:
        bonuses = {
            CorpusSourceType.sop: 0.08,
            CorpusSourceType.manual: 0.04,
            CorpusSourceType.repair_note: 0.02,
            CorpusSourceType.ticket: 0.0,
        }
        return bonuses.get(source_type, 0.0)

    @staticmethod
    def _normalize_step_key(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    @staticmethod
    def _citation_snippet(step_text: str, source_text: str) -> str:
        normalized_source = source_text.replace("\n", " ").strip()
        if len(normalized_source) <= 120:
            return normalized_source
        lowered_step = step_text.lower()
        lowered_source = normalized_source.lower()
        start = lowered_source.find(lowered_step[:30].lower())
        if start < 0:
            return normalized_source[:120].rstrip()
        end = min(len(normalized_source), start + 120)
        return normalized_source[start:end].rstrip()

    @staticmethod
    def _fallback_next_step(title: str, source_type: CorpusSourceType) -> str | None:
        if source_type == CorpusSourceType.ticket:
            return None
        return f"Review {title} for the next inspection step related to the current panel state"

    @staticmethod
    def _media_type_from_upload(*, filename: str, content_type: str) -> MediaType:
        lower_name = filename.lower()
        if content_type.startswith("image/") or lower_name.endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            return MediaType.image
        if content_type.startswith("video/") or lower_name.endswith(
            (".mp4", ".mov", ".avi", ".mkv")
        ):
            return MediaType.video
        raise ValueError(f"Unsupported upload type: {content_type or filename}")

    @staticmethod
    def _normalize_embedding(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(array)
        if norm < 1e-8:
            return cast(np.ndarray, array)
        return cast(np.ndarray, array / norm)
