"""In-memory metadata store used by the API and tests."""

from __future__ import annotations

from maintenance_triage_copilot.domain.models import (
    CorpusDocument,
    IncidentRecord,
    MediaAssetRecord,
    ReferenceState,
    TriageAuditRecord,
    TriageResponse,
)


class MemoryMetadataStore:
    def __init__(self) -> None:
        self.documents: dict[str, CorpusDocument] = {}
        self.incidents: dict[str, IncidentRecord] = {}
        self.reference_states: dict[str, ReferenceState] = {}
        self.media_assets: dict[str, MediaAssetRecord] = {}
        self.triage_audits: dict[str, TriageAuditRecord] = {}
        self.triage_history: list[dict[str, object]] = []

    def add_document(self, document: CorpusDocument) -> None:
        self.documents[document.document_id] = document

    def add_incident(self, incident: IncidentRecord) -> None:
        self.incidents[incident.incident_id] = incident

    def add_reference_state(self, reference_state: ReferenceState) -> None:
        self.reference_states[reference_state.state_id] = reference_state

    def get_reference_state(self, state_id: str) -> ReferenceState | None:
        return self.reference_states.get(state_id)

    def record_triage(self, request_id: str, response: TriageResponse) -> None:
        self.triage_history.append({"request_id": request_id, "response": response.model_dump()})

    def add_media_asset(self, asset: MediaAssetRecord) -> None:
        self.media_assets[asset.asset_id] = asset

    def get_media_asset(self, asset_id: str) -> MediaAssetRecord | None:
        return self.media_assets.get(asset_id)

    def record_triage_audit(self, audit: TriageAuditRecord) -> None:
        self.triage_audits[audit.audit_id] = audit

    def get_triage_audit(self, audit_id: str) -> TriageAuditRecord | None:
        return self.triage_audits.get(audit_id)

    def list_triage_audits(self, limit: int, offset: int) -> list[TriageAuditRecord]:
        audits = sorted(self.triage_audits.values(), key=lambda audit: audit.created_at, reverse=True)
        return audits[offset : offset + limit]

    def status(self) -> dict[str, str]:
        return {
            "metadata_store": "memory",
            "documents": str(len(self.documents)),
            "incidents": str(len(self.incidents)),
            "reference_states": str(len(self.reference_states)),
            "media_assets": str(len(self.media_assets)),
            "triage_audits": str(len(self.triage_audits)),
        }
