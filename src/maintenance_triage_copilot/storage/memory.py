"""In-memory metadata store used by the API and tests."""

from __future__ import annotations

from maintenance_triage_copilot.domain.models import (
    CorpusDocument,
    IncidentRecord,
    MediaAssetRecord,
    PilotUser,
    ReferenceState,
    TriageAuditRecord,
    TriageCase,
    TriageResponse,
)


class MemoryMetadataStore:
    def __init__(self) -> None:
        self.documents: dict[str, CorpusDocument] = {}
        self.incidents: dict[str, IncidentRecord] = {}
        self.reference_states: dict[str, ReferenceState] = {}
        self.triage_cases: dict[str, TriageCase] = {}
        self.pilot_users: dict[str, PilotUser] = {}
        self.media_assets: dict[str, MediaAssetRecord] = {}
        self.triage_audits: dict[str, TriageAuditRecord] = {}
        self.triage_history: list[dict[str, object]] = []

    def add_document(self, document: CorpusDocument) -> None:
        self.documents[document.document_id] = document

    def list_documents(self, limit: int, offset: int) -> list[CorpusDocument]:
        items = sorted(self.documents.values(), key=lambda item: item.document_id)
        return items[offset : offset + limit]

    def add_incident(self, incident: IncidentRecord) -> None:
        self.incidents[incident.incident_id] = incident

    def list_incidents(self, limit: int, offset: int) -> list[IncidentRecord]:
        items = sorted(self.incidents.values(), key=lambda item: item.incident_id)
        return items[offset : offset + limit]

    def add_reference_state(self, reference_state: ReferenceState) -> None:
        self.reference_states[reference_state.state_id] = reference_state

    def list_reference_states(self, limit: int, offset: int) -> list[ReferenceState]:
        items = sorted(self.reference_states.values(), key=lambda item: item.state_id)
        return items[offset : offset + limit]

    def get_reference_state(self, state_id: str) -> ReferenceState | None:
        return self.reference_states.get(state_id)

    def record_triage(self, request_id: str, response: TriageResponse) -> None:
        self.triage_history.append({"request_id": request_id, "response": response.model_dump()})

    def save_case(self, triage_case: TriageCase) -> None:
        self.triage_cases[triage_case.case_id] = triage_case

    def get_case(self, case_id: str) -> TriageCase | None:
        return self.triage_cases.get(case_id)

    def list_cases(self, limit: int, offset: int) -> list[TriageCase]:
        items = sorted(
            self.triage_cases.values(),
            key=lambda item: item.updated_at,
            reverse=True,
        )
        return items[offset : offset + limit]

    def add_pilot_user(self, pilot_user: PilotUser) -> None:
        self.pilot_users[pilot_user.user_id] = pilot_user

    def get_pilot_user(self, user_id: str) -> PilotUser | None:
        return self.pilot_users.get(user_id)

    def list_pilot_users(self, limit: int, offset: int) -> list[PilotUser]:
        items = sorted(
            self.pilot_users.values(),
            key=lambda item: item.created_at,
            reverse=True,
        )
        return items[offset : offset + limit]

    def add_media_asset(self, asset: MediaAssetRecord) -> None:
        self.media_assets[asset.asset_id] = asset

    def get_media_asset(self, asset_id: str) -> MediaAssetRecord | None:
        return self.media_assets.get(asset_id)

    def record_triage_audit(self, audit: TriageAuditRecord) -> None:
        self.triage_audits[audit.audit_id] = audit

    def get_triage_audit(self, audit_id: str) -> TriageAuditRecord | None:
        return self.triage_audits.get(audit_id)

    def list_triage_audits(self, limit: int, offset: int) -> list[TriageAuditRecord]:
        audits = sorted(
            self.triage_audits.values(),
            key=lambda audit: audit.created_at,
            reverse=True,
        )
        return audits[offset : offset + limit]

    def status(self) -> dict[str, str]:
        return {
            "metadata_store": "memory",
            "documents": str(len(self.documents)),
            "incidents": str(len(self.incidents)),
            "reference_states": str(len(self.reference_states)),
            "triage_cases": str(len(self.triage_cases)),
            "pilot_users": str(len(self.pilot_users)),
            "media_assets": str(len(self.media_assets)),
            "triage_audits": str(len(self.triage_audits)),
        }
