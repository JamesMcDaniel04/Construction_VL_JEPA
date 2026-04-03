"""In-memory metadata store used by the API and tests."""

from __future__ import annotations

from maintenance_triage_copilot.domain.models import (
    CorpusDocument,
    IncidentRecord,
    ReferenceState,
    TriageResponse,
)


class MemoryMetadataStore:
    def __init__(self) -> None:
        self.documents: dict[str, CorpusDocument] = {}
        self.incidents: dict[str, IncidentRecord] = {}
        self.reference_states: dict[str, ReferenceState] = {}
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

    def status(self) -> dict[str, str]:
        return {
            "metadata_store": "memory",
            "documents": str(len(self.documents)),
            "incidents": str(len(self.incidents)),
            "reference_states": str(len(self.reference_states)),
        }
