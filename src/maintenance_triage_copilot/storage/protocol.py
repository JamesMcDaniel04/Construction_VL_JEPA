"""Metadata store protocol."""

from __future__ import annotations

from typing import Protocol

from maintenance_triage_copilot.domain.models import (
    CorpusDocument,
    IncidentRecord,
    MediaAssetRecord,
    ReferenceState,
    TriageAuditRecord,
    TriageResponse,
)


class MetadataStore(Protocol):
    def add_document(self, document: CorpusDocument) -> None: ...

    def add_incident(self, incident: IncidentRecord) -> None: ...

    def add_reference_state(self, reference_state: ReferenceState) -> None: ...

    def get_reference_state(self, state_id: str) -> ReferenceState | None: ...

    def record_triage(self, request_id: str, response: TriageResponse) -> None: ...

    def add_media_asset(self, asset: MediaAssetRecord) -> None: ...

    def get_media_asset(self, asset_id: str) -> MediaAssetRecord | None: ...

    def record_triage_audit(self, audit: TriageAuditRecord) -> None: ...

    def get_triage_audit(self, audit_id: str) -> TriageAuditRecord | None: ...

    def list_triage_audits(self, limit: int, offset: int) -> list[TriageAuditRecord]: ...

    def status(self) -> dict[str, str]: ...
