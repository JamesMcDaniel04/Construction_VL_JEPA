"""Metadata store protocol."""

from __future__ import annotations

from typing import Protocol

from maintenance_triage_copilot.domain.models import (
    AssetRecord,
    CorpusDocument,
    IncidentRecord,
    MediaAssetRecord,
    PilotUser,
    ReferenceState,
    SiteRecord,
    TriageAuditRecord,
    TriageCase,
    TriageResponse,
)


class MetadataStore(Protocol):
    def add_document(self, document: CorpusDocument) -> None: ...

    def list_documents(self, limit: int, offset: int) -> list[CorpusDocument]: ...

    def add_incident(self, incident: IncidentRecord) -> None: ...

    def list_incidents(self, limit: int, offset: int) -> list[IncidentRecord]: ...

    def add_reference_state(self, reference_state: ReferenceState) -> None: ...

    def list_reference_states(self, limit: int, offset: int) -> list[ReferenceState]: ...

    def get_reference_state(self, state_id: str) -> ReferenceState | None: ...

    def record_triage(self, request_id: str, response: TriageResponse) -> None: ...

    def save_case(self, triage_case: TriageCase) -> None: ...

    def get_case(self, case_id: str) -> TriageCase | None: ...

    def list_cases(self, limit: int, offset: int) -> list[TriageCase]: ...

    def add_pilot_user(self, pilot_user: PilotUser) -> None: ...

    def get_pilot_user(self, user_id: str) -> PilotUser | None: ...

    def get_pilot_user_by_email(self, email: str) -> PilotUser | None: ...

    def list_pilot_users(self, limit: int, offset: int) -> list[PilotUser]: ...

    def add_site(self, site: SiteRecord) -> None: ...

    def get_site(self, site_id: str) -> SiteRecord | None: ...

    def list_sites(
        self,
        limit: int,
        offset: int,
        *,
        organization_id: str | None = None,
        query: str | None = None,
        active_only: bool = False,
    ) -> list[SiteRecord]: ...

    def add_asset_catalog_record(self, asset: AssetRecord) -> None: ...

    def get_asset_catalog_record(self, asset_id: str) -> AssetRecord | None: ...

    def list_asset_catalog_records(
        self,
        limit: int,
        offset: int,
        *,
        organization_id: str | None = None,
        site_id: str | None = None,
        query: str | None = None,
        active_only: bool = False,
    ) -> list[AssetRecord]: ...

    def add_media_asset(self, asset: MediaAssetRecord) -> None: ...

    def get_media_asset(self, asset_id: str) -> MediaAssetRecord | None: ...

    def record_triage_audit(self, audit: TriageAuditRecord) -> None: ...

    def get_triage_audit(self, audit_id: str) -> TriageAuditRecord | None: ...

    def list_triage_audits(self, limit: int, offset: int) -> list[TriageAuditRecord]: ...

    def status(self) -> dict[str, str]: ...
