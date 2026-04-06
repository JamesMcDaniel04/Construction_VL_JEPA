"""SQLAlchemy-backed metadata store."""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Column, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

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
from maintenance_triage_copilot.storage.migrations import run_migrations


class Base(DeclarativeBase):
    pass


class DocumentRecord(Base):
    __tablename__ = "corpus_documents"

    document_id = Column(String, primary_key=True)
    payload = Column(JSON, nullable=False)


class IncidentRecordORM(Base):
    __tablename__ = "incident_records"

    incident_id = Column(String, primary_key=True)
    payload = Column(JSON, nullable=False)


class ReferenceStateRecord(Base):
    __tablename__ = "reference_states"

    state_id = Column(String, primary_key=True)
    payload = Column(JSON, nullable=False)


class TriageHistoryRecord(Base):
    __tablename__ = "triage_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String, nullable=False, index=True)
    response = Column(JSON, nullable=False)


class TriageCaseRecordORM(Base):
    __tablename__ = "triage_cases"

    case_id = Column(String, primary_key=True)
    organization_id = Column(String, nullable=False, index=True)
    created_by_user_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False, index=True)
    created_at = Column(String, nullable=False, index=True)
    updated_at = Column(String, nullable=False, index=True)
    payload = Column(JSON, nullable=False)


class PilotUserRecordORM(Base):
    __tablename__ = "pilot_users"

    user_id = Column(String, primary_key=True)
    organization_id = Column(String, nullable=False, index=True)
    email = Column(String, nullable=False, index=True)
    role = Column(String, nullable=False, index=True)
    created_at = Column(String, nullable=False, index=True)
    payload = Column(JSON, nullable=False)


class SiteRecordORM(Base):
    __tablename__ = "sites"

    site_id = Column(String, primary_key=True)
    organization_id = Column(String, nullable=False, index=True)
    active = Column(String, nullable=False, index=True)
    payload = Column(JSON, nullable=False)


class AssetCatalogRecordORM(Base):
    __tablename__ = "assets"

    asset_id = Column(String, primary_key=True)
    organization_id = Column(String, nullable=False, index=True)
    site_id = Column(String, nullable=False, index=True)
    active = Column(String, nullable=False, index=True)
    payload = Column(JSON, nullable=False)


class MediaAssetRecordORM(Base):
    __tablename__ = "media_assets"

    asset_id = Column(String, primary_key=True)
    asset_type = Column(String, nullable=False, index=True)
    created_at = Column(String, nullable=False, index=True)
    payload = Column(JSON, nullable=False)


class TriageAuditRecordORM(Base):
    __tablename__ = "triage_audits"

    audit_id = Column(String, primary_key=True)
    request_id = Column(String, nullable=False, index=True)
    principal = Column(String, nullable=False, index=True)
    created_at = Column(String, nullable=False, index=True)
    payload = Column(JSON, nullable=False)


class SqlAlchemyMetadataStore:
    """Persists metadata and triage history to a SQL database."""

    def __init__(
        self,
        database_url: str,
        *,
        run_schema_migrations: bool = True,
        required: bool = False,
    ):
        if run_schema_migrations:
            run_migrations(database_url)

        engine_kwargs: dict[str, Any] = {"future": True}
        if database_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(database_url, **engine_kwargs)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False, class_=Session)
        if required:
            with self.engine.connect() as connection:
                connection.execute(select(1))

    def add_document(self, document: CorpusDocument) -> None:
        payload = document.model_dump(mode="json")
        with self.session_factory.begin() as session:
            session.merge(DocumentRecord(document_id=document.document_id, payload=payload))

    def list_documents(self, limit: int, offset: int) -> list[CorpusDocument]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(DocumentRecord)
                .order_by(DocumentRecord.document_id.asc())
                .offset(offset)
                .limit(limit)
            ).all()
        return [CorpusDocument.model_validate(self._payload(row.payload)) for row in rows]

    def add_incident(self, incident: IncidentRecord) -> None:
        payload = incident.model_dump(mode="json")
        with self.session_factory.begin() as session:
            session.merge(IncidentRecordORM(incident_id=incident.incident_id, payload=payload))

    def list_incidents(self, limit: int, offset: int) -> list[IncidentRecord]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(IncidentRecordORM)
                .order_by(IncidentRecordORM.incident_id.asc())
                .offset(offset)
                .limit(limit)
            ).all()
        return [IncidentRecord.model_validate(self._payload(row.payload)) for row in rows]

    def add_reference_state(self, reference_state: ReferenceState) -> None:
        payload = reference_state.model_dump(mode="json")
        with self.session_factory.begin() as session:
            session.merge(ReferenceStateRecord(state_id=reference_state.state_id, payload=payload))

    def list_reference_states(self, limit: int, offset: int) -> list[ReferenceState]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(ReferenceStateRecord)
                .order_by(ReferenceStateRecord.state_id.asc())
                .offset(offset)
                .limit(limit)
            ).all()
        return [ReferenceState.model_validate(self._payload(row.payload)) for row in rows]

    def get_reference_state(self, state_id: str) -> ReferenceState | None:
        with self.session_factory() as session:
            record = session.get(ReferenceStateRecord, state_id)
            if record is None:
                return None
            payload = self._payload(record.payload)
        return ReferenceState.model_validate(payload)

    def record_triage(self, request_id: str, response: TriageResponse) -> None:
        with self.session_factory.begin() as session:
            session.add(
                TriageHistoryRecord(
                    request_id=request_id,
                    response=response.model_dump(mode="json"),
                )
            )

    def save_case(self, triage_case: TriageCase) -> None:
        payload = triage_case.model_dump(mode="json")
        with self.session_factory.begin() as session:
            session.merge(
                TriageCaseRecordORM(
                    case_id=triage_case.case_id,
                    organization_id=triage_case.organization_id,
                    created_by_user_id=triage_case.created_by_user_id,
                    status=triage_case.status.value,
                    created_at=triage_case.created_at.isoformat(),
                    updated_at=triage_case.updated_at.isoformat(),
                    payload=payload,
                )
            )

    def get_case(self, case_id: str) -> TriageCase | None:
        with self.session_factory() as session:
            record = session.get(TriageCaseRecordORM, case_id)
            if record is None:
                return None
            payload = self._payload(record.payload)
        return TriageCase.model_validate(payload)

    def list_cases(self, limit: int, offset: int) -> list[TriageCase]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(TriageCaseRecordORM)
                .order_by(TriageCaseRecordORM.updated_at.desc())
                .offset(offset)
                .limit(limit)
            ).all()
        return [TriageCase.model_validate(self._payload(row.payload)) for row in rows]

    def add_pilot_user(self, pilot_user: PilotUser) -> None:
        payload = pilot_user.model_dump(mode="json")
        with self.session_factory.begin() as session:
            session.merge(
                PilotUserRecordORM(
                    user_id=pilot_user.user_id,
                    organization_id=pilot_user.organization_id,
                    email=pilot_user.email,
                    role=pilot_user.role.value,
                    created_at=pilot_user.created_at.isoformat(),
                    payload=payload,
                )
            )

    def get_pilot_user(self, user_id: str) -> PilotUser | None:
        with self.session_factory() as session:
            record = session.get(PilotUserRecordORM, user_id)
            if record is None:
                return None
            payload = self._payload(record.payload)
        return PilotUser.model_validate(payload)

    def get_pilot_user_by_email(self, email: str) -> PilotUser | None:
        with self.session_factory() as session:
            record = session.scalar(
                select(PilotUserRecordORM)
                .where(PilotUserRecordORM.email == email.strip().lower())
                .limit(1)
            )
            if record is None:
                return None
            payload = self._payload(record.payload)
        return PilotUser.model_validate(payload)

    def list_pilot_users(self, limit: int, offset: int) -> list[PilotUser]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(PilotUserRecordORM)
                .order_by(PilotUserRecordORM.created_at.desc())
                .offset(offset)
                .limit(limit)
            ).all()
        return [PilotUser.model_validate(self._payload(row.payload)) for row in rows]

    def add_site(self, site: SiteRecord) -> None:
        payload = site.model_dump(mode="json")
        with self.session_factory.begin() as session:
            session.merge(
                SiteRecordORM(
                    site_id=site.site_id,
                    organization_id=site.organization_id,
                    active=str(site.active).lower(),
                    payload=payload,
                )
            )

    def get_site(self, site_id: str) -> SiteRecord | None:
        with self.session_factory() as session:
            record = session.get(SiteRecordORM, site_id)
            if record is None:
                return None
            payload = self._payload(record.payload)
        return SiteRecord.model_validate(payload)

    def list_sites(
        self,
        limit: int,
        offset: int,
        *,
        organization_id: str | None = None,
        query: str | None = None,
        active_only: bool = False,
    ) -> list[SiteRecord]:
        with self.session_factory() as session:
            stmt = select(SiteRecordORM)
            if organization_id is not None:
                stmt = stmt.where(SiteRecordORM.organization_id == organization_id)
            if active_only:
                stmt = stmt.where(SiteRecordORM.active == "true")
            stmt = stmt.order_by(SiteRecordORM.site_id.asc())
            rows = session.scalars(stmt).all()
        items = [SiteRecord.model_validate(self._payload(row.payload)) for row in rows]
        if query:
            lowered = query.strip().lower()
            items = [
                item
                for item in items
                if lowered in item.name.lower()
                or lowered in (item.code or "").lower()
                or lowered in item.site_id.lower()
            ]
        return items[offset : offset + limit]

    def add_asset_catalog_record(self, asset: AssetRecord) -> None:
        payload = asset.model_dump(mode="json")
        with self.session_factory.begin() as session:
            session.merge(
                AssetCatalogRecordORM(
                    asset_id=asset.asset_id,
                    organization_id=asset.organization_id,
                    site_id=asset.site_id,
                    active=str(asset.active).lower(),
                    payload=payload,
                )
            )

    def get_asset_catalog_record(self, asset_id: str) -> AssetRecord | None:
        with self.session_factory() as session:
            record = session.get(AssetCatalogRecordORM, asset_id)
            if record is None:
                return None
            payload = self._payload(record.payload)
        return AssetRecord.model_validate(payload)

    def list_asset_catalog_records(
        self,
        limit: int,
        offset: int,
        *,
        organization_id: str | None = None,
        site_id: str | None = None,
        query: str | None = None,
        active_only: bool = False,
    ) -> list[AssetRecord]:
        with self.session_factory() as session:
            stmt = select(AssetCatalogRecordORM)
            if organization_id is not None:
                stmt = stmt.where(AssetCatalogRecordORM.organization_id == organization_id)
            if site_id is not None:
                stmt = stmt.where(AssetCatalogRecordORM.site_id == site_id)
            if active_only:
                stmt = stmt.where(AssetCatalogRecordORM.active == "true")
            stmt = stmt.order_by(AssetCatalogRecordORM.asset_id.asc())
            rows = session.scalars(stmt).all()
        items = [AssetRecord.model_validate(self._payload(row.payload)) for row in rows]
        if query:
            lowered = query.strip().lower()
            items = [
                item
                for item in items
                if lowered in item.display_name.lower()
                or lowered in item.asset_id.lower()
                or lowered in item.panel_family.lower()
                or lowered in (item.panel_id or "").lower()
            ]
        return items[offset : offset + limit]

    def add_media_asset(self, asset: MediaAssetRecord) -> None:
        payload = asset.model_dump(mode="json")
        with self.session_factory.begin() as session:
            session.merge(
                MediaAssetRecordORM(
                    asset_id=asset.asset_id,
                    asset_type=asset.asset_type.value,
                    created_at=asset.created_at.isoformat(),
                    payload=payload,
                )
            )

    def get_media_asset(self, asset_id: str) -> MediaAssetRecord | None:
        with self.session_factory() as session:
            record = session.get(MediaAssetRecordORM, asset_id)
            if record is None:
                return None
            payload = self._payload(record.payload)
        return MediaAssetRecord.model_validate(payload)

    def record_triage_audit(self, audit: TriageAuditRecord) -> None:
        payload = audit.model_dump(mode="json")
        with self.session_factory.begin() as session:
            session.merge(
                TriageAuditRecordORM(
                    audit_id=audit.audit_id,
                    request_id=audit.request_id,
                    principal=audit.principal,
                    created_at=audit.created_at.isoformat(),
                    payload=payload,
                )
            )

    def get_triage_audit(self, audit_id: str) -> TriageAuditRecord | None:
        with self.session_factory() as session:
            record = session.get(TriageAuditRecordORM, audit_id)
            if record is None:
                return None
            payload = self._payload(record.payload)
        return TriageAuditRecord.model_validate(payload)

    def list_triage_audits(self, limit: int, offset: int) -> list[TriageAuditRecord]:
        with self.session_factory() as session:
            rows = session.scalars(
                select(TriageAuditRecordORM)
                .order_by(TriageAuditRecordORM.created_at.desc())
                .offset(offset)
                .limit(limit)
            ).all()
        return [TriageAuditRecord.model_validate(self._payload(row.payload)) for row in rows]

    def status(self) -> dict[str, str]:
        with self.session_factory() as session:
            document_count = self._count(session, DocumentRecord)
            incident_count = self._count(session, IncidentRecordORM)
            reference_state_count = self._count(session, ReferenceStateRecord)
            triage_case_count = self._count(session, TriageCaseRecordORM)
            pilot_user_count = self._count(session, PilotUserRecordORM)
            site_count = self._count(session, SiteRecordORM)
            asset_count = self._count(session, AssetCatalogRecordORM)
            media_asset_count = self._count(session, MediaAssetRecordORM)
            audit_count = self._count(session, TriageAuditRecordORM)
        return {
            "metadata_store": "sqlalchemy",
            "documents": str(document_count),
            "incidents": str(incident_count),
            "reference_states": str(reference_state_count),
            "triage_cases": str(triage_case_count),
            "pilot_users": str(pilot_user_count),
            "sites": str(site_count),
            "assets": str(asset_count),
            "media_assets": str(media_asset_count),
            "triage_audits": str(audit_count),
        }

    @staticmethod
    def _count(session: Session, model: type[Base]) -> int:
        return int(session.scalar(select(func.count()).select_from(model)) or 0)

    @staticmethod
    def _payload(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        raise TypeError("Expected JSON payload dictionary from metadata store")
