"""SQLAlchemy-backed metadata store."""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, Column, Integer, String, create_engine, func, select
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from maintenance_triage_copilot.domain.models import (
    CorpusDocument,
    IncidentRecord,
    ReferenceState,
    TriageResponse,
)


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


class SqlAlchemyMetadataStore:
    """Persists metadata and triage history to a SQL database."""

    def __init__(self, database_url: str):
        engine_kwargs: dict[str, Any] = {"future": True}
        if database_url.startswith("sqlite"):
            engine_kwargs["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(database_url, **engine_kwargs)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False, class_=Session)
        Base.metadata.create_all(self.engine)

    def add_document(self, document: CorpusDocument) -> None:
        payload = document.model_dump(mode="json")
        with self.session_factory.begin() as session:
            session.merge(DocumentRecord(document_id=document.document_id, payload=payload))

    def add_incident(self, incident: IncidentRecord) -> None:
        payload = incident.model_dump(mode="json")
        with self.session_factory.begin() as session:
            session.merge(IncidentRecordORM(incident_id=incident.incident_id, payload=payload))

    def add_reference_state(self, reference_state: ReferenceState) -> None:
        payload = reference_state.model_dump(mode="json")
        with self.session_factory.begin() as session:
            session.merge(ReferenceStateRecord(state_id=reference_state.state_id, payload=payload))

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

    def status(self) -> dict[str, str]:
        with self.session_factory() as session:
            document_count = self._count(session, DocumentRecord)
            incident_count = self._count(session, IncidentRecordORM)
            reference_state_count = self._count(session, ReferenceStateRecord)
        return {
            "metadata_store": "sqlalchemy",
            "documents": str(document_count),
            "incidents": str(incident_count),
            "reference_states": str(reference_state_count),
        }

    @staticmethod
    def _count(session: Session, model: type[Base]) -> int:
        return int(session.scalar(select(func.count()).select_from(model)) or 0)

    @staticmethod
    def _payload(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        raise TypeError("Expected JSON payload dictionary from metadata store")
