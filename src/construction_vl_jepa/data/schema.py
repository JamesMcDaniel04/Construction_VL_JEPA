"""Data models / schemas for the PFIL platform.

These Pydantic models define the contracts for data flowing through the system.
SQLAlchemy models define the persistent storage schema.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


# ---------------------------------------------------------------------------
# Pydantic models (API / ingestion contracts)
# ---------------------------------------------------------------------------


class SourceType(str, Enum):
    SENSOR = "sensor"
    LOG = "log"
    VIDEO = "video"
    NOTE = "note"


class QualityFlag(str, Enum):
    GOOD = "good"
    SUSPECT = "suspect"
    BAD = "bad"


class EventType(str, Enum):
    FAILURE = "failure"
    REPAIR = "repair"
    INSPECTION = "inspection"
    SCHEDULED = "scheduled"


class IngestRecord(BaseModel):
    """Universal ingestion contract — every source must produce this."""

    machine_id: str
    timestamp: datetime
    source_type: SourceType
    payload: dict


class SensorReading(BaseModel):
    machine_id: str
    sensor_id: str
    timestamp: datetime
    value: float
    quality_flag: QualityFlag = QualityFlag.GOOD


class MaintenanceEvent(BaseModel):
    machine_id: str
    event_id: str
    timestamp: datetime
    event_type: EventType
    description: str
    root_cause: str | None = None


class AnomalyScore(BaseModel):
    machine_id: str
    timestamp: datetime
    score: float = Field(ge=0.0, le=1.0)
    contributing_signals: list[dict] = Field(default_factory=list)
    matched_pattern_ids: list[str] = Field(default_factory=list)


class AlertPayload(BaseModel):
    machine_id: str
    anomaly_id: str
    score: float
    trend: str | None = None
    contributing_signals: list[dict] = Field(default_factory=list)
    matched_patterns: list[dict] = Field(default_factory=list)
    recommended_action: str | None = None
    dashboard_url: str | None = None


class MachineHealth(BaseModel):
    machine_id: str
    current_score: float
    trend: str
    last_updated: datetime
    status: str  # green / yellow / red


# ---------------------------------------------------------------------------
# SQLAlchemy ORM models (persistent storage)
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class MachineORM(Base):
    __tablename__ = "machines"

    machine_id = Column(String, primary_key=True)
    machine_type = Column(String, nullable=False)
    location = Column(String)
    install_date = Column(DateTime)
    sensor_config = Column(JSONB, default=dict)
    created_at = Column(DateTime, server_default=func.now())

    sensor_readings = relationship("SensorReadingORM", back_populates="machine")
    maintenance_events = relationship("MaintenanceEventORM", back_populates="machine")
    anomaly_scores = relationship("AnomalyScoreORM", back_populates="machine")


class SensorReadingORM(Base):
    __tablename__ = "sensor_readings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String, ForeignKey("machines.machine_id"), nullable=False, index=True)
    sensor_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    value = Column(Float, nullable=False)
    quality_flag = Column(String, default="good")

    machine = relationship("MachineORM", back_populates="sensor_readings")


class MaintenanceEventORM(Base):
    __tablename__ = "maintenance_events"

    event_id = Column(String, primary_key=True)
    machine_id = Column(String, ForeignKey("machines.machine_id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    event_type = Column(String, nullable=False)
    description = Column(Text)
    root_cause = Column(Text, nullable=True)

    machine = relationship("MachineORM", back_populates="maintenance_events")


class AnomalyScoreORM(Base):
    __tablename__ = "anomaly_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    machine_id = Column(String, ForeignKey("machines.machine_id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    score = Column(Float, nullable=False)
    contributing_signals = Column(JSONB, default=list)
    matched_pattern_ids = Column(JSONB, default=list)

    machine = relationship("MachineORM", back_populates="anomaly_scores")
