"""Machine endpoints — health, anomalies, latent trajectory, feedback."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class MachineInfo(BaseModel):
    machine_id: str
    machine_type: str
    location: str | None = None
    install_date: str | None = None
    status: str = "unknown"


class HealthResponse(BaseModel):
    machine_id: str
    current_score: float
    normalized_score: float
    status: str  # green / yellow / red
    trend: str  # stable / rising
    baseline_mean: float
    baseline_std: float


class AnomalySummary(BaseModel):
    anomaly_id: str
    machine_id: str
    timestamp: str
    score: float
    is_trend_anomaly: bool
    contributing_signals: list[dict] = Field(default_factory=list)
    matched_pattern_ids: list[str] = Field(default_factory=list)


class AnomalyDetail(AnomalySummary):
    normalized_score: float
    pattern_matches: list[dict] = Field(default_factory=list)
    recommended_action: str | None = None


class FeedbackRequest(BaseModel):
    anomaly_id: str
    is_true_positive: bool
    notes: str | None = None
    actual_event_type: str | None = None


class FeedbackResponse(BaseModel):
    status: str
    message: str


class PatternInfo(BaseModel):
    pattern_id: str
    name: str | None = None
    event_type: str
    historical_lead_time_hours: float | None = None
    past_occurrences: int = 0
    description: str | None = None


# ---------------------------------------------------------------------------
# In-memory stores (replaced by DB in production)
# ---------------------------------------------------------------------------

# These are placeholder stores. In production, these would be backed by
# TimescaleDB + Qdrant queries via dependency injection.
_machines: dict[str, MachineInfo] = {}
_anomalies: dict[str, list[AnomalySummary]] = {}
_feedback: list[dict] = []


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=list[MachineInfo])
async def list_machines():
    """List all monitored machines."""
    return list(_machines.values())


@router.post("", response_model=MachineInfo, status_code=201)
async def register_machine(machine: MachineInfo):
    """Register a new machine for monitoring."""
    _machines[machine.machine_id] = machine
    return machine


@router.get("/{machine_id}", response_model=MachineInfo)
async def get_machine(machine_id: str):
    """Get machine details."""
    if machine_id not in _machines:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")
    return _machines[machine_id]


@router.get("/{machine_id}/health", response_model=HealthResponse)
async def get_machine_health(machine_id: str):
    """Get current anomaly score and health trend for a machine."""
    if machine_id not in _machines:
        raise HTTPException(status_code=404, detail=f"Machine {machine_id} not found")

    # In production, this queries the inference server's AnomalyDetector
    return HealthResponse(
        machine_id=machine_id,
        current_score=0.0,
        normalized_score=0.0,
        status="green",
        trend="stable",
        baseline_mean=0.0,
        baseline_std=1.0,
    )


@router.get("/{machine_id}/anomalies", response_model=list[AnomalySummary])
async def get_anomalies(
    machine_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """Get historical anomaly events for a machine."""
    anomalies = _anomalies.get(machine_id, [])
    return anomalies[offset : offset + limit]


@router.get("/{machine_id}/anomalies/{anomaly_id}", response_model=AnomalyDetail)
async def get_anomaly_detail(machine_id: str, anomaly_id: str):
    """Get detailed information about a specific anomaly."""
    anomalies = _anomalies.get(machine_id, [])
    for a in anomalies:
        if a.anomaly_id == anomaly_id:
            return AnomalyDetail(
                **a.model_dump(),
                normalized_score=0.0,
                pattern_matches=[],
                recommended_action=None,
            )
    raise HTTPException(status_code=404, detail=f"Anomaly {anomaly_id} not found")


@router.post("/{machine_id}/feedback", response_model=FeedbackResponse)
async def submit_feedback(machine_id: str, feedback: FeedbackRequest):
    """Submit operator feedback on an anomaly (true/false positive)."""
    _feedback.append({
        "machine_id": machine_id,
        **feedback.model_dump(),
        "submitted_at": datetime.now().isoformat(),
    })
    return FeedbackResponse(
        status="accepted",
        message="Feedback recorded. Will be used for threshold tuning and model retraining.",
    )


@router.get("/{machine_id}/latent-trajectory")
async def get_latent_trajectory(
    machine_id: str,
    hours: int = Query(default=24, ge=1, le=168),
):
    """Get the latent state trajectory over time for visualization.

    Returns time-series of 2D PCA-projected latent states for plotting.
    """
    # In production: query vector DB for embeddings, PCA project, return series
    return {
        "machine_id": machine_id,
        "hours": hours,
        "trajectory": [],
        "note": "Requires trained model and stored embeddings",
    }
