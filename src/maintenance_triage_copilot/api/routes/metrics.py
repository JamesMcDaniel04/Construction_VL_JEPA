"""Prometheus metrics endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Response

from maintenance_triage_copilot.telemetry import metrics_payload

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
async def metrics() -> Response:
    payload, content_type = metrics_payload()
    return Response(content=payload, media_type=content_type)
