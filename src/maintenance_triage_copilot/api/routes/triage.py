"""Triage endpoint."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request, Response

from maintenance_triage_copilot.domain.models import TriageRequest, TriageResponse
from maintenance_triage_copilot.telemetry import current_trace_id

router = APIRouter(prefix="/triage", tags=["triage"])


@router.post("/analyze", response_model=TriageResponse)
async def analyze(request_body: TriageRequest, request: Request, response: Response) -> TriageResponse:
    service = request.app.state.service
    triage_response = cast(TriageResponse, service.analyze(request_body))
    audit = service.record_triage_audit(
        request_id=request.state.request_id,
        principal=request.state.principal,
        trace_id=current_trace_id(),
        request_payload=request_body,
        response_payload=triage_response,
        linked_asset_ids=[],
        metadata={"route": "/triage/analyze"},
    )
    response.headers["X-Audit-ID"] = audit.audit_id
    return triage_response
