"""Triage endpoint."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request

from maintenance_triage_copilot.domain.models import TriageRequest, TriageResponse

router = APIRouter(prefix="/triage", tags=["triage"])


@router.post("/analyze", response_model=TriageResponse)
async def analyze(request_body: TriageRequest, request: Request) -> TriageResponse:
    return cast(TriageResponse, request.app.state.service.analyze(request_body))
