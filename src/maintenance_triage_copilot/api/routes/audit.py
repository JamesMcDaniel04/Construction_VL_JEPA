"""Audit history endpoints."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Query, Request

from maintenance_triage_copilot.domain.models import TriageAuditDetail, TriageAuditRecord

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/triage")
async def list_triage_audits(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    items = request.app.state.service.list_triage_audits(limit=limit, offset=offset)
    return {
        "items": [cast(TriageAuditRecord, item).model_dump(mode="json") for item in items],
        "limit": limit,
        "offset": offset,
    }


@router.get("/triage/{audit_id}")
async def get_triage_audit(audit_id: str, request: Request) -> dict[str, object]:
    detail = cast(
        TriageAuditDetail | None,
        request.app.state.service.get_triage_audit_detail(audit_id),
    )
    if detail is None:
        raise HTTPException(status_code=404, detail="Audit record not found")
    return cast(dict[str, object], detail.model_dump(mode="json"))
