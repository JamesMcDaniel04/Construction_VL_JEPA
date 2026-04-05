"""Authenticated identity endpoints for pilot users and service clients."""

from __future__ import annotations

from fastapi import APIRouter, Request

from maintenance_triage_copilot.api.security import current_identity

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/me")
async def me(request: Request) -> dict[str, object]:
    return current_identity(request)
