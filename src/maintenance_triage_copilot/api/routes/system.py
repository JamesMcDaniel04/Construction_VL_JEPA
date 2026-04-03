"""System health endpoints."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health")
async def health(request: Request) -> dict[str, object]:
    return cast(dict[str, object], request.app.state.service.health())
