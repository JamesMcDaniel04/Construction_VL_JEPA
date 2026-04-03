"""Reference state ingestion."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request

from maintenance_triage_copilot.domain.models import ReferenceState

router = APIRouter(prefix="/reference-states", tags=["reference-states"])


@router.post("")
async def add_reference_state(
    reference_state: ReferenceState, request: Request
) -> dict[str, object]:
    return cast(
        dict[str, object],
        request.app.state.service.add_reference_state(reference_state),
    )
