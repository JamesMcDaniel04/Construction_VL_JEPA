"""Admin endpoints for persisted pilot-user invites."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Query, Request

from maintenance_triage_copilot.api.security import require_role
from maintenance_triage_copilot.domain.models import (
    PilotUserInviteRequest,
    PilotUserInviteResponse,
    UserRole,
)

router = APIRouter(prefix="/admin/pilot-users", tags=["admin"])


@router.get("")
async def list_pilot_users(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    identity = require_role(request, UserRole.admin, UserRole.service)
    items = request.app.state.service.list_pilot_users(limit=limit, offset=offset)
    if identity["role"] == UserRole.admin.value:
        items = [
            item
            for item in items
            if item.organization_id == identity["organization_id"]
        ]
    return {
        "items": [
            request.app.state.service.pilot_user_view(item).model_dump(mode="json")
            for item in items
        ],
        "limit": limit,
        "offset": offset,
    }


@router.post("/invite", response_model=PilotUserInviteResponse)
async def invite_pilot_user(
    invite: PilotUserInviteRequest,
    request: Request,
) -> PilotUserInviteResponse:
    identity = require_role(request, UserRole.admin, UserRole.service)
    if (
        identity["role"] == UserRole.admin.value
        and invite.organization_id != identity["organization_id"]
    ):
        raise HTTPException(
            status_code=403,
            detail="Admins can only invite users into their own organization",
        )
    response = cast(
        PilotUserInviteResponse,
        None,
    )
    try:
        response = cast(
            PilotUserInviteResponse,
            request.app.state.service.invite_pilot_user(
                invite,
                invited_by_user_id=cast(str, identity["user_id"] or identity["principal"]),
            ),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    refresh = getattr(request.app.state, "refresh_pilot_user_lookup", None)
    if callable(refresh):
        refresh()
    return response
