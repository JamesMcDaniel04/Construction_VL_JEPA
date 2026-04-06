"""Auth helpers for pilot users and service integrations."""

from __future__ import annotations

from typing import Any, cast

from fastapi import HTTPException, Request

from maintenance_triage_copilot.domain.models import UserRole


def current_identity(request: Request) -> dict[str, Any]:
    return {
        "principal": getattr(request.state, "principal", "anonymous"),
        "principal_type": getattr(request.state, "principal_type", "anonymous"),
        "role": getattr(request.state, "role", "anonymous"),
        "user_id": getattr(request.state, "user_id", None),
        "organization_id": getattr(request.state, "organization_id", None),
        "display_name": getattr(request.state, "display_name", "anonymous"),
        "email": getattr(request.state, "email", None),
    }


def require_role(request: Request, *allowed: UserRole) -> dict[str, Any]:
    identity = current_identity(request)
    role_value = identity["role"]
    if role_value not in {item.value for item in allowed}:
        raise HTTPException(status_code=403, detail="Insufficient role for this endpoint")
    return identity


def require_human_user(request: Request) -> dict[str, Any]:
    identity = current_identity(request)
    if identity["principal_type"] != "human":
        raise HTTPException(status_code=403, detail="This endpoint requires an invited pilot user")
    return identity


def can_access_case(
    *,
    request: Request,
    organization_id: str,
    created_by_user_id: str,
) -> bool:
    identity = current_identity(request)
    role = str(identity["role"])
    if role == UserRole.service.value:
        return True
    request_organization_id = cast(str | None, identity["organization_id"])
    request_user_id = cast(str | None, identity["user_id"])
    if request_organization_id != organization_id:
        return False
    if role == UserRole.admin.value:
        return True
    return request_user_id == created_by_user_id
