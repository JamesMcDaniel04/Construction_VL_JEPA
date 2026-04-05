"""Admin dashboard endpoints for pilot corpus and case review."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, Request

from maintenance_triage_copilot.api.security import require_role
from maintenance_triage_copilot.domain.models import AdminDashboardMetrics, UserRole

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/dashboard")
async def dashboard(request: Request) -> dict[str, object]:
    identity = require_role(request, UserRole.admin, UserRole.service)
    organization_id = (
        None
        if identity["role"] == UserRole.service.value
        else identity["organization_id"]
    )
    metrics = cast(
        AdminDashboardMetrics,
        request.app.state.service.admin_dashboard(organization_id=organization_id),
    )
    return cast(dict[str, object], metrics.model_dump(mode="json"))
