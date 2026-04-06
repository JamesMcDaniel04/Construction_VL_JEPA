"""Site and asset catalog endpoints for pilot users."""

from __future__ import annotations

from typing import cast

from fastapi import APIRouter, HTTPException, Query, Request

from maintenance_triage_copilot.api.security import current_identity, require_role
from maintenance_triage_copilot.domain.models import (
    AssetCreateRequest,
    AssetPatchRequest,
    SiteCreateRequest,
    SitePatchRequest,
    UserRole,
)

router = APIRouter(prefix="/catalog", tags=["catalog"])


@router.get("/sites")
async def list_sites(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    q: str | None = Query(default=None),
    active_only: bool = Query(default=True),
) -> dict[str, object]:
    identity = current_identity(request)
    role = _require_catalog_role(identity["role"])
    items = request.app.state.service.list_sites(
        limit=limit,
        offset=offset,
        organization_id=cast(str | None, identity["organization_id"]),
        role=role,
        query=q,
        active_only=active_only,
    )
    return {
        "items": [item.model_dump(mode="json") for item in items],
        "limit": limit,
        "offset": offset,
    }


@router.post("/sites")
async def create_site(request_body: SiteCreateRequest, request: Request) -> dict[str, object]:
    identity = require_role(request, UserRole.admin, UserRole.service)
    organization_id = _resolve_target_organization(
        role=UserRole(identity["role"]),
        request_organization_id=cast(str | None, identity["organization_id"]),
        body_organization_id=request_body.organization_id,
    )
    try:
        site = request.app.state.service.add_site(
            request_body,
            organization_id=organization_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return cast(dict[str, object], site.model_dump(mode="json"))


@router.patch("/sites/{site_id}")
async def patch_site(
    site_id: str,
    patch: SitePatchRequest,
    request: Request,
) -> dict[str, object]:
    identity = require_role(request, UserRole.admin, UserRole.service)
    try:
        site = request.app.state.service.update_site(
            site_id,
            patch,
            organization_id=cast(str | None, identity["organization_id"]),
            role=UserRole(identity["role"]),
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if site is None:
        raise HTTPException(status_code=404, detail="Site not found")
    return cast(dict[str, object], site.model_dump(mode="json"))


@router.get("/assets")
async def list_assets(
    request: Request,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    site_id: str | None = Query(default=None),
    q: str | None = Query(default=None),
    active_only: bool = Query(default=True),
) -> dict[str, object]:
    identity = current_identity(request)
    role = _require_catalog_role(identity["role"])
    items = request.app.state.service.list_asset_catalog_records(
        limit=limit,
        offset=offset,
        organization_id=cast(str | None, identity["organization_id"]),
        role=role,
        site_id=site_id,
        query=q,
        active_only=active_only,
    )
    return {
        "items": [item.model_dump(mode="json") for item in items],
        "limit": limit,
        "offset": offset,
    }


@router.post("/assets")
async def create_asset(request_body: AssetCreateRequest, request: Request) -> dict[str, object]:
    identity = require_role(request, UserRole.admin, UserRole.service)
    organization_id = _resolve_target_organization(
        role=UserRole(identity["role"]),
        request_organization_id=cast(str | None, identity["organization_id"]),
        body_organization_id=request_body.organization_id,
    )
    try:
        asset = request.app.state.service.add_asset_catalog_record(
            request_body,
            organization_id=organization_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return cast(dict[str, object], asset.model_dump(mode="json"))


@router.patch("/assets/{asset_id}")
async def patch_asset(
    asset_id: str,
    patch: AssetPatchRequest,
    request: Request,
) -> dict[str, object]:
    identity = require_role(request, UserRole.admin, UserRole.service)
    try:
        asset = request.app.state.service.update_asset_catalog_record(
            asset_id,
            patch,
            organization_id=cast(str | None, identity["organization_id"]),
            role=UserRole(identity["role"]),
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if asset is None:
        raise HTTPException(status_code=404, detail="Asset not found")
    return cast(dict[str, object], asset.model_dump(mode="json"))


def _require_catalog_role(role_value: object) -> UserRole:
    try:
        role = UserRole(str(role_value))
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail="Authenticated catalog access is required",
        ) from exc
    if role not in {UserRole.technician, UserRole.admin, UserRole.service}:
        raise HTTPException(status_code=403, detail="Authenticated catalog access is required")
    return role


def _resolve_target_organization(
    *,
    role: UserRole,
    request_organization_id: str | None,
    body_organization_id: str | None,
) -> str:
    if role == UserRole.service:
        if body_organization_id is None:
            raise HTTPException(
                status_code=400,
                detail="organization_id is required for service-scoped catalog writes",
            )
        return body_organization_id
    if request_organization_id is None:
        raise HTTPException(status_code=400, detail="Organization context is required")
    if body_organization_id is not None and body_organization_id != request_organization_id:
        raise HTTPException(
            status_code=403,
            detail="Admins can only manage catalog records for their own organization",
        )
    return request_organization_id
