"""Technician-facing case lifecycle endpoints."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile

from maintenance_triage_copilot.api.media_quality import assess_image_capture
from maintenance_triage_copilot.api.security import can_access_case, current_identity
from maintenance_triage_copilot.domain.models import (
    TriageCase,
    TriageCaseCreateRequest,
    TriageCaseFeedback,
    TriageCaseSummary,
    UserRole,
)
from maintenance_triage_copilot.telemetry import current_trace_id

router = APIRouter(prefix="/cases", tags=["cases"])

_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
_VIDEO_CONTENT_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}


@router.post("", response_model=TriageCase)
async def create_case(request_body: TriageCaseCreateRequest, request: Request) -> TriageCase:
    identity = current_identity(request)
    role = UserRole(identity["role"])
    organization_id = cast(str | None, identity["organization_id"])
    if role == UserRole.service:
        organization_id = str(request_body.metadata.get("organization_id", "service-default"))
    if organization_id is None:
        raise HTTPException(status_code=400, detail="Organization context is required")
    triage_case = request.app.state.service.create_case(
        request_body,
        organization_id=organization_id,
        user_id=cast(str, identity["user_id"] or identity["principal"]),
        display_name=cast(str, identity["display_name"]),
        role=role,
    )
    return cast(TriageCase, triage_case)


@router.post("/{case_id}/analyze", response_model=TriageCase)
async def analyze_case(
    case_id: str,
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File()],
) -> TriageCase:
    service = request.app.state.service
    triage_case = cast(TriageCase | None, service.get_case(case_id))
    if triage_case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    if not can_access_case(
        request=request,
        organization_id=triage_case.organization_id,
        created_by_user_id=triage_case.created_by_user_id,
    ):
        raise HTTPException(status_code=403, detail="Not allowed to analyze this case")

    content_type = file.content_type or "application/octet-stream"
    media_type = _detect_media_type(file.filename, content_type)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")
    _enforce_upload_size(request, media_type=media_type, size=len(data))

    if media_type == "image":
        quality_hints = assess_image_capture(data)
        if quality_hints:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "Capture quality is too weak for trustworthy visual triage.",
                    "hints": quality_hints,
                },
            )

    updated_case, audit = service.analyze_case(
        triage_case=triage_case,
        filename=file.filename or "capture.bin",
        content_type=content_type,
        data=data,
        request_id=request.state.request_id,
        principal=request.state.principal,
        trace_id=current_trace_id(),
    )
    response.headers["X-Audit-ID"] = audit.audit_id
    return cast(TriageCase, updated_case)


@router.post("/{case_id}/feedback", response_model=TriageCase)
async def submit_feedback(
    case_id: str,
    feedback: TriageCaseFeedback,
    request: Request,
) -> TriageCase:
    service = request.app.state.service
    triage_case = cast(TriageCase | None, service.get_case(case_id))
    if triage_case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    if not can_access_case(
        request=request,
        organization_id=triage_case.organization_id,
        created_by_user_id=triage_case.created_by_user_id,
    ):
        raise HTTPException(status_code=403, detail="Not allowed to update this case")
    updated = service.submit_case_feedback(
        case_id,
        labels=feedback.labels,
        comment=feedback.comment,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return cast(TriageCase, updated)


@router.get("/{case_id}", response_model=TriageCase)
async def get_case(case_id: str, request: Request) -> TriageCase:
    service = request.app.state.service
    triage_case = cast(TriageCase | None, service.get_case(case_id))
    if triage_case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    if not can_access_case(
        request=request,
        organization_id=triage_case.organization_id,
        created_by_user_id=triage_case.created_by_user_id,
    ):
        raise HTTPException(status_code=403, detail="Not allowed to access this case")
    return triage_case


@router.get("")
async def list_cases(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    identity = current_identity(request)
    items = request.app.state.service.list_cases(
        limit=limit,
        offset=offset,
        organization_id=cast(str | None, identity["organization_id"]),
        user_id=cast(str | None, identity["user_id"]),
        role=UserRole(identity["role"]),
    )
    return {
        "items": [
            TriageCaseSummary.model_validate(
                request.app.state.service.case_summary(item)
            ).model_dump(mode="json")
            for item in items
        ],
        "limit": limit,
        "offset": offset,
    }


def _detect_media_type(filename: str | None, content_type: str) -> str:
    lower_name = (filename or "").lower()
    if content_type in _IMAGE_CONTENT_TYPES or lower_name.endswith(
        (".jpg", ".jpeg", ".png", ".webp")
    ):
        return "image"
    if content_type in _VIDEO_CONTENT_TYPES or lower_name.endswith(
        (".mp4", ".avi", ".mov", ".mkv")
    ):
        return "video"
    raise HTTPException(
        status_code=415,
        detail="Unsupported media type. Expected JPEG/PNG/WebP image or MP4/AVI/MOV video.",
    )


def _enforce_upload_size(request: Request, *, media_type: str, size: int) -> None:
    cfg = request.app.state.service.state.config.runtime
    limit = cfg.max_image_upload_bytes if media_type == "image" else cfg.max_video_upload_bytes
    if size > limit:
        raise HTTPException(status_code=413, detail=f"Upload exceeds maximum size of {limit} bytes")
