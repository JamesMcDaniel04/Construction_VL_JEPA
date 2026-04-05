"""Media upload endpoints for JPEG/PNG images and MP4 video files.

These endpoints accept raw binary uploads rather than requiring pre-computed
tensors, making the API usable from mobile/wearable cameras and field devices.
"""

from __future__ import annotations

import uuid
from typing import Annotated, Any, cast

from fastapi import APIRouter, File, Form, HTTPException, Request, Response, UploadFile

from maintenance_triage_copilot.domain.models import (
    AssetType,
    MediaType,
    TriageRequest,
    TriageResponse,
    VisualObservation,
)
from maintenance_triage_copilot.telemetry import current_trace_id

router = APIRouter(prefix="/media", tags=["media"])

_IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
_VIDEO_CONTENT_TYPES = {"video/mp4", "video/avi", "video/quicktime", "video/x-msvideo"}


@router.post("/triage", response_model=TriageResponse)
async def triage_from_upload(
    request: Request,
    response: Response,
    file: Annotated[UploadFile, File()],
    equipment_family: Annotated[str, Form()] = "electrical_panel_family_a",
    panel_id: Annotated[str | None, Form()] = None,
    question: Annotated[str | None, Form()] = None,
    operator_context: Annotated[str | None, Form()] = None,
    expected_state_label: Annotated[str | None, Form()] = None,
) -> TriageResponse:
    """Run triage on an uploaded image or video file.

    Accepts JPEG, PNG, WebP images and MP4, AVI, MOV video files.
    """
    content_type = file.content_type or ""
    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")
    _enforce_upload_size(
        size=len(data),
        limit=request.app.state.service.state.config.runtime.max_video_upload_bytes
        if content_type in _VIDEO_CONTENT_TYPES or _looks_like_video(file.filename)
        else request.app.state.service.state.config.runtime.max_image_upload_bytes,
    )

    service = request.app.state.service
    observation_id = f"upload-{uuid.uuid4().hex[:12]}"
    stored_asset = service.persist_uploaded_asset(
        asset_type=AssetType.triage_upload,
        filename=file.filename or "upload.bin",
        content_type=content_type or "application/octet-stream",
        data=data,
        metadata={"equipment_family": equipment_family, "panel_id": panel_id or ""},
    )

    if content_type in _IMAGE_CONTENT_TYPES or _looks_like_image(file.filename):
        embedding = service.state.image_backbone.encode_raw_image(data)
        media_type = MediaType.image
    elif content_type in _VIDEO_CONTENT_TYPES or _looks_like_video(file.filename):
        embedding = service.state.video_backbone.encode_raw_video(data)
        media_type = MediaType.video
    else:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type: {content_type}. "
            "Expected JPEG/PNG/WebP image or MP4/AVI/MOV video.",
        )

    observation = VisualObservation(
        observation_id=observation_id,
        equipment_family=equipment_family,
        panel_id=panel_id,
        media_type=media_type,
        embedding_values=embedding.tolist(),
    )
    triage_request = TriageRequest(
        observation=observation,
        question=question,
        operator_context=operator_context,
        expected_state_label=expected_state_label,
    )
    triage_response = cast(TriageResponse, service.analyze(triage_request))
    audit = service.record_triage_audit(
        request_id=request.state.request_id,
        principal=request.state.principal,
        trace_id=current_trace_id(),
        request_payload=triage_request,
        response_payload=triage_response,
        linked_asset_ids=[stored_asset.asset_id],
        metadata={"route": "/media/triage"},
    )
    response.headers["X-Audit-ID"] = audit.audit_id
    return triage_response


@router.post("/encode")
async def encode_media(
    request: Request,
    file: Annotated[UploadFile, File()],
) -> dict[str, Any]:
    """Encode an uploaded file and return the raw embedding vector.

    Useful for building custom retrieval pipelines or debugging.
    """
    content_type = file.content_type or ""
    data = await file.read()

    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")

    service = request.app.state.service

    if content_type in _IMAGE_CONTENT_TYPES or _looks_like_image(file.filename):
        _enforce_upload_size(
            size=len(data),
            limit=service.state.config.runtime.max_image_upload_bytes,
        )
        embedding = service.state.image_backbone.encode_raw_image(data)
        modality = "image"
    elif content_type in _VIDEO_CONTENT_TYPES or _looks_like_video(file.filename):
        _enforce_upload_size(
            size=len(data),
            limit=service.state.config.runtime.max_video_upload_bytes,
        )
        embedding = service.state.video_backbone.encode_raw_video(data)
        modality = "video"
    else:
        raise HTTPException(status_code=415, detail=f"Unsupported media type: {content_type}")

    return {
        "modality": modality,
        "embedding_dim": len(embedding),
        "embedding_values": embedding.tolist(),
    }


def _looks_like_image(filename: str | None) -> bool:
    if not filename:
        return False
    return filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))


def _looks_like_video(filename: str | None) -> bool:
    if not filename:
        return False
    return filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))


def _enforce_upload_size(*, size: int, limit: int) -> None:
    if size > limit:
        raise HTTPException(status_code=413, detail=f"Upload exceeds maximum size of {limit} bytes")
