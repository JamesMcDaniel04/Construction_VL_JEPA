"""Reference state ingestion."""

from __future__ import annotations

import json
import uuid
from typing import Annotated, cast

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from maintenance_triage_copilot.api.security import require_role
from maintenance_triage_copilot.domain.models import AssetType, MediaType, ReferenceState, UserRole

router = APIRouter(prefix="/reference-states", tags=["reference-states"])


@router.post("")
async def add_reference_state(
    reference_state: ReferenceState, request: Request
) -> dict[str, object]:
    require_role(request, UserRole.admin, UserRole.service)
    return cast(
        dict[str, object],
        request.app.state.service.add_reference_state(reference_state),
    )


@router.get("")
async def list_reference_states(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> dict[str, object]:
    require_role(request, UserRole.admin, UserRole.service)
    items = request.app.state.service.list_reference_states(limit=limit, offset=offset)
    return {
        "items": [cast(ReferenceState, item).model_dump(mode="json") for item in items],
        "limit": limit,
        "offset": offset,
    }


@router.post("/upload")
async def upload_reference_state(
    request: Request,
    file: Annotated[UploadFile, File()],
    state_label: Annotated[str, Form()],
    description: Annotated[str, Form()],
    caption: Annotated[str, Form()],
    state_id: Annotated[str | None, Form()] = None,
    equipment_family: Annotated[str, Form()] = "electrical_panel_family_a",
    allowed_variance_notes: Annotated[str | None, Form()] = None,
    metadata_json: Annotated[str | None, Form()] = None,
) -> dict[str, object]:
    require_role(request, UserRole.admin, UserRole.service)
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")
    filename = file.filename or "reference-state.bin"
    content_type = file.content_type or "application/octet-stream"
    media_type = _detect_media_type(filename, content_type)

    service = request.app.state.service
    asset = service.persist_uploaded_asset(
        asset_type=AssetType.reference_state_upload,
        filename=filename,
        content_type=content_type,
        data=data,
        metadata={"equipment_family": equipment_family, "state_label": state_label},
    )
    if media_type == MediaType.image:
        embedding = service.state.image_backbone.encode_raw_image(data)
    else:
        embedding = service.state.video_backbone.encode_raw_video(data)

    metadata: dict[str, str | int | float | bool] = {"asset_id": asset.asset_id}
    if metadata_json:
        metadata.update(_parse_metadata_json(metadata_json))
    reference_state = ReferenceState(
        state_id=state_id or f"state-{uuid.uuid4().hex[:12]}",
        equipment_family=equipment_family,
        state_label=state_label,
        description=description,
        allowed_variance_notes=allowed_variance_notes,
        media_type=media_type,
        caption=caption,
        embedding_values=embedding.tolist(),
        metadata=metadata,
    )
    result = service.add_reference_state(reference_state)
    result["asset_id"] = asset.asset_id
    return cast(dict[str, object], result)


def _detect_media_type(filename: str, content_type: str) -> MediaType:
    lower_name = filename.lower()
    if content_type.startswith("image/") or lower_name.endswith((".jpg", ".jpeg", ".png", ".webp")):
        return MediaType.image
    if content_type.startswith("video/") or lower_name.endswith((".mp4", ".avi", ".mov", ".mkv")):
        return MediaType.video
    raise HTTPException(status_code=415, detail="Unsupported reference-state media type")


def _parse_metadata_json(raw: str) -> dict[str, str | int | float | bool]:
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="metadata_json must encode an object")
    return {
        str(key): cast(str | int | float | bool, item)
        for key, item in value.items()
        if isinstance(item, (str, int, float, bool))
    }
