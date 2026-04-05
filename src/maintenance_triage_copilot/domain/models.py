"""Pydantic models for maintenance triage."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CorpusSourceType(StrEnum):
    manual = "manual"
    sop = "sop"
    ticket = "ticket"
    repair_note = "repair_note"


class MediaType(StrEnum):
    image = "image"
    video = "video"


class AssetType(StrEnum):
    triage_upload = "triage_upload"
    corpus_upload = "corpus_upload"


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    title: str
    source_type: CorpusSourceType
    chunk_id: str | None = None
    snippet: str | None = None


class CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str
    source_type: CorpusSourceType
    title: str
    body: str
    equipment_family: str = "electrical_panel_family_a"
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class IncidentRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str
    title: str
    summary: str
    issue_class: str
    fix_summary: str
    resolution_notes: str | None = None
    equipment_family: str = "electrical_panel_family_a"
    linked_document_ids: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    def searchable_text(self) -> str:
        parts = [self.title, self.summary, self.issue_class, self.fix_summary]
        if self.resolution_notes:
            parts.append(self.resolution_notes)
        return "\n".join(parts)


class TensorPayloadMixin(BaseModel):
    tensor_shape: list[int] | None = None
    tensor_values: list[float] | None = None
    media_uri: str | None = None

    @model_validator(mode="after")
    def _ensure_payload(self) -> TensorPayloadMixin:
        has_tensor_shape = self.tensor_shape is not None
        has_tensor_values = self.tensor_values is not None
        if has_tensor_shape != has_tensor_values:
            raise ValueError("tensor_shape and tensor_values must be provided together")
        if self.media_uri is not None and has_tensor_shape:
            raise ValueError("Provide either media_uri or tensor_shape + tensor_values, not both")
        return self

    def has_raw_payload(self) -> bool:
        return self.media_uri is not None or self.tensor_shape is not None

    def load_tensor(self) -> torch.Tensor:
        if self.tensor_shape and self.tensor_values is not None:
            array = np.asarray(self.tensor_values, dtype=np.float32).reshape(self.tensor_shape)
            return torch.from_numpy(array)

        if self.media_uri is None:
            raise ValueError("No raw tensor payload available")

        assert self.media_uri is not None
        path = Path(self.media_uri)
        if path.suffix == ".npy":
            return torch.from_numpy(np.load(path).astype(np.float32))
        if path.suffix in {".pt", ".pth"}:
            tensor = torch.load(path, map_location="cpu")
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected tensor at {path}")
            return tensor.float()
        raise ValueError(f"Unsupported media_uri format: {path.suffix}")


class ReferenceState(TensorPayloadMixin):
    model_config = ConfigDict(extra="forbid")

    state_id: str
    equipment_family: str = "electrical_panel_family_a"
    state_label: str
    description: str
    allowed_variance_notes: str | None = None
    media_type: MediaType
    caption: str
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_reference_payload(self) -> ReferenceState:
        if not self.has_raw_payload():
            raise ValueError("ReferenceState requires media_uri or tensor_shape + tensor_values")
        return self


class VisualObservation(TensorPayloadMixin):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    equipment_family: str = "electrical_panel_family_a"
    panel_id: str | None = None
    media_type: MediaType
    embedding_values: list[float] | None = None
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _ensure_single_payload_source(self) -> VisualObservation:
        has_embedding = self.embedding_values is not None
        has_raw = self.has_raw_payload()
        if has_embedding == has_raw:
            raise ValueError(
                "VisualObservation requires exactly one of embedding_values or "
                "media_uri / tensor_shape + tensor_values"
            )
        return self

    def has_precomputed_embedding(self) -> bool:
        return self.embedding_values is not None

    def load_embedding(self) -> np.ndarray:
        if self.embedding_values is None:
            raise ValueError("No precomputed embedding is available")
        return np.asarray(self.embedding_values, dtype=np.float32)

    def context_text(self) -> str:
        values = [f"{key}:{value}" for key, value in sorted(self.metadata.items())]
        if self.panel_id:
            values.append(f"panel_id:{self.panel_id}")
        return "\n".join(values)


class TriageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: VisualObservation
    question: str | None = None
    operator_context: str | None = None
    expected_state_label: str | None = None


class IssueCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_class: str
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class NextStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = Field(default_factory=list)


class SimilarIncident(BaseModel):
    model_config = ConfigDict(extra="forbid")

    incident_id: str
    title: str
    issue_class: str
    fix_summary: str
    similarity: float = Field(ge=0.0, le=1.0)


class StateAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    matched_state_id: str | None = None
    matched_state_label: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    matches_expected: bool | None = None
    summary: str


class TriageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue_candidates: list[IssueCandidate]
    state_assessment: StateAssessment
    next_steps: list[NextStep]
    similar_incidents: list[SimilarIncident]
    escalation_recommendation: str
    evidence_citations: list[Citation]


class MediaAssetRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    asset_type: AssetType
    filename: str
    content_type: str
    byte_size: int
    sha256: str
    object_uri: str
    created_at: datetime
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class TriageAuditRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit_id: str
    request_id: str
    observation_id: str
    principal: str
    trace_id: str | None = None
    request_payload: dict[str, Any]
    response_payload: dict[str, Any]
    linked_asset_ids: list[str] = Field(default_factory=list)
    outcome_status: str
    created_at: datetime
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class MediaAssetView(MediaAssetRecord):
    presigned_url: str | None = None


class TriageAuditDetail(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audit: TriageAuditRecord
    linked_assets: list[MediaAssetView] = Field(default_factory=list)
