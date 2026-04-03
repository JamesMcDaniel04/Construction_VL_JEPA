from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from maintenance_triage_copilot.api.main import create_app
from maintenance_triage_copilot.config import (
    AdapterConfig,
    AppConfig,
    ImageBackboneConfig,
    RetrievalConfig,
    TextEncoderConfig,
    TriageConfig,
    VideoBackboneConfig,
)
from maintenance_triage_copilot.encoding.text import MaintenanceTextEncoder
from maintenance_triage_copilot.models.adapter import VisualTextProjector
from maintenance_triage_copilot.models.backbones import IJEPAImageAdapter, VJEPAVideoAdapter
from maintenance_triage_copilot.retrieval.index import VectorIndex
from maintenance_triage_copilot.services.triage import AppState, TriageService
from maintenance_triage_copilot.storage.memory import MemoryMetadataStore


@pytest.fixture
def small_config() -> AppConfig:
    return AppConfig(
        retrieval=RetrievalConfig(
            top_k_documents=4,
            top_k_incidents=3,
            top_k_states=2,
            chunk_size=120,
            chunk_overlap=20,
        ),
        triage=TriageConfig(
            top_k_steps=3,
            state_match_threshold=0.8,
            escalation_threshold=0.5,
            video_num_frames=8,
        ),
        text_encoder=TextEncoderConfig(backend="hash", embedding_dim=192),
        adapter=AdapterConfig(hidden_dim=192, output_dim=192),
        image_backbone=ImageBackboneConfig(
            input_size=32,
            patch_size=8,
            embed_dim=192,
            depth=2,
            num_heads=3,
        ),
        video_backbone=VideoBackboneConfig(
            input_size=32,
            patch_size=8,
            num_frames=8,
            tubelet_size=2,
            embed_dim=192,
            depth=2,
            num_heads=3,
        ),
    )


@pytest.fixture
def triage_service(small_config: AppConfig) -> TriageService:
    text_encoder = MaintenanceTextEncoder(small_config.text_encoder)
    image_backbone = IJEPAImageAdapter(small_config.image_backbone)
    video_backbone = VJEPAVideoAdapter(small_config.video_backbone)
    projector = VisualTextProjector(
        input_dim=image_backbone.embedding_dim,
        hidden_dim=small_config.adapter.hidden_dim,
        output_dim=small_config.adapter.output_dim,
    )
    state = AppState(
        config=small_config,
        text_encoder=text_encoder,
        image_backbone=image_backbone,
        video_backbone=video_backbone,
        projector=projector,
        vector_index=VectorIndex(),
        metadata_store=MemoryMetadataStore(),
    )
    return TriageService(state)


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def image_values(fill: float) -> list[float]:
    tensor = np.full((3, 16, 16), fill_value=fill, dtype=np.float32)
    return tensor.reshape(-1).tolist()


def video_values(fill: float) -> list[float]:
    tensor = np.full((3, 10, 16, 16), fill_value=fill, dtype=np.float32)
    return tensor.reshape(-1).tolist()
