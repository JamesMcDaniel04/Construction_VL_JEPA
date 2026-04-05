from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from maintenance_triage_copilot.api.main import create_app
from maintenance_triage_copilot.config import (
    AdapterConfig,
    AppConfig,
    DatabaseConfig,
    ImageBackboneConfig,
    PolicyConfig,
    RetrievalConfig,
    RuntimeConfig,
    SecurityConfig,
    TextEncoderConfig,
    TriageConfig,
    VideoBackboneConfig,
)
from maintenance_triage_copilot.encoding.text import MaintenanceTextEncoder
from maintenance_triage_copilot.models.adapter import VisualTextProjector
from maintenance_triage_copilot.models.assets import checkpoint_sha256, directory_sha256
from maintenance_triage_copilot.models.backbones import IJEPAImageAdapter, VJEPAVideoAdapter
from maintenance_triage_copilot.models.policy import CalibratedTriagePolicy
from maintenance_triage_copilot.retrieval.index import VectorIndex
from maintenance_triage_copilot.services.triage import AppState, TriageService
from maintenance_triage_copilot.storage.object_store import MemoryObjectStore
from maintenance_triage_copilot.storage.sql import SqlAlchemyMetadataStore
from maintenance_triage_copilot.vendor.meta_ijepa import VisionTransformer
from maintenance_triage_copilot.vendor.meta_vjepa import VideoVisionTransformer


def _prepare_model_assets(model_dir) -> None:
    text_dir = model_dir / "text-encoder"
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / "config.json").write_text(json.dumps({"backend": "mock"}))

    projector_path = model_dir / "projector.pt"
    policy_path = model_dir / "triage-policy.json"
    image_path = model_dir / "image_backbone.pt"
    video_path = model_dir / "video_backbone.pt"

    torch.save(VisualTextProjector(192, 192, 192).state_dict(), projector_path)
    CalibratedTriagePolicy.bootstrap().save(policy_path)
    torch.save(
        {
            "encoder": VisionTransformer(
                img_size=32,
                patch_size=8,
                embed_dim=192,
                depth=2,
                num_heads=3,
            ).state_dict()
        },
        image_path,
    )
    torch.save(
        {
            "target_encoder": VideoVisionTransformer(
                img_size=32,
                patch_size=8,
                num_frames=8,
                tubelet_size=2,
                embed_dim=192,
                depth=2,
                num_heads=3,
            ).state_dict()
        },
        video_path,
    )

    manifest = {
        "version": "test-1",
        "assets": {
            "text_encoder": {
                "kind": "directory",
                "path": "text-encoder",
                "sha256": directory_sha256(text_dir),
            },
            "projector": {
                "kind": "file",
                "path": "projector.pt",
                "sha256": checkpoint_sha256(projector_path),
            },
            "triage_policy": {
                "kind": "file",
                "path": "triage-policy.json",
                "sha256": checkpoint_sha256(policy_path),
            },
            "image_backbone": {
                "kind": "file",
                "path": "image_backbone.pt",
                "sha256": checkpoint_sha256(image_path),
                "preset": "ijepa_vith14_224",
                "embedding_dim": 192,
                "checkpoint_keys": ["target_encoder", "encoder"],
                "smoke_stub": {
                    "input_size": 32,
                    "patch_size": 8,
                    "embed_dim": 192,
                    "depth": 2,
                    "num_heads": 3,
                },
            },
            "video_backbone": {
                "kind": "file",
                "path": "video_backbone.pt",
                "sha256": checkpoint_sha256(video_path),
                "preset": "vjepa_vith16_224_2x16x16",
                "embedding_dim": 192,
                "checkpoint_keys": ["target_encoder", "encoder"],
                "smoke_stub": {
                    "input_size": 32,
                    "patch_size": 8,
                    "num_frames": 8,
                    "tubelet_size": 2,
                    "embed_dim": 192,
                    "depth": 2,
                    "num_heads": 3,
                },
            },
        },
    }
    (model_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


@pytest.fixture
def small_config(tmp_path) -> AppConfig:
    model_dir = tmp_path / "models"
    _prepare_model_assets(model_dir)
    return AppConfig(
        runtime=RuntimeConfig(
            mode="development",
            model_dir=str(model_dir),
            allow_smoke_assets=True,
        ),
        database=DatabaseConfig(
            postgres_url=f"sqlite+pysqlite:///{tmp_path / 'mtc-test.db'}",
            qdrant_url=None,
            collection_prefix="test-maintenance-triage",
        ),
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
        policy=PolicyConfig(top_k_issues=3),
        security=SecurityConfig(service_tokens={"test-client": "secret-token"}),
        text_encoder=TextEncoderConfig(backend="mock", embedding_dim=192),
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
        triage_policy=CalibratedTriagePolicy.bootstrap(),
        vector_index=VectorIndex(),
        metadata_store=SqlAlchemyMetadataStore(small_config.database.postgres_url or ""),
        object_store=MemoryObjectStore(),
        asset_status={},
    )
    return TriageService(state)


@pytest.fixture
def client(small_config: AppConfig) -> TestClient:
    app = create_app(small_config)
    client = TestClient(app)
    client.headers.update({"Authorization": "Bearer secret-token"})
    return client


def image_values(fill: float) -> list[float]:
    tensor = np.full((3, 16, 16), fill_value=fill, dtype=np.float32)
    return tensor.reshape(-1).tolist()


def video_values(fill: float) -> list[float]:
    tensor = np.full((3, 10, 16, 16), fill_value=fill, dtype=np.float32)
    return tensor.reshape(-1).tolist()
