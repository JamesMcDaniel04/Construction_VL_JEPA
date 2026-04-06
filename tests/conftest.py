from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from maintenance_triage_copilot.api.main import create_app
from maintenance_triage_copilot.auth.supabase import SupabaseIdentity, SupabaseInviteResult
from maintenance_triage_copilot.config import (
    AdapterConfig,
    AppConfig,
    DatabaseConfig,
    ImageBackboneConfig,
    PolicyConfig,
    RetrievalConfig,
    RuntimeConfig,
    SecurityConfig,
    SupabaseConfig,
    TextEncoderConfig,
    TriageConfig,
    VideoBackboneConfig,
)
from maintenance_triage_copilot.domain.models import (
    AssetCreateRequest,
    SiteCreateRequest,
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
    torch.manual_seed(0)
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


class FakeSupabaseAuthProvider:
    def __init__(self) -> None:
        self._claims_by_token = {
            "supabase-tech-token": SupabaseIdentity(
                user_id="supabase-tech-1",
                email="alex@example.com",
                claims={"sub": "supabase-tech-1", "email": "alex@example.com"},
            ),
            "supabase-admin-token": SupabaseIdentity(
                user_id="supabase-admin-1",
                email="sam@example.com",
                claims={"sub": "supabase-admin-1", "email": "sam@example.com"},
            ),
        }

    def configured_for_human_auth(self) -> bool:
        return True

    def configured_for_invites(self) -> bool:
        return True

    def verify_access_token(self, token: str) -> SupabaseIdentity:
        if token not in self._claims_by_token:
            raise ValueError("invalid token")
        return self._claims_by_token[token]

    def invite_user(
        self,
        *,
        email: str,
        display_name: str,
        redirect_to: str | None = None,
    ) -> SupabaseInviteResult:
        normalized_email = email.strip().lower()
        local_part = normalized_email.split("@", 1)[0].replace(".", "-")
        return SupabaseInviteResult(
            user_id=f"supabase-{local_part}",
            email=normalized_email,
            invite_status="sent",
        )


def _seed_catalog(service: TriageService) -> None:
    sites = [
        SiteCreateRequest(site_id="site-a", name="Line A", code="A1"),
        SiteCreateRequest(site_id="site-b", name="Line B", code="B1"),
    ]
    assets = [
        AssetCreateRequest(
            asset_id="panel-42",
            site_id="site-a",
            display_name="Main breaker panel 42",
            panel_family="family-a",
            equipment_family="electrical_panel_family_a",
            panel_id="panel-42a",
        ),
        AssetCreateRequest(
            asset_id="panel-77",
            site_id="site-b",
            display_name="Secondary panel 77",
            panel_family="family-a",
            equipment_family="electrical_panel_family_a",
            panel_id="panel-77",
        ),
    ]
    for site in sites:
        service.add_site(site, organization_id="org-1")
    for asset in assets:
        service.add_asset_catalog_record(asset, organization_id="org-1")


def _wire_test_auth(app) -> None:
    fake_auth = FakeSupabaseAuthProvider()
    app.state.supabase_auth = fake_auth
    app.state.service.state.supabase_auth = fake_auth
    app.state.human_token_verifier = fake_auth.verify_access_token
    app.state.service.state.auth_mode = "hybrid_bearer"
    _seed_catalog(app.state.service)


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
        security=SecurityConfig(
            service_tokens={"test-client": "secret-token"},
            pilot_users=[
                {
                    "user_id": "supabase-tech-1",
                    "organization_id": "org-1",
                    "role": "technician",
                    "display_name": "Alex Technician",
                    "email": "alex@example.com",
                },
                {
                    "user_id": "supabase-admin-1",
                    "organization_id": "org-1",
                    "role": "admin",
                    "display_name": "Sam Supervisor",
                    "email": "sam@example.com",
                },
            ],
        ),
        supabase=SupabaseConfig(
            project_url="https://supabase.test",
            anon_key="anon-test",
            service_role_key="service-role-test",
            jwt_issuer="https://supabase.test/auth/v1",
            jwt_audience="authenticated",
            mobile_redirect_scheme="mtc://auth/callback",
            web_redirect_url="http://localhost:5173/auth/callback",
        ),
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
    service = TriageService(state)
    _seed_catalog(service)
    return service


@pytest.fixture
def client(small_config: AppConfig) -> TestClient:
    app = create_app(small_config)
    _wire_test_auth(app)
    client = TestClient(app)
    client.headers.update({"Authorization": "Bearer secret-token"})
    return client


@pytest.fixture
def technician_client(small_config: AppConfig) -> TestClient:
    app = create_app(small_config)
    _wire_test_auth(app)
    client = TestClient(app)
    client.headers.update({"Authorization": "Bearer supabase-tech-token"})
    return client


@pytest.fixture
def admin_client(small_config: AppConfig) -> TestClient:
    app = create_app(small_config)
    _wire_test_auth(app)
    client = TestClient(app)
    client.headers.update({"Authorization": "Bearer supabase-admin-token"})
    return client


def image_values(fill: float) -> list[float]:
    tensor = np.full((3, 16, 16), fill_value=fill, dtype=np.float32)
    return tensor.reshape(-1).tolist()


def video_values(fill: float) -> list[float]:
    tensor = np.full((3, 10, 16, 16), fill_value=fill, dtype=np.float32)
    return tensor.reshape(-1).tolist()
