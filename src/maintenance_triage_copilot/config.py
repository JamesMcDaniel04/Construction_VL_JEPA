"""Configuration for the maintenance triage copilot."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml  # type: ignore[import-untyped]


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class RuntimeConfig:
    mode: str = "development"
    model_dir: str = "/models"
    allow_smoke_assets: bool = False
    max_image_upload_bytes: int = 5 * 1024 * 1024
    max_video_upload_bytes: int = 50 * 1024 * 1024
    max_document_upload_bytes: int = 20 * 1024 * 1024

    def is_production(self) -> bool:
        return self.mode.lower() == "production"


@dataclass
class DatabaseConfig:
    postgres_url: str | None = "postgresql://mtc:mtc@localhost:5432/mtc"
    qdrant_url: str | None = "http://localhost:6333"
    collection_prefix: str = "maintenance-triage"
    run_migrations_on_startup: bool = True


@dataclass
class RetrievalConfig:
    top_k_documents: int = 5
    top_k_incidents: int = 3
    top_k_states: int = 3
    chunk_size: int = 320
    chunk_overlap: int = 40


@dataclass
class TriageConfig:
    top_k_steps: int = 3
    state_match_threshold: float = 0.82
    escalation_threshold: float = 0.55
    video_num_frames: int = 8


@dataclass
class PolicyConfig:
    checkpoint_path: str | None = None
    require_checkpoint: bool = False
    top_k_issues: int = 3


@dataclass
class TextEncoderConfig:
    backend: str = "sentence-transformer"
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    local_files_only: bool = False
    cache_folder: str | None = None


@dataclass
class AdapterConfig:
    hidden_dim: int = 384
    output_dim: int = 384
    checkpoint_path: str | None = None


@dataclass
class SecurityConfig:
    service_tokens: dict[str, str] = field(default_factory=dict)
    require_auth_in_production: bool = True


@dataclass
class TelemetryConfig:
    prometheus_enabled: bool = True
    otlp_endpoint: str | None = None
    service_name: str = "maintenance-triage-copilot"


@dataclass
class ObjectStoreConfig:
    endpoint_url: str | None = None
    bucket: str | None = None
    region: str | None = None
    access_key: str | None = None
    secret_key: str | None = None
    force_path_style: bool = True
    required_in_production: bool = True
    create_bucket_if_missing: bool = True


@dataclass
class ImageBackboneConfig:
    preset: str | None = None
    require_checkpoint: bool = False
    use_timm: bool = False
    timm_model_name: str = "vit_tiny_patch16_224"
    timm_pretrained: bool = True
    checkpoint_path: str | None = None
    input_size: int = 64
    patch_size: int = 8
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 3


@dataclass
class VideoBackboneConfig:
    preset: str | None = None
    require_checkpoint: bool = False
    use_timm: bool = False
    timm_model_name: str = "vit_tiny_patch16_224"
    timm_pretrained: bool = True
    checkpoint_path: str | None = None
    input_size: int = 64
    patch_size: int = 8
    num_frames: int = 8
    tubelet_size: int = 2
    embed_dim: int = 192
    depth: int = 4
    num_heads: int = 3


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    triage: TriageConfig = field(default_factory=TriageConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    object_store: ObjectStoreConfig = field(default_factory=ObjectStoreConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    image_backbone: ImageBackboneConfig = field(default_factory=ImageBackboneConfig)
    video_backbone: VideoBackboneConfig = field(default_factory=VideoBackboneConfig)


def _apply_dict(target: object, values: dict) -> None:
    for key, value in values.items():
        if not hasattr(target, key):
            continue
        current = getattr(target, key)
        if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
            _apply_dict(current, value)
        else:
            setattr(target, key, value)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML config and merge onto defaults."""
    cfg = AppConfig()
    candidate = path or os.getenv("MTC_CONFIG")
    if candidate is None:
        candidate = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    config_path = Path(candidate)
    if config_path.exists():
        with config_path.open() as handle:
            raw = yaml.safe_load(handle) or {}
        _apply_dict(cfg, raw)
    _apply_env_overrides(cfg)
    return cfg


def _apply_env_overrides(cfg: AppConfig) -> None:
    tokens_json = os.getenv("MTC_SERVICE_TOKENS_JSON")
    if tokens_json:
        cfg.security.service_tokens = {
            str(key): str(value) for key, value in json.loads(tokens_json).items()
        }

    access_key = os.getenv("MTC_OBJECT_STORE_ACCESS_KEY")
    if access_key:
        cfg.object_store.access_key = access_key
    secret_key = os.getenv("MTC_OBJECT_STORE_SECRET_KEY")
    if secret_key:
        cfg.object_store.secret_key = secret_key
    endpoint_url = os.getenv("MTC_OBJECT_STORE_ENDPOINT_URL")
    if endpoint_url:
        cfg.object_store.endpoint_url = endpoint_url
    bucket = os.getenv("MTC_OBJECT_STORE_BUCKET")
    if bucket:
        cfg.object_store.bucket = bucket
    otlp_endpoint = os.getenv("MTC_OTLP_ENDPOINT")
    if otlp_endpoint:
        cfg.telemetry.otlp_endpoint = otlp_endpoint
    if os.getenv("MTC_ALLOW_SMOKE_ASSETS") == "1":
        cfg.runtime.allow_smoke_assets = True
