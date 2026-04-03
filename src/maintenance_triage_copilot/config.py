"""Configuration for the maintenance triage copilot."""

from __future__ import annotations

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
class ImageBackboneConfig:
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
    return cfg
