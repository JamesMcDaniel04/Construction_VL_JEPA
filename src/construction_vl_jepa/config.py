"""Centralized configuration loaded from YAML with environment variable overrides."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    latent_dim: int = 256
    context_window: int = 1800
    target_window: int = 600
    predictor_depth: int = 6
    encoder_depth: int = 12
    n_heads: int = 8
    dropout: float = 0.1
    masking_ratio: float = 0.75


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1.5e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    total_epochs: int = 300
    ema_decay_start: float = 0.996
    ema_decay_end: float = 0.999
    gradient_clip: float = 1.0
    accumulate_grad_batches: int = 1
    precision: str = "16-mixed"


@dataclass
class DataConfig:
    sensor_patch_size: int = 60
    max_sensors: int = 64
    normalization: str = "per_sensor_zscore"
    train_split: str = "normal_periods_only"
    val_fraction: float = 0.1
    failure_exclusion_hours: int = 6


@dataclass
class InferenceConfig:
    scoring_interval: int = 60
    threshold_k: float = 3.5
    trend_window: int = 72
    trend_slope_threshold: float = 0.01
    alert_cooldown: int = 3600
    pattern_match_top_k: int = 5
    pattern_similarity_threshold: float = 0.80


@dataclass
class DatabaseConfig:
    timescale_url: str = "postgresql://pfil:pfil@localhost:5432/pfil"
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "latent_states"


@dataclass
class KafkaConfig:
    bootstrap_servers: str = "localhost:9092"
    sensor_topic: str = "sensor-readings"
    anomaly_topic: str = "anomaly-scores"
    consumer_group: str = "pfil-inference"


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass
class AlertConfig:
    webhook_url: str | None = None
    email_to: str | None = None
    slack_webhook: str | None = None
    pagerduty_key: str | None = None


@dataclass
class MlflowConfig:
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "pfil-jepa"


@dataclass
class PFILConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    mlflow: MlflowConfig = field(default_factory=MlflowConfig)


def _apply_dict(dc: object, d: dict) -> None:
    for k, v in d.items():
        if hasattr(dc, k):
            attr = getattr(dc, k)
            if isinstance(v, dict) and hasattr(attr, "__dataclass_fields__"):
                _apply_dict(attr, v)
            else:
                setattr(dc, k, v)


def load_config(path: str | Path | None = None) -> PFILConfig:
    """Load configuration from YAML file, falling back to defaults."""
    cfg = PFILConfig()
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    path = Path(path)
    if path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        _apply_dict(cfg, raw)
    return cfg
