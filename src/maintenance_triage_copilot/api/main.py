"""FastAPI app for the maintenance triage copilot."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from maintenance_triage_copilot.api.middleware import ApiKeyMiddleware, RequestMetricsMiddleware
from maintenance_triage_copilot.api.routes import corpus, media, reference_states, system, triage
from maintenance_triage_copilot.config import AppConfig, load_config
from maintenance_triage_copilot.encoding.text import MaintenanceTextEncoder
from maintenance_triage_copilot.models.adapter import VisualTextProjector
from maintenance_triage_copilot.models.backbones import IJEPAImageAdapter, VJEPAVideoAdapter
from maintenance_triage_copilot.models.policy import CalibratedTriagePolicy
from maintenance_triage_copilot.retrieval.index import VectorIndex
from maintenance_triage_copilot.services.triage import AppState, TriageService
from maintenance_triage_copilot.storage.memory import MemoryMetadataStore
from maintenance_triage_copilot.storage.protocol import MetadataStore
from maintenance_triage_copilot.storage.sql import SqlAlchemyMetadataStore
from maintenance_triage_copilot.utils.logging import setup_logging


def create_app(config_path: str | AppConfig | None = None) -> FastAPI:
    cfg = config_path if isinstance(config_path, AppConfig) else load_config(config_path)
    is_production = cfg.runtime.is_production()
    if is_production and (
        cfg.database.postgres_url is None
        or not cfg.database.postgres_url.startswith("postgresql")
    ):
        raise RuntimeError("Production mode requires a PostgreSQL database URL")
    if is_production and not cfg.database.qdrant_url:
        raise RuntimeError("Qdrant is required in production mode")
    app = FastAPI(
        title="Maintenance Triage Copilot",
        description="Vision-language maintenance triage API for industrial electrical panels.",
        version="0.1.0",
    )

    # Middleware stack (order matters — outermost first)
    app.add_middleware(RequestMetricsMiddleware)
    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    text_encoder = MaintenanceTextEncoder(cfg.text_encoder)
    image_backbone = IJEPAImageAdapter(cfg.image_backbone)
    video_backbone = VJEPAVideoAdapter(cfg.video_backbone)
    projector = VisualTextProjector(
        input_dim=image_backbone.embedding_dim,
        hidden_dim=cfg.adapter.hidden_dim,
        output_dim=cfg.adapter.output_dim,
    )

    projector_checkpoint = cfg.adapter.checkpoint_path
    projector_checkpoint_loaded = False
    if projector_checkpoint is not None:
        from pathlib import Path

        ckpt_path = Path(projector_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Projector checkpoint not found: {ckpt_path}")
        import torch

        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            projector.load_state_dict(state_dict)
        except Exception as exc:  # pragma: no cover - exact torch error varies by version
            raise RuntimeError(f"Failed to load projector checkpoint: {ckpt_path}") from exc
        projector_checkpoint_loaded = True
    projector.eval()

    policy_checkpoint = cfg.policy.checkpoint_path
    policy_checkpoint_loaded = False
    if policy_checkpoint is not None:
        from pathlib import Path

        policy_path = Path(policy_checkpoint)
        if not policy_path.exists():
            raise FileNotFoundError(f"Triage policy checkpoint not found: {policy_path}")
        triage_policy = CalibratedTriagePolicy.from_file(policy_path)
        policy_checkpoint_loaded = True
    elif cfg.policy.require_checkpoint:
        raise RuntimeError("Production mode requires a calibrated triage policy checkpoint")
    else:
        triage_policy = CalibratedTriagePolicy.bootstrap()

    metadata_store: MetadataStore
    if cfg.database.postgres_url:
        metadata_store = SqlAlchemyMetadataStore(
            cfg.database.postgres_url,
            run_schema_migrations=cfg.database.run_migrations_on_startup,
            required=is_production,
        )
    else:
        if is_production:
            raise RuntimeError("Postgres is required in production mode")
        metadata_store = MemoryMetadataStore()

    state = AppState(
        config=cfg,
        text_encoder=text_encoder,
        image_backbone=image_backbone,
        video_backbone=video_backbone,
        projector=projector,
        triage_policy=triage_policy,
        vector_index=VectorIndex(
            qdrant_url=cfg.database.qdrant_url,
            collection_prefix=cfg.database.collection_prefix,
            required=is_production,
        ),
        metadata_store=metadata_store,
        projector_checkpoint_path=projector_checkpoint,
        projector_checkpoint_loaded=projector_checkpoint_loaded,
        policy_checkpoint_path=policy_checkpoint,
        policy_checkpoint_loaded=policy_checkpoint_loaded,
    )
    app.state.service = TriageService(state)

    app.include_router(corpus.router)
    app.include_router(media.router)
    app.include_router(reference_states.router)
    app.include_router(system.router)
    app.include_router(triage.router)
    return app


def run() -> None:
    setup_logging()
    cfg = load_config()
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)


if __name__ == "__main__":
    run()
