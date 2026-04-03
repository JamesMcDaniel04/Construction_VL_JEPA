"""FastAPI app for the maintenance triage copilot."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from maintenance_triage_copilot.api.routes import corpus, reference_states, system, triage
from maintenance_triage_copilot.config import AppConfig, load_config
from maintenance_triage_copilot.encoding.text import MaintenanceTextEncoder
from maintenance_triage_copilot.models.adapter import VisualTextProjector
from maintenance_triage_copilot.models.backbones import IJEPAImageAdapter, VJEPAVideoAdapter
from maintenance_triage_copilot.retrieval.index import VectorIndex
from maintenance_triage_copilot.services.triage import AppState, TriageService
from maintenance_triage_copilot.storage.memory import MemoryMetadataStore
from maintenance_triage_copilot.storage.protocol import MetadataStore
from maintenance_triage_copilot.storage.sql import SqlAlchemyMetadataStore
from maintenance_triage_copilot.utils.logging import setup_logging


def create_app(config_path: str | AppConfig | None = None) -> FastAPI:
    cfg = config_path if isinstance(config_path, AppConfig) else load_config(config_path)
    app = FastAPI(
        title="Maintenance Triage Copilot",
        description="Vision-language maintenance triage API for industrial electrical panels.",
        version="0.1.0",
    )
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
    metadata_store: MetadataStore
    if cfg.database.postgres_url:
        metadata_store = SqlAlchemyMetadataStore(cfg.database.postgres_url)
    else:
        metadata_store = MemoryMetadataStore()
    state = AppState(
        config=cfg,
        text_encoder=text_encoder,
        image_backbone=image_backbone,
        video_backbone=video_backbone,
        projector=projector,
        vector_index=VectorIndex(
            qdrant_url=cfg.database.qdrant_url,
            collection_prefix=cfg.database.collection_prefix,
        ),
        metadata_store=metadata_store,
    )
    app.state.service = TriageService(state)

    app.include_router(corpus.router)
    app.include_router(reference_states.router)
    app.include_router(system.router)
    app.include_router(triage.router)
    return app


def run() -> None:
    setup_logging()
    cfg = load_config()
    app = create_app()
    uvicorn.run(app, host=cfg.api.host, port=cfg.api.port)


if __name__ == "__main__":
    run()
