"""FastAPI app for the maintenance triage copilot."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from maintenance_triage_copilot.api.middleware import (
    BearerAuthMiddleware,
    RequestContextMiddleware,
)
from maintenance_triage_copilot.api.routes import (
    audit,
    auth,
    cases,
    catalog,
    corpus,
    dashboard,
    media,
    metrics,
    pilot_users,
    reference_states,
    system,
    triage,
)
from maintenance_triage_copilot.auth.supabase import SupabaseAuthProvider
from maintenance_triage_copilot.config import AppConfig, load_config
from maintenance_triage_copilot.domain.models import PilotUserSeed
from maintenance_triage_copilot.encoding.text import MaintenanceTextEncoder
from maintenance_triage_copilot.models.adapter import VisualTextProjector
from maintenance_triage_copilot.models.assets import validate_model_assets
from maintenance_triage_copilot.models.backbones import (
    IJEPAImageAdapter,
    VJEPAVideoAdapter,
)
from maintenance_triage_copilot.models.policy import CalibratedTriagePolicy
from maintenance_triage_copilot.retrieval.index import VectorIndex
from maintenance_triage_copilot.services.triage import AppState, TriageService
from maintenance_triage_copilot.storage.memory import MemoryMetadataStore
from maintenance_triage_copilot.storage.object_store import (
    MemoryObjectStore,
    ObjectStore,
    S3ObjectStore,
)
from maintenance_triage_copilot.storage.protocol import MetadataStore
from maintenance_triage_copilot.storage.sql import SqlAlchemyMetadataStore
from maintenance_triage_copilot.telemetry import configure_telemetry
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
    pilot_user_seeds = [PilotUserSeed.model_validate(item) for item in cfg.security.pilot_users]
    supabase_auth = (
        SupabaseAuthProvider(cfg.supabase) if cfg.supabase.project_url is not None else None
    )
    if cfg.policy.require_checkpoint and cfg.policy.checkpoint_path is None:
        raise RuntimeError("Production mode requires a calibrated triage policy checkpoint")
    assets = validate_model_assets(cfg)
    app = FastAPI(
        title="Maintenance Triage Copilot",
        description="Vision-language maintenance triage API for industrial electrical panels.",
        version="0.1.0",
    )

    # Starlette wraps middleware in reverse registration order. RequestContext
    # must be outermost so auth failures still receive request IDs, logs, and metrics.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(BearerAuthMiddleware)
    app.add_middleware(RequestContextMiddleware)

    text_encoder = MaintenanceTextEncoder(cfg.text_encoder)
    image_backbone = IJEPAImageAdapter(cfg.image_backbone, runtime_spec=assets.image_backbone)
    video_backbone = VJEPAVideoAdapter(cfg.video_backbone, runtime_spec=assets.video_backbone)
    projector = VisualTextProjector(
        input_dim=image_backbone.embedding_dim,
        hidden_dim=cfg.adapter.hidden_dim,
        output_dim=cfg.adapter.output_dim,
    )

    projector_checkpoint = assets.projector_checkpoint_path or cfg.adapter.checkpoint_path
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

    policy_checkpoint = assets.policy_checkpoint_path or cfg.policy.checkpoint_path
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

    object_store: ObjectStore
    if cfg.object_store.endpoint_url and cfg.object_store.bucket:
        object_store = S3ObjectStore(cfg.object_store, required=is_production)
    else:
        if is_production and cfg.object_store.required_in_production:
            raise RuntimeError("Production mode requires S3-compatible object storage")
        object_store = MemoryObjectStore()

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
        object_store=object_store,
        asset_status=assets.status,
        supabase_auth=supabase_auth,
        auth_mode="disabled",
        telemetry_mode=(
            "prometheus+otlp"
            if cfg.telemetry.prometheus_enabled and cfg.telemetry.otlp_endpoint
            else "prometheus"
            if cfg.telemetry.prometheus_enabled
            else "disabled"
        ),
        projector_checkpoint_path=projector_checkpoint,
        projector_checkpoint_loaded=projector_checkpoint_loaded,
        policy_checkpoint_path=policy_checkpoint,
        policy_checkpoint_loaded=policy_checkpoint_loaded,
    )
    app.state.service = TriageService(state)
    for seed in pilot_user_seeds:
        app.state.service.seed_pilot_user(seed)

    def refresh_pilot_user_lookup() -> dict[str, object]:
        users = app.state.service.list_pilot_users(limit=5000, offset=0)
        lookup = {user.user_id: user for user in users if user.active}
        app.state.pilot_user_lookup = lookup
        return lookup

    pilot_lookup = refresh_pilot_user_lookup()
    human_auth_enabled = bool(
        supabase_auth is not None and supabase_auth.configured_for_human_auth()
    )
    auth_required = bool(cfg.security.service_tokens or pilot_lookup or human_auth_enabled) or (
        is_production and cfg.security.require_auth_in_production
    )
    if (
        is_production
        and cfg.security.require_auth_in_production
        and not cfg.security.service_tokens
        and not human_auth_enabled
    ):
        raise RuntimeError("Production mode requires service tokens or Supabase human auth")
    app.state.service_token_lookup = {
        token: principal for principal, token in cfg.security.service_tokens.items()
    }
    app.state.supabase_auth = supabase_auth
    app.state.refresh_pilot_user_lookup = refresh_pilot_user_lookup
    app.state.auth_required = auth_required
    if auth_required:
        if human_auth_enabled and cfg.security.service_tokens:
            app.state.service.state.auth_mode = "hybrid_bearer"
        elif human_auth_enabled:
            app.state.service.state.auth_mode = "supabase_bearer"
        else:
            app.state.service.state.auth_mode = "service_bearer"
    else:
        app.state.service.state.auth_mode = "disabled"
    configure_telemetry(
        cfg=cfg.telemetry,
        app=app,
        engine=getattr(metadata_store, "engine", None),
    )

    app.include_router(auth.router)
    app.include_router(audit.router)
    app.include_router(cases.router)
    app.include_router(catalog.router)
    app.include_router(corpus.router)
    app.include_router(dashboard.router)
    app.include_router(media.router)
    app.include_router(metrics.router)
    app.include_router(pilot_users.router)
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
