"""FastAPI application — REST API for the Pre-Failure Intelligence Layer.

Provides endpoints for machine health, anomaly history, pattern library,
operator feedback, and system health monitoring.
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from construction_vl_jepa.api.routes import machines, system
from construction_vl_jepa.config import load_config
from construction_vl_jepa.utils.logging import setup_logging

app = FastAPI(
    title="Pre-Failure Intelligence Layer",
    description="VL-JEPA powered predictive maintenance API",
    version="0.1.0",
)


def create_app(config_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    cfg = load_config(config_path)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store config in app state for dependency injection
    app.state.config = cfg

    # Include route modules
    app.include_router(machines.router, prefix="/machines", tags=["machines"])
    app.include_router(system.router, prefix="/system", tags=["system"])

    return app


def run():
    """CLI entry point for starting the API server."""
    setup_logging()
    cfg = load_config()
    application = create_app()
    uvicorn.run(application, host=cfg.api.host, port=cfg.api.port)


if __name__ == "__main__":
    run()
