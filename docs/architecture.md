# Architecture

The repo now targets a maintenance triage copilot for electrical panels.

## Core Flow

1. Ingest manuals, SOPs, and prior incidents into a vector-backed retrieval layer.
2. Encode panel photos with vendored Meta I-JEPA slices and short clips with vendored Meta V-JEPA slices.
3. Compare observations against curated reference states for panel-state assessment.
4. Fuse technician question text at retrieval time and return grounded issue candidates, next steps, similar incidents, and escalation guidance.

## Persistence

- Metadata uses SQLAlchemy against the configured database URL.
- Production mode requires PostgreSQL plus Qdrant and fails fast if either dependency is unavailable.
- Development mode can still use SQLite metadata and in-process vector fallback.
- Alembic owns schema creation and upgrades through `mtc-db-upgrade`.

## Backbones

- Image backbone source: `facebookresearch/ijepa` commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`
- Video backbone source: `facebookresearch/jepa` commit `51c59d518fc63c08464af6de585f78ac0c7ed4d5`

Only the minimal model slices needed for frozen feature extraction are vendored.

## Supported Training

- Supported training entrypoint: `mtc-train-adapter`
- Supported calibration entrypoint: `mtc-train-policy`
- Unsupported in this repo: removed third-party JEPA pretraining code and self-supervised JEPA pretraining flows

## Ingestion

- `POST /corpus/upload` supports plain text, Markdown, PDFs, and OCR-backed image documents.
- PDF ingestion first extracts embedded text and falls back to OCR when the document is sparse or scanned.

## Deployment

- Production containers mount `/models` read-only for the sentence-transformer directory, projector checkpoint, and calibrated triage-policy checkpoint.
- `tests/integration/test_docker_compose_smoke.py` exercises the compose stack with mounted model assets, Postgres, Qdrant, and the live API.
