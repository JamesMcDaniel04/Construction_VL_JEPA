# Architecture

The repo now targets a maintenance triage copilot for electrical panels.

## Core Flow

1. Ingest manuals, SOPs, and prior incidents into a vector-backed retrieval layer.
2. Encode panel photos with the vendored I-JEPA backbone and short clips with the vendored V-JEPA backbone.
3. Compare observations against curated reference states for panel-state assessment.
4. Fuse technician question text at retrieval time and return grounded issue candidates, next steps, similar incidents, and escalation guidance.

## Persistence

- Metadata uses SQLAlchemy against the configured database URL.
- Vector search uses Qdrant when configured and falls back to in-process vectors only when Qdrant is unavailable.

## Backbones

- Image backbone source: `facebookresearch/ijepa` commit `52c1ae95d05f743e000e8f10a1f3a79b10cff048`
- Video backbone source: `facebookresearch/jepa` commit `51c59d518fc63c08464af6de585f78ac0c7ed4d5`

Only the minimal model slices needed for frozen feature extraction are vendored.
