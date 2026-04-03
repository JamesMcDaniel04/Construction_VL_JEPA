# Maintenance Triage Copilot

API-first vision-language maintenance triage copilot for one electrical-panel family.

## MVP

- Photos use vendored Meta I-JEPA model slices.
- Short clips use vendored Meta V-JEPA model slices.
- Manuals, SOPs, and prior tickets are chunked and indexed for grounded retrieval.
- Triage responses return structured issue candidates, panel-state assessment, next steps, similar incidents, and escalation guidance.
- Metadata persists through SQLAlchemy to the configured database URL instead of an in-memory default.
- The text encoder requires a real sentence-transformer model in normal app runs. Tests use an explicit `mock` backend instead of a silent fallback.
- The supported training path in this repo is `mtc-train-adapter` for the visual-text projector. Self-supervised JEPA pretraining is not part of the supported runtime.

## Local Development

```bash
PYENV_VERSION=3.11.9 python -m pip install -e ".[dev]"
PYENV_VERSION=3.11.9 python -m pytest -q
PYENV_VERSION=3.11.9 mtc-api
```

## Endpoints

- `POST /corpus/documents`
- `POST /corpus/incidents`
- `POST /corpus/upload`
- `POST /media/triage`
- `POST /media/encode`
- `POST /reference-states`
- `POST /triage/analyze`
- `GET /system/health`

## Production Models

- Mount model assets into `/models`.
- Set the sentence-transformer directory at `/models/text-encoder`.
- Set the projector checkpoint at `/models/projector.pt`.
- Optional backbone checkpoints can also be mounted under `/models`.
