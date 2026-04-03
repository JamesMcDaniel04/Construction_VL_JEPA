# Maintenance Triage Copilot

API-first vision-language maintenance triage copilot for one electrical-panel family.

## MVP

- Photos use a vendored I-JEPA-style image backbone.
- Short clips use a vendored V-JEPA-style video backbone.
- Manuals, SOPs, and prior tickets are chunked and indexed for grounded retrieval.
- Triage responses return structured issue candidates, panel-state assessment, next steps, similar incidents, and escalation guidance.

## Local Development

```bash
PYENV_VERSION=3.11.9 python -m pip install -e ".[dev]"
PYENV_VERSION=3.11.9 python -m pytest -q
PYENV_VERSION=3.11.9 mtc-api
```

## Endpoints

- `POST /corpus/documents`
- `POST /corpus/incidents`
- `POST /reference-states`
- `POST /triage/analyze`
- `GET /system/health`
