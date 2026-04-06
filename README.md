# Maintenance Triage Copilot

Pilot-ready mobile field-tech troubleshooting product for one electrical-panel family.

## MVP

- Technician-facing Expo mobile app for capture, triage review, and feedback.
- Admin web console for corpus upload, reference-state management, and case review.
- Photos use vendored Meta I-JEPA model slices.
- Short clips use vendored Meta V-JEPA model slices.
- Manuals, SOPs, and prior tickets are chunked and indexed for grounded retrieval.
- OCR-backed ingestion supports scanned PDFs and uploaded image documents.
- Triage responses return structured likely issue candidates, panel-state assessment, next inspection steps, similar incidents, escalation guidance, and safety/uncertainty notices.
- First-class case/session APIs support site, asset, panel-family, media, and technician feedback history.
- Invited human pilot users are persisted in the metadata store and authenticate through Supabase magic links.
- Sites and assets are app-managed catalog records, so technicians select real pilot inventory instead of typing free-text panel metadata.
- Metadata persists through SQLAlchemy to the configured database URL instead of an in-memory default.
- The text encoder requires a real sentence-transformer model in normal app runs. Tests use an explicit `mock` backend instead of a silent fallback.
- The supported training path in this repo is `mtc-train-adapter` for the visual-text projector. Self-supervised JEPA pretraining is not part of the supported runtime.
- Calibrated issue ranking and escalation use a persisted triage-policy checkpoint trained with `mtc-train-policy`.

## Local Development

```bash
PYENV_VERSION=3.11.9 python -m pip install -e ".[dev]"
PYENV_VERSION=3.11.9 python -m pytest -q
PYENV_VERSION=3.11.9 mtc-api
npm install
npm run mobile
npm run admin
npm run frontend:test
```

Development mode can run with SQLite metadata and in-process vector fallback. Production mode requires PostgreSQL, Qdrant, and mounted model assets.

Frontend pilot builds expect baked environment variables:

- Mobile: `EXPO_PUBLIC_MTC_API_BASE_URL`, `EXPO_PUBLIC_SUPABASE_URL`, `EXPO_PUBLIC_SUPABASE_ANON_KEY`
- Admin: `VITE_MTC_API_BASE_URL`, `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`

Optional:

- Mobile redirect override: `EXPO_PUBLIC_SUPABASE_MOBILE_REDIRECT_URL`
- Admin redirect override: `VITE_SUPABASE_WEB_REDIRECT_URL`
- Hidden debug override toggle: `EXPO_PUBLIC_MTC_ENABLE_DEBUG_OVERRIDE=1` or `VITE_MTC_ENABLE_DEBUG_OVERRIDE=1`

## Dev Orchestration

- `docker compose up api postgres qdrant minio` starts the backend stack.
- `docker compose --profile ui up admin` starts the admin web console inside compose on port `5173`.
- `docker compose --profile mobile up mobile` starts the Expo dev service inside compose for optional containerized mobile development.
- Human pilot login is Supabase-backed. Set `MTC_SUPABASE_*`, `VITE_SUPABASE_*`, and `EXPO_PUBLIC_SUPABASE_*` env vars before using the admin/mobile UI profiles.
- Service tokens remain available for smoke tests and non-human integrations through `MTC_SERVICE_TOKENS_JSON`.

## Endpoints

- `GET /auth/me`
- `GET /admin/pilot-users`
- `POST /admin/pilot-users/invite`
- `GET /catalog/sites`
- `POST /catalog/sites`
- `PATCH /catalog/sites/{site_id}`
- `GET /catalog/assets`
- `POST /catalog/assets`
- `PATCH /catalog/assets/{asset_id}`
- `POST /cases`
- `POST /cases/{case_id}/analyze`
- `POST /cases/{case_id}/feedback`
- `GET /cases`
- `GET /cases/{case_id}`
- `POST /corpus/documents`
- `POST /corpus/incidents`
- `GET /corpus/documents`
- `GET /corpus/incidents`
- `POST /corpus/upload`
- `POST /media/triage`
- `POST /media/encode`
- `POST /reference-states`
- `POST /reference-states/upload`
- `GET /reference-states`
- `GET /admin/dashboard`
- `POST /triage/analyze`
- `GET /audit/triage`
- `GET /audit/triage/{audit_id}`
- `GET /metrics`
- `GET /system/health`

## Pilot Surfaces

- [`apps/mobile`](./apps/mobile): React Native + Expo technician app for on-site capture, analysis, retry, and feedback.
- [`apps/admin`](./apps/admin): React admin console for Supabase-backed login, invite issuance, catalog management, corpus uploads, reference-state curation, dashboard metrics, and case review.

## Production Models

- Mount model assets into `/models`.
- Include `/models/manifest.json` with SHA256 digests, official backbone presets, and asset paths.
- Set the sentence-transformer directory at `/models/text-encoder`.
- Set the projector checkpoint at `/models/projector.pt`.
- Set the calibrated triage-policy checkpoint at `/models/triage-policy.json`.
- Set the official I-JEPA encoder checkpoint at `/models/image_backbone.pt`.
- Set the official V-JEPA encoder checkpoint at `/models/video_backbone.pt`.
- Run `mtc-validate-model-assets --config /app/configs/production.yaml` as the canonical preflight check.

## Production Runtime

- `runtime.mode: production` fails fast unless `database.postgres_url` is PostgreSQL and `database.qdrant_url` is configured.
- Production also fails fast unless `/models/manifest.json`, the configured checkpoints, bearer service tokens, and S3-compatible object storage are all configured and reachable.
- Human pilot login expects Supabase config in `supabase.project_url`, `supabase.anon_key`, `supabase.service_role_key`, `supabase.jwt_issuer`, and `supabase.jwt_audience`.
- Alembic migrations run at startup by default and can also be applied with `mtc-db-upgrade --database-url ...`.
- Protected endpoints require `Authorization: Bearer <token>`. Only `GET /system/health` is unauthenticated.
- Uploaded media and corpus files are persisted to object storage and linked into triage audit records.
- `GET /metrics` exposes Prometheus metrics, and `GET /audit/triage` exposes persisted triage audit history.
- The compose smoke test mounts a real `/models` directory and can be run with `RUN_DOCKER_SMOKE=1 PYENV_VERSION=3.11.9 python -m pytest tests/integration/test_docker_compose_smoke.py -q`.

## Mobile Distribution

- Expo EAS internal distribution is configured in [`eas.json`](./eas.json).
- Use `eas build --platform ios --profile pilot-ios` for the iOS internal pilot build.
- Use `eas build --platform android --profile pilot-android` for the Android internal pilot build.
- The pilot flow is: invite technician in the admin console, send Supabase magic link, install internal build, tap link on device, app opens via `mtc://auth/callback`, and session restores automatically.
