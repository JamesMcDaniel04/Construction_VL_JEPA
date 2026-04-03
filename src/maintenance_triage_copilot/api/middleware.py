"""API middleware: authentication and request metrics."""

from __future__ import annotations

import os
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from maintenance_triage_copilot.utils.logging import get_logger

log = get_logger(__name__)

# Optional API key from environment — when set, all requests must provide it
_API_KEY = os.getenv("MTC_API_KEY")


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Reject requests missing a valid API key (when MTC_API_KEY is set)."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if _API_KEY is None:
            return await call_next(request)

        # Allow health checks without auth
        if request.url.path in {"/system/health", "/docs", "/openapi.json", "/redoc"}:
            return await call_next(request)

        provided = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if provided != _API_KEY:
            return Response(
                content='{"detail":"Invalid or missing API key"}',
                status_code=401,
                media_type="application/json",
            )
        return await call_next(request)


class RequestMetricsMiddleware(BaseHTTPMiddleware):
    """Log request duration and status for observability."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        log.info(
            "http_request method=%s path=%s status=%s duration_ms=%.1f",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-Duration-Ms"] = f"{duration_ms:.1f}"
        return response
