"""API middleware for auth, request IDs, and metrics."""

from __future__ import annotations

import time
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from maintenance_triage_copilot.telemetry import current_trace_id, record_auth_failure, record_request
from maintenance_triage_copilot.utils.logging import get_logger

log = get_logger(__name__)

_AUTH_EXEMPT_PATHS = {"/system/health"}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Require bearer service auth when the app is configured for it."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _AUTH_EXEMPT_PATHS:
            request.state.principal = "healthcheck"
            return await call_next(request)

        token_lookup: dict[str, str] = getattr(request.app.state, "service_token_lookup", {})
        auth_required = bool(getattr(request.app.state, "auth_required", False))
        if not token_lookup and not auth_required:
            request.state.principal = "anonymous"
            return await call_next(request)

        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            record_auth_failure(request.url.path)
            return Response(
                content='{"detail":"Missing bearer token"}',
                status_code=401,
                media_type="application/json",
            )

        token = header.removeprefix("Bearer ").strip()
        principal = token_lookup.get(token)
        if principal is None:
            record_auth_failure(request.url.path)
            return Response(
                content='{"detail":"Invalid bearer token"}',
                status_code=401,
                media_type="application/json",
            )

        request.state.principal = principal
        return await call_next(request)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach request IDs, emit logs, and record request metrics."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req-{uuid4().hex}"
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        duration_seconds = time.perf_counter() - start
        trace_id = current_trace_id()
        principal = getattr(request.state, "principal", "unknown")

        record_request(request.method, request.url.path, response.status_code, duration_seconds)
        log.info(
            "http_request",
            extra={
                "event": "http_request",
                "request_id": request_id,
                "trace_id": trace_id or "",
                "principal": principal,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration_seconds * 1000, 2),
            },
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Request-Duration-Ms"] = f"{duration_seconds * 1000:.1f}"
        if trace_id is not None:
            response.headers["X-Trace-ID"] = trace_id
        return response
