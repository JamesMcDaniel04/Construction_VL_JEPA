"""API middleware for auth, request IDs, and metrics."""

from __future__ import annotations

import time
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from maintenance_triage_copilot.domain.models import UserRole
from maintenance_triage_copilot.telemetry import (
    current_trace_id,
    record_auth_failure,
    record_request,
)
from maintenance_triage_copilot.utils.logging import get_logger

log = get_logger(__name__)

_AUTH_EXEMPT_PATHS = {"/system/health"}


class BearerAuthMiddleware(BaseHTTPMiddleware):
    """Require bearer auth for service tokens and Supabase-authenticated pilot users."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path in _AUTH_EXEMPT_PATHS:
            request.state.principal = "healthcheck"
            request.state.principal_type = "service"
            request.state.role = UserRole.service.value
            request.state.user_id = "healthcheck"
            request.state.organization_id = None
            request.state.display_name = "healthcheck"
            request.state.email = None
            return await call_next(request)

        service_lookup: dict[str, str] = getattr(request.app.state, "service_token_lookup", {})
        pilot_lookup = getattr(request.app.state, "pilot_user_lookup", {})
        auth_required = bool(getattr(request.app.state, "auth_required", False))
        if not service_lookup and not pilot_lookup and not auth_required:
            request.state.principal = "anonymous"
            request.state.principal_type = "anonymous"
            request.state.role = "anonymous"
            request.state.user_id = None
            request.state.organization_id = None
            request.state.display_name = "anonymous"
            request.state.email = None
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
        principal = service_lookup.get(token)
        pilot_user = None
        if principal is None:
            claims = None
            verifier = getattr(request.app.state, "human_token_verifier", None)
            if callable(verifier):
                claims = verifier(token)
            else:
                auth_provider = getattr(request.app.state, "supabase_auth", None)
                if auth_provider is not None and getattr(
                    auth_provider,
                    "configured_for_human_auth",
                    lambda: False,
                )():
                    try:
                        claims = auth_provider.verify_access_token(token)
                    except Exception:
                        claims = None
            if claims is not None:
                pilot_user = pilot_lookup.get(claims.user_id)

        if principal is None and pilot_user is None:
            record_auth_failure(request.url.path)
            return Response(
                content='{"detail":"Invalid bearer token"}',
                status_code=401,
                media_type="application/json",
            )

        if pilot_user is not None:
            request.state.principal = pilot_user.user_id
            request.state.principal_type = "human"
            request.state.role = pilot_user.role.value
            request.state.user_id = pilot_user.user_id
            request.state.organization_id = pilot_user.organization_id
            request.state.display_name = pilot_user.display_name
            request.state.email = pilot_user.email
        else:
            assert principal is not None
            request.state.principal = principal
            request.state.principal_type = "service"
            request.state.role = UserRole.service.value
            request.state.user_id = principal
            request.state.organization_id = None
            request.state.display_name = principal
            request.state.email = None
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
        principal_type = getattr(request.state, "principal_type", "unknown")
        role = getattr(request.state, "role", "unknown")

        record_request(request.method, request.url.path, response.status_code, duration_seconds)
        log.info(
            "http_request",
            extra={
                "event": "http_request",
                "request_id": request_id,
                "trace_id": trace_id or "",
                "principal": principal,
                "principal_type": principal_type,
                "role": role,
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
