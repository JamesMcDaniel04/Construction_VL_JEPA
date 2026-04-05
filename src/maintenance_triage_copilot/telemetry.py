"""Prometheus metrics and OpenTelemetry helpers."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from maintenance_triage_copilot.config import TelemetryConfig

REQUEST_COUNTER = Counter(
    "mtc_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "mtc_http_request_duration_seconds",
    "HTTP request duration",
    ["method", "path"],
)
AUTH_FAILURES = Counter(
    "mtc_auth_failures_total",
    "Authentication failures",
    ["path"],
)
OPERATION_LATENCY = Histogram(
    "mtc_operation_duration_seconds",
    "Duration of instrumented operations",
    ["operation"],
)


def metrics_payload() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


def record_request(method: str, path: str, status: int, duration_seconds: float) -> None:
    REQUEST_COUNTER.labels(method=method, path=path, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(duration_seconds)


def record_auth_failure(path: str) -> None:
    AUTH_FAILURES.labels(path=path).inc()


@contextmanager
def trace_operation(name: str) -> Iterator[None]:
    from opentelemetry import trace

    tracer = trace.get_tracer("maintenance-triage-copilot")
    start = time.perf_counter()
    with tracer.start_as_current_span(name):
        try:
            yield
        finally:
            OPERATION_LATENCY.labels(operation=name).observe(time.perf_counter() - start)


def current_trace_id() -> str | None:
    from opentelemetry import trace

    span_context = trace.get_current_span().get_span_context()
    if not span_context.is_valid:
        return None
    return f"{span_context.trace_id:032x}"


def configure_telemetry(
    *,
    cfg: TelemetryConfig,
    app=None,
    engine=None,
) -> None:
    if cfg.otlp_endpoint is None:
        return

    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": cfg.service_name}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.otlp_endpoint)))
    trace.set_tracer_provider(provider)

    if app is not None:
        FastAPIInstrumentor.instrument_app(app)
    if engine is not None:
        SQLAlchemyInstrumentor().instrument(engine=engine)
