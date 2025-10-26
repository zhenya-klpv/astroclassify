from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

logger = logging.getLogger("astroclassify.observability")

try:  # pragma: no cover - optional dependency
    from opentelemetry import trace  # type: ignore
    from opentelemetry.sdk.resources import Resource  # type: ignore
    from opentelemetry.sdk.trace import TracerProvider  # type: ignore
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
    try:  # prefer HTTP exporter when available
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore
            OTLPSpanExporter,
        )
    except Exception:  # pragma: no cover - fallback to gRPC exporter
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore
            OTLPSpanExporter,
        )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # type: ignore
    from opentelemetry.instrumentation.requests import RequestsInstrumentor  # type: ignore
    from opentelemetry.trace import Tracer as _Tracer  # type: ignore

    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency missing
    trace = None  # type: ignore
    Resource = None  # type: ignore
    TracerProvider = None  # type: ignore
    BatchSpanProcessor = None  # type: ignore
    OTLPSpanExporter = None  # type: ignore
    FastAPIInstrumentor = None  # type: ignore
    RequestsInstrumentor = None  # type: ignore
    _Tracer = None  # type: ignore
    _OTEL_AVAILABLE = False

DEFAULT_TRACE_ID = "0" * 32
DEFAULT_SPAN_ID = "0" * 16

_LOG_FILTER_ATTACHED = False
_MIDDLEWARE_ATTACHED = False
_REQUESTS_INSTRUMENTED = False
_CONFIGURED = False


class TraceContextFilter(logging.Filter):
    """Inject trace/span ids into every log record for formatting."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - simple setter
        trace_id, span_id = current_trace_ids()
        record.trace_id = trace_id
        record.span_id = span_id
        return True


def ensure_logging_filter() -> None:
    """Attach the trace-id logging filter to the root logger (idempotent)."""
    global _LOG_FILTER_ATTACHED
    if _LOG_FILTER_ATTACHED:
        return
    root = logging.getLogger()
    root.addFilter(TraceContextFilter())
    _LOG_FILTER_ATTACHED = True


def current_trace_ids() -> Tuple[str, str]:
    """Return (trace_id, span_id) as hex strings; zeroed if no active span."""
    if not _OTEL_AVAILABLE or trace is None:  # pragma: no cover - dependency missing
        return DEFAULT_TRACE_ID, DEFAULT_SPAN_ID
    try:
        span = trace.get_current_span()
        if span is None:
            return DEFAULT_TRACE_ID, DEFAULT_SPAN_ID
        ctx = span.get_span_context()
        if ctx is None or not ctx.is_valid:
            return DEFAULT_TRACE_ID, DEFAULT_SPAN_ID
        return f"{ctx.trace_id:032x}", f"{ctx.span_id:016x}"
    except Exception:  # pragma: no cover - defensive
        return DEFAULT_TRACE_ID, DEFAULT_SPAN_ID


def _parse_otlp_headers(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    headers: Dict[str, str] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            headers[key] = value
    return headers


def _build_otlp_exporter():
    if not _OTEL_AVAILABLE or OTLPSpanExporter is None:
        return None

    endpoint = os.environ.get("ASTRO_OTLP_ENDPOINT") or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/v1/traces"):
            endpoint = f"{endpoint}/v1/traces"
    else:
        endpoint = "http://localhost:4318/v1/traces"

    headers = _parse_otlp_headers(
        os.environ.get("ASTRO_OTLP_HEADERS") or os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")
    )

    try:
        kwargs = {"endpoint": endpoint}
        if headers:
            kwargs["headers"] = headers
        return OTLPSpanExporter(**kwargs)
    except Exception as exc:  # pragma: no cover - exporter misconfiguration
        logger.warning("OTLP exporter initialization failed: %s", exc)
        return None


async def _trace_headers_middleware(request, call_next):  # pragma: no cover - simple middleware
    response = await call_next(request)
    trace_id, span_id = current_trace_ids()
    response.headers.setdefault("x-trace-id", trace_id)
    response.headers.setdefault("x-span-id", span_id)
    response.headers.setdefault("traceparent", f"00-{trace_id}-{span_id}-01")
    return response


class _NoopTracer:
    """Fallback tracer when OpenTelemetry is unavailable."""

    def start_as_current_span(self, name: str, **kwargs):
        return nullcontext(None)


def get_tracer(name: str = "astroclassify") -> "_Tracer | _NoopTracer":
    if not _OTEL_AVAILABLE or trace is None:
        return _NoopTracer()
    return trace.get_tracer(name)


def configure_observability(app) -> None:
    """Configure tracing + logging if OpenTelemetry is present."""
    global _CONFIGURED, _MIDDLEWARE_ATTACHED, _REQUESTS_INSTRUMENTED

    if _CONFIGURED:
        return

    ensure_logging_filter()

    if not _OTEL_AVAILABLE:  # pragma: no cover - optional dependency
        if not _MIDDLEWARE_ATTACHED:
            app.middleware("http")(_trace_headers_middleware)
            _MIDDLEWARE_ATTACHED = True
        logger.info("OpenTelemetry packages not installed; tracing disabled.")
        _CONFIGURED = True
        return

    if trace.get_tracer_provider() is None or not isinstance(trace.get_tracer_provider(), TracerProvider):
        service_name = os.environ.get("OTEL_SERVICE_NAME") or os.environ.get("ASTRO_SERVICE_NAME") or "astroclassify-api"
        service_version = os.environ.get("ASTRO_API_VERSION") or os.environ.get("ASTRO_VERSION") or "unknown"
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.namespace": "astroclassify",
                "service.version": service_version,
            }
        )
        provider = TracerProvider(resource=resource)
        exporter = _build_otlp_exporter()
        if exporter is not None:
            processor = BatchSpanProcessor(exporter)
            provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

    if FastAPIInstrumentor is not None:
        try:
            FastAPIInstrumentor().instrument_app(app, tracer_provider=trace.get_tracer_provider())
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("FastAPI instrumentation failed: %s", exc)

    if RequestsInstrumentor is not None and not _REQUESTS_INSTRUMENTED:
        try:
            RequestsInstrumentor().instrument()
            _REQUESTS_INSTRUMENTED = True
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Requests instrumentation failed: %s", exc)

    if not _MIDDLEWARE_ATTACHED:
        app.middleware("http")(_trace_headers_middleware)
        _MIDDLEWARE_ATTACHED = True

    _CONFIGURED = True
