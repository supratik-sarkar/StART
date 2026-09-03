"""Institutional OpenTelemetry Tracing Engine for StART.

Production-grade OpenTelemetry implementation using the official OpenTelemetry Python SDK:
- Official `TracerProvider`, `Tracer`, and `ReadableSpan` hierarchy.
- Official `InMemorySpanExporter` for zero-egress offline verification.
- Context-propagating nested spans across the canonical lifecycle:
    review.run -> review.checkpoint -> agent.execution -> tool.execution ->
    evidence.commit -> artifact.generate -> policy.evaluate -> governance.evaluate -> attestation.seal
- Strict secret/credential redaction in all span attributes.
"""

from __future__ import annotations

import re
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Span, StatusCode

SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|secret|token|bearer|password|auth)[\s:=]+([^\s,;'\"]+)"),
    re.compile(r"sk-[a-zA-Z0-9_\-]{15,}"),
]
SENSITIVE_KEYS = {"api_key", "secret", "token", "password", "bearer", "authorization", "auth"}


def sanitize_attribute_value(val: Any, key: str = "") -> Any:
    """Recursively sanitize attribute values to prevent data or secret leakage."""
    if val is None:
        return None
    if key and any(k in key.lower() for k in SENSITIVE_KEYS):
        return "[REDACTED]"
    if isinstance(val, (int, float, bool)):
        return val
    if isinstance(val, (list, tuple)):
        return [sanitize_attribute_value(x) for x in val]
    if isinstance(val, dict):
        return {str(k): sanitize_attribute_value(v, str(k)) for k, v in val.items()}

    s = str(val)
    for pat in SENSITIVE_PATTERNS:
        s = pat.sub("[REDACTED]", s)
    return s


class OTelTracer:
    """Official OpenTelemetry SDK Tracer and In-Memory Collector for StART reviews."""

    def __init__(self, service_name: str = "start.review", privacy_mode: bool = True) -> None:
        self.service_name = service_name
        self.privacy_mode = privacy_mode

        # Initialize official OTel SDK provider and in-memory exporter
        self.provider = TracerProvider()
        self.exporter = InMemorySpanExporter()
        self.processor = SimpleSpanProcessor(self.exporter)
        self.provider.add_span_processor(self.processor)
        self.tracer = self.provider.get_tracer(self.service_name)
        self._active_spans: list[Span] = []

    def start_span(
        self,
        name: str,
        trace_id: str | None = None,
        parent_span: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Create and start a real OpenTelemetry Span with context propagation."""
        ctx = trace.set_span_in_context(parent_span) if parent_span else None
        span = self.tracer.start_span(name, context=ctx)
        if attributes:
            for k, v in attributes.items():
                sanitized = sanitize_attribute_value(v, str(k))
                if isinstance(sanitized, (str, bool, int, float)):
                    span.set_attribute(str(k), sanitized)
                elif isinstance(sanitized, (list, tuple)):
                    span.set_attribute(str(k), [str(x) for x in sanitized])
                else:
                    span.set_attribute(str(k), str(sanitized))
        self._active_spans.append(span)
        return span

    def end_span(self, span: Span, status: str = "OK", error_msg: str = "") -> None:
        """End an OpenTelemetry span with official StatusCode."""
        if status.upper() == "OK":
            span.set_status(StatusCode.OK)
        else:
            span.set_status(StatusCode.ERROR, description=error_msg)
        span.end()

    def get_finished_spans(self) -> list[ReadableSpan]:
        """Return all finished ReadableSpans collected by the in-memory exporter."""
        return list(self.exporter.get_finished_spans())

    def get_span_hierarchy(self) -> list[dict[str, Any]]:
        """Return parent-child hierarchical tree from recorded ReadableSpans."""
        finished = self.get_finished_spans()
        by_id: dict[int, dict[str, Any]] = {}
        for s in finished:
            s_ctx = s.get_span_context()
            p_ctx = s.parent
            by_id[s_ctx.span_id] = {
                "name": s.name,
                "span_id": f"{s_ctx.span_id:016x}",
                "trace_id": f"{s_ctx.trace_id:032x}",
                "parent_span_id": f"{p_ctx.span_id:016x}" if p_ctx else None,
                "status": s.status.status_code.name,
                "attributes": dict(s.attributes or {}),
                "duration_ns": s.end_time - s.start_time if s.end_time and s.start_time else 0,
                "children": [],
            }

        roots = []
        for s in finished:
            s_ctx = s.get_span_context()
            s_dict = by_id[s_ctx.span_id]
            if s.parent and s.parent.span_id in by_id:
                by_id[s.parent.span_id]["children"].append(s_dict)
            else:
                roots.append(s_dict)
        return roots

    def to_otlp_payload(self) -> dict[str, Any]:
        """Export in standard OTLP JSON dictionary structure."""
        finished = self.get_finished_spans()
        return {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [
                            {"key": "service.name", "value": {"stringValue": self.service_name}},
                            {"key": "privacy.zero_egress", "value": {"boolValue": self.privacy_mode}},
                        ]
                    },
                    "scopeSpans": [
                        {
                            "scope": {"name": "start.telemetry"},
                            "spans": [
                                {
                                    "name": s.name,
                                    "spanId": f"{s.get_span_context().span_id:016x}",
                                    "traceId": f"{s.get_span_context().trace_id:032x}",
                                    "parentSpanId": f"{s.parent.span_id:016x}" if s.parent else "",
                                    "status": {"code": s.status.status_code.name},
                                    "attributes": [{"key": k, "value": {"stringValue": str(v)}} for k, v in (s.attributes or {}).items()],
                                }
                                for s in finished
                            ],
                        }
                    ],
                }
            ]
        }
