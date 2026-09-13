from __future__ import annotations

import json

from start.ai_engineering import (
    ADAPTER_CLASSES,
    build_adapters,
    run_ai_engineering_layer,
)
from start.ai_engineering.base import BaseAdapter


def test_all_required_adapters_present():
    names = {cls.name for cls in ADAPTER_CLASSES}
    assert {
        "OPA",
        "MCP Server",
        "MCP SDK",
        "MCP Inspector",
        "Langfuse",
        "OpenTelemetry",
        "Garak",
        "Promptfoo",
        "Moonshot",
        "NeMo Guardrails",
        "DeepEval",
    } <= names


def test_every_adapter_implements_contract():
    for adapter in build_adapters():
        for method in ("available", "validate", "execute", "collect_artifacts", "emit_evidence"):
            assert callable(getattr(adapter, method)), f"{adapter.name} missing {method}"


def test_unavailable_adapters_are_honest(tmp_path):
    # In the test env, most backends are absent — must report not_installed,
    # never fabricate success, and still emit evidence + a finding + guidance.
    report = run_ai_engineering_layer({"run_id": "R"}, output_root=str(tmp_path))
    for result in report.results:
        assert result.status in {"complete", "not_installed", "error"}
        if not result.available:
            assert result.status == "not_installed"
            assert result.detail  # explains why
            assert result.findings and result.findings[0].recommendation  # install guidance
            assert result.evidence  # still auditable
            # install hint present
            d = result.detail.lower()
            assert "install" in d or "pip" in d or "npm" in d


def test_opentelemetry_really_executes(tmp_path):
    # OpenTelemetry is installed -> the adapter must genuinely run and emit a span.
    report = run_ai_engineering_layer(
        {"run_id": "RUN-OTEL", "n_stages": 22, "evidence_count": 6}, output_root=str(tmp_path)
    )
    otel = next(r for r in report.results if r.adapter == "OpenTelemetry")
    assert otel.available and otel.status == "complete"
    assert otel.summary["span_count"] == 1
    assert otel.runtime_seconds > 0
    # the artifact is a real serialized span
    assert any("telemetry" in a.name for a in otel.artifacts)
    payload = json.loads((tmp_path / "ai_engineering" / "RUN-OTEL" / "telemetry.json").read_text())
    assert payload["span_count"] == 1
    assert payload["spans"][0]["name"] == "model_review"
    assert payload["spans"][0]["attributes"]["start.run_id"] == "RUN-OTEL"


def test_layer_aggregates_artifacts_findings_evidence(tmp_path):
    report = run_ai_engineering_layer({"run_id": "R"}, output_root=str(tmp_path))
    assert report.total == len(ADAPTER_CLASSES)
    assert report.available_count >= 1  # at least OpenTelemetry
    # every adapter contributes exactly one evidence record
    assert len(report.evidence) == report.total
    rows = report.summary_rows()
    assert len(rows) == report.total
    assert all("runtime_s" in r for r in rows)


def test_streaming_callback_fires_per_adapter(tmp_path):
    seen = []
    run_ai_engineering_layer(
        {"run_id": "R"}, output_root=str(tmp_path), on_adapter=lambda r: seen.append(r.adapter)
    )
    assert len(seen) == len(ADAPTER_CLASSES)


def test_adapter_error_is_surfaced(tmp_path):
    class ExplodingAdapter(BaseAdapter):
        name = "Exploder"
        category = "test"

        def available(self) -> bool:
            return True

        def _run(self, context):
            raise RuntimeError("boom")

    result = ExplodingAdapter(output_root=str(tmp_path)).execute({"run_id": "R"})
    assert result.status == "error"
    assert "boom" in result.detail
    assert result.evidence  # error still produces an evidence record


def test_validate_method():
    for adapter in build_adapters():
        v = adapter.validate()
        # validate returns ok iff available
        assert v.ok == adapter.available()
        if not v.ok:
            assert v.detail


# -- v2.1.1 control surface (Sections L/M) ------------------------------------ #
def test_adapter_describe_has_control_surface_fields():
    from start.ai_engineering.adapters import OPAAdapter, OpenTelemetryAdapter

    for A in (OPAAdapter, OpenTelemetryAdapter):
        d = A().describe()
        for key in ("adapter", "purpose", "role", "status", "would_do",
                    "expected_outputs", "install_guidance"):
            assert key in d
        assert d["purpose"] and d["role"]


def test_all_adapters_have_purpose_and_role():
    from start.ai_engineering.adapters import build_adapters

    for adapter in build_adapters(output_root="/tmp/start_cs"):
        d = adapter.describe()
        assert d["purpose"], f"{d['adapter']} missing purpose"
        assert d["role"], f"{d['adapter']} missing role"


def test_report_control_surface_merges_status():
    import tempfile

    from start.ai_engineering.layer import run_ai_engineering_layer

    rep = run_ai_engineering_layer({}, output_root=tempfile.mkdtemp())
    cs = rep.control_surface()
    from start.ai_engineering.adapters import ADAPTER_CLASSES
    assert len(cs) == len(ADAPTER_CLASSES)
    valid = {"complete", "available", "not_installed", "error"}
    # Environment-robust: every adapter reports a valid status with purpose/role,
    # and unavailable adapters still carry install guidance. We do NOT assume a
    # fixed set of installed adapters — the machine may have OPA, Promptfoo, MCP
    # Inspector, OpenTelemetry, etc. installed (items 1 & 6).
    for row in cs:
        assert row["status"] in valid, f"{row['adapter']} has invalid status {row['status']}"
        assert row["purpose"] and row["role"], f"{row['adapter']} missing purpose/role"
        if row["status"] == "not_installed":
            assert row["install_guidance"], f"{row['adapter']} missing install guidance"
    # OpenTelemetry is the one adapter we always ship a working backend for, so
    # it must be available/complete in any StART environment.
    otel = next(r for r in cs if r["adapter"] == "OpenTelemetry")
    assert otel["status"] in ("complete", "available")
    # OPA must report a valid status and, regardless of install state, expose its
    # governance purpose — this stays correct whether or not the OPA binary exists.
    opa = next(r for r in cs if r["adapter"] == "OPA")
    assert opa["status"] in valid
    assert "policy" in opa["purpose"].lower() or "governance" in opa["role"].lower()
