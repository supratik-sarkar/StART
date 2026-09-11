"""Gate B: Engineering Trace, Observability & Policy Closure Test Suite.

Validates:
- Structured parent/child trace events and schema compliance
- Stable run identity across all trace spans
- Local trace export without external collector
- Machine-readable JSONL export
- Fail-closed optional OTLP export
- Security & secret redaction (zero API keys/tokens, zero full prompt leaks)
- Terminal engineering renderer exposing all 8 required sections
- Resource ledger showing LLM calls = 0 in deterministic mode
- Policy adapter contract: decision ID exists and references Evidence IDs
- Preservation invariant: Trace OFF vs Trace ENGINEERING produces 0 scientific drift
"""

import json
import os
import tempfile
from pathlib import Path

# Numerical safety invariants for macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["START_TORCH_DEVICE"] = "cpu"
os.environ["START_DISABLE_MPS"] = "1"
os.environ["START_DEVICE"] = "cpu"

import pytest

pytest.importorskip("opentelemetry")

from start.runtime.execution import CanonicalExecutionService
from start.telemetry.engineering_trace import (
    OP_DATASET_RESOLVE,
    OP_EVIDENCE_EMIT,
    OP_MODEL_EVALUATE,
    OP_MODEL_FIT,
    OP_RUN,
    EngineeringTracer,
    PolicyAdapter,
    TerminalEngineeringRenderer,
    sanitize_trace_value,
)


def test_native_trace_structured_spans_and_stable_run_identity():
    """Verify EngineeringTracer creates structured parent/child spans with stable run identity."""
    run_id = "RUN-TEST-001"
    tracer = EngineeringTracer(run_id=run_id, service_name="start.test")

    with tracer.span(OP_RUN, {"workflow": "predictive_ml"}):
        with tracer.span(OP_DATASET_RESOLVE, {"dataset": "institutional_credit_v1"}):
            pass
        with tracer.span(OP_MODEL_FIT, {"model": "random_forest"}):
            with tracer.span(OP_MODEL_EVALUATE, {"metric": "auc_roc"}):
                pass

    records = tracer.get_records()
    assert len(records) == 4

    for rec in records:
        assert rec.run_id == run_id
        assert len(rec.span_id) == 16
        assert len(rec.trace_id) == 32
        assert rec.status == "OK"
        assert rec.duration_ms >= 0.0

    names = [r.name for r in records]
    assert OP_DATASET_RESOLVE in names
    assert OP_MODEL_FIT in names
    assert OP_MODEL_EVALUATE in names
    assert OP_RUN in names


def test_machine_readable_jsonl_export():
    """Verify trace records export to machine-readable JSONL format matching specification."""
    run_id = "RUN-TEST-JSONL"
    tracer = EngineeringTracer(run_id=run_id)

    with tracer.span(OP_RUN, {"key": "val"}):
        with tracer.span(OP_EVIDENCE_EMIT, {"evidence_id": "EV-12345"}):
            pass

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "test_trace.jsonl"
        tracer.export_jsonl(jsonl_path)
        assert jsonl_path.exists()

        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "trace_id" in data
            assert "span_id" in data
            assert "name" in data
            assert "start_time" in data
            assert "end_time" in data
            assert "duration_ms" in data
            assert "status" in data
            assert data["run_id"] == run_id


def test_opentelemetry_compatibility_and_fail_closed_otlp():
    """Verify in-process OTel TracerProvider and fail-closed OTLP export behavior."""
    tracer = EngineeringTracer(run_id="RUN-OTEL")
    with tracer.span(OP_RUN):
        pass

    # In-memory readable spans
    finished = tracer.exporter.get_finished_spans()
    assert len(finished) >= 1

    # Fail-closed OTLP generator
    otlp_local = tracer.export_otlp(endpoint=None)
    assert "resourceSpans" in otlp_local
    assert len(otlp_local["resourceSpans"]) > 0

    # Non-existent endpoint fails safely without crashing
    otlp_fail = tracer.export_otlp(endpoint="http://127.0.0.1:4317")
    assert otlp_fail["status"] == "FAIL_CLOSED_NO_COLLECTOR"


def test_trace_security_and_secret_redaction():
    """Verify API keys, bearer tokens, and full LLM prompts are redacted from span attributes."""
    raw_attrs = {
        "api_key": "sk-proj-abc12345678901234567890",
        "secret_token": "bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "normal_key": "safe_value",
        "llm_prompt": "Classify whether this applicant defaults on loan. Target is binary 0 or 1.",
    }

    sanitized = {k: sanitize_trace_value(v, k) for k, v in raw_attrs.items()}
    assert sanitized["api_key"] == "[REDACTED]"
    assert sanitized["secret_token"] == "[REDACTED]"
    assert sanitized["normal_key"] == "safe_value"
    assert "[PROMPT_HASH:" in sanitized["llm_prompt"]
    assert "Classify whether this applicant" not in sanitized["llm_prompt"]


def test_terminal_engineering_renderer_eight_sections():
    """Verify terminal engineering renderer produces all 8 required sections."""
    rendered = TerminalEngineeringRenderer.render(
        run_id="RUN-RENDER-TEST",
        dataset_contract={"provider": "local_csv", "dataset_id": "adult_census.csv", "rows": 1000, "features": 14},
        orchestration={"execution_mode": "deterministic_run", "llm_provider": "none", "llm_model": "none"},
        dispatch={"requested_model": "RandomForestClassifier", "device": "cpu"},
        scientific_invariants={"leakage": "CLEAN", "metric_domains": "VALID", "finite_outputs": "VERIFIED"},
        evidence_lineage={"total_records": 47, "sample_evidence_ids": ["EV-001", "EV-002"]},
        governance_policy={"decision": "ACCEPT", "decision_id": "POL-DEC-001", "engine": "OPA_LOCAL", "evidence_ids": ["EV-001"]},
        reproducibility={"seed": 42, "merkle_root": "0a492d24589e4fa8"},
        resource_ledger={"wall_time_seconds": 1.25, "device": "cpu", "llm_calls": 0, "llm_tokens": 0},
        trace_mode="engineering",
    )

    required_sections = [
        "1. DATASET CONTRACT PROOF",
        "2. ORCHESTRATION",
        "3. DISPATCH PROOF",
        "4. SCIENTIFIC INVARIANTS",
        "5. EVIDENCE LINEAGE",
        "6. GOVERNANCE / POLICY",
        "7. REPRODUCIBILITY CAPSULE",
        "8. RESOURCE LEDGER",
        "LLM Calls:        0 (Deterministic mode LLM calls = 0)",
    ]

    for sec in required_sections:
        assert sec in rendered, f"Missing section in engineering trace: {sec}"


def test_policy_adapter_contract_and_decision_id():
    """Verify PolicyAdapter emits decision ID referencing Evidence IDs with authentic OPA or native fallback."""
    adapter = PolicyAdapter()
    evidence_ids = ["EV-abc1", "EV-abc2", "EV-abc3"]

    decision = adapter.evaluate_signoff(
        run_id="RUN-POL-TEST",
        evidence_ids=evidence_ids,
        disposition="ACCEPT",
        ungrounded_claims=0,
        validation_failures=0,
    )

    assert decision.decision_id.startswith("POL-DEC-")
    assert decision.decision == "ALLOW"
    assert decision.evidence_ids == evidence_ids
    assert decision.engine in ("OPA_LOCAL", "NATIVE_ADAPTER")
    assert len(decision.input_fingerprint) > 0


def test_preservation_trace_off_vs_trace_engineering():
    """Verify Trace OFF vs Trace ENGINEERING produces identical scientific metrics, records, and governance."""
    # Run with trace_mode="off"
    res_off = CanonicalExecutionService.execute(
        workflow_id="predictive_ml",
        context_id="institutional_credit_v1",
        request_params={"model": "logistic_regression"},
        seed=42,
        execution_mode="deterministic_run",
        trace_mode="off",
    )

    # Run with trace_mode="engineering"
    res_on = CanonicalExecutionService.execute(
        workflow_id="predictive_ml",
        context_id="institutional_credit_v1",
        request_params={"model": "logistic_regression"},
        seed=42,
        execution_mode="deterministic_run",
        trace_mode="engineering",
    )

    # 1. Zero evidence count drift
    assert len(res_off.records) == len(res_on.records)

    # 2. Zero test ID drift
    test_ids_off = [r.test_id for r in res_off.records]
    test_ids_on = [r.test_id for r in res_on.records]
    assert test_ids_off == test_ids_on

    # 3. Zero primary metric drift
    rec_disc_off = next(r for r in res_off.records if r.test_id == "supervised.discrimination")
    rec_disc_on = next(r for r in res_on.records if r.test_id == "supervised.discrimination")
    assert rec_disc_off.metrics["roc_auc"] == rec_disc_on.metrics["roc_auc"]

    # 4. Zero governance semantics drift
    assert res_off.governance_disposition == res_on.governance_disposition

    # 5. Trace existence
    assert res_off.tracer is None
    assert res_on.tracer is not None
    assert len(res_on.tracer.get_records()) > 0
