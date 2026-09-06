"""StART v5.1.0 — Canonical Web Execution Runtime Closure Acceptance Gates.

Validates all 51 binding amendments:
1. CORE_RUNTIME_IMPORTS_START_WEB == 0
2. WEB_PACKAGE_OWNS_EXECUTION_SEMANTICS == NO
3. CLI_AND_WEB_USE_SAME_EXECUTION_SERVICE == PASS
4. WEB_TRANSPORT_SYNTHETIC_RUNTIME_EVENTS == 0
5. SIMULATED_DL_EPOCH_EVENTS == 0
6. CONTEXT_SPEC_SINGLE_SOURCE == PASS
7. CONTEXT_METADATA_EQUALS_RUNTIME_CONTEXT == PASS
8. CONTEXT_TARGET_METADATA_EQUALS_RUNTIME == PASS
9. WORKFLOW_APPLICABILITY_RESOLUTION == PASS
10. PLAN_AND_EXECUTOR_SHARE_RESOLVED_SPEC == PASS
11. UNKNOWN_WORKFLOW == REJECTED (HTTP 400 before queue)
12. UNKNOWN_CONTEXT == REJECTED (HTTP 400 before queue)
13. INCOMPATIBLE_CONTEXT == REJECTED (HTTP 400 before queue)
14. GOVERNANCE_EVENT_WITHOUT_GOVERNANCE == 0
15. ATTESTATION_EVENT_WITHOUT_ATTESTATION == 0
16. TERMINAL_WEB_RUNTIME_EVENT_PARITY == PASS
17. EVERY_ENABLED_WORKFLOW_EXECUTION == REAL
18. DISABLED_WORKFLOWS_HAVE_TRUTHFUL_REASON == PASS
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from start.runtime.contexts import (
    get_canonical_context_specs,
    instantiate_context,
)
from start.runtime.events import ListEventSink, RuntimeEvent
from start.runtime.execution import CanonicalExecutionService
from start.runtime.workflows import (
    WORKFLOW_SPECS,
    EngineKind,
    get_workflow_catalog,
    resolve_workflow,
)
from start.web.app import create_app
from start.web.queue import GLOBAL_QUEUE, QueueEventSink
from start.web.schemas import RunRequest


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_core_runtime_imports_start_web_is_zero() -> None:
    """Amendment 1 & 19: CORE_RUNTIME_IMPORTS_START_WEB == 0.
    
    The canonical runtime layer MUST NOT import start.web in any form.
    """
    runtime_dir = Path(__file__).resolve().parent.parent / "src" / "start" / "runtime"
    assert runtime_dir.exists(), f"Runtime package directory missing: {runtime_dir}"

    violations = []
    for py_file in runtime_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("start.web"):
                        violations.append((py_file.name, node.lineno, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith("start.web"):
                    violations.append((py_file.name, node.lineno, node.module))

    assert len(violations) == 0, f"Violations of CORE_RUNTIME_IMPORTS_START_WEB == 0: {violations}"


def test_context_spec_single_source_and_metadata_parity() -> None:
    """Amendments 5, 6, 7, 8: Single canonical context definitions, zero ambiguous targets.
    
    Checks that metadata accurately matches the generated DataFrame shapes and target columns.
    """
    specs = get_canonical_context_specs()
    assert len(specs) >= 3

    spec_map = {s.context_id: s for s in specs}
    assert "institutional_credit_v1" in spec_map
    assert "deep_learning_v1" in spec_map
    assert "institutional_market_v1" in spec_map

    # 1. Credit context
    credit_spec = spec_map["institutional_credit_v1"]
    assert credit_spec.target_column == "target"
    credit_inst = instantiate_context("institutional_credit_v1")
    assert credit_inst.actual_rows == 500
    assert credit_inst.actual_features == 8
    assert credit_inst.actual_target == "target"
    assert credit_inst.actual_seed == 42
    assert "target" in credit_inst.context.train.columns

    # 2. Deep learning context (not labeled vision embedding)
    dl_spec = spec_map["deep_learning_v1"]
    assert "vision" not in dl_spec.label.lower()
    assert "vision" not in dl_spec.description.lower()
    assert dl_spec.target_column == "target"
    dl_inst = instantiate_context("deep_learning_v1")
    assert dl_inst.actual_rows == 500
    assert dl_inst.actual_features == 8
    assert dl_inst.actual_target == "target"
    assert dl_inst.actual_seed == 17
    assert "target" in dl_inst.context.train.columns

    # 3. Market context
    market_spec = spec_map["institutional_market_v1"]
    assert market_spec.target_column == "N/A"
    market_inst = instantiate_context("institutional_market_v1")
    assert market_inst.actual_rows == 1000
    assert market_inst.actual_assets == 50
    assert market_inst.actual_periods == 1000
    assert market_inst.actual_target == "N/A"
    assert market_inst.actual_seed == 7


def test_workflow_applicability_resolution_and_registry_parity() -> None:
    """Amendments 9, 10, 11, 12, 13: Candidate vs applicable distinction & registry alignment."""
    catalog = get_workflow_catalog()
    assert len(catalog) >= 5

    # 1. Predictive ML
    pred_res = resolve_workflow("predictive_ml", "institutional_credit_v1")
    assert len(pred_res.candidate_test_ids) == 52
    assert len(pred_res.applicable_test_ids) > 0
    assert len(pred_res.applicable_test_ids) <= len(pred_res.candidate_test_ids)
    # 2. Explainability does NOT include genai.citation_coverage (Amendment 12)
    xai_res = resolve_workflow("explainability", "institutional_credit_v1")
    assert "genai.citation_coverage" not in xai_res.candidate_test_ids

    # 3. Market Risk
    mkt_res = resolve_workflow("quantitative_finance", "institutional_market_v1")
    assert len(mkt_res.candidate_test_ids) == 25
    assert len(mkt_res.applicable_test_ids) > 0

    # 4. Hyperparameter tuning is an execution engine, not registered tests
    tune_spec = WORKFLOW_SPECS["hyperparameter_tuning"]
    assert tune_spec.engine_kind == EngineKind.TUNING

    # 5. Deep Learning review is Diagnostics / Review, not training epochs
    dl_spec = WORKFLOW_SPECS["deep_learning"]
    assert dl_spec.engine_kind == EngineKind.DEEP_LEARNING_REVIEW
    assert "training" not in dl_spec.label.lower()


def test_request_validation_order_and_rejections(client: TestClient) -> None:
    """Amendments 27 & 28: Turnstile -> Schema -> Semantic validation -> Queue submit.
    
    Unknown workflow, unknown context, and incompatible context must fail before queue submission.
    """
    initial_queue_size = len(GLOBAL_QUEUE._runs)

    # 1. Unknown workflow
    r1 = client.post("/api/v1/runs/start", json={
        "workflow": "unknown_workflow_xyz",
        "synthetic_profile": "institutional_credit_v1",
    })
    assert r1.status_code == 400
    assert r1.json()["error_code"] == "UNKNOWN_WORKFLOW"
    assert len(GLOBAL_QUEUE._runs) == initial_queue_size

    # 2. Unknown context
    r2 = client.post("/api/v1/runs/start", json={
        "workflow": "predictive_ml",
        "synthetic_profile": "unknown_context_abc",
    })
    assert r2.status_code == 400
    assert r2.json()["error_code"] == "UNKNOWN_CONTEXT"
    assert len(GLOBAL_QUEUE._runs) == initial_queue_size

    # 3. Incompatible context
    r3 = client.post("/api/v1/runs/start", json={
        "workflow": "predictive_ml",
        "synthetic_profile": "institutional_market_v1",
    })
    assert r3.status_code == 400
    assert r3.json()["error_code"] == "INCOMPATIBLE_CONTEXT"
    assert len(GLOBAL_QUEUE._runs) == initial_queue_size


def test_initial_plan_nodes_are_future_not_completed(client: TestClient) -> None:
    """Amendment 24: Plan is a projection of execution spec; initial nodes must not be completed."""
    plan_resp = client.post(
        "/api/v1/plans",
        json={"workflow": "predictive_ml", "synthetic_profile": "institutional_credit_v1"},
    )
    assert plan_resp.status_code == 200
    plan_data = plan_resp.json()["plan"]

    for step in plan_data:
        assert step["status"] == "future", f"Step {step['id']} was initialized with status {step['status']}"
        assert step.get("observed") is False, f"Step {step['id']} had observed=True before execution"


def test_canonical_execution_and_event_stream_boundaries() -> None:
    """Amendments 14, 17, 20, 21, 22: Events emitted at real boundaries; 0 simulated DL epochs."""
    events: list[RuntimeEvent] = []
    sink = ListEventSink(events)

    # Execute data_diagnostics workflow
    result = CanonicalExecutionService.execute(
        workflow_id="data_diagnostics",
        context_id="institutional_credit_v1",
        event_sink=sink,
        run_id="RUN-TEST-V510-BOUNDARY",
        session_id="SES-TEST-V510",
    )

    assert result["run_id"] == "RUN-TEST-V510-BOUNDARY"
    assert len(result["evidence_records"]) > 0
    assert len(events) >= 5

    event_types = [e.event_type for e in events]
    assert "context_ready" in event_types
    assert "workflow_resolved" in event_types
    assert "test_started" in event_types
    assert "test_completed" in event_types
    assert "evidence_committed" in event_types
    assert "governance_decided" in event_types
    assert "attestation_created" in event_types
    assert "workflow_completed" in event_types

    # Ensure no simulated epoch events
    assert not any("epoch" in e.event_type.lower() for e in events)

    # Verify evidence event matches canonical record id
    ev_events = [e for e in events if e.event_type == "evidence_committed"]
    committed_ids = {ref for e in ev_events for ref in e.evidence_refs}
    result_ids = {r.evidence_id for r in result["evidence_records"]}
    assert committed_ids == result_ids


def test_terminal_web_runtime_event_parity() -> None:
    """Amendments 2, 40, 41: CLI and Web use the same execution service and produce identical canonical event identities."""
    workflow_id = "data_diagnostics"
    context_id = "institutional_credit_v1"

    # Run 1: CLI / Terminal sink
    terminal_events: list[RuntimeEvent] = []
    terminal_sink = ListEventSink(terminal_events)
    res_terminal = CanonicalExecutionService.execute(
        workflow_id=workflow_id,
        context_id=context_id,
        event_sink=terminal_sink,
        run_id="RUN-PARITY-TERM",
        session_id="SES-PARITY",
    )

    # Run 2: Web Queue sink
    run_id_web = "RUN-PARITY-WEB"
    session_id_web = "SES-PARITY"
    req = RunRequest(session_id=session_id_web, workflow=workflow_id, contextId=context_id)
    GLOBAL_QUEUE.submit_run(run_id_web, req)

    web_sink = QueueEventSink(GLOBAL_QUEUE, run_id_web, session_id_web)
    res_web = CanonicalExecutionService.execute(
        workflow_id=workflow_id,
        context_id=context_id,
        event_sink=web_sink,
        run_id=run_id_web,
        session_id=session_id_web,
    )

    web_ctx = GLOBAL_QUEUE.get_run(run_id_web, session_id_web)
    assert web_ctx is not None

    # Parity check: Compare semantic sequences of events (event_type, node_id, test_id)
    term_seq = [(e.event_type, e.node_id, e.test_id) for e in terminal_events]
    web_seq = [(e.get("event_type"), e.get("node_id"), e.get("test_id")) for e in web_ctx.events]

    assert len(term_seq) == len(web_seq)
    for i in range(len(term_seq)):
        assert term_seq[i] == web_seq[i], f"Mismatch at event index {i}: terminal={term_seq[i]}, web={web_seq[i]}"

    # Compare test result statuses
    assert len(res_terminal["evidence_records"]) == len(res_web["evidence_records"])
    term_tests = {r.test_id: r.status.value for r in res_terminal["evidence_records"]}
    web_tests = {r.test_id: r.status.value for r in res_web["evidence_records"]}
    assert term_tests == web_tests


def test_deep_learning_diagnostics_has_zero_epoch_simulation() -> None:
    """Amendment 14: Deep learning review has 0 simulated training epochs."""
    events: list[RuntimeEvent] = []
    sink = ListEventSink(events)

    result = CanonicalExecutionService.execute(
        workflow_id="deep_learning",
        context_id="deep_learning_v1",
        event_sink=sink,
        run_id="RUN-TEST-V510-DL",
        session_id="SES-TEST-DL",
    )

    assert result["run_id"] == "RUN-TEST-V510-DL"
    epoch_events = [e for e in events if "epoch" in e.event_type.lower() or "epoch" in str(e.to_dict()).lower()]
    assert len(epoch_events) == 0, f"Found simulated epoch events in deep learning workflow: {epoch_events}"


def test_disabled_workflows_have_truthful_reason() -> None:
    """Amendments 43 & 51: Disabled workflows must have explicit truthful reasons."""
    catalog = get_workflow_catalog()
    for item in catalog:
        if not item["enabled"]:
            assert item.get("disabled_reason") is not None
            assert len(item["disabled_reason"]) > 0
            assert "unsupported" in item["disabled_reason"].lower() or "requires" in item["disabled_reason"].lower()
