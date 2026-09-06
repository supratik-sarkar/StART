"""StART v5.1.1 — Measured Acceptance, Deployment Parity & Provenance Closure Gates.

Validates the binding requirements for v5.1.1:
1. ARTIFACT_PRODUCER_GUESSES == 0
2. SYNTHETIC_FIXTURE_NUMERIC_TRUTH_AUDITED == PASS
3. CONTEXT_SEMANTIC_LABELS_TRUTHFUL == PASS
4. ACCEPTANCE_SELF_DECLARED_PASS_GATES == 0 (hardened semantic AST analyzer)
5. CHILD_EVIDENCE_OWNERSHIP_MEASURED == PASS
6. GRAPH_COMPARISON_EXACT_PARITY == PASS (extra/missing nodes & edges all 0)
7. SERVER_REVIEW_GATE == PASS
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from start.data.synthetic_dl import generate_dl_world
from start.runtime.contexts import get_canonical_context_specs
from start.runtime.events import ListEventSink
from start.runtime.execution import CanonicalExecutionService
from start.web.app import create_app
from start.web.queue import GLOBAL_QUEUE
from start.web.routes_workbench import get_workflow_definition, make_canonical_plan
from start.web.schemas import RunRequest


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_artifact_producer_guesses_is_zero() -> None:
    """Requirement 12: ARTIFACT_PRODUCER_GUESSES == 0.

    Execution must not assign producing_step_id = prev_node_id.
    If producer cannot be established, producing_step_id must be None,
    and no graph producer edge should be generated.
    """
    sink = ListEventSink()
    result = CanonicalExecutionService.execute(
        workflow_id="predictive_ml",
        context_id="institutional_credit_v1",
        event_sink=sink,
        run_id="TEST-ARTIFACT-PROVENANCE",
    )

    # Inspect emitted artifact events
    art_events = [e for e in sink.events if e.event_type == "artifact_created"]
    assert len(art_events) > 0, "Artifact creation events expected"

    for ev in art_events:
        # For review artifacts generated without step provenance, node_id must be None
        assert ev.node_id is None, (
            f"Artifact event {ev.artifact_refs} has guessed node_id={ev.node_id!r}; "
            "must be None when producer cannot be authoritatively established."
        )


def test_synthetic_fixture_numeric_truth_audited() -> None:
    """Requirement 13: SYNTHETIC_FIXTURE_NUMERIC_TRUTH_AUDITED == PASS.

    Uncomputed metrics must be stripped:
    - cv_5fold_auroc_mean (REMOVE)
    - cv_5fold_auroc_std (REMOVE)
    - subgroup_max_disparity (REMOVE)
    - feature_importance_stability_rank_corr (REMOVE)
    - best_epoch must be computed from actual val_loss minimum
    - loss history must not contain fallback constants
    """
    world = generate_dl_world(n_samples=500, n_features=8, seed=42)

    sens_meta = world["sensitivity_metadata"]
    assert "cv_5fold_auroc_mean" not in sens_meta, "cv_5fold_auroc_mean must be removed (not computed)"
    assert "cv_5fold_auroc_std" not in sens_meta, "cv_5fold_auroc_std must be removed (not computed)"
    assert "subgroup_max_disparity" not in sens_meta, "subgroup_max_disparity must be removed (not computed)"

    expl_meta = world["explainability_metadata"]
    assert "feature_importance_stability_rank_corr" not in expl_meta, (
        "feature_importance_stability_rank_corr must be removed (not computed)"
    )

    # Verify best_epoch is computed dynamically
    hist = world["history"]
    expected_best = int(np.argmin(hist["val_loss"]) + 1)
    arch_meta = world["architecture_metadata"]
    assert arch_meta["best_epoch"] == expected_best, (
        f"best_epoch {arch_meta['best_epoch']} does not match computed argmin(val_loss)={expected_best}"
    )
    assert f"best_epoch={expected_best}" in arch_meta["early_stopping"]


def test_context_semantic_labels_truthful() -> None:
    """Requirement 14: CONTEXT_SEMANTIC_LABELS_TRUTHFUL == PASS.

    institutional_credit_v1 must be named neutrally as a synthetic binary classification benchmark
    without ungrounded 'credit default' domain claims.
    """
    spec_map = {s.id: s for s in get_canonical_context_specs()}
    credit_spec = spec_map["institutional_credit_v1"]

    assert credit_spec.label == "Synthetic Binary Classification Benchmark"
    assert "credit default" not in credit_spec.description.lower()
    assert "credit" not in credit_spec.badges
    assert "benchmark" in credit_spec.badges


def test_hardened_ast_acceptance_analyzer_catches_violations() -> None:
    """Requirement 10: ACCEPTANCE_SELF_DECLARED_PASS_GATES == 0.

    Verify the hardened semantic AST checker rejects:
    - literal booleans
    - literal success strings ('PASS', 'OK', 'SUCCESS', 'true')
    - literal numbers (0, 0.0)
    - returncode-only ternary expressions
    - constant predicate lambdas
    - missing runtime artifact citation
    """
    from scripts.run_v511_acceptance import check_ast_gate_integrity

    snippet_bad_bool = """
def test_fake():
    record_gate("TEST_GATE", True, lambda x: x, "HTTP response")
"""
    violations = check_ast_gate_integrity(ast.parse(snippet_bad_bool))
    assert any("Boolean literal" in v[1] for v in violations)

    snippet_bad_str = """
def test_fake():
    record_gate("TEST_GATE", "PASS", lambda x: x == "PASS", "HTTP response")
"""
    violations = check_ast_gate_integrity(ast.parse(snippet_bad_str))
    assert any("Literal success string" in v[1] for v in violations)

    snippet_bad_num = """
def test_fake():
    record_gate("TEST_GATE", 0, lambda x: x == 0, "HTTP response")
"""
    violations = check_ast_gate_integrity(ast.parse(snippet_bad_num))
    assert any("Literal zero number" in v[1] for v in violations)

    snippet_bad_lambda = """
def test_fake():
    record_gate("TEST_GATE", observed_var, lambda x: True, "HTTP response")
"""
    violations = check_ast_gate_integrity(ast.parse(snippet_bad_lambda))
    assert any("Constant lambda" in v[1] for v in violations)

    snippet_bad_source = """
def test_fake():
    record_gate("TEST_GATE", observed_var, lambda x: x > 0, "Self declaration")
"""
    violations = check_ast_gate_integrity(ast.parse(snippet_bad_source))
    assert any("Source must cite a concrete runtime artifact" in v[1] for v in violations)


def test_server_review_gating_hydration(client: TestClient) -> None:
    """Requirement 7: SERVER_REVIEW_GATE == PASS.

    Proves server gate hydrates real metrics, enforces OPA gating, and rejects unknown evidence IDs.
    """
    # 1. Launch a real run
    session_id = "TEST-SES"
    resp = client.post(
        "/api/v1/runs",
        json={
            "workflow": "predictive_ml",
            "synthetic_profile": "institutional_credit_v1",
            "session_id": session_id,
        },
    )
    assert resp.status_code == 200
    run_id = resp.json()["data"]["run_id"]

    # Wait for execution completion
    for _ in range(30):
        time.sleep(0.5)
        st_resp = client.get(f"/api/v1/runs/{run_id}")
        if st_resp.status_code == 200 and st_resp.json()["data"]["phase"] == "completed":
            break

    # Get evidence records
    ev_resp = client.get(f"/api/v1/runs/{run_id}/evidence")
    assert ev_resp.status_code == 200
    evidence_list = ev_resp.json()["data"]["evidence_records"]
    assert len(evidence_list) > 0
    valid_ev_id = evidence_list[0]["evidence_id"]

    # 2. Submit reviewer output with valid evidence ID
    submission_valid = {
        "run_id": run_id,
        "session_id": "TEST-SES",
        "model_name": "SmolLM2-1.7B-Instruct-q4f16_1-MLC",
        "executive_summary": "Synthesized evidence review.",
        "findings": [
            {
                "finding_id": "FIND-01",
                "severity": "MEDIUM",
                "title": "Valid Observation",
                "description": "Evidence grounded.",
                "evidence_refs": [{"evidence_id": valid_ev_id, "metric_name": ""}],
                "recommendation": "Proceed.",
            }
        ],
        "limitations": [],
        "suggested_actions": [],
    }

    gate_resp = client.post(f"/api/v1/runs/{run_id}/reviewer/hydrate-and-gate", json=submission_valid)
    assert gate_resp.status_code == 200
    data = gate_resp.json()["data"]
    assert data["all_grounded"] is True
    assert len(data["hydrated_findings"]) == 1
    assert data["hydrated_findings"][0]["grounded"] is True

    # 3. Submit reviewer output with UNKNOWN evidence ID -> Must be flagged UNGROUNDED
    submission_invalid = {
        "run_id": run_id,
        "session_id": "TEST-SES",
        "model_name": "SmolLM2-1.7B-Instruct-q4f16_1-MLC",
        "executive_summary": "Fabricated review.",
        "findings": [
            {
                "finding_id": "FIND-02",
                "severity": "HIGH",
                "title": "Fabricated Observation",
                "description": "Citing nonexistent evidence.",
                "evidence_refs": [{"evidence_id": "EV-NONEXISTENT-999", "metric_name": ""}],
                "recommendation": "Reject.",
            }
        ],
        "limitations": [],
        "suggested_actions": [],
    }

    bad_gate_resp = client.post(f"/api/v1/runs/{run_id}/reviewer/hydrate-and-gate", json=submission_invalid)
    assert bad_gate_resp.status_code == 200
    bad_data = bad_gate_resp.json()["data"]
    assert bad_data["all_grounded"] is False
    assert bad_data["hydrated_findings"][0]["grounded"] is False


def test_child_evidence_ownership_must_fetch_child_evidence(client: TestClient) -> None:
    """Requirement 8: CHILD_EVIDENCE_OWNERSHIP_MEASURED == PASS.

    Parent run -> action -> child run -> GET child evidence:
    - child_run_id != parent_run_id
    - child_evidence_count > 0
    - every child EvidenceRecord.run_id == child_run_id
    - no parent EvidenceRecord returned by child endpoint
    """
    # 1. Launch parent run
    p_resp = client.post(
        "/api/v1/runs",
        json={"workflow": "predictive_ml", "synthetic_profile": "institutional_credit_v1"},
    )
    assert p_resp.status_code == 200
    parent_run_id = p_resp.json()["data"]["run_id"]

    for _ in range(30):
        time.sleep(0.5)
        st = client.get(f"/api/v1/runs/{parent_run_id}")
        if st.status_code == 200 and st.json()["data"]["phase"] == "completed":
            break

    p_ev_resp = client.get(f"/api/v1/runs/{parent_run_id}/evidence")
    p_records = p_ev_resp.json()["data"]["evidence_records"]
    assert len(p_records) > 0
    p_ev_ids = {r["evidence_id"] for r in p_records}

    # 2. Submit action to create child run
    act_resp = client.post(
        f"/api/v1/runs/{parent_run_id}/actions",
        json={
            "kind": "rerun",
            "label": "Rerun with tuned threshold",
            "parameters": {"threshold": 0.55},
            "sourceEvidenceId": p_records[0]["evidence_id"],
        },
    )
    assert act_resp.status_code == 200
    child_snapshot = act_resp.json()
    child_run_id = child_snapshot["runId"]

    assert child_run_id != parent_run_id, "child_run_id must be distinct from parent_run_id"

    # Wait for child completion
    for _ in range(30):
        time.sleep(0.5)
        c_st = client.get(f"/api/v1/runs/{child_run_id}")
        if c_st.status_code == 200 and c_st.json()["data"]["phase"] == "completed":
            break

    # 3. GET child evidence
    c_ev_resp = client.get(f"/api/v1/runs/{child_run_id}/evidence")
    assert c_ev_resp.status_code == 200
    c_records = c_ev_resp.json()["data"]["evidence_records"]
    assert len(c_records) > 0, "child_evidence_count must be > 0"

    # Verify ownership
    for rec in c_records:
        assert rec["run_id"] == child_run_id, f"Child evidence has wrong run_id: {rec['run_id']}"
        assert rec["evidence_id"] not in p_ev_ids, f"Parent evidence leaked into child: {rec['evidence_id']}"


def test_graph_comparison_exact_parity(client: TestClient) -> None:
    """Requirement 9: GRAPH COMPARISON MUST ACTUALLY COMPARE.

    Obtain canonical runtime semantic events and /graph.
    Calculate:
    - extra_observed_nodes
    - missing_observed_nodes
    - extra_observed_edges
    - missing_observed_edges
    Require all 4 sets empty.
    """
    # Launch a run
    resp = client.post(
        "/api/v1/runs",
        json={"workflow": "predictive_ml", "synthetic_profile": "institutional_credit_v1"},
    )
    run_id = resp.json()["data"]["run_id"]

    for _ in range(30):
        time.sleep(0.5)
        st = client.get(f"/api/v1/runs/{run_id}")
        if st.status_code == 200 and st.json()["data"]["phase"] == "completed":
            break

    ctx = GLOBAL_QUEUE.get_run(run_id)
    assert ctx is not None

    # Fetch /runs/{run_id}/graph
    g_resp = client.get(f"/api/v1/runs/{run_id}/graph")
    assert g_resp.status_code == 200
    graph = g_resp.json()

    # Observed nodes in returned graph
    actual_observed_node_ids = {n["id"] for n in graph["nodes"] if n.get("observed")}
    actual_observed_edge_ids = {e["id"] for e in graph["edges"] if e.get("edgeKind") == "observed"}

    # Build expected sets from canonical runtime events
    wdef = get_workflow_definition("predictive_ml")
    test_to_step = {tid: s_id for s_id, _, _, _, tids in wdef.step_specs for tid in tids}

    completed_step_ids = {
        ev.get("node_id")
        for ev in ctx.events
        if ev.get("node_id") and str(ev.get("status", "")).upper() in ("COMPLETED", "SUCCESS")
    }
    all_event_node_ids = {ev.get("node_id") for ev in ctx.events if ev.get("node_id")}
    observed_step_ids = completed_step_ids | all_event_node_ids

    expected_node_ids: set[str] = set(observed_step_ids)
    expected_edge_ids: set[str] = set()

    # Evidence nodes
    for r in ctx.evidence_records:
        expected_node_ids.add(r.evidence_id)
        step_id = test_to_step.get(r.test_id)
        if step_id and step_id in observed_step_ids:
            expected_edge_ids.add(f"edge-{step_id}-{r.evidence_id}")

    # Artifact nodes
    for art_id in ctx.artifacts:
        expected_node_ids.add(art_id)

    # Governance node (if disposition present)
    # Governance node (only if separate node generated when step-governance is not in plan)
    plan = make_canonical_plan("predictive_ml")
    pres = ctx.presentation or {}
    gov_disp = pres.get("governance_disposition")
    has_gov_step = any(s["id"] == "step-governance" for s in plan)
    last_step = plan[-1]["id"] if plan else None
    if gov_disp:
        if not has_gov_step and last_step:
            expected_node_ids.add("governance")
            expected_edge_ids.add(f"edge-{last_step}-governance")

    # Attestation node (if merkle root present)
    merkle_root = pres.get("attestation_seal_merkle_root")
    gov_node_id = "step-governance" if has_gov_step else ("governance" if gov_disp else last_step)
    if merkle_root and gov_node_id:
        expected_node_ids.add("attest")
        expected_edge_ids.add(f"edge-{gov_node_id}-attest")

    # Runtime event step-to-step edges
    seen_step_edges = set()
    for ev in ctx.events:
        pnid = ev.get("parent_node_id")
        nid = ev.get("node_id")
        if pnid and nid and pnid != nid:
            edge_key = (pnid, nid)
            if edge_key not in seen_step_edges and pnid in expected_node_ids and nid in expected_node_ids:
                seen_step_edges.add(edge_key)
                expected_edge_ids.add(f"edge-obs-{pnid}-{nid}")

    # Compute set differences
    extra_observed_nodes = actual_observed_node_ids - expected_node_ids
    missing_observed_nodes = expected_node_ids - actual_observed_node_ids
    extra_observed_edges = actual_observed_edge_ids - expected_edge_ids
    missing_observed_edges = expected_edge_ids - actual_observed_edge_ids

    assert len(extra_observed_nodes) == 0, f"extra_observed_nodes: {extra_observed_nodes}"
    assert len(missing_observed_nodes) == 0, f"missing_observed_nodes: {missing_observed_nodes}"
    assert len(extra_observed_edges) == 0, f"extra_observed_edges: {extra_observed_edges}"
    assert len(missing_observed_edges) == 0, f"missing_observed_edges: {missing_observed_edges}"
