"""
v5.0.2 Gate Acceptance Tests.

Validates the final binding amendments:
1. HEURISTIC_PLAN_EVENT_MATCHING = 0
2. PLANNED_VS_OBSERVED_GRAPH_DISTINCTION = PASS
3. EVIDENCE_WITH_UNKNOWN_PRODUCER_REMAINS_VISIBLE = PASS
4. ARTIFACT_WITH_UNKNOWN_PRODUCER_REMAINS_VISIBLE = PASS
5. STATUS_TO_SEVERITY_INVENTION = 0
6. ACTION_EXECUTION_SERVER_RESOLVED = PASS
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from start.core.schemas import EvidenceRecord, Status
from start.web.app import create_app
from start.web.queue import GLOBAL_QUEUE
from start.web.schemas import RunRequest


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_heuristic_plan_event_matching_is_zero(client: TestClient) -> None:
    """Amendment 1: Stable observed node identity; no heuristic plan matching."""
    run_id = "RUN-TEST-V502-01"
    session_id = "SES-TEST-V502-01"
    req = RunRequest(session_id=session_id, workflow="predictive_ml")
    GLOBAL_QUEUE.submit_run(run_id, req)
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    assert ctx is not None

    # Emit an event only for step-preflight
    ctx.events.append({
        "event_type": "STEP_COMPLETED",
        "node_id": "step-preflight",
        "status": "COMPLETED",
        "message": "Completed preflight step",
    })
    GLOBAL_QUEUE.mark_completed(run_id, evidence_records=[], presentation={})

    # Status / snapshot check
    status_resp = client.get(f"/api/v1/runs/{run_id}/status?session_id={session_id}")
    assert status_resp.status_code == 200
    snap = status_resp.json()["data"]
    step_map = {s["id"]: s for s in snap["plan"]}

    assert step_map["step-preflight"]["status"] == "completed"
    assert step_map["step-preflight"].get("observed") is True

    # Unexecuted plan steps must NOT be marked completed just because run completed
    assert step_map["step-features"]["status"] == "future"
    assert step_map["step-features"].get("observed") is False

    # Graph check
    graph_resp = client.get(f"/api/v1/runs/{run_id}/graph?session_id={session_id}")
    assert graph_resp.status_code == 200
    graph = graph_resp.json()
    node_map = {n["id"]: n for n in graph["nodes"]}

    assert node_map["step-preflight"]["status"] == "completed"
    assert node_map["step-preflight"].get("observed") is True
    assert node_map["step-features"]["status"] == "future"
    assert node_map["step-features"].get("observed") is False


def test_planned_vs_observed_graph_distinction(client: TestClient) -> None:
    """Amendment 2: Planned edges and observed edges are distinct concepts."""
    run_id = "RUN-TEST-V502-02"
    session_id = "SES-TEST-V502-02"
    req = RunRequest(session_id=session_id, workflow="predictive_ml")
    GLOBAL_QUEUE.submit_run(run_id, req)
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    assert ctx is not None

    ctx.events.extend([
        {
            "event_type": "STEP_COMPLETED",
            "node_id": "step-preflight",
            "status": "COMPLETED",
        },
        {
            "event_type": "STEP_COMPLETED",
            "node_id": "step-features",
            "parent_node_id": "step-preflight",
            "status": "COMPLETED",
        },
    ])
    GLOBAL_QUEUE.mark_completed(run_id, evidence_records=[], presentation={})

    graph_resp = client.get(f"/api/v1/runs/{run_id}/graph?session_id={session_id}")
    assert graph_resp.status_code == 200
    graph = graph_resp.json()

    planned_edges = [e for e in graph["edges"] if e.get("edgeKind") == "planned"]
    observed_edges = [e for e in graph["edges"] if e.get("edgeKind") == "observed"]

    assert len(planned_edges) >= 4
    assert len(observed_edges) >= 1

    # Observed edge specifically between step-preflight and step-features
    obs_1_2 = [e for e in observed_edges if e["source"] == "step-preflight" and e["target"] == "step-features"]
    assert len(obs_1_2) == 1
    assert obs_1_2[0]["edgeKind"] == "observed"


def test_evidence_and_artifact_with_unknown_producer_remains_visible(client: TestClient) -> None:
    """Amendment 4: Unknown producer means unknown; nodes remain visible with zero fabricated edges."""
    run_id = "RUN-TEST-V502-03"
    session_id = "SES-TEST-V502-03"
    req = RunRequest(session_id=session_id, workflow="predictive_ml")
    GLOBAL_QUEUE.submit_run(run_id, req)
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    assert ctx is not None

    rec_orphan = EvidenceRecord(
        evidence_id="EV-ORPHAN-001",
        test_id="unmapped.isolated_test",
        test_name="Unmapped Isolated Test",
        model_id="MOD-01",
        dataset_id="DS-01",
        run_id=run_id,
        status=Status.PASS,
        metrics={"score": 0.99},
    )

    ctx.artifacts["art-orphan-1"] = {
        "id": "art-orphan-1",
        "name": "Orphan Metric Table",
        "artifact_type": "table",
        "producing_step_id": None,  # Unknown producer
    }

    GLOBAL_QUEUE.mark_completed(run_id, evidence_records=[rec_orphan], presentation={})

    graph_resp = client.get(f"/api/v1/runs/{run_id}/graph?session_id={session_id}")
    assert graph_resp.status_code == 200
    graph = graph_resp.json()
    node_map = {n["id"]: n for n in graph["nodes"]}

    # Evidence node remains visible
    assert "EV-ORPHAN-001" in node_map
    ev_node = node_map["EV-ORPHAN-001"]
    assert ev_node.get("parentId") is None
    assert ev_node.get("subtitle") == "Producer lineage unavailable"

    # Artifact node remains visible
    assert "art-orphan-1" in node_map
    art_node = node_map["art-orphan-1"]
    assert art_node.get("parentId") is None
    assert art_node.get("subtitle") == "Producer lineage unavailable"

    # Zero fabricated edges pointing to orphan nodes
    fabricated_ev_edges = [e for e in graph["edges"] if e["target"] == "EV-ORPHAN-001"]
    fabricated_art_edges = [e for e in graph["edges"] if e["target"] == "art-orphan-1"]
    assert len(fabricated_ev_edges) == 0
    assert len(fabricated_art_edges) == 0


def test_status_to_severity_invention_is_zero(client: TestClient) -> None:
    """Amendment 6: Finding status and severity remain separate; zero severity invention."""
    run_id = "RUN-TEST-V502-04"
    session_id = "SES-TEST-V502-04"
    req = RunRequest(session_id=session_id, workflow="predictive_ml")
    GLOBAL_QUEUE.submit_run(run_id, req)

    rec_fail = EvidenceRecord(
        evidence_id="EV-FAIL-001",
        test_id="eda.missing_values",
        test_name="Missing Values",
        model_id="MOD-01",
        dataset_id="DS-01",
        run_id=run_id,
        status=Status.FAIL,
        metrics={"missing_pct": 0.45},
    )
    rec_warn = EvidenceRecord(
        evidence_id="EV-WARN-001",
        test_id="supervised.overfitting",
        test_name="Overfitting Detection",
        model_id="MOD-01",
        dataset_id="DS-01",
        run_id=run_id,
        status=Status.WARN,
        metrics={"train_test_gap": 0.22},
    )

    GLOBAL_QUEUE.mark_completed(run_id, evidence_records=[rec_fail, rec_warn], presentation={})

    resp = client.get(f"/api/v1/runs/{run_id}/findings?session_id={session_id}")
    assert resp.status_code == 200
    findings = resp.json()

    # Find the findings for EV-FAIL-001 and EV-WARN-001
    f_fail = next(f for f in findings if "EV-FAIL-001" in f["evidenceIds"])
    f_warn = next(f for f in findings if "EV-WARN-001" in f["evidenceIds"])

    # Test status preserved accurately
    assert f_fail["testStatus"] == "FAIL"
    assert f_warn["testStatus"] == "WARN"

    # Severity must NOT be invented
    assert f_fail.get("severity") is None
    assert f_warn.get("severity") is None


def test_action_execution_server_resolved(client: TestClient) -> None:
    """Amendments 10, 12, 13: Action execution resolution, rerun vs parameter delta semantics."""
    run_id = "RUN-TEST-V502-05"
    session_id = "SES-TEST-V502-05"
    req = RunRequest(session_id=session_id, workflow="predictive_ml", parameters={"perturbation_rate": 0.10})
    GLOBAL_QUEUE.submit_run(run_id, req)

    rec_parent = EvidenceRecord(
        evidence_id="EV-PARENT-001",
        test_id="eda.missing_values",
        test_name="Missing Values",
        model_id="MOD-01",
        dataset_id="DS-01",
        run_id=run_id,
        status=Status.WARN,
        metrics={"missing_pct": 0.15},
    )
    GLOBAL_QUEUE.mark_completed(run_id, evidence_records=[rec_parent], presentation={})

    # 1. Rerun with identical parameters succeeds and returns full resolution
    rerun_payload = {
        "kind": "rerun",
        "parameters": {"perturbation_rate": 0.10},
        "sourceEvidenceId": "EV-PARENT-001",
        "rationale": "Deterministic rerun",
    }
    resp_rerun = client.post(f"/api/v1/runs/{run_id}/actions?session_id={session_id}", json=rerun_payload)
    assert resp_rerun.status_code == 200
    res_rerun = resp_rerun.json()

    assert res_rerun["runId"] != run_id
    assert res_rerun["parentRunId"] == run_id
    assert res_rerun["action_kind"] == "rerun"
    assert "resolved_test_ids" in res_rerun
    assert "resolved_parameter_delta" in res_rerun
    assert res_rerun["source_evidence_id"] == "EV-PARENT-001"

    # 2. Change parameter requires genuine delta
    no_delta_payload = {
        "kind": "change_parameter",
        "parameters": {"perturbation_rate": 0.10},
        "sourceEvidenceId": "EV-PARENT-001",
        "rationale": "No delta provided",
    }
    resp_no_delta = client.post(f"/api/v1/runs/{run_id}/actions?session_id={session_id}", json=no_delta_payload)
    assert resp_no_delta.status_code == 400

    delta_payload = {
        "kind": "change_parameter",
        "parameters": {"perturbation_rate": 0.20},
        "sourceEvidenceId": "EV-PARENT-001",
        "rationale": "Increase perturbation rate",
    }
    resp_delta = client.post(f"/api/v1/runs/{run_id}/actions?session_id={session_id}", json=delta_payload)
    assert resp_delta.status_code == 200
    res_delta = resp_delta.json()
    assert res_delta["action_kind"] == "change_parameter"
    assert res_delta["resolved_parameter_delta"] == {"perturbation_rate": 0.20}

    # 3. Challenge rejected from creating child run
    chal_payload = {
        "kind": "challenge",
        "rationale": "Conversational only",
    }
    resp_chal = client.post(f"/api/v1/runs/{run_id}/actions?session_id={session_id}", json=chal_payload)
    assert resp_chal.status_code == 400
