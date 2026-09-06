"""Comprehensive Unit & Integration Test Suite for StART v4.5 Web API & Security."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from start.core.schemas import EvidenceRecord, Status
from start.web.app import create_app
from start.web.queue import GLOBAL_QUEUE
from start.web.schemas import (
    START_SCHEMA_VERSION,
    START_VERSION,
    EvidenceMetricRef,
    QualitativeFinding,
    RunRequest,
    WebReviewerSubmission,
)
from start.web.security import sanitize_artifact_id


@pytest.fixture
def client() -> TestClient:
    app = create_app()
    return TestClient(app)


def test_health_and_info_endpoints(client: TestClient) -> None:
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["data"]["status"] == "HEALTHY"
    assert data["data"]["version"] == START_VERSION
    assert data["data"]["schema_version"] == START_SCHEMA_VERSION

    info_resp = client.get("/api/v1/info")
    assert info_resp.status_code == 200
    info_data = info_resp.json()["data"]
    assert info_data["start_version"] == START_VERSION
    assert info_data["start_schema_version"] == START_SCHEMA_VERSION
    assert info_data["backend_build_version"] == f"{START_VERSION}-local"
    assert info_data["max_concurrency"] == 1


def test_zero_cost_attestation_removed(client: TestClient) -> None:
    """Amendment 26 & 28: Zero-cost attestation endpoint is removed."""
    resp = client.get("/api/v1/zero-cost-attestation")
    assert resp.status_code == 404



def test_synthetic_profiles_catalog(client: TestClient) -> None:
    resp = client.get("/api/v1/profiles")
    assert resp.status_code == 200
    profiles = resp.json()["data"]["profiles"]
    assert len(profiles) >= 3
    domains = {p["domain"] for p in profiles}
    assert "market" in domains
    assert "predictive" in domains


def test_run_submission_and_status(client: TestClient) -> None:
    req_payload = {
        "domain": "market",
        "mode": "deterministic",
        "materiality": "high",
        "synthetic_profile": "institutional_market_v1",
        "session_id": "SES-TEST-001",
    }
    resp = client.post("/api/v1/runs/start", json=req_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    run_id = data["run_id"]
    assert run_id.startswith("RUN-WEB-")

    # Check status
    st_resp = client.get(f"/api/v1/runs/{run_id}/status?session_id=SES-TEST-001")
    assert st_resp.status_code == 200
    st_data = st_resp.json()["data"]
    assert st_data["run_id"] == run_id
    assert st_data["status"] in ("QUEUED", "RUNNING", "COMPLETED")


def test_untrusted_reviewer_hydration_and_opa_gate(client: TestClient) -> None:
    # Setup a mock run with known evidence records
    run_id = "RUN-TEST-HYDRATE-01"
    session_id = "SES-HYDRATE-01"
    req = RunRequest(session_id=session_id, domain="market")
    GLOBAL_QUEUE.submit_run(run_id, req)

    rec1 = EvidenceRecord(
        evidence_id="EV-TEST-001",
        test_id="traded_risk.var_kupiec_pof",
        test_name="VaR Kupiec POF",
        model_id="MOD-01",
        dataset_id="DS-01",
        run_id=run_id,
        status=Status.PASS,
        metrics={"empirical_exceptions": 3, "p_value": 0.42},
    )
    rec2 = EvidenceRecord(
        evidence_id="EV-TEST-002",
        test_id="portfolio.sharpe_ratio",
        test_name="Sharpe Ratio",
        model_id="MOD-01",
        dataset_id="DS-01",
        run_id=run_id,
        status=Status.PASS,
        metrics={"sharpe": 1.85},
    )

    GLOBAL_QUEUE.mark_completed(
        run_id=run_id,
        evidence_records=[rec1, rec2],
        presentation={"run_id": run_id, "domains": ["market"], "blocks": {}},
    )

    # Browser WebLLM submits qualitative findings with metric references
    submission = WebReviewerSubmission(
        run_id=run_id,
        session_id=session_id,
        model_name="Llama-3.2-1B-Instruct-q4f32_1-MLC",
        executive_summary="Reviewer observes sound VaR calibration.",
        findings=[
            QualitativeFinding(
                finding_id="F-01",
                client_proposed_severity="LOW",
                title="VaR Exceedances Acceptable",
                description="Kupiec test indicates failure rate is within nominal coverage.",
                evidence_refs=[
                    EvidenceMetricRef(
                        evidence_id="EV-TEST-001",
                        metric_name="empirical_exceptions",
                        client_claimed_value=99999.0,
                    ),
                    EvidenceMetricRef(
                        evidence_id="EV-TEST-001",
                        metric_name="p_value",
                    ),
                ],
                recommendation="Maintain current VaR window.",
            ),
            QualitativeFinding(
                finding_id="F-02",
                client_proposed_severity="MEDIUM",
                title="Ungrounded Finding Test",
                description="Claim references nonexistent evidence.",
                evidence_refs=[
                    EvidenceMetricRef(evidence_id="EV-FAKE-999", metric_name="missing_metric"),
                ],
            ),
        ],
    )

    resp = client.post(f"/api/v1/runs/{run_id}/reviewer/hydrate-and-gate", json=submission.model_dump())
    assert resp.status_code == 200
    data = resp.json()["data"]

    assert data["run_id"] == run_id
    assert data["model_name"] == "Llama-3.2-1B-Instruct-q4f32_1-MLC"

    findings = data["hydrated_findings"]
    assert len(findings) == 2

    # F-01 was fully grounded: client numeric claim is untrusted; server hydrates canonical value
    f1 = findings[0]
    assert f1["grounded"] is True
    # CLIENT_NUMERIC_VALUE_TRUSTED = NO & SERVER_HYDRATED_VALUE = CANONICAL_EVIDENCE_VALUE
    assert f1["evidence_refs"][0]["client_claimed_value"] == 99999.0
    assert f1["evidence_refs"][0]["server_hydrated_value"] == 3
    assert f1["evidence_refs"][0]["canonical_value"] == 3
    assert f1["evidence_refs"][0]["grounding_status"] == "GROUNDED"

    assert f1["evidence_refs"][1]["server_hydrated_value"] == 0.42
    assert f1["evidence_refs"][1]["canonical_value"] == 0.42
    assert f1["evidence_refs"][1]["grounding_status"] == "GROUNDED"

    # F-02 had ungrounded evidence
    f2 = findings[1]
    assert f2["grounded"] is False
    assert f2["evidence_refs"][0]["grounding_status"] == "UNGROUNDED_EVIDENCE_ID"
    assert f2["evidence_refs"][0]["server_hydrated_value"] is None

    # Overall grounding reflects ungrounded F-02 citation
    assert data["all_grounded"] is False
    assert data["gate_status"] == "BLOCKED"

    # REVIEW_GATE_DEFAULT_ACCEPT = 0 & REVIEWER_ROUTE_OWNS_GOVERNANCE_SEMANTICS = NO:
    # Reviewer route does not fabricate an ACCEPT governance disposition
    assert data["governance_disposition"] is None

    # SYNTHETIC_ATTESTATION_FALLBACK = 0: No fake Merkle root emitted on ungrounded/unverified review
    assert data["attestation_seal_merkle_root"] is None


def test_artifact_id_sanitization_and_security() -> None:
    # Valid IDs pass
    assert sanitize_artifact_id("ART-CHART-01") == "ART-CHART-01"
    assert sanitize_artifact_id("dendrogram_svg_123") == "dendrogram_svg_123"

    # Traversal attempts are rejected
    with pytest.raises(ValueError, match="path traversal"):
        sanitize_artifact_id("../../../etc/passwd")

    with pytest.raises(ValueError, match="path traversal"):
        sanitize_artifact_id("foo/bar")

    with pytest.raises(ValueError, match="path traversal"):
        sanitize_artifact_id("..\\windows\\system32")


def test_pdf_generation_endpoint(client: TestClient) -> None:
    run_id = "RUN-TEST-PDF-01"
    session_id = "SES-PDF-01"
    req = RunRequest(session_id=session_id, domain="market")
    GLOBAL_QUEUE.submit_run(run_id, req)
    GLOBAL_QUEUE.mark_completed(
        run_id=run_id,
        presentation={
            "run_id": run_id,
            "domains": ["market"],
            "governance_disposition": "ACCEPT",
            "attestation_seal_merkle_root": "a1b2c3d4e5f67890",
            "blocks": {
                "PORTFOLIO": {
                    "title": "Portfolio Construction",
                    "domain": "market",
                    "rows": [
                        {"test_id": "hrp_weights", "metric": "effective_n", "value": 14.2, "status": "PASS"},
                    ],
                }
            },
        },
    )

    pdf_resp = client.get(f"/api/v1/runs/{run_id}/pdf?session_id={session_id}")
    assert pdf_resp.status_code == 200
    assert pdf_resp.headers["Content-Type"] == "application/pdf"
    assert pdf_resp.content.startswith(b"%PDF-1.4")


def test_fastapi_route_uniqueness(client: TestClient) -> None:
    """Amendment 2: Assert ZERO ambiguous or duplicate FastAPI path+method registrations."""
    all_endpoints = []
    app = client.app
    for r in app.router.routes:
        if hasattr(r, "original_router"):
            prefix = r.include_context.prefix if hasattr(r, "include_context") else ""
            for sub in r.original_router.routes:
                sub_path = getattr(sub, "path", "")
                full_path = f"{prefix}{sub_path}".replace("//", "/")
                for m in getattr(sub, "methods", ["ALL"]):
                    all_endpoints.append((m, full_path))
        elif hasattr(r, "methods") and hasattr(r, "path"):
            for m in r.methods:
                all_endpoints.append((m, r.path))

    from collections import Counter
    counts = Counter(all_endpoints)
    duplicates = [k for k, v in counts.items() if v > 1]
    assert len(duplicates) == 0, f"Found duplicate route registrations: {duplicates}"


def test_capabilities_truthful_catalog(client: TestClient) -> None:
    """Amendment 1: Capabilities must not hardcode all 10 as enabled; model_comparison disabled with reason."""
    resp = client.get("/api/v1/capabilities")
    assert resp.status_code == 200
    caps = resp.json()
    assert isinstance(caps, list)
    assert len(caps) == 10

    by_id = {c["id"]: c for c in caps}
    assert by_id["predictive_ml"]["enabled"] is True
    assert by_id["deep_learning"]["enabled"] is True
    assert by_id["quantitative_finance"]["enabled"] is True

    # model_comparison must be disabled with machine-readable reason
    mc = by_id["model_comparison"]
    assert mc["enabled"] is False
    assert "disabledReason" in mc
    assert len(mc["disabledReason"]) > 10


def test_execution_contexts_catalog(client: TestClient) -> None:
    """Amendment 19: Real execution context catalog returns public-safe contexts."""
    resp = client.get("/api/v1/execution-contexts")
    assert resp.status_code == 200
    contexts = resp.json()
    assert isinstance(contexts, list)
    assert len(contexts) >= 3
    ctx_ids = {c["id"] for c in contexts}
    assert "institutional_credit_v1" in ctx_ids
    assert "deep_learning_v1" in ctx_ids
    assert "institutional_market_v1" in ctx_ids


def test_create_agent_plan_no_fake_run_id(client: TestClient) -> None:
    """Amendment 3 & 4: createPlan must return AgentPlanPreview without inventing runId."""
    plan_req = {
        "workflowId": "predictive_ml",
        "contextId": "institutional_credit_v1",
        "goal": "Verify credit default model calibration",
    }
    resp = client.post("/api/v1/plans", json=plan_req)
    assert resp.status_code == 200
    plan_data = resp.json()

    # Must NOT have runId, startedAt, or elapsedMs
    assert "runId" not in plan_data
    assert "startedAt" not in plan_data
    assert "elapsedMs" not in plan_data

    # Must have AgentPlanPreview shape
    assert plan_data["workflowId"] == "predictive_ml"
    assert plan_data["contextId"] == "institutional_credit_v1"
    assert "plan" in plan_data
    assert len(plan_data["plan"]) >= 5
    assert plan_data["plan"][0]["kind"] in ("context", "test", "agent", "tool", "evidence", "governance")


def test_graph_lineage_and_action_validation(client: TestClient) -> None:
    """Amendment 7, 15, 32: Graph distinction, action validation boundary, and child lineage."""
    run_id = "RUN-PARENT-01"
    session_id = "SES-PARENT-01"
    req = RunRequest(session_id=session_id, workflow="predictive_ml", synthetic_profile="institutional_credit_v1")
    GLOBAL_QUEUE.submit_run(run_id, req)

    rec = EvidenceRecord(
        evidence_id="EV-PARENT-001",
        test_id="calibration.brier_score",
        test_name="Brier Score",
        model_id="MOD-01",
        dataset_id="DS-01",
        run_id=run_id,
        status=Status.WARN,
        metrics={"brier_score": 0.28},
    )
    GLOBAL_QUEUE.mark_completed(
        run_id=run_id,
        evidence_records=[rec],
        presentation={"run_id": run_id, "domains": ["predictive"], "blocks": {}},
    )

    # 1. Graph returns nodes with planned vs completed states
    g_resp = client.get(f"/api/v1/runs/{run_id}/graph?session_id={session_id}")
    assert g_resp.status_code == 200
    graph = g_resp.json()
    assert len(graph["nodes"]) >= 5
    assert len(graph["edges"]) >= 4

    # 2. Findings derived from ATTENTION record
    f_resp = client.get(f"/api/v1/runs/{run_id}/findings?session_id={session_id}")
    assert f_resp.status_code == 200
    findings = f_resp.json()
    assert len(findings) >= 1
    assert "EV-PARENT-001" in findings[0]["evidenceIds"]
    assert findings[0]["sourceNodeId"] not in ("branch-a", "branch-b")

    # 3. Action validation boundary: fail-closed on invalid parameters, accept valid parameters
    invalid_act_req = {
        "kind": "deeper_test",
        "sourceEvidenceId": "EV-PARENT-001",
        "parameters": {"depth": "focused", "perturbation_rate": 0.50},  # 0.50 is out of bounds (> 0.30)
    }
    inv_resp = client.post(f"/api/v1/runs/{run_id}/actions/validate?session_id={session_id}", json=invalid_act_req)
    assert inv_resp.status_code == 400
    assert "out of allowed bounds" in inv_resp.json()["detail"].lower()

    valid_act_req = {
        "kind": "deeper_test",
        "sourceEvidenceId": "EV-PARENT-001",
        "parameters": {"depth": "focused", "perturbation_rate": 0.20},
    }
    v_resp = client.post(f"/api/v1/runs/{run_id}/actions/validate?session_id={session_id}", json=valid_act_req)
    assert v_resp.status_code == 200
    val_action = v_resp.json()
    assert val_action["parameters"]["perturbation_rate"] == 0.20

    # 4. Execute validated human action -> child run with lineage
    exec_resp = client.post(f"/api/v1/runs/{run_id}/actions?session_id={session_id}", json=val_action)
    assert exec_resp.status_code == 200
    child_snapshot = exec_resp.json()
    assert child_snapshot["runId"] != run_id
    assert child_snapshot["parentRunId"] == run_id

    # 5. Child graph includes parent lineage
    child_run_id = child_snapshot["runId"]
    cg_resp = client.get(f"/api/v1/runs/{child_run_id}/graph?session_id={session_id}")
    assert cg_resp.status_code == 200
    c_graph = cg_resp.json()
    node_ids = {n["id"] for n in c_graph["nodes"]}
    assert "parent-run" in node_ids


def test_idor_cross_session_isolation(client: TestClient) -> None:
    """Amendment 23: Run endpoints must enforce session isolation."""
    run_id = "RUN-ISOLATED-01"
    session_owner = "SES-OWNER"
    req = RunRequest(session_id=session_owner, workflow="predictive_ml")
    GLOBAL_QUEUE.submit_run(run_id, req)

    # Owner can access
    resp = client.get(f"/api/v1/runs/{run_id}/status?session_id={session_owner}")
    assert resp.status_code == 200

    # Attacker session denied (404 / access denied)
    bad_resp = client.get(f"/api/v1/runs/{run_id}/status?session_id=SES-ATTACKER")
    assert bad_resp.status_code == 404

