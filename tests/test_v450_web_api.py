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
    assert data["data"]["schema_version"] == START_SCHEMA_VERSION

    info_resp = client.get("/api/v1/info")
    assert info_resp.status_code == 200
    info_data = info_resp.json()["data"]
    assert info_data["start_version"] == START_VERSION
    assert info_data["max_concurrency"] == 1


def test_zero_cost_attestation_endpoint(client: TestClient) -> None:
    resp = client.get("/api/v1/zero-cost-attestation")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert data["always_free_eligible"] is True
    assert data["recurring_monthly_charge_usd"] == 0.0
    assert data["attestation_status"] == "VERIFIED_ZERO_COST"


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
                severity="LOW",
                title="VaR Exceedances Acceptable",
                description="Kupiec test indicates failure rate is within nominal coverage.",
                evidence_refs=[
                    EvidenceMetricRef(evidence_id="EV-TEST-001", metric_name="empirical_exceptions"),
                    EvidenceMetricRef(evidence_id="EV-TEST-001", metric_name="p_value"),
                ],
                recommendation="Maintain current VaR window.",
            ),
            QualitativeFinding(
                finding_id="F-02",
                severity="MEDIUM",
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

    # F-01 was fully grounded
    f1 = findings[0]
    assert f1["grounded"] is True
    assert f1["evidence_refs"][0]["hydrated_value"] == 3
    assert f1["evidence_refs"][1]["hydrated_value"] == 0.42

    # F-02 had ungrounded evidence
    f2 = findings[1]
    assert f2["grounded"] is False
    assert f2["evidence_refs"][0]["status"] == "UNGROUNDED_EVIDENCE_ID"

    # Governance disposition evaluated server-side
    assert data["governance_disposition"] in ("ACCEPT", "CONDITIONAL_ACCEPT")
    assert len(data["attestation_seal_merkle_root"]) > 0


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
