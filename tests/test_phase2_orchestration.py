"""Tests for Phase 2 Live Agent Orchestration, Checkpoints, Human Control, and Artifacts."""

from fastapi.testclient import TestClient

from start.runtime.execution import CanonicalExecutionService
from start.web.app import create_app
from start.web.queue import GLOBAL_QUEUE
from start.web.schemas import RunRequest


def test_canonical_execution_checkpoints_and_artifacts():
    """Verify CanonicalExecutionService emits CP-001 through CP-006 and enriches artifacts."""
    res = CanonicalExecutionService.execute(
        workflow_id="predictive_ml",
        context_id="institutional_credit_v1",
        materiality="TIER_1",
    )

    assert res is not None
    assert len(res.records) > 0

    # Verify checkpoints
    cp_ids = [c["checkpoint_id"] for c in res.checkpoints]
    assert "CP-001" in cp_ids
    assert "CP-002" in cp_ids
    assert "CP-003" in cp_ids
    assert "CP-004" in cp_ids
    assert "CP-005" in cp_ids
    assert "CP-006" in cp_ids

    # Verify artifacts enrichment
    assert len(res.artifacts) > 0
    for _art_id, art_data in res.artifacts.items():
        assert "title" in art_data
        assert "kind" in art_data
        assert "producing_step_id" in art_data
        assert "data_fingerprint" in art_data


def test_phase2_web_endpoints(monkeypatch):
    """Verify workbench endpoints: checkpoints, decisions, question, artifacts."""
    from unittest.mock import MagicMock

    from start.providers.base import ProviderResult

    # Mock LLM provider so question endpoint succeeds offline
    mock_prov = MagicMock()
    mock_prov.available = True
    mock_prov.name = "openai"
    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: mock_prov)
    monkeypatch.setattr("start.providers.keys.ensure_provider_key", lambda *args, **kwargs: MagicMock(source="mock"))

    app = create_app()
    client = TestClient(app)

    # Execute a run
    run_id = "RUN-TEST-P2-01"
    req = RunRequest(
        domain="predictive",
        workflow="predictive_ml",
        synthetic_profile="institutional_credit_v1",
        session_id="sess-test-01",
    )
    GLOBAL_QUEUE.submit_run(run_id, req)
    GLOBAL_QUEUE.mark_running(run_id)

    res = CanonicalExecutionService.execute(
        workflow_id="predictive_ml",
        context_id="institutional_credit_v1",
        run_id=run_id,
    )
    GLOBAL_QUEUE.mark_completed(
        run_id=run_id,
        presentation=res.presentation_model.to_dict() if res.presentation_model else {},
        artifacts=res.artifacts,
        evidence_records=res.records,
        checkpoints=res.checkpoints,
    )

    # 1. Test GET /api/v1/runs/{run_id}/checkpoints
    cp_resp = client.get(f"/api/v1/runs/{run_id}/checkpoints")
    assert cp_resp.status_code == 200
    checkpoints = cp_resp.json()
    assert len(checkpoints) >= 6
    assert any(c["checkpoint_id"] == "CP-001" for c in checkpoints)
    assert any(c["checkpoint_id"] == "CP-006" for c in checkpoints)

    # 2. Test GET /api/v1/runs/{run_id}/artifacts
    art_resp = client.get(f"/api/v1/runs/{run_id}/artifacts")
    assert art_resp.status_code == 200
    artifacts = art_resp.json()
    assert len(artifacts) > 0
    first_art = artifacts[0]
    assert "artifactId" in first_art
    assert "kind" in first_art
    assert "content" in first_art

    # 3. Test POST /api/v1/runs/{run_id}/decisions (Accept / Challenge)
    dec_payload = {
        "action": "CHALLENGE",
        "target_stage": "step-features",
        "target_checkpoint": "CP-003",
        "evidence_ids": [res.records[0].evidence_id],
        "rationale": "Challenging the stationarity assumption for feature drift under stressed macro regime.",
        "author": "Risk Officer",
    }
    dec_resp = client.post(f"/api/v1/runs/{run_id}/decisions", json=dec_payload)
    assert dec_resp.status_code == 200
    receipt = dec_resp.json()
    assert receipt["receipt_id"].startswith("REC-")
    assert receipt["action"] == "CHALLENGE"
    assert "decision_hash" in receipt
    assert receipt["immutable_evidence_preserved"] is True

    # 4. Test POST /api/v1/runs/{run_id}/question
    q_payload = {
        "question": "What is the discrimination score and why is the model acceptable?",
        "target_stage": "step-supervised",
        "evidence_id": res.records[0].evidence_id,
    }
    mock_prov.complete_result.return_value = ProviderResult(
        status="completed",
        text=f"The model discrimination is verified [{res.records[0].evidence_id}].",
        response_id="resp-mock-01",
        model="gpt-5.1",
        latency_seconds=0.1,
    )
    q_resp = client.post(f"/api/v1/runs/{run_id}/question", json=q_payload)
    assert q_resp.status_code == 200
    q_data = q_resp.json()
    assert "answer" in q_data
    assert "citations" in q_data
    assert len(q_data["citations"]) > 0
    assert q_data["citations"][0]["evidence_id"] == res.records[0].evidence_id
