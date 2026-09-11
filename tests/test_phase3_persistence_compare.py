"""Tests for Phase 3: Persistence, Reload, Deterministic Compare, Lineage, Search, and Regressions."""

import time
from pathlib import Path

from fastapi.testclient import TestClient

from start.registry import list_tests
from start.runtime.execution import CanonicalExecutionService
from start.web.app import create_app
from start.web.queue import GLOBAL_QUEUE
from start.web.schemas import RunRequest


def test_phase3_run_persistence_and_reload_across_restart():
    """Verify runs are persisted to disk and reloadable after process restart."""
    app = create_app()
    client = TestClient(app)

    run_id = f"RUN-TEST-P3-PERSIST-{int(time.time())}"
    req = RunRequest(
        domain="predictive",
        workflow="predictive_ml",
        synthetic_profile="institutional_credit_v1",
        session_id="sess-persist-01",
    )

    # 1. Execute run
    res = CanonicalExecutionService.execute(
        workflow_id="predictive_ml",
        context_id="institutional_credit_v1",
        run_id=run_id,
    )
    GLOBAL_QUEUE.submit_run(run_id, req)
    GLOBAL_QUEUE.mark_running(run_id)
    GLOBAL_QUEUE.mark_completed(
        run_id=run_id,
        presentation={"governance_disposition": res.governance_disposition or "PASS"},
        artifacts=res.artifacts,
        evidence_records=res.records,
        checkpoints=getattr(res, "checkpoints", []),
    )

    # Verify run_context.json exists on disk
    ctx_file = Path("start_output") / run_id / "run_context.json"
    assert ctx_file.exists(), f"Expected {ctx_file} to exist on disk!"

    # 2. Simulate complete backend process restart by clearing in-memory runs
    with GLOBAL_QUEUE._lock:
        GLOBAL_QUEUE._runs.clear()

    assert run_id not in GLOBAL_QUEUE._runs

    # 3. Verify get_run reloads from disk
    reloaded_ctx = GLOBAL_QUEUE.get_run(run_id)
    assert reloaded_ctx is not None
    assert reloaded_ctx.run_id == run_id
    assert reloaded_ctx.status == "COMPLETED"
    assert len(reloaded_ctx.evidence_records) > 0
    assert len(reloaded_ctx.artifacts) > 0

    # 4. Verify list_runs and get_run_history find persisted run
    history = GLOBAL_QUEUE.get_run_history()
    hist_item = next((h for h in history if h["run_id"] == run_id), None)
    assert hist_item is not None
    assert hist_item["workflow"] == "predictive_ml"
    assert hist_item["status"] == "completed"
    assert hist_item["evidence_count"] == len(reloaded_ctx.evidence_records)

    # 5. Verify HTTP API endpoints reload persisted run cleanly
    resp_run = client.get(f"/api/v1/runs/{run_id}")
    assert resp_run.status_code == 200
    assert resp_run.json()["data"]["run_id"] == run_id

    resp_arts = client.get(f"/api/v1/workbench/runs/{run_id}/artifacts")
    assert resp_arts.status_code == 200
    assert len(resp_arts.json()) > 0

    resp_hist = client.get("/api/v1/workbench/runs")
    assert resp_hist.status_code == 200
    assert any(r["run_id"] == run_id for r in resp_hist.json())


def test_phase3_deterministic_compare_and_compatibility():
    """Verify deterministic Run Compare enforces scientific compatibility and computes exact deltas."""
    app = create_app()
    client = TestClient(app)

    # Create Run A (predictive)
    run_a = f"RUN-TEST-CMP-A-{int(time.time())}"
    req_a = RunRequest(
        domain="predictive",
        workflow="predictive_ml",
        synthetic_profile="institutional_credit_v1",
        session_id="sess-cmp-01",
        parameters={"alpha": 0.05, "confidence": 0.95},
    )
    res_a = CanonicalExecutionService.execute(
        workflow_id="predictive_ml",
        context_id="institutional_credit_v1",
        run_id=run_a,
    )
    GLOBAL_QUEUE.submit_run(run_a, req_a)
    GLOBAL_QUEUE.mark_running(run_a)
    GLOBAL_QUEUE.mark_completed(
        run_id=run_a,
        presentation={"governance_disposition": res_a.governance_disposition or "PASS"},
        artifacts=res_a.artifacts,
        evidence_records=res_a.records,
        checkpoints=getattr(res_a, "checkpoints", []),
    )

    # Create Run B (child rerun of A with parameter override)
    run_b = f"RUN-TEST-CMP-B-{int(time.time())}"
    req_b = RunRequest(
        domain="predictive",
        workflow="predictive_ml",
        synthetic_profile="institutional_credit_v1",
        session_id="sess-cmp-01",
        parent_run_id=run_a,
        intervention="stress_testing",
        parameters={"alpha": 0.01, "confidence": 0.99},
    )
    res_b = CanonicalExecutionService.execute(
        workflow_id="predictive_ml",
        context_id="institutional_credit_v1",
        run_id=run_b,
    )
    GLOBAL_QUEUE.submit_run(run_b, req_b)
    GLOBAL_QUEUE.mark_running(run_b)
    GLOBAL_QUEUE.mark_completed(
        run_id=run_b,
        presentation={"governance_disposition": res_b.governance_disposition or "PASS"},
        artifacts=res_b.artifacts,
        evidence_records=res_b.records,
        checkpoints=getattr(res_b, "checkpoints", []),
    )

    # Create Run C (market workflow - incompatible)
    run_c = f"RUN-TEST-CMP-C-{int(time.time())}"
    req_c = RunRequest(
        domain="market",
        workflow="quantitative_finance",
        synthetic_profile="institutional_market_v1",
        session_id="sess-cmp-02",
    )
    res_c = CanonicalExecutionService.execute(
        workflow_id="quantitative_finance",
        context_id="institutional_market_v1",
        run_id=run_c,
    )
    GLOBAL_QUEUE.submit_run(run_c, req_c)
    GLOBAL_QUEUE.mark_running(run_c)
    GLOBAL_QUEUE.mark_completed(
        run_id=run_c,
        presentation={"governance_disposition": res_c.governance_disposition or "PASS"},
        artifacts=res_c.artifacts,
        evidence_records=res_c.records,
        checkpoints=getattr(res_c, "checkpoints", []),
    )

    # Test 1: Incompatible runs rejected
    resp_incompat = client.get(f"/api/v1/workbench/compare?runA={run_a}&runB={run_c}")
    assert resp_incompat.status_code == 200
    data_incompat = resp_incompat.json()
    assert data_incompat["compatible"] is False
    assert "Incompatible workflows" in data_incompat["incompatibleReason"]

    # Test 2: Compatible runs compared deterministically
    resp_compat = client.get(f"/api/v1/workbench/compare?runA={run_a}&runB={run_b}")
    assert resp_compat.status_code == 200
    data_compat = resp_compat.json()
    assert data_compat["compatible"] is True
    assert data_compat["lineage"]["isLineage"] is True
    assert data_compat["lineage"]["parentRunId"] == run_a

    # Check metric comparisons
    metrics_summary = data_compat["metricsSummary"]
    assert metrics_summary["totalCommonTests"] > 0

    metric_comparisons = data_compat["metricComparisons"]
    assert len(metric_comparisons) > 0
    for mc in metric_comparisons:
        assert "testId" in mc
        assert "statusA" in mc
        assert "statusB" in mc
        assert "isChanged" in mc
        for m in mc["metrics"]:
            assert "metric" in m
            if m["delta"] is not None:
                assert isinstance(m["delta"], (int, float))

    # Test 3: Lineage endpoint
    resp_lineage = client.get(f"/api/v1/workbench/runs/{run_b}/lineage")
    assert resp_lineage.status_code == 200
    lineage_data = resp_lineage.json()
    assert lineage_data["parentRunId"] == run_a
    assert lineage_data["intervention"] == "stress_testing"
    assert "alpha" in lineage_data["parameterDelta"]


def test_phase3_search_indexing():
    """Verify search only indexes genuine entities and does not fabricate matches."""
    app = create_app()
    client = TestClient(app)

    # Search for credit
    resp = client.get("/api/v1/workbench/search?q=credit")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] > 0
    for r in data["results"]:
        assert r["category"] in ("run", "test", "scenario", "evidence", "context", "artifact")
        assert "credit" in (r["id"] + r["title"] + r["subtitle"]).lower()


def test_phase1_catalog_census_regression():
    """Verify Phase-1 52/25/2 catalog census remains strictly preserved."""
    all_tests = list_tests()
    phase1_tests = [t for t in all_tests if getattr(t, "family", "") != "recommender"]
    _CONTEXT_TO_DOMAIN = {
        "tabular": "predictive",
        "market": "market",
        "short_rate": "treasury",
    }
    counts = {"predictive": 0, "market": 0, "treasury": 0}
    for t in phase1_tests:
        ctx_type = getattr(t, "context_type", "tabular")
        d = _CONTEXT_TO_DOMAIN.get(ctx_type, "predictive")
        counts[d] += 1

    assert len(phase1_tests) == 79
    assert counts["predictive"] == 52
    assert counts["market"] == 25
    assert counts["treasury"] == 2
    assert len([t for t in all_tests if getattr(t, "family", "") == "recommender"]) == 7


def test_phase2_nullable_producer_regression():
    """Verify Phase-2 nullable producer state in execution artifacts."""
    res = CanonicalExecutionService.execute(
        workflow_id="predictive_ml",
        context_id="institutional_credit_v1",
        materiality="TIER_1",
    )
    assert len(res.artifacts) > 0
    for _art_id, art in res.artifacts.items():
        # Producer step is authoritatively None (no synthetic guessing)
        assert art.get("producing_step_id") is None
