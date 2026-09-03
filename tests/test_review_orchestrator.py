from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from start.ai_engineering import (
    STAGE_ADAPTERS,
    available_stages,
    run_all_stages,
    run_stage,
)
from start.modeling.review_orchestrator import STAGES, ReviewOrchestrator


# --------------------------------------------------------------------------- #
# Layer 8A: AI-engineering stages (honest availability)
# --------------------------------------------------------------------------- #
def test_all_stack_categories_present():
    categories = {a.category for a in STAGE_ADAPTERS}
    assert {
        "policy",
        "mcp",
        "observability",
        "telemetry",
        "redteam",
        "compliance",
        "guardrails",
        "evals",
        "orchestration",
    } <= categories


def test_stages_report_honest_status():
    results = run_all_stages()
    assert len(results) == len(STAGE_ADAPTERS)
    for r in results:
        # never a fabricated success: status is either complete (installed) or not_installed
        assert r.status in {"complete", "not_installed", "skipped"}
        if not r.available:
            assert r.status == "not_installed"
            assert r.detail  # explicit hint, never blank


def test_not_installed_stage_has_install_hint():
    # In the public/offline env most are absent; each must explain how to enable.
    results = run_all_stages()
    absent = [r for r in results if not r.available]
    assert absent, "expected some stages absent in the test environment"
    for r in absent:
        assert "install" in r.detail.lower() or "pip" in r.detail.lower() or "npm" in r.detail.lower()


def test_run_stage_unknown_raises():
    with pytest.raises(ValueError, match="Unknown stage"):
        run_stage("Quantum Validation")


def test_available_stages_subset():
    avail_names = {a.name for a in available_stages()}
    all_names = {a.name for a in STAGE_ADAPTERS}
    assert avail_names <= all_names


# --------------------------------------------------------------------------- #
# Layer 8: full review orchestrator
# --------------------------------------------------------------------------- #
@pytest.fixture()
def churn_frame():
    rng = np.random.default_rng(0)
    n = 400
    age = rng.integers(20, 70, n)
    # give the target a learnable signal
    churn = ((age > 50).astype(int) + rng.integers(0, 2, n) >= 1).astype(int)
    return pd.DataFrame(
        {
            "customer_id": range(n),
            "age": age,
            "balance": rng.normal(1000, 200, n),
            "tenure": rng.integers(0, 10, n),
            "churned": churn,
        }
    )


def test_full_pipeline_emits_all_stages(churn_frame, tmp_path):
    events = []
    orch = ReviewOrchestrator(on_stage=lambda e: events.append(e))
    outcome = orch.run(
        churn_frame,
        user_target="churned",
        agent_mode="deterministic",
        output_root=str(tmp_path),
        run_dl=True,
        seed=0,
    )
    emitted_complete = {e.stage for e in events if e.status in {"complete", "skipped"}}
    assert set(STAGES) <= emitted_complete  # every stage is visible
    assert outcome.task_type == "binary_classification"
    assert outcome.modality == "tabular"
    assert outcome.recommended_family == "mlp"


def test_pipeline_produces_evidence_and_report(churn_frame, tmp_path):
    orch = ReviewOrchestrator()
    outcome = orch.run(churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=True, seed=0)
    test_ids = {r.test_id for r in outcome.evidence}
    assert {
        "discovery.dataset_profile",
        "discovery.target_selection",
        "discovery.task_inference",
        "split.plan",
        "feature_engineering.diagnostics",
    } <= test_ids
    assert outcome.agent_review is not None and outcome.agent_review.signoff
    assert (tmp_path / "ledger.jsonl").exists()
    report = (tmp_path / "reports" / f"{outcome.run_id}.md").read_text()
    for section in (
        "## Review summary",
        "## Pipeline stages",
        "## Evidence ledger",
        "## AI-engineering stage surface",
        "## Validation recommendations",
    ):
        assert section in report
    assert "binary_classification" in report


def test_pipeline_diagnostics_only_without_run_dl(churn_frame, tmp_path):
    orch = ReviewOrchestrator()
    outcome = orch.run(churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=False, seed=0)
    # model execution skipped, but governance still runs and evidence still sealed
    exec_events = [e for e in outcome.stage_events if e.stage == "model_execution"]
    assert exec_events and exec_events[-1].status == "skipped"
    assert outcome.cohort_metrics == {}
    assert outcome.agent_review.signoff


def test_pipeline_requires_target_when_none_found():
    df = pd.DataFrame({"a": range(10), "b": range(10)})  # no obvious target
    orch = ReviewOrchestrator()
    # 'a'/'b' won't score as candidates; with no user_target it should raise
    with pytest.raises(ValueError, match="target"):
        orch.run(df, output_root=None, run_dl=False)


def test_ai_engineering_stage_in_outcome(churn_frame, tmp_path):
    outcome = ReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=False, seed=0
    )
    assert len(outcome.ai_engineering) == len(STAGE_ADAPTERS)
    ai_event = [e for e in outcome.stage_events if e.stage == "ai_engineering"][-1]
    assert ai_event.status == "complete"
