from __future__ import annotations

import tempfile

import pytest

pytest.importorskip("torch", reason="execution trains a small model")

from start.modeling.data import load_attrition_dataset  # noqa: E402
from start.modeling.enterprise_orchestrator import (  # noqa: E402
    EnterpriseReviewOrchestrator,
)
from start.review_session import Decision, ReviewSession  # noqa: E402


def test_rejecting_correlation_pruning_keeps_all_features():
    df = load_attrition_dataset(seed=0)
    s = ReviewSession(run_id="R")
    s.record_decision(Decision(
        key="fe:correlation_pruning", prompt="", recommended="apply",
        user_value="skip", effective="skip", choice="reject",
    ))
    out = EnterpriseReviewOrchestrator().run(
        df, user_target="attrition", output_root=tempfile.mkdtemp(),
        run_dl=True, session=s, seed=0,
    )
    assert out.copilot_execution.pruned_features == []
    honored = [t for t in out.trace_log.traces
               if "honored user rejection" in t.to_dict().get("action_taken", "")]
    assert len(honored) == 1


def test_not_rejecting_pruning_drops_correlated_features():
    df = load_attrition_dataset(seed=0)  # breast-cancer cohort has correlated cols
    s = ReviewSession(run_id="R")  # no rejection recorded
    out = EnterpriseReviewOrchestrator().run(
        df, user_target="attrition", output_root=tempfile.mkdtemp(),
        run_dl=True, session=s, seed=0,
    )
    assert len(out.copilot_execution.pruned_features) > 0


def test_architecture_override_produces_honored_trace():
    df = load_attrition_dataset(seed=0)
    s = ReviewSession(run_id="R")
    s.record_decision(Decision(
        key="architecture", prompt="", recommended="mlp",
        user_value="wide_deep", effective="wide_deep", choice="keep",
    ))
    out = EnterpriseReviewOrchestrator().run(
        df, user_target="attrition", output_root=tempfile.mkdtemp(),
        run_dl=False, architecture="wide_deep", session=s, seed=0,
    )
    honored = [t for t in out.trace_log.traces
               if t.to_dict().get("user_decision") == "override honored"]
    assert len(honored) == 1


def test_review_session_embedded_in_dashboard():
    import json

    df = load_attrition_dataset(seed=0)
    s = ReviewSession(run_id="R")
    s.record_decision(Decision(
        key="architecture", prompt="Family?", recommended="mlp",
        user_value="wide_deep", effective="wide_deep", choice="keep",
    ))
    out = EnterpriseReviewOrchestrator().run(
        df, user_target="attrition", output_root=tempfile.mkdtemp(),
        run_dl=False, architecture="wide_deep", session=s, seed=0,
    )
    md = open(out.dashboard_paths["md"]).read()
    d = json.loads(open(out.dashboard_paths["json"]).read())
    assert "## Review Journey" in md
    assert d.get("review_journey") is not None
