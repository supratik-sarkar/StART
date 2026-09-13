from __future__ import annotations

from start.interactive_checkpoints_flow import (
    run_feature_engineering_checkpoints,
    run_metric_checkpoint,
    run_target_checkpoint,
)
from start.review_session import ReviewSession


class _FakeRec:
    def __init__(self, step):
        self.step = step
        self.recommendation = f"apply {step}"
        self.reason = f"{step} needed"
        self.evidence_id = f"FE-{step}"
        self.risk_if_ignored = "some risk"


class _FakeFESet:
    def __init__(self, steps):
        self._steps = [_FakeRec(s) for s in steps]

    def applicable(self):
        return self._steps


def test_fe_checkpoint_reject_records_skip():
    s = ReviewSession(run_id="R")
    fe = _FakeFESet(["scaling", "correlation_pruning"])
    # scaling -> apply (Y), correlation_pruning -> skip (n)
    answers = iter(["Y", "n"])
    overrides = run_feature_engineering_checkpoints(
        fe, s, interactive=True, auto_accept=False,
        ask=lambda p: next(answers), emit=lambda m: None,
    )
    assert overrides["scaling"] == "apply"
    assert overrides["correlation_pruning"] == "skip"
    assert s.rejected("fe:correlation_pruning") is True
    assert s.rejected("fe:scaling") is False


def test_fe_checkpoint_auto_accept_applies_all():
    s = ReviewSession(run_id="R")
    fe = _FakeFESet(["scaling", "outliers"])
    overrides = run_feature_engineering_checkpoints(
        fe, s, interactive=False, auto_accept=True,
        ask=lambda p: "", emit=lambda m: None,
    )
    assert all(v == "apply" for v in overrides.values())


def test_fe_checkpoint_ask_then_apply():
    s = ReviewSession(run_id="R")
    fe = _FakeFESet(["scaling"])
    asked = []
    answers = iter(["Q", "Why scale?", "Y"])
    run_feature_engineering_checkpoints(
        fe, s, interactive=True, auto_accept=False,
        ask=lambda p: next(answers), emit=lambda m: asked.append(m),
    )
    # the question was routed to the agent and recorded in the session
    assert len(s.conversations) == 1
    assert s.conversations[0].question == "Why scale?"


def test_metric_checkpoint_records_and_returns():
    s = ReviewSession(run_id="R")
    answers = iter(["A"])  # accept recommendation
    eff = run_metric_checkpoint(
        "balanced", "false_negatives", "FN costlier", s,
        interactive=True, auto_accept=False,
        ask=lambda p: next(answers), emit=lambda m: None,
    )
    assert eff == "false_negatives"
    assert s.decision_for("metric_priority") is not None


def test_metric_checkpoint_ask_agent():
    s = ReviewSession(run_id="R")
    answers = iter(["Q", "Why PR-AUC?", "K"])  # ask, then keep user value
    run_metric_checkpoint(
        "balanced", "false_negatives", "reason", s,
        interactive=True, auto_accept=False,
        ask=lambda p: next(answers), emit=lambda m: None,
    )
    assert len(s.conversations) == 1


def test_target_checkpoint_records():
    s = ReviewSession(run_id="R")
    answers = iter(["A"])
    eff = run_target_checkpoint(
        "churn", "churn", "only binary column", s,
        interactive=True, auto_accept=False,
        ask=lambda p: next(answers), emit=lambda m: None,
    )
    assert eff == "churn"
    assert s.decision_for("target") is not None
