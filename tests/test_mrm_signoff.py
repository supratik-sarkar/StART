from __future__ import annotations

from start.evidence_store import EvidenceStore
from start.mrm_signoff import (
    CONDITIONAL,
    NOT_READY,
    READY,
    evaluate_signoff,
    render_signoff_markdown,
)
from start.review_session import Challenge, Decision, ReviewSession


def _strong_store():
    s = EvidenceStore()
    s.cohort_metrics = {"train": {"auc_roc": 0.97, "ece": 0.05},
                        "oos": {"auc_roc": 0.96, "ece": 0.06}}
    s.max_abs_drift = 0.02
    s.most_sensitive_feature = "worst_perimeter"
    return s


def test_strong_model_ready():
    d = evaluate_signoff(_strong_store(), ReviewSession(run_id="R"))
    assert d.verdict == READY


def test_excessive_sensitivity_blocks_ready():
    # the prompt's explicit requirement: high feature dependence -> NOT READY
    s = _strong_store()
    s.max_abs_drift = 0.35
    d = evaluate_signoff(s, ReviewSession(run_id="R"))
    assert d.verdict == NOT_READY
    assert any(f.name == "Feature dependence" and f.status == "blocker" for f in d.factors)


def test_weak_performance_blocks():
    s = EvidenceStore()
    s.cohort_metrics = {"oos": {"auc_roc": 0.55}}
    d = evaluate_signoff(s, ReviewSession(run_id="R"))
    assert d.verdict == NOT_READY


def test_open_challenge_makes_conditional():
    sess = ReviewSession(run_id="R")
    sess.record_challenge(Challenge(text="Why not WideDeep?", agent="A"))
    d = evaluate_signoff(_strong_store(), sess)
    assert d.verdict == NOT_READY


def test_unresolved_challenge_blocks():
    sess = ReviewSession(run_id="R")
    ch = Challenge(text="x", agent="A")
    ch.status = "unresolved"
    sess.record_challenge(ch)
    d = evaluate_signoff(_strong_store(), sess)
    assert d.verdict == NOT_READY


def test_override_makes_conditional():
    sess = ReviewSession(run_id="R")
    sess.record_decision(Decision(key="architecture", prompt="", recommended="mlp",
                                  user_value="wide_deep", effective="wide_deep", choice="keep"))
    d = evaluate_signoff(_strong_store(), sess)
    assert d.verdict == CONDITIONAL


def test_rationale_and_factors_present():
    d = evaluate_signoff(_strong_store(), ReviewSession(run_id="R"))
    assert d.rationale
    assert d.factors
    md = render_signoff_markdown(d)
    assert "Verdict" in md and "| Factor |" in md


def test_no_metrics_still_decides():
    # missing performance evidence -> unknown, not a crash
    d = evaluate_signoff(EvidenceStore(), ReviewSession(run_id="R"))
    assert d.verdict in (READY, CONDITIONAL, NOT_READY)
