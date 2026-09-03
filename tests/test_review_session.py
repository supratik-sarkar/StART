from __future__ import annotations

from start.review_session import Decision, Exchange, ReviewSession


def _session_with_choices():
    s = ReviewSession(run_id="R")
    s.record_decision(
        Decision(
            key="correlation_pruning",
            prompt="Prune?",
            recommended="prune",
            user_value="keep_all",
            effective="keep_all",
            choice="reject",
        )
    )
    s.record_decision(
        Decision(
            key="architecture",
            prompt="Family?",
            recommended="mlp",
            user_value="mlp",
            effective="mlp",
            choice="accept",
        )
    )
    return s


def test_decisions_persist_and_query():
    s = _session_with_choices()
    assert s.rejected("correlation_pruning") is True
    assert s.accepted("architecture") is True
    assert s.effective("architecture") == "mlp"
    assert s.effective("missing", "default") == "default"


def test_overrides_detected():
    s = _session_with_choices()
    keys = [d.key for d in s.overrides()]
    assert "correlation_pruning" in keys  # user diverged from recommendation
    assert "architecture" not in keys  # user took the recommendation


def test_context_banner_surfaces_prior_choices():
    s = _session_with_choices()
    s.add_clarification("false negatives are costlier")
    banner = s.context_banner()
    assert any("correlation_pruning: REJECTED" in line for line in banner)
    assert any("false negatives" in line for line in banner)


def test_conversations_recorded():
    s = ReviewSession(run_id="R")
    s.record_exchange(
        Exchange(agent="ArchitectureReviewAgent", question="Why MLP?", answer="lower overfitting")
    )
    assert len(s.conversations) == 1
    assert s.conversations[0].agent == "ArchitectureReviewAgent"


def test_serialization_complete():
    s = _session_with_choices()
    s.record_exchange(Exchange(agent="A", question="q", answer="a"))
    d = s.to_dict()
    for key in ("run_id", "decisions", "conversations", "clarifications", "overrides"):
        assert key in d
    assert len(d["decisions"]) == 2
    assert len(d["overrides"]) == 1


def test_latest_decision_wins():
    s = ReviewSession(run_id="R")
    s.record_decision(
        Decision(key="k", prompt="", recommended="a", user_value="a", effective="a", choice="accept")
    )
    s.record_decision(
        Decision(key="k", prompt="", recommended="a", user_value="b", effective="b", choice="modify")
    )
    assert s.effective("k") == "b"
