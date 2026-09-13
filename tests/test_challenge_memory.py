from __future__ import annotations

from start.agent_dialogue import AgentContext, ask_agent
from start.evidence_store import EvidenceStore
from start.review_session import Challenge, ReviewSession


def test_record_and_close_challenge():
    s = ReviewSession(run_id="R")
    s.record_challenge(Challenge(text="Why not WideDeep?", agent="ArchitectureReviewAgent"))
    assert len(s.open_challenges()) == 1
    s.close_challenge("Why not WideDeep?", response="MLP lowers overfitting")
    assert len(s.closed_challenges()) == 1
    assert len(s.open_challenges()) == 0


def test_challenge_summary_in_serialization():
    s = ReviewSession(run_id="R")
    s.record_challenge(Challenge(text="a", agent="X"))
    s.record_challenge(Challenge(text="b", agent="Y"))
    s.close_challenge("a")
    d = s.to_dict()
    assert d["challenge_summary"] == {"open": 1, "closed": 1, "unresolved": 0}
    assert len(d["challenges"]) == 2


def test_why_not_question_creates_challenge():
    s = ReviewSession(run_id="R")
    ctx = AgentContext(agent="ArchitectureReviewAgent", recommendation="mlp", reason="r")
    ask_agent("ArchitectureReviewAgent", "Why not wide_deep?", ctx, s)
    assert len(s.challenges) == 1
    assert s.challenges[0].status == "closed"  # answered immediately


def test_disagree_creates_challenge():
    s = ReviewSession(run_id="R")
    ctx = AgentContext(agent="FeatureEngineeringAgent", recommendation="prune", reason="r")
    ask_agent("FeatureEngineeringAgent", "I disagree with correlation pruning", ctx, s)
    assert len(s.challenges) == 1


def test_show_evidence_challenge_cites_source():
    df_store = EvidenceStore(outliers={"col_a": 12})
    s = ReviewSession(run_id="R")
    ctx = AgentContext(agent="FeatureEngineeringAgent", recommendation="clip",
                       reason="r", evidence=df_store)
    ask_agent("FeatureEngineeringAgent", "Show outlier evidence", ctx, s)
    assert s.challenges
    assert "data_statistics.outlier_summary" in s.challenges[0].evidence_used


def test_regular_question_no_challenge():
    s = ReviewSession(run_id="R")
    ctx = AgentContext(agent="ArchitectureReviewAgent", recommendation="mlp", reason="r")
    ask_agent("ArchitectureReviewAgent", "What is the recommendation?", ctx, s)
    assert len(s.challenges) == 0
