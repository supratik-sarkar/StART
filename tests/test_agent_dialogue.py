from __future__ import annotations

from start.agent_dialogue import AgentContext, ask_agent, compare_model_families
from start.review_session import ReviewSession


def _ctx():
    return AgentContext(
        agent="ArchitectureReviewAgent", recommendation="mlp",
        reason="Small tabular dataset; simpler MLP lowers overfitting risk.",
        risk_if_ignored="Higher overfitting risk and reduced interpretability.",
        alternatives=[{"family": "mlp"}, {"family": "wide_deep"}, {"family": "xgboost"}],
        dataset_summary="569 rows x 31 cols", checkpoint="architecture",
    )


def test_default_explains_recommendation():
    s = ReviewSession(run_id="R")
    ex = ask_agent("ArchitectureReviewAgent", "Why do you prefer MLP?", _ctx(), s)
    assert "mlp" in ex.answer.lower()
    assert ex.backend == "deterministic"


def test_show_alternatives_lists_families():
    s = ReviewSession(run_id="R")
    ex = ask_agent("ArchitectureReviewAgent", "Show alternatives.", _ctx(), s)
    assert "wide_deep" in ex.answer
    assert "xgboost" in ex.answer


def test_why_not_compares():
    s = ReviewSession(run_id="R")
    ex = ask_agent("ArchitectureReviewAgent", "Why not XGBoost?", _ctx(), s)
    assert "xgboost" in ex.answer.lower()
    assert "mlp" in ex.answer.lower()


def test_overfitting_question_uses_risk():
    s = ReviewSession(run_id="R")
    ex = ask_agent("ArchitectureReviewAgent", "Explain overfitting risk.", _ctx(), s)
    assert "overfitting" in ex.answer.lower()


def test_keep_all_features_question():
    s = ReviewSession(run_id="R")
    ex = ask_agent("FeatureEngineeringAgent", "What if I keep all features?", _ctx(), s)
    assert "keep" in ex.answer.lower() and "prun" in ex.answer.lower()


def test_exchange_recorded_in_session():
    s = ReviewSession(run_id="R")
    ask_agent("A", "Why MLP?", _ctx(), s)
    assert len(s.conversations) == 1


def test_llm_backend_used_when_connected():
    s = ReviewSession(run_id="R")

    class FakeLLM:
        name = "openai"

        def generate(self, prompt, *, system=None, metadata=None):
            return "MLP is preferred because the dataset is small and simple."

    ex = ask_agent("ArchitectureReviewAgent", "Why MLP?", _ctx(), s,
                   llm=FakeLLM(), llm_connected=True)
    assert ex.backend == "openai"
    assert "MLP" in ex.answer


def test_llm_failure_falls_back_to_deterministic():
    s = ReviewSession(run_id="R")

    class BrokenLLM:
        name = "openai"

        def generate(self, prompt, *, system=None, metadata=None):
            raise RuntimeError("api down")

    ex = ask_agent("ArchitectureReviewAgent", "Why MLP?", _ctx(), s,
                   llm=BrokenLLM(), llm_connected=True)
    assert ex.backend == "deterministic"  # honest fallback
    assert ex.answer


def test_compare_model_families_table():
    rows = compare_model_families(["mlp", "wide_deep", "xgboost"])
    assert len(rows) == 3
    for r in rows:
        for key in ("family", "performance", "interpretability", "maintenance", "governance"):
            assert key in r
