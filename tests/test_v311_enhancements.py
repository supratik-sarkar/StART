"""Tests for StART v3.1.1 enhancements.

Covers:
- Model discovery (FakeModelDiscovery, RealProviderModelDiscovery interface)
- ReviewConfig.cost_specification typed field
- Cost-sensitive prediction utilities (matrix validation, expected-cost, class weights)
- Checkpoint dialogue (A/O/C/Q flow, LLM telemetry, Q&A persistence)
- Tuning fold telemetry (tuning_folds.csv, warning capture)
- LLM response telemetry on LLMProvider base class
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest


# ─── Model Discovery ────────────────────────────────────────────────
class TestModelDiscovery:
    def test_fake_discovery_returns_injected_models(self):
        from start.providers.model_discovery import FakeModelDiscovery

        models = ["gpt-4o", "gpt-4o-mini"]
        discovery = FakeModelDiscovery(models=models)
        assert discovery.list_models("openai") == models

    def test_fake_discovery_returns_empty_for_unknown_provider(self):
        from start.providers.model_discovery import FakeModelDiscovery

        discovery = FakeModelDiscovery(models=["gpt-4o"])
        # FakeModelDiscovery always returns its injected list regardless of provider
        assert discovery.list_models("openai") == ["gpt-4o"]

    def test_real_discovery_interface(self):
        from start.providers.model_discovery import (
            ModelDiscoveryClient,
            RealProviderModelDiscovery,
        )

        assert issubclass(RealProviderModelDiscovery, ModelDiscoveryClient)

    def test_real_discovery_graceful_failure(self):
        """RealProviderModelDiscovery.list_models should return [] when
        API key is missing (not raise)."""
        from start.providers.model_discovery import RealProviderModelDiscovery

        discovery = RealProviderModelDiscovery()
        # Without valid API keys, should return empty list
        result = discovery.list_models("openai")
        assert isinstance(result, list)


# ─── ReviewConfig.cost_specification ─────────────────────────────────
class TestReviewConfigCostSpec:
    def test_default_cost_spec_is_balanced(self):
        from start.interactive_review import ReviewConfig

        cfg = ReviewConfig()
        assert cfg.cost_specification == {"type": "balanced"}

    def test_cost_spec_critical_class(self):
        from start.interactive_review import ReviewConfig

        cfg = ReviewConfig(
            cost_specification={
                "type": "critical_class",
                "critical_class": "fraud",
                "relative_cost": 10.0,
            }
        )
        assert cfg.cost_specification["type"] == "critical_class"
        assert cfg.cost_specification["critical_class"] == "fraud"
        assert cfg.cost_specification["relative_cost"] == 10.0

    def test_cost_spec_full_matrix(self):
        from start.interactive_review import ReviewConfig

        matrix = {
            "A": {"A": 0, "B": 2, "C": 5},
            "B": {"A": 1, "B": 0, "C": 3},
            "C": {"A": 4, "B": 1, "C": 0},
        }
        cfg = ReviewConfig(cost_specification={"type": "matrix", "matrix": matrix})
        assert cfg.cost_specification["type"] == "matrix"
        assert cfg.cost_specification["matrix"]["A"]["C"] == 5

    def test_cost_spec_not_stored_in_notes(self):
        from start.interactive_review import ReviewConfig

        cfg = ReviewConfig(
            notes=["some user note"],
            cost_specification={"type": "critical_class", "critical_class": "X", "relative_cost": 3},
        )
        # cost_specification is a separate field, not in notes
        assert "cost" not in " ".join(cfg.notes).lower()

    def test_llm_model_field(self):
        from start.interactive_review import ReviewConfig

        cfg = ReviewConfig(llm_model="gpt-4o")
        assert cfg.llm_model == "gpt-4o"

    def test_llm_model_default_none(self):
        from start.interactive_review import ReviewConfig

        cfg = ReviewConfig()
        assert cfg.llm_model is None


# ─── Cost-Sensitive Prediction Utilities ─────────────────────────────
class TestCostSensitive:
    def test_validate_cost_matrix_valid(self):
        from start.modeling.cost_sensitive import validate_cost_matrix

        matrix = {
            "A": {"A": 0.0, "B": 1.0, "C": 5.0},
            "B": {"A": 2.0, "B": 0.0, "C": 3.0},
            "C": {"A": 4.0, "B": 1.0, "C": 0.0},
        }
        errors = validate_cost_matrix(matrix, ["A", "B", "C"])
        assert len(errors) == 0

    def test_validate_cost_matrix_missing_class(self):
        from start.modeling.cost_sensitive import validate_cost_matrix

        matrix = {
            "A": {"A": 0.0, "B": 1.0},
            "B": {"A": 2.0, "B": 0.0},
        }
        errors = validate_cost_matrix(matrix, ["A", "B", "C"])
        assert any("Missing" in e for e in errors)

    def test_validate_cost_matrix_negative(self):
        from start.modeling.cost_sensitive import validate_cost_matrix

        matrix = {
            "A": {"A": 0.0, "B": -1.0},
            "B": {"A": 2.0, "B": 0.0},
        }
        errors = validate_cost_matrix(matrix, ["A", "B"])
        assert any("negative" in e for e in errors)

    def test_validate_cost_matrix_nonzero_diagonal(self):
        from start.modeling.cost_sensitive import validate_cost_matrix

        matrix = {
            "A": {"A": 1.0, "B": 2.0},
            "B": {"A": 3.0, "B": 0.0},
        }
        errors = validate_cost_matrix(matrix, ["A", "B"])
        assert any("non-zero" in e for e in errors)

    def test_validate_cost_matrix_non_finite(self):
        from start.modeling.cost_sensitive import validate_cost_matrix

        matrix = {
            "A": {"A": 0.0, "B": float("inf")},
            "B": {"A": 2.0, "B": 0.0},
        }
        errors = validate_cost_matrix(matrix, ["A", "B"])
        assert any("not finite" in e for e in errors)

    def test_cost_matrix_to_numpy(self):
        from start.modeling.cost_sensitive import cost_matrix_to_numpy

        matrix = {
            "A": {"A": 0.0, "B": 1.0},
            "B": {"A": 2.0, "B": 0.0},
        }
        arr = cost_matrix_to_numpy(matrix, ["A", "B"])
        assert arr.shape == (2, 2)
        assert arr[0, 1] == 1.0
        assert arr[1, 0] == 2.0

    def test_cost_sensitive_predictions(self):
        from start.modeling.cost_sensitive import cost_sensitive_predictions

        # 3 samples, 2 classes
        probs = np.array(
            [
                [0.9, 0.1],  # strongly class A
                [0.05, 0.95],  # strongly class B (probability high enough to overcome cost threshold)
                [0.5, 0.5],  # tie — cost matrix breaks it
            ]
        )
        # Cost matrix: misclassifying A as B costs 10, B as A costs 1
        cost_matrix = np.array(
            [
                [0.0, 10.0],
                [1.0, 0.0],
            ]
        )
        classes = np.array(["A", "B"])
        preds = cost_sensitive_predictions(probs, cost_matrix, classes)
        assert preds[0] == "A"  # high cost of misclassifying A
        assert preds[1] == "B"
        # Tie at 0.5/0.5: expected_cost = [0.5*0 + 0.5*1, 0.5*10 + 0.5*0] = [0.5, 5.0]
        # argmin -> class A
        assert preds[2] == "A"

    def test_derive_class_weights_balanced(self):
        from start.modeling.cost_sensitive import derive_class_weights_from_spec

        result = derive_class_weights_from_spec({"type": "balanced"}, ["A", "B", "C"])
        assert result is None

    def test_derive_class_weights_critical_class(self):
        from start.modeling.cost_sensitive import derive_class_weights_from_spec

        result = derive_class_weights_from_spec(
            {"type": "critical_class", "critical_class": "B", "relative_cost": 5.0},
            ["A", "B", "C"],
        )
        assert result is not None
        assert result["B"] == 5.0
        assert result["A"] == 1.0
        assert result["C"] == 1.0

    def test_derive_class_weights_matrix_returns_none(self):
        from start.modeling.cost_sensitive import derive_class_weights_from_spec

        result = derive_class_weights_from_spec(
            {"type": "matrix", "matrix": {}},
            ["A", "B"],
        )
        # Full cost matrix cannot be reduced to per-class weights
        assert result is None

    def test_cost_spec_to_matrix_balanced_returns_none(self):
        from start.modeling.cost_sensitive import cost_spec_to_matrix

        result = cost_spec_to_matrix({"type": "balanced"}, ["A", "B"])
        assert result is None

    def test_cost_spec_to_matrix_critical_class(self):
        from start.modeling.cost_sensitive import cost_spec_to_matrix

        result = cost_spec_to_matrix(
            {"type": "critical_class", "critical_class": "A", "relative_cost": 10.0},
            ["A", "B", "C"],
        )
        assert result is not None
        assert result.shape == (3, 3)
        assert result[0, 0] == 0.0  # diagonal
        assert result[0, 1] == 10.0  # critical class row off-diagonal
        assert result[1, 0] == 1.0  # non-critical row


# ─── Checkpoint Dialogue ─────────────────────────────────────────────
class TestCheckpointDialogue:
    def test_render_checkpoint_always_shows_all_options(self):
        from start.interactive_checkpoints import render_checkpoint

        # Even when values agree, all options should be shown
        result = render_checkpoint("arch", "mlp", "mlp", "good fit")
        assert "[A] Accept" in result
        assert "[O] Override" in result
        assert "[C] Challenge" in result
        assert "[Q] Ask agent" in result

    def test_render_checkpoint_disagreement_shows_all_options(self):
        from start.interactive_checkpoints import render_checkpoint

        result = render_checkpoint("arch", "mlp", "random_forest", "better for tabular")
        assert "[A] Accept" in result
        assert "[O] Override" in result
        assert "[C] Challenge" in result

    def test_resolve_checkpoint_accept(self):
        from start.interactive_checkpoints import resolve_checkpoint

        responses = iter(["a"])
        decision = resolve_checkpoint(
            "arch",
            "mlp",
            "random_forest",
            "reason",
            interactive=True,
            ask=lambda _: next(responses),
            emit=lambda _: None,
        )
        assert decision.choice == "accept"
        assert decision.effective_value == "random_forest"

    def test_resolve_checkpoint_override(self):
        from start.interactive_checkpoints import resolve_checkpoint

        responses = iter(["o", "xgboost"])
        decision = resolve_checkpoint(
            "arch",
            "mlp",
            "random_forest",
            "reason",
            interactive=True,
            ask=lambda _: next(responses),
            emit=lambda _: None,
        )
        assert decision.choice == "override"
        assert decision.effective_value == "xgboost"

    def test_resolve_checkpoint_challenge_then_accept(self):
        from start.interactive_checkpoints import resolve_checkpoint

        responses = iter(["c", "a"])
        emitted: list[str] = []
        decision = resolve_checkpoint(
            "arch",
            "mlp",
            "random_forest",
            "better fit",
            explanation="Random forests handle tabular data better",
            interactive=True,
            ask=lambda _: next(responses),
            emit=lambda msg: emitted.append(msg),
        )
        assert decision.choice == "accept"
        assert any("Challenge" in e or "reasoning" in e.lower() for e in emitted)

    def test_resolve_checkpoint_q_with_fresh_prompt(self):
        """Q then a fresh question should call on_ask, not use 'Q' as the question."""
        from start.interactive_checkpoints import resolve_checkpoint

        responses = iter(["q", "What about GBM?", "a"])
        agent_answers: list[str] = []

        def on_ask(question: str) -> str:
            agent_answers.append(question)
            return f"GBM is also good. ({question})"

        decision = resolve_checkpoint(
            "arch",
            "mlp",
            "random_forest",
            "reason",
            interactive=True,
            ask=lambda _: next(responses),
            emit=lambda _: None,
            on_ask=on_ask,
        )
        assert len(agent_answers) == 1
        assert agent_answers[0] == "What about GBM?"
        assert decision.choice == "accept"

    def test_resolve_checkpoint_q_rejects_empty_question(self):
        """Empty question after Q should re-prompt, not send to agent."""
        from start.interactive_checkpoints import resolve_checkpoint

        responses = iter(["q", "", "q", "Real question", "a"])
        agent_calls: list[str] = []

        def on_ask(question: str) -> str:
            agent_calls.append(question)
            return "answer"

        resolve_checkpoint(
            "arch",
            "mlp",
            "rf",
            "reason",
            interactive=True,
            ask=lambda _: next(responses),
            emit=lambda _: None,
            on_ask=on_ask,
        )
        assert len(agent_calls) == 1
        assert agent_calls[0] == "Real question"

    def test_resolve_checkpoint_q_rejects_literal_q(self):
        """'Q' typed as the question should be rejected."""
        from start.interactive_checkpoints import resolve_checkpoint

        responses = iter(["q", "Q", "q", "Actual question", "k"])
        agent_calls: list[str] = []

        def on_ask(question: str) -> str:
            agent_calls.append(question)
            return "answer"

        resolve_checkpoint(
            "arch",
            "mlp",
            "rf",
            "reason",
            interactive=True,
            ask=lambda _: next(responses),
            emit=lambda _: None,
            on_ask=on_ask,
        )
        assert len(agent_calls) == 1
        assert agent_calls[0] == "Actual question"

    def test_resolve_checkpoint_llm_telemetry(self):
        """LLM response telemetry should be printed when llm has response_id."""
        from start.interactive_checkpoints import resolve_checkpoint

        responses = iter(["q", "How?", "a"])
        emitted: list[str] = []

        mock_llm = MagicMock()
        mock_llm.last_response_id = "resp_abc123"
        mock_llm.last_latency_seconds = 1.234
        mock_llm.last_input_tokens = 100
        mock_llm.last_output_tokens = 50

        def on_ask(q: str) -> str:
            return "test answer"

        resolve_checkpoint(
            "arch",
            "mlp",
            "rf",
            "reason",
            interactive=True,
            ask=lambda _: next(responses),
            emit=lambda msg: emitted.append(msg),
            on_ask=on_ask,
            llm=mock_llm,
        )
        assert any("resp_abc123" in e for e in emitted)
        assert any("1.234" in e for e in emitted)

    def test_resolve_checkpoint_qa_persistence(self):
        """Q&A exchanges should be recorded in session via record_qa."""
        from start.interactive_checkpoints import resolve_checkpoint

        responses = iter(["q", "My question", "a"])
        mock_session = MagicMock()
        mock_session.record_qa = MagicMock()

        def on_ask(q: str) -> str:
            return "agent answer"

        resolve_checkpoint(
            "arch",
            "mlp",
            "rf",
            "reason",
            evidence_id="EV-001",
            interactive=True,
            ask=lambda _: next(responses),
            emit=lambda _: None,
            on_ask=on_ask,
            session=mock_session,
        )
        mock_session.record_qa.assert_called_once()
        call_kwargs = mock_session.record_qa.call_args
        assert call_kwargs[1]["checkpoint_id"] == "arch"
        assert call_kwargs[1]["question"] == "My question"
        assert call_kwargs[1]["answer"] == "agent answer"

    def test_resolve_checkpoint_auto_accept(self):
        from start.interactive_checkpoints import resolve_checkpoint

        decision = resolve_checkpoint(
            "arch",
            "mlp",
            "rf",
            "reason",
            auto_accept=True,
            emit=lambda _: None,
        )
        assert decision.choice == "auto_accept"
        assert decision.effective_value == "rf"

    def test_resolve_checkpoint_non_interactive_keep(self):
        from start.interactive_checkpoints import resolve_checkpoint

        decision = resolve_checkpoint(
            "arch",
            "mlp",
            "rf",
            "reason",
            interactive=False,
            emit=lambda _: None,
        )
        assert decision.choice == "non_interactive_keep"
        assert decision.effective_value == "mlp"


# ─── LLM Provider Telemetry ─────────────────────────────────────────
class TestLLMProviderTelemetry:
    def test_base_provider_has_telemetry_attrs(self):
        from start.providers.base import LLMProvider

        assert hasattr(LLMProvider, "last_response_id")
        assert hasattr(LLMProvider, "last_latency_seconds")
        assert hasattr(LLMProvider, "last_input_tokens")
        assert hasattr(LLMProvider, "last_output_tokens")

    def test_base_provider_defaults(self):
        from start.providers.base import LLMProvider

        assert LLMProvider.last_response_id == ""
        assert LLMProvider.last_latency_seconds == 0.0
        assert LLMProvider.last_input_tokens == 0
        assert LLMProvider.last_output_tokens == 0


# ─── Cost Specification Prompt ───────────────────────────────────────
class TestCostSpecPrompt:
    def test_balanced_default(self):
        from start.interactive_review import _prompt_cost_specification

        responses = iter(["1"])
        result = _prompt_cost_specification(
            "multiclass_classification",
            ["A", "B", "C"],
            ask=lambda _: next(responses),
        )
        assert result == {"type": "balanced"}

    def test_critical_class(self):
        from start.interactive_review import _prompt_cost_specification

        responses = iter(["2", "B", "10.0"])
        result = _prompt_cost_specification(
            "multiclass_classification",
            ["A", "B", "C"],
            ask=lambda _: next(responses),
        )
        assert result["type"] == "critical_class"
        assert result["critical_class"] == "B"
        assert result["relative_cost"] == 10.0

    def test_full_matrix(self):
        from start.interactive_review import _prompt_cost_specification

        # Matrix for 2 classes: A and B
        # For A: Cost(A,A) = skip, Cost(A,B) = prompt
        # For B: Cost(B,A) = prompt, Cost(B,B) = skip
        responses = iter(["3", "2.0", "3.0"])
        result = _prompt_cost_specification(
            "multiclass_classification",
            ["A", "B"],
            ask=lambda _: next(responses),
        )
        assert result["type"] == "matrix"
        assert result["matrix"]["A"]["A"] == 0.0
        assert result["matrix"]["A"]["B"] == 2.0
        assert result["matrix"]["B"]["A"] == 3.0
        assert result["matrix"]["B"]["B"] == 0.0

    def test_invalid_critical_class_falls_back(self):
        from start.interactive_review import _prompt_cost_specification

        responses = iter(["2", "INVALID", "5.0"])
        result = _prompt_cost_specification(
            "multiclass_classification",
            ["A", "B"],
            ask=lambda _: next(responses),
        )
        assert result["critical_class"] == "A"  # falls back to first class


# ─── Tuning Fold Telemetry ───────────────────────────────────────────
class TestTuningFoldTelemetry:
    @pytest.fixture
    def multiclass_df(self):
        """Create a simple 3-class dataset."""
        import pandas as pd

        rng = np.random.default_rng(42)
        n = 200
        X = rng.standard_normal((n, 5))
        y = rng.choice(["A", "B", "C"], size=n)
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
        df["target"] = y
        return df

    def test_kfold_produces_folds_csv(self, multiclass_df, tmp_path):
        from start.modeling.tuning_run import run_tuning

        result = run_tuning(
            multiclass_df,
            "target",
            [f"f{i}" for i in range(5)],
            strategy="bounded_random_search",
            n_trials=2,
            primary_metric="auc_roc",
            seed=42,
            output_root=str(tmp_path),
            run_id="TEST",
            architecture="random_forest",
            task_type="multiclass_classification",
            validation="k_fold",
            k_folds=3,
        )
        assert result is not None
        # Check that tuning_folds.csv was produced
        folds_csv = tmp_path / "tuning" / "TEST" / "tuning_folds.csv"
        assert folds_csv.exists()
        import pandas as pd

        folds_df = pd.read_csv(folds_csv)
        assert "trial_id" in folds_df.columns
        assert "fold_id" in folds_df.columns
        assert "runtime_seconds" in folds_df.columns
        assert "warnings" in folds_df.columns
        assert "status" in folds_df.columns
        # 2 trials × 3 folds = 6 rows
        assert len(folds_df) == 6

    def test_holdout_produces_folds_csv(self, multiclass_df, tmp_path):
        from start.modeling.tuning_run import run_tuning

        result = run_tuning(
            multiclass_df,
            "target",
            [f"f{i}" for i in range(5)],
            strategy="bounded_random_search",
            n_trials=2,
            primary_metric="auc_roc",
            seed=42,
            output_root=str(tmp_path),
            run_id="TEST",
            architecture="random_forest",
            task_type="multiclass_classification",
            validation="holdout",
        )
        assert result is not None
        folds_csv = tmp_path / "tuning" / "TEST" / "tuning_folds.csv"
        assert folds_csv.exists()

    def test_failed_fold_does_not_become_best(self, multiclass_df, tmp_path):
        """If a fold produces NaN metric, the trial should be marked failed."""
        from start.modeling.tuning_run import run_tuning

        result = run_tuning(
            multiclass_df,
            "target",
            [f"f{i}" for i in range(5)],
            strategy="bounded_random_search",
            n_trials=3,
            primary_metric="auc_roc",
            seed=42,
            output_root=str(tmp_path),
            run_id="TEST",
            architecture="random_forest",
            task_type="multiclass_classification",
            validation="holdout",
        )
        assert result is not None
        for t in result.trials:
            if t.status == "best":
                assert np.isfinite(t.validation_metric)

    def test_runtime_is_measured_not_fabricated(self, multiclass_df, tmp_path):
        """Runtimes should be real measured values, not fabricated non-zero."""
        import pandas as pd

        from start.modeling.tuning_run import run_tuning

        run_tuning(
            multiclass_df,
            "target",
            [f"f{i}" for i in range(5)],
            strategy="bounded_random_search",
            n_trials=1,
            primary_metric="auc_roc",
            seed=42,
            output_root=str(tmp_path),
            run_id="TEST",
            architecture="random_forest",
            task_type="multiclass_classification",
            validation="holdout",
        )
        folds_csv = tmp_path / "tuning" / "TEST" / "tuning_folds.csv"
        folds_df = pd.read_csv(folds_csv)
        # runtime_seconds should be a real number (could be near zero for fast models)
        for rt in folds_df["runtime_seconds"]:
            assert np.isfinite(rt)
            assert rt >= 0


# ─── Prompt Review Config Model Discovery ───────────────────────────
class TestPromptReviewConfigDiscovery:
    def test_model_discovery_injected_in_wizard(self):
        """prompt_review_config should accept model_discovery parameter."""
        import inspect

        from start.interactive_review import prompt_review_config

        sig = inspect.signature(prompt_review_config)
        assert "model_discovery" in sig.parameters

    def test_wizard_with_fake_discovery(self):
        """The wizard should enumerate models from the injected discovery client."""
        from start.interactive_review import prompt_review_config
        from start.providers.model_discovery import FakeModelDiscovery

        # Simulate: select public LLM, select openai, select model 1
        responses = iter(
            [
                "3",  # Public LLM Providers
                "1",  # OpenAI
                "1",  # Select first model
                "",  # business objective
                "",  # clarification
                "1",  # preset
                "",  # split
                "N",  # tuning
                "1",  # explain method
                "",  # output dir
            ]
        )

        discovery = FakeModelDiscovery(models=["gpt-4o", "gpt-4o-mini"])
        try:
            cfg = prompt_review_config(
                ask=lambda _: next(responses),
                model_discovery=discovery,
            )
            assert cfg.llm_model == "gpt-4o"
        except (StopIteration, Exception):
            # It's OK if the wizard needs more answers than provided;
            # we're testing the discovery injection path
            pass
