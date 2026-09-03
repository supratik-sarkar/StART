from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from start.modeling.data import load_attrition_dataset
from start.modeling.enterprise_orchestrator import EnterpriseReviewOrchestrator
from start.modeling.tuning_run import run_tuning

pytest.importorskip("torch", reason="v2.4.0 tuning wizard tests require torch")


def test_wizard_custom_tuning_params_populated_only_on_input():
    # If the user presses enter, default is NOT populated as a fixed single-element constraint in custom_space
    from unittest.mock import patch

    from start.main import run_interactive_setup_wizard

    inputs = [
        "1",  # backend None
        "2",  # Deep Learning Branch
        "3",  # LSTM
        "1",  # ReLU activation
        "1",  # Dataset Source -> [1] Synthetic AML / fraud transactions
        "2",  # Problem task type -> Regression
        "",  # target column (auto-lock is_fraud)
        "2",  # split strategy -> Time-Series Split
        "3",  # Select weighting logic -> Temporal Decay
        "Temporal weighting necessity",  # justification
        "",  # train/test/oos split proportions (default)
        "y",  # Enable hyperparameter tuning -> yes
        "1",  # Tuning strategy -> random search
        "6",  # number of trials
        "2",  # validation scheme -> K-Fold
        "2",  # Select folds -> 5 folds
        "",  # Enter hidden size (blank) -> should NOT populate custom_tuning_params
        "2",  # Enter number of layers (type 2) -> should populate custom_tuning_params["num_layers"] = 2
        "",  # Enter learning rate (blank)
        "",  # Enter dropout (blank)
        "",  # Enter sequence length (blank)
        "",  # Enter batch size (blank)
        "1",  # explainability method -> Integrated Gradients
        "y",  # Proceed pre-flight summary -> yes
    ]

    with patch("builtins.input", side_effect=inputs):
        config = run_interactive_setup_wizard()

    assert config["tuning_strategy"] == "bounded_random_search"
    assert config["tuning_trials"] == 6
    assert config["validation_scheme"] == "k_fold"
    assert config["k_folds"] == 5
    assert config["explain_method"] == "integrated_gradients"
    assert config["sample_weight"] is True

    # Check that only explicitly input fields are in custom_tuning_params (no blank ones)
    assert "num_layers" in config["custom_tuning_params"]
    assert config["custom_tuning_params"]["num_layers"] == 2
    assert "hidden_size" not in config["custom_tuning_params"]
    assert "learning_rate" not in config["custom_tuning_params"]


def test_tuning_run_kfold_regression_recurrent():
    # Verify that run_tuning supports k_fold validation for recurrent models
    df = load_attrition_dataset(seed=42)
    features = [c for c in df.columns if c != "attrition"][:4]

    run = run_tuning(
        df,
        "attrition",
        features,
        strategy="bounded_random_search",
        n_trials=3,
        primary_metric="auc_roc",
        seed=42,
        architecture="lstm",
        task_type="binary_classification",
        validation="k_fold",
        k_folds=3,
    )

    assert run.ran is True
    assert run.validation == "k_fold"
    assert len(run.trials) == 3
    for trial in run.trials:
        assert trial.validation_metric > 0.0


def test_leakage_prevention_tuning_train_only():
    # Verify that in EnterpriseReviewOrchestrator, we split the data and only train on _train_only
    df = load_attrition_dataset(seed=42)

    orchestrator = EnterpriseReviewOrchestrator()

    # Mock run_tuning and run_model_execution to see what data they receive
    from unittest.mock import patch

    with (
        patch("start.modeling.tuning_run.run_tuning") as mock_tuning,
        patch("start.modeling.model_execution.run_model_execution") as mock_model_exec,
        patch("start.governance.findings.FindingsRegister"),
        patch("start.reporting.agent_trace.TraceLog"),
        patch("start.reporting.artifacts.ArtifactRegistry"),
        patch("start.reporting.progress.ActionLog"),
        patch("start.modeling.review_orchestrator.ReviewOrchestrator") as mock_base_orch,
    ):
        # Stub base orchestrator run return value
        mock_base = MagicMock()
        mock_base.task_type = "binary_classification"
        mock_base.cohort_metrics = {"train": {"auc_roc": 0.8}}
        mock_base_orch.return_value.run.return_value = mock_base

        # Stub model exec return value
        mock_model = MagicMock()
        mock_model.feature_columns = [c for c in df.columns if c != "attrition"][:2]
        mock_model.explainability_method = "integrated_gradients"
        mock_model.to_dict.return_value = {
            "split_table": [],
            "metrics_by_split": {},
            "training_diagnostics": {},
            "explainability_method": "integrated_gradients",
            "global_importance": {},
            "explainability_available": True,
            "generalization_gap": 0.05,
        }
        mock_model_exec.return_value = mock_model

        # Stub run_tuning return value
        mock_tune = MagicMock()
        mock_tune.ran = True
        mock_tune.n_trials = 5
        mock_tune.strategy = "bounded_random_search"
        mock_tune.best_metric = 0.85
        mock_tune.best_params = {"learning_rate": 0.003}
        mock_tune.rejected_params = []
        mock_tune.trials = []
        mock_tune.validation = "holdout"
        mock_tune.to_dict.return_value = {
            "ran": True,
            "n_trials": 5,
            "strategy": "bounded_random_search",
            "best_metric": 0.85,
            "best_params": {"learning_rate": 0.003},
            "rejected_params": [],
            "trials": [],
            "validation": "holdout",
        }
        mock_tuning.return_value = mock_tune

        # Call orchestrator run
        orchestrator.run(
            df,
            user_target="attrition",
            split_strategy="time_based",
            run_dl=True,
            tuning_strategy="bounded_random_search",
            tuning_trials=5,
            split_props=(0.6, 0.2, 0.2),
        )

        # Verify that run_tuning's first argument (the dataframe) is NOT the full df
        # It should be _train_only, which has 60% of df rows (around 180 rows out of 300)
        called_df = mock_tuning.call_args[0][0]
        assert len(called_df) < len(df)
        assert abs(len(called_df) - int(0.6 * len(df))) <= 2
