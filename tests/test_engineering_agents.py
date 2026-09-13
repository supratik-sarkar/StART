from __future__ import annotations

from start.agents.engineering_agents import (
    ArchitectureReviewAgent,
    HyperparameterTuningAgent,
    select_primary_metric,
)


# -- metric priority routing ------------------------------------------------- #
def test_binary_default_metric():
    assert select_primary_metric("binary_classification")["primary_metric"] == "auc_roc"


def test_false_negatives_costly_routes_to_recall_family():
    r = select_primary_metric("binary_classification", costlier_errors="false_negatives")
    assert r["primary_metric"] == "pr_auc"
    assert "recall" in r["secondary_metrics"]


def test_false_positives_costly_routes_to_precision():
    r = select_primary_metric("binary_classification", costlier_errors="false_positives")
    assert r["primary_metric"] == "precision"


def test_multiclass_and_regression_metrics():
    assert select_primary_metric("multiclass_classification")["primary_metric"] == "f1_macro"
    assert select_primary_metric("regression")["primary_metric"] == "rmse"
    assert select_primary_metric("multilabel_classification")["primary_metric"] == "f1_micro"


# -- architecture review agent ----------------------------------------------- #
def test_complex_arch_on_small_tabular_recommends_mlp():
    review = ArchitectureReviewAgent().review(
        user_family="wide_deep", user_activation="gelu", modality="tabular",
        n_samples=569, n_features=30, task_type="binary_classification",
    )
    assert review.recommendation["family"] == "mlp"
    assert not review.agrees
    assert review.reason and review.evidence_id
    assert "overfitting" in review.risk_if_ignored.lower()


def test_appropriate_choice_agrees():
    review = ArchitectureReviewAgent().review(
        user_family="mlp", user_activation="relu", modality="tabular",
        n_samples=569, n_features=30, task_type="binary_classification",
    )
    assert review.agrees
    assert review.recommendation == review.user_choice


def test_sequence_modality_recommends_recurrent():
    review = ArchitectureReviewAgent().review(
        user_family="mlp", user_activation="relu", modality="sequence",
        n_samples=2000, n_features=5, task_type="binary_classification",
    )
    assert review.recommendation["family"] in ("lstm", "gru", "rnn", "bi_lstm")
    assert not review.agrees


def test_vision_modality_recommends_cnn():
    review = ArchitectureReviewAgent().review(
        user_family="mlp", user_activation="relu", modality="vision",
        n_samples=1000, n_features=0, task_type="multiclass_classification",
    )
    assert review.recommendation["family"].startswith("simple_cnn")


def test_large_dataset_keeps_complex_arch():
    review = ArchitectureReviewAgent().review(
        user_family="wide_deep", user_activation="relu", modality="tabular",
        n_samples=50000, n_features=40, task_type="binary_classification",
    )
    assert review.agrees  # enough data to justify the larger model


def test_review_to_dict_complete():
    review = ArchitectureReviewAgent().review(
        user_family="residual_mlp", user_activation="selu", modality="tabular",
        n_samples=300, n_features=10, task_type="binary_classification",
    )
    d = review.to_dict()
    for key in ("user_choice", "recommendation", "reason", "evidence_id",
                "risk_if_ignored", "agrees"):
        assert key in d


# -- hyperparameter tuning agent --------------------------------------------- #
def test_tuning_plan_is_leakage_safe():
    plan = HyperparameterTuningAgent().plan(task_type="binary_classification", n_samples=569)
    assert plan.validation == "train_internal_holdout"  # never test/OOS
    assert plan.early_stopping is True
    assert 5 <= plan.n_trials <= 15
    assert "learning_rate" in plan.search_space


def test_tuning_trials_scale_with_data():
    small = HyperparameterTuningAgent().plan(task_type="binary_classification", n_samples=500)
    large = HyperparameterTuningAgent().plan(task_type="binary_classification", n_samples=100000)
    assert small.n_trials <= large.n_trials


def test_tuning_metric_follows_cost_preference():
    plan = HyperparameterTuningAgent().plan(
        task_type="binary_classification", n_samples=1000, costlier_errors="false_negatives"
    )
    assert plan.primary_metric == "pr_auc"


def test_tuning_records_outcome():
    agent = HyperparameterTuningAgent()
    plan = agent.plan(task_type="binary_classification", n_samples=1000)
    plan = agent.record_outcome(plan, {"learning_rate": 3e-3}, [{"learning_rate": 1e-2}])
    assert plan.best_params == {"learning_rate": 3e-3}
    assert len(plan.rejected_params) == 1
    assert "best_params" in plan.to_dict()
