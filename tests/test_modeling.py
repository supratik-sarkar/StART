from __future__ import annotations

import numpy as np
import pytest

from start.modeling.data import (
    TARGET_COLUMN,
    feature_columns,
    load_attrition_dataset,
    three_way_split,
)
from start.modeling.explain import global_importance
from start.modeling.metrics import compute_cohort_metrics, top_decile_lift
from start.modeling.models import resolve_model
from start.modeling.sensitivity import DEFAULT_SHOCKS, run_feature_shocks
from start.modeling.tuning import tune_model


@pytest.fixture(scope="module")
def attrition_splits():
    df = load_attrition_dataset(seed=0)
    return three_way_split(df, TARGET_COLUMN, seed=0)


@pytest.fixture(scope="module")
def fitted_rf(attrition_splits):
    train, _, _ = attrition_splits
    features = feature_columns(train)
    model, name, note = resolve_model("random_forest", seed=0)
    model.fit(train[features], train[TARGET_COLUMN])
    return model, features


def test_three_way_split_proportions_and_stratification(attrition_splits):
    train, test, oos = attrition_splits
    total = len(train) + len(test) + len(oos)
    assert abs(len(train) / total - 0.6) < 0.02
    assert abs(len(test) / total - 0.2) < 0.02
    assert abs(len(oos) / total - 0.2) < 0.02
    overall = (train[TARGET_COLUMN].sum() + test[TARGET_COLUMN].sum() + oos[TARGET_COLUMN].sum()) / total
    for cohort in (train, test, oos):
        assert abs(cohort[TARGET_COLUMN].mean() - overall) < 0.05
    # splits are disjoint by index construction
    assert len(train) + len(test) + len(oos) == total


def test_top_decile_lift_known_values():
    # Perfect ranking: all 10 positives in the top decile of 100 rows.
    y = np.array([1] * 10 + [0] * 90)
    scores = np.linspace(1.0, 0.0, 100)
    assert top_decile_lift(y, scores) == pytest.approx(10.0)  # 100% vs 10% base rate
    # Random-like: positives spread uniformly -> lift ~ 1
    rng = np.random.default_rng(0)
    y2 = rng.integers(0, 2, 10_000)
    s2 = rng.random(10_000)
    assert top_decile_lift(y2, s2) == pytest.approx(1.0, abs=0.15)


def test_compute_cohort_metrics_keys_and_ranges(fitted_rf, attrition_splits):
    model, features = fitted_rf
    _, test, _ = attrition_splits
    scores = model.predict_proba(test[features])[:, 1]
    metrics = compute_cohort_metrics(test[TARGET_COLUMN].to_numpy(), scores)
    assert set(metrics) == {"auc_roc", "accuracy", "precision", "recall", "f1", "top_decile_lift"}
    assert 0.5 < metrics["auc_roc"] <= 1.0
    assert metrics["top_decile_lift"] > 1.0


def test_xgboost_unavailable_falls_back_cleanly(monkeypatch):
    import start.modeling.models as models

    monkeypatch.setattr(models, "xgboost_available", lambda: False)
    model, name, note = models.resolve_model("xgboost", seed=0)
    assert name == "random_forest"
    assert "xgboost is not installed" in note


def test_lightgbm_unavailable_falls_back_cleanly(monkeypatch):
    import start.modeling.models as models

    monkeypatch.setattr(models, "lightgbm_available", lambda: False)
    model, name, note = models.resolve_model("lightgbm", seed=0)
    assert name == "random_forest"
    assert "lightgbm is not installed" in note


def test_shap_fallback_to_permutation_is_honest(fitted_rf, attrition_splits):
    model, features = fitted_rf
    _, test, _ = attrition_splits
    result = global_importance(
        model, test[features], test[TARGET_COLUMN], seed=0, use_shap=False
    )
    assert result.method == "permutation"
    assert "permutation" in result.note
    assert result.local_examples == []  # no fabricated local attributions
    assert len(result.top_features(5)) == 5


def test_sensitivity_zero_shock_equals_baseline(fitted_rf, attrition_splits):
    model, features = fitted_rf
    _, test, _ = attrition_splits
    imp = global_importance(model, test[features], test[TARGET_COLUMN], seed=0, use_shap=False)
    rows = run_feature_shocks(
        model, test, imp.top_features(5), TARGET_COLUMN, features, DEFAULT_SHOCKS
    )
    zero = next(r for r in rows if r["shock"] == 0.0)
    from sklearn.metrics import roc_auc_score

    baseline = roc_auc_score(
        test[TARGET_COLUMN], model.predict_proba(test[features])[:, 1]
    )
    assert zero["auc_roc"] == round(baseline, 6)  # engine reports 6-decimal precision
    assert zero["auc_drift"] == 0.0
    assert len(rows) == len(DEFAULT_SHOCKS)


def test_optuna_unavailable_skips_gracefully(monkeypatch, fitted_rf, attrition_splits):
    import start.modeling.tuning as tuning

    monkeypatch.setattr(tuning, "optuna_available", lambda: False)
    model, features = fitted_rf
    train, _, _ = attrition_splits
    from start.modeling.models import HYPERPARAM_SPACES

    estimator, outcome = tuning.tune_model(
        model,
        train[features],
        train[TARGET_COLUMN],
        method="optuna",
        space=HYPERPARAM_SPACES["random_forest"],
        cv_folds=3,
        seed=0,
    )
    assert outcome.method == "none"
    assert "optuna is not installed" in outcome.note


def test_grid_search_with_kfold_runs_small(attrition_splits):
    train, _, _ = attrition_splits
    features = feature_columns(train)
    from sklearn.ensemble import RandomForestClassifier

    tiny_space = {
        "n_estimators": {"type": "int", "grid": [20, 40]},
        "max_depth": {"type": "int", "grid": [3, 5]},
    }
    estimator, outcome = tune_model(
        RandomForestClassifier(random_state=0, n_jobs=-1),
        train[features].head(200),
        train[TARGET_COLUMN].head(200),
        method="grid",
        space=tiny_space,
        cv_folds=3,
        seed=0,
    )
    assert outcome.method == "grid"
    assert outcome.cv_folds == 3
    assert outcome.n_candidates == 4
    assert set(outcome.best_params) == {"n_estimators", "max_depth"}
    assert outcome.best_cv_auc is not None and outcome.best_cv_auc > 0.8
