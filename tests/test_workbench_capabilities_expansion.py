"""Authoritative test suite for StART Agentic AI Engineering Workbench expansion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from start.data.uci_credit import fetch_or_load_german_credit
from start.modeling.models import HYPERPARAM_SPACES, resolve_model
from start.modeling.sensitivity_analysis import DEFAULT_SHOCKS, run_sensitivity_analysis
from start.recommender.fixtures import DATASET_C
from start.recommender.models import FieldAwareFactorizationMachineModel
from start.web.app import app


def test_capability_manifest_endpoint():
    client = TestClient(app)
    response = client.get("/api/v1/capability-manifest")
    assert response.status_code == 200
    data = response.json()

    # Workbench identity
    assert data["workbench"]["name"] == "StART — Agentic AI Engineering Workbench"
    assert data["workbench"]["tagline"] == "Build · Tune · Stress · Explain · Compare · Govern"
    assert data["workbench"]["version"] == "4.0.0"
    assert data["workbench"]["default_execution_mode"] == "hybrid_workbench"
    assert data["workbench"]["deterministic_science_invariant"] is True

    # Three execution modes
    modes = {m["id"] for m in data["execution_modes"]}
    assert "hybrid_workbench" in modes
    assert "agentic_session" in modes
    assert "deterministic_run" in modes

    # Predictive ML expansion
    pred = data["domains"]["predictive_ml"]
    assert "logistic_regression" in pred["supported_models"]
    assert "gradient_boosting" in pred["supported_models"]
    assert "lightgbm" in pred["supported_models"]
    assert "xgboost" in pred["supported_models"]
    assert "catboost" in pred["supported_models"]
    assert "random_forest" in pred["supported_models"]
    assert "optuna_bayesian" in pred["tuning_strategies"]
    assert len(pred["sensitivity_analysis"]["shocks_grid"]) == 9
    assert "parallel_basket" in pred["sensitivity_analysis"]["modes"]

    # Recommender systems
    rec = data["domains"]["recommender_systems"]
    assert "field_aware_factorization_machine" in rec["algorithms"]
    assert "factorization_machine" in rec["algorithms"]
    assert "neural_collaborative_filtering" in rec["algorithms"]
    assert "matrix_factorization" in rec["algorithms"]

    # Deferred items
    deferred_caps = {d["capability"] for d in data["deferred_capabilities"]}
    assert any("Monte Carlo" in c for c in deferred_caps)
    assert any("Ray" in c for c in deferred_caps)

    # AI provider (strict gpt-5.1)
    assert data["ai_provider"]["provider"] == "openai"
    assert data["ai_provider"]["model"] == "gpt-5.1"
    assert data["ai_provider"]["strict_zero_substitution"] is True


def test_predictive_models_expansion():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression

    lr, name_lr, note_lr = resolve_model("logistic_regression")
    assert isinstance(lr, LogisticRegression)
    assert name_lr == "logistic_regression"
    assert note_lr == ""
    assert "logistic_regression" in HYPERPARAM_SPACES

    gb, name_gb, note_gb = resolve_model("gradient_boosting")
    assert isinstance(gb, GradientBoostingClassifier)
    assert name_gb == "gradient_boosting"
    assert note_gb == ""
    assert "gradient_boosting" in HYPERPARAM_SPACES


def test_sensitivity_9point_and_basket_mode():
    from sklearn.linear_model import LogisticRegression

    assert len(DEFAULT_SHOCKS) == 9
    assert DEFAULT_SHOCKS == (-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30)

    np.random.seed(42)
    X = pd.DataFrame({
        "feat_a": np.random.randn(100),
        "feat_b": np.random.randn(100),
        "feat_c": np.random.randn(100),
    })
    y = (X["feat_a"] + 0.5 * X["feat_b"] > 0).astype(int).values

    model = LogisticRegression(random_state=42)
    model.fit(X, y)

    # One at a time
    res_oat = run_sensitivity_analysis(model, X, y, top_features=["feat_a", "feat_b"], mode="one_at_a_time")
    assert len(res_oat.shock_rows) == 2 * 9

    # Parallel basket
    res_basket = run_sensitivity_analysis(model, X, y, top_features=["feat_a", "feat_b"], mode="parallel_basket")
    assert len(res_basket.shock_rows) == 9
    assert all("[PARALLEL BASKET" in r.feature for r in res_basket.shock_rows)


def test_field_aware_factorization_machine():
    ffm = FieldAwareFactorizationMachineModel(latent_dim=4, epochs=3, seed=42)
    summary = ffm.fit(DATASET_C)

    assert summary.algorithm == "field_aware_factorization_machine"
    assert summary.training_summary["n_fields"] > 2
    assert summary.training_summary["n_features"] > 0

    score = ffm.predict_score("U1", "I1")
    assert 0.0 <= score <= 1.0

    recs = ffm.recommend("U1", k=5)
    assert len(recs) <= 5
    assert len(recs) == len(set(recs))


def test_semantic_dataset_uci_german_credit():
    df = fetch_or_load_german_credit()
    assert len(df) == 1000
    assert "credit_amount" in df.columns
    assert "duration_months" in df.columns
    assert "age_years" in df.columns
    assert "is_bad_credit" in df.columns
    assert len(df.columns) == 21
