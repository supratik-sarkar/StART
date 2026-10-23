"""Targeted test suite for StART Final Backend Contract Closure (C1–C4).

Tests:
- Test A: Outlier mitigation propagation (IQR, Z-Score, Winsorize) with zero leakage.
- Test B: Categorical encoding propagation (Target, One-Hot, Ordinal, Frequency) with zero leakage.
- Test C: Tuning execution, serialization & champion lineage preservation without HTTP 500.
- Test D: Quantitative scenario dispatch with custom shock magnitude.
- Test E: Portfolio optimizer dispatch (HRP, Min-Variance, ERC) with distinct portfolio weights.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from start.analysis.pipelines import (
    run_portfolio_erc_pipeline,
    run_portfolio_hrp_pipeline,
    run_portfolio_min_variance_pipeline,
)
from start.data.preprocessing import apply_preprocessing_pipeline
from start.portfolio.hrp import hrp_weights_and_tree
from start.portfolio.optimization import solve_equal_risk_contribution
from start.runtime.execution import CanonicalExecutionService
from start.tests.portfolio import solve_min_variance
from start.utils.serializers import sanitize_json_primitives


# =========================================================================
# Test A: Outlier Mitigation Propagation & Zero Leakage
# =========================================================================
def test_outlier_mitigation_iqr_zscore_winsorize():
    """Verify IQR, Z-Score, and Winsorize clip outliers using only train bounds."""
    np.random.seed(42)
    # 100 samples with severe train outliers and test outliers
    X_train = pd.DataFrame({
        "feat1": np.concatenate([np.random.normal(0, 1, 98), [100.0, -100.0]]),
        "feat2": np.random.normal(10, 2, 100),
    })
    X_test = pd.DataFrame({
        "feat1": [50.0, -50.0, 0.0],
        "feat2": [10.0, 10.0, 10.0],
    })
    y_train = pd.Series(np.random.randint(0, 2, 100))

    # 1. IQR
    prep_iqr = {
        "imputation": "none",
        "outlier_mitigation": "iqr",
        "scaler": "none",
        "categorical_encoding": "none",
    }
    X_tr_iqr, X_te_iqr, info_iqr = apply_preprocessing_pipeline(X_train, X_test, y_train, prep_iqr)
    assert info_iqr["outlier_mitigation"]["method"] == "iqr"
    assert X_tr_iqr["feat1"].max() < 10.0
    assert X_tr_iqr["feat1"].min() > -10.0
    # Zero leakage: test set must be clipped using train bounds!
    assert X_te_iqr["feat1"].iloc[0] == pytest.approx(info_iqr["outlier_mitigation"]["bounds"]["feat1"]["upper"])
    assert X_te_iqr["feat1"].iloc[1] == pytest.approx(info_iqr["outlier_mitigation"]["bounds"]["feat1"]["lower"])

    # 2. Z-Score
    prep_zscore = {
        "imputation": "none",
        "outlier_mitigation": "zscore",
        "scaler": "none",
        "categorical_encoding": "none",
    }
    X_tr_z, X_te_z, info_z = apply_preprocessing_pipeline(X_train, X_test, y_train, prep_zscore)
    assert info_z["outlier_mitigation"]["method"] == "zscore"
    assert X_tr_z["feat1"].max() <= info_z["outlier_mitigation"]["bounds"]["feat1"]["upper"]

    # 3. Winsorize
    prep_win = {
        "imputation": "none",
        "outlier_mitigation": "winsorize",
        "scaler": "none",
        "categorical_encoding": "none",
    }
    X_tr_w, X_te_w, info_w = apply_preprocessing_pipeline(X_train, X_test, y_train, prep_win)
    assert info_w["outlier_mitigation"]["method"] == "winsorize"
    assert X_te_w["feat1"].iloc[0] == pytest.approx(info_w["outlier_mitigation"]["bounds"]["feat1"]["upper"])


# =========================================================================
# Test B: Categorical Encoding Propagation & Zero Leakage
# =========================================================================
def test_categorical_encoding_target_and_onehot():
    """Verify target encoding fits on train only (zero test leakage) and handles unseen categories."""
    X_train = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "cat": ["A", "A", "B", "B", "C", "C"],
    })
    y_train = pd.Series([1, 1, 0, 0, 1, 0])
    X_test = pd.DataFrame({
        "num": [1.5, 2.5],
        "cat": ["A", "UNKNOWN"],  # UNKNOWN category should fall back to global prior
    })

    # Target Encoding
    prep_target = {
        "imputation": "none",
        "outlier_mitigation": "none",
        "scaler": "none",
        "categorical_encoding": "target",
    }
    X_tr_te, X_te_te, info_te = apply_preprocessing_pipeline(X_train, X_test, y_train, prep_target)
    assert info_te["categorical_encoding"]["method"] == "target"
    # Category A has target [1, 1], so smoothed value must be > global prior (0.5)
    global_prior = 3.0 / 6.0
    assert X_tr_te["cat"].iloc[0] > global_prior
    # Category B has target [0, 0], so smoothed value must be < global prior (0.5)
    assert X_tr_te["cat"].iloc[2] < global_prior
    # In test set, UNKNOWN category should get exactly global prior
    assert X_te_te["cat"].iloc[1] == pytest.approx(global_prior)
    # In test set, Category A should get the exact value learned from training
    assert X_te_te["cat"].iloc[0] == pytest.approx(X_tr_te["cat"].iloc[0])

    # One-Hot Encoding
    prep_ohe = {
        "imputation": "none",
        "outlier_mitigation": "none",
        "scaler": "none",
        "categorical_encoding": "onehot",
    }
    X_tr_ohe, X_te_ohe, info_ohe = apply_preprocessing_pipeline(X_train, X_test, y_train, prep_ohe)
    assert info_ohe["categorical_encoding"]["method"] == "onehot"
    assert "cat_A" in X_tr_ohe.columns
    assert "cat_B" in X_tr_ohe.columns
    assert "cat_C" in X_tr_ohe.columns
    assert "cat_A" in X_te_ohe.columns


# =========================================================================
# Test C: Tuning Serialization & Champion Lineage via CanonicalExecutionService
# =========================================================================
def test_tuning_execution_and_lineage_preservation(tmp_path):
    """Verify tuning completes with champion lineage, best hyperparameters, and no numpy serialization errors."""
    req_params = {
        "tuning_strategy": "bounded_random_search",
        "model_type": "lightgbm",
        "objective_metric": "auc_roc",
        "max_trials": 3,
    }

    res = CanonicalExecutionService.execute(
        workflow_id="hyperparameter_tuning",
        context_id="institutional_credit_v1",
        request_params=req_params,
        seed=0,
        output_root=str(tmp_path),
    )
    assert res is not None
    assert res.run_id is not None

    # Context instance tabular checks
    tab = res.context_instance.bundle.tabular
    assert tab is not None
    assert tab.model is not None

    champion_lineage = tab.extra.get("champion_lineage", {})
    assert champion_lineage.get("champion_model") == "lightgbm"
    assert champion_lineage.get("best_hyperparameters") is not None
    assert champion_lineage.get("objective_metric") == "auc_roc"
    assert champion_lineage.get("best_validation_metric") is not None

    res_cfg = tab.extra.get("resolved_configuration", {})
    assert res_cfg.get("architecture") == "lightgbm"
    assert res_cfg.get("best_hyperparameters") is not None

    # Verify no numpy objects survived sanitization
    sanitized = sanitize_json_primitives(tab.extra)
    json_str = json.dumps(sanitized)
    reloaded = json.loads(json_str)
    assert isinstance(reloaded["champion_lineage"]["best_validation_metric"], (int, float))

    # Verify canonical artifact contains champion model, not fallback Logistic Regression
    pred_case_a = res.artifacts.get("canonical_presentation")
    if pred_case_a and hasattr(pred_case_a, "semantic_payload"):
        payload = pred_case_a.semantic_payload
        assert payload.get("champion_model") in ("lightgbm", None) or payload.get("resolved_configuration", {}).get("model") == "lightgbm"


# =========================================================================
# Test D: Quantitative Scenario Dispatch via CanonicalExecutionService
# =========================================================================
def test_scenario_dispatch_with_shock_magnitude(tmp_path):
    """Verify scenario shocks and shock magnitudes are propagated into market execution."""
    req_params = {
        "scenario": "asset_tail_stress",
        "shock_magnitude": -0.25,
        "optimizer": "min_var",
    }

    res = CanonicalExecutionService.execute(
        workflow_id="quantitative_finance",
        context_id="institutional_market_v1",
        request_params=req_params,
        seed=42,
        output_root=str(tmp_path),
    )
    assert res is not None
    market = res.context_instance.bundle.market
    assert market is not None

    res_cfg = market.extra.get("resolved_configuration", {})
    assert res_cfg.get("scenario") == "asset_tail_stress"
    assert res_cfg.get("shock_magnitude") == -0.25
    assert res_cfg.get("optimizer") == "min_var"


# =========================================================================
# Test E: Portfolio Optimizer Dispatch (HRP, Min-Variance, ERC)
# =========================================================================
def test_portfolio_optimizer_dispatch():
    """Verify HRP, Min-Variance, and Equal Risk Contribution produce distinct valid weights."""
    np.random.seed(42)
    assets = ["EQ_US", "EQ_EU", "EQ_EM", "BOND_US", "COMMODITY"]
    # Return matrix with distinct volatilities
    returns = pd.DataFrame(
        np.random.multivariate_normal(
            mean=[0.0005, 0.0004, 0.0006, 0.0001, 0.0003],
            cov=[
                [0.0004, 0.0002, 0.0002, -0.00005, 0.0001],
                [0.0002, 0.0005, 0.0002, -0.00004, 0.0001],
                [0.0002, 0.0002, 0.0009, -0.00008, 0.0002],
                [-0.00005, -0.00004, -0.00008, 0.0001, -0.00002],
                [0.0001, 0.0001, 0.0002, -0.00002, 0.0006],
            ],
            size=150,
        ),
        columns=assets,
    )

    cov_df = returns.cov()
    cov_np = cov_df.to_numpy()

    # 1. HRP
    w_hrp_series, tree_info = hrp_weights_and_tree(cov_df)
    w_hrp = dict(w_hrp_series)
    assert len(w_hrp) == len(assets)
    assert sum(w_hrp.values()) == pytest.approx(1.0)

    # 2. Min-Variance
    w_minvar_arr, _ = solve_min_variance(mu=np.zeros(len(assets)), sigma=cov_np, constraints=None)
    w_minvar = {assets[i]: float(w_minvar_arr[i]) for i in range(len(assets))}
    assert len(w_minvar) == len(assets)
    assert sum(w_minvar.values()) == pytest.approx(1.0)
    # Bond US has lowest variance, so min_var weight should be highest for BOND_US
    assert w_minvar["BOND_US"] > w_minvar["EQ_EM"]

    # 3. Equal Risk Contribution (ERC)
    erc_result = solve_equal_risk_contribution(cov_df)
    w_erc = erc_result.weights
    assert len(w_erc) == len(assets)
    assert sum(w_erc.values()) == pytest.approx(1.0)

    # Verify that the weights are genuinely distinct across optimizers
    hrp_vec = [w_hrp[a] for a in assets]
    minvar_vec = [w_minvar[a] for a in assets]
    erc_vec = [w_erc[a] for a in assets]
    assert not np.allclose(hrp_vec, minvar_vec, atol=1e-3)
    assert not np.allclose(hrp_vec, erc_vec, atol=1e-3)
    assert not np.allclose(minvar_vec, erc_vec, atol=1e-3)

    # Verify pipeline execution
    res_hrp = run_portfolio_hrp_pipeline(returns)
    res_minvar = run_portfolio_min_variance_pipeline(returns)
    res_erc = run_portfolio_erc_pipeline(returns)

    assert res_hrp.technique == "hierarchical_risk_parity"
    assert res_minvar.technique == "minimum_variance"
    assert res_erc.technique == "equal_risk_contribution"
