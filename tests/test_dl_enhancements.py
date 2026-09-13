from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="DL enhancements test requires the [torch] extra")

from start.modeling.data import load_preset_dataset
from start.modeling.tabular_dl import TabularDLClassifier
from start.modeling.tabular_dl_metrics import dl_task_metrics
from start.modeling.tuning_run import run_tuning


def test_preset_datasets_nan_and_outliers():
    # Anomaly preset
    df_a = load_preset_dataset("A", seed=0)
    assert "is_fraud" in df_a.columns
    assert df_a.isna().any().any()
    
    # Regression presets
    df_b = load_preset_dataset("B", seed=0)
    assert "target_value" in df_b.columns
    assert df_b.isna().any().any()
    
    df_c = load_preset_dataset("C", seed=0)
    assert "adjusted_price" in df_c.columns
    assert df_c.isna().any().any()
    
    # Multiclass preset
    df_d = load_preset_dataset("D", seed=0)
    assert "decision_label" in df_d.columns
    assert df_d.isna().any().any()


def test_gnn_dcn_fitting():
    # Try GNN
    df = load_preset_dataset("A", seed=0)
    features = [c for c in df.columns if c != "is_fraud"]
    
    # GNN fit
    clf_gnn = TabularDLClassifier(
        task="binary_classification", family="gnn", epochs=3, random_state=0
    )
    clf_gnn.fit(df[features], df["is_fraud"])
    proba = clf_gnn.predict_proba(df[features])
    assert proba.shape == (len(df), 2)
    
    # DCN fit
    clf_dcn = TabularDLClassifier(
        task="binary_classification", family="dcn", epochs=3, random_state=0
    )
    clf_dcn.fit(df[features], df["is_fraud"])
    proba_dcn = clf_dcn.predict_proba(df[features])
    assert proba_dcn.shape == (len(df), 2)


def test_new_activations():
    df = load_preset_dataset("A", seed=0)
    features = [c for c in df.columns if c != "is_fraud"]
    for act in ("swish", "mish", "sigmoid", "softplus"):
        clf = TabularDLClassifier(
            task="binary_classification", activation=act, epochs=2, random_state=0
        )
        clf.fit(df[features], df["is_fraud"])
        assert clf.activation == act
        proba = clf.predict_proba(df[features])
        assert proba.shape == (len(df), 2)


def test_regression_pipeline_and_preprocessing():
    df = load_preset_dataset("B", seed=0)
    features = [c for c in df.columns if c != "target_value"]
    
    # Fit with imputation and winsorization
    clf = TabularDLClassifier(
        task="regression", family="mlp", epochs=5, random_state=0, winsorize=True
    )
    clf.fit(df[features], df["target_value"])
    
    # Preprocessing states should be fitted
    assert clf._medians is not None
    assert clf._mean is not None
    assert clf._std is not None
    assert clf._lower_bounds is not None
    assert clf._upper_bounds is not None

    
    preds = clf.predict(df[features])
    assert preds.shape == (len(df),)
    
    metrics = dl_task_metrics("regression", df["target_value"].to_numpy(), preds)
    for k in ("rmse", "mse", "mae", "r2"):
        assert k in metrics
        assert isinstance(metrics[k], float)


def test_tuning_minimization():
    df = load_preset_dataset("B", seed=0)
    features = [c for c in df.columns if c != "target_value"]
    
    # Test tuning under minimization (RMSE)
    run = run_tuning(
        df, "target_value", features, strategy="bounded_random_search",
        n_trials=3, primary_metric="rmse", seed=0,
        task_type="regression", architecture="mlp"
    )
    assert run is not None
    assert run.ran is True
    assert len(run.trials) == 3
    # The best metric should match the minimum validation metric across ok trials
    ok_metrics = [t.validation_metric for t in run.trials if t.status in ("best", "ok")]
    assert run.best_metric == min(ok_metrics)
