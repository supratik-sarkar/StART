from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("torch", reason="sensitivity test trains a small model")

from sklearn.datasets import make_classification  # noqa: E402

from start.modeling.sensitivity_analysis import (  # noqa: E402
    DEFAULT_SHOCKS,
    render_sensitivity_markdown,
    run_sensitivity_analysis,
)
from start.modeling.tabular_dl import TabularDLClassifier  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    X, y = make_classification(n_samples=400, n_features=8, n_informative=5, random_state=0)
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    clf = TabularDLClassifier(task="binary_classification", epochs=8, random_state=0).fit(X, y)
    return clf, X, y


def test_shock_grid_is_correct():
    assert DEFAULT_SHOCKS == (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30)


def test_zero_shock_equals_baseline(fitted):
    clf, X, y = fitted
    res = run_sensitivity_analysis(clf, X, y, top_features=["f0", "f1", "f2"], metric_name="auc_roc")
    zero_rows = [r for r in res.shock_rows if r.shock == 0.0]
    assert zero_rows
    for r in zero_rows:
        assert r.metric_value == res.baseline
        assert r.drift == 0.0


def test_top_features_capped(fitted):
    clf, X, y = fitted
    res = run_sensitivity_analysis(
        clf, X, y, top_features=["f0", "f1", "f2", "f3", "f4", "f5", "f6"], max_features=5
    )
    assert len({r.feature for r in res.shock_rows}) == 5


def test_rows_per_feature(fitted):
    clf, X, y = fitted
    res = run_sensitivity_analysis(clf, X, y, top_features=["f0", "f1"])
    per_feature = {}
    for r in res.shock_rows:
        per_feature.setdefault(r.feature, 0)
        per_feature[r.feature] += 1
    assert all(count == len(DEFAULT_SHOCKS) for count in per_feature.values())


def test_most_sensitive_feature_identified(fitted):
    clf, X, y = fitted
    res = run_sensitivity_analysis(clf, X, y, top_features=["f0", "f1", "f2", "f3", "f4"])
    assert res.most_sensitive_feature in {"f0", "f1", "f2", "f3", "f4"}
    assert res.max_abs_drift >= 0.0


def test_metric_routing(fitted):
    clf, X, y = fitted
    auc = run_sensitivity_analysis(clf, X, y, top_features=["f0"], metric_name="auc_roc")
    rec = run_sensitivity_analysis(clf, X, y, top_features=["f0"], metric_name="recall")
    assert auc.metric_name == "auc_roc"
    assert rec.metric_name == "recall"


def test_drift_table_structure(fitted):
    clf, X, y = fitted
    res = run_sensitivity_analysis(clf, X, y, top_features=["f0", "f1"])
    table = res.drift_table()
    assert set(table) == {"f0", "f1"}
    assert "+0%" in table["f0"] and table["f0"]["+0%"] == 0.0


def test_to_dict_and_markdown(fitted):
    clf, X, y = fitted
    res = run_sensitivity_analysis(clf, X, y, top_features=["f0", "f1"])
    d = res.to_dict()
    assert "drift_table" in d and "baseline" in d and "most_sensitive_feature" in d
    md = render_sensitivity_markdown(res)
    assert "### Sensitivity analysis" in md and "| Feature |" in md


def test_missing_feature_skipped(fitted):
    clf, X, y = fitted
    res = run_sensitivity_analysis(clf, X, y, top_features=["f0", "does_not_exist"])
    assert {r.feature for r in res.shock_rows} == {"f0"}


def test_sensitivity_markdown_has_risk_column(fitted):
    clf, X, y = fitted
    res = run_sensitivity_analysis(clf, X, y, top_features=["f0", "f1"])
    md = render_sensitivity_markdown(res)
    assert "Risk impact" in md
    assert "Shocked" in md and "Delta" in md


def test_sensitivity_rows_have_risk_and_baseline(fitted):
    clf, X, y = fitted
    res = run_sensitivity_analysis(clf, X, y, top_features=["f0"])
    rows = res.to_dict()["rows"]
    assert all("risk_impact" in r and "baseline" in r for r in rows)
