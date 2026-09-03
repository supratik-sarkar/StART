"""Gate A slice A2 — six EDA tests and two compatible preprocessing extensions."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from start.core.schemas import Status
from start.registry import TestContext
from start.tests.eda import (
    categorical_distribution,
    class_imbalance,
    correlation,
    descriptive_statistics,
    multicollinearity,
    numeric_distribution,
)
from start.tests.preprocessing import feature_drift, outliers


def _ctx(df, test=None, target=None, extra=None):
    return TestContext(train=df, test=test, target_column=target, extra=extra or {})


def _numeric(n=300, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})


# ------------------------------------------------- descriptive_statistics --
def test_descriptive_reports_moments_and_percentiles():
    r = descriptive_statistics(_ctx(_numeric()))
    assert r.status == Status.RECORDED
    for key in ("a.mean", "a.std", "a.min", "a.max", "a.skew", "a.excess_kurtosis"):
        assert key in r.metrics
    for pct in (1, 5, 25, 50, 75, 95, 99):
        assert f"a.p{pct}" in r.metrics


def test_descriptive_counts_missing_separately():
    df = _numeric()
    df.loc[:19, "a"] = np.nan
    r = descriptive_statistics(_ctx(df))
    assert r.metrics["a.n_missing"] == 20
    assert r.metrics["a.count"] == 280


def test_descriptive_excludes_rather_than_coerces_categoricals():
    """Coercing a category to numeric produces moments of an arbitrary encoding."""
    df = _numeric()
    df["cat"] = list("ABC") * 100
    r = descriptive_statistics(_ctx(df))
    assert r.metrics["n_excluded_non_numeric"] == 1
    assert "cat.mean" not in r.metrics
    assert any("excluded rather than coerced" in x for x in r.limitations)


def test_descriptive_handles_constant_column():
    df = _numeric()
    df["const"] = 7.0
    r = descriptive_statistics(_ctx(df))
    assert r.metrics["const.std"] == 0.0
    assert r.metrics["const.skew"] == 0.0


def test_descriptive_excludes_target_and_score_columns():
    df = _numeric()
    df["y"] = 1.0
    r = descriptive_statistics(_ctx(df, target="y"))
    assert "y.mean" not in r.metrics


def test_descriptive_skips_when_no_numeric_features():
    df = pd.DataFrame({"cat": list("ABC") * 10})
    assert descriptive_statistics(_ctx(df)).status == Status.SKIPPED


# --------------------------------------------------------------- correlation --
def test_correlation_recovers_known_structure():
    rng = np.random.default_rng(1)
    a = rng.normal(size=800)
    df = pd.DataFrame({"a": a, "pos": a * 2 + rng.normal(scale=0.05, size=800),
                       "neg": -a * 2 + rng.normal(scale=0.05, size=800),
                       "indep": rng.normal(size=800)})
    r = correlation(_ctx(df))
    assert r.metrics["max_abs_correlation"] > 0.95
    # The strongest pair may legitimately be the negative one; magnitude is what ranks.
    assert abs(r.metrics["strongest_pair_correlation"]) > 0.95
    assert r.metrics["pair.a~pos"] > 0.95
    assert r.metrics["pair.a~neg"] < -0.95


def test_correlation_excludes_the_diagonal():
    """Including it makes max_abs_correlation identically 1.0, which is not a finding."""
    df = _numeric()
    r = correlation(_ctx(df))
    assert r.metrics["max_abs_correlation"] < 0.5
    assert r.metrics["n_pairs_above_threshold"] == 0


def test_correlation_counts_pairs_not_cells():
    """Each unordered pair appears twice in the masked matrix."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=500)
    df = pd.DataFrame({"a": a, "b": a + rng.normal(scale=0.01, size=500)})
    r = correlation(_ctx(df), high=0.8)
    assert r.metrics["n_pairs_above_threshold"] == 1


@pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
def test_correlation_supports_all_three_methods(method):
    r = correlation(_ctx(_numeric()), method=method)
    assert r.params["method"] == method
    assert "max_abs_correlation" in r.metrics


def test_correlation_rejects_unknown_method():
    with pytest.raises(ValueError, match="not supported"):
        correlation(_ctx(_numeric()), method="cosine")


def test_correlation_skips_with_fewer_than_two_features():
    assert correlation(_ctx(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))).status == Status.SKIPPED


def test_correlation_evidence_is_complete_without_the_artifact():
    """No conclusion may depend on the figure."""
    r = correlation(_ctx(_numeric()), emit_heatmap=False)
    assert r.artifacts == {}
    assert r.metrics["max_abs_correlation"] is not None
    assert r.status in {Status.PASS, Status.WARN, Status.FAIL}


# ---------------------------------------------------------- multicollinearity --
def test_vif_near_one_for_independent_features():
    r = multicollinearity(_ctx(_numeric(n=600, seed=3)))
    assert r.metrics["max_vif"] < 1.5
    assert r.status == Status.PASS


def test_vif_rises_with_correlation():
    rng = np.random.default_rng(4)
    a = rng.normal(size=600)
    df = pd.DataFrame({"a": a, "b": a * 0.98 + rng.normal(scale=0.2, size=600),
                       "c": rng.normal(size=600)})
    r = multicollinearity(_ctx(df))
    assert r.metrics["max_vif"] > 5.0


def test_perfect_collinearity_is_infinite_not_a_crash():
    rng = np.random.default_rng(5)
    a = rng.normal(size=400)
    df = pd.DataFrame({"a": a, "double": a * 2.0, "c": rng.normal(size=400)})
    r = multicollinearity(_ctx(df))
    assert r.metrics["n_perfectly_collinear"] >= 2
    assert r.status == Status.FAIL


def test_perfect_collinearity_does_not_poison_the_finite_maximum():
    """max_vif = inf would hide how many features are actually affected."""
    rng = np.random.default_rng(6)
    a = rng.normal(size=400)
    df = pd.DataFrame({"a": a, "double": a * 2.0, "c": rng.normal(size=400),
                       "d": rng.normal(size=400)})
    r = multicollinearity(_ctx(df))
    assert math.isfinite(r.metrics["max_vif"])


def test_vif_excludes_categoricals_and_says_so():
    df = _numeric(n=400)
    df["cat"] = list("AB") * 200
    r = multicollinearity(_ctx(df))
    assert "vif.cat" not in r.metrics
    assert any("non-numeric" in x for x in r.limitations)


def test_vif_skips_with_insufficient_columns():
    assert multicollinearity(_ctx(pd.DataFrame({"a": [1.0, 2.0, 3.0]}))).status == Status.SKIPPED


def test_vif_reports_condition_number():
    assert "condition_number" in multicollinearity(_ctx(_numeric(n=400))).metrics


# ------------------------------------------------------- numeric_distribution --
def test_distribution_uses_shapiro_on_small_samples():
    r = numeric_distribution(_ctx(_numeric(n=300, seed=7)))
    assert r.metrics["a.normality_test"] == "shapiro_wilk"


def test_distribution_switches_to_jarque_bera_on_large_samples():
    """Shapiro rejects immaterial departures once n is large; the switch is recorded."""
    r = numeric_distribution(_ctx(_numeric(n=6000, seed=8)))
    assert r.metrics["a.normality_test"] == "jarque_bera"


def test_distribution_does_not_claim_normality():
    r = numeric_distribution(_ctx(_numeric(n=300, seed=9)))
    assert any("has not established normality" in x for x in r.limitations)
    assert "is normal" not in r.interpretation


def test_distribution_handles_constant_and_tiny_columns():
    df = _numeric(n=200)
    df["const"] = 3.0
    r = numeric_distribution(_ctx(df))
    assert r.metrics["const.normality_test"] == "none"
    assert r.status == Status.RECORDED


def test_distribution_drops_infinities_rather_than_propagating_nan():
    """One inf turns every moment into nan and the column's evidence disappears."""
    df = _numeric(n=200)
    df.loc[0, "a"] = np.inf
    r = numeric_distribution(_ctx(df))
    assert r.metrics["a.n"] == 199
    assert math.isfinite(r.metrics["a.skew"])


def test_distribution_skips_when_no_numeric_columns():
    df = pd.DataFrame({"cat": list("AB") * 10})
    assert numeric_distribution(_ctx(df)).status == Status.SKIPPED


# --------------------------------------------------- categorical_distribution --
def test_categorical_reports_structure():
    df = pd.DataFrame({"c": ["A"] * 900 + ["B"] * 95 + ["C"] * 5})
    r = categorical_distribution(_ctx(df))
    assert r.metrics["c.n_unique"] == 3
    assert abs(r.metrics["c.mode_share_pct"] - 90.0) < 1e-6
    assert r.metrics["c.n_rare_levels"] == 1        # C at 0.5% < rare_pct 1.0
    assert r.metrics["c.entropy"] > 0


def test_categorical_rare_boundary_is_strict():
    """A level sitting exactly on the threshold is not rare. Strict `<` is documented
    because implementations differ and the choice changes the count."""
    df = pd.DataFrame({"c": ["A"] * 99 + ["B"]})       # B is exactly 1.0%
    assert categorical_distribution(_ctx(df), rare_pct=1.0).metrics["c.n_rare_levels"] == 0
    assert categorical_distribution(_ctx(df), rare_pct=1.01).metrics["c.n_rare_levels"] == 1


def test_categorical_entropy_is_zero_for_a_single_level():
    df = pd.DataFrame({"c": ["A"] * 50})
    r = categorical_distribution(_ctx(df))
    assert r.metrics["c.entropy"] == 0.0
    assert r.metrics["c.normalised_entropy"] == 0.0


def test_categorical_counts_missing_without_making_it_a_level():
    """Whether missingness is a category is a feature-engineering decision."""
    df = pd.DataFrame({"c": ["A"] * 40 + ["B"] * 40 + [None] * 20})
    r = categorical_distribution(_ctx(df))
    assert r.metrics["c.n_missing"] == 20
    assert r.metrics["c.n_unique"] == 2
    assert any("not treated as a level" in x for x in r.limitations)


def test_categorical_excludes_the_target():
    df = pd.DataFrame({"c": list("AB") * 50, "y": list("XY") * 50})
    r = categorical_distribution(_ctx(df, target="y"))
    assert "y.n_unique" not in r.metrics


def test_categorical_skips_when_none_present():
    assert categorical_distribution(_ctx(_numeric())).status == Status.SKIPPED


# ------------------------------------------------------------ class_imbalance --
def test_imbalance_on_binary_target():
    y = np.array([0] * 950 + [1] * 50)
    df = pd.DataFrame({"a": np.arange(1000, dtype=float), "y": y})
    r = class_imbalance(_ctx(df, target="y"))
    assert abs(r.metrics["minority_class_share"] - 0.05) < 1e-9
    assert r.metrics["n_classes"] == 2
    assert r.status == Status.WARN


def test_imbalance_on_multiclass_target():
    y = np.array(["a"] * 500 + ["b"] * 400 + ["c"] * 100)
    df = pd.DataFrame({"x": np.arange(1000, dtype=float), "y": y})
    r = class_imbalance(_ctx(df, target="y"))
    assert r.metrics["n_classes"] == 3
    assert "class.c.count" in r.metrics


def test_imbalance_skips_cleanly_on_continuous_target():
    """A class distribution over rounded floats looks like a result and means nothing."""
    rng = np.random.default_rng(11)
    df = pd.DataFrame({"x": rng.normal(size=300), "y": rng.normal(size=300)})
    r = class_imbalance(_ctx(df, target="y"))
    assert r.status == Status.SKIPPED
    assert r.test_id == "eda.class_imbalance"
    assert "continuous" in r.interpretation


def test_imbalance_honours_explicit_target_type():
    counts = np.random.default_rng(12).integers(0, 8, 400)
    df = pd.DataFrame({"x": np.arange(400, dtype=float), "y": counts})
    assert class_imbalance(_ctx(df, target="y")).status != Status.SKIPPED
    forced = class_imbalance(_ctx(df, target="y", extra={"target_type": "continuous"}))
    assert forced.status == Status.SKIPPED


def test_imbalance_records_ambiguous_inference():
    counts = np.random.default_rng(13).integers(0, 8, 400)
    df = pd.DataFrame({"x": np.arange(400, dtype=float), "y": counts})
    r = class_imbalance(_ctx(df, target="y"))
    assert r.params["target_type_confidence"] == "ambiguous"
    assert any("inferred, not stated" in x for x in r.limitations)


def test_imbalance_threshold_direction_is_lower():
    """Lower minority share is worse — the opposite of most thresholds here."""
    y = np.array([0] * 999 + [1])
    df = pd.DataFrame({"x": np.arange(1000, dtype=float), "y": y})
    r = class_imbalance(_ctx(df, target="y"))
    assert r.status == Status.FAIL
    assert r.thresholds[0].direction == "lower"


# ------------------------------------------- extension: preprocessing.outliers --
def _outlier_frame():
    rng = np.random.default_rng(21)
    df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    df.loc[:9, "a"] = 50.0
    return df


def test_outliers_legacy_default_payload_is_unchanged():
    """The exact v4.1.1 invocation must produce the exact v4.1.1 evidence."""
    r = outliers(_ctx(_outlier_frame()))
    assert set(r.params) == {"iqr_multiplier", "warn_pct", "fail_pct"}
    assert set(r.metrics) == {"max_outlier_pct", "worst_column", "n_numeric_columns"}
    assert r.artifacts == {}
    assert r.thresholds[0].metric == "max_outlier_pct"


def test_outliers_multiplier_was_already_configurable():
    """The design called for adding `k`; inspection showed iqr_multiplier already exists.
    Two controls for one quantity is how they end up disagreeing."""
    tight = outliers(_ctx(_outlier_frame()), iqr_multiplier=1.0)
    loose = outliers(_ctx(_outlier_frame()), iqr_multiplier=5.0)
    assert tight.metrics["max_outlier_pct"] >= loose.metrics["max_outlier_pct"]


def test_outliers_boxplot_is_opt_in():
    r = outliers(_ctx(_outlier_frame()))
    assert "emit_boxplot" not in r.params


def test_outliers_numeric_evidence_is_identical_with_and_without_the_figure(tmp_path):
    legacy = outliers(_ctx(_outlier_frame()))
    with_fig = outliers(_ctx(_outlier_frame(), extra={"output_dir": str(tmp_path)}),
                        emit_boxplot=True)
    assert legacy.metrics == with_fig.metrics
    assert legacy.status == with_fig.status


# -------------------------------------- extension: preprocessing.feature_drift --
def _drift_frames(shift=0.0):
    rng = np.random.default_rng(22)
    train = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
    test = pd.DataFrame({"a": rng.normal(loc=shift, size=500), "b": rng.normal(size=500)})
    return train, test


def test_drift_legacy_default_payload_is_unchanged():
    train, test = _drift_frames()
    r = feature_drift(_ctx(train, test=test))
    assert set(r.params) == {"psi_warn", "psi_fail"}
    assert set(r.metrics) == {"max_psi", "worst_feature", "min_ks_pvalue", "n_features_checked"}


def test_drift_wasserstein_is_opt_in():
    train, test = _drift_frames()
    off = feature_drift(_ctx(train, test=test))
    on = feature_drift(_ctx(train, test=test), include_wasserstein=True)
    assert "max_wasserstein_normalised" not in off.metrics
    assert "max_wasserstein_normalised" in on.metrics
    assert on.params["include_wasserstein"] is True


def test_drift_legacy_metrics_survive_the_extension():
    """Old metric names and values must not move when the new one is enabled."""
    train, test = _drift_frames(shift=0.8)
    off = feature_drift(_ctx(train, test=test))
    on = feature_drift(_ctx(train, test=test), include_wasserstein=True)
    for key in ("max_psi", "worst_feature", "min_ks_pvalue", "n_features_checked"):
        assert off.metrics[key] == on.metrics[key]
    assert off.status == on.status


def test_drift_wasserstein_detects_a_shift():
    _, _ = _drift_frames()
    quiet = feature_drift(_ctx(*_drift_frames(shift=0.0)), include_wasserstein=True)
    shifted = feature_drift(_ctx(*_drift_frames(shift=1.5)), include_wasserstein=True)
    assert shifted.metrics["max_wasserstein_normalised"] > quiet.metrics["max_wasserstein_normalised"]
