"""A3 — preprocessing diagnostic siblings."""

from __future__ import annotations

import numpy as np
import pandas as pd

from start.core.schemas import Status
from start.registry import TestContext
from start.tests.preprocessing import (
    categorical_drift,
    dimensionality_diagnostic,
    feature_target_relationship,
    leakage_entity_overlap,
    leakage_high_correlation,
    leakage_name_heuristic,
    leakage_row_overlap,
    leakage_suspicious_predictivity,
    leakage_target_reconstruction,
    leakage_temporal,
    redundancy,
    target_analysis,
)


def _ctx(train, test=None, target=None, ts=None, entity=None, extra=None):
    return TestContext(
        train=train,
        test=test,
        target_column=target,
        timestamp_column=ts,
        entity_id_column=entity,
        extra=extra or {},
    )


# ------------------------------------------------- target reconstruction --
def test_exact_reconstruction_fails():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    df = pd.DataFrame({"noise": rng.normal(size=300), "copy": y.astype(float), "y": y})
    r = leakage_target_reconstruction(_ctx(df, target="y"))
    assert r.status == Status.FAIL
    assert r.metrics["reconstruction.copy"] == "exact"


def test_affine_reconstruction_fails():
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    df = pd.DataFrame({"noise": rng.normal(size=300), "aff": 3.0 * y + 7.0, "y": y})
    r = leakage_target_reconstruction(_ctx(df, target="y"))
    assert r.status == Status.FAIL
    assert r.metrics["reconstruction.aff"] == "affine"


def test_invertible_monotone_reconstruction_fails():
    y = np.linspace(0.1, 5.0, 200)
    df = pd.DataFrame({"mono": np.exp(y), "y": y})
    r = leakage_target_reconstruction(_ctx(df, target="y"))
    assert r.status == Status.FAIL
    assert r.metrics["reconstruction.mono"] == "invertible_monotone"


def test_high_correlation_alone_is_not_reconstruction():
    """The distinction the whole split exists for: 0.97 correlation is a finding,
    not a defect."""
    rng = np.random.default_rng(2)
    y = rng.normal(size=600)
    df = pd.DataFrame({"strong": y + rng.normal(scale=0.25, size=600), "y": y})
    assert abs(df["strong"].corr(df["y"])) > 0.95
    assert leakage_target_reconstruction(_ctx(df, target="y")).status == Status.PASS


def test_conflicting_target_for_same_feature_value_is_not_monotone():
    """A mapping is not a function if one x has two different y."""
    x = np.array([1.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([1.0, 9.0, 2.0, 3.0, 4.0, 5.0])
    df = pd.DataFrame({"x": x, "y": y})
    r = leakage_target_reconstruction(_ctx(df, target="y"))
    assert r.metrics.get("reconstruction.x") != "invertible_monotone"


def test_reconstruction_limitations_scope_the_claim():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"a": rng.normal(size=50), "y": rng.normal(size=50)})
    r = leakage_target_reconstruction(_ctx(df, target="y"))
    assert any("OBSERVED SUPPORT" in x for x in r.limitations)


# ------------------------------------------------------- high correlation --
def test_high_correlation_warns_never_fails():
    rng = np.random.default_rng(4)
    y = rng.normal(size=400)
    df = pd.DataFrame({"strong": y + rng.normal(scale=0.05, size=400), "y": y})
    r = leakage_high_correlation(_ctx(df, target="y"))
    assert r.status == Status.WARN
    assert any("WARN only" in x for x in r.limitations)


def test_high_correlation_passes_when_weak():
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"a": rng.normal(size=400), "y": rng.normal(size=400)})
    assert leakage_high_correlation(_ctx(df, target="y")).status == Status.PASS


# -------------------------------------------------------------- temporal --
def _temporal(train_end, test_start, n=100):
    tr = pd.DataFrame({"a": np.arange(n, dtype=float), "t": pd.date_range("2024-01-01", periods=n, freq="D")})
    te = pd.DataFrame({"a": np.arange(n, dtype=float), "t": pd.date_range(test_start, periods=n, freq="D")})
    return tr, te


def test_temporal_leakage_detected():
    tr, te = _temporal(None, "2024-02-01")
    r = leakage_temporal(_ctx(tr, test=te, ts="t"))
    assert r.status == Status.FAIL
    assert r.metrics["n_train_after_test_start"] > 0
    assert r.metrics["overlap_days"] > 0


def test_temporal_clean_split_passes():
    tr, te = _temporal(None, "2025-01-01")
    assert leakage_temporal(_ctx(tr, test=te, ts="t")).status == Status.PASS


def test_temporal_skips_without_configured_column():
    """A timestamp column is never inferred — a finding about a guess is not a finding."""
    tr, te = _temporal(None, "2024-02-01")
    r = leakage_temporal(_ctx(tr, test=te))
    assert r.status == Status.SKIPPED
    assert "not inferred" in r.interpretation


def test_temporal_skips_when_column_absent_from_cohort():
    tr, te = _temporal(None, "2025-01-01")
    r = leakage_temporal(_ctx(tr, test=te.drop(columns=["t"]), ts="t"))
    assert r.status == Status.SKIPPED


# --------------------------------------------------------- entity overlap --
def test_entity_overlap_detected():
    tr = pd.DataFrame({"e": [1, 2, 3, 4], "a": [1.0, 2, 3, 4]})
    te = pd.DataFrame({"e": [3, 4, 5], "a": [3.0, 4, 5]})
    r = leakage_entity_overlap(_ctx(tr, test=te, entity="e"))
    assert r.status == Status.FAIL
    assert r.metrics["n_shared_entities"] == 2


def test_entity_disjoint_passes():
    tr = pd.DataFrame({"e": [1, 2], "a": [1.0, 2]})
    te = pd.DataFrame({"e": [3, 4], "a": [3.0, 4]})
    assert leakage_entity_overlap(_ctx(tr, test=te, entity="e")).status == Status.PASS


def test_entity_overlap_does_not_claim_universal_invalidity():
    tr = pd.DataFrame({"e": [1, 2], "a": [1.0, 2]})
    te = pd.DataFrame({"e": [2, 3], "a": [2.0, 3]})
    r = leakage_entity_overlap(_ctx(tr, test=te, entity="e"))
    assert any("declared validation design" in x for x in r.limitations)


# ------------------------------------------------------------ row overlap --
def test_row_overlap_detected():
    tr = pd.DataFrame({"a": [1.0, 2, 3], "b": ["x", "y", "z"]})
    te = pd.DataFrame({"a": [3.0, 4], "b": ["z", "w"]})
    r = leakage_row_overlap(_ctx(tr, test=te))
    assert r.status == Status.FAIL
    assert r.metrics["n_duplicate_across_split"] == 1


def test_row_overlap_clean_passes():
    tr = pd.DataFrame({"a": [1.0, 2], "b": ["x", "y"]})
    te = pd.DataFrame({"a": [3.0, 4], "b": ["z", "w"]})
    assert leakage_row_overlap(_ctx(tr, test=te)).status == Status.PASS


def test_row_overlap_is_reproducible_across_processes():
    """String rendering, never Python hash() — which varies per PYTHONHASHSEED."""
    tr = pd.DataFrame({"a": [1.0, 2, 3], "b": ["x", "y", "z"]})
    te = pd.DataFrame({"a": [3.0], "b": ["z"]})
    a = leakage_row_overlap(_ctx(tr, test=te))
    b = leakage_row_overlap(_ctx(tr, test=te))
    assert a.metrics == b.metrics
    assert any("deterministic across processes" in x for x in a.limitations)


def test_row_overlap_honours_hash_columns():
    tr = pd.DataFrame({"a": [1.0, 2], "b": ["x", "y"]})
    te = pd.DataFrame({"a": [1.0], "b": ["DIFFERENT"]})
    assert leakage_row_overlap(_ctx(tr, test=te)).status == Status.PASS
    r = leakage_row_overlap(_ctx(tr, test=te), hash_columns=("a",))
    assert r.status == Status.FAIL


# -------------------------------------------------- suspicious predictivity --
def test_predictivity_binary_uses_auc():
    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, 400)
    df = pd.DataFrame({"sep": y + rng.normal(scale=0.01, size=400), "noise": rng.normal(size=400), "y": y})
    r = leakage_suspicious_predictivity(_ctx(df, target="y"))
    assert r.status == Status.WARN
    assert r.metrics["metric_used"] == "single_feature_auc"


def test_predictivity_continuous_uses_r_squared_not_auc():
    rng = np.random.default_rng(7)
    y = rng.normal(size=400)
    df = pd.DataFrame({"lin": y * 3 + rng.normal(scale=0.01, size=400), "y": y})
    r = leakage_suspicious_predictivity(_ctx(df, target="y"))
    assert r.metrics["metric_used"] == "univariate_r_squared"
    assert r.status == Status.WARN


def test_predictivity_never_fails():
    rng = np.random.default_rng(8)
    y = rng.integers(0, 2, 300)
    df = pd.DataFrame({"perfect": y.astype(float), "y": y})
    r = leakage_suspicious_predictivity(_ctx(df, target="y"))
    assert r.status != Status.FAIL
    assert any("NOT proof of leakage" in x for x in r.limitations)


# ---------------------------------------------------------- name heuristic --
def test_name_heuristic_flags_outcome_terms():
    df = pd.DataFrame({"target_leak": [1.0], "future_value": [1.0], "clean": [1.0]})
    r = leakage_name_heuristic(_ctx(df))
    assert r.status == Status.WARN
    assert r.metrics["n_flagged"] == 2


def test_name_heuristic_respects_token_boundaries():
    """Substring matching flags 'postcode' for 'post'; token matching does not."""
    df = pd.DataFrame({"postcode": [1.0], "labelled_region": [1.0]})
    r = leakage_name_heuristic(_ctx(df))
    assert r.metrics["n_flagged"] == 0


def test_name_heuristic_handles_camel_case():
    df = pd.DataFrame({"actualOutcome": [1.0]})
    assert leakage_name_heuristic(_ctx(df)).metrics["n_flagged"] == 1


def test_name_heuristic_is_framed_as_an_observation():
    df = pd.DataFrame({"target_x": [1.0]})
    r = leakage_name_heuristic(_ctx(df))
    assert "not a finding about the data" in r.interpretation
    assert any("NAMING OBSERVATION" in x for x in r.limitations)


# -------------------------------------------------------- target analysis --
def test_target_analysis_binary():
    y = np.array([0] * 900 + [1] * 100)
    df = pd.DataFrame({"a": np.arange(1000, dtype=float), "y": y})
    r = target_analysis(_ctx(df, target="y"))
    assert r.metrics["target_kind"] == "binary"
    assert r.metrics["n_positive"] == 100


def test_target_analysis_warns_on_too_few_positives():
    y = np.array([0] * 990 + [1] * 10)
    df = pd.DataFrame({"a": np.arange(1000, dtype=float), "y": y})
    assert target_analysis(_ctx(df, target="y")).status == Status.WARN


def test_target_analysis_multiclass_omits_positive_rate():
    """The binary vocabulary must not be forced onto a multiclass target."""
    y = np.array(["a"] * 400 + ["b"] * 400 + ["c"] * 200)
    df = pd.DataFrame({"x": np.arange(1000, dtype=float), "y": y})
    r = target_analysis(_ctx(df, target="y"))
    assert r.metrics["target_kind"] == "multiclass"
    assert "positive_rate" not in r.metrics
    assert any("does not apply" in x for x in r.limitations)


def test_target_analysis_continuous_gives_moments_not_classes():
    rng = np.random.default_rng(9)
    df = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    r = target_analysis(_ctx(df, target="y"))
    assert r.metrics["target_kind"] == "continuous"
    assert "mean" in r.metrics and "p50" in r.metrics
    assert "positive_rate" not in r.metrics


def test_target_analysis_records_dispatch_params():
    y = np.array([0] * 500 + [1] * 500)
    df = pd.DataFrame({"x": np.arange(1000, dtype=float), "y": y})
    r = target_analysis(_ctx(df, target="y"))
    for key in ("target_type_inferred", "target_type_confidence", "target_type_source"):
        assert key in r.params


# ------------------------------------------------ feature-target relationship --
def test_feature_target_binary_uses_auc_and_iv():
    rng = np.random.default_rng(10)
    y = rng.integers(0, 2, 500)
    df = pd.DataFrame({"a": y + rng.normal(scale=0.5, size=500), "b": rng.normal(size=500), "y": y})
    r = feature_target_relationship(_ctx(df, target="y"))
    assert "auc.a" in r.metrics
    assert "auc" in r.metrics["statistics_used"]


def test_feature_target_continuous_uses_correlation_not_iv():
    rng = np.random.default_rng(11)
    y = rng.normal(size=500)
    df = pd.DataFrame({"a": y + rng.normal(scale=0.5, size=500), "y": y})
    r = feature_target_relationship(_ctx(df, target="y"))
    assert "pearson.a" in r.metrics
    assert not any(k.startswith("iv.") for k in r.metrics)


def test_feature_target_records_exclusions():
    rng = np.random.default_rng(12)
    y = rng.integers(0, 2, 300)
    df = pd.DataFrame({"a": rng.normal(size=300), "cat": list("AB") * 150, "y": y})
    r = feature_target_relationship(_ctx(df, target="y"))
    assert r.metrics["n_features_excluded"] == 1
    assert "cat" in r.metrics["excluded_features"]


# ------------------------------------------------------------- redundancy --
def test_redundancy_detects_near_duplicates():
    rng = np.random.default_rng(13)
    a = rng.normal(size=400)
    df = pd.DataFrame({"a": a, "a_copy": a + rng.normal(scale=0.01, size=400), "c": rng.normal(size=400)})
    r = redundancy(_ctx(df))
    assert r.status == Status.WARN
    assert r.metrics["n_redundant_pairs"] == 1
    assert r.metrics["n_features_removable"] == 1


def test_redundancy_is_diagnostic_only():
    rng = np.random.default_rng(14)
    a = rng.normal(size=300)
    df = pd.DataFrame({"a": a, "b": a * 1.001})
    r = redundancy(_ctx(df))
    assert any("DIAGNOSTIC ONLY" in x for x in r.limitations)


def test_redundancy_passes_when_independent():
    assert (
        redundancy(
            _ctx(pd.DataFrame(np.random.default_rng(15).normal(size=(400, 3)), columns=list("abc")))
        ).status
        == Status.PASS
    )


# -------------------------------------------------- dimensionality diagnostic --
def test_dimensionality_reports_components():
    rng = np.random.default_rng(16)
    base = rng.normal(size=(400, 2))
    df = pd.DataFrame(
        {
            "a": base[:, 0],
            "b": base[:, 1],
            "c": base[:, 0] + 0.01 * rng.normal(size=400),
            "d": base[:, 1] + 0.01 * rng.normal(size=400),
        }
    )
    r = dimensionality_diagnostic(_ctx(df))
    assert r.metrics["n_components_for_target"] <= 3
    assert r.metrics["effective_rank"] < 4


def test_dimensionality_produces_no_transformed_data():
    """It reports a number; pca_transform produces components."""
    df = pd.DataFrame(np.random.default_rng(17).normal(size=(200, 4)), columns=list("abcd"))
    r = dimensionality_diagnostic(_ctx(df))
    assert any("DIAGNOSTIC ONLY" in x for x in r.limitations)
    assert not any(isinstance(v, (pd.DataFrame, np.ndarray)) for v in r.metrics.values())


def test_dimensionality_flags_p_greater_than_n():
    df = pd.DataFrame(np.random.default_rng(18).normal(size=(5, 12)), columns=[f"f{i}" for i in range(12)])
    r = dimensionality_diagnostic(_ctx(df))
    assert r.metrics["p_greater_than_n"] is True
    assert any("p > n" in x for x in r.limitations)


def test_dimensionality_skips_on_insufficient_features():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    assert dimensionality_diagnostic(_ctx(df)).status == Status.SKIPPED


def test_dimensionality_handles_constant_columns():
    rng = np.random.default_rng(19)
    df = pd.DataFrame({"a": rng.normal(size=200), "b": rng.normal(size=200), "const": 5.0})
    r = dimensionality_diagnostic(_ctx(df))
    assert r.metrics["n_features"] == 2


# -------------------------------------------------------- categorical drift --
def test_categorical_drift_detects_shift():
    tr = pd.DataFrame({"c": ["A"] * 500 + ["B"] * 500})
    te = pd.DataFrame({"c": ["A"] * 100 + ["B"] * 900})
    r = categorical_drift(_ctx(tr, test=te))
    assert r.metrics["n_columns_rejected"] == 1
    assert r.status == Status.WARN


def test_categorical_drift_reports_new_levels():
    tr = pd.DataFrame({"c": ["A"] * 200 + ["B"] * 200})
    te = pd.DataFrame({"c": ["A"] * 200 + ["Z"] * 200})
    r = categorical_drift(_ctx(tr, test=te))
    assert r.metrics["n_new_levels_total"] == 1
    assert "Z" in r.metrics["c.new_levels"]


def test_categorical_drift_skips_sparse_tables():
    """A chi-square with tiny expected counts has no inferential meaning."""
    tr = pd.DataFrame({"c": ["A", "B", "C", "D"]})
    te = pd.DataFrame({"c": ["A", "B", "C", "D"]})
    r = categorical_drift(_ctx(tr, test=te))
    assert r.metrics["n_columns_sparse"] == 1
    assert r.metrics["c.test_used"] == "none_sparse"


def test_categorical_drift_does_not_claim_equality():
    tr = pd.DataFrame({"c": ["A"] * 500 + ["B"] * 500})
    te = pd.DataFrame({"c": ["A"] * 500 + ["B"] * 500})
    r = categorical_drift(_ctx(tr, test=te))
    assert r.status == Status.PASS
    assert any("has NOT established" in x for x in r.limitations)
