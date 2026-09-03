"""A4 — execution contract, transformations, and the fitting-scope audit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from start.core.schemas import Status
from start.registry import TestContext
from start.tests.feature_engineering import (
    aggregation_features,
    categorical_encoding,
    fitting_scope_audit,
    imputation,
    interactions,
    monotonic_binning,
    numeric_transform,
    pca_transform,
    plan,
    rare_category_grouping,
    scaling,
    selection,
    temporal_features,
    winsorization,
    woe_iv,
)
from start.tests.feature_engineering.audit import _perturb_frame, audit_executor
from start.tests.feature_engineering.execution import (
    FittingScope,
    canonical_frame_hash,
    canonical_state_hash,
)
from start.tests.feature_engineering.leaky_fixtures import LEAKY_FIXTURES
from start.tests.feature_engineering.transforms import (
    run_aggregation_features,
    run_categorical_encoding,
    run_imputation,
    run_interactions,
    run_monotonic_binning,
    run_numeric_transform,
    run_pca_transform,
    run_rare_category_grouping,
    run_scaling,
    run_selection,
    run_temporal_features,
    run_winsorization,
    run_woe_iv,
)

RNG = np.random.default_rng(0)


def _frames(n=300, m=90):
    rng = np.random.default_rng(11)
    train = pd.DataFrame({
        "a": rng.normal(10, 3, n), "b": np.abs(rng.normal(5, 2, n)) + 0.5,
        "cat": rng.choice(list("ABCD"), n), "e": rng.choice(["e1", "e2", "e3"], n),
        "t": pd.date_range("2024-01-01", periods=n, freq="D"),
        "y": rng.integers(0, 2, n),
    })
    test = pd.DataFrame({
        "a": rng.normal(40, 3, m), "b": np.abs(rng.normal(9, 2, m)) + 0.5,
        "cat": rng.choice(list("ABCD"), m), "e": rng.choice(["e9"], m),
        "t": pd.date_range("2025-06-01", periods=m, freq="D"),
        "y": rng.integers(0, 2, m),
    })
    return train, test


def _ctx(train=None, test=None, **kw):
    tr, te = _frames()
    return TestContext(train=tr if train is None else train,
                       test=te if test is None else test,
                       target_column=kw.pop("target", "y"),
                       timestamp_column=kw.pop("ts", None),
                       entity_id_column=kw.pop("entity", None),
                       extra=kw.pop("extra", {}) or {})


# =============================================================== execution ==
def test_hash_is_stable_across_calls():
    tr, _ = _frames()
    assert canonical_frame_hash(tr) == canonical_frame_hash(tr)


def test_hash_ignores_column_order_but_not_row_order():
    """Row order is meaningful for temporal features; column order is not."""
    tr, _ = _frames()
    reordered_cols = tr[sorted(tr.columns, reverse=True)]
    assert canonical_frame_hash(tr) == canonical_frame_hash(reordered_cols)
    shuffled_rows = tr.sample(frac=1.0, random_state=5)
    assert canonical_frame_hash(tr) != canonical_frame_hash(shuffled_rows)


def test_hash_normalises_signed_zero_and_handles_nan():
    a = pd.DataFrame({"x": [0.0, np.nan, 1.0]})
    b = pd.DataFrame({"x": [-0.0, np.nan, 1.0]})
    assert canonical_frame_hash(a) == canonical_frame_hash(b)


def test_state_hash_tolerates_last_bit_noise():
    """Two honest fits can differ in the last bit; a raw hash would call that a leak."""
    a = {"scale": 1.0000000000000002}
    b = {"scale": 1.0}
    assert canonical_state_hash(a) == canonical_state_hash(b)


def test_state_hash_detects_a_real_difference():
    assert canonical_state_hash({"scale": 1.0}) != canonical_state_hash({"scale": 1.5})


def test_evidence_metrics_contain_no_frames():
    """The whole point of the contract: frames stay a runtime payload."""
    tr, te = _frames()
    result = run_scaling(tr, te, None, method="standard", exclude=("y",))
    metrics = result.evidence_metrics()
    for value in metrics.values():
        assert not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))


def test_execution_result_carries_the_frames():
    tr, te = _frames()
    result = run_scaling(tr, te, None, method="standard", exclude=("y",))
    assert isinstance(result.transformed_train, pd.DataFrame)
    assert isinstance(result.transformed_test, pd.DataFrame)
    assert result.transformed_oos is None
    assert result.evidence_metrics()["oos_present"] is False


# ========================================================== transformations ==
def test_imputation_fits_on_train_only():
    tr, te = _frames()
    tr.loc[:19, "a"] = np.nan
    result = run_imputation(tr, te, None, strategy="median", exclude=("y",))
    assert abs(result.fitted_state["fill_values"]["a"] - tr["a"].median()) < 1e-9


def test_imputation_leaves_all_missing_columns_alone():
    """Inventing a constant for a column with no data is worse than leaving the gap."""
    tr, te = _frames()
    tr["empty"] = np.nan
    te["empty"] = np.nan
    result = run_imputation(tr, te, None, strategy="median", exclude=("y",))
    assert "empty" not in result.fitted_state["fill_values"]
    assert any("entirely missing" in n for n in result.notes)


def test_imputation_adds_indicator_when_requested():
    tr, te = _frames()
    tr.loc[:19, "a"] = np.nan
    result = run_imputation(tr, te, None, strategy="median", add_indicator=True, exclude=("y",))
    assert "a__missing" in result.transformed_train.columns


def test_scaling_does_not_explode_on_zero_variance():
    """An inf column silently poisons every downstream fit."""
    tr, te = _frames()
    tr["const"] = 5.0
    te["const"] = 5.0
    result = run_scaling(tr, te, None, method="standard", exclude=("y",))
    assert np.isfinite(result.transformed_train["const"]).all()
    assert any("zero-variance" in n for n in result.notes)


@pytest.mark.parametrize("method", ["standard", "robust", "minmax", "maxabs"])
def test_scaling_methods(method):
    tr, te = _frames()
    result = run_scaling(tr, te, None, method=method, exclude=("y",))
    assert result.fitting_scope == FittingScope.TRAIN_ONLY
    assert np.isfinite(result.transformed_train["a"]).all()


def test_boxcox_skips_non_positive_rather_than_shifting():
    """Shifting to force positivity changes what the feature means."""
    tr, te = _frames()
    tr["signed"] = np.linspace(-5, 5, len(tr))
    te["signed"] = np.linspace(-5, 5, len(te))
    result = run_numeric_transform(tr, te, None, method="boxcox", exclude=("y",))
    assert "signed" not in result.fitted_state["lambdas"]
    assert any("domain-invalid" in n for n in result.notes)


def test_yeo_johnson_handles_signed_data():
    tr, te = _frames()
    tr["signed"] = np.linspace(-5, 5, len(tr))
    te["signed"] = np.linspace(-5, 5, len(te))
    result = run_numeric_transform(tr, te, None, method="yeo_johnson", exclude=("y",))
    assert "signed" in result.fitted_state["lambdas"]


def test_stateless_transform_is_declared_stateless():
    tr, te = _frames()
    result = run_numeric_transform(tr, te, None, method="rank", exclude=("y",))
    assert result.fitting_scope == FittingScope.STATELESS


def test_winsorization_bounds_come_from_train_only():
    tr, te = _frames()
    result = run_winsorization(tr, te, None, method="iqr", k=1.5, exclude=("y",))
    lo, hi = result.fitted_state["bounds"]["a"]
    q1, q3 = np.percentile(tr["a"], [25, 75])
    assert abs(lo - (q1 - 1.5 * (q3 - q1))) < 1e-9
    # Evaluation values above the train bound are clipped, not re-fitted.
    assert result.transformed_test["a"].max() <= hi + 1e-9


def test_rare_grouping_maps_unseen_levels_to_other():
    tr, te = _frames()
    te.loc[0, "cat"] = "ZZZ_UNSEEN"
    result = run_rare_category_grouping(tr, te, None, min_pct=1.0, exclude=("y",))
    assert result.transformed_test.loc[0, "cat"] == "__OTHER__"


def test_onehot_encoding_widens_and_drops_source():
    tr, te = _frames()
    result = run_categorical_encoding(tr, te, None, method="onehot",
                                      target_column="y", exclude=())
    assert "cat" not in result.transformed_train.columns
    assert any(c.startswith("cat__") for c in result.transformed_train.columns)


def test_ordinal_encoding_uses_sentinel_for_unseen():
    tr, te = _frames()
    te.loc[0, "cat"] = "ZZZ"
    result = run_categorical_encoding(tr, te, None, method="ordinal",
                                      target_column="y", exclude=())
    assert result.transformed_test.loc[0, "cat"] == -1


def test_target_encoding_is_declared_out_of_fold():
    tr, te = _frames()
    result = run_categorical_encoding(tr, te, None, method="target",
                                      target_column="y", exclude=())
    assert result.fitting_scope == FittingScope.TRAIN_FOLDS
    assert any("OUT-OF-FOLD" in n for n in result.notes)


def test_woe_requires_binary_target():
    tr, te = _frames()
    tr["y"] = np.linspace(0, 100, len(tr))
    with pytest.raises(ValueError, match="binary"):
        run_woe_iv(tr, te, None, target_column="y")


def test_woe_reports_information_value():
    tr, te = _frames()
    result = run_woe_iv(tr, te, None, target_column="y")
    assert result.fitted_state["iv"]
    assert result.fitting_scope == FittingScope.TRAIN_FOLDS


def test_monotonic_binning_documents_its_merge_rule():
    """Several algorithms exist; the one used must be stated."""
    tr, te = _frames()
    result = run_monotonic_binning(tr, te, None, target_column="y")
    assert any("lowest-indexed" in n for n in result.notes)
    assert any("not the uniquely canonical" in n for n in result.notes)


def test_monotonic_binning_is_deterministic():
    tr, te = _frames()
    a = run_monotonic_binning(tr, te, None, target_column="y")
    b = run_monotonic_binning(tr, te, None, target_column="y")
    assert a.state_hash() == b.state_hash()


def test_ratio_interaction_produces_no_infinities():
    tr, te = _frames()
    tr["zero"] = 0.0
    te["zero"] = 0.0
    result = run_interactions(tr, te, None, method="ratio", max_features=10, exclude=("y",))
    numeric = result.transformed_train.select_dtypes(include=[np.number])
    assert np.isfinite(numeric.to_numpy()).all()


def test_interactions_are_capped_and_deterministic():
    tr, te = _frames()
    a = run_interactions(tr, te, None, method="product", max_features=3, exclude=("y",))
    b = run_interactions(tr, te, None, method="product", max_features=3, exclude=("y",))
    assert a.fitted_state["pairs"] == b.fitted_state["pairs"]
    # Two numeric features (a, b) yield exactly one pair; the cap is not binding here.
    assert len(a.fitted_state["pairs"]) == 1
    # With more features the cap binds and truncation is recorded.
    wide = tr.assign(c=tr["a"] * 2, d=tr["b"] * 3)
    capped = run_interactions(wide, te, None, method="product", max_features=3,
                              exclude=("y",))
    assert len(capped.fitted_state["pairs"]) == 3
    assert any("capped" in n for n in capped.notes)


def test_temporal_features_use_the_training_origin():
    """Each cohort's own minimum would make the same date give different values."""
    tr, te = _frames()
    result = run_temporal_features(tr, te, None, timestamp_column="t", exclude=("y",))
    assert result.fitted_state["origin"].startswith("2024-01-01")
    assert result.transformed_test["t__days_since"].min() > 400


def test_aggregation_excludes_the_current_row():
    """closed='left': the row being predicted from must not be inside its own feature."""
    tr, te = _frames()
    result = run_aggregation_features(tr, te, None, entity_id_column="e",
                                      timestamp_column="t", windows=(7,), exclude=("y",))
    column = "a__mean_7d"
    assert column in result.transformed_train.columns
    # The first observation for an entity has no prior window. Uses head(1) rather than
    # groupby().first(), which skips NaN by default and would silently return the
    # SECOND observation — hiding exactly the property under test.
    first_index = tr.groupby("e").head(1).index
    assert result.transformed_train.loc[first_index, column].isna().all()


def test_pca_requires_two_features():
    tr, te = _frames()
    with pytest.raises(ValueError, match="at least two"):
        run_pca_transform(tr[["a", "y"]], te[["a", "y"]], None, exclude=("y",))


def test_pca_applies_train_components_to_test():
    tr, te = _frames()
    result = run_pca_transform(tr, te, None, n_components=2, exclude=("y", "t", "e", "cat"))
    assert "pc_1" in result.transformed_test.columns
    assert result.fitted_state["n_components_actual"] == 2


def test_selection_permutation_requires_a_model():
    tr, te = _frames()
    with pytest.raises(ValueError, match="requires a fitted model"):
        run_selection(tr, te, None, method="permutation", target_column="y")


def test_selection_states_the_cross_validation_limitation():
    tr, te = _frames()
    result = run_selection(tr, te, None, method="mutual_info", target_column="y", top_k=2)
    assert any("external cross-validation" in n for n in result.notes)
    assert any("column decision" in n for n in result.notes)


# ================================================================== audit ==
def test_perturbation_straddles_the_training_range():
    """Perturbing in one direction leaves rank-based statistics unchanged."""
    frame = pd.DataFrame({"x": np.linspace(0.0, 1.0, 50)})
    perturbed = _perturb_frame(frame)
    assert perturbed["x"].min() < frame["x"].min()
    assert perturbed["x"].max() > frame["x"].max()


@pytest.mark.parametrize("index", list(range(6)))
def test_every_leaky_fixture_is_caught(index):
    """The mandatory six. A detector never shown to fire should not be trusted."""
    fixture, _defect, expected = LEAKY_FIXTURES[index]
    tr, te = _frames()
    kwargs = {
        "leaky_scaler": {"method": "standard"},
        "leaky_imputer": {"strategy": "median"},
        "leaky_target_encoder": {"target_column": "y"},
        "leaky_aggregation": {"entity_id_column": "e", "timestamp_column": "t",
                              "windows": (7,)},
        "leaky_pca": {"n_components": 2, "exclude": ("y", "t", "e", "cat")},
        "leaky_selector": {"target_column": "y", "top_k": 1},
    }[fixture.__name__]
    call = dict(kwargs)
    target = call.pop("target_column", None)
    stamp = call.pop("timestamp_column", None)
    audit = audit_executor(fixture, tr, te, step=fixture.__name__,
                           target_column=target, timestamp_column=stamp, **call)
    fired = {f.check for f in audit.violations}
    assert all(check in fired for check in expected), (fixture.__name__, fired)


@pytest.mark.parametrize(
    "label,executor,kwargs,target,stamp",
    [
        ("scaling", run_scaling, {"method": "standard", "exclude": ("y",)}, None, None),
        ("imputation", run_imputation, {"strategy": "median", "exclude": ("y",)}, None, None),
        ("target_encoding", run_categorical_encoding, {"method": "target"}, "y", None),
        ("woe", run_woe_iv, {}, "y", None),
        ("aggregation", run_aggregation_features,
         {"entity_id_column": "e", "exclude": ("y",)}, None, "t"),
        ("pca", run_pca_transform,
         {"n_components": 2, "exclude": ("y", "t", "e", "cat")}, None, None),
        ("selection", run_selection, {"method": "mutual_info", "top_k": 2}, "y", None),
    ],
)
def test_correct_executors_pass_the_audit(label, executor, kwargs, target, stamp):
    tr, te = _frames()
    audit = audit_executor(executor, tr, te, step=label, target_column=target,
                           timestamp_column=stamp, **kwargs)
    assert audit.passed, (label, audit.summary())


def test_audit_forwards_target_column_to_the_executor():
    """Swallowing it would produce a spurious execution failure, not a real finding."""
    tr, te = _frames()
    audit = audit_executor(run_categorical_encoding, tr, te, step="enc",
                           target_column="y", method="target")
    assert not any(f.check == "check_0_execution" for f in audit.findings)


def test_out_of_fold_prior_is_also_out_of_fold():
    """A full-train prior leaks a row's own target through the smoothing term."""
    tr, te = _frames()
    audit = audit_executor(run_categorical_encoding, tr, te, step="enc",
                           target_column="y", method="target")
    check3 = [f for f in audit.findings if f.check == "check_3_out_of_fold_target_encoding"]
    assert check3 and check3[0].passed


def test_violation_names_the_step_and_check():
    """'Something leaked' is not actionable."""
    from start.tests.feature_engineering.leaky_fixtures import leaky_scaler
    tr, te = _frames()
    audit = audit_executor(leaky_scaler, tr, te, step="leaky_scaler", method="standard")
    assert not audit.passed
    assert "leaky_scaler" in audit.summary()
    assert audit.violations[0].step == "leaky_scaler"
    assert audit.violations[0].check


# ====================================================== registered surfaces ==
def test_plan_freezes_a_hash():
    result = plan(_ctx(), steps=("imputation", "scaling"))
    assert result.metrics["plan_hash"]
    assert result.status == Status.RECORDED


def test_plan_fails_when_mutated_during_execution():
    ctx = _ctx(extra={"feature_engineering_plan_hash": "0" * 64})
    result = plan(ctx, steps=("imputation",))
    assert result.status == Status.FAIL
    assert result.metrics["plan_mutated_during_execution"] is True


@pytest.mark.parametrize(
    "fn,kw",
    [
        (imputation, {}), (scaling, {}), (numeric_transform, {}),
        (winsorization, {}), (categorical_encoding, {}),
        (rare_category_grouping, {}), (interactions, {}), (selection, {}),
    ],
)
def test_registered_surfaces_emit_scalar_evidence(fn, kw):
    result = fn(_ctx(), **kw)
    assert result.status in {Status.RECORDED, Status.SKIPPED, Status.WARN}
    for value in result.metrics.values():
        assert not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))


def test_woe_warns_on_suspicious_iv():
    """An IV that high usually means the feature encodes the outcome."""
    tr, te = _frames()
    tr["leaky"] = tr["y"].astype(float) + np.random.default_rng(3).normal(0, 0.01, len(tr))
    te["leaky"] = te["y"].astype(float) + np.random.default_rng(4).normal(0, 0.01, len(te))
    result = woe_iv(_ctx(tr, te))
    assert result.status == Status.WARN
    assert result.metrics["max_iv_band"] == "suspicious"


def test_temporal_surface_skips_without_timestamp():
    result = temporal_features(_ctx())
    assert result.status == Status.SKIPPED


def test_aggregation_surface_runs_with_entity_and_timestamp():
    result = aggregation_features(_ctx(ts="t", entity="e"), windows=(7,))
    assert result.status == Status.RECORDED
    assert result.metrics["n_features_after"] > result.metrics["n_features_before"]


def test_pca_surface_reports_explained_variance():
    result = pca_transform(_ctx(), n_components=2)
    assert result.metrics["n_components_actual"] == 2
    assert 0.0 < result.metrics["explained_variance_total"] <= 1.0


def test_monotonic_binning_surface_reports_directions():
    result = monotonic_binning(_ctx())
    assert result.status == Status.RECORDED
    assert "directions" in result.metrics


def test_fitting_scope_audit_passes_on_correct_pipeline():
    result = fitting_scope_audit(_ctx())
    assert result.status == Status.PASS
    assert result.metrics["n_violations"] == 0
    assert result.metrics["n_steps_audited"] > 0


def test_fitting_scope_audit_skips_without_evaluation_cohort():
    """Check 2 perturbs evaluation values; without a test cohort it cannot run."""
    tr, _ = _frames()
    result = fitting_scope_audit(TestContext(train=tr, target_column="y"))
    assert result.status == Status.SKIPPED
    assert "perturbs evaluation values" in result.interpretation


def test_fitting_scope_audit_explains_why_shuffling_is_insufficient():
    result = fitting_scope_audit(_ctx())
    assert any("shuffling" in x for x in result.limitations)


def test_audit_evidence_never_contains_frames():
    result = fitting_scope_audit(_ctx())
    for value in result.metrics.values():
        assert not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))
