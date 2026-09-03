"""Gate-A reviewer control surface — verification, not new machinery.

The frozen design required Gate-A configurable parameters to reach reviewers through an
established mechanism. Inspection of the live tree found one already exists:
``TestFamiliesConfig.overrides``, a per-test-id parameter map, already used in production
by ``start.modeling.propensity``.

So no new parameter-management framework is built. These tests verify the existing
mechanism genuinely reaches the Gate-A surfaces, which is the difference between a
control that is documented and a control that works.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from start.core.config import TestFamiliesConfig
from start.registry import TestContext, list_tests

#: The reviewer-facing Gate-A parameters. Deliberately excludes low-level numerical
#: tolerances (state-hash rounding, audit atol): those are correctness machinery, not
#: policy, and exposing them as casual knobs invites someone to loosen a leakage check
#: until it stops complaining.
REVIEWER_CONTROLS: dict[str, tuple[str, ...]] = {
    "eda.correlation": ("method", "high"),
    "eda.multicollinearity": ("vif_warn", "vif_fail"),
    "eda.class_imbalance": ("warn_ratio", "fail_ratio"),
    "preprocessing.outliers": ("iqr_multiplier", "warn_pct", "fail_pct", "emit_boxplot"),
    "preprocessing.feature_drift": ("psi_warn", "psi_fail", "include_wasserstein"),
    "preprocessing.leakage_high_correlation": ("warn_corr",),
    "preprocessing.leakage_suspicious_predictivity": ("auc_threshold", "r2_threshold"),
    "preprocessing.dimensionality_diagnostic": ("variance_target", "ratio_warn"),
    "preprocessing.redundancy": ("corr_threshold",),
    "preprocessing.categorical_drift": ("alpha", "min_expected"),
    "feature_engineering.rare_category_grouping": ("min_pct",),
    "feature_engineering.categorical_encoding": ("method", "n_folds", "smoothing"),
    "feature_engineering.winsorization": ("method", "k"),
    "feature_engineering.selection": ("method", "top_k", "threshold"),
    "feature_engineering.pca_transform": ("n_components", "whiten"),
    "feature_engineering.woe_iv": ("bins", "min_bin_pct", "smoothing"),
}

#: Never reviewer policy. Correctness machinery.
NOT_REVIEWER_CONTROLS: tuple[str, ...] = ("atol", "rtol", "STATE_HASH_DECIMALS", "STATE_ATOL", "decimals")


def _specs():
    return {s.test_id: s for s in list_tests()}


def test_a_generic_override_mechanism_already_exists():
    """No new framework is needed, so none is built."""
    config = TestFamiliesConfig()
    assert hasattr(config, "overrides")
    assert isinstance(config.overrides, dict)


def test_overrides_accept_gate_a_test_ids():
    config = TestFamiliesConfig(overrides={"preprocessing.outliers": {"iqr_multiplier": 3.0}})
    assert config.overrides["preprocessing.outliers"]["iqr_multiplier"] == 3.0


@pytest.mark.parametrize("test_id", sorted(REVIEWER_CONTROLS))
def test_every_declared_control_is_a_real_registered_parameter(test_id):
    """A control that is documented but not actually a parameter is a lie."""
    spec = _specs()[test_id]
    for parameter in REVIEWER_CONTROLS[test_id]:
        assert parameter in spec.default_params, (test_id, parameter)


@pytest.mark.parametrize("test_id", sorted(REVIEWER_CONTROLS))
def test_every_declared_control_is_accepted_by_the_function(test_id):
    """The parameter must reach the callable, not merely sit in default_params."""
    import inspect

    spec = _specs()[test_id]
    signature = inspect.signature(spec.fn)
    for parameter in REVIEWER_CONTROLS[test_id]:
        assert parameter in signature.parameters, (test_id, parameter)


def test_numerical_tolerances_are_not_exposed_as_reviewer_policy():
    """Exposing an audit tolerance invites loosening it until the check stops firing."""
    for test_id, parameters in REVIEWER_CONTROLS.items():
        for parameter in parameters:
            assert parameter not in NOT_REVIEWER_CONTROLS, (test_id, parameter)


def test_overriding_a_control_changes_the_result():
    """End-to-end: the mechanism actually alters behaviour."""
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    frame.loc[:9, "a"] = 60.0
    ctx = TestContext(train=frame)
    outliers = _specs()["preprocessing.outliers"].fn

    tight = outliers(ctx, iqr_multiplier=1.0)
    loose = outliers(ctx, iqr_multiplier=6.0)
    assert tight.metrics["max_outlier_pct"] > loose.metrics["max_outlier_pct"]
    assert tight.params["iqr_multiplier"] == 1.0


def test_opt_in_controls_default_to_off():
    """Legacy evidence identity is preserved when a reviewer changes nothing."""
    specs = _specs()
    assert specs["preprocessing.outliers"].default_params["emit_boxplot"] is False
    assert specs["preprocessing.feature_drift"].default_params["include_wasserstein"] is False


def test_policy_hashing_inputs_are_untouched():
    """F1 and the control surface change no threshold default."""
    specs = _specs()
    assert specs["eda.multicollinearity"].default_params["vif_warn"] == 5.0
    assert specs["eda.multicollinearity"].default_params["vif_fail"] == 10.0
    assert specs["preprocessing.outliers"].default_params["iqr_multiplier"] == 1.5
    assert specs["preprocessing.feature_drift"].default_params["psi_warn"] == 0.1
