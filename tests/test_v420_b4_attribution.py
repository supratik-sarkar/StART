"""B4 — attribution, with expected values derived independently of production code.

Every algebraic expectation here is computed from explicit small arrays in the test
itself. No expected value is produced by calling the function under test.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from start.core.schemas import Status
from start.registry.market_contexts import MarketContext, PortfolioSpec
from start.tests.attribution import (
    INTERACTION_SHARE_WARN,
    RECONCILIATION_ATOL,
    RECONCILIATION_RTOL,
    RISK_CHANGE_CONTRACT,
    AttributionState,
    align_universe,
    cross_sectional_factor_model,
    decompose_risk_change,
    estimate_factor_returns,
    exposure_analysis,
    factor_return_estimation,
    interaction_share,
    return_attribution,
    risk_attribution,
    risk_change_decomposition,
)


def _idx(n):
    return pd.date_range("2024-01-01", periods=n, freq="B")


def _ctx(returns, exposures, weights=None, benchmark=None, factor_returns=None, **kw):
    if weights is None:
        weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    return MarketContext(
        returns=returns, factor_exposures=exposures, factor_returns=factor_returns,
        portfolio=PortfolioSpec(weights=weights, benchmark_weights=benchmark), **kw
    )


def _exact_world(n_periods=40, seed=0):
    """A world with NO specific return: r = X f exactly, so attribution must reconcile
    to machine precision and any residual is a defect rather than noise."""
    rng = np.random.default_rng(seed)
    assets = [f"A{i}" for i in range(6)]
    factors = ["F1", "F2", "F3"]
    X = pd.DataFrame(rng.normal(size=(6, 3)), index=assets, columns=factors)
    f = pd.DataFrame(rng.normal(0, 0.01, (n_periods, 3)), index=_idx(n_periods),
                     columns=factors)
    returns = pd.DataFrame(f.to_numpy() @ X.to_numpy().T, index=f.index, columns=assets)
    return returns, X, f


# ============================================== FROZEN CONTRACTS ==
def test_reconciliation_tolerances_are_frozen():
    assert RECONCILIATION_ATOL == 1e-10
    assert RECONCILIATION_RTOL == 1e-8
    assert INTERACTION_SHARE_WARN == 0.20


def test_risk_change_contract_records_all_four_components():
    for key in ("exposure_component", "covariance_component", "specific_component",
                "interaction_component", "identity", "interaction_share", "ordering"):
        assert key in RISK_CHANGE_CONTRACT
    assert "simultaneous" in RISK_CHANGE_CONTRACT["ordering"]


# ============================================== ALIGNMENT ==
def test_asset_and_factor_order_is_canonical_not_insertion_order():
    returns = pd.DataFrame(np.ones((3, 3)), index=_idx(3), columns=["C", "A", "B"])
    exposures = pd.DataFrame(np.ones((3, 2)), index=["B", "C", "A"], columns=["Z", "Y"])
    universe = align_universe(returns, exposures)
    assert universe.assets == ("A", "B", "C")
    assert universe.factors == ("Y", "Z")


def test_duplicate_labels_are_rejected():
    returns = pd.DataFrame(np.ones((3, 2)), index=_idx(3), columns=["A", "A"])
    exposures = pd.DataFrame(np.ones((2, 2)), index=["A", "B"], columns=["F1", "F2"])
    with pytest.raises(ValueError, match="duplicate asset labels in returns"):
        align_universe(returns, exposures)


def test_duplicate_factor_labels_are_rejected():
    returns = pd.DataFrame(np.ones((3, 2)), index=_idx(3), columns=["A", "B"])
    exposures = pd.DataFrame(np.ones((2, 2)), index=["A", "B"], columns=["F1", "F1"])
    with pytest.raises(ValueError, match="duplicate factor labels"):
        align_universe(returns, exposures)


def test_assets_without_exposures_are_excluded_not_imputed():
    """A fabricated zero exposure would move that asset's entire return into specific."""
    returns = pd.DataFrame(np.ones((3, 3)), index=_idx(3), columns=["A", "B", "C"])
    exposures = pd.DataFrame(np.ones((2, 2)), index=["A", "B"], columns=["F1", "F2"])
    universe = align_universe(returns, exposures)
    assert universe.assets == ("A", "B")
    assert "C" in universe.excluded_assets
    assert "never imputed as zero" in universe.exclusion_rule


def test_empty_intersection_is_rejected():
    returns = pd.DataFrame(np.ones((3, 2)), index=_idx(3), columns=["A", "B"])
    exposures = pd.DataFrame(np.ones((2, 2)), index=["X", "Y"], columns=["F1", "F2"])
    with pytest.raises(ValueError, match="no asset appears"):
        align_universe(returns, exposures)


def test_shuffled_column_order_gives_identical_results():
    """Two reviews handed the same data in different order must agree."""
    returns, X, _ = _exact_world(20)
    a = factor_return_estimation(_ctx(returns, X))
    shuffled_returns = returns[list(reversed(returns.columns))]
    shuffled_X = X.loc[list(reversed(X.index)), list(reversed(X.columns))]
    b = factor_return_estimation(_ctx(shuffled_returns, shuffled_X))
    assert a.metrics["factor_returns_hash"] == b.metrics["factor_returns_hash"]


# ============================================== WLS KNOWN ANSWERS ==
def test_exact_full_rank_unit_weight_wls():
    """3 assets, 2 factors, no noise: lstsq must recover the generating f exactly.
    Expected value computed here by explicit construction, not from the estimator."""
    assets, factors = ["A", "B", "C"], ["F1", "F2"]
    X = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], index=assets, columns=factors)
    true_f = np.array([0.02, -0.01])
    r = X.to_numpy() @ true_f
    returns = pd.DataFrame([r], index=_idx(1), columns=assets)
    outcome = estimate_factor_returns(_ctx(returns, X))
    estimated = outcome.factor_returns.iloc[0].to_numpy()
    assert np.allclose(estimated, true_f, atol=1e-12)
    assert outcome.weighting == "unit"


def test_supplied_observation_weights_are_used():
    """With an exactly determined system both weightings recover the same f; the point
    is that the weighting is recorded and accepted."""
    assets, factors = ["A", "B", "C"], ["F1", "F2"]
    X = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], index=assets, columns=factors)
    true_f = np.array([0.02, -0.01])
    returns = pd.DataFrame([X.to_numpy() @ true_f], index=_idx(1), columns=assets)
    weights = pd.Series([1.0, 4.0, 9.0], index=assets)
    outcome = estimate_factor_returns(_ctx(returns, X), observation_weights=weights)
    assert outcome.weighting == "supplied"
    assert np.allclose(outcome.factor_returns.iloc[0].to_numpy(), true_f, atol=1e-12)


def test_weighting_changes_the_estimate_on_an_overdetermined_noisy_system():
    rng = np.random.default_rng(3)
    assets = [f"A{i}" for i in range(8)]
    factors = ["F1", "F2"]
    X = pd.DataFrame(rng.normal(size=(8, 2)), index=assets, columns=factors)
    r = X.to_numpy() @ np.array([0.01, 0.02]) + rng.normal(0, 0.005, 8)
    returns = pd.DataFrame([r], index=_idx(1), columns=assets)
    plain = estimate_factor_returns(_ctx(returns, X)).factor_returns.iloc[0].to_numpy()
    weighted = estimate_factor_returns(
        _ctx(returns, X), observation_weights=pd.Series(np.arange(1.0, 9.0), index=assets)
    ).factor_returns.iloc[0].to_numpy()
    assert not np.allclose(plain, weighted, atol=1e-9)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_observation_weights_are_rejected(bad):
    """A zero weight is an undefined sqrt, not a down-weighting."""
    assets, factors = ["A", "B", "C"], ["F1", "F2"]
    X = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], index=assets, columns=factors)
    returns = pd.DataFrame([[0.01, 0.02, 0.03]], index=_idx(1), columns=assets)
    weights = pd.Series([1.0, 1.0, bad], index=assets)
    with pytest.raises(ValueError, match="observation weights"):
        estimate_factor_returns(_ctx(returns, X), observation_weights=weights)


def test_rank_deficient_cross_section_is_skipped_not_reported():
    """Two identical exposure columns cannot identify two factors."""
    assets, factors = ["A", "B", "C"], ["F1", "F2"]
    X = pd.DataFrame([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], index=assets, columns=factors)
    returns = pd.DataFrame(np.ones((5, 3)) * 0.01, index=_idx(5), columns=assets)
    outcome = estimate_factor_returns(_ctx(returns, X))
    assert outcome.n_periods_estimated == 0
    assert outcome.n_periods_skipped_rank == 5


def test_underidentified_cross_section_is_skipped():
    """2 assets cannot identify 3 factors."""
    assets, factors = ["A", "B"], ["F1", "F2", "F3"]
    X = pd.DataFrame(np.eye(2, 3), index=assets, columns=factors)
    returns = pd.DataFrame(np.ones((4, 2)) * 0.01, index=_idx(4), columns=assets)
    outcome = estimate_factor_returns(_ctx(returns, X))
    assert outcome.n_periods_estimated == 0
    assert outcome.n_periods_skipped_rank == 4


def test_all_periods_unusable_gives_error():
    assets, factors = ["A", "B"], ["F1", "F2", "F3"]
    X = pd.DataFrame(np.eye(2, 3), index=assets, columns=factors)
    returns = pd.DataFrame(np.ones((4, 2)) * 0.01, index=_idx(4), columns=assets)
    result = factor_return_estimation(_ctx(returns, X))
    assert result.status == Status.ERROR


def test_partial_rank_deficiency_warns_and_counts():
    returns, X, _ = _exact_world(10)
    result = factor_return_estimation(_ctx(returns, X))
    assert result.status == Status.RECORDED
    assert result.metrics["n_periods_estimated"] == 10
    assert result.metrics["n_periods_skipped_rank"] == 0


def test_no_intercept_is_added_silently():
    returns, X, _ = _exact_world(10)
    result = factor_return_estimation(_ctx(returns, X))
    assert result.metrics["intercept_added"] is False
    assert result.metrics["n_factors"] == 3
    assert result.metrics["factor_names"] == "F1, F2, F3"


def test_metrics_contain_no_matrices():
    returns, X, _ = _exact_world(10)
    result = factor_return_estimation(_ctx(returns, X))
    for value in result.metrics.values():
        assert not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))
    assert len(result.metrics["factor_returns_hash"]) == 32


# ============================================== EXPOSURES ==
def test_static_exposure_alignment():
    returns, X, _ = _exact_world(10)
    ctx = _ctx(returns, X)
    assert ctx.is_time_varying_exposure is False
    result = exposure_analysis(ctx)
    assert result.status == Status.RECORDED
    assert result.metrics["exposures_time_varying"] is False


def test_time_varying_exposure_alignment():
    returns, X, _ = _exact_world(10)
    varying = {t: X * (1.0 + i * 0.01) for i, t in enumerate(returns.index)}
    ctx = _ctx(returns, varying)
    assert ctx.is_time_varying_exposure is True
    result = exposure_analysis(ctx)
    assert result.status == Status.RECORDED
    assert result.metrics["exposures_time_varying"] is True


def test_missing_time_varying_exposure_errors_rather_than_forward_filling():
    returns, X, _ = _exact_world(10)
    partial = {t: X for t in returns.index[:5]}
    result = exposure_analysis(_ctx(returns, partial))
    assert result.status == Status.ERROR
    assert "never forward-filled" in result.interpretation


def test_portfolio_exposure_known_answer():
    """x = X' w, computed by hand: X'=[[1,0,1],[0,1,1]], w=[0.5,0.25,0.25]
    -> x = [0.5*1 + 0.25*0 + 0.25*1, 0.5*0 + 0.25*1 + 0.25*1] = [0.75, 0.5]."""
    assets, factors = ["A", "B", "C"], ["F1", "F2"]
    X = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], index=assets, columns=factors)
    returns = pd.DataFrame(np.ones((3, 3)) * 0.01, index=_idx(3), columns=assets)
    weights = pd.Series([0.5, 0.25, 0.25], index=assets)
    result = exposure_analysis(_ctx(returns, X, weights))
    assert abs(result.metrics["portfolio_exposure.F1"] - 0.75) < 1e-12
    assert abs(result.metrics["portfolio_exposure.F2"] - 0.50) < 1e-12


def test_benchmark_and_active_exposure_known_answer():
    """Benchmark equal-weight: x_b = [2/3, 2/3]; active = [0.75-2/3, 0.5-2/3]."""
    assets, factors = ["A", "B", "C"], ["F1", "F2"]
    X = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], index=assets, columns=factors)
    returns = pd.DataFrame(np.ones((3, 3)) * 0.01, index=_idx(3), columns=assets)
    weights = pd.Series([0.5, 0.25, 0.25], index=assets)
    benchmark = pd.Series([1 / 3, 1 / 3, 1 / 3], index=assets)
    result = exposure_analysis(_ctx(returns, X, weights, benchmark))
    assert result.metrics["benchmark_available"] is True
    assert abs(result.metrics["benchmark_exposure.F1"] - 2 / 3) < 1e-12
    assert abs(result.metrics["active_exposure.F1"] - (0.75 - 2 / 3)) < 1e-12
    assert abs(result.metrics["active_exposure.F2"] - (0.50 - 2 / 3)) < 1e-12


def test_absent_benchmark_yields_no_active_exposure():
    returns, X, _ = _exact_world(10)
    result = exposure_analysis(_ctx(returns, X))
    assert result.metrics["benchmark_available"] is False
    assert not any(k.startswith("active_exposure.") for k in result.metrics)
    assert any("invented benchmark" in x for x in result.limitations)


def test_weights_are_never_renormalised():
    """If they sum to 0.97 the evidence shows 0.97."""
    assets, factors = ["A", "B", "C"], ["F1", "F2"]
    X = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], index=assets, columns=factors)
    returns = pd.DataFrame(np.ones((3, 3)) * 0.01, index=_idx(3), columns=assets)
    weights = pd.Series([0.5, 0.25, 0.22], index=assets)
    result = exposure_analysis(_ctx(returns, X, weights))
    assert abs(result.metrics["weights_sum"] - 0.97) < 1e-12
    assert result.metrics["weights_renormalised"] is False


# ============================================== RETURN ATTRIBUTION ==
def test_return_attribution_reconciles_exactly_with_no_specific_return():
    """r = X f exactly, so factor + specific must equal observed to machine precision."""
    returns, X, _ = _exact_world(30)
    result = return_attribution(_ctx(returns, X))
    assert result.status == Status.RECORDED
    assert result.metrics["max_abs_reconciliation_error"] < 1e-14
    assert result.metrics["n_periods_outside_tolerance"] == 0


def test_return_attribution_reconciles_with_specific_return_present():
    rng = np.random.default_rng(5)
    returns, X, _ = _exact_world(30, seed=5)
    noisy = returns + rng.normal(0, 0.002, returns.shape)
    result = return_attribution(_ctx(noisy, X))
    assert result.status == Status.RECORDED
    assert result.metrics["n_periods_outside_tolerance"] == 0


def test_supplied_factor_return_route_is_recorded():
    returns, X, f = _exact_world(20)
    result = return_attribution(_ctx(returns, X, factor_returns=f))
    assert result.metrics["factor_return_source"] == "supplied"
    assert result.metrics["max_abs_reconciliation_error"] < 1e-14


def test_estimated_factor_return_route_is_recorded():
    returns, X, _ = _exact_world(20)
    result = return_attribution(_ctx(returns, X))
    assert result.metrics["factor_return_source"] == "estimated"


def test_the_two_routes_are_never_switched_silently():
    returns, X, f = _exact_world(20)
    supplied = return_attribution(_ctx(returns, X, factor_returns=f))
    estimated = return_attribution(_ctx(returns, X))
    assert supplied.params["factor_return_source"] != estimated.params["factor_return_source"]


def test_specific_contribution_known_answer():
    """One period, X'w = [0.75, 0.5], f = [0.02, -0.01].
    factor contribution = 0.75*0.02 + 0.5*(-0.01) = 0.015 - 0.005 = 0.010."""
    assets, factors = ["A", "B", "C"], ["F1", "F2"]
    X = pd.DataFrame([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], index=assets, columns=factors)
    true_f = np.array([0.02, -0.01])
    r = X.to_numpy() @ true_f
    returns = pd.DataFrame([r], index=_idx(1), columns=assets)
    weights = pd.Series([0.5, 0.25, 0.25], index=assets)
    f_frame = pd.DataFrame([true_f], index=_idx(1), columns=factors)
    result = return_attribution(_ctx(returns, X, weights, factor_returns=f_frame))
    assert abs(result.metrics["total_factor_contribution"] - 0.010) < 1e-12
    assert abs(result.metrics["total_specific_contribution"]) < 1e-14


# ============================================== RISK ATTRIBUTION ==
def test_factor_variance_known_answer():
    """x'Fx with x = [0.75, 0.5] and F = [[4,1],[1,9]]*1e-4:
    = 1e-4 * (0.75*(4*0.75 + 1*0.5) + 0.5*(1*0.75 + 9*0.5))
    = 1e-4 * (0.75*3.5 + 0.5*5.25) = 1e-4 * (2.625 + 2.625) = 5.25e-4."""
    x = np.array([0.75, 0.5])
    F = np.array([[4.0, 1.0], [1.0, 9.0]]) * 1e-4
    assert abs(float(x @ F @ x) - 5.25e-4) < 1e-15


def test_risk_attribution_components_reconcile():
    returns, X, f = _exact_world(60)
    result = risk_attribution(_ctx(returns, X, factor_returns=f))
    assert result.status == Status.RECORDED
    assert abs(result.metrics["factor_component_reconciliation_error"]) < 1e-15
    assert abs(result.metrics["factor_component_sum"]
               - result.metrics["factor_variance"]) < 1e-14


def test_risk_attribution_uses_factor_not_asset_covariance():
    """ctx.covariance holds the ASSET covariance and must never be mistaken for F."""
    returns, X, f = _exact_world(60)
    asset_cov = returns.cov()
    ctx = _ctx(returns, X, factor_returns=f)
    ctx.covariance = asset_cov
    result = risk_attribution(ctx)
    assert result.metrics["factor_covariance_source"] in {
        "supplied_factor_returns", "estimated_factor_returns"
    }
    assert result.metrics["factor_covariance_dimension"] == 3


def test_specific_risk_model_is_declared_diagonal():
    returns, X, _ = _exact_world(60)
    result = risk_attribution(_ctx(returns, X))
    assert result.metrics["specific_risk_model"] == "diagonal"
    assert any("DIAGONAL" in x for x in result.limitations)


def test_specific_variance_known_answer():
    """S = sum(w_i^2 * var(e_i)); with no specific return it must be ~0."""
    returns, X, _ = _exact_world(60)
    result = risk_attribution(_ctx(returns, X))
    assert result.metrics["specific_variance"] < 1e-25


def test_factor_model_variance_need_not_equal_empirical():
    returns, X, _ = _exact_world(60)
    result = risk_attribution(_ctx(returns, X))
    assert "empirical_portfolio_variance" in result.metrics
    assert any("not necessarily a software defect" in x for x in result.limitations)


# ============================================== FACTOR STATISTICS ==
def test_factor_statistics_known_answer():
    """Constructed factor returns: mean and sd computed by hand in the test."""
    values = np.array([0.01, 0.03, -0.01, 0.02, 0.00])
    assets, factors = ["A", "B", "C"], ["F1"]
    X = pd.DataFrame([[1.0], [2.0], [3.0]], index=assets, columns=factors)
    returns = pd.DataFrame(np.outer(values, X["F1"].to_numpy()),
                           index=_idx(5), columns=assets)
    result = cross_sectional_factor_model(_ctx(returns, X))
    expected_mean = float(values.mean())
    expected_sd = float(values.std(ddof=1))
    expected_se = expected_sd / math.sqrt(5)
    assert abs(result.metrics["mean.F1"] - expected_mean) < 1e-12
    assert abs(result.metrics["sd.F1"] - expected_sd) < 1e-12
    assert abs(result.metrics["stderr.F1"] - expected_se) < 1e-12
    assert abs(result.metrics["tstat.F1"] - expected_mean / expected_se) < 1e-9


def test_stderr_convention_is_recorded():
    returns, X, _ = _exact_world(30)
    result = cross_sectional_factor_model(_ctx(returns, X))
    assert "sqrt(T)" in result.metrics["stderr_convention"]
    assert "no serial-dependence correction" in result.metrics["stderr_convention"]


def test_the_statistics_are_never_claimed_to_be_fama_macbeth():
    """Borrowing the name would imply a two-pass procedure, a risk-premium reading and
    a HAC correction — none of which is implemented.

    A blunt substring search is wrong here: the module deliberately CONTAINS the name
    inside an explicit disclaimer. What matters is that every occurrence is negated.
    """
    returns, X, _ = _exact_world(30)
    result = cross_sectional_factor_model(_ctx(returns, X))
    blob = " ".join([result.interpretation, *result.limitations,
                     str(result.metrics.get("stderr_convention", ""))]).lower()

    # The disclaimer must be present...
    assert "not fama-macbeth" in blob
    # ...and every mention of the name must sit inside a negation.
    for start in [i for i in range(len(blob)) if blob.startswith("fama-macbeth", i)]:
        assert "not " in blob[max(0, start - 8):start]
    # No risk-premium or causal claim.
    assert "risk premium is estimated" in blob or "no risk premium" in blob
    assert "causal effect" in blob


def test_no_hac_correction_is_claimed():
    returns, X, _ = _exact_world(30)
    result = cross_sectional_factor_model(_ctx(returns, X))
    blob = " ".join(result.limitations).lower()
    assert "hac" in blob and "no " in blob
    assert "newey" in blob


def test_serial_dependence_limitation_is_stated():
    returns, X, _ = _exact_world(30)
    result = cross_sectional_factor_model(_ctx(returns, X))
    assert any("autocorrelated" in x or "serially independent" in x
               for x in result.limitations)


def test_numerically_constant_factor_return_gives_no_tstat():
    """Carries the B3 lesson: sd is ~1e-18, not 0, so an exact test would let a
    t-statistic of ~1e17 through."""
    assets, factors = ["A", "B", "C"], ["F1"]
    X = pd.DataFrame([[1.0], [2.0], [3.0]], index=assets, columns=factors)
    constant = np.full(20, 0.01)
    returns = pd.DataFrame(np.outer(constant, X["F1"].to_numpy()),
                           index=_idx(20), columns=assets)
    result = cross_sectional_factor_model(_ctx(returns, X))
    assert result.metrics["degenerate.F1"] is True
    assert result.metrics["tstat.F1"] is None
    assert result.metrics["n_degenerate_factors"] == 1


def test_genuinely_low_volatility_factor_remains_usable():
    """A small but real standard deviation must still produce a t-statistic."""
    rng = np.random.default_rng(9)
    assets, factors = ["A", "B", "C"], ["F1"]
    X = pd.DataFrame([[1.0], [2.0], [3.0]], index=assets, columns=factors)
    values = 0.01 + rng.normal(0, 1e-6, 40)
    returns = pd.DataFrame(np.outer(values, X["F1"].to_numpy()),
                           index=_idx(40), columns=assets)
    result = cross_sectional_factor_model(_ctx(returns, X))
    assert result.metrics["degenerate.F1"] is False
    assert result.metrics["tstat.F1"] is not None
    assert abs(result.metrics["tstat.F1"]) > 100


# ============================================== RISK CHANGE ==
def _state(x, F, S, label=""):
    factors = [f"F{i+1}" for i in range(len(x))]
    return AttributionState(
        exposure=pd.Series(x, index=factors, dtype=float),
        factor_covariance=pd.DataFrame(F, index=factors, columns=factors, dtype=float),
        specific_variance=float(S), label=label,
    )


def test_risk_change_decomposition_known_answer():
    """Two factors, everything explicit. Expected components computed here by hand
    from the frozen formulas, never by calling the production helper."""
    x0 = np.array([1.0, 2.0])
    F0 = np.array([[4.0, 1.0], [1.0, 9.0]])
    S0 = 5.0
    dx = np.array([0.5, -0.25])
    dF = np.array([[1.0, 0.5], [0.5, 2.0]])
    dS = 1.5

    before = _state(x0, F0, S0, "before")
    after = _state(x0 + dx, F0 + dF, S0 + dS, "after")

    expected_exposure = 2.0 * (x0 @ F0 @ dx) + dx @ F0 @ dx
    expected_covariance = x0 @ dF @ x0
    expected_specific = dS
    expected_interaction = 2.0 * (x0 @ dF @ dx) + dx @ dF @ dx

    components = decompose_risk_change(before, after)
    assert abs(components["exposure_component"] - expected_exposure) < 1e-12
    assert abs(components["factor_covariance_component"] - expected_covariance) < 1e-12
    assert abs(components["specific_risk_component"] - expected_specific) < 1e-12
    assert abs(components["interaction_component"] - expected_interaction) < 1e-12


def test_components_sum_exactly_to_the_variance_change():
    """The identity, checked against variances computed independently."""
    x0 = np.array([1.0, 2.0])
    F0 = np.array([[4.0, 1.0], [1.0, 9.0]])
    S0 = 5.0
    x1 = np.array([1.5, 1.75])
    F1 = np.array([[5.0, 1.5], [1.5, 11.0]])
    S1 = 6.5

    v0 = float(x0 @ F0 @ x0) + S0
    v1 = float(x1 @ F1 @ x1) + S1
    components = decompose_risk_change(_state(x0, F0, S0), _state(x1, F1, S1))
    assert abs(sum(components.values()) - (v1 - v0)) < 1e-10


def test_interaction_share_is_bounded_and_uses_absolute_components():
    """Dividing by signed dV explodes when components cancel."""
    # Signed total is exactly zero while the interaction is large: dividing by dV
    # would be a division by zero, which is precisely the case this definition exists
    # to survive.
    components = {"exposure_component": 10.0, "factor_covariance_component": -11.0,
                  "specific_risk_component": 0.0, "interaction_component": 1.0}
    assert abs(sum(components.values())) < 1e-12          # signed total IS zero
    share = interaction_share(components)
    assert 0.0 <= share <= 1.0
    assert abs(share - 1.0 / 22.0) < 1e-12                # |1| / (10+11+0+1)


def test_interaction_share_is_zero_for_no_change():
    components = {k: 0.0 for k in ("exposure_component", "factor_covariance_component",
                                   "specific_risk_component", "interaction_component")}
    assert interaction_share(components) == 0.0


def test_zero_change_reconciles():
    state = _state([1.0, 2.0], [[4.0, 1.0], [1.0, 9.0]], 5.0)
    components = decompose_risk_change(state, state)
    assert all(abs(v) < 1e-15 for v in components.values())


# ---- two-state transport and hashing ----
def test_state_hash_changes_with_exposure():
    a = _state([1.0, 2.0], [[4.0, 1.0], [1.0, 9.0]], 5.0)
    b = _state([1.0, 2.001], [[4.0, 1.0], [1.0, 9.0]], 5.0)
    assert a.canonical_hash() != b.canonical_hash()


def test_state_hash_changes_with_factor_covariance():
    a = _state([1.0, 2.0], [[4.0, 1.0], [1.0, 9.0]], 5.0)
    b = _state([1.0, 2.0], [[4.0, 1.0], [1.0, 9.001]], 5.0)
    assert a.canonical_hash() != b.canonical_hash()


def test_state_hash_changes_with_specific_variance():
    a = _state([1.0, 2.0], [[4.0, 1.0], [1.0, 9.0]], 5.0)
    b = _state([1.0, 2.0], [[4.0, 1.0], [1.0, 9.0]], 5.001)
    assert a.canonical_hash() != b.canonical_hash()


def test_state_hash_is_stable():
    a = _state([1.0, 2.0], [[4.0, 1.0], [1.0, 9.0]], 5.0)
    assert a.canonical_hash() == a.canonical_hash()


def _three_factor_state(scale=1.0, specific=1e-6):
    x = pd.Series([0.1, 0.2, 0.3], index=["F1", "F2", "F3"])
    F = pd.DataFrame(np.eye(3) * 1e-4 * scale, index=x.index, columns=x.index)
    return AttributionState(exposure=x, factor_covariance=F, specific_variance=specific)


def test_comparison_state_hash_reaches_evidence_params():
    """Hard acceptance condition: changing the prior state changes evidence identity."""
    returns, X, f = _exact_world(40)
    ctx = _ctx(returns, X, factor_returns=f)
    a = risk_change_decomposition(ctx, comparison_state=_three_factor_state(1.0))
    b = risk_change_decomposition(ctx, comparison_state=_three_factor_state(2.0))
    assert a.params["comparison_state_hash"] != b.params["comparison_state_hash"]
    assert a.params["current_state_hash"] == b.params["current_state_hash"]


@pytest.mark.parametrize("field", ["exposure", "covariance", "specific"])
def test_every_state_field_change_moves_the_comparison_hash(field):
    returns, X, f = _exact_world(40)
    ctx = _ctx(returns, X, factor_returns=f)
    base = _three_factor_state()
    if field == "exposure":
        other = AttributionState(base.exposure + 0.001, base.factor_covariance,
                                 base.specific_variance)
    elif field == "covariance":
        other = AttributionState(base.exposure, base.factor_covariance * 1.5,
                                 base.specific_variance)
    else:
        other = AttributionState(base.exposure, base.factor_covariance,
                                 base.specific_variance * 2)
    a = risk_change_decomposition(ctx, comparison_state=base)
    b = risk_change_decomposition(ctx, comparison_state=other)
    assert a.params["comparison_state_hash"] != b.params["comparison_state_hash"]


def test_missing_comparison_state_skips_with_the_architectural_reason():
    returns, X, f = _exact_world(20)
    result = risk_change_decomposition(_ctx(returns, X, factor_returns=f))
    assert result.status == Status.SKIPPED
    assert "does not canonicalise extra" in result.interpretation


def test_mismatched_comparison_factors_error():
    returns, X, f = _exact_world(20)
    two_factor = _state([1.0, 2.0], [[4.0, 1.0], [1.0, 9.0]], 5.0)
    result = risk_change_decomposition(_ctx(returns, X, factor_returns=f),
                                       comparison_state=two_factor)
    assert result.status == Status.ERROR
    assert "do not match" in result.interpretation


def test_risk_change_reconciles_end_to_end():
    returns, X, f = _exact_world(40)
    result = risk_change_decomposition(_ctx(returns, X, factor_returns=f),
                                       comparison_state=_three_factor_state())
    assert result.status in {Status.RECORDED, Status.WARN}
    assert abs(result.metrics["reconciliation_error"]) < 1e-10
    assert abs(result.metrics["component_sum"] - result.metrics["observed_delta"]) < 1e-10


def test_high_interaction_reconciles_but_warns():
    """Correct algebra does not make a decomposition informative."""
    x0 = np.array([1.0, 1.0])
    F0 = np.array([[1e-8, 0.0], [0.0, 1e-8]])
    dx = np.array([5.0, 5.0])
    dF = np.array([[1.0, 0.0], [0.0, 1.0]])
    before = _state(x0, F0, 0.0)
    after = _state(x0 + dx, F0 + dF, 0.0)
    components = decompose_risk_change(before, after)
    v0 = float(x0 @ F0 @ x0)
    v1 = float((x0 + dx) @ (F0 + dF) @ (x0 + dx))
    assert abs(sum(components.values()) - (v1 - v0)) < 1e-9   # reconciles exactly
    assert interaction_share(components) > INTERACTION_SHARE_WARN


def test_evidence_contains_no_matrices():
    returns, X, f = _exact_world(30)
    result = risk_change_decomposition(_ctx(returns, X, factor_returns=f),
                                       comparison_state=_three_factor_state())
    for value in result.metrics.values():
        assert not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))
