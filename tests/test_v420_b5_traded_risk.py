"""B5 — traded risk, with expected values derived independently of production code."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from start.core.schemas import Status
from start.data.synthetic_market import (
    barrier_crossing_probability,
    generate_market_world,
    generate_short_rate_path,
)
from start.registry.market_contexts import MarketContext, PortfolioSpec, ShortRateContext
from start.tests.traded_risk import (
    TRAFFIC_LIGHT_BANDS,
    TRAFFIC_LIGHT_CONFIDENCE,
    TRAFFIC_LIGHT_N,
    brownian_bridge_barrier,
    cev_elasticity,
    christoffersen_independence_lr,
    estimate_cev,
    exception_sequence,
    kupiec_lr,
    stanton_first_order,
    stanton_nonparametric,
    var_christoffersen_conditional,
    var_christoffersen_independence,
    var_exceptions,
    var_kupiec_pof,
    var_traffic_light,
)


def _idx(n, start="2024-01-01"):
    return pd.date_range(start, periods=n, freq="B")


def _short_rate_ctx(rates, ppy=252.0, min_obs=250):
    return ShortRateContext(rates=rates, units="decimal", periods_per_year=ppy, min_observations=min_obs)


def _var_ctx(pnl, var, confidence=0.99, hypothetical=None):
    index = pnl.index
    returns = pd.DataFrame({"A": pnl.to_numpy()}, index=index)
    return MarketContext(
        returns=returns,
        pnl=pnl,
        hypothetical_pnl=hypothetical,
        var_series=var,
        var_confidence=confidence,
        portfolio=PortfolioSpec(weights=pd.Series([1.0], index=["A"])),
    )


# ==================================================== CEV ==
def test_cev_recovers_a_deterministic_structural_relationship():
    """Construct increments so (dr)^2 = c * r^(2g) * dt EXACTLY with g = 0.5.
    Expected slope computed here from the construction, not from the estimator."""
    dt = 1.0 / 252.0
    gamma_true, c = 0.5, 4e-4
    # Build the series so that each increment satisfies (dr)^2 = c * r_prev^(2g) * dt
    # EXACTLY, alternating sign so the level stays in range.
    n = 400
    values = np.empty(n)
    values[0] = 0.04
    for i in range(1, n):
        step = math.sqrt(c * (values[i - 1] ** (2 * gamma_true)) * dt)
        values[i] = values[i - 1] + (step if i % 2 else -step)
    series = pd.Series(values, index=_idx(n))

    # Expected slope derived here, independently of the production helper.
    prev, dr = values[:-1], np.diff(values)
    x, y = np.log(prev), np.log(dr**2 / dt)
    expected_slope = float(np.polyfit(x, y, 1)[0])

    estimate = estimate_cev(series, dt)
    assert abs(estimate.gamma_hat - expected_slope / 2.0) < 1e-9
    # And the construction really does encode gamma = 0.5.
    assert abs(expected_slope / 2.0 - gamma_true) < 1e-6


def test_cev_gamma_is_half_the_regression_slope():
    """The defining relation, checked with an independent polyfit."""
    rates, _ = generate_short_rate_path(n_periods=600, gamma=0.5, seed=1)
    dt = 1.0 / 252.0
    values = rates.to_numpy()
    prev, dr = values[:-1], np.diff(values)
    keep = (prev > 0) & (dr != 0)
    slope = float(np.polyfit(np.log(prev[keep]), np.log(dr[keep] ** 2 / dt), 1)[0])
    assert abs(estimate_cev(rates, dt).gamma_hat - slope / 2.0) < 1e-10


def test_cev_drops_nonpositive_rates_without_shifting():
    """Shifting to make logs work would change the process being estimated."""
    values = np.linspace(0.01, 0.05, 300)
    values[10] = -0.001
    values[20] = 0.0
    series = pd.Series(values, index=_idx(300))
    estimate = estimate_cev(series, 1 / 252)
    assert estimate.n_nonpositive_dropped >= 2
    assert estimate.n_used < estimate.n_total - 1
    assert (series.to_numpy() == values).all()  # input untouched


def test_cev_counts_zero_increments_separately():
    values = np.concatenate([np.full(50, 0.03), np.linspace(0.03, 0.05, 250)])
    estimate = estimate_cev(pd.Series(values, index=_idx(300)), 1 / 252)
    assert estimate.n_zero_increment_dropped >= 49


def test_cev_insufficient_valid_observations_raises():
    series = pd.Series(np.full(300, 0.03), index=_idx(300))
    with pytest.raises(ValueError, match="usable increment"):
        estimate_cev(series, 1 / 252)


def test_cev_surface_skips_on_insufficient_observations():
    rates = pd.Series(np.linspace(0.02, 0.05, 100), index=_idx(100))
    result = cev_elasticity(_short_rate_ctx(rates))
    assert result.status == Status.SKIPPED


def test_cev_surface_reports_bootstrap_interval():
    rates, _ = generate_short_rate_path(n_periods=1200, gamma=0.5, seed=2)
    result = cev_elasticity(_short_rate_ctx(rates), bootstrap_draws=120)
    assert result.status == Status.RECORDED
    assert result.metrics["ci_low"] < result.metrics["gamma_hat"] < result.metrics["ci_high"]
    assert result.metrics["bootstrap_seed"] is not None
    assert "bootstrap" in result.metrics["ci_method"]


def test_cev_bootstrap_is_seeded_and_reproducible():
    rates, _ = generate_short_rate_path(n_periods=800, gamma=0.5, seed=3)
    ctx = _short_rate_ctx(rates)
    a = cev_elasticity(ctx, bootstrap_draws=100)
    b = cev_elasticity(ctx, bootstrap_draws=100)
    assert a.metrics["ci_low"] == b.metrics["ci_low"]
    assert a.metrics["ci_high"] == b.metrics["ci_high"]


def test_cev_stated_gamma_outside_interval_warns_never_fails():
    rates, _ = generate_short_rate_path(n_periods=1200, gamma=0.5, seed=4)
    result = cev_elasticity(_short_rate_ctx(rates), stated_gamma=-5.0, bootstrap_draws=120)
    assert result.status == Status.WARN
    assert result.status != Status.FAIL
    assert result.metrics["stated_gamma_inside_interval"] is False


def test_cev_claim_language_is_bounded():
    rates, _ = generate_short_rate_path(n_periods=800, gamma=0.5, seed=5)
    result = cev_elasticity(_short_rate_ctx(rates), bootstrap_draws=60)
    blob = " ".join(result.limitations)
    assert "APPROXIMATE FINITE-SAMPLE ESTIMATOR" in blob
    assert "not presented as a universally" in blob
    assert "would not be a valid diffusion confidence interval" in blob


# ==================================================== STANTON ==
def test_stanton_has_no_order_parameter():
    """First order only, structurally — not merely by default."""
    import inspect

    signature = inspect.signature(stanton_nonparametric)
    assert "order" not in signature.parameters
    assert "order" not in inspect.signature(stanton_first_order).parameters


def test_stanton_kernel_weights_match_an_independent_computation():
    """Nadaraya-Watson computed by hand from explicit arrays."""
    values = np.array([0.02, 0.03, 0.04, 0.05, 0.06])
    series = pd.Series(values, index=_idx(5))
    dt, h, point = 0.5, 0.01, 0.04
    prev, dr = values[:-1], np.diff(values)
    w = np.exp(-0.5 * ((prev - point) / h) ** 2)
    expected_mu = float((w @ dr) / (w.sum() * dt))
    expected_s2 = float((w @ (dr**2)) / (w.sum() * dt))
    frame = stanton_first_order(series, dt, np.array([point]), h)
    assert abs(float(frame["mu"].iloc[0]) - expected_mu) < 1e-12
    assert abs(float(frame["sigma2"].iloc[0]) - expected_s2) < 1e-12


def test_stanton_effective_sample_size_formula():
    """ESS = (sum w)^2 / sum(w^2), computed independently."""
    values = np.linspace(0.02, 0.06, 40)
    series = pd.Series(values, index=_idx(40))
    h, point = 0.005, 0.04
    prev = values[:-1]
    w = np.exp(-0.5 * ((prev - point) / h) ** 2)
    expected = float((w.sum() ** 2) / (w * w).sum())
    frame = stanton_first_order(series, 1 / 252, np.array([point]), h)
    assert abs(float(frame["ess"].iloc[0]) - expected) < 1e-10


def test_stanton_recovers_a_controlled_constant_diffusion():
    """Constant-volatility random walk: sigma2_hat should sit near the true value."""
    rng = np.random.default_rng(11)
    dt, sigma = 1 / 252, 0.01
    steps = rng.normal(0, sigma * math.sqrt(dt), 3000)
    values = 0.04 + np.cumsum(steps)
    values = np.clip(values, 1e-4, None)
    series = pd.Series(values, index=_idx(3000))
    frame = stanton_first_order(series, dt, np.array([float(np.median(values))]), 0.004)
    assert abs(float(frame["sigma2"].iloc[0]) - sigma**2) < sigma**2 * 0.5


def test_stanton_thin_support_is_flagged_not_hidden():
    rates, _ = generate_short_rate_path(n_periods=600, gamma=0.5, seed=6)
    result = stanton_nonparametric(_short_rate_ctx(rates), bandwidth=1e-6, n_grid=15)
    assert result.metrics["n_thin_support_points"] > 0
    assert result.status == Status.WARN
    assert any("thin-support" in x for x in result.limitations)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_stanton_rejects_invalid_bandwidth(bad):
    rates, _ = generate_short_rate_path(n_periods=400, seed=7)
    result = stanton_nonparametric(_short_rate_ctx(rates), bandwidth=bad)
    assert result.status == Status.ERROR


def test_stanton_silverman_bandwidth_is_recorded():
    rates, _ = generate_short_rate_path(n_periods=600, seed=8)
    result = stanton_nonparametric(_short_rate_ctx(rates))
    assert result.metrics["bandwidth_rule"] == "silverman"
    assert result.metrics["bandwidth"] > 0
    assert result.metrics["estimator_order"] == 1


def test_stanton_limitations_name_the_first_order_bound():
    rates, _ = generate_short_rate_path(n_periods=600, seed=9)
    blob = " ".join(stanton_nonparametric(_short_rate_ctx(rates)).limitations)
    assert "FIRST-ORDER ESTIMATOR ONLY" in blob
    assert "O(dt) discretisation bias" in blob
    assert "bandwidth" in blob


# ==================================================== EXCEPTIONS ==
def test_exception_sequence_hand_calculation():
    """VaR = 2.0 throughout; PnL below -2.0 is an exception. Counted by hand: 3."""
    pnl = pd.Series([1.0, -3.0, 0.5, -2.5, -1.0, -10.0, 2.0], index=_idx(7))
    var = pd.Series([2.0] * 7, index=_idx(7))
    sequence = exception_sequence(_var_ctx(pnl, var))
    assert sequence.n_exceptions == 3
    assert list(sequence.indicators.to_numpy()) == [0, 1, 0, 1, 0, 1, 0]


def test_exception_boundary_is_strictly_less_than():
    """PnL exactly equal to -VaR is NOT an exception under the stated convention."""
    pnl = pd.Series([-2.0, -2.0001], index=_idx(2))
    var = pd.Series([2.0, 2.0], index=_idx(2))
    sequence = exception_sequence(_var_ctx(pnl, var))
    assert list(sequence.indicators.to_numpy()) == [0, 1]


def test_actual_and_hypothetical_are_never_substituted():
    pnl = pd.Series([-3.0, 1.0], index=_idx(2))
    hypo = pd.Series([1.0, 1.0], index=_idx(2))
    var = pd.Series([2.0, 2.0], index=_idx(2))
    ctx = _var_ctx(pnl, var, hypothetical=hypo)
    assert exception_sequence(ctx, "actual").n_exceptions == 1
    assert exception_sequence(ctx, "hypothetical").n_exceptions == 0


def test_missing_requested_pnl_source_is_refused():
    pnl = pd.Series([-3.0, 1.0], index=_idx(2))
    var = pd.Series([2.0, 2.0], index=_idx(2))
    with pytest.raises(ValueError, match="never substituted"):
        exception_sequence(_var_ctx(pnl, var), "hypothetical")


def test_alignment_is_by_timestamp_not_position():
    """Two series of equal length with different indices must not be zipped."""
    pnl = pd.Series([-3.0, -3.0, 1.0], index=_idx(3, "2024-01-01"))
    var = pd.Series([2.0, 2.0, 2.0], index=_idx(3, "2024-06-03"))
    with pytest.raises(ValueError, match="share no timestamps"):
        exception_sequence(_var_ctx(pnl, var))


def test_partial_overlap_records_dropped_observations():
    full = _idx(10)
    pnl = pd.Series([-3.0] * 10, index=full)
    var = pd.Series([2.0] * 6, index=full[:6])
    sequence = exception_sequence(_var_ctx(pnl, var))
    assert sequence.n_aligned == 6
    assert sequence.n_dropped_alignment == 4


def test_duplicate_timestamps_are_rejected():
    index = pd.DatetimeIndex(["2024-01-01", "2024-01-01", "2024-01-02"])
    pnl = pd.Series([-3.0, -3.0, 1.0], index=index)
    var = pd.Series([2.0, 2.0, 2.0], index=index)
    with pytest.raises(ValueError, match="duplicate timestamps"):
        exception_sequence(_var_ctx(pnl, var))


def test_var_exceptions_surface_reports_counts():
    pnl = pd.Series([1.0, -3.0, 0.5, -2.5], index=_idx(4))
    var = pd.Series([2.0] * 4, index=_idx(4))
    result = var_exceptions(_var_ctx(pnl, var))
    assert result.metrics["n_exceptions"] == 2
    assert result.metrics["pnl_source"] == "actual"
    assert len(result.metrics["exception_indicator_hash"]) == 32


# ==================================================== KUPIEC ==
def test_kupiec_known_answer():
    """n=250, x=5, p=0.01. LR computed here from the explicit formula."""
    n, x, p = 250, 5, 0.01
    pi = x / n
    ll_null = x * math.log(p) + (n - x) * math.log(1 - p)
    ll_alt = x * math.log(pi) + (n - x) * math.log(1 - pi)
    expected = -2.0 * (ll_null - ll_alt)
    assert abs(kupiec_lr(n, x, p) - expected) < 1e-10


def test_kupiec_is_zero_when_the_rate_matches_exactly():
    """x/n == p means the null and alternative likelihoods coincide."""
    assert abs(kupiec_lr(100, 1, 0.01)) < 1e-12


def test_kupiec_x_equals_zero_is_finite():
    """Zero exceptions is a legitimate year, not an error. 0*log(0) must not be NaN."""
    n, p = 250, 0.01
    lr = kupiec_lr(n, 0, p)
    expected = -2.0 * (n * math.log(1 - p) - 0.0)
    assert math.isfinite(lr)
    assert abs(lr - expected) < 1e-10


def test_kupiec_x_equals_n_is_finite():
    n, p = 50, 0.01
    lr = kupiec_lr(n, n, p)
    expected = -2.0 * (n * math.log(p) - 0.0)
    assert math.isfinite(lr)
    assert abs(lr - expected) < 1e-10


def test_kupiec_surface_boundary_case_is_recorded():
    pnl = pd.Series(np.full(300, 1.0), index=_idx(300))
    var = pd.Series(np.full(300, 2.0), index=_idx(300))
    result = var_kupiec_pof(_var_ctx(pnl, var))
    assert result.metrics["boundary_case"] == "x=0"
    assert math.isfinite(result.metrics["lr_uc"])


def test_kupiec_rejects_a_badly_understated_var():
    rng = np.random.default_rng(21)
    pnl = pd.Series(rng.normal(0, 1, 500), index=_idx(500))
    var = pd.Series(np.full(500, 0.5), index=_idx(500))  # far too small
    result = var_kupiec_pof(_var_ctx(pnl, var, confidence=0.99))
    assert result.status == Status.FAIL
    assert result.metrics["rejected"] is True


def test_kupiec_non_rejection_carries_the_caveat():
    rng = np.random.default_rng(22)
    pnl = pd.Series(rng.normal(0, 1, 500), index=_idx(500))
    var = pd.Series(np.full(500, float(stats.norm.ppf(0.99))), index=_idx(500))
    result = var_kupiec_pof(_var_ctx(pnl, var, confidence=0.99))
    if result.status != Status.FAIL:
        assert "does not establish that the model is correct" in result.interpretation


# ==================================================== CHRISTOFFERSEN ==
def test_christoffersen_transition_counts_hand_calculation():
    """Sequence 0,1,1,0,0,1 -> pairs (0,1)(1,1)(1,0)(0,0)(0,1)
    n00=1, n01=2, n10=1, n11=1."""
    pnl = pd.Series([1.0, -3.0, -3.0, 1.0, 1.0, -3.0], index=_idx(6))
    var = pd.Series([2.0] * 6, index=_idx(6))
    sequence = exception_sequence(_var_ctx(pnl, var))
    assert sequence.transition_counts() == (1, 2, 1, 1)


def test_christoffersen_independence_known_answer():
    """LR_ind computed here from the explicit likelihood expression."""
    n00, n01, n10, n11 = 200, 20, 18, 4
    n0, n1 = n00 + n01, n10 + n11
    pi = (n01 + n11) / (n0 + n1)
    pi0, pi1 = n01 / n0, n11 / n1
    ll_null = (n01 + n11) * math.log(pi) + (n00 + n10) * math.log(1 - pi)
    ll_alt = n01 * math.log(pi0) + n00 * math.log(1 - pi0) + n11 * math.log(pi1) + n10 * math.log(1 - pi1)
    expected = -2.0 * (ll_null - ll_alt)
    assert abs(christoffersen_independence_lr(n00, n01, n10, n11) - expected) < 1e-10


def test_christoffersen_zero_cell_is_finite():
    """No exception ever follows another: n11 = 0. Must not produce NaN."""
    lr = christoffersen_independence_lr(200, 20, 20, 0)
    assert math.isfinite(lr)


def test_christoffersen_all_zero_exceptions_is_finite():
    lr = christoffersen_independence_lr(249, 0, 0, 0)
    assert math.isfinite(lr)
    assert abs(lr) < 1e-12


def test_independence_surface_records_transition_cells():
    pnl = pd.Series([1.0, -3.0, -3.0, 1.0, 1.0, -3.0] * 10, index=_idx(60))
    var = pd.Series([2.0] * 60, index=_idx(60))
    result = var_christoffersen_independence(_var_ctx(pnl, var))
    for key in ("n00", "n01", "n10", "n11", "pi_0", "pi_1"):
        assert key in result.metrics


def test_conditional_coverage_identity_holds_exactly():
    """LR_cc = LR_uc + LR_ind, the hard identity requirement."""
    rng = np.random.default_rng(23)
    pnl = pd.Series(rng.normal(0, 1, 400), index=_idx(400))
    var = pd.Series(np.full(400, 1.8), index=_idx(400))
    ctx = _var_ctx(pnl, var, confidence=0.99)
    cc = var_christoffersen_conditional(ctx)
    uc = var_kupiec_pof(ctx)
    ind = var_christoffersen_independence(ctx)
    assert abs(cc.metrics["lr_cc"] - (uc.metrics["lr_uc"] + ind.metrics["lr_ind"])) < 1e-9
    assert abs(cc.metrics["identity_residual"]) < 1e-12
    assert cc.metrics["degrees_of_freedom"] == 2


def test_all_three_tests_share_one_exception_sequence():
    """A convention mismatch would break the identity as a fake numerical problem."""
    rng = np.random.default_rng(24)
    pnl = pd.Series(rng.normal(0, 1, 300), index=_idx(300))
    var = pd.Series(np.full(300, 1.9), index=_idx(300))
    ctx = _var_ctx(pnl, var, confidence=0.99)
    hashes = {
        var_exceptions(ctx).metrics["exception_indicator_hash"],
        var_kupiec_pof(ctx).metrics["exception_indicator_hash"],
        var_christoffersen_independence(ctx).metrics["exception_indicator_hash"],
        var_christoffersen_conditional(ctx).metrics["exception_indicator_hash"],
        var_traffic_light(ctx, strict_applicability=False).metrics["exception_indicator_hash"],
    }
    assert len(hashes) == 1


# ==================================================== TRAFFIC LIGHT ==
def _traffic_ctx(n_exceptions, n=250, confidence=0.99):
    pnl = np.full(n, 1.0)
    pnl[:n_exceptions] = -3.0
    index = _idx(n)
    return _var_ctx(pd.Series(pnl, index=index), pd.Series(np.full(n, 2.0), index=index), confidence)


@pytest.mark.parametrize(
    "x,zone,status",
    [
        (0, "green", Status.RECORDED),
        (4, "green", Status.RECORDED),
        (5, "yellow", Status.WARN),
        (9, "yellow", Status.WARN),
        (10, "red", Status.FAIL),
        (25, "red", Status.FAIL),
    ],
)
def test_traffic_light_bands(x, zone, status):
    result = var_traffic_light(_traffic_ctx(x))
    assert result.metrics["zone"] == zone
    assert result.status == status


def test_traffic_light_band_boundaries_are_the_classical_ones():
    assert TRAFFIC_LIGHT_BANDS[0][:2] == (0, 4)
    assert TRAFFIC_LIGHT_BANDS[1][:2] == (5, 9)
    assert TRAFFIC_LIGHT_BANDS[2][0] == 10
    assert TRAFFIC_LIGHT_N == 250
    assert TRAFFIC_LIGHT_CONFIDENCE == 0.99


def test_traffic_light_skips_outside_its_applicability():
    """Reusing 250/99% bands elsewhere would be a generic classifier in regulatory dress."""
    result = var_traffic_light(_traffic_ctx(5, n=100))
    assert result.status == Status.SKIPPED
    assert "not applicable" in result.interpretation


def test_traffic_light_skips_on_wrong_confidence():
    result = var_traffic_light(_traffic_ctx(5, n=250, confidence=0.95))
    assert result.status == Status.SKIPPED


def test_traffic_light_can_be_forced_but_records_inapplicability():
    result = var_traffic_light(_traffic_ctx(5, n=100), strict_applicability=False)
    assert result.status != Status.SKIPPED
    assert result.metrics["applicable"] is False


def test_traffic_light_regulatory_scope_is_disclaimed():
    blob = " ".join(var_traffic_light(_traffic_ctx(3)).limitations)
    assert "HISTORICAL/CLASSICAL" in blob
    assert "NOT a complete implementation of the current Basel/FRTB" in blob


# ==================================================== BRIDGE ==
def test_bridge_known_answer_matches_an_independent_computation():
    """S0=100, S1=110, H=115, sigma=25%, 25-day interval. Expected computed here
    from the log-space formula directly, not by calling the helper."""
    a, b, H, sigma, dt = 100.0, 110.0, 115.0, 0.25, 25 / 252
    expected = math.exp(-2.0 * math.log(H / a) * math.log(H / b) / (sigma * sigma * dt))
    assert abs(barrier_crossing_probability(a, b, H, sigma, dt) - expected) < 1e-15
    assert 0.12 < expected < 0.15  # the live-verified ~0.135


def test_arithmetic_price_formula_is_not_used():
    """The arithmetic form collapses to ~0 where the log form gives 0.135."""
    a, b, H, sigma, dt = 100.0, 110.0, 115.0, 0.25, 25 / 252
    log_form = barrier_crossing_probability(a, b, H, sigma, dt)
    arithmetic = math.exp(-2.0 * (H - a) * (H - b) / (sigma * sigma * dt))
    assert log_form > 0.12
    assert arithmetic < 1e-6


def test_already_crossed_endpoint_returns_certainty():
    assert barrier_crossing_probability(100, 120, 115, 0.25, 0.1) == 1.0
    assert barrier_crossing_probability(120, 110, 115, 0.25, 0.1) == 1.0


def test_bridge_surface_reuses_the_b2_helper():
    """The equation must not be duplicated in an independent implementation."""
    import inspect

    from start.tests import traded_risk

    source = inspect.getsource(traded_risk)
    assert "barrier_crossing_probability" in source
    assert source.count("def barrier_crossing_probability") == 0  # imported, not redefined


def _price_ctx(world):
    """MarketContext carrying prices.

    world.market_context() supplies returns only, which is correct for the portfolio
    and attribution surfaces. The barrier test needs price levels, so the context is
    built explicitly here rather than widening the B2 factory.
    """
    return MarketContext(
        returns=world.returns,
        prices=world.prices,
        periods_per_year=world.periods_per_year,
        portfolio=PortfolioSpec(weights=world.weights),
    )


def test_bridge_surface_reports_under_detection():
    world = generate_market_world(n_assets=3, n_periods=200, seed=31)
    ctx = _price_ctx(world)
    barrier = float(ctx.prices.iloc[:, 0].max() * 0.98)
    result = brownian_bridge_barrier(ctx, barrier=barrier, sigma=0.25)
    assert result.metrics["space"] == "log_price"
    assert result.metrics["expected_continuous_crossings"] >= result.metrics["n_discrete_crossings"] - 1e-9


def test_bridge_requires_an_explicit_barrier():
    world = generate_market_world(n_assets=3, n_periods=100, seed=32)
    result = brownian_bridge_barrier(_price_ctx(world))
    assert result.status == Status.SKIPPED
    assert "never inferred" in result.interpretation


def test_bridge_is_not_described_as_interpolation():
    world = generate_market_world(n_assets=3, n_periods=100, seed=33)
    ctx = _price_ctx(world)
    result = brownian_bridge_barrier(ctx, barrier=float(ctx.prices.iloc[:, 0].max() * 0.99), sigma=0.25)
    blob = " ".join(result.limitations)
    assert "not missing-data interpolation" in blob


def test_bridge_rejects_non_positive_prices():
    index = _idx(5)
    prices = pd.DataFrame({"A": [100.0, 90.0, -1.0, 95.0, 100.0]}, index=index)
    ctx = MarketContext(prices=prices, portfolio=PortfolioSpec(weights=pd.Series([1.0], index=["A"])))
    result = brownian_bridge_barrier(ctx, barrier=115.0, sigma=0.25)
    assert result.status == Status.ERROR
