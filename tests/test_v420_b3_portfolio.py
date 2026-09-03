"""B3 — portfolio analytics, with independently derived known answers.

Expected values here are computed by hand or from closed-form algebra, never by calling
the production function whose output is under test.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from start.core.schemas import Status
from start.registry.market_contexts import (
    MarketContext,
    PortfolioConstraints,
    PortfolioSpec,
)
from start.tests.portfolio import (
    CONSTRAINT_TOLERANCE,
    EWMA_DECAY_DEFAULT,
    MAX_SHARPE_CONTRACT,
    annualised_geometric_return,
    covariance_conditioning,
    hierarchical_risk_parity,
    historical_returns,
    hrp_weights,
    max_drawdown,
    mean_variance,
    portfolio_wealth,
    risk_statistics,
    solve_min_variance,
)


def _market(returns, weights=None, **kw):
    if weights is None:
        weights = pd.Series(1.0 / returns.shape[1], index=returns.columns)
    kw.setdefault("portfolio", PortfolioSpec(weights=weights))
    return MarketContext(returns=returns, **kw)


def _series(values, freq="D"):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq=freq))


def _frame(data, freq="D"):
    return pd.DataFrame(
        data, index=pd.date_range("2024-01-01", periods=len(next(iter(data.values()))), freq=freq)
    )


# ================================================== FROZEN CONTRACT ==
def test_max_sharpe_contract_is_frozen_with_every_required_field():
    """The grid must be a specification, not an implementation detail."""
    for key in (
        "feasible_bound_method",
        "coarse_points",
        "refinement_points",
        "sharpe_basis",
        "annualisation",
        "tie_tolerance",
        "tie_break_order",
    ):
        assert key in MAX_SHARPE_CONTRACT
    assert MAX_SHARPE_CONTRACT["coarse_points"] == 101
    assert MAX_SHARPE_CONTRACT["refinement_points"] == 21
    assert MAX_SHARPE_CONTRACT["tie_tolerance"] == 1e-12
    assert len(MAX_SHARPE_CONTRACT["tie_break_order"]) == 4


def test_feasible_bounds_are_not_approximated_by_asset_means():
    assert "min/max of asset means" in MAX_SHARPE_CONTRACT["feasible_bound_method"]


def test_ewma_decay_is_frozen():
    assert EWMA_DECAY_DEFAULT == 0.94


# ============================================ KNOWN-ANSWER: helpers ==
def test_simple_wealth_compounds_multiplicatively():
    """Hand-computed: 1.10 * 1.10 = 1.21."""
    wealth = portfolio_wealth(_series([0.10, 0.10]), "simple")
    assert abs(float(wealth.iloc[-1]) - 1.21) < 1e-12


def test_log_wealth_compounds_exponentially():
    """exp(ln(1.1) + ln(1.1)) = 1.21."""
    r = math.log(1.10)
    wealth = portfolio_wealth(_series([r, r]), "log")
    assert abs(float(wealth.iloc[-1]) - 1.21) < 1e-12


def test_applying_the_wrong_wealth_rule_is_detectable():
    """prod(1+r) on log returns silently produces a plausible, wrong path."""
    r = math.log(1.10)
    correct = float(portfolio_wealth(_series([r, r]), "log").iloc[-1])
    wrong = float((1.0 + _series([r, r])).cumprod().iloc[-1])
    assert abs(correct - 1.21) < 1e-12
    assert abs(wrong - 1.21) > 1e-3


def test_geometric_annualisation_simple_known_answer():
    """4 periods doubling overall, ppy=4 -> exactly 100%."""
    r = _series([2**0.25 - 1] * 4)
    assert abs(annualised_geometric_return(r, 4.0, "simple") - 1.0) < 1e-12


def test_geometric_annualisation_log_known_answer():
    """mean log return * ppy, exponentiated: exp(ln2) - 1 = 1.0."""
    r = _series([math.log(2.0) / 4.0] * 4)
    assert abs(annualised_geometric_return(r, 4.0, "log") - 1.0) < 1e-12


def test_arithmetic_annualisation_would_differ():
    """Guards the single most common error in this calculation."""
    r = _series([0.10, -0.05, 0.08, 0.02])
    geometric = annualised_geometric_return(r, 4.0, "simple")
    arithmetic = float(r.mean()) * 4.0
    assert abs(geometric - arithmetic) > 1e-4


def test_max_drawdown_known_answer_and_dates():
    """Wealth 1.0 -> 1.2 -> 0.6 -> 0.9: peak 1.2, trough 0.6, MDD exactly -0.5."""
    r = _series([0.2, -0.5, 0.5])
    result = max_drawdown(r, "simple")
    assert abs(result["max_drawdown"] - (-0.5)) < 1e-12
    assert "2024-01-01" in result["drawdown_start"]
    assert "2024-01-02" in result["drawdown_trough"]


def test_unrecovered_drawdown_reports_none():
    r = _series([0.2, -0.5, 0.01])
    assert max_drawdown(r, "simple")["drawdown_recovery"] is None


def test_recovered_drawdown_reports_a_date():
    r = _series([0.2, -0.5, 1.5])
    assert max_drawdown(r, "simple")["drawdown_recovery"] is not None


# ==================================== historical_returns ==
def test_historical_returns_weights_correctly():
    """0.5*0.10 + 0.5*0.20 = 0.15, by hand."""
    frame = _frame({"A": [0.10, 0.0], "B": [0.20, 0.0]})
    weights = pd.Series([0.5, 0.5], index=["A", "B"])
    result = historical_returns(_market(frame, weights))
    assert result.status == Status.RECORDED
    assert abs(result.metrics["max_periodic_return"] - 0.15) < 1e-12


def test_equal_weights_are_never_invented():
    """A portfolio review without weights would report on a portfolio nobody specified."""
    frame = _frame({"A": [0.01, 0.02], "B": [0.03, 0.04]})
    result = historical_returns(MarketContext(returns=frame))
    assert result.status == Status.SKIPPED
    assert "not invented" in result.interpretation


def test_missing_weight_for_an_asset_is_refused():
    frame = _frame({"A": [0.01, 0.02], "B": [0.03, 0.04]})
    result = historical_returns(_market(frame, pd.Series([1.0], index=["A"])))
    assert result.status == Status.SKIPPED
    assert "weights missing" in result.interpretation


def test_return_basis_is_recorded_and_honoured():
    frame = _frame({"A": [0.01, 0.02], "B": [0.03, 0.04]})
    assert historical_returns(_market(frame)).metrics["return_basis"] == "simple"
    ctx = _market(frame, return_basis="log")
    assert historical_returns(ctx).metrics["return_basis"] == "log"


def test_returns_derived_from_prices_are_flagged():
    prices = _frame({"A": [100.0, 110.0, 121.0], "B": [50.0, 55.0, 60.5]})
    ctx = MarketContext(prices=prices, portfolio=PortfolioSpec(weights=pd.Series(0.5, index=["A", "B"])))
    result = historical_returns(ctx)
    assert result.metrics["derived_from_prices"] is True
    assert abs(result.metrics["mean_periodic_return"] - 0.10) < 1e-12


# ==================================== risk_statistics ==
def test_sharpe_known_answer_with_zero_risk_free():
    """mean/sd * sqrt(ppy), computed by hand from a fixed vector."""
    values = [0.01, 0.02, -0.01, 0.03, 0.00]
    frame = _frame({"A": values})
    result = risk_statistics(_market(frame, pd.Series([1.0], index=["A"])))
    expected = float(np.mean(values)) / float(np.std(values, ddof=1)) * math.sqrt(252.0)
    assert abs(result.metrics["sharpe_ratio"] - expected) < 1e-9


def test_sharpe_uses_periodic_excess_not_geometric_cagr():
    """CAGR compounds, sigma*sqrt(T) does not; their ratio is not the estimator."""
    values = [0.01, 0.02, -0.01, 0.03, 0.00]
    frame = _frame({"A": values})
    ctx = _market(frame, pd.Series([1.0], index=["A"]))
    result = risk_statistics(ctx)
    geo = result.metrics["annualised_geometric_return"]
    vol = result.metrics["annualised_volatility"]
    assert abs(result.metrics["sharpe_ratio"] - geo / vol) > 1e-6


def test_risk_free_is_converted_to_the_return_period():
    """Subtracting an annual rate from a daily return is wrong by ~252x."""
    values = [0.001] * 100
    frame = _frame({"A": values})
    ctx = _market(frame, pd.Series([1.0], index=["A"]), risk_free_rate=0.03, risk_free_frequency="annual")
    result = risk_statistics(ctx)
    assert result.metrics["risk_free_source_periods_per_year"] == 1.0
    assert result.metrics["risk_free_target_periods_per_year"] == 252.0
    assert 0 < result.metrics["risk_free_period_rate"] < 0.0005


def test_sortino_records_its_downside_convention():
    """The two conventions give materially different numbers."""
    frame = _frame({"A": [0.02, -0.01, 0.03, -0.02, 0.01]})
    result = risk_statistics(_market(frame, pd.Series([1.0], index=["A"])))
    assert "ALL observations" in result.metrics["downside_deviation_convention"]
    assert result.metrics["sortino_ratio"] is not None


def test_calmar_is_geometric_return_over_drawdown():
    frame = _frame({"A": [0.2, -0.5, 0.5]})
    result = risk_statistics(_market(frame, pd.Series([1.0], index=["A"])))
    expected = result.metrics["annualised_geometric_return"] / abs(result.metrics["max_drawdown"])
    assert abs(result.metrics["calmar_ratio"] - expected) < 1e-9


def test_zero_volatility_gives_none_not_infinity():
    """An infinite Sharpe is a degenerate input, not a good portfolio."""
    frame = _frame({"A": [0.01] * 20})
    result = risk_statistics(_market(frame, pd.Series([1.0], index=["A"])))
    assert result.metrics["sharpe_ratio"] is None
    assert "zero excess-return standard deviation" in result.metrics["sharpe_undefined_reason"]


def test_historical_var_known_answer():
    """95% VaR of 0..99 is the 5th percentile, negated."""
    values = list(np.arange(100, dtype=float) / 1000.0 - 0.05)
    frame = _frame({"A": values})
    result = risk_statistics(_market(frame, pd.Series([1.0], index=["A"])), var_confidence=0.95)
    expected = -float(np.percentile(values, 5))
    assert abs(result.metrics["historical_var"] - expected) < 1e-12


def test_es_requires_enough_tail_observations():
    """An ES from two observations is noise with a decimal point."""
    frame = _frame({"A": list(np.linspace(-0.05, 0.05, 20))})
    result = risk_statistics(
        _market(frame, pd.Series([1.0], index=["A"])), var_confidence=0.95, min_tail_observations=10
    )
    assert result.metrics["historical_es"] is None
    assert "tail observation" in result.metrics["es_insufficient_reason"]


def test_es_is_reported_with_enough_tail():
    frame = _frame({"A": list(np.linspace(-0.05, 0.05, 500))})
    result = risk_statistics(
        _market(frame, pd.Series([1.0], index=["A"])), var_confidence=0.95, min_tail_observations=10
    )
    assert result.metrics["historical_es"] is not None
    assert result.metrics["historical_es"] > result.metrics["historical_var"]


def test_var_basis_is_stated():
    frame = _frame({"A": list(np.linspace(-0.05, 0.05, 100))})
    result = risk_statistics(_market(frame, pd.Series([1.0], index=["A"])))
    assert result.metrics["var_basis"] == "simple_return"


# ==================================== covariance_conditioning ==
def test_conditioning_of_a_known_diagonal_matrix():
    """Eigenvalues are the diagonal; condition number is exactly 4/1."""
    cov = pd.DataFrame(np.diag([4.0, 2.0, 1.0]), index=list("ABC"), columns=list("ABC"))
    frame = _frame({c: list(np.random.default_rng(0).normal(size=50)) for c in "ABC"})
    result = covariance_conditioning(_market(frame, covariance=cov))
    assert abs(result.metrics["condition_number"] - 4.0) < 1e-9
    assert abs(result.metrics["min_eigenvalue"] - 1.0) < 1e-12
    assert result.metrics["rank"] == 3
    assert result.metrics["is_psd"] is True


def test_singular_covariance_is_detected():
    cov = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], index=list("AB"), columns=list("AB"))
    frame = _frame({c: list(np.random.default_rng(0).normal(size=50)) for c in "AB"})
    result = covariance_conditioning(_market(frame, covariance=cov))
    assert result.metrics["rank"] == 1
    assert result.metrics["full_rank"] is False


def test_non_psd_covariance_fails():
    cov = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=list("AB"), columns=list("AB"))
    frame = _frame({c: list(np.random.default_rng(0).normal(size=50)) for c in "AB"})
    result = covariance_conditioning(_market(frame, covariance=cov))
    assert result.metrics["is_psd"] is False
    assert result.status == Status.FAIL


def test_no_silent_repair():
    cov = pd.DataFrame([[1.0, 2.0], [2.0, 1.0]], index=list("AB"), columns=list("AB"))
    frame = _frame({c: list(np.random.default_rng(0).normal(size=50)) for c in "AB"})
    result = covariance_conditioning(_market(frame, covariance=cov))
    assert result.metrics["repair_applied"] is False
    assert any("NO REPAIR" in x for x in result.limitations)


# ==================================== mean_variance ==
def test_two_asset_minimum_variance_closed_form():
    """Uncorrelated assets: w1 = s2^2/(s1^2+s2^2), exactly."""
    v1, v2 = 0.04, 0.01
    sigma = np.diag([v1, v2])
    mu = np.array([0.01, 0.01])
    weights, diagnostics = solve_min_variance(mu, sigma, None)
    expected_first = v2 / (v1 + v2)
    assert diagnostics["converged"]
    assert abs(weights[0] - expected_first) < 1e-7
    assert abs(weights.sum() - 1.0) < 1e-9


def test_minimum_variance_with_correlation_closed_form():
    """w1 = (s2^2 - cov) / (s1^2 + s2^2 - 2cov), the standard two-asset result."""
    v1, v2, cov = 0.04, 0.01, 0.005
    sigma = np.array([[v1, cov], [cov, v2]])
    weights, _ = solve_min_variance(np.array([0.01, 0.01]), sigma, None)
    expected = (v2 - cov) / (v1 + v2 - 2 * cov)
    assert abs(weights[0] - expected) < 1e-7


def test_target_return_constraint_is_met_exactly():
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (300, 4)),
        index=pd.date_range("2024-01-01", periods=300, freq="D"),
        columns=list("ABCD"),
    )
    ctx = _market(frame)
    mu = frame.mean().to_numpy()
    target = float(np.median(mu))
    result = mean_variance(ctx, objective="target_return", target_return=target)
    assert result.status == Status.RECORDED
    assert abs(result.metrics["expected_return_periodic"] - target) < CONSTRAINT_TOLERANCE


def test_long_only_is_respected():
    rng = np.random.default_rng(5)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (200, 5)),
        index=pd.date_range("2024-01-01", periods=200, freq="D"),
        columns=list("ABCDE"),
    )
    ctx = _market(frame, weights=pd.Series(0.2, index=list("ABCDE")))
    ctx.portfolio.constraints = PortfolioConstraints(long_only=True)
    result = mean_variance(ctx, objective="min_variance")
    assert result.metrics["min_weight"] >= -CONSTRAINT_TOLERANCE


def test_max_weight_cap_is_respected():
    rng = np.random.default_rng(6)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (200, 5)),
        index=pd.date_range("2024-01-01", periods=200, freq="D"),
        columns=list("ABCDE"),
    )
    ctx = _market(frame, weights=pd.Series(0.2, index=list("ABCDE")))
    ctx.portfolio.constraints = PortfolioConstraints(long_only=True, max_weight=0.30)
    result = mean_variance(ctx, objective="min_variance")
    assert result.metrics["max_weight"] <= 0.30 + CONSTRAINT_TOLERANCE


def test_infeasible_constraints_error_before_solving():
    rng = np.random.default_rng(7)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (100, 3)),
        index=pd.date_range("2024-01-01", periods=100, freq="D"),
        columns=list("ABC"),
    )
    ctx = _market(frame, weights=pd.Series(1 / 3, index=list("ABC")))
    ctx.portfolio.constraints = PortfolioConstraints(min_weight=0.5, max_weight=0.2)
    result = mean_variance(ctx)
    assert result.status == Status.ERROR
    assert "infeasible" in result.interpretation


def test_concentration_with_long_short_is_rejected():
    """HHI is not a concentration measure on a long/short book."""
    rng = np.random.default_rng(8)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (100, 3)),
        index=pd.date_range("2024-01-01", periods=100, freq="D"),
        columns=list("ABC"),
    )
    ctx = _market(frame, weights=pd.Series(1 / 3, index=list("ABC")))
    ctx.portfolio.constraints = PortfolioConstraints(long_only=False, max_concentration=0.4)
    result = mean_variance(ctx)
    assert result.status == Status.ERROR
    assert "long_only" in result.interpretation


def test_max_sharpe_records_the_frozen_grid():
    rng = np.random.default_rng(9)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (250, 4)),
        index=pd.date_range("2024-01-01", periods=250, freq="D"),
        columns=list("ABCD"),
    )
    ctx = _market(frame)
    result = mean_variance(ctx, objective="max_sharpe")
    assert result.status == Status.RECORDED
    assert result.metrics["grid_coarse_points"] == 101
    assert result.metrics["grid_refinement_points"] == 21
    assert result.metrics["grid_mu_min"] <= result.metrics["grid_selected_target"]
    assert result.metrics["grid_selected_target"] <= result.metrics["grid_mu_max"]


def test_max_sharpe_beats_or_matches_min_variance_on_sharpe():
    rng = np.random.default_rng(10)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (250, 4)),
        index=pd.date_range("2024-01-01", periods=250, freq="D"),
        columns=list("ABCD"),
    )
    ctx = _market(frame)
    best = mean_variance(ctx, objective="max_sharpe").metrics["sharpe_periodic"]
    minvar = mean_variance(ctx, objective="min_variance").metrics["sharpe_periodic"]
    assert best >= minvar - 1e-9


def test_max_sharpe_is_deterministic():
    rng = np.random.default_rng(11)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (200, 4)),
        index=pd.date_range("2024-01-01", periods=200, freq="D"),
        columns=list("ABCD"),
    )
    ctx = _market(frame)
    a = mean_variance(ctx, objective="max_sharpe")
    b = mean_variance(ctx, objective="max_sharpe")
    assert a.metrics["grid_selected_target"] == b.metrics["grid_selected_target"]
    for asset in "ABCD":
        assert abs(a.metrics[f"weight.{asset}"] - b.metrics[f"weight.{asset}"]) < 1e-9


def test_ewma_expected_return_differs_from_the_mean():
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (200, 3)),
        index=pd.date_range("2024-01-01", periods=200, freq="D"),
        columns=list("ABC"),
    )
    ctx = _market(frame, weights=pd.Series(1 / 3, index=list("ABC")))
    plain = mean_variance(ctx, objective="max_sharpe", expected_return="mean")
    ewma = mean_variance(ctx, objective="max_sharpe", expected_return="ewma")
    assert plain.metrics["grid_mu_max"] != ewma.metrics["grid_mu_max"]
    assert ewma.params["ewma_decay"] == EWMA_DECAY_DEFAULT


def test_ledoit_wolf_covariance_is_selectable():
    rng = np.random.default_rng(13)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (60, 8)),
        index=pd.date_range("2024-01-01", periods=60, freq="D"),
        columns=[f"A{i}" for i in range(8)],
    )
    ctx = _market(frame, weights=pd.Series(0.125, index=frame.columns))
    result = mean_variance(ctx, objective="min_variance", covariance="ledoit_wolf")
    assert result.status == Status.RECORDED
    assert result.params["covariance"] == "ledoit_wolf"


def test_turnover_is_reported_against_prior_weights():
    rng = np.random.default_rng(14)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (150, 3)),
        index=pd.date_range("2024-01-01", periods=150, freq="D"),
        columns=list("ABC"),
    )
    prior = pd.Series([1.0, 0.0, 0.0], index=list("ABC"))
    ctx = _market(frame, weights=pd.Series(1 / 3, index=list("ABC")))
    ctx.portfolio.prior_weights = prior
    result = mean_variance(ctx, objective="min_variance")
    assert "one_way_turnover" in result.metrics
    assert 0.0 <= result.metrics["one_way_turnover"] <= 1.0


def test_unknown_objective_is_rejected():
    rng = np.random.default_rng(15)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (50, 3)),
        index=pd.date_range("2024-01-01", periods=50, freq="D"),
        columns=list("ABC"),
    )
    with pytest.raises(ValueError, match="not supported"):
        mean_variance(_market(frame, pd.Series(1 / 3, index=list("ABC"))), objective="max_return")


# ==================================== HRP ==
def test_hrp_single_asset_gets_everything():
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    weights, order = hrp_weights(cov)
    assert abs(float(weights.iloc[0]) - 1.0) < 1e-12
    assert order == ["A"]


def test_hrp_two_assets_is_inverse_variance():
    """Known answer: w1 = (1/v1)/((1/v1)+(1/v2)) = v2/(v1+v2)."""
    v1, v2 = 0.04, 0.01
    cov = pd.DataFrame(np.diag([v1, v2]), index=list("AB"), columns=list("AB"))
    weights, _ = hrp_weights(cov)
    assert abs(float(weights.loc["A"]) - v2 / (v1 + v2)) < 1e-12
    assert abs(float(weights.loc["B"]) - v1 / (v1 + v2)) < 1e-12


def test_hrp_equal_variance_uncorrelated_is_equal_weight():
    """Independently derivable: identical assets must receive identical weight."""
    cov = pd.DataFrame(np.diag([0.04] * 4), index=list("ABCD"), columns=list("ABCD"))
    weights, _ = hrp_weights(cov)
    assert np.allclose(weights.to_numpy(), 0.25, atol=1e-6)


def test_hrp_weights_sum_to_one_and_are_non_negative():
    rng = np.random.default_rng(16)
    data = rng.normal(size=(300, 6))
    cov = pd.DataFrame(np.cov(data, rowvar=False), index=list("ABCDEF"), columns=list("ABCDEF"))
    weights, _ = hrp_weights(cov)
    assert abs(float(weights.sum()) - 1.0) < 1e-9
    assert (weights >= -1e-12).all()


def test_hrp_gives_lower_weight_to_the_riskier_asset():
    cov = pd.DataFrame(np.diag([0.09, 0.01, 0.04, 0.01]), index=list("ABCD"), columns=list("ABCD"))
    weights, _ = hrp_weights(cov)
    assert float(weights.loc["A"]) < float(weights.loc["B"])


def test_hrp_non_positive_variance_errors():
    cov = pd.DataFrame(np.diag([0.04, 0.0, 0.01]), index=list("ABC"), columns=list("ABC"))
    with pytest.raises(ValueError, match="degenerate variance"):
        hrp_weights(cov)


def test_hrp_degeneracy_check_is_relative_not_exact():
    """A constant column has variance ~1e-36, not exactly 0. An exact test lets it
    through and the inverse-variance step then hands it almost the whole book."""
    rng = np.random.default_rng(31)
    frame = pd.DataFrame(
        {"A": rng.normal(size=200), "B": [0.01] * 200, "C": rng.normal(size=200)},
        index=pd.date_range("2024-01-01", periods=200, freq="D"),
    )
    variances = np.diag(frame.cov().to_numpy())
    assert 0.0 < float(variances.min()) < 1e-30  # not exactly zero
    with pytest.raises(ValueError, match="degenerate variance"):
        hrp_weights(frame.cov())


def test_hrp_test_records_the_quasi_diagonal_order():
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (200, 5)),
        index=pd.date_range("2024-01-01", periods=200, freq="D"),
        columns=list("ABCDE"),
    )
    result = hierarchical_risk_parity(_market(frame, pd.Series(0.2, index=list("ABCDE"))))
    assert result.status == Status.RECORDED
    assert result.metrics["quasi_diagonal_order"]
    assert result.metrics["linkage_method"] == "single"
    assert abs(result.metrics["weights_sum"] - 1.0) < 1e-9


def test_hrp_errors_cleanly_on_a_degenerate_asset():
    frame = _frame(
        {
            "A": list(np.random.default_rng(18).normal(size=100)),
            "B": [0.01] * 100,
            "C": list(np.random.default_rng(19).normal(size=100)),
        }
    )
    result = hierarchical_risk_parity(_market(frame, pd.Series(1 / 3, index=list("ABC"))))
    assert result.status == Status.ERROR


def test_hrp_is_deterministic():
    rng = np.random.default_rng(20)
    frame = pd.DataFrame(
        rng.normal(0.001, 0.01, (200, 5)),
        index=pd.date_range("2024-01-01", periods=200, freq="D"),
        columns=list("ABCDE"),
    )
    ctx = _market(frame, pd.Series(0.2, index=list("ABCDE")))
    a = hierarchical_risk_parity(ctx)
    b = hierarchical_risk_parity(ctx)
    assert a.metrics == b.metrics
