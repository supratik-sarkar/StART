"""B1 — market contexts, portfolio state, canonical fingerprinting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from start.registry.contexts import ContextKind, ReviewContext
from start.registry.market_contexts import (
    FREQUENCY_PERIODS,
    MarketContext,
    PortfolioConstraints,
    PortfolioSpec,
    ShortRateContext,
    canonical_frame_bytes,
    canonical_scalar,
    convert_rate,
)
from start.risk.coverage import normalise_context_type


def _returns(n=200, k=4, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 0.01, (n, k)),
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
        columns=[f"A{i}" for i in range(k)],
    )


def _spec(frame):
    return PortfolioSpec(weights=pd.Series(1.0 / frame.shape[1], index=frame.columns))


def _ctx(frame=None, **kw):
    frame = _returns() if frame is None else frame
    kw.setdefault("portfolio", _spec(frame))
    return MarketContext(returns=frame, **kw)


# ------------------------------------------------------------- protocol ==
def test_market_context_satisfies_review_context():
    assert isinstance(_ctx(), ReviewContext)


def test_short_rate_context_satisfies_review_context():
    rates = pd.Series(np.linspace(0.02, 0.05, 300),
                      index=pd.date_range("2024-01-01", periods=300, freq="D"))
    assert isinstance(ShortRateContext(rates=rates), ReviewContext)


def test_context_kinds_are_canonical():
    assert _ctx().context_kind() == ContextKind.MARKET.value == "market"
    rates = pd.Series([0.02] * 300, index=pd.date_range("2024-01-01", periods=300, freq="D"))
    assert ShortRateContext(rates=rates).context_kind() == "short_rate"


def test_gate_a_alias_normaliser_already_knows_the_new_kinds():
    assert normalise_context_type("MarketContext") == "market"
    assert normalise_context_type("ShortRateContext") == "short_rate"


def test_describe_carries_shapes_not_frames():
    for value in _ctx().describe().values():
        assert not isinstance(value, (pd.DataFrame, pd.Series))


# ------------------------------------------------- typed canonicalisation ==
@pytest.mark.parametrize(
    "left,right",
    [(1, "1"), (1, True), (0, False), (True, "True"), (1.0, 1), (1.0, "1.0")],
)
def test_typed_scalars_do_not_collide(left, right):
    """str(value) would collapse these onto one token."""
    assert canonical_scalar(left) != canonical_scalar(right)


def test_signed_zero_and_nan_are_normalised():
    assert canonical_scalar(-0.0) == canonical_scalar(0.0)
    assert canonical_scalar(float("nan")) == canonical_scalar(float("nan"))
    assert canonical_scalar(float("inf")) != canonical_scalar(float("-inf"))


def test_unicode_is_nfc_normalised():
    assert canonical_scalar("é") == canonical_scalar("e\u0301")


def test_mixed_type_column_labels_do_not_crash_sorting():
    """Ordinary sorted() raises on int and str labels together; pandas allows both."""
    frame = pd.DataFrame({1: [1.0, 2.0], "b": [3.0, 4.0], 2.5: [5.0, 6.0]})
    assert canonical_frame_bytes(frame)


def test_duplicate_columns_are_rejected():
    frame = pd.DataFrame(np.ones((3, 2)), columns=["a", "a"])
    with pytest.raises(ValueError, match="duplicate column"):
        canonical_frame_bytes(frame)


def test_duplicate_index_is_rejected():
    frame = pd.DataFrame({"a": [1.0, 2.0]}, index=[0, 0])
    with pytest.raises(ValueError, match="duplicate index"):
        canonical_frame_bytes(frame)


# -------------------------------------------------- fingerprint identity ==
def test_column_order_is_not_semantic():
    frame = _returns()
    a = _ctx(frame)
    b = MarketContext(returns=frame[sorted(frame.columns, reverse=True)],
                      portfolio=_spec(frame))
    assert a.fingerprint() == b.fingerprint()


def test_lossless_float32_equals_float64():
    values = np.array([[0.5, 0.25], [0.125, -0.75]], dtype=np.float64)
    index = pd.date_range("2024-01-01", periods=2, freq="D")
    f64 = pd.DataFrame(values, index=index, columns=["A", "B"])
    f32 = f64.astype("float32")
    assert (f32.astype("float64") == f64).all().all()
    assert MarketContext(returns=f64).fingerprint() == MarketContext(returns=f32).fingerprint()


def test_lossy_float32_correctly_differs():
    """A cast that loses precision changes the values, so it must change the hash."""
    index = pd.date_range("2024-01-01", periods=1, freq="D")
    exact = pd.DataFrame([[0.1, 0.2]], index=index, columns=["A", "B"])
    lossy = exact.astype("float32")
    assert MarketContext(returns=exact).fingerprint() != MarketContext(returns=lossy).fingerprint()


def test_naive_and_utc_timestamps_agree():
    frame = _returns()
    aware = frame.copy()
    aware.index = aware.index.tz_localize("UTC")
    assert _ctx(frame).fingerprint() == MarketContext(
        returns=aware, portfolio=_spec(frame)
    ).fingerprint()


def test_naive_assumption_is_recorded():
    assert _ctx()._naive_timestamps_assumed_utc is True


def test_static_and_fully_repeated_exposures_agree():
    frame = _returns()
    exposures = pd.DataFrame(np.arange(12.0).reshape(4, 3),
                             index=frame.columns, columns=["f1", "f2", "f3"])
    static = MarketContext(returns=frame, factor_exposures=exposures, portfolio=_spec(frame))
    repeated = MarketContext(returns=frame,
                             factor_exposures={t: exposures for t in frame.index},
                             portfolio=_spec(frame))
    assert static.fingerprint() == repeated.fingerprint()
    assert static.is_time_varying_exposure is False
    assert repeated.is_time_varying_exposure is False


def test_partial_exposure_coverage_does_not_collapse_to_static():
    """A dict covering only part of the support is NOT the same thing."""
    frame = _returns()
    exposures = pd.DataFrame(np.arange(12.0).reshape(4, 3),
                             index=frame.columns, columns=["f1", "f2", "f3"])
    static = MarketContext(returns=frame, factor_exposures=exposures, portfolio=_spec(frame))
    partial = MarketContext(returns=frame,
                            factor_exposures={t: exposures for t in frame.index[:50]},
                            portfolio=_spec(frame))
    assert static.fingerprint() != partial.fingerprint()
    assert partial.is_time_varying_exposure is True


@pytest.mark.parametrize("field", ["return", "weight", "ppy", "rf_frequency", "constraint"])
def test_semantic_change_moves_the_fingerprint(field):
    frame = _returns()
    base = _ctx(frame).fingerprint()
    if field == "return":
        changed = frame.copy()
        changed.iloc[0, 0] += 1e-6
        other = MarketContext(returns=changed, portfolio=_spec(frame)).fingerprint()
    elif field == "weight":
        weights = pd.Series(0.25, index=frame.columns)
        weights.iloc[0] = 0.30
        other = MarketContext(returns=frame,
                              portfolio=PortfolioSpec(weights=weights)).fingerprint()
    elif field == "ppy":
        other = MarketContext(returns=frame, periods_per_year=52.0,
                              portfolio=_spec(frame)).fingerprint()
    elif field == "rf_frequency":
        other = MarketContext(returns=frame, risk_free_rate=0.03,
                              risk_free_frequency="annual",
                              portfolio=_spec(frame)).fingerprint()
    else:
        other = MarketContext(
            returns=frame,
            portfolio=PortfolioSpec(weights=pd.Series(0.25, index=frame.columns),
                                    constraints=PortfolioConstraints(max_weight=0.4)),
        ).fingerprint()
    assert base != other


def test_fingerprint_is_stable_across_calls():
    ctx = _ctx()
    assert ctx.fingerprint() == ctx.fingerprint()


# ------------------------------------------------------- portfolio state ==
def test_portfolio_spec_is_the_only_weight_holder():
    """No parallel weights field anywhere on MarketContext."""
    forbidden = {"weights", "benchmark_weights", "prior_weights"}
    assert not (forbidden & set(vars(_ctx())))


def test_concentration_requires_long_only():
    """HHI is not a concentration measure on a long/short book."""
    problems = PortfolioConstraints(long_only=False, max_concentration=0.2).validate()
    assert problems and "long_only" in problems[0]


def test_constraint_contradictions_are_caught():
    assert PortfolioConstraints(min_weight=0.5, max_weight=0.2).validate()
    assert PortfolioConstraints(budget=1.0, max_leverage=0.5).validate()


def test_valid_constraints_report_no_problems():
    assert PortfolioConstraints(long_only=True, max_weight=0.4,
                                max_concentration=0.3).validate() == []


# ------------------------------------------------------- rate conventions ==
def test_annual_to_daily_conversion_is_compounding():
    daily = convert_rate(0.03, 1.0, 252.0)
    assert abs((1 + daily) ** 252 - 1.03) < 1e-12
    # Proportional division would give 1.19e-4; compounding gives less.
    assert daily < 0.03 / 252


def test_risk_free_conversion_is_recorded():
    ctx = _ctx(risk_free_rate=0.03, risk_free_frequency="annual")
    value, record = ctx.risk_free_per_period()
    assert record["risk_free_source_periods_per_year"] == 1.0
    assert record["risk_free_target_periods_per_year"] == 252.0
    assert "conversion" in " ".join(record).lower() or record["risk_free_conversion"]
    assert 0 < value < 0.001


def test_explicit_periods_per_year_overrides_the_label():
    """252, 260 and 365 are all legitimate 'daily'."""
    ctx = MarketContext(returns=_returns(), frequency="daily", periods_per_year=365.0)
    assert ctx.periods_per_year == 365.0


def test_frequency_label_resolves_when_ppy_is_default():
    assert MarketContext(returns=_returns(), frequency="monthly").periods_per_year == 12.0
    assert FREQUENCY_PERIODS["monthly"] == 12.0


# ---------------------------------------------------------- return basis ==
def test_simple_and_log_return_derivation_differ():
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0]},
                          index=pd.date_range("2024-01-01", periods=3, freq="D"))
    simple = MarketContext(prices=prices, return_basis="simple").effective_returns()
    log = MarketContext(prices=prices, return_basis="log").effective_returns()
    assert abs(simple.iloc[0, 0] - 0.10) < 1e-12
    assert abs(log.iloc[0, 0] - np.log(1.10)) < 1e-12
    assert not np.allclose(simple.to_numpy(), log.to_numpy())


def test_invalid_return_basis_is_rejected():
    with pytest.raises(ValueError, match="return_basis"):
        MarketContext(returns=_returns(), return_basis="continuous")


# -------------------------------------------------------------- validation ==
def test_context_without_returns_or_prices_is_invalid():
    assert "neither returns nor prices" in " ".join(MarketContext().validate_context())


def test_var_series_without_confidence_is_invalid():
    frame = _returns()
    ctx = MarketContext(returns=frame,
                        var_series=pd.Series(0.02, index=frame.index))
    assert any("var_confidence" in p for p in ctx.validate_context())


def test_clean_market_context_validates():
    assert _ctx().validate_context() == []


def test_short_rate_missing_observations_are_rejected_not_filled():
    """A diffusion estimate over interpolated values is biased with no visible symptom."""
    rates = pd.Series(np.linspace(0.02, 0.05, 300),
                      index=pd.date_range("2024-01-01", periods=300, freq="D"))
    rates.iloc[5] = np.nan
    problems = ShortRateContext(rates=rates).validate_context()
    assert any("rejected rather than filled" in p for p in problems)


def test_short_rate_below_minimum_observations_is_flagged():
    rates = pd.Series(np.linspace(0.02, 0.05, 100),
                      index=pd.date_range("2024-01-01", periods=100, freq="D"))
    assert any("below the required" in p for p in ShortRateContext(rates=rates).validate_context())


def test_percent_units_are_normalised_and_recorded():
    rates = pd.Series(np.linspace(2.0, 5.0, 300),
                      index=pd.date_range("2024-01-01", periods=300, freq="D"))
    ctx = ShortRateContext(rates=rates, units="percent")
    decimal = ctx.decimal_rates()
    assert abs(decimal.iloc[0] - 0.02) < 1e-12
    assert ctx.describe()["normalised_from_percent"] is True


def test_short_rate_dt_follows_periods_per_year():
    rates = pd.Series([0.02] * 300, index=pd.date_range("2024-01-01", periods=300, freq="D"))
    assert abs(ShortRateContext(rates=rates, periods_per_year=252.0).dt - 1 / 252) < 1e-15
