"""Focused unit tests for Gate 5 VaR Exception Backtesting and Tail Severity diagnostics.

Verifies:
1. Out-of-sample timing contract and timestamp alignment.
2. Strict separation of VaR confidence (alpha_var) and test significance (gamma_test).
3. Kupiec POF with mathematical limiting forms on edge cases (0 exceptions, in-line, excessive).
4. Christoffersen independence on known first-order Markov transition counts.
5. Degenerate sequence estimability handling.
6. Exact conditional coverage identity: LR_cc = LR_uc + LR_ind with 2 df.
7. Inter-exception duration and clustering diagnostics.
8. Tail severity and normalized exceedance calculations.
9. Proof-carrying negative showcase: Kupiec failure to reject + Christoffersen rejection on clustered failures.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from start.portfolio.tail_risk import (
    compute_exception_duration_diagnostics,
    compute_tail_severity,
    run_comprehensive_tail_backtest,
)


def test_out_of_sample_timing_and_alignment() -> None:
    """Realized loss at t must be aligned with forecast made at t-1."""
    dates = pd.date_range("2025-01-01", periods=10, freq="B")
    pnl = pd.Series([0.01, -0.02, 0.005, -0.03, 0.01, -0.01, 0.02, -0.04, 0.005, 0.01], index=dates)
    var = pd.Series([0.025] * 10, index=dates)

    # Exceptions should occur on dates with PnL < -0.025 (indices 3: -0.03, 7: -0.04)
    backtest = run_comprehensive_tail_backtest(
        pnl_or_losses=pnl,
        var_series=var,
        var_confidence=0.99,
        test_significance=0.05,
    )
    assert backtest.n_observations == 10
    assert backtest.n_exceptions == 2
    assert len(backtest.exception_dates) == 2
    assert backtest.indicators[3] == 1
    assert backtest.indicators[7] == 1


def test_confidence_and_significance_separation() -> None:
    """VaR confidence alpha_var must be distinct from statistical test significance gamma_test."""
    # alpha_var = 0.99 -> expected exception probability p0 = 0.01
    # gamma_test = 0.05 -> rejection critical threshold
    n = 500
    pnl = np.zeros(n)
    var = np.ones(n)  # No exceptions
    res = run_comprehensive_tail_backtest(
        pnl_or_losses=pnl,
        var_series=var,
        var_confidence=0.99,
        test_significance=0.05,
    )
    assert res.var_confidence == 0.99
    assert res.test_significance == 0.05
    assert math.isclose(res.expected_probability, 0.01, rel_tol=1e-6)
    assert math.isclose(res.expected_exceptions, 5.0, rel_tol=1e-6)


def test_kupiec_pof_limiting_cases() -> None:
    """Kupiec POF must calculate limit-safe likelihood ratio without NaN/log(0)."""
    # Case 1: 0 exceptions in 250 observations at p0 = 0.01
    # LR_uc = -2 * ln((1 - 0.01)^250) = -2 * 250 * ln(0.99) = 5.02525
    res_zero = run_comprehensive_tail_backtest(
        pnl_or_losses=np.zeros(250),
        var_series=np.ones(250),
        var_confidence=0.99,
        test_significance=0.05,
    )
    assert res_zero.n_exceptions == 0
    assert res_zero.kupiec_estimable is True
    expected_lr = -2.0 * 250.0 * math.log(0.99)
    assert math.isclose(res_zero.kupiec_lr, expected_lr, rel_tol=1e-4)
    # p-value = chi2.sf(5.025, 1) = 0.025 -> rejects at 0.05
    assert res_zero.kupiec_rejected is True

    # Case 2: Expected number of exceptions (3 exceptions in 250 obs at p0 = 0.01 -> rate = 0.012)
    # p-value should be high -> DOES NOT REJECT
    pnl_norm = np.zeros(250)
    pnl_norm[:3] = -2.0
    var_norm = np.ones(250)
    res_norm = run_comprehensive_tail_backtest(
        pnl_or_losses=pnl_norm,
        var_series=var_norm,
        var_confidence=0.99,
        test_significance=0.05,
    )
    assert res_norm.n_exceptions == 3
    assert res_norm.kupiec_rejected is False
    assert res_norm.kupiec_p_value > 0.50


def test_christoffersen_independence_known_transition_matrix() -> None:
    """Christoffersen test must compute exact 2x2 Markov transition counts and likelihood ratio."""
    # Indicator sequence: [0, 0, 1, 0, 1, 1, 0, 0, 0]
    # Pairs (prev -> curr):
    # 0 -> 0: (0,1), (6,7), (7,8) -> n00 = 3
    # 0 -> 1: (1,2), (3,4) -> n01 = 2
    # 1 -> 0: (2,3), (5,6) -> n10 = 2
    # 1 -> 1: (4,5) -> n11 = 1
    # Total transitions = 8
    pnl = np.array([0.0, 0.0, -2.0, 0.0, -2.0, -2.0, 0.0, 0.0, 0.0])
    var = np.ones(9)
    res = run_comprehensive_tail_backtest(
        pnl_or_losses=pnl,
        var_series=var,
        var_confidence=0.90,
        test_significance=0.05,
    )
    n00, n01, n10, n11 = res.transition_counts
    assert (n00, n01, n10, n11) == (3, 2, 2, 1)
    assert math.isclose(res.pi_01, 2.0 / 5.0, rel_tol=1e-6)
    assert math.isclose(res.pi_11, 1.0 / 3.0, rel_tol=1e-6)
    assert res.christoffersen_estimable is True


def test_christoffersen_degenerate_sequence_estimability() -> None:
    """Insufficient or degenerate exception sequences must set estimable=False cleanly."""
    # Very short sequence (N=2)
    res_short = run_comprehensive_tail_backtest(
        pnl_or_losses=np.array([0.0, -2.0]),
        var_series=np.array([1.0, 1.0]),
        var_confidence=0.99,
    )
    assert res_short.christoffersen_estimable is False
    assert math.isnan(res_short.christoffersen_p_value)


def test_joint_conditional_coverage_exact_identity() -> None:
    """Conditional coverage test statistic must satisfy exact identity LR_cc = LR_uc + LR_ind with 2 df."""
    # 250 observations with realistic exceptions
    rng = np.random.RandomState(42)
    pnl = rng.normal(loc=0.0, scale=0.01, size=250)
    var = np.full(250, 0.02)
    res = run_comprehensive_tail_backtest(
        pnl_or_losses=pnl,
        var_series=var,
        var_confidence=0.95,
        test_significance=0.05,
    )
    assert res.conditional_coverage_estimable is True
    assert math.isclose(res.conditional_coverage_lr, res.kupiec_lr + res.christoffersen_lr, rel_tol=1e-5)
    expected_p_cc = float(stats.chi2.sf(res.conditional_coverage_lr, df=2))
    assert math.isclose(res.conditional_coverage_p_value, expected_p_cc, rel_tol=1e-5)


def test_inter_exception_duration_diagnostics() -> None:
    """Inter-exception durations and streak statistics must match deterministic expectations."""
    # Exceptions at indices 1, 4, 6, 7 in a 10-period sequence
    indicators = np.array([0, 1, 0, 0, 1, 0, 1, 1, 0, 0])
    dur = compute_exception_duration_diagnostics(indicators)
    # Intervals: 4-1=3, 6-4=2, 7-6=1 -> (3, 2, 1)
    assert dur.n_durations == 3
    assert dur.durations == (3, 2, 1)
    assert math.isclose(dur.mean_duration, 2.0, rel_tol=1e-6)
    assert math.isclose(dur.median_duration, 2.0, rel_tol=1e-6)
    assert dur.min_duration == 1
    assert dur.max_duration == 3
    assert dur.max_run_length == 2  # indices 6 and 7 are consecutive 1s


def test_tail_severity_diagnostics() -> None:
    """Tail severity must compute exact absolute and normalized exceedance metrics."""
    losses = np.array([1.0, 3.0, 2.5, 4.0])
    var = 2.0
    sev = compute_tail_severity(losses=losses, var_forecasts=var)
    assert sev.n_exceptions == 3
    # Exceedances: [3.0 - 2.0, 2.5 - 2.0, 4.0 - 2.0] = [1.0, 0.5, 2.0]
    assert sev.absolute_exceedances == (1.0, 0.5, 2.0)
    assert math.isclose(sev.mean_absolute_exceedance, 3.5 / 3.0, rel_tol=1e-6)
    assert math.isclose(sev.median_absolute_exceedance, 1.0, rel_tol=1e-6)
    assert math.isclose(sev.max_absolute_exceedance, 2.0, rel_tol=1e-6)
    assert math.isclose(sev.total_tail_exceedance_loss, 3.5, rel_tol=1e-6)
    # Normalized: [3/2=1.5, 2.5/2=1.25, 4/2=2.0] -> max = 2.0, mean = 4.75 / 3 = 1.5833
    assert math.isclose(sev.max_normalized_exceedance, 2.0, rel_tol=1e-6)
    assert math.isclose(sev.mean_normalized_exceedance, 4.75 / 3.0, rel_tol=1e-6)


def test_negative_showcase_kupiec_no_reject_christoffersen_reject() -> None:
    """Proof-carrying test: Kupiec fails to reject while Christoffersen rejects on clustered exceptions."""
    n = 250
    alpha_var = 0.99
    gamma_test = 0.05
    pnl = np.zeros(n)
    # Inject 4 consecutive exceptions on days 50..53 (rate = 4/250 = 1.6%, expected = 1.0%)
    pnl[50] = -2.0
    pnl[51] = -2.0
    pnl[52] = -2.0
    pnl[53] = -2.0
    var = np.ones(n)

    res = run_comprehensive_tail_backtest(
        pnl_or_losses=pnl,
        var_series=var,
        var_confidence=alpha_var,
        test_significance=gamma_test,
    )

    # 1. Kupiec POF does NOT reject unconditional coverage
    assert res.kupiec_rejected is False
    assert res.kupiec_p_value > gamma_test

    # 2. Christoffersen independence REJECTS (p < 0.05) due to n11 = 3 transitions
    assert res.christoffersen_rejected is True
    assert res.christoffersen_p_value < gamma_test

    # 3. Joint Conditional Coverage REJECTS
    assert res.conditional_coverage_rejected is True
    assert res.conditional_coverage_p_value < gamma_test
