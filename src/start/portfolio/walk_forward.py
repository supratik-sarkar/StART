"""Non-leaky portfolio walk-forward evaluation harness.

Core principles:
- Strict time ordering: parameter estimation uses historical slice [t - W, t) only.
- Decision stamping: weights are stamped at decision date t.
- Out-of-sample evaluation: returns are measured over [t, t + F) only after the decision.
- Transaction cost modeling: explicit transaction_cost_bps subtracted from rebalance points.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from start.portfolio.contracts import WalkForwardResult
from start.portfolio.hrp import hrp_weights_and_tree
from start.portfolio.optimization import solve_equal_risk_contribution


def run_walk_forward_evaluation(
    returns: pd.DataFrame,
    method: str = "hrp",
    estimation_window: int = 126,
    rebalance_frequency: int = 21,
    transaction_cost_bps: float = 0.0,
    periods_per_year: float = 252.0,
    rf_periodic: float = 0.0,
) -> WalkForwardResult:
    """Run an expanding or rolling walk-forward out-of-sample portfolio simulation."""
    assets = list(returns.columns)
    n_assets = len(assets)
    n_obs = len(returns)

    if n_obs < estimation_window + rebalance_frequency:
        raise ValueError(
            f"Insufficient observations ({n_obs}) for estimation window ({estimation_window}) "
            f"+ rebalance frequency ({rebalance_frequency})"
        )

    ppy = float(periods_per_year)
    cost_factor = transaction_cost_bps * 1e-4

    rebalance_indices = list(range(estimation_window, n_obs, rebalance_frequency))
    rebalance_dates: list[str] = []
    oos_returns: list[float] = []
    turnovers: list[float] = []

    prior_w = np.zeros(n_assets, dtype=float)

    for idx_pos, t_idx in enumerate(rebalance_indices):
        # 1. Estimation slice: strictly historical [t - W, t)
        hist_slice = returns.iloc[t_idx - estimation_window : t_idx]
        cov_slice = hist_slice.cov().to_numpy(dtype=float)
        date_str = str(returns.index[t_idx])
        rebalance_dates.append(date_str)

        # 2. Allocation Decision
        if method == "equal_weight":
            w = np.full(n_assets, 1.0 / n_assets, dtype=float)
        elif method == "erc":
            erc_res = solve_equal_risk_contribution(cov_slice, assets=assets)
            w = np.array([erc_res.weights[a] for a in assets], dtype=float)
        elif method == "min_variance":
            from start.tests.portfolio import solve_min_variance

            w_sol, _ = solve_min_variance(np.zeros(n_assets), cov_slice, constraints=None)
            w = w_sol if w_sol is not None else np.full(n_assets, 1.0 / n_assets)
        else:  # default 'hrp'
            w_series, _ = hrp_weights_and_tree(cov_slice, assets=assets)
            w = w_series.reindex(assets).to_numpy(dtype=float)

        # 3. Turnover and Cost
        if idx_pos > 0:
            one_way_turnover = float(0.5 * np.sum(np.abs(w - prior_w)))
            turnover_cost = one_way_turnover * cost_factor
        else:
            one_way_turnover = 0.0
            turnover_cost = 0.0
        turnovers.append(one_way_turnover)
        prior_w = w.copy()

        # 4. Out-of-Sample Evaluation Slice: [t, min(t + F, n_obs))
        next_idx = min(t_idx + rebalance_frequency, n_obs)
        oos_slice = returns.iloc[t_idx:next_idx]
        oos_period_returns = (oos_slice.to_numpy(dtype=float) @ w).tolist()

        # Apply transaction drag on first day of new allocation
        if oos_period_returns:
            oos_period_returns[0] -= turnover_cost

        oos_returns.extend(oos_period_returns)

    # Performance Statistics
    r_arr = np.asarray(oos_returns, dtype=float)
    wealth = np.cumprod(1.0 + r_arr)
    cum_ret = (wealth - 1.0).tolist()

    n_oos = len(r_arr)
    geo_mean = float(np.prod(1.0 + r_arr) ** (ppy / max(n_oos, 1)) - 1.0) if n_oos > 0 else 0.0
    realized_vol = float(np.std(r_arr, ddof=1)) * math.sqrt(ppy) if n_oos > 1 else 0.0

    excess_ret = r_arr - rf_periodic
    excess_sd = float(np.std(excess_ret, ddof=1))
    realized_sharpe = float(np.mean(excess_ret) / excess_sd * math.sqrt(ppy)) if excess_sd > 1e-12 else None

    # Max Drawdown
    running_peak = np.maximum.accumulate(wealth)
    drawdown = (wealth - running_peak) / running_peak
    mdd = float(np.min(drawdown)) if len(drawdown) else 0.0

    mean_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    return WalkForwardResult(
        method=method,
        rebalance_dates=tuple(rebalance_dates),
        out_of_sample_returns=[round(float(x), 8) for x in r_arr],
        cumulative_returns=[round(float(x), 8) for x in cum_ret],
        annualised_return=round(geo_mean, 8),
        annualised_volatility=round(realized_vol, 8),
        realized_sharpe=round(realized_sharpe, 6) if realized_sharpe is not None else None,
        max_drawdown=round(mdd, 8),
        mean_one_way_turnover=round(mean_turnover, 6),
        transaction_cost_bps=transaction_cost_bps,
    )
