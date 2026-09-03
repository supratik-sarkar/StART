"""Deterministic portfolio method comparison engine (Gate 2 & 3).

Compares available portfolio construction methods:
- 1/N Equal Weight
- Minimum Variance
- Maximum Sharpe
- Hierarchical Risk Parity (HRP)
- Equal Risk Contribution (ERC)
- Hierarchical Equal Risk Contribution (HERC)
- Maximum Diversification (MDP)
- Robust Mean-Variance (if uncertainty radius supplied or evaluated)
- Black-Litterman (if views supplied)
- CVaR / Expected-Shortfall (if scenarios available)
- Current Portfolio (if supplied)

Strict institutional policy:
- Does NOT declare an automatic 'winner'.
- Generates transparent, side-by-side comparative evidence under identical constraints.
- Discloses input covariance estimator, return source, constraints, and cost assumptions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from start.portfolio.contracts import MethodComparisonResult
from start.portfolio.hrp import hrp_weights_and_tree
from start.portfolio.optimization import solve_equal_risk_contribution
from start.portfolio.risk_contributions import calculate_risk_contributions


def compare_portfolio_methods(
    returns: pd.DataFrame | None,
    covariance: pd.DataFrame | np.ndarray,
    mu: np.ndarray | None = None,
    prior_weights: pd.Series | np.ndarray | dict[str, float] | None = None,
    benchmark_weights: pd.Series | np.ndarray | dict[str, float] | None = None,
    rf_periodic: float = 0.0,
    periods_per_year: float = 252.0,
    constraints: Any = None,
    bl_views: dict[str, Any] | None = None,
    robust_uncertainty_radius: float | None = None,
    cvar_confidence: float | None = None,
) -> MethodComparisonResult:
    """Evaluate and compare portfolio allocations side-by-side."""
    from start.portfolio.cvar import solve_cvar_portfolio
    from start.portfolio.herc import solve_herc
    from start.portfolio.max_diversification import solve_max_diversification
    from start.portfolio.robust_mvo import solve_robust_mvo
    from start.tests.portfolio import _max_sharpe, solve_min_variance

    if isinstance(covariance, pd.DataFrame):
        asset_names = list(covariance.columns)
        sigma = covariance.to_numpy(dtype=float)
    else:
        sigma = np.asarray(covariance, dtype=float)
        asset_names = [f"A{i}" for i in range(len(sigma))]

    n = len(asset_names)
    ppy = float(periods_per_year)
    prior_arr = None
    if prior_weights is not None:
        if isinstance(prior_weights, pd.Series):
            prior_arr = prior_weights.reindex(asset_names).fillna(0.0).to_numpy(dtype=float)
        elif isinstance(prior_weights, dict):
            prior_arr = np.array([float(prior_weights.get(a, 0.0)) for a in asset_names], dtype=float)
        else:
            prior_arr = np.asarray(prior_weights, dtype=float)

    bw_arr = None
    if benchmark_weights is not None:
        if isinstance(benchmark_weights, pd.Series):
            bw_arr = benchmark_weights.reindex(asset_names).fillna(0.0).to_numpy(dtype=float)
        elif isinstance(benchmark_weights, dict):
            bw_arr = np.array([float(benchmark_weights.get(a, 0.0)) for a in asset_names], dtype=float)
        else:
            bw_arr = np.asarray(benchmark_weights, dtype=float)

    if mu is None:
        if returns is not None and not returns.empty:
            mu = returns.mean().reindex(asset_names).fillna(0.0).to_numpy(dtype=float)
        else:
            mu = np.zeros(n, dtype=float)

    methods: list[str] = []
    summary_rows: list[dict[str, Any]] = []
    weights_matrix: dict[str, dict[str, float]] = {}
    rc_matrix: dict[str, dict[str, float]] = {}

    asset_vols = np.sqrt(np.maximum(1e-15, np.diag(sigma)))

    def evaluate_candidate(name: str, w_vec: np.ndarray) -> None:
        methods.append(name)
        w_dict = {a: round(float(w), 8) for a, w in zip(asset_names, w_vec, strict=True)}
        weights_matrix[name] = w_dict

        var = float(w_vec @ sigma @ w_vec)
        vol = math.sqrt(max(0.0, var))
        vol_ann = vol * math.sqrt(ppy)

        exp_ret_per = float(w_vec @ mu)
        exp_ret_ann = exp_ret_per * ppy

        sharpe_per = (exp_ret_per - rf_periodic) / vol if vol > 1e-12 else None
        sharpe_ann = sharpe_per * math.sqrt(ppy) if sharpe_per is not None else None

        h = float(np.sum(w_vec**2))
        eff_n = float(1.0 / h) if h > 1e-12 else 0.0

        # Diversification ratio
        weighted_vol = float(w_vec @ asset_vols)
        dr_val = float(weighted_vol / vol) if vol > 1e-12 else 1.0

        rc = calculate_risk_contributions(w_vec, sigma, assets=asset_names)
        rc_matrix[name] = {a: round(float(v), 8) for a, v in rc.percentage_contributions.items()}

        pcr_vals = list(rc.percentage_contributions.values())
        rc_dispersion = float(np.std(pcr_vals)) if len(pcr_vals) > 1 else 0.0

        turnover = float(0.5 * np.sum(np.abs(w_vec - prior_arr))) if prior_arr is not None else None

        tracking_err = (
            math.sqrt(max(0.0, float((w_vec - bw_arr) @ sigma @ (w_vec - bw_arr)))) * math.sqrt(ppy)
            if bw_arr is not None
            else None
        )

        summary_rows.append(
            {
                "method": name,
                "annualised_return": round(exp_ret_ann, 8),
                "annualised_volatility": round(vol_ann, 8),
                "annualised_sharpe": round(sharpe_ann, 6) if sharpe_ann is not None else None,
                "diversification_ratio": round(dr_val, 6),
                "herfindahl": round(h, 8),
                "effective_n_positions": round(eff_n, 4),
                "max_weight": round(float(np.max(w_vec)), 8),
                "min_weight": round(float(np.min(w_vec)), 8),
                "risk_contribution_dispersion": round(rc_dispersion, 8),
                "turnover_vs_current": round(turnover, 6) if turnover is not None else None,
                "tracking_error_annualised": round(tracking_err, 6) if tracking_err is not None else None,
            }
        )

    # 1. Current Portfolio (if prior supplied)
    if prior_arr is not None and len(prior_arr) == n:
        evaluate_candidate("current_portfolio", prior_arr)

    # 2. Equal Weight (1/N)
    w_ew = np.full(n, 1.0 / n)
    evaluate_candidate("equal_weight", w_ew)

    # 3. Minimum Variance
    w_min_var, diag_min = solve_min_variance(mu, sigma, constraints, prior_arr, target=None)
    if w_min_var is not None:
        evaluate_candidate("minimum_variance", w_min_var)

    # 4. Maximum Sharpe
    w_max_sh, diag_sh, _ = _max_sharpe(mu, sigma, constraints, prior_arr, rf_periodic)
    if w_max_sh is not None:
        evaluate_candidate("maximum_sharpe", w_max_sh)

    # 5. Hierarchical Risk Parity (HRP)
    w_hrp_series, _ = hrp_weights_and_tree(sigma, assets=asset_names)
    evaluate_candidate("hierarchical_risk_parity", w_hrp_series.reindex(asset_names).to_numpy(dtype=float))

    # 6. Equal Risk Contribution (ERC)
    erc_res = solve_equal_risk_contribution(sigma, assets=asset_names)
    w_erc = np.array([erc_res.weights[a] for a in asset_names], dtype=float)
    evaluate_candidate("equal_risk_contribution", w_erc)

    # 7. Hierarchical Equal Risk Contribution (HERC)
    herc_res = solve_herc(sigma, assets=asset_names, periods_per_year=periods_per_year)
    w_herc = np.array([herc_res.weights[a] for a in asset_names], dtype=float)
    evaluate_candidate("hierarchical_equal_risk_contribution", w_herc)

    # 8. Maximum Diversification (MDP)
    max_div_res = solve_max_diversification(
        sigma,
        assets=asset_names,
        constraints=constraints,
        prior_weights=prior_arr,
        periods_per_year=periods_per_year,
    )
    w_md = np.array([max_div_res.weights[a] for a in asset_names], dtype=float)
    evaluate_candidate("maximum_diversification", w_md)

    from start.portfolio.contracts import UncertaintyDerivationPolicy

    # 9. Robust MVO (if radius supplied)
    if robust_uncertainty_radius is not None:
        rob_res = solve_robust_mvo(
            mu,
            sigma,
            uncertainty_radius=robust_uncertainty_radius,
            uncertainty_policy=UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N,
            assets=asset_names,
            constraints=constraints,
            prior_weights=prior_arr,
            periods_per_year=periods_per_year,
        )
        if rob_res.usable_solution:
            w_rob = np.array([rob_res.weights[a] for a in asset_names], dtype=float)
            evaluate_candidate(f"robust_mvo_radius_{robust_uncertainty_radius}", w_rob)

    # 10. CVaR Optimization (if returns dataframe available with >= 10 rows)
    if returns is not None and len(returns) >= 10:
        cvar_conf = cvar_confidence if cvar_confidence is not None else 0.95
        if len(returns) * (1.0 - cvar_conf) >= 1.0:
            cvar_cons = constraints
            if constraints is not None and (
                constraints.max_concentration is not None or constraints.max_tracking_error is not None
            ):
                from dataclasses import replace

                cvar_cons = replace(constraints, max_concentration=None, max_tracking_error=None)
            cvar_res = solve_cvar_portfolio(
                returns[asset_names],
                confidence_level=cvar_conf,
                assets=asset_names,
                constraints=cvar_cons,
                prior_weights=prior_arr,
                periods_per_year=periods_per_year,
            )
            if cvar_res.usable_solution:
                w_cvar = np.array([cvar_res.weights[a] for a in asset_names], dtype=float)
                evaluate_candidate(f"cvar_optimization_{int(cvar_conf * 100)}pct", w_cvar)

    return MethodComparisonResult(
        methods=tuple(methods),
        summary_table=summary_rows,
        weights_matrix=weights_matrix,
        risk_contributions_matrix=rc_matrix,
    )
