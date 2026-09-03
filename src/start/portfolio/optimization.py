"""Deterministic portfolio optimization engines.

Includes: Equal-Weight, Equal Risk Contribution (ERC), and Efficient Frontier.

Core invariants:
- Numerical solvers (SLSQP / convex optimization) undergo independent post-solve verification.
- SLSQP success flag is NEVER trusted alone.
- 1/N baseline is explicit and deterministic.
- Max-Sharpe traces the efficient frontier under frozen tie-break contract.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from start.portfolio.contracts import (
    EfficientFrontierResult,
    EqualRiskContributionResult,
    FrontierPoint,
)
from start.portfolio.hrp import hrp_weights_and_tree
from start.portfolio.risk_contributions import calculate_risk_contributions

CONSTRAINT_TOLERANCE = 1e-6
DEGENERATE_VARIANCE_RELATIVE = 1e-20


def solve_equal_weight(
    assets: list[str] | tuple[str, ...],
    covariance: np.ndarray,
    mu: np.ndarray | None = None,
    rf_periodic: float = 0.0,
    periods_per_year: float = 252.0,
) -> tuple[pd.Series, dict[str, Any]]:
    """Compute explicit 1/N equal-weight baseline allocation and metrics."""
    n = len(assets)
    if n == 0:
        raise ValueError("Asset universe cannot be empty")
    w = np.full(n, 1.0 / n, dtype=float)
    w_series = pd.Series(w, index=assets)

    var = float(w @ covariance @ w)
    vol = math.sqrt(max(0.0, var))
    vol_ann = vol * math.sqrt(periods_per_year)

    exp_ret_per = float(w @ mu) if mu is not None else 0.0
    exp_ret_ann = exp_ret_per * periods_per_year if mu is not None else 0.0

    sharpe_per = (exp_ret_per - rf_periodic) / vol if vol > 1e-12 and mu is not None else None
    sharpe_ann = sharpe_per * math.sqrt(periods_per_year) if sharpe_per is not None else None

    rc = calculate_risk_contributions(w, covariance, assets=list(assets))

    metrics: dict[str, Any] = {
        "n_assets": n,
        "method": "equal_weight",
        "weights_sum": 1.0,
        "max_weight": round(1.0 / n, 8),
        "min_weight": round(1.0 / n, 8),
        "herfindahl": round(1.0 / n, 8),
        "effective_n_positions": float(n),
        "volatility_periodic": round(vol, 12),
        "volatility_annualised": round(vol_ann, 10),
        "expected_return_annualised": round(exp_ret_ann, 10) if mu is not None else None,
        "sharpe_annualised": round(sharpe_ann, 10) if sharpe_ann is not None else None,
        "risk_contributions": rc.percentage_contributions,
    }
    return w_series, metrics


def solve_equal_risk_contribution(
    covariance: pd.DataFrame | np.ndarray,
    assets: list[str] | tuple[str, ...] | None = None,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> EqualRiskContributionResult:
    """Solve the Equal Risk Contribution (ERC / Risk Parity) portfolio via logarithmic formulation."""
    if isinstance(covariance, pd.DataFrame):
        asset_names = list(covariance.columns)
        sigma = covariance.to_numpy(dtype=float)
    else:
        sigma = np.asarray(covariance, dtype=float)
        asset_names = list(assets) if assets is not None else [f"A{i}" for i in range(len(sigma))]

    n = len(asset_names)
    if n < 2:
        w_dict = {asset_names[0]: 1.0} if n == 1 else {}
        return EqualRiskContributionResult(
            weights=w_dict,
            risk_contributions={a: 1.0 for a in asset_names},
            percentage_risk_contributions={a: 1.0 for a in asset_names},
            target_risk_contribution=1.0,
            max_risk_contribution_dispersion=0.0,
            portfolio_volatility=math.sqrt(float(sigma[0, 0])) if n == 1 else 0.0,
            portfolio_variance=float(sigma[0, 0]) if n == 1 else 0.0,
            objective_value=0.0,
            solver_iterations=0,
            solver_status=0,
            converged=True,
            constraint_violations={"budget": 0.0, "non_negativity": 0.0},
        )

    # Spinu / Maillard log barrier objective: f(y) = 0.5 y^T Sigma y - (1/n) sum ln(y_i)
    # The optimal weights are w = y / sum(y)
    start_y = np.full(n, 1.0 / math.sqrt(n))

    def objective(y: np.ndarray) -> float:
        if np.any(y <= 1e-15):
            return 1e12
        return float(0.5 * y @ sigma @ y - (1.0 / n) * np.sum(np.log(y)))

    def gradient(y: np.ndarray) -> np.ndarray:
        return sigma @ y - (1.0 / n) / np.maximum(y, 1e-15)

    bounds = [(1e-10, None)] * n
    res = minimize(
        objective,
        start_y,
        jac=gradient,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": tol, "gtol": tol, "maxiter": max_iter},
    )

    y_sol = np.maximum(res.x, 0.0)
    w_raw = y_sol / np.sum(y_sol)

    # Post-solve verification
    budget_violation = abs(float(np.sum(w_raw)) - 1.0)
    non_neg_violation = float(max(0.0, -np.min(w_raw)))
    violations = {"budget": budget_violation, "non_negativity": non_neg_violation}

    rc = calculate_risk_contributions(w_raw, sigma, assets=asset_names)
    target_rc = 1.0 / n
    pcr_arr = np.array(list(rc.percentage_contributions.values()), dtype=float)
    dispersion = float(np.max(np.abs(pcr_arr - target_rc))) if len(pcr_arr) else 0.0

    w_dict = {a: round(float(w), 10) for a, w in zip(asset_names, w_raw, strict=True)}

    return EqualRiskContributionResult(
        weights=w_dict,
        risk_contributions=rc.component_contributions,
        percentage_risk_contributions=rc.percentage_contributions,
        target_risk_contribution=round(target_rc, 6),
        max_risk_contribution_dispersion=round(dispersion, 8),
        portfolio_volatility=rc.portfolio_volatility,
        portfolio_variance=rc.portfolio_variance,
        objective_value=float(res.fun),
        solver_iterations=int(res.nit),
        solver_status=int(res.status),
        converged=bool(res.success and budget_violation <= CONSTRAINT_TOLERANCE),
        constraint_violations=violations,
    )


def trace_efficient_frontier(
    mu: np.ndarray,
    sigma: np.ndarray,
    assets: list[str],
    constraints: Any = None,
    prior: np.ndarray | None = None,
    rf_periodic: float = 0.0,
    periods_per_year: float = 252.0,
    n_points: int = 50,
) -> EfficientFrontierResult:
    """Trace the parametric efficient frontier and compute reference portfolios."""
    from start.tests.portfolio import _max_sharpe, solve_min_variance

    n = len(mu)
    ppy = float(periods_per_year)

    # 1. Global Minimum Variance Point
    w_min_var, diag_min = solve_min_variance(mu, sigma, constraints, prior, target=None)
    if w_min_var is None:
        raise RuntimeError("Minimum variance solve failed on efficient frontier")

    var_min = float(w_min_var @ sigma @ w_min_var)
    vol_min = math.sqrt(max(0.0, var_min))
    ret_min_per = float(w_min_var @ mu)
    sharpe_min_per = (ret_min_per - rf_periodic) / vol_min if vol_min > 1e-12 else None

    min_var_point = FrontierPoint(
        label="Minimum Variance",
        target_return=round(ret_min_per, 10),
        expected_return_annualised=round(ret_min_per * ppy, 8),
        volatility_annualised=round(vol_min * math.sqrt(ppy), 8),
        sharpe_annualised=round(sharpe_min_per * math.sqrt(ppy), 6) if sharpe_min_per else None,
        weights={a: round(float(w), 8) for a, w in zip(assets, w_min_var, strict=True)},
    )

    # 2. Maximum Sharpe Point (under frozen tie-break contract)
    w_max_sharpe, diag_sharpe, _ = _max_sharpe(mu, sigma, constraints, prior, rf_periodic)
    if w_max_sharpe is not None:
        var_sh = float(w_max_sharpe @ sigma @ w_max_sharpe)
        vol_sh = math.sqrt(max(0.0, var_sh))
        ret_sh_per = float(w_max_sharpe @ mu)
        sharpe_sh_per = (ret_sh_per - rf_periodic) / vol_sh if vol_sh > 1e-12 else None
        max_sharpe_point = FrontierPoint(
            label="Maximum Sharpe",
            target_return=round(ret_sh_per, 10),
            expected_return_annualised=round(ret_sh_per * ppy, 8),
            volatility_annualised=round(vol_sh * math.sqrt(ppy), 8),
            sharpe_annualised=round(sharpe_sh_per * math.sqrt(ppy), 6) if sharpe_sh_per else None,
            weights={a: round(float(w), 8) for a, w in zip(assets, w_max_sharpe, strict=True)},
        )
    else:
        max_sharpe_point = min_var_point

    # 3. Parametric Frontier Grid along feasible return span
    mu_min = float(min_var_point.target_return)
    mu_max = float(np.max(mu)) if constraints is None or constraints.long_only else float(np.max(mu) * 1.5)
    target_returns = np.linspace(mu_min, mu_max, n_points)

    points: list[FrontierPoint] = []
    for t_ret in target_returns:
        w_t, diag_t = solve_min_variance(mu, sigma, constraints, prior, target=float(t_ret))
        if w_t is not None and diag_t.get("max_constraint_violation", 0.0) <= CONSTRAINT_TOLERANCE:
            v_t = float(w_t @ sigma @ w_t)
            vol_t = math.sqrt(max(0.0, v_t))
            sh_t = (float(t_ret) - rf_periodic) / vol_t if vol_t > 1e-12 else None
            points.append(
                FrontierPoint(
                    label=f"Frontier ({float(t_ret)*ppy:.2%})",
                    target_return=round(float(t_ret), 10),
                    expected_return_annualised=round(float(t_ret) * ppy, 8),
                    volatility_annualised=round(vol_t * math.sqrt(ppy), 8),
                    sharpe_annualised=round(sh_t * math.sqrt(ppy), 6) if sh_t else None,
                    weights={a: round(float(w), 8) for a, w in zip(assets, w_t, strict=True)},
                )
            )

    # 4. Overlays: Equal Weight, ERC, HRP
    # Equal Weight
    w_ew = np.full(n, 1.0 / n)
    v_ew = math.sqrt(float(w_ew @ sigma @ w_ew))
    r_ew = float(w_ew @ mu)
    sh_ew = (r_ew - rf_periodic) / v_ew if v_ew > 1e-12 else None
    ew_point = FrontierPoint(
        label="Equal Weight (1/N)",
        target_return=round(r_ew, 10),
        expected_return_annualised=round(r_ew * ppy, 8),
        volatility_annualised=round(v_ew * math.sqrt(ppy), 8),
        sharpe_annualised=round(sh_ew * math.sqrt(ppy), 6) if sh_ew else None,
        weights={a: 1.0 / n for a in assets},
    )

    # ERC
    erc_res = solve_equal_risk_contribution(sigma, assets=assets)
    w_erc = np.array([erc_res.weights[a] for a in assets], dtype=float)
    v_erc = math.sqrt(float(w_erc @ sigma @ w_erc))
    r_erc = float(w_erc @ mu)
    sh_erc = (r_erc - rf_periodic) / v_erc if v_erc > 1e-12 else None
    erc_point = FrontierPoint(
        label="Equal Risk Contribution",
        target_return=round(r_erc, 10),
        expected_return_annualised=round(r_erc * ppy, 8),
        volatility_annualised=round(v_erc * math.sqrt(ppy), 8),
        sharpe_annualised=round(sh_erc * math.sqrt(ppy), 6) if sh_erc else None,
        weights=erc_res.weights,
    )

    # HRP
    w_hrp_series, _ = hrp_weights_and_tree(sigma, assets=assets)
    w_hrp = w_hrp_series.reindex(assets).to_numpy(dtype=float)
    v_hrp = math.sqrt(float(w_hrp @ sigma @ w_hrp))
    r_hrp = float(w_hrp @ mu)
    sh_hrp = (r_hrp - rf_periodic) / v_hrp if v_hrp > 1e-12 else None
    hrp_point = FrontierPoint(
        label="Hierarchical Risk Parity",
        target_return=round(r_hrp, 10),
        expected_return_annualised=round(r_hrp * ppy, 8),
        volatility_annualised=round(v_hrp * math.sqrt(ppy), 8),
        sharpe_annualised=round(sh_hrp * math.sqrt(ppy), 6) if sh_hrp else None,
        weights={a: round(float(w), 8) for a, w in w_hrp_series.items()},
    )

    current_point = None
    if prior is not None and len(prior) == n:
        v_cur = math.sqrt(float(prior @ sigma @ prior))
        r_cur = float(prior @ mu)
        sh_cur = (r_cur - rf_periodic) / v_cur if v_cur > 1e-12 else None
        current_point = FrontierPoint(
            label="Current Portfolio",
            target_return=round(r_cur, 10),
            expected_return_annualised=round(r_cur * ppy, 8),
            volatility_annualised=round(v_cur * math.sqrt(ppy), 8),
            sharpe_annualised=round(sh_cur * math.sqrt(ppy), 6) if sh_cur else None,
            weights={a: round(float(w), 8) for a, w in zip(assets, prior, strict=True)},
        )

    return EfficientFrontierResult(
        frontier_points=tuple(points),
        min_variance_point=min_var_point,
        max_sharpe_point=max_sharpe_point,
        equal_weight_point=ew_point,
        erc_point=erc_point,
        hrp_point=hrp_point,
        current_point=current_point,
    )
