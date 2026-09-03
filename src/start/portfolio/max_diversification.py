"""Deterministic Maximum Diversification Portfolio (MDP) Engine (Choueifaty & Coignard, 2008).

Core invariants:
- Objective: Maximize Diversification Ratio DR(w) = (w' sigma_asset) / sqrt(w' Sigma w).
- Quadratic convex variable transformation / SLSQP optimization.
- Post-solve constraint verification.
- Zero automatic superiority claims: reports DR(w) and concentration metrics objectively.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from start.portfolio.constraints import verify_portfolio_constraints
from start.portfolio.contracts import (
    DeterminismTier,
    MaxDiversificationResult,
    MethodApplicability,
)
from start.registry.market_contexts import PortfolioConstraints

MAX_DIV_APPLICABILITY = MethodApplicability(
    method_name="maximum_diversification",
    required_inputs=("covariance",),
    min_assets=2,
    min_observations=2,
    requires_psd_covariance=True,
    supports_bounds=True,
    supports_group_constraints=True,
    supports_turnover_constraints=True,
    determinism=DeterminismTier.NUMERICALLY_DETERMINISTIC,
    assumptions=(
        "Choueifaty & Coignard (2008) Diversification Ratio maximization",
        "Asset volatilities sigma_i = sqrt(Sigma_ii)",
        "DR(w) = (sum w_i * sigma_i) / sqrt(w' Sigma w)",
    ),
)


def solve_max_diversification(
    covariance: pd.DataFrame | np.ndarray,
    assets: list[str] | tuple[str, ...] | None = None,
    constraints: PortfolioConstraints | None = None,
    prior_weights: np.ndarray | pd.Series | dict[str, float] | None = None,
    periods_per_year: float = 252.0,
) -> MaxDiversificationResult:
    """Solve the Maximum Diversification portfolio."""
    if isinstance(covariance, pd.DataFrame):
        asset_names = list(covariance.columns)
        sigma = covariance.to_numpy(dtype=float)
    else:
        sigma = np.asarray(covariance, dtype=float)
        asset_names = list(assets) if assets is not None else [f"A{i}" for i in range(len(sigma))]

    n = len(asset_names)
    if n < 2:
        w_dict = {asset_names[0]: 1.0} if n == 1 else {}
        ver_res = verify_portfolio_constraints(w_dict, asset_names, constraints=constraints)
        return MaxDiversificationResult(
            weights=w_dict,
            diversification_ratio=1.0,
            weighted_asset_volatility_annualised=math.sqrt(float(sigma[0, 0])) * math.sqrt(periods_per_year) if n == 1 else 0.0,
            portfolio_volatility_annualised=math.sqrt(float(sigma[0, 0])) * math.sqrt(periods_per_year) if n == 1 else 0.0,
            effective_n_positions=1.0,
            constraint_verification=ver_res,
        )

    asset_vols = np.sqrt(np.maximum(1e-15, np.diag(sigma)))

    # Objective: minimize - DR(w) = - (w' sigma_asset) / sqrt(w' Sigma w)
    def objective(w: np.ndarray) -> float:
        port_var = float(w @ sigma @ w)
        port_vol = math.sqrt(max(1e-15, port_var))
        weighted_vol = float(w @ asset_vols)
        return - float(weighted_vol / port_vol)

    bounds: list[tuple[float | None, float | None]] = []
    for _i, a in enumerate(asset_names):
        lb = 0.0 if (constraints is None or constraints.long_only) else None
        if constraints is not None and constraints.min_weight is not None:
            lb = max(lb or 0.0, constraints.min_weight)
        if constraints is not None and constraints.asset_lower_bounds and a in constraints.asset_lower_bounds:
            lb = max(lb or 0.0, constraints.asset_lower_bounds[a])

        ub = None
        if constraints is not None and constraints.max_weight is not None:
            ub = constraints.max_weight
        if constraints is not None and constraints.asset_upper_bounds and a in constraints.asset_upper_bounds:
            ub = min(ub if ub is not None else 1.0, constraints.asset_upper_bounds[a])
        bounds.append((lb, ub))

    prior_arr = None
    if prior_weights is not None:
        if isinstance(prior_weights, pd.Series):
            prior_arr = prior_weights.reindex(asset_names).fillna(0.0).to_numpy(dtype=float)
        elif isinstance(prior_weights, dict):
            prior_arr = np.array([float(prior_weights.get(a, 0.0)) for a in asset_names], dtype=float)
        else:
            prior_arr = np.asarray(prior_weights, dtype=float)

    from start.portfolio.constraints import build_slsqp_constraints

    cons = build_slsqp_constraints(
        constraints=constraints,
        assets=asset_names,
        prior_weights=prior_arr,
        covariance=sigma,
    )

    init_w = np.full(n, 1.0 / n)
    res = minimize(
        objective,
        init_w,
        method="SLSQP",
        bounds=bounds,
        constraints=cons,
        options={"ftol": 1e-12, "maxiter": 500},
    )

    if not res.success:
        w_opt = np.zeros(n, dtype=float)
        converged = False
        usable_solution = False
        solver_status = f"FAILED: {res.message}"
        solver_message = res.message
        dr_val = 0.0
        weighted_vol_ann = 0.0
        port_vol_ann = 0.0
        eff_n = 0.0
        ver_res = verify_portfolio_constraints(
            weights=w_opt,
            assets=asset_names,
            constraints=constraints,
            covariance=sigma,
            prior_weights=prior_arr,
        )
    else:
        w_opt = res.x
        converged = True
        usable_solution = True
        solver_status = "OPTIMAL"
        solver_message = ""

        ver_res = verify_portfolio_constraints(
            weights=w_opt,
            assets=asset_names,
            constraints=constraints,
            covariance=sigma,
            prior_weights=prior_arr,
        )

        port_var = float(w_opt @ sigma @ w_opt)
        port_vol = math.sqrt(max(1e-15, port_var))
        weighted_vol = float(w_opt @ asset_vols)
        dr_val = float(weighted_vol / port_vol)

        ppy = float(periods_per_year)
        port_vol_ann = port_vol * math.sqrt(ppy)
        weighted_vol_ann = weighted_vol * math.sqrt(ppy)

        h = float(np.sum(w_opt**2))
        eff_n = float(1.0 / h) if h > 1e-12 else 0.0

    return MaxDiversificationResult(
        weights={a: round(float(w_opt[i]), 8) for i, a in enumerate(asset_names)} if usable_solution else {},
        diversification_ratio=round(dr_val, 6),
        weighted_asset_volatility_annualised=round(weighted_vol_ann, 8),
        portfolio_volatility_annualised=round(port_vol_ann, 8),
        effective_n_positions=round(eff_n, 4),
        constraint_verification=ver_res,
        converged=converged,
        usable_solution=usable_solution,
        solver_status=solver_status,
        solver_message=solver_message,
    )
