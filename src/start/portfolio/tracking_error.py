"""Deterministic Tracking-Error Constrained Portfolio Optimizer.

Core invariants:
- Benchmark weights w_b are mandatory. If missing: fail closed.
- Tracking error: TE(w) = sqrt((w - w_b)' Sigma (w - w_b)).
- Enforces hard tracking-error upper bound constraint.
- Post-solve constraint verification.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from start.portfolio.constraints import verify_portfolio_constraints
from start.portfolio.contracts import (
    DeterminismTier,
    MethodApplicability,
    TrackingErrorResult,
)
from start.registry.market_contexts import PortfolioConstraints

TRACKING_ERROR_APPLICABILITY = MethodApplicability(
    method_name="tracking_error_constrained_optimization",
    required_inputs=("mu", "covariance", "benchmark_weights", "max_tracking_error"),
    min_assets=2,
    min_observations=2,
    requires_psd_covariance=True,
    supports_bounds=True,
    supports_group_constraints=True,
    supports_turnover_constraints=True,
    determinism=DeterminismTier.NUMERICALLY_DETERMINISTIC,
    assumptions=(
        "Explicit benchmark portfolio weights w_b required",
        "Tracking error TE(w) = sqrt((w - w_b)' Sigma (w - w_b)) <= TE_max",
        "Active return: (w - w_b)' mu",
    ),
)


def solve_tracking_error_constrained(
    mu: np.ndarray | pd.Series | list[float],
    covariance: np.ndarray | pd.DataFrame,
    benchmark_weights: pd.Series | dict[str, float] | np.ndarray,
    max_tracking_error: float,
    assets: list[str] | tuple[str, ...] | None = None,
    objective_type: str = "max_active_return",
    risk_aversion: float = 1.0,
    constraints: PortfolioConstraints | None = None,
    prior_weights: np.ndarray | pd.Series | dict[str, float] | None = None,
    periods_per_year: float = 252.0,
) -> TrackingErrorResult:
    """Solve benchmark-relative portfolio optimization under a hard tracking-error bound."""
    if isinstance(covariance, pd.DataFrame):
        asset_names = list(covariance.columns)
        sigma = covariance.to_numpy(dtype=float)
    else:
        sigma = np.asarray(covariance, dtype=float)
        asset_names = list(assets) if assets is not None else [f"A{i}" for i in range(len(sigma))]

    n = len(asset_names)
    if isinstance(mu, (pd.Series, dict)):
        missing_mu = [a for a in asset_names if a not in mu]
        if missing_mu:
            raise ValueError(f"Asset(s) {missing_mu} missing from expected returns mu (fail-closed)")
        mu_vec = np.array([float(mu[a]) for a in asset_names], dtype=float)
    else:
        mu_vec = np.asarray(mu, dtype=float)

    if isinstance(benchmark_weights, (pd.Series, dict)):
        missing_bw = [a for a in asset_names if a not in benchmark_weights]
        if missing_bw:
            raise ValueError(f"Asset(s) {missing_bw} missing from benchmark_weights (fail-closed)")
        bw_arr = np.array([float(benchmark_weights[a]) for a in asset_names], dtype=float)
    else:
        bw_arr = np.asarray(benchmark_weights, dtype=float)

    if len(bw_arr) != n:
        raise ValueError(f"Benchmark weights length ({len(bw_arr)}) does not match asset count ({n})")

    te_cap = float(max_tracking_error)
    if te_cap < 0:
        raise ValueError(f"max_tracking_error must be non-negative, got {te_cap}")

    ppy = float(periods_per_year)

    # Objective: maximize active return (w - w_b)' mu (or quadratic trade-off)
    def objective(w: np.ndarray) -> float:
        active_ret = float((w - bw_arr) @ mu_vec)
        if objective_type == "min_active_variance":
            active_w = w - bw_arr
            return float(active_w @ sigma @ active_w)
        return - float(active_ret)

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
        if isinstance(prior_weights, (pd.Series, dict)):
            missing_prior = [a for a in asset_names if a not in prior_weights]
            if missing_prior:
                raise ValueError(f"Asset(s) {missing_prior} missing from prior_weights (fail-closed)")
            prior_arr = np.array([float(prior_weights[a]) for a in asset_names], dtype=float)
        else:
            prior_arr = np.asarray(prior_weights, dtype=float)

    from start.portfolio.constraints import build_slsqp_constraints

    cons = build_slsqp_constraints(
        constraints=constraints,
        assets=asset_names,
        prior_weights=prior_arr,
        benchmark_weights=bw_arr,
        covariance=sigma,
        max_tracking_error=te_cap,
    )

    if te_cap <= 1e-12:
        w_opt = bw_arr.copy()
        converged = True
        usable_solution = True
        solver_status = "OPTIMAL"
        solver_message = ""
    else:
        init_w = bw_arr.copy()
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
        benchmark_weights=bw_arr,
        prior_weights=prior_arr,
        max_tracking_error=te_cap,
    )

    if usable_solution:
        active_w = w_opt - bw_arr
        te_periodic = math.sqrt(max(0.0, float(active_w @ sigma @ active_w)))
        te_annualised = te_periodic * math.sqrt(ppy)

        active_ret_per = float(active_w @ mu_vec)
        active_ret_ann = active_ret_per * ppy

        ir = (active_ret_ann / te_annualised) if te_annualised > 1e-10 else None

        port_var = float(w_opt @ sigma @ w_opt)
        port_vol_ann = math.sqrt(max(0.0, port_var)) * math.sqrt(ppy)
    else:
        active_w = np.zeros(n, dtype=float)
        te_periodic = 0.0
        te_annualised = 0.0
        active_ret_ann = None
        ir = None
        port_vol_ann = 0.0

    return TrackingErrorResult(
        weights={a: round(float(w_opt[i]), 8) for i, a in enumerate(asset_names)} if usable_solution else {},
        benchmark_weights={a: round(float(bw_arr[i]), 8) for i, a in enumerate(asset_names)},
        active_weights={a: round(float(active_w[i]), 8) for i, a in enumerate(asset_names)} if usable_solution else {},
        tracking_error_periodic=round(te_periodic, 8),
        tracking_error_annualised=round(te_annualised, 8),
        active_return_annualised=round(active_ret_ann, 8) if active_ret_ann is not None else None,
        information_ratio=round(ir, 6) if ir is not None else None,
        portfolio_volatility_annualised=round(port_vol_ann, 8),
        constraint_verification=ver_res,
        converged=converged,
        usable_solution=usable_solution,
        solver_status=solver_status,
        solver_message=solver_message,
    )
