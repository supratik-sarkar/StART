"""Deterministic Rockafellar-Uryasev CVaR / Expected-Shortfall Optimization Engine.

Core invariants:
- Convex Linear Programming formulation (Rockafellar & Uryasev, 2000).
- Decision variables: weights w in R^N, auxiliary VaR variable alpha in R, tail loss slacks u in R^S_+.
- Solved via high-precision HiGHS LP solver.
- Never substitutes parametric Gaussian assumptions into empirical CVaR optimization.
- Validates tail sample support: S * (1 - beta) >= 1.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from start.portfolio.constraints import verify_portfolio_constraints
from start.portfolio.contracts import (
    CVaROptimizationResult,
    DeterminismTier,
    MethodApplicability,
)
from start.registry.market_contexts import PortfolioConstraints

DEFAULT_CONFIDENCE = 0.95

CVAR_APPLICABILITY = MethodApplicability(
    method_name="cvar_optimization",
    required_inputs=("scenario_returns",),
    min_assets=2,
    min_observations=10,
    requires_psd_covariance=False,
    supports_bounds=True,
    supports_group_constraints=True,
    supports_turnover_constraints=True,
    determinism=DeterminismTier.EXACT_DETERMINISTIC,
    assumptions=(
        "Rockafellar-Uryasev (2000) Linear Programming formulation of CVaR",
        "Empirical scenario distribution: loss L_s(w) = - w' r_s",
        "Confidence level beta in (0, 1), requiring tail support S * (1 - beta) >= 1",
    ),
)


def solve_cvar_portfolio(
    scenario_returns: pd.DataFrame | np.ndarray,
    confidence_level: float = DEFAULT_CONFIDENCE,
    target_return: float | None = None,
    assets: list[str] | tuple[str, ...] | None = None,
    constraints: PortfolioConstraints | None = None,
    prior_weights: np.ndarray | pd.Series | dict[str, float] | None = None,
    periods_per_year: float = 252.0,
) -> CVaROptimizationResult:
    """Solve the minimum CVaR portfolio allocation via linear programming."""
    if isinstance(scenario_returns, pd.DataFrame):
        asset_names = list(scenario_returns.columns)
        r_mat = scenario_returns.to_numpy(dtype=float)
    else:
        r_mat = np.asarray(scenario_returns, dtype=float)
        asset_names = list(assets) if assets is not None else [f"A{i}" for i in range(r_mat.shape[1])]

    s, n = r_mat.shape
    if n < 2:
        raise ValueError(f"Asset universe must contain at least 2 assets, got {n}")
    if s < 2:
        raise ValueError(f"Scenario count ({s}) is insufficient for CVaR optimization")

    beta = float(confidence_level)
    if not (0.0 < beta < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {beta}")

    tail_support = s * (1.0 - beta)
    if tail_support < 1.0:
        raise ValueError(
            f"Insufficient scenario tail support: S={s} at confidence {beta} gives "
            f"{tail_support:.2f} tail scenarios (< 1.0). Minimum scenarios required: {math.ceil(1.0 / (1.0 - beta))}"
        )

    if np.any(np.isnan(r_mat)) or np.any(np.isinf(r_mat)):
        raise ValueError("scenario_returns matrix contains NaN or Inf values")

    # 0. Strict Linear Constraint Enforcement: reject unsupported non-linear constraints
    if constraints is not None:
        if constraints.max_concentration is not None:
            raise ValueError(
                "solve_cvar_portfolio (HiGHS LP) does not support non-linear quadratic constraints (max_concentration Herfindahl cap). Supply linear constraints only."
            )
        if constraints.max_tracking_error is not None:
            raise ValueError(
                "solve_cvar_portfolio (HiGHS LP) does not support non-linear quadratic tracking-error constraints. Supply linear constraints only."
            )

    # State vector: x = [w_1, ..., w_N, alpha, u_1, ..., u_S]
    # Total dimension: N + 1 + S
    c = np.zeros(n + 1 + s, dtype=float)
    c[n] = 1.0  # alpha
    c[n + 1 :] = 1.0 / (s * (1.0 - beta))  # (1 / S(1-beta)) * sum(u_s)

    # Inequality constraints:
    # 1. - r_s' w - alpha - u_s <= 0  for each scenario s
    A_tail = np.zeros((s, n + 1 + s), dtype=float)
    A_tail[:, :n] = -r_mat
    A_tail[:, n] = -1.0
    A_tail[:, n + 1 :] = -np.eye(s)
    b_tail = np.zeros(s, dtype=float)

    A_ub_list = [A_tail]
    b_ub_list = [b_tail]

    # Target expected return constraint (if supplied):
    # - (1/S) sum_s r_s' w <= - target_return
    if target_return is not None:
        mean_ret = np.mean(r_mat, axis=0)
        A_ret = np.zeros((1, n + 1 + s), dtype=float)
        A_ret[0, :n] = -mean_ret
        b_ret = np.array([-float(target_return)], dtype=float)
        A_ub_list.append(A_ret)
        b_ub_list.append(b_ret)

    # Factor constraints (linear):
    if constraints is not None and constraints.factor_constraints is not None:
        factor_spec = constraints.factor_constraints
        factor_spec.validate_asset_coverage(asset_names)
        for f_name in factor_spec.factor_names:
            loadings_k = np.array([float(factor_spec.loadings[a][f_name]) for a in asset_names], dtype=float)
            if f_name in factor_spec.upper_bounds:
                A_f = np.zeros((1, n + 1 + s), dtype=float)
                A_f[0, :n] = loadings_k
                b_f = np.array([float(factor_spec.upper_bounds[f_name])], dtype=float)
                A_ub_list.append(A_f)
                b_ub_list.append(b_f)
            if f_name in factor_spec.lower_bounds:
                A_f = np.zeros((1, n + 1 + s), dtype=float)
                A_f[0, :n] = -loadings_k
                b_f = np.array([-float(factor_spec.lower_bounds[f_name])], dtype=float)
                A_ub_list.append(A_f)
                b_ub_list.append(b_f)

    # Group constraints (linear):
    if constraints is not None and constraints.group_constraints is not None:
        group_spec = constraints.group_constraints
        group_spec.validate_asset_coverage(asset_names)
        for g_label, members in group_spec.memberships.items():
            g_row = np.array([1.0 if a in members else 0.0 for a in asset_names], dtype=float)
            if g_label in group_spec.upper_bounds:
                A_g = np.zeros((1, n + 1 + s), dtype=float)
                A_g[0, :n] = g_row
                b_g = np.array([float(group_spec.upper_bounds[g_label])], dtype=float)
                A_ub_list.append(A_g)
                b_ub_list.append(b_g)
            if g_label in group_spec.lower_bounds:
                A_g = np.zeros((1, n + 1 + s), dtype=float)
                A_g[0, :n] = -g_row
                b_g = np.array([-float(group_spec.lower_bounds[g_label])], dtype=float)
                A_ub_list.append(A_g)
                b_ub_list.append(b_g)

    A_ub = np.vstack(A_ub_list)
    b_ub = np.concatenate(b_ub_list)

    # Equality constraints:
    # sum(w) = budget
    budget_val = constraints.budget if constraints is not None else 1.0
    A_eq = np.zeros((1, n + 1 + s), dtype=float)
    A_eq[0, :n] = 1.0
    b_eq = np.array([budget_val], dtype=float)

    # Variable bounds:
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

    # alpha bound: (-inf, inf)
    bounds.append((None, None))

    # u_s bounds: [0, inf)
    for _ in range(s):
        bounds.append((0.0, None))

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    prior_arr = None
    if prior_weights is not None:
        if isinstance(prior_weights, pd.Series):
            prior_arr = prior_weights.reindex(asset_names).fillna(0.0).to_numpy(dtype=float)
        elif isinstance(prior_weights, dict):
            prior_arr = np.array([float(prior_weights.get(a, 0.0)) for a in asset_names], dtype=float)
        else:
            prior_arr = np.asarray(prior_weights, dtype=float)

    if not res.success:
        w_opt = np.zeros(n, dtype=float)
        alpha_opt = 0.0
        cvar_opt = 0.0
        solver_status = f"FAILED_{res.status}: {res.message}"
        converged = False
        usable_solution = False
        tail_scenarios = 0
        mean_ret_per = 0.0
        mean_ret_ann = 0.0
        eff_n = 0.0
        turnover = None
        ver_res = verify_portfolio_constraints(
            weights=w_opt,
            assets=asset_names,
            constraints=constraints,
            prior_weights=prior_arr,
        )
    else:
        w_opt = res.x[:n]
        alpha_opt = float(res.x[n])
        cvar_opt = float(res.fun)
        solver_status = "OPTIMAL"
        converged = True
        usable_solution = True

        ver_res = verify_portfolio_constraints(
            weights=w_opt,
            assets=asset_names,
            constraints=constraints,
            prior_weights=prior_arr,
        )

        scenario_losses = -(r_mat @ w_opt)
        tail_scenarios = int(np.sum(scenario_losses >= alpha_opt - 1e-8))

        ppy = float(periods_per_year)
        mean_ret_per = float(np.mean(r_mat @ w_opt))
        mean_ret_ann = mean_ret_per * ppy

        h = float(np.sum(w_opt**2))
        eff_n = float(1.0 / h) if h > 1e-12 else 0.0
        turnover = float(0.5 * np.sum(np.abs(w_opt - prior_arr))) if prior_arr is not None else None

    return CVaROptimizationResult(
        weights={a: round(float(w), 8) for a, w in zip(asset_names, w_opt, strict=True)}
        if usable_solution
        else {},
        confidence_level=beta,
        cvar_at_scenario_horizon=round(cvar_opt, 8),
        var_at_scenario_horizon=round(alpha_opt, 8),
        tail_scenario_count=tail_scenarios,
        n_scenarios=s,
        expected_return_periodic=round(mean_ret_per, 8),
        effective_n_positions=round(eff_n, 4),
        constraint_verification=ver_res,
        converged=converged,
        usable_solution=usable_solution,
        solver_status=solver_status,
        solver_message=res.message if not res.success else "",
        scenario_horizon="1_PERIOD",
        cvar_periodic=round(cvar_opt, 8),
        cvar_annualised=None,
        var_auxiliary_periodic=round(alpha_opt, 8),
        var_auxiliary_annualised=None,
        expected_return_annualised=round(mean_ret_ann, 8),
        turnover_vs_prior=round(turnover, 6) if (usable_solution and turnover is not None) else None,
    )
