"""Independent Deterministic Portfolio Constraint Verification Engine.

Core invariants:
- Post-solve verification is deterministic and independent of solver return flags.
- Every constraint evaluation produces a typed ConstraintViolation entry.
- Returns typed ConstraintVerificationResult.
- Numerical tolerances are never confused with governance thresholds.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from start.portfolio.contracts import (
    ConstraintType,
    ConstraintVerificationResult,
    ConstraintViolation,
    FactorConstraintSpec,
    GroupConstraintSpec,
)
from start.registry.market_contexts import PortfolioConstraints

CONSTRAINT_TOLERANCE_DEFAULT = 1e-6


def verify_portfolio_constraints(
    weights: dict[str, float] | pd.Series | np.ndarray,
    assets: list[str] | tuple[str, ...],
    constraints: PortfolioConstraints | None = None,
    covariance: np.ndarray | None = None,
    benchmark_weights: dict[str, float] | pd.Series | np.ndarray | None = None,
    prior_weights: dict[str, float] | pd.Series | np.ndarray | None = None,
    factor_spec: FactorConstraintSpec | None = None,
    group_spec: GroupConstraintSpec | None = None,
    max_tracking_error: float | None = None,
    max_turnover: float | None = None,
    tolerance: float = CONSTRAINT_TOLERANCE_DEFAULT,
) -> ConstraintVerificationResult:
    """Deterministically audit a portfolio weight vector against all active constraints."""
    if isinstance(weights, pd.Series):
        w_dict = {str(k): float(v) for k, v in weights.items()}
        w_vec = np.array([w_dict.get(a, 0.0) for a in assets], dtype=float)
    elif isinstance(weights, dict):
        w_dict = {str(k): float(v) for k, v in weights.items()}
        w_vec = np.array([w_dict.get(a, 0.0) for a in assets], dtype=float)
    else:
        w_vec = np.asarray(weights, dtype=float)
        w_dict = {a: float(w_vec[i]) for i, a in enumerate(assets)}

    violations: list[ConstraintViolation] = []
    max_violation = 0.0

    # 1. Budget (Sum of Weights)
    target_budget = constraints.budget if constraints is not None else 1.0
    actual_sum = float(np.sum(w_vec))
    budget_err = abs(actual_sum - target_budget)
    budget_status = "SATISFIED" if budget_err <= tolerance else "VIOLATED"
    if budget_err > max_violation:
        max_violation = budget_err
    violations.append(
        ConstraintViolation(
            constraint=ConstraintType.BUDGET,
            observed_value=round(actual_sum, 10),
            required_bound=round(target_budget, 10),
            violation=round(budget_err, 10),
            tolerance=tolerance,
            provenance="model_constraint_budget",
            status=budget_status,
        )
    )

    # 2. Long-Only
    if constraints is None or constraints.long_only:
        min_w = float(np.min(w_vec)) if len(w_vec) else 0.0
        neg_err = float(max(0.0, -min_w))
        lo_status = "SATISFIED" if neg_err <= tolerance else "VIOLATED"
        if neg_err > max_violation:
            max_violation = neg_err
        violations.append(
            ConstraintViolation(
                constraint=ConstraintType.LONG_ONLY,
                observed_value=round(min_w, 10),
                required_bound=0.0,
                violation=round(neg_err, 10),
                tolerance=tolerance,
                provenance="model_constraint_long_only",
                status=lo_status,
            )
        )

    # 3. Global Min Weight
    if constraints is not None and constraints.min_weight is not None:
        min_w = float(np.min(w_vec)) if len(w_vec) else 0.0
        floor_err = float(max(0.0, constraints.min_weight - min_w))
        st = "SATISFIED" if floor_err <= tolerance else "VIOLATED"
        if floor_err > max_violation:
            max_violation = floor_err
        violations.append(
            ConstraintViolation(
                constraint=ConstraintType.MIN_WEIGHT,
                observed_value=round(min_w, 10),
                required_bound=round(constraints.min_weight, 10),
                violation=round(floor_err, 10),
                tolerance=tolerance,
                provenance="model_constraint_global_min_weight",
                status=st,
            )
        )

    # 4. Global Max Weight
    if constraints is not None and constraints.max_weight is not None:
        max_w = float(np.max(w_vec)) if len(w_vec) else 0.0
        cap_err = float(max(0.0, max_w - constraints.max_weight))
        st = "SATISFIED" if cap_err <= tolerance else "VIOLATED"
        if cap_err > max_violation:
            max_violation = cap_err
        violations.append(
            ConstraintViolation(
                constraint=ConstraintType.MAX_WEIGHT,
                observed_value=round(max_w, 10),
                required_bound=round(constraints.max_weight, 10),
                violation=round(cap_err, 10),
                tolerance=tolerance,
                provenance="model_constraint_global_max_weight",
                status=st,
            )
        )

    # 5. Per-Asset Lower Bounds
    asset_lbs = constraints.asset_lower_bounds if constraints is not None else None
    if asset_lbs:
        for a, lb in asset_lbs.items():
            w_i = w_dict.get(a, 0.0)
            err = float(max(0.0, lb - w_i))
            st = "SATISFIED" if err <= tolerance else "VIOLATED"
            if err > max_violation:
                max_violation = err
            violations.append(
                ConstraintViolation(
                    constraint=f"asset_lower_bound.{a}",
                    observed_value=round(w_i, 10),
                    required_bound=round(float(lb), 10),
                    violation=round(err, 10),
                    tolerance=tolerance,
                    provenance=f"asset_lower_bound_policy.{a}",
                    status=st,
                )
            )

    # 6. Per-Asset Upper Bounds
    asset_ubs = constraints.asset_upper_bounds if constraints is not None else None
    if asset_ubs:
        for a, ub in asset_ubs.items():
            w_i = w_dict.get(a, 0.0)
            err = float(max(0.0, w_i - ub))
            st = "SATISFIED" if err <= tolerance else "VIOLATED"
            if err > max_violation:
                max_violation = err
            violations.append(
                ConstraintViolation(
                    constraint=f"asset_upper_bound.{a}",
                    observed_value=round(w_i, 10),
                    required_bound=round(float(ub), 10),
                    violation=round(err, 10),
                    tolerance=tolerance,
                    provenance=f"asset_upper_bound_policy.{a}",
                    status=st,
                )
            )

    # 7. Gross Leverage
    if constraints is not None and constraints.max_leverage is not None:
        gross_lev = float(np.sum(np.abs(w_vec)))
        lev_err = float(max(0.0, gross_lev - constraints.max_leverage))
        st = "SATISFIED" if lev_err <= tolerance else "VIOLATED"
        if lev_err > max_violation:
            max_violation = lev_err
        violations.append(
            ConstraintViolation(
                constraint=ConstraintType.GROSS_LEVERAGE,
                observed_value=round(gross_lev, 10),
                required_bound=round(constraints.max_leverage, 10),
                violation=round(lev_err, 10),
                tolerance=tolerance,
                provenance="model_constraint_gross_leverage",
                status=st,
            )
        )

    # 8. Concentration (Herfindahl)
    if constraints is not None and constraints.max_concentration is not None:
        herf = float(np.sum(w_vec**2))
        herf_err = float(max(0.0, herf - constraints.max_concentration))
        st = "SATISFIED" if herf_err <= tolerance else "VIOLATED"
        if herf_err > max_violation:
            max_violation = herf_err
        violations.append(
            ConstraintViolation(
                constraint=ConstraintType.MAX_CONCENTRATION,
                observed_value=round(herf, 10),
                required_bound=round(constraints.max_concentration, 10),
                violation=round(herf_err, 10),
                tolerance=tolerance,
                provenance="model_constraint_max_concentration",
                status=st,
            )
        )

    # 9. Turnover
    eff_max_turnover = (
        max_turnover
        if max_turnover is not None
        else (constraints.max_turnover if constraints is not None else None)
    )
    if eff_max_turnover is not None and prior_weights is not None:
        if isinstance(prior_weights, pd.Series):
            pw_dict = {str(k): float(v) for k, v in prior_weights.items()}
        elif isinstance(prior_weights, dict):
            pw_dict = {str(k): float(v) for k, v in prior_weights.items()}
        else:
            pw_arr = np.asarray(prior_weights, dtype=float)
            pw_dict = {a: float(pw_arr[i]) for i, a in enumerate(assets)}
        pw_vec = np.array([pw_dict.get(a, 0.0) for a in assets], dtype=float)
        turnover = float(0.5 * np.sum(np.abs(w_vec - pw_vec)))
        to_err = float(max(0.0, turnover - eff_max_turnover))
        st = "SATISFIED" if to_err <= tolerance else "VIOLATED"
        if to_err > max_violation:
            max_violation = to_err
        violations.append(
            ConstraintViolation(
                constraint=ConstraintType.MAX_TURNOVER,
                observed_value=round(turnover, 10),
                required_bound=round(eff_max_turnover, 10),
                violation=round(to_err, 10),
                tolerance=tolerance,
                provenance="model_constraint_max_turnover",
                status=st,
            )
        )

    # 10. Tracking Error
    eff_max_te = (
        max_tracking_error
        if max_tracking_error is not None
        else (constraints.max_tracking_error if constraints is not None else None)
    )
    if eff_max_te is not None and benchmark_weights is not None and covariance is not None:
        if isinstance(benchmark_weights, pd.Series):
            bw_dict = {str(k): float(v) for k, v in benchmark_weights.items()}
        elif isinstance(benchmark_weights, dict):
            bw_dict = {str(k): float(v) for k, v in benchmark_weights.items()}
        else:
            bw_arr = np.asarray(benchmark_weights, dtype=float)
            bw_dict = {a: float(bw_arr[i]) for i, a in enumerate(assets)}
        bw_vec = np.array([bw_dict.get(a, 0.0) for a in assets], dtype=float)
        active_w = w_vec - bw_vec
        te = math.sqrt(max(0.0, float(active_w @ covariance @ active_w)))
        te_err = float(max(0.0, te - eff_max_te))
        st = "SATISFIED" if te_err <= tolerance else "VIOLATED"
        if te_err > max_violation:
            max_violation = te_err
        violations.append(
            ConstraintViolation(
                constraint=ConstraintType.TRACKING_ERROR,
                observed_value=round(te, 10),
                required_bound=round(eff_max_te, 10),
                violation=round(te_err, 10),
                tolerance=tolerance,
                provenance="model_constraint_max_tracking_error",
                status=st,
            )
        )

    # 11. Factor Constraints
    eff_factor_spec = factor_spec or (constraints.factor_constraints if constraints else None)
    if eff_factor_spec is not None and isinstance(eff_factor_spec, FactorConstraintSpec):
        eff_factor_spec.validate_asset_coverage(assets)
        for f_name in eff_factor_spec.factor_names:
            # Check exposure = sum_i X_ik * w_i
            loadings_k = np.array([
                float(eff_factor_spec.loadings[a][f_name]) for a in assets
            ], dtype=float)
            exp_val = float(loadings_k @ w_vec)
            f_lb = eff_factor_spec.lower_bounds.get(f_name)
            f_ub = eff_factor_spec.upper_bounds.get(f_name)
            err = 0.0
            if f_lb is not None and exp_val < f_lb - tolerance:
                err = max(err, f_lb - exp_val)
            if f_ub is not None and exp_val > f_ub + tolerance:
                err = max(err, exp_val - f_ub)
            st = "SATISFIED" if err <= tolerance else "VIOLATED"
            if err > max_violation:
                max_violation = err
            violations.append(
                ConstraintViolation(
                    constraint=f"factor_exposure.{f_name}",
                    observed_value=round(exp_val, 8),
                    required_bound=(f_lb if f_lb is not None else -math.inf, f_ub if f_ub is not None else math.inf),
                    violation=round(err, 8),
                    tolerance=tolerance,
                    provenance=f"factor_constraint_policy.{f_name}",
                    status=st,
                )
            )

    # 12. Group Constraints
    eff_group_spec = group_spec or (constraints.group_constraints if constraints else None)
    if eff_group_spec is not None and isinstance(eff_group_spec, GroupConstraintSpec):
        for g_label, members in eff_group_spec.memberships.items():
            g_weight = sum(w_dict.get(a, 0.0) for a in members)
            g_lb = eff_group_spec.lower_bounds.get(g_label)
            g_ub = eff_group_spec.upper_bounds.get(g_label)
            err = 0.0
            if g_lb is not None and g_weight < g_lb - tolerance:
                err = max(err, g_lb - g_weight)
            if g_ub is not None and g_weight > g_ub + tolerance:
                err = max(err, g_weight - g_ub)
            st = "SATISFIED" if err <= tolerance else "VIOLATED"
            if err > max_violation:
                max_violation = err
            violations.append(
                ConstraintViolation(
                    constraint=f"group_exposure.{eff_group_spec.group_name}.{g_label}",
                    observed_value=round(g_weight, 8),
                    required_bound=(g_lb if g_lb is not None else -math.inf, g_ub if g_ub is not None else math.inf),
                    violation=round(err, 8),
                    tolerance=tolerance,
                    provenance=f"group_constraint_policy.{eff_group_spec.group_name}.{g_label}",
                    status=st,
                )
            )

    is_valid = max_violation <= tolerance
    summary = {
        "is_valid": is_valid,
        "max_violation": round(max_violation, 10),
        "total_checks": len(violations),
        "satisfied_checks": sum(1 for v in violations if v.status == "SATISFIED"),
        "violated_checks": sum(1 for v in violations if v.status == "VIOLATED"),
    }

    return ConstraintVerificationResult(
        is_valid=is_valid,
        max_violation=round(max_violation, 10),
        tolerance=tolerance,
        violations=tuple(violations),
        summary=summary,
    )


def build_slsqp_constraints(
    constraints: PortfolioConstraints | None,
    assets: list[str] | tuple[str, ...],
    prior_weights: np.ndarray | None = None,
    benchmark_weights: np.ndarray | None = None,
    covariance: np.ndarray | None = None,
    max_tracking_error: float | None = None,
) -> list[dict[str, Any]]:
    """Construct complete SLSQP constraint dictionaries from PortfolioConstraints specification."""
    cons: list[dict[str, Any]] = []

    # 1. Budget equality
    budget_val = constraints.budget if constraints is not None else 1.0
    cons.append({
        "type": "eq",
        "fun": lambda w, b=budget_val: float(np.sum(w)) - b,
    })

    # 2. Tracking Error
    eff_te = max_tracking_error if max_tracking_error is not None else (constraints.max_tracking_error if constraints is not None else None)
    if eff_te is not None and benchmark_weights is not None and covariance is not None:
        te_cap_sq = float(eff_te) ** 2
        bw_vec = np.asarray(benchmark_weights, dtype=float)
        sigma = np.asarray(covariance, dtype=float)
        cons.append({
            "type": "ineq",
            "fun": lambda w, cap_sq=te_cap_sq, bw=bw_vec, cov=sigma: cap_sq - float((w - bw) @ cov @ (w - bw)),
        })

    if constraints is None:
        return cons

    # 3. Max Concentration (Herfindahl)
    if constraints.max_concentration is not None:
        h_cap = float(constraints.max_concentration)
        cons.append({
            "type": "ineq",
            "fun": lambda w, cap=h_cap: cap - float(np.sum(w**2)),
        })

    # 4. Gross Leverage
    if constraints.max_leverage is not None:
        lev_cap = float(constraints.max_leverage)
        cons.append({
            "type": "ineq",
            "fun": lambda w, cap=lev_cap: cap - float(np.sum(np.abs(w))),
        })

    # 5. Max Turnover
    if constraints.max_turnover is not None and prior_weights is not None:
        to_cap = float(constraints.max_turnover)
        pw_vec = np.asarray(prior_weights, dtype=float)
        cons.append({
            "type": "ineq",
            "fun": lambda w, cap=to_cap, pw=pw_vec: cap - 0.5 * float(np.sum(np.abs(w - pw))),
        })

    # 6. Factor Constraints
    factor_spec = constraints.factor_constraints
    if factor_spec is not None and isinstance(factor_spec, FactorConstraintSpec):
        factor_spec.validate_asset_coverage(assets)
        for f_name in factor_spec.factor_names:
            loadings_k = np.array([
                float(factor_spec.loadings[a][f_name]) for a in assets
            ], dtype=float)
            if f_name in factor_spec.upper_bounds:
                ub = float(factor_spec.upper_bounds[f_name])
                cons.append({
                    "type": "ineq",
                    "fun": lambda w, lk=loadings_k, b=ub: b - float(lk @ w),
                })
            if f_name in factor_spec.lower_bounds:
                lb = float(factor_spec.lower_bounds[f_name])
                cons.append({
                    "type": "ineq",
                    "fun": lambda w, lk=loadings_k, b=lb: float(lk @ w) - b,
                })

    # 7. Group Constraints
    group_spec = constraints.group_constraints
    if group_spec is not None and isinstance(group_spec, GroupConstraintSpec):
        for g_label, members in group_spec.memberships.items():
            idxs = [i for i, a in enumerate(assets) if a in members]
            if not idxs:
                continue
            if g_label in group_spec.upper_bounds:
                ub = float(group_spec.upper_bounds[g_label])
                cons.append({
                    "type": "ineq",
                    "fun": lambda w, ix=idxs, b=ub: b - float(np.sum(w[ix])),
                })
            if g_label in group_spec.lower_bounds:
                lb = float(group_spec.lower_bounds[g_label])
                cons.append({
                    "type": "ineq",
                    "fun": lambda w, ix=idxs, b=lb: float(np.sum(w[ix])) - b,
                })

    return cons
