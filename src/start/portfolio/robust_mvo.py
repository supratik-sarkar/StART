"""Deterministic Robust Mean-Variance Optimization Engine.

Core invariants:
- Mathematical formulation: Ellipsoidal expected-return uncertainty set.
- Worst-case expected return: R_wc(w) = w' mu_0 - kappa * sqrt(w' Sigma_mu w).
- Uncertainty parameter kappa (uncertainty radius) is an explicit input, never invented.
- Deterministic parameter sensitivity grid over supplied radii.
- Zero subjective LLM parameter selection without policy.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from start.portfolio.constraints import verify_portfolio_constraints
from start.portfolio.contracts import (
    DeterminismTier,
    MethodApplicability,
    MetricHorizon,
    RobustMVOResult,
    RobustSensitivityPoint,
    RobustSensitivityResult,
    UncertaintyDerivationPolicy,
    validate_horizon_alignment,
)
from start.registry.market_contexts import PortfolioConstraints

CONSTRAINT_TOLERANCE = 1e-6

ROBUST_MVO_APPLICABILITY = MethodApplicability(
    method_name="robust_mean_variance",
    required_inputs=("mu", "covariance", "uncertainty_radius"),
    min_assets=2,
    min_observations=2,
    requires_psd_covariance=True,
    supports_bounds=True,
    supports_group_constraints=True,
    supports_turnover_constraints=True,
    determinism=DeterminismTier.NUMERICALLY_DETERMINISTIC,
    assumptions=(
        "Convex ellipsoidal uncertainty set on expected returns",
        "Worst-case return: min_{mu in U} w'mu = w'mu_0 - kappa * sqrt(w' Sigma_mu w)",
        "Uncertainty radius kappa is an explicit input or derived from confidence level",
    ),
)


def solve_robust_mvo(
    mu: np.ndarray | pd.Series | list[float],
    covariance: np.ndarray | pd.DataFrame,
    uncertainty_radius: float,
    assets: list[str] | tuple[str, ...] | None = None,
    uncertainty_cov: np.ndarray | None = None,
    uncertainty_policy: UncertaintyDerivationPolicy | str | None = None,
    n_observations: int | None = None,
    risk_aversion: float = 3.0,
    constraints: PortfolioConstraints | None = None,
    prior_weights: np.ndarray | pd.Series | dict[str, float] | None = None,
    rf_periodic: float = 0.0,
    periods_per_year: float = 252.0,
    mu_horizon: MetricHorizon | str | None = None,
    cov_horizon: MetricHorizon | str | None = None,
) -> RobustMVOResult:
    """Solve convex robust mean-variance optimization under ellipsoidal uncertainty."""
    validate_horizon_alignment(mu_horizon, cov_horizon, periods_per_year=periods_per_year)
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

    if len(mu_vec) != n:
        raise ValueError(f"mu length ({len(mu_vec)}) must match asset count ({n})")

    kappa = float(uncertainty_radius)
    if kappa < 0:
        raise ValueError(f"uncertainty_radius must be non-negative, got {kappa}")

    if kappa == 0.0:
        eff_policy = UncertaintyDerivationPolicy.EXPLICIT_UNCERTAINTY_COV
        sigma_mu = np.zeros((n, n), dtype=float) if uncertainty_cov is None else np.asarray(uncertainty_cov, dtype=float)
    elif uncertainty_cov is None and uncertainty_policy is None:
        raise ValueError(
            "solve_robust_mvo requires either an explicit uncertainty_cov matrix or a named uncertainty_policy "
            "(e.g. SAMPLE_COVARIANCE_DIV_N, IDENTITY_ESTIMATION) (fail-closed)"
        )
    elif uncertainty_cov is not None:
        sigma_mu = np.asarray(uncertainty_cov, dtype=float)
        eff_policy = (
            UncertaintyDerivationPolicy(uncertainty_policy)
            if uncertainty_policy is not None
            else UncertaintyDerivationPolicy.EXPLICIT_UNCERTAINTY_COV
        )
    elif uncertainty_policy == UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N:
        n_obs = n_observations if n_observations is not None else 100
        sigma_mu = sigma / float(n_obs)
        eff_policy = UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N
    elif uncertainty_policy == UncertaintyDerivationPolicy.IDENTITY_ESTIMATION:
        sigma_mu = np.eye(n) * float(np.trace(sigma) / n)
        eff_policy = UncertaintyDerivationPolicy.IDENTITY_ESTIMATION
    else:
        sigma_mu = sigma.copy()
        eff_policy = (
            UncertaintyDerivationPolicy(uncertainty_policy)
            if uncertainty_policy is not None
            else UncertaintyDerivationPolicy.EXPLICIT_UNCERTAINTY_COV
        )

    unc_cov_hash = hashlib.sha256(sigma_mu.tobytes()).hexdigest()[:32]
    delta_val = float(risk_aversion)
    ppy = float(periods_per_year)

    prior_arr = None
    if prior_weights is not None:
        if isinstance(prior_weights, (pd.Series, dict)):
            missing_prior = [a for a in asset_names if a not in prior_weights]
            if missing_prior:
                raise ValueError(f"Asset(s) {missing_prior} missing from prior_weights (fail-closed)")
            prior_arr = np.array([float(prior_weights[a]) for a in asset_names], dtype=float)
        else:
            prior_arr = np.asarray(prior_weights, dtype=float)

    # Objective: maximize w' mu_0 - kappa * sqrt(w' Sigma_mu w) - 0.5 * delta * w' Sigma w
    def objective(w: np.ndarray) -> float:
        pen_var = float(w @ sigma_mu @ w)
        pen_vol = math.sqrt(max(0.0, pen_var))
        robust_ret = float(w @ mu_vec) - kappa * pen_vol
        var = float(w @ sigma @ w)
        return float(- (robust_ret - 0.5 * delta_val * var))

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

    w_candidate = res.x if res.success else np.zeros(n, dtype=float)
    ver_res = verify_portfolio_constraints(
        weights=w_candidate,
        assets=asset_names,
        constraints=constraints,
        covariance=sigma,
        prior_weights=prior_arr,
    )

    if not res.success or not ver_res.is_valid:
        w_opt = np.zeros(n, dtype=float)
        converged = False
        usable_solution = False
        solver_status = f"FAILED: {res.message if not res.success else 'Constraint verification failed'}"
        solver_message = res.message if not res.success else "Constraint verification failed"
        nom_ret_ann = 0.0
        wc_ret_ann = 0.0
        vol_ann = 0.0
        nom_sharpe = None
        wc_sharpe = None
        eff_n = 0.0
        turnover = None
    else:
        w_opt = w_candidate
        converged = True
        usable_solution = True
        solver_status = "OPTIMAL"
        solver_message = ""

        var_opt = float(w_opt @ sigma @ w_opt)
        vol_opt = math.sqrt(max(0.0, var_opt))
        vol_ann = vol_opt * math.sqrt(ppy)

        nom_ret_per = float(w_opt @ mu_vec)
        nom_ret_ann = nom_ret_per * ppy

        unc_var = float(w_opt @ sigma_mu @ w_opt)
        unc_vol = math.sqrt(max(0.0, unc_var))
        wc_ret_per = nom_ret_per - kappa * unc_vol
        wc_ret_ann = wc_ret_per * ppy

        nom_sharpe = (nom_ret_per - rf_periodic) / vol_opt * math.sqrt(ppy) if vol_opt > 1e-12 else None
        wc_sharpe = (wc_ret_per - rf_periodic) / vol_opt * math.sqrt(ppy) if vol_opt > 1e-12 else None

        h = float(np.sum(w_opt**2))
        eff_n = float(1.0 / h) if h > 1e-12 else 0.0

        turnover = (
            float(0.5 * np.sum(np.abs(w_opt - prior_arr))) if prior_arr is not None else None
        )

    return RobustMVOResult(
        weights={a: round(float(w_opt[i]), 8) for i, a in enumerate(asset_names)} if usable_solution else {},
        uncertainty_radius=kappa,
        nominal_expected_return_annualised=round(nom_ret_ann, 8),
        worst_case_expected_return_annualised=round(wc_ret_ann, 8),
        portfolio_volatility_annualised=round(vol_ann, 8),
        nominal_sharpe_annualised=round(nom_sharpe, 6) if nom_sharpe is not None else None,
        worst_case_sharpe_annualised=round(wc_sharpe, 6) if wc_sharpe is not None else None,
        effective_n_positions=round(eff_n, 4),
        turnover_vs_prior=round(turnover, 6) if (usable_solution and turnover is not None) else None,
        constraint_verification=ver_res,
        uncertainty_set_type="ellipsoidal_return",
        uncertainty_policy=eff_policy,
        uncertainty_covariance_fingerprint=unc_cov_hash,
        converged=converged,
        usable_solution=usable_solution,
        solver_status=solver_status,
        solver_message=solver_message,
    )


def robust_mvo_sensitivity_grid(
    mu: np.ndarray | pd.Series | list[float],
    covariance: np.ndarray | pd.DataFrame,
    assets: list[str] | tuple[str, ...],
    radii: list[float] | tuple[float, ...],
    uncertainty_cov: np.ndarray | None = None,
    uncertainty_policy: UncertaintyDerivationPolicy | str | None = None,
    n_observations: int | None = None,
    risk_aversion: float = 3.0,
    constraints: PortfolioConstraints | None = None,
    prior_weights: np.ndarray | pd.Series | dict[str, float] | None = None,
    rf_periodic: float = 0.0,
    periods_per_year: float = 252.0,
) -> RobustSensitivityResult:
    """Evaluate robust MVO across a grid of uncertainty radii."""
    points: list[RobustSensitivityPoint] = []
    baseline_r = float(radii[0]) if len(radii) else 0.0

    eff_policy = uncertainty_policy or (UncertaintyDerivationPolicy.EXPLICIT_UNCERTAINTY_COV if uncertainty_cov is not None else UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N)

    for r in radii:
        res = solve_robust_mvo(
            mu=mu,
            covariance=covariance,
            uncertainty_radius=float(r),
            assets=assets,
            uncertainty_cov=uncertainty_cov,
            uncertainty_policy=eff_policy,
            n_observations=n_observations,
            risk_aversion=risk_aversion,
            constraints=constraints,
            prior_weights=prior_weights,
            rf_periodic=rf_periodic,
            periods_per_year=periods_per_year,
        )
        pt = RobustSensitivityPoint(
            uncertainty_radius=float(r),
            nominal_expected_return_annualised=res.nominal_expected_return_annualised,
            worst_case_expected_return_annualised=res.worst_case_expected_return_annualised,
            portfolio_volatility_annualised=res.portfolio_volatility_annualised,
            worst_case_sharpe_annualised=res.worst_case_sharpe_annualised,
            effective_n_positions=res.effective_n_positions,
            turnover_vs_prior=res.turnover_vs_prior,
            weights=res.weights,
        )
        points.append(pt)

    return RobustSensitivityResult(
        points=tuple(points),
        radii_evaluated=tuple(float(r) for r in radii),
        baseline_radius=baseline_r,
    )
