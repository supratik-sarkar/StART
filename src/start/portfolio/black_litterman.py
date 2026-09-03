"""Deterministic Black-Litterman Portfolio Optimization Engine.

Core invariants:
- Implied equilibrium returns: Pi = delta * Sigma * w_m.
- Exact Bayesian updating with user views (P, Q, Omega).
- Never invents market weights, risk aversion, tau, Omega, or subjective views.
- Strict input dimension and applicability validation.
- Emits typed BlackLittermanResult with full constraint verification.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from start.portfolio.constraints import verify_portfolio_constraints
from start.portfolio.contracts import (
    BlackLittermanResult,
    DeterminismTier,
    MethodApplicability,
    MetricHorizon,
    ViewUncertaintyPolicy,
    validate_horizon_alignment,
)
from start.registry.market_contexts import PortfolioConstraints

CONSTRAINT_TOLERANCE = 1e-6

BLACK_LITTERMAN_APPLICABILITY = MethodApplicability(
    method_name="black_litterman",
    required_inputs=("covariance", "market_weights", "P", "Q"),
    min_assets=2,
    min_observations=2,
    requires_psd_covariance=True,
    supports_bounds=True,
    supports_group_constraints=True,
    supports_turnover_constraints=True,
    determinism=DeterminismTier.NUMERICALLY_DETERMINISTIC,
    assumptions=(
        "Equilibrium baseline implied returns: Pi = delta * Sigma * w_market",
        "Views formulated as P * r ~ N(Q, Omega)",
        "Tau scales uncertainty of the prior equilibrium distribution",
    ),
)


def compute_implied_equilibrium_returns(
    covariance: np.ndarray,
    market_weights: np.ndarray,
    risk_aversion: float = 3.0,
) -> np.ndarray:
    """Compute equilibrium implied excess returns: Pi = delta * Sigma * w_market."""
    if risk_aversion <= 0:
        raise ValueError(f"Risk aversion must be positive, got {risk_aversion}")
    sigma = np.asarray(covariance, dtype=float)
    w_m = np.asarray(market_weights, dtype=float)
    if sigma.shape[0] != sigma.shape[1] or sigma.shape[0] != len(w_m):
        raise ValueError(
            f"Dimension mismatch between covariance {sigma.shape} and market weights ({len(w_m)},)"
        )
    return float(risk_aversion) * (sigma @ w_m)


def compute_black_litterman_posterior(
    covariance: np.ndarray,
    market_weights: np.ndarray,
    P: np.ndarray,
    Q: np.ndarray,
    Omega: np.ndarray | None = None,
    risk_aversion: float = 3.0,
    tau: float = 0.05,
    uncertainty_policy: ViewUncertaintyPolicy | str | None = None,
    assets: list[str] | None = None,
    view_labels: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute Black-Litterman posterior expected returns and covariance matrix."""
    sigma = np.asarray(covariance, dtype=float)
    w_m = np.asarray(market_weights, dtype=float)
    n = len(w_m)

    if tau <= 0:
        raise ValueError(f"tau parameter must be positive, got {tau}")

    p_mat = np.asarray(P, dtype=float)
    q_vec = np.asarray(Q, dtype=float)

    if p_mat.ndim == 1:
        p_mat = p_mat.reshape(1, -1)
    if q_vec.ndim == 0:
        q_vec = np.array([float(q_vec)])

    k, p_cols = p_mat.shape
    if p_cols != n:
        raise ValueError(f"View matrix P columns ({p_cols}) must match asset count ({n})")
    if len(q_vec) != k:
        raise ValueError(f"View vector Q length ({len(q_vec)}) must match view count ({k})")

    eff_policy = uncertainty_policy or (ViewUncertaintyPolicy.EXPLICIT_OMEGA if Omega is not None else ViewUncertaintyPolicy.PROPORTIONAL_TAU_SIGMA)

    if Omega is None:
        # He & Litterman (1999) / Idzorek default: proportional view uncertainty
        p_tau_sigma_p = p_mat @ (tau * sigma) @ p_mat.T
        omega_mat = np.diag(np.diag(p_tau_sigma_p))
    else:
        omega_mat = np.asarray(Omega, dtype=float)
        if omega_mat.shape != (k, k):
            raise ValueError(f"Omega shape {omega_mat.shape} must be ({k}, {k})")

    pi = compute_implied_equilibrium_returns(sigma, w_m, risk_aversion=risk_aversion)

    # Master formula (Kalman / Woodbury form):
    # mu_BL = Pi + tau * Sigma * P' * (P * tau * Sigma * P' + Omega)^(-1) * (Q - P * Pi)
    m_inv_mid = np.linalg.inv(p_mat @ (tau * sigma) @ p_mat.T + omega_mat)
    view_residual = q_vec - p_mat @ pi
    gain_matrix = tau * sigma @ p_mat.T @ m_inv_mid

    mu_bl = pi + gain_matrix @ view_residual

    # Posterior covariance of asset returns:
    # Sigma_BL = Sigma + M_BL^(-1) = (1 + tau)*Sigma - tau^2 * Sigma * P' * M_inv_mid * P * Sigma
    sigma_bl = (1.0 + tau) * sigma - (tau**2) * sigma @ p_mat.T @ m_inv_mid @ p_mat @ sigma

    diagnostics = {
        "implied_returns": pi,
        "view_residuals": view_residual,
        "view_uncertainties": np.diag(omega_mat),
        "gain_matrix": gain_matrix,
        "omega_matrix": omega_mat,
        "p_matrix": p_mat,
        "q_vector": q_vec,
        "uncertainty_policy": str(eff_policy),
    }
    return mu_bl, sigma_bl, diagnostics


def solve_black_litterman(
    covariance: pd.DataFrame | np.ndarray,
    market_weights: pd.Series | dict[str, float] | np.ndarray,
    P: np.ndarray | list[list[float]],
    Q: np.ndarray | list[float],
    Omega: np.ndarray | list[list[float]] | None = None,
    risk_aversion: float = 3.0,
    tau: float = 0.05,
    uncertainty_policy: ViewUncertaintyPolicy | str | None = None,
    assets: list[str] | None = None,
    view_labels: list[str] | None = None,
    constraints: PortfolioConstraints | None = None,
    prior_weights: pd.Series | dict[str, float] | np.ndarray | None = None,
    rf_periodic: float = 0.0,
    periods_per_year: float = 252.0,
    returns_horizon: MetricHorizon | str | None = None,
    cov_horizon: MetricHorizon | str | None = None,
) -> BlackLittermanResult:
    """Execute complete institutional Black-Litterman optimization and constraint verification."""
    validate_horizon_alignment(returns_horizon, cov_horizon, periods_per_year=periods_per_year)
    if isinstance(covariance, pd.DataFrame):
        asset_names = list(covariance.columns)
        sigma = covariance.to_numpy(dtype=float)
    else:
        sigma = np.asarray(covariance, dtype=float)
        asset_names = list(assets) if assets is not None else [f"A{i}" for i in range(len(sigma))]

    n = len(asset_names)
    if isinstance(market_weights, (pd.Series, dict)):
        missing = [a for a in asset_names if a not in market_weights]
        if missing:
            raise ValueError(f"Asset(s) {missing} missing from market_weights (fail-closed)")
        w_m_arr = np.array([float(market_weights[a]) for a in asset_names], dtype=float)
    else:
        w_m_arr = np.asarray(market_weights, dtype=float)

    if len(w_m_arr) != n:
        raise ValueError(f"Market weights length {len(w_m_arr)} does not match asset count {n}")

    k = len(Q) if hasattr(Q, "__len__") else 1
    v_labels = tuple(view_labels) if view_labels is not None else tuple(f"View_{i+1}" for i in range(k))

    eff_policy = uncertainty_policy or (ViewUncertaintyPolicy.EXPLICIT_OMEGA if Omega is not None else ViewUncertaintyPolicy.PROPORTIONAL_TAU_SIGMA)

    mu_bl, sigma_bl, diag = compute_black_litterman_posterior(
        covariance=sigma,
        market_weights=w_m_arr,
        P=np.asarray(P, dtype=float),
        Q=np.asarray(Q, dtype=float),
        Omega=np.asarray(Omega, dtype=float) if Omega is not None else None,
        risk_aversion=risk_aversion,
        tau=tau,
        uncertainty_policy=eff_policy,
        assets=asset_names,
        view_labels=list(v_labels),
    )

    prior_arr = None
    if prior_weights is not None:
        if isinstance(prior_weights, (pd.Series, dict)):
            missing_prior = [a for a in asset_names if a not in prior_weights]
            if missing_prior:
                raise ValueError(f"Asset(s) {missing_prior} missing from prior_weights (fail-closed)")
            prior_arr = np.array([float(prior_weights[a]) for a in asset_names], dtype=float)
        else:
            prior_arr = np.asarray(prior_weights, dtype=float)
    else:
        prior_arr = w_m_arr.copy()

    # Optimization using posterior expected returns mu_bl and posterior covariance sigma_bl
    delta_val = float(risk_aversion)

    def objective(w: np.ndarray) -> float:
        return float(- (w @ mu_bl - 0.5 * delta_val * (w @ sigma_bl @ w)))

    def grad(w: np.ndarray) -> np.ndarray:
        return - (mu_bl - delta_val * (sigma_bl @ w))

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
        benchmark_weights=w_m_arr,
        covariance=sigma,
    )

    init_w = np.full(n, 1.0 / n)
    res = minimize(
        objective,
        init_w,
        jac=grad,
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
        benchmark_weights=w_m_arr,
        prior_weights=prior_arr,
    )

    if not res.success or not ver_res.is_valid:
        w_opt = np.zeros(n, dtype=float)
        converged = False
        usable_solution = False
        solver_status = f"FAILED: {res.message if not res.success else 'Constraint verification failed'}"
        solver_message = res.message if not res.success else "Constraint verification failed"
        post_vol_ann = 0.0
        sharpe_ann = None
        turnover = 0.0
    else:
        w_opt = w_candidate
        converged = True
        usable_solution = True
        solver_status = "OPTIMAL"
        solver_message = ""

        port_var = float(w_opt @ sigma_bl @ w_opt)
        post_vol = math.sqrt(max(0.0, port_var))
        ppy = float(periods_per_year)
        post_vol_ann = post_vol * math.sqrt(ppy)
        post_ret_per = float(w_opt @ mu_bl)

        sharpe_ann = (
            ((post_ret_per - rf_periodic) / post_vol) * math.sqrt(ppy)
            if post_vol > 1e-12
            else None
        )

        turnover = float(0.5 * np.sum(np.abs(w_opt - prior_arr)))

    cov_hash = hashlib.sha256(sigma_bl.tobytes()).hexdigest()[:32]

    return BlackLittermanResult(
        implied_returns={a: round(float(v), 8) for a, v in zip(asset_names, diag["implied_returns"], strict=True)},
        posterior_returns={a: round(float(v), 8) for a, v in zip(asset_names, mu_bl, strict=True)},
        prior_weights={a: round(float(v), 8) for a, v in zip(asset_names, w_m_arr, strict=True)},
        posterior_weights={a: round(float(w_opt[i]), 8) for i, a in enumerate(asset_names)} if usable_solution else {},
        view_residuals={v: round(float(res_val), 8) for v, res_val in zip(v_labels, diag["view_residuals"], strict=True)},
        view_uncertainties={v: round(float(u_val), 8) for v, u_val in zip(v_labels, diag["view_uncertainties"], strict=True)},
        risk_aversion=risk_aversion,
        tau=tau,
        turnover_vs_prior=round(turnover, 6),
        constraint_verification=ver_res,
        posterior_covariance_fingerprint=cov_hash,
        p_matrix=[[float(x) for x in row] for row in diag["p_matrix"]],
        q_vector=[float(x) for x in diag["q_vector"]],
        omega_matrix=[[float(x) for x in row] for row in diag["omega_matrix"]],
        view_labels=v_labels,
        posterior_volatility_annualised=round(post_vol_ann, 8),
        posterior_sharpe_annualised=round(sharpe_ann, 6) if sharpe_ann is not None else None,
        uncertainty_policy=eff_policy,
        converged=converged,
        usable_solution=usable_solution,
        solver_status=solver_status,
        solver_message=solver_message,
    )
