"""Institutional Linear Factor Risk Model and Euler Variance Decomposition Engine.

Core Invariants:
1. Mathematical Representation: r = B f + epsilon, Sigma_asset = B F B' + D
2. Strict Factor Alignment: All assets and factors must align strictly without silent zero imputation.
3. Euler-Consistent Factor Risk: Marginal factor risk m = F b_p, component variance C_k = b_{p,k} (F b_p)_k, sum_k C_k = b_p' F b_p.
4. Active Risk Decomposition: a = w - w_b, Delta b = B' a, TE^2 = Delta b' F Delta b + a' D a.
5. Deterministic Data Integrity Checking: Validates coverage, time-alignment, and missingness fail-closed.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from start.portfolio.contracts import (
    ActiveRiskDecompositionResult,
    FactorDataIntegrityResult,
    FactorRiskDecompositionResult,
    FactorRiskModelResult,
    MetricHorizon,
    validate_horizon_alignment,
)
from start.portfolio.covariance import diagnose_covariance


def _fingerprint_matrix(arr: np.ndarray) -> str:
    """Compute SHA-256 fingerprint for a 2D numerical array."""
    return hashlib.sha256(np.asarray(arr, dtype=float).tobytes()).hexdigest()[:32]


def validate_factor_alignment(
    exposures: pd.DataFrame | np.ndarray,
    factor_cov: pd.DataFrame | np.ndarray,
    specific_var: pd.Series | dict[str, float] | np.ndarray,
    assets: list[str] | tuple[str, ...] | None = None,
    factors: list[str] | tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...]]:
    """Strictly align and validate factor model components fail-closed."""
    # 1. Resolve asset and factor names
    if isinstance(exposures, pd.DataFrame):
        if exposures.index.has_duplicates:
            raise ValueError("Duplicate asset labels in exposure rows (fail-closed)")
        if exposures.columns.has_duplicates:
            raise ValueError("Duplicate factor labels in exposure columns (fail-closed)")
        asset_names = tuple(str(a) for a in exposures.index)
        factor_names = tuple(str(f) for f in exposures.columns)
        B = exposures.to_numpy(dtype=float)
    else:
        B = np.asarray(exposures, dtype=float)
        if assets is None or factors is None:
            raise ValueError("assets and factors tuples must be supplied for ndarray exposures")
        asset_names = tuple(str(a) for a in assets)
        factor_names = tuple(str(f) for f in factors)

    n_assets = len(asset_names)
    n_factors = len(factor_names)

    if B.shape != (n_assets, n_factors):
        raise ValueError(f"Exposure matrix shape {B.shape} does not match ({n_assets}, {n_factors})")

    # 2. Factor Covariance F
    if isinstance(factor_cov, pd.DataFrame):
        if factor_cov.index.has_duplicates or factor_cov.columns.has_duplicates:
            raise ValueError("Duplicate factor labels in factor covariance (fail-closed)")
        missing_f = [f for f in factor_names if f not in factor_cov.columns]
        if missing_f:
            raise ValueError(f"Factor(s) {missing_f} missing from factor covariance (fail-closed)")
        F = factor_cov.reindex(index=list(factor_names), columns=list(factor_names)).to_numpy(dtype=float)
    else:
        F = np.asarray(factor_cov, dtype=float)
        if F.shape != (n_factors, n_factors):
            raise ValueError(f"Factor covariance shape {F.shape} does not match ({n_factors}, {n_factors})")

    # 3. Specific Variances D
    if isinstance(specific_var, (pd.Series, dict)):
        missing_a = [a for a in asset_names if a not in specific_var]
        if missing_a:
            raise ValueError(f"Asset(s) {missing_a} missing from specific variances (fail-closed)")
        d_vec = np.array([float(specific_var[a]) for a in asset_names], dtype=float)
    else:
        d_vec = np.asarray(specific_var, dtype=float)
        if len(d_vec) != n_assets:
            raise ValueError(f"Specific variances length {len(d_vec)} does not match asset count {n_assets}")

    if not np.all(np.isfinite(B)):
        raise ValueError("Exposure matrix contains non-finite (NaN/Inf) elements")
    if not np.all(np.isfinite(F)):
        raise ValueError("Factor covariance matrix contains non-finite (NaN/Inf) elements")
    if not np.all(np.isfinite(d_vec)):
        raise ValueError("Specific variances contain non-finite (NaN/Inf) elements")
    if np.any(d_vec < 0.0):
        raise ValueError("Specific variances must be non-negative (fail-closed)")

    return B, F, d_vec, asset_names, factor_names


def build_linear_factor_model(
    exposures: pd.DataFrame | np.ndarray,
    factor_cov: pd.DataFrame | np.ndarray,
    specific_var: pd.Series | dict[str, float] | np.ndarray,
    assets: list[str] | tuple[str, ...] | None = None,
    factors: list[str] | tuple[str, ...] | None = None,
    time_alignment: str = "beginning_of_period_exposures",
    factor_cov_horizon: MetricHorizon | str = MetricHorizon.PERIODIC,
    specific_var_horizon: MetricHorizon | str = MetricHorizon.PERIODIC,
    frequency: str | None = None,
    periods_per_year: float = 252.0,
) -> FactorRiskModelResult:
    """Construct a linear factor risk model and reconstruct the asset covariance matrix Sigma = B F B' + D."""
    validate_horizon_alignment(
        mu_horizon=factor_cov_horizon,
        cov_horizon=specific_var_horizon,
        periods_per_year=periods_per_year,
        frequency=frequency,
    )

    B, F, d_vec, asset_names, factor_names = validate_factor_alignment(
        exposures=exposures,
        factor_cov=factor_cov,
        specific_var=specific_var,
        assets=assets,
        factors=factors,
    )

    # Reconstruct Sigma = B F B' + diag(D)
    sigma_asset = B @ F @ B.T + np.diag(d_vec)
    sigma_asset = (sigma_asset + sigma_asset.T) / 2.0

    diag_result = diagnose_covariance(sigma_asset, assets=list(asset_names))

    fp_exp = _fingerprint_matrix(B)
    fp_fcov = _fingerprint_matrix(F)
    fp_svar = _fingerprint_matrix(d_vec)
    fp_sigma = _fingerprint_matrix(sigma_asset)

    spec_var_dict = {a: float(d_vec[i]) for i, a in enumerate(asset_names)}

    return FactorRiskModelResult(
        asset_order=asset_names,
        factor_order=factor_names,
        exposure_matrix=B.tolist(),
        factor_covariance=F.tolist(),
        specific_variances=spec_var_dict,
        reconstructed_covariance=sigma_asset.tolist(),
        exposure_fingerprint=fp_exp,
        factor_covariance_fingerprint=fp_fcov,
        specific_variance_fingerprint=fp_svar,
        reconstructed_covariance_fingerprint=fp_sigma,
        diagnostics=diag_result,
        time_alignment=time_alignment,
        horizon=MetricHorizon(factor_cov_horizon)
        if isinstance(factor_cov_horizon, str)
        else factor_cov_horizon,
        frequency=frequency,
        periods_per_year=periods_per_year,
    )


def decompose_factor_risk(
    weights: dict[str, float] | pd.Series | np.ndarray,
    factor_model: FactorRiskModelResult,
    periods_per_year: float | None = None,
) -> FactorRiskDecompositionResult:
    """Perform Euler-consistent systematic and factor risk component decomposition."""
    eff_ppy = periods_per_year if periods_per_year is not None else factor_model.periods_per_year
    h_model = (
        MetricHorizon(factor_model.horizon) if isinstance(factor_model.horizon, str) else factor_model.horizon
    )

    validate_horizon_alignment(
        mu_horizon=h_model,
        cov_horizon=h_model,
        periods_per_year=eff_ppy,
        frequency=factor_model.frequency,
    )

    asset_names = factor_model.asset_order
    factor_names = factor_model.factor_order

    if isinstance(weights, (dict, pd.Series)):
        missing = [a for a in asset_names if a not in weights]
        if missing:
            raise ValueError(f"Asset(s) {missing} missing from portfolio weights (fail-closed)")
        w = np.array([float(weights[a]) for a in asset_names], dtype=float)
        w_dict = {a: float(weights[a]) for a in asset_names}
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != len(asset_names):
            raise ValueError(f"Weights length {len(w)} does not match asset count {len(asset_names)}")
        w_dict = {a: float(w[i]) for i, a in enumerate(asset_names)}

    B = np.asarray(factor_model.exposure_matrix, dtype=float)
    F = np.asarray(factor_model.factor_covariance, dtype=float)
    d_vec = np.array([factor_model.specific_variances[a] for a in asset_names], dtype=float)
    sigma = np.asarray(factor_model.reconstructed_covariance, dtype=float)

    # 1. Portfolio Factor Exposures b_p = B' w
    b_p = B.T @ w
    port_exp_dict = {f: float(b_p[i]) for i, f in enumerate(factor_names)}

    # 2. Systematic Variance V_sys = b_p' F b_p
    v_sys = float(b_p @ F @ b_p)

    # 3. Euler Factor Component Variance C_k = b_{p,k} * (F b_p)_k
    marginal_factor_risk = F @ b_p
    factor_comp_var = b_p * marginal_factor_risk
    factor_comp_dict = {f: float(factor_comp_var[i]) for i, f in enumerate(factor_names)}
    euler_error = float(abs(np.sum(factor_comp_var) - v_sys))

    # 4. Specific Variance V_spec = w' D w = sum w_i^2 * d_i
    asset_spec_contrib = (w**2) * d_vec
    asset_spec_dict = {a: float(asset_spec_contrib[i]) for i, a in enumerate(asset_names)}
    v_spec = float(np.sum(asset_spec_contrib))

    # 5. Total Variance & Volatilities
    v_total = v_sys + v_spec
    v_direct = float(w @ sigma @ w)
    total_recon_error = float(abs(v_total - v_direct))

    if h_model == MetricHorizon.ANNUAL:
        vol_total_ann = math.sqrt(max(0.0, v_total))
        vol_sys_ann = math.sqrt(max(0.0, v_sys))
        vol_spec_ann = math.sqrt(max(0.0, v_spec))
    else:
        vol_total_ann = math.sqrt(max(0.0, v_total * eff_ppy))
        vol_sys_ann = math.sqrt(max(0.0, v_sys * eff_ppy))
        vol_spec_ann = math.sqrt(max(0.0, v_spec * eff_ppy))

    sys_share = float(v_sys / v_total) if v_total > 1e-15 else 0.0
    spec_share = float(v_spec / v_total) if v_total > 1e-15 else 0.0

    factor_shares = {
        f: float(factor_comp_dict[f] / v_total) if v_total > 1e-15 else 0.0 for f in factor_names
    }

    return FactorRiskDecompositionResult(
        weights=w_dict,
        portfolio_factor_exposures=port_exp_dict,
        systematic_variance_periodic=round(v_sys, 15),
        specific_variance_periodic=round(v_spec, 15),
        total_variance_periodic=round(v_total, 15),
        portfolio_volatility_annualised=round(vol_total_ann, 8),
        systematic_volatility_annualised=round(vol_sys_ann, 8),
        specific_volatility_annualised=round(vol_spec_ann, 8),
        systematic_variance_share=round(sys_share, 6),
        specific_variance_share=round(spec_share, 6),
        factor_variance_contributions_periodic=factor_comp_dict,
        factor_variance_shares=factor_shares,
        asset_specific_variance_contributions=asset_spec_dict,
        euler_reconciliation_error=round(euler_error, 16),
        total_reconciliation_error=round(total_recon_error, 16),
        horizon=h_model,
        periods_per_year=eff_ppy,
    )


def decompose_active_risk(
    weights: dict[str, float] | pd.Series | np.ndarray,
    benchmark_weights: dict[str, float] | pd.Series | np.ndarray,
    factor_model: FactorRiskModelResult,
    periods_per_year: float | None = None,
) -> ActiveRiskDecompositionResult:
    """Decompose benchmark-relative active risk (tracking error) into factor and specific active risks."""
    eff_ppy = periods_per_year if periods_per_year is not None else factor_model.periods_per_year
    h_model = (
        MetricHorizon(factor_model.horizon) if isinstance(factor_model.horizon, str) else factor_model.horizon
    )

    validate_horizon_alignment(
        mu_horizon=h_model,
        cov_horizon=h_model,
        periods_per_year=eff_ppy,
        frequency=factor_model.frequency,
    )

    asset_names = factor_model.asset_order
    factor_names = factor_model.factor_order

    if isinstance(weights, (dict, pd.Series)):
        missing_w = [a for a in asset_names if a not in weights]
        if missing_w:
            raise ValueError(f"Asset(s) {missing_w} missing from portfolio weights (fail-closed)")
        w = np.array([float(weights[a]) for a in asset_names], dtype=float)
        w_dict = {a: float(weights[a]) for a in asset_names}
    else:
        w = np.asarray(weights, dtype=float)
        w_dict = {a: float(w[i]) for i, a in enumerate(asset_names)}

    if isinstance(benchmark_weights, (dict, pd.Series)):
        missing_b = [a for a in asset_names if a not in benchmark_weights]
        if missing_b:
            raise ValueError(f"Asset(s) {missing_b} missing from benchmark weights (fail-closed)")
        w_b = np.array([float(benchmark_weights[a]) for a in asset_names], dtype=float)
        w_b_dict = {a: float(benchmark_weights[a]) for a in asset_names}
    else:
        w_b = np.asarray(benchmark_weights, dtype=float)
        w_b_dict = {a: float(w_b[i]) for i, a in enumerate(asset_names)}

    a_vec = w - w_b
    a_dict = {a: float(a_vec[i]) for i, a in enumerate(asset_names)}

    B = np.asarray(factor_model.exposure_matrix, dtype=float)
    F = np.asarray(factor_model.factor_covariance, dtype=float)
    d_vec = np.array([factor_model.specific_variances[a] for a in asset_names], dtype=float)
    sigma = np.asarray(factor_model.reconstructed_covariance, dtype=float)

    # Active Factor Exposures Delta b = B' (w - w_b)
    delta_b = B.T @ a_vec
    active_exp_dict = {f: float(delta_b[i]) for i, f in enumerate(factor_names)}

    # Factor Active Variance
    v_active_sys = float(delta_b @ F @ delta_b)
    marginal_active_factor = F @ delta_b
    active_factor_comp = delta_b * marginal_active_factor
    active_factor_dict = {f: float(active_factor_comp[i]) for i, f in enumerate(factor_names)}

    # Specific Active Variance
    asset_active_spec = (a_vec**2) * d_vec
    asset_active_spec_dict = {a: float(asset_active_spec[i]) for i, a in enumerate(asset_names)}
    v_active_spec = float(np.sum(asset_active_spec))

    # Total Active Variance & Tracking Error
    v_active_total = v_active_sys + v_active_spec
    v_active_direct = float(a_vec @ sigma @ a_vec)
    recon_error = float(abs(v_active_total - v_active_direct))

    if h_model == MetricHorizon.ANNUAL:
        te_ann = math.sqrt(max(0.0, v_active_total))
    else:
        te_ann = math.sqrt(max(0.0, v_active_total * eff_ppy))

    sys_share = float(v_active_sys / v_active_total) if v_active_total > 1e-15 else 0.0
    spec_share = float(v_active_spec / v_active_total) if v_active_total > 1e-15 else 0.0

    return ActiveRiskDecompositionResult(
        weights=w_dict,
        benchmark_weights=w_b_dict,
        active_weights=a_dict,
        active_factor_exposures=active_exp_dict,
        factor_active_variance_periodic=round(v_active_sys, 15),
        specific_active_variance_periodic=round(v_active_spec, 15),
        total_active_variance_periodic=round(v_active_total, 15),
        tracking_error_annualised=round(te_ann, 8),
        factor_active_share=round(sys_share, 6),
        specific_active_share=round(spec_share, 6),
        active_factor_contributions_periodic=active_factor_dict,
        asset_specific_active_contributions=asset_active_spec_dict,
        reconciliation_error=round(recon_error, 16),
        horizon=h_model,
        periods_per_year=eff_ppy,
    )


def validate_factor_data_integrity(
    returns: pd.DataFrame | np.ndarray | None = None,
    exposures: pd.DataFrame | np.ndarray | None = None,
    factor_cov: pd.DataFrame | np.ndarray | None = None,
    specific_var: pd.Series | dict[str, float] | np.ndarray | None = None,
    factor_returns: pd.DataFrame | np.ndarray | None = None,
    weights: dict[str, float] | pd.Series | None = None,
    benchmark_weights: dict[str, float] | pd.Series | None = None,
    timestamp: Any = None,
) -> FactorDataIntegrityResult:
    """Deterministic pre-flight verification of factor model data integrity without LLM intervention."""
    issues: list[str] = []
    assets: list[str] = []
    factors: list[str] = []
    has_dup_assets = False
    has_dup_factors = False
    missing_exp = 0
    missing_fret = 0
    missing_svar = 0
    has_lookahead = False

    has_exp = exposures is not None
    has_fcov = factor_cov is not None
    has_svar = specific_var is not None
    has_fret = factor_returns is not None
    has_any = has_exp or has_fcov or has_svar or has_fret

    # Partial factor input validation (fail closed)
    if has_any:
        if has_exp and not has_fcov:
            issues.append(
                "Partial factor model specification: factor exposures provided but factor covariance is missing (fail-closed)."
            )
        if has_exp and not has_svar:
            missing_svar = 1
            issues.append(
                "Partial factor model specification: factor exposures provided but specific variance is missing (fail-closed)."
            )
        if has_fcov and not has_exp:
            missing_exp = 1
            issues.append(
                "Partial factor model specification: factor covariance provided but factor exposures are missing (fail-closed)."
            )
        if has_svar and not has_exp:
            missing_exp = 1
            issues.append(
                "Partial factor model specification: specific variance provided but factor exposures are missing (fail-closed)."
            )
        if has_fret and not has_exp:
            missing_exp = 1
            issues.append(
                "Partial factor model specification: factor returns provided but factor exposures are missing (fail-closed)."
            )

    if exposures is not None:
        if isinstance(exposures, pd.DataFrame):
            if exposures.index.has_duplicates:
                has_dup_assets = True
                issues.append("Duplicate asset index found in factor exposures.")
            if exposures.columns.has_duplicates:
                has_dup_factors = True
                issues.append("Duplicate factor columns found in factor exposures.")
            assets = [str(a) for a in exposures.index]
            factors = [str(f) for f in exposures.columns]
            if exposures.isna().any().any():
                missing_exp = int(exposures.isna().sum().sum())
                issues.append(f"Found {missing_exp} missing (NaN) values in factor exposures.")
        else:
            arr = np.asarray(exposures)
            if np.isnan(arr).any():
                missing_exp = int(np.isnan(arr).sum())
                issues.append(f"Found {missing_exp} missing (NaN) values in factor exposures array.")

    if returns is not None and isinstance(returns, pd.DataFrame):
        if returns.columns.has_duplicates:
            has_dup_assets = True
            issues.append("Duplicate asset columns found in returns dataframe.")
        if not assets:
            assets = [str(c) for c in returns.columns]
        else:
            diff_a = set(assets) - set(returns.columns)
            if diff_a:
                issues.append(f"Assets {diff_a} present in exposures but missing from returns.")

    if specific_var is not None and isinstance(specific_var, (dict, pd.Series)):
        missing_sv = set(assets) - set(specific_var.keys())
        if missing_sv:
            missing_svar = len(missing_sv)
            issues.append(f"Specific variances missing for asset(s): {missing_sv}")

    is_valid = len(issues) == 0

    return FactorDataIntegrityResult(
        is_valid=is_valid,
        n_assets=len(assets),
        n_factors=len(factors),
        assets=tuple(assets),
        factors=tuple(factors),
        has_duplicate_assets=has_dup_assets,
        has_duplicate_factors=has_dup_factors,
        missing_exposure_count=missing_exp,
        missing_factor_return_count=missing_fret,
        missing_specific_variance_count=missing_svar,
        has_lookahead_violation=has_lookahead,
        issues=tuple(issues),
    )
