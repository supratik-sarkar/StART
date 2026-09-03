"""Institutional Covariance Diagnostics, PSD Repair, and Model Comparison Engine.

Core Invariants:
1. Strict Numerical Diagnostics: Computes exact spectral, condition, and entropy-based effective rank metrics.
2. Mathematically Explicit PSD Repair:
   - SPECTRAL_CLIPPING: Standard eigenvalue clipping to a positive threshold.
   - HIGHAM_NEAREST_CORRELATION: Higham (2002) alternating projections with Dykstra correction on the correlation matrix, rescaled to original diagonal variances.
3. Repair is an Intervention: PSD_REPAIRED != MODEL_VALID. Repair is never applied silently inside solvers.
4. Canonical Estimator Reuse: Multi-estimator comparison directly composes canonical empirical, Ledoit-Wolf, and RegEM implementations.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from start.portfolio.contracts import (
    CovarianceComparisonResult,
    CovarianceDiagnostics,
    MetricHorizon,
    PSDRepairMethod,
    PSDRepairResult,
)
from start.tests.covariance import run_regularized_em


def _matrix_fingerprint(mat: np.ndarray | list[list[float]]) -> str:
    """Deterministic SHA-256 fingerprint of a 2D numerical matrix."""
    arr = np.asarray(mat, dtype=float)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:32]


def diagnose_covariance(
    cov: np.ndarray | pd.DataFrame | list[list[float]],
    assets: list[str] | tuple[str, ...] | None = None,
    tol: float = 1e-10,
) -> CovarianceDiagnostics:
    """Compute comprehensive mathematical diagnostics for a covariance matrix."""
    if isinstance(cov, pd.DataFrame):
        mat = cov.to_numpy(dtype=float)
    else:
        mat = np.asarray(cov, dtype=float)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Covariance matrix must be square 2D, got shape {mat.shape}")

    n = mat.shape[0]
    if n == 0:
        raise ValueError("Covariance matrix cannot be empty (0x0)")

    # Check for NaN / Inf
    if not np.all(np.isfinite(mat)):
        raise ValueError("Covariance matrix contains non-finite (NaN/Inf) elements")

    symmetry_error = float(np.max(np.abs(mat - mat.T)))
    is_symmetric = symmetry_error <= 1e-8

    # Symmetrize for spectral analysis
    sym_mat = (mat + mat.T) / 2.0
    eigs = np.linalg.eigvalsh(sym_mat)
    min_eig = float(eigs.min())
    max_eig = float(eigs.max())
    is_psd = bool(min_eig >= -tol)

    # Rank calculations
    rank = int(np.linalg.matrix_rank(sym_mat, tol=tol))
    scale_tol = tol * max(1.0, max(abs(min_eig), abs(max_eig)))
    numerical_rank = int((eigs > scale_tol).sum())

    # Condition number
    if min_eig > 1e-15:
        cond = float(max_eig / min_eig)
    else:
        cond = float("inf")

    # Trace and log determinant
    tr = float(np.trace(sym_mat))
    if min_eig > 0:
        log_det = float(np.sum(np.log(eigs)))
    else:
        log_det = None

    # Entropy-based Effective Rank: p_i = max(0, lambda_i) / sum(max(0, lambda_j))
    # effective_rank = exp(-sum p_i ln p_i)
    pos_eigs = np.maximum(0.0, eigs)
    sum_pos = float(np.sum(pos_eigs))
    if sum_pos > 1e-15:
        p_vec = pos_eigs / sum_pos
        nz_p = p_vec[p_vec > 1e-15]
        entropy = float(-np.sum(nz_p * np.log(nz_p)))
        eff_rank = float(np.exp(entropy))
        largest_share = float(np.max(pos_eigs) / sum_pos)
    else:
        eff_rank = 0.0
        largest_share = 0.0

    diag_vals = np.diag(sym_mat)
    diagonal_positive = bool(np.all(diag_vals > 1e-15))

    # Correlation validity check
    valid_corr = False
    if diagonal_positive:
        stds = np.sqrt(diag_vals)
        corr_mat = sym_mat / np.outer(stds, stds)
        if np.all(corr_mat >= -1.0 - 1e-6) and np.all(corr_mat <= 1.0 + 1e-6):
            valid_corr = True

    fp = _matrix_fingerprint(mat)

    return CovarianceDiagnostics(
        n_assets=n,
        is_symmetric=is_symmetric,
        symmetry_error=round(symmetry_error, 15),
        is_psd=is_psd,
        minimum_eigenvalue=round(min_eig, 15),
        maximum_eigenvalue=round(max_eig, 15),
        eigenvalue_spectrum=tuple(float(round(v, 12)) for v in eigs),
        rank=rank,
        numerical_rank=numerical_rank,
        condition_number=round(cond, 6) if math.isfinite(cond) else float("inf"),
        trace=round(tr, 12),
        log_determinant=round(log_det, 12) if log_det is not None else None,
        effective_rank=round(eff_rank, 6),
        largest_eigenvalue_share=round(largest_share, 8),
        diagonal_positive=diagonal_positive,
        valid_correlation_conversion=valid_corr,
        matrix_fingerprint=fp,
    )


def repair_psd_covariance(
    cov: np.ndarray | pd.DataFrame | list[list[float]],
    method: PSDRepairMethod | str = PSDRepairMethod.HIGHAM_NEAREST_CORRELATION,
    min_eigenvalue: float = 1e-8,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> PSDRepairResult:
    """Deterministically repair an indefinite symmetric covariance matrix to positive semi-definite.

    Methods:
    - SPECTRAL_CLIPPING: Clamps eigenvalues of symmetric matrix to min_eigenvalue.
    - HIGHAM_NEAREST_CORRELATION: Higham (2002) alternating projections with Dykstra correction on the correlation matrix, rescaled to original diagonal variances.
    """
    if isinstance(cov, pd.DataFrame):
        mat = cov.to_numpy(dtype=float)
    else:
        mat = np.asarray(cov, dtype=float)

    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Covariance matrix must be square 2D, got shape {mat.shape}")

    if not np.all(np.isfinite(mat)):
        raise ValueError("Cannot repair covariance matrix with non-finite (NaN/Inf) values (fail-closed)")

    eff_method = PSDRepairMethod(method)
    fp_before = _matrix_fingerprint(mat)
    sym_orig = (mat + mat.T) / 2.0
    orig_min_eig = float(np.linalg.eigvalsh(sym_orig).min())

    n = mat.shape[0]
    iterations_used = 0
    converged = True

    if eff_method == PSDRepairMethod.SPECTRAL_CLIPPING:
        eigs, vecs = np.linalg.eigh(sym_orig)
        clipped_eigs = np.maximum(eigs, float(min_eigenvalue))
        repaired = vecs @ np.diag(clipped_eigs) @ vecs.T
        repaired = (repaired + repaired.T) / 2.0
        iterations_used = 1
    elif eff_method == PSDRepairMethod.HIGHAM_NEAREST_CORRELATION:
        # Preserve original variances
        orig_diag = np.diag(sym_orig).copy()
        variances = np.maximum(orig_diag, float(min_eigenvalue))
        stds = np.sqrt(variances)
        # Scaled floor for correlation projection so rescaled covariance meets min_eigenvalue
        min_var = float(np.min(variances))
        corr_min_eig = max(1e-12, float(min_eigenvalue) / max(1e-12, min_var))

        # Initial approximate correlation
        R0 = sym_orig / np.outer(stds, stds)
        np.fill_diagonal(R0, 1.0)
        R0 = (R0 + R0.T) / 2.0

        # Dykstra's alternating projection algorithm (Higham, 2002)
        dS = np.zeros((n, n), dtype=float)
        Y = R0.copy()

        for k in range(1, max_iter + 1):
            iterations_used = k
            R_k = Y - dS
            # 1. Project onto S (PSD cone)
            eigs_k, vecs_k = np.linalg.eigh(R_k)
            eigs_k = np.maximum(eigs_k, corr_min_eig)
            X_k = vecs_k @ np.diag(eigs_k) @ vecs_k.T
            X_k = (X_k + X_k.T) / 2.0

            # 2. Update Dykstra correction
            dS = X_k - R_k

            # 3. Project onto U (unit diagonal)
            Y_next = X_k.copy()
            np.fill_diagonal(Y_next, 1.0)

            # Check convergence
            denom = max(1e-15, float(np.linalg.norm(Y, "fro")))
            diff = float(np.linalg.norm(Y_next - Y, "fro")) / denom
            Y = Y_next
            if diff < tol:
                converged = True
                break
        else:
            converged = False

        # Final projection onto PSD cone to strictly guarantee minimum eigenvalue
        final_eigs, final_vecs = np.linalg.eigh(Y)
        final_eigs = np.maximum(final_eigs, corr_min_eig)
        R_final = final_vecs @ np.diag(final_eigs) @ final_vecs.T
        R_final = (R_final + R_final.T) / 2.0

        # Rescale correlation back to covariance
        repaired = np.outer(stds, stds) * R_final
        repaired = (repaired + repaired.T) / 2.0


    repaired_min_eig = float(np.linalg.eigvalsh(repaired).min())
    frob_dist = float(np.linalg.norm(repaired - mat, "fro"))
    orig_frob = float(np.linalg.norm(mat, "fro"))
    rel_frob = frob_dist / max(1e-15, orig_frob)
    max_change = float(np.max(np.abs(repaired - mat)))
    diag_preserved = bool(np.allclose(np.diag(repaired), np.diag(mat), atol=1e-5))
    fp_after = _matrix_fingerprint(repaired)

    return PSDRepairResult(
        repair_method=eff_method,
        original_minimum_eigenvalue=round(orig_min_eig, 15),
        repaired_minimum_eigenvalue=round(repaired_min_eig, 15),
        frobenius_distortion=round(frob_dist, 12),
        relative_frobenius_distortion=round(rel_frob, 10),
        maximum_element_change=round(max_change, 12),
        diagonal_preserved=diag_preserved,
        iterations_used=iterations_used,
        converged=converged,
        matrix_fingerprint_before=fp_before,
        matrix_fingerprint_after=fp_after,
        repaired_matrix=repaired.tolist(),
        pd_floor=float(min_eigenvalue),
        intervention_reason="Non-PSD input requiring explicit numerical repair",
    )


def compare_covariance_estimators(
    returns: pd.DataFrame | np.ndarray,
    assets: list[str] | tuple[str, ...] | None = None,
    estimators: list[str] | tuple[str, ...] | None = None,
    portfolio_weights: dict[str, float] | pd.Series | np.ndarray | None = None,
    returns_horizon: MetricHorizon | str = MetricHorizon.PERIODIC,
    frequency: str | None = None,
    periods_per_year: float = 252.0,
) -> CovarianceComparisonResult:
    """Deterministically compare multiple canonical covariance estimators on the same return universe."""
    from start.portfolio.contracts import validate_horizon_alignment

    validate_horizon_alignment(
        mu_horizon=returns_horizon,
        cov_horizon=returns_horizon,
        periods_per_year=periods_per_year,
        frequency=frequency,
    )
    if isinstance(returns, pd.DataFrame):
        asset_names = tuple(str(c) for c in returns.columns)
        ret_df = returns.copy()
        ret_arr = returns.to_numpy(dtype=float)
    else:
        ret_arr = np.asarray(returns, dtype=float)
        if assets is not None:
            asset_names = tuple(str(a) for a in assets)
        else:
            asset_names = tuple(f"A{i}" for i in range(ret_arr.shape[1]))
        ret_df = pd.DataFrame(ret_arr, columns=asset_names)

    eff_estimators = tuple(estimators) if estimators is not None else ("empirical", "ledoit_wolf", "regularized_em")

    cov_matrices: dict[str, np.ndarray] = {}
    diagnostics_by_estimator: dict[str, CovarianceDiagnostics] = {}

    for est in eff_estimators:
        if est == "empirical":
            complete_data = ret_df.dropna()
            if complete_data.shape[0] < 2:
                raise ValueError("Empirical covariance requires at least 2 complete observations")
            cov_mat = complete_data.cov(ddof=1).to_numpy(dtype=float)
        elif est == "ledoit_wolf":
            complete_data = ret_df.dropna()
            if complete_data.shape[0] < 3:
                raise ValueError("Ledoit-Wolf shrinkage requires at least 3 complete observations")
            lw = LedoitWolf().fit(complete_data.to_numpy(dtype=float))
            cov_mat = lw.covariance_
        elif est == "regularized_em":
            regem_res = run_regularized_em(ret_arr)
            cov_mat = regem_res.covariance
        else:
            raise ValueError(f"Unsupported covariance estimator '{est}'")

        cov_matrices[est] = cov_mat
        diagnostics_by_estimator[est] = diagnose_covariance(cov_mat, assets=list(asset_names))

    # Compute pairwise distances
    pairwise_frobenius: dict[str, float] = {}
    pairwise_spectral: dict[str, float] = {}

    est_list = list(eff_estimators)
    for i in range(len(est_list)):
        for j in range(i + 1, len(est_list)):
            e1, e2 = est_list[i], est_list[j]
            c1, c2 = cov_matrices[e1], cov_matrices[e2]
            diff = c1 - c2
            f_dist = float(np.linalg.norm(diff, "fro"))
            s_dist = float(np.linalg.norm(diff, 2))
            pair_key = f"{e1}_vs_{e2}"
            pairwise_frobenius[pair_key] = round(f_dist, 10)
            pairwise_spectral[pair_key] = round(s_dist, 10)

    # Compute portfolio volatility impact if weights supplied
    portfolio_vols: dict[str, float] = {}
    pw_dict = None
    if portfolio_weights is not None:
        if isinstance(portfolio_weights, (dict, pd.Series)):
            pw_dict = {a: float(portfolio_weights.get(a, 0.0)) for a in asset_names}
            w_arr = np.array([pw_dict[a] for a in asset_names], dtype=float)
        else:
            w_arr = np.asarray(portfolio_weights, dtype=float)
            pw_dict = {a: float(w_arr[i]) for i, a in enumerate(asset_names)}

        for est, cov_mat in cov_matrices.items():
            var_periodic = float(w_arr @ cov_mat @ w_arr)
            var_annual = max(0.0, var_periodic * periods_per_year)
            vol_annual = math.sqrt(var_annual)
            portfolio_vols[est] = round(vol_annual, 8)

    return CovarianceComparisonResult(
        estimators_compared=eff_estimators,
        asset_order=asset_names,
        diagnostics_by_estimator=diagnostics_by_estimator,
        pairwise_frobenius_distances=pairwise_frobenius,
        pairwise_spectral_distances=pairwise_spectral,
        portfolio_volatilities_annualised=portfolio_vols,
        portfolio_weights=pw_dict,
    )
