"""Covariance estimation — three registered surfaces.

Why RegEM is a separate surface
-------------------------------

``covariance.empirical`` uses **complete cases** and says so. It does not silently
assemble a matrix from pairwise-complete blocks: a pairwise matrix is built from
different sample sizes per entry, is frequently not positive semi-definite, and looks
exactly like an ordinary covariance in the output. If missingness is the problem,
``covariance.regularized_em`` is the estimator built for it.

Ledoit–Wolf likewise refuses incomplete input rather than imputing behind the caller's
back.

What RegEM is, and is not
-------------------------

It is a **regularized covariance point estimator under multivariate Gaussian and MAR
working assumptions**. After ridge regularisation and eigenvalue clipping the result is
no longer the unconstrained maximum-likelihood estimate, so calling it "the MLE" would
be wrong. It produces no uncertainty quantification of any kind.

The E-step computes genuine conditional sufficient statistics. Mean-imputing and then
running an ordinary covariance is **not** EM: it omits the conditional covariance term
``Cov[X_M | X_O]``, which systematically understates variance on the imputed entries.
That term is included here, and a test pins it.

Eigenvalue clipping is a **StART numerical safeguard**, not part of Schneider's method,
and is attributed accordingly.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from start.core.schemas import Status, TestResult, ThresholdSpec
from start.registry import register_test
from start.registry.market_contexts import canonical_frame_bytes

__all__ = [
    "empirical",
    "ledoit_wolf_shrinkage",
    "regularized_em",
    "run_regularized_em",
    "RegEMResult",
    "PSD_EIGENVALUE_FLOOR",
    "REGEM_DEFAULT_RIDGE",
    "REGEM_DEFAULT_TOL",
    "REGEM_DEFAULT_MAX_ITER",
    "REGEM_DDOF",
]

# --------------------------------------------------------------------------- #
# FROZEN CONSTANTS
# --------------------------------------------------------------------------- #
#: PSD safeguard floor. A StART numerical safeguard, not part of any published method.
PSD_EIGENVALUE_FLOOR = 1e-12

#: RegEM defaults, frozen before any experiment.
REGEM_DEFAULT_RIDGE = 1e-6
REGEM_DEFAULT_TOL = 1e-6
REGEM_DEFAULT_MAX_ITER = 200

#: RegEM targets the ML covariance, denominator n. Stated explicitly because comparing
#: it against a sample covariance (denominator n-1) would compare different estimands
#: and produce a "failure" that is really a convention mismatch.
REGEM_DDOF = 0

_STRIPES = ("market",)
_OBJECTS = ("deterministic_calculator", "statistical_model")


def _skip(test_id: str, name: str, reason: str, **params: Any) -> TestResult:
    return TestResult(
        test_id=test_id, test_name=name, status=Status.SKIPPED, params=params, interpretation=reason
    )


def _error(
    test_id: str, name: str, reason: str, metrics: dict[str, Any] | None = None, **params: Any
) -> TestResult:
    return TestResult(
        test_id=test_id,
        test_name=name,
        status=Status.ERROR,
        params=params,
        metrics=metrics or {},
        interpretation=reason,
    )


def _conditioning(matrix: np.ndarray) -> dict[str, Any]:
    symmetric = (matrix + matrix.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(symmetric)
    minimum, maximum = float(eigenvalues.min()), float(eigenvalues.max())
    condition = float(maximum / minimum) if minimum > 0 else float("inf")
    return {
        "min_eigenvalue": round(minimum, 15),
        "max_eigenvalue": round(maximum, 15),
        "condition_number": round(condition, 6) if math.isfinite(condition) else float("inf"),
        "is_psd": bool(minimum >= -PSD_EIGENVALUE_FLOOR),
        "is_symmetric": bool(np.allclose(matrix, matrix.T, atol=1e-12)),
        "rank": int(np.linalg.matrix_rank(matrix, tol=1e-10)),
        "n_negative_eigenvalues": int((eigenvalues < -PSD_EIGENVALUE_FLOOR).sum()),
    }


def _matrix_hash(frame: pd.DataFrame) -> str:
    return hashlib.sha256(canonical_frame_bytes(frame, label="covariance")).hexdigest()[:32]


def _returns_or_reason(ctx: Any) -> tuple[pd.DataFrame | None, str]:
    returns = ctx.effective_returns()
    if returns is None or returns.empty:
        return None, "no returns available (neither returns nor prices supplied)"
    if returns.shape[1] < 2:
        return None, f"{returns.shape[1]} asset(s); at least 2 are required"
    return returns, ""


# =========================================================================== #
# 1. empirical
# =========================================================================== #
@register_test(
    "covariance.empirical",
    family="covariance",
    name="Empirical covariance",
    requires=("returns",),
    default_params={"ddof": 1, "missing_policy": "complete_case"},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification", "data_quality_lineage"),
    object_kinds=_OBJECTS,
)
def empirical(ctx: Any, ddof: int = 1, missing_policy: str = "complete_case") -> TestResult:
    """Sample covariance on complete cases. Determinism: numerical.

    ``complete_case`` is the only supported policy, and it is named in evidence. A
    pairwise-complete matrix is assembled from different sample sizes per entry, is
    frequently not PSD, and is indistinguishable from an ordinary covariance in the
    output — which is why it is refused here rather than offered as a convenience.
    """
    if missing_policy != "complete_case":
        return _error(
            "covariance.empirical",
            "Empirical covariance",
            f"missing_policy={missing_policy!r} is not supported. Only "
            "'complete_case' is available; for genuinely incomplete data use "
            "covariance.regularized_em, which is built for it.",
        )

    returns, reason = _returns_or_reason(ctx)
    if returns is None:
        return _skip("covariance.empirical", "Empirical covariance", reason, ddof=ddof)

    n_rows = int(returns.shape[0])
    complete = returns.dropna()
    n_complete = int(complete.shape[0])
    if n_complete < ddof + 2:
        return _skip(
            "covariance.empirical",
            "Empirical covariance",
            f"{n_complete} complete row(s) after dropping incomplete "
            f"observations; at least {ddof + 2} required",
            ddof=ddof,
        )

    matrix = complete.cov(ddof=ddof)
    values = matrix.to_numpy(dtype=float)
    metrics: dict[str, Any] = {
        "n_assets": int(matrix.shape[0]),
        "n_observations_supplied": n_rows,
        "n_observations_used": n_complete,
        "n_observations_dropped": n_rows - n_complete,
        "dropped_fraction": round((n_rows - n_complete) / n_rows, 10) if n_rows > 0 else 0.0,
        "missing_fraction": round((n_rows - n_complete) / n_rows, 10) if n_rows > 0 else 0.0,
        "missing_policy": "complete_case",
        "ddof": ddof,
        "estimand": "sample covariance (denominator n-1)" if ddof == 1 else f"covariance with ddof={ddof}",
        "covariance_hash": _matrix_hash(matrix),
        **_conditioning(values),
    }
    metrics["mean_variance"] = round(float(np.mean(np.diag(values))), 15)

    result = TestResult(
        test_id="covariance.empirical",
        test_name="Empirical covariance",
        params={"ddof": ddof, "missing_policy": missing_policy},
        metrics=metrics,
        thresholds=[ThresholdSpec(metric="condition_number", warn=1e4, fail=1e8)],
        interpretation=(
            f"{matrix.shape[0]}x{matrix.shape[0]} sample covariance from {n_complete:,} "
            f"complete observation(s)"
            + (f" ({n_rows - n_complete} incomplete row(s) dropped)" if n_rows != n_complete else "")
            + f"; condition number {metrics['condition_number']:.4g}."
        ),
        limitations=[
            "COMPLETE-CASE estimation. Rows with any missing value are dropped, and the "
            "count is reported. A pairwise-complete matrix is deliberately not offered: "
            "it mixes sample sizes across entries, is frequently not PSD, and looks "
            "identical to an ordinary covariance in the output.",
            "For genuinely incomplete data use covariance.regularized_em.",
            "The sample covariance is noisy when observations are few relative to "
            "assets; see covariance.ledoit_wolf_shrinkage.",
            "Determinism: numerical.",
        ],
    )
    applied = result.apply_thresholds()
    if not metrics["is_psd"]:
        applied.status = Status.FAIL
    return applied


# =========================================================================== #
# 2. ledoit_wolf_shrinkage
# =========================================================================== #
@register_test(
    "covariance.ledoit_wolf_shrinkage",
    family="covariance",
    name="Ledoit-Wolf shrinkage covariance",
    requires=("returns",),
    default_params={},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification", "benchmarking"),
    object_kinds=_OBJECTS,
)
def ledoit_wolf_shrinkage(ctx: Any) -> TestResult:
    """Ledoit-Wolf (2004) shrinkage. Determinism: numerical.

    Refuses incomplete input rather than imputing silently: an undocumented imputation
    would change the estimand while the output still said "Ledoit-Wolf".
    """
    returns, reason = _returns_or_reason(ctx)
    if returns is None:
        return _skip("covariance.ledoit_wolf_shrinkage", "Ledoit-Wolf shrinkage covariance", reason)

    if returns.isna().to_numpy().any():
        return _skip(
            "covariance.ledoit_wolf_shrinkage",
            "Ledoit-Wolf shrinkage covariance",
            f"{int(returns.isna().to_numpy().sum())} missing value(s) present. No "
            "imputation is applied here, because an undocumented fill would change the "
            "estimand while the output still said Ledoit-Wolf. Use "
            "covariance.regularized_em, or supply complete data.",
        )
    if returns.shape[0] < 3:
        return _skip(
            "covariance.ledoit_wolf_shrinkage",
            "Ledoit-Wolf shrinkage covariance",
            f"{returns.shape[0]} observation(s); at least 3 required",
        )

    from sklearn.covariance import LedoitWolf

    data = returns.to_numpy(dtype=float)
    try:
        fitted = LedoitWolf().fit(data)
    except Exception as exc:
        return _error(
            "covariance.ledoit_wolf_shrinkage",
            "Ledoit-Wolf shrinkage covariance",
            f"{type(exc).__name__}: {exc}",
        )

    shrunk = pd.DataFrame(fitted.covariance_, index=returns.columns, columns=returns.columns)
    sample = returns.cov(ddof=1)
    before = _conditioning(sample.to_numpy(dtype=float))
    after = _conditioning(shrunk.to_numpy(dtype=float))

    metrics: dict[str, Any] = {
        "n_assets": int(returns.shape[1]),
        "n_observations": int(returns.shape[0]),
        "ddof": 1,
        "observations_to_assets_ratio": round(returns.shape[0] / returns.shape[1], 6),
        "shrinkage_intensity": round(float(fitted.shrinkage_), 10),
        "condition_number_before": before["condition_number"],
        "condition_number_after": after["condition_number"],
        "min_eigenvalue_before": before["min_eigenvalue"],
        "min_eigenvalue_after": after["min_eigenvalue"],
        "is_psd": after["is_psd"],
        "rank_after": after["rank"],
        "covariance_hash": _matrix_hash(shrunk),
        "frobenius_distance_from_sample": round(
            float(np.linalg.norm(shrunk.to_numpy() - sample.to_numpy(), "fro")), 15
        ),
    }

    return TestResult(
        test_id="covariance.ledoit_wolf_shrinkage",
        test_name="Ledoit-Wolf shrinkage covariance",
        status=Status.RECORDED if after["is_psd"] else Status.FAIL,
        params={},
        metrics=metrics,
        interpretation=(
            f"Shrinkage intensity {fitted.shrinkage_:.4f}; condition number moved "
            f"{before['condition_number']:.4g} -> {after['condition_number']:.4g} over "
            f"{returns.shape[1]} asset(s) and {returns.shape[0]:,} observation(s)."
        ),
        limitations=[
            "Shrinkage trades variance for bias. It is NOT universally superior to the "
            "sample covariance: where observations are plentiful relative to assets the "
            "sample estimate may be preferable, and this test makes no claim either way.",
            "The shrinkage target is the scaled identity used by sklearn's LedoitWolf; "
            "a different target would give a different estimate.",
            "Missing values are refused rather than imputed, so the estimand stays what the name says it is.",
            "Determinism: numerical.",
        ],
    )


# =========================================================================== #
# 3. regularized_em
# =========================================================================== #
@dataclass
class RegEMResult:
    mean: np.ndarray
    covariance: np.ndarray
    n_iterations: int
    converged: bool
    final_change: float
    n_eigenvalue_clips: int
    min_eigenvalue_before_clip: float
    min_eigenvalue_after_clip: float
    ridge_used: float
    n_patterns: int
    n_pseudoinverse_fallbacks: int
    history: list[float] = field(default_factory=list)


def run_regularized_em(
    data: np.ndarray,
    ridge: float = REGEM_DEFAULT_RIDGE,
    tol: float = REGEM_DEFAULT_TOL,
    max_iter: int = REGEM_DEFAULT_MAX_ITER,
) -> RegEMResult:
    """Regularized EM for a Gaussian mean and covariance with missing entries.

    **Deterministic initialisation**, frozen before any experiment: column means from
    observed values, and an available-case covariance with the diagonal filled from
    observed per-column variances, stabilised by ridge. No random start, so the result
    is reproducible without a seed.

    **E-step.** For each missingness pattern, partition into observed ``O`` and missing
    ``M`` and compute both conditional sufficient statistics::

        E[X_M | X_O] = mu_M + Sigma_MO Sigma_OO^-1 (x_O - mu_O)
        Cov[X_M | X_O] = Sigma_MM - Sigma_MO Sigma_OO^-1 Sigma_OM

    The scatter accumulates the conditional mean outer product **plus** the conditional
    covariance. Omitting the second term is the difference between EM and mean
    imputation: mean imputation systematically understates variance on the imputed
    entries and biases correlations toward zero, and it is easy to mistake for EM
    because the code looks similar.

    **M-step.** Mean and covariance from the expected sufficient statistics
    (ML convention, denominator n), then ridge, then the PSD safeguard.

    ``np.linalg.solve`` is used for the conditional blocks; a pseudo-inverse fallback is
    taken only on a singular block and is counted, so the numerical fallback is visible
    rather than silent.
    """
    n, p = data.shape
    observed_mask = ~np.isnan(data)

    if not observed_mask.any(axis=0).all():
        empty = [int(j) for j in range(p) if not observed_mask[:, j].any()]
        raise ValueError(f"variable(s) {empty} have no observed values")

    # -- deterministic initialisation ------------------------------------
    mu = np.array([np.nanmean(data[:, j]) for j in range(p)], dtype=float)
    centred = np.where(observed_mask, data - mu, 0.0)
    pair_counts = observed_mask.astype(float).T @ observed_mask.astype(float)
    sigma = (centred.T @ centred) / np.maximum(pair_counts, 1.0)
    sigma = (sigma + sigma.T) / 2.0
    sigma += np.eye(p) * max(ridge, PSD_EIGENVALUE_FLOOR)

    patterns: dict[tuple[bool, ...], list[int]] = {}
    for i in range(n):
        patterns.setdefault(tuple(observed_mask[i]), []).append(i)

    history: list[float] = []
    clips = 0
    fallbacks = 0
    min_before = float("inf")
    min_after = float("inf")
    converged = False
    n_iterations = 0

    for step in range(1, max_iter + 1):
        n_iterations = step
        scatter = np.zeros((p, p), dtype=float)
        total_mean = np.zeros(p, dtype=float)

        for pattern, rows in patterns.items():
            obs = np.array(pattern, dtype=bool)
            mis = ~obs
            block = data[rows]

            if not mis.any():
                filled = block
                extra = np.zeros((p, p), dtype=float)
            else:
                s_oo = sigma[np.ix_(obs, obs)]
                s_mo = sigma[np.ix_(mis, obs)]
                s_mm = sigma[np.ix_(mis, mis)]
                rhs = (block[:, obs] - mu[obs]).T
                try:
                    solved = np.linalg.solve(s_oo, rhs)
                    coefficient = np.linalg.solve(s_oo, s_mo.T).T
                except np.linalg.LinAlgError:
                    fallbacks += 1
                    inverse = np.linalg.pinv(s_oo)
                    solved = inverse @ rhs
                    coefficient = s_mo @ inverse

                conditional_mean = mu[mis][:, None] + s_mo @ solved
                # The term that makes this EM rather than mean imputation.
                conditional_cov = s_mm - coefficient @ s_mo.T
                conditional_cov = (conditional_cov + conditional_cov.T) / 2.0

                filled = block.copy()
                filled[:, mis] = conditional_mean.T
                extra = np.zeros((p, p), dtype=float)
                extra[np.ix_(mis, mis)] = conditional_cov * len(rows)

            total_mean += filled.sum(axis=0)
            scatter += filled.T @ filled + extra

        new_mu = total_mean / n
        new_sigma = scatter / n - np.outer(new_mu, new_mu)  # ML convention
        new_sigma = (new_sigma + new_sigma.T) / 2.0
        new_sigma += np.eye(p) * ridge

        eigenvalues, vectors = np.linalg.eigh(new_sigma)
        min_before = float(eigenvalues.min())
        n_clipped = int((eigenvalues < PSD_EIGENVALUE_FLOOR).sum())
        if n_clipped:
            clips += n_clipped
            eigenvalues = np.maximum(eigenvalues, PSD_EIGENVALUE_FLOOR)
            new_sigma = vectors @ np.diag(eigenvalues) @ vectors.T
            new_sigma = (new_sigma + new_sigma.T) / 2.0
        min_after = float(np.linalg.eigvalsh(new_sigma).min())

        denominator = max(float(np.linalg.norm(sigma, "fro")), 1e-300)
        change = float(np.linalg.norm(new_sigma - sigma, "fro")) / denominator
        history.append(change)
        mu, sigma = new_mu, new_sigma
        if change < tol:
            converged = True
            break

    return RegEMResult(
        mean=mu,
        covariance=sigma,
        n_iterations=n_iterations,
        converged=converged,
        final_change=history[-1] if history else float("nan"),
        n_eigenvalue_clips=clips,
        min_eigenvalue_before_clip=min_before,
        min_eigenvalue_after_clip=min_after,
        ridge_used=ridge,
        n_patterns=len(patterns),
        n_pseudoinverse_fallbacks=fallbacks,
        history=history,
    )


@register_test(
    "covariance.regularized_em",
    family="covariance",
    name="Regularized EM covariance",
    requires=("returns",),
    default_params={
        "ridge": REGEM_DEFAULT_RIDGE,
        "tol": REGEM_DEFAULT_TOL,
        "max_iterations": REGEM_DEFAULT_MAX_ITER,
    },
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification", "assumption_validity"),
    object_kinds=_OBJECTS,
)
def regularized_em(
    ctx: Any,
    ridge: float = REGEM_DEFAULT_RIDGE,
    tol: float = REGEM_DEFAULT_TOL,
    max_iterations: int = REGEM_DEFAULT_MAX_ITER,
) -> TestResult:
    """Regularized EM covariance under Gaussian/MAR working assumptions.

    Determinism: numerical, with deterministic initialisation — no seed is needed.
    """
    returns, reason = _returns_or_reason(ctx)
    if returns is None:
        return _skip("covariance.regularized_em", "Regularized EM covariance", reason, ridge=ridge)
    if ridge < 0:
        return _error(
            "covariance.regularized_em",
            "Regularized EM covariance",
            f"ridge must be non-negative; got {ridge!r}",
        )

    data = returns.to_numpy(dtype=float)
    n_missing = int(np.isnan(data).sum())
    per_column = np.isnan(data).mean(axis=0)

    try:
        outcome = run_regularized_em(data, ridge=ridge, tol=tol, max_iter=max_iterations)
    except ValueError as exc:
        return _error("covariance.regularized_em", "Regularized EM covariance", str(exc), ridge=ridge)

    matrix = pd.DataFrame(outcome.covariance, index=returns.columns, columns=returns.columns)
    conditioning = _conditioning(outcome.covariance)

    metrics: dict[str, Any] = {
        "n_assets": int(returns.shape[1]),
        "n_observations": int(returns.shape[0]),
        "n_total_values": int(data.size),
        "n_missing_values": n_missing,
        "missing_fraction": round(n_missing / data.size, 10),
        "min_column_missing_fraction": round(float(per_column.min()), 10),
        "max_column_missing_fraction": round(float(per_column.max()), 10),
        "n_complete_rows": int((~np.isnan(data)).all(axis=1).sum()),
        "n_missingness_patterns": outcome.n_patterns,
        "initialisation": "deterministic: observed column means, available-case covariance, ridge-stabilised",
        "estimand_convention": "ML covariance, denominator n (ddof=0)",
        "ddof": 0,
        "ridge_used": outcome.ridge_used,
        "ridge_selection": "supplied or module default; no GCV search is performed",
        "tolerance": tol,
        "max_iterations": max_iterations,
        "n_iterations": outcome.n_iterations,
        "converged": outcome.converged,
        "final_relative_change": round(outcome.final_change, 15),
        "n_eigenvalue_clips": outcome.n_eigenvalue_clips,
        "min_eigenvalue_before_clip": round(outcome.min_eigenvalue_before_clip, 18),
        "min_eigenvalue_after_clip": round(outcome.min_eigenvalue_after_clip, 18),
        "psd_floor": PSD_EIGENVALUE_FLOOR,
        "n_pseudoinverse_fallbacks": outcome.n_pseudoinverse_fallbacks,
        "covariance_hash": _matrix_hash(matrix),
        **conditioning,
    }

    if not outcome.converged:
        # A non-converged result reported green would be a covariance nobody should use,
        # wearing an ordinary status.
        return _error(
            "covariance.regularized_em",
            "Regularized EM covariance",
            f"EM did not converge in {max_iterations} iteration(s); final relative "
            f"change {outcome.final_change:.3g} against tolerance {tol:g}",
            metrics=metrics,
            ridge=ridge,
            tol=tol,
            max_iterations=max_iterations,
        )

    return TestResult(
        test_id="covariance.regularized_em",
        test_name="Regularized EM covariance",
        status=Status.RECORDED if conditioning["is_psd"] else Status.FAIL,
        params={"ridge": ridge, "tol": tol, "max_iterations": max_iterations},
        metrics=metrics,
        interpretation=(
            f"Converged in {outcome.n_iterations} iteration(s) over "
            f"{outcome.n_patterns} missingness pattern(s); "
            f"{metrics['missing_fraction']:.1%} of values missing; condition number "
            f"{conditioning['condition_number']:.4g}."
        ),
        limitations=[
            "A REGULARIZED COVARIANCE POINT ESTIMATOR under multivariate Gaussian and "
            "MAR working assumptions. After ridge regularisation and eigenvalue "
            "clipping it is NOT the unconstrained maximum-likelihood estimate, and it "
            "is not described as one.",
            "NO UNCERTAINTY QUANTIFICATION is produced. There are no standard errors, "
            "no intervals and no convergence diagnostics beyond the reported change.",
            "MAR is a working assumption, not a verified property of the data. Under "
            "MNAR the estimate is biased and nothing here would reveal it.",
            "The E-step includes the conditional covariance term Cov[X_M | X_O]. "
            "Mean-imputing and running an ordinary covariance is NOT EM: it understates "
            "variance on imputed entries and biases correlations toward zero.",
            "Eigenvalue clipping at the reported floor is a StART numerical safeguard, "
            "not part of Schneider's method, and the number of clips is recorded.",
            "The ridge parameter materially affects the result and is reported. No GCV "
            "search is performed; the value is supplied or the module default.",
            "Determinism: numerical, with deterministic initialisation — no seed.",
        ],
    )
