"""Portfolio analytics — five registered tests.

Everything here operates on **periodic** quantities internally — periodic expected
returns, periodic covariance, periodic risk-free — and annualises only for reporting.
Mixing an annual risk-free rate into a daily optimisation is wrong by roughly 252x and
nothing downstream would reveal it.

The max-Sharpe contract
-----------------------

:data:`MAX_SHARPE_CONTRACT` is frozen in this module as a constant, written before the
optimiser below it. An optimiser whose answer depends on an undocumented grid is not
reproducible, and "we used a reasonable grid" is not a specification. The grid, the
refinement rule and the full tie-break ordering are fixed and recorded in evidence.

Wealth construction
-------------------

Simple returns compound as ``prod(1 + r)``; log returns as ``exp(cumsum(r))``. Applying
``prod(1 + r)`` to log returns is a real and silent error — it produces a wealth path
that looks plausible and is wrong, which then flows into maximum drawdown and Calmar.
The basis is carried on the context and never inferred.

Determinism
-----------

All five are ``numerical``. Covariance, eigenvalues, SLSQP and the HRP linkage all route
through BLAS/LAPACK, which is not bitwise reproducible across platforms. Observation
counts are exact; nothing computed from them is.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from start.core.schemas import Status, TestResult, ThresholdSpec
from start.registry import register_test

__all__ = [
    "historical_returns", "risk_statistics", "covariance_conditioning",
    "mean_variance", "hierarchical_risk_parity",
    "black_litterman", "cvar_optimization", "herc",
    "maximum_diversification", "constrained_optimization",
    "MAX_SHARPE_CONTRACT", "EWMA_DECAY_DEFAULT", "CONSTRAINT_TOLERANCE",
    "DEGENERATE_SCALE_TOLERANCE", "DEGENERATE_VARIANCE_RELATIVE",
    "portfolio_wealth", "annualised_geometric_return", "max_drawdown",
    "hrp_weights", "solve_min_variance",
]

# --------------------------------------------------------------------------- #
# FROZEN CONTRACTS — recorded before the optimiser was written
# --------------------------------------------------------------------------- #
#: The complete max-Sharpe specification. Immutable for v4.2.0.
MAX_SHARPE_CONTRACT: dict[str, Any] = {
    "feasible_bound_method": (
        "two constrained solves under the EXACT active constraint set: "
        "mu_min = min_w w'mu and mu_max = max_w w'mu. Not approximated as "
        "min/max of asset means, which is wrong whenever constraints bind."
    ),
    "coarse_points": 101,
    "coarse_spacing": "equally spaced target returns on [mu_min, mu_max], endpoints included",
    "refinement_points": 21,
    "refinement_rule": (
        "one refinement only, over the interval between the coarse grid points "
        "adjacent to the best coarse Sharpe; a single adjacent interval when the best "
        "point is an endpoint; duplicates removed; no recursive refinement"
    ),
    "sharpe_basis": "periodic: (w'mu - rf_periodic) / sqrt(w'Sigma w)",
    "annualisation": "S_annual = S_periodic * sqrt(periods_per_year); ranking unchanged",
    "tie_tolerance": 1e-12,
    "tie_break_order": (
        "1. lower portfolio volatility",
        "2. lower one-way turnover vs prior_weights, when prior_weights exists",
        "3. lower target expected return",
        "4. lexicographically smaller weight vector in canonical asset order",
    ),
}

#: RiskMetrics convention. Frozen before execution.
EWMA_DECAY_DEFAULT = 0.94

#: Post-solve constraint verification tolerance. SLSQP's success flag is not trusted:
#: a nominal success with a violated constraint is a silently wrong portfolio.
CONSTRAINT_TOLERANCE = 1e-6

#: Relative tolerance for detecting a degenerate (constant) return series.
#:
#: An exact ``== 0`` test does not work. A genuinely constant series that has passed
#: through a matrix multiply has a standard deviation of order 1e-18, not zero, so an
#: exact test never fires and the ratio becomes ~1e17 — which is infinity wearing a
#: disguise, and far more dangerous than an honest `inf` because it looks like a
#: number. The comparison is relative to the series' own scale so it behaves the same
#: for returns quoted in percent as in decimals.
DEGENERATE_SCALE_TOLERANCE = 1e-12

#: Relative floor below which a variance is treated as zero, as a fraction of the
#: largest variance in the same matrix. Variances share units, so a relative test is
#: the meaningful one.
DEGENERATE_VARIANCE_RELATIVE = 1e-20

_STRIPES = ("market",)
_OBJECTS = ("deterministic_calculator", "statistical_model")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _skip(test_id: str, name: str, reason: str, **params: Any) -> TestResult:
    return TestResult(test_id=test_id, test_name=name, status=Status.SKIPPED,
                      params=params, interpretation=reason)


def _error(test_id: str, name: str, reason: str, metrics: dict[str, Any] | None = None,
           **params: Any) -> TestResult:
    return TestResult(test_id=test_id, test_name=name, status=Status.ERROR,
                      params=params, metrics=metrics or {}, interpretation=reason)


def _portfolio_returns(ctx: Any) -> tuple[pd.Series | None, str]:
    """Weighted portfolio return series, or a reason it cannot be formed.

    Equal weights are **never** invented. A portfolio review without weights is a
    review of a different portfolio, and silently substituting 1/N would produce
    confident numbers about something nobody asked about.
    """
    returns = ctx.effective_returns()
    if returns is None or returns.empty:
        return None, "no returns available (neither returns nor prices supplied)"
    if ctx.portfolio is None or ctx.portfolio.weights is None:
        return None, (
            "no portfolio weights supplied. Equal weights are not invented: a "
            "portfolio review without weights would report confident statistics about "
            "a portfolio nobody specified"
        )
    weights = ctx.portfolio.weights
    if isinstance(weights, pd.DataFrame):
        weights = weights.iloc[-1]
    aligned = weights.reindex(returns.columns)
    if aligned.isna().any():
        missing = [str(c) for c in returns.columns[aligned.isna().to_numpy()]][:5]
        return None, f"weights missing for asset(s): {', '.join(missing)}"
    return (returns @ aligned).rename("portfolio_return"), ""


def portfolio_wealth(returns: pd.Series, basis: str) -> pd.Series:
    """Wealth index under the correct compounding rule for the basis."""
    if basis == "log":
        return np.exp(returns.cumsum())
    return (1.0 + returns).cumprod()


def annualised_geometric_return(returns: pd.Series, ppy: float, basis: str) -> float:
    """Geometric annualisation, reported on a simple-equivalent scale for both bases."""
    n = int(returns.size)
    if n == 0:
        return float("nan")
    if basis == "log":
        return float(np.exp(returns.mean() * ppy) - 1.0)
    growth = float((1.0 + returns).prod())
    if growth <= 0:
        return float("nan")
    return float(growth ** (ppy / n) - 1.0)


def max_drawdown(returns: pd.Series, basis: str) -> dict[str, Any]:
    """Maximum drawdown with its dates. Recovery is ``None`` when never recovered."""
    wealth = portfolio_wealth(returns, basis)
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    trough_idx = drawdown.idxmin()
    mdd = float(drawdown.loc[trough_idx])

    peak_before = wealth.loc[:trough_idx].idxmax()
    after = wealth.loc[trough_idx:]
    peak_value = float(wealth.loc[peak_before])
    recovered = after[after >= peak_value]
    return {
        "max_drawdown": mdd,
        "drawdown_start": str(peak_before),
        "drawdown_trough": str(trough_idx),
        "drawdown_recovery": str(recovered.index[0]) if len(recovered) else None,
    }


def _excess_returns(returns: pd.Series, ctx: Any) -> tuple[pd.Series, dict[str, Any]]:
    """Periodic excess returns. The risk-free is converted to the RETURN period first."""
    rf, record = ctx.risk_free_per_period()
    if rf is None:
        return returns, record
    if isinstance(rf, pd.Series):
        return (returns - rf.reindex(returns.index).fillna(0.0)), record
    return (returns - float(rf)), record


# --------------------------------------------------------------------------- #
# 1. historical_returns
# --------------------------------------------------------------------------- #
@register_test(
    "portfolio.historical_returns",
    family="portfolio",
    name="Historical portfolio returns",
    requires=("returns",),
    default_params={},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("data_quality_lineage", "implementation_verification"),
    object_kinds=_OBJECTS,
)
def historical_returns(ctx: Any) -> TestResult:
    """Portfolio return series under the declared basis. Determinism: numerical."""
    series, reason = _portfolio_returns(ctx)
    if series is None:
        return _skip("portfolio.historical_returns", "Historical portfolio returns", reason)

    basis = ctx.return_basis
    weights = ctx.portfolio.weights
    if isinstance(weights, pd.DataFrame):
        weights = weights.iloc[-1]

    metrics: dict[str, Any] = {
        "return_basis": basis,
        "derived_from_prices": ctx.returns is None,
        "n_observations": int(series.size),
        "n_assets": int(ctx.effective_returns().shape[1]),
        "periods_per_year": float(ctx.periods_per_year),
        "mean_periodic_return": round(float(series.mean()), 10),
        "std_periodic_return": round(float(series.std(ddof=1)), 10) if series.size > 1 else 0.0,
        "min_periodic_return": round(float(series.min()), 10),
        "max_periodic_return": round(float(series.max()), 10),
        "n_missing": int(series.isna().sum()),
        "weights_sum": round(float(weights.sum()), 10),
        "portfolio_fingerprint": ctx.fingerprint()[:32],
    }
    return TestResult(
        test_id="portfolio.historical_returns",
        test_name="Historical portfolio returns",
        status=Status.RECORDED,
        params={"return_basis": basis},
        metrics=metrics,
        interpretation=(
            f"{series.size:,} periodic portfolio return(s) on a {basis} basis over "
            f"{metrics['n_assets']} asset(s)"
            + (" (derived from prices)" if metrics["derived_from_prices"] else "")
            + "."
        ),
        limitations=[
            f"Returns are on a {basis} basis. Simple and log returns are NOT "
            "interchangeable: wealth compounds as prod(1+r) for simple and "
            "exp(cumsum(r)) for log, and mixing them silently corrupts drawdown.",
            "Weights are taken as supplied; equal weights are never substituted.",
            "Numerical, not bitwise reproducible across BLAS implementations.",
        ],
    )


# --------------------------------------------------------------------------- #
# 2. risk_statistics
# --------------------------------------------------------------------------- #
@register_test(
    "portfolio.risk_statistics",
    family="portfolio",
    name="Portfolio risk statistics",
    requires=("returns",),
    default_params={"var_confidence": 0.95, "mar": 0.0, "min_tail_observations": 10},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("accuracy_calibration", "implementation_verification"),
    object_kinds=_OBJECTS,
)
def risk_statistics(
    ctx: Any,
    var_confidence: float = 0.95,
    mar: float = 0.0,
    min_tail_observations: int = 10,
) -> TestResult:
    """Return and risk statistics under the frozen Sharpe/Sortino contracts.

    Sharpe and Sortino are computed from **periodic excess returns**, never from a
    geometric CAGR: CAGR compounds and ``sigma * sqrt(T)`` does not, so their ratio is
    not the conventional estimator and is not anything else either.
    """
    series, reason = _portfolio_returns(ctx)
    if series is None:
        return _skip("portfolio.risk_statistics", "Portfolio risk statistics", reason)
    if series.size < 2:
        return _skip("portfolio.risk_statistics", "Portfolio risk statistics",
                     f"{series.size} observation(s); at least 2 required")

    basis = ctx.return_basis
    ppy = float(ctx.periods_per_year)
    excess, rf_record = _excess_returns(series, ctx)

    geo = annualised_geometric_return(series, ppy, basis)
    vol = float(series.std(ddof=1)) * math.sqrt(ppy)
    metrics: dict[str, Any] = {
        "return_basis": basis,
        "n_observations": int(series.size),
        "annualised_geometric_return": round(geo, 10),
        "annualised_volatility": round(vol, 10),
        **rf_record,
    }
    limitations = [
        f"Returns are on a {basis} basis; wealth is built with the matching "
        "compounding rule.",
        "Annualised volatility uses sqrt-of-time scaling, which assumes iid returns. "
        "Under autocorrelation it is biased with no visible symptom.",
        "Sharpe and Sortino are computed from PERIODIC EXCESS returns. A geometric "
        "CAGR is never used as the numerator: CAGR compounds and sigma*sqrt(T) does "
        "not, so their ratio is not the conventional estimator.",
        "Numerical, not bitwise reproducible across BLAS implementations.",
    ]

    excess_sd = float(excess.std(ddof=1))
    excess_scale = max(1.0, float(np.abs(excess.to_numpy(dtype=float)).max()))
    if excess_sd <= DEGENERATE_SCALE_TOLERANCE * excess_scale:
        # Not infinity. An infinite Sharpe is a degenerate input, not a good portfolio.
        metrics["sharpe_ratio"] = None
        metrics["sharpe_undefined_reason"] = "zero excess-return standard deviation"
    else:
        metrics["sharpe_ratio"] = round(
            float(excess.mean()) / excess_sd * math.sqrt(ppy), 10
        )

    mar_periodic = mar / ppy if abs(mar) > 1e-12 else 0.0
    target_relative = series - mar_periodic
    downside = np.minimum(target_relative.to_numpy(dtype=float), 0.0)
    # Averaged over ALL observations, not only the downside ones. The two conventions
    # give materially different numbers, so the choice is stated rather than assumed.
    downside_dev = float(np.sqrt(np.mean(downside**2)))
    downside_scale = max(1.0, float(np.abs(target_relative.to_numpy(dtype=float)).max()))
    metrics["downside_deviation_convention"] = (
        "mean of squared shortfalls over ALL observations"
    )
    metrics["mar_annual"] = mar
    metrics["mar_periodic"] = round(mar_periodic, 12)
    if downside_dev <= DEGENERATE_SCALE_TOLERANCE * downside_scale:
        metrics["sortino_ratio"] = None
        metrics["sortino_undefined_reason"] = "zero downside deviation"
    else:
        metrics["sortino_ratio"] = round(
            float(target_relative.mean()) / downside_dev * math.sqrt(ppy), 10
        )

    dd = max_drawdown(series, basis)
    metrics.update({k: (round(v, 10) if isinstance(v, float) else v) for k, v in dd.items()})
    if abs(dd["max_drawdown"]) <= DEGENERATE_SCALE_TOLERANCE:
        metrics["calmar_ratio"] = None
        metrics["calmar_undefined_reason"] = "zero maximum drawdown"
    else:
        metrics["calmar_ratio"] = round(geo / abs(dd["max_drawdown"]), 10)
    if dd["drawdown_recovery"] is None:
        limitations.append(
            "The maximum drawdown had not recovered by the end of the sample."
        )

    values = series.to_numpy(dtype=float)
    quantile = float(np.percentile(values, (1.0 - var_confidence) * 100.0))
    metrics["var_confidence"] = var_confidence
    metrics["historical_var"] = round(-quantile, 10)
    metrics["var_basis"] = f"{basis}_return"
    metrics["var_quantile_method"] = "linear interpolation (numpy.percentile default)"

    tail = values[values <= quantile]
    metrics["n_tail_observations"] = int(tail.size)
    if tail.size < min_tail_observations:
        # An ES from two observations is noise with a decimal point.
        metrics["historical_es"] = None
        metrics["es_insufficient_reason"] = (
            f"{tail.size} tail observation(s); {min_tail_observations} required"
        )
        limitations.append(
            f"Expected shortfall not reported: only {tail.size} observation(s) in the "
            f"tail, below the {min_tail_observations} required. A number computed from "
            "two observations would be false precision."
        )
    else:
        metrics["historical_es"] = round(-float(tail.mean()), 10)

    return TestResult(
        test_id="portfolio.risk_statistics",
        test_name="Portfolio risk statistics",
        status=Status.RECORDED,
        params={"var_confidence": var_confidence, "mar": mar,
                "min_tail_observations": min_tail_observations},
        metrics=metrics,
        interpretation=(
            f"Annualised geometric return {geo:.4%}, volatility {vol:.4%}, "
            f"maximum drawdown {dd['max_drawdown']:.4%}; "
            f"{var_confidence:.0%} historical VaR {metrics['historical_var']:.4%} on a "
            f"{basis}-return basis."
        ),
        limitations=limitations,
    )


# --------------------------------------------------------------------------- #
# 3. covariance_conditioning
# --------------------------------------------------------------------------- #
@register_test(
    "portfolio.covariance_conditioning",
    family="portfolio",
    name="Covariance conditioning",
    requires=("returns",),
    default_params={"condition_warn": 1e4, "condition_fail": 1e8},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification", "assumption_validity"),
    object_kinds=_OBJECTS,
)
def covariance_conditioning(
    ctx: Any, condition_warn: float = 1e4, condition_fail: float = 1e8
) -> TestResult:
    """Numerical health of the covariance matrix. Nothing is repaired silently.

    An optimiser handed an ill-conditioned covariance produces weights that look
    reasonable and are dominated by estimation noise in the smallest eigen-directions.
    That is invisible in the weights themselves, which is why it is measured here.
    """
    supplied = ctx.covariance is not None
    if supplied:
        matrix = ctx.covariance
        n_observations = 0
    else:
        returns = ctx.effective_returns()
        if returns is None or returns.shape[0] < 2:
            return _skip("portfolio.covariance_conditioning", "Covariance conditioning",
                         "insufficient return observations to estimate a covariance")
        matrix = returns.cov()
        n_observations = int(returns.shape[0])

    values = matrix.to_numpy(dtype=float)
    n = values.shape[0]
    symmetric = bool(np.allclose(values, values.T, atol=1e-12))
    eigenvalues = np.linalg.eigvalsh((values + values.T) / 2.0)
    min_eig, max_eig = float(eigenvalues.min()), float(eigenvalues.max())
    is_psd = bool(min_eig >= -1e-12)
    rank = int(np.linalg.matrix_rank(values, tol=1e-10))
    condition = float(max_eig / min_eig) if min_eig > 0 else float("inf")

    positive = eigenvalues[eigenvalues > 1e-15]
    normalised = positive / positive.sum() if positive.size else np.array([])
    effective_rank = (
        float(np.exp(-np.sum(normalised * np.log(normalised)))) if normalised.size else 0.0
    )

    metrics: dict[str, Any] = {
        "n_assets": n,
        "covariance_source": "supplied" if supplied else "estimated_from_returns",
        "n_observations": n_observations,
        "is_symmetric": symmetric,
        "is_psd": is_psd,
        "rank": rank,
        "full_rank": bool(rank == n),
        "effective_rank": round(effective_rank, 6),
        "min_eigenvalue": round(min_eig, 12),
        "max_eigenvalue": round(max_eig, 12),
        "condition_number": round(condition, 6) if math.isfinite(condition) else float("inf"),
        "n_negative_eigenvalues": int((eigenvalues < -1e-12).sum()),
        "repair_applied": False,
    }

    result = TestResult(
        test_id="portfolio.covariance_conditioning",
        test_name="Covariance conditioning",
        params={"condition_warn": condition_warn, "condition_fail": condition_fail},
        metrics=metrics,
        thresholds=[ThresholdSpec(metric="condition_number",
                                  warn=condition_warn, fail=condition_fail)],
        interpretation=(
            f"{n}x{n} covariance, rank {rank}"
            + ("" if rank == n else f" (deficient by {n - rank})")
            + f", condition number {condition:.4g}, "
            + ("PSD" if is_psd else "NOT PSD")
            + "."
        ),
        limitations=[
            "NO REPAIR IS APPLIED HERE. This test reports conditioning; any numerical "
            "safeguard applied downstream is recorded by the test that applies it.",
            "An ill-conditioned covariance produces optimiser weights dominated by "
            "estimation noise in the smallest eigen-directions, which is invisible in "
            "the weights themselves.",
            "Effective rank is the participation-ratio measure, not the matrix rank.",
            "Numerical, not bitwise reproducible across BLAS implementations.",
        ],
    )
    applied = result.apply_thresholds()
    if not is_psd:
        applied.status = Status.FAIL
    return applied


# --------------------------------------------------------------------------- #
# 4. mean_variance
# --------------------------------------------------------------------------- #
def _expected_returns(returns: pd.DataFrame, method: str, decay: float,
                      supplied: pd.Series | None) -> pd.Series:
    if method == "supplied":
        if supplied is None:
            raise ValueError(
                "expected_return='supplied' requires ctx.extra['expected_returns']"
            )
        return supplied.reindex(returns.columns)
    if method == "ewma":
        weights = np.array([decay**i for i in range(len(returns) - 1, -1, -1)])
        weights /= weights.sum()
        return pd.Series(weights @ returns.to_numpy(dtype=float), index=returns.columns)
    return returns.mean()


def _covariance(returns: pd.DataFrame, method: str,
                supplied: pd.DataFrame | None) -> pd.DataFrame:
    if method == "supplied":
        if supplied is None:
            raise ValueError("covariance='supplied' requires ctx.covariance")
        return supplied.reindex(index=returns.columns, columns=returns.columns)
    if method == "ledoit_wolf":
        from sklearn.covariance import LedoitWolf

        fitted = LedoitWolf().fit(returns.to_numpy(dtype=float))
        return pd.DataFrame(fitted.covariance_, index=returns.columns,
                            columns=returns.columns)
    return returns.cov()


def _constraint_list(constraints: Any, n: int, mu: np.ndarray,
                     prior: np.ndarray | None,
                     target: float | None) -> list[dict[str, Any]]:
    """SLSQP constraint dicts implementing the frozen formulas."""
    budget = constraints.budget if constraints else 1.0
    items: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda w, b=budget: float(np.sum(w) - b)}
    ]
    if target is not None:
        items.append({"type": "eq", "fun": lambda w, m=mu, t=target: float(w @ m - t)})
    if constraints is None:
        return items
    if constraints.max_leverage is not None:
        items.append({
            "type": "ineq",
            "fun": lambda w, L=constraints.max_leverage: float(L - np.sum(np.abs(w))),
        })
    if constraints.max_turnover is not None and prior is not None:
        items.append({
            "type": "ineq",
            "fun": lambda w, T=constraints.max_turnover, p=prior: float(
                T - 0.5 * np.sum(np.abs(w - p))
            ),
        })
    if constraints.max_concentration is not None:
        items.append({
            "type": "ineq",
            "fun": lambda w, H=constraints.max_concentration: float(H - np.sum(w * w)),
        })
    return items


def _bounds(constraints: Any, n: int) -> list[tuple[float | None, float | None]]:
    lower: float | None = 0.0 if (constraints is None or constraints.long_only) else None
    if constraints is not None and constraints.min_weight is not None:
        lower = constraints.min_weight
    upper = constraints.max_weight if constraints is not None else None
    return [(lower, upper)] * n


def _verify_constraints(w: np.ndarray, constraints: Any, mu: np.ndarray,
                        prior: np.ndarray | None,
                        target: float | None) -> dict[str, float]:
    """Independent post-solve verification.

    SLSQP's success flag is not trusted. A nominal success with a violated constraint
    produces a portfolio that is silently outside its mandate, and the flag alone gives
    no way to tell.
    """
    violations: dict[str, float] = {"budget": abs(
        float(np.sum(w)) - (constraints.budget if constraints else 1.0)
    )}
    if target is not None:
        violations["target_return"] = abs(float(w @ mu) - target)
    if constraints is not None:
        if constraints.long_only:
            violations["long_only"] = float(max(0.0, -w.min()))
        if constraints.min_weight is not None:
            violations["min_weight"] = float(max(0.0, constraints.min_weight - w.min()))
        if constraints.max_weight is not None:
            violations["max_weight"] = float(max(0.0, w.max() - constraints.max_weight))
        if constraints.max_leverage is not None:
            violations["max_leverage"] = float(
                max(0.0, np.sum(np.abs(w)) - constraints.max_leverage)
            )
        if constraints.max_turnover is not None and prior is not None:
            violations["max_turnover"] = float(
                max(0.0, 0.5 * np.sum(np.abs(w - prior)) - constraints.max_turnover)
            )
        if constraints.max_concentration is not None:
            violations["max_concentration"] = float(
                max(0.0, np.sum(w * w) - constraints.max_concentration)
            )
    return violations


def solve_min_variance(mu: np.ndarray, sigma: np.ndarray, constraints: Any,
                       prior: np.ndarray | None = None,
                       target: float | None = None
                       ) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Minimum variance subject to the active constraints. Deterministic start."""
    from scipy.optimize import minimize

    n = len(mu)
    start = np.full(n, (constraints.budget if constraints else 1.0) / n)
    outcome = minimize(
        lambda w: float(w @ sigma @ w),
        start,
        jac=lambda w: 2.0 * sigma @ w,
        method="SLSQP",
        bounds=_bounds(constraints, n),
        constraints=_constraint_list(constraints, n, mu, prior, target),
        options={"ftol": 1e-10, "maxiter": 500},
    )
    diagnostics: dict[str, Any] = {
        "solver_status": int(outcome.status),
        "solver_message": str(outcome.message),
        "n_iterations": int(outcome.nit),
        "objective": float(outcome.fun) if outcome.success else float("nan"),
        "converged": bool(outcome.success),
    }
    if not outcome.success:
        return None, diagnostics
    weights = np.asarray(outcome.x, dtype=float)
    violations = _verify_constraints(weights, constraints, mu, prior, target)
    diagnostics["max_constraint_violation"] = float(max(violations.values()))
    diagnostics["violations"] = violations
    return weights, diagnostics


def _sharpe(w: np.ndarray, mu: np.ndarray, sigma: np.ndarray,
            rf: float) -> tuple[float, float]:
    variance = float(w @ sigma @ w)
    scale = float(np.max(np.abs(np.diag(sigma)))) if sigma.size else 1.0
    if variance <= DEGENERATE_VARIANCE_RELATIVE * max(scale, 1e-300):
        return float("-inf"), 0.0
    volatility = math.sqrt(variance)
    return (float(w @ mu) - rf) / volatility, volatility


def _max_sharpe(mu: np.ndarray, sigma: np.ndarray, constraints: Any,
                prior: np.ndarray | None, rf: float
                ) -> tuple[np.ndarray | None, dict[str, Any], dict[str, Any]]:
    """Frontier traversal under the frozen contract."""
    from scipy.optimize import minimize

    n = len(mu)
    bounds = _bounds(constraints, n)
    base = _constraint_list(constraints, n, mu, prior, None)
    start = np.full(n, (constraints.budget if constraints else 1.0) / n)

    # Feasible bounds under the EXACT active constraints, not min/max of asset means.
    lo = minimize(lambda w: float(w @ mu), start, jac=lambda w: mu, method="SLSQP",
                  bounds=bounds, constraints=base,
                  options={"ftol": 1e-10, "maxiter": 500})
    hi = minimize(lambda w: float(-(w @ mu)), start, jac=lambda w: -mu, method="SLSQP",
                  bounds=bounds, constraints=base,
                  options={"ftol": 1e-10, "maxiter": 500})
    if not (lo.success and hi.success):
        return None, {"converged": False, "solver_status": -1, "n_iterations": 0,
                      "solver_message": "feasible expected-return bound solve failed"}, {}

    mu_min, mu_max = float(lo.x @ mu), float(hi.x @ mu)
    record: dict[str, Any] = {
        "grid_mu_min": round(mu_min, 12), "grid_mu_max": round(mu_max, 12),
        "grid_coarse_points": MAX_SHARPE_CONTRACT["coarse_points"],
        "grid_refinement_points": MAX_SHARPE_CONTRACT["refinement_points"],
        "grid_tie_tolerance": MAX_SHARPE_CONTRACT["tie_tolerance"],
    }
    if mu_max - mu_min < 1e-15:
        weights, diagnostics = solve_min_variance(mu, sigma, constraints, prior, None)
        record["grid_degenerate"] = True
        return weights, diagnostics, record

    def evaluate(targets: list[float]) -> list[dict[str, Any]]:
        found: list[dict[str, Any]] = []
        for target in targets:
            w, diag = solve_min_variance(mu, sigma, constraints, prior, target)
            if w is None:
                continue
            if diag.get("max_constraint_violation", 1.0) > CONSTRAINT_TOLERANCE:
                continue
            sharpe, volatility = _sharpe(w, mu, sigma, rf)
            if not math.isfinite(sharpe):
                continue
            turnover = float(0.5 * np.sum(np.abs(w - prior))) if prior is not None else 0.0
            found.append({"target": float(target), "weights": w, "sharpe": sharpe,
                          "volatility": volatility, "turnover": turnover, "diag": diag})
        return found

    coarse_targets = [float(x) for x in
                      np.linspace(mu_min, mu_max, MAX_SHARPE_CONTRACT["coarse_points"])]
    coarse = evaluate(coarse_targets)
    record["grid_coarse_feasible"] = len(coarse)
    record["grid_coarse_failed"] = len(coarse_targets) - len(coarse)
    if not coarse:
        return None, {"converged": False, "solver_status": -1, "n_iterations": 0,
                      "solver_message": "no feasible frontier point"}, record

    best_index = max(range(len(coarse)), key=lambda i: coarse[i]["sharpe"])
    lower = coarse[max(0, best_index - 1)]["target"]
    upper = coarse[min(len(coarse) - 1, best_index + 1)]["target"]
    refine_targets = [
        float(t) for t in np.linspace(lower, upper, MAX_SHARPE_CONTRACT["refinement_points"])
        if not any(abs(t - c["target"]) < 1e-15 for c in coarse)
    ]
    refined = evaluate(refine_targets)
    record["grid_refined_feasible"] = len(refined)

    def tie_key(candidate: dict[str, Any]) -> tuple:
        # Frozen tie-break order, applied only within the tolerance band.
        return (candidate["volatility"], candidate["turnover"], candidate["target"],
                tuple(round(float(x), 12) for x in candidate["weights"]))

    everything = coarse + refined
    best_sharpe = max(c["sharpe"] for c in everything)
    tied = [c for c in everything
            if best_sharpe - c["sharpe"] <= MAX_SHARPE_CONTRACT["tie_tolerance"]]
    record["grid_n_tied"] = len(tied)
    winner = min(tied, key=tie_key)
    record["grid_selected_target"] = round(winner["target"], 12)
    return winner["weights"], winner["diag"], record


@register_test(
    "portfolio.mean_variance",
    family="portfolio",
    name="Mean-variance optimisation",
    requires=("returns",),
    default_params={
        "objective": "min_variance", "expected_return": "mean",
        "covariance": "empirical", "ewma_decay": EWMA_DECAY_DEFAULT,
        "risk_aversion": 1.0, "target_return": None,
    },
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("conceptual_soundness", "implementation_verification"),
    object_kinds=_OBJECTS,
)
def mean_variance(
    ctx: Any,
    objective: str = "min_variance",
    expected_return: str = "mean",
    covariance: str = "empirical",
    ewma_decay: float = EWMA_DECAY_DEFAULT,
    risk_aversion: float = 1.0,
    target_return: float | None = None,
) -> TestResult:
    """Mean-variance optimisation on periodic quantities. SLSQP, no CVXPY.

    ``max_sharpe`` follows the frozen :data:`MAX_SHARPE_CONTRACT` grid: direct Sharpe
    maximisation is non-convex, so the frontier is traced and the best point selected
    under a fully specified tie-break.
    """
    if objective not in {"min_variance", "target_return", "max_sharpe", "risk_adjusted"}:
        raise ValueError(f"objective={objective!r} is not supported")

    returns = ctx.effective_returns()
    if returns is None or returns.shape[0] < 2 or returns.shape[1] < 2:
        return _skip("portfolio.mean_variance", "Mean-variance optimisation",
                     "at least 2 assets and 2 observations are required")

    constraints = ctx.portfolio.constraints if ctx.portfolio else None
    if constraints is not None:
        problems = constraints.validate()
        if problems:
            return _error("portfolio.mean_variance", "Mean-variance optimisation",
                          "infeasible constraint set: " + "; ".join(problems))

    try:
        mu_series = _expected_returns(returns, expected_return, ewma_decay,
                                      (ctx.extra or {}).get("expected_returns"))
        sigma_frame = _covariance(returns, covariance, ctx.covariance)
    except ValueError as exc:
        return _skip("portfolio.mean_variance", "Mean-variance optimisation", str(exc))

    mu = mu_series.to_numpy(dtype=float)
    sigma = sigma_frame.to_numpy(dtype=float)
    assets = list(returns.columns)
    prior = None
    if ctx.portfolio and ctx.portfolio.prior_weights is not None:
        prior = ctx.portfolio.prior_weights.reindex(assets).to_numpy(dtype=float)

    rf_periodic, rf_record = ctx.risk_free_per_period()
    rf = float(rf_periodic) if isinstance(rf_periodic, (int, float)) else 0.0

    params = {
        "objective": objective, "expected_return": expected_return,
        "covariance": covariance, "ewma_decay": ewma_decay,
        "risk_aversion": risk_aversion, "target_return": target_return,
    }
    grid_record: dict[str, Any] = {}

    if objective in {"min_variance", "target_return"}:
        target = target_return if objective == "target_return" else None
        weights, diagnostics = solve_min_variance(mu, sigma, constraints, prior, target)
    elif objective == "risk_adjusted":
        from scipy.optimize import minimize

        n = len(mu)
        start = np.full(n, (constraints.budget if constraints else 1.0) / n)
        outcome = minimize(
            lambda w: float(-(w @ mu) + 0.5 * risk_aversion * (w @ sigma @ w)),
            start, method="SLSQP", bounds=_bounds(constraints, n),
            constraints=_constraint_list(constraints, n, mu, prior, None),
            options={"ftol": 1e-10, "maxiter": 500},
        )
        diagnostics = {
            "solver_status": int(outcome.status), "solver_message": str(outcome.message),
            "n_iterations": int(outcome.nit), "converged": bool(outcome.success),
            "objective": float(outcome.fun) if outcome.success else float("nan"),
        }
        weights = np.asarray(outcome.x, dtype=float) if outcome.success else None
        if weights is not None:
            violations = _verify_constraints(weights, constraints, mu, prior, None)
            diagnostics["max_constraint_violation"] = float(max(violations.values()))
            diagnostics["violations"] = violations
    else:
        weights, diagnostics, grid_record = _max_sharpe(mu, sigma, constraints, prior, rf)

    if weights is None:
        return _error("portfolio.mean_variance", "Mean-variance optimisation",
                      f"optimiser did not converge: {diagnostics.get('solver_message', '')}",
                      metrics={**{k: v for k, v in diagnostics.items() if k != 'violations'},
                               **grid_record}, **params)

    violation = float(diagnostics.get("max_constraint_violation", 0.0))
    if violation > CONSTRAINT_TOLERANCE:
        # A nominal solver success with a violated constraint is a silently wrong
        # portfolio. The success flag alone gives no way to tell.
        return _error(
            "portfolio.mean_variance", "Mean-variance optimisation",
            f"solver reported success but post-solve verification found a constraint "
            f"violation of {violation:.3g}, above the {CONSTRAINT_TOLERANCE:g} tolerance",
            metrics={**{k: v for k, v in diagnostics.items() if k != "violations"},
                     **grid_record}, **params,
        )

    sharpe_periodic, volatility = _sharpe(weights, mu, sigma, rf)
    ppy = float(ctx.periods_per_year)
    finite_sharpe = math.isfinite(sharpe_periodic)
    metrics: dict[str, Any] = {
        **{k: v for k, v in diagnostics.items() if k != "violations"},
        **grid_record, **rf_record,
        "n_assets": len(assets),
        "expected_return_periodic": round(float(weights @ mu), 12),
        "volatility_periodic": round(volatility, 12),
        "sharpe_periodic": round(sharpe_periodic, 10) if finite_sharpe else None,
        "sharpe_annualised": (
            round(sharpe_periodic * math.sqrt(ppy), 10) if finite_sharpe else None
        ),
        "expected_return_annualised": round(float(weights @ mu) * ppy, 10),
        "volatility_annualised": round(volatility * math.sqrt(ppy), 10),
        "n_active_positions": int((np.abs(weights) > 1e-8).sum()),
        "max_weight": round(float(weights.max()), 10),
        "min_weight": round(float(weights.min()), 10),
        "gross_leverage": round(float(np.sum(np.abs(weights))), 10),
        "herfindahl": round(float(np.sum(weights * weights)), 10),
    }
    if prior is not None:
        metrics["one_way_turnover"] = round(
            float(0.5 * np.sum(np.abs(weights - prior))), 10
        )
    for asset, weight in zip(assets, weights, strict=True):
        metrics[f"weight.{asset}"] = round(float(weight), 10)

    sharpe_text = (
        f"{metrics['sharpe_annualised']:.4f}" if finite_sharpe else "undefined"
    )
    return TestResult(
        test_id="portfolio.mean_variance",
        test_name="Mean-variance optimisation",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"{objective} solution over {len(assets)} asset(s): periodic volatility "
            f"{volatility:.6g}, annualised Sharpe {sharpe_text}, "
            f"{metrics['n_active_positions']} active position(s)."
        ),
        limitations=[
            "Optimised on PERIODIC quantities; annualised figures are derived for "
            "reporting only. Mixing annual and periodic inside the optimisation would "
            "be wrong by roughly the periods-per-year factor.",
            "Every constraint is verified independently after the solve; SLSQP's "
            "success flag alone is not accepted.",
            "Mean-variance weights are highly sensitive to the expected-return "
            "estimate. This test does not establish that the inputs are reliable.",
            "Numerical, atol 1e-6 on weights; not bitwise reproducible across BLAS.",
        ],
    )


# --------------------------------------------------------------------------- #
# 5. hierarchical_risk_parity
# --------------------------------------------------------------------------- #
def hrp_weights(covariance: pd.DataFrame,
                linkage_method: str = "single") -> tuple[pd.Series, list[str]]:
    """López de Prado HRP. Returns weights and the quasi-diagonal order."""
    from start.portfolio.hrp import hrp_weights_and_tree

    weights_series, tree_res = hrp_weights_and_tree(covariance, linkage_method=linkage_method)
    return weights_series, list(tree_res.quasi_diagonal_order)


@register_test(
    "portfolio.hierarchical_risk_parity",
    family="portfolio",
    name="Hierarchical risk parity",
    requires=("returns",),
    default_params={"linkage_method": "single"},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("benchmarking", "conceptual_soundness"),
    object_kinds=_OBJECTS,
)
def hierarchical_risk_parity(ctx: Any, linkage_method: str = "single") -> TestResult:
    """HRP allocation (López de Prado 2016). Determinism: numerical.

    Requires no matrix inversion, which is the point: it degrades gracefully where
    mean-variance becomes noise-dominated on an ill-conditioned covariance.
    """
    if ctx.covariance is not None:
        matrix = ctx.covariance
    else:
        returns = ctx.effective_returns()
        if returns is None or returns.shape[0] < 2:
            return _skip("portfolio.hierarchical_risk_parity", "Hierarchical risk parity",
                         "insufficient return observations to estimate a covariance")
        matrix = returns.cov()

    try:
        from start.portfolio.hrp import hrp_weights_and_tree

        weights, tree_result = hrp_weights_and_tree(matrix, linkage_method=linkage_method)
        order = list(tree_result.quasi_diagonal_order)
    except ValueError as exc:
        return _error("portfolio.hierarchical_risk_parity", "Hierarchical risk parity",
                      str(exc), linkage_method=linkage_method)

    values = weights.to_numpy(dtype=float)
    cov_arr = (
        matrix.to_numpy(dtype=float)
        if isinstance(matrix, pd.DataFrame)
        else np.asarray(matrix, dtype=float)
    )
    variance = float(values @ cov_arr @ values)
    metrics: dict[str, Any] = {
        "n_assets": int(len(weights)),
        "linkage_method": linkage_method,
        "quasi_diagonal_order": ", ".join(map(str, order)),
        "weights_sum": round(float(values.sum()), 10),
        "max_weight": round(float(values.max()), 10),
        "min_weight": round(float(values.min()), 10),
        "herfindahl": round(float(np.sum(values**2)), 10),
        "effective_n_positions": round(float(1.0 / np.sum(values**2)), 6),
        "portfolio_variance_periodic": round(variance, 14),
    }
    if tree_result.cophenetic_correlation is not None:
        metrics["cophenetic_correlation"] = tree_result.cophenetic_correlation

    for asset, weight in weights.items():
        metrics[f"weight.{asset}"] = round(float(weight), 10)

    return TestResult(
        test_id="portfolio.hierarchical_risk_parity",
        test_name="Hierarchical risk parity",
        status=Status.RECORDED,
        params={"linkage_method": linkage_method},
        metrics=metrics,
        interpretation=(
            f"HRP allocation over {len(weights)} asset(s) using {linkage_method} "
            f"linkage; effective positions {metrics['effective_n_positions']:.2f}, "
            f"largest weight {metrics['max_weight']:.4f}."
        ),
        limitations=[
            "HRP requires no matrix inversion, so it degrades gracefully where "
            "mean-variance becomes noise-dominated. It does not use expected returns "
            "at all and therefore makes no claim about expected performance.",
            "The allocation depends on the linkage method; single linkage is the "
            "published default and is recorded in evidence.",
            "The quasi-diagonal order follows scipy's linkage output, which fixes the "
            "tie-break; it is recorded so the ordering is auditable.",
            "Numerical, not bitwise reproducible across BLAS implementations.",
        ],
    )


# --------------------------------------------------------------------------- #
# 6. black_litterman
# --------------------------------------------------------------------------- #
@register_test(
    "portfolio.black_litterman",
    family="portfolio",
    name="Black-Litterman Bayesian portfolio optimization",
    requires=("covariance",),
    default_params={"risk_aversion": 3.0, "tau": 0.05},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification", "conceptual_soundness"),
    object_kinds=_OBJECTS,
)
def black_litterman(
    ctx: Any,
    risk_aversion: float = 3.0,
    tau: float = 0.05,
    **params: Any,
) -> TestResult:
    """Black-Litterman model combining market equilibrium with views. Determinism: numerical."""
    from start.portfolio.black_litterman import solve_black_litterman

    matrix = ctx.covariance if ctx.covariance is not None else (
        ctx.effective_returns().cov() if ctx.effective_returns() is not None else None
    )
    if matrix is None or (isinstance(matrix, pd.DataFrame) and matrix.shape[0] < 2):
        return _skip("portfolio.black_litterman", "Black-Litterman Bayesian portfolio optimization",
                     "insufficient covariance data")

    assets = (
        list(matrix.columns)
        if isinstance(matrix, pd.DataFrame)
        else [f"A{i}" for i in range(len(matrix))]
    )
    n = len(assets)

    # Market / reference weights
    market_w = None
    if ctx.portfolio is not None:
        if ctx.portfolio.benchmark_weights is not None:
            market_w = ctx.portfolio.benchmark_weights
        elif ctx.portfolio.weights is not None:
            market_w = ctx.portfolio.weights

    if market_w is None and "market_weights" in ctx.extra:
        market_w = ctx.extra["market_weights"]

    if market_w is None:
        return _skip("portfolio.black_litterman", "Black-Litterman Bayesian portfolio optimization",
                     "market/benchmark weights required for Black-Litterman equilibrium baseline")

    # Views P and Q
    bl_views = ctx.extra.get("bl_views") or ctx.extra.get("views")
    if bl_views is None and "P" in ctx.extra and "Q" in ctx.extra:
        bl_views = {"P": ctx.extra["P"], "Q": ctx.extra["Q"], "Omega": ctx.extra.get("Omega")}

    if bl_views is None:
        return _skip("portfolio.black_litterman", "Black-Litterman Bayesian portfolio optimization",
                     "no investor views (P, Q) supplied; views are never invented")

    P = bl_views["P"]
    Q = bl_views["Q"]
    Omega = bl_views.get("Omega")
    view_labels = bl_views.get("labels")

    constraints = ctx.portfolio.constraints if ctx.portfolio is not None else None
    prior_w = ctx.portfolio.prior_weights if ctx.portfolio is not None else None

    try:
        bl_res = solve_black_litterman(
            covariance=matrix,
            market_weights=market_w,
            P=P,
            Q=Q,
            Omega=Omega,
            risk_aversion=risk_aversion,
            tau=tau,
            assets=assets,
            view_labels=view_labels,
            constraints=constraints,
            prior_weights=prior_w,
            periods_per_year=ctx.periods_per_year,
        )
    except Exception as exc:
        return _error(
            "portfolio.black_litterman",
            "Black-Litterman Bayesian portfolio optimization",
            str(exc),
        )

    metrics: dict[str, Any] = {
        "n_assets": n,
        "n_views": len(bl_res.view_labels),
        "risk_aversion": bl_res.risk_aversion,
        "tau": bl_res.tau,
        "posterior_volatility_annualised": bl_res.posterior_volatility_annualised,
        "posterior_sharpe_annualised": bl_res.posterior_sharpe_annualised,
        "turnover_vs_prior": bl_res.turnover_vs_prior,
        "max_constraint_violation": bl_res.constraint_verification.max_violation,
    }
    for a, ret in bl_res.posterior_returns.items():
        metrics[f"posterior_return.{a}"] = ret
    for a, w in bl_res.posterior_weights.items():
        metrics[f"weight.{a}"] = w

    return TestResult(
        test_id="portfolio.black_litterman",
        test_name="Black-Litterman Bayesian portfolio optimization",
        status=Status.RECORDED,
        params={"risk_aversion": risk_aversion, "tau": tau, "n_views": len(bl_res.view_labels)},
        metrics=metrics,
        interpretation=(
            f"Black-Litterman Bayesian portfolio solved with {len(bl_res.view_labels)} views. "
            f"Annualized volatility: {bl_res.posterior_volatility_annualised:.2%}, "
            f"Turnover vs prior: {bl_res.turnover_vs_prior:.2%}."
        ),
        limitations=[
            "Equilibrium prior is anchored on CAPM market weights.",
            "Subjective views are user-supplied inputs; zero views invented.",
        ],
    )


# --------------------------------------------------------------------------- #
# 7. cvar_optimization
# --------------------------------------------------------------------------- #
@register_test(
    "portfolio.cvar_optimization",
    family="portfolio",
    name="Rockafellar-Uryasev CVaR optimization",
    requires=("returns",),
    default_params={"confidence_level": 0.95},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification", "conceptual_soundness"),
    object_kinds=_OBJECTS,
)
def cvar_optimization(
    ctx: Any,
    confidence_level: float = 0.95,
    target_return: float | None = None,
    **params: Any,
) -> TestResult:
    """Rockafellar-Uryasev CVaR Linear Programming optimization. Determinism: exact."""
    from start.portfolio.cvar import solve_cvar_portfolio

    returns = ctx.effective_returns()
    if returns is None or len(returns) < 10:
        return _skip("portfolio.cvar_optimization", "Rockafellar-Uryasev CVaR optimization",
                     "insufficient return observations for non-parametric CVaR scenario LP")

    constraints = ctx.portfolio.constraints if ctx.portfolio is not None else None
    prior_w = ctx.portfolio.prior_weights if ctx.portfolio is not None else None

    try:
        cvar_res = solve_cvar_portfolio(
            scenario_returns=returns,
            confidence_level=confidence_level,
            target_return=target_return,
            constraints=constraints,
            prior_weights=prior_w,
            periods_per_year=ctx.periods_per_year,
        )
    except Exception as exc:
        return _error("portfolio.cvar_optimization", "Rockafellar-Uryasev CVaR optimization", str(exc))

    metrics: dict[str, Any] = {
        "confidence_level": cvar_res.confidence_level,
        "cvar_annualised": cvar_res.cvar_annualised,
        "var_auxiliary_annualised": cvar_res.var_auxiliary_annualised,
        "tail_scenario_count": cvar_res.tail_scenario_count,
        "n_scenarios": cvar_res.n_scenarios,
        "nominal_tail_scenarios": int(round((1.0 - cvar_res.confidence_level) * cvar_res.n_scenarios)),
        "effective_n_positions": cvar_res.effective_n_positions,
        "expected_return_annualised": cvar_res.expected_return_annualised,
        "max_constraint_violation": cvar_res.constraint_verification.max_violation,
    }
    for a, w in cvar_res.weights.items():
        metrics[f"weight.{a}"] = w

    cvar_ann_str = f"{cvar_res.cvar_annualised:.2%}" if cvar_res.cvar_annualised is not None else "N/A"
    return TestResult(
        test_id="portfolio.cvar_optimization",
        test_name="Rockafellar-Uryasev CVaR optimization",
        status=Status.RECORDED,
        params={"confidence_level": confidence_level, "n_scenarios": cvar_res.n_scenarios},
        metrics=metrics,
        interpretation=(
            f"Minimum CVaR portfolio at {confidence_level:.1%} confidence level over "
            f"{cvar_res.n_scenarios} scenarios. Annualized CVaR: {cvar_ann_str}, "
            f"Tail scenarios: {cvar_res.tail_scenario_count}."
        ),
        limitations=[
            "Rockafellar-Uryasev LP formulation on empirical scenarios.",
            "Does not assume normality; tail support requires adequate historical sample length.",
        ],
    )


# --------------------------------------------------------------------------- #
# 8. herc
# --------------------------------------------------------------------------- #
@register_test(
    "portfolio.herc",
    family="portfolio",
    name="Hierarchical equal risk contribution",
    requires=("covariance",),
    default_params={"linkage_method": "single"},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification", "conceptual_soundness"),
    object_kinds=_OBJECTS,
)
def herc(
    ctx: Any,
    linkage_method: str = "single",
    **params: Any,
) -> TestResult:
    """Hierarchical Equal Risk Contribution (HERC) portfolio allocation. Determinism: numerical."""
    from start.portfolio.herc import solve_herc

    matrix = ctx.covariance if ctx.covariance is not None else (
        ctx.effective_returns().cov() if ctx.effective_returns() is not None else None
    )
    if matrix is None or (isinstance(matrix, pd.DataFrame) and matrix.shape[0] < 2):
        return _skip("portfolio.herc", "Hierarchical equal risk contribution",
                     "insufficient covariance data")

    try:
        herc_res = solve_herc(matrix, linkage_method=linkage_method, periods_per_year=ctx.periods_per_year)
    except Exception as exc:
        return _error("portfolio.herc", "Hierarchical equal risk contribution", str(exc))

    metrics: dict[str, Any] = {
        "n_assets": len(herc_res.weights),
        "linkage_method": linkage_method,
        "effective_n_positions": herc_res.effective_n_positions,
        "portfolio_volatility_annualised": herc_res.portfolio_volatility_annualised,
        "portfolio_variance": herc_res.portfolio_variance,
    }
    for a, w in herc_res.weights.items():
        metrics[f"weight.{a}"] = w

    return TestResult(
        test_id="portfolio.herc",
        test_name="Hierarchical equal risk contribution",
        status=Status.RECORDED,
        params={"linkage_method": linkage_method},
        metrics=metrics,
        interpretation=(
            f"HERC allocation computed with {linkage_method} linkage over {len(herc_res.weights)} assets. "
            f"Effective positions: {herc_res.effective_n_positions:.2f}, "
            f"Annualized volatility: {herc_res.portfolio_volatility_annualised:.2%}."
        ),
        limitations=[
            "Applies equal risk contribution budgeting across hierarchical clusters (Raffinot, 2018).",
            "Distance matrix uses angular correlation distance.",
        ],
    )


# --------------------------------------------------------------------------- #
# 9. maximum_diversification
# --------------------------------------------------------------------------- #
@register_test(
    "portfolio.maximum_diversification",
    family="portfolio",
    name="Maximum diversification portfolio",
    requires=("covariance",),
    default_params={},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification", "conceptual_soundness"),
    object_kinds=_OBJECTS,
)
def maximum_diversification(
    ctx: Any,
    **params: Any,
) -> TestResult:
    """Maximum Diversification Portfolio (MDP) maximizing diversification ratio. Determinism: numerical."""
    from start.portfolio.max_diversification import solve_max_diversification

    matrix = ctx.covariance if ctx.covariance is not None else (
        ctx.effective_returns().cov() if ctx.effective_returns() is not None else None
    )
    if matrix is None or (isinstance(matrix, pd.DataFrame) and matrix.shape[0] < 2):
        return _skip("portfolio.maximum_diversification", "Maximum diversification portfolio",
                     "insufficient covariance data")

    constraints = ctx.portfolio.constraints if ctx.portfolio is not None else None
    prior_w = ctx.portfolio.prior_weights if ctx.portfolio is not None else None

    try:
        md_res = solve_max_diversification(
            matrix,
            constraints=constraints,
            prior_weights=prior_w,
            periods_per_year=ctx.periods_per_year,
        )
    except Exception as exc:
        return _error("portfolio.maximum_diversification", "Maximum diversification portfolio", str(exc))

    metrics: dict[str, Any] = {
        "diversification_ratio": md_res.diversification_ratio,
        "weighted_asset_volatility_annualised": md_res.weighted_asset_volatility_annualised,
        "portfolio_volatility_annualised": md_res.portfolio_volatility_annualised,
        "effective_n_positions": md_res.effective_n_positions,
        "max_constraint_violation": md_res.constraint_verification.max_violation,
    }
    for a, w in md_res.weights.items():
        metrics[f"weight.{a}"] = w

    return TestResult(
        test_id="portfolio.maximum_diversification",
        test_name="Maximum diversification portfolio",
        status=Status.RECORDED,
        params={},
        metrics=metrics,
        interpretation=(
            f"Maximum Diversification Portfolio solved. Ratio: {md_res.diversification_ratio:.4f}, "
            f"Portfolio volatility: {md_res.portfolio_volatility_annualised:.2%}."
        ),
        limitations=[
            "Maximizes ratio of weighted individual asset volatilities to total portfolio volatility.",
            "Does not use expected return inputs.",
        ],
    )


# --------------------------------------------------------------------------- #
# 10. constrained_optimization
# --------------------------------------------------------------------------- #
@register_test(
    "portfolio.constrained_optimization",
    family="portfolio",
    name="Constrained institutional portfolio optimization",
    requires=("covariance",),
    default_params={},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification", "conceptual_soundness"),
    object_kinds=_OBJECTS,
)
def constrained_optimization(
    ctx: Any,
    **params: Any,
) -> TestResult:
    """Audit of portfolio allocation under institutional constraints. Determinism: numerical."""
    from start.portfolio.constraints import verify_portfolio_constraints

    if ctx.portfolio is None or ctx.portfolio.weights is None:
        return _skip("portfolio.constrained_optimization", "Constrained institutional portfolio optimization",
                     "no portfolio weights supplied")

    matrix = ctx.covariance if ctx.covariance is not None else (
        ctx.effective_returns().cov() if ctx.effective_returns() is not None else None
    )
    if isinstance(matrix, pd.DataFrame):
        assets = list(matrix.columns)
    elif isinstance(ctx.portfolio.weights, dict):
        assets = list(ctx.portfolio.weights.keys())
    else:
        assets = list(ctx.portfolio.weights.index)

    constraints = ctx.portfolio.constraints
    prior_w = ctx.portfolio.prior_weights
    bm_w = ctx.portfolio.benchmark_weights

    if isinstance(matrix, pd.DataFrame):
        cov_arr = matrix.to_numpy(dtype=float)
    elif matrix is not None:
        cov_arr = np.asarray(matrix, dtype=float)
    else:
        cov_arr = None

    ver_res = verify_portfolio_constraints(
        weights=ctx.portfolio.weights,
        assets=assets,
        constraints=constraints,
        covariance=cov_arr,
        benchmark_weights=bm_w,
        prior_weights=prior_w,
    )

    metrics: dict[str, Any] = {
        "is_valid": ver_res.is_valid,
        "max_violation": ver_res.max_violation,
        "total_checks": ver_res.summary.get("total_checks", len(ver_res.violations)),
        "satisfied_checks": ver_res.summary.get("satisfied_checks", 0),
        "violated_checks": ver_res.summary.get("violated_checks", 0),
    }

    return TestResult(
        test_id="portfolio.constrained_optimization",
        test_name="Constrained institutional portfolio optimization",
        status=Status.RECORDED,
        params={"tolerance": ver_res.tolerance},
        metrics=metrics,
        interpretation=(
            f"Constraint audit completed: {'VALID' if ver_res.is_valid else 'VIOLATED'}. "
            f"{ver_res.summary.get('satisfied_checks', 0)} of "
            f"{ver_res.summary.get('total_checks', 0)} checks satisfied. "
            f"Max violation: {ver_res.max_violation:.8f}."
        ),
        limitations=[
            "Independent deterministic post-solve constraint audit.",
        ],
    )
