"""Performance and risk attribution — six registered tests.

Terminology
-----------

This module implements a **cross-sectional factor model**: for each period, asset
returns are regressed on *observed* exposures to recover period factor returns. The
time-series statistics reported on those estimates are called **cross-sectional
factor-return statistics**.

They are deliberately **not** called Fama–MacBeth. Fama–MacBeth is a two-pass procedure
whose first pass estimates betas from time-series regressions and whose second pass
interprets the cross-sectional coefficients as risk premia. No two-pass procedure is
implemented here, no risk premium is estimated, and no Newey–West or other HAC
correction is applied. Borrowing the name would imply all three.

The two-state problem
---------------------

``risk_change_decomposition`` needs a *prior* state as well as the current one.
Inspection of the live ``MarketContext.fingerprint()`` confirmed it does **not**
canonicalise ``extra``. So routing a prior state through ``ctx.extra`` would make a
material analytical input invisible to evidence identity: two reviews comparing against
completely different prior states would produce the same input hash.

:class:`AttributionState` is the narrow alternative. It carries only what the
decomposition needs — exposure vector, factor covariance, specific variance — plus its
own canonical hash. The **hash string** is recorded in ``params``, which the existing
parameter hashing represents canonically, so changing either state changes evidence
identity. No DataFrame is passed through a hashed field, and ``EvidenceRecord`` is
untouched.

Determinism
-----------

All six are ``numerical``: ``lstsq``, covariance estimation and floating composition all
route through BLAS/LAPACK. Counts and hashes inside the results are exact components,
but no surface as a whole is bitwise reproducible across platforms.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from start.core.schemas import Status, TestResult
from start.registry import register_test
from start.registry.market_contexts import canonical_frame_bytes, canonical_series_bytes

__all__ = [
    "factor_return_estimation", "cross_sectional_factor_model", "exposure_analysis",
    "return_attribution", "risk_attribution", "risk_change_decomposition",
    "AttributionState", "estimate_factor_returns", "align_universe",
    "decompose_risk_change", "interaction_share",
    "RECONCILIATION_ATOL", "RECONCILIATION_RTOL", "INTERACTION_SHARE_WARN",
    "DEGENERATE_SCALE_TOLERANCE", "RISK_CHANGE_CONTRACT",
]

# --------------------------------------------------------------------------- #
# FROZEN CONTRACTS — recorded before any fixture was run
# --------------------------------------------------------------------------- #
#: Reconciliation tolerances. Frozen before implementation; never tuned to fixtures.
RECONCILIATION_ATOL = 1e-10
RECONCILIATION_RTOL = 1e-8

#: Bounded interaction share above which the decomposition WARNs, independently of
#: whether the algebra reconciles.
INTERACTION_SHARE_WARN = 0.20

#: Scale-relative degeneracy tolerance, carrying the B3 lesson: a numerically constant
#: series has a standard deviation of order 1e-18, not zero, so an exact `<= 0` test
#: never fires and the t-statistic becomes ~1e17 — infinity wearing a disguise.
DEGENERATE_SCALE_TOLERANCE = 1e-12

#: The exact risk-change decomposition. V = x'Fx + S, x1 = x0 + dx, F1 = F0 + dF,
#: S1 = S0 + dS. For symmetric F these four components sum to dV exactly.
RISK_CHANGE_CONTRACT: dict[str, str] = {
    "variance": "V = x' F x + S",
    "exposure_component": "2 * x0' F0 dx + dx' F0 dx",
    "covariance_component": "x0' dF x0",
    "specific_component": "dS",
    "interaction_component": "2 * x0' dF dx + dx' dF dx",
    "identity": "dV = exposure + covariance + specific + interaction, exactly, for symmetric F",
    "interaction_share": (
        "|interaction| / (|exposure| + |covariance| + |specific| + |interaction|); "
        "bounded in [0,1]; 0 when the denominator is numerically zero. Deliberately "
        "NOT divided by signed dV, which explodes when components cancel."
    ),
    "ordering": "simultaneous, not sequential — no order-dependent attribution",
}

_STRIPES = ("market", "valuation")
_OBJECTS = ("statistical_model", "deterministic_calculator")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _skip(test_id: str, name: str, reason: str, **params: Any) -> TestResult:
    return TestResult(test_id=test_id, test_name=name, status=Status.SKIPPED,
                      params=params, interpretation=reason)


def _error(test_id: str, name: str, reason: str, metrics: dict[str, Any] | None = None,
           **params: Any) -> TestResult:
    return TestResult(test_id=test_id, test_name=name, status=Status.ERROR,
                      params=params, metrics=metrics or {}, interpretation=reason)


def _within_tolerance(error: float, scale: float) -> bool:
    return abs(error) <= RECONCILIATION_ATOL + RECONCILIATION_RTOL * abs(scale)


@dataclass(frozen=True)
class Universe:
    """The canonical alignment for one attribution computation."""

    assets: tuple[str, ...]
    factors: tuple[str, ...]
    excluded_assets: tuple[str, ...]
    excluded_factors: tuple[str, ...]
    exclusion_rule: str


def align_universe(
    returns: pd.DataFrame,
    exposures: pd.DataFrame,
    weights: pd.Series | None = None,
    benchmark: pd.Series | None = None,
) -> Universe:
    """Deterministic canonical asset and factor ordering.

    Ordering is **sorted**, never insertion order. Two reviews handed the same data in
    different column orders must produce identical results, and relying on insertion
    order would make that quietly false.

    The documented exclusion rule: an asset is used only if it appears in the returns
    frame, the exposure rows **and** the weight vector when weights are supplied. An
    asset present in returns but absent from exposures cannot be attributed at all, so
    it is excluded and recorded rather than imputed with zero exposure — a fabricated
    zero would silently move that asset's whole return into the specific term.
    """
    if returns.columns.has_duplicates:
        raise ValueError("duplicate asset labels in returns")
    if exposures.index.has_duplicates:
        raise ValueError("duplicate asset labels in exposure rows")
    if exposures.columns.has_duplicates:
        raise ValueError("duplicate factor labels in exposure columns")

    return_assets = set(map(str, returns.columns))
    exposure_assets = set(map(str, exposures.index))
    usable = return_assets & exposure_assets
    if weights is not None:
        if weights.index.has_duplicates:
            raise ValueError("duplicate asset labels in portfolio weights")
        usable &= set(map(str, weights.index))
    if benchmark is not None:
        if benchmark.index.has_duplicates:
            raise ValueError("duplicate asset labels in benchmark weights")

    if not usable:
        raise ValueError(
            "no asset appears in returns, exposures and weights simultaneously"
        )

    assets = tuple(sorted(usable))
    factors = tuple(sorted(map(str, exposures.columns)))
    excluded = tuple(sorted((return_assets | exposure_assets) - usable))
    return Universe(
        assets=assets, factors=factors, excluded_assets=excluded, excluded_factors=(),
        exclusion_rule=(
            "an asset is used only when present in returns, exposure rows and (when "
            "supplied) the weight vector; missing exposures are never imputed as zero, "
            "which would move the asset's entire return into the specific term"
        ),
    )


def _exposures_for(ctx: Any, timestamp: Any, universe: Universe) -> pd.DataFrame | None:
    """Canonical exposure matrix at ``timestamp``, aligned. Never forward-filled."""
    frame = ctx.exposures_at(timestamp)
    if frame is None:
        return None
    return frame.reindex(index=list(universe.assets), columns=list(universe.factors))


# --------------------------------------------------------------------------- #
# Estimation core — shared by every surface that needs factor returns
# --------------------------------------------------------------------------- #
@dataclass
class EstimationOutcome:
    factor_returns: pd.DataFrame        # periods x factors
    specific_returns: pd.DataFrame      # periods x assets
    diagnostics: pd.DataFrame           # per-period rank, condition, residual norm
    universe: Universe
    n_periods_total: int
    n_periods_estimated: int
    n_periods_skipped_rank: int
    n_periods_skipped_missing: int
    weighting: str


def estimate_factor_returns(
    ctx: Any,
    observation_weights: pd.Series | None = None,
) -> EstimationOutcome:
    """Period-by-period cross-sectional weighted least squares.

    ``r_t = X_t f_t + e_t`` solved as ``lstsq(sqrt(W) X, sqrt(W) r)``. The normal
    equations ``(X'WX)^-1`` are never formed explicitly: squaring the condition number
    is exactly what destroys accuracy on the near-collinear exposure matrices this is
    most often applied to.

    **No intercept is added.** If a market factor is wanted it must be an explicit
    exposure column, so every factor in the output corresponds to a column someone
    deliberately supplied.
    """
    returns = ctx.effective_returns()
    if returns is None or returns.empty:
        raise ValueError("no returns available")
    if not ctx._exposures_canonical:
        raise ValueError("no factor exposures supplied")

    sample = next(iter(ctx._exposures_canonical.values()))
    weights_series = None
    if ctx.portfolio is not None and ctx.portfolio.weights is not None:
        weights_series = ctx.portfolio.weights
        if isinstance(weights_series, pd.DataFrame):
            weights_series = weights_series.iloc[-1]
    universe = align_universe(returns, sample, None)

    assets = list(universe.assets)
    factors = list(universe.factors)

    if observation_weights is not None:
        aligned = observation_weights.reindex(assets)
        if aligned.isna().any():
            raise ValueError("observation weights are not aligned to the asset universe")
        values = aligned.to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("observation weights contain non-finite values")
        if np.any(values <= 0):
            # A zero or negative weight is not a down-weighting, it is an undefined
            # sqrt in the transform and a silently different estimator.
            raise ValueError("observation weights must be strictly positive")
        weighting = "supplied"
        root_w = np.sqrt(values)
    else:
        weighting = "unit"
        root_w = np.ones(len(assets), dtype=float)

    factor_rows: dict[Any, np.ndarray] = {}
    specific_rows: dict[Any, np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    skipped_rank = 0
    skipped_missing = 0

    for timestamp in returns.index:
        exposures = _exposures_for(ctx, timestamp, universe)
        if exposures is None or exposures.isna().to_numpy().any():
            skipped_missing += 1
            continue
        r = returns.loc[timestamp, assets].to_numpy(dtype=float)
        if not np.all(np.isfinite(r)):
            skipped_missing += 1
            continue

        X = exposures.to_numpy(dtype=float)
        rank = int(np.linalg.matrix_rank(X, tol=1e-10))
        if len(assets) < len(factors) or rank < len(factors):
            # An underidentified cross-section produces coefficients that are not
            # determined by the data. Reporting them as factor returns would be a
            # confident number about something the period cannot answer.
            skipped_rank += 1
            diagnostics.append({
                "timestamp": str(timestamp), "rank": rank, "n_assets": len(assets),
                "n_factors": len(factors), "estimated": False,
                "reason": "rank deficient" if rank < len(factors) else "underidentified",
            })
            continue

        Xw = X * root_w[:, None]
        rw = r * root_w
        solution, _residuals, lstsq_rank, singular = np.linalg.lstsq(Xw, rw, rcond=None)
        condition = (
            float(singular.max() / singular.min())
            if singular.size and float(singular.min()) > 0 else float("inf")
        )
        residual = r - X @ solution
        factor_rows[timestamp] = solution
        specific_rows[timestamp] = residual
        diagnostics.append({
            "timestamp": str(timestamp), "rank": int(lstsq_rank),
            "n_assets": len(assets), "n_factors": len(factors), "estimated": True,
            "condition_number": condition,
            "residual_norm": float(np.linalg.norm(residual)),
        })

    return EstimationOutcome(
        factor_returns=pd.DataFrame.from_dict(factor_rows, orient="index", columns=factors),
        specific_returns=pd.DataFrame.from_dict(specific_rows, orient="index", columns=assets),
        diagnostics=pd.DataFrame(diagnostics),
        universe=universe,
        n_periods_total=int(len(returns.index)),
        n_periods_estimated=len(factor_rows),
        n_periods_skipped_rank=skipped_rank,
        n_periods_skipped_missing=skipped_missing,
        weighting=weighting,
    )


def _estimation_metrics(outcome: EstimationOutcome) -> dict[str, Any]:
    """Scalar summaries and content hashes. **No matrices.**"""
    diagnostics = outcome.diagnostics
    estimated = diagnostics[diagnostics.get("estimated", pd.Series(dtype=bool))] \
        if not diagnostics.empty and "estimated" in diagnostics else pd.DataFrame()
    conditions = (
        estimated["condition_number"].replace([np.inf], np.nan).dropna()
        if "condition_number" in estimated else pd.Series(dtype=float)
    )
    return {
        "n_periods_total": outcome.n_periods_total,
        "n_periods_estimated": outcome.n_periods_estimated,
        "n_periods_skipped_rank": outcome.n_periods_skipped_rank,
        "n_periods_skipped_missing": outcome.n_periods_skipped_missing,
        "n_assets": len(outcome.universe.assets),
        "n_factors": len(outcome.universe.factors),
        "factor_names": ", ".join(outcome.universe.factors),
        "n_assets_excluded": len(outcome.universe.excluded_assets),
        "excluded_assets": ", ".join(outcome.universe.excluded_assets[:20]),
        "weighting": outcome.weighting,
        "max_condition_number": round(float(conditions.max()), 6) if len(conditions) else None,
        "mean_abs_residual": (
            round(float(np.abs(outcome.specific_returns.to_numpy()).mean()), 12)
            if not outcome.specific_returns.empty else None
        ),
        "factor_returns_hash": hashlib.sha256(
            canonical_frame_bytes(outcome.factor_returns, label="factor_returns")
        ).hexdigest()[:32],
        "specific_returns_hash": hashlib.sha256(
            canonical_frame_bytes(outcome.specific_returns, label="specific_returns")
        ).hexdigest()[:32],
        "intercept_added": False,
    }


_ESTIMATION_LIMITATIONS = [
    "Exposures are OBSERVED characteristics, not causal effects. A factor return is a "
    "cross-sectional regression coefficient, not a structural risk premium.",
    "No intercept is added. A market factor must be supplied as an explicit exposure "
    "column so every reported factor corresponds to one someone deliberately defined.",
    "Solved via lstsq on the sqrt-weighted system; the normal equations are never "
    "formed, because squaring the condition number is what destroys accuracy on "
    "near-collinear exposures.",
    "Rank-deficient and underidentified periods are SKIPPED and counted, never "
    "reported as ordinary estimates.",
    "Numerical, not bitwise reproducible across BLAS implementations.",
]


# --------------------------------------------------------------------------- #
# 1. factor_return_estimation
# --------------------------------------------------------------------------- #
@register_test(
    "attribution.factor_return_estimation",
    family="attribution",
    name="Cross-sectional factor return estimation",
    requires=("returns", "factor_exposures"),
    default_params={},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification", "conceptual_soundness"),
    object_kinds=_OBJECTS,
)
def factor_return_estimation(ctx: Any) -> TestResult:
    """Period-by-period cross-sectional WLS. Determinism: numerical."""
    try:
        outcome = estimate_factor_returns(
            ctx, (ctx.extra or {}).get("observation_weights")
        )
    except ValueError as exc:
        return _skip("attribution.factor_return_estimation",
                     "Cross-sectional factor return estimation", str(exc))

    metrics = _estimation_metrics(outcome)
    if outcome.n_periods_estimated == 0:
        return _error(
            "attribution.factor_return_estimation",
            "Cross-sectional factor return estimation",
            f"no period could be estimated: {outcome.n_periods_skipped_rank} rank "
            f"deficient, {outcome.n_periods_skipped_missing} with missing data",
            metrics=metrics,
        )

    status = Status.RECORDED
    limitations = list(_ESTIMATION_LIMITATIONS)
    if outcome.n_periods_skipped_rank:
        status = Status.WARN
        limitations.append(
            f"{outcome.n_periods_skipped_rank} of {outcome.n_periods_total} period(s) "
            "were rank deficient or underidentified and contribute no factor return."
        )

    return TestResult(
        test_id="attribution.factor_return_estimation",
        test_name="Cross-sectional factor return estimation",
        status=status,
        params={"weighting": outcome.weighting},
        metrics=metrics,
        interpretation=(
            f"Estimated {len(outcome.universe.factors)} factor return(s) over "
            f"{outcome.n_periods_estimated} of {outcome.n_periods_total} period(s) "
            f"using {outcome.weighting}-weight cross-sectional least squares on "
            f"{len(outcome.universe.assets)} asset(s)."
        ),
        limitations=limitations,
    )


# --------------------------------------------------------------------------- #
# 2. cross_sectional_factor_model
# --------------------------------------------------------------------------- #
@register_test(
    "attribution.cross_sectional_factor_model",
    family="attribution",
    name="Cross-sectional factor-return statistics",
    requires=("returns", "factor_exposures"),
    default_params={},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("conceptual_soundness", "assumption_validity"),
    object_kinds=_OBJECTS,
)
def cross_sectional_factor_model(ctx: Any) -> TestResult:
    """Time-series summary of the estimated factor returns.

    Reuses the same estimation core as ``factor_return_estimation``; the WLS
    mathematics exists once.

    **Not Fama–MacBeth.** No two-pass procedure, no risk-premium interpretation, no HAC
    correction. The standard error is the naive ``sd / sqrt(T)``, which assumes serially
    independent period estimates — an assumption factor returns routinely violate.
    """
    try:
        outcome = estimate_factor_returns(
            ctx, (ctx.extra or {}).get("observation_weights")
        )
    except ValueError as exc:
        return _skip("attribution.cross_sectional_factor_model",
                     "Cross-sectional factor-return statistics", str(exc))

    frame = outcome.factor_returns
    if frame.empty:
        return _error("attribution.cross_sectional_factor_model",
                      "Cross-sectional factor-return statistics",
                      "no period could be estimated", metrics=_estimation_metrics(outcome))

    metrics: dict[str, Any] = _estimation_metrics(outcome)
    n = int(len(frame))
    degenerate: list[str] = []

    for factor in outcome.universe.factors:
        series = frame[factor].dropna()
        if series.empty:
            continue
        mean = float(series.mean())
        sd = float(series.std(ddof=1)) if len(series) > 1 else 0.0
        # Scale-relative, per the B3 lesson. A numerically constant series has an sd of
        # order 1e-18, so an exact `sd <= 0` test never fires and the t-statistic comes
        # out at ~1e17 — a number a reviewer might act on.
        scale = max(1.0, float(np.abs(series.to_numpy(dtype=float)).max()))
        metrics[f"mean.{factor}"] = round(mean, 12)
        metrics[f"sd.{factor}"] = round(sd, 12)
        metrics[f"n_periods.{factor}"] = int(len(series))
        if sd <= DEGENERATE_SCALE_TOLERANCE * scale or len(series) < 2:
            metrics[f"stderr.{factor}"] = None
            metrics[f"tstat.{factor}"] = None
            metrics[f"degenerate.{factor}"] = True
            degenerate.append(factor)
        else:
            stderr = sd / math.sqrt(len(series))
            metrics[f"stderr.{factor}"] = round(stderr, 12)
            metrics[f"tstat.{factor}"] = round(mean / stderr, 10)
            metrics[f"degenerate.{factor}"] = False

    metrics["n_degenerate_factors"] = len(degenerate)
    metrics["degenerate_factors"] = ", ".join(degenerate)
    metrics["stderr_convention"] = "sample_sd / sqrt(T), no serial-dependence correction"

    return TestResult(
        test_id="attribution.cross_sectional_factor_model",
        test_name="Cross-sectional factor-return statistics",
        status=Status.RECORDED,
        params={"weighting": outcome.weighting},
        metrics=metrics,
        interpretation=(
            f"Descriptive time-series statistics for {len(outcome.universe.factors)} "
            f"factor(s) over {n} estimated period(s)"
            + (f"; {len(degenerate)} factor(s) numerically constant" if degenerate else "")
            + "."
        ),
        limitations=[
            "These are DESCRIPTIVE cross-sectional factor-return statistics. They are "
            "NOT Fama-MacBeth statistics: no two-pass procedure is implemented, no "
            "risk premium is estimated, and no HAC/Newey-West correction is applied.",
            "The standard error is the naive sd/sqrt(T), which assumes serially "
            "independent period estimates. Factor returns are routinely autocorrelated, "
            "and where they are this understates the true standard error and inflates "
            "the t-statistic.",
            "A non-zero t-statistic is not evidence of a causal effect.",
            "Factors whose estimates are numerically constant report no t-statistic "
            "rather than an enormous finite one.",
            "Numerical, not bitwise reproducible across BLAS implementations.",
        ],
    )


# --------------------------------------------------------------------------- #
# 3. exposure_analysis
# --------------------------------------------------------------------------- #
@register_test(
    "attribution.exposure_analysis",
    family="attribution",
    name="Factor exposure analysis",
    requires=("factor_exposures",),
    default_params={},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("explainability", "implementation_verification"),
    object_kinds=_OBJECTS,
)
def exposure_analysis(ctx: Any) -> TestResult:
    """Portfolio, benchmark and active factor exposures.

    Weights are **never renormalised**. If they sum to 0.97 the evidence shows 0.97,
    because silently scaling to 1.0 would hide a real data problem behind a tidy number.
    """
    returns = ctx.effective_returns()
    if returns is None or not ctx._exposures_canonical:
        return _skip("attribution.exposure_analysis", "Factor exposure analysis",
                     "returns and factor exposures are both required")
    if ctx.portfolio is None or ctx.portfolio.weights is None:
        return _skip("attribution.exposure_analysis", "Factor exposure analysis",
                     "no portfolio weights supplied; exposures are not invented")

    weights = ctx.portfolio.weights
    if isinstance(weights, pd.DataFrame):
        weights = weights.iloc[-1]
    benchmark = ctx.portfolio.benchmark_weights
    if isinstance(benchmark, pd.DataFrame):
        benchmark = benchmark.iloc[-1]

    sample = next(iter(ctx._exposures_canonical.values()))
    try:
        universe = align_universe(returns, sample, weights, benchmark)
    except ValueError as exc:
        return _error("attribution.exposure_analysis", "Factor exposure analysis", str(exc))

    timestamp = returns.index[-1]
    exposures = _exposures_for(ctx, timestamp, universe)
    if exposures is None or exposures.isna().to_numpy().any():
        return _error(
            "attribution.exposure_analysis", "Factor exposure analysis",
            f"no canonical exposure matrix is available at {timestamp}; exposures are "
            "never forward-filled",
        )

    assets = list(universe.assets)
    X = exposures.to_numpy(dtype=float)
    w = weights.reindex(assets).to_numpy(dtype=float)
    if not np.all(np.isfinite(w)):
        return _error("attribution.exposure_analysis", "Factor exposure analysis",
                      "portfolio weights contain non-finite values")

    portfolio_exposure = X.T @ w
    metrics: dict[str, Any] = {
        "as_of": str(timestamp),
        "n_periods": len(returns),
        "n_assets": len(assets),
        "n_factors": len(universe.factors),
        "exposures_time_varying": bool(ctx.is_time_varying_exposure),
        "weights_sum": round(float(w.sum()), 12),
        "gross_exposure": round(float(np.abs(w).sum()), 12),
        "weights_renormalised": False,
        "n_assets_excluded": len(universe.excluded_assets),
        "exclusion_rule": universe.exclusion_rule,
    }
    for factor, value in zip(universe.factors, portfolio_exposure, strict=True):
        metrics[f"portfolio_exposure.{factor}"] = round(float(value), 12)

    benchmark_available = benchmark is not None
    metrics["benchmark_available"] = benchmark_available
    if benchmark_available:
        b = benchmark.reindex(assets)
        if b.isna().any() or not np.all(np.isfinite(b.to_numpy(dtype=float))):
            metrics["benchmark_available"] = False
            benchmark_available = False
        else:
            bv = b.to_numpy(dtype=float)
            benchmark_exposure = X.T @ bv
            metrics["benchmark_weights_sum"] = round(float(bv.sum()), 12)
            for factor, value in zip(universe.factors, benchmark_exposure, strict=True):
                metrics[f"benchmark_exposure.{factor}"] = round(float(value), 12)
                metrics[f"active_exposure.{factor}"] = round(
                    float(portfolio_exposure[list(universe.factors).index(factor)] - value), 12
                )

    limitations = [
        "Weights are reported as supplied and are NEVER renormalised. A weight vector "
        "summing to 0.97 is shown as 0.97, because scaling it to 1.0 would hide a real "
        "data problem behind a tidy number.",
        "Exposures are observed characteristics; an exposure is not a causal effect.",
        "Attribution is descriptive, so optimiser constraints are not enforced here — "
        "only finiteness and asset alignment are required.",
        "Numerical, not bitwise reproducible across BLAS implementations.",
    ]
    if not benchmark_available:
        limitations.append(
            "No usable benchmark weights: active exposure is not reported rather than "
            "computed against an invented benchmark."
        )

    return TestResult(
        test_id="attribution.exposure_analysis",
        test_name="Factor exposure analysis",
        status=Status.RECORDED,
        params={},
        metrics=metrics,
        interpretation=(
            f"Portfolio factor exposures across {len(universe.factors)} factor(s) as of "
            f"{timestamp}"
            + ("; active exposure reported against the supplied benchmark."
               if benchmark_available else "; no benchmark supplied.")
        ),
        limitations=limitations,
    )


# --------------------------------------------------------------------------- #
# 4. return_attribution
# --------------------------------------------------------------------------- #
@register_test(
    "attribution.return_attribution",
    family="attribution",
    name="Return attribution",
    requires=("returns", "factor_exposures"),
    default_params={},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification", "explainability"),
    object_kinds=_OBJECTS,
)
def return_attribution(ctx: Any) -> TestResult:
    """Decompose portfolio return into factor and specific contributions.

    ``(X_t' w)' f_t + w' e_t`` must equal ``w' r_t`` to tolerance. The residual stays
    visible: forcing it to zero by construction would make the reconciliation check
    vacuous, which is the opposite of what it exists for.
    """
    returns = ctx.effective_returns()
    if returns is None or not ctx._exposures_canonical:
        return _skip("attribution.return_attribution", "Return attribution",
                     "returns and factor exposures are both required")
    if ctx.portfolio is None or ctx.portfolio.weights is None:
        return _skip("attribution.return_attribution", "Return attribution",
                     "no portfolio weights supplied")

    weights = ctx.portfolio.weights
    if isinstance(weights, pd.DataFrame):
        weights = weights.iloc[-1]

    try:
        outcome = estimate_factor_returns(ctx, (ctx.extra or {}).get("observation_weights"))
    except ValueError as exc:
        return _skip("attribution.return_attribution", "Return attribution", str(exc))

    assets = list(outcome.universe.assets)
    factors = list(outcome.universe.factors)
    w = weights.reindex(assets).to_numpy(dtype=float)

    # Supplied factor returns take precedence, and the source is always recorded.
    supplied = ctx.factor_returns
    use_supplied = False
    if supplied is not None:
        candidate = supplied.reindex(columns=factors)
        if not candidate.isna().to_numpy().all():
            use_supplied = True
    source = "supplied" if use_supplied else "estimated"

    errors: list[float] = []
    factor_total = 0.0
    specific_total = 0.0
    observed_total = 0.0
    n_periods = 0

    for timestamp in outcome.factor_returns.index:
        exposures = _exposures_for(ctx, timestamp, outcome.universe)
        if exposures is None:
            continue
        X = exposures.to_numpy(dtype=float)
        r = returns.loc[timestamp, assets].to_numpy(dtype=float)

        if use_supplied:
            if timestamp not in supplied.index:
                continue
            f = supplied.loc[timestamp, factors].to_numpy(dtype=float)
            if not np.all(np.isfinite(f)):
                continue
            residual = r - X @ f
        else:
            f = outcome.factor_returns.loc[timestamp].to_numpy(dtype=float)
            residual = outcome.specific_returns.loc[timestamp].to_numpy(dtype=float)

        factor_contribution = float((X.T @ w) @ f)
        specific_contribution = float(w @ residual)
        observed = float(w @ r)
        errors.append(observed - (factor_contribution + specific_contribution))
        factor_total += factor_contribution
        specific_total += specific_contribution
        observed_total += observed
        n_periods += 1

    if n_periods == 0:
        return _error("attribution.return_attribution", "Return attribution",
                      "no period could be attributed",
                      metrics=_estimation_metrics(outcome))

    absolute = np.abs(errors)
    scale = max(1.0, abs(observed_total / n_periods))
    outside = int(sum(1 for e in errors if not _within_tolerance(e, scale)))

    metrics: dict[str, Any] = {
        **_estimation_metrics(outcome),
        "factor_return_source": source,
        "n_periods_attributed": n_periods,
        "total_factor_contribution": round(factor_total, 12),
        "total_specific_contribution": round(specific_total, 12),
        "total_observed_return": round(observed_total, 12),
        "max_abs_reconciliation_error": round(float(absolute.max()), 15),
        "mean_abs_reconciliation_error": round(float(absolute.mean()), 15),
        "n_periods_outside_tolerance": outside,
        "reconciliation_atol": RECONCILIATION_ATOL,
        "reconciliation_rtol": RECONCILIATION_RTOL,
    }

    status = Status.FAIL if outside else Status.RECORDED
    return TestResult(
        test_id="attribution.return_attribution",
        test_name="Return attribution",
        status=status,
        params={"factor_return_source": source, "weighting": outcome.weighting},
        metrics=metrics,
        interpretation=(
            f"Attributed {n_periods} period(s) using {source} factor returns; "
            f"largest reconciliation error {float(absolute.max()):.3g}"
            + (f", with {outside} period(s) outside tolerance." if outside else ".")
        ),
        limitations=[
            f"Factor returns were {source}. The source is always recorded; the two "
            "routes are never switched silently.",
            "The reconciliation residual is reported as computed and is never forced "
            "to zero by construction, which would make this check vacuous.",
            "Attribution is descriptive of the supplied factor structure; it does not "
            "establish that the factor model is well specified.",
            "Numerical, not bitwise reproducible across BLAS implementations.",
        ],
    )


# --------------------------------------------------------------------------- #
# 5. risk_attribution
# --------------------------------------------------------------------------- #
@register_test(
    "attribution.risk_attribution",
    family="attribution",
    name="Risk attribution",
    requires=("returns", "factor_exposures"),
    default_params={"ddof": 1},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification", "conceptual_soundness"),
    object_kinds=_OBJECTS,
)
def risk_attribution(ctx: Any, ddof: int = 1) -> TestResult:
    """Factor-model variance ``V = x'Fx + S``, with per-factor components.

    The factor covariance is **never** taken from ``ctx.covariance``: that field holds
    the *asset* covariance, and inferring which one it is from a coincidental shape
    match would be a silent category error.
    """
    returns = ctx.effective_returns()
    if returns is None or not ctx._exposures_canonical:
        return _skip("attribution.risk_attribution", "Risk attribution",
                     "returns and factor exposures are both required")
    if ctx.portfolio is None or ctx.portfolio.weights is None:
        return _skip("attribution.risk_attribution", "Risk attribution",
                     "no portfolio weights supplied")

    weights = ctx.portfolio.weights
    if isinstance(weights, pd.DataFrame):
        weights = weights.iloc[-1]

    try:
        outcome = estimate_factor_returns(ctx, (ctx.extra or {}).get("observation_weights"))
    except ValueError as exc:
        return _skip("attribution.risk_attribution", "Risk attribution", str(exc))
    if outcome.factor_returns.empty:
        return _error("attribution.risk_attribution", "Risk attribution",
                      "no period could be estimated", metrics=_estimation_metrics(outcome))

    assets = list(outcome.universe.assets)
    factors = list(outcome.universe.factors)

    if ctx.factor_returns is not None:
        candidate = ctx.factor_returns.reindex(columns=factors).dropna()
        if len(candidate) > 1:
            factor_covariance = candidate.cov(ddof=ddof)
            covariance_source = "supplied_factor_returns"
        else:
            factor_covariance = outcome.factor_returns.cov(ddof=ddof)
            covariance_source = "estimated_factor_returns"
    else:
        factor_covariance = outcome.factor_returns.cov(ddof=ddof)
        covariance_source = "estimated_factor_returns"

    exposures = _exposures_for(ctx, returns.index[-1], outcome.universe)
    if exposures is None or exposures.isna().to_numpy().any():
        return _error("attribution.risk_attribution", "Risk attribution",
                      "no canonical exposure matrix at the reporting date")

    w = weights.reindex(assets).to_numpy(dtype=float)
    X = exposures.to_numpy(dtype=float)
    x = X.T @ w
    F = factor_covariance.reindex(index=factors, columns=factors).to_numpy(dtype=float)

    factor_variance = float(x @ F @ x)
    specific_variance_by_asset = outcome.specific_returns.var(ddof=ddof)
    D = specific_variance_by_asset.reindex(assets).to_numpy(dtype=float)
    specific_variance = float((w**2) @ D)
    total = factor_variance + specific_variance

    metrics: dict[str, Any] = {
        **_estimation_metrics(outcome),
        "factor_covariance_source": covariance_source,
        "factor_covariance_dimension": len(factors),
        "specific_risk_model": "diagonal",
        "ddof": ddof,
        "factor_variance": round(factor_variance, 15),
        "specific_variance": round(specific_variance, 15),
        "total_factor_model_variance": round(total, 15),
        "factor_variance_share": round(factor_variance / total, 10) if total > 0 else None,
        "specific_variance_share": round(specific_variance / total, 10) if total > 0 else None,
    }
    for factor, value in zip(factors, x, strict=True):
        metrics[f"exposure.{factor}"] = round(float(value), 12)

    # Per-factor components: marginal = F x, component_i = x_i * marginal_i, which sums
    # exactly to x'Fx. Squared standalone contributions would not reconcile.
    marginal = F @ x
    components = x * marginal
    for factor, value in zip(factors, components, strict=True):
        metrics[f"factor_risk_component.{factor}"] = round(float(value), 15)
    component_error = float(components.sum() - factor_variance)
    metrics["factor_component_sum"] = round(float(components.sum()), 15)
    metrics["factor_component_reconciliation_error"] = round(component_error, 18)

    portfolio_returns = (returns[assets] @ weights.reindex(assets))
    empirical = float(portfolio_returns.var(ddof=ddof)) if len(portfolio_returns) > 1 else None
    metrics["empirical_portfolio_variance"] = (
        round(empirical, 15) if empirical is not None else None
    )
    if empirical is not None and empirical > 0:
        ratio = total / empirical
        metrics["factor_model_to_empirical_ratio"] = round(ratio, 10)
        metrics["variance_shortfall"] = round(1.0 - ratio, 10)

    status = Status.RECORDED
    if not _within_tolerance(component_error, factor_variance):
        status = Status.FAIL

    return TestResult(
        test_id="attribution.risk_attribution",
        test_name="Risk attribution",
        status=status,
        params={"ddof": ddof, "factor_covariance_source": covariance_source},
        metrics=metrics,
        interpretation=(
            f"Factor-model variance {total:.6g} = factor {factor_variance:.6g} + "
            f"specific {specific_variance:.6g}"
            + (f"; empirical portfolio variance {empirical:.6g}."
               if empirical is not None else ".")
        ),
        limitations=[
            "Specific risk uses a DIAGONAL model, D = diag(var(e_i)). Cross-sectional "
            "correlation among specific returns is not modelled, and where it exists "
            "this understates total risk. This is not full idiosyncratic covariance "
            "modelling.",
            "The factor covariance is taken from factor returns, never from "
            "ctx.covariance, which holds the ASSET covariance. Inferring which is which "
            "from a coincidental shape match would be a silent category error.",
            "Factor-model variance need not equal the empirical portfolio variance. Any "
            "difference is evidence about model specification, not necessarily a "
            "software defect.",
            "Per-factor components use marginal = F x and component_i = x_i * "
            "marginal_i, which sums exactly to x'Fx.",
            "Numerical, not bitwise reproducible across BLAS implementations.",
        ],
    )


# --------------------------------------------------------------------------- #
# 6. risk_change_decomposition
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AttributionState:
    """The narrow, hashable prior state for a risk-change comparison.

    Deliberately minimal. ``MarketContext.fingerprint()`` does not canonicalise
    ``extra``, so a prior state routed through ``ctx.extra`` would be a material
    analytical input invisible to evidence identity — two reviews comparing against
    entirely different prior states would share an input hash.

    This object carries only what the decomposition needs, and its canonical hash is
    recorded in ``params`` as a string, which the existing parameter hashing represents
    exactly. No DataFrame crosses into a hashed field and ``EvidenceRecord`` is
    unchanged.
    """

    exposure: pd.Series             # x0, indexed by factor
    factor_covariance: pd.DataFrame  # F0, factor x factor
    specific_variance: float         # S0
    label: str = ""

    def canonical_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"AttributionState/1")
        digest.update(canonical_series_bytes(self.exposure, label="exposure"))
        digest.update(canonical_frame_bytes(self.factor_covariance, label="factor_cov"))
        digest.update(f"S={float(self.specific_variance).hex()}".encode())
        digest.update(f"label={self.label}".encode())
        return digest.hexdigest()

    def variance(self) -> float:
        x = self.exposure.to_numpy(dtype=float)
        F = self.factor_covariance.to_numpy(dtype=float)
        return float(x @ F @ x) + float(self.specific_variance)


def decompose_risk_change(before: AttributionState,
                          after: AttributionState) -> dict[str, float]:
    """The frozen four-component decomposition. See :data:`RISK_CHANGE_CONTRACT`.

    Simultaneous, not sequential: an order-dependent attribution would give a different
    answer depending on which component was peeled off first, and nothing in the output
    would reveal which order had been chosen.
    """
    factors = list(before.exposure.index)
    x0 = before.exposure.reindex(factors).to_numpy(dtype=float)
    x1 = after.exposure.reindex(factors).to_numpy(dtype=float)
    F0 = before.factor_covariance.reindex(index=factors, columns=factors).to_numpy(dtype=float)
    F1 = after.factor_covariance.reindex(index=factors, columns=factors).to_numpy(dtype=float)
    dx = x1 - x0
    dF = F1 - F0
    dS = float(after.specific_variance) - float(before.specific_variance)

    exposure_component = float(2.0 * (x0 @ F0 @ dx) + dx @ F0 @ dx)
    covariance_component = float(x0 @ dF @ x0)
    specific_component = dS
    interaction_component = float(2.0 * (x0 @ dF @ dx) + dx @ dF @ dx)

    return {
        "exposure_component": exposure_component,
        "factor_covariance_component": covariance_component,
        "specific_risk_component": specific_component,
        "interaction_component": interaction_component,
    }


def interaction_share(components: dict[str, float]) -> float:
    """Bounded share in [0, 1].

    Divided by the sum of absolute components, not by signed dV. Dividing by dV
    explodes toward infinity whenever components cancel — the very case where the
    interaction term most needs to stay readable.
    """
    denominator = sum(abs(v) for v in components.values())
    if denominator <= RECONCILIATION_ATOL:
        return 0.0
    return abs(components["interaction_component"]) / denominator


@register_test(
    "attribution.risk_change_decomposition",
    family="attribution",
    name="Risk change decomposition",
    requires=("returns", "factor_exposures"),
    default_params={"interaction_warn": INTERACTION_SHARE_WARN, "ddof": 1},
    context_type="market",
    risk_stripes=_STRIPES,
    risk_dimensions=("sensitivity", "change_control"),
    object_kinds=_OBJECTS,
)
def risk_change_decomposition(
    ctx: Any,
    comparison_state: AttributionState | None = None,
    interaction_warn: float = INTERACTION_SHARE_WARN,
    ddof: int = 1,
) -> TestResult:
    """Decompose the change in factor-model variance between two states.

    The prior state arrives as an explicit typed :class:`AttributionState` parameter,
    never through ``ctx.extra``. Both state hashes are recorded in ``params`` so
    changing either state changes evidence identity.
    """
    if comparison_state is None:
        return _skip(
            "attribution.risk_change_decomposition", "Risk change decomposition",
            "no comparison state supplied. The prior state is an explicit typed "
            "AttributionState parameter rather than a value in ctx.extra, because "
            "MarketContext.fingerprint() does not canonicalise extra and a prior state "
            "routed through it would be invisible to evidence identity",
        )

    returns = ctx.effective_returns()
    if returns is None or not ctx._exposures_canonical:
        return _skip("attribution.risk_change_decomposition", "Risk change decomposition",
                     "returns and factor exposures are both required")
    if ctx.portfolio is None or ctx.portfolio.weights is None:
        return _skip("attribution.risk_change_decomposition", "Risk change decomposition",
                     "no portfolio weights supplied")

    weights = ctx.portfolio.weights
    if isinstance(weights, pd.DataFrame):
        weights = weights.iloc[-1]

    try:
        outcome = estimate_factor_returns(ctx, (ctx.extra or {}).get("observation_weights"))
    except ValueError as exc:
        return _skip("attribution.risk_change_decomposition", "Risk change decomposition",
                     str(exc))
    if outcome.factor_returns.empty:
        return _error("attribution.risk_change_decomposition", "Risk change decomposition",
                      "no period could be estimated", metrics=_estimation_metrics(outcome))

    assets = list(outcome.universe.assets)
    factors = list(outcome.universe.factors)
    exposures = _exposures_for(ctx, returns.index[-1], outcome.universe)
    if exposures is None or exposures.isna().to_numpy().any():
        return _error("attribution.risk_change_decomposition", "Risk change decomposition",
                      "no canonical exposure matrix at the reporting date")

    w = weights.reindex(assets).to_numpy(dtype=float)
    x1 = pd.Series(exposures.to_numpy(dtype=float).T @ w, index=factors)
    F1 = outcome.factor_returns.cov(ddof=ddof).reindex(index=factors, columns=factors)
    D = outcome.specific_returns.var(ddof=ddof).reindex(assets).to_numpy(dtype=float)
    S1 = float((w**2) @ D)

    current = AttributionState(exposure=x1, factor_covariance=F1,
                               specific_variance=S1, label="current")

    if list(comparison_state.exposure.index) != factors:
        return _error(
            "attribution.risk_change_decomposition", "Risk change decomposition",
            "comparison state factors do not match the current canonical factor order: "
            f"{list(comparison_state.exposure.index)} vs {factors}",
        )

    components = decompose_risk_change(comparison_state, current)
    v0 = comparison_state.variance()
    v1 = current.variance()
    observed = v1 - v0
    component_sum = sum(components.values())
    error = observed - component_sum
    share = interaction_share(components)

    metrics: dict[str, Any] = {
        "variance_before": round(v0, 15),
        "variance_after": round(v1, 15),
        "observed_delta": round(observed, 15),
        **{k: round(v, 15) for k, v in components.items()},
        "component_sum": round(component_sum, 15),
        "reconciliation_error": round(error, 18),
        "interaction_abs": round(abs(components["interaction_component"]), 15),
        "interaction_share": round(share, 10),
        "interaction_share_definition": RISK_CHANGE_CONTRACT["interaction_share"],
        "current_state_hash": current.canonical_hash()[:32],
        "comparison_state_hash": comparison_state.canonical_hash()[:32],
        "n_factors": len(factors),
        "reconciliation_atol": RECONCILIATION_ATOL,
        "reconciliation_rtol": RECONCILIATION_RTOL,
        "specific_risk_model": "diagonal",
    }

    reconciled = _within_tolerance(error, max(abs(observed), abs(v0), abs(v1)))
    if not reconciled:
        status = Status.FAIL
        note = f"reconciliation error {error:.3g} exceeds tolerance"
    elif share > interaction_warn:
        status = Status.WARN
        note = (
            f"the interaction term carries {share:.1%} of total absolute component "
            f"magnitude, above the {interaction_warn:.0%} threshold"
        )
    else:
        status = Status.RECORDED
        note = "reconciled, with a modest interaction term"

    return TestResult(
        test_id="attribution.risk_change_decomposition",
        test_name="Risk change decomposition",
        status=status,
        params={
            "interaction_warn": interaction_warn, "ddof": ddof,
            # Both hashes in params: changing either state changes evidence identity.
            "current_state_hash": current.canonical_hash(),
            "comparison_state_hash": comparison_state.canonical_hash(),
        },
        metrics=metrics,
        interpretation=(
            f"Factor-model variance moved {v0:.6g} -> {v1:.6g} (delta {observed:.6g}); "
            f"{note}."
        ),
        limitations=[
            "The decomposition is SIMULTANEOUS, not sequential. An order-dependent "
            "attribution would give different answers depending on which component was "
            "removed first, and nothing in the output would reveal the order chosen.",
            "Interaction share is |interaction| divided by the SUM OF ABSOLUTE "
            "components, bounded in [0,1]. Dividing by the signed change explodes when "
            "components cancel, which is exactly when the interaction most needs to "
            "stay readable.",
            "A large interaction term WARNs independently of reconciliation: correct "
            "algebra does not make a decomposition informative.",
            "Specific risk uses the diagonal model; see attribution.risk_attribution.",
            "Numerical, not bitwise reproducible across BLAS implementations.",
        ],
    )
