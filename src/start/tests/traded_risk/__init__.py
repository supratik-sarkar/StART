"""Traded risk — eight registered surfaces.

Two context kinds. CEV and Stanton consume a :class:`ShortRateContext`: a single scalar
diffusion with a known observation interval. The VaR family and the barrier test consume
a :class:`MarketContext`.

One canonical exception sequence
--------------------------------

Every VaR backtest calls :func:`exception_sequence`. Kupiec, both Christoffersen tests
and the traffic light share it rather than each re-deriving the indicator.

That matters more than it looks. ``LR_cc = LR_uc + LR_ind`` is an identity only when all
three statistics are computed from the *same* sequence. If the traffic light quietly used
``<=`` where Kupiec used ``<``, the three would disagree on a boundary day and the
identity would break in a way that looks like a numerical problem rather than a
convention mismatch.

Convention, stated once and applied everywhere::

    VaR_t is a POSITIVE loss magnitude
    I_t = 1  iff  PnL_t < -VaR_t

Actual and hypothetical P&L are different inputs and are never substituted for one
another. Hypothetical P&L revalues a frozen position; actual P&L includes intraday
trading and fees. A backtest run on the wrong one is a backtest of something else.

Limit-safe likelihood arithmetic
--------------------------------

Every log-likelihood here uses ``scipy.special.xlogy`` and ``xlog1py``, which return 0
when the multiplier is 0 rather than ``0 * -inf = nan``. Zero exceptions and
all-exceptions are legitimate observed sequences, not error conditions, and a backtest
that returns ``nan`` on a perfect year is useless exactly when someone most wants an
answer.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import special, stats

from start.core.schemas import Status, TestResult
from start.data.synthetic_market import barrier_crossing_probability
from start.registry import register_test

__all__ = [
    "cev_elasticity",
    "stanton_nonparametric",
    "var_exceptions",
    "var_kupiec_pof",
    "var_christoffersen_independence",
    "var_christoffersen_conditional",
    "var_traffic_light",
    "brownian_bridge_barrier",
    "ExceptionSequence",
    "exception_sequence",
    "kupiec_lr",
    "christoffersen_independence_lr",
    "estimate_cev",
    "stanton_first_order",
    "CEV_BOOTSTRAP_DRAWS",
    "CEV_BOOTSTRAP_SEED",
    "CEV_CI_LEVEL",
    "TRAFFIC_LIGHT_BANDS",
    "TRAFFIC_LIGHT_N",
    "TRAFFIC_LIGHT_CONFIDENCE",
    "SILVERMAN_FACTOR",
]

# --------------------------------------------------------------------------- #
# FROZEN CONSTANTS
# --------------------------------------------------------------------------- #
#: CEV interval controls. Bootstrap rather than a naive OLS interval: the log
#: squared-increment regression has heteroskedastic, non-normal errors, so the textbook
#: OLS standard error is not a defensible diffusion confidence interval.
CEV_BOOTSTRAP_DRAWS = 400
CEV_BOOTSTRAP_SEED = 20240501
CEV_CI_LEVEL = 0.95

#: Historical/classical traffic light. 250 observations at 99%.
TRAFFIC_LIGHT_N = 250
TRAFFIC_LIGHT_CONFIDENCE = 0.99
TRAFFIC_LIGHT_BANDS: tuple[tuple[int, int, str], ...] = (
    (0, 4, "green"),
    (5, 9, "yellow"),
    (10, 10**9, "red"),
)

#: Silverman's rule-of-thumb multiplier.
SILVERMAN_FACTOR = 1.06

_SHORT_RATE_STRIPES = ("market", "treasury_irrbb")
_MARKET_STRIPES = ("market",)
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


_NON_REJECTION_CAVEAT = (
    "Non-rejection does not establish that the model is correct. It means the observed "
    "exception process gave insufficient evidence against the stated hypothesis at this "
    "level, on this sample."
)


# =========================================================================== #
# CEV
# =========================================================================== #
@dataclass
class CevEstimate:
    gamma_hat: float
    log_sigma2_hat: float
    sigma_hat: float
    n_total: int
    n_used: int
    n_nonpositive_dropped: int
    n_zero_increment_dropped: int
    dt: float
    ci_low: float | None = None
    ci_high: float | None = None
    n_bootstrap_valid: int = 0


def estimate_cev(rates: pd.Series, dt: float) -> CevEstimate:
    """Approximate CEV elasticity from the log squared-increment regression.

    Under ``dr = (a + b r) dt + sigma r^gamma dW``, the squared increment satisfies
    ``E[(dr)^2 | r] ≈ sigma^2 r^(2 gamma) dt`` over a short interval, so::

        ln( (Δr_t)^2 / Δt )  ≈  ln(sigma^2)  +  2 gamma * ln(r_{t-1})  +  noise

    and ``gamma = slope / 2``.

    This is a **finite-sample approximation under the stated diffusion and
    discretisation assumptions**, not a universally consistent diffusion estimator. The
    log transform of a squared increment has strongly non-normal, heteroskedastic
    errors, and the drift term contributes an O(Δ) bias that does not vanish at fixed Δ.
    Its empirical behaviour is assessed by the pre-registered Monte Carlo study rather
    than asserted here.

    Non-positive rates are **dropped and counted**, never shifted or floored to make the
    logarithm defined: shifting changes the process being estimated, and doing it
    silently would make gamma an answer about different data.
    """
    values = rates.to_numpy(dtype=float)
    n_total = int(values.size)
    if n_total < 2:
        raise ValueError("at least two observations are required")

    previous = values[:-1]
    increments = np.diff(values)

    nonpositive = ~(previous > 0)
    zero_increment = increments == 0.0
    keep = (~nonpositive) & (~zero_increment) & np.isfinite(previous) & np.isfinite(increments)

    n_nonpositive = int(nonpositive.sum())
    n_zero = int((zero_increment & ~nonpositive).sum())
    if int(keep.sum()) < 10:
        raise ValueError(
            f"only {int(keep.sum())} usable increment(s) after dropping "
            f"{n_nonpositive} non-positive rate(s) and {n_zero} zero increment(s)"
        )

    x = np.log(previous[keep])
    y = np.log((increments[keep] ** 2) / dt)
    design = np.column_stack([np.ones_like(x), x])
    (intercept, slope), *_ = np.linalg.lstsq(design, y, rcond=None)

    return CevEstimate(
        gamma_hat=float(slope / 2.0),
        log_sigma2_hat=float(intercept),
        sigma_hat=float(math.sqrt(math.exp(intercept))) if intercept < 700 else float("inf"),
        n_total=n_total,
        n_used=int(keep.sum()),
        n_nonpositive_dropped=n_nonpositive,
        n_zero_increment_dropped=n_zero,
        dt=dt,
    )


def _bootstrap_cev(
    rates: pd.Series, dt: float, draws: int, seed: int, level: float
) -> tuple[float | None, float | None, int]:
    """Seeded pair bootstrap over usable increments.

    The resampling unit is the ``(r_{t-1}, Δr_t)`` pair, which keeps each observation's
    regressor and response together. Resampling residuals instead would impose the
    homoskedasticity the log transform demonstrably violates.
    """
    values = rates.to_numpy(dtype=float)
    previous = values[:-1]
    increments = np.diff(values)
    keep = (previous > 0) & (increments != 0.0) & np.isfinite(previous) & np.isfinite(increments)
    x_all = np.log(previous[keep])
    y_all = np.log((increments[keep] ** 2) / dt)
    n = x_all.size
    if n < 10:
        return None, None, 0

    rng = np.random.default_rng(seed)
    gammas: list[float] = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        x, y = x_all[idx], y_all[idx]
        if float(np.std(x)) <= 0:
            continue
        design = np.column_stack([np.ones_like(x), x])
        try:
            (_, slope), *_ = np.linalg.lstsq(design, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if math.isfinite(slope):
            gammas.append(float(slope / 2.0))

    if len(gammas) < 20:
        return None, None, len(gammas)
    alpha = (1.0 - level) / 2.0
    low, high = np.percentile(gammas, [alpha * 100.0, (1.0 - alpha) * 100.0])
    return float(low), float(high), len(gammas)


@register_test(
    "traded_risk.cev_elasticity",
    family="traded_risk",
    name="CEV volatility elasticity",
    requires=("rates",),
    default_params={
        "stated_gamma": None,
        "bootstrap_draws": CEV_BOOTSTRAP_DRAWS,
        "bootstrap_seed": CEV_BOOTSTRAP_SEED,
        "ci_level": CEV_CI_LEVEL,
    },
    context_type="short_rate",
    risk_stripes=_SHORT_RATE_STRIPES,
    risk_dimensions=("conceptual_soundness", "assumption_validity"),
    object_kinds=_OBJECTS,
)
def cev_elasticity(
    ctx: Any,
    stated_gamma: float | None = None,
    bootstrap_draws: int = CEV_BOOTSTRAP_DRAWS,
    bootstrap_seed: int = CEV_BOOTSTRAP_SEED,
    ci_level: float = CEV_CI_LEVEL,
) -> TestResult:
    """Volatility elasticity under a CEV short-rate model.

    Determinism: ``seeded`` — the point estimate is numerical, the bootstrap interval
    depends on the recorded seed.
    """
    problems = ctx.validate_context()
    if problems:
        return _skip(
            "traded_risk.cev_elasticity",
            "CEV volatility elasticity",
            "; ".join(problems),
            stated_gamma=stated_gamma,
        )

    rates = ctx.decimal_rates()
    try:
        estimate = estimate_cev(rates, ctx.dt)
    except ValueError as exc:
        return _skip(
            "traded_risk.cev_elasticity", "CEV volatility elasticity", str(exc), stated_gamma=stated_gamma
        )

    low, high, n_valid = _bootstrap_cev(rates, ctx.dt, bootstrap_draws, bootstrap_seed, ci_level)
    estimate.ci_low, estimate.ci_high, estimate.n_bootstrap_valid = low, high, n_valid

    metrics: dict[str, Any] = {
        "gamma_hat": round(estimate.gamma_hat, 10),
        "log_sigma2_hat": round(estimate.log_sigma2_hat, 10),
        "sigma_hat": round(estimate.sigma_hat, 10) if math.isfinite(estimate.sigma_hat) else None,
        "n_total": estimate.n_total,
        "n_used": estimate.n_used,
        "n_nonpositive_dropped": estimate.n_nonpositive_dropped,
        "n_zero_increment_dropped": estimate.n_zero_increment_dropped,
        "dt": estimate.dt,
        "periods_per_year": ctx.periods_per_year,
        "ci_level": ci_level,
        "ci_low": round(low, 10) if low is not None else None,
        "ci_high": round(high, 10) if high is not None else None,
        "ci_method": "seeded pair bootstrap over (r_prev, dr) pairs, percentile interval",
        "bootstrap_draws": bootstrap_draws,
        "bootstrap_seed": bootstrap_seed,
        "n_bootstrap_valid": n_valid,
        "stated_gamma": stated_gamma,
        "estimator": "ln((dr)^2/dt) regressed on ln(r_prev); gamma = slope / 2",
    }

    status = Status.RECORDED
    note = ""
    if stated_gamma is not None and low is not None and high is not None:
        inside = low <= float(stated_gamma) <= high
        metrics["stated_gamma_inside_interval"] = inside
        if not inside:
            status = Status.WARN
            note = (
                f" The reviewer-stated gamma {stated_gamma:g} lies outside the "
                f"{ci_level:.0%} bootstrap interval [{low:.4f}, {high:.4f}]."
            )

    return TestResult(
        test_id="traded_risk.cev_elasticity",
        test_name="CEV volatility elasticity",
        status=status,
        params={
            "stated_gamma": stated_gamma,
            "bootstrap_draws": bootstrap_draws,
            "bootstrap_seed": bootstrap_seed,
            "ci_level": ci_level,
        },
        metrics=metrics,
        interpretation=(
            f"Estimated elasticity gamma = {estimate.gamma_hat:.4f} from "
            f"{estimate.n_used:,} usable increment(s)"
            + (
                f", {ci_level:.0%} bootstrap interval [{low:.4f}, {high:.4f}]"
                if low is not None
                else ", bootstrap interval unavailable"
            )
            + "."
            + note
        ),
        limitations=[
            "APPROXIMATE FINITE-SAMPLE ESTIMATOR under the stated diffusion and "
            "discretisation assumptions. It is not presented as a universally "
            "consistent diffusion estimator, and its empirical behaviour is assessed "
            "through the pre-registered Monte Carlo study rather than asserted here.",
            "The log squared-increment regression has strongly non-normal, "
            "heteroskedastic errors, so a conventional OLS standard error would not be "
            "a valid diffusion confidence interval. The reported interval is a seeded "
            "pair bootstrap, and its own coverage is an empirical question.",
            "The drift term contributes an O(dt) bias that does not vanish at fixed dt.",
            "Non-positive rates and zero increments are DROPPED and counted, never "
            "shifted or floored: shifting would change the process being estimated.",
            "Determinism: seeded. The point estimate is numerical; the interval depends "
            "on the recorded bootstrap seed.",
        ],
    )


# =========================================================================== #
# Stanton — first order only
# =========================================================================== #
def stanton_first_order(rates: pd.Series, dt: float, grid: np.ndarray, bandwidth: float) -> pd.DataFrame:
    """First-order Nadaraya-Watson drift and diffusion estimates.

    Exact expressions, with ``w_i(r) = K((r_i - r)/h)`` a Gaussian kernel::

        mu_hat(r)     = sum_i w_i(r) * (r_{i+1} - r_i)   / ( sum_i w_i(r) * dt )
        sigma2_hat(r) = sum_i w_i(r) * (r_{i+1} - r_i)^2 / ( sum_i w_i(r) * dt )

    **First order only.** There is no ``order`` parameter and no higher-order path.
    Stanton's second- and third-order corrections reduce the O(dt) discretisation bias
    but require their own expressions and their own validation; claiming their
    properties for this estimator would be false.

    Effective sample size ``ESS = (sum w)^2 / sum(w^2)`` is reported per grid point so
    thin support stays visible rather than being smoothed into a confident-looking curve.
    """
    values = rates.to_numpy(dtype=float)
    previous = values[:-1]
    increments = np.diff(values)

    rows: list[dict[str, float]] = []
    for point in grid:
        z = (previous - point) / bandwidth
        w = np.exp(-0.5 * z * z)  # Gaussian kernel, constant factor cancels
        total = float(w.sum())
        # Both denominators must be guarded, not just the weight sum. With a very
        # narrow bandwidth every weight underflows: sum(w) can still be a tiny
        # positive number while sum(w^2) underflows to exactly 0.0, so guarding only
        # `total > 0` still divides by zero. That is precisely the thin-support case
        # this diagnostic exists to surface, so it must not crash on it.
        sum_squares = float((w * w).sum())
        ess = float((total**2) / sum_squares) if (total > 0 and sum_squares > 0) else 0.0
        if total <= 0 or sum_squares <= 0 or not math.isfinite(total):
            rows.append(
                {"r": float(point), "mu": float("nan"), "sigma2": float("nan"), "ess": 0.0, "weight_sum": 0.0}
            )
            continue
        mu = float((w @ increments) / (total * dt))
        sigma2 = float((w @ (increments**2)) / (total * dt))
        rows.append({"r": float(point), "mu": mu, "sigma2": sigma2, "ess": ess, "weight_sum": total})
    return pd.DataFrame(rows)


@register_test(
    "traded_risk.stanton_nonparametric",
    family="traded_risk",
    name="Stanton nonparametric drift and diffusion",
    requires=("rates",),
    default_params={"bandwidth": None, "n_grid": 25, "min_ess": 10.0},
    context_type="short_rate",
    risk_stripes=_SHORT_RATE_STRIPES,
    risk_dimensions=("conceptual_soundness", "assumption_validity"),
    object_kinds=_OBJECTS,
)
def stanton_nonparametric(
    ctx: Any,
    bandwidth: float | None = None,
    n_grid: int = 25,
    min_ess: float = 10.0,
) -> TestResult:
    """First-order nonparametric drift and diffusion. Determinism: numerical."""
    problems = ctx.validate_context()
    if problems:
        return _skip(
            "traded_risk.stanton_nonparametric",
            "Stanton nonparametric drift and diffusion",
            "; ".join(problems),
        )

    rates = ctx.decimal_rates()
    values = rates.to_numpy(dtype=float)
    if values.size < 3:
        return _skip(
            "traded_risk.stanton_nonparametric",
            "Stanton nonparametric drift and diffusion",
            "at least three observations are required",
        )

    if bandwidth is not None:
        if not (math.isfinite(bandwidth) and bandwidth > 0):
            return _error(
                "traded_risk.stanton_nonparametric",
                "Stanton nonparametric drift and diffusion",
                f"bandwidth must be finite and strictly positive; got {bandwidth!r}",
            )
        h = float(bandwidth)
        bandwidth_rule = "supplied"
    else:
        sd = float(np.std(values[:-1], ddof=1))
        h = SILVERMAN_FACTOR * sd * (values[:-1].size ** (-1.0 / 5.0))
        bandwidth_rule = "silverman"
        if not (math.isfinite(h) and h > 0):
            return _error(
                "traded_risk.stanton_nonparametric",
                "Stanton nonparametric drift and diffusion",
                "Silverman bandwidth is not positive; the rate series is numerically constant",
            )

    lo, hi = float(np.percentile(values, 5)), float(np.percentile(values, 95))
    grid = np.linspace(lo, hi, int(n_grid))
    estimates = stanton_first_order(rates, ctx.dt, grid, h)

    thin = estimates[estimates["ess"] < min_ess]
    usable = estimates[estimates["ess"] >= min_ess]

    metrics: dict[str, Any] = {
        "estimator_order": 1,
        "kernel": "gaussian",
        "bandwidth": round(h, 12),
        "bandwidth_rule": bandwidth_rule,
        "n_grid_points": int(n_grid),
        "grid_min": round(lo, 10),
        "grid_max": round(hi, 10),
        "n_observations": int(values.size),
        "n_increments": int(values.size - 1),
        "dt": ctx.dt,
        "min_ess_threshold": min_ess,
        "n_thin_support_points": int(len(thin)),
        "min_ess_observed": round(float(estimates["ess"].min()), 6),
        "max_ess_observed": round(float(estimates["ess"].max()), 6),
        "median_ess": round(float(estimates["ess"].median()), 6),
        "grid_hash": hashlib.sha256(",".join(f"{g:.12e}" for g in grid).encode()).hexdigest()[:32],
    }
    if len(usable):
        metrics["mu_min"] = round(float(usable["mu"].min()), 12)
        metrics["mu_max"] = round(float(usable["mu"].max()), 12)
        metrics["sigma2_min"] = round(float(usable["sigma2"].min()), 15)
        metrics["sigma2_max"] = round(float(usable["sigma2"].max()), 15)
        metrics["mean_sigma2"] = round(float(usable["sigma2"].mean()), 15)

    status = Status.WARN if len(thin) else Status.RECORDED
    return TestResult(
        test_id="traded_risk.stanton_nonparametric",
        test_name="Stanton nonparametric drift and diffusion",
        status=status,
        params={"bandwidth": bandwidth, "n_grid": n_grid, "min_ess": min_ess},
        metrics=metrics,
        interpretation=(
            f"First-order kernel estimates on {n_grid} grid point(s) with bandwidth "
            f"{h:.6g} ({bandwidth_rule})"
            + (
                f"; {len(thin)} point(s) below the effective-sample-size threshold of {min_ess:g}."
                if len(thin)
                else "; support adequate at every point."
            )
        ),
        limitations=[
            "FIRST-ORDER ESTIMATOR ONLY. There is no order parameter and no "
            "higher-order path. Stanton's second- and third-order corrections reduce "
            "the O(dt) discretisation bias but require their own expressions and their "
            "own validation; their properties are not claimed here.",
            "The first-order approximation carries O(dt) discretisation bias that does "
            "not vanish at fixed observation frequency.",
            "Results are sensitive to the bandwidth. A wider bandwidth smooths genuine "
            "structure; a narrower one produces noise that looks like structure.",
            "Effective sample size is reported per grid point. Estimates at thin-support "
            "points are not suppressed but are flagged, because a smooth-looking curve "
            "extrapolated from negligible local support is the failure mode here.",
            "Boundary grid points have one-sided support and are biased toward the interior.",
            "Determinism: numerical.",
        ],
    )


# =========================================================================== #
# The canonical exception sequence
# =========================================================================== #
@dataclass
class ExceptionSequence:
    """One exception series, shared by every VaR backtest."""

    indicators: pd.Series  # 0/1, timestamp-indexed
    pnl_source: str  # actual | hypothetical
    confidence: float
    n_pnl: int
    n_var: int
    n_aligned: int
    n_dropped_alignment: int

    @property
    def n(self) -> int:
        return int(self.indicators.size)

    @property
    def n_exceptions(self) -> int:
        return int(self.indicators.sum())

    @property
    def expected_probability(self) -> float:
        return 1.0 - self.confidence

    def transition_counts(self) -> tuple[int, int, int, int]:
        """(n00, n01, n10, n11) over consecutive pairs."""
        values = self.indicators.to_numpy(dtype=int)
        if values.size < 2:
            return 0, 0, 0, 0
        previous, current = values[:-1], values[1:]
        return (
            int(np.sum((previous == 0) & (current == 0))),
            int(np.sum((previous == 0) & (current == 1))),
            int(np.sum((previous == 1) & (current == 0))),
            int(np.sum((previous == 1) & (current == 1))),
        )

    def base_metrics(self) -> dict[str, Any]:
        return {
            "pnl_source": self.pnl_source,
            "confidence": self.confidence,
            "var_confidence": self.confidence,
            "n_observations": self.n,
            "n_exceptions": self.n_exceptions,
            "exception_rate": round(self.n_exceptions / self.n, 10) if self.n else None,
            "expected_probability": round(self.expected_probability, 10),
            "alpha_var": round(self.expected_probability, 10),
            "expected_exceptions": round(self.expected_probability * self.n, 6),
            "n_pnl": self.n_pnl,
            "n_var": self.n_var,
            "n_aligned": self.n_aligned,
            "n_dropped_alignment": self.n_dropped_alignment,
            "exception_convention": "I_t = 1 iff PnL_t < -VaR_t, VaR a positive loss magnitude",
            "exception_indicator_hash": hashlib.sha256(
                self.indicators.to_numpy(dtype=int).tobytes()
            ).hexdigest()[:32],
        }


def exception_sequence(ctx: Any, pnl_source: str = "actual") -> ExceptionSequence:
    """The one canonical exception derivation. Every VaR surface calls this.

    Alignment is by explicit timestamp intersection, never positional zip: two Series
    with different indices that happen to share a length would otherwise be compared
    row-by-row against the wrong days, and nothing in the output would reveal it.
    """
    if pnl_source not in {"actual", "hypothetical"}:
        raise ValueError(f"pnl_source={pnl_source!r} must be 'actual' or 'hypothetical'")
    pnl = ctx.pnl if pnl_source == "actual" else ctx.hypothetical_pnl
    if pnl is None:
        raise ValueError(
            f"no {pnl_source} P&L supplied. Actual and hypothetical P&L are different "
            "inputs and are never substituted for one another: hypothetical revalues a "
            "frozen position, actual includes intraday trading and fees"
        )
    if ctx.var_series is None:
        raise ValueError("no VaR series supplied")
    if ctx.var_confidence is None:
        raise ValueError("var_confidence is required when a VaR series is supplied")
    if pnl.index.has_duplicates or ctx.var_series.index.has_duplicates:
        raise ValueError("duplicate timestamps in the P&L or VaR series")

    common = pnl.index.intersection(ctx.var_series.index)
    if len(common) == 0:
        raise ValueError("P&L and VaR series share no timestamps")

    aligned_pnl = pnl.reindex(common).astype(float)
    aligned_var = ctx.var_series.reindex(common).astype(float)
    valid = aligned_pnl.notna() & aligned_var.notna()
    aligned_pnl, aligned_var = aligned_pnl[valid], aligned_var[valid]

    indicators = (aligned_pnl < -aligned_var).astype(int)
    return ExceptionSequence(
        indicators=indicators,
        pnl_source=pnl_source,
        confidence=float(ctx.var_confidence),
        n_pnl=int(pnl.size),
        n_var=int(ctx.var_series.size),
        n_aligned=int(indicators.size),
        n_dropped_alignment=int(max(len(pnl), len(ctx.var_series)) - indicators.size),
    )


def _sequence_or_reason(ctx: Any, pnl_source: str) -> tuple[ExceptionSequence | None, str]:
    try:
        return exception_sequence(ctx, pnl_source), ""
    except ValueError as exc:
        return None, str(exc)


# =========================================================================== #
# VaR surfaces
# =========================================================================== #
@register_test(
    "traded_risk.var_exceptions",
    family="traded_risk",
    name="VaR exception sequence",
    requires=("pnl", "var_series"),
    default_params={"pnl_source": "actual"},
    context_type="market",
    risk_stripes=_MARKET_STRIPES,
    risk_dimensions=("accuracy_calibration", "monitoring"),
    object_kinds=_OBJECTS,
)
def var_exceptions(ctx: Any, pnl_source: str = "actual") -> TestResult:
    """The exception sequence itself. Determinism: exact counts."""
    sequence, reason = _sequence_or_reason(ctx, pnl_source)
    if sequence is None:
        return _skip("traded_risk.var_exceptions", "VaR exception sequence", reason, pnl_source=pnl_source)

    metrics = sequence.base_metrics()
    dates = sequence.indicators[sequence.indicators == 1].index
    metrics["first_exception"] = str(dates[0]) if len(dates) else None
    metrics["last_exception"] = str(dates[-1]) if len(dates) else None
    metrics["exception_dates_sample"] = ", ".join(str(d) for d in dates[:10])

    return TestResult(
        test_id="traded_risk.var_exceptions",
        test_name="VaR exception sequence",
        status=Status.RECORDED,
        params={"pnl_source": pnl_source},
        metrics=metrics,
        interpretation=(
            f"{sequence.n_exceptions} exception(s) in {sequence.n:,} aligned "
            f"observation(s) against {sequence.expected_probability * sequence.n:.1f} "
            f"expected at {sequence.confidence:.0%} confidence, using {pnl_source} P&L."
        ),
        limitations=[
            "Counts only; no hypothesis is tested here.",
            "Actual and hypothetical P&L are different inputs. Actual includes intraday "
            "trading and fees; hypothetical revalues a frozen position. The source used "
            "is recorded and never substituted.",
            "The full indicator series is summarised by a content hash rather than "
            "dumped into scalar evidence.",
        ],
    )


def kupiec_lr(n: int, x: int, p: float) -> float:
    """Unconditional-coverage LR statistic, limit-safe at x=0 and x=n.

    ``xlogy(a, b)`` returns 0 when ``a == 0`` instead of ``0 * -inf = nan``. Zero
    exceptions is a legitimate observed year, not an error, and a backtest that returns
    ``nan`` there is useless exactly when someone most wants an answer.
    """
    if n <= 0:
        return float("nan")
    pi = x / n
    ll_null = special.xlogy(x, p) + special.xlog1py(n - x, -p)
    ll_alt = special.xlogy(x, pi) + special.xlog1py(n - x, -pi)
    return float(-2.0 * (ll_null - ll_alt))


@register_test(
    "traded_risk.var_kupiec_pof",
    family="traded_risk",
    name="Kupiec proportion-of-failures test",
    requires=("pnl", "var_series"),
    default_params={"pnl_source": "actual", "alpha": 0.05},
    context_type="market",
    risk_stripes=_MARKET_STRIPES,
    risk_dimensions=("accuracy_calibration", "implementation_verification"),
    object_kinds=_OBJECTS,
)
def var_kupiec_pof(ctx: Any, pnl_source: str = "actual", alpha: float = 0.05) -> TestResult:
    """Kupiec (1995) unconditional coverage. Determinism: numerical."""
    sequence, reason = _sequence_or_reason(ctx, pnl_source)
    if sequence is None:
        return _skip(
            "traded_risk.var_kupiec_pof", "Kupiec proportion-of-failures test", reason, pnl_source=pnl_source
        )
    if sequence.n < 2:
        return _skip(
            "traded_risk.var_kupiec_pof",
            "Kupiec proportion-of-failures test",
            f"{sequence.n} aligned observation(s); at least 2 required",
        )

    n, x, p = sequence.n, sequence.n_exceptions, sequence.expected_probability
    lr = kupiec_lr(n, x, p)
    p_value = float(stats.chi2.sf(lr, df=1)) if math.isfinite(lr) else float("nan")
    rejected = math.isfinite(p_value) and p_value < alpha

    metrics = {
        **sequence.base_metrics(),
        "lr_uc": round(lr, 10),
        "p_value": round(p_value, 10),
        "degrees_of_freedom": 1,
        "alpha": alpha,
        "gamma_test": alpha,
        "statistical_gamma_test": alpha,
        "statistical_criterion_source": "STATISTICAL_TEST_SPECIFICATION",
        "empirical_probability": round(x / n, 10),
        "critical_value": round(float(stats.chi2.ppf(1.0 - alpha, df=1)), 10),
        "rejected": rejected,
        "boundary_case": "x=0" if x == 0 else ("x=n" if x == n else "interior"),
    }

    return TestResult(
        test_id="traded_risk.var_kupiec_pof",
        test_name="Kupiec proportion-of-failures test",
        status=Status.FAIL if rejected else Status.RECORDED,
        params={"pnl_source": pnl_source, "alpha": alpha, "gamma_test": alpha},
        metrics=metrics,
        interpretation=(
            f"At the {alpha:.0%} level, the null of correct unconditional coverage was "
            f"{'REJECTED' if rejected else 'not rejected'} "
            f"(LR = {lr:.4f}, p = {p_value:.4f}, {x} exception(s) against "
            f"{p * n:.1f} expected over {n:,} observation(s))."
            + ("" if rejected else " " + _NON_REJECTION_CAVEAT)
        ),
        limitations=[
            _NON_REJECTION_CAVEAT,
            "Unconditional coverage only: this test is indifferent to clustering. A "
            "model whose exceptions all fall in one week can pass it.",
            "Limit-safe likelihood arithmetic (xlogy/xlog1py), so x=0 and x=n are "
            "handled as legitimate observed sequences rather than producing NaN.",
            "The chi-square reference distribution is asymptotic; with few exceptions "
            "the finite-sample p-value is approximate.",
        ],
    )


def christoffersen_independence_lr(n00: int, n01: int, n10: int, n11: int) -> float:
    """Independence LR statistic, limit-safe on zero transition cells."""
    n0, n1 = n00 + n01, n10 + n11
    total = n0 + n1
    if total == 0:
        return float("nan")
    pi = (n01 + n11) / total
    pi0 = n01 / n0 if n0 > 0 else 0.0
    pi1 = n11 / n1 if n1 > 0 else 0.0

    ll_null = special.xlogy(n01 + n11, pi) + special.xlog1py(n00 + n10, -pi)
    ll_alt = (
        special.xlogy(n01, pi0)
        + special.xlog1py(n00, -pi0)
        + special.xlogy(n11, pi1)
        + special.xlog1py(n10, -pi1)
    )
    return float(-2.0 * (ll_null - ll_alt))


@register_test(
    "traded_risk.var_christoffersen_independence",
    family="traded_risk",
    name="Christoffersen independence test",
    requires=("pnl", "var_series"),
    default_params={"pnl_source": "actual", "alpha": 0.05},
    context_type="market",
    risk_stripes=_MARKET_STRIPES,
    risk_dimensions=("accuracy_calibration", "stability"),
    object_kinds=_OBJECTS,
)
def var_christoffersen_independence(ctx: Any, pnl_source: str = "actual", alpha: float = 0.05) -> TestResult:
    """Christoffersen (1998) independence. Determinism: numerical."""
    sequence, reason = _sequence_or_reason(ctx, pnl_source)
    if sequence is None:
        return _skip(
            "traded_risk.var_christoffersen_independence",
            "Christoffersen independence test",
            reason,
            pnl_source=pnl_source,
        )
    if sequence.n < 3:
        return _skip(
            "traded_risk.var_christoffersen_independence",
            "Christoffersen independence test",
            f"{sequence.n} aligned observation(s); at least 3 required",
        )

    n00, n01, n10, n11 = sequence.transition_counts()
    lr = christoffersen_independence_lr(n00, n01, n10, n11)
    p_value = float(stats.chi2.sf(lr, df=1)) if math.isfinite(lr) else float("nan")
    rejected = math.isfinite(p_value) and p_value < alpha

    n0, n1 = n00 + n01, n10 + n11
    metrics = {
        **sequence.base_metrics(),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "n_transitions": int(n0 + n1),
        "pi": round((n01 + n11) / (n0 + n1), 10) if (n0 + n1) else None,
        "pi_0": round(n01 / n0, 10) if n0 else None,
        "pi_1": round(n11 / n1, 10) if n1 else None,
        "lr_ind": round(lr, 10),
        "p_value": round(p_value, 10),
        "degrees_of_freedom": 1,
        "alpha": alpha,
        "gamma_test": alpha,
        "statistical_gamma_test": alpha,
        "statistical_criterion_source": "STATISTICAL_TEST_SPECIFICATION",
        "rejected": rejected,
        "has_zero_transition_cell": bool(min(n00, n01, n10, n11) == 0),
    }

    return TestResult(
        test_id="traded_risk.var_christoffersen_independence",
        test_name="Christoffersen independence test",
        status=Status.FAIL if rejected else Status.RECORDED,
        params={"pnl_source": pnl_source, "alpha": alpha, "gamma_test": alpha},
        metrics=metrics,
        interpretation=(
            f"At the {alpha:.0%} level, the null of serially independent exceptions was "
            f"{'REJECTED' if rejected else 'not rejected'} "
            f"(LR = {lr:.4f}, p = {p_value:.4f}; transitions "
            f"n00={n00}, n01={n01}, n10={n10}, n11={n11})."
            + ("" if rejected else " " + _NON_REJECTION_CAVEAT)
        ),
        limitations=[
            _NON_REJECTION_CAVEAT,
            "Tests first-order Markov dependence only. Clustering at longer lags is not detected.",
            "Independence alone says nothing about whether the exception RATE is "
            "correct; see the Kupiec and conditional-coverage tests.",
            "Zero transition cells are handled with limit-safe arithmetic and never produce NaN.",
        ],
    )


@register_test(
    "traded_risk.var_christoffersen_conditional",
    family="traded_risk",
    name="Christoffersen conditional coverage test",
    requires=("pnl", "var_series"),
    default_params={"pnl_source": "actual", "alpha": 0.05},
    context_type="market",
    risk_stripes=_MARKET_STRIPES,
    risk_dimensions=("accuracy_calibration", "stability"),
    object_kinds=_OBJECTS,
)
def var_christoffersen_conditional(ctx: Any, pnl_source: str = "actual", alpha: float = 0.05) -> TestResult:
    """Joint coverage and independence. ``LR_cc = LR_uc + LR_ind`` by construction.

    Both components are recomputed from the **same** canonical exception sequence, so
    the identity holds exactly rather than approximately.
    """
    sequence, reason = _sequence_or_reason(ctx, pnl_source)
    if sequence is None:
        return _skip(
            "traded_risk.var_christoffersen_conditional",
            "Christoffersen conditional coverage test",
            reason,
            pnl_source=pnl_source,
        )
    if sequence.n < 3:
        return _skip(
            "traded_risk.var_christoffersen_conditional",
            "Christoffersen conditional coverage test",
            f"{sequence.n} aligned observation(s); at least 3 required",
        )

    lr_uc = kupiec_lr(sequence.n, sequence.n_exceptions, sequence.expected_probability)
    n00, n01, n10, n11 = sequence.transition_counts()
    lr_ind = christoffersen_independence_lr(n00, n01, n10, n11)
    lr_cc = lr_uc + lr_ind
    p_value = float(stats.chi2.sf(lr_cc, df=2)) if math.isfinite(lr_cc) else float("nan")
    rejected = math.isfinite(p_value) and p_value < alpha

    metrics = {
        **sequence.base_metrics(),
        "lr_uc": round(lr_uc, 10),
        "lr_ind": round(lr_ind, 10),
        "lr_cc": round(lr_cc, 10),
        "identity_residual": round(lr_cc - (lr_uc + lr_ind), 15),
        "p_value": round(p_value, 10),
        "degrees_of_freedom": 2,
        "alpha": alpha,
        "gamma_test": alpha,
        "statistical_gamma_test": alpha,
        "statistical_criterion_source": "STATISTICAL_TEST_SPECIFICATION",
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "rejected": rejected,
    }

    return TestResult(
        test_id="traded_risk.var_christoffersen_conditional",
        test_name="Christoffersen conditional coverage test",
        status=Status.FAIL if rejected else Status.RECORDED,
        params={"pnl_source": pnl_source, "alpha": alpha, "gamma_test": alpha},
        metrics=metrics,
        interpretation=(
            f"At the {alpha:.0%} level, the joint null of correct coverage AND "
            f"independence was {'REJECTED' if rejected else 'not rejected'} "
            f"(LR_cc = {lr_cc:.4f} = {lr_uc:.4f} + {lr_ind:.4f}, p = {p_value:.4f})."
            + ("" if rejected else " " + _NON_REJECTION_CAVEAT)
        ),
        limitations=[
            _NON_REJECTION_CAVEAT,
            "LR_cc = LR_uc + LR_ind holds exactly because all three statistics are "
            "computed from the SAME canonical exception sequence. Recomputing the "
            "indicator with a different comparison convention would break the identity "
            "in a way that looks numerical rather than definitional.",
            "The independence component tests first-order Markov dependence only.",
        ],
    )


@register_test(
    "traded_risk.var_traffic_light",
    family="traded_risk",
    name="Traffic-light exception classification",
    requires=("pnl", "var_series"),
    default_params={"pnl_source": "actual", "strict_applicability": True},
    context_type="market",
    risk_stripes=_MARKET_STRIPES,
    risk_dimensions=("accuracy_calibration", "monitoring"),
    object_kinds=_OBJECTS,
)
def var_traffic_light(ctx: Any, pnl_source: str = "actual", strict_applicability: bool = True) -> TestResult:
    """The historical/classical traffic-light rule. Determinism: exact classification.

    The 0-4 / 5-9 / 10+ bands are calibrated to **250 observations at 99%**. Applying
    them to a different sample size or confidence level would be a generic classifier
    wearing regulatory clothing, so outside that configuration the test SKIPs rather
    than reusing the bands.
    """
    sequence, reason = _sequence_or_reason(ctx, pnl_source)
    if sequence is None:
        return _skip(
            "traded_risk.var_traffic_light",
            "Traffic-light exception classification",
            reason,
            pnl_source=pnl_source,
        )

    applicable = sequence.n == TRAFFIC_LIGHT_N and abs(sequence.confidence - TRAFFIC_LIGHT_CONFIDENCE) < 1e-12
    if strict_applicability and not applicable:
        return _skip(
            "traded_risk.var_traffic_light",
            "Traffic-light exception classification",
            f"not applicable: the historical bands are calibrated to "
            f"{TRAFFIC_LIGHT_N} observations at {TRAFFIC_LIGHT_CONFIDENCE:.0%}, but this "
            f"sample has {sequence.n} observation(s) at {sequence.confidence:.0%}. The "
            "bands are not reused as a generic classifier.",
            pnl_source=pnl_source,
            strict_applicability=strict_applicability,
            band_n_observations=TRAFFIC_LIGHT_N,
            band_confidence=TRAFFIC_LIGHT_CONFIDENCE,
            n_observations=sequence.n,
            confidence=sequence.confidence,
        )

    x = sequence.n_exceptions
    zone = next(name for lo, hi, name in TRAFFIC_LIGHT_BANDS if lo <= x <= hi)
    cumulative = float(stats.binom.cdf(x, sequence.n, sequence.expected_probability))

    metrics = {
        **sequence.base_metrics(),
        "zone": zone,
        "applicable": applicable,
        "band_n_observations": TRAFFIC_LIGHT_N,
        "band_confidence": TRAFFIC_LIGHT_CONFIDENCE,
        "cumulative_probability": round(cumulative, 10),
        "bands": "green 0-4, yellow 5-9, red 10+",
    }
    status = {"green": Status.RECORDED, "yellow": Status.WARN, "red": Status.FAIL}[zone]

    return TestResult(
        test_id="traded_risk.var_traffic_light",
        test_name="Traffic-light exception classification",
        status=status,
        params={"pnl_source": pnl_source, "strict_applicability": strict_applicability},
        metrics=metrics,
        interpretation=(
            f"{x} exception(s) in {sequence.n} observation(s) places the model in the "
            f"{zone.upper()} zone under the historical traffic-light rule."
        ),
        limitations=[
            "This implements the HISTORICAL/CLASSICAL traffic-light backtesting rule "
            "and is NOT a complete implementation of the current Basel/FRTB market-risk "
            "capital framework. No capital multiplier or regulatory conclusion follows "
            "from this classification.",
            f"The bands are calibrated to {TRAFFIC_LIGHT_N} observations at "
            f"{TRAFFIC_LIGHT_CONFIDENCE:.0%} and are not reused as a generic "
            "classifier outside that configuration.",
            "Zone is a count-based classification, not a hypothesis test.",
        ],
    )


# =========================================================================== #
# Brownian bridge
# =========================================================================== #
@register_test(
    "traded_risk.brownian_bridge_barrier",
    family="traded_risk",
    name="Brownian bridge barrier monitoring",
    requires=("prices",),
    default_params={"barrier": None, "direction": "up", "sigma": None},
    context_type="market",
    risk_stripes=_MARKET_STRIPES,
    risk_dimensions=("stress_scenario", "implementation_verification"),
    object_kinds=_OBJECTS,
)
def brownian_bridge_barrier(
    ctx: Any,
    barrier: float | None = None,
    direction: str = "up",
    sigma: float | None = None,
    asset: str | None = None,
) -> TestResult:
    """Discrete-monitoring under-detection of barrier crossings.

    Reuses the live-verified B2 log-price bridge helper; the equation is not duplicated
    here. Discrete monitoring only sees the endpoints of each interval, so it
    systematically misses excursions that cross and return within one period. The
    magnitude of that under-detection is the finding.

    This is **barrier/excursion risk monitoring**, not missing-data interpolation.
    """
    if direction not in {"up", "down"}:
        return _error(
            "traded_risk.brownian_bridge_barrier",
            "Brownian bridge barrier monitoring",
            f"direction={direction!r} must be 'up' or 'down'",
        )
    if barrier is None:
        return _skip(
            "traded_risk.brownian_bridge_barrier",
            "Brownian bridge barrier monitoring",
            "no barrier level supplied; a barrier is never inferred from the data",
            direction=direction,
        )

    prices = ctx.prices
    if prices is None:
        return _skip(
            "traded_risk.brownian_bridge_barrier",
            "Brownian bridge barrier monitoring",
            "a price series is required; returns alone cannot locate a barrier",
            barrier=barrier,
            direction=direction,
        )

    column = asset if asset is not None else list(prices.columns)[0]
    if column not in prices.columns:
        return _error(
            "traded_risk.brownian_bridge_barrier",
            "Brownian bridge barrier monitoring",
            f"asset {column!r} is not present in the price frame",
        )

    series = prices[column].dropna().astype(float)
    if series.size < 2:
        return _skip(
            "traded_risk.brownian_bridge_barrier",
            "Brownian bridge barrier monitoring",
            "at least two price observations are required",
        )
    if (series <= 0).any():
        return _error(
            "traded_risk.brownian_bridge_barrier",
            "Brownian bridge barrier monitoring",
            "non-positive prices: the log-price bridge is undefined",
        )

    dt = 1.0 / float(ctx.periods_per_year)
    if sigma is None:
        log_returns = np.diff(np.log(series.to_numpy()))
        sigma_used = float(np.std(log_returns, ddof=1)) * math.sqrt(ctx.periods_per_year)
        sigma_rule = "estimated from realised log returns"
    else:
        if not (math.isfinite(sigma) and sigma > 0):
            return _error(
                "traded_risk.brownian_bridge_barrier",
                "Brownian bridge barrier monitoring",
                f"sigma must be finite and positive; got {sigma!r}",
            )
        sigma_used, sigma_rule = float(sigma), "supplied"

    values = series.to_numpy()
    starts, ends = values[:-1], values[1:]
    n_intervals = int(starts.size)

    if direction == "up":
        discrete = (starts >= barrier) | (ends >= barrier)
    else:
        discrete = (starts <= barrier) | (ends <= barrier)
    n_discrete = int(discrete.sum())

    probabilities = np.array(
        [
            barrier_crossing_probability(
                float(a), float(b), float(barrier), sigma_used, dt, direction=direction
            )
            for a, b in zip(starts, ends, strict=True)
        ]
    )
    expected_continuous = float(probabilities.sum())
    under_detection = expected_continuous - n_discrete

    metrics: dict[str, Any] = {
        "asset": str(column),
        "barrier": float(barrier),
        "direction": direction,
        "sigma": round(sigma_used, 12),
        "sigma_rule": sigma_rule,
        "dt": dt,
        "periods_per_year": float(ctx.periods_per_year),
        "n_intervals": n_intervals,
        "n_discrete_crossings": n_discrete,
        "discrete_crossing_rate": round(n_discrete / n_intervals, 10),
        "expected_continuous_crossings": round(expected_continuous, 10),
        "expected_continuous_rate": round(expected_continuous / n_intervals, 10),
        "under_detection_count": round(under_detection, 10),
        "under_detection_ratio": (round(expected_continuous / n_discrete, 10) if n_discrete else None),
        "max_interval_probability": round(float(probabilities.max()), 10),
        "space": "log_price",
    }

    return TestResult(
        test_id="traded_risk.brownian_bridge_barrier",
        test_name="Brownian bridge barrier monitoring",
        status=Status.WARN if under_detection > 0.5 else Status.RECORDED,
        params={"barrier": barrier, "direction": direction, "sigma": sigma, "asset": asset},
        metrics=metrics,
        interpretation=(
            f"Discrete monitoring detected {n_discrete} crossing(s) of {barrier:g} over "
            f"{n_intervals} interval(s); the continuous bridge implies "
            f"{expected_continuous:.2f} expected crossing(s), an under-detection of "
            f"{under_detection:.2f}."
        ),
        limitations=[
            "This is BARRIER/EXCURSION RISK MONITORING. It is not missing-data "
            "interpolation and makes no claim about unobserved prices.",
            "The bridge is computed in LOG-PRICE space, which is correct for GBM. The "
            "arithmetic-price form applied to prices materially understates crossing "
            "probability and is not used.",
            "The bridge assumes constant volatility within each interval; under "
            "stochastic volatility the correction is approximate.",
            "Volatility is " + sigma_rule + "; the result is sensitive to it.",
            "Determinism: numerical.",
        ],
    )
