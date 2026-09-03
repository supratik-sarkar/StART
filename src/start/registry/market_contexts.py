"""Market and short-rate review contexts.

Gate B adds two context kinds alongside the tabular one. They satisfy the same
structural ``ReviewContext`` protocol Gate A established, and ``TestContext`` is not
touched — that is the whole reason the protocol was chosen over a base class.

Why portfolio state lives in one place
--------------------------------------

``PortfolioSpec`` is the sole owner of weights, benchmark weights, prior weights and
constraints. There is deliberately no parallel ``weights`` field on ``MarketContext``,
no separate attribution weights and no optimiser weights field.

The reason is concrete: optimisation, attribution and VaR all need the portfolio, and if
each could be handed its own copy they would eventually disagree. A risk decomposition
computed on one weight vector and a VaR backtest computed on another produce a review
that is internally inconsistent in a way nothing would flag.

Canonical fingerprinting
------------------------

Both contexts implement ``fingerprint()``, feeding the **existing**
``EvidenceRecord.input_artifact_hash``. No schema changes anywhere.

The serializer is written here rather than borrowed from pandas. ``hash_pandas_object``
is fast and entirely adequate for equality checks inside one process, but its output is
an implementation detail that may vary between pandas versions — and a fingerprint that
silently changes when a dependency is upgraded would invalidate every cached evidence
record for no semantic reason.

Two hardening decisions worth stating:

**Typed scalars.** ``str(value)`` collapses ``1``, ``"1"`` and ``True`` onto the same
token. They are different values and must hash differently, so every scalar carries an
explicit type tag.

**Float rendering.** ``float.hex()`` rather than ``repr``: exact, round-trippable, and
stable across Python versions. ``repr`` has changed between versions and loses the last
bit.

Standard library plus numpy/pandas.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from start.registry.contexts import ContextKind

__all__ = [
    "MarketContext",
    "ShortRateContext",
    "PortfolioSpec",
    "PortfolioConstraints",
    "canonical_scalar",
    "canonical_frame_bytes",
    "canonical_series_bytes",
    "FREQUENCY_PERIODS",
    "convert_rate",
]

#: Label -> periods per year. ``periods_per_year`` is canonical; this is a convenience
#: that resolves to it. A supplied numeric value always overrides the label, because 252,
#: 260 and 365 are all legitimate "daily" conventions and the choice must be visible.
FREQUENCY_PERIODS: dict[str, float] = {
    "daily": 252.0,
    "business_daily": 252.0,
    "calendar_daily": 365.0,
    "weekly": 52.0,
    "monthly": 12.0,
    "quarterly": 4.0,
    "annual": 1.0,
    "yearly": 1.0,
}


def convert_rate(rate: float, from_periods: float, to_periods: float) -> float:
    """Convert an effective periodic rate between compounding frequencies.

    ``(1 + r_from) ** (from/to) - 1``. Compounding, not proportional: subtracting an
    annual risk-free rate from a daily return is wrong by roughly a factor of 252, and
    nothing downstream would show it.
    """
    if from_periods <= 0 or to_periods <= 0:
        raise ValueError("periods per year must be positive")
    if from_periods == to_periods:
        return float(rate)
    annual = (1.0 + float(rate)) ** from_periods - 1.0
    return (1.0 + annual) ** (1.0 / to_periods) - 1.0


# --------------------------------------------------------------------------- #
# Canonical serialisation
# --------------------------------------------------------------------------- #
def canonical_scalar(value: Any) -> bytes:
    """Type-tagged canonical bytes for one scalar.

    The tag matters. Without it ``1``, ``"1"`` and ``True`` all render as ``"1"`` and
    hash identically, so a context whose column label is the integer 1 would be
    indistinguishable from one whose label is the string "1".
    """
    if value is None or (isinstance(value, float) and value is None):
        return b"\x00N"
    if isinstance(value, (bool, np.bool_)):
        return b"\x00b1" if bool(value) else b"\x00b0"
    if isinstance(value, (int, np.integer)):
        return b"\x00i" + str(int(value)).encode("ascii")
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f):
            return b"\x00fNaN"
        if math.isinf(f):
            return b"\x00fInf" if f > 0 else b"\x00f-Inf"
        if f == 0.0:
            f = 0.0  # normalise -0.0; IEEE distinguishes them, semantics do not
        return b"\x00f" + f.hex().encode("ascii")
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        stamp = pd.Timestamp(value)
        stamp = stamp.tz_convert("UTC") if stamp.tzinfo is not None else stamp.tz_localize("UTC")
        return b"\x00t" + str(stamp.value).encode("ascii")
    if isinstance(value, (bytes, bytearray)):
        return b"\x00y" + bytes(value).hex().encode("ascii")
    text = unicodedata.normalize("NFC", str(value))
    return b"\x00s" + text.encode("utf-8")


def _canonical_index(index: pd.Index, *, label: str) -> bytes:
    """Canonicalise an index, rejecting the ambiguous cases rather than guessing."""
    if index.has_duplicates:
        raise ValueError(f"{label}: duplicate index entries make alignment ambiguous")

    if isinstance(index, pd.DatetimeIndex):
        aware = index.tz is not None
        # A mixed aware/naive index cannot be ordered unambiguously, and silently
        # picking one interpretation would shift every timestamp by an unknown offset.
        if aware:
            index = index.tz_convert("UTC")
        else:
            index = index.tz_localize("UTC")
    digest = bytearray()
    for entry in index:
        digest += canonical_scalar(entry)
    return bytes(digest)


def _sorted_labels(labels: Any) -> list[Any]:
    """Deterministic ordering across mixed label types.

    Ordinary ``sorted()`` raises on a frame with both integer and string column names,
    which is legal pandas. Sorting on ``(type name, canonical bytes)`` is total and
    stable regardless of what the labels are.
    """
    return sorted(labels, key=lambda x: (type(x).__name__, canonical_scalar(x)))


def canonical_series_bytes(series: pd.Series | None, *, label: str = "series") -> bytes:
    if series is None:
        return b"\x00absent"
    digest = bytearray(b"S")
    digest += _canonical_index(series.index, label=label)
    values = series
    if pd.api.types.is_float_dtype(values):
        values = values.astype("float64")
    elif pd.api.types.is_integer_dtype(values):
        values = values.astype("int64")
    for value in values.tolist():
        digest += canonical_scalar(value)
    return bytes(digest)


def canonical_frame_bytes(frame: pd.DataFrame | None, *, label: str = "frame") -> bytes:
    """Canonical bytes for a frame.

    Columns are sorted, because column order is presentation rather than meaning. The
    row index is **not** sorted into a new order — it is canonicalised in place after a
    duplicate check — because row order carries meaning for time series, and reordering
    would make a future-leaking construction hash identically to a correct one.
    """
    if frame is None:
        return b"\x00absent"
    if frame.columns.has_duplicates:
        raise ValueError(f"{label}: duplicate column labels make column identity ambiguous")

    digest = bytearray(b"F")
    digest += f"{frame.shape[0]}x{frame.shape[1]}".encode("ascii")
    digest += _canonical_index(frame.index, label=label)
    for column in _sorted_labels(frame.columns):
        digest += b"\x1fC" + canonical_scalar(column)
        series = frame[column]
        if pd.api.types.is_float_dtype(series):
            series = series.astype("float64")
        elif pd.api.types.is_integer_dtype(series):
            series = series.astype("int64")
        for value in series.tolist():
            digest += canonical_scalar(value)
    return bytes(digest)


# --------------------------------------------------------------------------- #
# Portfolio state
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PortfolioConstraints:
    """Optimisation constraints, with every formula stated.

    ``max_concentration`` is restricted to non-negative weights **by design**. The
    Herfindahl index is a concentration measure for a long-only allocation; applied to a
    long/short book it is not a concentration measure at all, because a large short and
    a large long contribute identically while representing opposite exposures. Rather
    than pretend it generalises, v4.2.0 rejects the combination explicitly.
    """

    #: Per asset: w_i >= 0.
    long_only: bool = True
    #: Per asset floor and cap.
    min_weight: float | None = None
    max_weight: float | None = None
    #: Global: sum(w) == budget.
    budget: float = 1.0
    #: Global gross leverage: sum(|w_i|) <= max_leverage.
    max_leverage: float | None = None
    #: Global one-way turnover: 0.5 * sum(|w_i - w_prior_i|) <= max_turnover.
    #: The 0.5 makes a complete replacement of the book equal 1.0 rather than 2.0.
    max_turnover: float | None = None
    #: Global Herfindahl: sum(w_i^2) <= max_concentration. Non-negative weights only.
    max_concentration: float | None = None
    #: Per-asset lower and upper bound maps
    asset_lower_bounds: dict[str, float] | None = None
    asset_upper_bounds: dict[str, float] | None = None
    #: Benchmark-relative tracking-error bound: sqrt((w-w_b)' Sigma (w-w_b)) <= max_tracking_error
    max_tracking_error: float | None = None
    #: Factor exposure bounds specification
    factor_constraints: Any = None
    #: Group / sector exposure bounds specification
    group_constraints: Any = None
    #: Transaction cost specification
    transaction_cost: Any = None

    def validate(self) -> list[str]:
        problems: list[str] = []
        if self.min_weight is not None and self.max_weight is not None:
            if self.min_weight > self.max_weight:
                problems.append("min_weight exceeds max_weight")
        if self.asset_lower_bounds and self.asset_upper_bounds:
            for k in set(self.asset_lower_bounds) & set(self.asset_upper_bounds):
                if self.asset_lower_bounds[k] > self.asset_upper_bounds[k]:
                    problems.append(f"asset_lower_bounds[{k}] exceeds asset_upper_bounds[{k}]")
        if self.max_concentration is not None and not self.long_only:
            problems.append(
                "max_concentration requires long_only: the Herfindahl index is not a "
                "concentration measure on a long/short book, where a large short and a "
                "large long contribute identically despite opposite exposure"
            )
        if self.max_leverage is not None and self.max_leverage < abs(self.budget):
            problems.append("max_leverage is below the budget and cannot be satisfied")
        if self.max_turnover is not None and self.max_turnover < 0:
            problems.append("max_turnover must be non-negative")
        if self.max_tracking_error is not None and self.max_tracking_error < 0:
            problems.append("max_tracking_error must be non-negative")
        return problems

    def as_dict(self) -> dict[str, Any]:
        return {
            "long_only": self.long_only,
            "min_weight": self.min_weight,
            "max_weight": self.max_weight,
            "budget": self.budget,
            "max_leverage": self.max_leverage,
            "max_turnover": self.max_turnover,
            "max_concentration": self.max_concentration,
            "asset_lower_bounds": self.asset_lower_bounds,
            "asset_upper_bounds": self.asset_upper_bounds,
            "max_tracking_error": self.max_tracking_error,
            "factor_constraints": (
                getattr(self.factor_constraints, "as_dict", lambda: self.factor_constraints)()
                if self.factor_constraints
                else None
            ),
            "group_constraints": (
                getattr(self.group_constraints, "as_dict", lambda: self.group_constraints)()
                if self.group_constraints
                else None
            ),
            "transaction_cost": (
                getattr(self.transaction_cost, "as_dict", lambda: self.transaction_cost)()
                if self.transaction_cost
                else None
            ),
        }


@dataclass
class PortfolioSpec:
    """The sole owner of portfolio state."""

    weights: pd.Series | pd.DataFrame | None = None
    benchmark_weights: pd.Series | pd.DataFrame | None = None
    prior_weights: pd.Series | None = None
    constraints: PortfolioConstraints | None = None
    currency: str = ""

    def canonical_bytes(self) -> bytes:
        digest = bytearray(b"P")
        for name, value in (
            ("weights", self.weights),
            ("benchmark_weights", self.benchmark_weights),
            ("prior_weights", self.prior_weights),
        ):
            digest += b"\x1f" + name.encode("ascii")
            if value is None:
                digest += b"\x00absent"
            elif isinstance(value, pd.DataFrame):
                digest += canonical_frame_bytes(value, label=name)
            else:
                digest += canonical_series_bytes(value, label=name)
        digest += b"\x1fconstraints"
        if self.constraints is None:
            digest += b"\x00absent"
        else:
            for key, value in sorted(self.constraints.as_dict().items()):
                digest += canonical_scalar(key) + canonical_scalar(value)
        digest += b"\x1fcurrency" + canonical_scalar(self.currency)
        return bytes(digest)

    def describe(self) -> dict[str, Any]:
        def shape(value: Any) -> Any:
            if value is None:
                return None
            return list(value.shape) if hasattr(value, "shape") else None

        return {
            "weights_shape": shape(self.weights),
            "benchmark_shape": shape(self.benchmark_weights),
            "has_prior_weights": self.prior_weights is not None,
            "has_constraints": self.constraints is not None,
            "currency": self.currency,
        }


# --------------------------------------------------------------------------- #
# MarketContext
# --------------------------------------------------------------------------- #
@dataclass
class MarketContext:
    """Time-indexed market observations. No train/test split, no target column."""

    returns: pd.DataFrame | None = None
    prices: pd.DataFrame | None = None
    periods_per_year: float = 252.0
    frequency: str | None = None
    return_basis: str = "simple"          # simple | log
    risk_free_rate: float | pd.Series | None = None
    risk_free_frequency: str | None = None
    factor_returns: pd.DataFrame | None = None
    factor_exposures: pd.DataFrame | dict[Any, pd.DataFrame] | None = None
    covariance: pd.DataFrame | None = None
    pnl: pd.Series | None = None
    hypothetical_pnl: pd.Series | None = None
    var_series: pd.Series | None = None
    var_confidence: float | None = None
    asset_metadata: pd.DataFrame | None = None
    portfolio: PortfolioSpec | None = None
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    #: Populated at construction: exposures are canonicalised exactly once.
    _exposures_canonical: dict[Any, pd.DataFrame] | None = field(default=None, repr=False)
    _exposures_time_varying: bool = field(default=False, repr=False)
    _naive_timestamps_assumed_utc: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.frequency and self.frequency in FREQUENCY_PERIODS:
            # An explicitly supplied periods_per_year always wins: 252, 260 and 365 are
            # all legitimate "daily", and the label must not silently override a choice.
            if self.periods_per_year == 252.0:
                self.periods_per_year = FREQUENCY_PERIODS[self.frequency]
        if self.return_basis not in {"simple", "log"}:
            raise ValueError(f"return_basis={self.return_basis!r} must be 'simple' or 'log'")
        self._canonicalise_exposures()
        for frame in (self.returns, self.prices):
            if frame is not None and isinstance(frame.index, pd.DatetimeIndex):
                if frame.index.tz is None:
                    self._naive_timestamps_assumed_utc = True

    # -- exposures ---------------------------------------------------------
    def _canonicalise_exposures(self) -> None:
        """Expand a static exposure matrix across the return support, once.

        A static frame and a time-varying dict that repeats the same frame across the
        **entire** return support are semantically identical and must fingerprint
        identically. A dict covering only part of the support is *not* the same thing
        and is left as supplied, so the difference stays visible.
        """
        if self.factor_exposures is None:
            self._exposures_canonical = None
            return
        if isinstance(self.factor_exposures, pd.DataFrame):
            support = self._return_support()
            self._exposures_time_varying = False
            self._exposures_canonical = {ts: self.factor_exposures for ts in support}
            return
        if isinstance(self.factor_exposures, dict):
            keys = list(self.factor_exposures)
            support = self._return_support()
            covers_all = bool(support) and set(keys) == set(support)
            frames = list(self.factor_exposures.values())
            identical = all(f.equals(frames[0]) for f in frames) if frames else False
            # Only a dict that repeats one matrix over the whole support collapses to
            # the static representation.
            self._exposures_time_varying = not (covers_all and identical)
            self._exposures_canonical = dict(self.factor_exposures)
            return
        raise TypeError("factor_exposures must be a DataFrame, a dict of them, or None")

    def _return_support(self) -> list[Any]:
        frame = self.returns if self.returns is not None else self.prices
        return list(frame.index) if frame is not None else []

    def exposures_at(self, timestamp: Any) -> pd.DataFrame | None:
        if not self._exposures_canonical:
            return None
        if timestamp in self._exposures_canonical:
            return self._exposures_canonical[timestamp]
        return None

    @property
    def is_time_varying_exposure(self) -> bool:
        return self._exposures_time_varying

    # -- derived -----------------------------------------------------------
    def effective_returns(self) -> pd.DataFrame | None:
        """Returns as supplied, or derived from prices under ``return_basis``."""
        if self.returns is not None:
            return self.returns
        if self.prices is None:
            return None
        if self.return_basis == "log":
            return np.log(self.prices / self.prices.shift(1)).iloc[1:]
        return (self.prices / self.prices.shift(1) - 1.0).iloc[1:]

    def risk_free_per_period(self) -> tuple[float | pd.Series | None, dict[str, Any]]:
        """Risk-free converted to the RETURN period, with the conversion recorded."""
        if self.risk_free_rate is None:
            return None, {"risk_free_supplied": False}
        source = FREQUENCY_PERIODS.get(
            (self.risk_free_frequency or "annual").lower(), 1.0
        )
        record = {
            "risk_free_supplied": True,
            "risk_free_source_frequency": self.risk_free_frequency or "annual",
            "risk_free_source_periods_per_year": source,
            "risk_free_target_periods_per_year": self.periods_per_year,
            "risk_free_conversion": "effective: (1+r)^(from/to) - 1",
        }
        if isinstance(self.risk_free_rate, pd.Series):
            converted = self.risk_free_rate.apply(
                lambda r: convert_rate(float(r), source, self.periods_per_year)
            )
            return converted, record
        value = convert_rate(float(self.risk_free_rate), source, self.periods_per_year)
        record["risk_free_period_rate"] = value
        return value, record

    # -- ReviewContext -----------------------------------------------------
    def context_kind(self) -> str:
        return ContextKind.MARKET.value

    def describe(self) -> dict[str, Any]:
        def shape(frame: Any) -> Any:
            return list(frame.shape) if frame is not None else None

        return {
            "kind": ContextKind.MARKET.value,
            "returns_shape": shape(self.returns),
            "prices_shape": shape(self.prices),
            "periods_per_year": self.periods_per_year,
            "return_basis": self.return_basis,
            "factor_returns_shape": shape(self.factor_returns),
            "has_exposures": self._exposures_canonical is not None,
            "exposures_time_varying": self._exposures_time_varying,
            "has_pnl": self.pnl is not None,
            "has_hypothetical_pnl": self.hypothetical_pnl is not None,
            "has_var_series": self.var_series is not None,
            "var_confidence": self.var_confidence,
            "portfolio": self.portfolio.describe() if self.portfolio else None,
            "seed": self.seed,
        }

    def validate_context(self) -> list[str]:
        problems: list[str] = []
        if self.returns is None and self.prices is None:
            problems.append("neither returns nor prices supplied")
        if self.periods_per_year <= 0:
            problems.append("periods_per_year must be positive")
        if self.var_series is not None and self.var_confidence is None:
            problems.append("var_series supplied without var_confidence")

        indexed = {
            "returns": self.returns, "prices": self.prices,
            "factor_returns": self.factor_returns, "pnl": self.pnl,
            "hypothetical_pnl": self.hypothetical_pnl, "var_series": self.var_series,
        }
        for name, obj in indexed.items():
            if obj is None:
                continue
            index = obj.index
            if index.has_duplicates:
                problems.append(f"{name}: duplicate index entries")
            if isinstance(index, pd.DatetimeIndex) and not index.is_monotonic_increasing:
                problems.append(f"{name}: index is not monotonically increasing")
        if self.portfolio and self.portfolio.constraints:
            problems.extend(
                f"constraints: {p}" for p in self.portfolio.constraints.validate()
            )
        return problems

    def fingerprint(self) -> str:
        """SHA-256 over canonicalised inputs. Feeds the existing input_artifact_hash."""
        digest = hashlib.sha256()
        digest.update(b"MarketContext/1")
        for name, frame in (
            ("returns", self.returns), ("prices", self.prices),
            ("factor_returns", self.factor_returns), ("covariance", self.covariance),
            ("asset_metadata", self.asset_metadata),
        ):
            digest.update(b"\x1e" + name.encode("ascii"))
            digest.update(canonical_frame_bytes(frame, label=name))
        for name, series in (
            ("pnl", self.pnl), ("hypothetical_pnl", self.hypothetical_pnl),
            ("var_series", self.var_series),
        ):
            digest.update(b"\x1e" + name.encode("ascii"))
            digest.update(canonical_series_bytes(series, label=name))

        digest.update(b"\x1eexposures")
        if self._exposures_canonical is None:
            digest.update(b"\x00absent")
        else:
            for key in _sorted_labels(self._exposures_canonical):
                digest.update(canonical_scalar(key))
                digest.update(canonical_frame_bytes(
                    self._exposures_canonical[key], label="exposures"
                ))

        digest.update(b"\x1eportfolio")
        digest.update(self.portfolio.canonical_bytes() if self.portfolio else b"\x00absent")

        for name, value in (
            ("periods_per_year", self.periods_per_year),
            ("return_basis", self.return_basis),
            ("risk_free_frequency", self.risk_free_frequency),
            ("var_confidence", self.var_confidence),
            ("seed", self.seed),
        ):
            digest.update(b"\x1e" + name.encode("ascii") + canonical_scalar(value))
        digest.update(b"\x1erisk_free")
        if isinstance(self.risk_free_rate, pd.Series):
            digest.update(canonical_series_bytes(self.risk_free_rate, label="rf"))
        else:
            digest.update(canonical_scalar(self.risk_free_rate))
        return digest.hexdigest()


# --------------------------------------------------------------------------- #
# ShortRateContext
# --------------------------------------------------------------------------- #
@dataclass
class ShortRateContext:
    """A single short-rate process with a known observation interval.

    Separate from ``MarketContext`` because the diffusion estimators need one scalar
    process and a known Δt. Folding it into a multi-asset container would make those
    assumptions invisible at the call site.
    """

    rates: pd.Series | None = None
    units: str = "decimal"                 # decimal | percent
    periods_per_year: float = 252.0
    frequency: str | None = None
    day_count: str = "act/365"
    min_observations: int = 250
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    _normalised_from_percent: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        if self.units not in {"decimal", "percent"}:
            raise ValueError(f"units={self.units!r} must be 'decimal' or 'percent'")
        if self.frequency and self.frequency in FREQUENCY_PERIODS:
            if self.periods_per_year == 252.0:
                self.periods_per_year = FREQUENCY_PERIODS[self.frequency]

    def decimal_rates(self) -> pd.Series | None:
        """Rates in decimal, with the normalisation recorded rather than implicit."""
        if self.rates is None:
            return None
        if self.units == "percent":
            self._normalised_from_percent = True
            return self.rates / 100.0
        return self.rates

    @property
    def dt(self) -> float:
        return 1.0 / self.periods_per_year

    def context_kind(self) -> str:
        return ContextKind.SHORT_RATE.value

    def describe(self) -> dict[str, Any]:
        return {
            "kind": ContextKind.SHORT_RATE.value,
            "n_observations": int(self.rates.size) if self.rates is not None else 0,
            "units": self.units,
            "normalised_from_percent": self._normalised_from_percent,
            "periods_per_year": self.periods_per_year,
            "day_count": self.day_count,
            "min_observations": self.min_observations,
            "dt": self.dt,
            "seed": self.seed,
        }

    def validate_context(self) -> list[str]:
        problems: list[str] = []
        if self.rates is None or self.rates.empty:
            problems.append("no short-rate observations supplied")
            return problems
        if self.rates.index.has_duplicates:
            problems.append("duplicate index entries")
        if isinstance(self.rates.index, pd.DatetimeIndex):
            if not self.rates.index.is_monotonic_increasing:
                problems.append("index is not monotonically increasing")
        # Missing observations are REJECTED, not filled. A diffusion estimate over a
        # forward-filled series is biased toward zero and the bias is invisible in the
        # output, which is the worst combination.
        n_missing = int(self.rates.isna().sum())
        if n_missing:
            problems.append(
                f"{n_missing} missing observation(s): short-rate data is rejected rather "
                "than filled, because a diffusion estimate over interpolated values is "
                "biased toward zero with no visible symptom"
            )
        if int(self.rates.dropna().size) < self.min_observations:
            problems.append(
                f"{self.rates.dropna().size} observation(s) below the required "
                f"{self.min_observations}"
            )
        return problems

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"ShortRateContext/1")
        digest.update(canonical_series_bytes(self.rates, label="rates"))
        for name, value in (
            ("units", self.units), ("periods_per_year", self.periods_per_year),
            ("day_count", self.day_count), ("min_observations", self.min_observations),
            ("seed", self.seed),
        ):
            digest.update(b"\x1e" + name.encode("ascii") + canonical_scalar(value))
        return digest.hexdigest()
