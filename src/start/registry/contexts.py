"""Review contexts — the structural contract every context satisfies.

Why a Protocol and not a base class
-----------------------------------

`TestContext` is a pydantic model constructed at roughly 250 call sites. Putting it under
a new base class changes its MRO and its model construction, and every one of those sites
becomes a place the change could surface. A structural `Protocol` gets the same dispatch
guarantee with **zero runtime change to any existing object**: `TestContext` already has
`seed` and `extra`, and the three methods below are added to it as new methods only.

That is the whole of the v4.2.0 context architecture's contact with existing code.

Determinism classes
-------------------

Every registered test declares how reproducible it is. This matters more than it looks.
An earlier release classified linear-algebra results as bitwise-exact; they are not.
`numpy.linalg.lstsq`, `cond`, eigendecomposition and most of `scipy.stats` route through
BLAS/LAPACK, and Accelerate on macOS does not agree bitwise with OpenBLAS on Linux. A test
asserting exact equality on those passes locally and fails in CI for reasons that have
nothing to do with correctness — and the usual response is to loosen the assertion until
it stops complaining, which destroys the guarantee entirely.

So the class is declared honestly per test:

``EXACT``
    Bitwise identical on every platform. Counts, category sets, canonical hashes,
    index and set operations. Nothing that goes through BLAS.
``SEEDED``
    Identical given the same seed. Permutation importance, bootstrap, fold assignment.
``NUMERICAL``
    Identical to a declared tolerance. Everything touching floating-point linear
    algebra, statistics or optimisation.

The precision budget (v4.1.1) measures ``SEEDED`` and ``NUMERICAL`` and skips ``EXACT``,
which is both correct and cheap.

Standard library only.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ReviewContext",
    "ContextKind",
    "ContextMismatch",
    "Determinism",
    "DEFAULT_TOLERANCE",
    "context_methods_for_test_context",
]


class ContextKind(StrEnum):
    """What kind of data a context carries. Drives registry dispatch."""

    TABULAR = "tabular"
    MARKET = "market"
    SHORT_RATE = "short_rate"


class ContextMismatch(TypeError):
    """A test was invoked with a context kind it does not accept.

    A typed error rather than letting the call fail deep inside numpy with an
    unreadable traceback about a missing attribute.
    """


class Determinism(StrEnum):
    EXACT = "exact"
    SEEDED = "seeded"
    NUMERICAL = "numerical"


#: Default tolerances by class, overridable per test.
#: Closed-form linear algebra is tighter than an iterative optimiser, and saying so is
#: more useful than one number that is wrong for both.
DEFAULT_TOLERANCE: dict[str, float] = {
    "closed_form": 1e-10,
    "iterative": 1e-6,
    "kernel": 1e-8,
}


@runtime_checkable
class ReviewContext(Protocol):
    """The surface the registry needs from any context.

    ``TestContext`` satisfies this without modification to its fields. ``MarketContext``
    and ``ShortRateContext`` (Gate B) satisfy it independently — no shared inheritance,
    so no shared breakage.
    """

    seed: int
    extra: dict[str, Any]

    def context_kind(self) -> str:
        """One of :class:`ContextKind`."""
        ...

    def describe(self) -> dict[str, Any]:
        """Loggable summary. Never carries data — shapes and names only."""
        ...

    def validate_context(self) -> list[str]:
        """Structural violations, empty when the context is usable.

        Returned rather than raised: a test whose context is unusable should report
        ``SKIPPED`` with the reason, not abort the review.
        """
        ...


# --------------------------------------------------------------------------- #
# The three methods added to TestContext
# --------------------------------------------------------------------------- #
def context_methods_for_test_context() -> dict[str, Any]:
    """The additive methods that make ``TestContext`` satisfy :class:`ReviewContext`.

    Returned as a mapping rather than written into the class here, so the caller
    attaches them explicitly and the change to an existing model is visible at the
    point it happens instead of by import side effect.
    """

    def context_kind(self: Any) -> str:
        return ContextKind.TABULAR.value

    def describe(self: Any) -> dict[str, Any]:
        def shape(frame: Any) -> tuple[int, int] | None:
            return tuple(frame.shape) if frame is not None else None  # type: ignore[return-value]

        return {
            "kind": ContextKind.TABULAR.value,
            "train_shape": shape(self.train),
            "test_shape": shape(self.test),
            "oos_shape": shape(self.extra.get("oos")) if self.extra else None,
            "target_column": self.target_column,
            "score_column": self.score_column,
            "prediction_column": self.prediction_column,
            "timestamp_column": self.timestamp_column,
            "entity_id_column": self.entity_id_column,
            "has_model": self.model is not None,
            "seed": self.seed,
        }

    def validate_context(self: Any) -> list[str]:
        violations: list[str] = []
        if self.train is None:
            violations.append("train frame is absent")
        elif getattr(self.train, "empty", False):
            violations.append("train frame is empty")

        if self.target_column and self.train is not None:
            columns = list(getattr(self.train, "columns", []))
            if self.target_column not in columns:
                violations.append(
                    f"target_column {self.target_column!r} is not a column in train"
                )

        if self.train is not None and self.test is not None:
            train_cols = set(getattr(self.train, "columns", []))
            test_cols = set(getattr(self.test, "columns", []))
            missing = train_cols - test_cols - {self.target_column or ""}
            if missing:
                violations.append(
                    f"test is missing {len(missing)} column(s) present in train: "
                    f"{', '.join(sorted(missing)[:5])}"
                )
        return violations

    return {
        "context_kind": context_kind,
        "describe": describe,
        "validate_context": validate_context,
    }
