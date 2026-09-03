"""Target-type dispatch.

Why this is a guard and not a `TestSpec` field
----------------------------------------------

Roughly half of v4.2.0's tests have no target at all — portfolio, attribution, traded
risk and covariance never see one. A `TestSpec` field applies to every registered test, so
`supported_target_types` would sit empty and meaningless on all of them. The registry does
not need it to dispatch; the test does. So it is enforced at the point of use.

Why inference is a recommendation, not a determination
------------------------------------------------------

The obvious rule — *integer with few distinct values is multiclass* — quietly misclassifies
a count-regression target. Number of claims, days delinquent and number of prior defaults
are all small non-negative integers, and all are regression targets. Routed to multiclass
they would be handed AUC and Information Value, statistics that mean nothing for a count,
and the resulting numbers would look perfectly reasonable on a report.

So the ambiguous case is inferred **and announced**. The test still runs; the reviewer is
told an inference was made, what it was based on, and that it can be overridden. That is
the difference between a tool that guesses and a tool that guesses out loud.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = [
    "TargetType",
    "TargetInference",
    "infer_target_type",
    "require_target_type",
    "AMBIGUOUS_INTEGER_MAX_UNIQUE",
]

#: Above this many distinct integer values a target reads as continuous; at or below it,
#: an integer target is genuinely ambiguous between multiclass and count regression.
AMBIGUOUS_INTEGER_MAX_UNIQUE = 20


class TargetType(StrEnum):
    BINARY = "binary"
    MULTICLASS = "multiclass"
    CONTINUOUS = "continuous"
    NONE = "none"


@dataclass(frozen=True)
class TargetInference:
    """What was concluded about the target, and how confidently."""

    target_type: TargetType
    source: str  # explicit | dtype | cardinality | ambiguous_integer | absent
    confidence: str  # high | ambiguous
    n_unique: int = 0
    detail: str = ""

    @property
    def is_ambiguous(self) -> bool:
        return self.confidence == "ambiguous"

    def as_params(self) -> dict[str, Any]:
        """Recorded on every target-dependent test, so the inference is visible."""
        return {
            "target_type_inferred": self.target_type.value,
            "target_type_source": self.source,
            "target_type_confidence": self.confidence,
            "target_n_unique": self.n_unique,
        }


def infer_target_type(ctx: Any) -> TargetInference:
    """Resolve the target type, most authoritative signal first.

    An explicit ``ctx.extra["target_type"]`` is never overridden — if a reviewer has
    stated what the target is, no heuristic gets to disagree.
    """
    explicit = (ctx.extra or {}).get("target_type") if hasattr(ctx, "extra") else None
    if explicit:
        try:
            return TargetInference(
                target_type=TargetType(str(explicit).lower()),
                source="explicit",
                confidence="high",
                detail="supplied by the reviewer",
            )
        except ValueError:
            valid = ", ".join(t.value for t in TargetType)
            raise ValueError(
                f"target_type={explicit!r} is not recognised. Valid: {valid}"
            ) from None

    column = getattr(ctx, "target_column", None)
    frame = getattr(ctx, "train", None)
    if not column or frame is None or column not in getattr(frame, "columns", []):
        return TargetInference(
            target_type=TargetType.NONE,
            source="absent",
            confidence="high",
            detail="no target column available",
        )

    series = frame[column].dropna()
    n_unique = int(series.nunique())
    dtype = str(series.dtype)

    if n_unique <= 1:
        return TargetInference(
            TargetType.NONE, "cardinality", "high", n_unique,
            "target has fewer than two distinct values",
        )

    if n_unique == 2:
        return TargetInference(
            TargetType.BINARY, "cardinality", "high", n_unique,
            "exactly two distinct values",
        )

    # Non-numeric dtypes are unambiguously categorical. The spellings differ across
    # pandas versions — 2.x reports "object" and "string", 3.x reports "str" — so match
    # on the family rather than an exact name, and check numeric-ness positively rather
    # than enumerating every non-numeric spelling.
    is_numeric_dtype = dtype.startswith(("int", "uint", "float", "Int", "UInt", "Float"))
    if not is_numeric_dtype:
        return TargetInference(
            TargetType.MULTICLASS, "dtype", "high", n_unique,
            f"non-numeric dtype {dtype!r} with {n_unique} levels",
        )

    is_integer = dtype.startswith(("int", "uint", "Int", "UInt")) or (
        dtype.startswith(("float", "Float")) and bool((series == series.round()).all())
    )

    if is_integer and n_unique <= AMBIGUOUS_INTEGER_MAX_UNIQUE:
        return TargetInference(
            TargetType.MULTICLASS,
            "ambiguous_integer",
            "ambiguous",
            n_unique,
            f"{column!r} is integer-valued with {n_unique} distinct values. Inferred as "
            "multiclass, but a count-regression target (claims, days delinquent, prior "
            "defaults) has the same signature. If it is a count, set "
            'ctx.extra["target_type"] = "continuous" — otherwise AUC and Information '
            "Value will be computed for it and will not mean what they appear to mean.",
        )

    return TargetInference(
        TargetType.CONTINUOUS, "dtype", "high", n_unique,
        f"numeric dtype {dtype!r} with {n_unique} distinct values",
    )


def require_target_type(ctx: Any, *supported: str) -> tuple[TargetInference, Any | None]:
    """Guard at the top of a target-dependent test.

    Returns ``(inference, skip_result)``. When ``skip_result`` is not ``None`` the test
    should return it immediately — the target type is unsupported and the reason is
    already recorded. When it is ``None``, proceed.

    Usage::

        inference, skip = require_target_type(ctx, "binary")
        if skip is not None:
            return skip
    """
    from start.core.schemas import Status, TestResult

    inference = infer_target_type(ctx)
    accepted = {str(s).lower() for s in supported}

    if inference.target_type.value in accepted:
        return inference, None

    if inference.target_type is TargetType.NONE:
        reason = (
            "no usable target column is available, and this test requires one "
            f"({', '.join(sorted(accepted))})"
        )
    else:
        reason = (
            f"target is {inference.target_type.value}; this test supports "
            f"{', '.join(sorted(accepted))}. {inference.detail}"
        )

    return inference, TestResult(
        test_id="",  # filled by the caller
        test_name="",
        status=Status.SKIPPED,
        params=inference.as_params(),
        interpretation=reason,
        limitations=[
            "Skipped on target type. A statistic computed for the wrong target type "
            "produces a number that looks valid and means nothing, which is worse than "
            "no number at all."
        ],
    )
