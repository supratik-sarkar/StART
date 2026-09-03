"""Test coverage for a risk plan.

What this bridges
-----------------

``start.risk`` answers *what does this review owe?* — a plan of dimensions, each with a
question, a phase and a rationale. The registry answers *what can this system compute?*
Until now nothing connected them, so a plan was a statement of obligation with no
executable meaning, and the registry was a pile of tests with no stated purpose.

``tests_for_plan`` joins the two: for each planned dimension, which registered tests
declare that they supply evidence for it.

The gap is the point
--------------------

The interesting output is not the tests that exist. It is
:attr:`PlanCoverage.uncovered` — planned dimensions with **no** candidate test at all.

A dimension that is owed and cannot be evidenced by anything in the registry is a real
finding about the review's scope, and it should be visible before a reviewer starts work
rather than discovered at sign-off. Reporting only the matches would make the system look
complete by hiding what it cannot do.

Coverage is candidacy, not discharge
------------------------------------

A dimension with three candidate tests is not thereby satisfied. The tests must actually
run, produce evidence, and be judged adequate. This module answers *"is there anything
that could speak to this?"* — nothing stronger, and every output says so.

Standard library only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

__all__ = [
    "DimensionCoverage",
    "PlanCoverage",
    "tests_for_plan",
    "coverage_for_plan",
    "normalise_context_type",
    "CONTEXT_ALIASES",
]

#: One coherent vocabulary. ``TestSpec.context_type`` is canonically ``"tabular"``;
#: callers sometimes reach for the class name instead. Normalising here rather than
#: matching loosely keeps a single contract: an unrecognised value still returns
#: nothing, so a typo is visible rather than silently matching everything.
CONTEXT_ALIASES: dict[str, str] = {
    "tabular": "tabular",
    "testcontext": "tabular",
    "market": "market",
    "marketcontext": "market",
    "short_rate": "short_rate",
    "shortratecontext": "short_rate",
}


def normalise_context_type(context_type: str | None) -> str | None:
    """Map a caller-supplied context name onto the canonical registry vocabulary."""
    if context_type is None:
        return None
    key = str(context_type).strip().lower().replace("-", "_")
    return CONTEXT_ALIASES.get(key, key)


@dataclass(frozen=True)
class DimensionCoverage:
    """Which registered tests could supply evidence for one planned dimension."""

    dimension_id: str
    label: str
    required: bool
    phase: int
    test_ids: tuple[str, ...]
    #: Tests matching the dimension but declaring a different context type — they
    #: cannot run against the cohort in hand, and saying so is more useful than
    #: silently omitting them.
    unavailable_test_ids: tuple[str, ...] = field(default=())

    @property
    def covered(self) -> bool:
        return bool(self.test_ids)

    @property
    def is_gap(self) -> bool:
        """A required dimension with nothing that can speak to it."""
        return self.required and not self.test_ids

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension_id,
            "label": self.label,
            "required": self.required,
            "phase": self.phase,
            "n_candidate_tests": len(self.test_ids),
            "test_ids": list(self.test_ids),
            "unavailable_test_ids": list(self.unavailable_test_ids),
            "covered": self.covered,
            "is_gap": self.is_gap,
        }


@dataclass
class PlanCoverage:
    """Candidate tests for every dimension in a plan, and the gaps."""

    stripe_id: str
    object_kind: str
    materiality: str
    context_type: str
    dimensions: tuple[DimensionCoverage, ...]

    @property
    def uncovered(self) -> tuple[DimensionCoverage, ...]:
        return tuple(d for d in self.dimensions if not d.covered)

    @property
    def gaps(self) -> tuple[DimensionCoverage, ...]:
        """Required dimensions with no candidate test. The finding that matters."""
        return tuple(d for d in self.dimensions if d.is_gap)

    def mapping(self) -> dict[str, list[str]]:
        """``dimension_id -> [test_id, ...]``, the plain form."""
        return {d.dimension_id: list(d.test_ids) for d in self.dimensions}

    def as_dict(self) -> dict[str, Any]:
        return {
            "stripe": self.stripe_id,
            "object_kind": self.object_kind,
            "materiality": self.materiality,
            "context_type": self.context_type,
            "n_dimensions": len(self.dimensions),
            "n_covered": sum(1 for d in self.dimensions if d.covered),
            "n_uncovered": len(self.uncovered),
            "n_required_gaps": len(self.gaps),
            "required_gap_dimensions": [d.dimension_id for d in self.gaps],
            "dimensions": [d.as_dict() for d in self.dimensions],
        }

    def coverage_hash(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    def summary_lines(self) -> list[str]:
        lines = [
            f"  Plan: {self.stripe_id} / {self.object_kind} / {self.materiality}"
            f"  ({self.context_type} context)",
            f"  {sum(1 for d in self.dimensions if d.covered)} of "
            f"{len(self.dimensions)} planned dimension(s) have at least one candidate test",
            "",
            f"  {'Dimension':<32}{'Req':<6}{'Tests':<7}Candidates",
        ]
        for dimension in self.dimensions:
            marker = "!" if dimension.is_gap else " "
            names = ", ".join(t.split(".", 1)[-1] for t in dimension.test_ids[:3])
            if len(dimension.test_ids) > 3:
                names += f", +{len(dimension.test_ids) - 3}"
            lines.append(
                f"  {marker}{dimension.dimension_id:<31}"
                f"{'yes' if dimension.required else 'no':<6}"
                f"{len(dimension.test_ids):<7}{names or '—'}"
            )
        if self.gaps:
            lines.append("")
            lines.append(
                f"  ! {len(self.gaps)} REQUIRED dimension(s) have no candidate test: "
                + ", ".join(d.dimension_id for d in self.gaps)
            )
            lines.append(
                "    These are owed by the plan and cannot be evidenced by anything currently registered."
            )
        lines.append("")
        lines.append("  Coverage means a test COULD supply evidence, not that the dimension is discharged.")
        return lines


def tests_for_plan(plan: Any, context_type: str | None = None) -> dict[str, list[str]]:
    """``dimension_id -> [test_id, ...]`` for one plan.

    ``context_type`` filters to tests that can run against the cohort in hand. Passing
    ``None`` returns every candidate regardless of context, which is the right default
    for planning — a market test is still a legitimate answer to a dimension even when
    today's review is tabular.
    """
    return coverage_for_plan(plan, context_type=context_type).mapping()


tests_for_plan.__test__ = False  # type: ignore[attr-defined]


def coverage_for_plan(plan: Any, context_type: str | None = None) -> PlanCoverage:
    """Full coverage analysis, including the gaps.

    Matching is on the ``risk_dimensions`` a test declares in its ``TestSpec``. It is
    deliberately not inferred from the test id or family: a declared mapping is a
    statement the test's author made and can be held to, whereas an inferred one is a
    guess that would quietly drift as names changed.
    """
    from start.registry import list_tests

    context_type = normalise_context_type(context_type)
    specs = list_tests()
    stripe_id = getattr(plan, "stripe_id", "")
    object_kind = getattr(plan, "object_kind", "")

    dimensions: list[DimensionCoverage] = []
    for planned in getattr(plan, "planned", ()):
        dimension_id = planned.dimension_id
        matching = [s for s in specs if dimension_id in getattr(s, "risk_dimensions", ())]

        # Prefer a test that also declares this stripe: a drift test written for the
        # market stripe and one written for the model stripe answer the same dimension
        # differently, and the stripe-specific one is the better candidate.
        stripe_specific = [s for s in matching if stripe_id in getattr(s, "risk_stripes", ())]
        preferred = stripe_specific or matching

        if context_type is None:
            available = preferred
            unavailable: list[Any] = []
        else:
            available = [s for s in preferred if getattr(s, "context_type", "tabular") == context_type]
            unavailable = [s for s in preferred if s not in available]

        dimensions.append(
            DimensionCoverage(
                dimension_id=dimension_id,
                label=planned.label,
                required=bool(planned.required),
                phase=int(planned.phase),
                test_ids=tuple(sorted(s.test_id for s in available)),
                unavailable_test_ids=tuple(sorted(s.test_id for s in unavailable)),
            )
        )

    return PlanCoverage(
        stripe_id=stripe_id,
        object_kind=object_kind,
        materiality=getattr(plan, "materiality", ""),
        context_type=context_type or "any",
        dimensions=tuple(dimensions),
    )
