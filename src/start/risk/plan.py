"""Deterministic review-plan synthesis.

Given a risk stripe, a risk object and a materiality, produce the plan: which
dimensions must be examined, in what order, why each is in scope, and what to
do about the ones that cannot be examined at all.

Two properties matter more than the content of any individual plan.

**Determinism.** The same inputs always produce the same plan, and the plan
carries a hash of its own content. That converts "what did you decide to look
at, and did you change your mind afterwards?" from a matter of recollection
into a matter of comparison. A plan hash computed before execution and again at
sign-off must match, or the scope moved.

**Burden conservation.** When a dimension is inapplicable, its obligation does
not evaporate. If the object cannot support ``overfitting_generalisation``, the
dimensions that dimension nominated as substitutes are promoted to mandatory
and marked with the burden they inherited. The plan therefore has no silent
holes — every dimension in the catalogue is either planned, substituted (with
the substitution named), or excluded with a stated reason.

Standard library only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from start.risk.controls import MAPPING_VERSION, coverage_report
from start.risk.dimensions import DIMENSIONS, Applicability, dimension_ids
from start.risk.objects import RiskObject, applicability
from start.risk.stripes import stripe as get_stripe

__all__ = [
    "PlannedDimension",
    "ReviewPlan",
    "synthesise_plan",
    "MATERIALITY_LEVELS",
]

MATERIALITY_LEVELS: tuple[str, ...] = ("low", "medium", "high")

#: Materiality does not change *what is true*, only *how much work is owed*.
#: Higher materiality promotes heightened dimensions to mandatory and raises the
#: minimum depth expected of every planned dimension.
_DEPTH_BY_MATERIALITY = {
    "low": "targeted",
    "medium": "standard",
    "high": "comprehensive",
}


@dataclass(frozen=True)
class PlannedDimension:
    """One dimension in the plan, with its provenance."""

    dimension_id: str
    label: str
    question: str
    phase: int
    required: bool
    depth: str
    #: Why this dimension is in the plan, in order of strength.
    rationale: tuple[str, ...]
    applicability: str
    evidence_classes: tuple[str, ...]
    inherited_burden_from: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension_id,
            "label": self.label,
            "question": self.question,
            "phase": self.phase,
            "required": self.required,
            "depth": self.depth,
            "rationale": list(self.rationale),
            "applicability": self.applicability,
            "evidence_classes": list(self.evidence_classes),
            "inherited_burden_from": list(self.inherited_burden_from),
        }


@dataclass(frozen=True)
class ReviewPlan:
    """A synthesised, hash-stable review plan."""

    stripe_id: str
    object_id: str
    object_kind: str
    materiality: str
    planned: tuple[PlannedDimension, ...]
    excluded: tuple[dict[str, Any], ...]
    substitutions: tuple[dict[str, Any], ...]
    control_frameworks: tuple[str, ...]
    mapping_version: str = MAPPING_VERSION

    # -- serialisation ------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "stripe": self.stripe_id,
            "object_id": self.object_id,
            "object_kind": self.object_kind,
            "materiality": self.materiality,
            "mapping_version": self.mapping_version,
            "control_frameworks": list(self.control_frameworks),
            "planned_dimensions": [p.as_dict() for p in self.planned],
            "excluded_dimensions": [dict(e) for e in self.excluded],
            "substitutions": [dict(s) for s in self.substitutions],
        }

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    def plan_hash(self) -> str:
        """Content hash of the plan. Compare before execution and at sign-off."""
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    # -- convenience --------------------------------------------------------
    def required_dimension_ids(self) -> tuple[str, ...]:
        return tuple(p.dimension_id for p in self.planned if p.required)

    def all_dimension_ids(self) -> tuple[str, ...]:
        return tuple(p.dimension_id for p in self.planned)

    def coverage(self, examined: set[str]) -> dict[str, Any]:
        """Control coverage given the dimensions that actually produced evidence."""
        return coverage_report(self.control_frameworks, examined)

    def summary_lines(self) -> list[str]:
        """Compact human rendering, used by the CLI and the demo script."""
        lines = [
            f"Review plan  {self.stripe_id} / {self.object_kind} / materiality={self.materiality}",
            f"Plan hash    {self.plan_hash()[:16]}…",
            f"Dimensions   {len(self.planned)} planned "
            f"({len(self.required_dimension_ids())} mandatory), "
            f"{len(self.excluded)} excluded, {len(self.substitutions)} substituted",
            "",
        ]
        current_phase = None
        for p in self.planned:
            if p.phase != current_phase:
                current_phase = p.phase
                lines.append(f"  -- phase {p.phase} --")
            if p.applicability == "substituted":
                flag = "sub"
            elif p.required:
                flag = "REQ"
            else:
                flag = "opt"
            inherited = (
                f"  <- burden from {', '.join(p.inherited_burden_from)}"
                if p.inherited_burden_from
                else ""
            )
            lines.append(f"  [{flag}] {p.dimension_id:<28} {p.depth:<14}{inherited}")
        return lines


def synthesise_plan(
    *,
    stripe_id: str,
    obj: RiskObject,
    materiality: str | None = None,
) -> ReviewPlan:
    """Build the plan for one (stripe, object, materiality) triple.

    The resolution order is fixed so the result is reproducible:

    1. Resolve applicability of every dimension against the object's
       capabilities.
    2. Collect burden transferred from substituted dimensions.
    3. Union the stripe's mandatory set, the object kind's always-required set,
       and the inherited burden. Anything in that union is mandatory.
    4. Promote the stripe's heightened dimensions to mandatory when materiality
       is high.
    5. Order by phase, then by dimension id.
    """
    strp = get_stripe(stripe_id)
    materiality = (materiality or obj.materiality or "medium").lower()
    if materiality not in MATERIALITY_LEVELS:
        raise ValueError(
            f"Unknown materiality {materiality!r}. Valid: {', '.join(MATERIALITY_LEVELS)}"
        )

    depth = _DEPTH_BY_MATERIALITY[materiality]

    # 1. applicability -------------------------------------------------------
    verdicts = {d: applicability(obj, d) for d in dimension_ids()}

    # 2. burden transfer -----------------------------------------------------
    inherited: dict[str, list[str]] = {}
    substitutions: list[dict[str, Any]] = []
    for did, verdict in verdicts.items():
        if verdict.applicability is Applicability.SUBSTITUTED:
            substitutions.append(verdict.as_dict())
            for target in verdict.burden_transferred_to:
                inherited.setdefault(target, []).append(did)

    # 3-4. mandatory set -----------------------------------------------------
    mandatory: set[str] = set(strp.mandatory_dimensions)
    mandatory |= set(obj.kind_spec().always_required)
    mandatory |= set(inherited)
    if materiality == "high":
        mandatory |= set(strp.heightened_dimensions)

    planned: list[PlannedDimension] = []
    excluded: list[dict[str, Any]] = []

    for did in dimension_ids():
        verdict = verdicts[did]
        dim = DIMENSIONS[did]

        if verdict.applicability is Applicability.NOT_APPLICABLE:
            excluded.append(verdict.as_dict())
            continue

        rationale: list[str] = []
        if did in strp.mandatory_dimensions:
            rationale.append(f"mandatory for the {strp.label.lower()} stripe")
        if did in obj.kind_spec().always_required:
            rationale.append(f"always required for a {obj.kind_spec().label.lower()}")
        if did in inherited:
            rationale.append(
                "carries burden transferred from "
                f"{', '.join(sorted(inherited[did]))}, which this object cannot support"
            )
        if materiality == "high" and did in strp.heightened_dimensions:
            rationale.append("elevated to mandatory by high materiality")
        if not rationale:
            rationale.append("in the standard catalogue; examined at proportionate depth")

        dim_depth = depth
        if did in strp.heightened_dimensions:
            dim_depth = "comprehensive" if materiality != "low" else "standard"

        planned.append(
            PlannedDimension(
                dimension_id=did,
                label=dim.label,
                question=dim.question,
                phase=dim.phase,
                required=did in mandatory,
                depth=dim_depth,
                rationale=tuple(rationale),
                applicability=verdict.applicability.value,
                evidence_classes=dim.evidence_classes,
                inherited_burden_from=tuple(sorted(inherited.get(did, ()))),
            )
        )

    planned.sort(key=lambda p: (p.phase, p.dimension_id))

    return ReviewPlan(
        stripe_id=strp.id,
        object_id=obj.object_id,
        object_kind=obj.kind,
        materiality=materiality,
        planned=tuple(planned),
        excluded=tuple(excluded),
        substitutions=tuple(substitutions),
        control_frameworks=strp.control_frameworks,
    )
