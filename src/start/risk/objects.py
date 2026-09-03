"""Risk objects — what is actually under review.

A model inventory is not a folder of scikit-learn pickles. Typically it holds,
in rough order of count: deterministic calculators, end-user spreadsheets,
rules engines and scorecards, vendor black boxes, expert-judgment overlays,
statistical models, and — a minority of entries — machine-learning models. More
recently it holds LLM and agentic systems, which break the assumptions of every
tool built for the previous six categories.

StART therefore reviews a *risk object*, characterised by what it can and
cannot support, rather than a "model" assumed to be a fitted estimator with a
``predict`` method.

The capability profile is the hinge. Given a profile, the applicable review
dimensions fall out deterministically:

    a deterministic pricing calculator has no training data, so
    ``overfitting_generalisation`` is not merely skipped — it is formally
    ``NOT_APPLICABLE``, and the burden it would have carried is transferred to
    ``implementation_verification`` and ``benchmarking``, which are then
    *mandatory*.

That transfer is the point. A dimension is never dropped silently; it is either
answered, or its burden is explicitly reassigned to dimensions that are
answerable, and the reassignment appears in the review plan and in the evidence.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from start.risk.dimensions import DIMENSIONS, Applicability, Dimension

__all__ = [
    "CapabilityProfile",
    "RiskObjectKind",
    "RiskObject",
    "OBJECT_KINDS",
    "object_kind",
    "object_kind_ids",
    "DimensionVerdict",
    "applicability",
]


# --------------------------------------------------------------------------- #
# Capabilities
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CapabilityProfile:
    """What an artefact can support, expressed as reviewable capabilities.

    These are deliberately phrased as properties of the artefact rather than of
    the technique used to build it, so a hand-written formula and a gradient
    boosting machine are described in the same vocabulary.
    """

    #: Ingests data at run time (as opposed to being a closed-form constant).
    consumes_data: bool = True
    #: Was estimated/fitted from a development sample.
    is_fitted: bool = False
    #: Emits an ordering or a continuous score, not just a category or a document.
    produces_scores: bool = True
    #: Labelled outcomes exist to test against.
    has_outcome_labels: bool = False
    #: Realised outcomes accumulate over time, enabling backtesting.
    has_realised_outcomes: bool = False
    #: Internals can be examined (white or grey box).
    is_inspectable: bool = True
    #: Repeated runs on identical input may differ.
    is_stochastic: bool = False
    #: Built or maintained outside the organisation.
    externally_sourced: bool = False
    #: Outputs bear on identifiable individuals (drives fair-lending style review).
    affects_individuals: bool = False
    #: Produces free-form natural language.
    natural_language_output: bool = False
    #: Can plan and take actions, not only produce an output.
    takes_actions: bool = False

    def as_dict(self) -> dict[str, bool]:
        return {
            "consumes_data": self.consumes_data,
            "is_fitted": self.is_fitted,
            "produces_scores": self.produces_scores,
            "has_outcome_labels": self.has_outcome_labels,
            "has_realised_outcomes": self.has_realised_outcomes,
            "is_inspectable": self.is_inspectable,
            "is_stochastic": self.is_stochastic,
            "externally_sourced": self.externally_sourced,
            "affects_individuals": self.affects_individuals,
            "natural_language_output": self.natural_language_output,
            "takes_actions": self.takes_actions,
        }


@dataclass(frozen=True)
class RiskObjectKind:
    """A class of reviewable artefact and its default capability profile."""

    id: str
    label: str
    description: str
    capabilities: CapabilityProfile
    #: Dimensions that are mandatory for this kind irrespective of stripe,
    #: usually because the kind is structurally weak somewhere.
    always_required: tuple[str, ...] = field(default=())
    notes: str = ""


_KINDS: tuple[RiskObjectKind, ...] = (
    RiskObjectKind(
        id="deterministic_calculator",
        label="Deterministic calculator",
        description="Closed-form computation: pricing formula, capital calculation, "
        "accrual engine, regulatory ratio.",
        capabilities=CapabilityProfile(
            is_fitted=False,
            produces_scores=True,
            has_outcome_labels=False,
            has_realised_outcomes=False,
            is_inspectable=True,
        ),
        always_required=("implementation_verification", "benchmarking", "assumption_validity"),
        notes="No development sample means statistical validation is meaningless here; "
        "the review turns on specification fidelity and independent recomputation.",
    ),
    RiskObjectKind(
        id="statistical_model",
        label="Statistical model",
        description="Estimated parametric or semi-parametric model: regression, survival, "
        "time-series, factor model.",
        capabilities=CapabilityProfile(
            is_fitted=True,
            has_outcome_labels=True,
            has_realised_outcomes=True,
            is_inspectable=True,
        ),
    ),
    RiskObjectKind(
        id="ml_model",
        label="Machine-learning model",
        description="Fitted supervised learner: tree ensembles, boosting, kernel methods.",
        capabilities=CapabilityProfile(
            is_fitted=True,
            has_outcome_labels=True,
            has_realised_outcomes=True,
            is_inspectable=True,
        ),
        always_required=("explainability", "overfitting_generalisation"),
    ),
    RiskObjectKind(
        id="deep_learning_model",
        label="Deep-learning model",
        description="Neural architectures over tabular, sequence, vision or graph inputs.",
        capabilities=CapabilityProfile(
            is_fitted=True,
            has_outcome_labels=True,
            has_realised_outcomes=True,
            is_inspectable=True,
            is_stochastic=True,
        ),
        always_required=("explainability", "robustness_adversarial", "reproducibility"),
    ),
    RiskObjectKind(
        id="rules_engine",
        label="Rules engine",
        description="Deterministic decision logic: AML/sanctions typologies, fraud rules, "
        "eligibility and limit checks.",
        capabilities=CapabilityProfile(
            is_fitted=False,
            produces_scores=False,
            has_outcome_labels=True,
            has_realised_outcomes=True,
            is_inspectable=True,
            affects_individuals=True,
        ),
        always_required=("implementation_verification", "outcomes_analysis", "use_boundary"),
        notes="Rules produce alerts rather than scores; discrimination is judged through "
        "alert productivity and coverage rather than rank statistics.",
    ),
    RiskObjectKind(
        id="scorecard",
        label="Scorecard",
        description="Points-based or weight-of-evidence scoring, typically with published reason codes.",
        capabilities=CapabilityProfile(
            is_fitted=True,
            has_outcome_labels=True,
            has_realised_outcomes=True,
            is_inspectable=True,
            affects_individuals=True,
        ),
        always_required=("bias_fairness", "explainability"),
    ),
    RiskObjectKind(
        id="vendor_model",
        label="Vendor / third-party model",
        description="Externally built component with limited or no internal visibility.",
        capabilities=CapabilityProfile(
            is_fitted=True,
            has_outcome_labels=False,
            has_realised_outcomes=True,
            is_inspectable=False,
            externally_sourced=True,
        ),
        always_required=("third_party_diligence", "benchmarking", "outcomes_analysis"),
        notes="Opacity does not reduce the obligation; it relocates it to outcome "
        "evidence, independent benchmarking and contractual assurance.",
    ),
    RiskObjectKind(
        id="spreadsheet_euc",
        label="End-user computing artefact",
        description="Spreadsheet or desktop tool performing a risk-relevant calculation.",
        capabilities=CapabilityProfile(
            is_fitted=False,
            has_outcome_labels=False,
            has_realised_outcomes=False,
            is_inspectable=True,
        ),
        always_required=(
            "implementation_verification",
            "change_control",
            "documentation_completeness",
        ),
        notes="The dominant failure mode is uncontrolled change, not statistical error.",
    ),
    RiskObjectKind(
        id="expert_judgment_overlay",
        label="Expert-judgment overlay",
        description="Human adjustment applied on top of a quantitative output: management "
        "overlay, post-model adjustment, qualitative override.",
        capabilities=CapabilityProfile(
            consumes_data=False,
            is_fitted=False,
            produces_scores=True,
            has_realised_outcomes=True,
            is_inspectable=True,
        ),
        always_required=("output_consumption", "documentation_completeness", "outcomes_analysis"),
        notes="Overlays are frequently the largest single driver of a reported number and "
        "the least evidenced part of the chain.",
    ),
    RiskObjectKind(
        id="data_pipeline",
        label="Data pipeline / feature store",
        description="Transformation layer feeding one or more downstream risk objects.",
        capabilities=CapabilityProfile(
            is_fitted=False,
            produces_scores=False,
            has_outcome_labels=False,
            is_inspectable=True,
        ),
        always_required=("data_quality_lineage", "implementation_verification", "monitoring"),
    ),
    RiskObjectKind(
        id="llm_system",
        label="LLM system",
        description="Prompted or retrieval-augmented language model producing text used in "
        "a risk-relevant process.",
        capabilities=CapabilityProfile(
            is_fitted=True,
            produces_scores=False,
            has_outcome_labels=False,
            has_realised_outcomes=True,
            is_inspectable=False,
            is_stochastic=True,
            externally_sourced=True,
            natural_language_output=True,
        ),
        always_required=(
            "robustness_adversarial",
            "third_party_diligence",
            "output_consumption",
            "use_boundary",
        ),
        notes="Non-determinism and free-form output mean the reviewable unit is the "
        "surrounding control envelope, not the weights.",
    ),
    RiskObjectKind(
        id="agentic_system",
        label="Agentic system",
        description="LLM-driven system that plans and takes actions through tools.",
        capabilities=CapabilityProfile(
            is_fitted=True,
            produces_scores=False,
            has_outcome_labels=False,
            has_realised_outcomes=True,
            is_inspectable=False,
            is_stochastic=True,
            externally_sourced=True,
            natural_language_output=True,
            takes_actions=True,
        ),
        always_required=(
            "use_boundary",
            "robustness_adversarial",
            "change_control",
            "monitoring",
            "output_consumption",
        ),
        notes="Action-taking converts an output-quality problem into a control problem: "
        "what the system may do matters more than what it says.",
    ),
    RiskObjectKind(
        id="monitoring_process",
        label="Monitoring process",
        description="The surveillance layer over other risk objects, itself reviewable.",
        capabilities=CapabilityProfile(
            is_fitted=False,
            produces_scores=False,
            has_realised_outcomes=True,
            is_inspectable=True,
        ),
        always_required=("monitoring", "outcomes_analysis", "change_control"),
    ),
)

OBJECT_KINDS: dict[str, RiskObjectKind] = {k.id: k for k in _KINDS}


def object_kind(kind_id: str) -> RiskObjectKind:
    try:
        return OBJECT_KINDS[kind_id]
    except KeyError:
        raise KeyError(
            f"Unknown risk object kind {kind_id!r}. Known kinds: {', '.join(sorted(OBJECT_KINDS))}"
        ) from None


def object_kind_ids() -> tuple[str, ...]:
    return tuple(sorted(OBJECT_KINDS))


# --------------------------------------------------------------------------- #
# A concrete object under review
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RiskObject:
    """A specific artefact submitted for review."""

    object_id: str
    kind: str
    name: str = ""
    owner: str = ""
    materiality: str = "medium"  # low | medium | high
    #: Capability overrides. A vendor model that *did* ship training statistics,
    #: or an internal model with no realised outcomes yet, differs from its kind's
    #: default; say so here rather than mis-classifying the kind.
    capability_overrides: dict[str, bool] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)

    def kind_spec(self) -> RiskObjectKind:
        return object_kind(self.kind)

    def capabilities(self) -> CapabilityProfile:
        base = self.kind_spec().capabilities
        if not self.capability_overrides:
            return base
        valid = base.as_dict()
        unknown = set(self.capability_overrides) - set(valid)
        if unknown:
            raise KeyError(
                f"Unknown capability override(s): {', '.join(sorted(unknown))}. "
                f"Valid capabilities: {', '.join(sorted(valid))}"
            )
        return replace(base, **self.capability_overrides)  # type: ignore[arg-type]

    def as_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "kind": self.kind,
            "name": self.name,
            "owner": self.owner,
            "materiality": self.materiality,
            "capabilities": self.capabilities().as_dict(),
            "capability_overrides": dict(sorted(self.capability_overrides.items())),
            "attributes": dict(sorted(self.attributes.items())),
        }


# --------------------------------------------------------------------------- #
# Applicability resolution
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DimensionVerdict:
    """Whether a dimension applies to an object, and what carries its burden."""

    dimension_id: str
    applicability: Applicability
    reason: str
    #: When SUBSTITUTED, the dimensions that inherit the burden. These become
    #: mandatory in the synthesised plan.
    burden_transferred_to: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension_id,
            "applicability": self.applicability.value,
            "reason": self.reason,
            "burden_transferred_to": list(self.burden_transferred_to),
        }


def _missing_capabilities(dim: Dimension, caps: CapabilityProfile) -> list[str]:
    have = caps.as_dict()
    return [c for c in dim.requires if not have.get(c, False)]


def applicability(obj: RiskObject, dimension_id: str) -> DimensionVerdict:
    """Resolve one dimension against one object.

    Three outcomes:

    * **applicable** — every required capability is present.
    * **substituted** — a capability is missing but the dimension declares
      substitutes, so the burden moves to those dimensions and they become
      mandatory.
    * **not_applicable** — a capability is missing and nothing can carry the
      burden. This is the only case where a dimension may be dropped, and even
      then the reason is recorded.
    """
    dim = DIMENSIONS[dimension_id]
    caps = obj.capabilities()
    missing = _missing_capabilities(dim, caps)

    if not missing:
        return DimensionVerdict(
            dimension_id=dimension_id,
            applicability=Applicability.APPLICABLE,
            reason="all required capabilities present",
        )

    missing_text = ", ".join(missing)
    kind_label = obj.kind_spec().label.lower()
    article = "an" if kind_label[:1] in "aeiou" else "a"

    if dim.substitutes:
        return DimensionVerdict(
            dimension_id=dimension_id,
            applicability=Applicability.SUBSTITUTED,
            reason=(
                f"{article} {kind_label} does not support: {missing_text}. The obligation is not "
                f"waived — it transfers to {', '.join(dim.substitutes)}, which become "
                "mandatory for this review."
            ),
            burden_transferred_to=dim.substitutes,
        )

    return DimensionVerdict(
        dimension_id=dimension_id,
        applicability=Applicability.NOT_APPLICABLE,
        reason=(
            f"{article} {kind_label} does not support: {missing_text}, and this dimension declares "
            "no substitute. Recorded as inapplicable with reason rather than skipped."
        ),
    )
