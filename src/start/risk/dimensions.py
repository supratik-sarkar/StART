"""Review dimensions — the model-agnostic vocabulary of a review.

The single largest design mistake in validation tooling is to organise the work
around *techniques* (AUC, SHAP, PSI) rather than around *questions*. Technique-
organised tooling only works on the artefacts those techniques accept, which is
why most of it silently assumes a trained supervised model with labelled
outcomes — an assumption that fails for the majority of what actually sits on a
bank's model inventory: pricing formulas, rules engines, vendor black boxes,
end-user spreadsheets, expert-judgment overlays, and now LLM systems.

A dimension here is a *question a reviewer must be able to answer*, expressed
independently of how it gets answered:

    "Does the thing discriminate between outcomes better than chance?"

is a question. ``roc_auc`` is one way to answer it for one class of artefact.
Separating the two is what lets the same review machinery run over a logistic
scorecard, a Monte-Carlo VaR engine, an AML rules tree, and a GenAI summariser
without any of them being a special case.

Each dimension declares the *capabilities* an artefact must have for the
dimension to be answerable at all, plus what to do when it is not — see
:mod:`start.risk.objects` for capability profiles and
:func:`start.risk.objects.applicability` for the resolution.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "Applicability",
    "Dimension",
    "DIMENSIONS",
    "dimension",
    "dimension_ids",
]


class Applicability(StrEnum):
    """Whether a dimension can be examined for a given artefact.

    ``SUBSTITUTED`` is the load-bearing value. Classical tooling has two states
    — it ran, or it did not — which pushes reviewers into recording "N/A" and
    moving on. "N/A" is where audit findings come from. A substituted dimension
    is still *owed an answer*; it is simply owed a different kind of evidence,
    and the substitution is recorded rather than assumed.
    """

    APPLICABLE = "applicable"
    SUBSTITUTED = "substituted"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class Dimension:
    """One question a review must be able to answer."""

    id: str
    label: str
    question: str
    #: Capability flags (see objects.CapabilityProfile) that must all be true.
    requires: tuple[str, ...] = field(default=())
    #: Dimensions that carry the burden when this one cannot be examined.
    substitutes: tuple[str, ...] = field(default=())
    #: Evidence classes that satisfy this dimension. These are contract names,
    #: not test IDs — a deterministic engine, an attestation, or a documented
    #: human judgement may all satisfy the same class.
    evidence_classes: tuple[str, ...] = field(default=())
    #: Ordering hint: lower runs earlier. Conceptual work precedes measurement;
    #: measurement precedes challenge; challenge precedes sign-off.
    phase: int = 5

    def __str__(self) -> str:  # pragma: no cover - display only
        return f"{self.id}: {self.label}"


# --------------------------------------------------------------------------- #
# The catalogue
# --------------------------------------------------------------------------- #
_DIMENSIONS: tuple[Dimension, ...] = (
    # ---- phase 1: is this the right thing at all -------------------------- #
    Dimension(
        id="conceptual_soundness",
        label="Conceptual soundness",
        question="Is the chosen approach defensible for the stated purpose, and is the "
        "theory linking inputs to outputs sound?",
        evidence_classes=("design_rationale", "literature_or_precedent", "assumption_register"),
        phase=1,
    ),
    Dimension(
        id="assumption_validity",
        label="Assumption validity",
        question="Are the stated assumptions true in the current environment, and what "
        "happens when each one breaks?",
        evidence_classes=("assumption_register", "assumption_breach_analysis"),
        phase=1,
    ),
    Dimension(
        id="use_boundary",
        label="Approved use and boundary conditions",
        question="Is the artefact being used only where it was approved to be used, and "
        "are the boundaries enforced rather than documented?",
        evidence_classes=("approved_use_statement", "usage_reconciliation", "boundary_controls"),
        phase=1,
    ),
    # ---- phase 2: does it rest on sound ground ---------------------------- #
    Dimension(
        id="data_quality_lineage",
        label="Data quality and lineage",
        question="Where did the inputs come from, what happened to them on the way, and "
        "are they fit for the purpose?",
        requires=("consumes_data",),
        substitutes=("third_party_diligence",),
        evidence_classes=("lineage_map", "quality_screens", "reconciliation"),
        phase=2,
    ),
    Dimension(
        id="implementation_verification",
        label="Implementation verification",
        question="Does the running implementation do what the specification says it does?",
        evidence_classes=("spec_to_code_trace", "recomputation_check", "unit_and_boundary_tests"),
        phase=2,
    ),
    Dimension(
        id="reproducibility",
        label="Reproducibility",
        question="Can an independent party re-run this and obtain the same numbers?",
        evidence_classes=("environment_manifest", "seed_and_determinism", "replay_attestation"),
        phase=2,
    ),
    # ---- phase 3: does it work -------------------------------------------- #
    Dimension(
        id="discriminatory_power",
        label="Discriminatory power",
        question="Does it separate outcomes better than the alternatives available?",
        requires=("produces_scores", "has_outcome_labels"),
        substitutes=("benchmarking", "implementation_verification"),
        evidence_classes=("rank_statistics", "cohort_comparison", "lift_analysis"),
        phase=3,
    ),
    Dimension(
        id="accuracy_calibration",
        label="Accuracy and calibration",
        question="Are the magnitudes right, not merely the ordering?",
        requires=("has_outcome_labels",),
        substitutes=("benchmarking", "outcomes_analysis"),
        evidence_classes=("calibration_curve", "error_decomposition", "bias_of_estimate"),
        phase=3,
    ),
    Dimension(
        id="stability",
        label="Stability over time and population",
        question="Does behaviour hold up as inputs, populations and regimes move?",
        evidence_classes=("population_shift", "temporal_performance", "segment_stability"),
        phase=3,
    ),
    Dimension(
        id="sensitivity",
        label="Input sensitivity",
        question="How much does the output move when the inputs move, and is that responsiveness intended?",
        evidence_classes=("shock_response", "elasticity_profile", "dominant_driver_ranking"),
        phase=3,
    ),
    Dimension(
        id="stress_scenario",
        label="Stress and scenario behaviour",
        question="What does it do under severe but plausible conditions, including "
        "conditions absent from its development sample?",
        evidence_classes=("scenario_grid", "reverse_stress", "tail_behaviour"),
        phase=3,
    ),
    Dimension(
        id="overfitting_generalisation",
        label="Overfitting and generalisation",
        question="Has it learned the sample rather than the phenomenon?",
        requires=("is_fitted",),
        substitutes=("benchmarking", "stress_scenario"),
        evidence_classes=("holdout_gap", "resampling_variance", "complexity_penalty"),
        phase=3,
    ),
    # ---- phase 4: is it challengeable ------------------------------------- #
    Dimension(
        id="benchmarking",
        label="Benchmarking and challenger comparison",
        question="Against what alternative was it judged, and would a simpler alternative have done as well?",
        evidence_classes=("challenger_result", "baseline_comparison", "champion_challenger_gap"),
        phase=4,
    ),
    Dimension(
        id="outcomes_analysis",
        label="Outcomes analysis and backtesting",
        question="What actually happened after the artefact spoke, and did reality agree?",
        requires=("has_realised_outcomes",),
        substitutes=("benchmarking", "monitoring"),
        evidence_classes=("backtest_series", "exception_analysis", "traffic_light_assessment"),
        phase=4,
    ),
    Dimension(
        id="explainability",
        label="Explainability and attribution",
        question="Can a reviewer say why a specific output was produced, at the level of "
        "detail the decision requires?",
        requires=("is_inspectable",),
        substitutes=("third_party_diligence", "sensitivity"),
        evidence_classes=("global_attribution", "local_attribution", "reason_code_mapping"),
        phase=4,
    ),
    Dimension(
        id="bias_fairness",
        label="Bias and fairness",
        question="Does it treat comparable subjects comparably, and is any disparity "
        "explainable and permissible?",
        requires=("affects_individuals",),
        evidence_classes=("disparity_measures", "proxy_analysis", "least_discriminatory_search"),
        phase=4,
    ),
    Dimension(
        id="robustness_adversarial",
        label="Robustness and adversarial resistance",
        question="What happens under malformed, gamed, or hostile input?",
        evidence_classes=("perturbation_suite", "gaming_analysis", "red_team_findings"),
        phase=4,
    ),
    # ---- phase 5: is it controlled ---------------------------------------- #
    Dimension(
        id="third_party_diligence",
        label="Third-party and dependency diligence",
        question="For components you did not build, what independent assurance exists?",
        requires=("externally_sourced",),
        evidence_classes=("vendor_documentation", "independent_testing", "exit_and_substitution"),
        phase=5,
    ),
    Dimension(
        id="output_consumption",
        label="Downstream consumption and overlays",
        question="What is done to the output before it reaches a decision, and is that "
        "adjustment itself governed?",
        evidence_classes=("downstream_map", "overlay_register", "adjustment_rationale"),
        phase=5,
    ),
    Dimension(
        id="monitoring",
        label="Ongoing monitoring and triggers",
        question="What would tell you it has stopped working, who is watching, and what "
        "happens when a threshold trips?",
        evidence_classes=("monitoring_plan", "trigger_thresholds", "escalation_path"),
        phase=5,
    ),
    Dimension(
        id="change_control",
        label="Change control and versioning",
        question="Can every version in production be tied to an approval, and can prior "
        "behaviour be reconstructed?",
        evidence_classes=("version_register", "approval_trail", "rollback_capability"),
        phase=5,
    ),
    Dimension(
        id="documentation_completeness",
        label="Documentation completeness",
        question="Could a competent successor take this over from the documentation alone?",
        evidence_classes=("document_inventory", "gap_assessment", "successor_test"),
        phase=5,
    ),
)

DIMENSIONS: dict[str, Dimension] = {d.id: d for d in _DIMENSIONS}


def dimension(dimension_id: str) -> Dimension:
    """Look up a dimension, with a helpful error listing valid ids."""
    try:
        return DIMENSIONS[dimension_id]
    except KeyError:
        raise KeyError(
            f"Unknown review dimension {dimension_id!r}. Known dimensions: {', '.join(sorted(DIMENSIONS))}"
        ) from None


def dimension_ids() -> tuple[str, ...]:
    """All dimension ids in deterministic phase-then-name order."""
    return tuple(d.id for d in sorted(_DIMENSIONS, key=lambda d: (d.phase, d.id)))
