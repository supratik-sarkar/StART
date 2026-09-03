"""F1 — risk metadata for the 19 pre-v4.2.0 registered tests.

Why the backfill exists
-----------------------

A1 added four defaulted fields to ``TestSpec`` — ``context_type``, ``risk_stripes``,
``risk_dimensions``, ``object_kinds`` — and deliberately left the 19 existing
registrations untouched so that no legacy behaviour changed. That was correct at the
time. The consequence surfaced at A5: ``coverage_for_plan`` reported
``accuracy_calibration``, ``discriminatory_power``, ``overfitting_generalisation``,
``outcomes_analysis`` and ``monitoring`` as **required gaps on a credit plan**, even
though ``supervised.calibration`` and ``supervised.discrimination`` plainly answer the
first two.

Those were *false* gaps: an artefact of missing metadata, not of missing capability. A
reviewer reading them would either distrust the tool or waste effort building evidence
that already exists.

The rule that governed every mapping below
------------------------------------------

    Map a dimension only where the engine's **actual emitted evidence** discharges the
    obligation the taxonomy describes.

A false claim of coverage is worse than an explicit gap. An explicit gap tells a
reviewer to go and find evidence; a false claim tells them not to bother. So each
mapping below was made after reading the implementation and asking what it actually
produces, and several obvious-sounding mappings were **declined** — those are recorded
in :data:`DECLINED` with the reason, because a decision not to map is as much a part of
the record as a decision to map.

Why this is applied rather than edited in place
-----------------------------------------------

The mappings are declared here as data and applied to the already-registered
``TestSpec`` objects at import. Two reasons:

* the legacy modules are not edited at all, so there is no possibility of a stray change
  to mathematics, parameters, thresholds or status logic;
* the justification for every mapping lives beside the mapping, which is where a
  reviewer will look for it — not scattered across four family modules.

``TestSpec`` is a frozen dataclass, so application uses ``dataclasses.replace`` and
rebinds the registry entry. Nothing else about the spec is touched.

**Analytical behaviour is unchanged.** ``TestSpec`` metadata is consumed only by
planning and coverage. No test function, parameter, threshold, metric, interpretation or
artifact is affected, and there is a regression test proving identical ``TestResult``
output before and after.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

__all__ = [
    "LEGACY_METADATA",
    "DECLINED",
    "apply_legacy_metadata",
    "LEGACY_TEST_IDS",
]

#: The 19 registrations that predate v4.2.0.
LEGACY_TEST_IDS: tuple[str, ...] = (
    "genai.citation_coverage",
    "preprocessing.constant_features",
    "preprocessing.duplicates",
    "preprocessing.feature_drift",
    "preprocessing.feature_ranges",
    "preprocessing.high_cardinality",
    "preprocessing.missingness",
    "preprocessing.outliers",
    "preprocessing.split_diagnostics",
    "preprocessing.target_leakage",
    "supervised.calibration",
    "supervised.classification_metrics",
    "supervised.cohort_metrics_comparison",
    "supervised.discrimination",
    "supervised.top_decile_lift",
    "xai.feature_sensitivity",
    "xai.global_importance",
    "xai.importance_stability",
    "xai.integrated_gradients",
)

_MODEL = ("model",)
_MODEL_CREDIT = ("model", "credit")
_ML_OBJECTS = ("ml_model", "statistical_model", "scorecard", "deep_learning_model")
_DATA_OBJECTS = ("ml_model", "statistical_model", "scorecard", "data_pipeline")

#: ``test_id -> (context_type, stripes, dimensions, object_kinds, justification)``.
#:
#: Every justification states what the engine *emits*, because that is what determines
#: whether it can discharge an obligation.
LEGACY_METADATA: dict[str, tuple[str, tuple[str, ...], tuple[str, ...], tuple[str, ...], str]] = {
    # ---------------------------------------------------------------- supervised --
    "supervised.calibration": (
        "tabular", _MODEL_CREDIT, ("accuracy_calibration",), _ML_OBJECTS,
        "Emits expected calibration error and Brier score over binned predicted "
        "probabilities. That is precisely the evidence accuracy_calibration asks for. "
        "Not mapped to discriminatory_power: a well-calibrated model can rank badly, "
        "and ECE says nothing about ordering.",
    ),
    "supervised.discrimination": (
        "tabular", _MODEL_CREDIT, ("discriminatory_power",), _ML_OBJECTS,
        "Emits holdout ROC-AUC, Gini and KS — rank-ordering power, which is the "
        "discriminatory_power obligation. Not mapped to accuracy_calibration: AUC is "
        "invariant to any monotone rescaling of scores, so it cannot evidence whether "
        "probabilities mean what they say.",
    ),
    "supervised.classification_metrics": (
        "tabular", _MODEL_CREDIT, ("discriminatory_power",), _ML_OBJECTS,
        "Emits accuracy, balanced accuracy, precision, recall and F1 at a single "
        "decision threshold — separation evidence at one operating point. "
        "outcomes_analysis is DECLINED: these are in-sample-style classification "
        "statistics against the modelling target, not realised outcomes observed after "
        "deployment.",
    ),
    "supervised.top_decile_lift": (
        "tabular", _MODEL_CREDIT, ("discriminatory_power",), _ML_OBJECTS,
        "Emits top-decile lift per cohort — event rate in the top decile relative to "
        "the base rate, which is rank-ordering power at a business-relevant cut. "
        "outcomes_analysis is DECLINED for the same reason as "
        "classification_metrics: lift on a scored cohort is not post-deployment "
        "outcome evidence.",
    ),
    "supervised.cohort_metrics_comparison": (
        "tabular", _MODEL_CREDIT,
        ("overfitting_generalisation", "discriminatory_power"), _ML_OBJECTS,
        "Emits per-cohort metrics across train/test/OOS AND an explicit "
        "auc_gap_train_test with warn/fail thresholds. The gap is the "
        "overfitting_generalisation obligation directly. discriminatory_power is also "
        "mapped because per-cohort AUC is emitted. monitoring is DECLINED: comparing "
        "fixed cohorts inside one review is not ongoing performance monitoring, which "
        "requires observation over time after deployment.",
    ),
    # ---------------------------------------------------------------------- xai --
    "xai.global_importance": (
        "tabular", _MODEL_CREDIT, ("explainability",), _ML_OBJECTS,
        "Emits global feature attribution with the method used recorded explicitly and "
        "no silent fallback. Directly the explainability obligation.",
    ),
    "xai.integrated_gradients": (
        "tabular", _MODEL, ("explainability",), ("deep_learning_model", "ml_model"),
        "Emits Integrated Gradients attributions for torch models. Object kinds are "
        "narrowed to deep-learning models because the method requires a differentiable "
        "model — claiming it for a rules engine or scorecard would be a false "
        "capability claim.",
    ),
    "xai.feature_sensitivity": (
        "tabular", _MODEL_CREDIT, ("sensitivity", "explainability"), _ML_OBJECTS,
        "Emits metric response under parallel multiplicative shocks (-30%..+30%) to "
        "top-k features. Shock response is the sensitivity obligation; it also "
        "evidences explainability because it shows which inputs move the output. "
        "stress_scenario is DECLINED: uniform parametric shocks are not the coherent "
        "adverse scenarios that obligation expects.",
    ),
    "xai.importance_stability": (
        "tabular", _MODEL_CREDIT, ("explainability", "reproducibility"), _ML_OBJECTS,
        "Emits overlap of top-k permutation-importance features across two seeds. That "
        "is evidence the explanation is reproducible, so both explainability and "
        "reproducibility apply. monitoring and stability are DECLINED: this measures "
        "seed-to-seed stability of an explanation within one fit, not population "
        "stability over time.",
    ),
    # ------------------------------------------------------------ preprocessing --
    "preprocessing.missingness": (
        "tabular", _MODEL_CREDIT, ("data_quality_lineage",), _DATA_OBJECTS,
        "Emits per-column missingness percentages with thresholds — completeness "
        "evidence, which is the data-quality obligation.",
    ),
    "preprocessing.duplicates": (
        "tabular", _MODEL_CREDIT, ("data_quality_lineage",), _DATA_OBJECTS,
        "Emits duplicate-row counts and rates over the training frame — record "
        "integrity, which is data-quality evidence. use_boundary is DECLINED: "
        "duplication WITHIN one cohort is a data defect, whereas duplication ACROSS "
        "the train/test boundary is an independence violation and is covered by "
        "preprocessing.split_diagnostics.",
    ),
    "preprocessing.constant_features": (
        "tabular", _MODEL_CREDIT, ("data_quality_lineage",), _DATA_OBJECTS,
        "Emits counts of constant and near-constant columns — degenerate features "
        "carrying no information, a data-quality finding.",
    ),
    "preprocessing.high_cardinality": (
        "tabular", _MODEL_CREDIT, ("data_quality_lineage",), _DATA_OBJECTS,
        "Emits per-column cardinality and flags high-cardinality categoricals. "
        "conceptual_soundness is DECLINED: cardinality is a property of the data, not "
        "an argument about whether the modelling approach suits the problem.",
    ),
    "preprocessing.feature_ranges": (
        "tabular", _MODEL_CREDIT, ("data_quality_lineage",), _DATA_OBJECTS,
        "Emits an informational summary of numeric ranges with no thresholds. Recorded "
        "as data-quality evidence; it asserts nothing, which is why it carries no other "
        "dimension.",
    ),
    "preprocessing.outliers": (
        "tabular", _MODEL_CREDIT, ("data_quality_lineage",), _DATA_OBJECTS,
        "Emits Tukey-fence outlier rates per column with thresholds and, when enabled, "
        "a boxplot artifact.",
    ),
    "preprocessing.split_diagnostics": (
        "tabular", _MODEL_CREDIT, ("use_boundary", "data_quality_lineage"), _DATA_OBJECTS,
        "Emits train/test row-overlap counts plus cohort size and class-balance checks. "
        "Overlap between cohorts is a use_boundary violation — the evaluation set is "
        "not independent — and the size/balance checks are data quality.",
    ),
    "preprocessing.target_leakage": (
        "tabular", _MODEL_CREDIT, ("use_boundary",), _DATA_OBJECTS,
        "Emits features with near-perfect absolute correlation to the target. The "
        "umbrella screen that predates the A3 specific detectors; retained alongside "
        "them.",
    ),
    "preprocessing.feature_drift": (
        "tabular", _MODEL_CREDIT, ("stability", "monitoring"), _DATA_OBJECTS,
        "Emits per-feature PSI and KS statistics between cohorts with thresholds. PSI "
        "is the standard population-stability measure, so stability applies directly. "
        "monitoring is mapped because this is the engine that would be re-run on "
        "production data to detect distribution shift — it produces exactly the "
        "quantity an ongoing monitoring programme tracks. This is the one legacy "
        "engine where monitoring is genuinely earned rather than adjacent.",
    ),
    # ---------------------------------------------------------------------- genai --
    "genai.citation_coverage": (
        "tabular", ("ai_genai",), ("documentation_completeness", "output_consumption"),
        ("llm_system", "agentic_system"),
        "Emits the fraction of numeric sentences in generated text lacking a citation "
        "tag. That is evidence about whether generated output is traceable to sources, "
        "so documentation_completeness and output_consumption apply. Deliberately NOT "
        "mapped to explainability, accuracy_calibration or conceptual_soundness: "
        "citation coverage says nothing about whether the underlying claims are "
        "correct, only whether they are attributed. Stripe is ai_genai, not model — "
        "forcing it into model-risk dimensions merely because every test should have "
        "metadata is exactly the false-coverage failure this backfill exists to avoid.",
    ),
}

#: Mappings that were considered and **declined**, with the reason. A decision not to
#: map is part of the record: without it, a later reader cannot tell whether a gap was
#: examined and rejected or simply overlooked.
DECLINED: tuple[tuple[str, str, str], ...] = (
    ("supervised.classification_metrics", "outcomes_analysis",
     "Classification statistics against the modelling target are not realised "
     "post-deployment outcomes."),
    ("supervised.top_decile_lift", "outcomes_analysis",
     "Lift on a scored evaluation cohort is not post-deployment outcome evidence."),
    ("supervised.cohort_metrics_comparison", "monitoring",
     "Comparing fixed cohorts inside one review is not ongoing monitoring over time."),
    ("supervised.calibration", "discriminatory_power",
     "A well-calibrated model can rank badly; ECE says nothing about ordering."),
    ("supervised.discrimination", "accuracy_calibration",
     "AUC is invariant to monotone rescaling, so it cannot evidence probability "
     "quality."),
    ("xai.feature_sensitivity", "stress_scenario",
     "Uniform parametric shocks are not coherent adverse scenarios."),
    ("xai.importance_stability", "monitoring",
     "Seed-to-seed stability of an explanation is not population stability over time."),
    ("preprocessing.high_cardinality", "conceptual_soundness",
     "Cardinality is a property of the data, not an argument about modelling approach."),
    ("genai.citation_coverage", "explainability",
     "Citation coverage establishes attribution, not whether claims are correct."),
    ("*", "outcomes_analysis",
     "NO legacy engine produces post-deployment outcome evidence. outcomes_analysis "
     "remains a GENUINE uncovered obligation on any plan that requires it, and that is "
     "the correct result."),
    ("*", "bias_fairness",
     "No legacy engine computes disparity across protected groups. A genuine gap."),
    ("*", "benchmarking",
     "No legacy engine compares against a challenger or trivial baseline. A genuine "
     "gap; the v4.0.2 stump benchmark lives outside the registry."),
)


def apply_legacy_metadata(registry: dict[str, Any] | None = None) -> dict[str, int]:
    """Attach the declared metadata to already-registered legacy specs.

    Idempotent, and never overwrites metadata a test declared for itself: a spec that
    already carries ``risk_dimensions`` is left exactly as it is, so this can never
    silently contradict an author's own declaration.

    Returns a small summary so a caller can assert what happened.
    """
    from start.registry import _REGISTRY

    target = _REGISTRY if registry is None else registry
    applied = 0
    skipped_present = 0
    absent = 0

    for test_id, (context, stripes, dimensions, objects, _why) in LEGACY_METADATA.items():
        spec = target.get(test_id)
        if spec is None:
            absent += 1
            continue
        if getattr(spec, "risk_dimensions", ()):
            skipped_present += 1
            continue
        target[test_id] = replace(
            spec,
            context_type=context,
            risk_stripes=stripes,
            risk_dimensions=dimensions,
            object_kinds=objects,
        )
        applied += 1

    return {"applied": applied, "already_present": skipped_present, "not_registered": absent}
