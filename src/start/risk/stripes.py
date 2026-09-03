"""Risk stripes — the review context.

A stripe answers "what kind of risk is this artefact bearing on?", which is a
different question from "what kind of artefact is it?" (:mod:`start.risk.objects`)
and from "what must a reviewer establish?" (:mod:`start.risk.dimensions`).

The three axes are orthogonal on purpose. The same gradient-boosting model is a
different review depending on whether it sets a credit limit, prices an
illiquid position, or triages sanctions alerts — different mandatory
dimensions, different tolerance for opacity, different evidence that counts.
Encoding the stripe separately is what lets one platform cover a bank rather
than one desk.

Stripes are extensible: an organisation can register its own via the
``start.risk_stripes`` entry point without forking this file. That matters
because internal stripe taxonomies rarely match anyone else's, and a taxonomy
you have to fork is a taxonomy that drifts out of date.

Standard library only. Regulatory references are to published, public
supervisory material and standards, cited by name so a reviewer can trace an
obligation to its source.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "RiskStripe",
    "STRIPES",
    "stripe",
    "stripe_ids",
    "register_stripe",
    "load_entry_point_stripes",
]


@dataclass(frozen=True)
class RiskStripe:
    """A risk domain with its own review expectations."""

    id: str
    label: str
    description: str
    #: Dimensions that must be examined for any object in this stripe.
    mandatory_dimensions: tuple[str, ...] = field(default=())
    #: Dimensions examined with elevated depth — extra evidence, tighter
    #: thresholds, senior challenge.
    heightened_dimensions: tuple[str, ...] = field(default=())
    #: Public frameworks whose expectations bear on this stripe (see controls.py).
    control_frameworks: tuple[str, ...] = field(default=())
    #: Object kinds most commonly found in this stripe. Advisory, not a constraint.
    typical_objects: tuple[str, ...] = field(default=())
    #: The measurement vocabulary a reviewer expects to see. Used to prompt for
    #: missing evidence, never to compute anything.
    measurement_vocabulary: tuple[str, ...] = field(default=())
    source: str = "builtin"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "mandatory_dimensions": list(self.mandatory_dimensions),
            "heightened_dimensions": list(self.heightened_dimensions),
            "control_frameworks": list(self.control_frameworks),
            "typical_objects": list(self.typical_objects),
            "measurement_vocabulary": list(self.measurement_vocabulary),
            "source": self.source,
        }


_BUILTIN: tuple[RiskStripe, ...] = (
    RiskStripe(
        id="credit",
        label="Credit risk",
        description="Probability of default, loss given default, exposure at default, "
        "impairment, origination and limit setting.",
        mandatory_dimensions=(
            "conceptual_soundness",
            "data_quality_lineage",
            "discriminatory_power",
            "accuracy_calibration",
            "stability",
            "outcomes_analysis",
            "bias_fairness",
            "monitoring",
        ),
        heightened_dimensions=("accuracy_calibration", "outcomes_analysis", "bias_fairness"),
        control_frameworks=("sr_11_7", "basel_irb", "ifrs9_cecl", "ecoa_reg_b", "bcbs_239"),
        typical_objects=("scorecard", "statistical_model", "ml_model", "expert_judgment_overlay"),
        measurement_vocabulary=(
            "rank_ordering",
            "calibration_to_observed_default",
            "population_stability",
            "vintage_performance",
            "override_rate",
        ),
    ),
    RiskStripe(
        id="market",
        label="Market risk",
        description="Valuation and risk measurement of traded positions: VaR, expected "
        "shortfall, sensitivities, P&L attribution.",
        mandatory_dimensions=(
            "conceptual_soundness",
            "assumption_validity",
            "implementation_verification",
            "sensitivity",
            "stress_scenario",
            "outcomes_analysis",
            "benchmarking",
        ),
        heightened_dimensions=("outcomes_analysis", "stress_scenario", "assumption_validity"),
        control_frameworks=("sr_11_7", "basel_frtb", "bcbs_239"),
        typical_objects=("deterministic_calculator", "statistical_model", "vendor_model"),
        measurement_vocabulary=(
            "backtest_exceptions",
            "pnl_attribution_residual",
            "risk_factor_coverage",
            "scenario_severity",
        ),
    ),
    RiskStripe(
        id="liquidity",
        label="Liquidity and funding risk",
        description="Cash-flow projection, survival horizon, deposit behaviouralisation, "
        "coverage and funding ratios.",
        mandatory_dimensions=(
            "conceptual_soundness",
            "assumption_validity",
            "stress_scenario",
            "sensitivity",
            "implementation_verification",
            "monitoring",
        ),
        heightened_dimensions=("assumption_validity", "stress_scenario"),
        control_frameworks=("sr_11_7", "basel_lcr_nsfr", "bcbs_239"),
        typical_objects=(
            "deterministic_calculator",
            "statistical_model",
            "expert_judgment_overlay",
        ),
        measurement_vocabulary=(
            "behavioural_assumption_sensitivity",
            "survival_horizon",
            "runoff_profile",
            "concentration",
        ),
    ),
    RiskStripe(
        id="capital_stress",
        label="Capital adequacy and stress testing",
        description="Firm-wide scenario projection, capital planning, and the model chains that feed them.",
        mandatory_dimensions=(
            "conceptual_soundness",
            "assumption_validity",
            "stress_scenario",
            "sensitivity",
            "output_consumption",
            "documentation_completeness",
            "change_control",
        ),
        heightened_dimensions=("output_consumption", "stress_scenario", "documentation_completeness"),
        control_frameworks=("sr_11_7", "ccar_dfast", "basel_icaap", "bcbs_239"),
        typical_objects=(
            "statistical_model",
            "deterministic_calculator",
            "expert_judgment_overlay",
            "spreadsheet_euc",
        ),
        measurement_vocabulary=(
            "scenario_conditional_projection",
            "overlay_contribution",
            "model_chain_sensitivity",
        ),
    ),
    RiskStripe(
        id="valuation",
        label="Valuation and fair value",
        description="Pricing of instruments including level-2 and level-3 positions, curve "
        "and surface construction, independent price verification.",
        mandatory_dimensions=(
            "conceptual_soundness",
            "assumption_validity",
            "implementation_verification",
            "benchmarking",
            "sensitivity",
            "data_quality_lineage",
        ),
        heightened_dimensions=("benchmarking", "implementation_verification"),
        control_frameworks=("sr_11_7", "ifrs13", "basel_pruval"),
        typical_objects=("deterministic_calculator", "vendor_model", "statistical_model"),
        measurement_vocabulary=(
            "independent_price_variance",
            "calibration_residual",
            "arbitrage_consistency",
            "input_observability",
        ),
    ),
    RiskStripe(
        id="operational",
        label="Operational risk",
        description="Loss event modelling, capital allocation for operational exposures, "
        "process and control failure.",
        mandatory_dimensions=(
            "conceptual_soundness",
            "data_quality_lineage",
            "assumption_validity",
            "stress_scenario",
            "outcomes_analysis",
        ),
        heightened_dimensions=("data_quality_lineage", "assumption_validity"),
        control_frameworks=("sr_11_7", "basel_sma", "bcbs_239"),
        typical_objects=("statistical_model", "spreadsheet_euc", "expert_judgment_overlay"),
        measurement_vocabulary=("loss_distribution_fit", "scenario_elicitation", "tail_dependence"),
    ),
    RiskStripe(
        id="financial_crime",
        label="Financial crime / AML and sanctions",
        description="Transaction monitoring, customer risk rating, sanctions and PEP "
        "screening, alert triage.",
        mandatory_dimensions=(
            "conceptual_soundness",
            "data_quality_lineage",
            "implementation_verification",
            "outcomes_analysis",
            "use_boundary",
            "robustness_adversarial",
            "monitoring",
        ),
        heightened_dimensions=("outcomes_analysis", "robustness_adversarial", "use_boundary"),
        control_frameworks=("sr_11_7", "bsa_aml", "ofac_sanctions", "fatf_recommendations"),
        typical_objects=("rules_engine", "ml_model", "vendor_model", "scorecard"),
        measurement_vocabulary=(
            "alert_productivity",
            "coverage_of_typologies",
            "below_the_line_testing",
            "threshold_tuning_impact",
        ),
    ),
    RiskStripe(
        id="fraud",
        label="Fraud risk",
        description="Real-time and near-real-time detection of fraudulent activity across "
        "payments, application and account takeover.",
        mandatory_dimensions=(
            "discriminatory_power",
            "stability",
            "robustness_adversarial",
            "outcomes_analysis",
            "monitoring",
            "output_consumption",
        ),
        heightened_dimensions=("robustness_adversarial", "stability"),
        control_frameworks=("sr_11_7", "nist_ai_rmf"),
        typical_objects=("ml_model", "rules_engine", "deep_learning_model", "vendor_model"),
        measurement_vocabulary=(
            "detection_rate_at_review_capacity",
            "false_positive_burden",
            "adversarial_drift",
            "latency_budget",
        ),
    ),
    RiskStripe(
        id="conduct_compliance",
        label="Conduct and compliance",
        description="Fair treatment, suitability, surveillance of communications and "
        "trading conduct, marketing and pricing fairness.",
        mandatory_dimensions=(
            "bias_fairness",
            "use_boundary",
            "explainability",
            "outcomes_analysis",
            "documentation_completeness",
        ),
        heightened_dimensions=("bias_fairness", "explainability"),
        control_frameworks=("sr_11_7", "ecoa_reg_b", "eu_ai_act", "nist_ai_rmf"),
        typical_objects=("scorecard", "rules_engine", "llm_system", "ml_model"),
        measurement_vocabulary=(
            "disparity_ratio",
            "proxy_leakage",
            "reason_code_fidelity",
            "adverse_action_traceability",
        ),
    ),
    RiskStripe(
        id="climate",
        label="Climate and environmental risk",
        description="Physical and transition risk projection, scenario-conditional exposure "
        "and counterparty impact.",
        mandatory_dimensions=(
            "conceptual_soundness",
            "assumption_validity",
            "data_quality_lineage",
            "stress_scenario",
            "documentation_completeness",
        ),
        heightened_dimensions=("assumption_validity", "data_quality_lineage"),
        control_frameworks=("sr_11_7", "ngfs_scenarios", "tcfd_issb", "bcbs_239"),
        typical_objects=("statistical_model", "vendor_model", "expert_judgment_overlay"),
        measurement_vocabulary=(
            "scenario_pathway_sensitivity",
            "proxy_data_reliance",
            "horizon_consistency",
        ),
    ),
    RiskStripe(
        id="treasury_irrbb",
        label="Treasury and interest-rate risk in the banking book",
        description="Balance-sheet rate sensitivity, behavioural maturity, economic value "
        "and earnings measures.",
        mandatory_dimensions=(
            "assumption_validity",
            "sensitivity",
            "stress_scenario",
            "implementation_verification",
            "outcomes_analysis",
        ),
        heightened_dimensions=("assumption_validity", "sensitivity"),
        control_frameworks=("sr_11_7", "basel_irrbb", "bcbs_239"),
        typical_objects=("deterministic_calculator", "statistical_model"),
        measurement_vocabulary=("eve_sensitivity", "nii_sensitivity", "behavioural_life"),
    ),
    RiskStripe(
        id="model",
        label="Model risk (the inventory itself)",
        description="Review of the model risk management process: inventory completeness, "
        "tiering, validation coverage, issue remediation.",
        mandatory_dimensions=(
            "documentation_completeness",
            "change_control",
            "monitoring",
            "use_boundary",
            "third_party_diligence",
        ),
        heightened_dimensions=("change_control", "documentation_completeness"),
        control_frameworks=("sr_11_7", "occ_2011_12", "pra_ss1_23", "ecb_trim"),
        typical_objects=("monitoring_process", "spreadsheet_euc", "data_pipeline"),
        measurement_vocabulary=("inventory_completeness", "validation_cycle_adherence", "issue_ageing"),
    ),
    RiskStripe(
        id="technology_cyber",
        label="Technology and cyber risk",
        description="Availability, integrity and security of the systems risk objects run on.",
        mandatory_dimensions=(
            "implementation_verification",
            "robustness_adversarial",
            "change_control",
            "monitoring",
            "third_party_diligence",
        ),
        heightened_dimensions=("robustness_adversarial", "change_control"),
        control_frameworks=("nist_csf", "nist_ai_rmf", "iso_27001"),
        typical_objects=("data_pipeline", "vendor_model", "agentic_system"),
        measurement_vocabulary=("failure_injection", "recovery_objective", "dependency_blast_radius"),
    ),
    RiskStripe(
        id="third_party",
        label="Third-party and vendor risk",
        description="Dependence on externally supplied models, data and inference services.",
        mandatory_dimensions=(
            "third_party_diligence",
            "benchmarking",
            "outcomes_analysis",
            "change_control",
            "use_boundary",
        ),
        heightened_dimensions=("third_party_diligence", "change_control"),
        control_frameworks=("sr_11_7", "interagency_tprm", "eu_dora"),
        typical_objects=("vendor_model", "llm_system", "data_pipeline"),
        measurement_vocabulary=(
            "independent_replication",
            "version_change_notification",
            "substitutability",
        ),
    ),
    RiskStripe(
        id="ai_genai",
        label="AI and generative-AI risk",
        description="Risk arising from language models and agentic systems used in "
        "risk-relevant processes, including StART itself.",
        mandatory_dimensions=(
            "use_boundary",
            "robustness_adversarial",
            "output_consumption",
            "third_party_diligence",
            "monitoring",
            "change_control",
            "reproducibility",
        ),
        heightened_dimensions=("use_boundary", "output_consumption", "robustness_adversarial"),
        control_frameworks=("eu_ai_act", "nist_ai_rmf", "iso_42001", "sr_11_7"),
        typical_objects=("llm_system", "agentic_system", "vendor_model"),
        measurement_vocabulary=(
            "grounding_rate",
            "unbound_claim_rate",
            "narrative_invariance",
            "tool_action_scope",
            "prompt_injection_resistance",
        ),
    ),
)

STRIPES: dict[str, RiskStripe] = {s.id: s for s in _BUILTIN}


def stripe(stripe_id: str) -> RiskStripe:
    try:
        return STRIPES[stripe_id]
    except KeyError:
        raise KeyError(
            f"Unknown risk stripe {stripe_id!r}. Known stripes: {', '.join(sorted(STRIPES))}. "
            "Organisations may register additional stripes via the 'start.risk_stripes' "
            "entry point."
        ) from None


def stripe_ids() -> tuple[str, ...]:
    return tuple(sorted(STRIPES))


def register_stripe(new_stripe: RiskStripe, *, overwrite: bool = False) -> None:
    """Register a stripe at runtime.

    Overwriting a builtin requires ``overwrite=True`` so a plugin cannot quietly
    redefine what "credit risk" means for everyone else in the process.
    """
    if new_stripe.id in STRIPES and not overwrite:
        raise ValueError(f"Risk stripe {new_stripe.id!r} already exists. Pass overwrite=True to replace it.")
    STRIPES[new_stripe.id] = new_stripe


def load_entry_point_stripes() -> list[str]:
    """Load organisation-defined stripes from the ``start.risk_stripes`` group.

    Each entry point should resolve to a :class:`RiskStripe` or a zero-argument
    callable returning one. Failures are collected and returned rather than
    raised: a malformed plugin must not stop StART from running.
    """
    loaded: list[str] = []
    try:
        from importlib import metadata as importlib_metadata

        for ep in importlib_metadata.entry_points().select(group="start.risk_stripes"):
            try:
                target = ep.load()
                candidate = target() if callable(target) else target
                if isinstance(candidate, RiskStripe):
                    register_stripe(candidate, overwrite=True)
                    loaded.append(candidate.id)
            except Exception:  # pragma: no cover - a bad plugin must not break startup
                continue
    except Exception:  # pragma: no cover
        return loaded
    return loaded
