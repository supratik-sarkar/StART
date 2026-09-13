"""Control frameworks and coverage.

A review that satisfies a reviewer and a review that satisfies an examiner are
not automatically the same review. The gap between them is usually not analytic
rigour — it is traceability: which specific expectation does this piece of work
discharge, and where is the evidence that it did?

This module holds that mapping in one direction only. A framework declares a
set of expectations; each expectation names the review dimensions that
discharge it. StART then computes coverage from evidence that actually exists,
so an uncovered expectation is a fact about the review, not an opinion about it.

Two deliberate limits, stated plainly because tools in this space routinely
overclaim:

* These mappings are **interpretations**, not legal advice, and not a
  certification of compliance. They exist to make an argument auditable, and
  they are versioned so a reviewer can see which interpretation was in force
  when a review was sealed.
* Coverage means *a dimension was examined and produced evidence*. It does not
  mean the evidence was adequate. Adequacy is a judgement, and this module does
  not pretend to make it.

All references are to published supervisory guidance and public standards.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from start.risk.dimensions import DIMENSIONS

__all__ = [
    "Expectation",
    "ControlFramework",
    "FRAMEWORKS",
    "framework",
    "framework_ids",
    "MAPPING_VERSION",
    "coverage_report",
]

#: Version of the interpretation encoded here. Stamped into every coverage
#: report and every seal, so a future reader knows which reading was applied.
MAPPING_VERSION = "2026.08-1"


@dataclass(frozen=True)
class Expectation:
    """One traceable expectation within a framework."""

    id: str
    text: str
    #: Dimensions that, if examined, discharge this expectation.
    dimensions: tuple[str, ...]
    #: If True, partial coverage is not enough — every listed dimension is needed.
    requires_all: bool = False


@dataclass(frozen=True)
class ControlFramework:
    """A published framework, reduced to traceable expectations."""

    id: str
    label: str
    issuer: str
    scope: str
    expectations: tuple[Expectation, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "issuer": self.issuer,
            "scope": self.scope,
            "expectations": [
                {
                    "id": e.id,
                    "text": e.text,
                    "dimensions": list(e.dimensions),
                    "requires_all": e.requires_all,
                }
                for e in self.expectations
            ],
        }


_FRAMEWORKS: tuple[ControlFramework, ...] = (
    ControlFramework(
        id="sr_11_7",
        label="SR 11-7 / OCC 2011-12 — Supervisory Guidance on Model Risk Management",
        issuer="Board of Governors of the Federal Reserve System; OCC",
        scope="Model development, implementation, use; validation; governance.",
        expectations=(
            Expectation(
                id="sr11_7.dev.conceptual",
                text="Evaluate the conceptual soundness of the design and construction, "
                "including the quality of the theory and the evidence supporting it.",
                dimensions=("conceptual_soundness", "assumption_validity"),
                requires_all=True,
            ),
            Expectation(
                id="sr11_7.dev.data",
                text="Assess the quality and relevance of the data and other information "
                "used, and the appropriateness of any proxies.",
                dimensions=("data_quality_lineage",),
            ),
            Expectation(
                id="sr11_7.val.outcomes",
                text="Perform outcomes analysis comparing outputs to actual outcomes, "
                "including backtesting where feasible.",
                dimensions=("outcomes_analysis", "accuracy_calibration"),
            ),
            Expectation(
                id="sr11_7.val.process",
                text="Verify that the model is implemented as intended and that it "
                "performs as expected in the production environment.",
                dimensions=("implementation_verification", "reproducibility"),
                requires_all=True,
            ),
            Expectation(
                id="sr11_7.val.benchmark",
                text="Compare against alternative theories and approaches, including "
                "benchmarking to challenger models.",
                dimensions=("benchmarking",),
            ),
            Expectation(
                id="sr11_7.val.sensitivity",
                text="Conduct sensitivity analysis and evaluate behaviour over a range of "
                "input values, including extreme values.",
                dimensions=("sensitivity", "stress_scenario"),
            ),
            Expectation(
                id="sr11_7.use.limits",
                text="Establish and enforce the range of use for which the model is "
                "approved, including limitations on use.",
                dimensions=("use_boundary", "output_consumption"),
            ),
            Expectation(
                id="sr11_7.gov.monitoring",
                text="Maintain ongoing monitoring to confirm the model is appropriate and "
                "performing as intended.",
                dimensions=("monitoring", "stability"),
            ),
            Expectation(
                id="sr11_7.gov.documentation",
                text="Maintain documentation sufficient for a knowledgeable third party to "
                "understand and evaluate the model.",
                dimensions=("documentation_completeness", "change_control"),
            ),
            Expectation(
                id="sr11_7.vendor.thirdparty",
                text="Apply the same validation expectations to vendor and third-party "
                "products, adapted for limited transparency.",
                dimensions=("third_party_diligence", "benchmarking"),
            ),
        ),
    ),
    ControlFramework(
        id="occ_2011_12",
        label="OCC Bulletin 2011-12 — Sound Practices for Model Risk Management",
        issuer="Office of the Comptroller of the Currency",
        scope="Companion issuance to SR 11-7; identical substantive expectations.",
        expectations=(
            Expectation(
                id="occ2011_12.equivalent",
                text="Expectations align with SR 11-7; coverage is assessed against that "
                "framework's expectation set.",
                dimensions=("conceptual_soundness", "outcomes_analysis", "monitoring"),
            ),
        ),
    ),
    ControlFramework(
        id="bcbs_239",
        label="BCBS 239 — Principles for effective risk data aggregation and reporting",
        issuer="Basel Committee on Banking Supervision",
        scope="Data architecture, accuracy, completeness, timeliness, adaptability.",
        expectations=(
            Expectation(
                id="bcbs239.accuracy",
                text="Risk data aggregation must be accurate and reliable, with controls "
                "as strong as those over accounting data.",
                dimensions=("data_quality_lineage", "implementation_verification"),
                requires_all=True,
            ),
            Expectation(
                id="bcbs239.completeness",
                text="Capture and aggregate all material risk data across the group.",
                dimensions=("data_quality_lineage", "use_boundary"),
            ),
            Expectation(
                id="bcbs239.adaptability",
                text="Generate aggregated data to meet ad hoc requests, including under "
                "stress.",
                dimensions=("stress_scenario", "reproducibility"),
            ),
        ),
    ),
    ControlFramework(
        id="eu_ai_act",
        label="EU AI Act — obligations for high-risk AI systems",
        issuer="European Union",
        scope="Risk management, data governance, transparency, human oversight, "
        "robustness and accuracy, logging.",
        expectations=(
            Expectation(
                id="ai_act.risk_mgmt",
                text="Establish a continuous risk management system across the lifecycle.",
                dimensions=("conceptual_soundness", "monitoring", "change_control"),
            ),
            Expectation(
                id="ai_act.data_governance",
                text="Training, validation and testing data must be relevant, "
                "representative and examined for bias.",
                dimensions=("data_quality_lineage", "bias_fairness"),
                requires_all=True,
            ),
            Expectation(
                id="ai_act.transparency",
                text="Operation must be sufficiently transparent for deployers to "
                "interpret output and use it appropriately.",
                dimensions=("explainability", "use_boundary", "documentation_completeness"),
            ),
            Expectation(
                id="ai_act.oversight",
                text="Enable effective human oversight, including the ability to disregard "
                "or reverse output.",
                dimensions=("output_consumption", "use_boundary"),
                requires_all=True,
            ),
            Expectation(
                id="ai_act.robustness",
                text="Achieve appropriate accuracy, robustness and cybersecurity, and "
                "resist attempts to alter use or performance.",
                dimensions=("robustness_adversarial", "accuracy_calibration", "stability"),
            ),
            Expectation(
                id="ai_act.logging",
                text="Automatically record events over the lifetime of the system to a "
                "degree appropriate to its purpose.",
                dimensions=("reproducibility", "monitoring", "change_control"),
            ),
        ),
    ),
    ControlFramework(
        id="nist_ai_rmf",
        label="NIST AI Risk Management Framework 1.0",
        issuer="National Institute of Standards and Technology",
        scope="Govern, Map, Measure, Manage.",
        expectations=(
            Expectation(
                id="nist.govern",
                text="GOVERN: policies, accountability and culture for AI risk.",
                dimensions=("change_control", "documentation_completeness", "use_boundary"),
            ),
            Expectation(
                id="nist.map",
                text="MAP: establish context, intended purpose and downstream impact.",
                dimensions=("conceptual_soundness", "use_boundary", "output_consumption"),
            ),
            Expectation(
                id="nist.measure",
                text="MEASURE: analyse and track identified risks with repeatable methods.",
                dimensions=(
                    "accuracy_calibration",
                    "robustness_adversarial",
                    "bias_fairness",
                    "reproducibility",
                ),
            ),
            Expectation(
                id="nist.manage",
                text="MANAGE: prioritise and act on risks, including monitoring and "
                "response.",
                dimensions=("monitoring", "output_consumption"),
            ),
        ),
    ),
    ControlFramework(
        id="iso_42001",
        label="ISO/IEC 42001 — AI management systems",
        issuer="ISO/IEC",
        scope="Management-system requirements for responsible AI.",
        expectations=(
            Expectation(
                id="iso42001.lifecycle",
                text="Define and control the AI system lifecycle, including change and "
                "impact assessment.",
                dimensions=("change_control", "documentation_completeness", "monitoring"),
            ),
            Expectation(
                id="iso42001.thirdparty",
                text="Manage risks arising from third-party AI components and suppliers.",
                dimensions=("third_party_diligence",),
            ),
        ),
    ),
    ControlFramework(
        id="ecoa_reg_b",
        label="ECOA / Regulation B — adverse action and non-discrimination",
        issuer="CFPB (implementing ECOA)",
        scope="Credit decisions affecting individuals.",
        expectations=(
            Expectation(
                id="regb.reasons",
                text="Provide specific and accurate principal reasons for adverse action.",
                dimensions=("explainability",),
                requires_all=True,
            ),
            Expectation(
                id="regb.disparity",
                text="Avoid discrimination on a prohibited basis, including through proxies.",
                dimensions=("bias_fairness",),
                requires_all=True,
            ),
        ),
    ),
    ControlFramework(
        id="basel_frtb",
        label="Basel — Minimum capital requirements for market risk (FRTB)",
        issuer="Basel Committee on Banking Supervision",
        scope="Internal models approach, backtesting, P&L attribution.",
        expectations=(
            Expectation(
                id="frtb.backtest",
                text="Backtest risk measures against realised outcomes with defined "
                "exception thresholds.",
                dimensions=("outcomes_analysis",),
                requires_all=True,
            ),
            Expectation(
                id="frtb.pla",
                text="Demonstrate alignment between front-office and risk-model P&L.",
                dimensions=("benchmarking", "implementation_verification"),
            ),
            Expectation(
                id="frtb.factors",
                text="Justify risk-factor selection and modellability.",
                dimensions=("conceptual_soundness", "data_quality_lineage"),
            ),
        ),
    ),
    ControlFramework(
        id="ifrs9_cecl",
        label="IFRS 9 / CECL — expected credit loss measurement",
        issuer="IASB / FASB",
        scope="Forward-looking impairment estimation.",
        expectations=(
            Expectation(
                id="ecl.forward_looking",
                text="Incorporate reasonable and supportable forward-looking information, "
                "including multiple scenarios.",
                dimensions=("stress_scenario", "assumption_validity"),
            ),
            Expectation(
                id="ecl.overlays",
                text="Support and govern post-model adjustments and management overlays.",
                dimensions=("output_consumption", "documentation_completeness"),
                requires_all=True,
            ),
        ),
    ),
    ControlFramework(
        id="interagency_tprm",
        label="Interagency Guidance on Third-Party Relationships: Risk Management (2023)",
        issuer="Federal Reserve, FDIC, OCC",
        scope="Third-party lifecycle risk management.",
        expectations=(
            Expectation(
                id="tprm.diligence",
                text="Conduct due diligence proportionate to the risk of the activity.",
                dimensions=("third_party_diligence",),
                requires_all=True,
            ),
            Expectation(
                id="tprm.monitoring",
                text="Monitor third-party performance and changes throughout the "
                "relationship.",
                dimensions=("monitoring", "change_control"),
            ),
        ),
    ),
)

FRAMEWORKS: dict[str, ControlFramework] = {f.id: f for f in _FRAMEWORKS}

#: Frameworks referenced by stripes for which no expectation set is encoded yet.
#: Naming them explicitly is more honest than letting a lookup fail or, worse,
#: silently reporting full coverage of a framework nobody mapped.
UNMAPPED_FRAMEWORKS: frozenset[str] = frozenset(
    {
        "basel_irb",
        "basel_lcr_nsfr",
        "basel_icaap",
        "basel_irrbb",
        "basel_pruval",
        "basel_sma",
        "ccar_dfast",
        "ifrs13",
        "bsa_aml",
        "ofac_sanctions",
        "fatf_recommendations",
        "ngfs_scenarios",
        "tcfd_issb",
        "pra_ss1_23",
        "ecb_trim",
        "eu_dora",
        "nist_csf",
        "iso_27001",
    }
)


def framework(framework_id: str) -> ControlFramework:
    try:
        return FRAMEWORKS[framework_id]
    except KeyError:
        if framework_id in UNMAPPED_FRAMEWORKS:
            raise KeyError(
                f"Framework {framework_id!r} is referenced by a stripe but has no encoded "
                "expectation set in this release. StART reports it as unmapped rather than "
                "claiming coverage it cannot substantiate."
            ) from None
        raise KeyError(
            f"Unknown control framework {framework_id!r}. Known: {', '.join(sorted(FRAMEWORKS))}"
        ) from None


def framework_ids() -> tuple[str, ...]:
    return tuple(sorted(FRAMEWORKS))


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
def coverage_report(
    framework_ids_in_scope: list[str] | tuple[str, ...],
    examined_dimensions: set[str],
) -> dict[str, Any]:
    """Compute which expectations are discharged by the dimensions examined.

    ``examined_dimensions`` should contain only dimensions that actually
    produced evidence. Passing planned-but-unexecuted dimensions here converts
    the report from a statement of fact into a statement of intent, which is
    exactly the failure mode this module exists to prevent.
    """
    unknown = examined_dimensions - set(DIMENSIONS)
    if unknown:
        raise KeyError(f"Unknown dimension(s) in coverage input: {', '.join(sorted(unknown))}")

    per_framework: list[dict[str, Any]] = []
    unmapped: list[str] = []
    total_expectations = 0
    total_covered = 0

    for fid in framework_ids_in_scope:
        if fid not in FRAMEWORKS:
            unmapped.append(fid)
            continue
        fw = FRAMEWORKS[fid]
        rows: list[dict[str, Any]] = []
        covered_count = 0
        for exp in fw.expectations:
            hits = [d for d in exp.dimensions if d in examined_dimensions]
            if exp.requires_all:
                covered = len(hits) == len(exp.dimensions)
            else:
                covered = bool(hits)
            covered_count += int(covered)
            rows.append(
                {
                    "expectation_id": exp.id,
                    "text": exp.text,
                    "required_dimensions": list(exp.dimensions),
                    "requires_all": exp.requires_all,
                    "satisfied_by": hits,
                    "missing": [d for d in exp.dimensions if d not in examined_dimensions],
                    "covered": covered,
                }
            )
        total_expectations += len(fw.expectations)
        total_covered += covered_count
        per_framework.append(
            {
                "framework_id": fw.id,
                "label": fw.label,
                "issuer": fw.issuer,
                "expectations_total": len(fw.expectations),
                "expectations_covered": covered_count,
                "coverage_ratio": round(covered_count / len(fw.expectations), 4)
                if fw.expectations
                else 0.0,
                "expectations": rows,
            }
        )

    return {
        "mapping_version": MAPPING_VERSION,
        "frameworks": per_framework,
        "unmapped_frameworks": sorted(unmapped),
        "expectations_total": total_expectations,
        "expectations_covered": total_covered,
        "overall_coverage_ratio": round(total_covered / total_expectations, 4)
        if total_expectations
        else 0.0,
        "caveat": (
            "Coverage means a dimension was examined and produced evidence. It is not an "
            "assessment of whether that evidence was sufficient, and it is not a compliance "
            "certification."
        ),
    }
