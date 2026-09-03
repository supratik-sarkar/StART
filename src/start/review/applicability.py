"""Applicable tests and the review plan preview.

The hard architectural rule
---------------------------

The CLI holds **no list of test IDs**. Applicability is derived by asking the registry
which registered surfaces declare a ``context_type`` the selected domains require.

A second hardcoded list is the specific failure worth avoiding. It looks harmless — a
tuple of twenty-two strings in a menu module — and then a new test is registered and the
menu silently never offers it. Nothing fails; the review is just quietly narrower than
the registry, and the gap widens with every release. Deriving the list means a newly
registered market surface appears in the next review with no CLI change at all.

Counts shown in the plan preview are computed the same way. The numbers in the tests
(market 20, treasury 2, market+treasury 22) assert *current registry behaviour*; they are
not inputs to the display logic.

Standard library only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from start.review.architecture import (
    DOMAIN_LABELS,
    LIFECYCLE_LABELS,
    MODE_LABELS,
    ReviewContextBundle,
    ReviewDomain,
    required_context_types,
)

__all__ = [
    "ApplicableTests", "applicable_tests", "ReviewPlanPreview", "build_plan_preview",
    "FAMILY_LABELS", "SCOPE_FAMILIES",
]

FAMILY_LABELS: dict[str, str] = {
    "portfolio": "Portfolio",
    "attribution": "Attribution",
    "traded_risk": "Traded Risk",
    "covariance": "Covariance",
    "preprocessing": "Preprocessing",
    "supervised": "Supervised",
    "xai": "Explainability",
    "eda": "Exploratory Analysis",
    "feature_engineering": "Feature Engineering",
    "genai": "GenAI",
}

#: Customisable scope groupings, per domain. Derived families only — nothing is offered
#: that the registry cannot supply.
SCOPE_FAMILIES: dict[ReviewDomain, tuple[str, ...]] = {
    ReviewDomain.MARKET: ("portfolio", "attribution", "traded_risk", "covariance"),
    ReviewDomain.TREASURY: ("traded_risk",),
    ReviewDomain.PREDICTIVE: (
        "eda", "preprocessing", "feature_engineering", "supervised", "xai",
    ),
}


@dataclass
class ApplicableTests:
    """Which registered surfaces apply, and why."""

    domains: tuple[ReviewDomain, ...]
    context_types: tuple[str, ...]
    test_ids: tuple[str, ...]
    by_context: dict[str, tuple[str, ...]] = field(default_factory=dict)
    by_family: dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.test_ids)

    def families(self) -> tuple[str, ...]:
        return tuple(sorted(self.by_family))

    def describe(self) -> dict[str, Any]:
        return {
            "domains": [str(d) for d in self.domains],
            "context_types": list(self.context_types),
            "n_applicable": self.count,
            "by_family": dict(sorted(self.by_family.items())),
            "by_context": {k: len(v) for k, v in sorted(self.by_context.items())},
        }


def applicable_tests(
    domains: tuple[ReviewDomain, ...],
    *,
    families: tuple[str, ...] | None = None,
) -> ApplicableTests:
    """Registered surfaces whose context type is required by the selected domains.

    ``families`` narrows a customised scope. It filters what the registry already
    offered; it can never add a surface the registry did not.
    """
    from start.registry import list_tests

    contexts = required_context_types(domains)
    wanted = set(contexts)

    selected = [
        spec for spec in list_tests()
        if getattr(spec, "context_type", "tabular") in wanted
        and (families is None or spec.family in families)
    ]
    selected.sort(key=lambda s: s.test_id)

    by_context: dict[str, tuple[str, ...]] = {}
    for context in contexts:
        by_context[context] = tuple(
            s.test_id for s in selected
            if getattr(s, "context_type", "tabular") == context
        )

    return ApplicableTests(
        domains=domains,
        context_types=contexts,
        test_ids=tuple(s.test_id for s in selected),
        by_context=by_context,
        by_family=dict(Counter(s.family for s in selected)),
    )


@dataclass
class ReviewPlanPreview:
    """What the reviewer is about to run, before anything executes."""

    bundle: ReviewContextBundle
    applicable: ApplicableTests
    evidence_enabled: bool = True
    ledger_enabled: bool = True
    narrative_enabled: bool = True
    critic_enabled: bool = True
    attestation_enabled: bool = True

    def lines(self) -> list[str]:
        bundle = self.bundle
        out = [
            "Review Plan",
            "=" * 44,
            "",
            f"  Mode:              {MODE_LABELS[bundle.mode][0]}",
            "  Domains:",
        ]
        for domain in bundle.domains:
            out.append(f"    [x] {DOMAIN_LABELS[domain]}")
        if bundle.technology is not None:
            from start.review.architecture import TECHNOLOGY_LABELS

            out.append(f"  Technology:        {TECHNOLOGY_LABELS[bundle.technology][0]}")
        out += [
            f"  Materiality:       {bundle.materiality.upper()}",
            f"  Lifecycle:         {LIFECYCLE_LABELS[bundle.lifecycle]}",
            "  Contexts:",
        ]
        for context in self.applicable.context_types:
            marker = "x" if bundle.context_for(context) is not None else " "
            out.append(f"    [{marker}] {context}")

        out += ["", f"  Applicable Registered Tests: {self.applicable.count}"]
        for family, count in sorted(self.applicable.by_family.items()):
            out.append(f"    [ROOT TEST] {FAMILY_LABELS.get(family, family):<20}{count}")

        if ReviewDomain.PREDICTIVE in bundle.domains:
            out += [
                "",
                "  Predictive Pre-Flight & Structural Diagnostics:",
                "    [DIAGNOSTIC] Outlier Detection: Robust interquartile range / z-score outliers",
                "    [DIAGNOSTIC] Correlation & Leakage: Pairwise collinearity and label leakage",
                "    [DIAGNOSTIC] Calibration & Drift: Expected calibration error and covariate drift",
            ]
        if ReviewDomain.MARKET in bundle.domains:
            out += [
                "",
                "  Pattern-B Subordinate Analytics:",
                "    [PATTERN-B] Linear Return Stress: Asset-level tail shocks",
                "    [PATTERN-B] Factor Linear Stress: Macro factor shift and specific risk",
                "    [PATTERN-B] Reverse Stress Testing: Mahalanobis minimum distance geometry",
                "    [PATTERN-B] Multi-Scenario Loss: Worst-scenario comparative ranking",
                "",
                "  Statistical & Structural Diagnostics:",
                "    [DIAGNOSTIC] Kupiec POF: Unconditional VaR coverage hypothesis test",
                "    [DIAGNOSTIC] Christoffersen: Independence & joint conditional coverage",
                "    [DIAGNOSTIC] Ledoit-Wolf Shrinkage: Condition improvement and intensity",
                "    [DIAGNOSTIC] RegEM Imputation: Structural cell pass rate under missingness",
            ]
        if ReviewDomain.TREASURY in bundle.domains:
            out += [
                "",
                "  Treasury Pre-Registered Validation Diagnostics:",
                "    [DIAGNOSTIC] CEV Consistency: Finite-sample nominal coverage [0.90, 1.00]",
                "    [DIAGNOSTIC] Stanton Bias: Non-zero drift wrong-sign rate <= 0.10",
            ]

        out += [
            "",
            "  Deferred Scope (Not offered in interactive selectors):",
            "    [DEFERRED] Monte Carlo VaR / ES revaluation",
            "    [DEFERRED] Formal Acerbi-Szekely ES spectral test",
            "    [DEFERRED] Delta-Gamma non-linear reverse stress",
            "    [DEFERRED] True Fama-MacBeth multi-pass cross-sectional estimator",
            "",
            "  Expected Review Checkpoints:",
        ]
        if ReviewDomain.PREDICTIVE in bundle.domains:
            out += [
                "    - Data Quality, Imbalance & Preprocessing Assumptions",
                "    - Model Architecture & Optimization Parameters",
                "    - Out-of-Sample Performance & Decision Metrics",
                "    - Feature Attribution & Explainability (XAI)",
                "    - Sensitivity, Robustness & Drift Analysis",
            ]
        if ReviewDomain.MARKET in bundle.domains:
            out += [
                "    1. Portfolio Risk & Volatility Assumptions",
                "    2. Factor Modeling & Attribution Assumptions",
                "    3. VaR Backtesting & Exception Frequency",
                "    4. Covariance Structure & Missing Data Treatment",
                "    5. Scenario Analysis & Stress Testing",
                "    6. Cross-Analytical Committee Synthesis",
            ]
        if ReviewDomain.TREASURY in bundle.domains:
            out += [
                "    - Short-Rate Diffusion & CEV Elasticity",
                "    - Stanton Nonparametric Drift & Diffusion",
            ]
        out += [
            "    - Barrier Validation (Conditional: omitted if barrier contracts N/A)",
            "    - Model Governance & Attestation Sign-off",
            "",
            "  Planned Visual & Tabular Artifacts:",
        ]
        if ReviewDomain.PREDICTIVE in bundle.domains:
            out += [
                "    - [ART] ROC & PR Performance Curves (JSON/PNG)",
                "    - [ART] Calibration Diagnostics (JSON/PNG)",
                "    - [ART] Feature Attribution & SHAP (JSON/PNG)",
            ]
        if ReviewDomain.MARKET in bundle.domains:
            out += [
                "    - [ART] Asset Weights Allocation (JSON)",
                "    - [ART] Covariance & Correlation Heatmaps (JSON)",
                "    - [ART] Factor Risk Model & Exposures (JSON)",
                "    - [ART] Reverse Stress Shock Geometry (JSON)",
            ]
        if ReviewDomain.TREASURY in bundle.domains:
            out += [
                "    - [ART] Short-Rate Path & Diffusion Diagnostics (JSON)",
                "    - [ART] CEV Elasticity Diagnostic Profile (JSON)",
                "    - [ART] Stanton Nonparametric Drift Profile (JSON)",
            ]

        llm_cfg = bundle.llm_config
        prov_map = {
            "openai": "OpenAI",
            "anthropic": "Anthropic",
            "gemini": "Gemini",
            "deepseek": "DeepSeek",
            "grok": "Grok",
            "enterprise_llm_gateway": "Enterprise LLM Gateway",
            "none": "None",
        }
        if llm_cfg.backend_mode == "public":
            prov_display = prov_map.get(llm_cfg.provider.lower(), llm_cfg.provider.title())
            out += [
                "",
                "  AI Reviewer Backend:       Public LLM",
                f"    Provider:                {prov_display}",
                f"    Model:                   {llm_cfg.model or 'default'}",
                f"    Backend Status:          {llm_cfg.status.title() if llm_cfg.status else 'Ready'}",
            ]
        elif llm_cfg.backend_mode == "enterprise":
            out += [
                "",
                "  AI Reviewer Backend:       Enterprise LLM Gateway",
                f"    Model:                   {llm_cfg.model or 'gateway-managed'}",
                f"    Backend Status:          {llm_cfg.status.title() if llm_cfg.status else 'Configured'}",
            ]
        else:
            out += [
                "",
                "  AI Reviewer Backend:       Deterministic Only",
            ]

        out += [
            "",
            f"  Evidence Store:            {'Enabled' if self.evidence_enabled else 'Disabled'}",
            f"  Ledger:                    "
            f"{'Chained / replayable' if self.ledger_enabled else 'Disabled'}",
            f"  Proof-Carrying Narrative:  "
            f"{'Enabled' if self.narrative_enabled else 'Disabled'}",
            f"  Evidence Critic:           {'Enabled' if self.critic_enabled else 'Disabled'}",
            f"  Attestation:               "
            f"{'Enabled' if self.attestation_enabled else 'Disabled'}",
        ]
        if not bundle.is_complete():
            out += [
                "",
                "  ! Missing required context(s): "
                + ", ".join(bundle.missing_context_types()),
                "    These must be supplied before the review can run.",
            ]
        return out

    def render(self) -> str:
        return "\n".join(self.lines())

    def reconcile_execution(
        self,
        executed_test_ids: set[str] | list[str],
        generated_artifact_ids: set[str] | list[str] | None = None,
    ) -> dict[str, Any]:
        """Reconcile the advertised review plan against actual executed surfaces and artifacts."""
        planned_tests = set(self.applicable.test_ids)
        executed = set(executed_test_ids)

        missing_planned = sorted(planned_tests - executed)
        unplanned_executed = sorted(executed - planned_tests)

        # Pre-registered validation or pattern-B analytics dynamically added during review
        accepted_extra_prefixes = (
            "scenario.",
            "validation.",
            "traded_risk.duration",
            "portfolio.adversarial.",
        )
        filtered_unplanned = [
            t for t in unplanned_executed
            if not any(t.startswith(p) for p in accepted_extra_prefixes)
        ]

        artifacts = set(generated_artifact_ids or ())

        return {
            "planned_count": len(planned_tests),
            "executed_count": len(executed),
            "reconciled": len(missing_planned) == 0 and len(filtered_unplanned) == 0,
            "missing_planned_tests": missing_planned,
            "unplanned_tests": filtered_unplanned,
            "artifacts_generated_count": len(artifacts),
        }


def build_plan_preview(bundle: ReviewContextBundle,
                       families: tuple[str, ...] | None = None) -> ReviewPlanPreview:
    """Assemble the preview. Every count is derived from the registry."""
    return ReviewPlanPreview(
        bundle=bundle,
        applicable=applicable_tests(bundle.domains, families=families),
    )
