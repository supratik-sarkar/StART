"""Review mode, domain and technology — the v4.3.0 routing vocabulary.

The problem this fixes
----------------------

The interactive shell used to open by asking users to choose between a *Propensity
Suite* and a *Deep Learning Suite*, then immediately offered Random Forest, XGBoost and
friends. That made sense when those were the product. It is now structurally wrong,
because it conflates three things that vary independently:

**Review mode** — is this one domain, or several composed into one governed review?

**Review domain** — which risk domain is under review? Predictive modelling, market
risk, or treasury/IRRBB. These are *what is being validated*.

**Predictive technology** — tree-based ML or deep learning. This is *how a predictive
model happens to be built*, and it only exists inside the predictive domain. Offering it
at the top level meant a market-risk reviewer was asked to choose a neural architecture
before being asked anything about their portfolio.

Separating the three is the whole point of the release. It also removes an entire class
of nonsense states: there is no longer any way to be doing a "Deep Learning review" of a
short-rate model.

Why there is no ``integrated`` domain
-------------------------------------

Market-plus-treasury is a **composition of two atomic domains**, not a third domain. If
it were its own value the set would grow combinatorially — market+treasury,
predictive+market, all three — and each new pairing would need its own context mapping,
its own applicability rule and its own test. Composition is expressed by selecting more
than one domain, and the context requirement is the union.

The same reasoning excludes ``deep_learning`` as a peer of ``market``: one is a
technology, the other is a domain.

GenAI is deliberately absent
----------------------------

StART uses agents internally, and there is a ``genai`` analytical family. But one
registered surface is not a validation domain, and presenting it as a peer of market
risk would promise something the analytical coverage does not support. A future release
can add ``ReviewDomain.GENAI_AGENTIC`` without disturbing any of this.

Standard library only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "ReviewMode", "ReviewDomain", "PredictiveTechnology", "ReviewLifecycle",
    "ReviewGroundingMode",
    "LLMReviewConfig", "ReviewContextBundle", "DOMAIN_CONTEXT", "DOMAIN_LABELS",
    "DOMAIN_DESCRIPTIONS", "TECHNOLOGY_LABELS", "LIFECYCLE_LABELS", "MODE_LABELS",
    "parse_domain_selection", "required_context_types", "requires_predictive_technology",
    "TRADITIONAL_ML_MODELS", "ExecutionProduct", "ReviewExecutionProducts",
]


class ReviewGroundingMode(StrEnum):
    """Execution mode for review claim grounding."""

    STRUCTURED = "STRUCTURED"
    LEGACY_FREEFORM = "LEGACY_FREEFORM"


class ReviewMode(StrEnum):
    """One domain, or several composed into a single governed review."""

    SINGLE_DOMAIN = "single_domain"
    CROSS_DOMAIN = "cross_domain"


class ReviewDomain(StrEnum):
    """The atomic, non-overlapping risk domains.

    Exactly three in v4.3.0. Compositions are expressed by selecting more than one, never
    by inventing a fourth value.
    """

    PREDICTIVE = "predictive"
    MARKET = "market"
    TREASURY = "treasury"


class PredictiveTechnology(StrEnum):
    """How a predictive model is built. Only meaningful inside the predictive domain."""

    TRADITIONAL_ML = "traditional_ml"
    DEEP_LEARNING = "deep_learning"


class ReviewLifecycle(StrEnum):
    """Where this review sits in the model's life.

    Governance metadata. It is recorded in evidence and shapes narrative framing, and it
    deliberately does **not** alter any analytical computation on its own — a periodic
    validation and an initial validation of the same model on the same data must produce
    the same numbers.
    """

    INITIAL_VALIDATION = "initial_validation"
    PERIODIC_VALIDATION = "periodic_validation"
    MATERIAL_MODEL_CHANGE = "material_model_change"
    ONGOING_MONITORING = "ongoing_monitoring"
    PRE_IMPLEMENTATION = "pre_implementation"


#: The mapping that makes routing work. Each domain requires exactly one typed context.
DOMAIN_CONTEXT: dict[ReviewDomain, str] = {
    ReviewDomain.PREDICTIVE: "tabular",
    ReviewDomain.MARKET: "market",
    ReviewDomain.TREASURY: "short_rate",
}

MODE_LABELS: dict[ReviewMode, tuple[str, str]] = {
    ReviewMode.SINGLE_DOMAIN: (
        "Single-Domain Review", "Review one risk/model domain independently"),
    ReviewMode.CROSS_DOMAIN: (
        "Cross-Domain Review", "Combine two or more domains within one governed review"),
}

DOMAIN_LABELS: dict[ReviewDomain, str] = {
    ReviewDomain.PREDICTIVE: "Predictive Modeling",
    ReviewDomain.MARKET: "Market Risk & Portfolio Analytics",
    ReviewDomain.TREASURY: "Treasury / IRRBB & Short-Rate Modeling",
}

DOMAIN_DESCRIPTIONS: dict[ReviewDomain, str] = {
    ReviewDomain.PREDICTIVE: "Supervised ML and deep-learning model validation",
    ReviewDomain.MARKET: (
        "Portfolio risk, attribution, VaR, covariance and barrier risk"),
    ReviewDomain.TREASURY: "Short-rate dynamics, CEV and Stanton diagnostics",
}

TECHNOLOGY_LABELS: dict[PredictiveTechnology, tuple[str, str]] = {
    PredictiveTechnology.TRADITIONAL_ML: (
        "Traditional / Tree-Based ML",
        "Random Forest, CatBoost, XGBoost, LightGBM, Extra Trees and supported "
        "classical models",
    ),
    PredictiveTechnology.DEEP_LEARNING: (
        "Deep Learning", "MLP, Wide & Deep and supported neural architectures"),
}

LIFECYCLE_LABELS: dict[ReviewLifecycle, str] = {
    ReviewLifecycle.INITIAL_VALIDATION: "Initial Validation",
    ReviewLifecycle.PERIODIC_VALIDATION: "Periodic Validation",
    ReviewLifecycle.MATERIAL_MODEL_CHANGE: "Material Model Change",
    ReviewLifecycle.ONGOING_MONITORING: "Ongoing Monitoring / Investigation",
    ReviewLifecycle.PRE_IMPLEMENTATION: "Pre-Implementation Design Review",
}

#: The legacy tree-model menu. Preserved exactly; it moves under
#: Predictive -> Traditional ML rather than disappearing.
TRADITIONAL_ML_MODELS: tuple[str, ...] = (
    "Random Forest", "CatBoost", "XGBoost", "LightGBM",
    "Distributed Random Forest", "Extra Trees", "Random Rotation Forest",
)

#: Menu order. Display index is 1-based and stable, so "2,3" always means the same thing.
_DOMAIN_ORDER: tuple[ReviewDomain, ...] = (
    ReviewDomain.PREDICTIVE, ReviewDomain.MARKET, ReviewDomain.TREASURY,
)


def parse_domain_selection(
    raw: str, *, mode: ReviewMode = ReviewMode.CROSS_DOMAIN
) -> tuple[ReviewDomain, ...]:
    """Parse a domain selection such as ``"2,3"`` into canonically ordered domains.

    Ordering is canonical (menu order), not input order, so ``"3,2"`` and ``"2,3"``
    produce the same review. Two reviewers describing the same scope differently must
    not get different plans, different context bundles or different evidence.

    Duplicates are rejected rather than silently deduplicated: ``"2,2"`` in cross-domain
    mode is far more likely a typo for ``"2,3"`` than a deliberate request, and quietly
    turning it into a single-domain review would hide the mistake.
    """
    if not raw or not raw.strip():
        raise ValueError("no domain selected")

    tokens = [token.strip() for token in raw.replace(" ", ",").split(",") if token.strip()]
    if not tokens:
        raise ValueError("no domain selected")

    seen: list[ReviewDomain] = []
    for token in tokens:
        if not token.isdigit():
            raise ValueError(
                f"invalid selection {token!r}: enter menu numbers separated by commas"
            )
        index = int(token)
        if not 1 <= index <= len(_DOMAIN_ORDER):
            raise ValueError(
                f"invalid selection {index}: choose between 1 and {len(_DOMAIN_ORDER)}"
            )
        domain = _DOMAIN_ORDER[index - 1]
        if domain in seen:
            raise ValueError(
                f"duplicate selection {index} ({DOMAIN_LABELS[domain]}): "
                "list each domain once"
            )
        seen.append(domain)

    if mode is ReviewMode.CROSS_DOMAIN and len(seen) < 2:
        raise ValueError(
            "a cross-domain review needs at least two distinct domains; "
            "choose Single-Domain Review for one"
        )
    if mode is ReviewMode.SINGLE_DOMAIN and len(seen) != 1:
        raise ValueError("a single-domain review takes exactly one domain")

    return tuple(d for d in _DOMAIN_ORDER if d in seen)


def required_context_types(domains: tuple[ReviewDomain, ...]) -> tuple[str, ...]:
    """The union of typed contexts the selected domains require.

    A union of existing context types — never a synthesised ``integrated`` or
    ``multi_context`` value. Those would be new analytical context types with no
    registered tests, and every applicability query would silently return nothing.
    """
    contexts: list[str] = []
    for domain in _DOMAIN_ORDER:
        if domain in domains:
            context = DOMAIN_CONTEXT[domain]
            if context not in contexts:
                contexts.append(context)
    return tuple(contexts)


def requires_predictive_technology(domains: tuple[ReviewDomain, ...]) -> bool:
    """Technology selection is offered only when the predictive domain is in scope."""
    return ReviewDomain.PREDICTIVE in domains


@dataclass
class LLMReviewConfig:
    """Explicit typed configuration for AI reviewer backend in interactive reviews."""

    backend_mode: str = "none"  # "none" | "enterprise" | "public"
    provider: str = "none"  # "openai" | "anthropic" | "gemini" | "deepseek" | "grok" | "none"
    model: str | None = None
    status: str = "DETERMINISTIC"  # "CONNECTED" | "CONFIGURED" | "FAILED" | "DETERMINISTIC"
    detail: str = ""

    def describe(self) -> dict[str, Any]:
        return {
            "backend_mode": self.backend_mode,
            "provider": self.provider,
            "model": self.model,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass
class ReviewContextBundle:
    """Orchestration-only container for the typed contexts a review needs.

    **This is not an analytical context type.** It has no ``context_type``, it is never
    registered, and no test dispatches on it. It exists so a cross-domain review can
    carry a ``TestContext`` and a ``MarketContext`` and a ``ShortRateContext`` together
    without anyone inventing a fourth analytical context to hold them.

    Making it an analytical context would be the seductive wrong move: every registered
    test would then need to know how to unpack it, and ``context_type`` filtering — the
    mechanism the whole applicability system rests on — would stop meaning anything.
    """

    tabular: Any | None = None
    market: Any | None = None
    short_rate: Any | None = None
    domains: tuple[ReviewDomain, ...] = field(default=())
    mode: ReviewMode = ReviewMode.SINGLE_DOMAIN
    technology: PredictiveTechnology | None = None
    lifecycle: ReviewLifecycle = ReviewLifecycle.INITIAL_VALIDATION
    materiality: str = "high"
    llm_config: LLMReviewConfig = field(default_factory=LLMReviewConfig)
    business_context: str = ""
    reviewer_clarification: str = ""
    intended_use: str = ""
    known_limitations: str = ""
    grounding_mode: ReviewGroundingMode = ReviewGroundingMode.LEGACY_FREEFORM
    structured_findings: list[Any] = field(default_factory=list)

    def context_for(self, context_type: str) -> Any | None:
        return {"tabular": self.tabular, "market": self.market,
                "short_rate": self.short_rate}.get(context_type)

    def available_context_types(self) -> tuple[str, ...]:
        return tuple(
            name for name in ("tabular", "market", "short_rate")
            if self.context_for(name) is not None
        )

    def missing_context_types(self) -> tuple[str, ...]:
        """Required by the selected domains but not populated."""
        return tuple(
            name for name in required_context_types(self.domains)
            if self.context_for(name) is None
        )

    def is_complete(self) -> bool:
        return not self.missing_context_types()

    def describe(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode),
            "domains": [str(d) for d in self.domains],
            "domain_labels": [DOMAIN_LABELS[d] for d in self.domains],
            "technology": str(self.technology) if self.technology else None,
            "lifecycle": str(self.lifecycle),
            "materiality": self.materiality,
            "llm_backend": self.llm_config.backend_mode,
            "llm_provider": self.llm_config.provider,
            "llm_model": self.llm_config.model,
            "llm_status": self.llm_config.status,
            "business_context": self.business_context,
            "reviewer_clarification": self.reviewer_clarification,
            "intended_use": self.intended_use,
            "known_limitations": self.known_limitations,
            "required_contexts": list(required_context_types(self.domains)),
            "available_contexts": list(self.available_context_types()),
            "missing_contexts": list(self.missing_context_types()),
            "complete": self.is_complete(),
        }


@dataclass
class ExecutionProduct:
    """Audit-grade container preserving a single typed deterministic execution output."""

    analytic_id: str
    result: Any
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    source_fingerprint: str = ""
    execution_provenance: str = "deterministic_engine"

    @property
    def result_object(self) -> Any:
        """Alias for result preserving typed analytical object."""
        return self.result


@dataclass
class ReviewExecutionProducts:
    """Preserves all typed deterministic result objects produced during review execution.

    Serves as the single source of truth for downstream Rich tables, semantic artifacts,
    and visual renderers without requiring ANY scientific recomputation.
    """

    products: dict[str, ExecutionProduct] = field(default_factory=dict)

    def register(
        self,
        analytic_id: str,
        result: Any,
        evidence_ids: tuple[str, ...] = (),
        source_fingerprint: str = "",
        provenance: str = "deterministic_engine",
    ) -> ExecutionProduct:
        prod = ExecutionProduct(
            analytic_id=analytic_id,
            result=result,
            evidence_ids=evidence_ids,
            source_fingerprint=source_fingerprint,
            execution_provenance=provenance,
        )
        self.products[analytic_id] = prod
        return prod

    def get_product(self, analytic_id: str) -> ExecutionProduct | None:
        prod = self.products.get(analytic_id)
        if prod is None and analytic_id == "scenario.asset_return":
            prod = self.products.get("scenario.linear_return")
        return prod

    def get_result(self, analytic_id: str) -> Any | None:
        p = self.get_product(analytic_id)
        return p.result if p is not None else None

    def has(self, analytic_id: str) -> bool:
        return analytic_id in self

    def __contains__(self, analytic_id: str) -> bool:
        if analytic_id in self.products:
            return True
        if analytic_id == "scenario.asset_return" and "scenario.linear_return" in self.products:
            return True
        return False

    def items(self) -> list[tuple[str, ExecutionProduct]]:
        return list(self.products.items())

    def fingerprint(self) -> str:
        raw = "|".join(sorted(f"{k}:{p.source_fingerprint}" for k, p in self.products.items()))
        return hashlib.sha256(raw.encode()).hexdigest()

    def summary(self) -> dict[str, str]:
        return {k: p.execution_provenance for k, p in self.products.items()}
