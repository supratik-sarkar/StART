"""Generalized workflow-coherence contracts for StART.

This module is a thin product-contract layer over existing registries and runtime
objects.  It does not execute scientific calculations, change EvidenceRecords, or
replace either the unified review executor or the canonical execution service.

The central invariant is that every supported route can describe the same truthful
envelope while retaining domain-specific meaning::

    data -> objective -> capability plan -> decisions -> deterministic execution
         -> evidence -> grounding (when applicable) -> governance/policy -> outcome

All builders fail closed when asked to represent a configuration that the owning
runtime route cannot execute.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from itertools import combinations
from typing import Any


class CapabilityState(StrEnum):
    """Applicability state for one capability or workflow stage."""

    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNSUPPORTED = "UNSUPPORTED"
    BLOCKED = "BLOCKED"
    AVAILABLE_OPTIONAL = "AVAILABLE_OPTIONAL"


class SupportStatus(StrEnum):
    """Product-support classification used by the capability inventory."""

    SUPPORTED = "SUPPORTED"
    SUPPORTED_WITH_CONSTRAINTS = "SUPPORTED_WITH_CONSTRAINTS"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    DEFERRED = "DEFERRED"


class OutputStructure(StrEnum):
    """Target/output forms that must never be silently conflated."""

    SINGLE_TARGET = "SINGLE_TARGET"
    MULTICLASS_SINGLE_TARGET = "MULTICLASS_SINGLE_TARGET"
    MULTILABEL = "MULTILABEL"
    MULTI_OUTPUT_REGRESSION = "MULTI_OUTPUT_REGRESSION"
    SEQUENCE_CLASSIFICATION = "SEQUENCE_CLASSIFICATION"
    SEQUENCE_OUTPUT = "SEQUENCE_OUTPUT"
    DOMAIN_ANALYTICAL_OUTPUT = "DOMAIN_ANALYTICAL_OUTPUT"
    NOT_APPLICABLE = "NOT_APPLICABLE"


COHERENCE_STAGE_ORDER: tuple[str, ...] = (
    "DATA_INTAKE",
    "PROBLEM_OBJECTIVE",
    "CAPABILITY_PLAN",
    "HUMAN_DECISIONS",
    "AGENT_DECISIONS",
    "DETERMINISTIC_EXECUTION",
    "EVIDENCE",
    "GROUNDING",
    "GOVERNANCE_POLICY",
    "OUTCOME",
)


class UnsupportedCapabilityError(ValueError):
    """Typed fail-closed result for a configuration with no truthful runtime path."""

    def __init__(
        self,
        code: str,
        reason: str,
        *,
        requested: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {reason}")
        self.code = code
        self.reason = reason
        self.requested = requested or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": CapabilityState.UNSUPPORTED.value,
            "code": self.code,
            "reason": self.reason,
            "requested": _json_safe(self.requested),
        }


def _json_safe(value: Any) -> Any:
    """Return deterministic JSON-compatible values without serializing raw data."""
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if hasattr(value, "describe"):
        return _json_safe(value.describe())
    return str(value)


def _fingerprint(payload: Any) -> str:
    canonical = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DataIntakeContract:
    """Canonical, non-data-bearing description of the resolved runtime input."""

    source_kind: str
    source_provider_or_adapter: str
    source_reference: str
    context_id: str
    provenance_fingerprint: str
    schema_summary: tuple[str, ...]
    dimensions: dict[str, int | str | None]
    observation_semantics: str
    entity_key: str | None = None
    temporal_key: str | None = None
    as_of: str | None = None
    freshness: str = "NOT_DECLARED"
    missingness_summary: str = "NOT_EVALUATED_AT_INTAKE"
    load_mode: str = "in_memory"
    provenance_metadata: dict[str, Any] = field(default_factory=dict)
    validation_status: str = "RESOLVED"
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_kind or not self.context_id:
            raise ValueError("DataIntakeContract requires source_kind and context_id")
        if not self.provenance_fingerprint:
            raise ValueError("DataIntakeContract requires a provenance fingerprint")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class ProblemContract:
    """Truthful problem/objective definition with domain-specific payloads."""

    domains: tuple[str, ...]
    workflow_purpose: str
    lifecycle: str
    objective: str
    task_or_risk_type: str
    decision_impact: str
    criterion_provenance: tuple[str, ...]
    validation_design: str
    candidate_methods: tuple[str, ...]
    constraints: tuple[str, ...]
    decision_metrics: tuple[str, ...]
    output_structure: OutputStructure
    target_specification: tuple[str, ...] = ()
    model_input_contract: str = "NOT_APPLICABLE"
    domain_payloads: dict[str, dict[str, Any]] = field(default_factory=dict)
    unsupported_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.domains:
            raise ValueError("ProblemContract requires at least one domain")
        if not self.objective:
            raise ValueError("ProblemContract requires an objective")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class PlannedCapability:
    capability_id: str
    state: CapabilityState
    owner: str
    reason: str
    registry_reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class CapabilityPlan:
    """Registry/configuration-derived applicability plan."""

    route_id: str
    review_mode: str
    stages: dict[str, CapabilityState]
    capabilities: tuple[PlannedCapability, ...]
    registered_test_ids: tuple[str, ...]
    artifact_families: tuple[str, ...]
    environment_projection: tuple[PlannedCapability, ...]
    unsupported_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        missing = set(COHERENCE_STAGE_ORDER) - set(self.stages)
        if missing:
            raise ValueError(f"CapabilityPlan is missing stages: {sorted(missing)}")
        if any(state is CapabilityState.UNSUPPORTED for state in self.stages.values()):
            if not self.unsupported_reasons:
                raise ValueError("Unsupported plan must include an explanatory reason")

    def state_for(self, stage: str) -> CapabilityState:
        return self.stages[stage]

    @property
    def supported(self) -> bool:
        return not any(
            state in {CapabilityState.UNSUPPORTED, CapabilityState.BLOCKED}
            for state in self.stages.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "review_mode": self.review_mode,
            "stages": {key: value.value for key, value in self.stages.items()},
            "capabilities": [item.to_dict() for item in self.capabilities],
            "registered_test_ids": list(self.registered_test_ids),
            "artifact_families": list(self.artifact_families),
            "environment_projection": [item.to_dict() for item in self.environment_projection],
            "unsupported_reasons": list(self.unsupported_reasons),
            "supported": self.supported,
        }


@dataclass(frozen=True)
class AgentDecisionTrace:
    """Auditable rationale summary; never private chain-of-thought or numeric authority."""

    agent_identity: str
    role: str
    trigger: str
    checkpoint: str
    canonical_input_references: tuple[str, ...]
    evidence_record_references: tuple[str, ...]
    applicable_alternatives: tuple[str, ...]
    recommendation: str
    rationale_summary: str
    limitations: tuple[str, ...]
    human_action: str
    resulting_action: str
    numeric_authority: str = "NONE"

    def __post_init__(self) -> None:
        if self.numeric_authority != "NONE":
            raise ValueError("AgentDecisionTrace numeric authority must remain NONE")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class DeterministicExecutionSummary:
    """Index over canonical deterministic outputs; it performs no recomputation."""

    engine_ids: tuple[str, ...]
    canonical_input_references: tuple[str, ...]
    operations: tuple[str, ...]
    authoritative_outputs: tuple[str, ...]
    statuses: tuple[str, ...]
    evidence_record_references: tuple[str, ...]
    artifact_references: tuple[str, ...]
    limitations: tuple[str, ...]
    skipped_or_not_applicable: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass(frozen=True)
class RunOutcomeCapsule:
    """Reusable terminal/report/frontend projection of completed canonical state."""

    run_id: str
    started_with: str
    objective: str
    alternatives: tuple[str, ...]
    human_decisions: tuple[str, ...]
    agent_contribution: tuple[str, ...] | str
    deterministic_results: tuple[str, ...]
    evidence_record_references: tuple[str, ...]
    grounding: str
    governance_policy: str
    achieved_result: str
    remaining_conditions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class WorkflowCoherenceEnvelope:
    """Canonical per-run coherence object shared by all presentation consumers."""

    run_id: str
    data_intake: DataIntakeContract
    problem: ProblemContract
    capability_plan: CapabilityPlan
    agent_decision_traces: list[AgentDecisionTrace] = field(default_factory=list)
    deterministic_summaries: list[DeterministicExecutionSummary] = field(default_factory=list)
    outcome: RunOutcomeCapsule | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "data_intake": self.data_intake.to_dict(),
            "problem": self.problem.to_dict(),
            "capability_plan": self.capability_plan.to_dict(),
            "agent_decision_traces": [trace.to_dict() for trace in self.agent_decision_traces],
            "deterministic_summaries": [item.to_dict() for item in self.deterministic_summaries],
            "outcome": self.outcome.to_dict() if self.outcome else None,
        }


def _columns_and_dtypes(frame: Any) -> tuple[str, ...]:
    raw_columns = getattr(frame, "columns", ())
    columns = list(raw_columns) if raw_columns is not None else []
    dtypes = getattr(frame, "dtypes", None)
    return tuple(
        f"{column}:{getattr(dtypes, 'get', lambda _name, _default='unknown': _default)(column, 'unknown')}"
        for column in columns
    )


def _target_tuple(target: Any) -> tuple[str, ...]:
    if target is None or target == "":
        return ()
    if isinstance(target, str):
        return (target,)
    if isinstance(target, (list, tuple, set, frozenset)):
        return tuple(str(item) for item in target)
    return (str(target),)


def canonical_predictive_explainability_identity(method: Any) -> str:
    """Return the product-contract identity for an actually executed method.

    The temporal executor exposes its human-readable method name while its
    deterministic algorithm identity is ``INPUT_GRADIENT``. Other predictive
    explainers already expose their canonical runtime identity and are
    preserved verbatim (for example ``permutation``).
    """
    resolved = str(method or "").strip()
    normalized = resolved.lower().replace("-", "_").replace(" ", "_")
    if normalized in {
        "input_gradient",
        "temporal_input_gradient_saliency",
        "temporal_input_gradient_saliency_/_input_gradient",
    }:
        return "Temporal Input-Gradient Saliency / INPUT_GRADIENT"
    return resolved or "NOT_APPLICABLE"


def synchronize_predictive_explainability(
    envelope: WorkflowCoherenceEnvelope,
    actual_method: Any,
) -> str:
    """Project the actually executed predictive method into its contract.

    This is a contract synchronization only. It does not select an explainer
    or alter any scientific output.
    """
    if "predictive" not in envelope.problem.domains:
        raise ValueError("predictive explainability can only be synchronized for predictive contracts")
    canonical = canonical_predictive_explainability_identity(actual_method)
    domain_payloads = {
        domain: dict(payload)
        for domain, payload in envelope.problem.domain_payloads.items()
    }
    domain_payloads.setdefault("predictive", {})["explainability"] = canonical
    envelope.problem = replace(envelope.problem, domain_payloads=domain_payloads)
    return canonical


def resolve_predictive_output_structure(
    *,
    task_type: str,
    targets: tuple[str, ...],
    route_supports_multi_output: bool,
    sequence_input: bool,
) -> OutputStructure:
    """Resolve output form and fail closed instead of silently dropping targets."""
    task = (task_type or "").lower()
    if len(targets) > 1 and not route_supports_multi_output:
        raise UnsupportedCapabilityError(
            "MULTI_TARGET_ROUTE_UNSUPPORTED",
            "the selected review route accepts one target and cannot silently collapse multiple targets",
            requested={"task_type": task_type, "targets": targets},
        )
    if task == "multilabel_classification":
        if not route_supports_multi_output:
            raise UnsupportedCapabilityError(
                "MULTILABEL_ROUTE_UNSUPPORTED",
                "multilabel estimation exists in TabularDLClassifier but is not wired to this review route",
                requested={"task_type": task_type, "targets": targets},
            )
        return OutputStructure.MULTILABEL
    if len(targets) > 1 and task in {"regression", "forecasting"}:
        return OutputStructure.MULTI_OUTPUT_REGRESSION
    if task in {"sequence_output", "sequence_to_sequence"}:
        raise UnsupportedCapabilityError(
            "SEQUENCE_OUTPUT_UNSUPPORTED",
            "registered sequence workflows classify whole sequences; sequence-output prediction is not wired",
            requested={"task_type": task_type},
        )
    if sequence_input:
        return OutputStructure.SEQUENCE_CLASSIFICATION
    if task == "multiclass_classification":
        return OutputStructure.MULTICLASS_SINGLE_TARGET
    return OutputStructure.SINGLE_TARGET


def _environment_projection(*, agent_assisted: bool) -> tuple[PlannedCapability, ...]:
    ai_state = CapabilityState.APPLICABLE if agent_assisted else CapabilityState.NOT_APPLICABLE
    return (
        PlannedCapability(
            "ai.policy_and_guardrails",
            ai_state,
            "start.ai_engineering + start.policies",
            "agent-facing policy/guardrail projection" if agent_assisted else "no AI agent executes on this route",
            "CAPABILITY_REGISTRY",
        ),
        PlannedCapability(
            "ai.tool_and_adapter_integration",
            ai_state,
            "start.ai_engineering.adapters",
            "registered adapters are projected by capability and availability",
            "build_adapters",
        ),
        PlannedCapability(
            "deterministic.policy",
            CapabilityState.APPLICABLE,
            "start.policies.opa_policy_plane",
            "OPA/local policy remains applicable independently of AI execution",
            "CAPABILITY_REGISTRY",
        ),
        PlannedCapability(
            "observability.otel",
            CapabilityState.APPLICABLE,
            "start.telemetry.engineering_trace",
            "in-process trace is available on all governed execution routes",
            "CAPABILITY_REGISTRY",
        ),
        PlannedCapability(
            "evidence.attestation",
            CapabilityState.APPLICABLE,
            "start.evidence + start.attestation",
            "canonical evidence and attestation remain deterministic",
            "CAPABILITY_REGISTRY",
        ),
    )


def _base_stage_plan(*, agent_assisted: bool, grounding: bool, human: bool) -> dict[str, CapabilityState]:
    return {
        "DATA_INTAKE": CapabilityState.APPLICABLE,
        "PROBLEM_OBJECTIVE": CapabilityState.APPLICABLE,
        "CAPABILITY_PLAN": CapabilityState.APPLICABLE,
        "HUMAN_DECISIONS": CapabilityState.APPLICABLE if human else CapabilityState.NOT_APPLICABLE,
        "AGENT_DECISIONS": CapabilityState.APPLICABLE if agent_assisted else CapabilityState.NOT_APPLICABLE,
        "DETERMINISTIC_EXECUTION": CapabilityState.APPLICABLE,
        "EVIDENCE": CapabilityState.APPLICABLE,
        "GROUNDING": CapabilityState.APPLICABLE if grounding else CapabilityState.NOT_APPLICABLE,
        "GOVERNANCE_POLICY": CapabilityState.APPLICABLE,
        "OUTCOME": CapabilityState.APPLICABLE,
    }


def build_predictive_coherence_envelope(
    *,
    run_id: str,
    config: Any,
    frame: Any,
    task_type: str,
    presentation_context: Any,
    dataset_selection: Any | None = None,
) -> WorkflowCoherenceEnvelope:
    """Adapt the actual interactive predictive route into the common envelope."""
    targets = _target_tuple(getattr(config, "target", None))
    output = resolve_predictive_output_structure(
        task_type=task_type,
        targets=targets,
        route_supports_multi_output=False,
        sequence_input=getattr(config, "sequence_bundle", None) is not None,
    )
    if dataset_selection is not None:
        provenance = dataset_selection.provenance_dict()
        source_kind = str(getattr(getattr(dataset_selection, "kind", ""), "value", ""))
        source_provider = "start.data.selection"
        source_reference = str(getattr(dataset_selection, "source_reference", ""))
        context_id = str(getattr(dataset_selection, "display_name", "interactive_dataset"))
        limitations = tuple(str(item) for item in getattr(dataset_selection, "notes", ()) or ())
    else:
        source_path = getattr(config, "data_path", None)
        provenance = {
            "source_path": source_path,
            "shape": list(getattr(frame, "shape", ())),
            "columns": list(getattr(frame, "columns", ())),
        }
        source_kind = "user_supplied" if source_path else "synthetic"
        source_provider = "start.data.loaders" if source_path else "start.modeling.data"
        source_reference = str(source_path or "built-in deterministic generator")
        context_id = str(source_path or "built_in_predictive_context")
        limitations = ()
    dimensions: dict[str, int | str | None] = {
        "samples": int(getattr(presentation_context, "model_sample_count", len(frame))),
        "features": int(getattr(presentation_context, "model_feature_count", 0)),
        "timesteps": getattr(presentation_context, "timesteps", None),
    }
    data = DataIntakeContract(
        source_kind=source_kind,
        source_provider_or_adapter=source_provider,
        source_reference=source_reference,
        context_id=context_id,
        provenance_fingerprint=_fingerprint(provenance),
        schema_summary=_columns_and_dtypes(frame),
        dimensions=dimensions,
        observation_semantics=str(getattr(presentation_context, "sample_structure", "tabular rows")),
        temporal_key=None,
        load_mode="prepared_in_memory" if dataset_selection is not None else "interactive_loader",
        provenance_metadata=provenance,
        validation_status="RESOLVED",
        limitations=limitations,
    )
    candidates = tuple(getattr(presentation_context, "architecture_alternatives", ()) or ())
    modality = str(getattr(presentation_context, "modality", "tabular"))
    resolved_explainability = (
        "Temporal Input-Gradient Saliency / INPUT_GRADIENT"
        if modality == "temporal_sequence"
        else "PENDING_RUNTIME_RESOLUTION"
    )
    problem = ProblemContract(
        domains=("predictive",),
        workflow_purpose=(
            "development-time model building + evidence-native self-review"
            if bool(getattr(config, "run_dl", False))
            else "predictive diagnostics and evidence-native review"
        ),
        lifecycle="interactive_review",
        objective=str(getattr(config, "objective", "") or f"review {task_type} model"),
        task_or_risk_type=task_type,
        decision_impact=str(getattr(config, "costlier_errors", "balanced")),
        criterion_provenance=("runtime configuration", "registered deterministic diagnostics"),
        validation_design=str(getattr(presentation_context, "split_description", "NOT_DECLARED")),
        candidate_methods=candidates,
        constraints=(
            f"architecture={getattr(config, 'architecture_family', None) or 'mlp'}",
            f"activation={getattr(config, 'activation', None) or 'relu'}",
            "LLM numeric authority=0",
        ),
        decision_metrics=("runtime-selected primary metric", "registered diagnostic statuses"),
        output_structure=output,
        target_specification=targets,
        model_input_contract=str(getattr(presentation_context, "model_input_contract", "NOT_DECLARED")),
        domain_payloads={
            "predictive": {
                "modality": modality,
                "split_strategy": getattr(presentation_context, "split_description", ""),
                "explainability": resolved_explainability,
                "requested_explainability": str(
                    getattr(config, "explain_method", "") or "NOT_DECLARED"
                ),
                "robustness_suite": getattr(config, "robustness_suite", ""),
            }
        },
    )
    agent_assisted = bool(getattr(config, "enterprise_mode", True))
    registered_tests = (
        "discovery.task_inference",
        "split.plan",
        "deep_learning.performance_diagnostics",
        "deep_learning.explainability_diagnostics",
        "deep_learning.robustness_diagnostics",
    )
    plan = CapabilityPlan(
        route_id="interactive_predictive_review",
        review_mode="single_domain",
        stages=_base_stage_plan(agent_assisted=agent_assisted, grounding=agent_assisted, human=True),
        capabilities=(
            PlannedCapability(
                "langgraph.live_review",
                CapabilityState.APPLICABLE if agent_assisted else CapabilityState.NOT_APPLICABLE,
                "start.orchestration.state_graph",
                "live review phases are entered by the enterprise review path",
                "build_live_review_graph",
            ),
            PlannedCapability(
                "predictive.model_execution",
                CapabilityState.APPLICABLE if getattr(config, "run_dl", False) else CapabilityState.NOT_APPLICABLE,
                "start.modeling.model_execution",
                "runtime configuration controls whether model execution is requested",
                "architecture_registry",
            ),
            PlannedCapability(
                "evidence.grounding",
                CapabilityState.APPLICABLE if agent_assisted else CapabilityState.NOT_APPLICABLE,
                "start.review.structured_contract",
                "agent claims must ground to canonical evidence",
                "grounding contract",
            ),
        ),
        registered_test_ids=registered_tests,
        artifact_families=("performance", "explainability", "robustness", "evidence"),
        environment_projection=_environment_projection(agent_assisted=agent_assisted),
    )
    return WorkflowCoherenceEnvelope(run_id=run_id, data_intake=data, problem=problem, capability_plan=plan)


def _context_description(context: Any) -> dict[str, Any]:
    if context is None:
        return {}
    if hasattr(context, "describe"):
        try:
            result = context.describe()
            return result if isinstance(result, dict) else {"description": str(result)}
        except Exception:
            return {"type": type(context).__name__}
    return {"type": type(context).__name__}


def build_unified_review_coherence_envelope(
    *,
    run_id: str,
    bundle: Any,
    applicable: Any,
) -> WorkflowCoherenceEnvelope:
    """Adapt Single/Cross-Domain unified review state without flattening domains."""
    domains = tuple(str(getattr(domain, "value", domain)) for domain in bundle.domains)
    context_payloads: dict[str, dict[str, Any]] = {}
    resolved_context_labels: list[str] = []
    for context_name in bundle.presentation_context_types():
        owner_name = "tabular" if context_name == "temporal_sequence" else context_name
        context = bundle.context_for(owner_name)
        context_payloads[context_name] = _context_description(context)
        resolved_context_labels.append(type(context).__name__)
    missing = tuple(bundle.missing_context_types())
    if missing:
        raise UnsupportedCapabilityError(
            "REQUIRED_CONTEXT_MISSING",
            "selected domains do not have every required typed context",
            requested={"domains": domains, "missing_contexts": missing},
        )
    mode = str(getattr(bundle.mode, "value", bundle.mode))
    data_kind = "composite" if len(domains) > 1 else next(iter(context_payloads), "structured")
    provenance = {
        "domains": domains,
        "contexts": context_payloads,
        "mode": mode,
        "selected_original_source": str(
            getattr(bundle, "selected_source", "") or ""
        ),
    }
    data = DataIntakeContract(
        source_kind=data_kind,
        source_provider_or_adapter="start.review.ReviewContextBundle",
        source_reference="typed in-memory review contexts",
        context_id=f"{mode}:{'+'.join(domains)}",
        provenance_fingerprint=_fingerprint(provenance),
        schema_summary=tuple(resolved_context_labels),
        dimensions={"domain_count": len(domains), "context_count": len(context_payloads)},
        observation_semantics="domain-specific typed contexts; constituents preserved",
        load_mode="prepared_in_memory_context",
        provenance_metadata=provenance,
        validation_status="RESOLVED",
        limitations=(
            (
                "Cross-domain execution composes registered domain surfaces; "
                "it does not create a new analytical context."
            ),
        )
        if len(domains) > 1
        else (),
    )
    test_ids = tuple(applicable.test_ids)
    families = tuple(sorted(applicable.by_family))
    context_by_domain = {
        "predictive": "tabular",
        "market": "market",
        "treasury": "short_rate",
    }
    surfaces_by_domain: dict[str, tuple[str, ...]] = {
        domain: tuple(applicable.by_context.get(context_by_domain[domain], ()))
        for domain in domains
    }
    domain_payloads: dict[str, dict[str, Any]] = {}
    if "predictive" in domains:
        domain_payloads["predictive"] = {
            "technology": str(getattr(getattr(bundle, "technology", None), "value", "NOT_DECLARED")),
            "context": context_payloads.get("tabular") or context_payloads.get("temporal_sequence", {}),
            "registered_families": sorted(
                {surface.split(".", 1)[0] for surface in surfaces_by_domain["predictive"]}
            ),
            "analytical_surfaces": list(surfaces_by_domain["predictive"]),
        }
    if "market" in domains:
        market = getattr(bundle, "market", None)
        returns = getattr(market, "returns", None) if market is not None else None
        domain_payloads["market"] = {
            "assets": list(getattr(returns, "columns", ())) if returns is not None else [],
            "observations": len(returns) if returns is not None else 0,
            "registered_families": [name for name in families if name in {"portfolio", "attribution", "traded_risk", "covariance"}],
            "analytical_surfaces": list(surfaces_by_domain["market"]),
        }
    if "treasury" in domains:
        short_rate = getattr(bundle, "short_rate", None)
        rates = getattr(short_rate, "rates", None) if short_rate is not None else None
        domain_payloads["treasury"] = {
            "observations": len(rates) if rates is not None else 0,
            "registered_surfaces": [tid for tid in test_ids if tid.startswith("traded_risk.cev_") or tid.startswith("traded_risk.stanton_")],
            "analytical_surfaces": list(surfaces_by_domain["treasury"]),
        }
    candidate_methods = (
        tuple(
            f"{domain}:{surface}"
            for domain in domains
            for surface in surfaces_by_domain[domain]
        )
        if len(domains) > 1
        else surfaces_by_domain[domains[0]]
    )
    problem = ProblemContract(
        domains=domains,
        workflow_purpose="governed evidence-native domain review",
        lifecycle=str(getattr(getattr(bundle, "lifecycle", ""), "value", getattr(bundle, "lifecycle", ""))),
        objective=str(getattr(bundle, "business_context", "") or f"review {' + '.join(domains)}"),
        task_or_risk_type="cross_domain_review" if len(domains) > 1 else f"{domains[0]}_review",
        decision_impact=str(getattr(bundle, "intended_use", "") or "NOT_SUPPLIED"),
        criterion_provenance=("registered test specifications", "runtime governance metadata"),
        validation_design=f"{len(test_ids)} registry-derived applicable analytical surfaces",
        candidate_methods=candidate_methods,
        constraints=(
            f"materiality={getattr(bundle, 'materiality', 'NOT_DECLARED')}",
            "LLM numeric authority=0",
        ),
        decision_metrics=tuple(sorted(applicable.by_family)),
        output_structure=OutputStructure.DOMAIN_ANALYTICAL_OUTPUT,
        model_input_contract="NOT_APPLICABLE" if "predictive" not in domains else "typed predictive context",
        domain_payloads=domain_payloads,
    )
    capabilities = tuple(
        PlannedCapability(
            f"test.{test_id}",
            CapabilityState.APPLICABLE,
            "start.registry",
            "selected by declared context_type and optional family scope",
            test_id,
        )
        for test_id in test_ids
    )
    plan = CapabilityPlan(
        route_id="unified_domain_review",
        review_mode=mode,
        stages=_base_stage_plan(agent_assisted=True, grounding=True, human=True),
        capabilities=capabilities,
        registered_test_ids=test_ids,
        artifact_families=families,
        environment_projection=_environment_projection(agent_assisted=True),
    )
    return WorkflowCoherenceEnvelope(run_id=run_id, data_intake=data, problem=problem, capability_plan=plan)


def build_canonical_runtime_coherence_envelope(
    *,
    run_id: str,
    workflow_id: str,
    context_id: str,
    context_metadata: dict[str, Any],
    resolved_metadata: dict[str, Any] | None = None,
) -> WorkflowCoherenceEnvelope:
    """Adapt canonical deterministic workflow configuration into the envelope."""
    from start.runtime.contexts import resolve_context_spec
    from start.runtime.workflows import get_canonical_workflow_specs, resolve_workflow

    specs = get_canonical_workflow_specs()
    if workflow_id not in specs:
        raise UnsupportedCapabilityError(
            "UNKNOWN_WORKFLOW", "workflow is not present in the canonical registry", requested={"workflow_id": workflow_id}
        )
    spec = specs[workflow_id]
    if not spec.enabled:
        raise UnsupportedCapabilityError(
            "WORKFLOW_DISABLED",
            spec.disabled_reason or "workflow is disabled",
            requested={"workflow_id": workflow_id},
        )
    try:
        resolved = resolve_workflow(workflow_id, context_id, None)
    except ValueError as exc:
        raise UnsupportedCapabilityError(
            "WORKFLOW_CONTEXT_UNSUPPORTED",
            str(exc),
            requested={"workflow_id": workflow_id, "context_id": context_id},
        ) from exc
    context_spec = resolve_context_spec(context_id)
    metadata = dict(context_metadata)
    provenance = {
        "context_spec": context_spec.to_dict(),
        "runtime_metadata": metadata,
    }
    data = DataIntakeContract(
        source_kind=context_spec.kind,
        source_provider_or_adapter="start.runtime.contexts",
        source_reference=context_spec.provenance,
        context_id=context_id,
        provenance_fingerprint=_fingerprint(provenance),
        schema_summary=tuple(context_spec.badges),
        dimensions={
            "samples": metadata.get("actual_samples"),
            "features": metadata.get("actual_features"),
            "assets": metadata.get("actual_assets"),
            "periods": metadata.get("actual_periods"),
        },
        observation_semantics=context_spec.description,
        load_mode="canonical_context_factory",
        provenance_metadata=provenance,
        validation_status="RESOLVED",
    )
    candidate_ids = tuple(resolved.candidate_test_ids)
    applicable_ids = tuple((resolved_metadata or {}).get("applicable_test_ids", resolved.applicable_test_ids))
    problem = ProblemContract(
        domains=(spec.category,),
        workflow_purpose=spec.label,
        lifecycle="deterministic_control_plane_execution",
        objective=f"execute canonical workflow {workflow_id}",
        task_or_risk_type=spec.engine_kind.value,
        decision_impact="deterministic registered workflow result",
        criterion_provenance=("canonical workflow registry", "registered TestSpec contracts"),
        validation_design=f"{len(candidate_ids)} candidate surfaces; {len(applicable_ids)} applicable",
        candidate_methods=candidate_ids,
        constraints=("deterministic-only", "LLM calls=0", "LLM numeric authority=0"),
        decision_metrics=applicable_ids,
        output_structure=OutputStructure.DOMAIN_ANALYTICAL_OUTPUT,
        target_specification=(str(metadata.get("actual_target")),) if metadata.get("actual_target") not in (None, "N/A") else (),
        model_input_contract="registered execution context",
        domain_payloads={spec.category: {"engine_kind": spec.engine_kind.value}},
    )
    capabilities = tuple(
        PlannedCapability(
            f"test.{test_id}",
            CapabilityState.APPLICABLE if test_id in applicable_ids else CapabilityState.NOT_APPLICABLE,
            "start.runtime.execution.CanonicalExecutionService",
            "resolved by workflow/context applicability",
            test_id,
        )
        for test_id in candidate_ids
    )
    plan = CapabilityPlan(
        route_id=workflow_id,
        review_mode="deterministic_only",
        stages=_base_stage_plan(agent_assisted=False, grounding=False, human=False),
        capabilities=capabilities,
        registered_test_ids=applicable_ids,
        artifact_families=tuple(sorted({test_id.split(".", 1)[0] for test_id in applicable_ids})),
        environment_projection=_environment_projection(agent_assisted=False),
    )
    return WorkflowCoherenceEnvelope(run_id=run_id, data_intake=data, problem=problem, capability_plan=plan)


def build_run_outcome_capsule(
    *,
    envelope: WorkflowCoherenceEnvelope,
    events: list[Any],
    presentation_state: Any,
) -> RunOutcomeCapsule:
    """Build one runtime-derived outcome from canonical observer state and events."""
    human: list[str] = []
    agents: list[str] = []
    deterministic: list[str] = []
    evidence: list[str] = []
    for event in events:
        kind = str(getattr(getattr(event, "kind", ""), "value", getattr(event, "kind", "")))
        payload = dict(getattr(event, "payload", {}) or {})
        if kind == "HUMAN_ACTION":
            action = str(payload.get("action", "ACTION"))
            checkpoint = str(payload.get("checkpoint", getattr(event, "semantic_checkpoint_id", "")))
            human.append(f"{action}@{checkpoint}")
        elif kind == "AGENT_DECISION_RECORDED":
            agent = str(payload.get("agent_identity", "agent"))
            recommendation = str(payload.get("recommendation", "decision recorded"))
            agents.append(f"{agent}: {recommendation}")
        elif kind == "ENGINE_COMPLETED":
            engine = str(payload.get("engine_name") or "deterministic engine")
            detail = str(payload.get("details") or "completed")
            deterministic.append(f"{engine}: {detail}")
        elif kind == "DETERMINISTIC_EXECUTION_SUMMARIZED":
            engines = [str(item) for item in payload.get("engine_ids", ())]
            operations = [str(item) for item in payload.get("operations", ())]
            outputs = [str(item) for item in payload.get("authoritative_outputs", ())]
            evidence_refs = [str(item) for item in payload.get("evidence_record_references", ())]
            engine_text = ", ".join(engines) or "deterministic engine"
            operation_text = "; ".join(operations) or "registered execution"
            deterministic.append(
                f"{engine_text} completed {operation_text}; "
                f"{len(outputs)} authoritative output(s); "
                f"{len(evidence_refs)} EvidenceRecord reference(s)"
            )
        elif kind == "EVIDENCE_COMMITTED":
            ev_id = str(payload.get("evidence_id", ""))
            if ev_id and ev_id not in evidence:
                evidence.append(ev_id)
    agent_stage = envelope.capability_plan.state_for("AGENT_DECISIONS")
    human_stage = envelope.capability_plan.state_for("HUMAN_DECISIONS")
    grounding_stage = envelope.capability_plan.state_for("GROUNDING")
    if agent_stage is CapabilityState.NOT_APPLICABLE:
        agent_contribution: tuple[str, ...] | str = CapabilityState.NOT_APPLICABLE.value
    else:
        agent_contribution = tuple(agents) if agents else ("APPLICABLE — no consolidated decision trace emitted",)
    grounding = (
        CapabilityState.NOT_APPLICABLE.value
        if grounding_stage is CapabilityState.NOT_APPLICABLE
        else str(getattr(presentation_state, "grounding_state", None) or "NOT_EVALUATED")
    )
    governance = str(getattr(presentation_state, "governance_disposition", None) or "NOT_EVALUATED")
    policy = str(getattr(presentation_state, "policy_decision", None) or "NOT_EVALUATED")
    unresolved_count = int(getattr(presentation_state, "governance_unresolved_count", 0) or 0)
    validation_failures = int(getattr(presentation_state, "governance_validation_failures", 0) or 0)
    governance_conditions = tuple(
        str(item)
        for item in getattr(presentation_state, "governance_conditions", ()) or ()
        if str(item).strip()
    )
    remaining: list[str] = []
    if human_stage is CapabilityState.APPLICABLE and not human:
        remaining.append("human decision stage applicable but no decision event was recorded")
    if agent_stage is CapabilityState.APPLICABLE and not agents:
        remaining.append("agent decision stage applicable but no consolidated trace was recorded")
    if grounding == "REJECTED":
        remaining.append("agent interpretation rejected; canonical deterministic evidence retained")
    if governance in {"NOT_EVALUATED", "UNRESOLVED", "DENY"}:
        remaining.append(f"governance={governance}")
    if policy in {"NOT_EVALUATED", "DENY"}:
        remaining.append(f"policy={policy}")
    if unresolved_count:
        remaining.append(f"{unresolved_count} unresolved governance item(s)")
    if validation_failures:
        remaining.append(f"{validation_failures} validation failure(s) require disposition")
    remaining.extend(governance_conditions)
    if governance == "ACCEPT_WITH_CONDITIONS" and not (
        unresolved_count or validation_failures or governance_conditions
    ):
        remaining.append(
            "GOVERNANCE_CONDITIONS_PRESENT — condition details unavailable in canonical presentation state"
        )
    if governance == "REMEDIATION_REQUIRED" and not (
        unresolved_count or validation_failures or governance_conditions
    ):
        remaining.append(
            "REMEDIATION_REQUIRED — remediation details unavailable in canonical presentation state"
        )
    remaining.extend(envelope.capability_plan.unsupported_reasons)
    domains = tuple(envelope.problem.domains)
    if envelope.capability_plan.route_id == "interactive_predictive_review":
        achieved_result = (
            "Predictive model development and evidence-native evaluation completed with "
            f"{len(evidence)} canonical EvidenceRecord reference(s)."
        )
    elif envelope.capability_plan.route_id == "unified_domain_review":
        if len(domains) > 1:
            scope = f"Cross-domain {' + '.join(domains)} review"
        elif domains == ("market",):
            scope = "Portfolio and market-risk analysis"
        elif domains == ("treasury",):
            scope = "Treasury and short-rate diagnostics"
        else:
            scope = f"{domains[0].title() if domains else 'Domain'} review"
        achieved_result = (
            f"{scope} completed with {len(evidence)} canonical EvidenceRecord reference(s)."
        )
    else:
        achieved_result = (
            f"Canonical deterministic workflow {envelope.capability_plan.route_id} completed with "
            f"{len(evidence)} canonical EvidenceRecord reference(s)."
        )
    capsule = RunOutcomeCapsule(
        run_id=envelope.run_id,
        started_with=(
            f"{envelope.data_intake.context_id} "
            f"({envelope.data_intake.observation_semantics}; "
            f"validation={envelope.problem.validation_design})"
        ),
        objective=envelope.problem.objective,
        alternatives=envelope.problem.candidate_methods,
        human_decisions=(
            tuple(human)
            if human
            else (
                (CapabilityState.NOT_APPLICABLE.value,)
                if human_stage is CapabilityState.NOT_APPLICABLE
                else ("APPLICABLE — no human decision event recorded",)
            )
        ),
        agent_contribution=agent_contribution,
        deterministic_results=tuple(dict.fromkeys(deterministic)) or ("canonical deterministic execution completed",),
        evidence_record_references=tuple(evidence),
        grounding=grounding,
        governance_policy=f"governance={governance}; policy={policy}",
        achieved_result=achieved_result,
        remaining_conditions=(
            tuple(dict.fromkeys(remaining))
            if remaining
            else ("NO_REMAINING_CONDITIONS",)
        ),
    )
    envelope.outcome = capsule
    return capsule


@dataclass(frozen=True)
class ContractCase:
    case_id: str
    route: str
    dimensions: dict[str, str]
    support: SupportStatus
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def enumerate_contract_cases() -> list[ContractCase]:
    """Enumerate registry/configuration contracts without scientific execution."""
    from start.modeling.architecture_registry import family_available, list_families
    from start.review.architecture import PredictiveTechnology, ReviewDomain, ReviewLifecycle
    from start.runtime.workflows import get_canonical_workflow_specs

    cases: list[ContractCase] = []
    for workflow_id, spec in sorted(get_canonical_workflow_specs().items()):
        for context_id in spec.compatible_contexts:
            cases.append(
                ContractCase(
                    case_id=f"runtime:{workflow_id}:{context_id}",
                    route="canonical_runtime",
                    dimensions={
                        "workflow": workflow_id,
                        "context": context_id,
                        "engine": spec.engine_kind.value,
                        "agent_mode": "deterministic_only",
                    },
                    support=SupportStatus.SUPPORTED if spec.enabled else SupportStatus.NOT_SUPPORTED,
                    reason="enabled canonical workflow" if spec.enabled else str(spec.disabled_reason),
                )
            )
    for modality in ("tabular", "sequence", "vision"):
        for family in list_families(modality):
            available, reason = family_available(family)
            cases.append(
                ContractCase(
                    case_id=f"architecture:{modality}:{family}",
                    route="predictive_architecture",
                    dimensions={
                        "domain": ReviewDomain.PREDICTIVE.value,
                        "technology": PredictiveTechnology.DEEP_LEARNING.value,
                        "modality": modality,
                        "model_family": family,
                    },
                    support=SupportStatus.SUPPORTED if available else SupportStatus.DEFERRED,
                    reason="registered implemented family" if available else reason,
                )
            )
    domain_values = [domain.value for domain in ReviewDomain]
    for size in (1, 2, 3):
        for selected in combinations(domain_values, size):
            mode = "single_domain" if size == 1 else "cross_domain"
            cases.append(
                ContractCase(
                    case_id=f"review:{'+'.join(selected)}",
                    route="unified_domain_review",
                    dimensions={
                        "mode": mode,
                        "domains": "+".join(selected),
                        "lifecycle": ReviewLifecycle.INITIAL_VALIDATION.value,
                        "agent_mode": "deterministic_or_llm",
                    },
                    support=(
                        SupportStatus.SUPPORTED
                        if size == 1
                        else SupportStatus.SUPPORTED_WITH_CONSTRAINTS
                    ),
                    reason=(
                        "registered atomic domain"
                        if size == 1
                        else "composed registered surfaces; no new cross-domain science is invented"
                    ),
                )
            )
    cases.extend(
        [
            ContractCase(
                "output:single_target",
                "predictive_output",
                {"output_structure": "single_target", "task": "classification_or_regression"},
                SupportStatus.SUPPORTED,
                "interactive predictive route accepts one explicit target",
            ),
            ContractCase(
                "output:multilabel_direct_estimator",
                "predictive_output",
                {"output_structure": "multilabel", "task": "multilabel_classification"},
                SupportStatus.SUPPORTED_WITH_CONSTRAINTS,
                "TabularDLClassifier supports it; interactive review target selection is not wired",
            ),
            ContractCase(
                "output:multioutput_regression_direct_estimator",
                "predictive_output",
                {"output_structure": "multi_output_regression", "task": "regression"},
                SupportStatus.SUPPORTED_WITH_CONSTRAINTS,
                "TabularDLClassifier supports matrix targets; end-to-end review route is not wired",
            ),
            ContractCase(
                "output:sequence_output",
                "predictive_output",
                {"output_structure": "sequence_output", "task": "sequence_to_sequence"},
                SupportStatus.NOT_SUPPORTED,
                "registered recurrent route provides whole-sequence classification only",
            ),
        ]
    )
    return cases


def deterministic_pairwise_cases(cases: list[ContractCase]) -> list[ContractCase]:
    """Greedy deterministic cover over pairs that exist in valid enumerated cases."""
    supported = [
        case
        for case in cases
        if case.support in {SupportStatus.SUPPORTED, SupportStatus.SUPPORTED_WITH_CONSTRAINTS}
    ]

    def pairs(case: ContractCase) -> set[tuple[str, str, str, str]]:
        items = sorted(case.dimensions.items())
        return {(a, av, b, bv) for (a, av), (b, bv) in combinations(items, 2)}

    universe: set[tuple[str, str, str, str]] = set()
    case_pairs: dict[str, set[tuple[str, str, str, str]]] = {}
    for case in supported:
        value = pairs(case)
        case_pairs[case.case_id] = value
        universe.update(value)
    uncovered = set(universe)
    selected: list[ContractCase] = []
    remaining = list(supported)
    while uncovered and remaining:
        best = max(
            remaining,
            key=lambda case: (len(case_pairs[case.case_id] & uncovered), -supported.index(case)),
        )
        gain = case_pairs[best.case_id] & uncovered
        if not gain:
            break
        selected.append(best)
        uncovered -= gain
        remaining.remove(best)
    return selected


def capability_inventory() -> dict[str, Any]:
    """Return selector-to-owner inventory from the actual source registries."""
    from start.data.providers.registry import list_provider_adapters
    from start.data.selection import WIZARD_OPTIONS
    from start.modeling.architecture_registry import family_available, list_families
    from start.modeling.models import MODEL_CHOICES
    from start.registry import list_tests
    from start.review.architecture import ReviewDomain, ReviewLifecycle, ReviewMode
    from start.runtime.contexts import get_canonical_context_specs
    from start.runtime.workflows import get_canonical_workflow_specs

    tests = list_tests(include_recommender=True)
    workflows = get_canonical_workflow_specs()
    architecture_rows: list[dict[str, Any]] = []
    for modality in ("tabular", "sequence", "vision"):
        for family in list_families(modality):
            available, reason = family_available(family)
            architecture_rows.append(
                {
                    "family": family,
                    "modality": modality,
                    "status": (
                        SupportStatus.SUPPORTED.value if available else SupportStatus.DEFERRED.value
                    ),
                    "reason": reason or "registered implemented architecture",
                    "selector_owner": "start.modeling.architecture_registry",
                    "execution_owner": "start.modeling.model_execution / modality-specific estimator",
                }
            )
    traditional_rows = []
    optional = {"xgboost", "lightgbm", "catboost"}
    unsupported = {"random_rotation_forest"}
    for model in MODEL_CHOICES:
        if model in unsupported:
            status = SupportStatus.NOT_SUPPORTED
            reason = "selector token exists but resolve_model fails closed"
        elif model in optional:
            status = SupportStatus.SUPPORTED_WITH_CONSTRAINTS
            reason = "optional package; fail-closed when unavailable"
        else:
            status = SupportStatus.SUPPORTED
            reason = "registered model resolution path"
        traditional_rows.append(
            {
                "model": model,
                "status": status.value,
                "reason": reason,
                "selector_owner": "start.modeling.models.MODEL_CHOICES",
                "execution_owner": "start.modeling.models.resolve_model",
            }
        )
    return {
        "review_modes": [mode.value for mode in ReviewMode],
        "domains": [domain.value for domain in ReviewDomain],
        "lifecycles": [lifecycle.value for lifecycle in ReviewLifecycle],
        "workflows": [
            {
                "workflow_id": key,
                "status": (
                    SupportStatus.SUPPORTED.value if spec.enabled else SupportStatus.NOT_SUPPORTED.value
                ),
                "reason": "enabled" if spec.enabled else spec.disabled_reason,
                "engine_kind": spec.engine_kind.value,
                "compatible_contexts": list(spec.compatible_contexts),
                "candidate_test_ids": list(spec.candidate_test_ids),
                "selector_owner": "start.runtime.workflows",
                "execution_owner": "start.runtime.execution.CanonicalExecutionService",
            }
            for key, spec in sorted(workflows.items())
        ],
        "contexts": [
            {
                **spec.to_dict(),
                "status": SupportStatus.SUPPORTED.value,
                "selector_owner": "start.runtime.contexts",
            }
            for spec in get_canonical_context_specs()
        ],
        "dataset_selection": [
            {
                "key": key,
                "label": label,
                "status": SupportStatus.SUPPORTED_WITH_CONSTRAINTS.value,
                "selector_owner": "start.data.selection.WIZARD_OPTIONS",
                "execution_owner": "start.data.selection.resolve_wizard_choice",
            }
            for key, label in WIZARD_OPTIONS
        ],
        "provider_adapters": list_provider_adapters(),
        "architectures": architecture_rows,
        "models": traditional_rows,
        "registered_tests": [
            {
                "test_id": spec.test_id,
                "family": spec.family,
                "context_type": spec.context_type,
                "requires": list(spec.requires),
                "status": SupportStatus.SUPPORTED.value,
                "execution_owner": f"{spec.fn.__module__}.{spec.fn.__name__}",
            }
            for spec in tests
        ],
        "target_output_modes": [
            {
                "mode": "single_target",
                "status": SupportStatus.SUPPORTED.value,
                "scope": "interactive review and canonical predictive workflows",
            },
            {
                "mode": "multilabel",
                "status": SupportStatus.SUPPORTED_WITH_CONSTRAINTS.value,
                "scope": "direct TabularDLClassifier API only; interactive review not wired",
            },
            {
                "mode": "multi_output_regression",
                "status": SupportStatus.SUPPORTED_WITH_CONSTRAINTS.value,
                "scope": "direct TabularDLClassifier API only; interactive review not wired",
            },
            {
                "mode": "sequence_output",
                "status": SupportStatus.NOT_SUPPORTED.value,
                "scope": "no registered end-to-end sequence-output route",
            },
        ],
        "cross_domain": {
            "status": SupportStatus.SUPPORTED_WITH_CONSTRAINTS.value,
            "selector_owner": "start.review.architecture.parse_domain_selection",
            "routing_owner": "start.review.applicability.applicable_tests",
            "execution_owner": "start.review.executor.run_market_treasury_review",
            "reason": "composes registered domain contexts/tests; no standalone canonical workflow ID",
        },
        "treasury": {
            "status": SupportStatus.SUPPORTED_WITH_CONSTRAINTS.value,
            "registered_surface_count": len([t for t in tests if t.context_type == "short_rate"]),
            "reason": "unified review path only; no standalone canonical workflow ID",
        },
    }


__all__ = [
    "AgentDecisionTrace",
    "CapabilityPlan",
    "CapabilityState",
    "COHERENCE_STAGE_ORDER",
    "ContractCase",
    "DataIntakeContract",
    "DeterministicExecutionSummary",
    "OutputStructure",
    "PlannedCapability",
    "ProblemContract",
    "RunOutcomeCapsule",
    "SupportStatus",
    "UnsupportedCapabilityError",
    "WorkflowCoherenceEnvelope",
    "build_canonical_runtime_coherence_envelope",
    "build_predictive_coherence_envelope",
    "canonical_predictive_explainability_identity",
    "build_run_outcome_capsule",
    "build_unified_review_coherence_envelope",
    "capability_inventory",
    "deterministic_pairwise_cases",
    "enumerate_contract_cases",
    "resolve_predictive_output_structure",
    "synchronize_predictive_explainability",
]
