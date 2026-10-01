"""Terminal-native observability for the unified StART review.

This module is deliberately a presentation adapter.  It never computes a
scientific result and it never promotes presentation events to EvidenceRecords.
Every card accepts an already-created canonical object (EvidenceRecord,
PolicyDecision, TraceRecord, structured grounding result, or LangGraph state).
"""

from __future__ import annotations

import base64
import json
import os
import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from rich.columns import Columns
from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from start.core.hashing import hash_obj
from start.core.schemas import EvidenceRecord


class PresentationEventKind(StrEnum):
    SESSION_STARTED = "SESSION_STARTED"
    DATA_SOURCE_RESOLVED = "DATA_SOURCE_RESOLVED"
    PROBLEM_CONTRACT_RESOLVED = "PROBLEM_CONTRACT_RESOLVED"
    CAPABILITY_PLAN_RESOLVED = "CAPABILITY_PLAN_RESOLVED"
    HUMAN_ACTION = "HUMAN_ACTION"
    GRAPH_TRANSITION = "GRAPH_TRANSITION"
    AGENT_ACTIVATED = "AGENT_ACTIVATED"
    ENGINE_STARTED = "ENGINE_STARTED"
    ENGINE_COMPLETED = "ENGINE_COMPLETED"
    EVIDENCE_COMMITTED = "EVIDENCE_COMMITTED"
    GROUNDING_ACCEPTED = "GROUNDING_ACCEPTED"
    GROUNDING_REJECTED = "GROUNDING_REJECTED"
    CHALLENGE_CREATED = "CHALLENGE_CREATED"
    DIAGNOSTIC_STARTED = "DIAGNOSTIC_STARTED"
    CHALLENGE_RESOLVED = "CHALLENGE_RESOLVED"
    GOVERNANCE_EVALUATED = "GOVERNANCE_EVALUATED"
    OPA_EVALUATED = "OPA_EVALUATED"
    TRACE_READY = "TRACE_READY"
    TRACE_SUMMARY = "TRACE_READY"  # compatibility alias
    ATTESTATION_SEALED = "ATTESTATION_SEALED"
    ATTESTATION_WITHHELD = "ATTESTATION_WITHHELD"
    ARTIFACT_AVAILABLE = "ARTIFACT_AVAILABLE"
    AGENT_DECISION_RECORDED = "AGENT_DECISION_RECORDED"
    DETERMINISTIC_EXECUTION_SUMMARIZED = "DETERMINISTIC_EXECUTION_SUMMARIZED"
    OUTCOME_READY = "OUTCOME_READY"
    SESSION_COMPLETED = "SESSION_COMPLETED"


STAGES: tuple[str, ...] = (
    "DATA",
    "OBJECTIVE",
    "PLAN",
    "HUMAN",
    "LANGGRAPH",
    "AGENT",
    "ENGINE",
    "EVIDENCE",
    "GROUNDING",
    "GOVERNANCE",
    "OPA",
    "TRACE",
    "ATTESTATION",
    "OUTCOME",
)

CONTROL_PLANE_STAGES: tuple[str, ...] = (
    "DATA",
    "OBJECTIVE",
    "PLAN",
    "ENGINE",
    "EVIDENCE",
    "GOVERNANCE",
    "OPA",
    "TRACE",
    "ATTESTATION",
    "OUTCOME",
)

HIGH_VALUE_EVENTS = frozenset(PresentationEventKind)


@dataclass(frozen=True)
class PresentationEvent:
    sequence: int
    kind: PresentationEventKind
    run_id: str
    semantic_checkpoint_id: str
    stage: str
    status: str
    source_component: str
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["kind"] = self.kind.value
        return data


@dataclass
class ReviewPresentationState:
    run_id: str
    mode: str = "LIVE REVIEW"
    stage_status: dict[str, str] = field(
        default_factory=lambda: {stage: "PENDING" for stage in STAGES}
    )
    active_stage: str | None = None
    completed_stages: list[str] = field(default_factory=list)
    active_stage_history: list[str] = field(default_factory=list)
    current_checkpoint: str | None = None
    last_human_action: str | None = None
    current_agent: str | None = None
    current_engine: str | None = None
    evidence_count: int = 0
    grounded_claims: int = 0
    ungrounded_claims: int = 0
    governance_disposition: str | None = None
    governance_unresolved_count: int = 0
    governance_validation_failures: int = 0
    governance_conditions: list[str] = field(default_factory=list)
    policy_decision: str | None = None
    policy_engine: str | None = None
    trace_id: str | None = None
    merkle_root: str | None = None
    grounding_state: str | None = None
    trace_state: str | None = None
    attestation_state: str | None = None
    llm_calls: int = 0
    llm_tokens: int = 0
    data_context_id: str | None = None
    objective: str | None = None
    capability_route: str | None = None
    outcome_state: str | None = None


@dataclass(frozen=True)
class PresentationPacing:
    """Display-only dwell policy. Disabled unless explicitly requested."""

    enabled: bool = False
    scale: float = 1.0

    @classmethod
    def from_environment(cls) -> PresentationPacing:
        enabled = presentation_mode_enabled()
        try:
            scale = max(0.0, float(os.environ.get("START_PRESENTATION_PACING_SCALE", "1")))
        except ValueError:
            scale = 1.0
        return cls(enabled=enabled, scale=scale)

    def seconds_for(self, kind: PresentationEventKind) -> float:
        if not self.enabled:
            return 0.0
        seconds = {
            PresentationEventKind.GRAPH_TRANSITION: 0.5,
            PresentationEventKind.AGENT_ACTIVATED: 0.7,
            PresentationEventKind.ENGINE_STARTED: 0.7,
            PresentationEventKind.EVIDENCE_COMMITTED: 0.6,
            PresentationEventKind.CHALLENGE_RESOLVED: 0.9,
            PresentationEventKind.GROUNDING_REJECTED: 1.7,
            PresentationEventKind.OPA_EVALUATED: 1.2,
            PresentationEventKind.TRACE_READY: 1.4,
            PresentationEventKind.SESSION_COMPLETED: 7.0,
        }.get(kind, 0.0)
        return seconds * self.scale


def presentation_mode_enabled() -> bool:
    """Return whether viewer-facing presentation controls are active."""
    return os.environ.get("START_PRESENTATION_MODE", "").lower() in {"1", "true", "yes"}


def predictive_runtime_context_title(*, model_development: bool) -> str:
    """Return a generic predictive product title with no demo-flight language."""
    activity = "MODEL DEVELOPMENT" if model_development else "DIAGNOSTICS"
    return f"PREDICTIVE {activity} + EVIDENCE-NATIVE REVIEW"


def workflow_runtime_context_title(domains: Iterable[Any]) -> str:
    """Build a generic context title from canonical review-domain labels."""
    from start.review.architecture import DOMAIN_LABELS, ReviewDomain

    labels = [
        DOMAIN_LABELS[domain if isinstance(domain, ReviewDomain) else ReviewDomain(str(domain))]
        for domain in domains
    ]
    if not labels:
        return "REVIEW"
    prefix = "CROSS-DOMAIN " if len(labels) > 1 else ""
    return prefix + " + ".join(labels) + " REVIEW"


def predictive_review_path(modality: str) -> str:
    """Describe only the predictive input contract that applies to this run."""
    normalized = str(modality or "tabular").lower().replace("-", "_")
    contracts = {
        "temporal": "TEMPORAL CONTRACT",
        "temporal_sequence": "TEMPORAL CONTRACT",
        "sequence": "TEMPORAL CONTRACT",
        "tabular": "TABULAR INPUT CONTRACT",
        "image": "IMAGE INPUT CONTRACT",
        "vision": "IMAGE INPUT CONTRACT",
        "recommender": "RECOMMENDER CONTEXT CONTRACT",
    }
    contract = contracts.get(normalized, f"{normalized.upper()} INPUT CONTRACT")
    return (
        f"DATA → {contract} → ARCHITECTURE → TRAIN/EVALUATE → PERFORMANCE → "
        "ATTRIBUTION → ROBUSTNESS → EVIDENCE → GOVERNANCE"
    )


def treasury_review_path(registered_surfaces: Iterable[str]) -> str:
    """Project the registered short-rate surfaces into a compact workflow path."""
    labels: list[str] = []
    for surface in registered_surfaces:
        value = str(surface).lower()
        if ".cev_" in f".{value}.":
            labels.append("CEV")
        elif ".stanton_" in f".{value}.":
            labels.append("STANTON")
    methods = list(dict.fromkeys(labels))
    if not methods:
        return ""
    return "SHORT-RATE CONTEXT → " + " → ".join(methods) + " → VALIDATION → EVIDENCE → GOVERNANCE"


@dataclass(frozen=True)
class FlightAPresentationContext:
    """Canonical viewer-facing interpretation of Flight A runtime inputs.

    The intake DataFrame can be useful review metadata for sequence workflows,
    but it is never allowed to masquerade as the rank-3 model input.
    """

    modality: str
    review_metadata_rows: int
    review_metadata_fields: int
    model_sample_count: int
    model_feature_count: int
    timesteps: int | None
    sample_structure: str
    model_input_contract: str
    split_description: str
    architecture_alternatives: tuple[str, ...]

    @property
    def review_metadata_display(self) -> str:
        return (
            "REVIEW METADATA — NOT MODEL INPUT: "
            f"{self.review_metadata_rows:,} rows × {self.review_metadata_fields} fields"
        )

    @property
    def model_input_display(self) -> str:
        return f"MODEL INPUT: {self.sample_structure}; {self.model_input_contract}"

    @property
    def discovery_trace_input(self) -> str:
        if self.modality == "temporal_sequence":
            return f"{self.review_metadata_display}; {self.model_input_display}"
        return self.model_input_display

    def architecture_evidence(self, task_type: str) -> list[str]:
        return [
            self.model_input_display,
            f"split: {self.split_description}",
            f"{task_type.replace('_', ' ')} task",
        ]

    def architecture_trace_input(self, family: str, activation: str) -> str:
        return (
            f"user choice {family}+{activation}; {self.model_input_display}; "
            f"modality={self.modality}; split={self.split_description}"
        )


def build_flight_a_presentation_context(
    *,
    df: Any,
    sequence_bundle: Any | None,
    selected_architecture: str | None,
    configured_split: str,
) -> FlightAPresentationContext:
    """Build the single presentation source for Flight A data/model facts."""
    from start.modeling.architecture_registry import family_available, list_families

    modality = "sequence" if sequence_bundle is not None else "tabular"
    candidates = [family for family in list_families(modality) if family_available(family)[0]]
    if selected_architecture and selected_architecture not in candidates:
        available, _ = family_available(selected_architecture)
        if available:
            candidates.insert(0, selected_architecture)
    alternatives = tuple(dict.fromkeys(candidates))

    if sequence_bundle is not None:
        train_count = len(sequence_bundle.X_train)
        test_count = len(sequence_bundle.X_test)
        oos_count = len(sequence_bundle.X_oos)
        total = train_count + test_count + oos_count
        timesteps = int(sequence_bundle.timesteps)
        n_features = int(sequence_bundle.n_features)
        return FlightAPresentationContext(
            modality="temporal_sequence",
            review_metadata_rows=int(len(df)),
            review_metadata_fields=int(df.shape[1]),
            model_sample_count=total,
            model_feature_count=n_features,
            timesteps=timesteps,
            sample_structure=(
                f"{total:,} independent sequences × {timesteps} timesteps × "
                f"{n_features} model features"
            ),
            model_input_contract=f"({total},{timesteps},{n_features}) (N,T,F)",
            split_description=(
                "order-preserving contiguous sequence holdout "
                f"(train/test/OOS={train_count}/{test_count}/{oos_count}; "
                "OOS is the most recent block; independent sequences, not forecasting)"
            ),
            architecture_alternatives=alternatives,
        )

    target_adjustment = 1 if getattr(df, "shape", (0, 0))[1] else 0
    n_features = max(0, int(df.shape[1]) - target_adjustment)
    return FlightAPresentationContext(
        modality="tabular",
        review_metadata_rows=int(len(df)),
        review_metadata_fields=int(df.shape[1]),
        model_sample_count=int(len(df)),
        model_feature_count=n_features,
        timesteps=None,
        sample_structure=f"{len(df):,} rows × {n_features} model features",
        model_input_contract=f"({len(df)},{n_features}) rank-2 tabular input",
        split_description=configured_split,
        architecture_alternatives=alternatives,
    )


@dataclass(frozen=True)
class AuthorityPresentation:
    reviewer_reasoning: str
    hosted_execution: str
    provider: str


def resolve_authority_presentation(
    *,
    reviewer_mode: str,
    backend_mode: str,
    provider: str,
    provider_status: str = "",
    deterministic_only: bool = False,
) -> AuthorityPresentation:
    """Resolve provider-aware authority labels without changing execution."""
    normalized_provider = (provider or "none").strip().lower()
    normalized_backend = (backend_mode or "none").strip().lower()
    normalized_reviewer = (reviewer_mode or "deterministic").strip().lower()
    normalized_status = (provider_status or "").strip().upper()
    if deterministic_only or normalized_reviewer == "deterministic" or normalized_backend == "none":
        return AuthorityPresentation(
            reviewer_reasoning="DETERMINISTIC_ONLY",
            hosted_execution="DISABLED",
            provider="none",
        )
    if normalized_provider == "offline_demo_twin" or normalized_backend == "offline":
        return AuthorityPresentation(
            reviewer_reasoning="OFFLINE_DEMO_TWIN",
            hosted_execution="DISABLED",
            provider="offline_demo_twin",
        )
    if normalized_status and normalized_status not in {"CONNECTED", "AVAILABLE"}:
        return AuthorityPresentation(
            reviewer_reasoning="DETERMINISTIC_FALLBACK",
            hosted_execution="UNAVAILABLE",
            provider=normalized_provider or "configured_provider",
        )
    return AuthorityPresentation(
        reviewer_reasoning="HOSTED_PROVIDER",
        hosted_execution="ENABLED",
        provider=normalized_provider or "configured_provider",
    )


def _presentation_signal(action: str, payload: dict[str, Any]) -> str:
    """Return an invisible controller signal for presentation-only behavior."""
    body = json.dumps({"action": action, **payload}, separators=(",", ":")).encode("utf-8")
    encoded = base64.urlsafe_b64encode(body).decode("ascii").rstrip("=")
    return f"\x1b]777;StART={encoded}\x07"


def viewer_suppression_signal(*, enabled: bool) -> str:
    """Return a controller-only signal that hides bytes from the live viewer."""
    return _presentation_signal("viewer_suppression", {"enabled": bool(enabled)})


class EvidenceDisplayPolicy:
    """Deterministic, session-specific selection of visible commit cards."""

    _A_EXACT = {
        "discovery.task_inference",
        "split.plan",
        "deep_learning.performance_diagnostics",
        "deep_learning.explainability_diagnostics",
        "deep_learning.robustness_diagnostics",
    }
    _B_EXACT = {
        "portfolio.hierarchical_risk_parity",
        "portfolio.risk_statistics.euler_decomposition",
        "attribution.factor_return_estimation",
        "attribution.return_attribution",
        "traded_risk.var_kupiec_pof",
        "validation.var_size_power",
        "covariance.ledoit_wolf_shrinkage",
        "validation.regem_structural",
        "scenario.linear_return",
        "scenario.reverse_stress",
    }

    def __init__(self, session_kind: str) -> None:
        self.session_kind = session_kind.upper()

    def category(self, test_id: str) -> str:
        prefix = test_id.split(".", 1)[0]
        return {
            "deep_learning": "model_validation",
            "split": "data_contract",
            "portfolio": "portfolio",
            "attribution": "attribution",
            "traded_risk": "market_risk",
            "covariance": "covariance",
            "scenario": "scenario",
            "validation": "validation",
            "diagnostic": "challenge_diagnostic",
        }.get(prefix, "other")

    def visible(self, test_id: str, status: str) -> bool:
        if status.lower() in {"fail", "error", "warn"} or test_id.startswith("diagnostic."):
            return True
        if self.session_kind == "A":
            return test_id in self._A_EXACT or "governance" in test_id
        if self.session_kind in {"B", "C"}:
            return test_id in self._B_EXACT or "governance" in test_id
        return False


class TerminalReviewObserver:
    """Collect and render truthful high-value review events.

    Rendering is append-only rather than a constantly-cleared live display, so
    scrollback remains an audit surface and non-colour terminals remain usable.
    """

    def __init__(
        self,
        run_id: str,
        *,
        console: Any,
        domains: Iterable[Any] = (),
        session_kind: str = "",
        control_plane: bool = False,
        max_width: int = 132,
        pacing: PresentationPacing | None = None,
    ) -> None:
        self.console = console
        self.state = ReviewPresentationState(run_id=run_id)
        self.session_kind = session_kind.upper()
        self.control_plane = control_plane
        self.max_width = max(80, min(int(max_width), 140))
        self.pacing = pacing or PresentationPacing.from_environment()
        self.evidence_policy = EvidenceDisplayPolicy(self.session_kind)
        self.flight_a_context: FlightAPresentationContext | None = None
        self.coherence_envelope: Any | None = None
        self.outcome_capsule: Any | None = None
        self._evidence_categories: Counter[str] = Counter()
        self._visible_evidence_count = 0
        self._last_spine_signature: tuple[Any, ...] | None = None
        self.events: list[PresentationEvent] = []
        if self.control_plane:
            for stage in ("HUMAN", "LANGGRAPH", "AGENT", "GROUNDING"):
                self.state.stage_status[stage] = "NOT_APPLICABLE"

    def bind_flight_a_context(self, context: FlightAPresentationContext) -> None:
        """Bind the canonical Flight A facts used by later presentation cards."""
        self.flight_a_context = context

    def bind_coherence_envelope(self, envelope: Any) -> None:
        """Bind the canonical cross-surface contract used by terminal/report export."""
        if str(getattr(envelope, "run_id", "")) != self.state.run_id:
            raise ValueError("coherence envelope run_id must match observer run_id")
        self.coherence_envelope = envelope

    def _publish_problem_contract(self, *, checkpoint: str) -> None:
        envelope = self.coherence_envelope
        if envelope is None:
            raise RuntimeError("bind_coherence_envelope() must be called before publishing")
        problem = envelope.problem
        problem_grid = Table.grid(padding=(0, 2))
        problem_grid.add_column(style="bold magenta", no_wrap=True)
        problem_grid.add_column()
        problem_grid.add_row("Domain(s)", " + ".join(problem.domains))
        problem_grid.add_row("Purpose", problem.workflow_purpose)
        problem_grid.add_row("Objective", problem.objective)
        problem_grid.add_row("Task / risk", problem.task_or_risk_type)
        problem_grid.add_row("Output", problem.output_structure.value)
        if problem.target_specification:
            problem_grid.add_row("Target(s)", ", ".join(problem.target_specification))
        predictive = problem.domain_payloads.get("predictive", {})
        if predictive.get("explainability"):
            problem_grid.add_row("Explainability", str(predictive["explainability"]))
        problem_grid.add_row("Validation", problem.validation_design)
        self.emit(
            PresentationEventKind.PROBLEM_CONTRACT_RESOLVED,
            stage="OBJECTIVE",
            status="COMPLETE",
            payload=problem.to_dict(),
            renderable=Panel(
                problem_grid,
                title="PROBLEM / OBJECTIVE CONTRACT",
                title_align="left",
                border_style="magenta",
            ),
            semantic_checkpoint_id=checkpoint,
            source_component="start.review.workflow_coherence",
        )

    def refresh_problem_contract(self) -> None:
        """Publish a runtime-resolved objective contract without replaying intake/plan."""
        self._publish_problem_contract(checkpoint="coherence.problem_objective.runtime_resolved")

    def publish_coherence_contract(self) -> None:
        """Synchronously publish and render the bound intake/objective/plan contracts."""
        envelope = self.coherence_envelope
        if envelope is None:
            raise RuntimeError("bind_coherence_envelope() must be called before publishing")
        data = envelope.data_intake
        data_grid = Table.grid(padding=(0, 2))
        data_grid.add_column(style="bold cyan", no_wrap=True)
        data_grid.add_column()
        data_grid.add_row("Source", f"{data.source_kind} · {data.source_provider_or_adapter}")
        original_source = str(
            (data.provenance_metadata or {}).get("selected_original_source", "")
        )
        if original_source:
            data_grid.add_row("Selected/original source", original_source)
        data_grid.add_row("Context", data.context_id)
        if data.schema_summary:
            data_grid.add_row("Resolved runtime context", " + ".join(data.schema_summary))
        data_grid.add_row("Reference", data.source_reference)
        dimensions = ", ".join(
            f"{key}={value}" for key, value in data.dimensions.items() if value is not None
        )
        data_grid.add_row("Dimensions", dimensions or "NOT_DECLARED")
        data_grid.add_row("Observations", data.observation_semantics)
        data_grid.add_row("Provenance", f"sha256:{data.provenance_fingerprint[:16]}…")
        data_grid.add_row("Validation", data.validation_status)
        self.emit(
            PresentationEventKind.DATA_SOURCE_RESOLVED,
            stage="DATA",
            status="COMPLETE",
            payload=data.to_dict(),
            renderable=Panel(
                data_grid,
                title="DATA SOURCE RESOLVED",
                title_align="left",
                border_style="bright_blue",
            ),
            semantic_checkpoint_id="coherence.data_intake",
            source_component=data.source_provider_or_adapter,
        )

        self._publish_problem_contract(checkpoint="coherence.problem_objective")

        plan = envelope.capability_plan
        plan_grid = Table.grid(padding=(0, 2))
        plan_grid.add_column(style="bold green", no_wrap=True)
        plan_grid.add_column()
        plan_grid.add_row("Route", plan.route_id)
        plan_grid.add_row("Mode", plan.review_mode)
        for stage_name in (
            "HUMAN_DECISIONS",
            "AGENT_DECISIONS",
            "DETERMINISTIC_EXECUTION",
            "GROUNDING",
            "GOVERNANCE_POLICY",
        ):
            plan_grid.add_row(stage_name.replace("_", " ").title(), plan.stages[stage_name].value)
        plan_grid.add_row("Registered surfaces", str(len(plan.registered_test_ids)))
        plan_grid.add_row("Artifact families", ", ".join(plan.artifact_families) or "NONE")
        for capability in plan.environment_projection:
            plan_grid.add_row(
                f"Environment · {capability.capability_id}",
                f"{capability.state.value} — {capability.owner}",
            )
        self.emit(
            PresentationEventKind.CAPABILITY_PLAN_RESOLVED,
            stage="PLAN",
            status="COMPLETE",
            payload=plan.to_dict(),
            renderable=Panel(
                plan_grid,
                title="CAPABILITY PLAN",
                title_align="left",
                border_style="green",
            ),
            semantic_checkpoint_id="coherence.capability_plan",
            source_component="start.review.workflow_coherence",
        )

    def emit(
        self,
        kind: PresentationEventKind,
        *,
        stage: str,
        status: str,
        payload: dict[str, Any] | None = None,
        renderable: RenderableType | None = None,
        semantic_checkpoint_id: str = "",
        source_component: str = "runtime",
        recurrent: bool = True,
    ) -> PresentationEvent:
        stage = stage.upper()
        normalized_status = status.upper()
        event = PresentationEvent(
            sequence=len(self.events) + 1,
            kind=kind,
            run_id=self.state.run_id,
            semantic_checkpoint_id=semantic_checkpoint_id,
            stage=stage,
            status=normalized_status,
            source_component=source_component,
            payload=payload or {},
        )
        self.events.append(event)
        if stage in self.state.stage_status:
            previous = self.state.active_stage
            if previous and previous != stage and self.state.stage_status.get(previous) == "ACTIVE":
                self._complete_stage(previous)
            self.state.active_stage = stage
            self.state.stage_status[stage] = "ACTIVE"
            if not self.state.active_stage_history or self.state.active_stage_history[-1] != stage:
                self.state.active_stage_history.append(stage)
        if semantic_checkpoint_id:
            self.state.current_checkpoint = semantic_checkpoint_id
        self._consume(event)
        if recurrent and kind in HIGH_VALUE_EVENTS:
            self._print_spine_if_changed()
        if renderable is not None:
            self._print(renderable)
        if stage in self.state.stage_status and normalized_status not in {"ACTIVE", "RUNNING"}:
            self.state.stage_status[stage] = normalized_status
            if normalized_status in {"COMPLETE", "PASSED", "SEALED", "ALLOW", "RECORDED"}:
                self._complete_stage(stage)
            if self.state.active_stage == stage:
                self.state.active_stage = None
        dwell = self.pacing.seconds_for(kind)
        if dwell:
            self._signal_controller("dwell", seconds=dwell, reason=kind.value)
        return event

    def _signal_controller(self, action: str, **payload: Any) -> None:
        if not self.pacing.enabled:
            return
        stream = getattr(self.console, "file", None)
        if stream is None:
            return
        stream.write(_presentation_signal(action, payload))
        stream.flush()

    def _presentation_card(self, renderable: RenderableType, *, dwell: float) -> None:
        self._print(renderable)
        if self.pacing.enabled and dwell > 0:
            self._signal_controller("dwell", seconds=dwell, reason="presentation_card")

    def _consume(self, event: PresentationEvent) -> None:
        payload = event.payload
        if event.kind is PresentationEventKind.HUMAN_ACTION:
            self.state.last_human_action = str(payload.get("action", ""))
        elif event.kind is PresentationEventKind.AGENT_ACTIVATED:
            self.state.current_agent = str(payload.get("agent_name", "")) or None
        elif event.kind in {
            PresentationEventKind.ENGINE_STARTED,
            PresentationEventKind.DIAGNOSTIC_STARTED,
        }:
            self.state.current_engine = str(payload.get("engine_name", "")) or None
        elif event.kind is PresentationEventKind.GROUNDING_ACCEPTED:
            self.state.grounding_state = "ACCEPTED"
            self._consume_grounding_counts(payload)
        elif event.kind is PresentationEventKind.GROUNDING_REJECTED:
            self.state.grounding_state = "REJECTED"
            self._consume_grounding_counts(payload)
        elif event.kind is PresentationEventKind.EVIDENCE_COMMITTED:
            self._record_evidence_category(
                str(payload.get("test_id", "unknown")), bool(payload.get("visible", False))
            )
        elif event.kind is PresentationEventKind.GOVERNANCE_EVALUATED:
            self.state.governance_disposition = str(payload.get("disposition", "")) or None
            self.state.governance_unresolved_count = int(payload.get("unresolved_count", 0) or 0)
            self.state.governance_validation_failures = int(
                payload.get("validation_failures", 0) or 0
            )
            self.state.governance_conditions = [
                str(item) for item in payload.get("conditions", ()) if str(item).strip()
            ]
        elif event.kind is PresentationEventKind.OPA_EVALUATED:
            self.state.policy_decision = str(payload.get("decision", "")) or None
            self.state.policy_engine = str(payload.get("engine", "")) or None
        elif event.kind is PresentationEventKind.TRACE_READY:
            self.state.trace_state = event.status
            self.state.trace_id = str(payload.get("trace_id", "")) or None
        elif event.kind is PresentationEventKind.ATTESTATION_SEALED:
            self.state.attestation_state = "SEALED"
            self.state.merkle_root = str(payload.get("merkle_root", "")) or None
        elif event.kind is PresentationEventKind.ATTESTATION_WITHHELD:
            self.state.attestation_state = "WITHHELD"
        elif event.kind is PresentationEventKind.DATA_SOURCE_RESOLVED:
            self.state.data_context_id = str(payload.get("context_id", "")) or None
        elif event.kind is PresentationEventKind.PROBLEM_CONTRACT_RESOLVED:
            self.state.objective = str(payload.get("objective", "")) or None
        elif event.kind is PresentationEventKind.CAPABILITY_PLAN_RESOLVED:
            self.state.capability_route = str(payload.get("route_id", "")) or None
        elif event.kind is PresentationEventKind.OUTCOME_READY:
            self.state.outcome_state = "READY"

    def _consume_grounding_counts(self, payload: dict[str, Any]) -> None:
        quantitative = int(
            payload.get("grounding_required_claims", payload.get("quantitative_claims", 0))
        )
        grounded = int(payload.get("grounded_claims", 0))
        self.state.grounded_claims += grounded
        self.state.ungrounded_claims += max(0, quantitative - grounded)

    def _complete_stage(self, stage: str) -> None:
        self.state.stage_status[stage] = "COMPLETE"
        if stage not in self.state.completed_stages:
            self.state.completed_stages.append(stage)

    def _print(self, renderable: RenderableType) -> None:
        if isinstance(renderable, Panel):
            renderable.width = self.max_width
        self.console.print(renderable)

    def _spine_signature(self) -> tuple[Any, ...]:
        stages = CONTROL_PLANE_STAGES if self.control_plane else STAGES
        return (
            self.state.active_stage,
            tuple((stage, self.state.stage_status[stage]) for stage in stages),
        )

    def _print_spine_if_changed(self) -> None:
        """Coalesce only consecutive identical architecture-state renders."""
        signature = self._spine_signature()
        if signature == self._last_spine_signature:
            return
        self._print(self.architecture_spine())
        self._last_spine_signature = signature

    def architecture_spine(self) -> Panel:
        cells: list[Text] = []
        stages = CONTROL_PLANE_STAGES if self.control_plane else STAGES
        for stage in stages:
            status = self.state.stage_status[stage]
            if stage == self.state.active_stage and status in {"ACTIVE", "RUNNING"}:
                marker, style = "◉", "bold cyan"
            elif status in {"COMPLETE", "PASSED", "SEALED", "ALLOW", "RECORDED"}:
                marker, style = "●", "green"
            elif status in {"FAILED", "REJECTED", "DENY", "ERROR", "UNRESOLVED"}:
                marker, style = "●", "red"
            elif status == "NOT_APPLICABLE":
                marker, style = "·", "dim"
            else:
                marker, style = "○", "dim"
            cells.append(Text(f"{marker} {stage}", style=style))
        surface = "CONTROL-PLANE RIBBON" if self.control_plane else "ARCHITECTURE SPINE"
        title = f"{surface} | StART v6.0.2 | {self.state.run_id} | {self.state.mode}"
        return Panel(
            Columns(cells, equal=False, expand=True, padding=(0, 1)),
            title=title,
            title_align="left",
            border_style="bright_black",
            padding=(0, 1),
            width=self.max_width,
        )

    def show_spine(self) -> None:
        self._print_spine_if_changed()

    def flight_context(self, title: str, rows: Iterable[tuple[str, Any]]) -> None:
        """Render canonical runtime metadata without creating a presentation event."""
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", no_wrap=True)
        grid.add_column()
        for label, value in rows:
            if value is None or value == "":
                continue
            if isinstance(value, (list, tuple, set)):
                text = ", ".join(str(item) for item in value) or "NOT_APPLICABLE"
            else:
                text = str(value)
            grid.add_row(label, text)
        self._presentation_card(
            Panel(
                grid,
                title=f"{title} — RUNTIME CONTEXT",
                title_align="left",
                border_style="bright_blue",
            ),
            dwell=1.4,
        )

    def review_story(
        self,
        title: str,
        *,
        question: str = "",
        alternatives: Iterable[str] = (),
        human_action: str = "",
        agent_recommendation: str = "",
        deterministic_result: str = "",
        evidence: Iterable[str] = (),
        outcome: str = "",
        prominence: str = "normal",
    ) -> None:
        """Summarize an already-observed decision without becoming a truth source."""
        rows = Table.grid(padding=(0, 2))
        rows.add_column(style="bold", no_wrap=True)
        rows.add_column()
        values = (
            ("Question", question),
            ("Alternatives", " · ".join(str(item) for item in alternatives)),
            ("Human action", human_action),
            ("Agent recommendation", agent_recommendation),
            ("Deterministic result", deterministic_result),
            ("Evidence", ", ".join(str(item) for item in evidence)),
            ("Outcome", outcome),
        )
        for label, value in values:
            if value:
                rows.add_row(label, value)
        hero = prominence == "hero"
        self._presentation_card(
            Panel(
                rows,
                title=title,
                title_align="left",
                border_style="bold red" if hero else "bright_magenta",
            ),
            dwell=2.2 if hero else 1.5,
        )

    def artifact_board_ready(self, manifest_path: str | Path, *, run_id: str) -> None:
        """Announce one exact current-run manifest to the private viewer controller."""
        manifest = Path(manifest_path).expanduser().resolve()
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold")
        grid.add_column()
        grid.add_row("Run", run_id)
        grid.add_row("Manifest", str(manifest))
        grid.add_row("Layout", "Terminal 62% · scientific artifacts 38%")
        self._presentation_card(
            Panel(
                grid,
                title="CURRENT-RUN SCIENTIFIC ARTIFACTS READY",
                title_align="left",
                border_style="bright_cyan",
            ),
            dwell=1.0,
        )
        self._signal_controller(
            "artifact_board",
            manifest_path=str(manifest),
            run_id=run_id,
        )

    def mark_not_applicable(self, *stages: str) -> None:
        for stage in stages:
            normalized = stage.upper()
            if normalized in self.state.stage_status:
                self.state.stage_status[normalized] = "NOT_APPLICABLE"

    def session_started(self, *, source_component: str, checkpoint: str = "session.start") -> None:
        self.emit(
            PresentationEventKind.SESSION_STARTED,
            stage="SESSION",
            status="COMPLETE",
            payload={"checkpoint": checkpoint},
            semantic_checkpoint_id=checkpoint,
            source_component=source_component,
        )

    def publish_outcome_capsule(self) -> Any | None:
        """Build/render the common outcome capsule from this run's canonical state."""
        if self.coherence_envelope is None:
            return None
        from start.review.workflow_coherence import build_run_outcome_capsule

        capsule = build_run_outcome_capsule(
            envelope=self.coherence_envelope,
            events=self.events,
            presentation_state=self.state,
        )
        self.outcome_capsule = capsule
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", no_wrap=True)
        grid.add_column()
        alternatives_display = " · ".join(capsule.alternatives) or "NONE"
        if self.coherence_envelope.capability_plan.route_id == "unified_domain_review":
            grouped: list[str] = []
            family_labels = {
                "portfolio": "portfolio construction",
                "covariance": "covariance",
                "attribution": "factor attribution",
                "traded_risk": "VaR / backtesting",
            }
            for domain, payload in self.coherence_envelope.problem.domain_payloads.items():
                surfaces = [str(item) for item in payload.get("analytical_surfaces", ())]
                if not surfaces:
                    continue
                families: dict[str, int] = {}
                for surface in surfaces:
                    family = surface.split(".", 1)[0]
                    label = family_labels.get(family, family.replace("_", " "))
                    families[label] = families.get(label, 0) + 1
                grouped.append(
                    f"{domain.title()}: "
                    + ", ".join(f"{label} ({count})" for label, count in families.items())
                )
            if grouped:
                alternatives_display = " · ".join(grouped)
        outcome_exceptions = [
            f"{output}={status}"
            for summary in self.coherence_envelope.deterministic_summaries
            for output, status in zip(summary.authoritative_outputs, summary.statuses)
            if str(status).upper() in {"WARN", "FAIL", "ERROR", "SKIPPED"}
        ]
        grid.add_row("Started with", capsule.started_with)
        grid.add_row("Objective", capsule.objective)
        grid.add_row("Alternatives", alternatives_display)
        grid.add_row("Human decisions", " · ".join(capsule.human_decisions))
        if isinstance(capsule.agent_contribution, str):
            agent_text = capsule.agent_contribution
        else:
            agent_text = " · ".join(capsule.agent_contribution)
        grid.add_row("Agent contribution", agent_text)
        grid.add_row("Deterministic results", " · ".join(capsule.deterministic_results))
        if outcome_exceptions:
            grid.add_row("Non-pass execution results", " · ".join(outcome_exceptions))
        grid.add_row("Evidence", f"{len(capsule.evidence_record_references)} canonical reference(s)")
        grid.add_row("Grounding", capsule.grounding)
        grid.add_row("Governance / policy", capsule.governance_policy)
        grid.add_row("Achieved result", capsule.achieved_result)
        grid.add_row("Remaining conditions", " · ".join(capsule.remaining_conditions))
        self.emit(
            PresentationEventKind.OUTCOME_READY,
            stage="OUTCOME",
            status="COMPLETE",
            payload=capsule.to_dict(),
            renderable=Panel(
                grid,
                title="RUN OUTCOME CAPSULE",
                title_align="left",
                border_style="cyan",
            ),
            semantic_checkpoint_id="coherence.outcome",
            source_component="start.review.workflow_coherence",
        )
        return capsule

    def session_completed(self, *, source_component: str) -> None:
        if not self.control_plane and self.state.grounding_state == "REJECTED":
            self.review_story(
                "EVIDENCE-NATIVE REVIEW — VERIFIED CHAIN",
                question="Can an AI quantitative claim enter governance without canonical support?",
                human_action=self.state.last_human_action or "REVIEWED",
                agent_recommendation="Claim submitted for canonical grounding",
                deterministic_result="Canonical EvidenceRecords remain authoritative",
                evidence=(f"{self.state.evidence_count} committed record(s)",),
                outcome=(
                    "Invalid claim rejected · deterministic evidence retained · "
                    f"governance {self.state.governance_disposition or 'NOT_EVALUATED'} · "
                    f"policy {self.state.policy_decision or 'NOT_EVALUATED'} · "
                    f"attestation {self.state.attestation_state or 'NOT_EVALUATED'}"
                ),
                prominence="hero",
            )
        self.publish_outcome_capsule()
        self.emit(
            PresentationEventKind.SESSION_COMPLETED,
            stage="SESSION",
            status="COMPLETE",
            payload={"evidence_count": self.state.evidence_count},
            source_component=source_component,
            renderable=self.completion_panel(),
            recurrent=False,
        )

    def human_action(
        self,
        action: str,
        checkpoint: str,
        note: str = "",
        *,
        source_component: str = "interactive_review",
    ) -> None:
        action = action.upper()
        label = {
            "A": "ACCEPT",
            "ACCEPT": "ACCEPT",
            "C": "CHALLENGE",
            "CHALLENGE": "CHALLENGE",
            "Q": "QUESTION",
            "QUESTION": "QUESTION",
            "O": "OVERRIDE",
            "OVERRIDE": "OVERRIDE",
            "V": "VIEW",
            "VIEW": "VIEW",
        }.get(action, action)
        lines = [f"[bold]HUMAN  [{action[:1]}] {label}[/bold]", f"[dim]{checkpoint}[/dim]"]
        if note:
            lines.append(note)
        if label == "OVERRIDE":
            lines.append("[yellow]Override recorded ≠ numerical recomputation[/yellow]")
        self.emit(
            PresentationEventKind.HUMAN_ACTION,
            stage="HUMAN",
            status="COMPLETE",
            payload={"action": label, "checkpoint": checkpoint, "note": note},
            renderable=Panel("\n".join(lines), border_style="blue", padding=(0, 1)),
            semantic_checkpoint_id=checkpoint,
            source_component=source_component,
        )

    def graph_transition(
        self,
        source: str,
        target: str,
        *,
        trigger: str,
        state_hash: str,
        thread_id: str = "",
        status: str = "COMPLETE",
        checkpoint: str = "",
    ) -> None:
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="bold cyan")
        grid.add_column()
        grid.add_row("Transition", f"{source} → {target}")
        grid.add_row("Trigger", trigger)
        grid.add_row("State hash", _short_hash(state_hash))
        if thread_id:
            grid.add_row("Thread", thread_id)
        self.emit(
            PresentationEventKind.GRAPH_TRANSITION,
            stage="LANGGRAPH",
            status=status,
            payload={
                "source": source,
                "target": target,
                "trigger": trigger,
                "state_hash": state_hash,
                "thread_id": thread_id,
            },
            renderable=Panel(grid, title="LANGGRAPH TRANSITION", title_align="left", border_style="cyan"),
            semantic_checkpoint_id=checkpoint or target,
            source_component="start.orchestration.state_graph",
        )

    def agent_card(
        self,
        agent_name: str,
        *,
        role: str,
        evidence_ids: Iterable[str] = (),
        activity: str = "reasoning / challenge",
        deterministic_context: str = "",
        checkpoint: str = "",
    ) -> None:
        scoped = list(evidence_ids)
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Role", role)
        table.add_row("Activity", activity)
        table.add_row("EvidenceRecords", ", ".join(scoped[:5]) if scoped else "NONE")
        if deterministic_context:
            table.add_row("Deterministic context", deterministic_context)
        table.add_row("Numeric authority", "[bold yellow]NONE[/bold yellow]")
        self.emit(
            PresentationEventKind.AGENT_ACTIVATED,
            stage="AGENT",
            status="ACTIVE",
            payload={
                "agent_name": agent_name,
                "role": role,
                "activity": activity,
                "evidence_ids": scoped,
                "deterministic_context": deterministic_context,
                "numeric_authority": "NONE",
            },
            renderable=Panel(table, title=agent_name, title_align="left", border_style="magenta"),
            semantic_checkpoint_id=checkpoint,
            source_component=agent_name,
        )

    def agent_decision_trace(self, trace: Any) -> None:
        """Publish an auditable decision summary without exposing private reasoning."""
        if getattr(trace, "numeric_authority", "NONE") != "NONE":
            raise ValueError("agent decision traces cannot carry numeric authority")
        if self.coherence_envelope is not None:
            self.coherence_envelope.agent_decision_traces.append(trace)
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold magenta", no_wrap=True)
        grid.add_column()
        grid.add_row("Agent", str(trace.agent_identity))
        grid.add_row("Role", str(trace.role))
        grid.add_row("Trigger", str(trace.trigger))
        grid.add_row("Alternatives", " · ".join(trace.applicable_alternatives) or "NONE")
        grid.add_row("Recommendation", str(trace.recommendation))
        grid.add_row("Bounded rationale", str(trace.rationale_summary))
        grid.add_row(
            "EvidenceRecords",
            ", ".join(trace.evidence_record_references) or "NONE",
        )
        grid.add_row("Human action", str(trace.human_action or "PENDING"))
        grid.add_row("Result", str(trace.resulting_action))
        grid.add_row("Numeric authority", "[bold yellow]NONE[/bold yellow]")
        self.emit(
            PresentationEventKind.AGENT_DECISION_RECORDED,
            stage="AGENT",
            status="COMPLETE",
            payload=trace.to_dict(),
            renderable=Panel(
                grid,
                title="AGENT DECISION TRACE",
                title_align="left",
                border_style="magenta",
            ),
            semantic_checkpoint_id=str(trace.checkpoint),
            source_component=str(trace.agent_identity),
        )

    def engine_card(
        self,
        engine_name: str,
        *,
        operation: str,
        device: str = "",
        input_contract: str = "",
        status: str = "ACTIVE",
        checkpoint: str = "",
    ) -> None:
        table = Table.grid(padding=(0, 1))
        table.add_column(style="bold")
        table.add_column()
        table.add_row("Engine", engine_name)
        table.add_row("Operation", operation)
        if device:
            table.add_row("Device", device)
        if input_contract:
            table.add_row("Input", input_contract)
        table.add_row("Numeric authority", "[bold green]AUTHORITATIVE[/bold green]")
        self.emit(
            PresentationEventKind.ENGINE_STARTED,
            stage="ENGINE",
            status=status,
            payload={
                "engine_name": engine_name,
                "operation": operation,
                "device": device,
                "input_contract": input_contract,
                "numeric_authority": "AUTHORITATIVE",
            },
            renderable=Panel(
                table, title="DETERMINISTIC ENGINE", title_align="left", border_style="green"
            ),
            semantic_checkpoint_id=checkpoint,
            source_component=engine_name,
        )

    def diagnostic_started(
        self,
        engine_name: str,
        *,
        operation: str,
        checkpoint: str,
    ) -> None:
        body = (
            f"[bold]{engine_name}[/bold]\n{operation}\n"
            "[bold green]Numeric authority: AUTHORITATIVE[/bold green]"
        )
        self.emit(
            PresentationEventKind.DIAGNOSTIC_STARTED,
            stage="ENGINE",
            status="ACTIVE",
            payload={"engine_name": engine_name, "operation": operation},
            renderable=Panel(body, title="DETERMINISTIC DIAGNOSTIC", border_style="yellow"),
            semantic_checkpoint_id=checkpoint,
            source_component=engine_name,
        )

    def engine_completed(
        self,
        engine_name: str,
        *,
        output_count: int | None = None,
        details: str = "",
        checkpoint: str = "",
    ) -> None:
        payload: dict[str, Any] = {"engine_name": engine_name, "details": details}
        if output_count is not None:
            payload["output_count"] = int(output_count)
        self.emit(
            PresentationEventKind.ENGINE_COMPLETED,
            stage="ENGINE",
            status="COMPLETE",
            payload=payload,
            semantic_checkpoint_id=checkpoint,
            source_component=engine_name,
        )

    def deterministic_execution_summary(self, summary: Any, *, checkpoint: str = "") -> None:
        """Publish an index over authoritative outputs without recomputing them."""
        if self.coherence_envelope is not None:
            self.coherence_envelope.deterministic_summaries.append(summary)
        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold green", no_wrap=True)
        grid.add_column()
        grid.add_row("Engine(s)", ", ".join(summary.engine_ids) or "NONE")
        grid.add_row("Operation(s)", " · ".join(summary.operations) or "NONE")
        grid.add_row("Authoritative outputs", " · ".join(summary.authoritative_outputs) or "NONE")
        grid.add_row("Statuses", ", ".join(summary.statuses) or "NONE")
        grid.add_row(
            "EvidenceRecords",
            ", ".join(summary.evidence_record_references[:8])
            or "NONE",
        )
        if summary.skipped_or_not_applicable:
            grid.add_row("Skipped / N/A", " · ".join(summary.skipped_or_not_applicable))
        self.emit(
            PresentationEventKind.DETERMINISTIC_EXECUTION_SUMMARIZED,
            stage="ENGINE",
            status="COMPLETE",
            payload=summary.to_dict(),
            renderable=Panel(
                grid,
                title="DETERMINISTIC EXECUTION SUMMARY",
                title_align="left",
                border_style="green",
            ),
            semantic_checkpoint_id=checkpoint or "coherence.deterministic_execution",
            source_component=",".join(summary.engine_ids) or "deterministic_engine",
        )

    def artifact_available(
        self,
        *,
        artifact_id: str,
        file_path: str,
        evidence_ids: Iterable[str],
        checkpoint: str,
    ) -> None:
        self.emit(
            PresentationEventKind.ARTIFACT_AVAILABLE,
            stage="ENGINE",
            status="COMPLETE",
            payload={
                "artifact_id": artifact_id,
                "file_path": file_path,
                "evidence_ids": list(evidence_ids),
                "checkpoint": checkpoint,
            },
            semantic_checkpoint_id=checkpoint,
            source_component="artifact_registry",
        )

    def authority_card(
        self,
        *,
        reviewer_mode: str,
        backend_mode: str,
        provider: str,
        provider_status: str = "",
        deterministic_only: bool = False,
        llm_calls: int = 0,
        llm_tokens: int = 0,
    ) -> Panel:
        self.state.llm_calls = int(llm_calls)
        self.state.llm_tokens = int(llm_tokens)
        authority = resolve_authority_presentation(
            reviewer_mode=reviewer_mode,
            backend_mode=backend_mode,
            provider=provider,
            provider_status=provider_status,
            deterministic_only=deterministic_only,
        )
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="bold")
        grid.add_column(justify="right")
        grid.add_row("Reviewer reasoning", authority.reviewer_reasoning)
        grid.add_row("Reasoning provider", authority.provider)
        grid.add_row("Hosted LLM execution", authority.hosted_execution)
        grid.add_row("Hosted LLM calls", str(self.state.llm_calls))
        grid.add_row("Hosted LLM tokens", str(self.state.llm_tokens))
        grid.add_row("LLM numeric authority", "[bold yellow]0[/bold yellow]")
        grid.add_row("Deterministic engines", "[bold green]AUTHORITATIVE[/bold green]")
        return Panel(
            grid,
            title="AUTHORITY BOUNDARY",
            title_align="left",
            border_style="bright_black",
            width=self.max_width,
        )

    def show_authority_boundary(
        self,
        *,
        reviewer_mode: str,
        backend_mode: str,
        provider: str,
        provider_status: str = "",
        deterministic_only: bool = False,
        llm_calls: int = 0,
        llm_tokens: int = 0,
    ) -> None:
        self._print(
            self.authority_card(
                reviewer_mode=reviewer_mode,
                backend_mode=backend_mode,
                provider=provider,
                provider_status=provider_status,
                deterministic_only=deterministic_only,
                llm_calls=llm_calls,
                llm_tokens=llm_tokens,
            )
        )

    def evidence_committed(
        self,
        record: EvidenceRecord,
        *,
        producer: str | None = None,
        render: bool | None = None,
    ) -> None:
        canonical = record.model_dump(mode="json")
        digest = hash_obj(canonical)
        metric_name, metric_value = _representative_metric(record.metrics)
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="bold")
        grid.add_column()
        grid.add_row("Evidence", f"[bold cyan]{record.evidence_id}[/bold cyan]")
        grid.add_row("Producer", producer or record.test_id)
        grid.add_row("Status", str(getattr(record.status, "value", record.status)).upper())
        flight_a_context = self.flight_a_context
        temporal_split_metadata = (
            self.session_kind == "A"
            and record.test_id == "split.plan"
            and flight_a_context is not None
            and flight_a_context.modality == "temporal_sequence"
        )
        if temporal_split_metadata and flight_a_context is not None:
            grid.add_row("Scope", "REVIEW METADATA — NOT MODEL INPUT")
            if metric_name:
                grid.add_row("Metadata-only plan", f"metrics.{metric_name} = {metric_value}")
            grid.add_row("Actual model split", flight_a_context.split_description)
        elif metric_name:
            grid.add_row("Metric", f"metrics.{metric_name} = {metric_value}")
        grid.add_row("Run", record.run_id)
        grid.add_row("Record digest", f"sha256:{digest[:16]}…")
        grid.add_row("Lineage", f"{record.dataset_id} → {record.model_id} → {record.test_id}")
        record_status = str(getattr(record.status, "value", record.status)).upper()
        test_id = producer or record.test_id
        visible = self.evidence_policy.visible(test_id, record_status) if render is None else render
        self.emit(
            PresentationEventKind.EVIDENCE_COMMITTED,
            stage="EVIDENCE",
            status="RECORDED",
            payload={
                "evidence_id": record.evidence_id,
                "producer": producer or record.test_id,
                "record_status": record_status,
                "test_id": test_id,
                "run_id": record.run_id,
                "record_digest": digest,
                "visible": visible,
            },
            renderable=(
                Panel(grid, title="EVIDENCE COMMITTED", title_align="left", border_style="cyan")
                if visible
                else None
            ),
            semantic_checkpoint_id=test_id,
            source_component=producer or record.test_id,
            recurrent=visible,
        )

    def evidence_reference_committed(
        self,
        *,
        evidence_id: str,
        test_id: str,
        record_status: str,
        metrics: dict[str, Any] | None = None,
        source_component: str = "evidence_ledger",
    ) -> None:
        visible = self.evidence_policy.visible(test_id, record_status)
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="bold")
        grid.add_column()
        grid.add_row("Evidence", f"[bold cyan]{evidence_id}[/bold cyan]")
        grid.add_row("Producer", test_id)
        grid.add_row("Status", record_status.upper())
        metric_name, metric_value = _representative_metric(metrics or {})
        if metric_name:
            grid.add_row("Metric", f"metrics.{metric_name} = {metric_value}")
        self.emit(
            PresentationEventKind.EVIDENCE_COMMITTED,
            stage="EVIDENCE",
            status="RECORDED",
            payload={
                "evidence_id": evidence_id,
                "test_id": test_id,
                "record_status": record_status.upper(),
                "visible": visible,
            },
            renderable=(
                Panel(grid, title="EVIDENCE COMMITTED", title_align="left", border_style="cyan")
                if visible
                else None
            ),
            semantic_checkpoint_id=test_id,
            source_component=source_component,
            recurrent=visible,
        )

    def _record_evidence_category(self, test_id: str, visible: bool) -> None:
        self.state.evidence_count += 1
        self._evidence_categories[self.evidence_policy.category(test_id)] += 1
        if visible:
            self._visible_evidence_count += 1

    def evidence_batch_summary(self) -> None:
        additional = max(0, self.state.evidence_count - self._visible_evidence_count)
        categories = ", ".join(
            f"{name}={count}" for name, count in sorted(self._evidence_categories.items())
        )
        body = f"+ {additional} additional canonical EvidenceRecords committed"
        if categories:
            body += f"\n[dim]{categories}[/dim]"
        self._print(Panel(body, title="EVIDENCE LEDGER", border_style="cyan"))

    def grounding_result(
        self,
        *,
        accepted: bool,
        quantitative_claims: int,
        grounded_claims: int,
        grounding_required_claims: int | None = None,
        other_exempt_claims: int = 0,
        invalid_details: Iterable[dict[str, Any]] = (),
        continuation: str = "NOT_APPLICABLE",
        deterministic_evidence_retained: bool | None = None,
        checkpoint: str = "",
    ) -> None:
        details = list(invalid_details)
        retained = (not accepted) if deterministic_evidence_retained is None else bool(
            deterministic_evidence_retained
        )
        required = (
            int(quantitative_claims)
            if grounding_required_claims is None
            else max(0, int(grounding_required_claims))
        )
        exempt = max(0, int(other_exempt_claims))
        rejected = max(0, required - int(grounded_claims))
        accepted = bool(accepted) and int(grounded_claims) == required and not details
        if accepted:
            body = (
                f"Quantitative claims  {quantitative_claims}\n"
                f"Grounding-required claims  {required}\n"
                f"Canonically grounded  {grounded_claims}/{required}\n"
                + (f"Other (soft/exempt) claims  {exempt}\n" if exempt else "")
                + "Invalid references  0"
            )
            title, style, status = "GROUNDING ACCEPTED", "green", "PASSED"
            kind = PresentationEventKind.GROUNDING_ACCEPTED
        else:
            rows = [
                f"[bold]Grounding-required claims[/bold]  {required}",
                f"Canonically grounded  {grounded_claims}/{required}",
                *([f"Other (soft/exempt) claims  {exempt}"] if exempt else []),
                f"[bold]Rejected references[/bold]  {max(rejected, len(details))}",
                "[dim]Deterministic evidence remains unchanged.[/dim]",
                f"Continuation  {continuation}",
            ]
            for raw_item in details[:4]:
                item = raw_item if isinstance(raw_item, dict) else getattr(raw_item, "__dict__", {})
                ev = item.get("evidence_id", item.get("citation", "UNKNOWN"))
                path = item.get("metric_path", item.get("path", "UNKNOWN"))
                reason = item.get("error", item.get("reason", "canonical grounding failure"))
                rows.append(f"[red]✗[/red] {ev} → {path} — {reason}")
            body = "\n".join(rows)
            title, style, status = "AGENT RESPONSE REJECTED", "red", "REJECTED"
            kind = PresentationEventKind.GROUNDING_REJECTED
        self.emit(
            kind,
            stage="GROUNDING",
            status=status,
            payload={
                "accepted": accepted,
                "quantitative_claims": quantitative_claims,
                "grounded_claims": grounded_claims,
                "grounding_required_claims": required,
                "other_exempt_claims": exempt,
                "invalid_details": details,
                "continuation": continuation,
                "deterministic_evidence_retained": retained,
            },
            renderable=Panel(body, title=title, title_align="left", border_style=style),
            semantic_checkpoint_id=checkpoint,
            source_component="grounding_validator",
        )

    def challenge_lineage(
        self,
        *,
        source_evidence_ids: Iterable[str],
        diagnostic: str,
        generated_evidence_ids: Iterable[str],
        resolution_status: str,
        checkpoint: str = "",
    ) -> None:
        source = list(source_evidence_ids)
        generated = list(generated_evidence_ids)
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="bold")
        grid.add_column()
        grid.add_row("Source evidence", ", ".join(source) if source else "NONE")
        grid.add_row("Deterministic diagnostic", diagnostic or "NO_REGISTERED_DIAGNOSTIC")
        grid.add_row("Subordinate evidence", ", ".join(generated) if generated else "NONE")
        grid.add_row("Resolution", resolution_status)
        self.emit(
            PresentationEventKind.CHALLENGE_RESOLVED,
            stage="EVIDENCE",
            status=resolution_status,
            payload={
                "source_evidence_ids": source,
                "diagnostic": diagnostic,
                "generated_evidence_ids": generated,
                "resolution_status": resolution_status,
            },
            renderable=Panel(grid, title="CHALLENGE LINEAGE", title_align="left", border_style="yellow"),
            semantic_checkpoint_id=checkpoint,
            source_component="challenge_orchestrator",
        )

    def challenge_created(
        self,
        *,
        checkpoint: str,
        note: str,
        source_evidence_ids: Iterable[str],
    ) -> None:
        """Record the actual reviewer challenge before a diagnostic is dispatched."""
        source = list(source_evidence_ids)
        self.emit(
            PresentationEventKind.CHALLENGE_CREATED,
            stage="AGENT",
            status="ACTIVE",
            payload={
                "checkpoint": checkpoint,
                "note": note,
                "source_evidence_ids": source,
            },
            semantic_checkpoint_id=checkpoint,
            source_component="interactive_review",
        )

    def policy_gate(self, decision: Any) -> None:
        evidence_ids = list(
            getattr(decision, "evidence_ids", None)
            or getattr(decision, "evidence_refs", None)
            or []
        )
        value = str(getattr(decision, "decision", "UNKNOWN"))
        engine = str(getattr(decision, "engine", "NATIVE_ADAPTER"))
        decision_id = str(getattr(decision, "decision_id", ""))
        if not decision_id:
            fingerprint = str(getattr(decision, "input_fingerprint", ""))
            decision_id = f"POL-DEC-{fingerprint[:8]}" if fingerprint else "UNAVAILABLE"
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="bold")
        grid.add_column()
        grid.add_row("Engine", engine)
        grid.add_row("Package", str(getattr(decision, "policy_package", "UNKNOWN")))
        grid.add_row("Rule", str(getattr(decision, "rule_name", "UNKNOWN")))
        grid.add_row("Evidence", f"{len(evidence_ids)} record(s)")
        grid.add_row("Decision", value)
        grid.add_row("Decision ID", decision_id)
        self.emit(
            PresentationEventKind.OPA_EVALUATED,
            stage="OPA",
            status=value,
            payload={
                "engine": engine,
                "policy_package": getattr(decision, "policy_package", ""),
                "rule_name": getattr(decision, "rule_name", ""),
                "decision": value,
                "decision_id": decision_id,
                "evidence_ids": evidence_ids,
            },
            renderable=Panel(grid, title="POLICY GATE", title_align="left", border_style="bright_red"),
            semantic_checkpoint_id="policy.governance_attestation",
            source_component=engine,
        )

    def policy_gate_payload(self, payload: dict[str, Any]) -> None:
        """Render a serialized PolicyEvaluationResult from a runtime event."""

        class _Decision:
            pass

        decision = _Decision()
        for key, value in payload.items():
            setattr(decision, key, value)
        self.policy_gate(decision)

    def governance_card(
        self,
        *,
        disposition: str,
        evidence_count: int,
        unresolved_count: int,
        validation_failures: int,
        conditions: Iterable[str] = (),
    ) -> None:
        canonical_conditions = tuple(dict.fromkeys(str(item) for item in conditions if str(item).strip()))
        grid = Table.grid(padding=(0, 1))
        grid.add_column(style="bold")
        grid.add_column()
        grid.add_row("Disposition", disposition)
        grid.add_row("EvidenceRecords", str(evidence_count))
        grid.add_row("Unresolved items", str(unresolved_count))
        grid.add_row("Validation failures", str(validation_failures))
        if canonical_conditions:
            grid.add_row("Conditions", " · ".join(canonical_conditions))
        self.emit(
            PresentationEventKind.GOVERNANCE_EVALUATED,
            stage="GOVERNANCE",
            status="COMPLETE",
            payload={
                "disposition": disposition,
                "evidence_count": evidence_count,
                "unresolved_count": unresolved_count,
                "validation_failures": validation_failures,
                "conditions": list(canonical_conditions),
            },
            renderable=Panel(grid, title="GOVERNANCE", title_align="left", border_style="magenta"),
            semantic_checkpoint_id="governance.signoff",
            source_component="model_governance",
        )

    def attestation_sealed(self, *, merkle_root: str, leaf_count: int | None = None) -> None:
        body = f"Merkle root  {_short_hash(merkle_root)}"
        if leaf_count is not None:
            body += f"\nLeaves  {leaf_count}"
        self.emit(
            PresentationEventKind.ATTESTATION_SEALED,
            stage="ATTESTATION",
            status="SEALED",
            payload={"merkle_root": merkle_root, "leaf_count": leaf_count},
            renderable=Panel(body, title="CRYPTOGRAPHIC ATTESTATION", border_style="green"),
            semantic_checkpoint_id="attestation.seal",
            source_component="start.attestation.seal",
        )

    def attestation_withheld(self, *, reason: str) -> None:
        self.emit(
            PresentationEventKind.ATTESTATION_WITHHELD,
            stage="ATTESTATION",
            status="DENY",
            payload={"reason": reason},
            renderable=Panel(reason, title="ATTESTATION WITHHELD", border_style="red"),
            semantic_checkpoint_id="attestation.seal",
            source_component="start.attestation.seal",
        )

    def trace_waterfall(self, records: Iterable[Any], *, drift: float | None = None) -> None:
        def field(rec: Any, name: str, default: Any = None) -> Any:
            if isinstance(rec, dict):
                return rec.get(name, default)
            return getattr(rec, name, default)

        spans = sorted(list(records), key=lambda rec: float(field(rec, "start_time", 0.0)))
        by_span_id = {str(field(rec, "span_id", "")): rec for rec in spans}

        def depth(rec: Any) -> int:
            count = 0
            parent_id = field(rec, "parent_span_id")
            visited: set[str] = set()
            while parent_id and str(parent_id) in by_span_id and str(parent_id) not in visited:
                visited.add(str(parent_id))
                count += 1
                parent_id = field(by_span_id[str(parent_id)], "parent_span_id")
            return count

        table = Table(title="OPENTELEMETRY TRACE", title_style="bold green", box=None)
        table.add_column("Span", style="white")
        table.add_column("Duration", justify="right", style="cyan")
        table.add_column("Status", justify="right")
        for rec in spans:
            duration = float(field(rec, "duration_ms", 0.0))
            table.add_row(
                f"{'  ' * depth(rec)}{field(rec, 'name', 'unknown')}",
                f"{duration:.2f} ms",
                str(field(rec, "status", "UNKNOWN")),
            )
        trace_ids = sorted(
            {str(field(rec, "trace_id", "")) for rec in spans if field(rec, "trace_id", "")}
        )
        trace_id = trace_ids[0] if len(trace_ids) == 1 else ""
        if trace_id:
            footer = f"trace_id  {_short_hash(trace_id)}"
        elif trace_ids:
            footer = f"trace_ids  {len(trace_ids)} independent local traces"
        else:
            footer = "trace_id  UNAVAILABLE"
        if drift is not None:
            footer += f"\nscientific drift with tracing = {drift:g}"
        self.emit(
            PresentationEventKind.TRACE_READY,
            stage="TRACE",
            status="COMPLETE" if spans else "NOT_APPLICABLE",
            payload={
                "trace_id": trace_id,
                "trace_ids": trace_ids,
                "span_count": len(spans),
                "scientific_drift": drift,
            },
            renderable=Panel(Group(table, Text(footer, style="dim")), border_style="green"),
            semantic_checkpoint_id="telemetry.trace",
            source_component="start.telemetry.engineering_trace",
        )

    def completion_panel(self) -> Panel:
        rows = Table.grid(padding=(0, 2))
        rows.add_column(style="bold")
        rows.add_column(justify="right")
        rows.add_row("Data intake contract", _stage_label(self.state.stage_status["DATA"]))
        rows.add_row("Problem / objective", _stage_label(self.state.stage_status["OBJECTIVE"]))
        rows.add_row("Capability plan", _stage_label(self.state.stage_status["PLAN"]))
        if self.control_plane:
            rows.add_row("LangGraph orchestration", "NOT_APPLICABLE")
            rows.add_row("Human review actions", "NOT_APPLICABLE")
            rows.add_row("Agent reasoning", "NOT_APPLICABLE")
        else:
            rows.add_row("LangGraph orchestration", _stage_label(self.state.stage_status["LANGGRAPH"]))
            rows.add_row("Human review actions", _stage_label(self.state.stage_status["HUMAN"]))
        rows.add_row("Deterministic scientific execution", _stage_label(self.state.stage_status["ENGINE"]))
        rows.add_row("EvidenceRecords", str(self.state.evidence_count))
        rows.add_row("Evidence grounding", _stage_label(self.state.stage_status["GROUNDING"]))
        rows.add_row("Governance", self.state.governance_disposition or "NOT_EVALUATED")
        policy_label = "OPA local policy" if self.state.policy_engine == "OPA_LOCAL" else "Policy gate"
        rows.add_row(policy_label, self.state.policy_decision or "NOT_EVALUATED")
        rows.add_row("OpenTelemetry trace", _stage_label(self.state.stage_status["TRACE"]))
        rows.add_row("Merkle attestation", "SEALED" if self.state.merkle_root else "NOT_SEALED")
        rows.add_row("LLM numeric authority", "0")
        footer = Text(
            "\nLangGraph orchestrates. Agents reason. Deterministic engines calculate.\n"
            "EvidenceRecords prove. OPA governs. OpenTelemetry observes.",
            justify="center",
            style="bold",
        )
        return Panel(
            Group(rows, footer),
            title="StART — REVIEW COMPLETE",
            title_align="center",
            border_style="cyan",
            padding=(1, 2),
            width=self.max_width,
        )

    def export(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "schema": "start.presentation-events/2",
                    "run_id": self.state.run_id,
                    "state": asdict(self.state),
                    "events": [event.to_dict() for event in self.events],
                    "workflow_coherence": (
                        self.coherence_envelope.to_dict()
                        if self.coherence_envelope is not None
                        else None
                    ),
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return target


class RuntimePresentationSink:
    """Translate genuine CanonicalExecutionService events synchronously."""

    def __init__(self, observer: TerminalReviewObserver, *, workflow_id: str) -> None:
        self.observer = observer
        self.workflow_id = workflow_id
        self._results: dict[str, dict[str, Any]] = {}
        self._engine_started = False
        self._engine_completed = False
        self._context_metadata: dict[str, Any] = {}
        self._context_id = ""

    def emit(self, event: Any) -> None:
        event_type = str(getattr(event, "event_type", ""))
        if event_type == "context_ready":
            metadata = dict(getattr(event, "metadata", {}) or {})
            self._context_metadata = metadata
            self._context_id = str(metadata.get("spec_id", "")) or (
                "institutional_market_v1"
                if self.workflow_id == "quantitative_finance"
                else "deep_learning_v1"
                if self.workflow_id == "deep_learning"
                else "institutional_credit_v1"
            )
            self.observer.session_started(
                source_component="start.runtime.execution",
                checkpoint=str(getattr(event, "node_id", "step-context")),
            )
            if metadata.get("actual_assets") is not None:
                shape = (
                    f"{metadata.get('actual_assets')} assets × "
                    f"{metadata.get('actual_periods')} observations"
                )
            elif metadata.get("actual_samples") is not None:
                shape = (
                    f"{metadata.get('actual_samples')} samples × "
                    f"{metadata.get('actual_features')} features"
                )
            else:
                shape = "runtime-resolved structured context"
            self.observer.flight_context(
                "SESSION C — DETERMINISTIC CONTROL PLANE",
                (
                    ("Workflow", self.workflow_id),
                    ("Dataset / context", metadata.get("spec_id", "runtime-resolved")),
                    ("Structure", shape),
                    ("Execution nature", "deterministic-only; LLM numeric authority = 0"),
                    (
                        "Purpose",
                        "Execute registered science → commit evidence → apply governance/policy → attest",
                    ),
                ),
            )
            self.observer.show_authority_boundary(
                reviewer_mode="deterministic",
                backend_mode="none",
                provider="none",
                deterministic_only=True,
            )
        elif event_type == "workflow_resolved" and not self._engine_started:
            metadata = dict(getattr(event, "metadata", {}) or {})
            from start.review.workflow_coherence import (
                build_canonical_runtime_coherence_envelope,
            )

            context_id = self._context_id or str(self._context_metadata.get("spec_id", ""))
            envelope = build_canonical_runtime_coherence_envelope(
                run_id=self.observer.state.run_id,
                workflow_id=self.workflow_id,
                context_id=context_id,
                context_metadata=self._context_metadata,
                resolved_metadata=metadata,
            )
            self.observer.bind_coherence_envelope(envelope)
            self.observer.publish_coherence_contract()
            self.observer.engine_card(
                "CanonicalDeterministicExecutionService",
                operation=f"execute {self.workflow_id}",
                device="cpu",
                input_contract=(
                    f"{len(metadata.get('applicable_test_ids', []))} registered analytical surfaces"
                ),
                checkpoint=str(getattr(event, "node_id", "step-context")),
            )
            self._engine_started = True
        elif event_type == "test_completed":
            for evidence_id in list(getattr(event, "evidence_refs", []) or []):
                self._results[str(evidence_id)] = {
                    "test_id": str(getattr(event, "test_id", "")),
                    "status": str(getattr(event, "status", "COMPLETED")),
                    "metrics": dict(getattr(event, "metadata", {}) or {}).get("metrics", {}),
                }
        elif event_type == "evidence_committed":
            for evidence_id in list(getattr(event, "evidence_refs", []) or []):
                cached = self._results.get(str(evidence_id), {})
                self.observer.evidence_reference_committed(
                    evidence_id=str(evidence_id),
                    test_id=str(getattr(event, "test_id", "") or cached.get("test_id", "unknown")),
                    record_status=str(cached.get("status", "RECORDED")),
                    metrics=dict(cached.get("metrics", {}) or {}),
                    source_component="start.evidence.ledger",
                )
        elif event_type == "artifact_created":
            refs = list(getattr(event, "artifact_refs", []) or [])
            for artifact_id in refs:
                self.observer.artifact_available(
                    artifact_id=str(artifact_id),
                    file_path=str(dict(getattr(event, "metadata", {}) or {}).get("file_path", "")),
                    evidence_ids=list(getattr(event, "evidence_refs", []) or []),
                    checkpoint=str(getattr(event, "checkpoint_id", "") or getattr(event, "node_id", "")),
                )
        elif event_type == "checkpoint_committed" and str(
            getattr(event, "checkpoint_id", "")
        ) == "CP-005":
            if self._engine_started and not self._engine_completed:
                self.observer.engine_completed(
                    "CanonicalDeterministicExecutionService",
                    output_count=self.observer.state.evidence_count,
                    details="registered deterministic workflow completed",
                    checkpoint="CP-005",
                )
                self._engine_completed = True
                from start.review.workflow_coherence import DeterministicExecutionSummary

                cached_results = list(self._results.items())
                self.observer.deterministic_execution_summary(
                    DeterministicExecutionSummary(
                        engine_ids=("CanonicalDeterministicExecutionService",),
                        canonical_input_references=(self._context_id,),
                        operations=(f"execute {self.workflow_id}",),
                        authoritative_outputs=tuple(
                            str(value.get("test_id", "unknown"))
                            for _, value in cached_results
                        ),
                        statuses=tuple(
                            str(value.get("status", "RECORDED"))
                            for _, value in cached_results
                        ),
                        evidence_record_references=tuple(key for key, _ in cached_results),
                        artifact_references=(),
                        limitations=(),
                        skipped_or_not_applicable=(
                            "Human decisions",
                            "Agent decisions",
                            "Grounding",
                        ),
                    ),
                    checkpoint="CP-005",
                )
            self.observer.evidence_batch_summary()
        elif event_type == "governance_decided":
            metadata = dict(getattr(event, "metadata", {}) or {})
            self.observer.governance_card(
                disposition=str(metadata.get("governance_disposition", "NOT_EVALUATED")),
                evidence_count=self.observer.state.evidence_count,
                unresolved_count=int(metadata.get("unresolved_count", 0)),
                validation_failures=int(metadata.get("validation_failures", 0)),
            )
            policy = metadata.get("policy_decision")
            if isinstance(policy, dict):
                self.observer.policy_gate_payload(policy)
        elif event_type == "trace_ready":
            metadata = dict(getattr(event, "metadata", {}) or {})
            records = metadata.get("records", [])
            self.observer.trace_waterfall(records if isinstance(records, list) else [])
        elif event_type == "attestation_created":
            metadata = dict(getattr(event, "metadata", {}) or {})
            self.observer.attestation_sealed(
                merkle_root=str(metadata.get("merkle_root", "")),
                leaf_count=metadata.get("leaf_count"),
            )
        elif event_type == "workflow_completed":
            if self._engine_started and not self._engine_completed:
                self.observer.engine_completed(
                    "CanonicalDeterministicExecutionService",
                    output_count=self.observer.state.evidence_count,
                    details="registered deterministic workflow completed",
                    checkpoint="workflow.completed",
                )
                self._engine_completed = True


def render_temporal_contract(
    *,
    n_sequences: int,
    timesteps: int,
    n_features: int,
    task: str,
    feature_names: Iterable[str] = (),
) -> Panel:
    names = list(feature_names)
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="bold")
    grid.add_column()
    grid.add_row("N", f"{n_sequences:,} sequences")
    grid.add_row("T", f"{timesteps} timesteps")
    grid.add_row("F", f"{n_features} features")
    grid.add_row("Input", f"({n_sequences}, {timesteps}, {n_features}) rank-3 tensor")
    grid.add_row("Task", task)
    if names:
        grid.add_row("Features", ", ".join(names))
    grid.add_row("Semantics", "sequence classification; not forecasting")
    grid.add_row("Sample axis", "independent sequences; not chronological future periods")
    return Panel(grid, title="TEMPORAL CONTRACT", title_align="left", border_style="cyan")


def _short_hash(value: str) -> str:
    if not value:
        return "UNAVAILABLE"
    return f"{value[:12]}…" if len(value) > 12 else value


def _representative_metric(metrics: dict[str, Any]) -> tuple[str, str]:
    for key, value in metrics.items():
        if isinstance(value, bool):
            return key, str(value).lower()
        if isinstance(value, int):
            return key, str(value)
        if isinstance(value, float):
            return key, f"{value:.6g}"
        if isinstance(value, str) and len(value) <= 40:
            return key, value
    return "", ""


def _stage_label(status: str) -> str:
    normalized = status.upper()
    if normalized in {"COMPLETE", "PASSED", "SEALED", "ALLOW", "RECORDED"}:
        return "✓"
    if normalized in {"FAILED", "REJECTED", "DENY", "ERROR", "UNRESOLVED"}:
        return normalized
    return normalized


__all__ = [
    "EvidenceDisplayPolicy",
    "PresentationEvent",
    "PresentationEventKind",
    "PresentationPacing",
    "ReviewPresentationState",
    "RuntimePresentationSink",
    "TerminalReviewObserver",
    "render_temporal_contract",
]
