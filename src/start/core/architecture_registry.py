"""Truthful, Machine-Readable Architecture Capability Registry for StART.

Strict Invariants:
1. No Exaggeration / No Marketing: Classifies every component strictly by its verified runtime capabilities.
2. Clear Classification Taxonomy:
   - PROVEN_ADVANCED: Core functionality fully verified and running in runtime with comprehensive test suites.
   - OPTIONAL_ADVANCED: Production-grade optional module activating when configured/installed; core operates independently.
   - OPTIONAL_FUNCTIONAL: Fully implemented functional adapter for enterprise ecosystem integration.
   - ADAPTER_ONLY: Typed boundary adapter provided for external service integration.
   - LEGACY_ONLY: Historical reference module preserved for backward compatibility.
   - NOT_INSTALLED: Package not installed in current environment; graceful deterministic fallback active.
3. Privacy & Offline Invariance: Core StART operation runs 100% locally and deterministically without network dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rich.table import Table


@dataclass(frozen=True)
class CapabilityRecord:
    """Audit-grade record describing an architectural integration."""

    name: str
    component_type: str  # "Deep Learning" | "Orchestration" | "Telemetry" | "Policy" | "Security" | "Adapter"
    classification: str  # "PROVEN_ADVANCED" | "OPTIONAL_ADVANCED" | "OPTIONAL_FUNCTIONAL" | "ADAPTER_ONLY"
    purpose: str
    implementation_location: str
    runtime_profile: str
    privacy_behavior: str
    test_verification: str
    fallback_behavior: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CAPABILITY_REGISTRY: tuple[CapabilityRecord, ...] = (
    CapabilityRecord(
        name="Deep Learning Institutional UX",
        component_type="Deep Learning",
        classification="PROVEN_ADVANCED",
        purpose="Complete PyTorch tabular DL architecture inspection, layer-by-layer summaries, loss convergence, tuning, ECE calibration, SHAP, and vector SVG artifacts.",
        implementation_location="src/start/modeling/tabular_dl.py, src/start/modeling/dl_artifacts.py, src/start/review/tables.py",
        runtime_profile="All profiles (Local PyTorch CPU/MPS/CUDA)",
        privacy_behavior="100% in-process local execution, deterministic seed synchronization, zero data egress.",
        test_verification="tests/test_v440_institutional_ux_and_architecture.py, tests/test_v441_advanced_langgraph.py",
        fallback_behavior="Deterministic sklearn linear/tree baseline with full metric equivalence.",
    ),
    CapabilityRecord(
        name="StateGraph / LangGraph Runtime",
        component_type="Orchestration",
        classification="PROVEN_ADVANCED",
        purpose="Compiled StateGraph runtime with typed state, conditional edge routing, checkpointer persistence (MemorySaver/FileSaver), resumability, and bounded retry.",
        implementation_location="src/start/orchestration/state_graph.py, src/start/orchestration/cyclic_executor.py",
        runtime_profile="All profiles (Core, Enterprise, Public)",
        privacy_behavior="100% local, zero data egress, state snapshots cryptographically hashed and persisted locally.",
        test_verification="tests/test_v441_advanced_langgraph.py, tests/test_v403_graph.py",
        fallback_behavior="Deterministic linear/cyclic graph execution with full state isolation.",
    ),
    CapabilityRecord(
        name="OpenTelemetry Tracing",
        component_type="Telemetry",
        classification="PROVEN_ADVANCED",
        purpose="Hierarchical span model (run -> checkpoint -> agent -> tool -> evidence -> governance -> attestation) with automated secret/credential redaction.",
        implementation_location="src/start/telemetry/otel_tracer.py, src/start/orchestration/events.py",
        runtime_profile="All profiles (local in-memory span collector active by default)",
        privacy_behavior="Local-only by default; sanitized span attributes; API keys, passwords, and raw datasets strictly redacted.",
        test_verification="tests/test_v441_ai_engineering.py",
        fallback_behavior="In-memory trace collector with zero network egress.",
    ),
    CapabilityRecord(
        name="Open Policy Agent (OPA)",
        component_type="Policy",
        classification="PROVEN_ADVANCED",
        purpose="Fail-closed policy control plane enforcing Rego rules for network egress, tool allowlists, agent permissions, artifact export, and governance sign-off.",
        implementation_location="src/start/policies/opa_policy_plane.py",
        runtime_profile="All profiles (in-process policy evaluator active; optional external daemon)",
        privacy_behavior="Local evaluation; policy rules evaluated in-process with cryptographic provenance.",
        test_verification="tests/test_v441_ai_engineering.py",
        fallback_behavior="Local deterministic rule engine enforcing strict fail-closed security.",
    ),
    CapabilityRecord(
        name="LangSmith Tracer",
        component_type="Telemetry",
        classification="OPTIONAL_ADVANCED",
        purpose="Optional external telemetry exporter for LLM reasoning traces and latency tracking over canonical event model.",
        implementation_location="src/start/ai_engineering/langsmith_tracer.py",
        runtime_profile="Enterprise / Public (when LANGSMITH_API_KEY is configured)",
        privacy_behavior="Disabled by default; inputs/outputs sanitized; never required for core operation.",
        test_verification="tests/test_langsmith_tracer.py",
        fallback_behavior="Complete no-op; core execution runs without external telemetry.",
    ),
    CapabilityRecord(
        name="NeMo Guardrails",
        component_type="Security",
        classification="OPTIONAL_ADVANCED",
        purpose="Optional input/output safety boundary for prompt injection defense, unauthorized tool blocking, and EvidenceRecord immutability enforcement.",
        implementation_location="src/start/ai_engineering/nemo_guardrails.py",
        runtime_profile="Enterprise / Public LLM mode",
        privacy_behavior="In-process guardrail evaluation; mathematically forbidden from mutating numerical EvidenceRecords.",
        test_verification="tests/test_v441_ai_engineering.py",
        fallback_behavior="Deterministic schema validation and regex grounding gates.",
    ),
    CapabilityRecord(
        name="MCP Server Integration",
        component_type="Adapter",
        classification="OPTIONAL_ADVANCED",
        purpose="Standardized Model Context Protocol adapter for external tool discovery, typed interface inspection, and capability validation.",
        implementation_location="src/start/ai_engineering/adapters.py",
        runtime_profile="Enterprise",
        privacy_behavior="Tool interfaces typed; data exchange constrained to declared parameters.",
        test_verification="tests/test_ai_engineering_adapters.py",
        fallback_behavior="Internal deterministic tool registry with 79 verified analytical tools.",
    ),
    CapabilityRecord(
        name="Garak LLM Vulnerability Scanner",
        component_type="Security",
        classification="OPTIONAL_FUNCTIONAL",
        purpose="Automated LLM vulnerability probing and adversarial prompt evaluation harness.",
        implementation_location="src/start/ai_engineering/adapters.py",
        runtime_profile="Validation / Security Audit",
        privacy_behavior="Local test harnesses; synthetic probe inputs only.",
        test_verification="tests/test_ai_engineering_adapters.py",
        fallback_behavior="Static guardrail validation and prompt confinement tests.",
    ),
    CapabilityRecord(
        name="Promptfoo / DeepEval",
        component_type="Adapter",
        classification="OPTIONAL_FUNCTIONAL",
        purpose="Unit-testing and regression evaluation harnesses for LLM prompt variations.",
        implementation_location="src/start/ai_engineering/adapters.py",
        runtime_profile="Development / Validation",
        privacy_behavior="Synthetic test fixtures only; zero sensitive client data.",
        test_verification="tests/test_ai_engineering_adapters.py",
        fallback_behavior="In-tree deterministic test suites and mock provider fixtures.",
    ),
    CapabilityRecord(
        name="Langfuse / Phoenix",
        component_type="Telemetry",
        classification="OPTIONAL_FUNCTIONAL",
        purpose="Optional trace capture and observability adapters consuming the unified event model.",
        implementation_location="src/start/ai_engineering/adapters.py",
        runtime_profile="Enterprise / Observability",
        privacy_behavior="Local trace capture; cloud export disabled by default.",
        test_verification="tests/test_ai_engineering_adapters.py",
        fallback_behavior="In-memory OpenTelemetry tracer.",
    ),
)


def get_architecture_capability_table() -> Table:
    """Build a rich table presenting the verified architecture capabilities."""
    table = Table(
        title="StART Architecture & AI-Engineering Capability Registry",
        title_style="bold cyan",
        header_style="bold",
        show_lines=False,
    )
    table.add_column("Component", style="bold white", no_wrap=True)
    table.add_column("Type", style="cyan")
    table.add_column("Classification", justify="center")
    table.add_column("Verified Runtime Purpose", style="dim")
    table.add_column("Privacy & Offline Behavior", style="dim")

    badge_colors = {
        "PROVEN_ADVANCED": "bold green",
        "OPTIONAL_ADVANCED": "bold blue",
        "OPTIONAL_FUNCTIONAL": "cyan",
        "ADAPTER_ONLY": "yellow",
        "NOT_INSTALLED": "dim",
    }

    for cap in CAPABILITY_REGISTRY:
        color = badge_colors.get(cap.classification, "white")
        table.add_row(
            cap.name,
            cap.component_type,
            f"[{color}]{cap.classification}[/{color}]",
            cap.purpose,
            cap.privacy_behavior,
        )
    return table


def get_architecture_registry_dict() -> dict[str, Any]:
    """Return machine-readable dictionary representation of capability registry."""
    return {
        "capabilities": [cap.to_dict() for cap in CAPABILITY_REGISTRY],
        "count": len(CAPABILITY_REGISTRY),
    }
