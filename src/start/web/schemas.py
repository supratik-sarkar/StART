"""Authoritative Pydantic Schemas for StART v4.5 Web Transport.

Serves as the single authoritative schema source of truth for:
- API requests / responses
- Session & run lifecycle
- Typed SSE event streaming envelopes
- Untrusted browser WebLLM reviewer submissions
- Server-side hydrated findings and metric groundings
- OPA policy decisions and Merkle attestation views
- Zero-cost deployment attestations
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

START_SCHEMA_VERSION: str = "4.6.0"
START_VERSION: str = "4.6.2"


# --------------------------------------------------------------------------- #
# Common / Envelope Schemas
# --------------------------------------------------------------------------- #
class SystemInfo(BaseModel):
    start_version: str = START_VERSION
    start_schema_version: str = START_SCHEMA_VERSION
    backend_build_version: str = "4.6.0-arm64-prod"
    git_sha: str | None = None
    compute_runtime: str = "local"  # "local" | "oracle_a1_arm64"
    max_concurrency: int = 1
    engine_status: Literal["READY", "BUSY", "MAINTENANCE"] = "READY"
    supported_domains: list[str] = Field(default_factory=lambda: ["predictive", "deep_learning", "market"])
    synthetic_profiles: list[str] = Field(
        default_factory=lambda: [
            "institutional_credit_v1",
            "deep_learning_v1",
            "institutional_market_v1",
        ]
    )


class APIResponseEnvelope(BaseModel):
    success: bool = True
    schema_version: str = START_SCHEMA_VERSION
    run_id: str | None = None
    timestamp: float = Field(default_factory=time.time)
    data: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    error_code: str | None = None


class SSEEnvelope(BaseModel):
    """Typed Server-Sent Event envelope bridging canonical RuntimeEvents with truthful progress."""

    event_id: str = Field(default_factory=lambda: f"EVT-{uuid.uuid4().hex[:8]}")
    sequence: int = 0
    run_id: str
    timestamp: float = Field(default_factory=time.time)
    # event_type: agent_transition | tool_execution | policy_decision
    #             evidence_commit | artifact_generate | governance_seal | progress_update
    event_type: str = "agent_transition"
    schema_version: str = START_SCHEMA_VERSION
    source_agent: str = "Director"
    target_agent: str = "DeterministicEngine"
    stage: str = "PLANNING"
    action: str = ""
    status: str = "RUNNING"  # RUNNING | SUCCESS | FAILED
    latency_ms: float = 0.0
    evidence_refs: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    policy_decision: str = "ALLOW"
    payload: dict[str, Any] = Field(default_factory=dict)

    # Truthful progress fields
    phase: str = ""
    step: int = 0
    completed: int = 0
    total: int = 0
    percent: float = 0.0
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float | None = None
    message: str = ""


# --------------------------------------------------------------------------- #
# Run Request & Session Management
# --------------------------------------------------------------------------- #
class RunRequest(BaseModel):
    """Request to initiate a deterministic StART analytical run."""

    domain: Literal["predictive", "deep_learning", "market"] = "predictive"
    mode: Literal["deterministic", "llm"] = "deterministic"
    materiality: Literal["low", "medium", "high"] = "high"
    lifecycle: Literal["pre_implementation", "validation", "annual_review", "monitoring"] = "validation"
    synthetic_profile: str = "institutional_credit_v1"
    synthetic_profile_version: str = "1.0.0"
    seed: int = 42
    turnstile_token: str | None = None
    session_id: str = Field(default_factory=lambda: f"SES-{uuid.uuid4().hex[:12]}")
    workflow: str = "predictive_ml"
    parameters: dict[str, Any] = Field(default_factory=dict)
    parent_run_id: str | None = None
    intervention: str | None = None


class RunStatusResponse(BaseModel):
    run_id: str
    session_id: str
    status: Literal[
        "CONFIGURING",
        "VALIDATING",
        "QUEUED",
        "INITIALIZING",
        "RUNNING",
        "PARTIAL",
        "COMPLETED",
        "RECOVERABLE_FAILURE",
        "FAILED",
        "BUSY",
    ]
    domain: str
    synthetic_profile: str
    created_at: float
    completed_at: float | None = None
    event_count: int = 0
    evidence_count: int = 0
    artifact_count: int = 0
    error_message: str | None = None
    error_code: str | None = None
    queue_position: int = 0
    phase: str = ""
    step: int = 0
    completed: int = 0
    total: int = 0
    percent: float = 0.0
    elapsed_seconds: float = 0.0


# --------------------------------------------------------------------------- #
# Presentation & Artifact Schemas
# --------------------------------------------------------------------------- #
class MetricRowView(BaseModel):
    test_id: str
    metric: str
    value: Any
    unit: str = ""
    status: str = "PASS"
    evidence_id: str = ""
    artifact_id: str | None = None


class PresentationBlockView(BaseModel):
    block_id: str
    title: str
    domain: str
    rows: list[MetricRowView] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)


class ReviewPresentationExport(BaseModel):
    run_id: str
    mode: str
    domains: list[str]
    materiality: str
    lifecycle: str
    governance_disposition: str
    attestation_seal_merkle_root: str
    blocks: dict[str, PresentationBlockView] = Field(default_factory=dict)
    orchestration_events: list[dict[str, Any]] = Field(default_factory=list)


class LogicalArtifactMetadata(BaseModel):
    artifact_id: str
    run_id: str
    title: str
    artifact_type: Literal["svg", "json", "html", "pdf", "table", "csv"]
    evidence_ids: list[str] = Field(default_factory=list)
    description: str = ""
    size_bytes: int = 0
    sha256: str = ""


# --------------------------------------------------------------------------- #
# Untrusted Browser Reviewer & Server Hydration Schemas
# --------------------------------------------------------------------------- #
class EvidenceMetricRef(BaseModel):
    evidence_id: str
    metric_name: str
    # Note: Client may propose finding text, but authoritative numeric value is hydrated server-side.
    client_claimed_value: Any = None
    server_hydrated_value: Any = None


class QualitativeFinding(BaseModel):
    finding_id: str = Field(default_factory=lambda: f"FND-{uuid.uuid4().hex[:6]}")
    severity: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = "MEDIUM"
    title: str
    description: str
    evidence_refs: list[EvidenceMetricRef] = Field(default_factory=list)
    recommendation: str = ""


class WebReviewerSubmission(BaseModel):
    """Untrusted payload submitted by browser WebLLM."""

    run_id: str
    session_id: str
    model_name: str = "Llama-3.2-1B-Instruct-q4f32_1-MLC"
    executive_summary: str = ""
    findings: list[QualitativeFinding] = Field(default_factory=list, max_length=20)
    limitations: list[str] = Field(default_factory=list, max_length=20)
    suggested_actions: list[str] = Field(default_factory=list, max_length=20)


class HydratedFindingView(BaseModel):
    finding_id: str
    severity: str
    title: str
    description: str
    grounded: bool
    evidence_refs: list[dict[str, Any]] = Field(default_factory=list)
    recommendation: str = ""


class ReviewerHydrationResponse(BaseModel):
    run_id: str
    schema_version: str = START_SCHEMA_VERSION
    model_name: str
    is_grounded: bool
    hydrated_findings: list[HydratedFindingView] = Field(default_factory=list)
    opa_policy_decision: Literal["ALLOW", "WARN", "BLOCK", "DENY"] = "ALLOW"
    opa_reasons: list[str] = Field(default_factory=list)
    governance_disposition: Literal["ACCEPT", "CONDITIONAL_ACCEPT", "REJECT"] = "ACCEPT"
    attestation_seal_merkle_root: str = ""
    attestation_timestamp: float = Field(default_factory=time.time)


# --------------------------------------------------------------------------- #
# Zero-Cost Provisioning Attestation Schema
# --------------------------------------------------------------------------- #
class ZeroCostAttestation(BaseModel):
    provider: str = "Oracle Cloud Infrastructure"
    resource_type: str = "Compute Instance"
    tier_shape: str = "VM.Standard.A1.Flex (2 OCPU / 12 GB RAM)"
    oci_a1_ocpu: int = 2
    oci_a1_memory_gb: int = 12
    within_always_free_allowance: Literal["YES", "NO"] = "YES"
    expected_recurring_charge: float = 0.0
    always_free_eligible: bool = True
    recurring_monthly_charge_usd: float = 0.0
    storage_tier: str = "Always Free Boot Volume (50 GB)"
    network_egress_allowance: str = "10 TB / month (Always Free)"
    cloudflare_plan: str = "Free Tier ($0/month)"
    huggingface_space_sdk: str = "static (Free)"
    verification_timestamp: float = Field(default_factory=time.time)
    attestation_status: Literal["VERIFIED_ZERO_COST", "NON_FREE_BLOCKED"] = "VERIFIED_ZERO_COST"
