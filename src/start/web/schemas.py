"""Authoritative Pydantic Schemas for StART v4.5 Web Transport.

Serves as the single authoritative schema source of truth for:
- API requests / responses
- Session & run lifecycle
- Typed SSE event streaming envelopes
- Untrusted browser WebLLM reviewer submissions
- Server-side hydrated findings and metric groundings
- OPA policy decisions and Merkle attestation views
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

START_SCHEMA_VERSION: str = "5.0.0"
START_VERSION: str = "5.1.3"


def get_backend_build_version() -> str:
    """Return runtime-driven backend build version without hardcoding ARM64."""
    return os.environ.get("START_BACKEND_BUILD_VERSION", f"{START_VERSION}-local")


# --------------------------------------------------------------------------- #
# Common / Envelope Schemas
# --------------------------------------------------------------------------- #
class SystemInfo(BaseModel):
    start_version: str = START_VERSION
    start_schema_version: str = START_SCHEMA_VERSION
    backend_build_version: str = Field(default_factory=get_backend_build_version)
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
    synthetic_profile: str | None = None
    synthetic_profile_version: str = "1.0.0"
    seed: int = 42
    turnstile_token: str | None = None
    session_id: str = Field(default_factory=lambda: f"SES-{uuid.uuid4().hex[:12]}")
    workflow: str | None = None
    workflowId: str | None = None
    contextId: str | None = None
    goal: str | None = None
    sourceEvidenceId: str | None = None
    parentRunId: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    parent_run_id: str | None = None
    intervention: str | None = None

    def model_post_init(self, __context: Any) -> None:
        if self.workflowId and not self.workflow:
            self.workflow = self.workflowId
        elif self.workflowId:
            self.workflow = self.workflowId

        if self.contextId and not self.synthetic_profile:
            self.synthetic_profile = self.contextId
        elif self.contextId:
            self.synthetic_profile = self.contextId

        if self.parentRunId and not self.parent_run_id:
            self.parent_run_id = self.parentRunId

        # Derive workflow if not explicitly provided
        if not self.workflow:
            if self.domain == "market" or self.synthetic_profile == "institutional_market_v1":
                self.workflow = "quantitative_finance"
            elif self.domain == "deep_learning" or self.synthetic_profile == "deep_learning_v1":
                self.workflow = "deep_learning"
            else:
                self.workflow = "predictive_ml"

        # Set default synthetic_profile only if none was provided
        if not self.synthetic_profile:
            if self.workflow in ("deep_learning", "neural_diagnostics"):
                self.synthetic_profile = "deep_learning_v1"
            elif self.workflow in ("quantitative_finance", "market_risk", "portfolio_stress"):
                self.synthetic_profile = "institutional_market_v1"
            else:
                self.synthetic_profile = "institutional_credit_v1"

        # Align domain from workflow
        if self.workflow in ("deep_learning", "neural_diagnostics"):
            self.domain = "deep_learning"
        elif self.workflow in ("quantitative_finance", "market_risk", "portfolio_stress"):
            self.domain = "market"
        else:
            self.domain = "predictive"


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
# Untrusted Browser Reviewer Request & Server Hydration Response Schemas
# --------------------------------------------------------------------------- #
class EvidenceCitationRequest(BaseModel):
    """Untrusted client citation request citing an EvidenceRecord.

    Includes optional metric name and claimed value.
    """

    model_config = {"extra": "forbid"}

    evidence_id: str
    metric_name: str | None = None
    client_claimed_value: Any = None


# Compatibility alias for code referencing EvidenceMetricRef in web context
class EvidenceMetricRef(EvidenceCitationRequest):
    pass


class QualitativeFinding(BaseModel):
    """Untrusted qualitative finding submitted by browser WebLLM.

    Client severity is audit-only metadata. The client cannot set canonical_severity.
    """

    model_config = {"extra": "forbid"}

    finding_id: str = Field(default_factory=lambda: f"FND-{uuid.uuid4().hex[:6]}")
    client_proposed_severity: str | None = None
    severity: str | None = None  # optional legacy alias for client_proposed_severity
    title: str
    description: str
    evidence_refs: list[EvidenceCitationRequest] = Field(default_factory=list)
    recommendation: str = ""

    def model_post_init(self, __context: Any) -> None:
        if self.severity and not self.client_proposed_severity:
            self.client_proposed_severity = self.severity


class WebReviewerSubmission(BaseModel):
    """Untrusted payload submitted by browser WebLLM."""

    model_config = {"extra": "forbid"}

    run_id: str
    session_id: str
    model_name: str = "SmolLM2-1.7B-Instruct-q4f16_1-MLC"
    executive_summary: str = ""
    findings: list[QualitativeFinding] = Field(default_factory=list, max_length=20)
    limitations: list[str] = Field(default_factory=list, max_length=20)
    suggested_actions: list[str] = Field(default_factory=list, max_length=20)


class HydratedEvidenceCitation(BaseModel):
    """Server-hydrated authoritative evidence citation."""

    evidence_id: str
    metric_name: str | None = None
    client_claimed_value: Any = None
    canonical_value: Any = None
    server_hydrated_value: Any = None  # alias for canonical_value
    # GROUNDED | UNGROUNDED_EVIDENCE_ID | UNGROUNDED_METRIC_PATH | NON_NUMERIC_METRIC_CLAIM
    grounding_status: str
    grounding_reason: str | None = None
    test_id: str | None = None
    record_status: str | None = None


class HydratedFindingView(BaseModel):
    finding_id: str
    canonical_severity: str | None = None
    client_proposed_severity: str | None = None
    severity: str | None = None  # matches canonical_severity; None if no server severity exists
    title: str
    description: str
    grounded: bool
    evidence_refs: list[HydratedEvidenceCitation | dict[str, Any]] = Field(default_factory=list)
    recommendation: str = ""


class ReviewerHydrationResponse(BaseModel):
    run_id: str
    schema_version: str = START_SCHEMA_VERSION
    model_name: str
    gate_status: Literal["ACCEPTED", "BLOCKED", "ERROR"] = "BLOCKED"
    is_grounded: bool = False
    all_grounded: bool = False
    hydrated_findings: list[HydratedFindingView] = Field(default_factory=list)
    opa_policy_decision: Literal["ALLOW", "WARN", "BLOCK", "DENY", "ERROR"] | None = None
    opa_reasons: list[str] = Field(default_factory=list)
    governance_disposition: (
        Literal["ACCEPT", "ACCEPT_WITH_CONDITIONS", "REMEDIATION_REQUIRED", "REJECT"] | None
    ) = None
    attestation_seal_merkle_root: str | None = None
    attestation_timestamp: float = Field(default_factory=time.time)

