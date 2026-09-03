"""Typed schemas for StART.

Everything that crosses an agent or provider boundary is a Pydantic v2 model.
Evidence records are the canonical audit artifact: deterministic engines
produce ``TestResult`` objects, the execution layer enriches them into
``EvidenceRecord`` objects, and the ledger persists them with content hashes.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Status(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    ERROR = "error"
    SKIPPED = "skipped"
    INFORMATIONAL = "informational"
    RECORDED = "recorded"


class TaskType(StrEnum):
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    RANKING = "ranking"
    RECOMMENDER = "recommender"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    PERFORMANCE_ATTRIBUTION = "performance_attribution"
    DEEP_LEARNING = "deep_learning"
    GENAI = "genai"


class ComputeDevice(StrEnum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class Materiality(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def new_evidence_id() -> str:
    return f"EV-{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Test layer
# --------------------------------------------------------------------------- #
class ThresholdSpec(BaseModel):
    """A single threshold applied to a metric, with directionality."""

    metric: str
    warn: float | None = None
    fail: float | None = None
    direction: str = Field(
        default="upper",
        description="'upper' = breach when metric > threshold; 'lower' = breach when metric < threshold.",
    )

    def evaluate(self, value: float) -> Status:
        def breached(limit: float | None) -> bool:
            if limit is None:
                return False
            return value > limit if self.direction == "upper" else value < limit

        if breached(self.fail):
            return Status.FAIL
        if breached(self.warn):
            return Status.WARN
        return Status.PASS


class TestResult(BaseModel):
    """Output of a single deterministic test invocation."""

    __test__ = False  # not a pytest class

    test_id: str
    test_name: str
    status: Status = Status.PASS
    metrics: dict[str, bool | float | int | str | None] = Field(default_factory=dict)
    thresholds: list[ThresholdSpec] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    interpretation: str = ""
    limitations: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(
        default_factory=dict, description="Name -> path/URI of generated artifacts."
    )

    def apply_thresholds(self) -> TestResult:
        if not self.thresholds:
            return self
        worst = Status.PASS
        order = {
            Status.RECORDED: -1,
            Status.INFORMATIONAL: -1,
            Status.SKIPPED: 0,
            Status.PASS: 0,
            Status.WARN: 1,
            Status.FAIL: 2,
            Status.ERROR: 3,
        }
        for spec in self.thresholds:
            value = self.metrics.get(spec.metric)
            if isinstance(value, (int, float)):
                outcome = spec.evaluate(float(value))
                if order.get(outcome, 0) > order.get(worst, 0):
                    worst = outcome
        self.status = worst
        return self


class ReproducibilityMeta(BaseModel):
    seed: int | None = None
    device: ComputeDevice = ComputeDevice.CPU
    python_version: str = ""
    package_versions: dict[str, str] = Field(default_factory=dict)
    git_sha: str | None = None
    runtime: str = "local"


class EvidenceRecord(BaseModel):
    """Audit-grade wrapper around a TestResult. Persisted to the ledger."""

    model_config = ConfigDict(protected_namespaces=())

    evidence_id: str = Field(default_factory=new_evidence_id)
    test_id: str
    test_name: str
    model_id: str
    dataset_id: str
    run_id: str
    enterprise_run_id: str | None = None
    timestamp: datetime = Field(default_factory=_utcnow)
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, bool | float | int | str | None] = Field(default_factory=dict)
    thresholds: list[ThresholdSpec] = Field(default_factory=list)
    status: Status
    interpretation: str = ""
    limitations: list[str] = Field(default_factory=list)
    input_artifact_hash: str | None = None
    policy_hash: str | None = None
    repro: ReproducibilityMeta = Field(default_factory=ReproducibilityMeta)
    artifacts: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        result: TestResult,
        *,
        model_id: str = "MOD-DEFAULT",
        dataset_id: str = "DS-DEFAULT",
        run_id: str = "RUN-DEFAULT",
        enterprise_run_id: str | None = None,
        input_artifact_hash: str | None = None,
        policy_hash: str | None = None,
        repro: ReproducibilityMeta | None = None,
    ) -> EvidenceRecord:
        norm_metrics = dict(result.metrics) if result.metrics else {}
        if result.test_id in {
            "traded_risk.var_kupiec_pof",
            "traded_risk.var_christoffersen_independence",
            "traded_risk.var_christoffersen_conditional",
        }:
            # Evidence bridge normalization: map statistical alpha to gamma_test with provenance
            if "gamma_test" not in norm_metrics and "statistical_gamma_test" not in norm_metrics:
                if "alpha" in norm_metrics:
                    norm_metrics["gamma_test"] = norm_metrics["alpha"]
                    norm_metrics["statistical_gamma_test"] = norm_metrics["alpha"]
                    norm_metrics["statistical_criterion_source"] = "STATISTICAL_TEST_SPECIFICATION"
                elif result.params and "alpha" in result.params:
                    norm_metrics["gamma_test"] = result.params["alpha"]
                    norm_metrics["statistical_gamma_test"] = result.params["alpha"]
                    norm_metrics["statistical_criterion_source"] = "STATISTICAL_TEST_SPECIFICATION"
            # Separate VaR tail probability from test significance (never normalize into gamma_test)
            if "alpha_var" not in norm_metrics and "expected_probability" in norm_metrics:
                norm_metrics["alpha_var"] = norm_metrics["expected_probability"]

        return cls(
            test_id=result.test_id,
            test_name=result.test_name,
            model_id=model_id,
            dataset_id=dataset_id,
            run_id=run_id,
            enterprise_run_id=enterprise_run_id,
            params=result.params,
            metrics=norm_metrics,
            thresholds=result.thresholds,
            status=result.status,
            interpretation=result.interpretation,
            limitations=result.limitations,
            artifacts=result.artifacts,
            input_artifact_hash=input_artifact_hash,
            policy_hash=policy_hash,
            repro=repro or ReproducibilityMeta(),
        )


# --------------------------------------------------------------------------- #
# Planning / agent message layer
# --------------------------------------------------------------------------- #
class ModelMetadata(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    task_type: TaskType
    materiality: Materiality = Materiality.MEDIUM
    description: str = ""
    target_column: str | None = None
    prediction_column: str | None = None
    score_column: str | None = None


class DatasetSummary(BaseModel):
    dataset_id: str
    n_rows: int = 0
    n_columns: int = 0
    columns: list[str] = Field(default_factory=list)
    source: str = ""


class PlannedTest(BaseModel):
    test_id: str
    reason: str = ""
    params: dict[str, Any] = Field(default_factory=dict)


class ValidationPlan(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    plan_id: str = Field(default_factory=lambda: f"PLAN-{uuid.uuid4().hex[:8]}")
    model_id: str
    dataset_id: str
    task_type: TaskType
    materiality: Materiality
    planned_tests: list[PlannedTest] = Field(default_factory=list)
    planner: str = "rule_based"
    notes: str = ""


class CritiqueIssue(BaseModel):
    severity: str = "warn"  # warn | block
    code: str
    message: str
    evidence_id: str | None = None


class CritiqueResult(BaseModel):
    ok: bool
    issues: list[CritiqueIssue] = Field(default_factory=list)


class PolicyDecision(BaseModel):
    allowed: bool
    reasons: list[str] = Field(default_factory=list)
    policy_hash: str | None = None


class Narrative(BaseModel):
    """Reviewer-facing narrative. Every quantitative claim must cite [EV-...]."""

    run_id: str
    summary: str
    findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    signoff: str = ""
    generator: str = "template"
    cited_evidence_ids: list[str] = Field(default_factory=list)


class AgentReview(BaseModel):
    """Output of the dual-mode agent review flow. In deterministic mode the
    sections come from rules/templates; in LLM mode they come from the
    configured provider, constrained to the evidence bundle and gated by
    EvidenceCriticAgent. `rejected_sections` lists sections whose LLM output
    failed critique twice and was replaced by the deterministic fallback."""

    mode: Literal["deterministic", "llm"] = "deterministic"
    llm_provider: str = "none"
    review_plan: list[str] = Field(default_factory=list)
    suggested_tests: list[str] = Field(default_factory=list)
    findings: list[str] = Field(default_factory=list)
    challenge_memo: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    governance: list[str] = Field(default_factory=list)
    signoff: str = ""
    critique_ok: bool = True
    rejected_sections: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    run_id: str
    plan: ValidationPlan
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    critique: CritiqueResult | None = None
    narrative: Narrative | None = None
    agent_review: AgentReview | None = None
    policy: PolicyDecision | None = None


class VisualArtifact(BaseModel):
    """Audit-grade visualization/tabular artifact record with Merkle evidence linkage."""

    artifact_id: str
    title: str
    artifact_type: str = "svg"
    file_path: str
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple)
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
