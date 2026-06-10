"""Typed schemas for StART.

Everything that crosses an agent or provider boundary is a Pydantic v2 model.
Evidence records are the canonical audit artifact: deterministic engines
produce ``TestResult`` objects, the execution layer enriches them into
``EvidenceRecord`` objects, and the ledger persists them with content hashes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Status(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    ERROR = "error"
    SKIPPED = "skipped"


class TaskType(str, Enum):
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


class ComputeDevice(str, Enum):
    CUDA = "cuda"
    MPS = "mps"
    CPU = "cpu"


class Materiality(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


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
    """Raw, deterministic output of a test engine. No LLM content allowed."""

    __test__ = False  # not a pytest class
    model_config = ConfigDict(protected_namespaces=())

    test_id: str
    test_name: str
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    thresholds: list[ThresholdSpec] = Field(default_factory=list)
    status: Status = Status.PASS
    params: dict[str, Any] = Field(default_factory=dict)
    interpretation: str = ""
    limitations: list[str] = Field(default_factory=list)
    artifacts: dict[str, str] = Field(
        default_factory=dict, description="Name -> path/URI of generated artifacts."
    )

    def apply_thresholds(self) -> TestResult:
        worst = Status.PASS
        order = {Status.PASS: 0, Status.WARN: 1, Status.FAIL: 2}
        for spec in self.thresholds:
            value = self.metrics.get(spec.metric)
            if isinstance(value, (int, float)):
                outcome = spec.evaluate(float(value))
                if order[outcome] > order[worst]:
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
    timestamp: datetime = Field(default_factory=_utcnow)
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
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
        model_id: str,
        dataset_id: str,
        run_id: str,
        input_artifact_hash: str | None = None,
        policy_hash: str | None = None,
        repro: ReproducibilityMeta | None = None,
    ) -> EvidenceRecord:
        return cls(
            test_id=result.test_id,
            test_name=result.test_name,
            model_id=model_id,
            dataset_id=dataset_id,
            run_id=run_id,
            params=result.params,
            metrics=result.metrics,
            thresholds=result.thresholds,
            status=result.status,
            interpretation=result.interpretation,
            limitations=result.limitations,
            input_artifact_hash=input_artifact_hash,
            policy_hash=policy_hash,
            repro=repro or ReproducibilityMeta(),
            artifacts=result.artifacts,
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
    generator: str = "template"
    cited_evidence_ids: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    run_id: str
    plan: ValidationPlan
    evidence: list[EvidenceRecord] = Field(default_factory=list)
    critique: CritiqueResult | None = None
    narrative: Narrative | None = None
    policy: PolicyDecision | None = None
