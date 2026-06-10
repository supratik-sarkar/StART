"""Agentic orchestration roles.

Design rule: agents plan, route, critique, and narrate — they never compute
metrics. Every agent has a deterministic (no-LLM) code path so a full run is
possible with ``NoLLMProvider``. When an LLM is available it may *propose*
(plans, narratives), but proposals are validated against the registry and the
citation policy before acceptance.
"""

from __future__ import annotations

import re
import traceback
from typing import Any

from start.core.config import PolicyConfig, StartConfig
from start.core.hashing import (
    current_git_sha,
    package_versions,
    python_version,
)
from start.core.schemas import (
    CritiqueIssue,
    CritiqueResult,
    DatasetSummary,
    EvidenceRecord,
    Materiality,
    ModelMetadata,
    Narrative,
    PlannedTest,
    PolicyDecision,
    ReproducibilityMeta,
    Status,
    TaskType,
    TestResult,
    ValidationPlan,
)
from start.providers.base import ComputeProvider, LLMProvider
from start.registry import TestContext, get_test, list_tests

EV_CITATION_RE = re.compile(r"\[EV-[A-Za-z0-9]+\]")
_NUMERIC_RE = re.compile(r"\d")
# Identifiers (run/plan/evidence IDs) contain digits but are not
# quantitative claims; strip them before numeric-claim detection.
_ID_TOKEN_RE = re.compile(r"\b(?:RUN|PLAN|EV|SRC|DOC)-[A-Za-z0-9\-]+\b")

_FAMILIES_BY_TASK: dict[TaskType, list[str]] = {
    TaskType.BINARY_CLASSIFICATION: ["preprocessing", "supervised", "xai"],
    TaskType.MULTICLASS_CLASSIFICATION: ["preprocessing", "supervised", "xai"],
    TaskType.REGRESSION: ["preprocessing", "xai"],
    TaskType.CLUSTERING: ["preprocessing", "unsupervised"],
    TaskType.RECOMMENDER: ["preprocessing", "recommender"],
    TaskType.RANKING: ["preprocessing", "recommender"],
    TaskType.PORTFOLIO_OPTIMIZATION: ["preprocessing", "portfolio"],
    TaskType.PERFORMANCE_ATTRIBUTION: ["preprocessing", "attribution"],
    TaskType.DEEP_LEARNING: ["preprocessing", "deep_learning", "xai"],
    TaskType.GENAI: ["genai"],
}


class ReviewPlannerAgent:
    """Produces a ValidationPlan from model metadata + dataset summary.

    Deterministic path: select all registered tests in the families mapped to
    the task type, filtered by enabled/disabled families in config. LLM path
    (when available) may re-order and annotate reasons, but may only reference
    registered test IDs.
    """

    def __init__(self, config: StartConfig, llm: LLMProvider) -> None:
        self.config = config
        self.llm = llm

    def plan(self, model_meta: ModelMetadata, dataset: DatasetSummary) -> ValidationPlan:
        families = _FAMILIES_BY_TASK.get(model_meta.task_type, ["preprocessing"])
        enabled = set(self.config.test_families.enabled)
        disabled = set(self.config.test_families.disabled)
        families = [f for f in families if f in enabled and f not in disabled]
        planned: list[PlannedTest] = []
        for family in families:
            for spec in list_tests(family):
                params = dict(spec.default_params)
                params.update(self.config.test_families.overrides.get(spec.test_id, {}))
                planned.append(
                    PlannedTest(
                        test_id=spec.test_id,
                        reason=f"Registered {family} check applicable to {model_meta.task_type.value}.",
                        params=params,
                    )
                )
        return ValidationPlan(
            model_id=model_meta.model_id,
            dataset_id=dataset.dataset_id,
            task_type=model_meta.task_type,
            materiality=model_meta.materiality,
            planned_tests=planned,
            planner="rule_based" if not self.llm.available else "rule_based+llm_annotated",
        )


class TestRouterAgent:
    """Maps planned tests to registered deterministic engines; drops unknowns."""

    def route(self, plan: ValidationPlan) -> tuple[ValidationPlan, list[str]]:
        routed: list[PlannedTest] = []
        unknown: list[str] = []
        for item in plan.planned_tests:
            try:
                get_test(item.test_id)
                routed.append(item)
            except KeyError:
                unknown.append(item.test_id)
        plan.planned_tests = routed
        return plan, unknown


class ExecutionAgent:
    """Runs deterministic engines through a compute provider.

    Must not invent metrics: on engine failure it emits an ERROR evidence
    record with the traceback rather than substituting values.
    """

    def __init__(self, compute: ComputeProvider, policy_hash: str | None, run_id: str) -> None:
        self.compute = compute
        self.policy_hash = policy_hash
        self.run_id = run_id

    def _repro(self, seed: int) -> ReproducibilityMeta:
        return ReproducibilityMeta(
            seed=seed,
            device=self.compute.device(),
            python_version=python_version(),
            package_versions=package_versions(),
            git_sha=current_git_sha(),
            runtime=self.compute.name,
        )

    def execute(
        self,
        plan: ValidationPlan,
        ctx: TestContext,
        *,
        input_artifact_hash: str | None = None,
    ) -> list[EvidenceRecord]:
        records: list[EvidenceRecord] = []
        for item in plan.planned_tests:
            spec = get_test(item.test_id)
            try:
                result: TestResult = self.compute.run(spec.fn, ctx, **item.params)
            except Exception as exc:  # noqa: BLE001 - converted to ERROR evidence
                result = TestResult(
                    test_id=item.test_id,
                    test_name=spec.name,
                    status=Status.ERROR,
                    params=item.params,
                    interpretation=f"Engine raised {type(exc).__name__}: {exc}",
                    limitations=[traceback.format_exc(limit=3)],
                )
            records.append(
                EvidenceRecord.from_result(
                    result,
                    model_id=plan.model_id,
                    dataset_id=plan.dataset_id,
                    run_id=self.run_id,
                    input_artifact_hash=input_artifact_hash,
                    policy_hash=self.policy_hash,
                    repro=self._repro(ctx.seed),
                )
            )
        return records


class EvidenceCriticAgent:
    """Checks evidence completeness and narrative citation discipline."""

    REQUIRED_FIELDS = ("test_id", "model_id", "dataset_id", "run_id", "status")

    def critique_evidence(self, records: list[EvidenceRecord]) -> CritiqueResult:
        issues: list[CritiqueIssue] = []
        for rec in records:
            data = rec.model_dump()
            for field in self.REQUIRED_FIELDS:
                if not data.get(field):
                    issues.append(
                        CritiqueIssue(
                            severity="block",
                            code="missing_field",
                            message=f"Evidence missing required field '{field}'.",
                            evidence_id=rec.evidence_id,
                        )
                    )
            if rec.status not in (Status.SKIPPED, Status.ERROR) and not rec.metrics:
                issues.append(
                    CritiqueIssue(
                        severity="block",
                        code="no_metrics",
                        message="Completed test produced no metrics.",
                        evidence_id=rec.evidence_id,
                    )
                )
            if rec.policy_hash is None:
                issues.append(
                    CritiqueIssue(
                        severity="warn",
                        code="no_policy_hash",
                        message="Evidence not stamped with a policy hash.",
                        evidence_id=rec.evidence_id,
                    )
                )
            if not rec.interpretation:
                issues.append(
                    CritiqueIssue(
                        severity="warn",
                        code="no_interpretation",
                        message="Evidence lacks an interpretation string.",
                        evidence_id=rec.evidence_id,
                    )
                )
        return CritiqueResult(ok=not any(i.severity == "block" for i in issues), issues=issues)

    def critique_narrative(
        self, narrative: Narrative, records: list[EvidenceRecord]
    ) -> CritiqueResult:
        issues: list[CritiqueIssue] = []
        valid_ids = {r.evidence_id for r in records}
        texts = [narrative.summary, *narrative.findings, *narrative.limitations]
        for text in texts:
            # Don't split before a citation bracket: "AUC is 0.81. [EV-x]"
            # must remain one sentence so the claim keeps its citation.
            for sentence in re.split(r"(?<=[.!?])\s+(?!\[)", text):
                if not sentence.strip():
                    continue
                cited = EV_CITATION_RE.findall(sentence)
                for cite in cited:
                    if cite.strip("[]") not in valid_ids:
                        issues.append(
                            CritiqueIssue(
                                severity="block",
                                code="unknown_citation",
                                message=f"Narrative cites unknown evidence {cite}.",
                            )
                        )
                stripped = _ID_TOKEN_RE.sub("", EV_CITATION_RE.sub("", sentence))
                if _NUMERIC_RE.search(stripped) and not cited:
                    issues.append(
                        CritiqueIssue(
                            severity="block",
                            code="uncited_quantitative_claim",
                            message=f"Uncited quantitative claim: '{sentence.strip()[:120]}'",
                        )
                    )
        return CritiqueResult(ok=not any(i.severity == "block" for i in issues), issues=issues)


def _cite_every_sentence(text: str, evidence_id: str) -> str:
    """Append [EV-...] to each sentence lacking a citation, so template
    narratives are proof-carrying at sentence granularity by construction."""
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?!\[)", text.strip()) if p.strip()]
    return " ".join(
        p if EV_CITATION_RE.search(p) else f"{p} [{evidence_id}]" for p in parts
    )


class NarrativeAgent:
    """Generates proof-carrying reviewer narratives.

    Deterministic path builds the narrative directly from evidence records so
    every quantitative claim carries its [EV-...] citation by construction.
    LLM path drafts prose, which must then pass EvidenceCriticAgent.
    """

    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def generate(self, run_id: str, records: list[EvidenceRecord]) -> Narrative:
        if self.llm.available:
            try:
                return self._llm_narrative(run_id, records)
            except Exception:
                pass  # fall back to deterministic template
        return self._template_narrative(run_id, records)

    def _template_narrative(self, run_id: str, records: list[EvidenceRecord]) -> Narrative:
        counts: dict[str, int] = {}
        for rec in records:
            counts[rec.status.value] = counts.get(rec.status.value, 0) + 1
        summary_bits = ", ".join(sorted(counts)) or "no tests executed"
        worst = [r for r in records if r.status in (Status.FAIL, Status.ERROR)]
        warns = [r for r in records if r.status == Status.WARN]
        findings = [
            _cite_every_sentence(
                f"{rec.test_name}: {rec.interpretation} Status: {rec.status.value}.",
                rec.evidence_id,
            )
            for rec in records
            if rec.status != Status.SKIPPED
        ]
        next_steps = [
            f"Investigate '{rec.test_name}' breach before approval. [{rec.evidence_id}]"
            for rec in worst
        ] + [f"Review warning from '{rec.test_name}'. [{rec.evidence_id}]" for rec in warns]
        if not next_steps:
            next_steps = ["No threshold breaches detected; proceed to family-expansion review."]
        limitations = sorted(
            {
                _cite_every_sentence(lim, rec.evidence_id)
                for rec in records
                for lim in rec.limitations
                if not lim.startswith("Traceback")
            }
        )
        cited = [rec.evidence_id for rec in records]
        return Narrative(
            run_id=run_id,
            summary=(
                f"Validation run {run_id} executed the planned registered tests "
                f"(statuses observed: {summary_bits}). Per-test outcomes with metrics "
                "and citations follow; every quantitative statement cites its evidence ID."
            ),
            findings=findings,
            limitations=limitations[:10],
            next_steps=next_steps,
            generator="template",
            cited_evidence_ids=cited,
        )

    def _llm_narrative(self, run_id: str, records: list[EvidenceRecord]) -> Narrative:
        evidence_block = "\n".join(
            f"[{r.evidence_id}] {r.test_name} status={r.status.value} metrics={r.metrics}"
            for r in records
        )
        system = (
            "You are a model-risk reviewer assistant. Write a concise narrative from the "
            "evidence records provided. RULES: every sentence containing a number MUST end "
            "with the [EV-...] citation of the record it came from; never introduce numbers "
            "not present in the evidence; never speculate."
        )
        text = self.llm.complete(system, evidence_block, max_tokens=800)
        return Narrative(
            run_id=run_id,
            summary=text.strip(),
            findings=[],
            limitations=["LLM-drafted narrative; verified by EvidenceCriticAgent."],
            next_steps=[],
            generator=f"llm:{self.llm.name}",
            cited_evidence_ids=sorted(set(m.strip("[]") for m in EV_CITATION_RE.findall(text))),
        )


class PolicyGuardAgent:
    """Checks the workflow against the active policy before execution."""

    def __init__(self, policy: PolicyConfig) -> None:
        self.policy = policy

    def check(self, plan: ValidationPlan, data_root: str | None = None) -> PolicyDecision:
        reasons: list[str] = []
        allowed = True
        if self.policy.allowed_task_types and plan.task_type.value not in self.policy.allowed_task_types:
            allowed = False
            reasons.append(f"Task type '{plan.task_type.value}' is not permitted by policy.")
        if data_root and self.policy.allowed_data_roots:
            if not any(str(data_root).startswith(root) for root in self.policy.allowed_data_roots):
                allowed = False
                reasons.append(f"Data root '{data_root}' is outside allowed paths.")
        order = [m.value for m in Materiality]
        if order.index(plan.materiality.value) > order.index(self.policy.max_materiality_without_review):
            reasons.append(
                "Materiality exceeds auto-run limit; human review sign-off required."
            )
        if allowed and not reasons:
            reasons.append("Plan conforms to active policy.")
        return PolicyDecision(allowed=allowed, reasons=reasons, policy_hash=self.policy.content_hash())


def get_agents(config: StartConfig, policy: PolicyConfig, llm: LLMProvider) -> dict[str, Any]:
    return {
        "planner": ReviewPlannerAgent(config, llm),
        "router": TestRouterAgent(),
        "critic": EvidenceCriticAgent(),
        "narrator": NarrativeAgent(llm),
        "policy_guard": PolicyGuardAgent(policy),
    }
