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
        texts = [narrative.summary, *narrative.findings, *narrative.limitations, narrative.signoff]
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
        "risk_findings": ModelRiskFindingAgent(),
        "test_suggestions": TestSuggestionAgent(),
    }


class ModelRiskFindingAgent:
    """Deterministic cross-evidence reasoning: turns patterns ACROSS records
    into reviewer-ready, citation-carrying findings. Reads metrics from
    evidence only — never recomputes or invents numbers."""

    def __init__(self, gap_concern: float = 0.05, oos_concern: float = 0.03) -> None:
        self.gap_concern = gap_concern
        self.oos_concern = oos_concern

    def findings(self, records: list[EvidenceRecord]) -> list[str]:
        by_test = {rec.test_id: rec for rec in records}
        out: list[str] = []

        comparison = by_test.get("supervised.cohort_metrics_comparison")
        if comparison and comparison.status not in (Status.SKIPPED, Status.ERROR):
            gap = comparison.metrics.get("auc_gap_train_test")
            if isinstance(gap, (int, float)) and gap >= self.gap_concern:
                out.append(
                    f"Overfitting signal: train-test AUC gap is {gap:.4f}, which exceeds "
                    f"the concern level of {self.gap_concern:.2f}. [{comparison.evidence_id}]"
                )
            test_oos = comparison.metrics.get("auc_gap_test_oos")
            if isinstance(test_oos, (int, float)) and abs(test_oos) >= self.oos_concern:
                out.append(
                    f"Cohort instability: test-vs-OOS AUC gap is {test_oos:+.4f}, beyond the "
                    f"stability band of \u00b1{self.oos_concern:.2f}. [{comparison.evidence_id}]"
                )

        sensitivity = by_test.get("xai.feature_sensitivity")
        if sensitivity and sensitivity.status not in (Status.SKIPPED, Status.ERROR):
            drift = sensitivity.metrics.get("max_abs_auc_drift")
            if isinstance(drift, (int, float)):
                out.append(
                    f"Sensitivity profile: parallel shocks to top features moved AUC by at most "
                    f"{drift:.4f} on the '{sensitivity.metrics.get('cohort', 'test')}' cohort. "
                    f"[{sensitivity.evidence_id}]"
                )

        breaches = [r for r in records if r.status in (Status.FAIL, Status.ERROR)]
        if breaches:
            cites = " ".join(f"[{r.evidence_id}]" for r in breaches)
            out.append(
                f"Threshold breaches or engine errors require disposition before "
                f"approval: {len(breaches)} record(s). {cites}"
            )
        return out


class TestSuggestionAgent:
    """Deterministic next-step recommender: explains skipped tests, identifies
    missing evidence, and suggests applicable follow-up validation work.
    Suggestions are non-quantitative or cite evidence IDs, so they pass the
    citation gate."""

    def suggest(self, records: list[EvidenceRecord], ctx: Any = None) -> list[str]:
        by_test = {rec.test_id: rec for rec in records}
        executed = set(by_test)
        out: list[str] = []

        for rec in records:
            if rec.status == Status.SKIPPED:
                out.append(
                    f"'{rec.test_name}' was skipped — {rec.interpretation.rstrip('.')}. "
                    f"Provide the missing artifact to enable it. [{rec.evidence_id}]"
                )

        importance = by_test.get("xai.global_importance")
        if importance and importance.metrics.get("method") == "permutation":
            out.append(
                "Global importance used the permutation fallback; install the xai extra "
                "(pip install -e \".[xai]\") to enable SHAP values and local attributions. "
                f"[{importance.evidence_id}]"
            )
        if importance and "xai.feature_sensitivity" not in executed:
            out.append(
                "Global importance is available; run the top-feature shock sensitivity test "
                f"to quantify robustness of the ranking. [{importance.evidence_id}]"
            )

        has_oos = bool(ctx is not None and getattr(ctx, "extra", {}).get("oos") is not None)
        if "supervised.cohort_metrics_comparison" in executed and not has_oos:
            out.append(
                "No out-of-sample cohort was provided; add an OOS split to test "
                "generalization beyond the holdout."
            )

        meta = (getattr(ctx, "extra", None) or {}).get("demo_meta", {}) if ctx is not None else {}
        if meta.get("tuning_method") == "none":
            out.append(
                "The model was fitted with default hyperparameters; consider a tuned "
                "challenger (grid, random, or Optuna) and compare cohort metrics."
            )
        if not any(t.startswith("supervised.calibration") for t in executed):
            out.append("Calibration has not been assessed; add the calibration check.")
        return out


class ModelRecommendationAgent:
    """Deterministic model recommendations from dataset metadata: dataset
    type, feature structure, time/entity structure, and target type drive a
    type-aware candidate list, honestly labeled available-now vs roadmap."""

    def recommend(self, profile: Any) -> list[str]:
        from start.taxonomy import MODEL_RECOMMENDATIONS

        candidates = MODEL_RECOMMENDATIONS.get(profile.dataset_type, MODEL_RECOMMENDATIONS["tabular"])
        lines = [f"Detected: {profile.describe()}"]
        for rank, (model, rationale, implemented) in enumerate(candidates, start=1):
            tag = "available now" if implemented else "roadmap"
            lines.append(f"{rank}. {model} — {rationale} ({tag})")
        if profile.target_type == "continuous" and profile.dataset_type == "tabular":
            lines.append(
                "Target appears continuous; current registered metric engines focus on "
                "binary classification — regression metrics are on the roadmap."
            )
        lines.extend(profile.notes)
        return lines


class ValidationPlannerAgent:
    """Model- and dataset-type-specific validation plan: combines the
    dataset-type plan with model-family additions and the explainability
    route, distinguishing checks runnable today from roadmap items."""

    def plan_for(self, profile: Any, model: Any = None, model_family: str | None = None) -> dict:
        from start.modeling.explain import detect_model_family, route_explainability
        from start.taxonomy import MODEL_FAMILY_PLANS, VALIDATION_PLANS

        family = detect_model_family(model, model_family)
        items = list(VALIDATION_PLANS.get(profile.dataset_type, VALIDATION_PLANS["tabular"]))
        if family == "tree":
            family_key = "tree"
        elif family in {"deep_learning", "transformer"}:
            family_key = "deep_learning"
        else:
            family_key = ""
        items += MODEL_FAMILY_PLANS.get(family_key, [])
        seen: set[str] = set()
        available, roadmap = [], []
        for check, ref, ok in items:
            if ref in seen:
                continue
            seen.add(ref)
            (available if ok else roadmap).append(f"{check} ({ref})")
        explain = route_explainability(model, family if family != "unknown" else None)
        return {
            "dataset_type": profile.dataset_type,
            "model_family": explain.model_family,
            "available_now": available,
            "roadmap": roadmap,
            "explainability": {
                "implemented": explain.implemented(),
                "roadmap": explain.roadmap(),
            },
        }


class ChallengeAgent:
    """Deterministic adversarial pass: attempts to invalidate or weaken the
    run's own conclusions using only the evidence, citing every record it
    leans on. LLM-assisted challenge drafting can be layered when an LLM
    provider is configured; the deterministic core always runs."""

    def challenge(self, records: list[EvidenceRecord]) -> list[str]:
        by_test = {rec.test_id: rec for rec in records}
        out: list[str] = []

        comparison = by_test.get("supervised.cohort_metrics_comparison")
        if comparison and comparison.metrics.get("train_auc_roc") == 1.0:
            out.append(
                "Challenge: the model separates training data perfectly, which is "
                "consistent with memorization; holdout metrics carry the burden of "
                f"proof here. [{comparison.evidence_id}]"
            )

        drift = by_test.get("preprocessing.feature_drift")
        sensitivity = by_test.get("xai.feature_sensitivity")
        if (
            drift
            and drift.status == Status.WARN
            and sensitivity
            and sensitivity.metrics.get("cohort") == "test"
        ):
            out.append(
                f"Challenge: feature drift was flagged between cohorts [{drift.evidence_id}] "
                "while sensitivity was evaluated on the test cohort only; re-running "
                f"sensitivity on the OOS or development cohort would be more conservative. "
                f"[{sensitivity.evidence_id}]"
            )

        small = next(
            (
                rec
                for rec in records
                if isinstance(rec.metrics.get("n_rows"), int) and rec.metrics["n_rows"] < 1000
            ),
            None,
        )
        if small:
            out.append(
                "Challenge: cohort sizes are modest, so point metrics carry sampling "
                f"uncertainty; treat small metric gaps as indicative, not conclusive. "
                f"[{small.evidence_id}]"
            )
        return out


class GovernanceAgent:
    """Deterministic governance gate over the whole run: policy stamping,
    missing/failed evidence, skipped planned tests, and narrative integrity.
    Produces reviewer-facing items and an overall ok flag."""

    def review(
        self,
        records: list[EvidenceRecord],
        narrative_ok: bool | None = None,
    ) -> tuple[bool, list[str]]:
        items: list[str] = []
        unstamped = [r for r in records if not r.policy_hash]
        if unstamped:
            cites = " ".join(f"[{r.evidence_id}]" for r in unstamped[:5])
            items.append(f"Governance: evidence found without a policy hash. {cites}")
        unresolved = [r for r in records if r.status in (Status.FAIL, Status.ERROR)]
        for rec in unresolved:
            items.append(
                f"Governance: '{rec.test_name}' ended in status {rec.status.value} and "
                f"requires documented disposition before sign-off. [{rec.evidence_id}]"
            )
        skipped = [r for r in records if r.status == Status.SKIPPED]
        for rec in skipped:
            items.append(
                f"Governance: planned test '{rec.test_name}' did not execute; record "
                f"a justification or supply the missing artifact. [{rec.evidence_id}]"
            )
        if narrative_ok is False:
            items.append("Governance: the narrative failed the citation gate and was rejected.")
        ok = not unresolved and not unstamped and narrative_ok is not False
        return ok, items


class SignoffAgent:
    """Reviewer-ready conclusion. Recommends sign-off only when governance is
    clean; otherwise states what remains outstanding. Wording avoids uncited
    quantities so it always passes the citation gate."""

    def conclude(
        self,
        records: list[EvidenceRecord],
        governance_ok: bool,
        governance_items: list[str],
    ) -> str:
        breaches = [r for r in records if r.status in (Status.FAIL, Status.ERROR)]
        warns = [r for r in records if r.status == Status.WARN]
        if governance_ok and not breaches:
            base = (
                "Reviewer recommendation: READY FOR SIGN-OFF. All executed checks "
                "completed without fail or error statuses, every evidence record is "
                "policy-stamped, and the narrative passed the citation gate."
            )
            if warns:
                cites = " ".join(f"[{r.evidence_id}]" for r in warns)
                base += (
                    f" Warnings remain open for reviewer judgment and are cited here: {cites}"
                )
            return base
        cites = " ".join(f"[{r.evidence_id}]" for r in breaches)
        return (
            "Reviewer recommendation: NOT READY FOR SIGN-OFF. Outstanding items "
            "require disposition before approval"
            + (f"; breached or errored evidence: {cites}." if cites else ".")
            + " See the governance findings above for the complete list."
        )
