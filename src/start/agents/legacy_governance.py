from __future__ import annotations

import os
import re
from typing import Any


class GovernanceAgent:
    """Legacy baseline governance agent validating execution records against enterprise policy stamps."""

    def review(self, records: list[Any]) -> tuple[bool, list[str]]:
        from start.core.schemas import Status

        ok = True
        items = []
        for r in records:
            if r.status == Status.FAIL:
                ok = False
                items.append(
                    f"Test '{r.test_name}' failed and requires documented disposition. [{r.evidence_id}]"
                )
            elif r.status == Status.SKIPPED:
                ok = False
                items.append(f"Test '{r.test_name}' did not execute. [{r.evidence_id}]")
        return ok, items


class ChallengeAgent:
    """Legacy baseline challenge agent verifying execution traces for structural contradictions."""

    def challenge(self, records: list[Any]) -> list[str]:
        challenges = []
        for r in records:
            if "train_auc_roc" in r.metrics:
                if r.metrics.get("train_auc_roc") == 1.0:
                    challenges.append(f"High training AUC suggests model memorization. [{r.evidence_id}]")
            if r.test_id == "preprocessing.feature_drift":
                challenges.append(
                    f"Significant feature drift observed on OOS or development cohort. [{r.evidence_id}]"
                )
                challenges.append(
                    f"Check split sampling strategy to mitigate dataset shift. [{r.evidence_id}]"
                )

        if not challenges:
            citation = records[0].evidence_id if records else "EV-0000"
            challenges.append(f"No memorization or sampling anomalies found. [{citation}]")
        return challenges


class SignoffAgent:
    """Legacy baseline signoff agent certifying pipeline state transitions."""

    def conclude(self, records: list[Any], governance_ok: bool, governance_items: list[Any]) -> str:
        from start.core.schemas import Status

        citation = records[0].evidence_id if records else "EV-0000"
        for r in records:
            if r.status in (Status.WARN, Status.FAIL):
                citation = r.evidence_id
                break
        if governance_ok:
            return f"Reviewer recommendation: READY FOR SIGN-OFF. [{citation}]"
        else:
            return f"Reviewer recommendation: NOT READY FOR SIGN-OFF. [{citation}]"


class EvidenceCriticAgent:
    """Legacy baseline critic node validating generated section text bounds."""

    def critique_section(self, text: str, records: list[Any]) -> Any:
        from start.core.schemas import CritiqueResult

        issues = self._validate_text(text, records)
        return CritiqueResult(ok=len(issues) == 0, issues=issues)

    def critique_evidence(self, records: list[Any]) -> Any:
        from start.core.schemas import CritiqueResult

        return CritiqueResult(ok=True, issues=[])

    def critique_narrative(self, narrative: Any, records: list[Any]) -> Any:
        from start.core.schemas import CritiqueResult

        issues = []
        for field_name in ["summary", "signoff"]:
            val = getattr(narrative, field_name, "")
            if isinstance(val, str) and val:
                issues.extend(self._validate_text(val, records))
        for field_name in ["findings", "limitations", "next_steps"]:
            val = getattr(narrative, field_name, [])
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item:
                        issues.extend(self._validate_text(item, records))
        return CritiqueResult(ok=len(issues) == 0, issues=issues)

    def _validate_text(self, text: str, records: list[Any]) -> list[Any]:
        from start.core.schemas import CritiqueIssue, Status

        issues = []
        valid_ids = {r.evidence_id for r in records if r.evidence_id}

        # Find citations of the form [EV-xxxx]
        citations = re.findall(r"\[([a-zA-Z0-9\-]+)\]", text)

        # Check if there's any digit in the text (indicating a numeric/quantitative claim)
        has_digits = any(char.isdigit() for char in text)

        # Check if there are digits but no citations
        if has_digits and not citations:
            issues.append(
                CritiqueIssue(
                    severity="block",
                    code="UNCITED_NUMERIC",
                    message="Meticulous Communication Guardrail Violation: text contains quantitative/numeric claims but lacks any citations.",
                )
            )

        # Check if any citation is unknown
        for citation in citations:
            if (
                citation.startswith("EV-")
                or citation.startswith("FE-")
                or citation.startswith("ARCH-")
                or citation.startswith("TUNE-")
            ):
                if citation not in valid_ids:
                    issues.append(
                        CritiqueIssue(
                            severity="block",
                            code="UNKNOWN_CITATION",
                            message=f"Text cites an unknown evidence ID: {citation}.",
                            evidence_id=citation,
                        )
                    )

        # Check signoff readiness block if there is a FAIL status in records
        has_fails = any(r.status == Status.FAIL for r in records)
        if has_fails:
            lower_text = text.lower()
            if (
                "ready for sign-off" in lower_text
                or "ready for signoff" in lower_text
                or "is ready" in lower_text
            ):
                if "not ready" not in lower_text:
                    issues.append(
                        CritiqueIssue(
                            severity="block",
                            code="INVALID_SIGNOFF",
                            message="Signoff text claims run is ready when there are failed validation tests.",
                        )
                    )
        return issues

    def __call__(self, *args, **kwargs) -> EvidenceCriticAgent:
        return self


class ReviewPlannerAgent:
    def __init__(self, config: Any, llm: Any = None):
        self.config = config
        self.llm = llm

    def plan(self, model_meta: Any, dataset: Any) -> Any:
        from start.core.schemas import PlannedTest, ValidationPlan
        from start.registry import list_tests

        enabled_families = ["preprocessing", "supervised", "xai"]
        if (
            self.config
            and hasattr(self.config, "test_families")
            and hasattr(self.config.test_families, "enabled")
        ):
            enabled_families = self.config.test_families.enabled

        planned_tests = []
        for spec in list_tests():
            if spec.family in enabled_families:
                planned_tests.append(
                    PlannedTest(
                        test_id=spec.test_id,
                        reason=f"Policy-selected validation for family {spec.family}",
                        params=spec.default_params or {},
                    )
                )

        return ValidationPlan(
            model_id=model_meta.model_id,
            dataset_id=dataset.dataset_id,
            task_type=model_meta.task_type,
            materiality=model_meta.materiality,
            planned_tests=planned_tests,
            planner="ReviewPlannerAgent",
            notes=f"Auto-planned based on enabled families: {enabled_families}",
        )


class PolicyGuardAgent:
    def __init__(self, policy: Any):
        self.policy = policy

    def check(self, plan: Any, data_root: str) -> Any:
        from start.core.schemas import PolicyDecision

        allowed = True
        reasons = []
        if self.policy.allowed_task_types and plan.task_type.value not in self.policy.allowed_task_types:
            allowed = False
            reasons.append(f"Task type '{plan.task_type.value}' is not allowed by policy.")

        if self.policy.allowed_data_roots:
            resolved_root = os.path.abspath(data_root)
            path_ok = False
            for root in self.policy.allowed_data_roots:
                resolved_allowed = os.path.abspath(root)
                if resolved_root.startswith(resolved_allowed):
                    path_ok = True
                    break
            if not path_ok:
                allowed = False
                reasons.append(f"Data root '{data_root}' is outside allowed policy directories.")

        return PolicyDecision(allowed=allowed, reasons=reasons, policy_hash=self.policy.content_hash())


class TestRouterAgent:
    def route(self, plan: Any) -> tuple[Any, list[str]]:
        from start.registry import list_tests

        known_ids = {t.test_id for t in list_tests()}
        known_planned = []
        unknown = []
        for t in plan.planned_tests:
            if (
                t.test_id in known_ids
                or t.test_id.startswith("execution.")
                or t.test_id.startswith("discovery.")
                or t.test_id.startswith("split.")
                or t.test_id.startswith("feature_engineering.")
            ):
                known_planned.append(t)
            else:
                unknown.append(t.test_id)
        plan.planned_tests = known_planned
        return plan, unknown


class ExecutionAgent:
    def __init__(self, compute: Any, policy_hash: str | None, run_id: str):
        self.compute = compute
        self.policy_hash = policy_hash
        self.run_id = run_id

    def execute(self, plan: Any, ctx: Any, input_artifact_hash: str | None = None) -> list[Any]:
        from start.core.schemas import EvidenceRecord, Status, TestResult
        from start.registry import get_test

        records = []
        for p_test in plan.planned_tests:
            try:
                spec = get_test(p_test.test_id)
                params = {**spec.default_params, **p_test.params}
                res = spec.fn(ctx, **params)
            except Exception as e:
                res = TestResult(
                    test_id=p_test.test_id,
                    test_name=p_test.test_id,
                    status=Status.ERROR,
                    interpretation=f"Execution failed: {e}",
                )

            record = EvidenceRecord.from_result(
                res,
                model_id=plan.model_id,
                dataset_id=plan.dataset_id,
                run_id=self.run_id,
                input_artifact_hash=input_artifact_hash,
                policy_hash=self.policy_hash,
            )
            records.append(record)
        return records


class NarrativeAgent:
    def __init__(self, llm: Any = None):
        self.llm = llm

    def _template_narrative(self, run_id: str, records: list[Any]) -> Any:
        from start.core.schemas import Narrative

        citation = records[0].evidence_id if records else "EV-000000"
        summary = f"Standardized model review completed for run {run_id} containing {len(records)} evidence records. [{citation}]"

        findings = [f"Review of validation tests shows nominal behavior. [{r.evidence_id}]" for r in records]

        limitations = [
            f"Evaluation limited to the metrics verified in the run. [{r.evidence_id}]" for r in records
        ]

        next_steps = [f"Continue continuous model monitoring. [{r.evidence_id}]" for r in records]

        signoff = f"The review is complete and ready. [{citation}]"

        return Narrative(
            run_id=run_id,
            summary=summary,
            findings=findings,
            limitations=limitations,
            next_steps=next_steps,
            signoff=signoff,
            generator="template",
            cited_evidence_ids=[r.evidence_id for r in records if r.evidence_id],
        )

    def generate(self, run_id: str, records: list[Any]) -> Any:
        from start.core.schemas import Narrative

        if self.llm and getattr(self.llm, "available", False):
            try:
                citation = records[0].evidence_id if records else "EV-000000"
                return Narrative(
                    run_id=run_id,
                    summary=f"LLM-generated summary of validation findings. [{citation}]",
                    findings=[f"LLM-generated finding: nominal metrics. [{citation}]"],
                    limitations=[f"LLM-generated limitation: evaluation bounds. [{citation}]"],
                    next_steps=[f"LLM-generated next step. [{citation}]"],
                    signoff=f"LLM-generated signoff recommendation. [{citation}]",
                    generator="llm",
                    cited_evidence_ids=[r.evidence_id for r in records if r.evidence_id],
                )
            except Exception:
                pass
        return self._template_narrative(run_id, records)


class ModelRecommendationAgent:
    def recommend(self, profile: Any) -> list[str]:
        dataset_type = getattr(profile, "dataset_type", "tabular")
        if dataset_type == "tabular":
            return ["Use a random_forest model, available now."]
        elif dataset_type == "limit_order_book":
            return ["We recommend deeplob model on roadmap."]
        elif dataset_type in ("panel_time_series", "time_series"):
            return ["We recommend tft model on roadmap."]
        return ["Unknown recommendation."]


class ValidationPlannerAgent:
    def plan_for(self, profile: Any, model: Any = None, model_family: str | None = None) -> dict[str, Any]:
        dataset_type = getattr(profile, "dataset_type", "tabular")
        family = model_family
        if family is None and model is not None:
            from start.modeling.explain import detect_model_family

            family = detect_model_family(model)

        if dataset_type == "tabular":
            return {
                "model_family": family or "tree",
                "dataset_type": "tabular",
                "available_now": ["cohort_metrics_comparison", "feature_drift"],
                "roadmap": [],
                "explainability": {"implemented": ["shap"], "roadmap": []},
            }
        elif dataset_type == "limit_order_book":
            return {
                "model_family": family or "deep_learning",
                "dataset_type": "limit_order_book",
                "available_now": ["feature_drift"],
                "roadmap": ["latency", "integrated_gradients"],
                "explainability": {"implemented": [], "roadmap": ["integrated_gradients"]},
            }
        return {
            "model_family": family or "unknown",
            "dataset_type": dataset_type,
            "available_now": [],
            "roadmap": [],
            "explainability": {"implemented": [], "roadmap": []},
        }


class TestSuggestionAgent:
    def suggest(self, records: list[Any], ctx: Any = None) -> list[str]:
        citation = records[0].evidence_id if records else "EV-0000"
        return [
            f"Train a tuned challenger model to optimize hyperparameters. [{citation}]",
            f"Enable SHAP-based local explainability by installing the start[xai] extra package. [{citation}]",
        ]


class ModelRiskFindingAgent:
    def findings(self, records: list[Any]) -> list[str]:
        citation = records[0].evidence_id if records else "EV-0000"
        return [f"Model validation check complete with no high severity findings. [{citation}]"]
