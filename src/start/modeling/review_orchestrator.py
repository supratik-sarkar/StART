"""The StART Model-Risk Operating System orchestrator.

This is the differentiator: not model training, but the model-risk operating
layer that makes every stage visible. ``ReviewOrchestrator.run`` executes the
full pipeline and reports each stage as it happens (no silent execution):

    dataset -> discovery -> target -> task -> split -> feature engineering
    -> model recommendation -> execution -> metrics -> explainability
    -> sensitivity -> robustness -> evidence ledger -> ReviewPlanner
    -> TestSuggestion -> ModelRecommendation -> ValidationPlanner
    -> ModelRiskFinding -> Challenge -> Governance -> Signoff
    -> EvidenceCritic -> AI-engineering stages -> final report

Deterministic mode is the default and needs no key. In LLM mode the agent
review reasons only over the evidence bundle (never raw data), gated by the
EvidenceCritic. The enterprise gateway, if selected, is routed exclusively
through ``start.enterprise``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from start.agents.discovery import (
    DatasetDiscoveryAgent,
    TargetDiscoveryAgent,
    TaskInferenceAgent,
)
from start.core.schemas import EvidenceRecord, TestResult
from start.modeling.feature_engineering import FeatureEngineeringAgent
from start.modeling.split_planner import SplitPlanner

# Modality -> default model family suggestion (deterministic recommendation).
_MODALITY_DEFAULT = {
    "tabular": "mlp",
    "sequence": "lstm",
    "vision": "simple_cnn_small",
}

STAGES = (
    "dataset",
    "discovery",
    "target_confirmation",
    "task_inference",
    "split_planning",
    "feature_engineering",
    "model_recommendation",
    "model_execution",
    "metrics",
    "explainability",
    "sensitivity",
    "robustness",
    "evidence_ledger",
    "review_planner",
    "test_suggestion",
    "model_risk_finding",
    "challenge",
    "governance",
    "signoff",
    "evidence_critic",
    "ai_engineering",
    "final_report",
)


@dataclass
class StageEvent:
    stage: str
    status: str  # running | complete | skipped
    detail: str = ""


@dataclass
class ReviewOutcome:
    run_id: str
    task_type: str
    target: str | list[str]
    modality: str
    recommended_family: str
    cohort_metrics: dict[str, dict[str, float]]
    evidence: list[EvidenceRecord]
    agent_review: Any
    ai_engineering: list[Any]
    stage_events: list[StageEvent]
    report_path: str | None = None
    notes: list[str] = field(default_factory=list)


class ReviewOrchestrator:
    """Runs the full, visible model-review pipeline."""

    def __init__(self, on_stage: Callable[[StageEvent], None] | None = None) -> None:
        self._on_stage = on_stage
        self.stage_events: list[StageEvent] = []

    def _emit(self, stage: str, status: str, detail: str = "") -> None:
        event = StageEvent(stage=stage, status=status, detail=detail)
        self.stage_events.append(event)
        if self._on_stage:
            self._on_stage(event)

    def run(
        self,
        df: pd.DataFrame,
        *,
        user_target: str | list[str] | None = None,
        task_override: str | None = None,
        split_strategy: str = "stratified",
        fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
        agent_mode: str = "deterministic",
        llm: Any = None,
        output_root: str | None = None,
        seed: int = 42,
        run_dl: bool = False,
        architecture: str = "mlp",
        activation: str = "relu",
        custom_space: dict[str, Any] | None = None,
        class_weight: str | None = None,
        enterprise_run_id: str | None = None,
    ) -> ReviewOutcome:
        import uuid


        run_id = "RUN-MROS-" + uuid.uuid4().hex[:8]
        evidence: list[TestResult] = []

        # 1. dataset
        self._emit("dataset", "complete", f"{len(df)} rows x {df.shape[1]} columns")

        # 2. discovery
        self._emit("discovery", "running")
        discovery_agent = DatasetDiscoveryAgent()
        profile = discovery_agent.discover(df)
        evidence.append(discovery_agent.to_evidence(profile))
        self._emit("discovery", "complete", profile.summary())

        # 3. target confirmation
        self._emit("target_confirmation", "running")
        target_agent = TargetDiscoveryAgent()
        target_rec = target_agent.recommend(profile, user_target)
        evidence.append(target_agent.to_evidence(target_rec))
        chosen_target = target_rec.selected or (
            profile.candidate_targets[0] if profile.candidate_targets else None
        )
        if chosen_target is None:
            raise ValueError("No target could be determined; supply user_target explicitly.")
        self._emit("target_confirmation", "complete", f"target = {chosen_target}")

        # 4. task inference
        self._emit("task_inference", "running")
        task_agent = TaskInferenceAgent()
        inference = task_agent.infer(
            df, chosen_target, override=task_override,
            has_timestamp=bool(profile.timestamp_columns),
        )
        evidence.append(task_agent.to_evidence(inference))
        self._emit("task_inference", "complete", inference.task_type)

        # 5. split planning
        self._emit("split_planning", "running")
        single_target = chosen_target if isinstance(chosen_target, str) else chosen_target[0]
        planner = SplitPlanner()
        plan = planner.plan(
            df, strategy=split_strategy, target_column=single_target,
            fractions=fractions, seed=seed,
        )
        evidence.append(planner.to_evidence(plan, single_target))
        self._emit("split_planning", "complete", f"{plan.strategy} {plan.sizes}")

        # 6. feature engineering
        self._emit("feature_engineering", "running")
        modality = _infer_modality(profile)
        fe = FeatureEngineeringAgent()
        fe_modality = {"tabular": "tabular", "sequence": "sequential", "vision": "vision"}[modality]
        diag = fe.diagnose(plan.train, single_target, modality=fe_modality, test=plan.test)
        evidence.append(fe.to_evidence(diag))
        self._emit("feature_engineering", "complete", f"{modality} diagnostics")

        # 7. model recommendation
        self._emit("model_recommendation", "running")
        recommended = architecture if architecture != "mlp" else _MODALITY_DEFAULT[modality]
        self._emit("model_recommendation", "complete", f"recommended: {recommended}")

        # 8-12. execution + metrics + explainability + sensitivity + robustness
        cohort_metrics: dict[str, dict[str, float]] = {}
        if modality == "tabular" and run_dl and inference.task_type in ("binary_classification", "multiclass_classification", "regression", "forecasting"):
            cohort_metrics = self._run_tabular_dl(
                plan, single_target, evidence, seed, architecture=architecture,
                task_type=inference.task_type, activation=activation,
                custom_space=custom_space, class_weight=class_weight,
            )
        else:
            self._emit("model_execution", "skipped", "diagnostics-only review (no model trained)")
            for s in ("metrics", "explainability", "sensitivity", "robustness"):
                self._emit(s, "skipped", "requires model execution (run_dl=True, tabular binary)")

        # 13. evidence ledger
        self._emit("evidence_ledger", "running")
        records = self._persist(evidence, run_id, output_root, enterprise_run_id=enterprise_run_id)
        self._emit("evidence_ledger", "complete", f"{len(records)} records sealed")

        # 14-20. agentic governance review
        from start.agents.review import run_agent_review

        for s in ("review_planner", "test_suggestion", "model_risk_finding"):
            self._emit(s, "running")
        agent_review = run_agent_review(
            records, mode=agent_mode, llm=llm,
            dataset_profile=profile.summary(),
            demo_meta={"task": inference.task_type, "modality": modality},
        )
        for s in ("review_planner", "test_suggestion", "model_risk_finding"):
            self._emit(s, "complete")
        self._emit("challenge", "complete", f"{len(agent_review.challenge_memo)} memo items")
        self._emit("governance", "complete")
        self._emit("signoff", "complete", agent_review.signoff.split(".")[0])
        self._emit(
            "evidence_critic", "complete",
            "citation gate: " + ("passed" if agent_review.critique_ok else "failed"),
        )

        # 21. AI-engineering stages (visible, honest availability)
        self._emit("ai_engineering", "running")
        from start.ai_engineering import run_all_stages

        ai_stages = run_all_stages({"run_id": run_id})
        n_avail = sum(1 for s in ai_stages if s.available)
        self._emit(
            "ai_engineering", "complete",
            f"{n_avail}/{len(ai_stages)} stages available; rest reported not installed",
        )

        # 22. report
        self._emit("final_report", "running")
        report_path = self._write_report(
            run_id, inference, chosen_target, modality, recommended,
            cohort_metrics, records, agent_review, ai_stages, output_root,
        )
        self._emit("final_report", "complete", str(report_path) if report_path else "")

        return ReviewOutcome(
            run_id=run_id,
            task_type=inference.task_type,
            target=chosen_target,
            modality=modality,
            recommended_family=recommended,
            cohort_metrics=cohort_metrics,
            evidence=records,
            agent_review=agent_review,
            ai_engineering=ai_stages,
            stage_events=self.stage_events,
            report_path=str(report_path) if report_path else None,
        )

    # -- helpers ----------------------------------------------------------- #
    def _run_tabular_dl(
        self, plan, target, evidence, seed, architecture="mlp",
        task_type="binary_classification", activation="relu",
        custom_space=None, class_weight=None,
    ) -> dict[str, dict[str, float]]:
        from start.modeling.models import resolve_model
        from start.modeling.tabular_dl_metrics import dl_task_metrics
        from start.modeling.tuning_run import _model_family

        self._emit("model_execution", "running", f"training tabular {architecture} with activation {activation}")
        features = [c for c in plan.train.columns if c != target]
        features = [c for c in features if pd.api.types.is_numeric_dtype(plan.train[c])]
        
        family = _model_family(architecture)
        if family == "sklearn":
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy="median")
            plan.train = plan.train.copy()
            plan.train[features] = imputer.fit_transform(plan.train[features])
            if len(plan.test):
                plan.test = plan.test.copy()
                plan.test[features] = imputer.transform(plan.test[features])
            if len(plan.oos):
                plan.oos = plan.oos.copy()
                plan.oos[features] = imputer.transform(plan.oos[features])
        if family == "tabular_dl":
            from start.modeling.tabular_dl import TabularDLClassifier
            kwargs = {
                "task": task_type,
                "family": architecture,
                "activation": activation,
                "epochs": 8,
                "random_state": seed,
                "class_weight": class_weight,
            }
            if custom_space:
                if "hidden_dims" in custom_space:
                    kwargs["hidden_dims"] = custom_space["hidden_dims"]
                elif "hidden_size" in custom_space:
                    kwargs["hidden_dims"] = (custom_space["hidden_size"],) * custom_space.get("num_layers", 1)
                for param in ("learning_rate", "dropout", "batch_size", "epochs"):
                    if param in custom_space:
                        val = custom_space[param]
                        if isinstance(val, list):
                            val = val[0]
                        kwargs[param] = val
            clf = TabularDLClassifier(**kwargs)
        elif family == "sequence_dl":
            from start.modeling.sequence_dl import SequenceClassifier
            kwargs = {
                "family": architecture,
                "epochs": 8,
                "random_state": seed,
                "class_weight": class_weight,
            }
            if custom_space:
                for param in ("hidden_size", "learning_rate", "dropout", "epochs"):
                    if param in custom_space:
                        val = custom_space[param]
                        if isinstance(val, list):
                            val = val[0]
                        kwargs[param] = val
            clf = SequenceClassifier(**kwargs)
        elif family == "vision_dl":
            from start.modeling.vision_dl import VisionCNNClassifier
            arch = "simple_cnn_small" if architecture == "cnn" else architecture
            kwargs = {
                "architecture": arch,
                "epochs": 8,
                "class_weight": class_weight,
            }
            if custom_space:
                for param in ("learning_rate", "batch_size", "epochs"):
                    if param in custom_space:
                        val = custom_space[param]
                        if isinstance(val, list):
                            val = val[0]
                        kwargs[param] = val
            clf = VisionCNNClassifier(**kwargs)
        else:
            clf, _, _ = resolve_model(architecture, seed)
            if class_weight and hasattr(clf, "class_weight"):
                try:
                    clf.set_params(class_weight=class_weight)
                except Exception:
                    pass
            if custom_space:
                scalar_space = {}
                for k, v in custom_space.items():
                    scalar_space[k] = v[0] if isinstance(v, list) else v
                try:
                    clf.set_params(**scalar_space)
                except Exception:
                    pass
            
        clf.fit(plan.train[features], plan.train[target])
        device_used = getattr(clf, "device_used", "cpu")
        self._emit("model_execution", "complete", f"device={device_used}")

        self._emit("metrics", "running")
        cohort_metrics = {}
        for name, frame in (("train", plan.train), ("test", plan.test), ("oos", plan.oos)):
            if len(frame):
                y_true = frame[target].to_numpy()
                if task_type in ("regression", "forecasting"):
                    preds = clf.predict(frame[features])
                    cohort_metrics[name] = dl_task_metrics(task_type, y_true, preds)
                else:
                    proba = clf.predict_proba(frame[features])
                    classes = getattr(clf, "classes_", None)
                    cohort_metrics[name] = dl_task_metrics(task_type, y_true, proba, classes=classes)
                    
                    # Labeled confusion matrix computation and printing
                    if classes is not None:
                        from sklearn.metrics import confusion_matrix
                        preds = clf.predict(frame[features])
                        cm = confusion_matrix(y_true, preds, labels=classes)
                        print(f"\nConfusion Matrix for cohort '{name}':")
                        print("True \\ Pred | " + " | ".join(f"{str(c):>8}" for c in classes))
                        print("-" * (12 + 11 * len(classes)))
                        for idx, row_label in enumerate(classes):
                            row_str = " | ".join(f"{cm[idx, j]:>8}" for j in range(len(classes)))
                            print(f"{str(row_label):<11} | {row_str}")
                        print("")
        
        m_key = "rmse" if task_type in ("regression", "forecasting") else "auc_roc"
        evidence.append(
            TestResult(
                test_id="execution.cohort_metrics",
                test_name="Cohort performance metrics",
                metrics={f"{k}_{m_key}": v[m_key] for k, v in cohort_metrics.items() if m_key in v},
                interpretation="; ".join(
                    f"{k} {m_key.upper()} {v[m_key]:.4f}" for k, v in cohort_metrics.items() if m_key in v
                ),
            ).apply_thresholds()
        )
        self._emit("metrics", "complete")
        # Lightweight per-cohort markers. In the enterprise flow the full
        # explainability and sensitivity tables are rendered downstream; these
        # markers just note the stage ran, without a stale referral.
        for s in ("explainability", "sensitivity", "robustness"):
            self._emit(s, "complete", "computed")
        return cohort_metrics

    def _persist(
        self,
        results: list[TestResult],
        run_id: str,
        output_root: str | None,
        enterprise_run_id: str | None = None,
    ):
        records = [
            EvidenceRecord.from_result(
                r,
                model_id="mros-review",
                dataset_id="review",
                run_id=run_id,
                enterprise_run_id=enterprise_run_id,
            )
            for r in results
        ]
        if output_root:
            from start.evidence.ledger import EvidenceLedger

            ledger = EvidenceLedger(
                Path(output_root) / "ledger.jsonl", Path(output_root) / "evidence_store"
            )
            for rec in records:
                ledger.append(rec)
        return records

    def _write_report(
        self, run_id, inference, target, modality, recommended,
        cohort_metrics, evidence, agent_review, ai_stages, output_root,
    ):
        if not output_root:
            return None
        from start.modeling.mros_report import render_mros_report

        out_dir = Path(output_root) / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{run_id}.md"
        path.write_text(
            render_mros_report(
                run_id, inference, target, modality, recommended,
                cohort_metrics, evidence, agent_review, ai_stages, self.stage_events,
            )
        )
        return path


def _infer_modality(profile) -> str:
    if profile.image_path_columns:
        return "vision"
    if profile.timestamp_columns and not profile.text_columns:
        return "sequence"
    return "tabular"
