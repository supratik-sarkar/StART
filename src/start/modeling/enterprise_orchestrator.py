"""Enterprise layered review orchestrator (v2.0.0).

Wraps the existing visible pipeline into the explicit layered architecture the
enterprise spec requires:

    Data -> Model -> Validation -> Governance -> AI-Engineering -> Evidence
    -> Reporting

Each layer emits status, runtime, warnings, findings, artifacts, and evidence
IDs. This composes the frozen Layer 1-9 components and the v2 additions
(governance findings engine, executable AI-engineering adapters, enterprise
dashboard) WITHOUT modifying them. The original ``ReviewOrchestrator.run`` and
its outputs are unchanged; this is the opt-in enterprise flow.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from start.ai_engineering import run_ai_engineering_layer
from start.governance.findings import (
    Finding,
    FindingsRegister,
    Materiality,
    Severity,
    derive_findings_from_evidence,
)
from start.modeling.review_orchestrator import ReviewOrchestrator
from start.reporting.dashboard import DashboardModel, write_dashboard

LAYER_NAMES = (
    "Data",
    "Model",
    "Validation",
    "Governance",
    "AI-Engineering",
    "Evidence",
    "Reporting",
)


@dataclass
class LayerResult:
    name: str
    status: str = "pending"  # pending | running | complete | error | skipped
    runtime_seconds: float = 0.0
    warnings: list[str] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.name,
            "status": self.status,
            "runtime_seconds": self.runtime_seconds,
            "warnings": self.warnings,
            "findings": len(self.findings),
            "artifacts": len(self.artifacts),
            "evidence_ids": self.evidence_ids,
            "detail": self.detail,
        }


@dataclass
class EnterpriseOutcome:
    run_id: str
    task_type: str
    target: Any
    modality: str
    recommended_family: str
    layers: list[LayerResult]
    findings_register: FindingsRegister
    ai_engineering: Any
    dashboard_paths: dict[str, str]
    base_outcome: Any
    dashboard_model: Any = None
    graph_paths: list[str] = field(default_factory=list)
    data_statistics: Any = None
    fe_recommendations: Any = None
    architecture_review: Any = None
    tuning_plan: Any = None
    sensitivity: Any = None
    action_log: Any = None
    metric_choice: dict[str, Any] = field(default_factory=dict)
    dataset_source: Any = None
    trace_log: Any = None
    artifact_registry: Any = None
    activation_report: Any = None
    copilot_execution: Any = None
    tuning_run: Any = None
    kfold_tuning: Any = None
    review_session: Any = None

    @property
    def critique_ok(self) -> bool:
        return self.base_outcome.agent_review.critique_ok


class EnterpriseReviewOrchestrator:
    """Runs the review as explicit layers and produces the audit package."""

    def __init__(
        self,
        on_stage: Callable[[Any], None] | None = None,
        on_layer: Callable[[LayerResult], None] | None = None,
        on_adapter: Callable[[Any], None] | None = None,
        on_adapter_start: Callable[[str, str], None] | None = None,
    ) -> None:
        self.on_stage = on_stage
        self.on_layer = on_layer
        self.on_adapter = on_adapter
        self.on_adapter_start = on_adapter_start
        self.layers: list[LayerResult] = []

    def _on_adapter(self, result: Any) -> None:
        if self.on_adapter:
            self.on_adapter(result)

    def _on_adapter_start(self, name: str, activity: str) -> None:
        if self.on_adapter_start:
            self.on_adapter_start(name, activity)

    def _layer(self, name: str) -> LayerResult:
        lr = LayerResult(name=name, status="running")
        self.layers.append(lr)
        if self.on_layer:
            self.on_layer(lr)
        return lr

    def _finish(self, lr: LayerResult, t0: float, detail: str = "") -> None:
        lr.runtime_seconds = round(time.perf_counter() - t0, 4)
        lr.status = "complete"
        lr.detail = detail
        if self.on_layer:
            self.on_layer(lr)

    def run(
        self,
        df: pd.DataFrame,
        *,
        user_target: str | list[str] | None = None,
        task_override: str | None = None,
        split_strategy: str = "stratified",
        agent_mode: str = "deterministic",
        llm: Any = None,
        output_root: str = "start_output",
        run_dl: bool = False,
        enterprise_mode: bool = False,
        seed: int = 42,
        cnn_config: dict[str, Any] | None = None,
        architecture: str = "mlp",
        activation: str = "relu",
        costlier_errors: str = "balanced",
        dataset_source: Any = None,
        requested_provider: str | None = None,
        split_props: tuple[float, float, float] = (0.60, 0.20, 0.20),
        explain_method: str = "integrated_gradients",
        tuning_strategy: str = "bounded_random_search",
        tuning_trials: int = 5,
        session: Any = None,
        k_folds: int = 5,
        validation: str = "holdout",
        class_weight: str | None = None,
        custom_space: dict[str, Any] | None = None,
        cost_specification: dict[str, Any] | None = None,
    ) -> EnterpriseOutcome:
        from start.agents.engineering_agents import (
            ArchitectureReviewAgent,
            HyperparameterTuningAgent,
            select_primary_metric,
        )
        from start.modeling.data_statistics import compute_data_statistics
        from start.modeling.dataset_source import describe_demo_dataset
        from start.modeling.fe_recommendations import recommend_feature_engineering
        from start.reporting.agent_trace import TraceLog
        from start.reporting.artifacts import ArtifactRegistry
        from start.reporting.progress import ActionLog

        run_id = "RUN-ENT-" + uuid.uuid4().hex[:8]
        register = FindingsRegister()
        action_log = ActionLog()
        trace_log = TraceLog()
        artifact_registry = ArtifactRegistry()

        # --- LLM activation preflight (Section A): visible, never silent ---
        from start.providers.llm_activation import preflight_llm
        # Prefer the explicitly requested provider so a fallback is reported
        # against what the user actually chose (e.g. "openai -> FALLBACK"),
        # not the degraded object's name ("none").
        if requested_provider and requested_provider not in ("none", ""):
            provider_name = requested_provider
        else:
            provider_name = getattr(llm, "name", "none") if llm is not None else "none"
        activation_report = preflight_llm(provider_name, llm)
        # Backend/llm_used context for agent traces (Section N): an agent used
        # the LLM only when the provider is genuinely CONNECTED.
        _llm_used = activation_report.status == "CONNECTED"
        _backend = provider_name if _llm_used else "deterministic"
        _fallback = (
            activation_report.detail
            if (not _llm_used and activation_report.status == "FALLBACK")
            else ""
        )

        # --- Data + Model + Validation + Governance via the visible pipeline ---
        # The base orchestrator runs the full Layer 1-9 flow with stage streaming.
        t0 = time.perf_counter()
        data_layer = self._layer("Data")
        base = ReviewOrchestrator(on_stage=self.on_stage)
        base_outcome = base.run(
            df,
            user_target=user_target,
            task_override=task_override,
            split_strategy=split_strategy,
            agent_mode=agent_mode,
            llm=llm,
            output_root=output_root,
            run_dl=run_dl,
            seed=seed,
            architecture=architecture,
            activation=activation,
        )
        evidence_ids = [getattr(r, "evidence_id", r.test_id) for r in base_outcome.evidence]

        # Attribute evidence to layers by test_id prefix.
        data_ids = [
            getattr(r, "evidence_id", r.test_id)
            for r in base_outcome.evidence
            if r.test_id.startswith(("discovery", "split", "feature_engineering"))
        ]
        data_layer.evidence_ids = data_ids
        self._finish(data_layer, t0, f"{len(df)} rows; modality {base_outcome.modality}")

        # --- Co-pilot: data statistics + FE recommendations (visible intel) ---
        single_target = (
            base_outcome.target if isinstance(base_outcome.target, str)
            else base_outcome.target[0]
        )
        data_stats = compute_data_statistics(df, single_target)
        if dataset_source is None:
            dataset_source = describe_demo_dataset(df, single_target)
        action_log.record(
            "DatasetDiscoveryAgent", f"{len(df)} rows x {df.shape[1]} cols",
            "computed data statistics",
            recommendation=f"suggested split: {data_stats.suggested_split}",
            evidence_ids=data_ids[:1],
        )
        trace_log.record(
            "DatasetDiscoveryAgent",
            inputs=f"{len(df)} rows x {df.shape[1]} cols",
            decision=f"profiled dataset; suggested split = {data_stats.suggested_split}",
            reasoning=f"{data_stats.n_numeric} numeric / {data_stats.n_categorical} "
            f"categorical; imbalance {data_stats.imbalance_warning}",
            evidence_ids=data_ids[:1], confidence=0.9,
            alternative_considered="random split",
            action_taken="emitted initial data statistics",
        )
        trace_log.record(
            "TaskInferenceAgent",
            inputs=f"target '{single_target}', type {data_stats.target_type}",
            decision=base_outcome.task_type,
            reasoning=f"target has type {data_stats.target_type}",
            confidence=0.95, alternative_considered="regression"
            if data_stats.target_type != "continuous" else "classification",
            action_taken="set task type for downstream agents",
        )
        fe_modality = {
            "tabular": "tabular", "sequence": "sequential", "vision": "vision"
        }.get(base_outcome.modality, "tabular")
        fe_recs = recommend_feature_engineering(
            data_stats,
            modality=fe_modality,
            cost_specification=cost_specification,
        )
        if fe_recs.applicable():
            action_log.record(
                "FeatureEngineeringAgent", "data statistics",
                f"recommended {len(fe_recs.applicable())} preprocessing step(s)",
                recommendation="; ".join(r.step for r in fe_recs.applicable()[:5]),
                evidence_ids=[r.evidence_id for r in fe_recs.applicable()[:3]],
            )
            trace_log.record(
                "FeatureEngineeringAgent",
                inputs="initial data statistics",
                decision=f"recommended {len(fe_recs.applicable())} preprocessing step(s)",
                reasoning="; ".join(r.step for r in fe_recs.applicable()[:5]),
                evidence_ids=[r.evidence_id for r in fe_recs.applicable()[:3]],
                confidence=0.85, alternative_considered="use raw features",
                action_taken="surfaced recommendations for user accept/override",
            )

        # --- Co-pilot: metric choice + architecture review + tuning plan ---
        metric_choice = select_primary_metric(
            base_outcome.task_type, costlier_errors=costlier_errors
        )
        arch_review = ArchitectureReviewAgent().review(
            user_family=architecture, user_activation=activation,
            modality=base_outcome.modality, n_samples=len(df),
            n_features=df.shape[1] - 1, task_type=base_outcome.task_type,
            imbalanced="severe" in data_stats.imbalance_warning
            or "moderate" in data_stats.imbalance_warning,
        )
        action_log.record(
            "ArchitectureReviewAgent", f"{architecture}+{activation}",
            "reviewed architecture choice",
            recommendation=f"{arch_review.recommendation['family']}+"
            f"{arch_review.recommendation['activation']}",
            evidence_ids=[arch_review.evidence_id],
            user_decision="agrees" if arch_review.agrees else "review needed",
        )
        trace_log.record(
            "ArchitectureReviewAgent",
            inputs=f"user choice {architecture}+{activation}; "
            f"{df.shape[1] - 1} features, {len(df)} rows, {base_outcome.modality}",
            decision=f"recommend {arch_review.recommendation['family']}+"
            f"{arch_review.recommendation['activation']}",
            reasoning=arch_review.reason,
            evidence_ids=[arch_review.evidence_id],
            confidence=0.75 if not arch_review.agrees else 0.9,
            alternative_considered=f"{architecture}+{activation}",
            action_taken="surfaced choice for user accept/keep",
            user_decision="agrees" if arch_review.agrees else "review needed",
        )
        # #2: if the user overrode the recommendation (via the interactive
        # session), the execution trace must explicitly say the override was
        # honored — the model that actually trains is the user's choice.
        if session is not None:
            arch_dec = session.decision_for("architecture")
            if arch_dec and arch_dec.effective != arch_dec.recommended:
                trace_log.record(
                    "ModelExecutionAgent",
                    inputs=f"user override: {arch_dec.effective} "
                    f"(recommended {arch_dec.recommended})",
                    decision=f"train user-selected {arch_dec.effective}",
                    reasoning="User override honored: the model under review is "
                    "the user's architecture, not the recommendation.",
                    evidence_ids=[arch_review.evidence_id], confidence=1.0,
                    alternative_considered=f"recommended {arch_dec.recommended}",
                    action_taken="honored user override in execution",
                    user_decision="override honored",
                )
        tuning_plan = HyperparameterTuningAgent().plan(
            task_type=base_outcome.task_type, family=architecture,
            n_samples=len(df), costlier_errors=costlier_errors,
            n_trials=tuning_trials,
        )
        action_log.record(
            "HyperparameterTuningAgent", "search space",
            f"planned {tuning_plan.n_trials}-trial {tuning_plan.strategy}",
            recommendation=f"metric: {tuning_plan.primary_metric}",
            evidence_ids=[tuning_plan.evidence_id],
        )
        trace_log.record(
            "HyperparameterTuningAgent",
            inputs=f"task {base_outcome.task_type}, {len(df)} rows, cost={costlier_errors}",
            decision=f"{tuning_plan.n_trials}-trial {tuning_plan.strategy}, "
            f"metric {tuning_plan.primary_metric}",
            reasoning=f"metric routed by cost preference '{costlier_errors}'; "
            f"{tuning_plan.validation} (no test/OOS leakage)",
            evidence_ids=[tuning_plan.evidence_id], confidence=0.8,
            alternative_considered="exhaustive grid search",
            action_taken="planned bounded, leakage-safe tuning",
        )

        t0 = time.perf_counter()
        model_layer = self._layer("Model")
        model_layer.detail = f"recommended {base_outcome.recommended_family}"
        self._finish(model_layer, t0, model_layer.detail)

        t0 = time.perf_counter()
        val_layer = self._layer("Validation")
        val_ids = [
            getattr(r, "evidence_id", r.test_id)
            for r in base_outcome.evidence
            if r.test_id.startswith(("execution", "deep_learning"))
        ]
        val_layer.evidence_ids = val_ids
        # --- Co-pilot: sensitivity analysis (real, when a tabular model trained) ---
        sensitivity_result = None
        copilot_exec = None
        tuning_run = None
        kfold_tuning = None
        if run_dl and base_outcome.cohort_metrics and base_outcome.modality == "tabular":
            _apply_winsorize = True
            if session is not None and session.rejected("fe:outliers"):
                _apply_winsorize = False

            sensitivity_result = self._run_sensitivity(
                df, single_target, metric_choice["primary_metric"], seed,
                architecture=architecture, task_type=base_outcome.task_type,
                activation=activation, winsorize=_apply_winsorize
            )
            if sensitivity_result:
                action_log.record(
                    "ValidationPlannerAgent",
                    f"top features, metric={sensitivity_result.metric_name}",
                    "ran feature-shock sensitivity analysis",
                    recommendation=f"most sensitive: {sensitivity_result.most_sensitive_feature}",
                    evidence_ids=val_ids[:1],
                )
            # --- Co-pilot execution: split table, metrics-by-split, training,
            # explainability (Sections D/G/I/J/K), all as registered artifacts ---
            from start.modeling.copilot_execution import run_copilot_execution

            # #2: the user's FE decision on correlation pruning drives execution.
            # Default is to prune; if the user explicitly rejected it, keep all.
            _apply_pruning = True
            if session is not None and session.rejected("fe:correlation_pruning"):
                _apply_pruning = False
            copilot_exec = run_copilot_execution(
                df, single_target, split_props=split_props,
                metric_name=metric_choice["primary_metric"],
                explain_method=explain_method, seed=seed,
                output_root=output_root, run_id=run_id, registry=artifact_registry,
                apply_correlation_pruning=_apply_pruning,
                architecture=architecture,
                stratify=(split_strategy == "stratified"),
                class_weight=class_weight,
                task_type=base_outcome.task_type,
                activation=activation,
                winsorize=_apply_winsorize,
                custom_space=custom_space,
            )
            if copilot_exec:
                if copilot_exec.pruned_features:
                    trace_log.record(
                        "FeatureEngineeringAgent",
                        inputs=f"{len(copilot_exec.feature_columns)} features after pruning",
                        decision=f"dropped {len(copilot_exec.pruned_features)} "
                        "highly-correlated feature(s)",
                        reasoning="Correlation pruning applied (>0.95); user did not reject it.",
                        evidence_ids=[], confidence=0.8,
                        alternative_considered="keep all features",
                        action_taken="pruned correlated features before training",
                    )
                elif session is not None and session.rejected("fe:correlation_pruning"):
                    trace_log.record(
                        "FeatureEngineeringAgent",
                        inputs="user rejected correlation pruning",
                        decision="kept all features",
                        reasoning="User override honored: correlation pruning was "
                        "declined, so no features were dropped.",
                        evidence_ids=[], confidence=1.0,
                        alternative_considered="prune correlated features",
                        action_taken="honored user rejection of pruning",
                    )
                trace_log.record(
                    "ModelExecutionAgent",
                    inputs=f"train/test/oos split {split_props}",
                    decision=f"trained {architecture}; explainability via {copilot_exec.explainability_method}",
                    reasoning=f"generalization gap {copilot_exec.generalization_gap}",
                    evidence_ids=val_ids[:1], confidence=0.85,
                    alternative_considered="diagnostics-only (no training)",
                    action_taken=f"emitted {len(copilot_exec.artifacts)} execution artifact(s)",
                )
                # --- Real hyperparameter tuning (Section H): actually runs ---
                from start.modeling.copilot_execution import _stratified_split
                from start.modeling.tuning_run import run_tuning

                _effective_stratify = (split_strategy == "stratified") and (base_outcome.task_type not in ("regression", "forecasting"))
                _splits = _stratified_split(df, single_target, split_props, seed, stratify=_effective_stratify)
                _train_only = _splits["train"]

                tuning_run = run_tuning(
                    _train_only, single_target, copilot_exec.feature_columns,
                    strategy=tuning_strategy, n_trials=tuning_trials,
                    primary_metric=metric_choice["primary_metric"], seed=seed,
                    output_root=output_root, run_id=run_id, registry=artifact_registry,
                    architecture=architecture, activation=activation, task_type=base_outcome.task_type,
                    custom_space=custom_space, validation=validation, k_folds=k_folds,
                    cost_specification=cost_specification,
                )
                if tuning_run:
                    # record the executed outcome on the planned tuning object
                    if tuning_run.ran:
                        tuning_plan = HyperparameterTuningAgent().record_outcome(
                            tuning_plan, tuning_run.best_params, tuning_run.rejected_params
                        )
                    trace_log.record(
                        "HyperparameterTuningAgent",
                        inputs=f"{tuning_run.n_trials}-trial {tuning_run.strategy}",
                        decision=(f"best metric {tuning_run.best_metric:.4f} "
                                  f"@ {tuning_run.best_params}") if tuning_run.ran
                        else tuning_run.note,
                        reasoning=f"train-internal {tuning_run.validation} only (no test/OOS leakage)",
                        evidence_ids=[tuning_plan.evidence_id],
                        confidence=0.85 if tuning_run.ran else 0.5,
                        alternative_considered="grid search",
                        action_taken=f"ran {len(tuning_run.trials)} trial(s)"
                        if tuning_run.ran else "tuning disabled by user",
                    )

                # v3.1.1: legacy run_kfold_tuning (logistic regression
                # C/class_weight tuner) removed. The unified run_tuning now
                # handles K-fold CV for the selected architecture.
                kfold_tuning = None
        self._finish(val_layer, t0, f"{len(base_outcome.cohort_metrics)} cohorts scored")

        # --- Governance layer: derive findings from evidence ---
        t0 = time.perf_counter()
        gov_layer = self._layer("Governance")
        derived = derive_findings_from_evidence(base_outcome.evidence)
        register.extend(derived)
        gov_layer.findings = list(derived)
        # informational governance finding tying the sign-off to evidence
        signoff_finding = Finding(
            title="Governance sign-off disposition",
            description=base_outcome.agent_review.signoff,
            severity=Severity.LOW if base_outcome.agent_review.critique_ok else Severity.MEDIUM,
            materiality=Materiality.MEDIUM,
            risk_category="Governance",
            evidence_ids=evidence_ids[:3],
            recommendation="Review the governance findings before approval.",
            source="governance",
        )
        register.add(signoff_finding)
        gov_layer.findings.append(signoff_finding)
        self._finish(gov_layer, t0, f"{len(register.findings)} findings")

        # --- AI-Engineering layer: executable adapters ---
        t0 = time.perf_counter()
        ai_layer = self._layer("AI-Engineering")
        ai_report = run_ai_engineering_layer(
            {
                "run_id": run_id,
                "n_stages": len(base_outcome.stage_events),
                "evidence_count": len(base_outcome.evidence),
            },
            output_root=output_root,
            on_adapter=self._on_adapter,
            on_adapter_start=self._on_adapter_start,
        )
        register.extend(ai_report.findings)
        ai_layer.findings = list(ai_report.findings)
        ai_layer.artifacts = [a.path for a in ai_report.artifacts]
        ai_layer.evidence_ids = [e.test_id for e in ai_report.evidence]
        if ai_report.available_count < ai_report.total:
            ai_layer.warnings.append(
                f"{ai_report.total - ai_report.available_count} AI-engineering "
                "backends not installed (reported explicitly)."
            )
        self._finish(
            ai_layer, t0,
            f"{ai_report.available_count}/{ai_report.total} adapters available",
        )

        # --- optional enterprise graph execution mode ---
        graph_paths: list[str] = []
        if enterprise_mode:
            graph_paths = self._run_graph_mode(run_id, base_outcome, output_root)
            for gp in graph_paths:
                artifact_registry.register(gp, category="graph")
        # register AI-engineering artifacts produced by adapters
        for art in ai_report.artifacts:
            artifact_registry.register(
                getattr(art, "path", str(art)), category="ai_engineering"
            )

        # --- Evidence layer: EvidenceCritic gate over findings ---
        t0 = time.perf_counter()
        ev_layer = self._layer("Evidence")
        uncited = register.uncited()
        if uncited:
            ev_layer.warnings.append(f"{len(uncited)} uncited finding(s) flagged by EvidenceCritic.")
        ev_layer.evidence_ids = evidence_ids
        self._finish(
            ev_layer, t0,
            f"{len(evidence_ids)} evidence records; critique "
            f"{'PASSED' if base_outcome.agent_review.critique_ok else 'FAILED'}",
        )

        # --- Reporting layer: enterprise dashboard ---
        action_log.record(
            "GovernanceSignoffAgent", f"{len(register.findings)} findings",
            "produced sign-off disposition",
            recommendation=base_outcome.agent_review.signoff.split(".")[0],
            evidence_ids=evidence_ids[:2],
        )
        trace_log.record(
            "GovernanceSignoffAgent",
            inputs=f"{len(register.findings)} findings, {len(evidence_ids)} evidence records",
            decision=base_outcome.agent_review.signoff.split(".")[0],
            reasoning=f"{register.summary()['total']} findings weighed against acceptance criteria",
            evidence_ids=evidence_ids[:2], confidence=0.85,
            alternative_considered="conditional sign-off",
            action_taken="recorded sign-off disposition",
        )
        action_log.record(
            "EvidenceCriticAgent", "all findings + narrative",
            "ran citation gate",
            recommendation="PASSED" if base_outcome.agent_review.critique_ok else "FAILED",
            evidence_ids=evidence_ids[:1],
        )
        trace_log.record(
            "EvidenceCriticAgent",
            inputs="all findings, recommendations, sign-off narrative",
            decision="PASSED" if base_outcome.agent_review.critique_ok else "FAILED",
            reasoning="every claim must cite evidence; uncited claims are flagged",
            evidence_ids=evidence_ids[:1],
            confidence=1.0 if base_outcome.agent_review.critique_ok else 0.5,
            alternative_considered="allow uncited narrative",
            action_taken=f"{len(uncited)} uncited finding(s) flagged"
            if uncited else "all findings cited",
        )
        t0 = time.perf_counter()
        rep_layer = self._layer("Reporting")
        # Section N: stamp every trace with the real backend / llm_used context.
        for _tr in trace_log.traces:
            _tr.backend = _backend
            _tr.llm_used = _llm_used
            _tr.fallback_reason = _fallback
        dashboard_paths = self._build_dashboard(
            run_id, base_outcome, register, ai_report, cnn_config, output_root,
            data_stats=data_stats, fe_recs=fe_recs, arch_review=arch_review,
            tuning_plan=tuning_plan, sensitivity=sensitivity_result,
            action_log=action_log, metric_choice=metric_choice,
            dataset_source=dataset_source, trace_log=trace_log,
            activation_report=activation_report, artifact_registry=artifact_registry,
            copilot_exec=copilot_exec, tuning_run=tuning_run,
            review_session=session,
        )
        for dp in dashboard_paths.values():
            artifact_registry.register(dp, category="report")
        rep_layer.artifacts = list(dashboard_paths.values())
        self._finish(rep_layer, t0, "dashboard.html/.json/.md generated")

        return EnterpriseOutcome(
            run_id=run_id,
            task_type=base_outcome.task_type,
            target=base_outcome.target,
            modality=base_outcome.modality,
            recommended_family=base_outcome.recommended_family,
            layers=self.layers,
            findings_register=register,
            ai_engineering=ai_report,
            dashboard_paths=dashboard_paths,
            dashboard_model=getattr(self, "_last_dashboard_model", None),
            base_outcome=base_outcome,
            graph_paths=graph_paths,
            data_statistics=data_stats,
            fe_recommendations=fe_recs,
            architecture_review=arch_review,
            tuning_plan=tuning_plan,
            sensitivity=sensitivity_result,
            action_log=action_log,
            metric_choice=metric_choice,
            dataset_source=dataset_source,
            trace_log=trace_log,
            artifact_registry=artifact_registry,
            activation_report=activation_report,
            copilot_execution=copilot_exec,
            tuning_run=tuning_run,
            kfold_tuning=kfold_tuning,
            review_session=session,
        )

    def _run_sensitivity(self, df, target, metric_name, seed, architecture="mlp", task_type="binary_classification", activation="relu", winsorize=False):
        """Train a quick tabular model on the same data and shock its top
        features. Real computation; uses model coefficients/importance proxy
        via variance of standardized features to pick the top set."""
        try:

            from start.modeling.models import resolve_model
            from start.modeling.sensitivity_analysis import run_sensitivity_analysis
            from start.modeling.tuning_run import _model_family

            features = [
                c for c in df.columns
                if c != target and pd.api.types.is_numeric_dtype(df[c])
            ]
            if len(features) < 2:
                return None
            X, y = df[features], df[target].to_numpy()
            
            family = _model_family(architecture)
            if family == "sklearn":
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy="median")
                X = pd.DataFrame(imputer.fit_transform(X), columns=features)
            if family == "tabular_dl":
                from start.modeling.tabular_dl import TabularDLClassifier
                clf = TabularDLClassifier(
                    task=task_type, family=architecture, activation=activation, epochs=8, random_state=seed, winsorize=winsorize
                )
            elif family == "sequence_dl":
                from start.modeling.sequence_dl import SequenceClassifier
                clf = SequenceClassifier(
                    family=architecture, epochs=8, random_state=seed,
                )
            elif family == "vision_dl":
                from start.modeling.vision_dl import VisionCNNClassifier
                arch = "simple_cnn_small" if architecture == "cnn" else architecture
                clf = VisionCNNClassifier(
                    architecture=arch, epochs=8, random_state=seed,
                )
            else:
                clf, _, _ = resolve_model(architecture, seed)
            
            clf.fit(X, y)
            # top features by standardized variance (cheap, model-agnostic proxy)
            variances = X.std().sort_values(ascending=False)
            top = list(variances.index[:5])
            metric = metric_name if metric_name in ("auc_roc", "pr_auc", "recall", "f1", "rmse", "mae", "r2") else ("rmse" if task_type in ("regression", "forecasting") else "auc_roc")
            return run_sensitivity_analysis(clf, X, y, top_features=top, metric_name=metric)
        except Exception:
            return None

    def _run_graph_mode(self, run_id, base_outcome, output_root) -> list[str]:
        from start.modeling.graph_orchestrator import GraphReviewOrchestrator

        g = GraphReviewOrchestrator(output_root=output_root)
        # Represent the completed pipeline as a DAG for the audit artifact.
        prev = None
        for stage in ("data", "model", "validation", "governance", "ai_engineering", "reporting"):
            g.add_node(stage, lambda s: {}, depends_on=[prev] if prev else [])
            prev = stage
        state = g.run(run_id, initial_state={"run_id": run_id})
        return g.write_graph_artifacts(run_id, state)

    def _build_dashboard(
        self, run_id, base_outcome, register, ai_report, cnn_config, output_root,
        *, data_stats=None, fe_recs=None, arch_review=None, tuning_plan=None,
        sensitivity=None, action_log=None, metric_choice=None, dataset_source=None,
        trace_log=None, activation_report=None, artifact_registry=None,
        copilot_exec=None, tuning_run=None, review_session=None,
    ) -> dict[str, str]:
        ai_rows = ai_report.summary_rows()
        evidence_rows = [
            {
                "evidence_id": getattr(r, "evidence_id", r.test_id),
                "test_name": r.test_name,
                "status": r.status.value,
            }
            for r in base_outcome.evidence
        ]
        model = DashboardModel(
            run_id=run_id,
            task_type=base_outcome.task_type,
            target=base_outcome.target,
            modality=base_outcome.modality,
            recommended_family=base_outcome.recommended_family,
            cohort_metrics=base_outcome.cohort_metrics,
            dataset_summary=next(
                (r.interpretation for r in base_outcome.evidence
                 if r.test_id.startswith("discovery.dataset")),
                "",
            ),
            model_summary=f"Recommended family: {base_outcome.recommended_family}.",
            cnn_config=cnn_config,
            explainability=(
                {"note": f"Global feature importance shown below "
                 f"({copilot_exec.explainability_method})."}
                if copilot_exec and copilot_exec.global_importance
                else {"note": "No model execution (diagnostics-only review)."}
            ),
            robustness=(
                {"note": "Feature-shock sensitivity analysis shown below."}
                if sensitivity
                else {"note": "No sensitivity run (diagnostics-only review)."}
            ),
            ai_engineering_rows=ai_rows,
            findings=register.to_list(),
            evidence_summary=register.summary() | {"total": len(base_outcome.evidence)},
            evidence_rows=evidence_rows,
            signoff=base_outcome.agent_review.signoff,
            critique_ok=base_outcome.agent_review.critique_ok,
            stage_timeline=[
                {"stage": e.stage, "status": e.status, "detail": e.detail}
                for e in base_outcome.stage_events
            ],
            data_statistics=data_stats.to_dict() if data_stats else None,
            fe_recommendations=fe_recs.to_list() if fe_recs else [],
            architecture_review=arch_review.to_dict() if arch_review else None,
            tuning_plan=tuning_plan.to_dict() if tuning_plan else None,
            sensitivity=sensitivity.to_dict() if sensitivity else None,
            action_log=action_log.to_list() if action_log else [],
            metric_choice=metric_choice,
            dataset_source=dataset_source.to_dict() if dataset_source else None,
            agent_traces=trace_log.to_list() if trace_log else [],
            activation_report=activation_report.to_dict() if activation_report else None,
            control_surface=ai_report.control_surface(),
            artifact_catalog=artifact_registry.to_list() if artifact_registry else [],
            copilot_execution=copilot_exec.to_dict() if copilot_exec else None,
            tuning_run=tuning_run.to_dict() if tuning_run else None,
            review_journey=review_session.to_dict() if review_session else None,
        )
        self._last_dashboard_model = model
        return write_dashboard(model, output_root, run_id)
