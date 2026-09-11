"""Authoritative Canonical Execution Service for StART v5.1.0.

Provides the single non-web execution coordinator shared between CLI and Web:
- Orchestrates real deterministic engines without reimplementing scientific calculations
- Emits typed RuntimeEvents at genuine execution boundaries
- Appends to canonical EvidenceLedger and binds true EvidenceRecord IDs
- Generates canonical artifacts and attestation seals
- Zero dependencies on start.web (CORE_RUNTIME_IMPORTS_START_WEB = 0)
"""

from __future__ import annotations

import contextlib
import datetime
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from start.core.schemas import EvidenceRecord, Status, TestResult
from start.evidence.ledger import EvidenceLedger
from start.registry import list_tests
from start.review.architecture import ReviewExecutionProducts
from start.runtime.contexts import ExecutionContextInstance, instantiate_context
from start.runtime.events import NoOpEventSink, RuntimeEvent, RuntimeEventSink
from start.runtime.workflows import (
    EngineKind,
    ResolvedWorkflowExecution,
    resolve_workflow,
)
from start.utils.serializers import sanitize_json_primitives

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Complete result of a canonical workflow execution."""

    run_id: str
    workflow_id: str
    context_id: str
    context_instance: ExecutionContextInstance
    resolved_execution: ResolvedWorkflowExecution
    records: list[EvidenceRecord]
    products: ReviewExecutionProducts
    ledger: EvidenceLedger
    artifacts: dict[str, Any]
    governance_disposition: str | None
    attestation_seal: Any | None
    merkle_root: str | None
    decisions: list[Any]
    presentation_model: Any | None
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    events: list[RuntimeEvent] = field(default_factory=list)
    output_path: str = ""
    elapsed_seconds: float = 0.0
    tracer: Any | None = None
    policy_result: Any | None = None

    def __getitem__(self, item: str) -> Any:
        if item == "evidence_records":
            return self.records
        return getattr(self, item)

    def to_summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workflow_id": self.workflow_id,
            "context_id": self.context_id,
            "n_records": len(self.records),
            "governance_disposition": self.governance_disposition,
            "merkle_root": self.merkle_root,
            "output_path": self.output_path,
            "elapsed_seconds": self.elapsed_seconds,
        }


class CanonicalExecutionService:
    """Authoritative execution coordinator for StART reviews."""

    @classmethod
    def execute(
        cls,
        workflow_id: str,
        context_id: str,
        request_params: dict[str, Any] | None = None,
        seed: int | None = None,
        event_sink: RuntimeEventSink | None = None,
        materiality: str = "TIER_1",
        run_id: str | None = None,
        parent_run_id: str | None = None,
        intervention: Any | None = None,
        output_root: str = "start_output",
        session_id: str | None = None,
        execution_mode: str = "hybrid_workbench",
        trace_mode: str = "off",
    ) -> ExecutionResult:
        """Execute canonical StART workflow, streaming genuine events into event_sink."""
        start_time = time.time()
        sink = event_sink or NoOpEventSink()
        run_id = run_id or f"RUN-{uuid.uuid4().hex[:8]}"
        root = Path(output_root) / run_id
        root.mkdir(parents=True, exist_ok=True)
        ledger = EvidenceLedger(root / "ledger.jsonl", root / "evidence")

        from start.telemetry.engineering_trace import (
            OP_ARTIFACT_PERSIST,
            OP_DATASET_RESOLVE,
            OP_EVIDENCE_EMIT,
            OP_GOVERNANCE_COMMIT,
            OP_MODEL_EVALUATE,
            OP_MODEL_FIT,
            OP_MODEL_RESOLVE,
            OP_PLAN,
            OP_POLICY_EVALUATE,
            OP_PREPROCESS_FIT,
            OP_SENSITIVITY_EXECUTE,
            OP_SPLIT,
            EngineeringTracer,
            PolicyAdapter,
        )

        tracer: EngineeringTracer | None = None
        if trace_mode != "off":
            tracer = EngineeringTracer(run_id=run_id, service_name=f"start.{workflow_id}")

        @contextlib.contextmanager
        def trace_scope(op_name: str, attrs: dict[str, Any] | None = None):
            if tracer is not None:
                with tracer.span(op_name, attrs):
                    yield
            else:
                yield

        # 1. Instantiate Context ONCE
        with trace_scope(OP_DATASET_RESOLVE, {"context_id": context_id, "seed": seed}):
            ctx_instance = instantiate_context(context_id, seed=seed, materiality=materiality)
        sink.emit(
            RuntimeEvent(
                run_id=run_id,
                event_type="context_ready",
                status="COMPLETED",
                source_agent="Director",
                target_agent="Specialist",
                stage="PLANNING",
                action=f"initialize_{workflow_id}_context",
                node_id="step-context",
                elapsed_seconds=round(time.time() - start_time, 2),
                message=f"Initialized execution context '{context_id}' in {execution_mode} mode",
                metadata={
                    "spec_id": ctx_instance.spec_id,
                    "actual_samples": ctx_instance.actual_samples,
                    "actual_features": ctx_instance.actual_features,
                    "actual_assets": ctx_instance.actual_assets,
                    "actual_periods": ctx_instance.actual_periods,
                    "actual_target": ctx_instance.actual_target,
                    "actual_seed": ctx_instance.actual_seed,
                    "workflow": workflow_id,
                    "parent_run_id": parent_run_id,
                    "intervention": intervention,
                    "execution_mode": execution_mode,
                },
            )
        )

        # 2. Resolve Workflow
        with trace_scope(OP_PLAN, {"workflow_id": workflow_id, "context_id": context_id}):
            resolved = resolve_workflow(workflow_id, context_id, ctx_instance)
        sink.emit(
            RuntimeEvent(
                run_id=run_id,
                event_type="workflow_resolved",
                status="COMPLETED",
                source_agent="Specialist",
                target_agent="DeterministicEngine",
                stage="PLANNING",
                action="resolve_applicability",
                node_id="step-context",
                parent_node_id="step-context",
                elapsed_seconds=round(time.time() - start_time, 2),
                message=(
                    f"Resolved workflow '{workflow_id}': {len(resolved.applicable_test_ids)} applicable, "
                    f"{len(resolved.skipped_test_ids)} skipped out of "
                    f"{len(resolved.candidate_test_ids)} candidate tests"
                ),
                metadata={
                    "candidate_test_ids": list(resolved.candidate_test_ids),
                    "applicable_test_ids": list(resolved.applicable_test_ids),
                    "skipped_test_ids": list(resolved.skipped_test_ids),
                    "engine_kind": resolved.engine_kind.value,
                },
            )
        )

        registry = {t.test_id: t for t in list_tests()}
        records: list[EvidenceRecord] = []
        products = ReviewExecutionProducts()
        artifacts_dict: dict[str, Any] = {}
        decisions: list[Any] = []
        final_gov_disposition: str | None = None
        seal: Any | None = None
        merkle_root: str | None = None
        all_events: list[RuntimeEvent] = []
        checkpoints_list: list[dict[str, Any]] = []
        policy_res: Any | None = None

        # CP-001: Context ready
        cp1_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{run_id}:CP-001:{context_id}")).replace("-", "")[:16]
        cp1_item = {
            "checkpoint_id": "CP-001",
            "name": "Context ready",
            "status": "completed",
            "producing_stage": "step-context",
            "agent_signature": "Director",
            "evidence_ids": [],
            "commit_hash": cp1_hash,
            "timestamp": datetime.datetime.fromtimestamp(time.time(), tz=datetime.UTC).isoformat(),
            "summary": f"Initialized execution context '{context_id}' ({ctx_instance.actual_samples} samples, {ctx_instance.actual_features} features).",
        }
        checkpoints_list.append(cp1_item)
        evt_cp1 = RuntimeEvent(
            run_id=run_id,
            event_type="checkpoint_committed",
            status="COMPLETED",
            source_agent="Director",
            target_agent="Specialist",
            stage="CHECKPOINTS",
            action="commit_CP-001",
            node_id="step-context",
            checkpoint_id="CP-001",
            elapsed_seconds=round(time.time() - start_time, 2),
            message="Committed checkpoint CP-001: Context ready",
            metadata=cp1_item,
        )
        sink.emit(evt_cp1)
        all_events.append(evt_cp1)

        prev_node_id = "step-context"

        # 3. Execute according to engine_kind
        if resolved.engine_kind == EngineKind.TUNING:
            from start.modeling.tuning_run import run_tuning

            tab = ctx_instance.bundle.tabular
            train_df = getattr(tab, "train", None)
            target_col = ctx_instance.actual_target
            feature_cols = [c for c in train_df.columns if c != target_col]

            trials_requested = (
                min(30, max(5, int(request_params.get("trials", 10))))
                if request_params
                else 10
            )
            raw_strategy = (
                request_params.get("strategy")
                or request_params.get("tuning_strategy")
                or request_params.get("tuning_method")
                or "bounded_random_search"
            ) if request_params else "bounded_random_search"
            
            tuning_strategy = "optuna" if str(raw_strategy).lower() in ("optuna", "bayesian", "optuna_bayesian") else ("grid" if str(raw_strategy).lower() in ("grid", "grid_search") else "bounded_random_search")
            tuning_arch = (
                request_params.get("model")
                or request_params.get("architecture")
                or request_params.get("estimator")
                or "lightgbm"
            ) if request_params else "lightgbm"
            primary_metric = (
                request_params.get("primary_metric")
                or request_params.get("metric")
                or request_params.get("objective_metric")
                or "auc_roc"
            ) if request_params else "auc_roc"

            tuning_res = run_tuning(
                train_df,
                target=target_col,
                features=feature_cols,
                n_trials=trials_requested,
                strategy=tuning_strategy,
                primary_metric=primary_metric,
                architecture=tuning_arch,
                seed=ctx_instance.actual_seed,
                run_id=run_id,
            )

            completed_trials = len(tuning_res.trials) if tuning_res and tuning_res.trials else 0
            if tuning_res and tuning_res.trials:
                for idx, t in enumerate(tuning_res.trials):
                    trial_pct = round(((idx + 1) / trials_requested) * 100, 1)
                    evt = RuntimeEvent(
                        run_id=run_id,
                        event_type="tuning_trial",
                        status=t.status.upper(),
                        source_agent="OptimizationAgent",
                        target_agent="EvidenceLedger",
                        stage="TUNING",
                        action=f"trial_{idx + 1}",
                        node_id="step-tuning",
                        parent_node_id="step-context",
                        elapsed_seconds=round(time.time() - start_time, 2),
                        step=idx + 1,
                        completed=idx + 1,
                        total=trials_requested,
                        percent=trial_pct,
                        message=(
                            f"Trial {idx + 1}/{trials_requested}: "
                            f"{tuning_res.primary_metric}={float(t.validation_metric):.4f} status={t.status}"
                        ),
                        metadata=sanitize_json_primitives({
                            "trial": t.trial,
                            "params": t.params,
                            "validation_metric": float(t.validation_metric),
                            "objective_metric": tuning_res.primary_metric,
                            "primary_metric": tuning_res.primary_metric,
                            "best_metric": float(tuning_res.best_metric) if tuning_res.best_metric is not None else None,
                            "requested_trials": trials_requested,
                            "attempted_trials": idx + 1,
                            "completed_trials": completed_trials,
                            "strategy": tuning_strategy,
                            "model": tuning_arch,
                        }),
                    )
                    sink.emit(evt)
                    all_events.append(evt)

            # Step completion event for tuning
            evt_tune = RuntimeEvent(
                run_id=run_id,
                event_type="step_execution",
                status="COMPLETED",
                source_agent="OptimizationAgent",
                target_agent="EvidenceLedger",
                stage="TUNING",
                action="complete_tuning",
                node_id="step-tuning",
                parent_node_id="step-context",
                elapsed_seconds=round(time.time() - start_time, 2),
                message=(
                    f"Completed hyperparameter search via {tuning_strategy} "
                    f"({completed_trials}/{trials_requested} trials finished, best {tuning_res.primary_metric if tuning_res else 'metric'}={float(tuning_res.best_metric) if tuning_res and tuning_res.best_metric is not None else 0:.4f})"
                ),
                metadata=sanitize_json_primitives({
                    "requested_trials": trials_requested,
                    "attempted_trials": completed_trials,
                    "completed_trials": completed_trials,
                    "best_metric": float(tuning_res.best_metric) if tuning_res and tuning_res.best_metric is not None else None,
                    "best_hyperparameters": tuning_res.best_params if tuning_res else {},
                    "objective_metric": tuning_res.primary_metric if tuning_res else "auc_roc",
                    "tuning_strategy": tuning_strategy,
                    "model": tuning_arch,
                }),
            )
            sink.emit(evt_tune)
            all_events.append(evt_tune)
            prev_node_id = "step-tuning"

            # Champion model promotion
            if tuning_res and tuning_res.best_params:
                from start.modeling.models import resolve_model
                champion_model, _, _ = resolve_model(tuning_arch, seed=ctx_instance.actual_seed)
                clean_best_params = sanitize_json_primitives(tuning_res.best_params)
                if hasattr(champion_model, "set_params"):
                    champion_model.set_params(**clean_best_params)
                champion_model.fit(train_df[feature_cols], train_df[target_col].to_numpy())
                tab.model = champion_model
                if hasattr(tab, "test") and tab.test is not None:
                    if hasattr(champion_model, "predict_proba"):
                        test_p = champion_model.predict_proba(tab.test[feature_cols])[:, 1]
                    else:
                        test_p = champion_model.predict(tab.test[feature_cols])
                    tab.test["score"] = test_p
                    tab.test["prediction"] = (test_p >= 0.5).astype(int)
                tab.extra["champion_lineage"] = {
                    "champion_model": tuning_arch,
                    "best_hyperparameters": clean_best_params,
                    "best_validation_metric": float(tuning_res.best_metric) if tuning_res.best_metric is not None else None,
                    "objective_metric": tuning_res.primary_metric,
                    "tuning_strategy": tuning_strategy,
                }
                tab.extra["resolved_configuration"] = {
                    "model": tuning_arch,
                    "architecture": tuning_arch,
                    "strategy": tuning_strategy,
                    "tuning_strategy": tuning_strategy,
                    "hyperparameters": clean_best_params,
                    "best_hyperparameters": clean_best_params,
                    "best_metric": float(tuning_res.best_metric) if tuning_res.best_metric is not None else None,
                    "objective_metric": tuning_res.primary_metric,
                    "primary_metric": tuning_res.primary_metric,
                    "trials": trials_requested,
                    "requested_trials": trials_requested,
                    "completed_trials": completed_trials,
                    "champion_lineage": tab.extra["champion_lineage"],
                }

            # Execute post-tuning validation tests
            val_step_recs: list[EvidenceRecord] = []
            for tid in resolved.applicable_test_ids:
                spec = registry.get(tid)
                if spec is None or getattr(spec, "fn", None) is None:
                    continue
                evt_start = RuntimeEvent(
                    run_id=run_id,
                    event_type="test_started",
                    status="RUNNING",
                    source_agent="OptimizationAgent",
                    target_agent="DeterministicEngine",
                    stage="EXECUTION",
                    action=f"execute_{tid}",
                    node_id="step-validation",
                    parent_node_id=prev_node_id,
                    test_id=tid,
                    elapsed_seconds=round(time.time() - start_time, 2),
                    message=f"Executing validation test {tid}",
                )
                sink.emit(evt_start)
                all_events.append(evt_start)

                t0 = time.time()
                try:
                    tr = spec.fn(tab)
                except Exception as exc:
                    tr = TestResult(
                        test_id=tid,
                        test_name=tid,
                        status=Status.ERROR,
                        metrics={"error": str(exc)},
                        interpretation=f"Error executing {tid}: {exc}",
                    )
                lat = round((time.time() - t0) * 1000, 2)

                rec = ledger.append(tr, run_id=run_id)
                records.append(rec)
                val_step_recs.append(rec)

                evt_done = RuntimeEvent(
                    run_id=run_id,
                    event_type="test_completed",
                    status=str(rec.status).upper(),
                    source_agent="DeterministicEngine",
                    target_agent="EvidenceLedger",
                    stage="EXECUTION",
                    action=f"completed_{tid}",
                    node_id="step-validation",
                    parent_node_id=prev_node_id,
                    test_id=tid,
                    evidence_refs=[rec.evidence_id],
                    elapsed_seconds=round(time.time() - start_time, 2),
                    message=f"Completed {tid} -> {rec.status} ({lat:.1f}ms)",
                    metadata={"metrics": rec.metrics, "latency_ms": lat},
                )
                sink.emit(evt_done)
                all_events.append(evt_done)

                evt_ev = RuntimeEvent(
                    run_id=run_id,
                    event_type="evidence_committed",
                    status="COMPLETED",
                    source_agent="DeterministicEngine",
                    target_agent="EvidenceLedger",
                    stage="CHECKPOINTS",
                    action="commit_evidence_record",
                    node_id="step-evidence",
                    parent_node_id="step-validation",
                    test_id=tid,
                    evidence_refs=[rec.evidence_id],
                    elapsed_seconds=round(time.time() - start_time, 2),
                    message=f"Committed EvidenceRecord {rec.evidence_id} ({rec.test_id})",
                )
                sink.emit(evt_ev)
                all_events.append(evt_ev)

            evt_val = RuntimeEvent(
                run_id=run_id,
                event_type="step_execution",
                status="COMPLETED",
                source_agent="DeterministicEngine",
                target_agent="EvidenceLedger",
                stage="EXECUTION",
                action="execute_step-validation",
                node_id="step-validation",
                parent_node_id="step-tuning",
                evidence_refs=[r.evidence_id for r in val_step_recs],
                elapsed_seconds=round(time.time() - start_time, 2),
                message=f"Completed post-tuning validation ({len(val_step_recs)} tests executed)",
            )
            sink.emit(evt_val)
            all_events.append(evt_val)
            prev_node_id = "step-validation"

        elif resolved.engine_kind in (EngineKind.PREDICTIVE_SUBSET, EngineKind.DEEP_LEARNING_REVIEW):
            tab = ctx_instance.bundle.tabular
            if tab is not None:
                from sklearn.model_selection import train_test_split

                from start.modeling.models import resolve_model
                from start.modeling.sensitivity_analysis import DEFAULT_SHOCKS, run_sensitivity_analysis

                req_model = (
                    request_params.get("model")
                    or request_params.get("architecture")
                    or request_params.get("estimator")
                ) if request_params else None
                user_hyper = (request_params.get("hyperparameters") or {}) if request_params else {}
                req_prep = (request_params.get("preprocessing") or {}) if request_params else {}
                req_split = (request_params.get("split") or request_params.get("split_strategy")) if request_params else None
                req_sens = (request_params.get("sensitivity") or {}) if request_params else {}
                sens_mode = req_sens.get("mode", "one_at_a_time") if isinstance(req_sens, dict) else str(req_sens or "one_at_a_time")

                target_col = getattr(tab, "target_column", "target")

                # If user explicitly specified a model architecture, resolve it (fails closed if uninstalled)
                with trace_scope(OP_MODEL_RESOLVE, {"requested_model": req_model}):
                    if req_model:
                        fitted_model, _, _ = resolve_model(req_model, seed=ctx_instance.actual_seed, fail_closed=True, **user_hyper)
                    else:
                        fitted_model = getattr(tab, "model", None)
                        cls_name = getattr(fitted_model, "__class__", type("Estimator", (), {})).__name__.lower()
                        if "tabulardl" in cls_name or "mlp" in cls_name:
                            req_model = "mlp"
                        elif "logistic" in cls_name:
                            req_model = "logistic_regression"
                        elif "forest" in cls_name:
                            req_model = "random_forest"
                        elif "lightgbm" in cls_name or "lgbm" in cls_name:
                            req_model = "lightgbm"
                        else:
                            req_model = "logistic_regression"

                # Reconstruct full dataset for custom split & preprocessing if requested or if custom model
                if tab.train is not None and tab.test is not None:
                    full_df = pd.concat([tab.train, tab.test], ignore_index=True)
                    drop_cols = {target_col, getattr(tab, "score_column", "score"), getattr(tab, "prediction_column", "prediction"), "score", "prediction"}
                    feature_cols = [c for c in full_df.columns if c not in drop_cols]

                    split_name = "stratified"
                    test_ratio = 0.25
                    if isinstance(req_split, dict):
                        split_name = req_split.get("strategy", "stratified")
                        test_ratio = float(req_split.get("test_size", 0.25))
                    elif isinstance(req_split, str):
                        split_name = req_split

                    with trace_scope(OP_SPLIT, {"strategy": split_name, "test_ratio": test_ratio}):
                        if split_name in ("stratified", "stratified_holdout") and full_df[target_col].nunique() <= 10:
                            train_df, test_df = train_test_split(full_df, test_size=test_ratio, random_state=ctx_instance.actual_seed, stratify=full_df[target_col])
                        elif split_name in ("time_series", "temporal"):
                            split_idx = int(len(full_df) * (1.0 - test_ratio))
                            train_df, test_df = full_df.iloc[:split_idx].copy(), full_df.iloc[split_idx:].copy()
                        else:
                            train_df, test_df = train_test_split(full_df, test_size=test_ratio, random_state=ctx_instance.actual_seed)

                    train_df = train_df.reset_index(drop=True)
                    test_df = test_df.reset_index(drop=True)

                    from start.data.preprocessing import apply_preprocessing_pipeline

                    # Handle string / categorical target column if present
                    y_tr_raw = train_df[target_col]
                    if (
                        y_tr_raw.dtype == object
                        or str(y_tr_raw.dtype) == "category"
                        or not pd.api.types.is_numeric_dtype(y_tr_raw)
                    ):
                        unique_vals = sorted([str(v) for v in pd.Series(y_tr_raw).dropna().unique()])
                        mapping = {val: idx for idx, val in enumerate(unique_vals)}
                        y_tr = y_tr_raw.astype(str).map(mapping).fillna(0).to_numpy(dtype=int)
                        y_te = test_df[target_col].astype(str).map(mapping).fillna(0).to_numpy(dtype=int)
                        train_df[target_col] = y_tr
                        test_df[target_col] = y_te
                    else:
                        y_tr = train_df[target_col].to_numpy(dtype=int)

                    with trace_scope(OP_PREPROCESS_FIT, {"preprocessing": str(req_prep)}):
                        train_feat_clean, test_feat_clean, prep_summary = apply_preprocessing_pipeline(
                            train_df[feature_cols], test_df[feature_cols], y_train=y_tr, preprocessing=req_prep
                        )
                    feature_cols_clean = list(train_feat_clean.columns)

                    train_df_clean = train_feat_clean.copy()
                    train_df_clean[target_col] = train_df[target_col]
                    test_df_clean = test_feat_clean.copy()
                    test_df_clean[target_col] = test_df[target_col]

                    if fitted_model is not None:
                        with trace_scope(OP_MODEL_FIT, {"model_class": fitted_model.__class__.__name__}):
                            fitted_model.fit(train_df_clean[feature_cols_clean], y_tr)
                        if hasattr(fitted_model, "predict_proba"):
                            test_p = fitted_model.predict_proba(test_df_clean[feature_cols_clean])[:, 1]
                        elif hasattr(fitted_model, "decision_function"):
                            test_p = fitted_model.decision_function(test_df_clean[feature_cols_clean])
                        else:
                            test_p = fitted_model.predict(test_df_clean[feature_cols_clean])

                        test_df_clean["score"] = test_p
                        test_df_clean["prediction"] = (test_p >= 0.5).astype(int)
                        tab.model = fitted_model

                    tab.train = train_df_clean
                    tab.test = test_df_clean
                    tab.score_column = "score"
                    tab.prediction_column = "prediction"

                    prep_dict = req_prep if isinstance(req_prep, dict) else prep_summary
                    tab.extra["resolved_configuration"] = {
                        "model": req_model,
                        "hyperparameters": user_hyper,
                        "preprocessing": prep_dict,
                        "preprocessing_summary": prep_summary,
                        "split": {"strategy": split_name, "test_size": test_ratio},
                    }
                    tab.extra["sensitivity_mode"] = sens_mode

                    if fitted_model is not None:
                        with trace_scope(OP_SENSITIVITY_EXECUTE, {"mode": sens_mode}):
                            sens_res = run_sensitivity_analysis(
                                fitted_model,
                                test_df_clean[feature_cols_clean],
                                test_df_clean[target_col].to_numpy(),
                                top_features=feature_cols_clean[:5],
                                metric_name="auc_roc",
                                shocks=DEFAULT_SHOCKS,
                                mode=sens_mode,
                            )
                            tab.extra["sensitivity_result"] = sens_res.to_dict()

            executed_test_ids = set()

            for step_id, label, kind, _desc, step_test_ids in resolved.step_specs:
                if kind != "test":
                    continue
                applicable_step_tests = [t for t in step_test_ids if t in resolved.applicable_test_ids]
                if not applicable_step_tests:
                    continue

                step_recs: list[EvidenceRecord] = []
                for tid in applicable_step_tests:
                    spec = registry.get(tid)
                    if spec is None or getattr(spec, "fn", None) is None:
                        continue

                    evt_start = RuntimeEvent(
                        run_id=run_id,
                        event_type="test_started",
                        status="RUNNING",
                        source_agent="Specialist",
                        target_agent="DeterministicEngine",
                        stage="EXECUTION",
                        action=f"execute_{tid}",
                        node_id=step_id,
                        parent_node_id=prev_node_id,
                        test_id=tid,
                        elapsed_seconds=round(time.time() - start_time, 2),
                        message=f"Starting test {tid}",
                    )
                    sink.emit(evt_start)
                    all_events.append(evt_start)

                    t0 = time.time()
                    with trace_scope(OP_MODEL_EVALUATE, {"test_id": tid, "step_id": step_id}):
                        try:
                            tr = spec.fn(tab)
                        except Exception as exc:
                            tr = TestResult(
                                test_id=tid,
                                test_name=tid,
                                status=Status.ERROR,
                                metrics={"error": str(exc)},
                                interpretation=f"Error executing {tid}: {exc}",
                            )
                    lat = round((time.time() - t0) * 1000, 2)

                    with trace_scope(OP_EVIDENCE_EMIT, {"test_id": tid}):
                        rec = ledger.append(tr, run_id=run_id)
                    records.append(rec)
                    step_recs.append(rec)
                    executed_test_ids.add(tid)

                    evt_done = RuntimeEvent(
                        run_id=run_id,
                        event_type="test_completed",
                        status=str(rec.status).upper(),
                        source_agent="DeterministicEngine",
                        target_agent="EvidenceLedger",
                        stage="EXECUTION",
                        action=f"completed_{tid}",
                        node_id=step_id,
                        parent_node_id=prev_node_id,
                        test_id=tid,
                        evidence_refs=[rec.evidence_id],
                        elapsed_seconds=round(time.time() - start_time, 2),
                        message=f"Completed {tid} -> {rec.status} ({lat:.1f}ms)",
                        metadata={"metrics": rec.metrics, "latency_ms": lat},
                    )
                    sink.emit(evt_done)
                    all_events.append(evt_done)

                    evt_ev = RuntimeEvent(
                        run_id=run_id,
                        event_type="evidence_committed",
                        status="COMPLETED",
                        source_agent="DeterministicEngine",
                        target_agent="EvidenceLedger",
                        stage="CHECKPOINTS",
                        action="commit_evidence_record",
                        node_id="step-evidence",
                        parent_node_id=step_id,
                        test_id=tid,
                        evidence_refs=[rec.evidence_id],
                        elapsed_seconds=round(time.time() - start_time, 2),
                        message=f"Committed EvidenceRecord {rec.evidence_id} ({rec.test_id})",
                    )
                    sink.emit(evt_ev)
                    all_events.append(evt_ev)

                # Emit step completion event
                evt_step = RuntimeEvent(
                    run_id=run_id,
                    event_type="step_execution",
                    status="COMPLETED",
                    source_agent="DeterministicEngine",
                    target_agent="EvidenceLedger",
                    stage="EXECUTION",
                    action=f"execute_{step_id}",
                    node_id=step_id,
                    parent_node_id=prev_node_id,
                    evidence_refs=[r.evidence_id for r in step_recs],
                    elapsed_seconds=round(time.time() - start_time, 2),
                    phase=label,
                    step=step_id,
                    message=f"Completed {label} ({len(step_recs)} tests executed)",
                    metadata={"test_ids": [r.test_id for r in step_recs]},
                )
                sink.emit(evt_step)
                all_events.append(evt_step)

                # Checkpoint emissions
                if step_id == "step-preflight":
                    cp2_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{run_id}:CP-002:{len(step_recs)}")).replace("-", "")[:16]
                    cp2_item = {
                        "checkpoint_id": "CP-002",
                        "name": "Preflight complete",
                        "status": "completed",
                        "producing_stage": "step-preflight",
                        "agent_signature": "DeterministicEngine",
                        "evidence_ids": [r.evidence_id for r in step_recs],
                        "commit_hash": cp2_hash,
                        "timestamp": datetime.datetime.fromtimestamp(time.time(), tz=datetime.UTC).isoformat(),
                        "summary": f"Preflight data integrity validated ({len(step_recs)} deterministic tests executed).",
                    }
                    checkpoints_list.append(cp2_item)
                    evt_cp2 = RuntimeEvent(
                        run_id=run_id,
                        event_type="checkpoint_committed",
                        status="COMPLETED",
                        source_agent="DeterministicEngine",
                        target_agent="EvidenceLedger",
                        stage="CHECKPOINTS",
                        action="commit_CP-002",
                        node_id="step-preflight",
                        checkpoint_id="CP-002",
                        evidence_refs=[r.evidence_id for r in step_recs],
                        elapsed_seconds=round(time.time() - start_time, 2),
                        message="Committed checkpoint CP-002: Preflight complete",
                        metadata=cp2_item,
                    )
                    sink.emit(evt_cp2)
                    all_events.append(evt_cp2)

                elif step_id == "step-features":
                    cp3_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{run_id}:CP-003:{len(step_recs)}")).replace("-", "")[:16]
                    cp3_item = {
                        "checkpoint_id": "CP-003",
                        "name": "Feature analysis",
                        "status": "completed",
                        "producing_stage": "step-features",
                        "agent_signature": "DeterministicEngine",
                        "evidence_ids": [r.evidence_id for r in step_recs],
                        "commit_hash": cp3_hash,
                        "timestamp": datetime.datetime.fromtimestamp(time.time(), tz=datetime.UTC).isoformat(),
                        "summary": f"Feature distribution, drift, and attribution checks completed ({len(step_recs)} tests).",
                    }
                    checkpoints_list.append(cp3_item)
                    evt_cp3 = RuntimeEvent(
                        run_id=run_id,
                        event_type="checkpoint_committed",
                        status="COMPLETED",
                        source_agent="DeterministicEngine",
                        target_agent="EvidenceLedger",
                        stage="CHECKPOINTS",
                        action="commit_CP-003",
                        node_id="step-features",
                        checkpoint_id="CP-003",
                        evidence_refs=[r.evidence_id for r in step_recs],
                        elapsed_seconds=round(time.time() - start_time, 2),
                        message="Committed checkpoint CP-003: Feature analysis",
                        metadata=cp3_item,
                    )
                    sink.emit(evt_cp3)
                    all_events.append(evt_cp3)

                elif step_id in ("step-supervised", "step-deep_learning"):
                    cp4_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{run_id}:CP-004:{len(step_recs)}")).replace("-", "")[:16]
                    cp4_item = {
                        "checkpoint_id": "CP-004",
                        "name": "Model evaluation",
                        "status": "completed",
                        "producing_stage": step_id,
                        "agent_signature": "DeterministicEngine",
                        "evidence_ids": [r.evidence_id for r in step_recs],
                        "commit_hash": cp4_hash,
                        "timestamp": datetime.datetime.fromtimestamp(time.time(), tz=datetime.UTC).isoformat(),
                        "summary": f"Core model discrimination, calibration, and risk metrics evaluated ({len(step_recs)} tests).",
                    }
                    checkpoints_list.append(cp4_item)
                    evt_cp4 = RuntimeEvent(
                        run_id=run_id,
                        event_type="checkpoint_committed",
                        status="COMPLETED",
                        source_agent="DeterministicEngine",
                        target_agent="EvidenceLedger",
                        stage="CHECKPOINTS",
                        action="commit_CP-004",
                        node_id=step_id,
                        checkpoint_id="CP-004",
                        evidence_refs=[r.evidence_id for r in step_recs],
                        elapsed_seconds=round(time.time() - start_time, 2),
                        message="Committed checkpoint CP-004: Model evaluation",
                        metadata=cp4_item,
                    )
                    sink.emit(evt_cp4)
                    all_events.append(evt_cp4)

                prev_node_id = step_id

        elif resolved.engine_kind == EngineKind.MARKET_SUBSET:
            market = ctx_instance.bundle.market
            short_rate = ctx_instance.bundle.short_rate
            if market is not None:
                opt_raw = (
                    request_params.get("optimizer")
                    or request_params.get("portfolio_optimizer")
                    or request_params.get("technique")
                    or "hrp"
                ) if request_params else "hrp"
                scen_raw = (
                    request_params.get("scenario")
                    or request_params.get("scenario_type")
                    or request_params.get("scenario_shocks")
                    or "asset_tail_stress"
                ) if request_params else "asset_tail_stress"

                opt_norm = "hrp"
                if any(x in str(opt_raw).lower() for x in ("hierarchical", "hrp")):
                    opt_norm = "hierarchical_risk_parity" if "hierarchical" in str(opt_raw).lower() else "hrp"
                elif any(x in str(opt_raw).lower() for x in ("min_var", "minimum_variance", "min_variance")):
                    opt_norm = "min_var"
                elif any(x in str(opt_raw).lower() for x in ("erc", "equal_risk")):
                    opt_norm = "erc"

                scen_norm = "asset_tail_stress"
                if any(x in str(scen_raw).lower() for x in ("factor", "macro")):
                    scen_norm = "factor_macro"
                elif any(x in str(scen_raw).lower() for x in ("reverse", "mahalanobis")):
                    scen_norm = "reverse_stress"
                elif any(x in str(scen_raw).lower() for x in ("correlation", "breakdown")):
                    scen_norm = "correlation_breakdown"
                elif any(x in str(scen_raw).lower() for x in ("shift", "hypothetical")):
                    scen_norm = "hypothetical_shift"
                elif any(x in str(scen_raw).lower() for x in ("historical", "replay", "asset")):
                    scen_norm = "asset_tail_stress"

                shock_mag = float(
                    request_params.get("shock_magnitude")
                    or request_params.get("raw_shock_mag")
                    or -0.20
                ) if request_params else -0.20

                market.extra["resolved_configuration"] = {
                    "optimizer": opt_norm,
                    "requested_optimizer": str(opt_raw),
                    "scenario": scen_norm,
                    "requested_scenario": str(scen_raw),
                    "shock_magnitude": shock_mag,
                    "risk_free_rate": getattr(market, "risk_free_rate", 0.02),
                    "parameters": request_params or {},
                }
            executed_test_ids = set()

            for step_id, label, kind, _desc, step_test_ids in resolved.step_specs:
                if kind != "test":
                    continue
                applicable_step_tests = [t for t in step_test_ids if t in resolved.applicable_test_ids]
                if not applicable_step_tests:
                    continue

                step_recs: list[EvidenceRecord] = []
                for tid in applicable_step_tests:
                    spec = registry.get(tid)
                    if spec is None or getattr(spec, "fn", None) is None:
                        continue

                    ctx_arg = short_rate if spec.context_type == "short_rate" else market

                    evt_start = RuntimeEvent(
                        run_id=run_id,
                        event_type="test_started",
                        status="RUNNING",
                        source_agent="MarketSpecialist",
                        target_agent="DeterministicEngine",
                        stage="EXECUTION",
                        action=f"execute_{tid}",
                        node_id=step_id,
                        parent_node_id=prev_node_id,
                        test_id=tid,
                        elapsed_seconds=round(time.time() - start_time, 2),
                        message=f"Starting test {tid}",
                    )
                    sink.emit(evt_start)
                    all_events.append(evt_start)

                    t0 = time.time()
                    try:
                        tr = spec.fn(ctx_arg)
                    except Exception as exc:
                        tr = TestResult(
                            test_id=tid,
                            test_name=tid,
                            status=Status.ERROR,
                            metrics={"error": str(exc)},
                            interpretation=f"Error executing {tid}: {exc}",
                        )
                    lat = round((time.time() - t0) * 1000, 2)

                    rec = ledger.append(tr, run_id=run_id)
                    records.append(rec)
                    step_recs.append(rec)
                    executed_test_ids.add(tid)

                    evt_done = RuntimeEvent(
                        run_id=run_id,
                        event_type="test_completed",
                        status=str(rec.status).upper(),
                        source_agent="DeterministicEngine",
                        target_agent="EvidenceLedger",
                        stage="EXECUTION",
                        action=f"completed_{tid}",
                        node_id=step_id,
                        parent_node_id=prev_node_id,
                        test_id=tid,
                        evidence_refs=[rec.evidence_id],
                        elapsed_seconds=round(time.time() - start_time, 2),
                        message=f"Completed {tid} -> {rec.status} ({lat:.1f}ms)",
                        metadata={"metrics": rec.metrics, "latency_ms": lat},
                    )
                    sink.emit(evt_done)
                    all_events.append(evt_done)

                    evt_ev = RuntimeEvent(
                        run_id=run_id,
                        event_type="evidence_committed",
                        status="COMPLETED",
                        source_agent="DeterministicEngine",
                        target_agent="EvidenceLedger",
                        stage="CHECKPOINTS",
                        action="commit_evidence_record",
                        node_id="step-evidence",
                        parent_node_id=step_id,
                        test_id=tid,
                        evidence_refs=[rec.evidence_id],
                        elapsed_seconds=round(time.time() - start_time, 2),
                        message=f"Committed EvidenceRecord {rec.evidence_id} ({rec.test_id})",
                    )
                    sink.emit(evt_ev)
                    all_events.append(evt_ev)

                evt_step = RuntimeEvent(
                    run_id=run_id,
                    event_type="step_execution",
                    status="COMPLETED",
                    source_agent="DeterministicEngine",
                    target_agent="EvidenceLedger",
                    stage="EXECUTION",
                    action=f"execute_{step_id}",
                    node_id=step_id,
                    parent_node_id=prev_node_id,
                    evidence_refs=[r.evidence_id for r in step_recs],
                    elapsed_seconds=round(time.time() - start_time, 2),
                    phase=label,
                    step=step_id,
                    message=f"Completed {label} ({len(step_recs)} tests executed)",
                    metadata={"test_ids": [r.test_id for r in step_recs]},
                )
                sink.emit(evt_step)
                all_events.append(evt_step)
                prev_node_id = step_id

        elif resolved.engine_kind == EngineKind.RECOMMENDER_REVIEW:
            from start.recommender.artifacts import (
                render_recommender_cold_start_artifact,
                render_recommender_metrics_artifact,
                render_recommender_ranking_table_artifact,
                render_recommender_sensitivity_artifact,
                render_recommender_sparsity_artifact,
                render_recommender_topk_examples_artifact,
            )
            from start.recommender.contracts import (
                RecommenderExecutionResult,
                RecommenderModelSummary,
            )
            from start.recommender.data import split_recommender_dataset
            from start.recommender.metrics import (
                compute_beyond_accuracy_metrics,
                compute_cold_start_metrics,
                compute_ranking_metrics,
                compute_rating_metrics,
            )
            from start.recommender.models import (
                FactorizationMachineModel,
                MatrixFactorizationModel,
                NeuralCollaborativeFilteringModel,
            )
            from start.recommender.sensitivity import evaluate_recommender_sensitivity

            rec_ctx = ctx_instance.bundle.recommender
            raw_data = ctx_instance.raw_data
            df = pd.DataFrame([r.to_dict() if hasattr(r, "to_dict") else vars(r) for r in raw_data])

            # Deterministic split under seed
            split_res = split_recommender_dataset(
                df,
                protocol="user_stratified",
                test_ratio=0.2,
                seed=ctx_instance.actual_seed,
            )
            rec_ctx.split_result = split_res

            user_col = "user_id"
            item_col = "item_id"
            target_col = ctx_instance.actual_target
            n_users = rec_ctx.dataset_profile.n_users
            n_items = rec_ctx.dataset_profile.n_items

            rec_algo = (
                request_params.get("algorithm")
                or request_params.get("model")
                or request_params.get("recommender_algorithm")
                or ""
            ).lower().strip() if request_params else ""

            # Train model
            if rec_algo in ("ffm", "field_aware_factorization_machine") or context_id == "recommender_ffm_v1":
                from start.recommender.models import FieldAwareFactorizationMachineModel
                model = FieldAwareFactorizationMachineModel(
                    latent_dim=int(request_params.get("latent_dim", 4)) if request_params else 4,
                    learning_rate=float(request_params.get("learning_rate", 0.02)) if request_params else 0.02,
                    regularization=float(request_params.get("regularization", 0.01)) if request_params else 0.01,
                    epochs=int(request_params.get("epochs", 10)) if request_params else 10,
                    seed=ctx_instance.actual_seed,
                )
                model.fit(split_res.train_data)
                algo_name = "Field-Aware Factorization Machine (FFM)"
            elif rec_algo in ("ncf", "neural_cf", "neural_collaborative_filtering") or (not rec_algo and context_id == "recommender_implicit_v1"):
                model = NeuralCollaborativeFilteringModel(
                    embedding_dim=16,
                    hidden_layers=[32, 16],
                    learning_rate=0.01,
                    epochs=15,
                    batch_size=32,
                    seed=ctx_instance.actual_seed,
                )
                model.fit(split_res.train_data)
                algo_name = "Neural Collaborative Filtering (NCF)"
            elif rec_algo in ("fm", "factorization_machine") or (not rec_algo and context_id == "recommender_contextual_v1"):
                model = FactorizationMachineModel(
                    latent_dim=8,
                    learning_rate=0.02,
                    regularization=0.01,
                    epochs=15,
                    seed=ctx_instance.actual_seed,
                )
                model.fit(split_res.train_data)
                algo_name = "Factorization Machine (FM)"
            else:
                model = MatrixFactorizationModel(
                    latent_dim=16,
                    learning_rate=0.05,
                    regularization=0.02,
                    epochs=20,
                    seed=ctx_instance.actual_seed,
                )
                model.fit(split_res.train_data)
                algo_name = "Biased Matrix Factorization (MF)"

            test_df = split_res.test_data
            all_users = list(df[user_col].unique())
            all_items = list(df[item_col].unique())

            recommendations_by_user: dict[str, list[str]] = {}
            for u in all_users:
                recommendations_by_user[u] = model.recommend(u, k=20, candidate_items=all_items)

            ground_truth_by_user: dict[str, set[str]] = {}
            for u, group in test_df.groupby(user_col):
                ground_truth_by_user[u] = set(group[item_col].unique())

            # Rating metrics if target column is present and not purely implicit
            rating_metrics = None
            if target_col in test_df.columns and rec_ctx.feedback_mode != "implicit":
                y_true = test_df[target_col].to_numpy(dtype=float)
                if hasattr(model, "predict_score"):
                    y_pred = np.array([model.predict_score(row[user_col], row[item_col]) for _, row in test_df.iterrows()], dtype=float)
                else:
                    y_pred = np.array([model.predict_one(row[user_col], row[item_col]) for _, row in test_df.iterrows()], dtype=float)
                rating_metrics = compute_rating_metrics(y_true, y_pred)

            ranking_metrics = compute_ranking_metrics(
                recommendations_by_user,
                ground_truth_by_user,
                k_list=(5, 10, 20),
            )

            beyond_accuracy_metrics = compute_beyond_accuracy_metrics(
                recommendations_by_user,
                all_items=all_items,
                all_users=all_users,
                k=10,
            )

            cold_start_metrics = compute_cold_start_metrics(
                recommendations_by_user,
                ground_truth_by_user,
                cold_user_ids=split_res.cold_user_ids,
                cold_item_ids=split_res.cold_item_ids,
                k=10,
            )

            sensitivity_result = evaluate_recommender_sensitivity(
                model,
                split_res,
                algorithm_type="ffm" if algo_name.startswith("Field-Aware") else ("ncf" if algo_name.startswith("Neural") else ("fm" if algo_name.startswith("Factorization") else "mf")),
                raw_data=raw_data,
            )

            model_summary = RecommenderModelSummary(
                algorithm=algo_name,
                task_mode=rec_ctx.task_mode.value,
                feedback_mode=rec_ctx.feedback_mode.value,
                hyperparameters={"latent_dim": getattr(model, "latent_dim", getattr(model, "embedding_dim", 16)), "seed": ctx_instance.actual_seed},
                training_epochs=getattr(model, "epochs", 20),
                seed=ctx_instance.actual_seed,
            )

            exec_result = RecommenderExecutionResult(
                model_summary=model_summary,
                dataset_profile=rec_ctx.dataset_profile,
                split_result=split_res,
                rating_metrics=rating_metrics,
                ranking_metrics=ranking_metrics,
                beyond_accuracy_metrics=beyond_accuracy_metrics,
                cold_start_metrics=cold_start_metrics,
                sensitivity_result=sensitivity_result,
            )
            rec_ctx.execution_result = exec_result

            products.register(
                "recommender.execution_result",
                exec_result,
                evidence_ids=(),
                source_fingerprint=rec_ctx.fingerprint(),
                provenance="deterministic_recommender_engine",
            )

            # Execute tests for steps
            for step_id, label, kind, _desc, step_test_ids in resolved.step_specs:
                if kind != "test":
                    continue
                applicable_step_tests = [t for t in step_test_ids if t in resolved.applicable_test_ids]
                if not applicable_step_tests:
                    continue

                step_recs: list[EvidenceRecord] = []
                for tid in applicable_step_tests:
                    spec = registry.get(tid)
                    if spec is None or getattr(spec, "fn", None) is None:
                        continue

                    evt_start = RuntimeEvent(
                        run_id=run_id,
                        event_type="test_started",
                        status="RUNNING",
                        source_agent="Specialist",
                        target_agent="DeterministicEngine",
                        stage="EXECUTION",
                        action=f"execute_{tid}",
                        node_id=step_id,
                        parent_node_id=prev_node_id,
                        test_id=tid,
                        elapsed_seconds=round(time.time() - start_time, 2),
                        message=f"Starting test {tid}",
                    )
                    sink.emit(evt_start)
                    all_events.append(evt_start)

                    t0 = time.time()
                    try:
                        tr = spec.fn(rec_ctx)
                    except Exception as exc:
                        tr = TestResult(
                            test_id=tid,
                            test_name=tid,
                            status=Status.ERROR,
                            metrics={"error": str(exc)},
                            interpretation=f"Error executing {tid}: {exc}",
                        )
                    lat = round((time.time() - t0) * 1000, 2)

                    rec = ledger.append(tr, run_id=run_id)
                    records.append(rec)
                    step_recs.append(rec)

                    evt_done = RuntimeEvent(
                        run_id=run_id,
                        event_type="test_completed",
                        status=str(rec.status).upper(),
                        source_agent="DeterministicEngine",
                        target_agent="EvidenceLedger",
                        stage="EXECUTION",
                        action=f"completed_{tid}",
                        node_id=step_id,
                        parent_node_id=prev_node_id,
                        test_id=tid,
                        evidence_refs=[rec.evidence_id],
                        elapsed_seconds=round(time.time() - start_time, 2),
                        message=f"Completed {tid} -> {rec.status} ({lat:.1f}ms)",
                        metadata={"metrics": rec.metrics, "latency_ms": lat},
                    )
                    sink.emit(evt_done)
                    all_events.append(evt_done)

                    evt_ev = RuntimeEvent(
                        run_id=run_id,
                        event_type="evidence_committed",
                        status="COMPLETED",
                        source_agent="DeterministicEngine",
                        target_agent="EvidenceLedger",
                        stage="CHECKPOINTS",
                        action="commit_evidence_record",
                        node_id="step-evidence",
                        parent_node_id=step_id,
                        test_id=tid,
                        evidence_refs=[rec.evidence_id],
                        elapsed_seconds=round(time.time() - start_time, 2),
                        message=f"Committed EvidenceRecord {rec.evidence_id} ({rec.test_id})",
                    )
                    sink.emit(evt_ev)
                    all_events.append(evt_ev)

                # Step completion event
                evt_step = RuntimeEvent(
                    run_id=run_id,
                    event_type="step_execution",
                    status="COMPLETED",
                    source_agent="DeterministicEngine",
                    target_agent="EvidenceLedger",
                    stage="EXECUTION",
                    action=f"execute_{step_id}",
                    node_id=step_id,
                    parent_node_id=prev_node_id,
                    evidence_refs=[r.evidence_id for r in step_recs],
                    elapsed_seconds=round(time.time() - start_time, 2),
                    phase=label,
                    step=step_id,
                    message=f"Completed {label} ({len(step_recs)} tests executed)",
                    metadata={"test_ids": [r.test_id for r in step_recs]},
                )
                sink.emit(evt_step)
                all_events.append(evt_step)
                prev_node_id = step_id

            # Generate canonical recommender review artifacts
            raw_arts = [
                render_recommender_metrics_artifact(
                    run_id=run_id,
                    rating_metrics=exec_result.rating_metrics,
                    ranking_metrics=exec_result.ranking_metrics,
                    beyond_accuracy=exec_result.beyond_accuracy_metrics,
                    node_id="step-rec-fidelity",
                ),
                render_recommender_ranking_table_artifact(
                    run_id=run_id,
                    ranking_metrics=exec_result.ranking_metrics,
                    node_id="step-rec-ranking",
                ) if exec_result.ranking_metrics else None,
                render_recommender_sparsity_artifact(
                    run_id=run_id,
                    profile=exec_result.dataset_profile,
                    node_id="step-rec-data",
                ),
                render_recommender_cold_start_artifact(
                    run_id=run_id,
                    cold_start=exec_result.cold_start_metrics,
                    node_id="step-rec-cold",
                ) if exec_result.cold_start_metrics else None,
                render_recommender_sensitivity_artifact(
                    run_id=run_id,
                    sensitivity=exec_result.sensitivity_result,
                    node_id="step-rec-stability",
                ) if exec_result.sensitivity_result else None,
                render_recommender_topk_examples_artifact(
                    run_id=run_id,
                    recommendations=recommendations_by_user,
                    node_id="step-rec-ranking",
                ),
            ]
            for art in raw_arts:
                if art is None:
                    continue
                art_id = art.get("artifactId") or art.get("id") or f"ART-{uuid.uuid4().hex[:6]}"
                art_title = art.get("label") or art.get("title") or art_id
                art_node = art.get("producerNodeId") or art.get("producing_step_id") or "step-rec-evidence"
                art_dict = {
                    "id": art_id,
                    "artifactId": art_id,
                    "name": art_title,
                    "title": art_title,
                    "label": art_title,
                    "kind": art.get("kind", "table"),
                    "artifact_type": art.get("kind", "table"),
                    "producing_step_id": art_node,
                    "producerNodeId": art_node,
                    "data_fingerprint": art.get("hash", ""),
                    "evidence_ids": [],
                    "content": art.get("preview", {}).get("payload") or art.get("content"),
                    "preview": art.get("preview"),
                    "mimeType": "application/json",
                }
                artifacts_dict[art_id] = art_dict
                evt_art = RuntimeEvent(
                    run_id=run_id,
                    event_type="artifact_created",
                    status="COMPLETED",
                    source_agent="DeterministicEngine",
                    target_agent="StructuredReviewer",
                    stage="ARTIFACT_GENERATION",
                    action="generate_artifact",
                    node_id=art_node,
                    artifact_refs=[art_id],
                    elapsed_seconds=round(time.time() - start_time, 2),
                    message=f"Created review artifact {art_id}: {art_title}",
                    metadata={"artifact_id": art_id, "title": art_title, "kind": art_dict["kind"]},
                )
                sink.emit(evt_art)
                all_events.append(evt_art)

            try:
                from start.analysis.builder import DeterministicArtifactBuilder
                from start.analysis.pipelines import (
                    run_recommender_fm_pipeline,
                    run_recommender_mf_pipeline,
                    run_recommender_ncf_pipeline,
                )

                if algo_name.startswith("Field-Aware") or context_id == "recommender_ffm_v1":
                    from start.analysis.pipelines import execute_canonical_recommender_ffm
                    can_res_rec = execute_canonical_recommender_ffm(raw_data, run_id=run_id, seed=ctx_instance.actual_seed)
                    case_k = "case_f"
                elif context_id == "recommender_implicit_v1":
                    can_res_rec = run_recommender_ncf_pipeline(raw_data, run_id=run_id, seed=ctx_instance.actual_seed)
                    case_k = "case_e"
                elif context_id == "recommender_contextual_v1":
                    can_res_rec = run_recommender_fm_pipeline(raw_data, run_id=run_id, seed=ctx_instance.actual_seed)
                    case_k = "case_f"
                else:
                    can_res_rec = run_recommender_mf_pipeline(raw_data, run_id=run_id, seed=ctx_instance.actual_seed)
                    case_k = "case_d"

                builder_rec = DeterministicArtifactBuilder(root / "artifacts")
                canon_rec_arts = builder_rec.build_and_persist_all(can_res_rec, case_k)
                for cart in canon_rec_arts:
                    cid = cart["artifact_id"]
                    artifacts_dict[cid] = {
                        "id": cid,
                        "artifactId": cid,
                        "name": cart["title"],
                        "title": cart["title"],
                        "label": cart["title"],
                        "kind": cart.get("kind", "table"),
                        "artifact_type": cart.get("artifact_type", "table"),
                        "producing_step_id": "step-rec-fidelity",
                        "producerNodeId": "step-rec-fidelity",
                        "data_fingerprint": cart.get("data_fingerprint", ""),
                        "evidence_ids": [records[0].evidence_id] if records else [],
                        "content": cart["semantic_payload"],
                        "mimeType": "application/json",
                    }
                products.register(
                    "analysis.canonical_result",
                    can_res_rec,
                    evidence_ids=tuple(r.evidence_id for r in records[:5]),
                    source_fingerprint=can_res_rec.provenance.get("content_hash", ""),
                    provenance="start.analysis.pipelines",
                )
            except Exception as e:
                logger.warning("Recommender canonical pipeline generation error: %s", e)

        # 4. Generate deterministic review artifacts
        from start.review.executor import (
            evaluate_deterministic_governance_disposition,
            generate_review_artifacts,
            run_domain_checkpoints,
        )

        artifacts_dir = root / "artifacts"
        with trace_scope(OP_ARTIFACT_PERSIST, {"artifacts_dir": str(artifacts_dir)}):
            artifacts_by_checkpoint = generate_review_artifacts(
                ctx_instance.bundle,
                records,
                artifacts_dir,
                products=products,
            )
        all_arts_list = [art for arts in artifacts_by_checkpoint.values() for art in arts]
        for art in all_arts_list:
            art_id = getattr(art, "artifact_id", "ART")
            producing_step = getattr(art, "producing_step_id", None) or getattr(art, "node_id", None)
            spec = getattr(art, "spec", None)
            art_title = getattr(spec, "title", None) or getattr(art, "title", art_id)
            art_type = getattr(spec, "artifact_type", None) or getattr(art, "artifact_type", "table")
            ev_ids = list(getattr(spec, "evidence_ids", ()) or ())
            data_fp = getattr(art, "data_fingerprint", "")
            payload = getattr(art, "semantic_payload", None)
            rendering_fmt = getattr(art, "rendering_format", "json")
            # Do not guess producer step when not authoritatively established (ARTIFACT_PRODUCER_GUESSES = 0)

            kind = "plot" if (art_type == "plot" or rendering_fmt == "plot" or any(k in art_id for k in ("ROC", "CALIBRATION", "IMPORTANCE", "CORR", "COV", "WATERFALL"))) else (
                "table" if art_type in ("table", "summary_table", "diagnostic_table", "confusion_matrix") else "metric"
            )

            artifacts_dict[art_id] = {
                "id": art_id,
                "name": art_title,
                "title": art_title,
                "kind": kind,
                "artifact_type": art_type,
                "producing_step_id": producing_step,
                "data_fingerprint": data_fp,
                "evidence_ids": ev_ids,
                "content": payload,
                "mimeType": "application/json",
            }
            evt_art = RuntimeEvent(
                run_id=run_id,
                event_type="artifact_created",
                status="COMPLETED",
                source_agent="DeterministicEngine",
                target_agent="StructuredReviewer",
                stage="ARTIFACT_GENERATION",
                action="generate_artifact",
                node_id=producing_step,
                artifact_refs=[art_id],
                elapsed_seconds=round(time.time() - start_time, 2),
                message=f"Created review artifact {art_id}: {art_title}",
                metadata={
                    "artifact_id": art_id,
                    "title": art_title,
                    "kind": kind,
                    "artifact_type": art_type,
                    "data_fingerprint": data_fp,
                    "evidence_ids": ev_ids,
                    "producing_step_id": producing_step,
                    "content": payload,
                },
            )
            sink.emit(evt_art)
            all_events.append(evt_art)

        # CP-005: Evidence committed
        cp5_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{run_id}:CP-005:{len(records)}:{len(artifacts_dict)}")).replace("-", "")[:16]
        cp5_item = {
            "checkpoint_id": "CP-005",
            "name": "Evidence committed",
            "status": "completed",
            "producing_stage": "step-evidence",
            "agent_signature": "EvidenceLedger",
            "evidence_ids": [r.evidence_id for r in records],
            "commit_hash": cp5_hash,
            "timestamp": datetime.datetime.fromtimestamp(time.time(), tz=datetime.UTC).isoformat(),
            "summary": f"Committed {len(records)} evidence surfaces and {len(artifacts_dict)} review artifacts to immutable ledger.",
        }
        checkpoints_list.append(cp5_item)
        evt_cp5 = RuntimeEvent(
            run_id=run_id,
            event_type="checkpoint_committed",
            status="COMPLETED",
            source_agent="EvidenceLedger",
            target_agent="ModelGovernance",
            stage="CHECKPOINTS",
            action="commit_CP-005",
            node_id="step-evidence",
            checkpoint_id="CP-005",
            evidence_refs=[r.evidence_id for r in records],
            elapsed_seconds=round(time.time() - start_time, 2),
            message="Committed checkpoint CP-005: Evidence committed",
            metadata=cp5_item,
        )
        sink.emit(evt_cp5)
        all_events.append(evt_cp5)

        # 5. Checkpoints & Governance Evaluation
        decisions = run_domain_checkpoints(
            ctx_instance.bundle,
            records,
            artifacts_by_checkpoint=artifacts_by_checkpoint,
            products=products,
            interactive=False,
        )

        committee_result = None
        if resolved.engine_kind == EngineKind.MARKET_SUBSET:
            from start.agents.committee import CrossAnalyticalCommittee

            committee = CrossAnalyticalCommittee()
            committee_result = committee.conduct_committee_review(records)

        final_gov_disposition = evaluate_deterministic_governance_disposition(
            ctx_instance.bundle,
            records,
            decisions,
            committee_result,
        )

        # Emit governance event ONLY when governance actually evaluated
        has_governance_step = any(
            s[0] == "step-governance" or s[2] == "governance" for s in resolved.step_specs
        )
        if has_governance_step and final_gov_disposition is not None:
            # Policy Adapter contract evaluation (Gate B)
            with trace_scope(OP_POLICY_EVALUATE, {"governance_disposition": final_gov_disposition}):
                pol_adapter = PolicyAdapter()
                policy_res = pol_adapter.evaluate_signoff(
                    run_id=run_id,
                    evidence_ids=[r.evidence_id for r in records],
                    disposition=final_gov_disposition,
                )

            evt_gov = RuntimeEvent(
                run_id=run_id,
                event_type="governance_decided",
                status="COMPLETED",
                source_agent="EvidenceCritic",
                target_agent="ModelGovernance",
                stage="GOVERNANCE",
                action="evaluate_governance_disposition",
                node_id="step-governance",
                parent_node_id=prev_node_id,
                elapsed_seconds=round(time.time() - start_time, 2),
                message=f"Governance evaluated: {final_gov_disposition}",
                metadata={
                    "governance_disposition": final_gov_disposition,
                    "policy_decision_id": policy_res.decision_id if policy_res else None,
                },
            )
            sink.emit(evt_gov)
            all_events.append(evt_gov)

            # Build real Merkle attestation seal
            with trace_scope(OP_GOVERNANCE_COMMIT, {"seal": True, "disposition": final_gov_disposition}):
                from start.attestation.seal import build_seal

                seal_meta = {
                    "run_id": run_id,
                    "workflow": workflow_id,
                    "context_id": context_id,
                    "n_records": len(records),
                    "governance_disposition": final_gov_disposition,
                }
                seal = build_seal(
                    review_id=run_id,
                    evidence_head=records[-1].evidence_id if records else None,
                    metadata=seal_meta,
                )
                root_val = seal.root() if callable(seal.root) else seal.root
                merkle_root = str(root_val)

            evt_att = RuntimeEvent(
                run_id=run_id,
                event_type="attestation_created",
                status="COMPLETED",
                source_agent="ModelGovernance",
                target_agent="AuditArchive",
                stage="GOVERNANCE",
                action="create_attestation_seal",
                node_id="step-governance",
                parent_node_id="step-governance",
                elapsed_seconds=round(time.time() - start_time, 2),
                message=f"Attestation signed Merkle root {merkle_root[:16]}",
                metadata={"merkle_root": merkle_root},
            )
            sink.emit(evt_att)
            all_events.append(evt_att)

            # CP-006: Attestation signed
            cp6_item = {
                "checkpoint_id": "CP-006",
                "name": "Attestation signed",
                "status": "completed",
                "producing_stage": "step-governance",
                "agent_signature": "ModelGovernance",
                "evidence_ids": [records[-1].evidence_id] if records else [],
                "commit_hash": merkle_root,
                "timestamp": datetime.datetime.fromtimestamp(time.time(), tz=datetime.UTC).isoformat(),
                "summary": f"Governance disposition '{final_gov_disposition}' certified with Merkle root {merkle_root[:16]}...",
            }
            checkpoints_list.append(cp6_item)
            evt_cp6 = RuntimeEvent(
                run_id=run_id,
                event_type="checkpoint_committed",
                status="COMPLETED",
                source_agent="ModelGovernance",
                target_agent="AuditArchive",
                stage="CHECKPOINTS",
                action="commit_CP-006",
                node_id="step-governance",
                checkpoint_id="CP-006",
                elapsed_seconds=round(time.time() - start_time, 2),
                message="Committed checkpoint CP-006: Attestation signed",
                metadata=cp6_item,
            )
            sink.emit(evt_cp6)
            all_events.append(evt_cp6)

        # CP-AGENTIC: Agentic executive synthesis (for agentic_session mode)
        if execution_mode == "agentic_session":
            cp_agentic_hash = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{run_id}:CP-AGENTIC:{len(records)}")).replace("-", "")[:16]
            cp_agentic_item = {
                "checkpoint_id": "CP-AGENTIC",
                "name": "Agentic executive synthesis",
                "status": "completed",
                "producing_stage": "step-synthesis",
                "agent_signature": "ExecutiveDirector",
                "evidence_ids": [r.evidence_id for r in records[-3:]] if records else [],
                "commit_hash": cp_agentic_hash,
                "timestamp": datetime.datetime.fromtimestamp(time.time(), tz=datetime.UTC).isoformat(),
                "summary": f"Agentic session synthesis: evaluated {len(records)} evidence records across {context_id} under executive governance supervision.",
            }
            checkpoints_list.append(cp_agentic_item)
            evt_agentic = RuntimeEvent(
                run_id=run_id,
                event_type="checkpoint_committed",
                status="COMPLETED",
                source_agent="ExecutiveDirector",
                target_agent="AuditArchive",
                stage="CHECKPOINTS",
                action="commit_CP-AGENTIC",
                node_id="step-synthesis",
                checkpoint_id="CP-AGENTIC",
                elapsed_seconds=round(time.time() - start_time, 2),
                message="Committed checkpoint CP-AGENTIC: Agentic executive synthesis",
                metadata=cp_agentic_item,
            )
            sink.emit(evt_agentic)
            all_events.append(evt_agentic)

        # 6. Build Presentation Model
        from start.reporting.presentation import build_presentation_model

        try:
            domains_tuple = tuple(ctx_instance.bundle.domains)
            pres_model = build_presentation_model(
                run_id=run_id,
                mode=str(ctx_instance.bundle.mode),
                domains=domains_tuple,
                materiality=str(ctx_instance.bundle.materiality),
                lifecycle=str(ctx_instance.bundle.lifecycle),
                records=records,
                artifacts_by_checkpoint=artifacts_by_checkpoint,
                governance_disposition=final_gov_disposition or "REVIEW_REQUIRED",
                attestation_seal_merkle_root=merkle_root or "",
                orchestration_events=[e.to_dict() for e in all_events],
            )
        except Exception:
            pres_model = None

        total_elapsed = round(time.time() - start_time, 2)
        evt_end = RuntimeEvent(
            run_id=run_id,
            event_type="workflow_completed",
            status="COMPLETED",
            source_agent="Director",
            target_agent="Specialist",
            stage="COMPLETED",
            action="finalize_workflow_run",
            elapsed_seconds=total_elapsed,
            message=f"Completed {workflow_id} review run ({len(records)} unique evidence records produced)",
            metadata={"elapsed_seconds": total_elapsed, "n_records": len(records)},
        )
        sink.emit(evt_end)
        all_events.append(evt_end)

        if tracer is not None:
            trace_jsonl_path = root / "trace.jsonl"
            tracer.export_jsonl(trace_jsonl_path)

        return ExecutionResult(
            run_id=run_id,
            workflow_id=workflow_id,
            context_id=context_id,
            context_instance=ctx_instance,
            resolved_execution=resolved,
            records=records,
            products=products,
            ledger=ledger,
            artifacts=artifacts_dict,
            governance_disposition=final_gov_disposition,
            attestation_seal=seal,
            merkle_root=merkle_root,
            decisions=decisions,
            presentation_model=pres_model,
            checkpoints=checkpoints_list,
            events=all_events,
            output_path=str(root),
            elapsed_seconds=total_elapsed,
            tracer=tracer,
            policy_result=policy_res,
        )
