"""Authoritative Canonical Execution Service for StART v5.1.0.

Provides the single non-web execution coordinator shared between CLI and Web:
- Orchestrates real deterministic engines without reimplementing scientific calculations
- Emits typed RuntimeEvents at genuine execution boundaries
- Appends to canonical EvidenceLedger and binds true EvidenceRecord IDs
- Generates canonical artifacts and attestation seals
- Zero dependencies on start.web (CORE_RUNTIME_IMPORTS_START_WEB = 0)
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    events: list[RuntimeEvent] = field(default_factory=list)
    output_path: str = ""
    elapsed_seconds: float = 0.0

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
    ) -> ExecutionResult:
        """Execute canonical StART workflow, streaming genuine events into event_sink."""
        start_time = time.time()
        sink = event_sink or NoOpEventSink()
        run_id = run_id or f"RUN-{uuid.uuid4().hex[:8]}"
        root = Path(output_root) / run_id
        root.mkdir(parents=True, exist_ok=True)
        ledger = EvidenceLedger(root / "ledger.jsonl", root / "evidence")

        # 1. Instantiate Context ONCE
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
                message=f"Initialized execution context '{context_id}'",
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
                },
            )
        )

        # 2. Resolve Workflow
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

            tuning_res = run_tuning(
                train_df,
                target=target_col,
                features=feature_cols,
                n_trials=trials_requested,
                strategy="bounded_random_search",
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
                            f"metric={t.validation_metric:.4f} status={t.status}"
                        ),
                        metadata={
                            "trial": t.trial,
                            "params": t.params,
                            "validation_metric": t.validation_metric,
                            "best_metric": tuning_res.best_metric,
                            "requested_trials": trials_requested,
                            "attempted_trials": idx + 1,
                            "completed_trials": completed_trials,
                        },
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
                    f"Completed hyperparameter search "
                    f"({completed_trials}/{trials_requested} trials finished)"
                ),
                metadata={
                    "requested_trials": trials_requested,
                    "attempted_trials": completed_trials,
                    "completed_trials": completed_trials,
                    "best_metric": tuning_res.best_metric if tuning_res else None,
                },
            )
            sink.emit(evt_tune)
            all_events.append(evt_tune)
            prev_node_id = "step-tuning"

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
                prev_node_id = step_id

        elif resolved.engine_kind == EngineKind.MARKET_SUBSET:
            market = ctx_instance.bundle.market
            short_rate = ctx_instance.bundle.short_rate
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

        # 4. Generate deterministic review artifacts
        from start.review.executor import (
            evaluate_deterministic_governance_disposition,
            generate_review_artifacts,
            run_domain_checkpoints,
        )

        artifacts_dir = root / "artifacts"
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
            artifacts_dict[art_id] = {
                "id": art_id,
                "name": getattr(art, "title", art_id),
                "artifact_type": getattr(art, "artifact_type", "table"),
                "producing_step_id": producing_step,
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
                message=f"Created review artifact {art_id}",
            )
            sink.emit(evt_art)
            all_events.append(evt_art)

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
                metadata={"governance_disposition": final_gov_disposition},
            )
            sink.emit(evt_gov)
            all_events.append(evt_gov)

            # Build real Merkle attestation seal
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
            message=f"Completed {workflow_id} review run ({len(records)} evidence surfaces produced)",
            metadata={"elapsed_seconds": total_elapsed, "n_records": len(records)},
        )
        sink.emit(evt_end)
        all_events.append(evt_end)

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
            events=all_events,
            output_path=str(root),
            elapsed_seconds=total_elapsed,
        )
