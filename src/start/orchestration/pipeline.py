"""End-to-end review pipeline.

Flow: plan -> policy guard -> route -> execute (deterministic engines) ->
evidence critique -> narrative -> narrative critique -> ledger append.
A blocked policy decision stops execution; a blocked narrative critique
replaces the LLM narrative with the deterministic template narrative.
"""

from __future__ import annotations

from pathlib import Path

from start.agents import (
    EvidenceCriticAgent,
    ExecutionAgent,
    NarrativeAgent,
    PolicyGuardAgent,
    ReviewPlannerAgent,
    TestRouterAgent,
)
from start.core.config import StartConfig, load_policy
from start.core.hashing import hash_dataframe
from start.core.schemas import (
    DatasetSummary,
    Materiality,
    ModelMetadata,
    RunResult,
    TaskType,
)
from start.evidence.ledger import EvidenceLedger
from start.providers.compute import get_compute_provider
from start.providers.experiment import get_experiment_provider
from start.providers.llm import get_llm_provider
from start.registry import TestContext


def build_context(config: StartConfig, train, test=None, model=None, extra=None) -> TestContext:
    return TestContext(
        train=train,
        test=test,
        target_column=config.model.target_column,
        prediction_column=config.model.prediction_column,
        score_column=config.model.score_column,
        model=model,
        seed=config.seed,
        extra=extra or {},
    )


def run_review(config: StartConfig, ctx: TestContext) -> RunResult:
    policy = load_policy(config.policy_file)
    llm = get_llm_provider(config.llm)
    compute = get_compute_provider(config.compute)
    out_root = Path(config.output.root)
    ledger = EvidenceLedger(
        out_root / config.output.ledger_file, out_root / config.output.evidence_store
    )
    experiments = get_experiment_provider(
        config.experiment.provider,
        config.experiment.tracking_uri,
        config.experiment.experiment_name,
        config.output.root,
    )

    model_meta = ModelMetadata(
        model_id=config.model.model_id,
        task_type=TaskType(config.model.task_type),
        materiality=Materiality(config.model.materiality),
        target_column=config.model.target_column,
        prediction_column=config.model.prediction_column,
        score_column=config.model.score_column,
    )
    n_rows = len(ctx.train) if ctx.train is not None else 0
    dataset = DatasetSummary(
        dataset_id=config.data.dataset_id,
        n_rows=n_rows,
        n_columns=len(ctx.train.columns) if ctx.train is not None else 0,
        columns=list(ctx.train.columns) if ctx.train is not None else [],
        source=config.data.path,
    )

    run_id = experiments.start_run(f"{config.project_name}:{model_meta.model_id}")

    planner = ReviewPlannerAgent(config, llm)
    plan = planner.plan(model_meta, dataset)

    guard = PolicyGuardAgent(policy)
    decision = guard.check(plan, data_root=config.data.path)
    if not decision.allowed:
        experiments.end_run(run_id)
        return RunResult(run_id=run_id, plan=plan, policy=decision)

    plan, _unknown = TestRouterAgent().route(plan)

    input_hash = hash_dataframe(ctx.train) if ctx.train is not None else None
    executor = ExecutionAgent(compute, decision.policy_hash, run_id)
    records = executor.execute(plan, ctx, input_artifact_hash=input_hash)

    critic = EvidenceCriticAgent()
    evidence_critique = critic.critique_evidence(records)

    narrator = NarrativeAgent(llm)
    narrative = narrator.generate(run_id, records)
    narrative_critique = critic.critique_narrative(narrative, records)
    if not narrative_critique.ok and narrative.generator.startswith("llm"):
        narrative = narrator._template_narrative(run_id, records)
        narrative_critique = critic.critique_narrative(narrative, records)

    for record in records:
        ledger.append(record)
        numeric = {k: float(v) for k, v in record.metrics.items() if isinstance(v, (int, float))}
        if numeric:
            experiments.log_metrics(run_id, numeric)
    experiments.end_run(run_id)

    combined = evidence_critique
    combined.issues.extend(narrative_critique.issues)
    combined.ok = evidence_critique.ok and narrative_critique.ok

    return RunResult(
        run_id=run_id,
        plan=plan,
        evidence=records,
        critique=combined,
        narrative=narrative,
        policy=decision,
    )
