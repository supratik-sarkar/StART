"""End-to-end review pipeline.

Flow: plan -> policy guard -> route -> execute (deterministic engines) ->
evidence critique -> narrative -> narrative critique -> ledger append.
A blocked policy decision stops execution; a blocked narrative critique
replaces the LLM narrative with the deterministic template narrative.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from start.agents import (
    EvidenceCriticAgent,
    ExecutionAgent,
    NarrativeAgent,
    PolicyGuardAgent,
    ReviewPlannerAgent,
    TestRouterAgent,
)
from start.cli.view import ProgressDashboardUI
from start.core.config import StartConfig, load_policy
from start.core.hashing import hash_dataframe
from start.core.schemas import (
    DatasetSummary,
    Materiality,
    ModelMetadata,
    RunResult,
    TaskType,
)
from start.engine.state import StepCheckpointer
from start.evidence.ledger import EvidenceLedger
from start.orchestration.middleware import SelfHealingExecutionInterceptor
from start.providers.compute import get_compute_provider
from start.providers.experiment import get_experiment_provider
from start.providers.llm import get_llm_provider
from start.registry import TestContext
from start.telemetry.bus import TelemetryBus


def build_context(config: StartConfig, train, test=None, model=None, extra=None) -> TestContext:
    return TestContext(
        train=train,
        test=test,
        target_column=config.model.target_column,
        prediction_column=config.model.prediction_column,
        score_column=config.model.score_column,
        timestamp_column=config.data.timestamp_column,
        entity_id_column=config.data.entity_id_column,
        model=model,
        seed=config.seed,
        extra=extra or {},
    )


def _enrich_narrative(narrative, records, ctx) -> None:
    """Agentic governance layer: cross-evidence findings, next-step
    suggestions, adversarial challenges, governance gate, and a reviewer
    sign-off recommendation — all deterministic and citation-carrying."""
    from start.agents import (
        ChallengeAgent,
        GovernanceAgent,
        ModelRiskFindingAgent,
        SignoffAgent,
        TestSuggestionAgent,
    )

    narrative.findings.extend(ModelRiskFindingAgent().findings(records))
    narrative.findings.extend(ChallengeAgent().challenge(records))
    narrative.next_steps.extend(TestSuggestionAgent().suggest(records, ctx))
    governance_ok, governance_items = GovernanceAgent().review(records)
    narrative.findings.extend(governance_items)
    narrative.signoff = SignoffAgent().conclude(records, governance_ok, governance_items)


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
    _enrich_narrative(narrative, records, ctx)
    narrative_critique = critic.critique_narrative(narrative, records)
    if not narrative_critique.ok and narrative.generator.startswith("llm"):
        narrative = narrator._template_narrative(run_id, records)
        _enrich_narrative(narrative, records, ctx)
        narrative_critique = critic.critique_narrative(narrative, records)

    # Dual-mode agent review: deterministic governance fallback, or
    # LLM-assisted evidence-grounded review gated by the citation critic.
    from start.agents.review import run_agent_review

    agent_llm = llm
    if config.agent.llm_provider and config.agent.llm_provider != config.llm.provider:
        from start.core.config import LLMConfig

        agent_llm = get_llm_provider(
            LLMConfig(provider=config.agent.llm_provider, model=config.llm.model)
        )
    agent_review = run_agent_review(
        records,
        mode=config.agent.mode,
        llm=agent_llm,
        ctx=ctx,
        policy_hash=records[0].policy_hash if records else None,
        demo_meta=(ctx.extra or {}).get("demo_meta"),
    )

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
        agent_review=agent_review,
        policy=decision,
    )


def review_dataframes(
    train_df,
    test_df=None,
    oos_df=None,
    *,
    target_column: str,
    score_column: str | None = "score",
    prediction_column: str | None = None,
    model=None,
    config: StartConfig | None = None,
    extra: dict | None = None,
    seed: int = 42,
) -> RunResult:
    """First-class pandas API: run the full review directly on in-memory
    DataFrames. If only `train_df` is given, a stratified 60/20/20
    train/test/OOS split is applied. No framework changes needed for user data.
    """
    from start.connectors import PandasConnector

    bundle = PandasConnector(
        train_df,
        test_df,
        oos_df,
        seed=seed,
        target_column=target_column,
        score_column=score_column,
    ).load_bundle()

    cfg = config or StartConfig()
    cfg.seed = seed
    cfg.model.target_column = target_column
    cfg.model.score_column = score_column
    cfg.model.prediction_column = prediction_column
    cfg.data.source = "pandas"
    cfg.data.dataset_id = bundle.source

    merged_extra = {"oos": bundle.oos, "data_notes": bundle.notes}
    merged_extra.update(extra or {})
    ctx = build_context(cfg, bundle.train, bundle.test, model=model, extra=merged_extra)
    return run_review(cfg, ctx)


# --- StART v2.4.0 Core Interceptor & Telemetry UI Bridge ---


async def execute_pipeline_v240_bridge(workflow_id: str, stage_name: str, core_runner_func, context_payload, violation_type="class_imbalance"):
    bus = TelemetryBus()
    checkpointer = StepCheckpointer()
    ui = ProgressDashboardUI(bus)
    interceptor = SelfHealingExecutionInterceptor(telemetry_bus=bus, checkpointer=checkpointer)
    
    async def wrapped_execution():
        ui.set_global_progress(15.0)
        # Wrap the legacy test function execution inside the self-healing interceptor framework
        result = await interceptor.execute_test_step(
            workflow_id=workflow_id,
            stage_name=stage_name,
            test_callable=lambda ctx: core_runner_func(ctx),
            test_context=context_payload
        )
        ui.set_global_progress(100.0)
        await asyncio.sleep(0.5)
        return result

    return await ui.monitor_execution(wrapped_execution())

def run_v240_sync_entry(workflow_id: str, stage_name: str, core_runner_func, context_payload, violation_type="class_imbalance"):
    """Synchronous entry point proxy to execute the async TUI inside the legacy CLI thread context."""
    try:
        return asyncio.run(execute_pipeline_v240_bridge(workflow_id, stage_name, core_runner_func, context_payload, violation_type))
    except RuntimeError:
        # Fallback if an event loop is already running in the current thread
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(execute_pipeline_v240_bridge(workflow_id, stage_name, core_runner_func, context_payload, violation_type))
