"""Interactive ``start review`` flow.

Drives the ReviewOrchestrator with user prompts that expose the full
capability surface (dataset, target, task, split strategy, architecture family,
activation, explainability, robustness, agent mode, provider). LLM-mode-only
prompts (free-form objective, target/task clarification) appear ONLY when the
user selects ``llm`` mode; deterministic mode shows no LLM prompts.

The console renderer prints each pipeline stage as it runs, so the user always
sees progress — never a blank screen.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console

from start.modeling.architecture_registry import ACTIVATIONS, list_families
from start.modeling.review_orchestrator import ReviewOrchestrator, StageEvent

console = Console()


@dataclass
class ReviewConfig:
    data_path: str | None = None
    target: str | None = None
    task_override: str | None = None
    split_strategy: str = "stratified"
    architecture_family: str | None = None
    activation: str | None = None
    explain_method: str = "integrated_gradients"
    robustness_suite: str = "standard"
    agent_mode: str = "deterministic"
    llm_provider: str = "none"
    objective: str = ""  # LLM-mode free-form objective
    run_dl: bool = False
    enterprise_mode: bool = False  # layered orchestrator + dashboard + graph
    governance_mode: bool = False  # surface governance findings prominently
    costlier_errors: str = "balanced"  # balanced | false_negatives | false_positives
    accept_recommendations: bool = False  # auto-accept agent recommendations
    show_progress: bool = True
    non_interactive: bool = False  # set by CLI --non-interactive
    train_prop: float = 0.60
    test_prop: float = 0.20
    oos_prop: float = 0.20
    tuning_strategy: str = "bounded_random_search"
    tuning_trials: int = 5
    output_root: str = "start_output"
    seed: int = 42
    notes: list[str] = field(default_factory=list)


def _ask(prompt: str, default: str, choices: list[str] | None = None, ask: Any = input) -> str:
    rendered = f"{prompt}"
    if choices:
        rendered += " [" + "/".join(c.upper() if c == default else c for c in choices) + "]"
    raw = ask(f"{rendered} (default: {default}): ").strip()
    if not raw:
        return default
    if choices and raw.lower() not in choices:
        return default
    return raw.lower() if choices else raw


def prompt_review_config(initial: ReviewConfig | None = None, ask: Any = input) -> ReviewConfig:
    cfg = initial or ReviewConfig()
    console.print("\n[bold]StART model review — interactive setup[/bold]\n")

    if not cfg.data_path:
        raw = ask("Dataset path (blank = built-in demo dataset): ").strip()
        cfg.data_path = raw or None

    if not cfg.target:
        raw = ask("Target column (blank = let discovery propose): ").strip()
        cfg.target = raw or None

    cfg.split_strategy = _ask(
        "Split strategy", cfg.split_strategy,
        ["random", "stratified", "time_based", "group", "custom"], ask,
    )
    tabular = list_families("tabular")
    cfg.architecture_family = _ask(
        "Architecture family", cfg.architecture_family or "mlp", tabular, ask
    )
    cfg.activation = _ask("Activation", cfg.activation or "relu", list(ACTIVATIONS), ask)
    cfg.explain_method = _ask(
        "Explainability", cfg.explain_method,
        ["integrated_gradients", "gradient_shap", "permutation"], ask,
    )
    cfg.robustness_suite = _ask(
        "Robustness suite", cfg.robustness_suite, ["standard", "extended"], ask
    )
    cfg.agent_mode = _ask("Agent mode", cfg.agent_mode, ["deterministic", "llm"], ask)

    # Section C: ask whether to actually train (default Yes). Without this,
    # enterprise review silently runs diagnostics-only and skips execution.
    run_dl_ans = ask(
        "Run model training, metrics, explainability, sensitivity, and robustness? [Y/n]: "
    ).strip().lower()
    cfg.run_dl = run_dl_ans in ("", "y", "yes")
    if not cfg.run_dl:
        console.print(
            "[yellow]Model execution, metrics, explainability, sensitivity, and "
            "robustness will be skipped (diagnostics-only review).[/yellow]"
        )

    # Section D: user-controlled train/test/OOS split proportions.
    if cfg.run_dl:
        cfg.train_prop, cfg.test_prop, cfg.oos_prop = _ask_split_proportions(ask)

    # Section H: tuning strategy + trials (only meaningful when training).
    if cfg.run_dl:
        cfg.tuning_strategy = _ask(
            "Tuning strategy", cfg.tuning_strategy,
            ["none", "bounded_random_search", "grid_search", "optuna_if_available"], ask,
        )
        if cfg.tuning_strategy != "none":
            raw = ask("Number of trials [default 5]: ").strip()
            cfg.tuning_trials = int(raw) if raw.isdigit() and int(raw) > 0 else 5

    # LLM-mode-only prompts
    if cfg.agent_mode == "llm":
        cfg.llm_provider = _ask(
            "LLM provider", cfg.llm_provider,
            ["none", "openai", "anthropic", "grok", "enterprise_llm_gateway"], ask,
        )
        cfg.objective = ask(
            "Business objective (free-form, sent only as context, never raw data): "
        ).strip()
        clar = ask("Any target/task clarification for the model-risk reviewer? ").strip()
        if clar:
            cfg.notes.append(f"User clarification: {clar}")
            # Section M: infer cost priority from free-text clarification.
            inferred = _infer_cost_priority(clar)
            if inferred:
                cfg.costlier_errors = inferred
                console.print(
                    f"[cyan]Cost priority inferred from your note: "
                    f"{inferred}.[/cyan]"
                )
    return cfg


def _ask_split_proportions(ask: Any = input) -> tuple[float, float, float]:
    """Prompt for train/test/OOS proportions; validate they sum to ~1.0."""
    def _f(prompt: str, default: float) -> float:
        raw = ask(f"{prompt} [default {default:.2f}]: ").strip()
        try:
            return float(raw) if raw else default
        except ValueError:
            return default

    for _ in range(3):
        train = _f("Train proportion", 0.60)
        test = _f("Test proportion", 0.20)
        oos = _f("OOS proportion", 0.20)
        total = train + test + oos
        if abs(total - 1.0) < 1e-6 and min(train, test, oos) > 0:
            return train, test, oos
        console.print(
            f"[yellow]Proportions must be positive and sum to 1.0 "
            f"(got {total:.2f}); using 0.60/0.20/0.20.[/yellow]"
        )
        break
    return 0.60, 0.20, 0.20


def _infer_cost_priority(text: str) -> str | None:
    """Section M: map free-text clarification to a cost priority."""
    t = text.lower()
    fn_signals = ("false negative", "missed", "miss ", "recall", "catch all",
                  "sensitivity", "attrition", "churn")
    fp_signals = ("false positive", "unnecessary", "avoid flagging",
                  "precision", "specificity", "false alarm")
    if any(s in t for s in fn_signals):
        return "false_negatives"
    if any(s in t for s in fp_signals):
        return "false_positives"
    return None


def _render_stage(event: StageEvent) -> None:
    color = {"running": "yellow", "complete": "green", "skipped": "dim"}.get(event.status, "white")
    label = event.stage.replace("_", " ").title()
    detail = f" — {event.detail}" if event.detail else ""
    console.print(f"  [{color}]{event.status.upper():9s}[/{color}] {label}{detail}")


def run_interactive_review(cfg: ReviewConfig) -> Any:

    # resolve LLM provider with strict trust-domain enforcement
    llm = None
    if cfg.agent_mode == "llm" and cfg.llm_provider not in ("none", ""):
        from start.core.config import LLMConfig
        from start.providers.keys import ensure_provider_key, key_required
        from start.providers.llm import get_llm_provider
        from start.providers.trust_domains import trust_domain

        # Section B: securely prompt for a missing public-provider key (hidden
        # input, session-only, never persisted). Enterprise gateway needs no
        # public key, so this is a no-op there.
        interactive = not cfg.non_interactive
        if key_required(cfg.llm_provider):
            status = ensure_provider_key(
                cfg.llm_provider,
                prompt_for_key=True if interactive else False,
                interactive=interactive,
            )
            if status.source == "missing" and interactive:
                console.print(
                    f"[yellow]No key provided for {cfg.llm_provider}; "
                    "continuing in deterministic fallback.[/yellow]"
                )
        domain = trust_domain(cfg.llm_provider).value  # 'public' | 'private' | 'none'
        expected = domain if domain in ("public", "private") else None
        llm = get_llm_provider(LLMConfig(provider=cfg.llm_provider), expected_domain=expected)

    # load data (demo or user path)
    if cfg.data_path:
        from start.data.loaders import load_any_tabular

        df = load_any_tabular(cfg.data_path)
    else:
        from start.modeling.data import load_attrition_dataset

        df = load_attrition_dataset(seed=cfg.seed)
        if not cfg.target:
            cfg.target = "attrition"

    if cfg.enterprise_mode:
        return _run_enterprise(cfg, df, llm)

    console.print(f"\n[bold]Running review[/bold] (agent mode: {cfg.agent_mode})\n")
    orch = ReviewOrchestrator(on_stage=_render_stage)
    outcome = orch.run(
        df,
        user_target=cfg.target,
        task_override=cfg.task_override,
        split_strategy=cfg.split_strategy,
        agent_mode=cfg.agent_mode,
        llm=llm,
        output_root=cfg.output_root,
        seed=cfg.seed,
        run_dl=cfg.run_dl,
    )
    console.print(
        f"\n[bold green]Review complete[/bold green] — {outcome.run_id}\n"
        f"  task: {outcome.task_type} | modality: {outcome.modality} | "
        f"recommended: {outcome.recommended_family}\n"
        f"  evidence records: {len(outcome.evidence)} | "
        f"critique: {'PASSED' if outcome.agent_review.critique_ok else 'FAILED'}\n"
        f"  sign-off: {outcome.agent_review.signoff.split('.')[0]}.\n"
        f"  report: {outcome.report_path}"
    )
    return outcome


def _render_layer(lr: Any) -> None:
    if lr.status == "running":
        return
    color = {"complete": "green", "error": "red", "skipped": "dim"}.get(lr.status, "white")
    console.print(
        f"  [{color}]{lr.status.upper():9s}[/{color}] {lr.name:16s} "
        f"{lr.runtime_seconds:.3f}s  findings={len(lr.findings)} "
        f"artifacts={len(lr.artifacts)} evidence={len(lr.evidence_ids)}"
    )


def _render_adapter(result: Any) -> None:
    color = {"complete": "green", "not_installed": "yellow", "error": "red"}.get(
        result.status, "white"
    )
    console.print(
        f"    [{color}]{result.status.upper():13s}[/{color}] {result.adapter:16s} "
        f"{result.runtime_seconds:.3f}s  artifacts={len(result.artifacts)} "
        f"evidence={len(result.evidence)}"
    )


def _run_enterprise(cfg: ReviewConfig, df: Any, llm: Any) -> Any:
    from start.modeling.dataset_source import (
        describe_custom_dataset,
        describe_demo_dataset,
    )
    from start.modeling.enterprise_orchestrator import EnterpriseReviewOrchestrator

    console.print(
        f"\n[bold]Running ENTERPRISE review[/bold] (agent mode: {cfg.agent_mode})\n"
    )
    # Item 1: introduce the review committee at startup.
    from start.agent_roster import render_agent_roster
    from start.review_session import ReviewSession

    console.print(render_agent_roster())
    console.print("")
    session = ReviewSession(run_id="RUN")
    # Section A: visible LLM activation before any agent runs
    from start.providers.llm_activation import preflight_llm

    if cfg.agent_mode == "llm" and cfg.llm_provider not in ("none", ""):
        provider_name = cfg.llm_provider
    else:
        provider_name = getattr(llm, "name", "none") if llm is not None else "none"
    activation = preflight_llm(provider_name, llm)
    console.print(activation.render_terminal())
    console.print("")

    # Item 9: show the AI-engineering environment (adapters) up front.
    try:
        import tempfile as _tmp

        from start.agent_roster import render_adapter_panel
        from start.ai_engineering.layer import run_ai_engineering_layer

        _probe = run_ai_engineering_layer({}, output_root=_tmp.mkdtemp())
        console.print(render_adapter_panel(_probe.control_surface()))
        console.print("")
    except Exception:
        pass

    # dataset provenance: custom path vs built-in demo
    if cfg.data_path:
        dataset_source = describe_custom_dataset(df, cfg.data_path, cfg.target)
    else:
        dataset_source = describe_demo_dataset(df, cfg.target or "attrition")

    # Section B: interactive decision checkpoints (architecture, metric).
    # Recommendations come from the same agents the orchestrator uses; the
    # user's resolution is recorded and applied (never a silent override).
    interactive = not cfg.non_interactive if hasattr(cfg, "non_interactive") else False
    arch_choice = cfg.architecture_family or "mlp"
    cost_choice = cfg.costlier_errors
    checkpoint_decisions = []
    if interactive or cfg.accept_recommendations:
        from start.agent_dialogue import AgentContext, ask_agent
        from start.agents.engineering_agents import (
            ArchitectureReviewAgent,
        )
        from start.interactive_checkpoints import resolve_checkpoint
        from start.review_session import Decision

        ar = ArchitectureReviewAgent().review(
            user_family=arch_choice, user_activation=cfg.activation or "relu",
            modality="tabular", n_samples=len(df), n_features=df.shape[1] - 1,
            task_type="binary_classification",
        )
        # Item 2: let the user interrogate the agent live at this checkpoint.
        _llm_connected = activation.status == "CONNECTED"
        _arch_ctx = AgentContext(
            agent="ArchitectureReviewAgent",
            recommendation=ar.recommendation["family"],
            reason=ar.reason, risk_if_ignored=ar.risk_if_ignored,
            alternatives=[{"family": ar.recommendation["family"]},
                          {"family": arch_choice}, {"family": "xgboost"}],
            dataset_summary=f"{len(df)} rows x {df.shape[1]} cols",
            checkpoint="architecture",
        )

        def _ask_arch(question: str) -> str:
            return ask_agent("ArchitectureReviewAgent", question, _arch_ctx,
                             session, llm=llm, llm_connected=_llm_connected).answer

        arch_dec = resolve_checkpoint(
            "architecture", arch_choice, ar.recommendation["family"], ar.reason,
            evidence_id=ar.evidence_id, explanation=ar.reason,
            interactive=interactive, auto_accept=cfg.accept_recommendations,
            ask=input, emit=lambda m: console.print(m), on_ask=_ask_arch,
        )
        arch_choice = arch_dec.effective_value
        checkpoint_decisions.append(arch_dec.to_dict())
        # Item 3: persist the decision so downstream agents/dashboard see it.
        session.record_decision(Decision(
            key="architecture", prompt="Model family?",
            recommended=ar.recommendation["family"], user_value=cfg.architecture_family or "mlp",
            effective=arch_choice, choice=arch_dec.choice, rationale=ar.reason,
            evidence_ids=[ar.evidence_id],
        ))

        # #1/#2: feature-engineering and metric checkpoints — each with Ask
        # Agent, recorded to the session so they drive downstream execution
        # (e.g. rejecting correlation pruning keeps all features).
        if cfg.run_dl:
            from start.interactive_checkpoints_flow import (
                run_feature_engineering_checkpoints,
                run_metric_checkpoint,
            )
            from start.modeling.data_statistics import compute_data_statistics
            from start.modeling.fe_recommendations import recommend_feature_engineering

            try:
                _stats = compute_data_statistics(df, cfg.target)
                _fe = recommend_feature_engineering(_stats)
                run_feature_engineering_checkpoints(
                    _fe, session, interactive=interactive,
                    auto_accept=cfg.accept_recommendations, llm=llm,
                    llm_connected=_llm_connected, ask=input,
                    emit=lambda m: console.print(m),
                )
            except Exception:
                pass
            # metric / cost-priority checkpoint
            from start.agents.engineering_agents import select_primary_metric

            _mc = select_primary_metric("binary_classification", costlier_errors=cost_choice)
            cost_choice = run_metric_checkpoint(
                cost_choice, cost_choice, _mc.get("reason", ""), session,
                interactive=interactive, auto_accept=cfg.accept_recommendations,
                llm=llm, llm_connected=_llm_connected, ask=input,
                emit=lambda m: console.print(m),
            )

    from start.agent_roster import announce_adapter_activity

    orch = EnterpriseReviewOrchestrator(
        on_stage=_render_stage, on_layer=_render_layer, on_adapter=_render_adapter,
        on_adapter_start=lambda name, activity: console.print(
            f"    [cyan]{announce_adapter_activity(name, activity)}[/cyan]"
        ),
    )
    outcome = orch.run(
        df,
        user_target=cfg.target,
        task_override=cfg.task_override,
        split_strategy=cfg.split_strategy,
        agent_mode=cfg.agent_mode,
        llm=llm,
        output_root=cfg.output_root,
        run_dl=cfg.run_dl,
        enterprise_mode=True,
        seed=cfg.seed,
        architecture=arch_choice,
        activation=cfg.activation or "relu",
        costlier_errors=cost_choice,
        dataset_source=dataset_source,
        requested_provider=cfg.llm_provider if cfg.agent_mode == "llm" else None,
        split_props=(cfg.train_prop, cfg.test_prop, cfg.oos_prop),
        explain_method=cfg.explain_method,
        tuning_strategy=cfg.tuning_strategy,
        tuning_trials=cfg.tuning_trials,
        session=session,
    )
    # Section K: agent reasoning traces (thinking visibility)
    if outcome.trace_log and outcome.trace_log.traces:
        console.print("\n[bold]Agent reasoning traces[/bold]")
        console.print(outcome.trace_log.render_terminal())
    # Section O: educative adapter control surface (purpose/role/install)
    cs = outcome.ai_engineering.control_surface()
    if cs:
        console.print("\n[bold]AI-engineering control surface[/bold]")
        for r in cs:
            console.print(
                f"  {r['adapter']:16s} [{r['status']}]\n"
                f"      purpose: {r['purpose']}\n"
                f"      role   : {r['role']}\n"
                f"      install: {r.get('install_guidance', '') or '—'}"
            )
    # Sections D/J/K: split table, metrics-by-split, explainability (visible)
    if outcome.copilot_execution:
        from start.modeling.copilot_execution import render_copilot_execution_markdown

        console.print("\n[bold]Model execution[/bold]")
        console.print(render_copilot_execution_markdown(outcome.copilot_execution))
    # Section H: real tuning trials table (visible)
    if outcome.tuning_run:
        from start.modeling.tuning_run import render_tuning_run_markdown

        console.print("\n[bold]Hyperparameter tuning[/bold]")
        console.print(render_tuning_run_markdown(outcome.tuning_run))
    # #4: sensitivity analysis table (feature/shock/baseline/shocked/delta/risk)
    if outcome.sensitivity:
        from start.modeling.sensitivity_analysis import render_sensitivity_markdown

        console.print("\n[bold]Sensitivity analysis[/bold]")
        console.print(render_sensitivity_markdown(outcome.sensitivity))
    # Section N: artifact discovery (everything generated, discoverable)
    if outcome.artifact_registry and outcome.artifact_registry.artifacts:
        console.print("\n[bold]Artifacts generated[/bold]")
        for art in outcome.artifact_registry.artifacts:
            console.print(f"  {art.name:24s} [{art.artifact_type}]  -> {art.path}")
    # Item 11: write the committee transcript (conversations, decisions,
    # overrides) alongside the dashboards.
    from start.reporting.review_transcript import write_transcript

    session.run_id = outcome.run_id
    transcript_paths = write_transcript(
        session, cfg.output_root, outcome.run_id, sensitivity=outcome.sensitivity
    )

    summary = outcome.findings_register.summary()
    console.print(
        f"\n[bold green]Enterprise review complete[/bold green] — {outcome.run_id}\n"
        f"  task: {outcome.task_type} | modality: {outcome.modality} | "
        f"recommended: {outcome.recommended_family}\n"
        f"  findings: {summary['total']} "
        f"(Critical={summary['Critical']} High={summary['High']} "
        f"Medium={summary['Medium']} Low={summary['Low']})\n"
        f"  AI-engineering: {outcome.ai_engineering.available_count}"
        f"/{outcome.ai_engineering.total} adapters available\n"
        f"  evidence critique: {'PASSED' if outcome.critique_ok else 'FAILED'}\n"
        f"  dashboard: {outcome.dashboard_paths['html']}\n"
        f"  transcript: {transcript_paths['html']}"
    )
    return outcome
