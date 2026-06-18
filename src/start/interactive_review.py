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
    trust_domain: str = "public"  # public | enterprise (set by gateway prompt)
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


def _ask_yes_no(prompt: str, default_no: bool = True, ask: Any = input) -> bool:
    """Ask a yes/no question. Defaults to No when default_no is True."""
    suffix = "[y/N]" if default_no else "[Y/n]"
    raw = (ask(f"{prompt} {suffix}: ") or "").strip().lower()
    if not raw:
        return not default_no
    return raw in ("y", "yes")


def prompt_review_config(initial: ReviewConfig | None = None, ask: Any = input) -> ReviewConfig:
    cfg = initial or ReviewConfig()
    console.print("\n[bold]StART model review — interactive setup[/bold]\n")

    # v2.3.0: the evidence-driven committee review is the default path. The
    # reviewer can opt down to the legacy/basic review explicitly.
    cfg.enterprise_mode = _ask_yes_no(
        "Use AI review committee workflow?", default_no=False, ask=ask
    )
    if cfg.enterprise_mode:
        console.print(
            "[cyan]AI review committee selected (evidence-first cards, "
            "ValidationAgent sensitivity review, MRM-grade signoff).[/cyan]"
        )
    else:
        console.print(
            "[yellow]Legacy/basic review selected — the v2.3.0 committee "
            "workflow, evidence cards, and MRM signoff will not be shown.[/yellow]"
        )

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
        # v2.3.0 gateway UX: ask about the enterprise gateway FIRST. Choosing it
        # sets the enterprise trust domain and skips the public provider menu and
        # all public API-key prompts — the request routes only through the
        # generic enterprise gateway adapter (which degrades to deterministic
        # review, never to a public provider, if the private package is absent).
        if _ask_yes_no("Use enterprise LLM gateway?", default_no=True, ask=ask):
            cfg.llm_provider = "enterprise_llm_gateway"
            cfg.trust_domain = "enterprise"
            console.print(
                "[cyan]Enterprise trust domain selected. Public provider keys "
                "(OpenAI/Anthropic/Grok) will not be requested. Requests route "
                "through the enterprise gateway adapter.[/cyan]"
            )
        else:
            cfg.trust_domain = "public"
            cfg.llm_provider = _ask(
                "LLM provider", cfg.llm_provider,
                ["none", "openai", "anthropic", "grok"], ask,
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
    _target_supplied_by_user = bool(cfg.target)
    if cfg.data_path:
        from start.data.loaders import load_any_tabular

        df = load_any_tabular(cfg.data_path)
        # v2.3.1 #2: dataset transparency for a user-supplied file.
        console.print(
            f"[bold]Dataset:[/bold] {cfg.data_path} "
            f"(user-supplied) — {len(df)} rows x {df.shape[1]} columns"
        )
    else:
        from start.modeling.data import load_attrition_dataset

        df = load_attrition_dataset(seed=cfg.seed)
        if not cfg.target:
            cfg.target = "attrition"
        # v2.3.1 #2: name the demo dataset, its source, and a public URL.
        console.print(
            "[bold]Demo dataset:[/bold] sklearn breast-cancer "
            "(reframed as binary attrition)\n"
            "  source: scikit-learn datasets (UCI ML Breast Cancer Wisconsin, "
            "diagnostic)\n"
            "  url: https://archive.ics.uci.edu/dataset/17/"
            "breast+cancer+wisconsin+diagnostic\n"
            f"  loaded: {len(df)} rows x {df.shape[1]} columns"
        )

    # v2.3.1 #2: target transparency — what was selected, by whom, and the
    # candidate columns discovery would consider.
    _binary_candidates = [
        c for c in df.columns
        if c != cfg.target and df[c].dropna().nunique() == 2
    ][:8]
    if _target_supplied_by_user:
        console.print(
            f"[bold]Target:[/bold] {cfg.target} (supplied by user)"
        )
    else:
        console.print(
            f"[bold]Target:[/bold] {cfg.target} (inferred by discovery)"
        )
    if _binary_candidates:
        console.print(
            "  candidate target columns: " + ", ".join(_binary_candidates)
        )
    console.print("")

    if cfg.enterprise_mode:
        return _run_enterprise(cfg, df, llm)

    console.print(
        f"\n[bold yellow]Running legacy/basic review[/bold yellow] "
        f"(agent mode: {cfg.agent_mode}) — for the full v2.3.0 committee review, "
        f"re-run with --enterprise.\n"
    )
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
        f"\n[bold]Running AI REVIEW COMMITTEE workflow[/bold] (agent mode: {cfg.agent_mode})\n"
    )
    # Item 1 / v2.3.1 #3: introduce the committee in a boxed panel with colored
    # agent names.
    from start.agent_roster import render_agent_roster_panel
    from start.review_session import ReviewSession

    console.print(render_agent_roster_panel())
    console.print("")
    session = ReviewSession(run_id="RUN")
    # Section A / v2.3.1 #3,#5: visible LLM activation as its own boxed panel.
    from start.providers.llm_activation import preflight_llm
    from start.review_tables import llm_activation_panel

    if cfg.agent_mode == "llm" and cfg.llm_provider not in ("none", ""):
        provider_name = cfg.llm_provider
    else:
        provider_name = getattr(llm, "name", "none") if llm is not None else "none"
    activation = preflight_llm(provider_name, llm)
    console.print(llm_activation_panel(activation))
    console.print("")

    # Item 9 / v2.3.1 #4: AI-engineering environment as a colored Rich table.
    try:
        import tempfile as _tmp

        from start.ai_engineering.layer import run_ai_engineering_layer
        from start.progress import progress_bar
        from start.review_tables import adapter_inventory_table

        # v2.3.1 #6: real percentage progress over the adapter sweep — the total
        # is the actual adapter count, so the percentage shown is real.
        _probe_dir = _tmp.mkdtemp()
        _n_adapters = len(run_ai_engineering_layer({}, output_root=_tmp.mkdtemp()).descriptions)
        with progress_bar(_n_adapters, "AI-engineering adapter sweep",
                          enabled=not cfg.non_interactive, console=console) as _adv:
            _probe = run_ai_engineering_layer(
                {}, output_root=_probe_dir, on_adapter=lambda _r: _adv(1)
            )
        console.print(adapter_inventory_table(_probe.control_surface()))
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
        from start.evidence_store import EvidenceStore
        from start.interactive_checkpoints import resolve_checkpoint
        from start.modeling.data_statistics import compute_data_statistics
        from start.review_session import Decision

        # v2.3.0 #1: assemble the evidence store from real diagnostics so every
        # Ask-Agent answer is grounded (or explicitly refused), never fabricated.
        try:
            _ev_stats = compute_data_statistics(df, cfg.target)
            _evidence = EvidenceStore.from_artifacts(data_stats=_ev_stats)
        except Exception:
            _evidence = EvidenceStore()

        # v2.3.0 #5: DatasetDiscoveryAgent transparency BEFORE any recommendation.
        from start.review_tables import (
            correlation_evidence_table,
            dataset_discovery_table,
            outlier_evidence_table,
        )

        _cand_targets = [cfg.target] if cfg.target else _evidence.candidate_targets
        console.print(dataset_discovery_table(
            _evidence, _cand_targets,
            (cfg.train_prop, cfg.test_prop, cfg.oos_prop)))
        console.print("")
        # v2.3.0 #6: FeatureEngineeringAgent evidence tables (real diagnostics).
        _otab, _has_out = outlier_evidence_table(_evidence, 10)
        if _has_out:
            console.print(_otab)
            console.print("")
        _ctab, _has_corr = correlation_evidence_table(_evidence, 10)
        if _has_corr:
            console.print(_ctab)
            console.print("")

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
            checkpoint="architecture", evidence=_evidence,
        )

        def _ask_arch(question: str) -> str:
            return ask_agent("ArchitectureReviewAgent", question, _arch_ctx,
                             session, llm=llm, llm_connected=_llm_connected).answer

        # v2.3.0 #1/#4: present the evidence-first committee card BEFORE the
        # decision — Evidence -> Recommendation -> Alternatives -> Risks.
        from start.committee_card import CommitteeCard, render_card_rich

        _arch_card = CommitteeCard(
            agent="ArchitectureReviewAgent", purpose="Model selection",
            evidence=[
                f"dataset size {len(df)} rows",
                f"{df.shape[1] - 1} candidate features",
                "binary classification task",
            ],
            recommendation=f"{ar.recommendation['family']} "
            f"({ar.recommendation.get('activation', 'relu')})",
            alternatives=[f"{ar.recommendation['family']} (recommended)",
                          arch_choice, "xgboost"],
            risks=[ar.risk_if_ignored] if ar.risk_if_ignored else [],
            artifacts_used=["data_statistics"],
        )
        console.print(render_card_rich(_arch_card))

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
                    emit=lambda m: console.print(m), evidence=_evidence,
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
                emit=lambda m: console.print(m), evidence=_evidence,
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
    # Section O: adapter transparency (purpose/status/time/artifacts/evidence #10)
    cs = outcome.ai_engineering.control_surface()
    if cs:
        from start.review_tables import adapter_inventory_table

        console.print("\n[bold]AI-engineering adapter inventory[/bold]")
        console.print(adapter_inventory_table(cs))
    # v2.3.0: assemble the post-run evidence store ONCE (sensitivity, metrics,
    # tuning) — reused by the ValidationAgent review and the MRM signoff.
    from start.evidence_store import EvidenceStore as _EvStore

    _final_store = _EvStore.from_artifacts(
        copilot_exec=outcome.copilot_execution, sensitivity=outcome.sensitivity,
        tuning_run=outcome.tuning_run,
    )

    # Sections D/J/K: split table, metrics-by-split, explainability (Rich #9)
    if outcome.copilot_execution:
        from start.review_tables import importance_table, metrics_table

        ce = outcome.copilot_execution
        console.print("\n[bold]Model execution[/bold]")
        if ce.metrics_by_split:
            console.print(metrics_table(ce.metrics_by_split))
        if ce.global_importance:
            console.print("")
            console.print(importance_table(ce.global_importance))
    # Section H: real tuning trials table (Rich #9)
    if outcome.tuning_run:
        from start.review_tables import tuning_table

        console.print("\n[bold]Hyperparameter tuning (single-split validation)[/bold]")
        console.print(tuning_table(outcome.tuning_run.to_dict().get("trials", [])))
    # v2.3.1 #7: real stratified K-fold tuning (train-only) per-fold table.
    if outcome.kfold_tuning:
        from start.review_tables import kfold_table

        console.print("\n[bold]K-fold tuning (train-only, stratified)[/bold]")
        console.print(kfold_table(outcome.kfold_tuning))
    # v2.3.0 #7/#8: ValidationAgent review (sensitivity ranking, shock table,
    # business interpretation, signoff impact) as an explicit checkpoint before
    # signoff — with [A]ccept / [Q] ask / [C]hallenge when interactive.
    _validation_summary = None
    if outcome.sensitivity:
        from start.validation_review import run_validation_checkpoint

        _validation_summary = run_validation_checkpoint(
            outcome.sensitivity, session, _final_store, console,
            interactive=interactive, auto_accept=cfg.accept_recommendations,
            llm=llm, llm_connected=(activation.status == "CONNECTED"), ask=input,
        )
        session.validation_review = _validation_summary
    # Section N: artifact discovery (everything generated, discoverable)
    if outcome.artifact_registry and outcome.artifact_registry.artifacts:
        console.print("\n[bold]Artifacts generated[/bold]")
        for art in outcome.artifact_registry.artifacts:
            console.print(f"  {art.name:24s} [{art.artifact_type}]  -> {art.path}")

    # v2.3.0 #8/#11: MRM-grade signoff that weighs performance, generalization,
    # calibration, feature dependence (sensitivity), and reviewer activity.
    from start.mrm_signoff import evaluate_signoff, render_signoff_rich

    _mrm = evaluate_signoff(_final_store, session, primary_metric="auc_roc")
    _st, _sp = render_signoff_rich(_mrm)
    console.print("")
    console.print(_st)
    console.print(_sp)
    # expose on the session for transcript/dashboard
    session.mrm_signoff = _mrm.to_dict()

    # v2.3.1 #8: compact review decision ledger (checkpoint/choice/recommendation/
    # status/evidence/impact) in the terminal.
    from start.review_tables import decision_ledger_table

    _ledger_rows = session.to_dict().get("decisions", [])
    if _ledger_rows:
        console.print("")
        console.print(decision_ledger_table(_ledger_rows))

    # v2.3.0 #12: refresh the dashboard so challenges, validation review, and the
    # MRM signoff (all computed post-run) are embedded in the primary dashboard.
    if outcome.dashboard_model is not None:
        try:
            from start.reporting.dashboard import write_dashboard

            outcome.dashboard_model.review_journey = session.to_dict()
            if outcome.kfold_tuning is not None:
                outcome.dashboard_model.kfold = outcome.kfold_tuning.to_dict()
            write_dashboard(outcome.dashboard_model, cfg.output_root, outcome.run_id)
        except Exception:
            pass

    # Item 11: write the committee transcript (conversations, decisions,
    # overrides) alongside the dashboards.
    from start.reporting.review_transcript import write_transcript

    session.run_id = outcome.run_id
    transcript_paths = write_transcript(
        session, cfg.output_root, outcome.run_id, sensitivity=outcome.sensitivity,
        kfold=outcome.kfold_tuning,
    )

    summary = outcome.findings_register.summary()
    console.print(
        f"\n[bold green]AI review committee complete[/bold green] — {outcome.run_id}\n"
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
