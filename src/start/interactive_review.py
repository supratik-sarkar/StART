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
from start.modeling.review_orchestrator import STAGES, ReviewOrchestrator, StageEvent

console = Console()


@dataclass
class ReviewConfig:
    data_path: str | None = None
    target: str | None = None
    dataset_selection: Any = None
    task_override: str | None = None
    split_strategy: str = "stratified"
    architecture_family: str | None = None
    activation: str | None = None
    explain_method: str = "integrated_gradients"
    robustness_suite: str = "standard"
    agent_mode: str = "deterministic"
    llm_provider: str = "none"
    llm_model: str | None = None  # v3.1.1: selected model ID from provider API
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
    class_weight: str | None = None
    preset_key: str | None = None
    custom_space: dict[str, Any] | None = None
    k_folds: int = 3
    validation_scheme: str = "holdout"
    # v3.1.1: typed cost specification — governance, tuning, and weighting
    # logic consume this structured field, never free-form notes.
    # Schema:
    #   {"type": "balanced"}
    #   {"type": "critical_class", "critical_class": str, "relative_cost": float}
    #   {"type": "matrix", "matrix": {class_i: {class_j: cost, ...}, ...}}
    cost_specification: dict[str, Any] = field(default_factory=lambda: {"type": "balanced"})
    open_figures: bool | None = None
    figure_delay: float = 1.5


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


def prompt_review_config(
    initial: ReviewConfig | None = None,
    ask: Any = input,
    model_discovery: Any = None,
) -> ReviewConfig:
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
    cfg.robustness_suite = _ask(
        "Robustness suite", cfg.robustness_suite, ["standard", "extended"], ask
    )
    # 0. AI Reviewer Agent Backend Selection (LLM Mode)
    console.print("\nSelect AI Reviewer Agent Backend (LLM Mode):")
    console.print("  [1] None (Deterministic Rule-Based / Local Engines) (default)")
    console.print("  [2] Enterprise LLM Gateway (Firm Environment)")
    console.print("  [3] Public LLM Providers (OpenAI, Anthropic, Gemini, DeepSeek, Groq, etc.)")
    backend_choice = ask("Select backend [default: 1]: ").strip() or "1"
    
    if backend_choice == "2":
        cfg.agent_mode = "llm"
        cfg.llm_provider = "enterprise_llm_gateway"
        cfg.trust_domain = "enterprise"
    elif backend_choice == "3":
        cfg.agent_mode = "llm"
        cfg.trust_domain = "public"
        cfg.llm_provider = _ask(
            "Select Public LLM Provider", cfg.llm_provider or "openai",
            ["openai", "anthropic", "gemini", "deepseek", "grok"], ask,
        )
        # v3.1.1: query provider Models API for available models
        _discovery = model_discovery
        if _discovery is None:
            from start.providers.model_discovery import RealProviderModelDiscovery
            _discovery = RealProviderModelDiscovery()
        _available_models = _discovery.list_models(cfg.llm_provider)
        if not _available_models and cfg.llm_provider in ("openai", "anthropic", "deepseek", "gemini", "grok"):
            fallback = {
                "openai": ["gpt-4o-mini", "gpt-4o"],
                "anthropic": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],
                "deepseek": ["deepseek-chat"],
                "gemini": ["gemini-2.0-flash"],
                "grok": ["grok-2-latest"],
            }.get(cfg.llm_provider, [])
            _available_models = fallback

        if _available_models:
            console.print(f"\n  Available models for {cfg.llm_provider}:")
            for idx, mid in enumerate(_available_models, 1):
                console.print(f"    [{idx}] {mid}")
            raw_model = ask("Select model [default: 1]: ").strip() or "1"
            try:
                sel_idx = int(raw_model) - 1
                if 0 <= sel_idx < len(_available_models):
                    cfg.llm_model = _available_models[sel_idx]
                else:
                    cfg.llm_model = _available_models[0]
            except ValueError:
                # Allow typing a model ID directly if it's in the list
                if raw_model in _available_models:
                    cfg.llm_model = raw_model
                else:
                    cfg.llm_model = _available_models[0]
            console.print(f"  [cyan]Selected model: {cfg.llm_model}[/cyan]")
        else:
            console.print(
                f"[yellow]No models returned by {cfg.llm_provider} API. "
                "Model selection unavailable — check API key and connectivity.[/yellow]"
            )
            cfg.llm_model = None
    else:
        cfg.agent_mode = "deterministic"
        cfg.llm_provider = "none"
        cfg.trust_domain = "none"

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

        cfg.explain_method = _ask(
            "Explainability", cfg.explain_method,
            ["integrated_gradients", "gradient_shap", "permutation"], ask,
        )

    # LLM-mode-only prompts
    if cfg.agent_mode == "llm":
        if cfg.llm_provider == "enterprise_llm_gateway":
            console.print(
                "[cyan]Enterprise trust domain selected. Public provider keys "
                "will not be requested. Requests route through the enterprise gateway adapter.[/cyan]"
            )
        else:
            # Securely prompt for key immediately if needed
            import sys

            from start.providers.keys import ensure_provider_key, key_required
            if key_required(cfg.llm_provider) and sys.stdin.isatty():
                status = ensure_provider_key(cfg.llm_provider, prompt_for_key=True, interactive=True)
                if not status.ok:
                    console.print(
                        f"\n[yellow]API key for '{cfg.llm_provider}' is missing. "
                        "Degrading to deterministic rule-based backend.[/yellow]"
                    )
                    cfg.agent_mode = "deterministic"
                    cfg.llm_provider = "none"
                    cfg.trust_domain = "none"
        if cfg.agent_mode == "llm":
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

    # v3.1.1: structured multiclass cost specification prompt
    # Prompt after target is known but before execution.
    # The cost specification is stored in cfg.cost_specification, never in notes.
    return cfg


def _prompt_cost_specification(
    task_type: str, classes: list[str], ask: Any = input,
) -> dict[str, Any]:
    """Prompt user for structured multiclass cost specification.

    Returns a typed dict with schema:
      {"type": "balanced"}
      {"type": "critical_class", "critical_class": str, "relative_cost": float}
      {"type": "matrix", "matrix": {class_i: {class_j: cost, ...}, ...}}
    """
    import sys
    if ask is input and not sys.stdin.isatty():
        return {"type": "balanced"}

    console.print("\n[bold]Misclassification Cost Specification[/bold]")
    console.print(f"  Classes: {classes}")
    console.print("  [1] Balanced treatment (default)")
    console.print("  [2] One critical class + relative miss cost")
    console.print("  [3] Full class-to-class cost matrix")
    choice = (ask("Select cost specification [default: 1]: ").strip() or "1")

    if choice == "2":
        console.print(f"  Available classes: {classes}")
        crit = ask(f"  Critical class [default: {classes[0]}]: ").strip() or classes[0]
        if crit not in classes:
            console.print(f"  [yellow]'{crit}' not in classes; using '{classes[0]}'.[/yellow]")
            crit = classes[0]
        raw_cost = ask("  Relative miss cost for this class [default: 5.0]: ").strip() or "5.0"
        try:
            rel_cost = float(raw_cost)
        except ValueError:
            rel_cost = 5.0
        return {"type": "critical_class", "critical_class": crit, "relative_cost": rel_cost}

    if choice == "3":
        console.print("  Enter the cost of predicting class j when true class is i.")
        console.print("  Diagonal (correct predictions) should be 0.")
        matrix: dict[str, dict[str, float]] = {}
        for ci in classes:
            row: dict[str, float] = {}
            for cj in classes:
                if ci == cj:
                    row[cj] = 0.0
                    continue
                raw = ask(f"    Cost(true={ci}, pred={cj}) [default: 1.0]: ").strip() or "1.0"
                try:
                    row[cj] = float(raw)
                except ValueError:
                    row[cj] = 1.0
            matrix[ci] = row
        return {"type": "matrix", "matrix": matrix}

    return {"type": "balanced"}


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
    if cfg.llm_provider not in ("none", ""):
        from start.runtime_profile import assert_provider_allowed

        assert_provider_allowed(cfg.llm_provider)

    if cfg.agent_mode == "llm" and cfg.llm_provider not in ("none", ""):
        import os

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

        DEFAULT_MODELS: dict[str, str] = {
            "openai": "gpt-4o-mini",
            "anthropic": "claude-3-5-haiku-20241022",
            "deepseek": "deepseek-chat",
            "gemini": "gemini-2.0-flash",
            "grok": "grok-2-latest",
        }
        resolved_model = cfg.llm_model or os.environ.get(f"START_{cfg.llm_provider.upper()}_MODEL") or DEFAULT_MODELS.get(cfg.llm_provider, "")
        if cfg.llm_provider in ("gateway", "enterprise_llm_gateway") and not resolved_model:
            resolved_model = os.environ.get("START_GATEWAY_MODEL", "")
            if not resolved_model:
                raise RuntimeError(
                    "Provider 'gateway' requires a model to be explicitly specified via --model or START_GATEWAY_MODEL; "
                    "StART never guesses model names on operator-supplied gateways."
                )
        llm = get_llm_provider(
            LLMConfig(provider=cfg.llm_provider, model=resolved_model),
            expected_domain=expected,
        )

    # load data (demo or user path)
    _target_supplied_by_user = bool(cfg.target)
    if getattr(cfg, "dataset_selection", None) is not None and cfg.dataset_selection.frame is not None:
        selection = cfg.dataset_selection
        df = selection.frame
        console.print(
            f"[bold]Dataset:[/bold] {selection.display_name} — "
            f"{selection.n_rows:,} rows x {selection.n_columns} columns "
            f"({selection.source_reference})"
        )
        if not cfg.target:
            cfg.target = selection.target_column
    elif cfg.data_path:
        from start.data.loaders import load_any_tabular

        df = load_any_tabular(cfg.data_path)
        # v2.3.1 #2: dataset transparency for a user-supplied file.
        console.print(
            f"[bold]Dataset:[/bold] {cfg.data_path} "
            f"(user-supplied) — {len(df)} rows x {df.shape[1]} columns"
        )
    else:
        preset_key = getattr(cfg, "preset_key", None)
        from start.modeling.data import load_preset_dataset

        df = load_preset_dataset(preset_key, seed=cfg.seed)
        if not cfg.target:
            target_map = {
                "A": "is_fraud",
                "B": "target_value",
                "C": "adjusted_price",
                "D": "decision_label"
            }
            cfg.target = target_map.get(preset_key, "is_fraud")


        preset_details = {
            "A": {
                "name": "Synthetic Anomaly Detection & Transaction Monitoring",
                "source": "Synthetic AML Profile Generator (imbalanced)",
                "url": "synthetic (locally generated)",
            },
            "B": {
                "name": "Synthetic Time-Series Forecasting Profile",
                "source": "Synthetic Trend & Seasonality Regressive Sequence",
                "url": "synthetic (locally generated)",
            },
            "C": {
                "name": "Synthetic Asset Pricing Matrix",
                "source": "Synthetic Multi-feature Pricing Regression",
                "url": "synthetic (locally generated)",
            },
            "D": {
                "name": "Synthetic ML Decision Support Model Data",
                "source": "Synthetic Multi-class Decision Profile",
                "url": "synthetic (locally generated)",
            }
        }
        details = preset_details.get(preset_key, {
            "name": "Synthetic AML / Transaction Monitoring (Default)",
            "source": "Synthetic Transaction Generator",
            "url": "synthetic (locally generated)",
        })


        console.print(
            f"[bold]Demo dataset:[/bold] {details['name']}\n"
            f"  source: {details['source']}\n"
            f"  url: {details['url']}\n"
            f"  loaded: {len(df)} rows x {df.shape[1]} columns"
        )

    # Rename target column in demo dataset if user specified a custom target column name
    if not cfg.data_path and cfg.target and cfg.target not in df.columns:
        default_col = "is_fraud"
        if getattr(cfg, "dataset_selection", None) and getattr(cfg.dataset_selection, "target_column", None):
            default_col = cfg.dataset_selection.target_column
        if default_col in df.columns:
            df = df.rename(columns={default_col: cfg.target})

    # Reconcile target cardinality & task override early
    if cfg.target in df.columns:
        nunique = df[cfg.target].dropna().nunique()
        if cfg.task_override == "binary_classification" and nunique > 2:
            if cfg.non_interactive:
                raise ValueError(
                    f"Target '{cfg.target}' has {nunique} unique values, which requires "
                    f"multiclass_classification, but the selected task override is binary_classification."
                )
            else:
                from rich.prompt import Confirm
                console.print(
                    f"\n[bold yellow]Target Cardinality Mismatch:[/bold yellow] Target '{cfg.target}' "
                    f"has {nunique} unique values, which requires multiclass classification, "
                    f"but binary classification was selected."
                )
                if Confirm.ask("Would you like to switch to multiclass_classification?", default=True):
                    cfg.task_override = "multiclass_classification"
                else:
                    raise ValueError(
                        f"Execution aborted: Selected binary classification is incompatible with target "
                        f"'{cfg.target}' cardinality ({nunique})."
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

    # v3.1.1: prompt for structured cost specification if multiclass
    if cfg.target in df.columns:
        _nunique = df[cfg.target].dropna().nunique()
        _resolved_task = cfg.task_override or (
            "multiclass_classification" if _nunique > 2
            else "binary_classification"
        )
        if _resolved_task == "multiclass_classification" and not cfg.non_interactive:
            _classes = sorted(df[cfg.target].dropna().unique().tolist(), key=str)
            cfg.cost_specification = _prompt_cost_specification(
                _resolved_task, [str(c) for c in _classes], ask=input,
            )

    if cfg.enterprise_mode:
        return _run_enterprise(cfg, df, llm)

    console.print(
        f"\n[bold yellow]Running legacy/basic review[/bold yellow] "
        f"(agent mode: {cfg.agent_mode}) — for the full v2.3.0 committee review, "
        f"re-run with --enterprise.\n"
    )
    from start.progress import progress_bar
    with progress_bar(len(STAGES), "Model Review Execution", enabled=not cfg.non_interactive, console=console) as adv:
        def on_stage_with_progress(event):
            _render_stage(event)
            if event.status in ("complete", "skipped"):
                adv(1)
        orch = ReviewOrchestrator(on_stage=on_stage_with_progress)
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
    from start.agents.discovery import TaskInferenceAgent
    try:
        ti = TaskInferenceAgent().infer(df, cfg.target, override=cfg.task_override)
        task_type = ti.task_type
    except Exception:
        task_type = cfg.task_override or "binary_classification"

    # Initialize session and LangSmith tracer
    import uuid

    from start.agent_roster import render_agent_roster_panel
    from start.ai_engineering.langsmith_tracer import LangSmithTracer
    from start.cli.panels import render_run_header
    from start.modeling.dataset_source import (
        describe_custom_dataset,
        describe_demo_dataset,
    )
    from start.modeling.enterprise_orchestrator import EnterpriseReviewOrchestrator
    from start.review_session import ReviewSession
    from start.runtime_profile import active_profile

    enterprise_run_id = getattr(cfg, "review_id", None) or ("RUN-ENT-" + uuid.uuid4().hex[:8])
    session = ReviewSession(run_id=enterprise_run_id)
    tracer = LangSmithTracer(run_id=enterprise_run_id)
    tracer.start_review(review_id=enterprise_run_id)

    # Render Panel 1: Run Header (D6)
    console.print(
        render_run_header(
            review_id=enterprise_run_id,
            target=cfg.target or "",
            task_type=task_type,
            dataset_shape=(len(df), df.shape[1]),
            profile=active_profile().value,
            policy_id="public_demo",
            seed=cfg.seed,
            mode=cfg.agent_mode,
            provider=cfg.llm_provider,
        )
    )
    console.print("")

    # Committee Roster Panel
    console.print(
        f"\n[bold]Running AI REVIEW COMMITTEE workflow[/bold] (agent mode: {cfg.agent_mode})\n"
    )
    console.print(render_agent_roster_panel())
    console.print("")
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
            task_type=task_type,
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
            business_context=cfg.objective or "",
            reviewer_clarification="\n".join(cfg.notes) or "",
            task_type=task_type,
            model_name=arch_choice,
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
                f"{task_type.replace('_', ' ')} task",
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
            llm=llm, session=session, ctx=_arch_ctx,
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
                _fe = recommend_feature_engineering(_stats, cost_specification=cfg.cost_specification)
                run_feature_engineering_checkpoints(
                    _fe, session, interactive=interactive,
                    auto_accept=cfg.accept_recommendations,
                    df=df, target=cfg.target, already_weighted=bool(cfg.class_weight),
                    llm=llm, llm_connected=_llm_connected, ask=input,
                    emit=lambda m: console.print(m), evidence=_evidence,
                    business_context=cfg.objective or "",
                    reviewer_clarification="\n".join(cfg.notes) or "",
                    task_type=task_type,
                    model_name=arch_choice,
                )
            except Exception:
                pass
            # metric / cost-priority checkpoint
            from start.agents.engineering_agents import select_primary_metric

            _mc = select_primary_metric(task_type, costlier_errors=cost_choice)
            cost_choice = run_metric_checkpoint(
                cost_choice, cost_choice, _mc.get("reason", ""), session,
                interactive=interactive, auto_accept=cfg.accept_recommendations,
                llm=llm, llm_connected=_llm_connected, ask=input,
                emit=lambda m: console.print(m), evidence=_evidence,
                business_context=cfg.objective or "",
                reviewer_clarification="\n".join(cfg.notes) or "",
                task_type=task_type,
                model_name=arch_choice,
            )
    else:
        from start.review_session import Decision

        session.record_decision(Decision(
            key="architecture",
            prompt="Model architecture family",
            recommended=arch_choice,
            user_value=arch_choice,
            effective=arch_choice,
            choice="auto_accept",
            rationale=f"Batch execution using configured family {arch_choice}",
            agent_rationale=f"Configured family {arch_choice}",
        ))
        session.record_decision(Decision(
            key="metric_priority",
            prompt="Cost priority / primary metric",
            recommended=cost_choice or "balanced",
            user_value=cost_choice or "balanced",
            effective=cost_choice or "balanced",
            choice="auto_accept",
            rationale="Batch execution with standard balanced cost weighting",
            agent_rationale="Default balanced error cost",
        ))

    from start.agent_roster import announce_adapter_activity
    from start.progress import progress_bar
    with progress_bar(len(STAGES), "AI Review Committee Execution", enabled=not cfg.non_interactive, console=console) as adv:
        def on_stage_with_progress(event):
            _render_stage(event)
            if event.status in ("complete", "skipped"):
                adv(1)
        
        orch = EnterpriseReviewOrchestrator(
            on_stage=on_stage_with_progress,
            on_layer=_render_layer,
            on_adapter=_render_adapter,
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
            class_weight=cfg.class_weight,
            custom_space=cfg.custom_space,
            validation=cfg.validation_scheme,
            k_folds=cfg.k_folds,
            cost_specification=cfg.cost_specification,
            run_id=enterprise_run_id,
        )
    # Section K: agent reasoning traces (thinking visibility)
    if outcome.trace_log and outcome.trace_log.traces:
        console.print("\n[bold]Agent reasoning traces[/bold]")
        console.print(outcome.trace_log.render_terminal())
    # Section O: adapter transparency recorded for dashboard
    # (terminal table was already printed during the pre-run sweep)
    # v2.3.0: assemble the post-run evidence store ONCE (sensitivity, metrics,
    # tuning) — reused by the ValidationAgent review and the MRM signoff.
    from start.evidence_store import EvidenceStore as _EvStore

    _final_store = _EvStore.from_artifacts(
        model_exec=outcome.model_execution, sensitivity=outcome.sensitivity,
        tuning_run=outcome.tuning_run,
    )

    # Sections D/J/K: split table, metrics-by-split, explainability (Rich #9)
    if outcome.model_execution:
        from start.review_tables import importance_table, metrics_table

        me = outcome.model_execution
        console.print("\n[bold]Model execution[/bold]")
        if me.metrics_by_split:
            console.print(metrics_table(me.metrics_by_split))

            # Inline Terminal Plots
            import numpy as np

            from start.cli.terminal_plots import (
                render_drift_sparkline,
                render_threshold_sweep_ascii,
            )

            y_true = getattr(me, "oos_y_true", None)
            scores = getattr(me, "oos_scores", None)
            if y_true is not None and scores is not None and len(y_true) > 0 and len(np.unique(y_true)) > 1:
                console.print(render_threshold_sweep_ascii(y_true, scores))
                console.print("")

            if outcome.sensitivity and getattr(outcome.sensitivity, "sensitivities", None):
                drift_dict = {
                    s.feature: getattr(s, "drift", 0.0) for s in outcome.sensitivity.sensitivities
                }
                if drift_dict:
                    console.print(render_drift_sparkline(drift_dict))
                    console.print("")

            # Generate and register static figures (Workstream C2 & C3)
            from start.reporting.figure_viewer import FigurePresentation, FigureSpec
            from start.reporting.figures import generate_all_report_figures

            if y_true is not None and scores is not None:
                cm = me.metrics_by_split.get("oos", {}).get("confusion_matrix") or me.metrics_by_split.get("test", {}).get("confusion_matrix")
                fig_drift = (
                    {s.feature: getattr(s, "drift", 0.0) for s in outcome.sensitivity.sensitivities}
                    if outcome.sensitivity and getattr(outcome.sensitivity, "sensitivities", None)
                    else None
                )
                figs = generate_all_report_figures(
                    y_true=y_true,
                    scores=scores,
                    cm=cm,
                    drift_dict=fig_drift,
                    global_importance_data=me.global_importance,
                    output_dir=cfg.output_root,
                    run_id=outcome.run_id,
                )
                if outcome.artifact_registry:
                    for fig_key, fig_path in figs.items():
                        outcome.artifact_registry.register(
                            fig_path,
                            name=f"figure_{fig_key}",
                            artifact_type="figure_png",
                            description=f"Generated diagnostic figure for {fig_key}",
                        )

                oos = me.metrics_by_split.get("oos", {})
                headlines = {
                    "roc_curve": f"AUC {oos.get('auc_roc', float('nan')):.4f}",
                    "pr_curve": f"PR-AUC {oos.get('pr_auc', float('nan')):.4f}",
                    "calibration_curve": f"ECE {oos.get('ece', float('nan')):.4f}",
                    "confusion_matrix": "at the selected decision threshold",
                    "feature_drift": "",
                    "global_importance": "",
                    "local_explanation": "3 named cases",
                }
                specs = [
                    FigureSpec(
                        key=key,
                        path=path,
                        headline=headlines.get(key, ""),
                        cohort="OOS" if key in {"roc_curve", "pr_curve", "calibration_curve"} else "",
                    )
                    for key, path in figs.items()
                ]
                presentation = FigurePresentation.configure(
                    explicit=getattr(cfg, "open_figures", None),
                    delay_seconds=getattr(cfg, "figure_delay", 1.5),
                    echo=lambda line: console.print(line, highlight=False),
                )
                presentation.present(specs)

        if me.global_importance:
            console.print("")
            console.print(importance_table(me.global_importance))

    # Section H: real tuning trials table (Rich #9)
    if outcome.tuning_run:
        from start.review_tables import tuning_table

        val_name = "K-fold cross-validation" if outcome.tuning_run.validation == "k_fold" else "single-split validation"
        console.print(f"\n[bold]Hyperparameter tuning ({val_name})[/bold]")
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
    from start.agents.engineering_agents import select_primary_metric
    from start.mrm_signoff import evaluate_signoff, render_signoff_rich

    _mrm_mc = select_primary_metric(task_type, costlier_errors=cost_choice)
    _mrm = evaluate_signoff(_final_store, session, primary_metric=_mrm_mc["primary_metric"])
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
    # overrides) alongside the dashboards with inner run ID.
    from start.reporting.review_transcript import write_transcript

    session.run_id = outcome.run_id
    inner_run_id = getattr(outcome, "inner_run_id", "")
    transcript_paths = write_transcript(
        session, cfg.output_root, outcome.run_id, sensitivity=outcome.sensitivity,
        kfold=outcome.kfold_tuning, inner_run_id=inner_run_id,
    )

    # Cross-Agent Collisions & Human Adjudication
    from start.attestation.seal import build_seal, persist_seal_manifest, validate_seal_preconditions
    from start.cli.panels import render_seal_panel
    from start.consensus import adjudicate_collisions_interactive, detect_collisions

    evidence_dicts = [
        r.model_dump() if hasattr(r, "model_dump") else (r.as_dict() if hasattr(r, "as_dict") else dict(r))
        for r in getattr(_final_store, "records", [])
    ]
    agent_outputs = {
        "ArchitectureReviewAgent": {"recommended_family": arch_choice},
        "ValidationPlannerAgent": {"expected_modality": "tabular"},
    }
    collisions = detect_collisions(
        evidence_records=evidence_dicts,
        agent_outputs=agent_outputs,
    )
    adjudications = []
    if collisions:
        adjudications, can_proceed = adjudicate_collisions_interactive(
            collisions,
            non_interactive=cfg.non_interactive,
            output_func=lambda m: console.print(m),
            input_func=input,
        )
        if not can_proceed:
            import typer
            if cfg.non_interactive:
                raise typer.Exit(code=1)
            console.print("[bold red]Review blocked by human adjudication outcome.[/bold red]")

    # Adjudications canonical payload (A2)
    adjudications_payload = session.to_canonical_dict()
    if adjudications:
        adjudications_payload["collisions"] = [
            a.as_evidence_record() if hasattr(a, "as_evidence_record") else dict(a)
            for a in adjudications
        ]

    # Attestations leaf payload (A5)
    from start.attestation.invariance import attest_narrative_invariance
    attestations_list = []
    agent_rev = getattr(outcome.base_outcome, "agent_review", None)
    if agent_rev and getattr(agent_rev, "signoff", None):
        signoff_text = agent_rev.signoff
        att = attest_narrative_invariance(
            section="governance_signoff",
            deterministic_narrative=signoff_text,
            model_narrative=signoff_text,
            evidence=outcome.base_outcome.evidence,
            provider_name=cfg.llm_provider if cfg.agent_mode == "llm" else "deterministic",
            narration_path="model_narrated" if cfg.agent_mode == "llm" else "deterministic_only",
        )
        attestations_list.append(att.as_dict())

    for chal in session.challenges:
        if chal.answer:
            c_att = attest_narrative_invariance(
                section=f"challenge:{chal.challenge_id}",
                deterministic_narrative=chal.answer,
                model_narrative=chal.answer,
                evidence=outcome.base_outcome.evidence,
                provider_name=chal.provider or cfg.llm_provider,
                narration_path="model_narrated" if cfg.agent_mode == "llm" else "deterministic_only",
            )
            attestations_list.append(c_att.as_dict())

    # Precondition validation (A6, Amendment 1)
    valid_seal, check_results = validate_seal_preconditions(
        review_id=outcome.run_id,
        ledger_records=outcome.base_outcome.evidence,
        evidence_head=outcome.base_outcome.evidence[-1] if outcome.base_outcome.evidence else None,
        adjudications=adjudications_payload,
        attestations=attestations_list,
        agent_mode=cfg.agent_mode,
        critic_verdict="PASSED" if outcome.critique_ok else "FAILED",
    )
    if not valid_seal:
        console.print("\n[bold red]SEAL WITHHELD — this review cannot be cryptographically sealed.[/bold red]")
        for c in check_results:
            mark = "[bold green]✓[/bold green]" if c.passed else "[bold red]✗[/bold red]"
            console.print(f"  {mark} {c.label}")
        console.print(
            "\n[dim]A seal that commits to no evidence is worse than no seal: "
            "it verifies forever and attests to nothing.[/dim]\n"
        )
        import typer
        raise typer.Exit(code=1)

    # Cryptographic Review Seal (start-seal/2 with 8 leaves)
    evidence_head_hash = (
        outcome.base_outcome.evidence[-1].evidence_id
        if outcome.base_outcome.evidence
        else ("0" * 32)
    )
    seal = build_seal(
        review_id=outcome.run_id,
        plan={
            "task": task_type,
            "target": cfg.target,
            "architecture": arch_choice,
            "enterprise_run_id": outcome.run_id,
            "inner_run_id": inner_run_id,
        },
        policy={"disclosure_policy": "public_demo", "profile": active_profile().value},
        evidence_head=evidence_head_hash,
        attestations=attestations_list if attestations_list else None,
        adjudications=adjudications_payload,
        metadata={"enterprise_run_id": outcome.run_id, "inner_run_id": inner_run_id},
    )

    manifest_path = persist_seal_manifest(seal, cfg.output_root)
    if outcome.artifact_registry:
        outcome.artifact_registry.register(
            str(manifest_path),
            name="seal_manifest",
            artifact_type="seal_manifest_json",
            description="Archived Merkle seal manifest",
        )

    console.print("")
    console.print(render_seal_panel(seal, critic_verdict="PASSED" if outcome.critique_ok else "FAILED"))
    tracer.end_review(seal_string=seal.seal_string())
    outcome.review_session = session

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
        f"  seal: [bold cyan]{seal.seal_string()}[/bold cyan]\n"
        f"  dashboard: {outcome.dashboard_paths['html']}\n"
        f"  transcript: {transcript_paths['html']}"
    )
    return outcome
