"""StART command-line interface.

Commands:
  start init    Scaffold configs and output directories in the current folder.
  start plan    Show the validation plan for a config (no execution).
  start run     Execute a full review run against a dataset.
  start report  Render the Markdown report for the latest (or given) run.
  start doctor  Diagnose the environment: device, runtimes, providers.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from start.core.config import StartConfig, load_config, load_policy
from start.core.schemas import DatasetSummary, Materiality, ModelMetadata, TaskType

app = typer.Typer(name="start", help="StART: Standardized Agentic Reusable Tests.", no_args_is_help=True)
console = Console()

DEFAULT_CONFIG = "configs/default.yaml"


@app.command()
def init(path: str = typer.Option(".", help="Project root to initialize.")) -> None:
    """Create config templates and output directories."""
    root = Path(path)
    (root / "configs" / "policy").mkdir(parents=True, exist_ok=True)
    (root / "start_output").mkdir(exist_ok=True)
    cfg_path = root / "configs" / "default.yaml"
    pol_path = root / "configs" / "policy" / "default_policy.yaml"
    if not cfg_path.exists():
        import yaml

        cfg_path.write_text(yaml.safe_dump(StartConfig().model_dump(), sort_keys=False))
        console.print(f"[green]Wrote[/green] {cfg_path}")
    if not pol_path.exists():
        pol_path.write_text(
            "name: default\nversion: 0.1.0\nallowed_task_types: []\n"
            "allowed_data_roots: []\nmax_materiality_without_review: high\n"
            "require_citations: true\nthresholds: {}\n"
        )
        console.print(f"[green]Wrote[/green] {pol_path}")
    console.print("[bold]StART project initialized.[/bold] Run `start doctor` next.")


@app.command()
def plan(config: str = typer.Option(DEFAULT_CONFIG, help="Path to YAML config.")) -> None:
    """Show the rule-based validation plan without executing anything."""
    from start.agents import ReviewPlannerAgent
    from start.providers.llm import get_llm_provider

    cfg = load_config(config)
    meta = ModelMetadata(
        model_id=cfg.model.model_id,
        task_type=TaskType(cfg.model.task_type),
        materiality=Materiality(cfg.model.materiality),
    )
    dataset = DatasetSummary(dataset_id=cfg.data.dataset_id, source=cfg.data.path)
    validation_plan = ReviewPlannerAgent(cfg, get_llm_provider(cfg.llm)).plan(meta, dataset)
    table = Table(title=f"Validation plan {validation_plan.plan_id}")
    table.add_column("Test ID")
    table.add_column("Reason")
    for item in validation_plan.planned_tests:
        table.add_row(item.test_id, item.reason)
    console.print(table)


@app.command()
def run(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to YAML config."),
    train: str = typer.Argument(..., help="Train dataset file (csv/parquet)."),
    test: str = typer.Option(None, help="Holdout dataset file (csv/parquet/feather)."),
    oos: str = typer.Option(None, help="Out-of-sample dataset file (csv/parquet/feather)."),
) -> None:
    """Execute a full review run and append evidence to the ledger."""
    from start.connectors import load_local_file
    from start.orchestration.pipeline import build_context, run_review

    cfg = load_config(config)
    train_df = load_local_file(train)
    test_df = load_local_file(test) if test else None
    extra = {"oos": load_local_file(oos)} if oos else None
    result = run_review(cfg, build_context(cfg, train_df, test_df, extra=extra))

    out_dir = Path(cfg.output.root) / cfg.output.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    from start.reporting import render_markdown

    report_path = out_dir / f"{result.run_id}.md"
    report_path.write_text(render_markdown(result))
    (out_dir / f"{result.run_id}.json").write_text(result.model_dump_json(indent=2))

    table = Table(title=f"Run {result.run_id}")
    table.add_column("Evidence ID")
    table.add_column("Test")
    table.add_column("Status")
    for rec in result.evidence:
        color = {"pass": "green", "warn": "yellow", "fail": "red", "error": "red"}.get(
            rec.status.value, "white"
        )
        table.add_row(rec.evidence_id, rec.test_name, f"[{color}]{rec.status.value}[/{color}]")
    console.print(table)
    console.print(f"Report: [bold]{report_path}[/bold]")


@app.command()
def report(
    config: str = typer.Option(DEFAULT_CONFIG, help="Path to YAML config."),
    run_id: str = typer.Option(None, help="Run ID; defaults to latest report."),
) -> None:
    """Print a previously generated report."""
    cfg = load_config(config)
    reports = sorted((Path(cfg.output.root) / cfg.output.reports_dir).glob("RUN-*.md"))
    if not reports:
        console.print("[red]No reports found. Run `start run` first.[/red]")
        raise typer.Exit(1)
    target = next((p for p in reports if run_id and run_id in p.name), reports[-1])
    console.print(target.read_text())


@app.command()
def doctor(config: str = typer.Option(None, help="Optional YAML config to validate.")) -> None:
    """Diagnose environment: devices, runtimes, providers, ledger integrity."""
    from start.providers.compute import detect_device, is_databricks_runtime, mlflow_available
    from start.providers.llm import _PROVIDERS
    from start.registry import list_families, list_tests

    table = Table(title="start doctor")
    table.add_column("Check")
    table.add_column("Result")
    table.add_row("Detected device (CUDA→MPS→CPU)", detect_device().value)
    table.add_row("Databricks runtime", str(is_databricks_runtime()))
    table.add_row("MLFlow importable", str(mlflow_available()))
    for name, cls in _PROVIDERS.items():
        try:
            table.add_row(f"LLM provider '{name}' available", str(cls().available))
        except Exception:
            table.add_row(f"LLM provider '{name}' available", "False")
    table.add_row("Registered test families", ", ".join(list_families()))
    table.add_row("Registered tests", str(len(list_tests())))
    if config:
        cfg = load_config(config)
        policy = load_policy(cfg.policy_file)
        table.add_row("Config valid", "True")
        table.add_row("Policy hash", policy.content_hash()[:16] + "…")
        ledger_path = Path(cfg.output.root) / cfg.output.ledger_file
        if ledger_path.exists():
            from start.evidence.ledger import EvidenceLedger

            ledger = EvidenceLedger(ledger_path, Path(cfg.output.root) / cfg.output.evidence_store)
            table.add_row("Ledger integrity", str(ledger.verify()))
    console.print(table)


@app.command("list-tests")
def list_tests_cmd(family: str = typer.Option(None, help="Filter by family.")) -> None:
    """List registered deterministic tests."""
    from start.registry import list_tests

    payload = [
        {"test_id": s.test_id, "family": s.family, "name": s.name, "description": s.description}
        for s in list_tests(family)
    ]
    print(json.dumps(payload, indent=2))  # plain stdout: pipeable JSON


if __name__ == "__main__":
    app()


@app.command("propensity-demo")
def propensity_demo(
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="Run with safe defaults, no prompts."
    ),
    model: str = typer.Option("random_forest", help="random_forest | xgboost | lightgbm."),
    tuning: str = typer.Option("none", help="none | grid | random | optuna."),
    cv: int = typer.Option(None, help="K for K-fold CV (3 or 5); omit for holdout."),
    cohort: str = typer.Option("test", help="Sensitivity cohort: test | oos | development."),
    seed: int = typer.Option(42, help="Random seed."),
    output_root: str = typer.Option("start_output", help="Output directory root."),
    train: str = typer.Option(None, help="Your train data (csv/parquet/feather/Delta dir)."),
    test: str = typer.Option(None, help="Your test data; omit to auto-split train."),
    oos: str = typer.Option(None, help="Your out-of-sample data; omit to auto-split train."),
    target: str = typer.Option(None, help="Target column name in your data."),
    agent_mode: str = typer.Option(
        "deterministic", "--agent-mode", help="deterministic | llm (evidence-grounded)."
    ),
    llm_provider: str = typer.Option(
        None, help="Agent LLM provider override (none | openai | anthropic | ... )."
    ),
) -> None:
    """Propensity-style model review demo: 60/20/20 split, feature checks,
    model + tuning choice, train/test/OOS metrics, explainability with honest
    SHAP fallback, top-feature shock sensitivity, full evidence pipeline."""
    from start.modeling.data import TARGET_COLUMN
    from start.modeling.propensity import PropensityOptions, prompt_options, run_propensity_demo

    opts = PropensityOptions(
        model=model,
        tuning=tuning,
        cv_folds=cv,
        sensitivity_cohort=cohort,
        seed=seed,
        output_root=output_root,
        train_path=train,
        test_path=test,
        oos_path=oos,
        target_column=target or TARGET_COLUMN,
        agent_mode=agent_mode,
        llm_provider=llm_provider or "",
    )
    if not non_interactive:
        opts = prompt_options(initial=opts)
    run_propensity_demo(opts)


@app.command()
def recommend(
    train: str = typer.Argument(..., help="Dataset file (csv/parquet/feather/Delta dir)."),
    target: str = typer.Option(None, help="Target column name."),
    timestamp_col: str = typer.Option(None, help="Timestamp column, if time-indexed."),
    entity_col: str = typer.Option(None, help="Entity/ID column, if panel-structured."),
    dataset_type: str = typer.Option(
        "auto",
        help="Declare a domain type (limit_order_book | tick_events | volatility_surface | "
        "panel_time_series | time_series | text_alternative | tabular); 'auto' infers.",
    ),
) -> None:
    """Profile a dataset, recommend candidate models, and produce a
    model/dataset-specific validation plan (available-now vs roadmap)."""
    from start.agents import ModelRecommendationAgent, ValidationPlannerAgent
    from start.connectors import load_local_file
    from start.taxonomy import profile_dataset

    df = load_local_file(train)
    profile = profile_dataset(
        df,
        target_column=target,
        timestamp_column=timestamp_col,
        entity_id_column=entity_col,
        declared_type=dataset_type,
    )
    console.print("\n[bold]Dataset profile[/bold]")
    console.print(f"  {profile.describe()}")

    console.print("\n[bold]Model recommendations[/bold]")
    for line in ModelRecommendationAgent().recommend(profile):
        console.print(f"  {line}")

    plan = ValidationPlannerAgent().plan_for(profile)
    console.print("\n[bold]Validation plan[/bold]")
    console.print(f"  model family: {plan['model_family']} | dataset type: {plan['dataset_type']}")
    console.print("  available now:")
    for item in plan["available_now"] or ["  (none registered for this type yet)"]:
        console.print(f"    - {item}")
    console.print("  roadmap:")
    for item in plan["roadmap"] or ["  (none)"]:
        console.print(f"    - {item}")
    console.print(
        "\n  explainability route — implemented: "
        f"{', '.join(plan['explainability']['implemented']) or 'none'}; "
        f"roadmap: {', '.join(plan['explainability']['roadmap']) or 'none'}"
    )


def _resolve_agent_llm(
    llm_provider: str | None,
    cfg,
    *,
    agent_mode: str = "deterministic",
    prompt_for_key: bool | None = None,
    allow_fallback: bool = False,
) -> tuple[str, object, bool]:
    """Resolve the agent-layer LLM provider from CLI flag > config.

    In llm mode, makes sure the provider's API key is available for this
    session: read from the environment, or — on an interactive terminal, or
    when --prompt-for-key is passed — collected via a hidden getpass prompt.
    The key is set for the current process only and never echoed, logged, or
    persisted. Missing key with prompting disabled fails clearly unless
    --allow-deterministic-fallback was given; never a silent fallback."""
    from start.core.config import LLMConfig
    from start.providers.keys import ensure_provider_key, key_required
    from start.providers.llm import get_llm_provider

    name = llm_provider or cfg.agent.llm_provider or cfg.llm.provider or "none"
    if agent_mode == "llm" and key_required(name):
        status = ensure_provider_key(name, prompt_for_key=prompt_for_key)
        if not status.ok:
            if allow_fallback:
                console.print(
                    f"[yellow]WARNING: {status.env_var} is not set and prompting is "
                    f"disabled; falling back to deterministic mode as requested.[/yellow]"
                )
                return "none", get_llm_provider(LLMConfig(provider="none")), True
            console.print(
                f"[red]Missing {status.env_var} for provider '{name}'. Set the "
                f"environment variable, re-run with --prompt-for-key to enter it "
                f"securely, or pass --allow-deterministic-fallback.[/red]"
            )
            raise typer.Exit(code=1)
    return name, get_llm_provider(LLMConfig(provider=name, model=cfg.llm.model)), False


def _load_review(
    config: str,
    agent_mode: str,
    llm_provider: str | None,
    run_id: str,
    evidence_dir: str,
    ledger: str,
    prompt_for_key: bool | None = None,
    allow_fallback: bool = False,
):
    from start.agents.review import load_run_records, run_agent_review

    cfg = load_config(config)
    root = evidence_dir or cfg.output.root
    resolved_run, records = load_run_records(root, ledger, run_id)
    name, llm, fell_back = _resolve_agent_llm(
        llm_provider,
        cfg,
        agent_mode=agent_mode,
        prompt_for_key=prompt_for_key,
        allow_fallback=allow_fallback,
    )
    if fell_back:
        agent_mode = "deterministic"  # CLI already printed the explicit warning
    review = run_agent_review(
        records,
        mode=agent_mode,
        llm=llm,
        policy_hash=records[0].policy_hash if records else None,
    )
    return resolved_run, records, review


def _print_review_sections(run_id: str, review, sections: tuple[str, ...]) -> None:
    console.print(
        f"\n[bold]Agent review — run {run_id}[/bold] | mode: {review.mode}"
        + (f" | provider: {review.llm_provider}" if review.mode == "llm" else "")
    )
    for note in review.notes:
        console.print(f"  [yellow]{note}[/yellow]")
    titles = {
        "review_plan": "Review plan",
        "suggested_tests": "Suggested next tests",
        "findings": "Model-risk findings",
        "challenge_memo": "Challenge memo",
        "missing_evidence": "Missing evidence",
        "governance": "Governance assessment",
    }
    for key in sections:
        if key == "signoff":
            console.print("\n[bold]Sign-off recommendation[/bold]")
            console.print(f"  {review.signoff}")
            continue
        items = getattr(review, key)
        console.print(f"\n[bold]{titles[key]}[/bold]")
        for item in items:
            console.print(f"  - {item}")
    console.print(
        f"\nEvidence critique status: "
        f"{'[green]PASSED[/green]' if review.critique_ok else '[red]FAILED[/red]'}"
    )


_AGENT_OPTS = {
    "config": typer.Option(DEFAULT_CONFIG, help="Path to YAML config."),
    "agent_mode": typer.Option("deterministic", help="deterministic | llm."),
    "llm_provider": typer.Option(
        None, help="none | openai | anthropic | grok | huggingface | hf_local | enterprise_llm_gateway."
    ),
    "run_id": typer.Option("latest", help="RUN-... id from the ledger, or 'latest'."),
    "evidence_dir": typer.Option("", help="Output root holding the ledger (default: config output root)."),
    "ledger": typer.Option("ledger.jsonl", help="Ledger filename."),
}


@app.command("agent-review")
def agent_review_cmd(
    config: str = _AGENT_OPTS["config"],
    prompt_for_key: bool = typer.Option(
        None,
        "--prompt-for-key/--no-prompt-for-key",
        help="Securely prompt for a missing API key (default: only on interactive terminals).",
    ),
    allow_fallback: bool = typer.Option(
        False, "--allow-deterministic-fallback", help="Fall back instead of failing on a missing key."
    ),
    agent_mode: str = typer.Option("deterministic", "--agent-mode", help="deterministic | llm."),
    llm_provider: str = _AGENT_OPTS["llm_provider"],
    run_id: str = _AGENT_OPTS["run_id"],
    evidence_dir: str = _AGENT_OPTS["evidence_dir"],
    ledger: str = _AGENT_OPTS["ledger"],
) -> None:
    """Full dual-mode agent review over stored evidence: plan, suggestions,
    findings, challenge memo, missing evidence, governance, and sign-off."""
    resolved_run, _, review = _load_review(
        config, agent_mode, llm_provider, run_id, evidence_dir, ledger, prompt_for_key, allow_fallback
    )
    _print_review_sections(
        resolved_run,
        review,
        (
            "review_plan",
            "suggested_tests",
            "findings",
            "challenge_memo",
            "missing_evidence",
            "governance",
            "signoff",
        ),
    )


@app.command("review-plan")
def review_plan_cmd(
    config: str = _AGENT_OPTS["config"],
    prompt_for_key: bool = typer.Option(
        None,
        "--prompt-for-key/--no-prompt-for-key",
        help="Securely prompt for a missing API key (default: only on interactive terminals).",
    ),
    allow_fallback: bool = typer.Option(
        False, "--allow-deterministic-fallback", help="Fall back instead of failing on a missing key."
    ),
    agent_mode: str = typer.Option("deterministic", "--agent-mode"),
    llm_provider: str = _AGENT_OPTS["llm_provider"],
    run_id: str = _AGENT_OPTS["run_id"],
    evidence_dir: str = _AGENT_OPTS["evidence_dir"],
    ledger: str = _AGENT_OPTS["ledger"],
) -> None:
    """Review plan for a stored run."""
    resolved_run, _, review = _load_review(
        config, agent_mode, llm_provider, run_id, evidence_dir, ledger, prompt_for_key, allow_fallback
    )
    _print_review_sections(resolved_run, review, ("review_plan",))


@app.command("suggest-tests")
def suggest_tests_cmd(
    config: str = _AGENT_OPTS["config"],
    prompt_for_key: bool = typer.Option(
        None,
        "--prompt-for-key/--no-prompt-for-key",
        help="Securely prompt for a missing API key (default: only on interactive terminals).",
    ),
    allow_fallback: bool = typer.Option(
        False, "--allow-deterministic-fallback", help="Fall back instead of failing on a missing key."
    ),
    agent_mode: str = typer.Option("deterministic", "--agent-mode"),
    llm_provider: str = _AGENT_OPTS["llm_provider"],
    run_id: str = _AGENT_OPTS["run_id"],
    evidence_dir: str = _AGENT_OPTS["evidence_dir"],
    ledger: str = _AGENT_OPTS["ledger"],
) -> None:
    """Suggested next validation tests for a stored run."""
    resolved_run, _, review = _load_review(
        config, agent_mode, llm_provider, run_id, evidence_dir, ledger, prompt_for_key, allow_fallback
    )
    _print_review_sections(resolved_run, review, ("suggested_tests", "missing_evidence"))


@app.command("challenge-findings")
def challenge_findings_cmd(
    config: str = _AGENT_OPTS["config"],
    prompt_for_key: bool = typer.Option(
        None,
        "--prompt-for-key/--no-prompt-for-key",
        help="Securely prompt for a missing API key (default: only on interactive terminals).",
    ),
    allow_fallback: bool = typer.Option(
        False, "--allow-deterministic-fallback", help="Fall back instead of failing on a missing key."
    ),
    agent_mode: str = typer.Option("deterministic", "--agent-mode"),
    llm_provider: str = _AGENT_OPTS["llm_provider"],
    run_id: str = _AGENT_OPTS["run_id"],
    evidence_dir: str = _AGENT_OPTS["evidence_dir"],
    ledger: str = _AGENT_OPTS["ledger"],
) -> None:
    """Adversarial challenge memo for a stored run."""
    resolved_run, _, review = _load_review(
        config, agent_mode, llm_provider, run_id, evidence_dir, ledger, prompt_for_key, allow_fallback
    )
    _print_review_sections(resolved_run, review, ("findings", "challenge_memo"))


@app.command("signoff")
def signoff_cmd(
    config: str = _AGENT_OPTS["config"],
    prompt_for_key: bool = typer.Option(
        None,
        "--prompt-for-key/--no-prompt-for-key",
        help="Securely prompt for a missing API key (default: only on interactive terminals).",
    ),
    allow_fallback: bool = typer.Option(
        False, "--allow-deterministic-fallback", help="Fall back instead of failing on a missing key."
    ),
    agent_mode: str = typer.Option("deterministic", "--agent-mode"),
    llm_provider: str = _AGENT_OPTS["llm_provider"],
    run_id: str = _AGENT_OPTS["run_id"],
    evidence_dir: str = _AGENT_OPTS["evidence_dir"],
    ledger: str = _AGENT_OPTS["ledger"],
) -> None:
    """Governance assessment and sign-off recommendation for a stored run."""
    resolved_run, _, review = _load_review(
        config, agent_mode, llm_provider, run_id, evidence_dir, ledger, prompt_for_key, allow_fallback
    )
    _print_review_sections(resolved_run, review, ("governance", "signoff"))


@app.command("llm-check")
def llm_check(
    llm_provider: str = typer.Option(..., "--llm-provider", help="Provider to verify."),
    prompt_for_key: bool = typer.Option(
        None,
        "--prompt-for-key/--no-prompt-for-key",
        help="Securely prompt for a missing API key (default: only on interactive terminals).",
    ),
) -> None:
    """Verify an LLM provider end to end: dependency installed, key available
    (or securely prompted), one synthetic-evidence test call, and the output
    checked against the evidence citation gate. No raw data is ever sent."""
    from start.providers.keys import (
        PROVIDER_KEY_ENV,
        dependency_available,
        ensure_provider_key,
        run_llm_check,
    )

    if llm_provider not in PROVIDER_KEY_ENV:
        console.print(
            f"[red]Unknown provider '{llm_provider}'. "
            f"Known: {', '.join(sorted(PROVIDER_KEY_ENV))}[/red]"
        )
        raise typer.Exit(code=1)

    dep_ok, dep_msg = dependency_available(llm_provider)
    console.print(f"Provider: {llm_provider}")
    console.print(f"Dependency: {dep_msg}")
    if not dep_ok:
        raise typer.Exit(code=1)

    if llm_provider == "none":
        console.print("Mode: deterministic (no key, no LLM — this is the default and is fully supported)")
        return
    if llm_provider == "enterprise_llm_gateway":
        console.print(
            "Mode: placeholder — enterprise_llm_gateway has no public implementation; "
            "provide a private one outside this repository."
        )
        return

    status = ensure_provider_key(llm_provider, prompt_for_key=prompt_for_key)
    console.print(f"Key source: {status.source}")
    if not status.ok and llm_provider != "hf_local":
        console.print(
            f"[red]Missing {status.env_var}; set it or re-run with --prompt-for-key.[/red]"
        )
        raise typer.Exit(code=1)

    result = run_llm_check(llm_provider)
    console.print(f"Mode: {result['mode']}")
    console.print(f"Synthetic evidence sent: {result['synthetic_evidence_sent']}")
    console.print(f"Raw dataset sent: {result['raw_dataset_sent']}")
    console.print(f"Critique: {result['critique']}")
    if result["critique"] == "failed":
        raise typer.Exit(code=1)
