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
