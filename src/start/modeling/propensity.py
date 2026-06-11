"""Propensity-style model review workflow (the flagship demo).

A realistic client-attrition/propensity-style binary classification review on
public sklearn data: 60/20/20 stratified train/test/OOS split, feature
engineering checks, model choice (Random Forest default; XGBoost/LightGBM
optional), tuning choice (none/grid/random/Optuna), K-fold options,
train/test/OOS metrics comparison, explainability with honest SHAP fallback,
top-feature shock sensitivity, and the full StART evidence pipeline.

Used by both `examples/propensity_interactive.py` and `start propensity-demo`.
All quantitative results flow through registered deterministic engines into
evidence records; the console tables are rendered from those same functions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.table import Table

from start.core.config import StartConfig
from start.core.schemas import RunResult
from start.modeling.data import (
    SCORE_COLUMN,
    TARGET_COLUMN,
    feature_columns,
)
from start.modeling.metrics import METRIC_NAMES, cohort_comparison
from start.modeling.models import HYPERPARAM_SPACES, MODEL_CHOICES, resolve_model
from start.modeling.sensitivity import DEFAULT_SHOCKS
from start.modeling.tuning import TUNING_CHOICES, TuningOutcome, tune_model

console = Console()

COHORT_CHOICES = ("test", "oos", "development")


@dataclass
class PropensityOptions:
    model: str = "random_forest"
    tuning: str = "none"
    cv_folds: int | None = None  # None = holdout; 3 or 5 = K-fold
    sensitivity_cohort: str = "test"
    seed: int = 42
    output_root: str = "start_output"
    custom_space: dict[str, dict[str, Any]] | None = None
    notes: list[str] = field(default_factory=list)
    # Bring-your-own-data (demo dataset is used only when train_path is None):
    train_path: str | None = None
    test_path: str | None = None
    oos_path: str | None = None
    target_column: str = TARGET_COLUMN
    agent_mode: str = "deterministic"
    llm_provider: str = ""


# --------------------------------------------------------------------------- #
# Interactive prompting (injectable input function for testability)
# --------------------------------------------------------------------------- #
def _choose(
    prompt: str, choices: tuple[str, ...], default: str, ask: Callable[[str], str]
) -> str:
    labels = "/".join(c.upper() if c == default else c for c in choices)
    raw = ask(f"{prompt} [{labels}] (default: {default}): ").strip().lower()
    return raw if raw in choices else default


def _parse_values(raw: str, kind: str) -> list[Any] | None:
    raw = raw.strip()
    if not raw:
        return None
    items: list[Any] = []
    for token in raw.split(","):
        token = token.strip()
        if token.lower() in {"none", "null"}:
            items.append(None)
        elif kind == "int":
            items.append(int(token))
        elif kind == "float":
            items.append(float(token))
        else:
            items.append(token)
    return items


def prompt_options(
    initial: PropensityOptions | None = None, ask: Callable[[str], str] = input
) -> PropensityOptions:
    """Interactive flow: model -> tuning -> (search space) -> CV -> cohort."""
    opts = initial or PropensityOptions()

    console.print("\n[bold]StART propensity model review — interactive setup[/bold]\n")
    opts.model = _choose("Model", MODEL_CHOICES, opts.model, ask)
    opts.tuning = _choose("Hyperparameter tuning", TUNING_CHOICES, opts.tuning, ask)

    if opts.tuning != "none":
        space_key = opts.model if opts.model in HYPERPARAM_SPACES else "random_forest"
        space = {k: dict(v) for k, v in HYPERPARAM_SPACES[space_key].items()}
        console.print("\nFive standard hyperparameters and suggested values (press Enter to accept):")
        for param, spec in space.items():
            if opts.tuning == "grid":
                hint = f"grid values e.g. {spec['grid']}"
                raw = ask(f"  {param} — {hint}. Custom comma-separated values or Enter: ")
                custom = _parse_values(raw, spec["type"])
                if custom:
                    spec["grid"] = custom
            else:  # random / optuna share range-style input
                if spec["type"] == "cat":
                    hint = f"choices e.g. {spec['choices']}"
                    raw = ask(f"  {param} — {hint}. Custom comma-separated choices or Enter: ")
                    custom = _parse_values(raw, "cat")
                    if custom:
                        spec["choices"] = custom
                else:
                    hint = f"range e.g. {spec['low']}..{spec['high']}"
                    raw = ask(f"  {param} — {hint}. Custom 'low,high' or Enter: ")
                    custom = _parse_values(raw, spec["type"])
                    if custom and len(custom) >= 2:
                        spec["low"], spec["high"] = custom[0], custom[1]
        opts.custom_space = space

        cv_mode = _choose("Validation scheme", ("holdout", "kfold"), "kfold", ask)
        if cv_mode == "kfold":
            k = _choose("K for K-fold CV", ("3", "5"), "3", ask)
            opts.cv_folds = int(k)
        else:
            opts.cv_folds = None

    opts.sensitivity_cohort = _choose(
        "Sensitivity evaluation cohort", COHORT_CHOICES, opts.sensitivity_cohort, ask
    )
    return opts


# --------------------------------------------------------------------------- #
# Workflow
# --------------------------------------------------------------------------- #
def _build_config(opts: PropensityOptions, model_name: str) -> StartConfig:
    config = StartConfig()
    config.project_name = "start-propensity-demo"
    config.seed = opts.seed
    config.output.root = opts.output_root
    config.data.dataset_id = "sklearn-breast-cancer-as-attrition"
    config.model.model_id = f"attrition-propensity-{model_name}"
    config.model.task_type = "binary_classification"
    config.model.materiality = "medium"
    config.model.target_column = TARGET_COLUMN
    config.model.score_column = SCORE_COLUMN
    config.test_families.enabled = ["preprocessing", "supervised", "xai"]
    config.agent.mode = opts.agent_mode  # type: ignore[assignment]
    config.agent.llm_provider = opts.llm_provider  # type: ignore[assignment]
    config.test_families.overrides = {
        "xai.feature_sensitivity": {"cohort": opts.sensitivity_cohort, "top_k": 5},
    }
    return config


def run_propensity_demo(opts: PropensityOptions) -> RunResult:
    from start.connectors import DemoConnector, LocalFileConnector
    from start.orchestration.pipeline import build_context, run_review

    target = opts.target_column
    if opts.train_path:
        console.print("\n[bold]1/5 Data[/bold] — loading user data via the files connector")
        connector = LocalFileConnector(
            opts.train_path, opts.test_path, opts.oos_path,
            seed=opts.seed, target_column=target,
        )
    else:
        console.print(
            "\n[bold]1/5 Data[/bold] — loading public demo dataset "
            "(bring your own with --train/--test/--oos)"
        )
        connector = DemoConnector(seed=opts.seed, target_column=target)
    bundle = connector.load_bundle()
    train, test, oos = bundle.train, bundle.test, bundle.oos
    if target not in train.columns:
        raise ValueError(
            f"Target column '{target}' not found in the data; pass --target to set it."
        )
    for note in bundle.notes:
        console.print(f"    [dim]{note}[/dim]")
    features = feature_columns(train, target)
    console.print(
        f"    source: {bundle.source} | rows: train={len(train)}, test={len(test)}, "
        f"oos={len(oos)}; features={len(features)}; event rate={train[target].mean():.3f}"
    )

    console.print("[bold]2/5 Model[/bold] — resolving choice and tuning")
    estimator, model_name, note = resolve_model(opts.model, opts.seed)
    if note:
        console.print(f"    [yellow]{note}[/yellow]")
        opts.notes.append(note)

    space = opts.custom_space or HYPERPARAM_SPACES.get(model_name)
    estimator, tuning = tune_model(
        estimator,
        train[features],
        train[target],
        method=opts.tuning,
        space=space if opts.tuning != "none" else None,
        cv_folds=opts.cv_folds,
        seed=opts.seed,
    )
    if tuning.note:
        console.print(f"    [yellow]{tuning.note}[/yellow]")
        opts.notes.append(tuning.note)
    _print_tuning(model_name, tuning)

    console.print("[bold]3/5 Fit & score[/bold] — fitting on train, scoring all cohorts")
    estimator.fit(train[features], train[target])
    cohorts = {}
    for name, frame in (("train", train), ("test", test), ("oos", oos)):
        frame = frame.copy()
        frame[SCORE_COLUMN] = estimator.predict_proba(frame[features])[:, 1]
        cohorts[name] = frame
    _print_metrics_table(cohorts, target)

    console.print("[bold]4/5 Evidence pipeline[/bold] — feature checks, metrics, XAI, sensitivity")
    config = _build_config(opts, model_name)
    config.model.target_column = target
    config.data.dataset_id = bundle.source
    config.data.source = "files" if opts.train_path else "demo"
    ctx = build_context(
        config,
        cohorts["train"],
        cohorts["test"],
        model=estimator,
        extra={
            "oos": cohorts["oos"],
            "demo_meta": {
                "model": model_name,
                "tuning_method": tuning.method,
                "cv_folds": tuning.cv_folds,
                "best_params": tuning.best_params,
                "sensitivity_cohort": opts.sensitivity_cohort,
            },
        },
    )
    result = run_review(config, ctx)
    _print_evidence(result)
    _print_sensitivity(result)

    console.print("[bold]5/5 Report[/bold]")
    from pathlib import Path

    from start.reporting import render_markdown

    out_dir = Path(config.output.root) / config.output.reports_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{result.run_id}.md"
    report_path.write_text(render_markdown(result))
    console.print(f"    report: {report_path}")
    console.print(f"    critique OK: {result.critique.ok if result.critique else 'n/a'}")
    if result.agent_review is not None:
        ar = result.agent_review
        mode_label = f"llm-assisted ({ar.llm_provider})" if ar.mode == "llm" else "deterministic"
        review_status = 'OK' if ar.critique_ok else 'FAILED'
        console.print(f"    agent mode: {mode_label} | review critique: {review_status}")
        for note in ar.notes:
            console.print(f"    [yellow]{note}[/yellow]")
        console.print(f"    sign-off: {ar.signoff.split('.')[0]}.")
    return result


# --------------------------------------------------------------------------- #
# Console rendering (numbers come from the same deterministic functions that
# feed the evidence records)
# --------------------------------------------------------------------------- #
def _print_tuning(model_name: str, tuning: TuningOutcome) -> None:
    if tuning.method == "none":
        console.print(f"    model: {model_name} (default hyperparameters, no tuning)")
        return
    console.print(
        f"    model: {model_name} | tuning: {tuning.method} | candidates: {tuning.n_candidates} "
        f"| CV: {'holdout' if not tuning.cv_folds else f'{tuning.cv_folds}-fold'} "
        f"| best CV AUC: {tuning.best_cv_auc}"
    )
    if tuning.best_params:
        console.print(f"    best params: {tuning.best_params}")


def _print_metrics_table(cohorts: dict[str, Any], target: str = TARGET_COLUMN) -> None:
    comparison = cohort_comparison(
        {
            name: (frame[target].to_numpy(), frame[SCORE_COLUMN].to_numpy())
            for name, frame in cohorts.items()
        }
    )
    table = Table(title="Cohort metrics comparison")
    table.add_column("Cohort")
    pretty = {
        "auc_roc": "AUC-ROC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "top_decile_lift": "Top 10% Lift",
    }
    for metric in METRIC_NAMES:
        table.add_column(pretty[metric], justify="right")
    for name in ("train", "test", "oos"):
        table.add_row(name, *[f"{comparison[name][m]:.4f}" for m in METRIC_NAMES])
    console.print(table)


def _print_evidence(result: RunResult) -> None:
    table = Table(title=f"Evidence — run {result.run_id}")
    table.add_column("Evidence ID")
    table.add_column("Test")
    table.add_column("Status")
    for rec in result.evidence:
        color = {"pass": "green", "warn": "yellow", "fail": "red", "error": "red"}.get(
            rec.status.value, "white"
        )
        table.add_row(rec.evidence_id, rec.test_name, f"[{color}]{rec.status.value}[/{color}]")
    console.print(table)


def _print_sensitivity(result: RunResult) -> None:
    rec = next((r for r in result.evidence if r.test_id == "xai.feature_sensitivity"), None)
    if rec is None or rec.status.value in {"skipped", "error"}:
        return
    table = Table(
        title=(
            f"Sensitivity — parallel shocks to top 5 features "
            f"(cohort: {rec.metrics.get('cohort')}, ranking: {rec.metrics.get('importance_method')})"
        )
    )
    table.add_column("Shock", justify="right")
    table.add_column("AUC-ROC", justify="right")
    table.add_column("AUC drift vs baseline", justify="right")
    for shock in DEFAULT_SHOCKS:
        label = f"{int(shock * 100):+d}pct"
        auc = rec.metrics.get(f"auc_{label}")
        drift = rec.metrics.get(f"drift_{label}")
        if auc is None:
            continue
        table.add_row(f"{int(shock * 100):+d}%", f"{auc:.4f}", f"{drift:+.4f}")
    console.print(table)
    console.print(f"    shocked features: {rec.metrics.get('shocked_features')} [{rec.evidence_id}]")
