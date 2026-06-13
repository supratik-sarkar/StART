"""Deep-learning model review workflow — the reusable orchestration layer.

Mirrors the propensity workflow's rigor and UX for deep learning, callable
identically from a terminal script, a Jupyter notebook, or a Databricks
notebook. Produces the full evidence set (EV-DL-0001..0007), figures, the
dual-mode agent review, and a proof-carrying Markdown report.

    data -> build model -> train -> evaluate -> evidence pipeline
         -> agent review -> figures -> report

The LLM (when enabled) reasons only over the evidence bundle and never sees
raw data. Default mode is deterministic and requires no key.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from start.core.schemas import EvidenceRecord, Status, TestResult
from start.modeling.dl_data import (
    SCORE_COLUMN,
    TARGET_COLUMN,
    feature_columns,
    load_dl_bundle,
)
from start.modeling.dl_metrics import DL_METRIC_NAMES, compute_dl_cohort_metrics

console = Console()

DL_ARCHITECTURES = ("mlp", "leaky_relu_mlp", "residual_mlp", "wide_deep")
COHORT_CHOICES = ("test", "oos", "development")

# Stable, human-readable evidence labels (carried alongside the ledger's
# content-addressed hash IDs).
DL_EVIDENCE_LABELS = {
    "deep_learning.training_diagnostics": "EV-DL-0001",
    "deep_learning.performance_diagnostics": "EV-DL-0002",
    "deep_learning.calibration_diagnostics": "EV-DL-0003",
    "deep_learning.explainability_diagnostics": "EV-DL-0004",
    "deep_learning.sensitivity_diagnostics": "EV-DL-0005",
    "deep_learning.robustness_diagnostics": "EV-DL-0006",
    "deep_learning.figure_diagnostics": "EV-DL-0007",
}


@dataclass
class DLReviewOptions:
    architecture: str = "mlp"
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 1e-3
    dropout: float = 0.1
    sensitivity_cohort: str = "test"
    explain_method: str = "integrated_gradients"  # or gradient_shap
    agent_mode: str = "deterministic"
    llm_provider: str = ""
    seed: int = 42
    output_root: str = "start_output"
    # bring-your-own-data
    data_source: str = "demo"
    train_path: str | None = None
    test_path: str | None = None
    oos_path: str | None = None
    target_column: str = TARGET_COLUMN
    notes: list[str] = field(default_factory=list)


@dataclass
class DLReviewResult:
    run_id: str
    architecture: str
    device: str
    cohort_metrics: dict[str, dict[str, float]]
    evidence: list[EvidenceRecord]
    figures: dict[str, str]
    agent_review: Any
    report_path: str
    narrative_ok: bool


def _result_to_record(result: TestResult, opts: DLReviewOptions, run_id: str, policy_hash: str):
    rec = EvidenceRecord.from_result(
        result,
        model_id=f"attrition-dl-{opts.architecture}",
        dataset_id="dl-review",
        run_id=run_id,
        policy_hash=policy_hash,
    )
    # carry the stable human-readable label as an artifact tag
    label = DL_EVIDENCE_LABELS.get(result.test_id)
    if label:
        rec.artifacts["dl_evidence_label"] = label
    return rec


def run_dl_review(opts: DLReviewOptions, ask: Any = None) -> DLReviewResult:
    """Execute the full deep-learning review and return a structured result."""
    from start.agents.review import run_agent_review
    from start.core.config import StartConfig, load_policy
    from start.evidence.ledger import EvidenceLedger
    from start.modeling.deep_learning import build_classifier, torch_available

    if not torch_available():
        raise ImportError(
            "Deep learning requires the torch extra: pip install -e \".[torch]\""
        )

    import uuid

    run_id = "RUN-DL-" + uuid.uuid4().hex[:8]
    config = StartConfig()
    config.seed = opts.seed
    config.output.root = opts.output_root
    policy = load_policy(config.policy_file)
    policy_hash = policy.content_hash()

    # 1. data
    console.print("\n[bold]1/8 Data[/bold] — loading and splitting 60/20/20 (stratified)")
    bundle = load_dl_bundle(
        opts.data_source,
        train_path=opts.train_path,
        test_path=opts.test_path,
        oos_path=opts.oos_path,
        target_column=opts.target_column,
        seed=opts.seed,
    )
    target = opts.target_column if opts.target_column in bundle.train.columns else bundle.target_column
    for note in bundle.notes:
        console.print(f"    [dim]{note}[/dim]")
    features = feature_columns(bundle.train, target)
    console.print(
        f"    source: {bundle.source} | train/test/oos: "
        f"{len(bundle.train)}/{len(bundle.test)}/{len(bundle.oos)} | features: {len(features)}"
    )

    # 2. build + 3. train
    console.print(f"[bold]2/8 Build[/bold] — {opts.architecture}")
    model = build_classifier(
        opts.architecture,
        epochs=opts.epochs,
        batch_size=opts.batch_size,
        learning_rate=opts.learning_rate,
        dropout=opts.dropout,
        random_state=opts.seed,
    )
    console.print("[bold]3/8 Train[/bold]")
    t0 = time.time()
    model.fit(bundle.train[features], bundle.train[target])
    device = model.device_used
    console.print(
        f"    trained in {time.time() - t0:.1f}s on device={device} | "
        f"best epoch {model.best_epoch_}"
        + (" (early-stopped)" if model.stopped_early_ else "")
    )

    # score all cohorts
    cohorts = {}
    for name, frame in (("train", bundle.train), ("test", bundle.test), ("oos", bundle.oos)):
        frame = frame.copy()
        frame[SCORE_COLUMN] = model.predict_proba(frame[features])[:, 1]
        cohorts[name] = frame

    # 4. evaluate
    console.print("[bold]4/8 Evaluate[/bold] — cohort metrics")
    cohort_metrics = {
        name: compute_dl_cohort_metrics(frame[target].to_numpy(), frame[SCORE_COLUMN].to_numpy())
        for name, frame in cohorts.items()
    }
    _print_metrics_table(cohort_metrics)

    # 5. evidence pipeline (EV-DL-0001..0007)
    console.print("[bold]5/8 Evidence[/bold] — diagnostics")
    evidence: list[EvidenceRecord] = []
    from start.modeling.dl_review_engines import (
        build_calibration_evidence,
        build_explainability_evidence,
        build_performance_evidence,
        build_robustness_evidence,
        build_sensitivity_evidence,
        build_training_evidence,
    )

    train_result = build_training_evidence(model)
    perf_result, perf_extras = build_performance_evidence(cohort_metrics)
    calib_result = build_calibration_evidence(cohort_metrics)
    explain_result, importance = build_explainability_evidence(
        model, cohorts, features, target, opts
    )
    sens_result, shock_rows = build_sensitivity_evidence(
        model, cohorts, features, target, importance, opts
    )
    robust_result = build_robustness_evidence(model, cohorts, features, target, importance, opts)

    for result in (
        train_result,
        perf_result,
        calib_result,
        explain_result,
        sens_result,
        robust_result,
    ):
        evidence.append(_result_to_record(result, opts, run_id, policy_hash))

    # 6. agent review (deterministic or LLM-assisted, both gated)
    console.print("[bold]6/8 Agent review[/bold]")
    agent_llm = None
    if opts.agent_mode == "llm":
        agent_llm = _resolve_llm(opts)
    agent_review = run_agent_review(
        evidence,
        mode=opts.agent_mode,
        llm=agent_llm,
        policy_hash=policy_hash,
        demo_meta={"architecture": opts.architecture, "device": device},
    )

    # 7. figures
    console.print("[bold]7/8 Figures[/bold]")
    figures = _build_figures(model, cohorts, target, importance, shock_rows, opts, run_id)
    fig_result = _figure_evidence(figures)
    evidence.append(_result_to_record(fig_result, opts, run_id, policy_hash))

    # persist evidence to the tamper-evident ledger
    ledger = EvidenceLedger(
        Path(opts.output_root) / "ledger.jsonl", Path(opts.output_root) / "evidence_store"
    )
    for rec in evidence:
        ledger.append(rec)

    # 8. report
    console.print("[bold]8/8 Report[/bold]")
    report_path = _render_report(
        run_id, opts, device, cohort_metrics, evidence, figures, agent_review, perf_extras
    )
    console.print(f"    report: {report_path}")
    _print_agent_summary(agent_review)

    return DLReviewResult(
        run_id=run_id,
        architecture=opts.architecture,
        device=device,
        cohort_metrics=cohort_metrics,
        evidence=evidence,
        figures=figures,
        agent_review=agent_review,
        report_path=str(report_path),
        narrative_ok=agent_review.critique_ok,
    )


def _resolve_llm(opts: DLReviewOptions):
    from start.core.config import LLMConfig
    from start.providers.llm import get_llm_provider

    provider = opts.llm_provider or "none"
    return get_llm_provider(LLMConfig(provider=provider))


def _build_figures(model, cohorts, target, importance, shock_rows, opts, run_id) -> dict[str, str]:
    from start.modeling import dl_figures

    test = cohorts["test"]
    figures = {
        "learning_curve": dl_figures.learning_curve_figure(model.history_, opts.output_root, run_id),
        "calibration_curve": dl_figures.calibration_curve_figure(
            test[target].to_numpy(), test[SCORE_COLUMN].to_numpy(), opts.output_root, run_id
        ),
        "attribution_top_features": dl_figures.attribution_figure(
            importance.global_importance, importance.method, opts.output_root, run_id
        ),
        "top_feature_shock_sensitivity": dl_figures.shock_sensitivity_figure(
            shock_rows, opts.output_root, run_id
        ),
    }
    return {k: v for k, v in figures.items() if v}


def _figure_evidence(figures: dict[str, str]) -> TestResult:
    from start.modeling.dl_figures import matplotlib_available

    if not matplotlib_available():
        return TestResult(
            test_id="deep_learning.figure_diagnostics",
            test_name="Figure generation diagnostics",
            status=Status.SKIPPED,
            interpretation="matplotlib not installed; figures were not generated.",
            limitations=["Install matplotlib to enable figure generation."],
        )
    result = TestResult(
        test_id="deep_learning.figure_diagnostics",
        test_name="Figure generation diagnostics",
        status=Status.PASS,
        metrics={"n_figures": len(figures), "figures": ", ".join(sorted(figures))},
        interpretation=f"Generated {len(figures)} review figures.",
        limitations=["Figures are visual aids; quantitative verdicts come from the evidence."],
    )
    return result.apply_thresholds()


# --------------------------------------------------------------------------- #
# Console rendering
# --------------------------------------------------------------------------- #
def _print_metrics_table(cohort_metrics: dict[str, dict[str, float]]) -> None:
    table = Table(title="Deep-learning cohort metrics")
    table.add_column("Cohort")
    pretty = {
        "auc_roc": "AUC-ROC",
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "top_decile_lift": "Top 10% Lift",
        "brier_score": "Brier",
        "ece": "ECE",
    }
    for metric in DL_METRIC_NAMES:
        table.add_column(pretty[metric], justify="right")
    for name in ("train", "test", "oos"):
        if name in cohort_metrics:
            table.add_row(name, *[f"{cohort_metrics[name][m]:.4f}" for m in DL_METRIC_NAMES])
    console.print(table)


def _print_agent_summary(agent_review: Any) -> None:
    mode = (
        f"llm-assisted ({agent_review.llm_provider})"
        if agent_review.mode == "llm"
        else "deterministic"
    )
    console.print(
        f"    agent mode: {mode} | review critique: "
        f"{'OK' if agent_review.critique_ok else 'FAILED'}"
    )
    for note in agent_review.notes:
        console.print(f"    [yellow]{note}[/yellow]")
    console.print(f"    sign-off: {agent_review.signoff.split('.')[0]}.")


def _render_report(
    run_id, opts, device, cohort_metrics, evidence, figures, agent_review, perf_extras
) -> Path:
    from start.modeling.dl_report import render_dl_report

    out_dir = Path(opts.output_root) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{run_id}.md"
    report_path.write_text(
        render_dl_report(
            run_id, opts, device, cohort_metrics, evidence, figures, agent_review, perf_extras
        )
    )
    return report_path
