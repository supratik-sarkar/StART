"""Builders for the deep-learning diagnostic evidence records.

Each returns a TestResult (the orchestrator wraps them into EvidenceRecords
with EV-DL-000x labels). Every quantitative interpretation references the
metrics actually computed; explainability never claims a method it did not
run. Thresholds drive pass/warn/fail exactly like the classical engines.
"""

from __future__ import annotations

from typing import Any

from start.core.schemas import TestResult, ThresholdSpec
from start.modeling.dl_metrics import training_diagnostics


def build_training_evidence(model: Any) -> TestResult:
    """EV-DL-0001 — training diagnostics: curves, generalization gap, early stop."""
    diag = training_diagnostics(model.history_, model.best_epoch_, model.stopped_early_)
    metrics: dict[str, Any] = {
        "epochs_run": diag["epochs_run"],
        "best_epoch": diag["best_epoch"],
        "stopped_early": str(diag["stopped_early"]),
        "final_train_loss": diag["final_train_loss"],
    }
    thresholds = []
    interpretation = (
        f"Trained {diag['epochs_run']} epoch(s); best epoch {diag['best_epoch']}"
        + (" (early-stopped)" if diag["stopped_early"] else "")
        + f"; final train loss {diag['final_train_loss']:.4f}."
    )
    if "generalization_gap" in diag:
        metrics["final_val_loss"] = diag["final_val_loss"]
        metrics["generalization_gap"] = diag["generalization_gap"]
        metrics["val_increase_from_min"] = diag["val_increase_from_min"]
        thresholds = [
            ThresholdSpec(metric="generalization_gap", warn=0.10, fail=0.30),
            ThresholdSpec(metric="val_increase_from_min", warn=0.05, fail=0.20),
        ]
        interpretation += (
            f" Validation loss {diag['final_val_loss']:.4f}; generalization gap "
            f"{diag['generalization_gap']:.4f}."
        )
    result = TestResult(
        test_id="deep_learning.training_diagnostics",
        test_name="Training diagnostics",
        metrics=metrics,
        thresholds=thresholds,
        interpretation=interpretation,
        limitations=[
            "Generalization gap uses an internal validation split, not the OOS cohort.",
            "Loss is BCE on standardized features.",
        ],
    )
    return result.apply_thresholds()


def build_performance_evidence(
    cohort_metrics: dict[str, dict[str, float]],
) -> tuple[TestResult, dict[str, dict[str, float]]]:
    """EV-DL-0002 — performance diagnostics across cohorts + overfit gap."""
    metrics: dict[str, Any] = {}
    for cohort, m in cohort_metrics.items():
        for key in ("auc_roc", "accuracy", "precision", "recall", "f1", "top_decile_lift"):
            metrics[f"{cohort}_{key}"] = m[key]
    thresholds = []
    if "train" in cohort_metrics and "test" in cohort_metrics:
        gap = round(cohort_metrics["train"]["auc_roc"] - cohort_metrics["test"]["auc_roc"], 6)
        metrics["auc_gap_train_test"] = gap
        thresholds.append(ThresholdSpec(metric="auc_gap_train_test", warn=0.05, fail=0.10))
    if "test" in cohort_metrics and "oos" in cohort_metrics:
        metrics["auc_gap_test_oos"] = round(
            cohort_metrics["test"]["auc_roc"] - cohort_metrics["oos"]["auc_roc"], 6
        )
    bits = [f"{c} AUC-ROC {m['auc_roc']:.4f}" for c, m in cohort_metrics.items()]
    result = TestResult(
        test_id="deep_learning.performance_diagnostics",
        test_name="Performance diagnostics",
        metrics=metrics,
        thresholds=thresholds,
        interpretation="; ".join(bits) + ".",
        limitations=["AUC is threshold-independent; operating-point metrics use 0.5."],
    )
    return result.apply_thresholds(), cohort_metrics


def build_calibration_evidence(cohort_metrics: dict[str, dict[str, float]]) -> TestResult:
    """EV-DL-0003 — calibration diagnostics (Brier, ECE) anchored on test."""
    anchor = "test" if "test" in cohort_metrics else next(iter(cohort_metrics))
    metrics: dict[str, Any] = {}
    for cohort, m in cohort_metrics.items():
        metrics[f"{cohort}_brier_score"] = m["brier_score"]
        metrics[f"{cohort}_ece"] = m["ece"]
    metrics["anchor_cohort"] = anchor
    result = TestResult(
        test_id="deep_learning.calibration_diagnostics",
        test_name="Calibration diagnostics",
        metrics=metrics,
        thresholds=[
            ThresholdSpec(metric=f"{anchor}_ece", warn=0.05, fail=0.15),
            ThresholdSpec(metric=f"{anchor}_brier_score", warn=0.15, fail=0.25),
        ],
        interpretation=(
            f"On the {anchor} cohort, Brier score is {cohort_metrics[anchor]['brier_score']:.4f} "
            f"and ECE is {cohort_metrics[anchor]['ece']:.4f}."
        ),
        limitations=["ECE uses 10 equal-width probability bins."],
    )
    return result.apply_thresholds()


def build_explainability_evidence(
    model: Any,
    cohorts: dict[str, Any],
    features: list[str],
    target: str,
    opts: Any,
):
    """EV-DL-0004 — explainability diagnostics with honest method routing."""
    from start.modeling.dl_explain import dl_global_importance

    frame = cohorts["test"]
    importance = dl_global_importance(
        model,
        frame[features],
        frame[target].to_numpy(),
        prefer=opts.explain_method,
        seed=opts.seed,
    )
    top = importance.global_importance[:5]
    metrics = {
        "method": importance.method,
        "available_methods": ", ".join(importance.available_methods),
        "top_features": ", ".join(name for name, _ in top),
        "top_feature": top[0][0] if top else "",
        "top_feature_importance": top[0][1] if top else 0.0,
    }
    note = f" Note: {importance.note}" if importance.note else ""
    result = TestResult(
        test_id="deep_learning.explainability_diagnostics",
        test_name="Explainability diagnostics",
        params={"preferred_method": opts.explain_method},
        metrics=metrics,
        interpretation=(
            f"Global importance computed via {importance.method}; the most influential "
            f"feature is '{top[0][0] if top else 'n/a'}'.{note}"
        ),
        limitations=(
            ["Permutation fallback used; gradient attributions unavailable."]
            if importance.method == "permutation"
            else ["Attributions use a zero baseline in standardized feature space."]
        ),
    )
    return result.apply_thresholds(), importance


def build_sensitivity_evidence(
    model: Any,
    cohorts: dict[str, Any],
    features: list[str],
    target: str,
    importance: Any,
    opts: Any,
):
    """EV-DL-0005 — top-feature shock sensitivity on the chosen cohort."""
    from start.modeling.dl_sensitivity import DEFAULT_SHOCKS, feature_shock_sensitivity

    frame = _cohort_frame(cohorts, opts.sensitivity_cohort)
    top_features = importance.top_features(5)
    rows = feature_shock_sensitivity(
        model, frame[features], frame[target].to_numpy(), top_features, DEFAULT_SHOCKS
    )
    metrics: dict[str, Any] = {
        "cohort": opts.sensitivity_cohort,
        "shocked_features": ", ".join(top_features),
        "baseline_auc": next(r["auc_roc"] for r in rows if r["shock"] == 0.0),
    }
    for row in rows:
        label = f"{int(row['shock'] * 100):+d}pct"
        metrics[f"auc_{label}"] = row["auc_roc"]
        metrics[f"drift_{label}"] = row["auc_drift"]
    metrics["max_abs_auc_drift"] = round(max(abs(r["auc_drift"]) for r in rows), 6)
    result = TestResult(
        test_id="deep_learning.sensitivity_diagnostics",
        test_name="Sensitivity diagnostics",
        params={"cohort": opts.sensitivity_cohort},
        metrics=metrics,
        thresholds=[ThresholdSpec(metric="max_abs_auc_drift", warn=0.02, fail=0.10)],
        interpretation=(
            f"Parallel shocks to the top 5 features on the {opts.sensitivity_cohort} cohort "
            f"produced a maximum absolute AUC drift of {metrics['max_abs_auc_drift']:.4f} "
            f"from baseline {metrics['baseline_auc']:.4f}."
        ),
        limitations=["Parallel multiplicative shocks; large shocks may leave training support."],
    )
    return result.apply_thresholds(), rows


def build_robustness_evidence(
    model: Any,
    cohorts: dict[str, Any],
    features: list[str],
    target: str,
    importance: Any,
    opts: Any,
) -> TestResult:
    """EV-DL-0006 — input-noise and feature-masking robustness."""
    from start.modeling.dl_sensitivity import (
        DEFAULT_MASK_COUNTS,
        DEFAULT_NOISE_LEVELS,
        feature_masking_robustness,
        input_noise_robustness,
    )

    frame = _cohort_frame(cohorts, opts.sensitivity_cohort)
    X, y = frame[features], frame[target].to_numpy()
    top_features = importance.top_features(5)
    noise_rows = input_noise_robustness(model, X, y, features, DEFAULT_NOISE_LEVELS, seed=opts.seed)
    mask_rows = feature_masking_robustness(model, X, y, top_features, DEFAULT_MASK_COUNTS)

    metrics: dict[str, Any] = {"cohort": opts.sensitivity_cohort}
    for row in noise_rows:
        metrics[f"noise_{row['noise']:.2f}_auc"] = row["auc_roc"]
        metrics[f"noise_{row['noise']:.2f}_drift"] = row["auc_drift"]
    for row in mask_rows:
        metrics[f"mask_top{row['masked_top_k']}_auc"] = row["auc_roc"]
        metrics[f"mask_top{row['masked_top_k']}_drift"] = row["auc_drift"]
    metrics["max_abs_noise_drift"] = round(max(abs(r["auc_drift"]) for r in noise_rows), 6)
    metrics["max_abs_mask_drift"] = round(max(abs(r["auc_drift"]) for r in mask_rows), 6)
    result = TestResult(
        test_id="deep_learning.robustness_diagnostics",
        test_name="Robustness diagnostics",
        params={"cohort": opts.sensitivity_cohort},
        metrics=metrics,
        thresholds=[
            ThresholdSpec(metric="max_abs_noise_drift", warn=0.03, fail=0.12),
            ThresholdSpec(metric="max_abs_mask_drift", warn=0.10, fail=0.30),
        ],
        interpretation=(
            f"Under input noise the maximum AUC drift was {metrics['max_abs_noise_drift']:.4f}; "
            f"masking the top features moved AUC by up to {metrics['max_abs_mask_drift']:.4f}."
        ),
        limitations=[
            "Noise is Gaussian, scaled per-feature by standard deviation.",
            "Masking replaces a feature with its cohort mean.",
        ],
    )
    return result.apply_thresholds()


def _cohort_frame(cohorts: dict[str, Any], cohort: str):
    import pandas as pd

    if cohort == "development":
        return pd.concat(list(cohorts.values()), ignore_index=True)
    return cohorts.get(cohort, cohorts["test"])
