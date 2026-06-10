"""Supervised (binary classification) test family. Deterministic engines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from start.core.schemas import Status, TestResult, ThresholdSpec
from start.registry import TestContext, register_test


def _binary_arrays(ctx: TestContext) -> tuple[np.ndarray, np.ndarray] | None:
    df: pd.DataFrame = ctx.test if ctx.test is not None else ctx.train
    if df is None or ctx.target_column is None or ctx.score_column is None:
        return None
    if ctx.target_column not in df.columns or ctx.score_column not in df.columns:
        return None
    frame = df[[ctx.target_column, ctx.score_column]].dropna()
    return frame[ctx.target_column].to_numpy(), frame[ctx.score_column].to_numpy()


def _skipped(test_id: str, name: str) -> TestResult:
    return TestResult(
        test_id=test_id,
        test_name=name,
        status=Status.SKIPPED,
        interpretation="Required target/score columns not available; test skipped.",
    )


@register_test(
    "supervised.discrimination",
    family="supervised",
    name="Discrimination (AUC / Gini / KS)",
    requires=("test", "target_column", "score_column"),
    default_params={"auc_warn": 0.65, "auc_fail": 0.55},
)
def discrimination(ctx: TestContext, auc_warn: float = 0.65, auc_fail: float = 0.55) -> TestResult:
    """ROC-AUC, Gini, and KS statistic on holdout scores."""
    arrays = _binary_arrays(ctx)
    if arrays is None:
        return _skipped("supervised.discrimination", "Discrimination (AUC / Gini / KS)")
    y, score = arrays
    auc = float(skm.roc_auc_score(y, score))
    fpr, tpr, _ = skm.roc_curve(y, score)
    ks = float(np.max(tpr - fpr))
    result = TestResult(
        test_id="supervised.discrimination",
        test_name="Discrimination (AUC / Gini / KS)",
        params={"auc_warn": auc_warn, "auc_fail": auc_fail},
        metrics={
            "roc_auc": round(auc, 6),
            "gini": round(2 * auc - 1, 6),
            "ks_statistic": round(ks, 6),
            "n_holdout": int(len(y)),
            "positive_rate": round(float(np.mean(y)), 6),
        },
        thresholds=[ThresholdSpec(metric="roc_auc", warn=auc_warn, fail=auc_fail, direction="lower")],
        interpretation=f"Holdout ROC-AUC is {auc:.4f} (Gini {2 * auc - 1:.4f}, KS {ks:.4f}).",
        limitations=["AUC is insensitive to calibration and to class-imbalance costs."],
    )
    return result.apply_thresholds()


@register_test(
    "supervised.classification_metrics",
    family="supervised",
    name="Thresholded classification metrics",
    requires=("test", "target_column", "score_column"),
    default_params={"decision_threshold": 0.5},
)
def classification_metrics(ctx: TestContext, decision_threshold: float = 0.5) -> TestResult:
    """Accuracy, precision, recall, F1, balanced accuracy at a decision threshold."""
    arrays = _binary_arrays(ctx)
    if arrays is None:
        return _skipped("supervised.classification_metrics", "Thresholded classification metrics")
    y, score = arrays
    pred = (score >= decision_threshold).astype(int)
    result = TestResult(
        test_id="supervised.classification_metrics",
        test_name="Thresholded classification metrics",
        params={"decision_threshold": decision_threshold},
        metrics={
            "accuracy": round(float(skm.accuracy_score(y, pred)), 6),
            "balanced_accuracy": round(float(skm.balanced_accuracy_score(y, pred)), 6),
            "precision": round(float(skm.precision_score(y, pred, zero_division=0)), 6),
            "recall": round(float(skm.recall_score(y, pred, zero_division=0)), 6),
            "f1": round(float(skm.f1_score(y, pred, zero_division=0)), 6),
        },
        interpretation=(
            f"At threshold {decision_threshold:.2f}, F1 is "
            f"{skm.f1_score(y, pred, zero_division=0):.4f}."
        ),
        limitations=["Single-threshold metrics; threshold choice should match business costs."],
    )
    return result.apply_thresholds()


@register_test(
    "supervised.calibration",
    family="supervised",
    name="Calibration (Brier / ECE)",
    requires=("test", "target_column", "score_column"),
    default_params={"n_bins": 10, "ece_warn": 0.05, "ece_fail": 0.15},
)
def calibration(
    ctx: TestContext, n_bins: int = 10, ece_warn: float = 0.05, ece_fail: float = 0.15
) -> TestResult:
    """Brier score and expected calibration error over equal-width bins."""
    arrays = _binary_arrays(ctx)
    if arrays is None:
        return _skipped("supervised.calibration", "Calibration (Brier / ECE)")
    y, score = arrays
    brier = float(skm.brier_score_loss(y, np.clip(score, 0, 1)))
    edges = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.clip(np.digitize(score, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        ece += (mask.mean()) * abs(float(y[mask].mean()) - float(score[mask].mean()))
    result = TestResult(
        test_id="supervised.calibration",
        test_name="Calibration (Brier / ECE)",
        params={"n_bins": n_bins, "ece_warn": ece_warn, "ece_fail": ece_fail},
        metrics={"brier_score": round(brier, 6), "ece": round(float(ece), 6), "n_bins": n_bins},
        thresholds=[ThresholdSpec(metric="ece", warn=ece_warn, fail=ece_fail)],
        interpretation=f"Expected calibration error is {ece:.4f}; Brier score is {brier:.4f}.",
        limitations=["ECE depends on binning scheme; equal-width bins used here."],
    )
    return result.apply_thresholds()
