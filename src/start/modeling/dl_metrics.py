"""Deep-learning metrics: cohort performance (reusing the shared metric core)
extended with Brier score and Expected Calibration Error, plus training-curve
diagnostics (generalization gap, overfitting signal, early-stopping metadata).
"""

from __future__ import annotations

import numpy as np

from start.modeling.metrics import compute_cohort_metrics

DL_METRIC_NAMES = (
    "auc_roc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "top_decile_lift",
    "brier_score",
    "ece",
)


def brier_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    return float(np.mean((scores - y_true) ** 2))


def expected_calibration_error(
    y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10
) -> float:
    """ECE: weighted average gap between confidence and accuracy across bins."""
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(y_true) == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(scores)
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        in_bin = (scores > lo) & (scores <= hi) if lo > 0 else (scores >= lo) & (scores <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        confidence = float(scores[in_bin].mean())
        accuracy = float(y_true[in_bin].mean())
        ece += (count / n) * abs(confidence - accuracy)
    return float(ece)


def compute_dl_cohort_metrics(
    y_true: np.ndarray, scores: np.ndarray, decision_threshold: float = 0.5
) -> dict[str, float]:
    """Shared cohort metrics plus calibration (Brier, ECE)."""
    metrics = compute_cohort_metrics(y_true, scores, decision_threshold)
    metrics["brier_score"] = round(brier_score(y_true, scores), 6)
    metrics["ece"] = round(expected_calibration_error(y_true, scores), 6)
    return metrics


def training_diagnostics(history: dict[str, list[float]], best_epoch: int, stopped_early: bool) -> dict:
    """Generalization gap and overfitting signal from the learning curves."""
    train = history.get("train_loss", [])
    val = history.get("val_loss", [])
    diagnostics: dict[str, float | int | bool | str] = {
        "epochs_run": len(train),
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "final_train_loss": round(train[-1], 6) if train else 0.0,
    }
    if val:
        gap = val[-1] - train[-1] if train else 0.0
        min_val = min(val)
        diagnostics["final_val_loss"] = round(val[-1], 6)
        diagnostics["min_val_loss"] = round(min_val, 6)
        diagnostics["generalization_gap"] = round(gap, 6)
        # overfitting signal: validation loss climbed meaningfully off its min
        diagnostics["val_increase_from_min"] = round(val[-1] - min_val, 6)
    return diagnostics
