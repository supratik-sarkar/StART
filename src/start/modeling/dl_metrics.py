"""Deep-learning evaluation metrics with train/test/OOS parity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class CohortPrediction:
    cohort: str
    y_true: np.ndarray
    y_score: np.ndarray


def top_decile_lift(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.10) -> float:
    n = max(1, int(np.ceil(len(y_true) * fraction)))
    order = np.argsort(-y_score)[:n]
    base_rate = float(np.mean(y_true))
    if base_rate == 0:
        return 0.0
    return float(np.mean(y_true[order]) / base_rate)


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (y_score >= lo) & (y_score < hi if hi < 1 else y_score <= hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask) * abs(np.mean(y_true[mask]) - np.mean(y_score[mask])))
    return ece


def cohort_metrics_table(predictions: list[CohortPrediction], threshold: float = 0.5) -> pd.DataFrame:
    rows = []
    for pred in predictions:
        y_hat = (pred.y_score >= threshold).astype(int)
        rows.append(
            {
                "cohort": pred.cohort,
                "auc_roc": float(roc_auc_score(pred.y_true, pred.y_score)),
                "accuracy": float(accuracy_score(pred.y_true, y_hat)),
                "precision": float(precision_score(pred.y_true, y_hat, zero_division=0)),
                "recall": float(recall_score(pred.y_true, y_hat, zero_division=0)),
                "f1": float(f1_score(pred.y_true, y_hat, zero_division=0)),
                "top_10_lift": top_decile_lift(pred.y_true, pred.y_score),
                "brier": float(brier_score_loss(pred.y_true, pred.y_score)),
                "ece": expected_calibration_error(pred.y_true, pred.y_score),
            }
        )
    return pd.DataFrame(rows)


def generalization_gap(metrics: pd.DataFrame) -> float:
    train = metrics.loc[metrics["cohort"] == "train", "auc_roc"].iloc[0]
    test = metrics.loc[metrics["cohort"] == "test", "auc_roc"].iloc[0]
    return float(train - test)
