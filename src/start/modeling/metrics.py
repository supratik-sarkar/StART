"""Deterministic cohort metrics: AUC-ROC, Accuracy, Precision, Recall, F1,
and top-10% lift, computed identically for train / test / OOS cohorts."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as skm

METRIC_NAMES = ("auc_roc", "accuracy", "precision", "recall", "f1", "top_decile_lift")


def top_decile_lift(y_true: np.ndarray, scores: np.ndarray, fraction: float = 0.10) -> float:
    """Lift = event rate in the top `fraction` of scores / overall event rate."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    overall = float(np.mean(y_true))
    if overall == 0.0 or len(y_true) == 0:
        return 0.0
    n_top = max(int(np.ceil(len(scores) * fraction)), 1)
    top_idx = np.argsort(scores)[::-1][:n_top]
    top_rate = float(np.mean(y_true[top_idx]))
    return top_rate / overall


def compute_cohort_metrics(
    y_true: np.ndarray, scores: np.ndarray, decision_threshold: float = 0.5
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pred = (scores >= decision_threshold).astype(int)
    return {
        "auc_roc": round(float(skm.roc_auc_score(y_true, scores)), 6),
        "accuracy": round(float(skm.accuracy_score(y_true, pred)), 6),
        "precision": round(float(skm.precision_score(y_true, pred, zero_division=0)), 6),
        "recall": round(float(skm.recall_score(y_true, pred, zero_division=0)), 6),
        "f1": round(float(skm.f1_score(y_true, pred, zero_division=0)), 6),
        "top_decile_lift": round(top_decile_lift(y_true, scores), 6),
    }


def cohort_comparison(
    cohorts: dict[str, tuple[np.ndarray, np.ndarray]], decision_threshold: float = 0.5
) -> dict[str, dict[str, float]]:
    """cohorts: name -> (y_true, scores). Returns name -> metric dict."""
    return {
        name: compute_cohort_metrics(y, s, decision_threshold)
        for name, (y, s) in cohorts.items()
    }
