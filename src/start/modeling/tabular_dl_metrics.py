"""Task-branching metrics for tabular deep learning.

Metrics automatically branch by task type so the report can state exactly
what was measured:

    binary       -> AUC-ROC, accuracy, precision, recall, F1, top-decile lift,
                    Brier, ECE
    multiclass   -> accuracy, macro/weighted F1, macro precision/recall,
                    log loss, per-class support
    multilabel   -> subset accuracy, micro/macro F1, Hamming loss, mean AUC
"""

from __future__ import annotations

from typing import Any

import numpy as np


def binary_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    from start.modeling.dl_metrics import compute_dl_cohort_metrics

    scores = proba[:, 1] if proba.ndim == 2 else proba
    return compute_dl_cohort_metrics(np.asarray(y_true).reshape(-1), scores)


def multiclass_metrics(y_true: np.ndarray, proba: np.ndarray, classes: Any) -> dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true).reshape(-1)
    preds_idx = proba.argmax(axis=1)
    classes = np.asarray(classes)
    preds = classes[preds_idx]
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, preds)), 6),
        "f1_macro": round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 6),
        "f1_weighted": round(float(f1_score(y_true, preds, average="weighted", zero_division=0)), 6),
        "precision_macro": round(
            float(precision_score(y_true, preds, average="macro", zero_division=0)), 6
        ),
        "recall_macro": round(
            float(recall_score(y_true, preds, average="macro", zero_division=0)), 6
        ),
        "n_classes": int(len(classes)),
    }
    try:
        metrics["log_loss"] = round(float(log_loss(y_true, proba, labels=list(classes))), 6)
    except (ValueError, IndexError):
        metrics["log_loss"] = float("nan")
    return metrics


def multilabel_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import f1_score, hamming_loss, roc_auc_score

    y_true = np.asarray(y_true)
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "subset_accuracy": round(float((preds == y_true).all(axis=1).mean()), 6),
        "f1_micro": round(float(f1_score(y_true, preds, average="micro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 6),
        "hamming_loss": round(float(hamming_loss(y_true, preds)), 6),
        "n_labels": int(y_true.shape[1]),
    }
    # mean AUC across labels that have both classes present
    aucs = []
    for j in range(y_true.shape[1]):
        col = y_true[:, j]
        if len(np.unique(col)) == 2:
            aucs.append(roc_auc_score(col, proba[:, j]))
    metrics["mean_auc"] = round(float(np.mean(aucs)), 6) if aucs else float("nan")
    return metrics


def dl_task_metrics(
    task: str, y_true: np.ndarray, proba: np.ndarray, classes: Any = None
) -> dict[str, float]:
    """Dispatch to the metric set for the task."""
    if task == "binary_classification":
        return binary_metrics(y_true, proba)
    if task == "multiclass_classification":
        return multiclass_metrics(y_true, proba, classes)
    if task == "multilabel_classification":
        return multilabel_metrics(y_true, proba)
    raise ValueError(f"No metric branch for task '{task}'.")
