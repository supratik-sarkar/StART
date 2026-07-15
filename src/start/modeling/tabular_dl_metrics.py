"""Task-branching metrics for tabular deep learning.

Metrics automatically branch by task type so the report can state exactly
what was measured:

    binary       -> AUC-ROC, accuracy, precision, recall, F1, top-decile lift,
                    Brier, ECE
    multiclass   -> accuracy, macro/weighted F1, macro precision/recall,
                    log loss, per-class support
    multilabel   -> subset accuracy, micro/macro F1, Hamming loss, mean AUC
    regression   -> RMSE, MSE, MAE, R2
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
        average_precision_score,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize

    y_true = np.asarray(y_true).reshape(-1)
    preds_idx = proba.argmax(axis=1)
    classes = np.asarray(classes)
    preds = classes[preds_idx]
    
    metrics = {
        "accuracy": round(float(accuracy_score(y_true, preds)), 6),
        "f1_macro": round(float(f1_score(y_true, preds, average="macro", zero_division=0)), 6),
        "f1_micro": round(float(f1_score(y_true, preds, average="micro", zero_division=0)), 6),
        "f1_weighted": round(float(f1_score(y_true, preds, average="weighted", zero_division=0)), 6),
        "precision_macro": round(
            float(precision_score(y_true, preds, average="macro", zero_division=0)), 6
        ),
        "precision_micro": round(
            float(precision_score(y_true, preds, average="micro", zero_division=0)), 6
        ),
        "precision_weighted": round(
            float(precision_score(y_true, preds, average="weighted", zero_division=0)), 6
        ),
        "recall_macro": round(
            float(recall_score(y_true, preds, average="macro", zero_division=0)), 6
        ),
        "recall_micro": round(
            float(recall_score(y_true, preds, average="micro", zero_division=0)), 6
        ),
        "recall_weighted": round(
            float(recall_score(y_true, preds, average="weighted", zero_division=0)), 6
        ),
        "n_classes": int(len(classes)),
    }
    
    # OVR ROC-AUC
    try:
        metrics["auc_roc_macro"] = round(
            float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)), 6
        )
        metrics["auc_roc_weighted"] = round(
            float(roc_auc_score(y_true, proba, multi_class="ovr", average="weighted", labels=classes)), 6
        )
    except Exception:
        metrics["auc_roc_macro"] = float("nan")
        metrics["auc_roc_weighted"] = float("nan")
        
    # Macro PR-AUC using binarized labels
    try:
        y_bin = label_binarize(y_true, classes=classes)
        if y_bin.shape[1] == 1 and proba.shape[1] == 2:
            y_bin = np.column_stack([1 - y_bin, y_bin])
        metrics["pr_auc_macro"] = round(
            float(average_precision_score(y_bin, proba, average="macro")), 6
        )
    except Exception:
        metrics["pr_auc_macro"] = float("nan")

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


def regression_metrics(y_true: np.ndarray, preds: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    mse = float(mean_squared_error(y_true, preds))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, preds))
    r2 = float(r2_score(y_true, preds))
    try:
        from sklearn.metrics import mean_absolute_percentage_error
        mape = float(mean_absolute_percentage_error(y_true, preds))
    except Exception:
        # manual fallback for MAPE
        idx = y_true != 0
        mape = float(np.mean(np.abs((y_true[idx] - preds[idx]) / y_true[idx]))) if np.any(idx) else 0.0
    return {
        "rmse": round(rmse, 6),
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "r2": round(r2, 6),
        "mape": round(mape, 6),
    }


def dl_task_metrics(
    task: str, y_true: np.ndarray, proba: np.ndarray, classes: Any = None
) -> dict[str, float]:
    """Dispatch to the metric set for the task."""
    if task in ("regression", "forecasting"):
        return regression_metrics(y_true, proba)
    if task == "binary_classification":
        return binary_metrics(y_true, proba)
    if task == "multiclass_classification":
        return multiclass_metrics(y_true, proba, classes)
    if task == "multilabel_classification":
        return multilabel_metrics(y_true, proba)
    raise ValueError(f"No metric branch for task '{task}'.")
