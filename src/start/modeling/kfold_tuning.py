"""Stratified K-fold hyperparameter tuning for tabular classification (v2.3.1 #7).

Real K-fold model selection that operates STRICTLY inside the training split:

- The train/test/OOS split is reproduced deterministically (identical to the
  copilot execution split), and only the training rows are passed to K-fold.
- Stratified K-fold folds are created over those training rows only. Test and
  OOS rows never enter any fold, so they are never used for model selection.
- Each candidate is scored by the primary metric routed from the reviewer's
  cost priority (false negatives -> recall/PR-AUC, false positives ->
  precision/specificity, balanced -> ROC-AUC/PR-AUC).
- Artifacts: fold_metrics.csv, tuning_trials.csv, tuning_summary.json.

The estimator is a light, dependency-free sklearn pipeline (scaling + logistic
regression with a configurable C / class_weight grid), so K-fold is fast and
stable enough to run in tests and on the propensity/tabular suite. The DL
tabular path keeps its existing single-split validation and is labelled as such
(see ``tuning_run.run_tuning``) — this module never claims to K-fold a DL model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Candidate hyperparameters for the tabular logistic-regression estimator.
_C_GRID = [0.1, 1.0, 3.0]
_CLASS_WEIGHT_GRID: list[Any] = [None, "balanced"]


@dataclass
class FoldResult:
    fold: int
    metric: float
    n_train: int
    n_val: int

    def to_dict(self) -> dict[str, Any]:
        return {"fold": self.fold, "metric": self.metric,
                "n_train": self.n_train, "n_val": self.n_val}


@dataclass
class KFoldTrial:
    trial: int
    params: dict[str, Any]
    mean_metric: float
    std_metric: float
    fold_metrics: list[float] = field(default_factory=list)
    status: str = "ok"  # best | ok

    def to_dict(self) -> dict[str, Any]:
        return {"trial": self.trial, "params": self.params,
                "mean_metric": self.mean_metric, "std_metric": self.std_metric,
                "fold_metrics": self.fold_metrics, "status": self.status}


@dataclass
class KFoldTuningRun:
    method: str  # "stratified_kfold"
    primary_metric: str
    n_folds: int
    trials: list[KFoldTrial] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)
    best_mean_metric: float = 0.0
    best_std_metric: float = 0.0
    best_fold_results: list[FoldResult] = field(default_factory=list)
    train_rows: int = 0
    excluded_rows: int = 0  # test + OOS rows held out of selection
    artifacts: list[str] = field(default_factory=list)
    ran: bool = True
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "primary_metric": self.primary_metric,
            "n_folds": self.n_folds,
            "ran": self.ran,
            "trials": [t.to_dict() for t in self.trials],
            "best_params": self.best_params,
            "best_mean_metric": self.best_mean_metric,
            "best_std_metric": self.best_std_metric,
            "best_fold_results": [f.to_dict() for f in self.best_fold_results],
            "train_rows": self.train_rows,
            "excluded_rows": self.excluded_rows,
            "note": self.note,
        }


def _replace_inf_with_nan(X):
    """Replace non-finite values (inf, -inf) in feature matrix with NaN."""
    X_arr = np.asarray(X, dtype=float)
    return np.where(np.isfinite(X_arr), X_arr, np.nan)


def _metric_fn(primary_metric: str, is_multiclass: bool, classes_order: Any):
    """Map the routed primary metric to a scorer(y_true, proba)."""
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.preprocessing import label_binarize

    def auc(y, p):
        if is_multiclass:
            return float(roc_auc_score(y, p, multi_class="ovr", average="macro", labels=classes_order))
        else:
            return float(roc_auc_score(y, p))

    def prauc(y, p):
        if is_multiclass:
            y_bin = label_binarize(y, classes=classes_order)
            return float(average_precision_score(y_bin, p, average="macro"))
        else:
            return float(average_precision_score(y, p))

    def rec(y, p):
        if is_multiclass:
            yhat = classes_order[np.argmax(p, axis=1)]
            return float(recall_score(y, yhat, average="macro", labels=classes_order, zero_division=0))
        else:
            yhat = (np.asarray(p) >= 0.5).astype(int)
            return float(recall_score(y, yhat, zero_division=0))

    def prec(y, p):
        if is_multiclass:
            yhat = classes_order[np.argmax(p, axis=1)]
            return float(precision_score(y, yhat, average="macro", labels=classes_order, zero_division=0))
        else:
            yhat = (np.asarray(p) >= 0.5).astype(int)
            return float(precision_score(y, yhat, zero_division=0))

    def f1(y, p):
        if is_multiclass:
            yhat = classes_order[np.argmax(p, axis=1)]
            return float(f1_score(y, yhat, average="macro", labels=classes_order, zero_division=0))
        else:
            yhat = (np.asarray(p) >= 0.5).astype(int)
            return float(f1_score(y, yhat, zero_division=0))

    def specificity(y, p):
        if is_multiclass:
            from sklearn.metrics import confusion_matrix
            yhat = classes_order[np.argmax(p, axis=1)]
            cm = confusion_matrix(y, yhat, labels=classes_order)
            specs = []
            for i in range(len(cm)):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - (tp + fn + fp)
                spec = float(tn / (tn + fp)) if (tn + fp) else 0.0
                specs.append(spec)
            return float(np.mean(specs))
        else:
            yhat = (np.asarray(p) >= 0.5).astype(int)
            tn = int(((yhat == 0) & (np.asarray(y) == 0)).sum())
            fp = int(((yhat == 1) & (np.asarray(y) == 0)).sum())
            return float(tn / (tn + fp)) if (tn + fp) else 0.0

    return {
        "auc_roc": auc, "pr_auc": prauc, "recall": rec,
        "precision": prec, "specificity": specificity,
        "f1": f1, "macro_f1": f1,
    }.get(primary_metric, auc)


def _make_estimator(C: float, class_weight: Any, seed: int):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer, StandardScaler

    return make_pipeline(
        FunctionTransformer(_replace_inf_with_nan),
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=C, class_weight=class_weight, max_iter=1000,
                           random_state=seed),
    )


def run_kfold_tuning(
    train_df: pd.DataFrame,
    target: str,
    features: list[str],
    *,
    n_folds: int = 5,
    primary_metric: str = "auc_roc",
    seed: int = 42,
    output_root: str = "start_output",
    run_id: str = "RUN",
    registry: Any = None,
    excluded_rows: int = 0,
    task_type: str = "binary_classification",
) -> KFoldTuningRun | None:
    """Run stratified K-fold tuning over TRAIN-ONLY rows. ``train_df`` must
    already exclude test/OOS. Returns None if infeasible (too few rows/features
    or a degenerate target)."""
    if len(features) < 2 or len(train_df) < 2 * n_folds:
        return None
    if target not in train_df or train_df[target].nunique() < 2:
        return None

    is_multiclass = (task_type == "multiclass_classification")

    metric_name = (primary_metric if primary_metric in
                   ("auc_roc", "pr_auc", "recall", "precision", "specificity", "f1", "macro_f1")
                   else "auc_roc")

    X = train_df[features].to_numpy(dtype=float)
    y = train_df[target].to_numpy()

    # smallest class count caps the number of stratified folds
    _, counts = np.unique(y, return_counts=True)
    folds = max(2, min(n_folds, int(counts.min())))

    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    candidates = [{"C": c, "class_weight": cw} for c in _C_GRID
                  for cw in _CLASS_WEIGHT_GRID]

    run = KFoldTuningRun(
        method="stratified_kfold", primary_metric=metric_name, n_folds=folds,
        train_rows=len(train_df), excluded_rows=excluded_rows,
    )
    best_mean, best_params, best_folds = -1.0, {}, []
    for ti, params in enumerate(candidates, start=1):
        fold_scores: list[float] = []
        fold_results: list[FoldResult] = []
        for fi, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
            est = _make_estimator(params["C"], params["class_weight"], seed)
            est.fit(X[tr_idx], y[tr_idx])
            classes_order = est.classes_
            score = _metric_fn(metric_name, is_multiclass, classes_order)

            if is_multiclass:
                proba = est.predict_proba(X[va_idx])
            else:
                proba = est.predict_proba(X[va_idx])[:, 1]

            m = round(float(score(y[va_idx], proba)), 6)
            fold_scores.append(m)
            fold_results.append(FoldResult(fold=fi, metric=m,
                                           n_train=len(tr_idx), n_val=len(va_idx)))
        mean_m = round(float(np.mean(fold_scores)), 6)
        std_m = round(float(np.std(fold_scores)), 6)
        run.trials.append(KFoldTrial(
            trial=ti, params=dict(params), mean_metric=mean_m,
            std_metric=std_m, fold_metrics=fold_scores,
        ))
        if mean_m > best_mean:
            best_mean, best_params, best_folds = mean_m, dict(params), fold_results

    for t in run.trials:
        t.status = "best" if t.params == best_params else "ok"
    run.best_params = best_params
    run.best_mean_metric = best_mean
    run.best_std_metric = next(
        (t.std_metric for t in run.trials if t.status == "best"), 0.0)
    run.best_fold_results = best_folds

    # --- artifacts ---
    out_dir = Path(output_root) / "tuning" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_csv = out_dir / "fold_metrics.csv"
    pd.DataFrame([
        {"trial": t.trial, "C": t.params["C"],
         "class_weight": str(t.params["class_weight"]), "fold": i + 1,
         "metric": fm}
        for t in run.trials for i, fm in enumerate(t.fold_metrics)
    ]).to_csv(fold_csv, index=False)
    trials_csv = out_dir / "tuning_trials.csv"
    pd.DataFrame([
        {"trial": t.trial, "C": t.params["C"],
         "class_weight": str(t.params["class_weight"]),
         "mean_metric": t.mean_metric, "std_metric": t.std_metric,
         "status": t.status}
        for t in run.trials
    ]).to_csv(trials_csv, index=False)
    summary_json = out_dir / "tuning_summary.json"
    summary_json.write_text(json.dumps(run.to_dict(), indent=2, default=str))
    run.artifacts = [str(fold_csv), str(trials_csv), str(summary_json)]
    if registry is not None:
        for path in run.artifacts:
            registry.register(path, category="tuning")
    return run


def render_kfold_markdown(run: KFoldTuningRun) -> str:
    """Markdown table for dashboard/transcript/notebook (#7)."""
    if not run.ran:
        return f"### K-fold tuning\n\n{run.note}\n"
    lines = [
        "### K-fold tuning",
        "",
        f"- Method: {run.method} ({run.n_folds}-fold, stratified)",
        f"- Primary metric: {run.primary_metric}",
        f"- Train rows used: {run.train_rows} "
        f"(test/OOS rows excluded from selection: {run.excluded_rows})",
        f"- Best params: {run.best_params}",
        f"- Best mean {run.primary_metric}: {run.best_mean_metric:.4f} "
        f"(std {run.best_std_metric:.4f})",
        "",
        "| Fold | Metric (best params) | n_train | n_val |",
        "| --- | --- | --- | --- |",
    ]
    for f in run.best_fold_results:
        lines.append(f"| {f.fold} | {f.metric:.4f} | {f.n_train} | {f.n_val} |")
    lines += ["", f"**Mean:** {run.best_mean_metric:.4f}  |  "
              f"**Std:** {run.best_std_metric:.4f}"]
    return "\n".join(lines) + "\n"
