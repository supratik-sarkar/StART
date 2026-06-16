"""Co-pilot execution artifacts (v2.1.1 remediation Sections D/G/I/J/K).

When the enterprise review runs with training enabled, this produces the
visible, exportable tables the reviewer needs — split distribution, metrics by
split (with calibration), training diagnostics, and a global explainability
table — and writes them as CSV/JSON artifacts registered in the
ArtifactRegistry. It reuses the existing DL building blocks (no new model
abstractions) and degrades honestly if torch is unavailable.

All outputs are deterministic given a seed and contain no raw user rows beyond
the standard evidence posture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class CopilotExecution:
    split_table: list[dict[str, Any]] = field(default_factory=list)
    metrics_by_split: dict[str, dict[str, float]] = field(default_factory=dict)
    training_diagnostics: dict[str, Any] = field(default_factory=dict)
    explainability_method: str = ""
    global_importance: list[dict[str, Any]] = field(default_factory=list)
    explainability_available: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    pruned_features: list[str] = field(default_factory=list)
    generalization_gap: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_table": self.split_table,
            "metrics_by_split": self.metrics_by_split,
            "training_diagnostics": self.training_diagnostics,
            "explainability_method": self.explainability_method,
            "global_importance": self.global_importance,
            "explainability_available": self.explainability_available,
            "generalization_gap": self.generalization_gap,
        }


def _stratified_split(
    df: pd.DataFrame, target: str, props: tuple[float, float, float], seed: int
) -> dict[str, pd.DataFrame]:
    """Stratified train/test/OOS split honoring user proportions."""
    train_p, test_p, _ = props
    rng = np.random.default_rng(seed)
    parts: dict[str, list[pd.DataFrame]] = {"train": [], "test": [], "oos": []}
    for _, grp in df.groupby(target, observed=True):
        idx = grp.index.to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * train_p))
        n_te = int(round(n * test_p))
        parts["train"].append(grp.loc[idx[:n_tr]])
        parts["test"].append(grp.loc[idx[n_tr:n_tr + n_te]])
        parts["oos"].append(grp.loc[idx[n_tr + n_te:]])
    return {k: pd.concat(v).sample(frac=1.0, random_state=seed) if v else pd.DataFrame()
            for k, v in parts.items()}


def _split_rows(splits: dict[str, pd.DataFrame], target: str) -> list[dict[str, Any]]:
    total = sum(len(f) for f in splits.values()) or 1
    rows = []
    for name, frame in splits.items():
        n = len(frame)
        pos = float((frame[target] == frame[target].max()).mean()) if n else 0.0
        rows.append({
            "split": name,
            "rows": n,
            "percent": round(100.0 * n / total, 2),
            "positive_rate": round(pos, 4),
            "negative_rate": round(1.0 - pos, 4),
        })
    return rows


def run_copilot_execution(
    df: pd.DataFrame,
    target: str,
    *,
    split_props: tuple[float, float, float] = (0.60, 0.20, 0.20),
    metric_name: str = "auc_roc",
    explain_method: str = "integrated_gradients",
    seed: int = 42,
    output_root: str = "start_output",
    run_id: str = "RUN",
    registry: Any = None,
    apply_correlation_pruning: bool = False,
) -> CopilotExecution | None:
    """Train a tabular model on a user-proportioned split and emit the visible
    tables + artifacts. Returns None if torch is unavailable (caller surfaces
    the honest fallback).

    ``apply_correlation_pruning`` reflects the user's FE decision: when True,
    one feature from each highly-correlated pair is dropped before training;
    when False (e.g. user rejected pruning), all features are retained (#2)."""
    try:
        from start.modeling.deep_learning import torch_available
        if not torch_available():
            return None
        from start.modeling.dl_explain import dl_global_importance
        from start.modeling.dl_metrics import compute_dl_cohort_metrics
        from start.modeling.tabular_dl import TabularDLClassifier
    except Exception:
        return None

    features = [
        c for c in df.columns
        if c != target and pd.api.types.is_numeric_dtype(df[c])
    ]
    # #2: honor the user's correlation-pruning decision in actual execution.
    pruned_features: list[str] = []
    if apply_correlation_pruning and len(features) > 2:
        corr = df[features].corr().abs()
        dropped = set()
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            if a in dropped:
                continue
            for b in cols[i + 1:]:
                if b in dropped:
                    continue
                if corr.loc[a, b] > 0.95:
                    dropped.add(b)
        if dropped:
            pruned_features = sorted(dropped)
            features = [c for c in features if c not in dropped]
    if len(features) < 2:
        return None

    splits = _stratified_split(df, target, split_props, seed)
    if any(len(f) == 0 for f in splits.values()):
        return None

    out_dir = Path(output_root) / "copilot" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result = CopilotExecution(feature_columns=features)
    result.pruned_features = pruned_features

    # --- split distribution table (Section D) ---
    result.split_table = _split_rows(splits, target)
    split_csv = out_dir / "split_distribution.csv"
    pd.DataFrame(result.split_table).to_csv(split_csv, index=False)
    split_json = out_dir / "split_distribution.json"
    split_json.write_text(json.dumps(result.split_table, indent=2))
    result.artifacts += [str(split_csv), str(split_json)]

    # --- train (Section I) ---
    clf = TabularDLClassifier(
        task="binary_classification", family="mlp", epochs=8, random_state=seed
    )
    clf.fit(splits["train"][features], splits["train"][target])

    # --- metrics by split (Section J) ---
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        precision_score,
        recall_score,
    )
    for name, frame in splits.items():
        proba = clf.predict_proba(frame[features])[:, 1]
        y_true = frame[target].to_numpy()
        m = compute_dl_cohort_metrics(y_true, proba)
        preds = (proba >= 0.5).astype(int)
        # add the full binary metric set the reviewer expects
        try:
            m["pr_auc"] = round(float(average_precision_score(y_true, proba)), 6)
        except Exception:
            m["pr_auc"] = float("nan")
        m["precision"] = round(float(precision_score(y_true, preds, zero_division=0)), 6)
        m["recall"] = round(float(recall_score(y_true, preds, zero_division=0)), 6)
        try:
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            m["specificity"] = round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 6)
            m["confusion_matrix"] = [int(tn), int(fp), int(fn), int(tp)]
        except Exception:
            m["specificity"] = float("nan")
        result.metrics_by_split[name] = m
    metrics_csv = out_dir / "metrics_by_split.csv"
    scalar_metrics = {
        split: {k: v for k, v in m.items() if not isinstance(v, list)}
        for split, m in result.metrics_by_split.items()
    }
    pd.DataFrame(scalar_metrics).T.to_csv(metrics_csv)
    result.artifacts.append(str(metrics_csv))
    # confusion matrices exported separately (Section J)
    cm_rows = [
        {"split": s, "tn": m["confusion_matrix"][0], "fp": m["confusion_matrix"][1],
         "fn": m["confusion_matrix"][2], "tp": m["confusion_matrix"][3]}
        for s, m in result.metrics_by_split.items() if "confusion_matrix" in m
    ]
    if cm_rows:
        cm_csv = out_dir / "confusion_matrix.csv"
        pd.DataFrame(cm_rows).to_csv(cm_csv, index=False)
        result.artifacts.append(str(cm_csv))
    if "train" in result.metrics_by_split and "oos" in result.metrics_by_split:
        m = metric_name if metric_name in result.metrics_by_split["train"] else "auc_roc"
        result.generalization_gap = round(
            result.metrics_by_split["train"][m] - result.metrics_by_split["oos"][m], 6
        )

    # --- training diagnostics (Section I) ---
    history = getattr(clf, "history_", None) or {}
    result.training_diagnostics = {
        "device": getattr(clf, "device_used", "cpu"),
        "best_epoch": getattr(clf, "best_epoch_", None),
        "stopped_early": getattr(clf, "stopped_early_", False),
        "epochs_run": len(history.get("train_loss", [])) if history else None,
        "architecture": "mlp",
    }
    train_json = out_dir / "training_summary.json"
    train_json.write_text(json.dumps(result.training_diagnostics, indent=2, default=str))
    result.artifacts.append(str(train_json))
    if history:
        hist_csv = out_dir / "training_history.csv"
        pd.DataFrame(history).to_csv(hist_csv, index_label="epoch")
        result.artifacts.append(str(hist_csv))

    # --- global explainability table (Section K) ---
    try:
        imp = dl_global_importance(
            clf, splits["test"][features], splits["test"][target].to_numpy(),
            prefer=explain_method, seed=seed,
        )
        result.explainability_method = imp.method
        result.explainability_available = imp.available_methods
        ranked = imp.global_importance[:20]
        result.global_importance = [
            {"rank": i + 1, "feature": f, "importance": round(float(v), 6),
             "direction": "positive" if v >= 0 else "negative"}
            for i, (f, v) in enumerate(ranked)
        ]
        imp_csv = out_dir / "global_feature_importance.csv"
        pd.DataFrame(result.global_importance).to_csv(imp_csv, index=False)
        result.artifacts.append(str(imp_csv))
    except Exception:
        result.explainability_method = "unavailable"

    if registry is not None:
        for path in result.artifacts:
            category = (
                "split" if "split" in path else
                "metrics" if "metrics" in path else
                "training" if "training" in path else
                "explainability" if "importance" in path else "execution"
            )
            registry.register(path, category=category)

    return result


def render_copilot_execution_markdown(ex: CopilotExecution) -> str:
    lines = ["### Train/Test/OOS split", "", "| Split | Rows | Percent | Positive rate | Negative rate |",
             "| --- | --- | --- | --- | --- |"]
    for r in ex.split_table:
        lines.append(
            f"| {r['split']} | {r['rows']} | {r['percent']}% "
            f"| {r['positive_rate']} | {r['negative_rate']} |"
        )
    if ex.metrics_by_split:
        keys = ["auc_roc", "pr_auc", "accuracy", "precision", "recall", "f1", "brier_score", "ece"]
        lines += ["", "### Metrics by split", "",
                  "| Split | " + " | ".join(keys) + " |",
                  "| --- " * (len(keys) + 1) + "|"]
        for split, m in ex.metrics_by_split.items():
            cells = " | ".join(f"{m.get(k, float('nan')):.4f}" for k in keys)
            lines.append(f"| {split} | {cells} |")
        if ex.generalization_gap is not None:
            lines += ["", f"Generalization gap (train - OOS): {ex.generalization_gap:.4f}"]
    if ex.global_importance:
        lines += ["", f"### Global explainability ({ex.explainability_method})", "",
                  "| Rank | Feature | Importance | Direction |", "| --- | --- | --- | --- |"]
        for r in ex.global_importance:
            lines.append(
                f"| {r['rank']} | {r['feature']} | {r['importance']} | {r['direction']} |"
            )
    return "\n".join(lines) + "\n"
