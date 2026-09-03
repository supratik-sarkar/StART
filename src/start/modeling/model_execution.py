"""Model execution artifacts (v2.1.1 remediation Sections D/G/I/J/K).

When the enterprise review runs with training enabled, this produces the
visible, exportable tables the reviewer needs — split distribution, metrics by
split (with calibration), training diagnostics, and a global explainability
table — and writes them as CSV/JSON artifacts registered in the
ArtifactRegistry.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelExecution:
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
    oos_y_true: Any = None
    oos_scores: Any = None
    benchmark: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_table": self.split_table,
            "metrics_by_split": self.metrics_by_split,
            "training_diagnostics": self.training_diagnostics,
            "explainability_method": self.explainability_method,
            "global_importance": self.global_importance,
            "explainability_available": self.explainability_available,
            "generalization_gap": self.generalization_gap,
            "benchmark": self.benchmark,
        }


def _stratified_split(
    df: pd.DataFrame, target: str, props: tuple[float, float, float], seed: int, stratify: bool = True
) -> dict[str, pd.DataFrame]:
    """Train/test/OOS split honoring user proportions, with optional stratification."""
    train_p, test_p, _ = props

    # If the task is regression, or classes < 2, stratify is not possible
    if stratify and target in df.columns and df[target].dropna().nunique() >= 2:
        try:
            rng = np.random.default_rng(seed)
            parts: dict[str, list[pd.DataFrame]] = {"train": [], "test": [], "oos": []}
            for _, grp in df.groupby(target, observed=True):
                idx = grp.index.to_numpy().copy()
                rng.shuffle(idx)
                n = len(idx)
                n_tr = int(round(n * train_p))
                n_te = int(round(n * test_p))
                parts["train"].append(grp.loc[idx[:n_tr]])
                parts["test"].append(grp.loc[idx[n_tr : n_tr + n_te]])
                parts["oos"].append(grp.loc[idx[n_tr + n_te :]])
            return {
                k: pd.concat(v).sample(frac=1.0, random_state=seed) if v else pd.DataFrame()
                for k, v in parts.items()
            }
        except Exception:
            pass  # Fall back to random split

    rng = np.random.default_rng(seed)
    idx = df.index.to_numpy().copy()
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(round(n * train_p))
    n_te = int(round(n * test_p))
    return {
        "train": df.loc[idx[:n_tr]].sample(frac=1.0, random_state=seed) if n_tr > 0 else pd.DataFrame(),
        "test": df.loc[idx[n_tr : n_tr + n_te]].sample(frac=1.0, random_state=seed)
        if n_te > 0
        else pd.DataFrame(),
        "oos": df.loc[idx[n_tr + n_te :]].sample(frac=1.0, random_state=seed)
        if (n - n_tr - n_te) > 0
        else pd.DataFrame(),
    }


def _split_rows(splits: dict[str, pd.DataFrame], target: str) -> list[dict[str, Any]]:
    total = sum(len(f) for f in splits.values()) or 1
    rows = []
    for name, frame in splits.items():
        n = len(frame)
        try:
            pos = float((frame[target] == frame[target].max()).mean()) if n else 0.0
        except Exception:
            pos = 0.0
        rows.append(
            {
                "split": name,
                "rows": n,
                "percent": round(100.0 * n / total, 2),
                "positive_rate": round(pos, 4),
                "negative_rate": round(1.0 - pos, 4),
            }
        )
    return rows


def run_model_execution(
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
    architecture: str = "mlp",
    stratify: bool = True,
    class_weight: str | None = None,
    task_type: str = "binary_classification",
    activation: str | None = None,
    winsorize: bool = False,
    custom_space: dict[str, Any] | None = None,
    costlier_errors: str | None = None,
    tuning_params: dict[str, Any] | None = None,
) -> ModelExecution | None:
    """Train a tabular model and emit the visible tables + artifacts."""
    from start.modeling.tuning_run import _model_family

    family = _model_family(architecture)
    try:
        from start.modeling.models import resolve_model

        if family != "sklearn":
            from start.modeling.deep_learning import torch_available

            if not torch_available():
                return None
            if family == "tabular_dl":
                from start.modeling.tabular_dl import TabularDLClassifier
            elif family == "sequence_dl":
                from start.modeling.sequence_dl import SequenceClassifier
            elif family == "vision_dl":
                from start.modeling.vision_dl import VisionCNNClassifier
        from start.modeling.dl_explain import dl_global_importance
        from start.modeling.tabular_dl_metrics import dl_task_metrics
    except Exception:
        return None

    features = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    # honor the user's correlation-pruning decision in actual execution.
    pruned_features: list[str] = []
    if apply_correlation_pruning and len(features) > 2:
        corr = df[features].corr().abs()
        dropped = set()
        cols = list(corr.columns)
        for i, a in enumerate(cols):
            if a in dropped:
                continue
            for b in cols[i + 1 :]:
                if b in dropped:
                    continue
                if corr.loc[a, b] > 0.95:
                    dropped.add(b)
        if dropped:
            pruned_features = sorted(dropped)
            features = [c for c in features if c not in dropped]
    if len(features) < 2:
        return None

    # For regression, disable stratification
    effective_stratify = stratify and (task_type not in ("regression", "forecasting"))
    splits = _stratified_split(df, target, split_props, seed, stratify=effective_stratify)
    if any(len(f) == 0 for f in splits.values()):
        return None

    if family == "sklearn":
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="median")
        for k in ("train", "test", "oos"):
            splits[k] = splits[k].copy()
        splits["train"][features] = imputer.fit_transform(splits["train"][features])
        if len(splits["test"]):
            splits["test"][features] = imputer.transform(splits["test"][features])
        if len(splits["oos"]):
            splits["oos"][features] = imputer.transform(splits["oos"][features])

    out_dir = Path(output_root) / "model_execution" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    result = ModelExecution(feature_columns=features)
    result.pruned_features = pruned_features

    # --- split distribution table (Section D) ---
    result.split_table = _split_rows(splits, target)
    split_csv = out_dir / "split_distribution.csv"
    pd.DataFrame(result.split_table).to_csv(split_csv, index=False)
    split_json = out_dir / "split_distribution.json"
    split_json.write_text(json.dumps(result.split_table, indent=2))
    result.artifacts += [str(split_csv), str(split_json)]

    # --- train (Section I) ---
    if family == "tabular_dl":
        kwargs = {
            "task": task_type,
            "family": architecture,
            "random_state": seed,
            "activation": activation or "relu",
            "epochs": 10,
            "early_stopping_patience": 3,
        }
        if class_weight:
            kwargs["class_weight"] = class_weight
        if custom_space:
            scalar_space = {}
            for k, v in custom_space.items():
                scalar_space[k] = v[0] if isinstance(v, list) else v
            kwargs.update(scalar_space)
        clf = TabularDLClassifier(**kwargs)
        if winsorize:
            from scipy.stats import mstats

            X_train = splits["train"][features].copy()
            for col in features:
                X_train[col] = mstats.winsorize(X_train[col], limits=[0.01, 0.01])
            clf.fit(X_train, splits["train"][target])
        else:
            clf.fit(splits["train"][features], splits["train"][target])
    elif family == "sequence_dl":
        kwargs = {
            "task": task_type,
            "family": architecture,
            "epochs": 10,
            "early_stopping_patience": 3,
            "random_state": seed,
        }
        if custom_space:
            scalar_space = {}
            for k, v in custom_space.items():
                scalar_space[k] = v[0] if isinstance(v, list) else v
            kwargs.update(scalar_space)
        clf = SequenceClassifier(**kwargs)
        clf.fit(splits["train"][features], splits["train"][target])
    elif family == "vision_dl":
        kwargs = {
            "task": task_type,
            "architecture": "simple_cnn_small" if architecture == "cnn" else architecture,
            "epochs": 10,
            "early_stopping_patience": 3,
            "random_state": seed,
        }
        if custom_space:
            scalar_space = {}
            for k, v in custom_space.items():
                scalar_space[k] = v[0] if isinstance(v, list) else v
            kwargs.update(scalar_space)
        clf = VisionCNNClassifier(**kwargs)
        clf.fit(splits["train"][features], splits["train"][target])
    else:
        clf, _, _ = resolve_model(architecture, seed)
        fit_kwargs = {}
        if class_weight and hasattr(clf, "class_weight"):
            try:
                clf.set_params(class_weight=class_weight)
            except Exception:
                pass
        elif class_weight == "balanced":
            try:
                from sklearn.utils.class_weight import compute_sample_weight

                sample_weight = compute_sample_weight("balanced", splits["train"][target])
                fit_kwargs["sample_weight"] = sample_weight
            except Exception:
                pass
        if custom_space:
            scalar_space = {}
            for k, v in custom_space.items():
                scalar_space[k] = v[0] if isinstance(v, list) else v
            try:
                clf.set_params(**scalar_space)
            except Exception:
                pass
        clf.fit(splits["train"][features], splits["train"][target], **fit_kwargs)

    from start.modeling.config_propagation import audit_propagation

    class _CapturedConfig:
        pass

    _cfg = _CapturedConfig()
    _cfg.class_weight = class_weight
    _cfg.architecture_family = architecture
    _cfg.activation = activation
    _cfg.seed = seed
    _cfg.explain_method = explain_method
    _cfg.stratify = stratify
    _cfg.train_prop = split_props[0] if split_props else None
    _cfg.tuning_strategy = (tuning_params or {}).get("strategy")
    _cfg.tuning_trials = (tuning_params or {}).get("trials")
    _cfg.validation_scheme = (tuning_params or {}).get("validation")
    _cfg.costlier_errors = costlier_errors

    propagation = audit_propagation(
        _cfg,
        {
            "estimator": clf,
            "fit_kwargs": locals().get("fit_kwargs", {}),
            "model_params": locals().get("kwargs", {}),
            "split_plan": result.split_table,
            "split_params": {
                "stratify": "stratified" if stratify else "random",
                "train_prop": split_props[0] if split_props else 0.60,
                "seed": seed,
            },
            "tuning": locals().get("tuning_summary", {}),
            "tuning_params": tuning_params or {},
            "explainability": {"method": explain_method},
            "metric_routing": {"primary": metric_name},
        },
    )
    for line in propagation.summary_lines():
        print(line)

    # --- metrics by split (Section J) ---
    from sklearn.metrics import (
        confusion_matrix,
        precision_score,
        recall_score,
    )

    for name, frame in splits.items():
        y_true = frame[target].to_numpy()
        if task_type in ("regression", "forecasting"):
            preds = clf.predict(frame[features])
            m = dl_task_metrics(task_type, y_true, preds)
        else:
            proba = clf.predict_proba(frame[features])
            m = dl_task_metrics(task_type, y_true, proba, classes=getattr(clf, "classes_", None))
            if task_type == "binary_classification":
                p1 = proba[:, 1]
                preds = (p1 >= 0.5).astype(int)
                m["precision"] = round(float(precision_score(y_true, preds, zero_division=0)), 6)
                m["recall"] = round(float(recall_score(y_true, preds, zero_division=0)), 6)
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
                    m["specificity"] = round(float(tn / (tn + fp)) if (tn + fp) else 0.0, 6)
                    m["confusion_matrix"] = [int(tn), int(fp), int(fn), int(tp)]
                except Exception:
                    m["specificity"] = float("nan")
        result.metrics_by_split[name] = m

    if "oos" in splits and len(splits["oos"]):
        oos_frame = splits["oos"]
        result.oos_y_true = oos_frame[target].to_numpy()
        if task_type in ("regression", "forecasting"):
            result.oos_scores = clf.predict(oos_frame[features])
        else:
            proba = clf.predict_proba(oos_frame[features])
            result.oos_scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    elif "test" in splits and len(splits["test"]):
        te_frame = splits["test"]
        result.oos_y_true = te_frame[target].to_numpy()
        if task_type in ("regression", "forecasting"):
            result.oos_scores = clf.predict(te_frame[features])
        else:
            proba = clf.predict_proba(te_frame[features])
            result.oos_scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]

    # --- benchmarking (Section 6) ---
    if result.oos_scores is not None and task_type == "binary_classification":
        try:
            from start.modeling.benchmark import benchmark_against_baselines

            oos_split = "oos" if ("oos" in splits and len(splits["oos"])) else "test"
            if oos_split in splits and len(splits[oos_split]) > 0:
                X_tr = splits["train"][features]
                y_tr = splits["train"][target].to_numpy()
                X_eval = splits[oos_split][features]
                y_eval = splits[oos_split][target].to_numpy()
                if len(result.oos_scores) == len(y_eval):
                    bench = benchmark_against_baselines(
                        X_tr, y_tr, X_eval, y_eval, result.oos_scores, seed=seed
                    )
                    for line in bench.summary_lines():
                        print(line)
                    result.benchmark = bench.as_dict()
                    bench_json = out_dir / "benchmark.json"
                    bench_json.write_text(json.dumps(result.benchmark, indent=2))
                    result.artifacts.append(str(bench_json))
        except Exception:
            pass

    metrics_csv = out_dir / "metrics_by_split.csv"
    scalar_metrics = {
        split: {k: v for k, v in m.items() if not isinstance(v, list)}
        for split, m in result.metrics_by_split.items()
    }
    pd.DataFrame(scalar_metrics).T.to_csv(metrics_csv)
    result.artifacts.append(str(metrics_csv))

    # confusion matrices exported separately (Section J)
    cm_rows = [
        {
            "split": s,
            "tn": m["confusion_matrix"][0],
            "fp": m["confusion_matrix"][1],
            "fn": m["confusion_matrix"][2],
            "tp": m["confusion_matrix"][3],
        }
        for s, m in result.metrics_by_split.items()
        if "confusion_matrix" in m
    ]
    if cm_rows:
        cm_csv = out_dir / "confusion_matrix.csv"
        pd.DataFrame(cm_rows).to_csv(cm_csv, index=False)
        result.artifacts.append(str(cm_csv))

    if "train" in result.metrics_by_split and "oos" in result.metrics_by_split:
        m = (
            metric_name
            if metric_name in result.metrics_by_split["train"]
            else ("rmse" if task_type in ("regression", "forecasting") else "auc_roc")
        )
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
        "architecture": architecture,
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
            clf,
            splits["test"][features],
            splits["test"][target].to_numpy(),
            prefer=explain_method,
            seed=seed,
        )
        result.explainability_method = imp.method
        result.explainability_available = imp.available_methods
        ranked = imp.global_importance[:20]
        result.global_importance = [
            {
                "rank": i + 1,
                "feature": f,
                "importance": round(float(v), 6),
                "direction": "positive" if v >= 0 else "negative",
            }
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
                "split"
                if "split" in path
                else "metrics"
                if "metrics" in path
                else "training"
                if "training" in path
                else "explainability"
                if "importance" in path
                else "execution"
            )
            registry.register(path, category=category)

    return result


def render_model_execution_markdown(ex: ModelExecution) -> str:
    lines = [
        "### Train/Test/OOS split",
        "",
        "| Split | Rows | Percent | Positive rate | Negative rate |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in ex.split_table:
        lines.append(
            f"| {r['split']} | {r['rows']} | {r['percent']}% | {r['positive_rate']} | {r['negative_rate']} |"
        )
    if ex.metrics_by_split:
        first_split = next(iter(ex.metrics_by_split.values()))
        keys = [
            k for k in first_split.keys() if k != "confusion_matrix" and not isinstance(first_split[k], list)
        ]
        lines += [
            "",
            "### Metrics by split",
            "",
            "| Split | " + " | ".join(keys) + " |",
            "| --- " * (len(keys) + 1) + "|",
        ]
        for split, m in ex.metrics_by_split.items():
            cells = " | ".join(
                f"{m.get(k, float('nan')):.4f}" if isinstance(m.get(k), (int, float)) else str(m.get(k))
                for k in keys
            )
            lines.append(f"| {split} | {cells} |")
        if ex.generalization_gap is not None:
            lines += ["", f"Generalization gap (train - OOS): {ex.generalization_gap:.4f}"]
    if ex.global_importance:
        lines += [
            "",
            f"### Global explainability ({ex.explainability_method})",
            "",
            "| Rank | Feature | Importance | Direction |",
            "| --- | --- | --- | --- |",
        ]
        for r in ex.global_importance:
            lines.append(f"| {r['rank']} | {r['feature']} | {r['importance']} | {r['direction']} |")
    return "\n".join(lines) + "\n"
