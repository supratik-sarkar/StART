"""Real hyperparameter tuning (v2.1.1 remediation Section H).

Runs an actual bounded randomized search: each trial trains a model on a
train-internal split and is scored on an internal holdout (never test/OOS — no
leakage).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TuningTrial:
    trial: int
    params: dict[str, Any]
    validation_metric: float
    status: str  # "best" | "ok" | "rejected"

    def to_dict(self) -> dict[str, Any]:
        return {
            "trial": self.trial,
            "params": self.params,
            "validation_metric": self.validation_metric,
            "status": self.status,
        }


@dataclass
class TuningRun:
    strategy: str
    primary_metric: str
    n_trials: int
    trials: list[TuningTrial] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)
    best_metric: float = 0.0
    rejected_params: list[dict[str, Any]] = field(default_factory=list)
    search_space: dict[str, list[Any]] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    ran: bool = True
    note: str = ""
    validation: str = "holdout"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "primary_metric": self.primary_metric,
            "n_trials": self.n_trials,
            "ran": self.ran,
            "trials": [t.to_dict() for t in self.trials],
            "best_params": self.best_params,
            "best_metric": self.best_metric,
            "rejected_params": self.rejected_params,
            "search_space": self.search_space,
            "note": self.note,
            "validation": self.validation,
        }


_TABULAR_DL_SEARCH_SPACE = {
    "learning_rate": [1e-3, 3e-3, 1e-2],
    "hidden_dims": [(32,), (64, 32), (128, 64)],
    "dropout": [0.0, 0.1, 0.2],
}

_SEQUENCE_DL_SEARCH_SPACE = {
    "learning_rate": [1e-3, 3e-3, 1e-2],
    "hidden_size": [16, 32, 64],
    "dropout": [0.0, 0.1, 0.2],
}

_VISION_DL_SEARCH_SPACE = {
    "learning_rate": [1e-3, 3e-3, 1e-2],
    "batch_size": [32, 64],
}

_SEQUENCE_FAMILIES = ()  # Tabular recurrent nets route to tabular_dl
_VISION_FAMILIES = ("simple_cnn_small", "simple_cnn_medium", "simple_cnn_deep")
_TABULAR_DL_FAMILIES = (
    "mlp",
    "wide_deep",
    "residual_mlp",
    "dcn",
    "leaky_relu_mlp",
    "rnn",
    "lstm",
    "gru",
    "bi_lstm",
    "cnn",
    "gnn",
)


def _model_family(architecture: str) -> str:
    """Classify architecture into 'tabular_dl', 'sequence_dl', 'vision_dl', or 'sklearn'."""
    if architecture in _TABULAR_DL_FAMILIES:
        return "tabular_dl"
    if architecture in _VISION_FAMILIES:
        return "vision_dl"
    return "sklearn"


def _scorer(metric_name: str, is_multiclass: bool = False, classes_order: Any = None):
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
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

    def f1(y, p):
        if is_multiclass:
            yhat = classes_order[np.argmax(p, axis=1)]
            return float(f1_score(y, yhat, average="macro", labels=classes_order, zero_division=0))
        else:
            yhat = (np.asarray(p) >= 0.5).astype(int)
            return float(f1_score(y, yhat, zero_division=0))

    def rmse(y, p):
        return float(np.sqrt(mean_squared_error(y, p)))

    def mae(y, p):
        return float(mean_absolute_error(y, p))

    def r2(y, p):
        return float(r2_score(y, p))

    return {"auc_roc": auc, "pr_auc": prauc, "recall": rec, "f1": f1, "rmse": rmse, "mae": mae, "r2": r2}.get(
        metric_name, auc
    )


def run_tuning(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    *,
    strategy: str = "bounded_random_search",
    n_trials: int = 5,
    primary_metric: str = "auc_roc",
    seed: int = 42,
    output_root: str = "start_output",
    run_id: str = "RUN",
    registry: Any = None,
    architecture: str = "mlp",
    activation: str | None = None,
    task_type: str = "binary_classification",
    custom_space: dict[str, Any] | None = None,
    validation: str = "holdout",
    k_folds: int = 3,
    cost_specification: dict[str, Any] | None = None,
) -> TuningRun | None:
    """Execute the tuning search. Returns None if disabled or torch missing for DL path."""
    if strategy == "none":
        return TuningRun(
            strategy="none",
            primary_metric=primary_metric,
            n_trials=0,
            ran=False,
            note="Hyperparameter tuning disabled by user.",
            validation=validation,
        )

    family = _model_family(architecture)

    try:
        from start.modeling.models import HYPERPARAM_SPACES, resolve_model

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
    except Exception:
        return None

    if len(features) < 2 or len(df) < 40:
        return None

    metric_name = (
        primary_metric
        if primary_metric in ("auc_roc", "pr_auc", "recall", "f1", "rmse", "mae", "r2")
        else ("rmse" if task_type in ("regression", "forecasting") else "auc_roc")
    )

    rng = np.random.default_rng(seed)

    # train-internal holdout (stratified) — never touches test/OOS
    if task_type not in ("regression", "forecasting") and df[target].nunique() > 1:
        from sklearn.model_selection import train_test_split

        tr, va = train_test_split(df, test_size=0.25, random_state=seed, stratify=df[target])
    else:
        idx = df.index.to_numpy().copy()
        rng.shuffle(idx)
        cut = int(len(idx) * 0.75)
        tr, va = df.loc[idx[:cut]], df.loc[idx[cut:]]
    if family == "sklearn":
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="median")
        tr = tr.copy()
        va = va.copy()
        tr[features] = imputer.fit_transform(tr[features])
        va[features] = imputer.transform(va[features])
    if len(va) < 5:
        return None
    if task_type not in ("regression", "forecasting") and va[target].nunique() < 2:
        return None

    # sample bounded trial configs deterministically per model family
    if family == "tabular_dl":
        if architecture in ("rnn", "lstm", "gru", "bi_lstm"):
            space = {
                "learning_rate": [1e-3, 3e-3, 1e-2],
                "hidden_size": [16, 32, 64],
                "num_layers": [1, 2],
                "dropout": [0.0, 0.1, 0.2],
            }
            if custom_space:
                for k, v in custom_space.items():
                    if k in space:
                        if not isinstance(v, list):
                            space[k] = [v]
                        else:
                            space[k] = v
            combos = [
                {"learning_rate": lr, "hidden_size": hs, "num_layers": nl, "dropout": dr}
                for lr in space["learning_rate"]
                for hs in space["hidden_size"]
                for nl in space["num_layers"]
                for dr in space["dropout"]
            ]
        else:
            space = dict(_TABULAR_DL_SEARCH_SPACE)
            if custom_space:
                for k, v in custom_space.items():
                    if not isinstance(v, list):
                        space[k] = [v]
                    else:
                        space[k] = v
            if "hidden_dims" in space:
                hd = space["hidden_dims"]
                if hd and not isinstance(hd[0], (list, tuple)):
                    space["hidden_dims"] = [tuple(hd)]
            combos = [
                {"learning_rate": lr, "hidden_dims": hd, "dropout": dr}
                for lr in space["learning_rate"]
                for hd in space["hidden_dims"]
                for dr in space["dropout"]
            ]
        rng.shuffle(combos)
        n_trials = max(1, min(n_trials, 15, len(combos)))
        chosen = combos[:n_trials]
        run = TuningRun(
            strategy=strategy,
            primary_metric=metric_name,
            n_trials=n_trials,
            search_space={k: [list(x) if isinstance(x, tuple) else x for x in v] for k, v in space.items()},
            validation=validation,
        )
    elif family == "sequence_dl":
        space = dict(_SEQUENCE_DL_SEARCH_SPACE)
        if custom_space:
            for k, v in custom_space.items():
                if not isinstance(v, list):
                    space[k] = [v]
                else:
                    space[k] = v
        combos = [
            {"learning_rate": lr, "hidden_size": hs, "dropout": dr}
            for lr in space["learning_rate"]
            for hs in space["hidden_size"]
            for dr in space["dropout"]
        ]
        rng.shuffle(combos)
        n_trials = max(1, min(n_trials, 15, len(combos)))
        chosen = combos[:n_trials]
        run = TuningRun(
            strategy=strategy,
            primary_metric=metric_name,
            n_trials=n_trials,
            search_space=space,
            validation=validation,
        )
    elif family == "vision_dl":
        space = dict(_VISION_DL_SEARCH_SPACE)
        if custom_space:
            for k, v in custom_space.items():
                if not isinstance(v, list):
                    space[k] = [v]
                else:
                    space[k] = v
        combos = [
            {"learning_rate": lr, "batch_size": bs}
            for lr in space["learning_rate"]
            for bs in space["batch_size"]
        ]
        rng.shuffle(combos)
        n_trials = max(1, min(n_trials, 15, len(combos)))
        chosen = combos[:n_trials]
        run = TuningRun(
            strategy=strategy,
            primary_metric=metric_name,
            n_trials=n_trials,
            search_space=space,
            validation=validation,
        )
    else:
        # sklearn / tree-based models
        orig_space = HYPERPARAM_SPACES.get(architecture, HYPERPARAM_SPACES["random_forest"])
        space = {}
        for k, spec in orig_space.items():
            space[k] = dict(spec)
        if custom_space:
            for k, v in custom_space.items():
                v_list = v if isinstance(v, list) else [v]
                if k in space:
                    if "grid" in space[k]:
                        space[k]["grid"] = v_list
                    elif "choices" in space[k]:
                        space[k]["choices"] = v_list
                    else:
                        valid_vals = [x for x in v_list if x is not None]
                        if valid_vals:
                            space[k]["low"] = min(valid_vals)
                            space[k]["high"] = max(valid_vals)
                else:
                    space[k] = {"type": "cat", "choices": v_list}
        combos = []
        for _ in range(50):
            c = {}
            for k, spec in space.items():
                if "grid" in spec:
                    c[k] = rng.choice(spec["grid"])
                elif "choices" in spec:
                    c[k] = rng.choice(spec["choices"])
                else:
                    if spec.get("type") == "int":
                        val = int(rng.integers(spec["low"], spec["high"] + 1))
                        c[k] = val
                    elif spec.get("type") == "float":
                        c[k] = float(rng.uniform(spec["low"], spec["high"]))
                    else:
                        c[k] = None
            if c not in combos:
                combos.append(c)
        rng.shuffle(combos)
        n_trials = max(1, min(n_trials, 15, len(combos)))
        chosen = combos[:n_trials]
        run = TuningRun(
            strategy=strategy,
            primary_metric=metric_name,
            n_trials=n_trials,
            search_space={
                k: spec.get("grid") or spec.get("choices") or [spec.get("low"), spec.get("high")]
                for k, spec in space.items()
            },
            validation=validation,
        )

    def _instantiate_model(params):
        if family == "tabular_dl":
            h_dims = params.get("hidden_dims")
            if h_dims is None and "hidden_size" in params:
                h_dims = (params["hidden_size"],) * params.get("num_layers", 1)
            return TabularDLClassifier(
                task=task_type,
                family=architecture,
                activation=activation,
                hidden_dims=h_dims,
                epochs=8,
                learning_rate=params["learning_rate"],
                dropout=params["dropout"],
                random_state=seed,
                cost_specification=cost_specification,
            )
        elif family == "sequence_dl":
            return SequenceClassifier(
                family=architecture,
                hidden_size=params["hidden_size"],
                learning_rate=params["learning_rate"],
                dropout=params["dropout"],
                epochs=8,
                random_state=seed,
                cost_specification=cost_specification,
            )
        elif family == "vision_dl":
            arch = "simple_cnn_small" if architecture == "cnn" else architecture
            return VisionCNNClassifier(
                architecture=arch,
                learning_rate=params["learning_rate"],
                batch_size=params["batch_size"],
                epochs=8,
                random_state=seed,
            )
        else:
            clf_obj, _, _ = resolve_model(architecture, seed)
            clf_obj.set_params(**params)
            return clf_obj

    minimize = metric_name in ("rmse", "mae")
    best_metric = float("inf") if minimize else -1.0

    if validation == "k_fold":
        if task_type in ("regression", "forecasting"):
            from sklearn.model_selection import KFold

            kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)
            folds = list(kf.split(df))
        else:
            from sklearn.model_selection import StratifiedKFold

            kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=seed)
            folds = list(kf.split(df, df[target]))
    else:
        folds = None

    from start.progress import progress_bar

    fold_telemetry_rows: list[dict[str, Any]] = []
    total_fit_count = 0
    failed_fit_count = 0

    with progress_bar(len(chosen), f"Tuning {architecture} ({len(chosen)} trials)") as adv:
        for i, params in enumerate(chosen, start=1):
            is_multiclass = task_type == "multiclass_classification"
            trial_failed = False
            fold_status_h = "ok"
            if validation == "k_fold":
                fold_metrics = []
                for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
                    import time as _time
                    import warnings as _warnings

                    fold_t0 = _time.perf_counter()
                    fold_warnings: list[str] = []
                    fold_status = "ok"
                    fold_metric = float("nan")
                    fold_epochs = 0
                    fold_early_stop = ""

                    try:
                        with _warnings.catch_warnings(record=True) as caught:
                            _warnings.simplefilter("always")

                            tr_fold = df.iloc[train_idx].copy()
                            va_fold = df.iloc[val_idx].copy()

                            # Fold-local preprocessing
                            if family == "sklearn":
                                from sklearn.impute import SimpleImputer

                                imputer = SimpleImputer(strategy="median")
                                tr_fold[features] = imputer.fit_transform(tr_fold[features])
                                va_fold[features] = imputer.transform(va_fold[features])
                            else:
                                # Finite-value check for DL models
                                _tr_X = tr_fold[features].to_numpy(dtype=float)
                                _va_X = va_fold[features].to_numpy(dtype=float)
                                _tr_mean = np.nanmean(_tr_X, axis=0)
                                _tr_std = np.nanstd(_tr_X, axis=0)
                                _tr_std[_tr_std == 0] = 1.0
                                _tr_X = np.nan_to_num(_tr_X, nan=0.0)
                                _va_X = np.nan_to_num(_va_X, nan=0.0)
                                _tr_X = (_tr_X - _tr_mean) / _tr_std
                                _va_X = (_va_X - _tr_mean) / _tr_std
                                tr_fold = tr_fold.copy()
                                va_fold = va_fold.copy()
                                tr_fold[features] = _tr_X
                                va_fold[features] = _va_X

                            # Freshly initialized model per fold
                            clf = _instantiate_model(params)
                            total_fit_count += 1
                            clf.fit(tr_fold[features], tr_fold[target])

                            fold_epochs = getattr(clf, "best_epoch_", getattr(clf, "epochs", 0))
                            fold_early_stop = "yes" if getattr(clf, "stopped_early_", False) else "no"

                            classes_order = getattr(clf, "classes_", None)
                            local_score = _scorer(metric_name, is_multiclass, classes_order)
                            if task_type in ("regression", "forecasting"):
                                proba = clf.predict(va_fold[features])
                            else:
                                if is_multiclass:
                                    proba = clf.predict_proba(va_fold[features])
                                else:
                                    proba = clf.predict_proba(va_fold[features])[:, 1]

                            fold_metric = local_score(va_fold[target].to_numpy(), proba)

                            # Check for NaN/inf metric — critical failure
                            if not np.isfinite(fold_metric):
                                fold_status = "failed"
                                fold_warnings.append("CRITICAL: NaN or infinite metric value")
                                trial_failed = True
                            elif (
                                task_type not in ("regression", "forecasting")
                                and (va_fold[target] == 1).sum() < 5
                            ):
                                fold_status = "degenerate"
                                fold_warnings.append(
                                    "Degenerate fold: fewer than 5 positive cases in validation fold"
                                )
                                trial_failed = True

                        # Classify and deduplicate captured warnings
                        _seen_warnings: set[str] = set()
                        _critical = False
                        for w in caught:
                            key = f"{w.category.__name__}: {str(w.message)}"
                            if key not in _seen_warnings:
                                _seen_warnings.add(key)
                                fold_warnings.append(key)
                            # Critical warning classification
                            if any(
                                kw in str(w.message).lower()
                                for kw in ("overflow", "invalid value", "divide by zero", "nan", "non-finite")
                            ):
                                _critical = True
                        if _critical:
                            fold_status = "failed"
                            trial_failed = True

                    except Exception as exc:
                        fold_status = "failed"
                        fold_warnings.append(f"EXCEPTION: {type(exc).__name__}: {exc}")
                        trial_failed = True
                        failed_fit_count += 1

                    fold_runtime = _time.perf_counter() - fold_t0

                    fold_telemetry_rows.append(
                        {
                            "trial_id": i,
                            "fold_id": fold_idx,
                            "model_family": architecture,
                            "params": str(params),
                            "preprocessing_scope": "fold-local",
                            "epochs_completed": fold_epochs,
                            "early_stopping": fold_early_stop,
                            "device": str(getattr(clf, "_device_used", "cpu") if "clf" in dir() else "cpu"),
                            "runtime_seconds": fold_runtime,
                            "metric_name": metric_name,
                            "metric_value": fold_metric,
                            "status": fold_status,
                            "warnings": "; ".join(fold_warnings) if fold_warnings else "",
                            "failure_reason": fold_warnings[0]
                            if fold_status == "failed" and fold_warnings
                            else "",
                        }
                    )

                    if fold_status == "ok":
                        fold_metrics.append(fold_metric)

                # Trial result: failed if any fold failed
                if trial_failed or not fold_metrics:
                    metric = float("nan")
                else:
                    metric = round(float(np.mean(fold_metrics)), 6)
            else:
                # Holdout validation (original path)
                import time as _time
                import warnings as _warnings

                fold_t0 = _time.perf_counter()
                fold_warnings_list: list[str] = []
                fold_status_h = "ok"

                try:
                    with _warnings.catch_warnings(record=True) as caught:
                        _warnings.simplefilter("always")
                        clf = _instantiate_model(params)
                        total_fit_count += 1
                        clf.fit(tr[features], tr[target])
                        classes_order = getattr(clf, "classes_", None)
                        local_score = _scorer(metric_name, is_multiclass, classes_order)
                        if task_type in ("regression", "forecasting"):
                            proba = clf.predict(va[features])
                        else:
                            if is_multiclass:
                                proba = clf.predict_proba(va[features])
                            else:
                                proba = clf.predict_proba(va[features])[:, 1]
                        metric = round(local_score(va[target].to_numpy(), proba), 6)

                        if not np.isfinite(metric):
                            fold_status_h = "failed"
                            fold_warnings_list.append("CRITICAL: NaN or infinite metric value")
                        elif task_type not in ("regression", "forecasting") and (va[target] == 1).sum() < 5:
                            fold_status_h = "degenerate"
                            fold_warnings_list.append(
                                "Degenerate holdout: fewer than 5 positive cases in validation set"
                            )

                    _seen = set()
                    for w in caught:
                        key = f"{w.category.__name__}: {str(w.message)}"
                        if key not in _seen:
                            _seen.add(key)
                            fold_warnings_list.append(key)

                except Exception as exc:
                    metric = float("nan")
                    fold_status_h = "failed"
                    fold_warnings_list.append(f"EXCEPTION: {type(exc).__name__}: {exc}")
                    failed_fit_count += 1

                fold_runtime_h = _time.perf_counter() - fold_t0
                fold_telemetry_rows.append(
                    {
                        "trial_id": i,
                        "fold_id": 1,
                        "model_family": architecture,
                        "params": str(params),
                        "preprocessing_scope": "holdout-split",
                        "epochs_completed": getattr(clf, "best_epoch_", 0) if "clf" in dir() else 0,
                        "early_stopping": "yes"
                        if getattr(clf, "stopped_early_", False)
                        else "no"
                        if "clf" in dir()
                        else "",
                        "device": str(getattr(clf, "_device_used", "cpu") if "clf" in dir() else "cpu"),
                        "runtime_seconds": fold_runtime_h,
                        "metric_name": metric_name,
                        "metric_value": metric,
                        "status": fold_status_h,
                        "warnings": "; ".join(fold_warnings_list) if fold_warnings_list else "",
                        "failure_reason": fold_warnings_list[0]
                        if fold_status_h == "failed" and fold_warnings_list
                        else "",
                    }
                )

            # Only consider non-NaN and non-degenerate trials for best
            is_valid = (
                (np.isfinite(metric) if isinstance(metric, float) else True)
                and fold_status_h != "degenerate"
                and not trial_failed
            )
            if is_valid:
                is_best = (metric < best_metric) if minimize else (metric > best_metric)
                if is_best:
                    best_metric = metric

            # Ensure hidden_dims remains a list if stored in params for deep learning
            saved_params = dict(params)
            if "hidden_dims" in saved_params and isinstance(saved_params["hidden_dims"], tuple):
                saved_params["hidden_dims"] = list(saved_params["hidden_dims"])

            t_status = (
                "degenerate"
                if (fold_status_h == "degenerate" or trial_failed)
                else ("ok" if is_valid else "failed")
            )
            run.trials.append(
                TuningTrial(
                    trial=i,
                    params=saved_params,
                    validation_metric=metric if is_valid else float("nan"),
                    status=t_status,
                )
            )
            adv(1)

    # Print concise warning summary
    _all_warnings = [r["warnings"] for r in fold_telemetry_rows if r["warnings"]]
    if _all_warnings:
        _unique_warnings: set[str] = set()
        for w_str in _all_warnings:
            for w in w_str.split("; "):
                _unique_warnings.add(w)
        import sys

        print(
            f"  [Tuning Warning Summary] {len(_unique_warnings)} unique warning(s) "
            f"across {len(fold_telemetry_rows)} fold(s):",
            file=sys.stderr,
        )
        for uw in sorted(_unique_warnings)[:10]:
            print(f"    - {uw}", file=sys.stderr)
        if len(_unique_warnings) > 10:
            print(f"    ... and {len(_unique_warnings) - 10} more", file=sys.stderr)

    # mark best + rejected — single best trial only
    found_best = False
    for t in run.trials:
        if t.status in ("failed", "degenerate"):
            run.rejected_params.append(t.params)
            continue
        is_best = not found_best and t.validation_metric == best_metric
        if is_best:
            t.status = "best"
            run.best_params = t.params
            run.best_metric = t.validation_metric
            found_best = True
        else:
            t.status = "ok"
            run.rejected_params.append(t.params)

    # write artifacts
    out_dir = Path(output_root) / "tuning" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Trial-level summary CSV
    row_dicts = []
    for t in run.trials:
        d = {"trial": t.trial}
        for k, v in t.params.items():
            if isinstance(v, (list, tuple)):
                d[k] = str(list(v))
            else:
                d[k] = v
        d["validation_metric"] = t.validation_metric
        d["status"] = t.status
        row_dicts.append(d)

    trials_csv = out_dir / "tuning_trials.csv"
    pd.DataFrame(row_dicts).to_csv(trials_csv, index=False)

    # v3.1.1: Fold-level telemetry CSV
    folds_csv = out_dir / "tuning_folds.csv"
    pd.DataFrame(fold_telemetry_rows).to_csv(folds_csv, index=False)

    summary_json = out_dir / "tuning_summary.json"
    summary_json.write_text(json.dumps(run.to_dict(), indent=2, default=str))

    run.artifacts = [str(trials_csv), str(folds_csv), str(summary_json)]
    if registry is not None:
        for a_path in run.artifacts:
            registry.register(a_path, category="tuning")

    return run


def render_tuning_run_markdown(run: TuningRun) -> str:
    """Markdown table for dashboard/transcript/notebook."""
    if not run.ran:
        return f"### Hyperparameter tuning\n\n{run.note}\n"

    lines = [
        "### Hyperparameter tuning",
        "",
        f"- Strategy: {run.strategy}",
        f"- Primary metric: {run.primary_metric}",
        f"- Number of trials: {run.n_trials}",
        "",
    ]

    # collect all param keys present across all trials
    param_keys = []
    for t in run.trials:
        for k in t.params.keys():
            if k not in param_keys:
                param_keys.append(k)

    # Build markdown table header
    header_cols = ["Trial"] + param_keys + ["Validation metric", "Status"]
    header_row = "| " + " | ".join(header_cols) + " |"
    separator_row = "| " + " | ".join(["---"] * len(header_cols)) + " |"
    lines.append(header_row)
    lines.append(separator_row)

    # Build rows
    for t in run.trials:
        row_vals = [str(t.trial)]
        for k in param_keys:
            val = t.params.get(k, "-")
            row_vals.append(str(val))
        row_vals.append(f"{t.validation_metric:.6f}")
        status_str = f"**{t.status}**" if t.status == "best" else t.status
        row_vals.append(status_str)
        lines.append("| " + " | ".join(row_vals) + " |")

    lines.append("")
    if run.best_params:
        lines.append(f"**Best Parameters:** `{run.best_params}` with metric `{run.best_metric:.6f}`")
    lines.append("")

    return "\n".join(lines)
