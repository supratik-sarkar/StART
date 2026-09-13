"""Hyperparameter tuning: none / grid / random / Bayesian (Optuna).

Grid and random search use sklearn only. Optuna is optional: if it is not
installed, tuning degrades to the default model with an explicit message
telling the user how to enable it (`pip install -e ".[optuna]"`). Hyperopt is
intentionally not the first-supported backend; Optuna is preferred.

Cross-validation: holdout (cv_folds=None) or K-fold (K=3 default, K=5
optional). Grid/random use (Stratified) K-fold via sklearn CV searchers;
the Optuna objective uses cross_val_score with the same K, or an internal
80/20 holdout when CV is disabled.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TUNING_CHOICES = ("none", "grid", "random", "optuna")


@dataclass
class TuningOutcome:
    method: str
    best_params: dict[str, Any] = field(default_factory=dict)
    best_cv_auc: float | None = None
    n_candidates: int = 0
    cv_folds: int | None = None
    note: str = ""


def optuna_available() -> bool:
    try:
        import optuna  # noqa: F401

        return True
    except ImportError:
        return False


def _grid_from_space(space: dict[str, dict[str, Any]]) -> dict[str, list[Any]]:
    return {param: list(spec["grid"]) for param, spec in space.items()}


def _random_distributions(space: dict[str, dict[str, Any]], seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    dists: dict[str, Any] = {}
    for param, spec in space.items():
        if spec["type"] == "cat":
            dists[param] = list(spec["choices"])
        elif spec["type"] == "int":
            step = int(spec.get("step", 1))
            dists[param] = list(range(int(spec["low"]), int(spec["high"]) + 1, step))
        else:  # float
            if spec.get("log"):
                dists[param] = list(
                    np.round(np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]), 25)), 5)
                )
            else:
                dists[param] = list(np.round(np.linspace(spec["low"], spec["high"], 25), 5))
    return dists


def tune_model(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    method: str = "none",
    space: dict[str, dict[str, Any]] | None = None,
    cv_folds: int | None = 3,
    seed: int = 42,
    n_random_iter: int = 15,
    n_optuna_trials: int = 20,
) -> tuple[Any, TuningOutcome]:
    """Returns (fitted-or-configured estimator, outcome). The returned
    estimator is NOT yet fitted on the full training data; callers fit it."""
    method = method.lower().strip()
    if method not in TUNING_CHOICES:
        return estimator, TuningOutcome(method="none", note=f"Unknown tuning '{method}'; using defaults.")
    if method == "none" or not space:
        return estimator, TuningOutcome(method="none", cv_folds=cv_folds)

    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed) if cv_folds else 3

    if method == "grid":
        search = GridSearchCV(estimator, _grid_from_space(space), scoring="roc_auc", cv=cv, n_jobs=-1)
        search.fit(X, y)
        return search.best_estimator_, TuningOutcome(
            method="grid",
            best_params=dict(search.best_params_),
            best_cv_auc=round(float(search.best_score_), 6),
            n_candidates=len(search.cv_results_["params"]),
            cv_folds=cv_folds,
        )

    if method == "random":
        search = RandomizedSearchCV(
            estimator,
            _random_distributions(space, seed),
            n_iter=n_random_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=seed,
            n_jobs=-1,
        )
        search.fit(X, y)
        return search.best_estimator_, TuningOutcome(
            method="random",
            best_params=dict(search.best_params_),
            best_cv_auc=round(float(search.best_score_), 6),
            n_candidates=n_random_iter,
            cv_folds=cv_folds,
        )

    # method == "optuna"
    if not optuna_available():
        return estimator, TuningOutcome(
            method="none",
            note=(
                "optuna is not installed; skipped Bayesian optimization and used the "
                "default model. Enable with: pip install -e \".[optuna]\""
            ),
            cv_folds=cv_folds,
        )
    return _optuna_tune(estimator, X, y, space, cv_folds, seed, n_optuna_trials)


def _optuna_tune(
    estimator: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    space: dict[str, dict[str, Any]],
    cv_folds: int | None,
    seed: int,
    n_trials: int,
) -> tuple[Any, TuningOutcome]:
    import optuna
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def suggest(trial: optuna.Trial) -> dict[str, Any]:
        params: dict[str, Any] = {}
        for name, spec in space.items():
            if spec["type"] == "cat":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            elif spec["type"] == "int":
                params[name] = trial.suggest_int(
                    name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
                )
            else:
                params[name] = trial.suggest_float(
                    name, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
                )
        return params

    if cv_folds:
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

        def objective(trial: optuna.Trial) -> float:
            model = clone(estimator).set_params(**suggest(trial))
            return float(np.mean(cross_val_score(model, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)))

    else:
        X_fit, X_val, y_fit, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )

        def objective(trial: optuna.Trial) -> float:
            model = clone(estimator).set_params(**suggest(trial))
            model.fit(X_fit, y_fit)
            return float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = clone(estimator).set_params(**study.best_params)
    return best, TuningOutcome(
        method="optuna",
        best_params=dict(study.best_params),
        best_cv_auc=round(float(study.best_value), 6),
        n_candidates=n_trials,
        cv_folds=cv_folds,
    )
