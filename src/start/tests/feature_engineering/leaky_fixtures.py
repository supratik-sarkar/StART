"""Deliberately leaky transformation fixtures.

Why these exist in the shipped package rather than only in tests
---------------------------------------------------------------

A leakage detector that has never been shown to fire is a leakage detector nobody should
trust. These six executors are *wrong on purpose*, each in a way that occurs in real
pipelines, and the audit is required to catch each one for the stated reason.

They are importable so the audit's sensitivity can be demonstrated on demand — during a
review, in a regression run, or by a sceptical reader — not only inside a test file that
most people never open.

Every fixture is prefixed ``leaky_`` and none is registered as a test. They cannot be
reached by the registry and will never run in a review.

The six defects
---------------

============================  ==========================================  =========
Fixture                       Defect                                      Caught by
============================  ==========================================  =========
``leaky_scaler``              scaler fitted on train + test               Check 1, 2
``leaky_imputer``             imputer fitted on train + test              Check 1, 2
``leaky_target_encoder``      full-train mapping reused on the same       Check 3
                              train rows that produced it
``leaky_aggregation``         rolling window that includes future rows    Check 4
``leaky_pca``                 PCA fitted on train + test                  Check 1, 2
``leaky_selector``            selector that consumes evaluation labels    Check 1, 2
============================  ==========================================  =========

The two that matter most are ``leaky_scaler`` and ``leaky_aggregation``. The first is
the defect that motivated Check 2, because shuffling evaluation rows leaves its fitted
state **identical** — a mean does not depend on row order. The second is the one that
produces a model with excellent backtest performance and no forward performance at all.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from start.tests.feature_engineering.execution import (
    FittingScope,
    TransformExecutionResult,
)
from start.tests.feature_engineering.transforms import (
    _smoothed_target_map,
    numeric_columns,
    run_imputation,
    run_pca_transform,
    run_scaling,
)

__all__ = [
    "leaky_scaler",
    "leaky_imputer",
    "leaky_target_encoder",
    "leaky_aggregation",
    "leaky_pca",
    "leaky_selector",
    "LEAKY_FIXTURES",
]


def _wrap(
    step: str,
    state: dict[str, Any],
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    oos: pd.DataFrame | None,
    scope: str = FittingScope.TRAIN_ONLY,
    train_out: pd.DataFrame | None = None,
    affected: tuple[str, ...] = (),
) -> TransformExecutionResult:
    output = train.copy() if train_out is None else train_out
    return TransformExecutionResult(
        step=step,
        transformed_train=output,
        transformed_test=test,
        transformed_oos=oos,
        fitted_state=state,
        fitting_scope=scope,
        input_feature_names=tuple(map(str, train.columns)),
        output_feature_names=tuple(map(str, output.columns)),
        affected_features=affected,
    ).with_input_hashes(train, test, oos)


# --------------------------------------------------------------------------- #
# 1. Scaler fitted on train + test
# --------------------------------------------------------------------------- #
def leaky_scaler(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    **kwargs: Any,
) -> TransformExecutionResult:
    """Fits the scaler on ``concat(train, test)``.

    The canonical leak, and the reason Check 2 perturbs values rather than order: the
    fitted mean and standard deviation are **unchanged** by shuffling the evaluation
    rows, so a shuffle-based check passes this happily.
    """
    combined = pd.concat([train, test]) if test is not None else train
    fitted = run_scaling(combined, None, None, **kwargs)
    return _wrap("leaky_scaler", fitted.fitted_state, train, test, oos, affected=fitted.affected_features)


# --------------------------------------------------------------------------- #
# 2. Imputer fitted on train + test
# --------------------------------------------------------------------------- #
def leaky_imputer(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    **kwargs: Any,
) -> TransformExecutionResult:
    """Learns fill values from ``concat(train, test)``.

    Subtler than the scaler in practice, because a median is robust and the leaked value
    often looks reasonable. It is still evaluation information entering the training
    representation.
    """
    combined = pd.concat([train, test]) if test is not None else train
    fitted = run_imputation(combined, None, None, **kwargs)
    return _wrap("leaky_imputer", fitted.fitted_state, train, test, oos, affected=fitted.affected_features)


# --------------------------------------------------------------------------- #
# 3. Target encoder reusing the full-train mapping on the same train rows
# --------------------------------------------------------------------------- #
def leaky_target_encoder(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    target_column: str | None = None,
    smoothing: float = 10.0,
    **kwargs: Any,
) -> TransformExecutionResult:
    """Encodes train rows with a mapping fitted on all of train, including themselves.

    Every train row's encoded value contains its own target. The feature looks
    outstanding in training and collapses in production, and nothing about the
    train-side state is anomalous — which is why Check 3 flips a single row's target and
    watches that row's own encoded value rather than inspecting the state.
    """
    if not target_column or target_column not in train.columns:
        raise ValueError("leaky_target_encoder requires a target_column.")

    y = pd.to_numeric(train[target_column], errors="coerce")
    prior = float(y.mean())
    categorical = [c for c in train.columns if c != target_column and c not in set(numeric_columns(train))]
    train_out = train.copy()
    maps: dict[str, Any] = {}
    for column in categorical:
        mapping = _smoothed_target_map(train[column], y, prior, smoothing)
        maps[column] = mapping
        train_out[column] = [mapping.get(str(v), prior) for v in train[column]]

    return _wrap(
        "leaky_target_encoder",
        {"maps": maps, "prior": prior},
        train,
        test,
        oos,
        FittingScope.TRAIN_FOLDS,
        train_out,
        tuple(categorical),
    )


# --------------------------------------------------------------------------- #
# 4. Rolling aggregation that includes future observations
# --------------------------------------------------------------------------- #
def leaky_aggregation(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    entity_id_column: str | None = None,
    timestamp_column: str | None = None,
    windows: tuple[int, ...] = (7,),
    **kwargs: Any,
) -> TransformExecutionResult:
    """Uses a centred window, so each row's aggregate includes later observations.

    The most damaging defect in the set. The model backtests beautifully and has no
    forward performance whatsoever, because at scoring time the future half of every
    window does not exist.
    """
    if not entity_id_column or not timestamp_column:
        raise ValueError("leaky_aggregation requires entity and timestamp columns.")

    columns = [c for c in numeric_columns(train, (entity_id_column,))]
    work = train.copy()
    work["__ts"] = pd.to_datetime(work[timestamp_column], errors="coerce")
    work = work.sort_values([entity_id_column, "__ts"], kind="mergesort")

    for column in columns:
        for window in windows:
            name = f"{column}__mean_{window}d"
            pieces = []
            for _, group in work.groupby(entity_id_column, sort=False):
                # center=True is the defect: the window straddles the current row.
                rolled = group[column].rolling(window, center=True, min_periods=1).mean()
                pieces.append(pd.Series(rolled.to_numpy(), index=group.index))
            work[name] = pd.concat(pieces).reindex(work.index)

    work = work.drop(columns=["__ts"]).reindex(train.index)
    return _wrap(
        "leaky_aggregation",
        {"windows": list(windows)},
        train,
        test,
        oos,
        FittingScope.STATELESS,
        work,
        tuple(columns),
    )


# --------------------------------------------------------------------------- #
# 5. PCA fitted on train + test
# --------------------------------------------------------------------------- #
def leaky_pca(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    **kwargs: Any,
) -> TransformExecutionResult:
    """Fits components on ``concat(train, test)``.

    The components are rotated toward evaluation variance, so every downstream metric is
    optimistic by an amount nobody can estimate after the fact.
    """
    combined = pd.concat([train, test]) if test is not None else train
    fitted = run_pca_transform(combined, None, None, **kwargs)
    return _wrap("leaky_pca", fitted.fitted_state, train, test, oos, affected=fitted.affected_features)


# --------------------------------------------------------------------------- #
# 6. Selector consuming evaluation labels
# --------------------------------------------------------------------------- #
def leaky_selector(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    target_column: str | None = None,
    top_k: int = 2,
    **kwargs: Any,
) -> TransformExecutionResult:
    """Ranks features using ``concat(train, test)`` labels.

    Distinct from the target-encoding defect, and worth keeping separate: supervised
    selection is *allowed* to use every training label. What it may never see is an
    evaluation label. This fixture crosses exactly that line.
    """
    if not target_column or target_column not in train.columns:
        raise ValueError("leaky_selector requires a target_column.")

    combined = pd.concat([train, test]) if test is not None else train
    features = numeric_columns(combined, (target_column,))
    y = pd.to_numeric(combined[target_column], errors="coerce")
    scores = {c: abs(float(pd.to_numeric(combined[c], errors="coerce").corr(y))) for c in features}
    scores = {c: (0.0 if not np.isfinite(v) else v) for c, v in scores.items()}
    keep = sorted(sorted(scores, key=lambda c: (-scores[c], c))[:top_k])
    dropped = sorted(set(features) - set(keep))

    return _wrap(
        "leaky_selector",
        {"scores": {k: round(v, 10) for k, v in scores.items()}, "kept": keep, "dropped": dropped},
        train,
        test,
        oos,
        FittingScope.TRAIN_ONLY,
        train.drop(columns=[c for c in dropped if c in train.columns]),
        tuple(dropped),
    )


#: (fixture, defect description, checks expected to fire)
LEAKY_FIXTURES: tuple[tuple[Any, str, tuple[str, ...]], ...] = (
    (
        leaky_scaler,
        "scaler fitted on train + test",
        ("check_1_train_only_reproduction", "check_2_evaluation_influence"),
    ),
    (
        leaky_imputer,
        "imputer fitted on train + test",
        ("check_1_train_only_reproduction", "check_2_evaluation_influence"),
    ),
    (
        leaky_target_encoder,
        "full-train target mapping reused on its own train rows",
        ("check_3_out_of_fold_target_encoding",),
    ),
    (leaky_aggregation, "rolling window includes future observations", ("check_4_future_influence",)),
    (
        leaky_pca,
        "PCA fitted on train + test",
        ("check_1_train_only_reproduction", "check_2_evaluation_influence"),
    ),
    (
        leaky_selector,
        "selector consumes evaluation labels",
        ("check_1_train_only_reproduction", "check_2_evaluation_influence"),
    ),
)
