"""Feature-engineering transformations.

Every executor here takes ``(train, test, oos, params)`` and returns a
:class:`TransformExecutionResult`. The registered test wrappers call these same
functions and emit audit evidence from the result — the mathematics exists once, so a
transformation cannot pass its audit while behaving differently in the pipeline.

The invariant every stateful executor obeys
-------------------------------------------

    Learned parameters come from the permitted training scope only, and are applied
    unchanged to evaluation cohorts.

Two executors need the stronger form. Target encoding and WoE derive values *from the
target*, so a train row encoded with a mapping that included its own target has seen its
own answer. Those fit out-of-fold on the train side and on the full training data for
evaluation cohorts. That asymmetry is deliberate and is what
``fitting_scope_audit`` Check 3 verifies.

Determinism is ``numerical`` throughout: quantiles, PCA, standardisation and every
sklearn call go through BLAS.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from start.tests.feature_engineering.execution import (
    FittingScope,
    TransformExecutionResult,
)

__all__ = [
    "run_imputation",
    "run_scaling",
    "run_numeric_transform",
    "run_winsorization",
    "run_categorical_encoding",
    "run_rare_category_grouping",
    "run_woe_iv",
    "run_monotonic_binning",
    "run_interactions",
    "run_temporal_features",
    "run_aggregation_features",
    "run_pca_transform",
    "run_selection",
    "numeric_columns",
    "categorical_columns",
]


def numeric_columns(df: pd.DataFrame, exclude: tuple[str | None, ...] = ()) -> list[str]:
    drop = {c for c in exclude if c}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]


def categorical_columns(df: pd.DataFrame, exclude: tuple[str | None, ...] = ()) -> list[str]:
    drop = {c for c in exclude if c}
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    return [c for c in df.columns if c not in numeric and c not in drop]


def _apply(frames: list[pd.DataFrame | None], fn: Any) -> list[pd.DataFrame | None]:
    return [None if f is None else fn(f.copy()) for f in frames]


def _result(
    step: str,
    outputs: list[pd.DataFrame | None],
    inputs: tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None],
    state: dict[str, Any],
    scope: str,
    params: dict[str, Any],
    affected: list[str],
    notes: list[str] | None = None,
) -> TransformExecutionResult:
    train_out = outputs[0]
    assert train_out is not None
    return TransformExecutionResult(
        step=step,
        transformed_train=train_out,
        transformed_test=outputs[1],
        transformed_oos=outputs[2],
        fitted_state=state,
        fitting_scope=scope,
        input_feature_names=tuple(map(str, inputs[0].columns)),
        output_feature_names=tuple(map(str, train_out.columns)),
        params=params,
        affected_features=tuple(sorted(affected)),
        notes=notes or [],
    ).with_input_hashes(*inputs)


# --------------------------------------------------------------------------- #
# Imputation
# --------------------------------------------------------------------------- #
def run_imputation(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    strategy: str = "median",
    add_indicator: bool = False,
    fill_value: Any = 0,
    exclude: tuple[str | None, ...] = (),
) -> TransformExecutionResult:
    """Fill missing values from training statistics only.

    An all-missing column has no learnable fill value. It is left untouched and recorded
    rather than filled with a fabricated constant — silently inventing a column of zeros
    is worse than leaving the gap visible.
    """
    if strategy not in {"median", "mean", "mode", "constant"}:
        raise ValueError(f"strategy={strategy!r} is not supported.")

    numeric = numeric_columns(train, exclude)
    categorical = categorical_columns(train, exclude)
    fills: dict[str, Any] = {}
    all_missing: list[str] = []

    for column in numeric:
        series = train[column]
        if series.isna().all():
            all_missing.append(column)
            continue
        if strategy == "median":
            fills[column] = float(series.median())
        elif strategy == "mean":
            fills[column] = float(series.mean())
        elif strategy == "mode":
            modes = series.mode()
            fills[column] = float(modes.iloc[0]) if len(modes) else 0.0
        else:
            fills[column] = fill_value

    for column in categorical:
        series = train[column].dropna()
        if series.empty:
            all_missing.append(column)
            continue
        # mode/constant are the only meaningful categorical policies; median and mean
        # are undefined for a category and are NOT silently substituted.
        if strategy == "constant":
            fills[column] = fill_value
        else:
            fills[column] = series.mode().iloc[0]

    affected = [c for c in fills if train[c].isna().any()]
    indicator_columns = sorted(affected) if add_indicator else []

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        for column in indicator_columns:
            if column in frame.columns:
                frame[f"{column}__missing"] = frame[column].isna().astype(int)
        for column, value in fills.items():
            if column in frame.columns:
                frame[column] = frame[column].fillna(value)
        return frame

    outputs = _apply([train, test, oos], transform)
    notes = []
    if all_missing:
        notes.append(
            f"{len(all_missing)} column(s) are entirely missing in training and were "
            f"left untouched rather than filled with a fabricated value: "
            f"{', '.join(all_missing[:8])}."
        )
    return _result(
        "imputation",
        outputs,
        (train, test, oos),
        {"fill_values": fills, "indicator_columns": indicator_columns},
        FittingScope.TRAIN_ONLY,
        {"strategy": strategy, "add_indicator": add_indicator, "fill_value": fill_value},
        affected + indicator_columns,
        notes,
    )


# --------------------------------------------------------------------------- #
# Scaling
# --------------------------------------------------------------------------- #
def run_scaling(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    method: str = "standard",
    quantile_range: tuple[float, float] = (25.0, 75.0),
    exclude: tuple[str | None, ...] = (),
) -> TransformExecutionResult:
    """Scale numeric features using training statistics only.

    A zero-variance column would divide by zero. The divisor is floored at 1.0 for such
    columns, leaving them shifted but not exploded — an ``inf`` column silently poisons
    every downstream fit and is far harder to trace back than a constant one.
    """
    if method not in {"standard", "robust", "minmax", "maxabs"}:
        raise ValueError(f"method={method!r} is not supported.")

    numeric = numeric_columns(train, exclude)
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    degenerate: list[str] = []

    for column in numeric:
        values = pd.to_numeric(train[column], errors="coerce").dropna()
        if values.empty:
            centers[column], scales[column] = 0.0, 1.0
            degenerate.append(column)
            continue
        if method == "standard":
            center = float(values.mean())
            scale = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        elif method == "robust":
            center = float(values.median())
            lo, hi = np.percentile(values, quantile_range)
            scale = float(hi - lo)
        elif method == "minmax":
            center = float(values.min())
            scale = float(values.max() - values.min())
        else:
            center = 0.0
            scale = float(np.max(np.abs(values)))
        if not math.isfinite(scale) or scale <= 0.0:
            scale = 1.0
            degenerate.append(column)
        centers[column], scales[column] = center, scale

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        for column in numeric:
            if column in frame.columns:
                frame[column] = (pd.to_numeric(frame[column], errors="coerce") - centers[column]) / scales[
                    column
                ]
        return frame

    outputs = _apply([train, test, oos], transform)
    notes = (
        [
            f"{len(degenerate)} zero-variance column(s) scaled with a unit divisor rather "
            f"than producing infinities: {', '.join(sorted(set(degenerate))[:8])}."
        ]
        if degenerate
        else []
    )
    return _result(
        "scaling",
        outputs,
        (train, test, oos),
        {"centers": centers, "scales": scales, "method": method},
        FittingScope.TRAIN_ONLY,
        {"method": method, "quantile_range": list(quantile_range)},
        numeric,
        notes,
    )


# --------------------------------------------------------------------------- #
# Numeric transforms
# --------------------------------------------------------------------------- #
def run_numeric_transform(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    method: str = "yeo_johnson",
    exclude: tuple[str | None, ...] = (),
) -> TransformExecutionResult:
    """Reshape numeric distributions.

    ``log`` and ``boxcox`` require strictly positive input. A column violating that is
    **skipped and recorded**, never shifted to make the transform succeed — adding a
    constant to force positivity changes what the feature means, and doing it silently
    is how a model ends up trained on a quantity nobody can describe.
    """
    if method not in {"log", "log1p", "boxcox", "yeo_johnson", "quantile", "rank"}:
        raise ValueError(f"method={method!r} is not supported.")

    numeric = numeric_columns(train, exclude)
    stateless = method in {"log", "log1p", "rank"}
    state: dict[str, Any] = {"method": method, "lambdas": {}, "quantiles": {}}
    applied: list[str] = []
    skipped: list[str] = []

    for column in numeric:
        values = pd.to_numeric(train[column], errors="coerce").dropna()
        if values.empty:
            skipped.append(column)
            continue
        if method in {"log", "boxcox"} and float(values.min()) <= 0:
            skipped.append(column)
            continue
        if method == "log1p" and float(values.min()) <= -1:
            skipped.append(column)
            continue
        applied.append(column)
        if method == "boxcox":
            from scipy import stats as sp

            _, lam = sp.boxcox(values.to_numpy(dtype=float))
            state["lambdas"][column] = float(lam)
        elif method == "yeo_johnson":
            from scipy import stats as sp

            _, lam = sp.yeojohnson(values.to_numpy(dtype=float))
            state["lambdas"][column] = float(lam)
        elif method == "quantile":
            state["quantiles"][column] = [float(x) for x in np.percentile(values, np.linspace(0, 100, 101))]

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        for column in applied:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce")
            if method == "log":
                frame[column] = np.log(values.where(values > 0))
            elif method == "log1p":
                frame[column] = np.log1p(values.where(values > -1))
            elif method == "rank":
                frame[column] = values.rank(pct=True)
            elif method == "boxcox":
                lam = state["lambdas"][column]
                arr = values.to_numpy(dtype=float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    frame[column] = np.log(arr) if abs(lam) < 1e-12 else (np.power(arr, lam) - 1) / lam
            elif method == "yeo_johnson":
                from scipy import stats as sp

                lam = state["lambdas"][column]
                arr = values.to_numpy(dtype=float)
                out = np.full_like(arr, np.nan, dtype=float)
                finite = np.isfinite(arr)
                if finite.any():
                    out[finite] = sp.yeojohnson(arr[finite], lmbda=lam)
                frame[column] = out
            elif method == "quantile":
                grid = np.asarray(state["quantiles"][column], dtype=float)
                frame[column] = np.interp(values.to_numpy(dtype=float), grid, np.linspace(0, 1, len(grid)))
        return frame

    outputs = _apply([train, test, oos], transform)
    notes = (
        [
            f"{len(skipped)} column(s) skipped as domain-invalid for {method} and left "
            f"unchanged rather than shifted to force the transform: "
            f"{', '.join(skipped[:8])}."
        ]
        if skipped
        else []
    )
    return _result(
        "numeric_transform",
        outputs,
        (train, test, oos),
        state,
        FittingScope.STATELESS if stateless else FittingScope.TRAIN_ONLY,
        {"method": method},
        applied,
        notes,
    )


# --------------------------------------------------------------------------- #
# Winsorization
# --------------------------------------------------------------------------- #
def run_winsorization(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    method: str = "iqr",
    k: float = 1.5,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    z_threshold: float = 3.0,
    exclude: tuple[str | None, ...] = (),
) -> TransformExecutionResult:
    """Clip extreme values to bounds learned on training data.

    A transformation, distinct from ``preprocessing.outliers`` which only reports. The
    bounds are fitted once on train and applied unchanged — recomputing them per cohort
    would clip evaluation data to its own extremes and hide exactly the distribution
    shift a reviewer needs to see.
    """
    if method not in {"iqr", "percentile", "zscore"}:
        raise ValueError(f"method={method!r} is not supported.")

    numeric = numeric_columns(train, exclude)
    bounds: dict[str, tuple[float, float]] = {}

    for column in numeric:
        values = pd.to_numeric(train[column], errors="coerce").dropna()
        if values.empty:
            continue
        if method == "iqr":
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            bounds[column] = (float(q1 - k * iqr), float(q3 + k * iqr))
        elif method == "percentile":
            lo, hi = np.percentile(values, [lower_pct, upper_pct])
            bounds[column] = (float(lo), float(hi))
        else:
            mean, std = float(values.mean()), float(values.std(ddof=1))
            std = std if std > 0 else 1.0
            bounds[column] = (mean - z_threshold * std, mean + z_threshold * std)

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        for column, (lo, hi) in bounds.items():
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce").clip(lo, hi)
        return frame

    outputs = _apply([train, test, oos], transform)
    return _result(
        "winsorization",
        outputs,
        (train, test, oos),
        {"bounds": {c: list(v) for c, v in bounds.items()}, "method": method},
        FittingScope.TRAIN_ONLY,
        {
            "method": method,
            "k": k,
            "lower_pct": lower_pct,
            "upper_pct": upper_pct,
            "z_threshold": z_threshold,
        },
        list(bounds),
    )


# --------------------------------------------------------------------------- #
# Rare category grouping
# --------------------------------------------------------------------------- #
def run_rare_category_grouping(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    min_pct: float = 1.0,
    other_label: str = "__OTHER__",
    exclude: tuple[str | None, ...] = (),
) -> TransformExecutionResult:
    """Collapse infrequent levels, using training frequencies only.

    A level unseen in training also maps to ``other_label``. It has no learned
    representation, so treating it as its own category would create a column value the
    model has never encountered.
    """
    categorical = categorical_columns(train, exclude)
    keep: dict[str, list[str]] = {}

    for column in categorical:
        series = train[column].dropna()
        if series.empty:
            keep[column] = []
            continue
        shares = series.value_counts() / len(series) * 100.0
        keep[column] = sorted(str(level) for level in shares[shares >= min_pct].index)

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        for column, kept in keep.items():
            if column in frame.columns:
                allowed = set(kept)
                frame[column] = frame[column].apply(
                    lambda v, _allowed=allowed: (
                        v if (pd.notna(v) and str(v) in _allowed) else (other_label if pd.notna(v) else v)
                    )
                )
        return frame

    affected = [c for c, kept in keep.items() if len(kept) < train[c].dropna().nunique()]
    outputs = _apply([train, test, oos], transform)
    return _result(
        "rare_category_grouping",
        outputs,
        (train, test, oos),
        {"kept_levels": keep, "other_label": other_label},
        FittingScope.TRAIN_ONLY,
        {"min_pct": min_pct, "other_label": other_label},
        affected,
        [
            "Levels unseen in training also map to the other-label: they have no learned "
            "representation, so treating them as their own category would create a value "
            "the model has never encountered."
        ],
    )


# --------------------------------------------------------------------------- #
# Categorical encoding — target encoding is out-of-fold on the train side
# --------------------------------------------------------------------------- #
def _fold_assignment(n: int, n_folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    folds = np.arange(n) % n_folds
    rng.shuffle(folds)
    return folds


def _smoothed_target_map(levels: pd.Series, y: pd.Series, prior: float, smoothing: float) -> dict[str, float]:
    frame = pd.DataFrame({"lvl": levels.astype(str), "y": y})
    grouped = frame.groupby("lvl")["y"].agg(["mean", "count"])
    weight = grouped["count"] / (grouped["count"] + smoothing)
    blended = weight * grouped["mean"] + (1 - weight) * prior
    return {str(k): float(v) for k, v in blended.items()}


def run_categorical_encoding(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    method: str = "onehot",
    target_column: str | None = None,
    n_folds: int = 5,
    smoothing: float = 10.0,
    seed: int = 42,
    unseen: str = "prior",
    exclude: tuple[str | None, ...] = (),
) -> TransformExecutionResult:
    """Encode categorical features.

    ``target`` encoding is the one that can leak, and it is handled specially. A train
    row encoded using a mapping that included its own target has seen its own answer,
    and the resulting feature looks far more predictive in training than it will ever be
    in production.

    So the train side is encoded **out-of-fold**: each row's value comes from a mapping
    fitted on the other folds only. Evaluation cohorts use the full-training mapping,
    which is correct — no evaluation row contributed to it. That asymmetry is the point,
    and ``fitting_scope_audit`` Check 3 verifies it directly.
    """
    if method not in {"onehot", "ordinal", "frequency", "target"}:
        raise ValueError(f"method={method!r} is not supported.")
    exclude = tuple(exclude) + (target_column,)
    categorical = categorical_columns(train, exclude)
    state: dict[str, Any] = {"method": method, "maps": {}}

    if method == "target":
        if not target_column or target_column not in train.columns:
            raise ValueError("target encoding requires a target_column present in train.")
        y = pd.to_numeric(train[target_column], errors="coerce")
        prior = float(y.mean())
        state["prior"] = prior
        folds = _fold_assignment(len(train), max(2, n_folds), seed)
        state["n_folds"] = int(max(2, n_folds))

        train_out = train.copy()
        for column in categorical:
            full_map = _smoothed_target_map(train[column], y, prior, smoothing)
            state["maps"][column] = full_map
            encoded = np.full(len(train), prior, dtype=float)
            for fold in range(state["n_folds"]):
                holdout = folds == fold
                fitting = ~holdout
                if not fitting.any() or not holdout.any():
                    continue
                # The PRIOR must also be out-of-fold. Computing it over all of train
                # leaks a held-out row's own target into its own encoded value through
                # the smoothing term (1 - weight) * prior. The effect is small, which
                # is precisely why it survives casual review — the audit's Check 3
                # caught it here by flipping one row's target and observing that row's
                # own encoding move.
                fold_prior = float(y[fitting].mean())
                fold_map = _smoothed_target_map(train.loc[fitting, column], y[fitting], fold_prior, smoothing)
                encoded[holdout] = [fold_map.get(str(v), fold_prior) for v in train.loc[holdout, column]]
            train_out[column] = encoded

        def transform_eval(frame: pd.DataFrame) -> pd.DataFrame:
            for column in categorical:
                if column in frame.columns:
                    mapping = state["maps"][column]
                    frame[column] = [mapping.get(str(v), prior) for v in frame[column]]
            return frame

        outputs = [train_out, *(_apply([test, oos], transform_eval))]
        return _result(
            "categorical_encoding",
            outputs,
            (train, test, oos),
            state,
            FittingScope.TRAIN_FOLDS,
            {
                "method": method,
                "n_folds": state["n_folds"],
                "smoothing": smoothing,
                "seed": seed,
                "unseen": unseen,
            },
            categorical,
            [
                "Train-side values are OUT-OF-FOLD: no row's encoding used its own target. "
                "Evaluation cohorts use the full-training mapping, which no evaluation row "
                "contributed to."
            ],
        )

    for column in categorical:
        series = train[column].dropna().astype(str)
        if method == "onehot":
            state["maps"][column] = sorted(series.unique())
        elif method == "ordinal":
            state["maps"][column] = {lvl: i for i, lvl in enumerate(sorted(series.unique()))}
        else:
            counts = series.value_counts()
            state["maps"][column] = {str(k): float(v / len(series)) for k, v in counts.items()}

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        for column in categorical:
            if column not in frame.columns:
                continue
            if method == "onehot":
                # Built as one block and concatenated once. Inserting columns in a loop
                # fragments the frame and pandas warns about it; with a wide categorical
                # the repeated reallocation is also genuinely slow.
                levels = state["maps"][column]
                as_str = frame[column].astype(str)
                block = pd.DataFrame(
                    {f"{column}__{level}": (as_str == level).astype(int) for level in levels},
                    index=frame.index,
                )
                frame = pd.concat([frame.drop(columns=[column]), block], axis=1)
            elif method == "ordinal":
                mapping = state["maps"][column]
                frame[column] = [mapping.get(str(v), -1) for v in frame[column]]
            else:
                mapping = state["maps"][column]
                frame[column] = [mapping.get(str(v), 0.0) for v in frame[column]]
        return frame

    outputs = _apply([train, test, oos], transform)
    return _result(
        "categorical_encoding",
        outputs,
        (train, test, oos),
        state,
        FittingScope.TRAIN_ONLY,
        {"method": method, "unseen": unseen},
        categorical,
        [
            f"Unseen levels map to {'-1' if method == 'ordinal' else '0.0'} — an explicit "
            "sentinel rather than a silent failure."
        ],
    )


# --------------------------------------------------------------------------- #
# WoE / IV
# --------------------------------------------------------------------------- #
def run_woe_iv(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    target_column: str | None = None,
    bins: int = 10,
    min_bin_pct: float = 0.05,
    smoothing: float = 0.5,
    n_folds: int = 5,
    seed: int = 42,
    exclude: tuple[str | None, ...] = (),
) -> TransformExecutionResult:
    """Weight-of-evidence transformation with information value. Binary target only.

    WoE is derived from the target, so it leaks exactly as target encoding does and is
    handled the same way: out-of-fold on the train side, full-training mapping for
    evaluation cohorts.

    IV bands (weak / medium / strong / suspicious) are **industry practice from credit
    scoring**, not a result with a citation. They are reported as convention.
    """
    if not target_column or target_column not in train.columns:
        raise ValueError("WoE requires a target_column present in train.")
    y = pd.to_numeric(train[target_column], errors="coerce")
    if y.dropna().nunique() != 2:
        raise ValueError("WoE requires a binary target.")

    exclude = tuple(exclude) + (target_column,)
    numeric = numeric_columns(train, exclude)
    state: dict[str, Any] = {"edges": {}, "woe": {}, "iv": {}, "n_folds": max(2, n_folds)}

    def woe_map(x: pd.Series, yy: pd.Series, edges: np.ndarray) -> tuple[dict[int, float], float]:
        binned = np.digitize(x.to_numpy(dtype=float), edges[1:-1], right=True)
        frame = pd.DataFrame({"b": binned, "y": yy.to_numpy(dtype=float)})
        pos, neg = float(frame["y"].sum()), float(len(frame) - frame["y"].sum())
        mapping, iv = {}, 0.0
        for bucket, group in frame.groupby("b"):
            good = (float(group["y"].sum()) + smoothing) / (pos + smoothing)
            bad = (float(len(group) - group["y"].sum()) + smoothing) / (neg + smoothing)
            value = math.log(good / bad)
            mapping[int(bucket)] = value
            iv += (good - bad) * value
        return mapping, float(iv)

    folds = _fold_assignment(len(train), state["n_folds"], seed)
    train_out = train.copy()
    applied: list[str] = []

    for column in numeric:
        values = pd.to_numeric(train[column], errors="coerce")
        try:
            edges = np.unique(np.nanpercentile(values.dropna(), np.linspace(0, 100, bins + 1)))
        except Exception:
            continue
        min_count = max(1, int(min_bin_pct * len(train)))
        if len(edges) < 3 or values.dropna().size < min_count * 2:
            continue
        applied.append(column)
        state["edges"][column] = [float(e) for e in edges]
        full_map, iv = woe_map(values.fillna(values.median()), y.fillna(0), edges)
        state["woe"][column] = {str(k): v for k, v in full_map.items()}
        state["iv"][column] = iv

        encoded = np.zeros(len(train), dtype=float)
        for fold in range(state["n_folds"]):
            holdout = folds == fold
            fitting = ~holdout
            if not fitting.any() or not holdout.any():
                continue
            # As for target encoding, everything the fold map depends on is computed
            # from the fitting folds only — including the median used to fill missing
            # values, which would otherwise carry the held-out rows' own values.
            fold_median = values[fitting].median()
            fold_map, _ = woe_map(values[fitting].fillna(fold_median), y[fitting].fillna(0), edges)
            binned = np.digitize(
                values[holdout].fillna(fold_median).to_numpy(dtype=float),
                edges[1:-1],
                right=True,
            )
            encoded[holdout] = [fold_map.get(int(b), 0.0) for b in binned]
        train_out[column] = encoded

    def transform_eval(frame: pd.DataFrame) -> pd.DataFrame:
        for column in applied:
            if column not in frame.columns:
                continue
            edges = np.asarray(state["edges"][column], dtype=float)
            mapping = state["woe"][column]
            values = pd.to_numeric(frame[column], errors="coerce")
            binned = np.digitize(
                values.fillna(values.median()).to_numpy(dtype=float), edges[1:-1], right=True
            )
            frame[column] = [mapping.get(str(int(b)), 0.0) for b in binned]
        return frame

    outputs = [train_out, *(_apply([test, oos], transform_eval))]
    return _result(
        "woe_iv",
        outputs,
        (train, test, oos),
        state,
        FittingScope.TRAIN_FOLDS,
        {
            "bins": bins,
            "min_bin_pct": min_bin_pct,
            "smoothing": smoothing,
            "n_folds": state["n_folds"],
            "seed": seed,
        },
        applied,
        [
            "Train-side WoE values are OUT-OF-FOLD, as for target encoding.",
            "IV bands (0.02 / 0.1 / 0.3 / 0.5) are credit-scoring industry practice, not "
            "a result with a single originating citation.",
        ],
    )


# --------------------------------------------------------------------------- #
# Monotonic binning
# --------------------------------------------------------------------------- #
def run_monotonic_binning(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    target_column: str | None = None,
    max_bins: int = 10,
    min_bin_pct: float = 0.05,
    direction: str = "auto",
    exclude: tuple[str | None, ...] = (),
) -> TransformExecutionResult:
    """Bin numeric features so the target rate is monotone across bins.

    **The specific algorithm implemented here**, stated because several competing
    monotonic-binning procedures exist and none is uniquely canonical:

    1. Initial bins: ``max_bins`` quantile bins of the training data, duplicate edges
       dropped.
    2. Bin target rates computed on training data.
    3. Direction: ``auto`` takes the sign of the Spearman correlation between bin index
       and bin target rate; ties resolve to increasing.
    4. Merge loop: repeatedly find the **lowest-indexed** adjacent pair violating the
       required direction and merge it into its right neighbour. Lowest-indexed is the
       fixed tie-break rule — without one, two implementations produce different bins
       from the same data.
    5. Minimum size: any bin below ``min_bin_pct`` of training rows is merged into its
       right neighbour, or its left neighbour if it is the last bin.
    6. Boundaries are then frozen and applied unchanged to evaluation cohorts.

    Output is the bin index, not the target rate — the index carries no target
    information and therefore does not need out-of-fold treatment.
    """
    if not target_column or target_column not in train.columns:
        raise ValueError("monotonic binning requires a target_column present in train.")
    if direction not in {"auto", "increasing", "decreasing"}:
        raise ValueError(f"direction={direction!r} is not supported.")

    exclude = tuple(exclude) + (target_column,)
    numeric = numeric_columns(train, exclude)
    y = pd.to_numeric(train[target_column], errors="coerce")
    min_count = max(1, int(min_bin_pct * len(train)))
    state: dict[str, Any] = {"edges": {}, "direction": {}, "rates": {}}
    applied: list[str] = []

    for column in numeric:
        values = pd.to_numeric(train[column], errors="coerce")
        clean = values.dropna()
        if clean.nunique() < 3:
            continue
        try:
            edges = list(np.unique(np.nanpercentile(clean, np.linspace(0, 100, max_bins + 1))))
        except Exception:
            continue
        if len(edges) < 3:
            continue

        def rates_for(edge_list: list[float], vals: pd.Series = values) -> tuple[list[float], list[int]]:
            idx = np.digitize(vals.to_numpy(dtype=float), np.asarray(edge_list[1:-1]), right=True)
            frame = pd.DataFrame({"b": idx, "y": y})
            grouped = frame.groupby("b")["y"].agg(["mean", "count"])
            full = [float(grouped["mean"].get(i, np.nan)) for i in range(len(edge_list) - 1)]
            counts = [int(grouped["count"].get(i, 0)) for i in range(len(edge_list) - 1)]
            return full, counts

        rates, counts = rates_for(edges)
        finite = [(i, r) for i, r in enumerate(rates) if math.isfinite(r)]
        if len(finite) < 2:
            continue
        if direction == "auto":
            from scipy import stats as sp

            corr = sp.spearmanr([i for i, _ in finite], [r for _, r in finite]).statistic
            want_increasing = not (math.isfinite(corr) and corr < 0)
        else:
            want_increasing = direction == "increasing"
        state["direction"][column] = "increasing" if want_increasing else "decreasing"

        # Merge loop — lowest-indexed violation first, deterministically.
        guard = 0
        while len(edges) > 3 and guard < 100:
            guard += 1
            rates, counts = rates_for(edges)
            violation = None
            for i in range(len(rates) - 1):
                a, b = rates[i], rates[i + 1]
                if not (math.isfinite(a) and math.isfinite(b)):
                    continue
                if (want_increasing and b < a) or ((not want_increasing) and b > a):
                    violation = i
                    break
            if violation is None:
                break
            del edges[violation + 1]

        guard = 0
        while len(edges) > 3 and guard < 100:
            guard += 1
            rates, counts = rates_for(edges)
            small = next((i for i, c in enumerate(counts) if c < min_count), None)
            if small is None:
                break
            del edges[small + 1 if small < len(edges) - 2 else small]

        applied.append(column)
        state["edges"][column] = [float(e) for e in edges]
        final_rates, _ = rates_for(edges)
        state["rates"][column] = [None if not math.isfinite(r) else round(r, 8) for r in final_rates]

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        for column in applied:
            if column in frame.columns:
                edges = np.asarray(state["edges"][column], dtype=float)
                frame[column] = np.digitize(
                    pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float),
                    edges[1:-1],
                    right=True,
                )
        return frame

    outputs = _apply([train, test, oos], transform)
    return _result(
        "monotonic_binning",
        outputs,
        (train, test, oos),
        state,
        FittingScope.TRAIN_ONLY,
        {"max_bins": max_bins, "min_bin_pct": min_bin_pct, "direction": direction},
        applied,
        [
            "The merge rule is: lowest-indexed adjacent violation merged into its right "
            "neighbour. This is a deterministic choice among several published "
            "monotonic-binning procedures, not the uniquely canonical method.",
            "Output is the bin index, which carries no target information and therefore "
            "needs no out-of-fold treatment.",
        ],
    )


# --------------------------------------------------------------------------- #
# Interactions
# --------------------------------------------------------------------------- #
def run_interactions(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    method: str = "product",
    max_features: int = 50,
    denominator_floor: float = 1e-6,
    exclude: tuple[str | None, ...] = (),
) -> TransformExecutionResult:
    """Pairwise feature interactions, deterministically named and bounded in count.

    ``max_features`` is a hard cap. n features produce n(n-1)/2 pairs — 50 features is
    1,225 new columns — and an uncontrolled expansion is how a feature table becomes
    unreviewable. Pairs are taken in sorted order so the selection is reproducible
    rather than dependent on column order.
    """
    if method not in {"product", "ratio", "difference"}:
        raise ValueError(f"method={method!r} is not supported.")

    numeric = sorted(numeric_columns(train, exclude))
    pairs: list[tuple[str, str]] = []
    for i in range(len(numeric)):
        for j in range(i + 1, len(numeric)):
            pairs.append((numeric[i], numeric[j]))
            if len(pairs) >= max_features:
                break
        if len(pairs) >= max_features:
            break

    truncated = len(numeric) * (len(numeric) - 1) // 2 > len(pairs)

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        for left, right in pairs:
            if left not in frame.columns or right not in frame.columns:
                continue
            a = pd.to_numeric(frame[left], errors="coerce")
            b = pd.to_numeric(frame[right], errors="coerce")
            if method == "product":
                frame[f"{left}__x__{right}"] = a * b
            elif method == "difference":
                frame[f"{left}__minus__{right}"] = a - b
            else:
                # Floor the magnitude, preserving sign: a raw division produces inf
                # that silently poisons every downstream fit.
                safe = b.where(b.abs() >= denominator_floor, np.sign(b).replace(0, 1) * denominator_floor)
                frame[f"{left}__over__{right}"] = a / safe
        return frame

    outputs = _apply([train, test, oos], transform)
    notes = (
        [
            f"Denominator magnitudes below {denominator_floor:g} are floored with the "
            "sign preserved; no infinities are produced."
        ]
        if method == "ratio"
        else []
    )
    if truncated:
        notes.append(
            f"Pair generation capped at {max_features}; pairs are taken in sorted "
            "column order so the selection is reproducible."
        )
    return _result(
        "interactions",
        outputs,
        (train, test, oos),
        {"pairs": [list(p) for p in pairs]},
        FittingScope.STATELESS,
        {"method": method, "max_features": max_features, "denominator_floor": denominator_floor},
        [c for pair in pairs for c in pair],
        notes,
    )


# --------------------------------------------------------------------------- #
# Temporal features
# --------------------------------------------------------------------------- #
def run_temporal_features(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    timestamp_column: str | None = None,
    features: tuple[str, ...] = ("hour", "dow", "month", "days_since"),
    exclude: tuple[str | None, ...] = (),
) -> TransformExecutionResult:
    """Calendar features derived from each row's own timestamp.

    Every feature is a function of that row's timestamp alone. ``days_since`` uses the
    **training** origin, fitted once, so an evaluation row's value is measured on the
    same scale as training. Using each cohort's own minimum would make the same
    calendar date produce different values in different cohorts.
    """
    if not timestamp_column or timestamp_column not in train.columns:
        raise ValueError("temporal features require a timestamp_column present in train.")

    origin = pd.to_datetime(train[timestamp_column], errors="coerce").min()
    state = {"origin": str(origin), "features": list(features)}

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        if timestamp_column not in frame.columns:
            return frame
        ts = pd.to_datetime(frame[timestamp_column], errors="coerce")
        if "hour" in features:
            frame[f"{timestamp_column}__hour"] = ts.dt.hour
        if "dow" in features:
            frame[f"{timestamp_column}__dow"] = ts.dt.dayofweek
        if "month" in features:
            frame[f"{timestamp_column}__month"] = ts.dt.month
        if "days_since" in features:
            frame[f"{timestamp_column}__days_since"] = (ts - origin).dt.total_seconds() / 86400.0
        return frame

    outputs = _apply([train, test, oos], transform)
    return _result(
        "temporal_features",
        outputs,
        (train, test, oos),
        state,
        FittingScope.TRAIN_ONLY,
        {"timestamp_column": timestamp_column, "features": list(features)},
        [timestamp_column],
        [
            "Every feature depends only on its own row's timestamp; none references a future observation.",
            "The days-since origin is fitted on training data so the same calendar date "
            "yields the same value in every cohort.",
        ],
    )


# --------------------------------------------------------------------------- #
# Aggregation features — the causal invariant
# --------------------------------------------------------------------------- #
def run_aggregation_features(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    entity_id_column: str | None = None,
    timestamp_column: str | None = None,
    value_columns: tuple[str, ...] | None = None,
    windows: tuple[int, ...] = (7, 30, 90),
    exclude: tuple[str | None, ...] = (),
) -> TransformExecutionResult:
    """Rolling per-entity statistics, **strictly backward-looking**.

    The invariant, and the reason this executor is written the way it is:

        A row's aggregate feature depends only on observations at or before its own
        timestamp. Changing a future observation must not change it.

    Implementation detail that makes it hold: within each entity, rows are sorted by
    timestamp and the window is closed on the left with ``closed="left"``, so the
    current row is **excluded** from its own aggregate. Including the current row is
    the subtler leak — it is not future information, but it means the feature contains
    the observation the model is being asked to predict from.

    ``fitting_scope_audit`` Check 4 perturbs future rows and asserts historical values
    are unchanged.
    """
    if not entity_id_column or entity_id_column not in train.columns:
        raise ValueError("aggregation features require an entity_id_column present in train.")
    if not timestamp_column or timestamp_column not in train.columns:
        raise ValueError("aggregation features require a timestamp_column present in train.")

    columns = (
        list(value_columns) if value_columns else numeric_columns(train, tuple(exclude) + (entity_id_column,))
    )
    state = {"value_columns": columns, "windows": list(windows)}

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        if entity_id_column not in frame.columns or timestamp_column not in frame.columns:
            return frame
        work = frame.copy()
        work["__ts"] = pd.to_datetime(work[timestamp_column], errors="coerce")
        work = work.sort_values([entity_id_column, "__ts"], kind="mergesort")
        for column in columns:
            if column not in work.columns:
                continue
            for window in windows:
                name = f"{column}__mean_{window}d"
                pieces = []
                for _, group in work.groupby(entity_id_column, sort=False):
                    series = group.set_index("__ts")[column]
                    # closed="left": the current row is excluded from its own window.
                    rolled = series.rolling(f"{window}D", closed="left").mean()
                    pieces.append(pd.Series(rolled.to_numpy(), index=group.index))
                work[name] = pd.concat(pieces).reindex(work.index)
        work = work.drop(columns=["__ts"])
        return work.reindex(frame.index)

    outputs = _apply([train, test, oos], transform)
    return _result(
        "aggregation_features",
        outputs,
        (train, test, oos),
        state,
        FittingScope.STATELESS,
        {
            "entity_id_column": entity_id_column,
            "timestamp_column": timestamp_column,
            "windows": list(windows),
        },
        columns,
        [
            "Windows are closed on the left, so a row is EXCLUDED from its own aggregate. "
            "Including it is not future information, but it puts the observation being "
            "predicted from inside the feature.",
            "Verified by the fitting-scope audit: perturbing future rows leaves historical "
            "feature values unchanged.",
        ],
    )


# --------------------------------------------------------------------------- #
# PCA
# --------------------------------------------------------------------------- #
def run_pca_transform(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    n_components: float | int = 0.95,
    whiten: bool = False,
    seed: int = 42,
    exclude: tuple[str | None, ...] = (),
) -> TransformExecutionResult:
    """PCA fitted on training data only, applied unchanged to evaluation cohorts.

    Distinct from ``preprocessing.dimensionality_diagnostic``, which reports how many
    components would be needed and produces no data.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    numeric = numeric_columns(train, exclude)
    frame = train[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(frame.median())
    usable = [c for c in numeric if frame[c].nunique() > 1]
    if len(usable) < 2:
        raise ValueError("PCA requires at least two non-constant numeric features.")

    scaler = StandardScaler().fit(frame[usable].to_numpy(dtype=float))
    pca = PCA(n_components=n_components, random_state=seed, whiten=whiten)
    pca.fit(scaler.transform(frame[usable].to_numpy(dtype=float)))
    n_actual = int(pca.n_components_)
    names = [f"pc_{i + 1}" for i in range(n_actual)]

    state = {
        "components": pca.components_.tolist(),
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "feature_order": usable,
        "n_components_actual": n_actual,
    }

    def transform(f: pd.DataFrame) -> pd.DataFrame:
        block = f[usable].apply(pd.to_numeric, errors="coerce")
        block = block.replace([np.inf, -np.inf], np.nan).fillna(frame.median())
        projected = pca.transform(scaler.transform(block.to_numpy(dtype=float)))
        out = f.drop(columns=usable).copy()
        for index, name in enumerate(names):
            out[name] = projected[:, index]
        return out

    outputs = _apply([train, test, oos], transform)
    return _result(
        "pca_transform",
        outputs,
        (train, test, oos),
        state,
        FittingScope.TRAIN_ONLY,
        {"n_components": n_components, "whiten": whiten, "seed": seed},
        usable,
        [
            "Components are fitted on training data and applied unchanged to evaluation "
            "cohorts. Refitting per cohort would produce different axes and make the "
            "cohorts incomparable."
        ],
    )


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def run_selection(
    train: pd.DataFrame,
    test: pd.DataFrame | None = None,
    oos: pd.DataFrame | None = None,
    *,
    method: str = "mutual_info",
    target_column: str | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
    vif_threshold: float = 10.0,
    seed: int = 42,
    model: Any = None,
    exclude: tuple[str | None, ...] = (),
) -> TransformExecutionResult:
    """Choose a feature subset using training labels only.

    An important distinction from target encoding, and one that is easy to get
    backwards: supervised selection **may** use the training labels of every training
    row. It is a decision about which columns to keep, not a per-row value, so there is
    no row-level self-influence to eliminate. Imposing out-of-fold discipline here would
    be wrong.

    What it may **not** do is see evaluation labels — that is what Check 6 of the
    fitting-scope audit verifies.

    Limitation worth stating plainly: if selection is later embedded in external
    cross-validation, the entire plan must be refitted inside each training fold.
    Selecting once on the full training set and then cross-validating gives optimistic
    estimates.
    """
    if method not in {"iv", "permutation", "correlation_vif", "mutual_info"}:
        raise ValueError(f"method={method!r} is not supported.")

    exclude = tuple(exclude) + (target_column,)
    numeric = numeric_columns(train, exclude)
    if not numeric:
        raise ValueError("selection requires at least one numeric feature.")

    scores: dict[str, float] = {}
    notes: list[str] = []

    if method == "correlation_vif":
        frame = train[numeric].apply(pd.to_numeric, errors="coerce").dropna()
        usable = [c for c in numeric if len(frame) and frame[c].nunique() > 1]
        matrix = frame[usable].to_numpy(dtype=float) if usable else np.empty((0, 0))
        for index, column in enumerate(usable):
            others = np.delete(matrix, index, axis=1)
            if others.size == 0:
                scores[column] = 1.0
                continue
            design = np.column_stack([np.ones(len(others)), others])
            coef, *_ = np.linalg.lstsq(design, matrix[:, index], rcond=None)
            residual = matrix[:, index] - design @ coef
            ss_tot = float(np.sum((matrix[:, index] - matrix[:, index].mean()) ** 2))
            r2 = 1.0 - float(np.sum(residual**2)) / ss_tot if ss_tot > 0 else 0.0
            vif = float("inf") if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
            scores[column] = -vif  # lower VIF is better, so negate to rank uniformly
        keep = [c for c in usable if -scores[c] < vif_threshold]
    else:
        if not target_column or target_column not in train.columns:
            raise ValueError(f"method={method!r} requires a target_column present in train.")
        frame = train[numeric + [target_column]].dropna()
        X = frame[numeric].to_numpy(dtype=float)
        yy = frame[target_column]

        if method == "permutation":
            if model is None:
                raise ValueError(
                    "permutation selection requires a fitted model exposing predict/"
                    "predict_proba; none was supplied."
                )
            from sklearn.inspection import permutation_importance

            imp = permutation_importance(model, X, yy, n_repeats=5, random_state=seed)
            scores = dict(zip(numeric, map(float, imp.importances_mean), strict=True))
        elif method == "iv":
            from start.tests.preprocessing._a3_diagnostics import _information_value

            yb = pd.to_numeric(yy, errors="coerce")
            for column in numeric:
                iv = _information_value(frame[column], yb)
                scores[column] = 0.0 if not math.isfinite(iv) else iv
        else:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

            is_classification = yy.nunique() <= 20 and str(yy.dtype).startswith(
                ("int", "bool", "str", "obj", "cat")
            )
            fn = mutual_info_classif if is_classification else mutual_info_regression
            values = fn(X, yy, random_state=seed)
            scores = dict(zip(numeric, map(float, values), strict=True))

        ranked = sorted(scores, key=lambda c: (-scores[c], c))
        if top_k:
            keep = sorted(ranked[:top_k])
        elif threshold is not None:
            keep = sorted(c for c in ranked if scores[c] >= threshold)
        else:
            keep = sorted(ranked)
            notes.append("No top_k or threshold supplied; all features retained and ranked.")

    dropped = sorted(set(numeric) - set(keep))

    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        present = [c for c in dropped if c in frame.columns]
        return frame.drop(columns=present)

    outputs = _apply([train, test, oos], transform)
    notes.append(
        "Supervised selection legitimately uses TRAINING labels for every training row "
        "— it is a column decision, not a per-row value, so out-of-fold discipline does "
        "not apply. It must not see evaluation labels."
    )
    notes.append(
        "If this selection is later embedded in external cross-validation, the whole "
        "plan must be refitted inside each training fold; selecting once on the full "
        "training set and then cross-validating gives optimistic estimates."
    )
    return _result(
        "selection",
        outputs,
        (train, test, oos),
        {
            "scores": {k: round(v, 10) for k, v in scores.items()},
            "kept": keep,
            "dropped": dropped,
            "method": method,
        },
        FittingScope.TRAIN_ONLY,
        {
            "method": method,
            "top_k": top_k,
            "threshold": threshold,
            "vif_threshold": vif_threshold,
            "seed": seed,
        },
        dropped,
        notes,
    )
