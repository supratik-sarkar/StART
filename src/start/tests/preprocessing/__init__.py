"""Preprocessing / data-quality test family. Pure, deterministic engines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from start.core.schemas import Status, TestResult, ThresholdSpec
from start.registry import TestContext, register_test


def _numeric_columns(df: pd.DataFrame, exclude: tuple[str | None, ...] = ()) -> list[str]:
    drop = {c for c in exclude if c}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]


def _non_feature_columns(ctx: TestContext) -> tuple[str | None, ...]:
    """Columns that are outputs/labels, not features: excluded from feature scans."""
    return (ctx.target_column, ctx.score_column, ctx.prediction_column)


@register_test(
    "preprocessing.missingness",
    family="preprocessing",
    name="Missingness profile",
    requires=("train",),
    default_params={"warn_pct": 5.0, "fail_pct": 30.0},
)
def missingness(ctx: TestContext, warn_pct: float = 5.0, fail_pct: float = 30.0) -> TestResult:
    """Per-column and overall missingness with thresholded status."""
    df: pd.DataFrame = ctx.train
    per_col = (df.isna().mean() * 100).round(4)
    overall = float(df.isna().to_numpy().mean() * 100)
    worst_col = str(per_col.idxmax()) if len(per_col) else ""
    result = TestResult(
        test_id="preprocessing.missingness",
        test_name="Missingness profile",
        params={"warn_pct": warn_pct, "fail_pct": fail_pct},
        metrics={
            "overall_missing_pct": round(overall, 4),
            "max_column_missing_pct": float(per_col.max()) if len(per_col) else 0.0,
            "worst_column": worst_col,
            "n_columns_over_warn": int((per_col > warn_pct).sum()),
        },
        thresholds=[
            ThresholdSpec(metric="overall_missing_pct", warn=warn_pct, fail=fail_pct),
            ThresholdSpec(metric="max_column_missing_pct", warn=warn_pct * 2, fail=fail_pct * 2),
        ],
        interpretation=(
            f"Overall missingness is {overall:.2f}%; "
            f"the most affected column is '{worst_col}'."
        ),
        limitations=["Missingness mechanism (MCAR/MAR/MNAR) is not inferred."],
    )
    return result.apply_thresholds()


@register_test(
    "preprocessing.outliers",
    family="preprocessing",
    name="IQR outlier scan",
    requires=("train",),
    default_params={"iqr_multiplier": 1.5, "warn_pct": 5.0, "fail_pct": 15.0},
)
def outliers(
    ctx: TestContext,
    iqr_multiplier: float = 1.5,
    warn_pct: float = 5.0,
    fail_pct: float = 15.0,
) -> TestResult:
    """Share of values outside Tukey fences per numeric column."""
    df: pd.DataFrame = ctx.train
    cols = _numeric_columns(df, exclude=_non_feature_columns(ctx))
    rates: dict[str, float] = {}
    for col in cols:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - iqr_multiplier * iqr, q3 + iqr_multiplier * iqr
        rates[col] = float(((series < lo) | (series > hi)).mean() * 100)
    max_rate = max(rates.values()) if rates else 0.0
    worst = max(rates, key=rates.get) if rates else ""  # type: ignore[arg-type]
    result = TestResult(
        test_id="preprocessing.outliers",
        test_name="IQR outlier scan",
        params={"iqr_multiplier": iqr_multiplier, "warn_pct": warn_pct, "fail_pct": fail_pct},
        metrics={
            "max_outlier_pct": round(max_rate, 4),
            "worst_column": worst,
            "n_numeric_columns": len(cols),
        },
        thresholds=[ThresholdSpec(metric="max_outlier_pct", warn=warn_pct, fail=fail_pct)],
        interpretation=(
            f"Maximum Tukey-fence outlier rate is {max_rate:.2f}% (column '{worst}')."
        ),
        limitations=["IQR fences assume roughly unimodal distributions."],
    )
    return result.apply_thresholds()


def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    eps = 1e-6
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    e_counts, _ = np.histogram(expected, bins=edges)
    a_counts, _ = np.histogram(actual, bins=edges)
    e_pct = np.clip(e_counts / max(e_counts.sum(), 1), eps, None)
    a_pct = np.clip(a_counts / max(a_counts.sum(), 1), eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


@register_test(
    "preprocessing.feature_drift",
    family="preprocessing",
    name="Feature drift (PSI + KS)",
    requires=("train", "test"),
    default_params={"psi_warn": 0.1, "psi_fail": 0.25},
)
def feature_drift(ctx: TestContext, psi_warn: float = 0.1, psi_fail: float = 0.25) -> TestResult:
    """PSI and KS statistics between train and test for numeric features."""
    train, test = ctx.train, ctx.test
    cols = [c for c in _numeric_columns(train, exclude=_non_feature_columns(ctx)) if c in test.columns]
    psis: dict[str, float] = {}
    ks_ps: dict[str, float] = {}
    for col in cols:
        a, b = train[col].dropna().to_numpy(), test[col].dropna().to_numpy()
        if len(a) < 10 or len(b) < 10:
            continue
        psis[col] = _psi(a, b)
        ks_ps[col] = float(stats.ks_2samp(a, b).pvalue)
    max_psi = max(psis.values()) if psis else 0.0
    worst = max(psis, key=psis.get) if psis else ""  # type: ignore[arg-type]
    result = TestResult(
        test_id="preprocessing.feature_drift",
        test_name="Feature drift (PSI + KS)",
        params={"psi_warn": psi_warn, "psi_fail": psi_fail},
        metrics={
            "max_psi": round(max_psi, 6),
            "worst_feature": worst,
            "min_ks_pvalue": round(min(ks_ps.values()), 6) if ks_ps else 1.0,
            "n_features_checked": len(psis),
        },
        thresholds=[ThresholdSpec(metric="max_psi", warn=psi_warn, fail=psi_fail)],
        interpretation=f"Maximum feature PSI between train and test is {max_psi:.4f} ('{worst}').",
        limitations=["Categorical drift and multivariate drift are not covered by this check."],
    )
    return result.apply_thresholds()


@register_test(
    "preprocessing.target_leakage",
    family="preprocessing",
    name="Target leakage screen",
    requires=("train", "target_column"),
    default_params={"warn_corr": 0.95, "fail_corr": 0.99},
)
def target_leakage(ctx: TestContext, warn_corr: float = 0.95, fail_corr: float = 0.99) -> TestResult:
    """Flags features with near-perfect absolute correlation to the target."""
    df: pd.DataFrame = ctx.train
    target = ctx.target_column
    if target is None or target not in df.columns:
        return TestResult(
            test_id="preprocessing.target_leakage",
            test_name="Target leakage screen",
            status=Status.SKIPPED,
            interpretation="No target column configured; leakage screen skipped.",
        )
    cols = _numeric_columns(df, exclude=_non_feature_columns(ctx))
    y = df[target]
    corrs = {c: float(abs(df[c].corr(y))) for c in cols if df[c].nunique() > 1}
    corrs = {c: (0.0 if np.isnan(v) else v) for c, v in corrs.items()}
    max_corr = max(corrs.values()) if corrs else 0.0
    worst = max(corrs, key=corrs.get) if corrs else ""  # type: ignore[arg-type]
    result = TestResult(
        test_id="preprocessing.target_leakage",
        test_name="Target leakage screen",
        params={"warn_corr": warn_corr, "fail_corr": fail_corr},
        metrics={"max_abs_target_corr": round(max_corr, 6), "worst_feature": worst},
        thresholds=[ThresholdSpec(metric="max_abs_target_corr", warn=warn_corr, fail=fail_corr)],
        interpretation=(
            f"Maximum absolute feature-target correlation is {max_corr:.4f} ('{worst}')."
        ),
        limitations=["Correlation screening misses non-linear and conditional leakage."],
    )
    return result.apply_thresholds()


@register_test(
    "preprocessing.split_diagnostics",
    family="preprocessing",
    name="Train/test split diagnostics",
    requires=("train", "test"),
    default_params={"warn_overlap_pct": 0.0, "fail_overlap_pct": 1.0},
)
def split_diagnostics(
    ctx: TestContext, warn_overlap_pct: float = 0.0, fail_overlap_pct: float = 1.0
) -> TestResult:
    """Row-overlap between train and test, plus size and class-balance checks."""
    train, test = ctx.train, ctx.test
    drop = {c for c in (ctx.score_column, ctx.prediction_column) if c}
    common_cols = sorted((set(train.columns) & set(test.columns)) - drop)
    train_hashes = pd.util.hash_pandas_object(train[common_cols], index=False)
    test_hashes = pd.util.hash_pandas_object(test[common_cols], index=False)
    overlap = float(test_hashes.isin(set(train_hashes)).mean() * 100)
    metrics: dict = {
        "test_rows_seen_in_train_pct": round(overlap, 4),
        "train_rows": len(train),
        "test_rows": len(test),
        "test_fraction": round(len(test) / max(len(train) + len(test), 1), 4),
    }
    if ctx.target_column and ctx.target_column in train.columns and ctx.target_column in test.columns:
        metrics["train_positive_rate"] = round(float(train[ctx.target_column].mean()), 4)
        metrics["test_positive_rate"] = round(float(test[ctx.target_column].mean()), 4)
    result = TestResult(
        test_id="preprocessing.split_diagnostics",
        test_name="Train/test split diagnostics",
        params={"warn_overlap_pct": warn_overlap_pct, "fail_overlap_pct": fail_overlap_pct},
        metrics=metrics,
        thresholds=[
            ThresholdSpec(
                metric="test_rows_seen_in_train_pct",
                warn=warn_overlap_pct,
                fail=fail_overlap_pct,
            )
        ],
        interpretation=f"{overlap:.2f}% of test rows are exact duplicates of train rows.",
        limitations=["Exact-duplicate detection misses near-duplicate contamination."],
    )
    return result.apply_thresholds()


@register_test(
    "preprocessing.duplicates",
    family="preprocessing",
    name="Duplicate row scan",
    requires=("train",),
    default_params={"warn_pct": 0.5, "fail_pct": 5.0},
)
def duplicates(ctx: TestContext, warn_pct: float = 0.5, fail_pct: float = 5.0) -> TestResult:
    """Share of exactly duplicated rows in the training data."""
    df: pd.DataFrame = ctx.train
    dup_pct = float(df.duplicated().mean() * 100)
    result = TestResult(
        test_id="preprocessing.duplicates",
        test_name="Duplicate row scan",
        params={"warn_pct": warn_pct, "fail_pct": fail_pct},
        metrics={"duplicate_row_pct": round(dup_pct, 4), "n_rows": len(df)},
        thresholds=[ThresholdSpec(metric="duplicate_row_pct", warn=warn_pct, fail=fail_pct)],
        interpretation=f"{dup_pct:.2f}% of training rows are exact duplicates.",
        limitations=["Near-duplicate rows are not detected by exact matching."],
    )
    return result.apply_thresholds()


@register_test(
    "preprocessing.constant_features",
    family="preprocessing",
    name="Constant / near-constant features",
    requires=("train",),
    default_params={"near_constant_top_freq": 0.99},
)
def constant_features(ctx: TestContext, near_constant_top_freq: float = 0.99) -> TestResult:
    """Counts features that are constant or dominated by a single value."""
    df: pd.DataFrame = ctx.train
    drop = {c for c in _non_feature_columns(ctx) if c}
    cols = [c for c in df.columns if c not in drop]
    constant, near_constant = [], []
    for col in cols:
        series = df[col].dropna()
        if series.empty or series.nunique() <= 1:
            constant.append(col)
            continue
        top_freq = float(series.value_counts(normalize=True).iloc[0])
        if top_freq >= near_constant_top_freq:
            near_constant.append(col)
    result = TestResult(
        test_id="preprocessing.constant_features",
        test_name="Constant / near-constant features",
        params={"near_constant_top_freq": near_constant_top_freq},
        metrics={
            "n_constant_features": len(constant),
            "n_near_constant_features": len(near_constant),
            "constant_features": ", ".join(constant[:10]),
            "near_constant_features": ", ".join(near_constant[:10]),
        },
        thresholds=[
            ThresholdSpec(metric="n_constant_features", warn=0.5, fail=5.5),
            ThresholdSpec(metric="n_near_constant_features", warn=0.5, fail=10.5),
        ],
        interpretation=(
            f"Found {len(constant)} constant and {len(near_constant)} near-constant features."
        ),
        limitations=["Near-constant detection uses a single top-frequency cutoff."],
    )
    return result.apply_thresholds()


@register_test(
    "preprocessing.high_cardinality",
    family="preprocessing",
    name="High-cardinality categorical scan",
    requires=("train",),
    default_params={"warn_unique_ratio": 0.5},
)
def high_cardinality(ctx: TestContext, warn_unique_ratio: float = 0.5) -> TestResult:
    """Flags object/categorical columns whose unique-value ratio is ID-like."""
    df: pd.DataFrame = ctx.train
    cat_cols = [
        c
        for c in df.select_dtypes(include=["object", "category"]).columns
        if c != ctx.target_column
    ]
    ratios = {
        c: float(df[c].nunique() / max(len(df), 1)) for c in cat_cols if len(df) > 0
    }
    max_ratio = max(ratios.values()) if ratios else 0.0
    worst = max(ratios, key=ratios.get) if ratios else ""  # type: ignore[arg-type]
    result = TestResult(
        test_id="preprocessing.high_cardinality",
        test_name="High-cardinality categorical scan",
        params={"warn_unique_ratio": warn_unique_ratio},
        metrics={
            "n_categorical_columns": len(cat_cols),
            "max_unique_ratio": round(max_ratio, 6),
            "worst_column": worst,
        },
        thresholds=[ThresholdSpec(metric="max_unique_ratio", warn=warn_unique_ratio, fail=0.95)],
        interpretation=(
            f"{len(cat_cols)} categorical columns scanned; maximum unique-value ratio "
            f"is {max_ratio:.2f}."
            if cat_cols
            else "No categorical columns present; nothing to scan."
        ),
        limitations=["Encoded-categorical columns stored as numerics are not scanned."],
    )
    return result.apply_thresholds()


@register_test(
    "preprocessing.feature_ranges",
    family="preprocessing",
    name="Numerical feature range summary",
    requires=("train",),
)
def feature_ranges(ctx: TestContext) -> TestResult:
    """Informational summary of numeric feature ranges (no thresholds)."""
    df: pd.DataFrame = ctx.train
    cols = _numeric_columns(df, exclude=_non_feature_columns(ctx))
    if not cols:
        return TestResult(
            test_id="preprocessing.feature_ranges",
            test_name="Numerical feature range summary",
            status=Status.SKIPPED,
            interpretation="No numeric feature columns present.",
        )
    desc = df[cols].describe()
    global_min = float(desc.loc["min"].min())
    global_max = float(desc.loc["max"].max())
    widest = str((desc.loc["max"] - desc.loc["min"]).idxmax())
    result = TestResult(
        test_id="preprocessing.feature_ranges",
        test_name="Numerical feature range summary",
        metrics={
            "n_numeric_features": len(cols),
            "global_min": round(global_min, 6),
            "global_max": round(global_max, 6),
            "widest_range_feature": widest,
            "n_negative_min_features": int((desc.loc["min"] < 0).sum()),
        },
        interpretation=(
            f"{len(cols)} numeric features span [{global_min:.4g}, {global_max:.4g}]; "
            f"the widest range belongs to '{widest}'."
        ),
        limitations=["Informational summary; range plausibility needs domain review."],
    )
    return result.apply_thresholds()
