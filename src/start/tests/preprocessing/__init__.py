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
    cols = _numeric_columns(df, exclude=(ctx.target_column,))
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
    cols = [c for c in _numeric_columns(train, exclude=(ctx.target_column,)) if c in test.columns]
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
    cols = _numeric_columns(df, exclude=(target,))
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
    common_cols = sorted(set(train.columns) & set(test.columns))
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
