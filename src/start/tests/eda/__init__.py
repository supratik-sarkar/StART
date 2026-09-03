"""Exploratory data analysis — six net-new registered tests.

Scope note (why this family is small)
-------------------------------------

An earlier draft of v4.2.0 proposed thirteen EDA tests. Auditing against the live
registry found that eight of them reimplemented engines that already work:
``preprocessing.missingness``, ``duplicates``, ``constant_features``,
``high_cardinality``, ``outliers`` (already an IQR scan), ``feature_drift`` (already
PSI + KS) and ``split_diagnostics``. Building those again would have been the opposite
of strengthening the codebase.

So this family contains only what genuinely did not exist: descriptive moments,
correlation structure, multicollinearity, distribution shape, categorical structure and
class balance.

Determinism
-----------

Every test here is ``numerical``, not ``exact``. Correlation, VIF, condition number,
skew, kurtosis and every ``scipy.stats`` routine go through BLAS/LAPACK, and Accelerate
on macOS does not agree bitwise with OpenBLAS on Linux. Claiming exactness would assert
a property the platform does not provide, and the usual consequence is that someone
loosens the assertion until CI stops complaining — which destroys the guarantee outright.
Counts of columns and levels are exact; the numbers computed from them are not.

Statistical discipline
----------------------

A normality test that fails to reject does not establish normality. Every such result is
worded as a non-rejection and carries the caveat. This is the same discipline the VaR
family will need in Gate B, and it is cheaper to establish here.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from start.core.schemas import Status, TestResult, ThresholdSpec
from start.registry import TestContext, register_test
from start.tests._target_dispatch import require_target_type

__all__ = [
    "descriptive_statistics",
    "correlation",
    "multicollinearity",
    "numeric_distribution",
    "categorical_distribution",
    "class_imbalance",
    "DEFAULT_PERCENTILES",
    "SHAPIRO_MAX_N",
    "SHAPIRO_MIN_N",
]

DEFAULT_PERCENTILES: tuple[float, ...] = (1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0)

#: Shapiro-Wilk loses reliability on large samples — it rejects trivially small,
#: immaterial departures from normality once n is big enough. SciPy itself warns above
#: 5000. Below 3 observations it is undefined.
SHAPIRO_MAX_N = 5000
SHAPIRO_MIN_N = 3

#: Shared metadata. These IDs are taken from the live ``start.risk`` taxonomy, not
#: invented: verified against ``stripe_ids()``, ``dimension_ids()`` and
#: ``object_kind_ids()``.
_EDA_STRIPES = ("model", "credit")
_EDA_OBJECTS = ("ml_model", "statistical_model", "scorecard", "deep_learning_model")


# --------------------------------------------------------------------------- #
# Shared helpers — mirrors the conventions already used by the preprocessing family
# --------------------------------------------------------------------------- #
def _non_feature_columns(ctx: TestContext) -> tuple[str | None, ...]:
    """Outputs and labels are not features and must not be scanned as such.

    Same convention as ``start.tests.preprocessing``; duplicated rather than imported
    to avoid a cross-family import for three lines, and kept identical deliberately.
    """
    return (ctx.target_column, ctx.score_column, ctx.prediction_column)


def _numeric_columns(df: pd.DataFrame, exclude: tuple[str | None, ...] = ()) -> list[str]:
    drop = {c for c in exclude if c}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]


def _categorical_columns(df: pd.DataFrame, exclude: tuple[str | None, ...] = ()) -> list[str]:
    drop = {c for c in exclude if c}
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    return [c for c in df.columns if c not in numeric and c not in drop]


def _finite(series: pd.Series) -> np.ndarray:
    """Drop NaN and infinities.

    Infinities are dropped rather than propagated: a single ``inf`` turns every moment
    into ``nan`` and the whole column's evidence silently disappears. The count of
    dropped non-finite values is reported wherever this is used.
    """
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _skipped(test_id: str, name: str, reason: str, **params: Any) -> TestResult:
    return TestResult(
        test_id=test_id,
        test_name=name,
        status=Status.SKIPPED,
        params=params,
        interpretation=reason,
    )


# --------------------------------------------------------------------------- #
# 1. descriptive_statistics
# --------------------------------------------------------------------------- #
@register_test(
    "eda.descriptive_statistics",
    family="eda",
    name="Descriptive statistics",
    requires=("train",),
    default_params={"percentiles": DEFAULT_PERCENTILES},
    context_type="tabular",
    risk_stripes=("model", "credit"),
    risk_dimensions=("data_quality_lineage",),
    object_kinds=("ml_model", "statistical_model", "scorecard", "deep_learning_model"),
)
def descriptive_statistics(
    ctx: TestContext,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
) -> TestResult:
    """Moments and percentiles for numeric features. Records what was excluded and why.

    Determinism: ``numerical``. Counts are exact; moments are not.
    """
    df: pd.DataFrame = ctx.train
    excluded_non_feature = [c for c in _non_feature_columns(ctx) if c]
    numeric = _numeric_columns(df, exclude=_non_feature_columns(ctx))
    non_numeric = _categorical_columns(df, exclude=_non_feature_columns(ctx))

    if not numeric:
        return _skipped(
            "eda.descriptive_statistics",
            "Descriptive statistics",
            "No numeric feature columns are available after excluding target, score and "
            "prediction columns. Categorical columns are not coerced to numeric — doing "
            "so would produce moments of an arbitrary encoding.",
            percentiles=list(percentiles),
        )

    metrics: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_numeric_columns": len(numeric),
        "n_excluded_non_numeric": len(non_numeric),
    }
    per_column: dict[str, dict[str, float]] = {}
    total_non_finite = 0

    for column in numeric:
        raw = pd.to_numeric(df[column], errors="coerce")
        values = _finite(raw)
        n_missing = int(raw.isna().sum())
        n_non_finite = int(len(raw) - n_missing - len(values))
        total_non_finite += n_non_finite

        if values.size == 0:
            per_column[column] = {"count": 0.0, "n_missing": float(n_missing)}
            continue

        summary: dict[str, float] = {
            "count": float(values.size),
            "n_missing": float(n_missing),
            "n_non_finite": float(n_non_finite),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
        for pct in percentiles:
            summary[f"p{pct:g}"] = float(np.percentile(values, pct))
        # Moments are undefined for a constant column: scipy returns nan with a warning.
        # Reporting 0.0 with the reason is more useful than a nan nobody can act on.
        if values.size > 2 and float(np.std(values)) > 0:
            summary["skew"] = float(stats.skew(values, bias=False))
            summary["excess_kurtosis"] = float(stats.kurtosis(values, fisher=True, bias=False))
        else:
            summary["skew"] = 0.0
            summary["excess_kurtosis"] = 0.0
        per_column[column] = summary

    metrics["n_non_finite_values"] = total_non_finite
    metrics["columns_included"] = ", ".join(numeric)
    metrics["columns_excluded_non_numeric"] = ", ".join(non_numeric)
    for column, summary in per_column.items():
        for stat_name, value in summary.items():
            metrics[f"{column}.{stat_name}"] = round(value, 6)

    limitations = [
        "Descriptive only; asserts nothing and applies no threshold.",
        "Moments are numerical, not bitwise reproducible across BLAS implementations.",
    ]
    if non_numeric:
        limitations.append(
            f"{len(non_numeric)} non-numeric column(s) excluded rather than coerced: "
            f"{', '.join(non_numeric[:8])}."
        )
    if excluded_non_feature:
        limitations.append(
            f"Excluded as outputs rather than features: {', '.join(excluded_non_feature)}."
        )
    if total_non_finite:
        limitations.append(
            f"{total_non_finite} non-finite value(s) dropped before computing moments."
        )

    return TestResult(
        test_id="eda.descriptive_statistics",
        test_name="Descriptive statistics",
        status=Status.RECORDED,
        params={"percentiles": list(percentiles)},
        metrics=metrics,
        interpretation=(
            f"Profiled {len(numeric)} numeric feature(s) over {len(df):,} rows; "
            f"{len(non_numeric)} non-numeric column(s) excluded."
        ),
        limitations=limitations,
    )


# --------------------------------------------------------------------------- #
# 2. correlation
# --------------------------------------------------------------------------- #
@register_test(
    "eda.correlation",
    family="eda",
    name="Correlation structure",
    requires=("train",),
    default_params={"method": "pearson", "high": 0.80, "emit_heatmap": False},
    context_type="tabular",
    risk_stripes=("model", "credit"),
    risk_dimensions=("data_quality_lineage", "conceptual_soundness"),
    object_kinds=("ml_model", "statistical_model", "scorecard"),
)
def correlation(
    ctx: TestContext,
    method: str = "pearson",
    high: float = 0.80,
    emit_heatmap: bool = False,
) -> TestResult:
    """Pairwise correlation structure among numeric features.

    The diagonal is excluded from every maximum and count. Including it would make
    ``max_abs_correlation`` identically 1.0 for any frame, which is not a finding.

    The heatmap is an artifact; the statistics are the evidence. No conclusion depends
    on the figure, and a reviewer who cannot see it reaches the same verdict from
    ``metrics``. Determinism: ``numerical``.
    """
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError(
            f"method={method!r} is not supported. Use pearson, spearman or kendall."
        )

    df: pd.DataFrame = ctx.train
    numeric = _numeric_columns(df, exclude=_non_feature_columns(ctx))
    usable = [c for c in numeric if pd.to_numeric(df[c], errors="coerce").nunique() > 1]
    degenerate = [c for c in numeric if c not in usable]

    if len(usable) < 2:
        return _skipped(
            "eda.correlation",
            "Correlation structure",
            f"Correlation requires at least two non-constant numeric features; "
            f"{len(usable)} available.",
            method=method,
            high=high,
        )

    matrix = df[usable].corr(method=method)
    values = matrix.to_numpy(dtype=float)
    # Exclude the diagonal explicitly rather than relying on it being 1.0 — a constant
    # column produces a nan diagonal, and masking by position is unambiguous.
    mask = ~np.eye(len(usable), dtype=bool)
    off_diagonal = np.abs(values[mask])
    off_diagonal = off_diagonal[np.isfinite(off_diagonal)]

    if off_diagonal.size == 0:
        return _skipped(
            "eda.correlation",
            "Correlation structure",
            "No finite off-diagonal correlations could be computed.",
            method=method,
            high=high,
        )

    # Each unordered pair appears twice in the masked array; count pairs, not cells.
    max_abs = float(off_diagonal.max())
    mean_abs = float(off_diagonal.mean())
    n_above = int((off_diagonal >= high).sum() // 2)

    pairs: list[tuple[str, str, float]] = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            value = float(values[i, j])
            if math.isfinite(value) and abs(value) >= high:
                pairs.append((usable[i], usable[j], value))
    pairs.sort(key=lambda p: -abs(p[2]))

    strongest = pairs[0] if pairs else None
    if strongest is None:
        flat: list[tuple[str, str, float]] = []
        for i in range(len(usable)):
            for j in range(i + 1, len(usable)):
                value = float(values[i, j])
                if math.isfinite(value):
                    flat.append((usable[i], usable[j], value))
        strongest = max(flat, key=lambda p: abs(p[2])) if flat else None

    metrics: dict[str, Any] = {
        "max_abs_correlation": round(max_abs, 6),
        "mean_abs_correlation": round(mean_abs, 6),
        "n_pairs_above_threshold": n_above,
        "n_features": len(usable),
        "n_pairs_evaluated": len(usable) * (len(usable) - 1) // 2,
    }
    if strongest is not None:
        metrics["strongest_pair_left"] = strongest[0]
        metrics["strongest_pair_right"] = strongest[1]
        metrics["strongest_pair_correlation"] = round(float(strongest[2]), 6)
    for left, right, value in pairs[:20]:
        metrics[f"pair.{left}~{right}"] = round(float(value), 6)

    artifacts: dict[str, str] = {}
    if emit_heatmap:
        path = _render_correlation_heatmap(matrix, method, high, ctx)
        if path:
            artifacts["correlation_heatmap"] = path

    return TestResult(
        test_id="eda.correlation",
        test_name="Correlation structure",
        params={"method": method, "high": high, "emit_heatmap": emit_heatmap},
        metrics=metrics,
        thresholds=[ThresholdSpec(metric="max_abs_correlation", warn=high, fail=0.99)],
        interpretation=(
            f"Strongest absolute {method} correlation among {len(usable)} features is "
            f"{max_abs:.4f}; {n_above} pair(s) at or above {high:g}."
        ),
        limitations=[
            f"{method.title()} correlation measures "
            + (
                "linear association only and is sensitive to outliers."
                if method == "pearson"
                else "monotone association only."
            ),
            "Pairwise association is not evidence of redundancy on its own; see "
            "eda.multicollinearity for the multivariate view.",
            "Numerical, not bitwise reproducible across BLAS implementations.",
        ]
        + (
            [f"{len(degenerate)} constant column(s) excluded: {', '.join(degenerate[:8])}."]
            if degenerate
            else []
        ),
        artifacts=artifacts,
    ).apply_thresholds()


def _render_correlation_heatmap(
    matrix: pd.DataFrame, method: str, high: float, ctx: TestContext
) -> str:
    """Deterministic heatmap. Returns "" when matplotlib is unavailable.

    Determinism here is *semantic*: fixed figure size, DPI, column ordering, colour map
    and annotation. It is deliberately not byte identity — matplotlib PNG output varies
    with FreeType version, backend and encoder, and asserting byte equality would fail
    in CI for reasons unrelated to correctness.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    try:
        from pathlib import Path

        columns = sorted(matrix.columns)
        ordered = matrix.loc[columns, columns]
        n = len(columns)

        fig, ax = plt.subplots(figsize=(max(4.0, 0.5 * n + 2), max(3.5, 0.5 * n + 1.5)), dpi=110)
        # Diverging, colourblind-safe, symmetric about zero so sign is readable.
        image = ax.imshow(ordered.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, cmap="RdBu_r")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(columns, fontsize=8)
        ax.set_title(f"{method.title()} correlation  (|r| >= {high:g} boxed)", fontsize=10)

        for i in range(n):
            for j in range(n):
                value = float(ordered.iat[i, j])
                if not math.isfinite(value):
                    continue
                # Magnitude is printed, so the reading never depends on colour alone.
                ax.text(
                    j, i, f"{value:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(value) > 0.55 else "black",
                )
                if i != j and abs(value) >= high:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.5, i - 0.5), 1, 1,
                            fill=False, edgecolor="black", linewidth=1.8,
                        )
                    )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()

        out_dir = Path(str((ctx.extra or {}).get("output_dir", "start_output/figures")))
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "eda_correlation_heatmap.png"
        fig.savefig(path)
        plt.close(fig)
        return str(path)
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# 3. multicollinearity
# --------------------------------------------------------------------------- #
@register_test(
    "eda.multicollinearity",
    family="eda",
    name="Multicollinearity (VIF)",
    requires=("train",),
    default_params={"vif_warn": 5.0, "vif_fail": 10.0},
    context_type="tabular",
    risk_stripes=("model", "credit"),
    risk_dimensions=("conceptual_soundness",),
    object_kinds=("statistical_model", "scorecard", "ml_model"),
)
def multicollinearity(
    ctx: TestContext,
    vif_warn: float = 5.0,
    vif_fail: float = 10.0,
) -> TestResult:
    """Variance inflation factors via auxiliary OLS regressions.

    For each numeric feature *j*, regress it on the remaining numeric features and take
    ``VIF_j = 1 / (1 - R²_j)``. Solved with ``numpy.linalg.lstsq`` — statsmodels is not
    needed for what is ultimately one division.

    Perfect collinearity yields ``R² = 1`` and an infinite VIF. That is reported as
    ``inf`` and counted separately rather than being allowed to poison ``max_vif`` and
    ``mean_vif``, which would otherwise both become ``inf`` and hide how many features
    are actually affected. Determinism: ``numerical``.
    """
    df: pd.DataFrame = ctx.train
    numeric = _numeric_columns(df, exclude=_non_feature_columns(ctx))
    frame = df[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    constant = [c for c in numeric if frame[c].nunique() <= 1] if len(frame) else list(numeric)
    usable = [c for c in numeric if c not in constant]
    non_numeric = _categorical_columns(df, exclude=_non_feature_columns(ctx))

    if len(usable) < 2 or len(frame) < len(usable) + 1:
        return _skipped(
            "eda.multicollinearity",
            "Multicollinearity (VIF)",
            f"VIF requires at least two non-constant numeric features and more complete "
            f"rows than features; {len(usable)} feature(s), {len(frame)} complete row(s).",
            vif_warn=vif_warn,
            vif_fail=vif_fail,
        )

    matrix = frame[usable].to_numpy(dtype=float)
    vifs: dict[str, float] = {}

    for index, column in enumerate(usable):
        y = matrix[:, index]
        others = np.delete(matrix, index, axis=1)
        design = np.column_stack([np.ones(len(others)), others])
        try:
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            residuals = y - design @ coefficients
            ss_res = float(np.sum(residuals**2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        except np.linalg.LinAlgError:
            r_squared = 1.0
        # Guard the division rather than letting 1/0 raise: an exactly collinear
        # feature is a finding to report, not an exception to propagate.
        vifs[column] = float("inf") if r_squared >= 1.0 - 1e-12 else 1.0 / (1.0 - r_squared)

    finite = {c: v for c, v in vifs.items() if math.isfinite(v)}
    n_perfect = len(vifs) - len(finite)
    max_vif = max(finite.values()) if finite else float("inf")
    mean_vif = float(np.mean(list(finite.values()))) if finite else float("inf")

    try:
        correlation_matrix = np.corrcoef(matrix, rowvar=False)
        condition_number = float(np.linalg.cond(correlation_matrix))
    except Exception:
        condition_number = float("nan")

    metrics: dict[str, Any] = {
        "max_vif": round(max_vif, 6) if math.isfinite(max_vif) else float("inf"),
        "mean_vif": round(mean_vif, 6) if math.isfinite(mean_vif) else float("inf"),
        "n_above_warn": int(sum(1 for v in vifs.values() if v >= vif_warn)),
        "n_above_fail": int(sum(1 for v in vifs.values() if v >= vif_fail)),
        "n_perfectly_collinear": n_perfect,
        "n_features_evaluated": len(usable),
        "n_complete_rows": int(len(frame)),
        "condition_number": round(condition_number, 6)
        if math.isfinite(condition_number)
        else float("inf"),
    }
    for column, value in vifs.items():
        metrics[f"vif.{column}"] = round(value, 6) if math.isfinite(value) else float("inf")

    limitations = [
        "VIF measures linear dependence only; a non-linear relationship between "
        "features produces a low VIF and is not detected here.",
        "Numerical, not bitwise reproducible across BLAS implementations.",
    ]
    if non_numeric:
        limitations.append(
            f"{len(non_numeric)} non-numeric column(s) excluded; VIF is undefined for "
            f"them unless they are already encoded: {', '.join(non_numeric[:8])}."
        )
    if constant:
        limitations.append(f"{len(constant)} constant column(s) excluded: {', '.join(constant[:8])}.")
    if n_perfect:
        limitations.append(
            f"{n_perfect} feature(s) are exactly collinear with the others (VIF = inf) "
            "and are excluded from max_vif and mean_vif so those statistics stay readable."
        )

    if n_perfect:
        interpretation = (
            f"{n_perfect} feature(s) are perfectly collinear with the remainder. Among the "
            f"rest, the highest VIF is {max_vif:.3f}."
            if finite
            else f"All {n_perfect} evaluated feature(s) are perfectly collinear."
        )
    else:
        interpretation = (
            f"Highest VIF among {len(usable)} feature(s) is {max_vif:.3f}; "
            f"{metrics['n_above_warn']} at or above the warning level of {vif_warn:g}."
        )

    result = TestResult(
        test_id="eda.multicollinearity",
        test_name="Multicollinearity (VIF)",
        params={"vif_warn": vif_warn, "vif_fail": vif_fail},
        metrics=metrics,
        thresholds=[ThresholdSpec(metric="max_vif", warn=vif_warn, fail=vif_fail)],
        interpretation=interpretation,
        limitations=limitations,
    )
    applied = result.apply_thresholds()
    # A perfectly collinear feature is a FAIL regardless of the finite maximum: the
    # design matrix is singular and any coefficient on those features is arbitrary.
    if n_perfect:
        applied.status = Status.FAIL
    return applied


# --------------------------------------------------------------------------- #
# 4. numeric_distribution
# --------------------------------------------------------------------------- #
@register_test(
    "eda.numeric_distribution",
    family="eda",
    name="Numeric distribution shape",
    requires=("train",),
    default_params={"normality_alpha": 0.05},
    context_type="tabular",
    risk_stripes=("model", "credit"),
    risk_dimensions=("data_quality_lineage", "assumption_validity"),
    object_kinds=("statistical_model", "ml_model", "deep_learning_model"),
)
def numeric_distribution(
    ctx: TestContext,
    normality_alpha: float = 0.05,
) -> TestResult:
    """Distribution shape per numeric feature, with a normality screen.

    Shapiro-Wilk is used within its reliable range and Jarque-Bera outside it. Which was
    applied is recorded per column, because the two answer subtly different questions
    and a reader comparing columns needs to know they were not tested the same way.

    Wording matters here. A non-rejection is reported as a non-rejection, never as
    evidence that a distribution *is* normal. Determinism: ``numerical``.
    """
    df: pd.DataFrame = ctx.train
    numeric = _numeric_columns(df, exclude=_non_feature_columns(ctx))
    if not numeric:
        return _skipped(
            "eda.numeric_distribution",
            "Numeric distribution shape",
            "No numeric feature columns available.",
            normality_alpha=normality_alpha,
        )

    metrics: dict[str, Any] = {"n_numeric_columns": len(numeric)}
    tested = 0
    rejected = 0
    skipped_columns: list[str] = []
    min_p = 1.0

    for column in numeric:
        raw = pd.to_numeric(df[column], errors="coerce")
        values = _finite(raw)
        n = int(values.size)
        metrics[f"{column}.n"] = n

        if n < SHAPIRO_MIN_N or float(np.std(values)) == 0.0:
            skipped_columns.append(column)
            metrics[f"{column}.normality_test"] = "none"
            continue

        metrics[f"{column}.skew"] = round(float(stats.skew(values, bias=False)), 6)
        metrics[f"{column}.excess_kurtosis"] = round(
            float(stats.kurtosis(values, fisher=True, bias=False)), 6
        )

        # Shapiro-Wilk rejects on immaterial departures once n is large, and SciPy warns
        # above 5000. Switching to Jarque-Bera there is a deliberate choice, recorded so
        # the reader is not comparing two different tests unknowingly.
        try:
            if n <= SHAPIRO_MAX_N:
                statistic, p_value = stats.shapiro(values)
                test_used = "shapiro_wilk"
            else:
                statistic, p_value = stats.jarque_bera(values)
                test_used = "jarque_bera"
        except Exception:
            skipped_columns.append(column)
            metrics[f"{column}.normality_test"] = "none"
            continue

        tested += 1
        metrics[f"{column}.normality_test"] = test_used
        metrics[f"{column}.normality_statistic"] = round(float(statistic), 6)
        metrics[f"{column}.normality_p_value"] = round(float(p_value), 6)
        if float(p_value) < normality_alpha:
            rejected += 1
        min_p = min(min_p, float(p_value))

    metrics["n_columns_tested"] = tested
    metrics["n_normality_rejected"] = rejected
    metrics["n_columns_skipped"] = len(skipped_columns)
    metrics["min_normality_p_value"] = round(min_p, 6)

    limitations = [
        "A normality test that does not reject has not established normality; it has "
        "failed to find evidence against it at the stated level.",
        "Shapiro-Wilk rejects immaterial departures at large n, which is why "
        f"Jarque-Bera is substituted above {SHAPIRO_MAX_N:,} observations. The test "
        "applied is recorded per column.",
        "No multiplicity correction is applied across columns; with many features some "
        "rejections are expected by chance alone.",
        "Numerical, not bitwise reproducible across BLAS implementations.",
    ]
    if skipped_columns:
        limitations.append(
            f"{len(skipped_columns)} column(s) not tested (constant, too few finite "
            f"observations, or the test failed): {', '.join(skipped_columns[:8])}."
        )

    return TestResult(
        test_id="eda.numeric_distribution",
        test_name="Numeric distribution shape",
        status=Status.RECORDED,
        params={"normality_alpha": normality_alpha},
        metrics=metrics,
        interpretation=(
            f"Tested {tested} of {len(numeric)} numeric feature(s); normality was "
            f"rejected at the {normality_alpha:g} level for {rejected}."
        ),
        limitations=limitations,
    )


# --------------------------------------------------------------------------- #
# 5. categorical_distribution
# --------------------------------------------------------------------------- #
@register_test(
    "eda.categorical_distribution",
    family="eda",
    name="Categorical distribution",
    requires=("train",),
    default_params={"rare_pct": 1.0},
    context_type="tabular",
    risk_stripes=("model", "credit"),
    risk_dimensions=("data_quality_lineage",),
    object_kinds=("scorecard", "ml_model", "statistical_model"),
)
def categorical_distribution(ctx: TestContext, rare_pct: float = 1.0) -> TestResult:
    """Level structure of categorical features: concentration, rarity and entropy.

    Target, score and prediction columns are excluded — they are outputs, not
    predictors, and the existing preprocessing family already applies this convention.

    Missing values are counted but never treated as a level. Whether missingness is a
    category is a modelling decision the reviewer makes at feature engineering, and
    silently folding it in here would prejudge it. Determinism: ``numerical`` (entropy);
    level counts are exact.
    """
    df: pd.DataFrame = ctx.train
    categorical = _categorical_columns(df, exclude=_non_feature_columns(ctx))
    if not categorical:
        return _skipped(
            "eda.categorical_distribution",
            "Categorical distribution",
            "No categorical feature columns available after excluding target, score and "
            "prediction columns.",
            rare_pct=rare_pct,
        )

    metrics: dict[str, Any] = {"n_categorical_columns": len(categorical), "n_rows": int(len(df))}
    max_mode_share = 0.0
    total_rare = 0
    worst_column = ""

    for column in categorical:
        series = df[column]
        n_missing = int(series.isna().sum())
        observed = series.dropna()
        n_observed = int(observed.size)

        metrics[f"{column}.n_missing"] = n_missing
        metrics[f"{column}.missing_pct"] = round(100.0 * n_missing / max(len(df), 1), 4)

        if n_observed == 0:
            metrics[f"{column}.n_unique"] = 0
            metrics[f"{column}.mode_share_pct"] = 0.0
            metrics[f"{column}.n_rare_levels"] = 0
            metrics[f"{column}.entropy"] = 0.0
            continue

        counts = observed.value_counts()
        shares = counts / n_observed
        mode_share = float(shares.iloc[0] * 100)
        # Strictly less than: a level sitting exactly on the threshold is not rare.
        # Implementations differ on this boundary and the choice changes the count, so
        # it is stated rather than left to the reader to infer.
        n_rare = int((shares * 100 < rare_pct).sum())
        # Shannon entropy in nats; 0 for a single level, which is correct and not an error.
        entropy = float(-(shares * np.log(shares)).sum()) if len(shares) > 1 else 0.0

        metrics[f"{column}.n_unique"] = int(counts.size)
        metrics[f"{column}.mode_share_pct"] = round(mode_share, 4)
        metrics[f"{column}.mode_level"] = str(counts.index[0])
        metrics[f"{column}.n_rare_levels"] = n_rare
        metrics[f"{column}.entropy"] = round(entropy, 6)
        # Normalised entropy makes columns with different level counts comparable.
        metrics[f"{column}.normalised_entropy"] = (
            round(entropy / math.log(counts.size), 6) if counts.size > 1 else 0.0
        )

        total_rare += n_rare
        if mode_share > max_mode_share:
            max_mode_share = mode_share
            worst_column = column

    metrics["max_mode_share_pct"] = round(max_mode_share, 4)
    metrics["most_concentrated_column"] = worst_column
    metrics["n_rare_levels_total"] = total_rare

    return TestResult(
        test_id="eda.categorical_distribution",
        test_name="Categorical distribution",
        status=Status.RECORDED,
        params={"rare_pct": rare_pct},
        metrics=metrics,
        interpretation=(
            f"Profiled {len(categorical)} categorical feature(s); highest single-level "
            f"concentration is {max_mode_share:.2f}%"
            + (f" in '{worst_column}'" if worst_column else "")
            + f"; {total_rare} level(s) below {rare_pct:g}% of observations."
        ),
        limitations=[
            "Descriptive only; asserts nothing and applies no threshold.",
            "Missing values are counted but not treated as a level — whether "
            "missingness is itself a category is a feature-engineering decision.",
            "Rare levels are counted, not merged; grouping is a separate decision.",
            "Entropy is reported in nats.",
        ],
    )


# --------------------------------------------------------------------------- #
# 6. class_imbalance
# --------------------------------------------------------------------------- #
@register_test(
    "eda.class_imbalance",
    family="eda",
    name="Class imbalance",
    requires=("train", "target_column"),
    default_params={"warn_ratio": 0.10, "fail_ratio": 0.01},
    context_type="tabular",
    risk_stripes=("model", "credit", "fraud"),
    risk_dimensions=("conceptual_soundness", "data_quality_lineage"),
    object_kinds=("ml_model", "scorecard", "deep_learning_model"),
)
def class_imbalance(
    ctx: TestContext,
    warn_ratio: float = 0.10,
    fail_ratio: float = 0.01,
) -> TestResult:
    """Class balance for a classification target.

    Uses the A1 target dispatch guard. A continuous target returns ``SKIPPED`` rather
    than a class distribution over rounded floats, which would look like a result and
    mean nothing.

    Thresholds are on the *minority share*, so lower is worse — the opposite direction
    to most thresholds in the registry, which is stated in the interpretation so nobody
    reads a low number as good.
    """
    inference, skip = require_target_type(ctx, "binary", "multiclass")
    if skip is not None:
        skip.test_id = "eda.class_imbalance"
        skip.test_name = "Class imbalance"
        skip.params = {**skip.params, "warn_ratio": warn_ratio, "fail_ratio": fail_ratio}
        return skip

    df: pd.DataFrame = ctx.train
    series = df[ctx.target_column].dropna()
    n_missing = int(df[ctx.target_column].isna().sum())

    if series.empty:
        return _skipped(
            "eda.class_imbalance",
            "Class imbalance",
            "Target column contains no non-missing values.",
            warn_ratio=warn_ratio,
            fail_ratio=fail_ratio,
        )

    counts = series.value_counts()
    shares = counts / int(series.size)
    minority_share = float(shares.min())
    majority_share = float(shares.max())
    imbalance_ratio = float(majority_share / minority_share) if minority_share > 0 else float("inf")

    metrics: dict[str, Any] = {
        "minority_class_share": round(minority_share, 6),
        "majority_class_share": round(majority_share, 6),
        "imbalance_ratio": round(imbalance_ratio, 6)
        if math.isfinite(imbalance_ratio)
        else float("inf"),
        "n_classes": int(counts.size),
        "n_observations": int(series.size),
        "n_missing_target": n_missing,
        "minority_class": str(shares.idxmin()),
        "majority_class": str(shares.idxmax()),
        **inference.as_params(),
    }
    for level, count in counts.items():
        metrics[f"class.{level}.count"] = int(count)
        metrics[f"class.{level}.share"] = round(float(count / series.size), 6)

    limitations = [
        "Thresholds are on the minority share, where a LOWER value is worse — the "
        "opposite direction to most thresholds in this registry.",
        "Class balance is a property of the sample, not of the population; a "
        "deliberately stratified sample will not reflect the true prevalence.",
        "Counts are exact; shares and the imbalance ratio are numerical.",
    ]
    if inference.is_ambiguous:
        limitations.append(
            "Target type was inferred, not stated: " + inference.detail
        )

    result = TestResult(
        test_id="eda.class_imbalance",
        test_name="Class imbalance",
        params={
            "warn_ratio": warn_ratio,
            "fail_ratio": fail_ratio,
            **inference.as_params(),
        },
        metrics=metrics,
        thresholds=[
            ThresholdSpec(
                metric="minority_class_share",
                warn=warn_ratio,
                fail=fail_ratio,
                direction="lower",
            )
        ],
        interpretation=(
            f"{counts.size} class(es) over {series.size:,} labelled observations; the "
            f"smallest class holds {minority_share:.2%} of them "
            f"(majority-to-minority ratio {imbalance_ratio:.1f}:1)."
        ),
        limitations=limitations,
    )
    return result.apply_thresholds()
