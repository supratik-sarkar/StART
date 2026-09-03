"""A3 — preprocessing diagnostic siblings.

Twelve new tests in the **existing** ``preprocessing`` family. Nothing here removes,
renames or replaces the v4.1.1 engines: ``preprocessing.target_leakage`` remains the
umbrella screen and these sit alongside it as specific detectors.

Why the leakage detectors are split rather than one test
--------------------------------------------------------

The v4.1.1 umbrella screen answers "is anything suspiciously correlated with the
target?". That is one question with one answer, and it cannot express the distinction
that actually matters to a reviewer: **reconstruction is a defect, correlation is a
finding**.

A feature that deterministically reproduces the target is a broken pipeline — someone
joined the outcome back onto the feature table. A feature correlating 0.97 with the
target might be that, or might be a genuinely excellent predictor. Behaviour score
against default. Days-past-due against default. Bureau score against default. All
legitimately correlate very highly, and a tool that FAILs them teaches reviewers to
ignore it.

So the severities differ by design:

    reconstruction (exact / affine / invertible monotone)  ->  FAIL
    high correlation                                       ->  WARN, never FAIL
    suspicious single-feature predictivity                 ->  WARN, never FAIL
    naming heuristic                                       ->  WARN, framed as an
                                                               observation about names

Only the first is a statement about the data-generating process. The others are
statements about what the reviewer should look at next.

Statistical discipline
----------------------

Non-rejection is reported as non-rejection. A chi-square test that fails to reject has
not established that two categorical distributions are equal, and every such result says
so.

Determinism
-----------

All of these are ``numerical``: correlation, mutual information, PCA and every
``scipy.stats`` routine go through BLAS/LAPACK, which is not bitwise reproducible across
platforms. Counts of rows, features and levels are exact.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from start.core.schemas import Status, TestResult, ThresholdSpec
from start.registry import TestContext, register_test
from start.tests._target_dispatch import require_target_type

__all__ = [
    "leakage_target_reconstruction",
    "leakage_high_correlation",
    "leakage_temporal",
    "leakage_entity_overlap",
    "leakage_row_overlap",
    "leakage_suspicious_predictivity",
    "leakage_name_heuristic",
    "target_analysis",
    "feature_target_relationship",
    "redundancy",
    "dimensionality_diagnostic",
    "categorical_drift",
    "LEAKAGE_NAME_PATTERNS",
]

#: Word-boundary aware, case-normalised. Substring matching without boundaries flags
#: "postcode" for "post" and "labelled_region" for "label", which is exactly the noise
#: that gets a WARN-only check switched off.
LEAKAGE_NAME_PATTERNS: tuple[str, ...] = (
    "target", "label", "outcome", "future", "actual", "post",
)

_STRIPES = ("model", "credit")
_OBJECTS = ("ml_model", "statistical_model", "scorecard", "data_pipeline")


# --------------------------------------------------------------------------- #
# Local helpers. Deliberately not imported from the v4.1.1 module block above —
# these are appended to the same file, so the originals are already in scope.
# --------------------------------------------------------------------------- #
def _a3_numeric(df: pd.DataFrame, exclude: tuple[str | None, ...] = ()) -> list[str]:
    drop = {c for c in exclude if c}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in drop]


def _a3_categorical(df: pd.DataFrame, exclude: tuple[str | None, ...] = ()) -> list[str]:
    drop = {c for c in exclude if c}
    numeric = set(df.select_dtypes(include=[np.number]).columns)
    return [c for c in df.columns if c not in numeric and c not in drop]


def _a3_outputs(ctx: TestContext) -> tuple[str | None, ...]:
    return (ctx.target_column, ctx.score_column, ctx.prediction_column)


def _a3_skip(test_id: str, name: str, reason: str, **params: Any) -> TestResult:
    return TestResult(
        test_id=test_id, test_name=name, status=Status.SKIPPED,
        params=params, interpretation=reason,
    )


# --------------------------------------------------------------------------- #
# A3.1 target reconstruction
# --------------------------------------------------------------------------- #
def _affine_fit(x: np.ndarray, y: np.ndarray, tol: float) -> tuple[bool, float, float]:
    """Is y == a*x + b exactly, within tolerance, across the observed support?"""
    if x.size < 3 or float(np.std(x)) == 0.0:
        return False, 0.0, 0.0
    design = np.column_stack([x, np.ones_like(x)])
    try:
        (slope, intercept), *_ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return False, 0.0, 0.0
    residual = np.max(np.abs(y - (slope * x + intercept)))
    scale = max(1.0, float(np.max(np.abs(y))))
    return bool(residual / scale <= tol), float(slope), float(intercept)


def _invertible_monotone(x: np.ndarray, y: np.ndarray, tol: float) -> bool:
    """A deterministic one-to-one mapping over the observed support.

    Deliberately NOT high Spearman correlation. Rank correlation of 0.99 means the
    ordering mostly agrees; it says nothing about whether the mapping is a function.
    What is required here is stronger and checkable: every distinct x maps to exactly
    one y (no conflicting target for the same feature value, within tolerance), every
    distinct y comes from exactly one x, and the mapping is strictly monotone when the
    pairs are sorted by x.
    """
    if x.size < 4:
        return False
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if frame.empty:
        return False

    grouped = frame.groupby("x")["y"]
    # Conflicting targets for the same feature value: not a function, so not a
    # reconstruction however well the ranks line up.
    if float(grouped.apply(lambda s: s.max() - s.min()).max()) > tol:
        return False

    mapping = grouped.mean().sort_index()
    if mapping.size < 4:
        return False
    if mapping.nunique() != mapping.size:      # not injective
        return False

    diffs = np.diff(mapping.to_numpy(dtype=float))
    return bool(np.all(diffs > tol) or np.all(diffs < -tol))


@register_test(
    "preprocessing.leakage_target_reconstruction",
    family="preprocessing",
    name="Leakage: target reconstruction",
    requires=("train", "target_column"),
    default_params={"tol": 1e-9},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("use_boundary", "data_quality_lineage"),
    object_kinds=_OBJECTS,
)
def leakage_target_reconstruction(ctx: TestContext, tol: float = 1e-9) -> TestResult:
    """Detects features that deterministically reproduce the target.

    Three reconstruction kinds, all FAIL: exact equality, affine recovery
    (``y = a*x + b``), and an invertible monotone mapping over the observed support.

    This is a defect, not a finding. A feature that reproduces the target means the
    outcome was joined back onto the feature table, and no amount of model quality
    argument survives it.
    """
    df: pd.DataFrame = ctx.train
    target = ctx.target_column
    if not target or target not in df.columns:
        return _a3_skip(
            "preprocessing.leakage_target_reconstruction", "Leakage: target reconstruction",
            "No target column configured.", tol=tol,
        )

    y_raw = pd.to_numeric(df[target], errors="coerce")
    if y_raw.isna().all():
        return _a3_skip(
            "preprocessing.leakage_target_reconstruction", "Leakage: target reconstruction",
            "Target is not numeric; reconstruction detection requires a numeric target.",
            tol=tol,
        )

    flagged: dict[str, str] = {}
    detail: dict[str, str] = {}
    for column in _a3_numeric(df, exclude=_a3_outputs(ctx)):
        pair = pd.DataFrame({"x": pd.to_numeric(df[column], errors="coerce"), "y": y_raw}).dropna()
        pair = pair[np.isfinite(pair["x"]) & np.isfinite(pair["y"])]
        if len(pair) < 4:
            continue
        x = pair["x"].to_numpy(dtype=float)
        y = pair["y"].to_numpy(dtype=float)

        scale = max(1.0, float(np.max(np.abs(y))))
        if float(np.max(np.abs(x - y))) / scale <= tol:
            flagged[column] = "exact"
            detail[column] = "feature equals the target"
            continue
        is_affine, slope, intercept = _affine_fit(x, y, tol)
        if is_affine and abs(slope) > tol:
            flagged[column] = "affine"
            detail[column] = f"y = {slope:.6g}*x + {intercept:.6g}"
            continue
        if _invertible_monotone(x, y, tol):
            flagged[column] = "invertible_monotone"
            detail[column] = "one-to-one strictly monotone mapping over the observed support"

    metrics: dict[str, Any] = {
        "n_flagged": len(flagged),
        "n_exact": sum(1 for v in flagged.values() if v == "exact"),
        "n_affine": sum(1 for v in flagged.values() if v == "affine"),
        "n_monotone": sum(1 for v in flagged.values() if v == "invertible_monotone"),
        "flagged_features": ", ".join(sorted(flagged)),
        "tolerance": tol,
    }
    for column, kind in sorted(flagged.items()):
        metrics[f"reconstruction.{column}"] = kind
        metrics[f"relationship.{column}"] = detail[column]

    strongest = ""
    for kind in ("exact", "affine", "invertible_monotone"):
        matches = sorted(c for c, k in flagged.items() if k == kind)
        if matches:
            strongest = f"{matches[0]} ({kind})"
            break
    metrics["strongest_relationship"] = strongest

    return TestResult(
        test_id="preprocessing.leakage_target_reconstruction",
        test_name="Leakage: target reconstruction",
        status=Status.FAIL if flagged else Status.PASS,
        params={"tol": tol},
        metrics=metrics,
        interpretation=(
            f"{len(flagged)} feature(s) deterministically reconstruct the target: "
            f"{', '.join(f'{c} ({k})' for c, k in sorted(flagged.items()))}."
            if flagged
            else "No feature deterministically reconstructs the target."
        ),
        limitations=[
            "Reconstruction is established over the OBSERVED SUPPORT of the training "
            "sample. It is evidence that the relationship holds for the rows examined, "
            "not proof of a universal mathematical identity.",
            "Monotone reconstruction requires an observed one-to-one mapping with no "
            "conflicting target value for the same feature value — high rank "
            "correlation alone is NOT treated as reconstruction.",
            "Non-numeric features are not examined; a categorical key that encodes the "
            "target would not be detected here.",
            "Numerical, not bitwise reproducible across BLAS implementations.",
        ],
    )


# --------------------------------------------------------------------------- #
# A3.2 high correlation — WARN only
# --------------------------------------------------------------------------- #
@register_test(
    "preprocessing.leakage_high_correlation",
    family="preprocessing",
    name="Leakage: high target correlation",
    requires=("train", "target_column"),
    default_params={"warn_corr": 0.95},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("use_boundary",),
    object_kinds=_OBJECTS,
)
def leakage_high_correlation(ctx: TestContext, warn_corr: float = 0.95) -> TestResult:
    """Features correlating strongly with the target. WARN only, never FAIL.

    A behaviour score against default, or days-past-due against default, legitimately
    correlates above 0.95. FAILing those would teach reviewers to ignore this check,
    which is worse than not having it. Reconstruction is the FAIL condition and lives in
    its own test.
    """
    df: pd.DataFrame = ctx.train
    target = ctx.target_column
    if not target or target not in df.columns:
        return _a3_skip(
            "preprocessing.leakage_high_correlation", "Leakage: high target correlation",
            "No target column configured.", warn_corr=warn_corr,
        )

    y = pd.to_numeric(df[target], errors="coerce")
    if y.isna().all() or y.nunique() <= 1:
        return _a3_skip(
            "preprocessing.leakage_high_correlation", "Leakage: high target correlation",
            "Target is non-numeric or constant; correlation is undefined.",
            warn_corr=warn_corr,
        )

    correlations: dict[str, float] = {}
    for column in _a3_numeric(df, exclude=_a3_outputs(ctx)):
        series = pd.to_numeric(df[column], errors="coerce")
        if series.nunique() <= 1:
            continue
        value = float(series.corr(y))
        if math.isfinite(value):
            correlations[column] = value

    if not correlations:
        return _a3_skip(
            "preprocessing.leakage_high_correlation", "Leakage: high target correlation",
            "No non-constant numeric features available.", warn_corr=warn_corr,
        )

    flagged = {c: v for c, v in correlations.items() if abs(v) >= warn_corr}
    worst = max(correlations, key=lambda c: abs(correlations[c]))

    metrics: dict[str, Any] = {
        "max_abs_corr": round(abs(correlations[worst]), 6),
        "max_corr_feature": worst,
        "n_flagged": len(flagged),
        "n_features_examined": len(correlations),
        "flagged_features": ", ".join(sorted(flagged)),
    }
    for column, value in sorted(flagged.items()):
        metrics[f"corr.{column}"] = round(value, 6)

    return TestResult(
        test_id="preprocessing.leakage_high_correlation",
        test_name="Leakage: high target correlation",
        status=Status.WARN if flagged else Status.PASS,
        params={"warn_corr": warn_corr},
        metrics=metrics,
        interpretation=(
            f"{len(flagged)} feature(s) correlate at or above {warn_corr:g} with the "
            f"target; strongest is '{worst}' at {correlations[worst]:.4f}."
            if flagged
            else f"No feature correlates at or above {warn_corr:g}; strongest is "
                 f"'{worst}' at {correlations[worst]:.4f}."
        ),
        limitations=[
            "WARN only by design. High correlation is not leakage — legitimately strong "
            "predictors exist, and a check that FAILs them gets switched off.",
            "Correlation is not causation and not reconstruction; see "
            "preprocessing.leakage_target_reconstruction for the FAIL condition.",
            "Pearson correlation measures linear association only.",
            "Numerical, not bitwise reproducible across BLAS implementations.",
        ],
    )


# --------------------------------------------------------------------------- #
# A3.3 temporal leakage
# --------------------------------------------------------------------------- #
@register_test(
    "preprocessing.leakage_temporal",
    family="preprocessing",
    name="Leakage: temporal ordering",
    requires=("train", "test", "timestamp_column"),
    default_params={},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("use_boundary", "data_quality_lineage"),
    object_kinds=_OBJECTS,
)
def leakage_temporal(ctx: TestContext) -> TestResult:
    """Training observations dated at or after the evaluation period begins.

    Requires an explicitly configured ``timestamp_column``. A timestamp column is never
    inferred: guessing which column is time and then declaring a chronology violation
    would be a finding about the guess, not about the data.
    """
    column = ctx.timestamp_column
    if not column:
        return _a3_skip(
            "preprocessing.leakage_temporal", "Leakage: temporal ordering",
            "No timestamp_column configured. A timestamp column is not inferred — "
            "guessing which column is time would make any finding a statement about "
            "the guess rather than the data.",
        )
    if ctx.test is None:
        return _a3_skip(
            "preprocessing.leakage_temporal", "Leakage: temporal ordering",
            "No test cohort available; chronology cannot be compared.",
        )
    for name, frame in (("train", ctx.train), ("test", ctx.test)):
        if column not in frame.columns:
            return _a3_skip(
                "preprocessing.leakage_temporal", "Leakage: temporal ordering",
                f"timestamp_column {column!r} is not present in the {name} cohort.",
            )

    train_ts = pd.to_datetime(ctx.train[column], errors="coerce").dropna()
    test_ts = pd.to_datetime(ctx.test[column], errors="coerce").dropna()
    if train_ts.empty or test_ts.empty:
        return _a3_skip(
            "preprocessing.leakage_temporal", "Leakage: temporal ordering",
            f"timestamp_column {column!r} contains no parseable timestamps in one or "
            "both cohorts.",
        )

    train_max, test_min = train_ts.max(), test_ts.min()
    n_after = int((train_ts >= test_min).sum())
    overlap_days = float((train_max - test_min).total_seconds() / 86400.0) if train_max >= test_min else 0.0

    metrics: dict[str, Any] = {
        "n_train_after_test_start": n_after,
        "train_rows_after_test_start_pct": round(100.0 * n_after / max(len(train_ts), 1), 4),
        "overlap_days": round(overlap_days, 6),
        "train_max_timestamp": str(train_max),
        "test_min_timestamp": str(test_min),
        "train_min_timestamp": str(train_ts.min()),
        "test_max_timestamp": str(test_ts.max()),
        "n_train_timestamps": int(train_ts.size),
        "n_test_timestamps": int(test_ts.size),
    }

    return TestResult(
        test_id="preprocessing.leakage_temporal",
        test_name="Leakage: temporal ordering",
        status=Status.FAIL if n_after > 0 else Status.PASS,
        params={"timestamp_column": column},
        metrics=metrics,
        interpretation=(
            f"{n_after:,} training observation(s) are dated at or after the evaluation "
            f"period begins ({test_min}); the cohorts overlap by {overlap_days:.1f} days."
            if n_after
            else f"Training data ends at {train_max} and evaluation begins at "
                 f"{test_min}; chronology is respected."
        ),
        limitations=[
            "Detects cohort-level chronology violations only. Leakage from a feature "
            "computed using information from after its own observation timestamp is a "
            "different defect and is not detected here.",
            "Unparseable timestamps are dropped and counted, not inferred.",
        ],
    )


# --------------------------------------------------------------------------- #
# A3.4 entity overlap
# --------------------------------------------------------------------------- #
@register_test(
    "preprocessing.leakage_entity_overlap",
    family="preprocessing",
    name="Leakage: entity overlap",
    requires=("train", "test", "entity_id_column"),
    default_params={},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("use_boundary",),
    object_kinds=_OBJECTS,
)
def leakage_entity_overlap(ctx: TestContext) -> TestResult:
    """Entities appearing in both cohorts.

    Whether this is a defect depends on the declared validation design. Entity-disjoint
    evaluation is the right choice when the model will score unseen entities; repeated
    observations of the same entity across periods are entirely normal in longitudinal
    problems, and calling that leakage universally would be wrong.

    The test is invoked when entity-disjoint evaluation is expected, and the result is
    reported as a violation of *that* expectation rather than as a universal law.
    """
    column = ctx.entity_id_column
    if not column:
        return _a3_skip(
            "preprocessing.leakage_entity_overlap", "Leakage: entity overlap",
            "No entity_id_column configured.",
        )
    if ctx.test is None:
        return _a3_skip(
            "preprocessing.leakage_entity_overlap", "Leakage: entity overlap",
            "No test cohort available.",
        )
    for name, frame in (("train", ctx.train), ("test", ctx.test)):
        if column not in frame.columns:
            return _a3_skip(
                "preprocessing.leakage_entity_overlap", "Leakage: entity overlap",
                f"entity_id_column {column!r} is not present in the {name} cohort.",
            )

    train_ids = set(ctx.train[column].dropna().unique())
    test_ids = set(ctx.test[column].dropna().unique())
    shared = train_ids & test_ids
    overlap_rate = len(shared) / len(test_ids) if test_ids else 0.0
    n_test_rows = int(ctx.test[column].isin(shared).sum())

    metrics: dict[str, Any] = {
        "n_shared_entities": len(shared),
        "overlap_rate": round(overlap_rate, 6),
        "n_train_entities": len(train_ids),
        "n_test_entities": len(test_ids),
        "n_test_rows_affected": n_test_rows,
        "test_rows_affected_pct": round(100.0 * n_test_rows / max(len(ctx.test), 1), 4),
    }

    return TestResult(
        test_id="preprocessing.leakage_entity_overlap",
        test_name="Leakage: entity overlap",
        status=Status.FAIL if shared else Status.PASS,
        params={"entity_id_column": column},
        metrics=metrics,
        interpretation=(
            f"{len(shared):,} entity/entities appear in both cohorts, affecting "
            f"{overlap_rate:.1%} of evaluation entities."
            if shared
            else "No entity appears in both cohorts; the split is entity-disjoint."
        ),
        limitations=[
            "Whether entity overlap is a defect depends on the declared validation "
            "design. It is a violation when entity-disjoint evaluation is intended; "
            "repeated observations of the same entity are normal in longitudinal "
            "problems and are not leakage there.",
            "Entity aliasing — the same real entity under two different identifiers — "
            "is not detected.",
        ],
    )


# --------------------------------------------------------------------------- #
# A3.5 row overlap
# --------------------------------------------------------------------------- #
@register_test(
    "preprocessing.leakage_row_overlap",
    family="preprocessing",
    name="Leakage: row overlap",
    requires=("train", "test"),
    default_params={"hash_columns": None},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("use_boundary", "data_quality_lineage"),
    object_kinds=_OBJECTS,
)
def leakage_row_overlap(
    ctx: TestContext, hash_columns: tuple[str, ...] | None = None
) -> TestResult:
    """Exact duplicate records crossing the train/test boundary.

    Hashing is over the *string rendering* of the values, not Python object identity.
    ``hash()`` on a tuple of objects varies per process under PYTHONHASHSEED and would
    make this test non-reproducible in a way that only shows up intermittently.
    """
    if ctx.test is None:
        return _a3_skip(
            "preprocessing.leakage_row_overlap", "Leakage: row overlap",
            "No test cohort available.", hash_columns=hash_columns,
        )

    shared_columns = [c for c in ctx.train.columns if c in ctx.test.columns]
    if hash_columns:
        missing = [c for c in hash_columns if c not in shared_columns]
        if missing:
            return _a3_skip(
                "preprocessing.leakage_row_overlap", "Leakage: row overlap",
                f"hash_columns not present in both cohorts: {', '.join(missing)}.",
                hash_columns=list(hash_columns),
            )
        columns = list(hash_columns)
    else:
        # Compare on features, excluding outputs: an identical feature row with a
        # different label is still a duplicate observation.
        columns = [c for c in shared_columns if c not in {x for x in _a3_outputs(ctx) if x}]

    if not columns:
        return _a3_skip(
            "preprocessing.leakage_row_overlap", "Leakage: row overlap",
            "No comparable columns shared between cohorts.",
            hash_columns=list(hash_columns) if hash_columns else None,
        )

    def keys(frame: pd.DataFrame) -> pd.Series:
        # Deterministic across processes: string rendering, never hash().
        return frame[columns].astype(str).agg("\x1f".join, axis=1)

    train_keys = keys(ctx.train)
    test_keys = keys(ctx.test)
    shared = set(train_keys) & set(test_keys)
    n_test_rows = int(test_keys.isin(shared).sum())

    metrics: dict[str, Any] = {
        "n_duplicate_across_split": len(shared),
        "n_test_rows_duplicated": n_test_rows,
        "overlap_rate": round(n_test_rows / max(len(ctx.test), 1), 6),
        "n_columns_compared": len(columns),
        "columns_compared": ", ".join(columns[:20]),
    }

    return TestResult(
        test_id="preprocessing.leakage_row_overlap",
        test_name="Leakage: row overlap",
        status=Status.FAIL if shared else Status.PASS,
        params={"hash_columns": list(hash_columns) if hash_columns else None},
        metrics=metrics,
        interpretation=(
            f"{len(shared):,} distinct feature row(s) appear in both cohorts, covering "
            f"{n_test_rows:,} evaluation row(s)."
            if shared
            else "No exact feature row appears in both cohorts."
        ),
        limitations=[
            "Exact matches only. Near-duplicates differing in a single float are not "
            "detected.",
            "Comparison is over the string rendering of values, which is deterministic "
            "across processes — Python object hashing is not, and would make this "
            "check intermittently non-reproducible.",
            "Float formatting affects the comparison; two values equal to within "
            "floating-point noise but rendered differently are treated as distinct.",
        ],
    )


# --------------------------------------------------------------------------- #
# A3.6 suspicious predictivity — WARN only, target dispatched
# --------------------------------------------------------------------------- #
@register_test(
    "preprocessing.leakage_suspicious_predictivity",
    family="preprocessing",
    name="Leakage: suspicious single-feature predictivity",
    requires=("train", "target_column"),
    default_params={"auc_threshold": 0.95, "r2_threshold": 0.95},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("use_boundary",),
    object_kinds=_OBJECTS,
)
def leakage_suspicious_predictivity(
    ctx: TestContext, auc_threshold: float = 0.95, r2_threshold: float = 0.95
) -> TestResult:
    """A single feature separating the target implausibly well. WARN only.

    The statistic depends on the target type and the one actually used is recorded —
    comparing an AUC against an R² without knowing which is which would be meaningless.

        binary      single-feature AUC
        multiclass  one-vs-rest macro AUC
        continuous  univariate R² (squared Pearson correlation), NOT an AUC
    """
    inference, skip = require_target_type(ctx, "binary", "multiclass", "continuous")
    if skip is not None:
        skip.test_id = "preprocessing.leakage_suspicious_predictivity"
        skip.test_name = "Leakage: suspicious single-feature predictivity"
        return skip

    df: pd.DataFrame = ctx.train
    y = df[ctx.target_column]
    kind = inference.target_type.value
    threshold = r2_threshold if kind == "continuous" else auc_threshold
    metric_name = {
        "binary": "single_feature_auc",
        "multiclass": "one_vs_rest_macro_auc",
        "continuous": "univariate_r_squared",
    }[kind]

    scores: dict[str, float] = {}
    for column in _a3_numeric(df, exclude=_a3_outputs(ctx)):
        x = pd.to_numeric(df[column], errors="coerce")
        pair = pd.DataFrame({"x": x, "y": y}).dropna()
        if len(pair) < 10 or pair["x"].nunique() <= 1:
            continue
        try:
            if kind == "continuous":
                r = float(pair["x"].corr(pd.to_numeric(pair["y"], errors="coerce")))
                value = r * r if math.isfinite(r) else 0.0
            else:
                from sklearn.metrics import roc_auc_score

                if kind == "binary":
                    if pair["y"].nunique() != 2:
                        continue
                    auc = float(roc_auc_score(pair["y"], pair["x"]))
                    # Direction-free: a feature that perfectly anti-separates is just
                    # as suspicious as one that perfectly separates.
                    value = max(auc, 1.0 - auc)
                else:
                    if pair["y"].nunique() < 3:
                        continue
                    value = float(
                        roc_auc_score(
                            pair["y"], pair[["x"]].to_numpy().repeat(pair["y"].nunique(), axis=1),
                            multi_class="ovr", average="macro",
                        )
                    )
        except Exception:
            continue
        if math.isfinite(value):
            scores[column] = value

    if not scores:
        return _a3_skip(
            "preprocessing.leakage_suspicious_predictivity",
            "Leakage: suspicious single-feature predictivity",
            "No numeric feature could be scored against the target.",
            **inference.as_params(),
        )

    worst = max(scores, key=lambda c: scores[c])
    flagged = {c: v for c, v in scores.items() if v >= threshold}

    metrics: dict[str, Any] = {
        "max_single_feature_score": round(scores[worst], 6),
        "max_score_feature": worst,
        "metric_used": metric_name,
        "threshold_used": threshold,
        "n_flagged": len(flagged),
        "n_features_scored": len(scores),
        "flagged_features": ", ".join(sorted(flagged)),
        **inference.as_params(),
    }
    for column, value in sorted(flagged.items()):
        metrics[f"score.{column}"] = round(value, 6)

    return TestResult(
        test_id="preprocessing.leakage_suspicious_predictivity",
        test_name="Leakage: suspicious single-feature predictivity",
        status=Status.WARN if flagged else Status.PASS,
        params={
            "auc_threshold": auc_threshold,
            "r2_threshold": r2_threshold,
            **inference.as_params(),
        },
        metrics=metrics,
        interpretation=(
            f"{len(flagged)} feature(s) reach {metric_name} at or above "
            f"{threshold:g}; highest is '{worst}' at {scores[worst]:.4f}."
            if flagged
            else f"No single feature reaches {metric_name} {threshold:g}; highest is "
                 f"'{worst}' at {scores[worst]:.4f}."
        ),
        limitations=[
            "WARN only. High single-feature predictivity is NOT proof of leakage — "
            "legitimately strong predictors exist, and in some domains a single "
            "feature dominating is expected rather than suspicious.",
            f"The statistic used is {metric_name}, chosen by target type; scores are "
            "not comparable across target types.",
            "Distributed leakage spread across several features is not detected.",
            "Numerical, not bitwise reproducible across BLAS implementations.",
        ],
    )


# --------------------------------------------------------------------------- #
# A3.7 name heuristic — WARN only
# --------------------------------------------------------------------------- #
@register_test(
    "preprocessing.leakage_name_heuristic",
    family="preprocessing",
    name="Leakage: feature naming observation",
    requires=("train",),
    default_params={"patterns": LEAKAGE_NAME_PATTERNS},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("documentation_completeness", "use_boundary"),
    object_kinds=_OBJECTS,
)
def leakage_name_heuristic(
    ctx: TestContext, patterns: tuple[str, ...] = LEAKAGE_NAME_PATTERNS
) -> TestResult:
    """Feature names containing outcome-like terms. An observation about names only.

    Matching is on token boundaries after splitting on underscores, hyphens, spaces and
    camel-case transitions. Raw substring matching flags ``postcode`` for ``post`` and
    ``labelled_region`` for ``label`` — noise that gets a WARN-only check ignored.
    """
    df: pd.DataFrame = ctx.train
    columns = [c for c in df.columns if c not in {x for x in _a3_outputs(ctx) if x}]

    def tokens(name: str) -> set[str]:
        spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
        return {t for t in re.split(r"[^A-Za-z0-9]+", spaced.lower()) if t}

    flagged: dict[str, str] = {}
    for column in columns:
        matched = sorted(tokens(column) & {p.lower() for p in patterns})
        if matched:
            flagged[column] = ", ".join(matched)

    metrics: dict[str, Any] = {
        "n_flagged": len(flagged),
        "n_columns_examined": len(columns),
        "flagged_features": ", ".join(sorted(flagged)),
        "patterns": ", ".join(patterns),
    }
    for column, matched in sorted(flagged.items()):
        metrics[f"name_match.{column}"] = matched

    return TestResult(
        test_id="preprocessing.leakage_name_heuristic",
        test_name="Leakage: feature naming observation",
        status=Status.WARN if flagged else Status.PASS,
        params={"patterns": list(patterns)},
        metrics=metrics,
        interpretation=(
            f"{len(flagged)} feature name(s) contain outcome-like terms: "
            f"{', '.join(f'{c} ({m})' for c, m in sorted(flagged.items()))}. This is an "
            "observation about naming, not a finding about the data."
            if flagged
            else "No feature name contains an outcome-like term."
        ),
        limitations=[
            "This is a NAMING OBSERVATION, not evidence of leakage. A feature called "
            "'target_market_segment' is not leakage; a leaking feature called 'x7' "
            "would not be flagged.",
            "Matching is on token boundaries, so 'postcode' does not match 'post'. "
            "Concatenated names without separators may still be missed.",
            "Exact, not numerical — this test performs no floating-point computation.",
        ],
    )


# --------------------------------------------------------------------------- #
# A3.8 target analysis — dispatched
# --------------------------------------------------------------------------- #
@register_test(
    "preprocessing.target_analysis",
    family="preprocessing",
    name="Target analysis",
    requires=("train", "target_column"),
    default_params={"min_positive": 30, "rare_class_pct": 1.0},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("conceptual_soundness", "data_quality_lineage"),
    object_kinds=_OBJECTS,
)
def target_analysis(
    ctx: TestContext, min_positive: int = 30, rare_class_pct: float = 1.0
) -> TestResult:
    """Profile of the target, dispatched by its type.

    A continuous target gets moments and percentiles, not a "positive rate". Forcing
    the binary vocabulary onto a regression target is how a review ends up reporting a
    positive class share for a loss amount.
    """
    inference, skip = require_target_type(ctx, "binary", "multiclass", "continuous")
    if skip is not None:
        skip.test_id = "preprocessing.target_analysis"
        skip.test_name = "Target analysis"
        return skip

    df: pd.DataFrame = ctx.train
    raw = df[ctx.target_column]
    series = raw.dropna()
    kind = inference.target_type.value

    metrics: dict[str, Any] = {
        "target_kind": kind,
        "n_observations": int(series.size),
        "n_missing_target": int(raw.isna().sum()),
        "missing_target_pct": round(100.0 * raw.isna().sum() / max(len(raw), 1), 4),
        **inference.as_params(),
    }
    status = Status.RECORDED
    limitations = [
        "Descriptive profile of the training sample only.",
        "Counts are exact; moments and entropy are numerical.",
    ]

    if kind in {"binary", "multiclass"}:
        counts = series.value_counts()
        shares = counts / max(int(series.size), 1)
        entropy = float(-(shares * np.log(shares)).sum()) if len(shares) > 1 else 0.0
        metrics["n_classes"] = int(counts.size)
        metrics["entropy"] = round(entropy, 6)
        metrics["normalised_entropy"] = (
            round(entropy / math.log(counts.size), 6) if counts.size > 1 else 0.0
        )
        metrics["n_rare_classes"] = int((shares * 100 < rare_class_pct).sum())
        for level, count in counts.items():
            metrics[f"class.{level}.count"] = int(count)
            metrics[f"class.{level}.share"] = round(float(count / series.size), 6)

        if kind == "binary":
            positive = counts.min()
            metrics["n_positive"] = int(positive)
            metrics["positive_rate"] = round(float(positive / series.size), 6)
            metrics["minority_class"] = str(shares.idxmin())
            if int(positive) < min_positive:
                status = Status.WARN
                limitations.append(
                    f"Only {positive} minority-class observation(s); below {min_positive} "
                    "most supervised statistics are unstable."
                )
            interpretation = (
                f"Binary target with {positive:,} minority-class observation(s) of "
                f"{series.size:,} ({positive / series.size:.2%})."
            )
        else:
            interpretation = (
                f"Multiclass target with {counts.size} class(es) over {series.size:,} "
                f"observations; normalised entropy {metrics['normalised_entropy']:.3f}."
            )
            limitations.append(
                "'Positive rate' is not reported for a multiclass target — the binary "
                "notion does not apply."
            )
    else:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return _a3_skip(
                "preprocessing.target_analysis", "Target analysis",
                "Continuous target contains no finite values.",
                **inference.as_params(),
            )
        metrics["mean"] = round(float(np.mean(values)), 6)
        metrics["std"] = round(float(np.std(values, ddof=1)), 6) if values.size > 1 else 0.0
        metrics["min"] = round(float(np.min(values)), 6)
        metrics["max"] = round(float(np.max(values)), 6)
        for pct in (1, 5, 25, 50, 75, 95, 99):
            metrics[f"p{pct}"] = round(float(np.percentile(values, pct)), 6)
        if values.size > 2 and float(np.std(values)) > 0:
            metrics["skew"] = round(float(stats.skew(values, bias=False)), 6)
            metrics["excess_kurtosis"] = round(
                float(stats.kurtosis(values, fisher=True, bias=False)), 6
            )
        else:
            metrics["skew"] = 0.0
            metrics["excess_kurtosis"] = 0.0
        interpretation = (
            f"Continuous target over {values.size:,} finite observation(s); "
            f"mean {metrics['mean']:.4g}, sd {metrics['std']:.4g}."
        )
        limitations.append(
            "'Positive rate' and class counts are not reported for a continuous "
            "target — the binary notion does not apply."
        )

    if inference.is_ambiguous:
        limitations.append("Target type was inferred, not stated: " + inference.detail)

    return TestResult(
        test_id="preprocessing.target_analysis",
        test_name="Target analysis",
        status=status,
        params={
            "min_positive": min_positive,
            "rare_class_pct": rare_class_pct,
            **inference.as_params(),
        },
        metrics=metrics,
        interpretation=interpretation,
        limitations=limitations,
    )


# --------------------------------------------------------------------------- #
# A3.9 feature-target relationship — dispatched
# --------------------------------------------------------------------------- #
def _information_value(x: pd.Series, y: pd.Series, bins: int = 10) -> float:
    """IV over quantile bins with 0.5 smoothing. Binary target only."""
    try:
        binned = pd.qcut(x, q=bins, duplicates="drop")
    except (ValueError, TypeError):
        return float("nan")
    frame = pd.DataFrame({"bin": binned, "y": y}).dropna()
    if frame.empty:
        return float("nan")
    positives = float(frame["y"].sum())
    negatives = float(len(frame) - positives)
    if positives <= 0 or negatives <= 0:
        return float("nan")
    total = 0.0
    for _, group in frame.groupby("bin", observed=True):
        good = (float(group["y"].sum()) + 0.5) / (positives + 0.5)
        bad = (float(len(group) - group["y"].sum()) + 0.5) / (negatives + 0.5)
        total += (good - bad) * math.log(good / bad)
    return float(total)


@register_test(
    "preprocessing.feature_target_relationship",
    family="preprocessing",
    name="Feature-target relationship",
    requires=("train", "target_column"),
    default_params={"method": "auto", "top_n": 20},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("conceptual_soundness",),
    object_kinds=_OBJECTS,
)
def feature_target_relationship(
    ctx: TestContext, method: str = "auto", top_n: int = 20
) -> TestResult:
    """Association between each feature and the target, using valid statistics only.

    ``method="auto"`` selects by target type. Nothing is computed that does not apply:
    no IV for a continuous target, no Pearson correlation against a multiclass label.
    Excluded features and the reason are recorded.
    """
    inference, skip = require_target_type(ctx, "binary", "multiclass", "continuous")
    if skip is not None:
        skip.test_id = "preprocessing.feature_target_relationship"
        skip.test_name = "Feature-target relationship"
        return skip
    if method not in {"auto", "mutual_info", "correlation", "iv"}:
        raise ValueError(f"method={method!r} is not supported.")

    df: pd.DataFrame = ctx.train
    kind = inference.target_type.value
    numeric = _a3_numeric(df, exclude=_a3_outputs(ctx))
    excluded = _a3_categorical(df, exclude=_a3_outputs(ctx))
    if not numeric:
        return _a3_skip(
            "preprocessing.feature_target_relationship", "Feature-target relationship",
            "No numeric feature columns available.", method=method, **inference.as_params(),
        )

    frame = df[numeric + [ctx.target_column]].dropna()
    if len(frame) < 10:
        return _a3_skip(
            "preprocessing.feature_target_relationship", "Feature-target relationship",
            f"Only {len(frame)} complete row(s); at least 10 required.",
            method=method, **inference.as_params(),
        )

    X = frame[numeric].to_numpy(dtype=float)
    y_series = frame[ctx.target_column]
    statistics_used: list[str] = []
    metrics: dict[str, Any] = {
        "n_features_scored": len(numeric),
        "n_features_excluded": len(excluded),
        "excluded_features": ", ".join(excluded[:20]),
        "target_kind": kind,
        **inference.as_params(),
    }

    try:
        if kind == "continuous":
            from sklearn.feature_selection import mutual_info_regression

            mi = mutual_info_regression(X, y_series.to_numpy(dtype=float), random_state=ctx.seed)
        else:
            from sklearn.feature_selection import mutual_info_classif

            mi = mutual_info_classif(X, y_series, random_state=ctx.seed)
        statistics_used.append("mutual_information")
        for column, value in zip(numeric, mi, strict=True):
            metrics[f"mi.{column}"] = round(float(value), 6)
        metrics["max_mutual_information"] = round(float(np.max(mi)), 6)
        metrics["top_feature_by_mi"] = numeric[int(np.argmax(mi))]
    except Exception as exc:
        metrics["mutual_information_error"] = f"{type(exc).__name__}"

    if kind == "continuous":
        y = y_series.to_numpy(dtype=float)
        statistics_used += ["pearson_r", "spearman_r"]
        for column in numeric:
            x = frame[column].to_numpy(dtype=float)
            if float(np.std(x)) == 0:
                continue
            metrics[f"pearson.{column}"] = round(float(np.corrcoef(x, y)[0, 1]), 6)
            metrics[f"spearman.{column}"] = round(float(stats.spearmanr(x, y).statistic), 6)
    elif kind == "binary":
        statistics_used += ["auc", "information_value"]
        try:
            from sklearn.metrics import roc_auc_score

            for column in numeric:
                x = frame[column]
                if x.nunique() <= 1 or y_series.nunique() != 2:
                    continue
                auc = float(roc_auc_score(y_series, x))
                metrics[f"auc.{column}"] = round(max(auc, 1.0 - auc), 6)
                iv = _information_value(x, pd.to_numeric(y_series, errors="coerce"))
                if math.isfinite(iv):
                    metrics[f"iv.{column}"] = round(iv, 6)
        except Exception as exc:
            metrics["auc_error"] = f"{type(exc).__name__}"
    else:
        statistics_used.append("mutual_information_only")

    metrics["statistics_used"] = ", ".join(statistics_used)

    return TestResult(
        test_id="preprocessing.feature_target_relationship",
        test_name="Feature-target relationship",
        status=Status.RECORDED,
        params={"method": method, "top_n": top_n, **inference.as_params()},
        metrics=metrics,
        interpretation=(
            f"Scored {len(numeric)} numeric feature(s) against a {kind} target using "
            f"{', '.join(statistics_used)}; {len(excluded)} non-numeric feature(s) "
            "excluded."
        ),
        limitations=[
            "Descriptive only; asserts nothing and applies no threshold.",
            "Statistics are selected by target type — IV and AUC are not computed for "
            "a continuous target, and correlation is not computed against a "
            "multiclass label. Scores are not comparable across target types.",
            "Univariate association only; a feature useful only in combination with "
            "another will score low here.",
            "Mutual information is estimated and depends on the seed; it is 'seeded', "
            "not exact.",
            "Non-numeric features are excluded and listed.",
        ],
    )


# --------------------------------------------------------------------------- #
# A3.10 redundancy
# --------------------------------------------------------------------------- #
@register_test(
    "preprocessing.redundancy",
    family="preprocessing",
    name="Feature redundancy",
    requires=("train",),
    default_params={"corr_threshold": 0.95},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("conceptual_soundness",),
    object_kinds=_OBJECTS,
)
def redundancy(ctx: TestContext, corr_threshold: float = 0.95) -> TestResult:
    """Near-duplicate numeric features. Diagnostic only — nothing is dropped.

    ``n_features_removable`` is the size of a greedy cover, not an instruction. Which
    member of a redundant pair to keep is a modelling decision that depends on
    interpretability, data lineage and cost of collection, none of which this test can
    see.
    """
    df: pd.DataFrame = ctx.train
    numeric = _a3_numeric(df, exclude=_a3_outputs(ctx))
    usable = [c for c in numeric if pd.to_numeric(df[c], errors="coerce").nunique() > 1]
    if len(usable) < 2:
        return _a3_skip(
            "preprocessing.redundancy", "Feature redundancy",
            f"At least two non-constant numeric features are required; {len(usable)} available.",
            corr_threshold=corr_threshold,
        )

    matrix = df[usable].corr()
    pairs: list[tuple[str, str, float]] = []
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            value = float(matrix.iat[i, j])
            if math.isfinite(value) and abs(value) >= corr_threshold:
                pairs.append((usable[i], usable[j], value))
    pairs.sort(key=lambda p: (-abs(p[2]), p[0], p[1]))

    implicated = sorted({c for pair in pairs for c in pair[:2]})
    # Greedy cover: repeatedly drop the feature appearing in most remaining pairs.
    remaining = list(pairs)
    removable: list[str] = []
    while remaining:
        counts: dict[str, int] = {}
        for left, right, _ in remaining:
            counts[left] = counts.get(left, 0) + 1
            counts[right] = counts.get(right, 0) + 1
        victim = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        removable.append(victim)
        remaining = [p for p in remaining if victim not in p[:2]]

    metrics: dict[str, Any] = {
        "n_redundant_pairs": len(pairs),
        "n_features_implicated": len(implicated),
        "n_features_removable": len(removable),
        "n_features_examined": len(usable),
        "implicated_features": ", ".join(implicated[:30]),
        "removable_features": ", ".join(sorted(removable)[:30]),
    }
    for left, right, value in pairs[:20]:
        metrics[f"pair.{left}~{right}"] = round(value, 6)

    return TestResult(
        test_id="preprocessing.redundancy",
        test_name="Feature redundancy",
        status=Status.WARN if pairs else Status.PASS,
        params={"corr_threshold": corr_threshold},
        metrics=metrics,
        interpretation=(
            f"{len(pairs)} feature pair(s) correlate at or above {corr_threshold:g}, "
            f"implicating {len(implicated)} feature(s)."
            if pairs
            else f"No feature pair correlates at or above {corr_threshold:g}."
        ),
        limitations=[
            "DIAGNOSTIC ONLY. No feature is dropped, and n_features_removable is the "
            "size of a greedy cover rather than a recommendation — which member of a "
            "redundant pair to keep depends on interpretability, lineage and cost of "
            "collection, none of which this test can see.",
            "Pairwise linear correlation only; three features that are jointly "
            "collinear but pairwise weak are not detected here (see "
            "eda.multicollinearity).",
            "Numerical, not bitwise reproducible across BLAS implementations.",
        ],
    )


# --------------------------------------------------------------------------- #
# A3.11 dimensionality diagnostic — no transformation
# --------------------------------------------------------------------------- #
@register_test(
    "preprocessing.dimensionality_diagnostic",
    family="preprocessing",
    name="Dimensionality diagnostic",
    requires=("train",),
    default_params={"variance_target": 0.95, "ratio_warn": 0.80},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("conceptual_soundness",),
    object_kinds=_OBJECTS,
)
def dimensionality_diagnostic(
    ctx: TestContext, variance_target: float = 0.95, ratio_warn: float = 0.80
) -> TestResult:
    """How much of the feature space is genuinely distinct. Diagnostic only.

    **This produces no transformed data.** It reports how many components would be
    needed; ``feature_engineering.pca_transform`` is what actually produces them. The
    separation is deliberate: a test that reports a number and a step that changes the
    data are different things and must not be the same registered surface.
    """
    df: pd.DataFrame = ctx.train
    numeric = _a3_numeric(df, exclude=_a3_outputs(ctx))
    frame = df[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    usable = [c for c in numeric if len(frame) and frame[c].nunique() > 1]

    if len(usable) < 2:
        return _a3_skip(
            "preprocessing.dimensionality_diagnostic", "Dimensionality diagnostic",
            f"At least two non-constant numeric features are required; {len(usable)} available.",
            variance_target=variance_target,
        )
    if len(frame) < 3:
        return _a3_skip(
            "preprocessing.dimensionality_diagnostic", "Dimensionality diagnostic",
            f"Only {len(frame)} complete row(s); at least 3 required.",
            variance_target=variance_target,
        )

    matrix = frame[usable].to_numpy(dtype=float)
    n_samples, n_features = matrix.shape
    max_components = min(n_samples, n_features)

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        # Standardised: PCA on raw features is dominated by whichever has the largest
        # units, which is a statement about measurement scale, not dimensionality.
        scaled = StandardScaler().fit_transform(matrix)
        pca = PCA(n_components=max_components, random_state=ctx.seed)
        pca.fit(scaled)
        ratios = np.asarray(pca.explained_variance_ratio_, dtype=float)
    except Exception as exc:
        return TestResult(
            test_id="preprocessing.dimensionality_diagnostic",
            test_name="Dimensionality diagnostic",
            status=Status.ERROR,
            params={"variance_target": variance_target},
            metrics={"n_features": len(usable), "n_complete_rows": int(n_samples)},
            interpretation=f"PCA failed: {type(exc).__name__}: {exc}",
            limitations=["Decomposition did not converge on this data."],
        )

    cumulative = np.cumsum(ratios)
    n_needed = int(np.searchsorted(cumulative, variance_target) + 1)
    n_needed = min(n_needed, len(ratios))
    ratio = n_needed / len(usable)

    positive = ratios[ratios > 1e-12]
    # Participation-ratio effective rank: how many components genuinely carry variance.
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive)))) if positive.size else 0.0

    metrics: dict[str, Any] = {
        "n_features": len(usable),
        "n_complete_rows": int(n_samples),
        "n_components_for_target": n_needed,
        "components_to_features_ratio": round(ratio, 6),
        "variance_target": variance_target,
        "effective_rank": round(effective_rank, 6),
        "n_nonzero_components": int(positive.size),
        "p_greater_than_n": bool(n_features > n_samples),
    }
    for index, value in enumerate(ratios[:10], start=1):
        metrics[f"explained_variance_ratio_{index}"] = round(float(value), 6)

    limitations = [
        "DIAGNOSTIC ONLY. No transformed data is produced here; "
        "feature_engineering.pca_transform performs the transformation.",
        "PCA is fitted on standardised features — on raw features the result would "
        "describe measurement scale rather than dimensionality.",
        "Linear structure only; features related non-linearly appear independent.",
        "Numerical, not bitwise reproducible across BLAS implementations.",
    ]
    if n_features > n_samples:
        limitations.append(
            f"p > n ({n_features} features, {n_samples} complete rows): at most "
            f"{max_components} components exist and the covariance estimate is singular."
        )

    result = TestResult(
        test_id="preprocessing.dimensionality_diagnostic",
        test_name="Dimensionality diagnostic",
        params={"variance_target": variance_target, "ratio_warn": ratio_warn},
        metrics=metrics,
        thresholds=[
            ThresholdSpec(metric="components_to_features_ratio", warn=ratio_warn, fail=None)
        ],
        interpretation=(
            f"{n_needed} of {len(usable)} component(s) capture {variance_target:.0%} of "
            f"variance (ratio {ratio:.2f}); effective rank {effective_rank:.2f}."
        ),
        limitations=limitations,
    )
    return result.apply_thresholds()


# --------------------------------------------------------------------------- #
# A3.12 categorical drift
# --------------------------------------------------------------------------- #
@register_test(
    "preprocessing.categorical_drift",
    family="preprocessing",
    name="Categorical drift",
    requires=("train", "test"),
    default_params={"alpha": 0.05, "min_expected": 5.0},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("stability",),
    object_kinds=_OBJECTS,
)
def categorical_drift(
    ctx: TestContext, alpha: float = 0.05, min_expected: float = 5.0
) -> TestResult:
    """Level-distribution shift between cohorts, plus new and vanished levels.

    New levels in evaluation are reported separately from the chi-square result and
    are often the more actionable finding: a level the model never saw during training
    has no learned representation at all, regardless of whether the overall
    distribution shifted.
    """
    if ctx.test is None:
        return _a3_skip(
            "preprocessing.categorical_drift", "Categorical drift",
            "No test cohort available.", alpha=alpha,
        )

    train, test = ctx.train, ctx.test
    columns = [
        c for c in _a3_categorical(train, exclude=_a3_outputs(ctx)) if c in test.columns
    ]
    if not columns:
        return _a3_skip(
            "preprocessing.categorical_drift", "Categorical drift",
            "No categorical feature columns shared between cohorts.", alpha=alpha,
        )

    metrics: dict[str, Any] = {"n_columns_examined": len(columns)}
    min_p = 1.0
    n_rejected = 0
    n_tested = 0
    total_new = 0
    total_missing = 0
    sparse: list[str] = []
    worst = ""

    for column in columns:
        train_counts = train[column].dropna().value_counts()
        test_counts = test[column].dropna().value_counts()
        new_levels = sorted(set(test_counts.index) - set(train_counts.index), key=str)
        missing_levels = sorted(set(train_counts.index) - set(test_counts.index), key=str)
        total_new += len(new_levels)
        total_missing += len(missing_levels)

        metrics[f"{column}.n_new_levels"] = len(new_levels)
        metrics[f"{column}.n_missing_levels"] = len(missing_levels)
        if new_levels:
            metrics[f"{column}.new_levels"] = ", ".join(str(x) for x in new_levels[:10])

        levels = sorted(set(train_counts.index) | set(test_counts.index), key=str)
        if len(levels) < 2 or train_counts.sum() == 0 or test_counts.sum() == 0:
            metrics[f"{column}.test_used"] = "none"
            continue

        observed = np.array([
            [float(train_counts.get(level, 0)) for level in levels],
            [float(test_counts.get(level, 0)) for level in levels],
        ])
        # A chi-square with tiny expected counts is not valid. Reporting a p-value
        # anyway would be a number with no inferential meaning attached to it.
        row_totals = observed.sum(axis=1, keepdims=True)
        col_totals = observed.sum(axis=0, keepdims=True)
        expected = row_totals @ col_totals / observed.sum()
        if float(expected.min()) < min_expected:
            sparse.append(column)
            metrics[f"{column}.test_used"] = "none_sparse"
            metrics[f"{column}.min_expected_count"] = round(float(expected.min()), 4)
            continue

        try:
            chi2, p_value, dof, _ = stats.chi2_contingency(observed)
        except Exception:
            metrics[f"{column}.test_used"] = "none"
            continue

        n_tested += 1
        metrics[f"{column}.test_used"] = "chi_square"
        metrics[f"{column}.chi2"] = round(float(chi2), 6)
        metrics[f"{column}.p_value"] = round(float(p_value), 6)
        metrics[f"{column}.dof"] = int(dof)
        if float(p_value) < min_p:
            min_p = float(p_value)
            worst = column
        if float(p_value) < alpha:
            n_rejected += 1

    metrics["min_p_value"] = round(min_p, 6)
    metrics["n_columns_tested"] = n_tested
    metrics["n_columns_rejected"] = n_rejected
    metrics["n_columns_sparse"] = len(sparse)
    metrics["n_new_levels_total"] = total_new
    metrics["n_missing_levels_total"] = total_missing
    metrics["worst_feature"] = worst

    status = Status.PASS
    if n_rejected or total_new:
        status = Status.WARN

    return TestResult(
        test_id="preprocessing.categorical_drift",
        test_name="Categorical drift",
        status=status,
        params={"alpha": alpha, "min_expected": min_expected},
        metrics=metrics,
        interpretation=(
            f"Of {len(columns)} categorical feature(s), {n_tested} were testable; the "
            f"null hypothesis of an unchanged level distribution was rejected at the "
            f"{alpha:g} level for {n_rejected}"
            + (f" (smallest p-value {min_p:.4g} in '{worst}')" if worst else "")
            + f". {total_new} level(s) appear in evaluation but not in training."
        ),
        limitations=[
            "A chi-square test that does not reject has NOT established that the two "
            "level distributions are equal; it has failed to find evidence against "
            "equality at the stated level.",
            f"Columns with any expected cell count below {min_expected:g} are not "
            "tested — a chi-square statistic there has no valid inferential meaning. "
            f"{len(sparse)} column(s) were skipped for this reason.",
            "No multiplicity correction across columns; with many features some "
            "rejections are expected by chance.",
            "New levels are reported separately and are often more actionable than the "
            "test result: a level never seen in training has no learned representation.",
            "Numerical, not bitwise reproducible across BLAS implementations.",
        ],
    )
