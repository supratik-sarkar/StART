"""Comprehensive dataset statistics for the reviewer co-pilot.

Produces the rich "what does this data look like" picture the v2.1.0 UX needs,
on top of (not replacing) the frozen DatasetDiscoveryAgent. Everything here is
deterministic and emits a single ``DataStatistics`` object that renders to a
terminal table, a notebook table, the dashboard, and the markdown report.

Cross-platform: pure pandas/numpy, no OS-specific paths or calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ColumnStat:
    name: str
    dtype: str
    role: str  # numeric | categorical | datetime | text | image_path | boolean | identifier
    missing_pct: float
    n_unique: int
    is_high_cardinality: bool = False
    is_low_variance: bool = False
    n_outliers: int = 0


@dataclass
class DataStatistics:
    n_rows: int
    n_columns: int
    target_column: str | None
    target_type: str | None
    class_distribution: dict[str, float]
    n_numeric: int
    n_categorical: int
    n_datetime: int
    n_text: int
    n_image_path: int
    n_boolean: int
    n_identifier: int
    n_duplicate_rows: int
    high_cardinality_columns: list[str]
    low_variance_columns: list[str]
    leakage_candidates: list[str]
    imbalance_warning: str
    suggested_split: str
    column_stats: list[ColumnStat] = field(default_factory=list)
    missing_by_column: dict[str, float] = field(default_factory=dict)
    correlation_summary: dict[str, Any] = field(default_factory=dict)
    outlier_summary: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "target_column": self.target_column,
            "target_type": self.target_type,
            "class_distribution": self.class_distribution,
            "n_numeric": self.n_numeric,
            "n_categorical": self.n_categorical,
            "n_datetime": self.n_datetime,
            "n_text": self.n_text,
            "n_image_path": self.n_image_path,
            "n_boolean": self.n_boolean,
            "n_identifier": self.n_identifier,
            "n_duplicate_rows": self.n_duplicate_rows,
            "high_cardinality_columns": self.high_cardinality_columns,
            "low_variance_columns": self.low_variance_columns,
            "leakage_candidates": self.leakage_candidates,
            "imbalance_warning": self.imbalance_warning,
            "suggested_split": self.suggested_split,
            "missing_by_column": self.missing_by_column,
            "correlation_summary": self.correlation_summary,
            "outlier_summary": self.outlier_summary,
        }

    def summary_rows(self) -> list[tuple[str, Any]]:
        return [
            ("Rows", self.n_rows),
            ("Columns", self.n_columns),
            ("Target", self.target_column),
            ("Target type", self.target_type),
            ("Numeric / Categorical", f"{self.n_numeric} / {self.n_categorical}"),
            ("Datetime / Text / Image", f"{self.n_datetime} / {self.n_text} / {self.n_image_path}"),
            ("Duplicate rows", self.n_duplicate_rows),
            ("High-cardinality cols", len(self.high_cardinality_columns)),
            ("Low-variance cols", len(self.low_variance_columns)),
            ("Leakage candidates", len(self.leakage_candidates)),
            ("Imbalance", self.imbalance_warning),
            ("Suggested split", self.suggested_split),
        ]


def _role(series: pd.Series, name: str) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        nunique = series.nunique(dropna=True)
        if nunique > 0.9 * len(series) and ("id" in name.lower().split("_")):
            return "identifier"
        return "numeric"
    # object / string
    lname = name.lower()
    if any(t in lname.split("_") for t in ("path", "image", "img", "file", "filepath")):
        return "image_path"
    avg_len = series.dropna().astype(str).str.len().mean() if len(series.dropna()) else 0
    return "text" if avg_len and avg_len > 40 else "categorical"


def _outlier_count(series: pd.Series) -> int:
    """IQR-rule outlier count for a numeric series."""
    s = series.dropna()
    if len(s) < 4:
        return 0
    q1, q3 = np.percentile(s, [25, 75])
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lo) | (s > hi)).sum())


def compute_data_statistics(
    df: pd.DataFrame, target_column: str | None = None
) -> DataStatistics:
    n_rows, n_cols = len(df), df.shape[1]
    column_stats: list[ColumnStat] = []
    role_counts = {
        "numeric": 0, "categorical": 0, "datetime": 0, "text": 0,
        "image_path": 0, "boolean": 0, "identifier": 0,
    }
    missing_by_column: dict[str, float] = {}
    high_card: list[str] = []
    low_var: list[str] = []
    outlier_summary: dict[str, int] = {}

    for name in df.columns:
        series = df[name]
        role = _role(series, name)
        role_counts[role] = role_counts.get(role, 0) + 1
        missing = round(float(series.isna().mean() * 100), 4)
        missing_by_column[name] = missing
        nunique = int(series.nunique(dropna=True))

        is_high_card = role in ("categorical", "text") and nunique > 0.5 * n_rows and nunique > 20
        is_low_var = False
        n_out = 0
        if role == "numeric":
            n_out = _outlier_count(series)
            if n_out:
                outlier_summary[name] = n_out
            std = float(series.std()) if len(series.dropna()) else 0.0
            is_low_var = std == 0.0 or (nunique <= 1)
        elif role in ("categorical", "boolean"):
            is_low_var = nunique <= 1

        if is_high_card:
            high_card.append(name)
        if is_low_var:
            low_var.append(name)

        column_stats.append(
            ColumnStat(
                name=name, dtype=str(series.dtype), role=role, missing_pct=missing,
                n_unique=nunique, is_high_cardinality=is_high_card,
                is_low_variance=is_low_var, n_outliers=n_out,
            )
        )

    n_dupes = int(df.duplicated().sum())

    # target analysis
    target_type, class_dist, imbalance = None, {}, "n/a"
    if target_column and target_column in df.columns:
        t = df[target_column]
        if pd.api.types.is_numeric_dtype(t) and t.nunique() > 20:
            target_type = "continuous"
        else:
            nuniq = t.nunique()
            target_type = "binary" if nuniq == 2 else "multiclass" if nuniq <= 20 else "high_cardinality"
            vc = t.value_counts(normalize=True)
            class_dist = {str(k): round(float(v), 4) for k, v in vc.items()}
            if class_dist:
                minority = min(class_dist.values())
                if minority < 0.10:
                    imbalance = f"severe (minority class {minority:.1%})"
                elif minority < 0.30:
                    imbalance = f"moderate (minority class {minority:.1%})"
                else:
                    imbalance = "balanced"

    # leakage candidates: numeric features near-perfectly correlated with a numeric target
    leakage: list[str] = []
    corr_summary: dict[str, Any] = {}
    if target_column and target_column in df.columns and pd.api.types.is_numeric_dtype(df[target_column]):
        numeric_feats = [
            c for c in df.columns
            if c != target_column and pd.api.types.is_numeric_dtype(df[c])
        ]
        y = df[target_column]
        y_std = float(y.std()) if len(y.dropna()) else 0.0
        high_corr_pairs = []
        for c in numeric_feats:
            if y_std == 0 or float(df[c].std() or 0) == 0:
                continue  # correlation undefined for a constant column
            corr = df[c].corr(y)
            if corr == corr and abs(corr) > 0.98:
                leakage.append(c)
            if corr == corr and abs(corr) > 0.7:
                high_corr_pairs.append((c, round(float(corr), 3)))
        corr_summary = {
            "n_features_corr_gt_0.7": len(high_corr_pairs),
            "top_correlated": sorted(high_corr_pairs, key=lambda x: -abs(x[1]))[:5],
        }

    # suggested split
    has_datetime = role_counts["datetime"] > 0
    if has_datetime:
        suggested_split = "time_based"
    elif target_type in ("binary", "multiclass"):
        suggested_split = "stratified"
    else:
        suggested_split = "random"

    notes = []
    if n_dupes:
        notes.append(f"{n_dupes} duplicate row(s) detected.")
    if leakage:
        notes.append(f"Potential leakage: {', '.join(leakage[:5])}.")

    return DataStatistics(
        n_rows=n_rows, n_columns=n_cols, target_column=target_column,
        target_type=target_type, class_distribution=class_dist,
        n_numeric=role_counts["numeric"], n_categorical=role_counts["categorical"],
        n_datetime=role_counts["datetime"], n_text=role_counts["text"],
        n_image_path=role_counts["image_path"], n_boolean=role_counts["boolean"],
        n_identifier=role_counts["identifier"], n_duplicate_rows=n_dupes,
        high_cardinality_columns=high_card, low_variance_columns=low_var,
        leakage_candidates=leakage, imbalance_warning=imbalance,
        suggested_split=suggested_split, column_stats=column_stats,
        missing_by_column=missing_by_column, correlation_summary=corr_summary,
        outlier_summary=outlier_summary, notes=notes,
    )


def render_statistics_markdown(stats: DataStatistics) -> str:
    lines = ["### Initial data statistics", "", "| Metric | Value |", "| --- | --- |"]
    for label, value in stats.summary_rows():
        lines.append(f"| {label} | {value} |")
    if stats.class_distribution:
        lines += ["", "**Class distribution:** " + ", ".join(
            f"{k}={v:.1%}" for k, v in stats.class_distribution.items()
        )]
    cols_with_missing = {k: v for k, v in stats.missing_by_column.items() if v > 0}
    if cols_with_missing:
        lines += ["", "**Columns with missing values:** " + ", ".join(
            f"{k} ({v:.1f}%)" for k, v in sorted(cols_with_missing.items(), key=lambda x: -x[1])[:10]
        )]
    return "\n".join(lines) + "\n"
