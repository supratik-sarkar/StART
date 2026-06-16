"""Dataset source provenance for reviewer verification.

For the built-in demo dataset, exposes its real name, a public source URL, why
it was selected, and its task suitability — so a reviewer can independently
verify the data from a public web link. For custom datasets, exposes file path,
detected format, loading route, shape, and a content hash.

Cross-platform: uses pathlib and hashlib only; no OS-specific calls.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class DatasetSource:
    kind: str  # "builtin_demo" | "custom" | "synthetic_fallback"
    name: str
    n_rows: int
    n_columns: int
    target_column: str | None = None
    public_url: str | None = None
    reason_selected: str = ""
    task_suitability: str = ""
    file_path: str | None = None
    detected_format: str | None = None
    loading_route: str | None = None
    data_hash: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "target_column": self.target_column,
            "public_url": self.public_url,
            "reason_selected": self.reason_selected,
            "task_suitability": self.task_suitability,
            "file_path": self.file_path,
            "detected_format": self.detected_format,
            "loading_route": self.loading_route,
            "data_hash": self.data_hash,
        }


def frame_hash(df: pd.DataFrame) -> str:
    """Stable content hash of a dataframe (shape + column names + values)."""
    h = hashlib.sha256()
    h.update(str(df.shape).encode())
    h.update("|".join(map(str, df.columns)).encode())
    try:
        h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    except Exception:
        h.update(df.to_csv(index=False).encode())
    return h.hexdigest()[:16]


# Public provenance for the built-in demo dataset (scikit-learn breast cancer,
# used as a clean, verifiable binary-classification demonstration cohort).
_DEMO_PROVENANCE = {
    "name": "scikit-learn Breast Cancer Wisconsin (Diagnostic)",
    "public_url": "https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
    "reason_selected": (
        "A small, clean, fully public binary-classification dataset bundled with "
        "scikit-learn — ideal for a reproducible, verifiable model-review demo with "
        "no download or credentials required."
    ),
    "task_suitability": (
        "Binary classification (30 numeric features, 2 balanced classes); suitable "
        "for tabular DL, calibration, explainability, and sensitivity demonstrations."
    ),
}


def describe_demo_dataset(df: pd.DataFrame, target_column: str = "attrition") -> DatasetSource:
    is_real = df.shape[1] >= 30  # the real breast-cancer frame has 30 features + target
    if is_real:
        return DatasetSource(
            kind="builtin_demo",
            name=_DEMO_PROVENANCE["name"],
            n_rows=len(df),
            n_columns=df.shape[1],
            target_column=target_column,
            public_url=_DEMO_PROVENANCE["public_url"],
            reason_selected=_DEMO_PROVENANCE["reason_selected"],
            task_suitability=_DEMO_PROVENANCE["task_suitability"],
            loading_route="sklearn.datasets.load_breast_cancer",
            data_hash=frame_hash(df),
            notes=[
                "Target relabeled to 'attrition' for the demo narrative; underlying "
                "data is the public diagnostic dataset above.",
            ],
        )
    return DatasetSource(
        kind="synthetic_fallback",
        name="StART synthetic binary-classification cohort",
        n_rows=len(df),
        n_columns=df.shape[1],
        target_column=target_column,
        public_url=None,
        reason_selected="scikit-learn unavailable; generated a reproducible synthetic cohort.",
        task_suitability="Binary classification demo.",
        loading_route="start.modeling.data._synthetic_fallback",
        data_hash=frame_hash(df),
    )


def describe_custom_dataset(
    df: pd.DataFrame, file_path: str, target_column: str | None = None
) -> DatasetSource:
    p = Path(file_path)
    fmt = p.suffix.lower().lstrip(".") or "unknown"
    route = {
        "csv": "load_any_tabular -> read_csv",
        "tsv": "load_any_tabular -> read_csv(sep=tab)",
        "parquet": "load_any_tabular -> read_parquet",
        "feather": "load_any_tabular -> read_feather",
        "json": "load_any_tabular -> read_json",
        "jsonl": "load_any_tabular -> read_json(lines)",
        "xlsx": "load_any_tabular -> read_excel",
    }.get(fmt, "load_any_tabular")
    return DatasetSource(
        kind="custom",
        name=p.name,
        n_rows=len(df),
        n_columns=df.shape[1],
        target_column=target_column,
        file_path=str(p),
        detected_format=fmt,
        loading_route=route,
        data_hash=frame_hash(df),
    )


def render_dataset_source_markdown(src: DatasetSource) -> str:
    lines = ["### Dataset source", "", "| Field | Value |", "| --- | --- |",
             f"| Name | {src.name} |", f"| Kind | {src.kind} |",
             f"| Rows / Columns | {src.n_rows} / {src.n_columns} |",
             f"| Target | {src.target_column} |"]
    if src.public_url:
        lines.append(f"| Public source | {src.public_url} |")
    if src.file_path:
        lines.append(f"| File path | {src.file_path} |")
        lines.append(f"| Detected format | {src.detected_format} |")
    if src.loading_route:
        lines.append(f"| Loading route | {src.loading_route} |")
    if src.data_hash:
        lines.append(f"| Data hash | {src.data_hash} |")
    if src.reason_selected:
        lines += ["", f"**Why selected:** {src.reason_selected}"]
    if src.task_suitability:
        lines += ["", f"**Task suitability:** {src.task_suitability}"]
    return "\n".join(lines) + "\n"
