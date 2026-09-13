"""Explicit data loaders for every supported input format.

Tabular: CSV, TSV, TXT (delimited), Parquet, Feather, Excel (.xlsx/.xls),
JSON, JSONL, Pickle (gated behind an explicit allow flag because unpickling
executes arbitrary code). Vision: image-folder discovery into a manifest
DataFrame with one row per image and a label inferred from the subdirectory.

Optional formats degrade with a clear install hint; nothing here imports a
heavy dependency at module load.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUPPORTED_TABULAR_FORMATS = (
    ".csv",
    ".tsv",
    ".txt",
    ".parquet",
    ".pq",
    ".feather",
    ".ft",
    ".xlsx",
    ".xls",
    ".json",
    ".jsonl",
    ".ndjson",
    ".pkl",
    ".pickle",
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff")


def sniff_format(path: str | Path) -> str:
    """Classify a path as 'tabular', 'image_folder', 'delta', or 'unknown'."""
    p = Path(path)
    if p.is_dir():
        # an image folder has image files (possibly nested one level for labels)
        if any(_iter_images(p)):
            return "image_folder"
        if (p / "_delta_log").exists():
            return "delta"
        return "unknown"
    if p.suffix.lower() in SUPPORTED_TABULAR_FORMATS:
        return "tabular"
    return "unknown"


def load_any_tabular(path: str | Path, *, allow_pickle: bool = False) -> pd.DataFrame:
    """Load any supported tabular format into a DataFrame."""
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix == ".tsv":
        return pd.read_csv(p, sep="\t")
    if suffix == ".txt":
        # delimited text: let pandas sniff the separator
        return pd.read_csv(p, sep=None, engine="python")
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(p)
    if suffix in {".feather", ".ft"}:
        return pd.read_feather(p)
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(p)
        except ImportError as exc:
            raise ImportError(
                "Reading Excel requires openpyxl: pip install openpyxl"
            ) from exc
    if suffix == ".json":
        return pd.read_json(p)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(p, lines=True)
    if suffix in {".pkl", ".pickle"}:
        if not allow_pickle:
            raise ValueError(
                f"Refusing to load pickle '{p.name}' by default: unpickling executes "
                "arbitrary code. Pass allow_pickle=True (or --allow-pickle) only for "
                "files you fully trust."
            )
        obj = pd.read_pickle(p)
        if not isinstance(obj, pd.DataFrame):
            raise ValueError(f"Pickle did not contain a DataFrame (got {type(obj).__name__}).")
        return obj
    raise ValueError(
        f"Unsupported tabular format '{suffix}'. Supported: "
        f"{', '.join(SUPPORTED_TABULAR_FORMATS)}."
    )


def _iter_images(root: Path):
    """Yield image files directly under root or one subdirectory deep."""
    for child in sorted(root.iterdir()):
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
            yield child
        elif child.is_dir():
            for f in sorted(child.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    yield f


def discover_image_folder(
    root: str | Path, label_column: str = "label", path_column: str = "image_path"
) -> pd.DataFrame:
    """Build an image manifest DataFrame from a folder.

    Layout convention (ImageFolder-style): ``root/<class_name>/<image>``. The
    immediate subdirectory name becomes the label. Flat folders (images
    directly under root) yield a single unlabeled manifest.
    """
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Image folder '{root}' is not a directory.")
    rows: list[dict[str, str]] = []
    for child in sorted(root.iterdir()):
        if child.is_dir():
            label = child.name
            for f in sorted(child.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({path_column: str(f), label_column: label})
        elif child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
            rows.append({path_column: str(child), label_column: ""})
    if not rows:
        raise ValueError(f"No images found under '{root}'.")
    return pd.DataFrame(rows)
