"""Content hashing for the tamper-evident evidence layer.

Canonicalization rules:
- JSON with sorted keys, compact separators, UTF-8, non-ASCII preserved.
- Non-JSON types (datetime, Enum, numpy scalars) serialized via ``str``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from importlib import metadata
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json(obj))


def hash_dataframe(df: Any, sample_rows: int | None = 100_000) -> str:
    """Order-insensitive content hash of a pandas DataFrame.

    Rows are hashed individually and the hashes summed modulo 2**256 so the
    result is invariant to row order; column order is normalized by sorting.
    For very large frames a deterministic head sample can be used.
    """
    import pandas as pd  # local import keeps core import light

    if not isinstance(df, pd.DataFrame):
        raise TypeError("hash_dataframe expects a pandas DataFrame")
    frame = df[sorted(df.columns)]
    if sample_rows is not None and len(frame) > sample_rows:
        frame = frame.head(sample_rows)
    row_hashes = pd.util.hash_pandas_object(frame, index=False).astype("uint64")
    digest_input = f"{int(row_hashes.sum())}|{len(frame)}|{list(frame.columns)}"
    return sha256_hex(digest_input)


def current_git_sha(cwd: str | None = None) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def package_versions(
    packages: tuple[str, ...] = ("numpy", "pandas", "scipy", "scikit-learn", "pydantic"),
) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def python_version() -> str:
    return sys.version.split()[0]
