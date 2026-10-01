"""Canonical current-review presentation root and atomic artifact manifest."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunLineage:
    review_run_id: str
    execution_run_ids: tuple[str, ...] = ()
    report_run_id: str | None = None

    def all_owner_ids(self) -> tuple[str, ...]:
        values = (self.review_run_id, *self.execution_run_ids)
        if self.report_run_id:
            values += (self.report_run_id,)
        return tuple(dict.fromkeys(values))


def current_run_presentation_root(output_root: str | Path, review_run_id: str) -> Path:
    """Create the sole presentation root for an exact review run."""
    if not review_run_id or Path(review_run_id).name != review_run_id:
        raise ValueError(f"invalid review run ID: {review_run_id!r}")
    root = Path(output_root).expanduser().resolve() / review_run_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_artifact_entry(
    *,
    path: str | Path,
    owner_run_id: str,
    lineage: RunLineage,
    artifact_id: str,
    artifact_type: str,
    title: str,
    checkpoint: str,
    test_id: str = "",
    evidence_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Validate and describe a real artifact owned by one exact run."""
    artifact_path = Path(path).expanduser().resolve()
    if owner_run_id not in lineage.all_owner_ids():
        raise ValueError(f"artifact owner {owner_run_id!r} is absent from run lineage")
    if not artifact_path.is_file():
        raise FileNotFoundError(f"current-run artifact does not exist: {artifact_path}")
    if owner_run_id not in artifact_path.parts:
        raise ValueError(
            f"artifact {artifact_path} is not scoped beneath its declared owner run {owner_run_id}"
        )
    companion = artifact_path.with_suffix(".json")
    return {
        "artifact_id": artifact_id,
        "artifact_type": artifact_type,
        "title": title,
        "test_id": test_id,
        "evidence_ids": list(evidence_ids),
        "owner_run_id": owner_run_id,
        "parent_review_run_id": lineage.review_run_id,
        "file_path": str(artifact_path),
        "semantic_companion": str(companion) if companion.is_file() else None,
        "checkpoint": checkpoint,
    }


def write_current_run_manifest(
    *,
    presentation_root: Path,
    lineage: RunLineage,
    groups: dict[str, list[dict[str, Any]]],
) -> Path:
    """Atomically write a validated exact-run manifest; never select ``latest``."""
    root = presentation_root.expanduser().resolve()
    expected_root = root.parent / lineage.review_run_id
    if root != expected_root:
        raise ValueError(
            f"presentation root {root} does not match review run {lineage.review_run_id}"
        )
    if not root.is_dir():
        raise FileNotFoundError(f"presentation root does not exist: {root}")

    artifact_roots: set[str] = set()
    for entries in groups.values():
        for entry in entries:
            path = Path(str(entry.get("file_path", ""))).resolve()
            owner = str(entry.get("owner_run_id", ""))
            if owner not in lineage.all_owner_ids() or not path.is_file() or owner not in path.parts:
                raise ValueError(f"inconsistent current-run artifact entry: {entry!r}")
            artifact_roots.add(str(path.parent))

    payload = {
        "schema": "start.current-run-artifacts/2",
        "review_run_id": lineage.review_run_id,
        "execution_run_ids": list(lineage.execution_run_ids),
        "report_run_id": lineage.report_run_id,
        "presentation_root": str(root),
        "scientific_artifact_roots": sorted(artifact_roots),
        "groups": groups,
    }
    manifest = root / "artifact_manifest.json"
    fd, temp_name = tempfile.mkstemp(prefix=".artifact_manifest.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, manifest)
    except BaseException:
        try:
            Path(temp_name).unlink(missing_ok=True)
        finally:
            raise
    return manifest
