#!/usr/bin/env python3
"""Manifest-Driven Synchronizer from Development Tree to Protected Git Tree for v5.1.3."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
DEST_ROOT = SRC_ROOT.parent / "My_Git" / "StART"
MANIFEST_PATH = SRC_ROOT / "start_output" / "v513_release" / "v513_candidate_manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("=" * 70)
    print("StART v5.1.3 — Manifest-Driven Protected Git Tree Synchronization")
    print(f"Source: {SRC_ROOT}")
    print(f"Destination: {DEST_ROOT}")
    print("=" * 70)

    if not DEST_ROOT.exists():
        raise RuntimeError(f"Destination Git repository missing at: {DEST_ROOT}")
    if not (DEST_ROOT / ".git").exists():
        raise RuntimeError(f"Destination is not a git repository: {DEST_ROOT}/.git missing")
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"Release manifest missing at: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    synced_files = []
    errors = []

    files_data = manifest.get("files", [])

    for item in files_data:
        rel = item["path"]
        if rel.startswith("start_output/"):
            continue

        src_fp = SRC_ROOT / rel
        dest_fp = DEST_ROOT / rel

        if not src_fp.exists():
            errors.append(f"Source file missing: {src_fp}")
            continue

        dest_fp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_fp, dest_fp)

        dest_sha = sha256_file(dest_fp)
        if dest_sha != item["sha256"]:
            errors.append(f"SHA mismatch for {rel}: expected {item['sha256']}, got {dest_sha}")
        else:
            synced_files.append(
                {
                    "path": rel,
                    "sha256": dest_sha,
                    "size_bytes": dest_fp.stat().st_size,
                }
            )

    if errors:
        print("SYNCHRONIZATION ERRORS ENCOUNTERED:")
        for err in errors:
            print(f"  - {err}")
        raise RuntimeError(f"{len(errors)} errors during synchronization.")

    print(f"Successfully synchronized {len(synced_files)} files with 100% SHA-256 match.")


if __name__ == "__main__":
    main()
