#!/usr/bin/env python3
"""Manifest-Driven Synchronizer from Development Tree to Protected Git Tree.

Transfers only frozen, verified publication files to /Users/.../Desktop/My_Git/StART,
protecting .git and validating bit-for-bit SHA-256 hash matches.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parent.parent
DEST_ROOT = SRC_ROOT.parent / "My_Git" / "StART"
MANIFEST_PATH = SRC_ROOT / "start_output" / "v45_release_closure" / "v450_release_freeze_manifest.json"
REPORT_PATH = SRC_ROOT / "start_output" / "v45_release_closure" / "phase21_git_sync_report.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    print("=" * 70)
    print("StART v4.5 — Manifest-Driven Protected Git Tree Synchronization")
    print(f"Source: {SRC_ROOT}")
    print(f"Destination: {DEST_ROOT}")
    print("=" * 70)

    if not DEST_ROOT.exists():
        raise RuntimeError(f"Destination Git repository missing at: {DEST_ROOT}")
    if not (DEST_ROOT / ".git").exists():
        raise RuntimeError(f"Destination is not a git repository: {DEST_ROOT}/.git missing")
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"Release freeze manifest missing at: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    synced_files = []
    errors = []

    # 1. Sync every frozen publication file
    for item in manifest.get("files", []):
        rel = item["path"]
        # Skip local output artifacts from copying directly into git root if not intended
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

    # 2. Save Sync Report
    report = {
        "phase": "PHASE_21_MANIFEST_GIT_SYNC",
        "timestamp": os.path.getmtime(MANIFEST_PATH),
        "source_root": str(SRC_ROOT),
        "dest_root": str(DEST_ROOT),
        "synced_file_count": len(synced_files),
        "status": "ALL_SYNCHRONIZED_AND_VERIFIED",
        "synced_files": synced_files,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved Git sync report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
