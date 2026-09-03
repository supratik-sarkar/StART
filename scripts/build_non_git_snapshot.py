#!/usr/bin/env python3
"""Build Non-Git Pre-Acceptance Source Snapshot Manifest for StART v4.5."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "start_output" / "v45_post_push_recovery"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest_entries = []
    scan_dirs = ["src", "web/src", "deploy", "scripts", "tests"]
    scan_files = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "requirements.txt",
        "web/package.json",
        "web/index.html",
        "web/vite.config.ts",
        "web/tsconfig.json",
        "web/tailwind.config.js",
    ]

    for d in scan_dirs:
        dp = ROOT / d
        if not dp.exists():
            continue
        for r, _, files in os.walk(dp):
            if any(p in ("__pycache__", ".venv-start", "node_modules", "dist") for p in Path(r).parts):
                continue
            for f in sorted(files):
                if f.startswith(".") or f.endswith(".pyc"):
                    continue
                fp = Path(r) / f
                rel = str(fp.relative_to(ROOT))
                manifest_entries.append(
                    {
                        "path": rel,
                        "sha256": sha256_file(fp),
                        "size_bytes": fp.stat().st_size,
                    }
                )

    for sf in scan_files:
        sfp = ROOT / sf
        if sfp.exists():
            rel = str(sfp.relative_to(ROOT))
            manifest_entries.append(
                {
                    "path": rel,
                    "sha256": sha256_file(sfp),
                    "size_bytes": sfp.stat().st_size,
                }
            )

    manifest_entries.sort(key=lambda x: x["path"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = OUTPUT_DIR / "non_git_pre_acceptance_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "snapshot_id": "v4.5.0-non-git-pre-acceptance",
                "timestamp": time.time(),
                "start_version": "4.5.0",
                "start_schema_version": "4.5.0",
                "frontend_build_version": "4.5.0-prod",
                "quarantined_git_commit": "e256c4e",
                "quarantined_git_tag": "v4.5.0",
                "file_count": len(manifest_entries),
                "files": manifest_entries,
            },
            f,
            indent=2,
        )

    print(f"Recorded non-git pre-acceptance snapshot with {len(manifest_entries)} files: {manifest_file}")


if __name__ == "__main__":
    main()
