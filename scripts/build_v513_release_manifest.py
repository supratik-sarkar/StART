#!/usr/bin/env python3
"""Build Cryptographic Release Freeze Candidate Manifest for StART v5.1.3."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "start_output" / "v513_release"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_tree_digest(files: list[dict[str, Any]], prefix: str = "") -> str:
    """Compute deterministic root digest over filtered file entries."""
    matching = [f for f in files if f["path"].startswith(prefix)]
    matching.sort(key=lambda x: x["path"])
    h = hashlib.sha256()
    for m in matching:
        h.update(f"{m['path']}:{m['sha256']}".encode())
    return h.hexdigest()


def build_candidate_manifest() -> dict[str, Any]:
    manifest_entries = []

    # Tracked source directories
    scan_dirs = ["src", "webapp/src", "webapp/docs", "deploy", "scripts", "tests", "webapp/dist"]
    scan_files = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "requirements.txt",
        ".gitignore",
        "webapp/package.json",
        "webapp/package-lock.json",
        "webapp/index.html",
        "webapp/vite.config.ts",
        "webapp/tsconfig.json",
        "webapp/README.md",
    ]

    for d in scan_dirs:
        dp = ROOT / d
        if not dp.exists():
            continue
        for r, _, files in os.walk(dp):
            if any(p in ("__pycache__", ".venv-start", "node_modules", ".wrangler") for p in Path(r).parts):
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
            if not any(e["path"] == rel for e in manifest_entries):
                manifest_entries.append(
                    {
                        "path": rel,
                        "sha256": sha256_file(sfp),
                        "size_bytes": sfp.stat().st_size,
                    }
                )

    # Sort entries deterministically
    manifest_entries.sort(key=lambda x: x["path"])

    backend_digest = compute_tree_digest(manifest_entries, prefix="src/")
    frontend_dist_digest = compute_tree_digest(manifest_entries, prefix="webapp/dist/")
    overall_digest = compute_tree_digest(manifest_entries, prefix="")

    manifest_data = {
        "release": "StART v5.1.3 Candidate Freeze",
        "version": "5.1.3",
        "timestamp": time.time(),
        "file_count": len(manifest_entries),
        "backend_digest": backend_digest,
        "frontend_dist_digest": frontend_dist_digest,
        "candidate_root_digest": overall_digest,
        "files": manifest_entries,
    }

    manifest_file = OUTPUT_DIR / "v513_candidate_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Generated v5.1.3 candidate manifest with {len(manifest_entries)} files: {manifest_file}")
    print(f"Backend Digest: {backend_digest}")
    print(f"Frontend Dist Digest: {frontend_dist_digest}")
    print(f"Candidate Root Digest: {overall_digest}")
    return manifest_data


if __name__ == "__main__":
    build_candidate_manifest()
