#!/usr/bin/env python3
"""Build Cryptographic Release Freeze Manifest for StART v4.6.2."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "start_output" / "v462_remote_release"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    manifest_entries = []

    # Directories to scan
    scan_dirs = ["src", "web/src", "deploy", "scripts", "tests", ".github"]
    scan_files = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "requirements.txt",
        "web/package.json",
        "web/package-lock.json",
        "web/index.html",
        "web/vite.config.ts",
        "web/tsconfig.json",
        "web/tailwind.config.js",
        "web/postcss.config.js",
        "web/.env.production",
        "web/.env.development",
    ]

    for d in scan_dirs:
        dp = ROOT / d
        if not dp.exists():
            continue
        for r, _, files in os.walk(dp):
            if any(p in ("__pycache__", ".venv-start", "node_modules", "dist", ".wrangler") for p in Path(r).parts):
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

    # Sort entries deterministically
    manifest_entries.sort(key=lambda x: x["path"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_file = OUTPUT_DIR / "v462_release_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "release": "StART v4.6.2",
                "timestamp": time.time(),
                "file_count": len(manifest_entries),
                "files": manifest_entries,
            },
            f,
            indent=2,
        )

    print(f"Generated v4.6.2 release manifest with {len(manifest_entries)} files at: {manifest_file}")


if __name__ == "__main__":
    main()
