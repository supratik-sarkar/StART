#!/usr/bin/env python3
"""Build Cryptographic Release Freeze Manifest for StART v4.6.3."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "start_output" / "v463_remote_release"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INCLUDED_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".css", ".html", ".js", ".mjs", ".json", ".toml", ".yaml", ".yml", ".sh"
}

EXCLUDED_PATHS = {
    ".venv-start", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
    "web/node_modules", "web/dist", "start_output", ".agents", ".gemini"
}


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_manifest() -> dict[str, str]:
    manifest_entries: dict[str, str] = {}

    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        rel_str = str(rel)

        if any(rel_str.startswith(exc) or f"/{exc}/" in f"/{rel_str}/" for exc in EXCLUDED_PATHS):
            continue

        if p.suffix in INCLUDED_EXTENSIONS or p.name in {"Dockerfile", "Makefile"}:
            # Avoid hashing gitignored environment or private key files
            if rel_str.endswith(".env") or "id_ed25519" in rel_str:
                continue
            manifest_entries[rel_str] = sha256_file(p)

    return manifest_entries


def main() -> None:
    print("Building StART v4.6.3 Cryptographic Release Manifest...")
    manifest_entries = build_manifest()

    manifest_file = OUTPUT_DIR / "v463_release_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "release": "StART v4.6.3",
                "timestamp": time.time(),
                "file_count": len(manifest_entries),
                "files": manifest_entries,
            },
            f,
            indent=2,
        )

    print(f"Generated v4.6.3 release manifest with {len(manifest_entries)} files at: {manifest_file}")


if __name__ == "__main__":
    main()
