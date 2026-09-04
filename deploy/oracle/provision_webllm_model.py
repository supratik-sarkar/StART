#!/usr/bin/env python3
"""Idempotent Pinned WebLLM Model Provisioner for Oracle Cloud Origin.

Downloads, verifies, and provisions the exact pinned model assets for in-browser
WebGPU/WebLLM inference without proxying through Cloudflare Worker.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path


def sha256_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def provision_model(
    manifest_path: Path,
    target_base_dir: Path = Path("/opt/start/static/webllm-models"),
) -> bool:
    """Provision pinned model assets idempotently."""
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}", file=sys.stderr)
        return False

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    model_id = manifest["model_id"]
    revision = manifest["upstream_revision"]
    upstream_repo = manifest["upstream_repository"]
    files_spec = manifest["files"]

    model_dir = target_base_dir / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checking pinned WebLLM model: {model_id} (revision: {revision[:8]}...)")

    # 1. Verify existing files
    all_valid = True
    missing_files = []

    for fname, meta in files_spec.items():
        fpath = model_dir / fname
        if not fpath.exists():
            all_valid = False
            missing_files.append(fname)
            continue

        if fpath.stat().st_size != meta["size"]:
            all_valid = False
            missing_files.append(fname)
            continue

        actual_hash = sha256_file(fpath)
        if actual_hash != meta["sha256"]:
            print(
                f"Checksum mismatch for {fname}: expected {meta['sha256']}, got {actual_hash}",
                file=sys.stderr,
            )
            all_valid = False
            missing_files.append(fname)

    if all_valid and len(missing_files) == 0:
        print(f"✅ All {len(files_spec)} model artifacts present and verified with SHA-256.")
        _setup_compatibility_symlinks(model_dir)
        return True

    print(
        f"Missing/corrupted artifacts ({len(missing_files)}/{len(files_spec)}). "
        f"Provisioning from {upstream_repo}..."
    )

    # 2. Check disk space
    stat = shutil.disk_usage(model_dir)
    if stat.free < 3 * 1024 * 1024 * 1024:  # Require at least 3 GB free
        print(
            f"ERROR: Insufficient disk space ({stat.free / 1024 / 1024:.1f} MB free, need 3 GB)",
            file=sys.stderr,
        )
        return False

    # 3. Download missing files to staging
    staging_dir = target_base_dir / f".staging_{model_id}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        for fname in missing_files:
            meta = files_spec[fname]
            url = f"{upstream_repo}/resolve/{revision}/{fname}"
            stage_path = staging_dir / fname

            print(f"Downloading {fname} ({meta['size'] / 1024 / 1024:.2f} MB)...")
            req = urllib.request.Request(url, headers={"User-Agent": "StART-Model-Provisioner"})
            with urllib.request.urlopen(req, timeout=300) as resp, open(stage_path, "wb") as out_f:
                while True:
                    buf = resp.read(1024 * 1024)
                    if not buf:
                        break
                    out_f.write(buf)

            # Verify downloaded file
            actual_hash = sha256_file(stage_path)
            if actual_hash != meta["sha256"]:
                raise ValueError(
                    f"Downloaded file {fname} failed SHA-256 check. "
                    f"Expected: {meta['sha256']}, got: {actual_hash}"
                )

            # Atomically move to final destination
            shutil.move(str(stage_path), str(model_dir / fname))
            print(f"  Verified {fname}")

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)

    # 4. Setup runtime compatibility symlinks
    _setup_compatibility_symlinks(model_dir)

    print(f"✅ WebLLM model {model_id} successfully provisioned and verified.")
    return True


def _setup_compatibility_symlinks(model_dir: Path) -> None:
    """Setup WebLLM version compatibility symlinks."""
    # WebLLM resolve/main subdirectory alias
    resolve_dir = model_dir / "resolve"
    resolve_dir.mkdir(exist_ok=True)
    resolve_main = resolve_dir / "main"
    if not resolve_main.exists() and not resolve_main.is_symlink():
        try:
            resolve_main.symlink_to(model_dir, target_is_directory=True)
        except Exception:
            pass

    # tensor-cache.json alias for ndarray-cache.json
    tensor_cache = model_dir / "tensor-cache.json"
    ndarray_cache = model_dir / "ndarray-cache.json"
    if ndarray_cache.exists() and not tensor_cache.exists() and not tensor_cache.is_symlink():
        try:
            tensor_cache.symlink_to(ndarray_cache.name)
        except Exception:
            pass


def main() -> None:
    root = Path(__file__).resolve().parent.parent.parent
    manifest = Path(__file__).resolve().parent / "webllm_model_manifest.json"
    if not manifest.exists():
        manifest = root / "deploy" / "oracle" / "webllm_model_manifest.json"

    ok = provision_model(manifest)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
