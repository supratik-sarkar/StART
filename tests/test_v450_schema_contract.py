"""Test Schema Contract and Python/TypeScript Synchronization for StART v4.5."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_schema_version_consistency() -> None:
    from start.web.schemas import START_SCHEMA_VERSION, START_VERSION, SystemInfo

    assert START_SCHEMA_VERSION.startswith("4.")
    assert START_VERSION.startswith("4.")
    info = SystemInfo()
    assert info.start_version.startswith("4.")
    assert info.start_schema_version.startswith("4.")


def test_typescript_definitions_in_sync() -> None:
    res = subprocess.run(
        [sys.executable, "scripts/export_web_schemas.py", "--check"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"TypeScript schema definitions drift detected: {res.stdout} {res.stderr}"
