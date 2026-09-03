#!/usr/bin/env python3
"""Regenerate requirements.txt from pyproject.toml.

requirements.txt exists because tooling and reviewers expect to find one. It is
NOT a source of truth — having two files that both claim to define dependencies
is how the previous "the full install is missing the OpenAI SDK" problem
happened, and the fix is not to keep them in sync by hand.

So this file is generated, marked as generated, and checked by
``tests/test_packaging_contract.py``. Editing it directly will fail CI.

    python scripts/sync_requirements.py          # rewrite
    python scripts/sync_requirements.py --check  # verify, exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path

HEADER = """\
# ─────────────────────────────────────────────────────────────────────────────
# GENERATED FILE — DO NOT EDIT.
#
# Source of truth: pyproject.toml
# Regenerate:      python scripts/sync_requirements.py
# Verified by:     tests/test_packaging_contract.py
#
# This mirror is for tooling that expects a requirements.txt. The supported way
# to install StART is:
#
#     python scripts/bootstrap.py
# ─────────────────────────────────────────────────────────────────────────────
"""


def render(pyproject: Path) -> str:
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    project = data["project"]
    extras = project.get("optional-dependencies", {})

    lines = [HEADER, f"# requires-python: {project['requires-python']}", "", "# --- core ---"]
    lines += sorted(project.get("dependencies", []))

    # "everything" and the compatibility aliases are unions of the others;
    # emitting them would duplicate every line.
    for name in sorted(extras):
        if name in {"everything", "all", "torch"}:
            continue
        lines += ["", f"# --- extra: {name} ---", *sorted(extras[name])]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify without writing")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    expected = render(root / "pyproject.toml")
    target = root / "requirements.txt"

    if args.check:
        actual = target.read_text(encoding="utf-8") if target.exists() else ""
        if actual != expected:
            print("requirements.txt is stale. Run: python scripts/sync_requirements.py")
            return 1
        print("requirements.txt is in sync with pyproject.toml")
        return 0

    target.write_text(expected, encoding="utf-8")
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
