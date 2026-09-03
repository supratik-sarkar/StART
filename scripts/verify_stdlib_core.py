#!/usr/bin/env python3
"""Standard-library-only verification of StART core risk and attestation packages."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src is at the head of sys.path to simulate zero-installed environment
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from start.risk import RiskObject, stripe_ids, synthesise_plan  # noqa: E402
from start.runtime_profile import profile_banner  # noqa: E402


def main() -> None:
    plan = synthesise_plan(stripe_id="credit", obj=RiskObject(object_id="X", kind="scorecard"))
    assert plan.plan_hash(), "Plan hash must not be empty"
    assert len(stripe_ids()) >= 10, f"Expected >= 10 stripes, got {len(stripe_ids())}"
    print("stdlib core OK —", profile_banner())


if __name__ == "__main__":
    main()
