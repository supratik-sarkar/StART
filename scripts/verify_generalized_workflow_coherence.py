#!/usr/bin/env python3
"""Write the deterministic generalized-workflow coherence verification artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from start.review.coherence_verifier import VerificationPathsV4, write_verification_v4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="StART private workspace root (default: current directory)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Machine-readable verification JSON path",
    )
    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    result_json = (
        args.output.expanduser().resolve()
        if args.output is not None
        else workspace / "StART_GENERALIZED_WORKFLOW_COHERENCE_VERIFICATION_V4.json"
    )
    paths = VerificationPathsV4(
        workspace=workspace,
        result_json=result_json,
        closure_report=workspace / "StART_GENERALIZED_WORKFLOW_COHERENCE_FINAL_CLOSURE.md",
    )
    result = write_verification_v4(paths)
    print(
        json.dumps(
            {
                "schema": result["schema"],
                "status": result["status"],
                "result_json": str(paths.result_json),
                "closure_report": str(paths.closure_report),
                "hosted_calls": 0,
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
