"""Tiny process fixture for demo-controller exit semantics tests."""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("success", "cancelled", "crashed", "wrong_prompt"))
parser.add_argument("manifest", nargs="?")
args = parser.parse_args()

if args.mode == "wrong_prompt":
    print("unexpected output", flush=True)
    raise SystemExit(0)

print("Executing Canonical Workflow 'quantitative_finance'", flush=True)
if args.mode == "success":
    print("Review run: RUN-REVIEW-FIXTURE", flush=True)
    print("Execution run: RUN-ENT-FIXTURE", flush=True)
    if args.manifest:
        print(f"Artifact manifest: {args.manifest}", flush=True)
    print("Policy OPA_LOCAL: ALLOW", flush=True)
    print("Merkle: fixture-root", flush=True)
    print("3 EvidenceRecords committed", flush=True)
    raise SystemExit(0)
if args.mode == "cancelled":
    print("ReviewCancelled: reviewer cancelled; fail closed", flush=True)
    raise SystemExit(7)

print("Traceback (most recent call last):", flush=True)
print("RuntimeError: fixture crash", flush=True)
raise SystemExit(9)
