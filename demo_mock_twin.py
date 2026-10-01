"""Deterministic offline twin harness for StART v6.0.2 demo controller verification.

Emits the complete interactive prompt sequence for Sessions A, B, and C
to prove controller state graph traversal offline at $0 with zero network calls.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def run_mock_session_a(artifact_manifest: str | None = None) -> None:
    """Simulate terminal interactions for Session A (Predictive / Temporal LSTM)."""
    # 1. Wizard Setup Sequence
    sys.stdout.write("\nSelect Review Mode:\n  [1] Single-Domain Review\n  [2] Cross-Domain Review\nSelect option [1-2] [default: 1]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans == "1", f"Unexpected ans for A_MODE: {ans}"

    sys.stdout.write("\nSelect Review Domain:\n  [1] Predictive Modeling (ML/DL)\n  [2] Market & Treasury\nSelect option [1-4] [default: 1]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans == "1", f"Unexpected ans for A_DOMAIN: {ans}"

    sys.stdout.write("\nSelect Predictive Modeling Technology:\n  [1] Tabular Machine Learning\n  [2] Deep Learning Suite\nSelect option [1-2] [default: 1]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans == "2", f"Unexpected ans for A_TECH: {ans}"

    sys.stdout.write("\nSelect AI Reviewer Agent Backend:\n  [1] Deterministic Fallback\n  [4] Offline Demo Twin\nSelect option [1-4] [default: 1]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans == "4", f"Unexpected ans for A_BACKEND: {ans}"

    sys.stdout.write("\nSelect Model Materiality:\n  [1] High Materiality (Tier 1)\n  [2] Medium Materiality\nSelect option [1-3] [default: 1]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans == "1", f"Unexpected ans for A_MATERIALITY: {ans}"

    sys.stdout.write("\nSelect Review Lifecycle:\n  [1] Pre-Implementation / Initial Validation\n  [2] Periodic Validation\nSelect option [1-3] [default: 1]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans == "1", f"Unexpected ans for A_LIFECYCLE: {ans}"

    # Governance Multiline inputs
    sys.stdout.write("\nEnter Business Context (paste text, type END on empty line to finish, or press Enter on empty line only: END to skip.)\n")
    sys.stdout.flush()
    lines = []
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "END":
            break
        lines.append(line)

    sys.stdout.write("\nEnter Reviewer Clarification (type END to skip.)\n")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "END":
            break

    sys.stdout.write("\nEnter Intended Use / Decision Impact (type END to skip.)\n")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "END":
            break

    sys.stdout.write("\nEnter Known Limitations / Reviewer Concerns (type END to skip.)\n")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "END":
            break

    # Technology & Architecture
    sys.stdout.write("\nSelect Neural Network Architecture:\n  [1] MLP\n  [2] Residual MLP\n  [3] LSTM (Long Short-Term Memory)\nSelect option [1-6] [default: 1]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans == "3", f"Unexpected ans for A_ARCH_SELECT: {ans}"

    sys.stdout.write("\nSelect Activation Function:\n  [1] ReLU\n  [2] Leaky ReLU\nSelect option [1-5] [default: 1]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans == "1", f"Unexpected ans for A_ACT_SELECT: {ans}"

    sys.stdout.write("\nSelect Predictive Dataset Source:\n  [1] Synthetic Credit Default\n  [5] Synthetic Temporal Sequence\nSelect option [1-5] [default: 5]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans == "5", f"Unexpected ans for A_DATA_SELECT: {ans}"

    sys.stdout.write("\nSelect Review Scope:\n  [1] Full Recommended Suite\n  [2] Custom Scope\nSelect option [1-2] [default: 1]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans == "1", f"Unexpected ans for A_SCOPE_SELECT: {ans}"

    sys.stdout.write("\nProceed to execute review? [Y/n]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "Y", f"Unexpected ans for A_PROCEED_WIZ: {ans}"

    # 2. Interactive Checkpoints
    sys.stdout.write("\n[architecture] Accept (A) / Override (O) / Challenge (C) / [Q] ask ArchitectureReviewAgent? ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "C", f"Unexpected ans for A_ARCH_CHECKPOINT_1: {ans}"

    sys.stdout.write("\nEnter challenge to ArchitectureReviewAgent: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nDoes this response change the disposition? [y/N]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "N", f"Unexpected ans for A_CONCEDE_1: {ans}"

    sys.stdout.write("\n[architecture] Accept (A) / Override (O) / Challenge (C) / [Q] ask ArchitectureReviewAgent? ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "C", f"Unexpected ans for A_ARCH_CHECKPOINT_2: {ans}"

    sys.stdout.write("\nEnter challenge to ArchitectureReviewAgent: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nDoes this response change the disposition? [y/N]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "N", f"Unexpected ans for A_CONCEDE_2: {ans}"

    sys.stdout.write("\n[architecture] Accept (A) / Override (O) / Challenge (C) / [Q] ask ArchitectureReviewAgent? ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "Q", f"Unexpected ans for A_ARCH_CHECKPOINT_3: {ans}"

    sys.stdout.write("\nAsk ArchitectureReviewAgent: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\n[architecture] Accept (A) / Override (O) / Challenge (C) / [Q] ask ArchitectureReviewAgent? ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "A", f"Unexpected ans for A_ARCH_DECISION: {ans}"

    # Final Execution & Evidence Separation
    sys.stdout.write("\nReview complete! Merklized seal: start-seal-4a8f9c\n")
    sys.stdout.write("Review run: RUN-ENT-TEMPORAL-FIXTURE\n")
    sys.stdout.write("Execution run: RUN-ENT-TEMPORAL-FIXTURE\n")
    if artifact_manifest:
        manifest = Path(artifact_manifest).expanduser().resolve()
        if not manifest.is_file():
            raise RuntimeError(f"fixture artifact manifest does not exist: {manifest}")
        sys.stdout.write(f"Artifact manifest: {manifest}\n")
    sys.stdout.write("evidence records: 8 records generated | sign-off: conditional | seal: verified | dashboard: dashboard.html\n")
    sys.stdout.flush()


def run_mock_session_b() -> None:
    """Simulate terminal interactions for Session B (Market / Portfolio / Risk / Governance)."""
    # 1. Wizard Setup
    sys.stdout.write("\nSelect Review Mode:\n  [1] Single-Domain Review\nSelect option [1-2] [default: 1]: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nSelect Review Domain:\n  [2] Market Risk & Portfolio\nSelect option [1-4] [default: 1]: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nSelect AI Reviewer Agent Backend:\n  [4] Offline Demo Twin\nSelect option [1-4] [default: 1]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans == "4", f"Unexpected ans for B_BACKEND: {ans}"

    sys.stdout.write("\nSelect Model Materiality:\n  [1] High Materiality\nSelect option [1-3] [default: 1]: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nSelect Review Lifecycle:\n  [1] Initial Validation\nSelect option [1-3] [default: 1]: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nEnter Business Context (paste text, type END on empty line to finish, or press Enter on empty line only: END to skip.)\n")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "END":
            break

    sys.stdout.write("\nEnter Reviewer Clarification (type END to skip.)\n")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "END":
            break

    sys.stdout.write("\nEnter Intended Use / Decision Impact (type END to skip.)\n")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "END":
            break

    sys.stdout.write("\nEnter Known Limitations / Reviewer Concerns (type END to skip.)\n")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline()
        if not line or line.strip() == "END":
            break

    sys.stdout.write("\nSelect Market & Treasury Data Source:\n  [1] Synthetic 50-asset world\nSelect option [1-3] [default: 1]: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nSelect Review Scope:\n  [1] Full Recommended\nSelect option [1-2] [default: 1]: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nProceed to execute review? [Y/n]: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    # 2. Checkpoints 1 through 7
    # Chk 1: Portfolio
    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "C"

    sys.stdout.write("\nEnter reviewer challenge note: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "Q"

    sys.stdout.write("\nAsk agent committee: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "O"

    sys.stdout.write("\nEnter reviewer override justification: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    # Chk 2: Attribution
    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "Q"

    sys.stdout.write("\nAsk agent committee: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "A"

    # Chk 3: VaR Backtesting
    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "Q"

    sys.stdout.write("\nAsk agent committee: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "A"

    # Chk 4: Covariance Structure
    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "Q"

    sys.stdout.write("\nAsk agent committee: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "A"

    # Chk 5: Scenario Analysis & Stress Testing
    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "C"

    sys.stdout.write("\nEnter reviewer challenge note: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "A"

    # Chk 6: Cross-Analytical Committee Synthesis
    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "Q"

    sys.stdout.write("\nAsk agent committee: ")
    sys.stdout.flush()
    _ = sys.stdin.readline()

    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "A"

    # Chk 7: Model Governance & Attestation Sign-off
    sys.stdout.write("\nAction [[A]ccept (default) / [O]verride / [C]hallenge / [Q]uestion / [S]kip / [V]iew / [E]vidence / [D]ashboard / [H]elp]: ")
    sys.stdout.flush()
    ans = sys.stdin.readline().strip()
    assert ans.upper() == "A"

    # Final Execution & Attestation Seal
    sys.stdout.write("\nAttestation Seal: start-seal-mkt-91b402\n")
    sys.stdout.write("Merklized Ledger Replay: INTACT | 0 divergences | Review complete!\n")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Deterministic Mock Twin Runner")
    parser.add_argument("--session", required=True, choices=["SESSION_A_TEMPORAL", "SESSION_B_MARKET", "SESSION_C_POLICY"])
    parser.add_argument("--artifact-manifest")
    args = parser.parse_args()

    if args.session == "SESSION_A_TEMPORAL":
        run_mock_session_a(args.artifact_manifest)
    elif args.session == "SESSION_B_MARKET":
        run_mock_session_b()
    elif args.session == "SESSION_C_POLICY":
        from start.runtime import CanonicalExecutionService
        CanonicalExecutionService.execute(
            workflow_id="quantitative_finance",
            context_id="institutional_market_v1",
            seed=42,
            output_root="start_output",
            trace_mode="engineering",
        )
        print("Executing Canonical Workflow 'quantitative_finance' on Context 'institutional_market_v1'...")
        print("Workflow 'quantitative_finance' completed successfully -> start_output/RUN-mock")


if __name__ == "__main__":
    main()
