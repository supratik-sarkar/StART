"""Governance findings example.

Shows the governance findings engine directly: how findings are derived from
evidence, rated by severity/materiality, mapped to a risk category, linked to
evidence IDs, and prioritized into a register — the building blocks behind the
enterprise dashboard's Governance Findings section.

    python examples/governance_findings_demo.py
"""

from __future__ import annotations

from start.core.schemas import Status, TestResult
from start.governance import (
    Finding,
    FindingsRegister,
    Materiality,
    Severity,
    derive_findings_from_evidence,
)


def main() -> None:
    register = FindingsRegister()

    # 1. A finding authored directly (e.g. by a reviewer or governance agent).
    register.add(
        Finding(
            title="Target Leakage",
            description="A post-event variable is almost perfectly correlated with the target.",
            severity=Severity.HIGH,
            materiality=Materiality.HIGH,
            risk_category="Data Quality",
            evidence_ids=["EV-LEAK-01"],
            recommendation="Remove post-event variables and re-run the review.",
            source="reviewer",
        )
    )

    # 2. Findings derived automatically from warn/fail evidence records.
    evidence = [
        TestResult(
            test_id="deep_learning.calibration_diagnostics",
            test_name="Calibration (ECE)",
            status=Status.WARN,
            interpretation="Expected calibration error is elevated on the OOS cohort.",
        ),
        TestResult(
            test_id="execution.cohort_metrics",
            test_name="OOS performance",
            status=Status.FAIL,
            interpretation="OOS AUC fell below the acceptance threshold.",
        ),
        TestResult(test_id="discovery.dataset_profile", test_name="Profile", status=Status.PASS),
    ]
    register.extend(derive_findings_from_evidence(evidence))

    print("Governance findings register\n" + "=" * 60)
    s = register.summary()
    print(
        f"Total: {s['total']}  |  Critical={s['Critical']} High={s['High']} "
        f"Medium={s['Medium']} Low={s['Low']}\n"
    )
    for f in register.sorted():
        print(f"[{f.severity.value}/{f.materiality.value}] {f.risk_category}: {f.title}")
        print(f"    {f.description}")
        print(f"    evidence: {', '.join(f.evidence_ids) or '—'}")
        print(f"    recommendation: {f.recommendation}\n")

    blocking = register.blocking()
    uncited = register.uncited()
    print(f"Blocking findings (High/Critical): {len(blocking)}")
    print(f"Uncited findings (flagged by EvidenceCritic): {len(uncited)}")
    if not uncited:
        print("All findings are evidence-backed — no uncited governance findings.")


if __name__ == "__main__":
    main()
