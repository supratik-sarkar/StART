from __future__ import annotations

import pytest

from start.governance import (
    Finding,
    FindingsRegister,
    Materiality,
    Severity,
    derive_findings_from_evidence,
)
from start.governance.findings import RISK_CATEGORIES


def test_finding_fields_and_priority():
    f = Finding(
        title="Target Leakage",
        description="Post-event variable detected.",
        severity="High",
        materiality="High",
        risk_category="Data Quality",
        evidence_ids=["EV-1"],
        recommendation="Remove post-event variables.",
    )
    assert f.severity == Severity.HIGH and f.materiality == Materiality.HIGH
    assert f.is_cited
    assert f.priority == Severity.HIGH.rank * 10 + Materiality.HIGH.rank
    d = f.to_dict()
    assert d["severity"] == "High" and d["evidence_ids"] == ["EV-1"]


def test_severity_materiality_ranks():
    assert Severity.CRITICAL.rank > Severity.HIGH.rank > Severity.MEDIUM.rank > Severity.LOW.rank
    assert Materiality.HIGH.rank > Materiality.MEDIUM.rank > Materiality.LOW.rank


def test_unknown_category_normalized():
    f = Finding("X", "desc", "Low", "Low", "NotARealCategory")
    assert f.risk_category == "Governance"
    assert all(c in RISK_CATEGORIES for c in [f.risk_category])


def test_register_sorting_and_filters():
    reg = FindingsRegister()
    reg.add(Finding("low", "d", "Low", "Low", "Operational", evidence_ids=["E1"]))
    reg.add(Finding("crit", "d", "Critical", "High", "Security", evidence_ids=["E2"]))
    reg.add(Finding("med", "d", "Medium", "Medium", "Robustness", evidence_ids=["E3"]))
    ordered = reg.sorted()
    assert ordered[0].title == "crit" and ordered[-1].title == "low"
    assert len(reg.blocking()) == 1  # only Critical
    assert reg.summary()["total"] == 3
    assert reg.summary()["Critical"] == 1


def test_uncited_findings_detected():
    reg = FindingsRegister()
    reg.add(Finding("cited", "d", "High", "High", "Data Quality", evidence_ids=["E1"]))
    reg.add(Finding("uncited", "d", "High", "High", "Data Quality"))
    assert len(reg.uncited()) == 1
    assert reg.uncited()[0].title == "uncited"


def test_derive_findings_from_evidence():
    from start.core.schemas import Status, TestResult

    fail = TestResult(
        test_id="feature_engineering.diagnostics",
        test_name="Leakage check",
        status=Status.FAIL,
        interpretation="Leakage detected.",
    )
    warn = TestResult(
        test_id="deep_learning.calibration_diagnostics",
        test_name="Calibration",
        status=Status.WARN,
        interpretation="ECE elevated.",
    )
    ok = TestResult(test_id="x.y", test_name="OK", status=Status.PASS)
    findings = derive_findings_from_evidence([fail, warn, ok])
    assert len(findings) == 2  # pass produces no finding
    sev = {f.severity for f in findings}
    assert Severity.HIGH in sev and Severity.MEDIUM in sev
    assert findings[0].risk_category == "Data Quality"
    assert all(f.is_cited for f in findings)


def test_finding_requires_severity_materiality():
    with pytest.raises(ValueError):
        Finding("x", "d", "Severe", "High", "Data Quality")  # invalid severity
