"""Governance findings engine.

A structured finding is the unit of model-risk governance. Every finding
carries a severity, materiality, risk category (control), the evidence IDs that
support it, and a remediation recommendation. Findings are produced by the
validation/governance layers and the AI-engineering adapters, collected into a
register, and rendered into the dashboard and reports.

No finding is uncited: every finding references at least one evidence ID (or is
explicitly marked as an informational/operational note), enforced by the
EvidenceCritic gate downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

    @property
    def rank(self) -> int:
        return {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}[self.value]


class Materiality(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

    @property
    def rank(self) -> int:
        return {"Low": 1, "Medium": 2, "High": 3}[self.value]


# Risk categories (controls) findings can map to.
RISK_CATEGORIES = (
    "Data Quality",
    "Model Performance",
    "Explainability",
    "Robustness",
    "Calibration",
    "Bias & Fairness",
    "Security",
    "Compliance",
    "Operational",
    "Governance",
)


@dataclass
class Finding:
    title: str
    description: str
    severity: Severity
    materiality: Materiality
    risk_category: str
    evidence_ids: list[str] = field(default_factory=list)
    recommendation: str = ""
    source: str = ""  # which agent/adapter raised it

    def __post_init__(self) -> None:
        if isinstance(self.severity, str):
            self.severity = Severity(self.severity)
        if isinstance(self.materiality, str):
            self.materiality = Materiality(self.materiality)
        if self.risk_category not in RISK_CATEGORIES:
            # accept unknown categories but normalize to Governance with a note
            self.description += f" (category '{self.risk_category}' normalized)"
            self.risk_category = "Governance"

    @property
    def priority(self) -> int:
        """Composite priority for sorting (severity dominates, materiality breaks ties)."""
        return self.severity.rank * 10 + self.materiality.rank

    @property
    def is_cited(self) -> bool:
        return bool(self.evidence_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "materiality": self.materiality.value,
            "risk_category": self.risk_category,
            "evidence_ids": list(self.evidence_ids),
            "recommendation": self.recommendation,
            "source": self.source,
        }


@dataclass
class FindingsRegister:
    findings: list[Finding] = field(default_factory=list)

    def add(self, finding: Finding) -> None:
        self.findings.append(finding)

    def extend(self, findings: list[Finding]) -> None:
        self.findings.extend(findings)

    def sorted(self) -> list[Finding]:
        return sorted(self.findings, key=lambda f: -f.priority)

    def by_severity(self, severity: Severity) -> list[Finding]:
        return [f for f in self.findings if f.severity == severity]

    def blocking(self) -> list[Finding]:
        """High/Critical findings that block an unconditional sign-off."""
        return [f for f in self.findings if f.severity.rank >= Severity.HIGH.rank]

    def uncited(self) -> list[Finding]:
        return [f for f in self.findings if not f.is_cited]

    def summary(self) -> dict[str, int]:
        counts = {s.value: 0 for s in Severity}
        for f in self.findings:
            counts[f.severity.value] += 1
        counts["total"] = len(self.findings)
        return counts

    def to_list(self) -> list[dict[str, Any]]:
        return [f.to_dict() for f in self.sorted()]


def derive_findings_from_evidence(evidence: list[Any]) -> list[Finding]:
    """Map warn/fail evidence records into governance findings, so every
    diagnostic breach becomes a cited, severity-rated finding."""
    findings: list[Finding] = []
    for rec in evidence:
        status = getattr(rec.status, "value", str(rec.status))
        ev_id = getattr(rec, "evidence_id", getattr(rec, "test_id", "unknown"))
        if status == "fail":
            findings.append(
                Finding(
                    title=f"{rec.test_name}: failure",
                    description=rec.interpretation or f"{rec.test_name} failed its threshold.",
                    severity=Severity.HIGH,
                    materiality=Materiality.HIGH,
                    risk_category=_category_for(rec.test_id),
                    evidence_ids=[ev_id],
                    recommendation="Investigate and remediate before sign-off.",
                    source="evidence",
                )
            )
        elif status == "warn":
            findings.append(
                Finding(
                    title=f"{rec.test_name}: warning",
                    description=rec.interpretation or f"{rec.test_name} raised a warning.",
                    severity=Severity.MEDIUM,
                    materiality=Materiality.MEDIUM,
                    risk_category=_category_for(rec.test_id),
                    evidence_ids=[ev_id],
                    recommendation="Review and disposition the warning.",
                    source="evidence",
                )
            )
    return findings


def _category_for(test_id: str) -> str:
    tid = test_id.lower()
    if "leak" in tid or "feature_engineering" in tid or "discovery" in tid:
        return "Data Quality"
    if "calibration" in tid:
        return "Calibration"
    if "robust" in tid:
        return "Robustness"
    if "explain" in tid or "importance" in tid or "saliency" in tid:
        return "Explainability"
    if "sensitivity" in tid:
        return "Robustness"
    if "performance" in tid or "cohort" in tid or "metric" in tid:
        return "Model Performance"
    return "Governance"
