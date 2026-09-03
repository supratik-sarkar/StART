"""Typed analytical claims for StART cross-analytical reasoning.

Invariants:
- Quantitative claim assertions bind directly to EvidenceRecord metric paths.
- Zero free-floating LLM arithmetic.
- Objective, non-normative classification: sensitivity/disagreement without
  policy thresholds remains EVIDENCE_ONLY.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from start.core.hashing import canonical_json, sha256_hex


class ClaimType(StrEnum):
    """Category of analytical claim."""

    OBSERVATION = "OBSERVATION"
    RECONCILIATION = "RECONCILIATION"
    SENSITIVITY = "SENSITIVITY"
    DEPENDENCY = "DEPENDENCY"
    CONTRADICTION = "CONTRADICTION"
    UNRESOLVED_RISK = "UNRESOLVED_RISK"
    METHOD_DISAGREEMENT = "METHOD_DISAGREEMENT"


class ClaimStatus(StrEnum):
    """Validation status of an analytical claim."""

    VERIFIED = "VERIFIED"
    EVIDENCE_ONLY = "EVIDENCE_ONLY"
    UNRESOLVED = "UNRESOLVED"
    CONTRADICTED = "CONTRADICTED"


def new_claim_id() -> str:
    return f"CLM-{uuid.uuid4().hex[:10]}"


@dataclass(frozen=True)
class AnalyticalClaim:
    """A typed, deterministic assertion derived from one or more EvidenceRecords."""

    claim_id: str
    source_evidence_ids: tuple[str, ...]
    claim_type: ClaimType
    domain: str
    metric_paths: dict[str, str]
    status: ClaimStatus
    statement: str
    limitations: tuple[str, ...] = field(default_factory=tuple)
    threshold_provenance: str | None = None
    statistical_criterion_source: str | None = None
    statistical_gamma_test: float | None = None
    materiality_criterion_source: str = "NONE"
    payload: dict[str, Any] = field(default_factory=dict)
    data_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "source_evidence_ids": list(self.source_evidence_ids),
            "claim_type": self.claim_type.value,
            "domain": self.domain,
            "metric_paths": self.metric_paths,
            "status": self.status.value,
            "statement": self.statement,
            "limitations": list(self.limitations),
            "threshold_provenance": self.threshold_provenance,
            "statistical_criterion_source": self.statistical_criterion_source,
            "statistical_gamma_test": self.statistical_gamma_test,
            "materiality_criterion_source": self.materiality_criterion_source,
            "payload": self.payload,
            "data_fingerprint": self.data_fingerprint,
        }

    def canonical_hash(self) -> str:
        serialized = canonical_json(self.to_dict())
        return sha256_hex(serialized)
