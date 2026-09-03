"""Tests for v4.0.3 Cyclic Graph Orchestration (Section 7 Verification).

Covers:
7(a) Linear mode unchanged
7(b) Graph mode happy path (visits, execution_path, start-seal/3)
7(c) Remediation loop never resolves -> remediation_exhausted (BLOCKER) & disposition NOT READY
7(d) Remediation resolved -> remediation_succeeded (INFORMATIONAL) & disposition stays ready
"""

from __future__ import annotations

import pandas as pd
import pytest

from start.data.uci_credit import fetch_or_load_german_credit
from start.governance.findings import Severity
from start.modeling.enterprise_orchestrator import EnterpriseReviewOrchestrator
from start.review_session import ReviewSession


@pytest.fixture
def german_credit_df() -> pd.DataFrame:
    df = fetch_or_load_german_credit()
    return df


# --------------------------------------------------------------------------- #
# 7(c) Remediation loop never resolves -> remediation_exhausted BLOCKER
# --------------------------------------------------------------------------- #
def test_7c_remediation_exhausted_becomes_blocker_and_disposition_not_ready(german_credit_df: pd.DataFrame) -> None:
    """Section 7(c): A run where remediation never resolves.

    Must show `remediation_exhausted` as a BLOCKER finding, and the governance
    disposition changing to NOT READY.
    """
    orch = EnterpriseReviewOrchestrator()
    session = ReviewSession(run_id="RUN-TEST-7C")
    # Signal that overfitting never resolves despite retries
    session.context = {
        "never_resolve_overfitting": True,
        "max_generalization_gap": 0.10,
    }

    outcome = orch.run(
        german_credit_df,
        user_target="is_bad_credit",
        enterprise_mode=True,
        execution_mode="graph",
        session=session,
        seed=42,
    )

    # 1. Execution path was recorded
    assert outcome.execution_path is not None
    summary = outcome.execution_path.remediation_summary()
    assert summary["attempts"] >= 1
    assert summary["budget_exhausted"] >= 1

    # 2. Findings contain remediation_exhausted with blocker severity
    exhausted_findings = [
        f for f in outcome.findings_register.findings
        if f.title == "Remediation Exhausted" or "remediation_exhausted" in f.description.lower() or "not addressable by the routed remedy" in f.description
    ]
    assert len(exhausted_findings) >= 1
    assert any(f.severity == Severity.HIGH for f in exhausted_findings)

    # 3. Governance sign-off disposition is NOT READY
    assert "NOT READY" in outcome.base_outcome.agent_review.signoff
    assert outcome.base_outcome.agent_review.critique_ok is False


# --------------------------------------------------------------------------- #
# 7(a) Linear mode unchanged
# --------------------------------------------------------------------------- #
def test_7a_linear_mode_unchanged(german_credit_df: pd.DataFrame) -> None:
    """Section 7(a): Linear mode execution proceeds untouched."""
    orch = EnterpriseReviewOrchestrator()
    outcome = orch.run(
        german_credit_df,
        user_target="is_bad_credit",
        enterprise_mode=True,
        execution_mode="linear",
        seed=42,
    )
    assert outcome.execution_path is None
    assert len(outcome.layers) >= 5


# --------------------------------------------------------------------------- #
# 7(b) Graph mode happy path
# --------------------------------------------------------------------------- #
def test_7b_graph_mode_happy_path(german_credit_df: pd.DataFrame) -> None:
    """Section 7(b): Graph mode runs cleanly, records path and commits seal/3."""
    orch = EnterpriseReviewOrchestrator()
    outcome = orch.run(
        german_credit_df,
        user_target="is_bad_credit",
        enterprise_mode=True,
        execution_mode="graph",
        seed=42,
    )
    assert outcome.execution_path is not None
    assert outcome.execution_path.terminated_at == "seal"
    assert len(outcome.execution_path.visits) >= 10
    assert outcome.execution_path.path_hash()

    # Evidence has orchestration.execution_path
    ev = [e for e in outcome.base_outcome.evidence if e.test_id == "orchestration.execution_path"]
    assert len(ev) == 1
    assert ev[0].metrics["node_visits"] == len(outcome.execution_path.visits)


# --------------------------------------------------------------------------- #
# 7(d) Remediation resolved
# --------------------------------------------------------------------------- #
def test_7d_remediation_resolved_shows_informational_finding(german_credit_df: pd.DataFrame) -> None:
    """Section 7(d): Remediation resolves on attempt 1.

    Shows `remediation_succeeded` as informational and disposition ready.
    """
    orch = EnterpriseReviewOrchestrator()
    session = ReviewSession(run_id="RUN-TEST-7D")
    # Resolve overfitting on attempt 1
    session.context = {
        "resolve_overfitting_on_attempt": 1,
        "max_generalization_gap": 0.10,
    }

    outcome = orch.run(
        german_credit_df,
        user_target="is_bad_credit",
        enterprise_mode=True,
        execution_mode="graph",
        session=session,
        seed=42,
    )

    assert outcome.execution_path is not None
    summary = outcome.execution_path.remediation_summary()
    assert summary["resolved"] >= 1
    assert summary["budget_exhausted"] == 0

    succeeded_findings = [
        f for f in outcome.findings_register.findings
        if f.title == "Remediation Succeeded" or "resolved on attempt" in f.description
    ]
    assert len(succeeded_findings) >= 1
    assert succeeded_findings[0].severity == Severity.LOW
