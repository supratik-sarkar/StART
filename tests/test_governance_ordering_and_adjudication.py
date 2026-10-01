"""Regression tests for governance ordering and adjudication integration.

Verifies that:
1. Formal MRM governance sign-off must NOT finalize as READY / READY WITH CONDITIONS
   if an unresolved, deferred, or rejected cross-agent collision remains.
2. An adjudication decision of 'defer' or 'reject_run' produces a BLOCKER factor in evaluate_signoff.
3. Successfully resolved adjudications ('uphold_a', 'uphold_b', 'reconcile_partial')
   are recorded as passing factors.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from start.mrm_signoff import (
    NOT_READY,
    READY,
    evaluate_signoff,
)


def _make_mock_store(pass_all_metrics: bool = True) -> MagicMock:
    store = MagicMock()
    if pass_all_metrics:
        store.cohort_metrics = {
            "oos": {
                "auc_roc": 0.95,
                "ece": 0.05,
            },
            "train": {
                "auc_roc": 0.96,
            },
        }
        store.max_abs_drift = 0.05
        store.most_sensitive_feature = "feat_0"
        store.benchmark = {"status": "ok", "verdict": "Outperformed baseline"}
    return store


def test_deferred_adjudication_blocks_formal_governance() -> None:
    """When a collision is deferred to Senior Committee, formal sign-off must be NOT READY."""
    store = _make_mock_store()
    session = MagicMock()
    session.challenges = []
    session.overrides.return_value = []
    session.adjudications = [
        {
            "collision_id": "col-001",
            "rule_name": "architecture_contradiction",
            "decision": "defer",
            "reviewer": "MRO_LEAD",
            "evidence_id": "EV-ADJ-col-001",
            "rationale": "Deferred for senior review.",
        }
    ]

    signoff = evaluate_signoff(store, session, primary_metric="auc_roc")

    # Must be NOT READY because defer is a blocker
    assert signoff.verdict == NOT_READY
    blocker_factors = [f for f in signoff.factors if f.status == "blocker"]
    assert len(blocker_factors) >= 1
    assert any("Cross-agent collisions" in f.name for f in blocker_factors)
    assert any("blocks formal sign-off" in f.detail for f in blocker_factors)


def test_rejected_run_adjudication_blocks_formal_governance() -> None:
    """When a run is rejected via adjudication, formal sign-off must be NOT READY."""
    store = _make_mock_store()
    session = MagicMock()
    session.challenges = []
    session.overrides.return_value = []
    session.adjudications = [
        {
            "collision_id": "col-002",
            "rule_name": "architecture_contradiction",
            "decision": "reject_run",
            "reviewer": "MRO_LEAD",
            "evidence_id": "EV-ADJ-col-002",
            "rationale": "Premise rejected.",
        }
    ]

    signoff = evaluate_signoff(store, session, primary_metric="auc_roc")

    assert signoff.verdict == NOT_READY
    assert any(f.status == "blocker" and "Cross-agent collisions" in f.name for f in signoff.factors)


def test_resolved_adjudication_passes_formal_governance() -> None:
    """When a collision is successfully resolved (e.g. uphold_a), it contributes a passing factor."""
    store = _make_mock_store()
    session = MagicMock()
    session.challenges = []
    session.overrides.return_value = []
    session.adjudications = [
        {
            "collision_id": "col-003",
            "rule_name": "architecture_contradiction",
            "decision": "uphold_a",
            "reviewer": "MRO_LEAD",
            "evidence_id": "EV-ADJ-col-003",
            "rationale": "Upholding Agent A sequence contract.",
        }
    ]

    signoff = evaluate_signoff(store, session, primary_metric="auc_roc")

    # All other factors pass, so verdict is READY
    assert signoff.verdict == READY
    col_factor = next(f for f in signoff.factors if f.name == "Cross-agent collisions")
    assert col_factor.status == "ok"
    assert "resolved via human adjudication (uphold_a)" in col_factor.detail
