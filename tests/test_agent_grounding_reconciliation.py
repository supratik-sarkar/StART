"""Deterministic regression tests for Agent Grounding Reconciliation.

Verifies that immutable deterministic DataContract / EvidenceRecord facts dominate agent claims:
- Modality: sequence vs tabular
- Tensor rank: 3
- Shape: (800, 24, 3), T=24, F=3
- Architecture: lstm

Proves that malformed agent output (e.g. ValidationPlannerAgent claiming tabular) is
deterministically rejected and invalidated, rather than generating an ungrounded cross-agent
collision in the Human Adjudication Council.
"""

from __future__ import annotations

from typing import Any

from start.consensus import (
    FactualGroundingReconciliation,
    detect_collisions,
    ground_agent_claims,
)


def _make_sequence_discovery_evidence() -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": "EV-DISC-01",
            "test_id": "discovery",
            "status": "pass",
            "metrics": {
                "modality": "sequence",
                "tensor_rank": 3,
                "tensor_shape": "[800, 24, 3]",
                "timesteps": 24,
                "n_features": 3,
                "n_samples": 800,
            },
        },
        {
            "evidence_id": "EV-ARCH-01",
            "test_id": "architecture",
            "status": "pass",
            "metrics": {"selected_architecture": "lstm"},
        },
    ]


def test_deterministic_sequence_fact_dominates_malformed_tabular_claim() -> None:
    """Exact live-failure regression:

    Canonical contract:
      modality = sequence
      shape = [800, 24, 3]
      T = 24, F = 3
      architecture = lstm

    Malformed agent output:
      ValidationPlannerAgent asserts modality = tabular

    Must prove:
    1. Deterministic grounding rejection occurs.
    2. Structured invalidation of the claim.
    3. Canonical sequence contract survives unchanged.
    4. False claim does NOT create an ungrounded Human Adjudication Council collision.
    5. An explicit FactualGroundingReconciliation record is generated.
    """
    evidence = _make_sequence_discovery_evidence()
    malformed_agent_outputs = {
        "ArchitectureReviewAgent": {"recommended_family": "lstm", "modality": "sequence"},
        "ValidationPlannerAgent": {"expected_modality": "tabular"},
    }

    # 1. Direct grounding validation test
    grounded_outputs, reconciliations = ground_agent_claims(
        malformed_agent_outputs,
        evidence_records=evidence,
        plan={"modality": "sequence", "architecture": "lstm"},
    )

    # Assert reconciliation record was generated
    assert len(reconciliations) == 1
    rec = reconciliations[0]
    assert isinstance(rec, FactualGroundingReconciliation)
    assert rec.agent == "ValidationPlannerAgent"
    assert rec.property_name == "expected_modality"
    assert rec.claimed_value == "tabular"
    assert rec.canonical_value == "sequence"
    assert rec.reconciliation == "invalidated_by_deterministic_contract"
    assert "EV-DISC-01" in rec.evidence_id
    assert "contradicts immutable deterministic contract" in rec.detail

    # Assert grounded output has invalidated the false claim
    val_grounded = grounded_outputs["ValidationPlannerAgent"]
    assert val_grounded["expected_modality"] == "sequence"
    assert val_grounded["grounding_status"] == "invalidated_by_deterministic_contract"
    assert val_grounded["grounding_reconciliation"]["canonical_value"] == "sequence"

    # 2. Collision detector integration test
    # Because the false claim was invalidated against canonical evidence,
    # it must NOT create an architecture_contradiction collision!
    cols = detect_collisions(
        evidence_records=evidence,
        agent_outputs=malformed_agent_outputs,
        plan={"modality": "sequence", "architecture": "lstm"},
    )
    assert len(cols) == 0, f"Expected 0 collisions after grounding reconciliation, got: {cols}"

    # Verify last_reconciliations attribute on detect_collisions
    last_recs = getattr(detect_collisions, "last_reconciliations", [])
    assert len(last_recs) == 1
    assert last_recs[0].agent == "ValidationPlannerAgent"


def test_case_1_all_agents_grounded_correctly_no_council_collision() -> None:
    """Case 1: When all agents are grounded with canonical sequence contract,

    no collision is raised and no Human Adjudication Council is needed.
    """
    evidence = _make_sequence_discovery_evidence()
    valid_outputs = {
        "ArchitectureReviewAgent": {"recommended_family": "lstm", "modality": "sequence"},
        "ValidationPlannerAgent": {"expected_modality": "sequence"},
    }
    cols = detect_collisions(
        evidence_records=evidence,
        agent_outputs=valid_outputs,
        plan={"modality": "sequence", "architecture": "lstm"},
    )
    assert len(cols) == 0
    last_recs = getattr(detect_collisions, "last_reconciliations", [])
    assert len(last_recs) == 0


def test_unspecified_evidence_preserves_legacy_collision_for_council() -> None:
    """When evidence is empty/unspecified, the claim cannot be automatically reconciled

    and must proceed to the Human Adjudication Council for audit transparency.
    """
    empty_evidence: list[dict[str, Any]] = []
    outputs = {
        "ArchitectureReviewAgent": {"recommended_family": "lstm"},
        "ValidationPlannerAgent": {"expected_modality": "tabular"},
    }
    cols = detect_collisions(evidence_records=empty_evidence, agent_outputs=outputs)
    assert len(cols) == 1
    assert cols[0].rule_name == "architecture_contradiction"
