"""Tests for StART EvidenceGraph and AnalyticalClaim architecture (Combined Gate 7-9 Slice A)."""

from __future__ import annotations

from start.core.schemas import EvidenceRecord, Status
from start.evidence.claims import (
    AnalyticalClaim,
    ClaimStatus,
    ClaimType,
    new_claim_id,
)
from start.evidence.graph import (
    EvidenceGraph,
    RelationshipType,
)


def _make_dummy_evidence(test_id: str, name: str, metrics: dict, status: Status = Status.PASS) -> EvidenceRecord:
    return EvidenceRecord(
        test_id=test_id,
        test_name=name,
        model_id="MOD-TEST",
        dataset_id="DS-SYNTH",
        run_id="RUN-001",
        metrics=metrics,
        status=status,
        interpretation="Test interpretation.",
        limitations=["Limitation 1"],
    )


def test_evidence_node_and_graph_construction():
    """Verify node creation from EvidenceRecord and graph node addition."""
    rec1 = _make_dummy_evidence("portfolio.mvo", "Mean Variance", {"sharpe": 1.5})
    rec2 = _make_dummy_evidence("covariance.ledoit_wolf", "Ledoit-Wolf", {"condition_number": 12.0})

    graph = EvidenceGraph()
    node1 = graph.add_node(rec1, domain="portfolio")
    node2 = graph.add_node(rec2, domain="covariance")

    assert graph.node_count == 2
    assert graph.edge_count == 0
    assert node1.evidence_id == rec1.evidence_id
    assert node1.domain == "portfolio"
    assert node2.domain == "covariance"
    assert node1.metric_paths["sharpe"] == 1.5


def test_evidence_graph_typed_relationships():
    """Verify typed edges and query methods across all required relationship concepts."""
    rec1 = _make_dummy_evidence("portfolio.mvo", "MVO", {"status": "converged"})
    rec2 = _make_dummy_evidence("covariance.ledoit_wolf", "LW", {"is_psd": True})
    rec3 = _make_dummy_evidence("scenario.reverse_stress", "Reverse Stress", {"target_loss": 0.20})

    graph = EvidenceGraph()
    graph.add_node(rec1)
    graph.add_node(rec2)
    graph.add_node(rec3)

    edge1 = graph.add_edge(
        source_id=rec1.evidence_id,
        target_id=rec2.evidence_id,
        relation=RelationshipType.DEPENDS_ON,
        provenance_rule="mvo_covariance_dependency",
    )
    edge2 = graph.add_edge(
        source_id=rec3.evidence_id,
        target_id=rec1.evidence_id,
        relation=RelationshipType.DIAGNOSTIC_OF,
        provenance_rule="reverse_stress_mvo_probe",
    )

    assert graph.edge_count == 2
    assert edge1.relation == RelationshipType.DEPENDS_ON
    assert edge2.relation == RelationshipType.DIAGNOSTIC_OF

    # Out/In edges
    out_edges_1 = graph.get_out_edges(rec1.evidence_id)
    assert len(out_edges_1) == 1
    assert out_edges_1[0].target_id == rec2.evidence_id

    in_edges_1 = graph.get_in_edges(rec1.evidence_id)
    assert len(in_edges_1) == 1
    assert in_edges_1[0].source_id == rec3.evidence_id

    # Filter relationships
    rel_edges = graph.get_relationships(rec1.evidence_id, relation=RelationshipType.DEPENDS_ON)
    assert len(rel_edges) == 1


def test_evidence_graph_canonical_serialization():
    """Verify deterministic serialization and fingerprinting of EvidenceGraph."""
    rec1 = _make_dummy_evidence("portfolio.mvo", "MVO", {"ret": 0.10})
    rec2 = _make_dummy_evidence("covariance.sample", "Sample", {"var": 0.04})

    graph1 = EvidenceGraph()
    graph1.add_node(rec1)
    graph1.add_node(rec2)
    graph1.add_edge(rec1.evidence_id, rec2.evidence_id, RelationshipType.DEPENDS_ON)

    fp1 = graph1.canonical_fingerprint()
    assert isinstance(fp1, str)
    assert len(fp1) == 64

    # Identical graph yields identical fingerprint
    graph2 = EvidenceGraph()
    graph2.add_node(rec1)
    graph2.add_node(rec2)
    graph2.add_edge(rec1.evidence_id, rec2.evidence_id, RelationshipType.DEPENDS_ON)
    assert graph2.canonical_fingerprint() == fp1


def test_analytical_claim_typed_structure_and_hash():
    """Verify AnalyticalClaim fields, non-normative classification, and deterministic hashing."""
    claim = AnalyticalClaim(
        claim_id=new_claim_id(),
        source_evidence_ids=("EV-001", "EV-002"),
        claim_type=ClaimType.SENSITIVITY,
        domain="covariance",
        metric_paths={"cov_diff": "EV-001.metrics.turnover"},
        status=ClaimStatus.EVIDENCE_ONLY,
        statement="Measured turnover across covariance models is 0.15 without external policy threshold.",
        limitations=("Requires institutional threshold.",),
        payload={"turnover": 0.15},
    )

    d = claim.to_dict()
    assert d["claim_type"] == "SENSITIVITY"
    assert d["status"] == "EVIDENCE_ONLY"
    assert d["metric_paths"]["cov_diff"] == "EV-001.metrics.turnover"

    h = claim.canonical_hash()
    assert len(h) == 64
