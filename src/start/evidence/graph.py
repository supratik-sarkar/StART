"""Deterministic EvidenceGraph structure for StART cross-analytical reasoning.

Invariants:
- Graph construction is deterministic and grounded in EvidenceRecord provenance.
- Relationships are typed explicitly (DEPENDS_ON, DERIVED_FROM, CHALLENGES, CONTRADICTS,
  SUPPORTS, ALTERNATIVE_METHOD, DIAGNOSTIC_OF, GOVERNED_BY).
- Zero LLM arithmetic or non-deterministic edge generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from start.core.hashing import canonical_json, sha256_hex
from start.core.schemas import EvidenceRecord, Status


class RelationshipType(StrEnum):
    """Typed relationship between two EvidenceRecord nodes."""

    DEPENDS_ON = "DEPENDS_ON"
    DERIVED_FROM = "DERIVED_FROM"
    CHALLENGES = "CHALLENGES"
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"
    ALTERNATIVE_METHOD = "ALTERNATIVE_METHOD"
    DIAGNOSTIC_OF = "DIAGNOSTIC_OF"
    GOVERNED_BY = "GOVERNED_BY"


@dataclass(frozen=True)
class EvidenceNode:
    """A node in the EvidenceGraph representing an EvidenceRecord."""

    evidence_id: str
    test_id: str
    test_name: str
    domain: str
    status: str
    source_provenance: str
    metric_paths: dict[str, Any]
    limitations: tuple[str, ...]
    data_fingerprint: str

    @classmethod
    def from_record(cls, record: EvidenceRecord, domain: str = "") -> EvidenceNode:
        return cls(
            evidence_id=record.evidence_id,
            test_id=record.test_id,
            test_name=record.test_name,
            domain=domain or "portfolio",
            status=record.status.value if isinstance(record.status, Status) else str(record.status),
            source_provenance=record.model_id or record.dataset_id or "deterministic_engine",
            metric_paths=dict(record.metrics),
            limitations=tuple(record.limitations),
            data_fingerprint=record.input_artifact_hash or "",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "test_id": self.test_id,
            "test_name": self.test_name,
            "domain": self.domain,
            "status": self.status,
            "source_provenance": self.source_provenance,
            "metric_paths": self.metric_paths,
            "limitations": list(self.limitations),
            "data_fingerprint": self.data_fingerprint,
        }


@dataclass(frozen=True)
class EvidenceEdge:
    """A directed edge representing a typed relationship between evidence records."""

    source_id: str
    target_id: str
    relation: RelationshipType
    provenance_rule: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation.value,
            "provenance_rule": self.provenance_rule,
            "payload": self.payload,
        }


class EvidenceGraph:
    """Deterministic directed graph of evidence nodes and typed relationship edges."""

    def __init__(self) -> None:
        self._nodes: dict[str, EvidenceNode] = {}
        self._edges: list[EvidenceEdge] = []

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def add_node(self, record_or_node: EvidenceRecord | EvidenceNode, domain: str = "") -> EvidenceNode:
        """Add an evidence node to the graph."""
        if isinstance(record_or_node, EvidenceNode):
            node = record_or_node
        elif isinstance(record_or_node, EvidenceRecord):
            node = EvidenceNode.from_record(record_or_node, domain=domain)
        else:
            raise TypeError(f"Expected EvidenceRecord or EvidenceNode, got {type(record_or_node)}")

        self._nodes[node.evidence_id] = node
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: RelationshipType | str,
        provenance_rule: str = "",
        payload: dict[str, Any] | None = None,
    ) -> EvidenceEdge:
        """Add a directed relationship edge between two evidence nodes."""
        if source_id not in self._nodes:
            raise KeyError(f"Source evidence node '{source_id}' does not exist in graph.")
        if target_id not in self._nodes:
            raise KeyError(f"Target evidence node '{target_id}' does not exist in graph.")

        rel_enum = relation if isinstance(relation, RelationshipType) else RelationshipType(str(relation))
        edge = EvidenceEdge(
            source_id=source_id,
            target_id=target_id,
            relation=rel_enum,
            provenance_rule=provenance_rule,
            payload=payload or {},
        )
        self._edges.append(edge)
        return edge

    def get_node(self, evidence_id: str) -> EvidenceNode | None:
        return self._nodes.get(evidence_id)

    def get_nodes(self) -> dict[str, EvidenceNode]:
        return dict(self._nodes)

    def get_edges(self) -> list[EvidenceEdge]:
        return list(self._edges)

    def get_in_edges(self, target_id: str) -> list[EvidenceEdge]:
        return [e for e in self._edges if e.target_id == target_id]

    def get_out_edges(self, source_id: str) -> list[EvidenceEdge]:
        return [e for e in self._edges if e.source_id == source_id]

    def get_relationships(
        self,
        evidence_id: str,
        relation: RelationshipType | str | None = None,
    ) -> list[EvidenceEdge]:
        rel_enum = (
            relation
            if (relation is None or isinstance(relation, RelationshipType))
            else RelationshipType(str(relation))
        )
        edges = [e for e in self._edges if e.source_id == evidence_id or e.target_id == evidence_id]
        if rel_enum is not None:
            edges = [e for e in edges if e.relation == rel_enum]
        return edges

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": {eid: n.to_dict() for eid, n in self._nodes.items()},
            "edges": [e.to_dict() for e in self._edges],
            "node_count": self.node_count,
            "edge_count": self.edge_count,
        }

    def canonical_fingerprint(self) -> str:
        serialized = canonical_json(self.to_dict())
        return sha256_hex(serialized)
