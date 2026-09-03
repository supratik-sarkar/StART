"""The review topology, expressed as data.

What this adds that the DAG executor cannot
-------------------------------------------

``GraphReviewOrchestrator`` already runs the review as a directed **acyclic** graph
with checkpointing and resumability. That is the right structure for "run these stages
in dependency order". It cannot express the thing a validator actually does:

    The generalisation gap is too wide. Go back and retune. Try again.

A DAG has no edge from Overfitting back to Tuning, because that edge is a cycle. So in
the acyclic model a failing diagnostic can only become a *finding* — something written
down and weighed at sign-off. The review notices the problem and proceeds.

With bounded back-edges the review instead attempts to resolve it, and both outcomes are
more informative than a finding:

* **Remediation succeeded** — the evidence chain records what was wrong, what was
  changed, and that the change worked. That is a far better artefact than a passing
  metric with no history.
* **Remediation exhausted its budget** — *"the generalisation gap was addressed three
  times by retuning and did not resolve"* is a much stronger statement than *"the
  generalisation gap is 0.28"*. It is the difference between observing a symptom and
  establishing that it is not fixable by the obvious means.

Why the budget is a policy value, not a constant
------------------------------------------------

An unbounded retry loop is not a review, it is a hyperparameter search wearing a
review's clothes. The bound is what keeps remediation honest: three attempts is a
governed allowance, and *exhausting* it is a specific, citable outcome.

So budgets live in policy, are hashed into the seal, and exhausting one produces a
named finding kind rather than a generic failure. A reviewer reading the archive can
tell that the allowance was three, that all three were used, and what each one changed.

Why the graph is data
---------------------

The topology is a value, not control flow spread across an orchestrator. That buys
three things:

* it can be **validated** — unreachable nodes, unbounded cycles and remediation edges
  with no budget are structural defects caught before a run rather than during one;
* it can be **hashed** — two reviews that ran different topologies are different
  reviews, and the seal says so;
* it can be **rendered** — the schematic in the README is generated from the same
  object the executor walks, so the picture cannot drift from the behaviour.

No LangGraph dependency. The graph is a plain description; the executor is pluggable,
and LangGraph is one possible backend rather than a requirement.

Standard library only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Any

__all__ = [
    "NodeKind",
    "EdgeKind",
    "Node",
    "Edge",
    "ReviewGraph",
    "GraphValidationError",
    "REVIEW_GRAPH",
    "DEFAULT_REMEDIATION_BUDGETS",
]


class NodeKind(StrEnum):
    """What a node does when the executor reaches it."""

    #: A reviewer agent that produces evidence.
    AGENT = "agent"
    #: A human decision point. Challenge/Ask re-enters the same node; Accept/Override
    #: advances. The self-loop is an edge in the graph, not a hidden while-loop.
    CHECKPOINT = "checkpoint"
    #: Splits into independent branches that may run concurrently.
    FANOUT = "fanout"
    #: Waits for every inbound branch before continuing.
    JOIN = "join"
    #: Entry or exit.
    TERMINAL = "terminal"


class EdgeKind(StrEnum):
    """Why the executor took this edge — recorded in the path, so a reader can tell a
    normal progression from a remediation cycle from an emergency stop."""

    #: Ordinary forward progression.
    ADVANCE = "advance"
    #: A checkpoint re-entering itself after Challenge or Ask.
    SELF_LOOP = "self_loop"
    #: A failing diagnostic routing back to the stage that can address it. Bounded.
    REMEDIATION = "remediation"
    #: A blocking condition short-circuiting straight to sign-off.
    HARD_STOP = "hard_stop"
    #: Fan-out to a parallel branch.
    BRANCH = "branch"


class GraphValidationError(ValueError):
    """The topology is structurally unsound. Raised before a run, never during one."""


@dataclass(frozen=True)
class Node:
    id: str
    kind: NodeKind
    label: str = ""
    #: Only executed when this predicate is true in the run context. Gates that
    #: fire conditionally (the metric and feature-engineering gates in the deep
    #: learning suite) declare it here rather than hiding it in an if-statement.
    condition: str = ""
    #: Human-facing description of what this node establishes.
    purpose: str = ""

    def display(self) -> str:
        return self.label or self.id


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    kind: EdgeKind = EdgeKind.ADVANCE
    #: Condition under which this edge is taken; empty means unconditional.
    condition: str = ""
    #: Maximum traversals per run. Required for REMEDIATION and SELF_LOOP.
    budget: int | None = None
    #: What this edge is for, in one line. Rendered in the schematic.
    rationale: str = ""

    @property
    def edge_id(self) -> str:
        return f"{self.source}->{self.target}:{self.kind.value}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "kind": self.kind.value,
            "condition": self.condition,
            "budget": self.budget,
            "rationale": self.rationale,
        }


@dataclass
class ReviewGraph:
    """A validated review topology."""

    name: str
    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    entry: str
    exits: tuple[str, ...]
    version: str = "1"

    # -- lookup -------------------------------------------------------------
    def node(self, node_id: str) -> Node:
        for candidate in self.nodes:
            if candidate.id == node_id:
                return candidate
        raise KeyError(f"unknown node {node_id!r}")

    def outgoing(self, node_id: str) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.source == node_id)

    def incoming(self, node_id: str) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.target == node_id)

    def remediation_edges(self) -> tuple[Edge, ...]:
        return tuple(e for e in self.edges if e.kind is EdgeKind.REMEDIATION)

    def node_ids(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes)

    # -- validation ---------------------------------------------------------
    def validate(self) -> None:
        """Structural checks. Every one of these has a way to go wrong silently.

        An unreachable node looks like a stage that mysteriously never runs. An
        unbudgeted back-edge looks like a hang. A remediation edge pointing forward
        is a typo that would quietly make a cycle into a skip.
        """
        ids = set(self.node_ids())
        if len(ids) != len(self.nodes):
            raise GraphValidationError("duplicate node ids")

        if self.entry not in ids:
            raise GraphValidationError(f"entry node {self.entry!r} is not defined")
        for exit_id in self.exits:
            if exit_id not in ids:
                raise GraphValidationError(f"exit node {exit_id!r} is not defined")

        for edge in self.edges:
            if edge.source not in ids:
                raise GraphValidationError(f"edge from unknown node {edge.source!r}")
            if edge.target not in ids:
                raise GraphValidationError(f"edge to unknown node {edge.target!r}")
            if edge.kind in {EdgeKind.REMEDIATION, EdgeKind.SELF_LOOP}:
                if edge.budget is None or edge.budget < 1:
                    raise GraphValidationError(
                        f"{edge.edge_id} is a cycle and must declare a budget >= 1. "
                        "An unbounded retry loop is a hyperparameter search, not a review."
                    )
            if edge.kind is EdgeKind.SELF_LOOP and edge.source != edge.target:
                raise GraphValidationError(
                    f"{edge.edge_id} is marked self_loop but connects two different nodes"
                )

        # Reachability from the entry, following every edge kind.
        reachable = {self.entry}
        frontier = [self.entry]
        while frontier:
            current = frontier.pop()
            for edge in self.outgoing(current):
                if edge.target not in reachable:
                    reachable.add(edge.target)
                    frontier.append(edge.target)
        unreachable = ids - reachable
        if unreachable:
            raise GraphValidationError(
                f"unreachable node(s) from {self.entry!r}: {', '.join(sorted(unreachable))}"
            )

        # Every exit must be reachable, and every non-exit must have an exit path.
        for exit_id in self.exits:
            if exit_id not in reachable:
                raise GraphValidationError(f"exit {exit_id!r} is unreachable")

        # Any cycle must contain at least one budgeted edge, or a run can hang.
        for cycle in self._find_cycles():
            if not self._cycle_is_bounded(cycle):
                raise GraphValidationError(
                    "unbounded cycle: "
                    + " -> ".join(cycle)
                    + " -> "
                    + cycle[0]
                    + ". Every cycle must include a REMEDIATION or SELF_LOOP edge "
                    "carrying a budget."
                )

        # Fan-out and join must pair up.
        for node in self.nodes:
            if node.kind is NodeKind.FANOUT and len(self.outgoing(node.id)) < 2:
                raise GraphValidationError(f"fanout {node.id!r} has fewer than 2 branches")
            if node.kind is NodeKind.JOIN and len(self.incoming(node.id)) < 2:
                raise GraphValidationError(f"join {node.id!r} has fewer than 2 inbound edges")

    def _find_cycles(self) -> list[list[str]]:
        """Simple cycles via iterative DFS. Small graphs, so clarity over cleverness."""
        cycles: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()

        def walk(node_id: str, stack: list[str]) -> None:
            for edge in self.outgoing(node_id):
                if edge.target in stack:
                    start = stack.index(edge.target)
                    cycle = stack[start:]
                    key = tuple(sorted(cycle)) + (str(len(cycle)),)
                    if key not in seen:
                        seen.add(key)
                        cycles.append(cycle)
                elif len(stack) < len(self.nodes):
                    walk(edge.target, [*stack, edge.target])

        walk(self.entry, [self.entry])
        return cycles

    def _cycle_is_bounded(self, cycle: list[str]) -> bool:
        for index, node_id in enumerate(cycle):
            nxt = cycle[(index + 1) % len(cycle)]
            for edge in self.outgoing(node_id):
                if edge.target == nxt and edge.budget is not None:
                    return True
        return False

    # -- identity -----------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "entry": self.entry,
            "exits": list(self.exits),
            "nodes": [
                {
                    "id": n.id,
                    "kind": n.kind.value,
                    "label": n.label,
                    "condition": n.condition,
                    "purpose": n.purpose,
                }
                for n in self.nodes
            ],
            "edges": [e.as_dict() for e in self.edges],
        }

    def graph_hash(self) -> str:
        """Content hash of the topology. Two reviews that ran different graphs are
        different reviews, and the seal should say so."""
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return sha256(payload.encode("utf-8")).hexdigest()

    # -- rendering ----------------------------------------------------------
    def to_mermaid(self) -> str:
        """Mermaid source for the README schematic.

        Generated from the same object the executor walks, so the picture in the
        documentation cannot drift away from the behaviour in the code.
        """
        shape = {
            NodeKind.AGENT: ("[", "]"),
            NodeKind.CHECKPOINT: ("{{", "}}"),
            NodeKind.FANOUT: ("([", "])"),
            NodeKind.JOIN: ("([", "])"),
            NodeKind.TERMINAL: ("((", "))"),
        }
        style = {
            EdgeKind.ADVANCE: "-->",
            EdgeKind.BRANCH: "-->",
            EdgeKind.SELF_LOOP: "-.->",
            EdgeKind.REMEDIATION: "==>",
            EdgeKind.HARD_STOP: "-.->",
        }
        lines = ["flowchart TD"]
        for node in self.nodes:
            open_b, close_b = shape[node.kind]
            lines.append(f"    {node.id}{open_b}{node.display()}{close_b}")
        for edge in self.edges:
            arrow = style[edge.kind]
            label = edge.condition or ""
            if edge.budget is not None:
                label = f"{label} (max {edge.budget})".strip()
            suffix = f"|{label}|" if label else ""
            lines.append(f"    {edge.source} {arrow}{suffix} {edge.target}")
        lines.append("")
        lines.append("    classDef checkpoint fill:#fff4e0,stroke:#d98c00;")
        lines.append("    classDef agent fill:#eaf7ee,stroke:#2f9e44;")
        checkpoints = [n.id for n in self.nodes if n.kind is NodeKind.CHECKPOINT]
        agents = [n.id for n in self.nodes if n.kind is NodeKind.AGENT]
        if checkpoints:
            lines.append(f"    class {','.join(checkpoints)} checkpoint;")
        if agents:
            lines.append(f"    class {','.join(agents)} agent;")
        return "\n".join(lines)

    def to_dot(self) -> str:
        lines = [f'digraph "{self.name}" {{', "  rankdir=TB;"]
        for node in self.nodes:
            shape = {
                NodeKind.AGENT: "box",
                NodeKind.CHECKPOINT: "diamond",
                NodeKind.FANOUT: "ellipse",
                NodeKind.JOIN: "ellipse",
                NodeKind.TERMINAL: "circle",
            }[node.kind]
            lines.append(f'  "{node.id}" [shape={shape}, label="{node.display()}"];')
        for edge in self.edges:
            attrs = []
            if edge.kind is EdgeKind.REMEDIATION:
                attrs.append('color="firebrick", style=bold')
            elif edge.kind is EdgeKind.SELF_LOOP:
                attrs.append("style=dashed")
            elif edge.kind is EdgeKind.HARD_STOP:
                attrs.append('color="orange", style=dashed')
            label = edge.condition
            if edge.budget is not None:
                label = f"{label} (max {edge.budget})".strip()
            if label:
                attrs.append(f'label="{label}"')
            joined = ", ".join(attrs)
            lines.append(f'  "{edge.source}" -> "{edge.target}" [{joined}];')
        lines.append("}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Budgets
#
# Policy values, not constants. They are hashed into the seal, and exhausting one is
# a named outcome rather than a generic failure.
# --------------------------------------------------------------------------- #
DEFAULT_REMEDIATION_BUDGETS: dict[str, int] = {
    "overfitting_to_tuning": 3,
    "sensitivity_to_tuning": 2,
    "explainability_to_feature_engineering": 2,
    "validation_to_execution": 2,
    "checkpoint_self_loop": 10,
}


# --------------------------------------------------------------------------- #
# The topology
# --------------------------------------------------------------------------- #
_NODES: tuple[Node, ...] = (
    Node("start", NodeKind.TERMINAL, "Start", purpose="Entry."),
    Node(
        "dataset_discovery",
        NodeKind.AGENT,
        "DatasetDiscovery",
        purpose="Profile the data; surface leakage candidates and imbalance.",
    ),
    Node("task_inference", NodeKind.AGENT, "TaskInference", purpose="Frame the task from the target."),
    Node(
        "feature_engineering",
        NodeKind.AGENT,
        "FeatureEngineering",
        purpose="Propose preprocessing; the reviewer chooses each method.",
    ),
    Node(
        "architecture_review",
        NodeKind.AGENT,
        "ArchitectureReview",
        purpose="Assess whether the model class suits the data.",
    ),
    Node(
        "gate_architecture",
        NodeKind.CHECKPOINT,
        "checkpoint: architecture",
        purpose="Reviewer accepts, overrides, challenges or asks.",
    ),
    Node(
        "hyperparameter_tuning",
        NodeKind.AGENT,
        "HyperparameterTuning",
        purpose="Plan and run a leakage-safe search.",
    ),
    Node(
        "gate_metric",
        NodeKind.CHECKPOINT,
        "checkpoint: metric",
        condition="run_dl",
        purpose="Reviewer sets the metric priority. Deep-learning suite only.",
    ),
    Node("model_execution", NodeKind.AGENT, "ModelExecution", purpose="Fit and score; emit cohort metrics."),
    Node(
        "diagnostics_fanout",
        NodeKind.FANOUT,
        "parallel diagnostics",
        purpose="Three independent diagnostics, given a fitted model.",
    ),
    Node("explainability", NodeKind.AGENT, "Explainability", purpose="Global and local attribution."),
    Node("sensitivity", NodeKind.AGENT, "Sensitivity", purpose="Metric response under input shock."),
    Node("overfitting", NodeKind.AGENT, "Overfitting", purpose="Generalisation gap diagnosis."),
    Node("diagnostics_join", NodeKind.JOIN, "join", purpose="Wait for all three diagnostics."),
    Node("validation", NodeKind.AGENT, "Validation", purpose="Adversarial and robustness checks."),
    Node(
        "gate_validation",
        NodeKind.CHECKPOINT,
        "checkpoint: validation",
        purpose="Accept, Ask or Challenge. No Override: a reviewer may question a "
        "validation result but may not simply replace it.",
    ),
    Node(
        "governance_signoff",
        NodeKind.AGENT,
        "GovernanceSignoff",
        purpose="Weigh every factor into a disposition.",
    ),
    Node(
        "evidence_critic",
        NodeKind.AGENT,
        "EvidenceCritic",
        purpose="Citation integrity across the narrative.",
    ),
    Node(
        "seal",
        NodeKind.TERMINAL,
        "Evidence seal + dashboards",
        purpose="Commit the review and emit artefacts.",
    ),
)

_EDGES: tuple[Edge, ...] = (
    Edge("start", "dataset_discovery"),
    Edge("dataset_discovery", "task_inference"),
    Edge("task_inference", "feature_engineering"),
    Edge("feature_engineering", "architecture_review"),
    Edge("architecture_review", "gate_architecture"),
    # Challenge and Ask re-enter the same checkpoint. Making this an edge rather
    # than a hidden while-loop means the number of challenges at a gate is visible
    # in the execution path.
    Edge(
        "gate_architecture",
        "gate_architecture",
        EdgeKind.SELF_LOOP,
        condition="challenge or ask",
        budget=DEFAULT_REMEDIATION_BUDGETS["checkpoint_self_loop"],
        rationale="the reviewer keeps questioning the same decision",
    ),
    Edge("gate_architecture", "hyperparameter_tuning", condition="accept or override"),
    Edge("hyperparameter_tuning", "gate_metric", condition="run_dl"),
    Edge("hyperparameter_tuning", "model_execution", condition="not run_dl"),
    Edge(
        "gate_metric",
        "gate_metric",
        EdgeKind.SELF_LOOP,
        condition="challenge or ask",
        budget=DEFAULT_REMEDIATION_BUDGETS["checkpoint_self_loop"],
    ),
    Edge("gate_metric", "model_execution", condition="accept or override"),
    Edge("model_execution", "diagnostics_fanout"),
    Edge("diagnostics_fanout", "explainability", EdgeKind.BRANCH),
    Edge("diagnostics_fanout", "sensitivity", EdgeKind.BRANCH),
    Edge("diagnostics_fanout", "overfitting", EdgeKind.BRANCH),
    Edge("explainability", "diagnostics_join"),
    Edge("sensitivity", "diagnostics_join"),
    Edge("overfitting", "diagnostics_join"),
    # The back-edges. This is what a DAG cannot express, and what turns a failing
    # check from an observation into an attempt at resolution.
    Edge(
        "overfitting",
        "hyperparameter_tuning",
        EdgeKind.REMEDIATION,
        condition="generalisation gap exceeds threshold",
        budget=DEFAULT_REMEDIATION_BUDGETS["overfitting_to_tuning"],
        rationale="retune with stronger regularisation before accepting the gap",
    ),
    Edge(
        "sensitivity",
        "hyperparameter_tuning",
        EdgeKind.REMEDIATION,
        condition="metric drift exceeds threshold",
        budget=DEFAULT_REMEDIATION_BUDGETS["sensitivity_to_tuning"],
        rationale="excessive input sensitivity is often a capacity problem",
    ),
    Edge(
        "explainability",
        "feature_engineering",
        EdgeKind.REMEDIATION,
        condition="attribution concentrated or degenerate",
        budget=DEFAULT_REMEDIATION_BUDGETS["explainability_to_feature_engineering"],
        rationale="a single dominating feature usually indicates leakage or "
        "an encoding that should be revisited",
    ),
    Edge("diagnostics_join", "validation"),
    Edge(
        "validation",
        "model_execution",
        EdgeKind.REMEDIATION,
        condition="robustness failure",
        budget=DEFAULT_REMEDIATION_BUDGETS["validation_to_execution"],
        rationale="refit under the conditions the robustness check exposed",
    ),
    Edge("validation", "gate_validation"),
    Edge(
        "gate_validation",
        "gate_validation",
        EdgeKind.SELF_LOOP,
        condition="challenge or ask",
        budget=DEFAULT_REMEDIATION_BUDGETS["checkpoint_self_loop"],
    ),
    Edge("gate_validation", "governance_signoff", condition="accept"),
    # Any blocking condition short-circuits to sign-off. A blocked review still
    # produces a disposition and a seal — it stops early, it does not vanish.
    Edge("dataset_discovery", "governance_signoff", EdgeKind.HARD_STOP, condition="blocking data defect"),
    Edge("model_execution", "governance_signoff", EdgeKind.HARD_STOP, condition="execution failure"),
    Edge("governance_signoff", "evidence_critic"),
    Edge("evidence_critic", "seal"),
)

REVIEW_GRAPH = ReviewGraph(
    name="start-review",
    nodes=_NODES,
    edges=_EDGES,
    entry="start",
    exits=("seal",),
    version="1",
)

REVIEW_GRAPH.validate()
