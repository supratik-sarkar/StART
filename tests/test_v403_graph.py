"""v4.0.3 — bounded cyclic review graph.

The property under test throughout: a failing diagnostic becomes an *attempt at
resolution* rather than an observation, and both outcomes — resolved, or exhausted —
are stronger governance statements than a metric on its own.
"""

from __future__ import annotations

import pytest

from start.orchestration.cyclic_executor import (
    GraphExecutor,
    NodeOutcome,
    NodeResult,
)
from start.orchestration.review_graph_spec import (
    REVIEW_GRAPH,
    Edge,
    EdgeKind,
    GraphValidationError,
    Node,
    NodeKind,
    ReviewGraph,
)


def _handlers(**overrides):
    base = {n.id: (lambda nid, ctx: NodeResult()) for n in REVIEW_GRAPH.nodes}
    base.update(overrides)
    return base


CTX = {"run_dl": False, "accept": True}


# --------------------------------------------------------------------------- #
# Topology validation
# --------------------------------------------------------------------------- #
def test_shipped_graph_is_valid() -> None:
    REVIEW_GRAPH.validate()
    assert REVIEW_GRAPH.graph_hash()


def test_every_cycle_is_bounded() -> None:
    """An unbounded retry loop is a hyperparameter search, not a review."""
    for cycle in REVIEW_GRAPH._find_cycles():
        assert REVIEW_GRAPH._cycle_is_bounded(cycle), cycle


def test_remediation_edges_declare_a_budget() -> None:
    for edge in REVIEW_GRAPH.remediation_edges():
        assert edge.budget and edge.budget >= 1, edge.edge_id
        assert edge.rationale, f"{edge.edge_id} must say what it is for"


def test_unbudgeted_cycle_is_rejected() -> None:
    graph = ReviewGraph(
        name="bad",
        nodes=(
            Node("a", NodeKind.AGENT),
            Node("b", NodeKind.AGENT),
            Node("end", NodeKind.TERMINAL),
        ),
        edges=(
            Edge("a", "b"),
            Edge("b", "a", EdgeKind.REMEDIATION),  # no budget
            Edge("b", "end"),
        ),
        entry="a",
        exits=("end",),
    )
    with pytest.raises(GraphValidationError, match="budget"):
        graph.validate()


def test_unreachable_node_is_rejected() -> None:
    graph = ReviewGraph(
        name="bad",
        nodes=(Node("a", NodeKind.AGENT), Node("orphan", NodeKind.AGENT),
               Node("end", NodeKind.TERMINAL)),
        edges=(Edge("a", "end"),),
        entry="a",
        exits=("end",),
    )
    with pytest.raises(GraphValidationError, match="unreachable"):
        graph.validate()


def test_self_loop_must_connect_a_node_to_itself() -> None:
    graph = ReviewGraph(
        name="bad",
        nodes=(Node("a", NodeKind.AGENT), Node("end", NodeKind.TERMINAL)),
        edges=(Edge("a", "end", EdgeKind.SELF_LOOP, budget=2),),
        entry="a",
        exits=("end",),
    )
    with pytest.raises(GraphValidationError, match="self_loop"):
        graph.validate()


def test_graph_hash_changes_with_topology() -> None:
    """Two reviews that ran different graphs are different reviews."""
    modified = ReviewGraph(
        name=REVIEW_GRAPH.name,
        nodes=REVIEW_GRAPH.nodes,
        edges=REVIEW_GRAPH.edges[:-1],
        entry=REVIEW_GRAPH.entry,
        exits=REVIEW_GRAPH.exits,
    )
    assert modified.graph_hash() != REVIEW_GRAPH.graph_hash()


def test_schematic_renders_from_the_same_object_the_executor_walks() -> None:
    """The picture in the docs cannot drift from the behaviour in the code."""
    mermaid = REVIEW_GRAPH.to_mermaid()
    assert "flowchart TD" in mermaid
    for node in REVIEW_GRAPH.nodes:
        assert node.id in mermaid
    dot = REVIEW_GRAPH.to_dot()
    assert dot.startswith("digraph")
    assert "firebrick" in dot  # remediation edges are visually distinct


# --------------------------------------------------------------------------- #
# Execution — the capability a DAG cannot provide
# --------------------------------------------------------------------------- #
def test_all_three_diagnostics_run() -> None:
    """Regression: picking the first branch edge ran one diagnostic and skipped two,
    which looked exactly like success."""
    seen: list[str] = []
    handlers = _handlers(
        explainability=lambda nid, c: (seen.append(nid), NodeResult())[1],
        sensitivity=lambda nid, c: (seen.append(nid), NodeResult())[1],
        overfitting=lambda nid, c: (seen.append(nid), NodeResult())[1],
    )
    GraphExecutor(REVIEW_GRAPH, handlers).run(dict(CTX))
    assert set(seen) == {"explainability", "sensitivity", "overfitting"}


def test_failing_diagnostic_routes_back_and_resolves() -> None:
    state = {"tune": 0, "check": 0}

    def tuning(nid, ctx):
        state["tune"] += 1
        return NodeResult(fingerprint=f"lr={state['tune']}")

    def overfitting(nid, ctx):
        state["check"] += 1
        if state["check"] < 3:
            return NodeResult(outcome=NodeOutcome.FAIL, detail="gap 0.28 exceeds 0.10")
        return NodeResult(detail="gap 0.04")

    path = GraphExecutor(
        REVIEW_GRAPH, _handlers(hyperparameter_tuning=tuning, overfitting=overfitting)
    ).run(dict(CTX))

    assert state["tune"] == 3, "tuning must actually re-run on each remediation"
    summary = path.remediation_summary()
    assert summary["attempts"] == 2
    assert summary["resolved"] == 1
    assert summary["budget_exhausted"] == 0
    assert path.terminated_at == "seal"


def test_only_the_final_attempt_is_credited_as_resolved() -> None:
    """Marking every attempt resolved would read as three successful remediations
    when the truth is two failures and a success."""
    state = {"n": 0}

    def overfitting(nid, ctx):
        state["n"] += 1
        return NodeResult(outcome=NodeOutcome.FAIL) if state["n"] < 3 else NodeResult()

    path = GraphExecutor(REVIEW_GRAPH, _handlers(overfitting=overfitting)).run(dict(CTX))
    outcomes = [r.outcome for r in path.remediations]
    assert outcomes.count("resolved") == 1
    assert "still_failing" in outcomes


def test_exhausted_budget_produces_a_blocking_finding() -> None:
    """The statement a linear review cannot make."""
    handlers = _handlers(
        overfitting=lambda nid, c: NodeResult(outcome=NodeOutcome.FAIL, detail="gap 0.28"),
    )
    path = GraphExecutor(REVIEW_GRAPH, handlers).run(dict(CTX))

    assert path.remediation_summary()["budget_exhausted"] >= 1
    findings = [f for f in path.governance_findings() if f["kind"] == "remediation_exhausted"]
    assert findings
    assert findings[0]["severity"] == "blocker"
    assert "did not resolve" in findings[0]["detail"]


def test_review_still_reaches_the_seal_after_exhaustion() -> None:
    """A review that gives up on remediation must still produce a disposition."""
    handlers = _handlers(
        overfitting=lambda nid, c: NodeResult(outcome=NodeOutcome.FAIL, detail="gap"),
    )
    path = GraphExecutor(REVIEW_GRAPH, handlers).run(dict(CTX))
    assert path.terminated_at == "seal"


def test_budget_is_never_exceeded() -> None:
    handlers = _handlers(
        overfitting=lambda nid, c: NodeResult(outcome=NodeOutcome.FAIL),
    )
    path = GraphExecutor(REVIEW_GRAPH, handlers).run(dict(CTX))
    for edge_id, used in path.consumed.items():
        assert used <= path.budgets.get(edge_id, 0), edge_id


def test_budget_override_is_respected() -> None:
    """The bound is a policy value, not a constant."""
    handlers = _handlers(overfitting=lambda nid, c: NodeResult(outcome=NodeOutcome.FAIL))
    edge_id = "overfitting->hyperparameter_tuning:remediation"
    path = GraphExecutor(REVIEW_GRAPH, handlers, budgets={edge_id: 1}).run(dict(CTX))
    assert path.consumed.get(edge_id, 0) <= 1
    assert path.budgets[edge_id] == 1


def test_a_blocking_outcome_hard_stops_to_signoff() -> None:
    """A blocking defect must not be quietly retried."""
    handlers = _handlers(
        model_execution=lambda nid, c: NodeResult(outcome=NodeOutcome.BLOCK, detail="fit failed"),
    )
    path = GraphExecutor(REVIEW_GRAPH, handlers).run(dict(CTX))
    sequence = path.node_sequence
    assert "governance_signoff" in sequence
    assert "explainability" not in sequence, "diagnostics must not run after a hard stop"
    assert path.terminated_at == "seal"


def test_conditional_gate_is_skipped_when_its_condition_is_false() -> None:
    path = GraphExecutor(REVIEW_GRAPH, _handlers()).run({"run_dl": False, "accept": True})
    assert "gate_metric" not in path.node_sequence


def test_conditional_gate_runs_when_enabled() -> None:
    path = GraphExecutor(REVIEW_GRAPH, _handlers()).run({"run_dl": True, "accept": True})
    assert "gate_metric" in path.node_sequence


def test_a_handler_raising_does_not_lose_the_path() -> None:
    def explode(nid, ctx):
        raise RuntimeError("boom")

    path = GraphExecutor(REVIEW_GRAPH, _handlers(sensitivity=explode)).run(dict(CTX))
    assert path.visits
    failed = [v for v in path.visits if v.node_id == "sensitivity"]
    assert failed and failed[0].outcome == NodeOutcome.FAIL
    assert "RuntimeError" in failed[0].detail


def test_execution_never_runs_forever() -> None:
    handlers = _handlers(
        overfitting=lambda nid, c: NodeResult(outcome=NodeOutcome.FAIL),
        sensitivity=lambda nid, c: NodeResult(outcome=NodeOutcome.FAIL),
        explainability=lambda nid, c: NodeResult(outcome=NodeOutcome.FAIL),
        validation=lambda nid, c: NodeResult(outcome=NodeOutcome.FAIL),
    )
    path = GraphExecutor(REVIEW_GRAPH, handlers, max_visits=200).run(dict(CTX))
    assert len(path.visits) < 200


# --------------------------------------------------------------------------- #
# The path as evidence
# --------------------------------------------------------------------------- #
def test_path_hash_is_stable_for_identical_runs() -> None:
    a = GraphExecutor(REVIEW_GRAPH, _handlers()).run(dict(CTX))
    b = GraphExecutor(REVIEW_GRAPH, _handlers()).run(dict(CTX))
    assert a.path_hash() == b.path_hash()


def test_path_hash_differs_when_the_route_differs() -> None:
    """Two runs can agree on every metric and still have done different work."""
    clean = GraphExecutor(REVIEW_GRAPH, _handlers()).run(dict(CTX))
    remediated = GraphExecutor(
        REVIEW_GRAPH,
        _handlers(overfitting=lambda nid, c: NodeResult(outcome=NodeOutcome.FAIL)),
    ).run(dict(CTX))
    assert clean.path_hash() != remediated.path_hash()


def test_path_is_evidence_shaped() -> None:
    path = GraphExecutor(REVIEW_GRAPH, _handlers()).run(dict(CTX))
    block = path.as_dict()
    for key in ("graph_hash", "path", "remediations", "budgets", "consumed", "summary"):
        assert key in block


def test_successful_remediation_records_what_changed() -> None:
    state = {"n": 0}

    def tuning(nid, ctx):
        state["n"] += 1
        return NodeResult(fingerprint=f"dropout=0.{state['n']}")

    def overfitting(nid, ctx):
        return NodeResult() if state["n"] >= 2 else NodeResult(outcome=NodeOutcome.FAIL)

    path = GraphExecutor(
        REVIEW_GRAPH, _handlers(hyperparameter_tuning=tuning, overfitting=overfitting)
    ).run(dict(CTX))
    resolved = [r for r in path.remediations if r.outcome == "resolved"]
    assert resolved
    assert resolved[0].changed and resolved[0].changed != "nothing"
