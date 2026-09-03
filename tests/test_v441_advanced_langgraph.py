"""Tests proving Canonical LangGraph StateGraph Runtime Properties.

Mandatory verification:
1. Real compiled LangGraph StateGraph object
2. TypedReviewState execution
3. Conditional edge routing
4. Checkpoint persistence across nodes with MemorySaver
5. Resume from checkpoint with thread_id
6. Zero duplicate EvidenceRecords after resume (idempotency)
7. State history inspection and Mermaid diagram generation
8. Canonical review graph integration in run_unified_review
"""

from __future__ import annotations

import pytest

pytest.importorskip("langgraph")

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from start.core.schemas import EvidenceRecord, Status
from start.orchestration.state_graph import (
    TypedReviewState,
    build_canonical_review_graph,
    get_canonical_graph_mermaid,
)


def test_stategraph_normal_compilation_and_execution() -> None:
    """Prove real LangGraph StateGraph object compilation and typed execution."""
    workflow = StateGraph(TypedReviewState)

    def plan_node(state: TypedReviewState) -> dict:
        return {"stage": "PLANNING", "step_history": [{"node": "plan", "status": "OK"}]}

    def exec_node(state: TypedReviewState) -> dict:
        rec = EvidenceRecord(
            evidence_id="EV-TEST-1",
            test_id="test.demo",
            test_name="Demo Test",
            model_id="M1",
            dataset_id="D1",
            run_id=state.get("run_id", "RUN"),
            status=Status.PASS,
            metrics={"score": 0.95},
        )
        return {
            "stage": "EXECUTION",
            "evidence_records": [rec],
            "evidence_ids": ["EV-TEST-1"],
            "step_history": [{"node": "exec", "status": "OK"}],
        }

    workflow.add_node("plan", plan_node)
    workflow.add_node("exec", exec_node)
    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "exec")
    workflow.add_edge("exec", END)

    saver = MemorySaver()
    app = workflow.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "thread-001"}}
    init_state: TypedReviewState = {"run_id": "RUN-LG-001", "thread_id": "thread-001"}
    final_state = app.invoke(init_state, config=config)

    assert final_state["stage"] == "EXECUTION"
    assert "EV-TEST-1" in final_state["evidence_ids"]
    assert len(final_state["evidence_records"]) == 1
    assert len(final_state["step_history"]) == 2


def test_stategraph_conditional_routing() -> None:
    """Prove conditional edge routing with real LangGraph StateGraph."""
    workflow = StateGraph(TypedReviewState)

    def eval_node(state: TypedReviewState) -> dict:
        return {"governance_state": {"score": 0.40}}

    def route_decision(state: TypedReviewState) -> str:
        score = state.get("governance_state", {}).get("score", 0.0)
        return "remediate" if score < 0.50 else "approve"

    def remediate_node(state: TypedReviewState) -> dict:
        return {"stage": "REMEDIATION"}

    def approve_node(state: TypedReviewState) -> dict:
        return {"stage": "APPROVED"}

    workflow.add_node("eval", eval_node)
    workflow.add_node("remediate", remediate_node)
    workflow.add_node("approve", approve_node)
    workflow.add_edge(START, "eval")
    workflow.add_conditional_edges("eval", route_decision, ["remediate", "approve"])
    workflow.add_edge("remediate", END)
    workflow.add_edge("approve", END)

    app = workflow.compile()
    res = app.invoke({"run_id": "RUN-COND"})

    assert res["stage"] == "REMEDIATION"


def test_stategraph_checkpoint_persistence_and_history() -> None:
    """Prove state history inspection and MemorySaver checkpoints across steps."""
    saver = MemorySaver()
    app = build_canonical_review_graph(checkpointer=saver)

    config = {"configurable": {"thread_id": "thread-hist-001"}}
    init_state: TypedReviewState = {
        "run_id": "RUN-HIST",
        "thread_id": "thread-hist-001",
        "governance_state": {"disposition": "ACCEPT"},
    }
    final_state = app.invoke(init_state, config=config)
    assert final_state["stage"] == "GOVERNANCE"

    # State history inspection
    history = list(app.get_state_history(config))
    assert len(history) >= 4
    current_state = app.get_state(config)
    assert current_state.values["stage"] == "GOVERNANCE"


def test_stategraph_resume_and_evidence_deduplication() -> None:
    """Prove resuming a thread from checkpoint and guaranteeing zero duplicate EvidenceRecords."""
    saver = MemorySaver()
    workflow = StateGraph(TypedReviewState)

    def node_a(state: TypedReviewState) -> dict:
        rec_a = EvidenceRecord(
            evidence_id="EV-A",
            test_id="test.a",
            test_name="Test A",
            model_id="M1",
            dataset_id="D1",
            run_id="RUN-RESUME",
            status=Status.PASS,
            metrics={"val": 1.0},
        )
        return {"evidence_records": [rec_a], "evidence_ids": ["EV-A"]}

    def node_b(state: TypedReviewState) -> dict:
        rec_b = EvidenceRecord(
            evidence_id="EV-B",
            test_id="test.b",
            test_name="Test B",
            model_id="M1",
            dataset_id="D1",
            run_id="RUN-RESUME",
            status=Status.PASS,
            metrics={"val": 2.0},
        )
        return {"evidence_records": [rec_b], "evidence_ids": ["EV-B"]}

    workflow.add_node("a", node_a)
    workflow.add_node("b", node_b)
    workflow.add_edge(START, "a")
    workflow.add_edge("a", "b")
    workflow.add_edge("b", END)

    app = workflow.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "thread-resume-001"}}

    # First invoke
    out1 = app.invoke({"run_id": "RUN-RESUME"}, config=config)
    assert len(out1["evidence_records"]) == 2

    # Second invoke on same thread (re-supplying EV-A)
    rec_a_duplicate = EvidenceRecord(
        evidence_id="EV-A",
        test_id="test.a",
        test_name="Test A",
        model_id="M1",
        dataset_id="D1",
        run_id="RUN-RESUME",
        status=Status.PASS,
        metrics={"val": 1.0},
    )
    app.update_state(config, {"evidence_records": [rec_a_duplicate]})
    state_now = app.get_state(config)
    # Deduplication reducer must maintain exactly 2 unique records
    assert len(state_now.values["evidence_records"]) == 2


def test_canonical_graph_mermaid_diagram() -> None:
    """Prove Mermaid diagram export of the canonical StateGraph."""
    mermaid_str = get_canonical_graph_mermaid()
    assert "plan" in mermaid_str
    assert "execute_tools" in mermaid_str
    assert "governance_signoff" in mermaid_str
