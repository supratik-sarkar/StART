from __future__ import annotations

import json

import pytest

from start.modeling.graph_orchestrator import (
    GraphReviewOrchestrator,
    GraphRunState,
    langgraph_available,
)


def _linear_orch(tmp_path):
    orch = GraphReviewOrchestrator(output_root=str(tmp_path))
    orch.add_node("data", lambda s: {"rows": 100})
    orch.add_node("model", lambda s: {"trained": True}, depends_on=["data"])
    orch.add_node("validation", lambda s: {"auc": 0.8}, depends_on=["model"])
    orch.add_node("governance", lambda s: {"findings": 2}, depends_on=["validation"])
    orch.add_node("report", lambda s: {"report": "done"}, depends_on=["governance"])
    return orch


def test_engine_selection():
    orch = GraphReviewOrchestrator()
    assert orch.engine == ("langgraph" if langgraph_available() else "builtin_dag")


def test_dag_executes_in_dependency_order(tmp_path):
    orch = _linear_orch(tmp_path)
    seen = []
    orch.on_node = lambda n: seen.append((n.name, n.status))
    state = orch.run("RUN-1", initial_state={"dataset": "demo"})
    assert state.completed == ["data", "model", "validation", "governance", "report"]
    assert state.state["rows"] == 100 and state.state["auc"] == 0.8
    assert state.state["dataset"] == "demo"  # initial state preserved
    # each node emitted a running then complete event
    completes = [name for name, status in seen if status == "complete"]
    assert completes == ["data", "model", "validation", "governance", "report"]


def test_diamond_dependencies(tmp_path):
    orch = GraphReviewOrchestrator(output_root=str(tmp_path))
    orch.add_node("root", lambda s: {"r": 1})
    orch.add_node("left", lambda s: {"l": s["r"] + 1}, depends_on=["root"])
    orch.add_node("right", lambda s: {"rt": s["r"] + 2}, depends_on=["root"])
    orch.add_node("merge", lambda s: {"m": s["l"] + s["rt"]}, depends_on=["left", "right"])
    state = orch.run("RUN-DIAMOND")
    assert state.state["m"] == 2 + 3
    assert state.completed.index("root") < state.completed.index("merge")


def test_cycle_detection(tmp_path):
    orch = GraphReviewOrchestrator(output_root=str(tmp_path))
    orch.add_node("a", lambda s: {}, depends_on=["b"])
    orch.add_node("b", lambda s: {}, depends_on=["a"])
    with pytest.raises(ValueError, match="Cycle detected"):
        orch.run("RUN-CYCLE")


def test_unknown_dependency(tmp_path):
    orch = GraphReviewOrchestrator(output_root=str(tmp_path))
    orch.add_node("a", lambda s: {}, depends_on=["ghost"])
    with pytest.raises(ValueError, match="unknown node"):
        orch.run("RUN-GHOST")


def test_checkpoint_and_resume(tmp_path):
    calls = {"n": 0}

    def flaky(s):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return {"auc": 0.8}

    orch = GraphReviewOrchestrator(output_root=str(tmp_path))
    orch.add_node("data", lambda s: {"rows": 10})
    orch.add_node("validation", flaky, depends_on=["data"])
    state = GraphRunState(run_id="RUN-R")
    with pytest.raises(RuntimeError):
        orch.run("RUN-R", resume_from=state)
    assert state.completed == ["data"]  # checkpoint holds completed work

    resumed = orch.run("RUN-R", resume_from=state)
    assert resumed.completed == ["data", "validation"]
    assert calls["n"] == 2  # data not re-run; validation retried
    assert resumed.state["auc"] == 0.8


def test_graph_artifacts_written(tmp_path):
    orch = _linear_orch(tmp_path)
    state = orch.run("RUN-ART")
    paths = orch.write_graph_artifacts("RUN-ART", state)
    names = [p.split("/")[-1] for p in paths]
    assert "review_graph.json" in names
    # visualization present (png if matplotlib, else dot+mmd)
    assert any(n.endswith((".png", ".dot", ".mmd")) for n in names)
    graph = json.loads((tmp_path / "ai_engineering" / "RUN-ART" / "review_graph.json").read_text())
    assert len(graph["nodes"]) == 5
    assert len(graph["edges"]) == 4
    assert all(n["status"] == "complete" for n in graph["nodes"])
    assert graph["engine"] in {"langgraph", "builtin_dag"}


def test_checkpoint_serialization(tmp_path):
    orch = _linear_orch(tmp_path)
    state = orch.run("RUN-CK")
    ck = state.checkpoint()
    assert ck["run_id"] == "RUN-CK"
    assert set(ck["completed"]) == {"data", "model", "validation", "governance", "report"}
    assert ck["node_status"]["report"] == "complete"
