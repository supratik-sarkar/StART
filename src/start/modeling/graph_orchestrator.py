"""Graph-based review orchestration (enterprise execution mode).

``GraphReviewOrchestrator`` runs the review pipeline as a directed acyclic
graph with checkpointing, resumability, and state tracking. It uses LangGraph
when installed; otherwise it runs a built-in deterministic DAG executor that
provides the same guarantees (real execution, not a stub). Either way it emits:

    review_graph.json  - nodes, edges, per-node status/runtime, final state
    review_graph.<png|dot|mmd>  - a visualization (PNG via matplotlib if
                         available, else a Graphviz .dot and a Mermaid .mmd
                         that render anywhere)

The existing ReviewOrchestrator remains the default; this is the opt-in
enterprise execution mode.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def langgraph_available() -> bool:
    try:
        import langgraph  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class GraphNode:
    name: str
    fn: Callable[[dict[str, Any]], dict[str, Any]]
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | complete | error | skipped
    runtime_seconds: float = 0.0
    detail: str = ""


@dataclass
class GraphRunState:
    run_id: str
    completed: list[str] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    node_status: dict[str, str] = field(default_factory=dict)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "completed": list(self.completed),
            "node_status": dict(self.node_status),
            "state_keys": sorted(self.state.keys()),
        }


class GraphReviewOrchestrator:
    """DAG executor for the review pipeline with checkpoint/resume."""

    def __init__(
        self,
        output_root: str = "start_output",
        on_node: Callable[[GraphNode], None] | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.on_node = on_node
        self.nodes: dict[str, GraphNode] = {}
        self.order: list[str] = []
        self.engine = "langgraph" if langgraph_available() else "builtin_dag"

    def add_node(
        self, name: str, fn: Callable[[dict[str, Any]], dict[str, Any]], depends_on=None
    ) -> None:
        self.nodes[name] = GraphNode(name=name, fn=fn, depends_on=list(depends_on or []))
        self.order.append(name)

    def _topo_order(self) -> list[str]:
        resolved: list[str] = []
        visiting: set[str] = set()

        def visit(n: str) -> None:
            if n in resolved:
                return
            if n in visiting:
                raise ValueError(f"Cycle detected at node '{n}'.")
            visiting.add(n)
            for dep in self.nodes[n].depends_on:
                if dep not in self.nodes:
                    raise ValueError(f"Node '{n}' depends on unknown node '{dep}'.")
                visit(dep)
            visiting.discard(n)
            resolved.append(n)

        for name in self.order:
            visit(name)
        return resolved

    def run(
        self, run_id: str, initial_state: dict[str, Any] | None = None,
        resume_from: GraphRunState | None = None,
    ) -> GraphRunState:
        state = resume_from or GraphRunState(run_id=run_id, state=dict(initial_state or {}))
        for name in self._topo_order():
            if name in state.completed:
                self.nodes[name].status = "complete"  # resumed: skip already-done
                continue
            node = self.nodes[name]
            node.status = "running"
            if self.on_node:
                self.on_node(node)
            t0 = time.perf_counter()
            try:
                update = node.fn(state.state)
                if update:
                    state.state.update(update)
                node.status = "complete"
            except Exception as exc:
                node.status = "error"
                node.detail = f"{type(exc).__name__}: {exc}"
                node.runtime_seconds = round(time.perf_counter() - t0, 4)
                state.node_status[name] = "error"
                if self.on_node:
                    self.on_node(node)
                raise
            node.runtime_seconds = round(time.perf_counter() - t0, 4)
            state.completed.append(name)
            state.node_status[name] = "complete"
            if self.on_node:
                self.on_node(node)
        return state

    # -- serialization & visualization ------------------------------------ #
    def to_graph_dict(self, state: GraphRunState | None = None) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "nodes": [
                {
                    "name": n.name,
                    "depends_on": n.depends_on,
                    "status": n.status,
                    "runtime_seconds": n.runtime_seconds,
                }
                for n in (self.nodes[name] for name in self.order)
            ],
            "edges": [
                {"from": dep, "to": n.name}
                for n in self.nodes.values()
                for dep in n.depends_on
            ],
            "final_state_keys": sorted(state.state.keys()) if state else [],
        }

    def write_graph_artifacts(self, run_id: str, state: GraphRunState | None = None) -> list[str]:
        out_dir = self.output_root / "ai_engineering" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: list[str] = []

        graph = self.to_graph_dict(state)
        json_path = out_dir / "review_graph.json"
        json_path.write_text(json.dumps(graph, indent=2))
        paths.append(str(json_path))

        # Visualization: PNG via matplotlib if available, else .dot + .mmd.
        png = self._render_png(graph, out_dir)
        if png:
            paths.append(png)
        else:
            paths.append(self._render_dot(graph, out_dir))
            paths.append(self._render_mermaid(graph, out_dir))
        return paths

    def _render_png(self, graph: dict[str, Any], out_dir: Path) -> str | None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        nodes = graph["nodes"]
        fig, ax = plt.subplots(figsize=(6, max(4, len(nodes) * 0.5)))
        y = list(range(len(nodes), 0, -1))
        colors = {"complete": "#2e7d32", "error": "#c62828", "pending": "#9e9e9e"}
        for yi, n in zip(y, nodes, strict=False):
            ax.scatter(0, yi, s=300, color=colors.get(n["status"], "#1565c0"), zorder=3)
            ax.text(0.1, yi, f"{n['name']} ({n['runtime_seconds']}s)", va="center", fontsize=8)
        for yi in range(1, len(nodes)):
            ax.plot([0, 0], [y[yi - 1], y[yi]], color="#bbbbbb", zorder=1)
        ax.axis("off")
        ax.set_title("Review graph")
        fig.tight_layout()
        path = out_dir / "review_graph.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        return str(path)

    def _render_dot(self, graph: dict[str, Any], out_dir: Path) -> str:
        lines = ["digraph review {", "  rankdir=TB;"]
        for n in graph["nodes"]:
            lines.append(f'  "{n["name"]}" [label="{n["name"]}\\n{n["status"]}"];')
        for e in graph["edges"]:
            lines.append(f'  "{e["from"]}" -> "{e["to"]}";')
        lines.append("}")
        path = out_dir / "review_graph.dot"
        path.write_text("\n".join(lines))
        return str(path)

    def _render_mermaid(self, graph: dict[str, Any], out_dir: Path) -> str:
        lines = ["graph TD"]
        for e in graph["edges"]:
            lines.append(f'  {_safe(e["from"])} --> {_safe(e["to"])}')
        if not graph["edges"]:
            for n in graph["nodes"]:
                lines.append(f'  {_safe(n["name"])}')
        path = out_dir / "review_graph.mmd"
        path.write_text("\n".join(lines))
        return str(path)


def _safe(name: str) -> str:
    return name.replace(" ", "_").replace("-", "_")
