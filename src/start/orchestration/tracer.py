"""Real Event-Driven Agent Orchestration Tracer for StART.

Strict Invariants:
1. No Fictional Transitions: Only instruments and renders nodes/agents/tools that actually execute.
2. Privacy Preserving: Never captures or displays API keys, raw datasets, or private reasoning.
3. Multi-Format Rendering:
   - Rich Terminal Orchestration Trace Table
   - Machine-readable agent_orchestration.json
   - Standard Mermaid Flowchart (agent_orchestration.mmd)
   - Visual SVG Schematic Diagram (agent_orchestration.svg)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.table import Table


@dataclass
class TraceEvent:
    """An atomic, auditable transition event in the agent orchestration graph."""

    step_index: int
    timestamp: float = field(default_factory=time.time)
    source_agent: str = "Director"
    target_agent: str = "Specialist"
    stage: str = "INITIALIZATION"
    node: str = "start"
    tool_name: str | None = None
    tool_allowlist_status: str = "ALLOWED"
    source_evidence_ids: list[str] = field(default_factory=list)
    emitted_evidence_ids: list[str] = field(default_factory=list)
    emitted_artifact_ids: list[str] = field(default_factory=list)
    policy_guardrail_decision: str = "PASS"
    provider: str = "DETERMINISTIC"
    model: str = "DETERMINISTIC_ENGINE"
    latency_ms: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    status: str = "SUCCESS"
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AgentExecutionTracer:
    """Collector and visualizer for real agent orchestration events."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    def record(
        self,
        source_agent: str,
        target_agent: str,
        stage: str,
        node: str,
        tool_name: str | None = None,
        tool_allowlist_status: str = "ALLOWED",
        source_evidence_ids: list[str] | None = None,
        emitted_evidence_ids: list[str] | None = None,
        emitted_artifact_ids: list[str] | None = None,
        policy_guardrail_decision: str = "PASS",
        provider: str = "DETERMINISTIC",
        model: str = "DETERMINISTIC_ENGINE",
        latency_ms: float = 0.0,
        token_usage: dict[str, int] | None = None,
        status: str = "SUCCESS",
        detail: str = "",
    ) -> TraceEvent:
        """Record an actual runtime transition event."""
        event = TraceEvent(
            step_index=len(self.events) + 1,
            timestamp=time.time(),
            source_agent=source_agent,
            target_agent=target_agent,
            stage=stage,
            node=node,
            tool_name=tool_name,
            tool_allowlist_status=tool_allowlist_status,
            source_evidence_ids=source_evidence_ids or [],
            emitted_evidence_ids=emitted_evidence_ids or [],
            emitted_artifact_ids=emitted_artifact_ids or [],
            policy_guardrail_decision=policy_guardrail_decision,
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            token_usage=token_usage or {},
            status=status,
            detail=detail,
        )
        self.events.append(event)
        return event

    def build_rich_table(self, title: str = "Agent Orchestration & Tool Execution Trace") -> Table:
        """Render a clean, color-coded Rich table of real orchestration events."""
        table = Table(
            title=title,
            title_style="bold cyan",
            header_style="bold",
            show_lines=False,
        )
        table.add_column("Step", justify="right", style="dim", no_wrap=True)
        table.add_column("Transition (Source → Target)", style="bold white")
        table.add_column("Stage / Node", style="cyan")
        table.add_column("Tool / Action", style="bold yellow")
        table.add_column("Evidence & Artifact Flow", style="dim")
        table.add_column("Guardrail / Policy", justify="center")
        table.add_column("Latency / Status", justify="right")

        for e in self.events:
            trans = f"{e.source_agent} → {e.target_agent}"
            tool_str = e.tool_name or e.node
            if e.tool_allowlist_status == "ALLOWED" and e.tool_name:
                tool_str = f"🔧 {tool_str}"

            ev_art_flow = []
            if e.emitted_evidence_ids:
                ev_art_flow.append(f"+{len(e.emitted_evidence_ids)} EV")
            if e.emitted_artifact_ids:
                ev_art_flow.append(f"+{len(e.emitted_artifact_ids)} ART")
            flow_str = " | ".join(ev_art_flow) if ev_art_flow else "—"

            guard_style = "green" if e.policy_guardrail_decision == "PASS" else "yellow"
            guard_str = f"[{guard_style}]{e.policy_guardrail_decision}[/{guard_style}]"

            st_style = "green" if e.status == "SUCCESS" else "red"
            lat_str = f"{e.latency_ms:.0f}ms [{st_style}]{e.status}[/{st_style}]" if e.latency_ms > 0 else f"[{st_style}]{e.status}[/{st_style}]"

            table.add_row(
                str(e.step_index),
                trans,
                f"{e.stage} / {e.node}",
                tool_str,
                flow_str,
                guard_str,
                lat_str,
            )

        return table

    def export_json(self, output_path: Path | str) -> str:
        """Export raw event log to JSON."""
        data = {
            "total_events": len(self.events),
            "events": [e.to_dict() for e in self.events],
        }
        text = json.dumps(data, indent=2)
        Path(output_path).write_text(text, encoding="utf-8")
        return text

    def export_mermaid(self, output_path: Path | str) -> str:
        """Export real execution flow as Mermaid flowchart."""
        lines = ["flowchart TD", "    classDef agent fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;"]
        lines.append("    classDef tool fill:#0f172a,stroke:#f59e0b,stroke-width:1px,color:#f8fafc;")
        lines.append("    classDef gov fill:#1e1b4b,stroke:#a855f7,stroke-width:2px,color:#f8fafc;")

        # Track unique nodes and transitions
        nodes_seen: set[str] = set()
        for e in self.events:
            src = e.source_agent.replace(" ", "_")
            tgt = e.target_agent.replace(" ", "_")
            if src not in nodes_seen:
                lines.append(f'    {src}["{e.source_agent}"]:::agent')
                nodes_seen.add(src)
            if tgt not in nodes_seen:
                cls_name = "gov" if "Governance" in tgt or "Committee" in tgt else "agent"
                lines.append(f'    {tgt}["{e.target_agent}"]:::{cls_name}')
                nodes_seen.add(tgt)

            lbl = e.tool_name or e.node
            lines.append(f"    {src} -->|{lbl}| {tgt}")

        content = "\n".join(lines)
        Path(output_path).write_text(content, encoding="utf-8")
        return content

    def export_svg(self, output_path: Path | str) -> str:
        """Export a clean, standalone SVG diagram of the real execution trace."""
        w = 900
        h = max(400, 60 + len(self.events) * 45)
        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" height="100%" '
            'style="background:#090d16; font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;">',
            '  <defs>',
            '    <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">',
            '      <stop offset="0%" stop-color="#38bdf8" />',
            '      <stop offset="100%" stop-color="#818cf8" />',
            '    </linearGradient>',
            '  </defs>',
            f'  <text x="30" y="35" fill="#f8fafc" font-size="18" font-weight="bold">StART Agent Orchestration Trace ({len(self.events)} Transitions)</text>',
            f'  <line x1="30" y1="50" x2="{w - 30}" y2="50" stroke="#334155" stroke-width="1"/>',
        ]

        y = 80
        for e in self.events:
            # Step badge
            svg_lines.append(f'  <rect x="30" y="{y - 15}" width="36" height="24" rx="4" fill="#1e293b" stroke="#475569" stroke-width="1"/>')
            svg_lines.append(f'  <text x="48" y="{y + 2}" fill="#94a3b8" font-size="11" font-weight="bold" text-anchor="middle">{e.step_index}</text>')

            # Transition text
            svg_lines.append(f'  <text x="80" y="{y + 2}" fill="#38bdf8" font-size="13" font-weight="bold">{e.source_agent} → {e.target_agent}</text>')

            # Node / Tool badge
            tool_txt = e.tool_name or e.node
            svg_lines.append(f'  <text x="320" y="{y + 2}" fill="#fbbf24" font-size="12">{tool_txt}</text>')

            # Flow summary
            ev_summary = f"+{len(e.emitted_evidence_ids)} EV" if e.emitted_evidence_ids else ""
            if e.emitted_artifact_ids:
                ev_summary += f" +{len(e.emitted_artifact_ids)} ART"
            svg_lines.append(f'  <text x="580" y="{y + 2}" fill="#94a3b8" font-size="11">{ev_summary}</text>')

            # Status pill
            status_color = "#4ade80" if e.status == "SUCCESS" else "#f87171"
            svg_lines.append(f'  <rect x="{w - 110}" y="{y - 14}" width="80" height="22" rx="11" fill="#1e293b" stroke="{status_color}" stroke-width="1"/>')
            svg_lines.append(f'  <text x="{w - 70}" y="{y + 1}" fill="{status_color}" font-size="10" font-weight="bold" text-anchor="middle">{e.status}</text>')

            y += 42

        svg_lines.append('</svg>')
        content = "\n".join(svg_lines)
        Path(output_path).write_text(content, encoding="utf-8")
        return content
