"""Agent roster + adapter activity announcer (v2.2.0 items 1 & 9).

Provides the canonical list of review agents with one-line purposes for the
startup display, and a small helper to render adapter activity announcements so
the user can see the AI-engineering ecosystem working in real time.

Pure presentation/metadata — no model logic here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentRole:
    name: str
    purpose: str


# The committee. Order = the order the user meets them in a review.
AGENT_ROSTER: tuple[AgentRole, ...] = (
    AgentRole("DatasetDiscoveryAgent", "Dataset understanding"),
    AgentRole("TaskInferenceAgent", "Task framing & inference"),
    AgentRole("FeatureEngineeringAgent", "Data preparation & checks"),
    AgentRole("ArchitectureReviewAgent", "Model selection & verification"),
    AgentRole("HyperparameterTuningAgent", "Optimization strategy"),
    AgentRole("ModelExecutionAgent", "Training execution & telemetry"),
    AgentRole("ExplainabilityAgent", "Model interpretability & attributions"),
    AgentRole("SensitivityAgent", "Metric shock response"),
    AgentRole("OverfittingAgent", "Generalization gap diagnosis"),
    AgentRole("ValidationAgent", "Adversarial & robustness checks"),
    AgentRole("GovernanceSignoffAgent", "MRM signoff decision"),
    AgentRole("EvidenceCriticAgent", "Evidence & citation integrity"),
)


def render_agent_roster() -> str:
    """Startup display: each agent and its purpose (item 1)."""
    lines = ["Review committee — your AI reviewers:", ""]
    for role in AGENT_ROSTER:
        lines.append(f"  {role.name}")
        lines.append(f"      Purpose: {role.purpose}")
    return "\n".join(lines)


def roster_as_list() -> list[dict[str, str]]:
    """Roster for the dashboard/notebook (item 1 & 8)."""
    return [{"agent": r.name, "purpose": r.purpose} for r in AGENT_ROSTER]


# Stable terminal color per agent (v2.3.1 #3) — colored, never plain white.
AGENT_COLORS: dict[str, str] = {
    "DatasetDiscoveryAgent": "bright_cyan",
    "TaskInferenceAgent": "orange1",
    "FeatureEngineeringAgent": "bright_green",
    "ArchitectureReviewAgent": "bright_blue",
    "HyperparameterTuningAgent": "bright_magenta",
    "ModelExecutionAgent": "bright_yellow",
    "ExplainabilityAgent": "spring_green1",
    "SensitivityAgent": "purple",
    "OverfittingAgent": "salmon",
    "ValidationAgent": "cyan",
    "GovernanceSignoffAgent": "green",
    "EvidenceCriticAgent": "magenta",
}


def agent_color(name: str) -> str:
    from start.cli.view import AGENT_COLOR_REGISTRY

    return AGENT_COLOR_REGISTRY.get(name, "white")


def render_agent_roster_panel() -> Any:
    """Rich panel: committee roster with colored agent names (v2.3.1 #3)."""
    from rich.panel import Panel
    from rich.table import Table

    from start.cli.view import get_styled_agent_name

    grid = Table.grid(padding=(0, 2))
    grid.add_column(no_wrap=True)
    grid.add_column()
    for r in AGENT_ROSTER:
        grid.add_row(get_styled_agent_name(r.name), r.purpose)
    return Panel(
        grid,
        title="[bold]Review committee — your AI reviewers[/bold]",
        border_style="cyan",
        title_align="left",
    )


def render_adapter_panel(control_surface: list[dict]) -> str:
    """Startup AI-engineering environment panel with check/cross (item 9)."""
    lines = ["AI Engineering Environment:", ""]
    for row in control_surface:
        available = row.get("status") in ("complete", "available")
        mark = "[+]" if available else "[-]"
        lines.append(f"  {mark} {row.get('adapter')}")
        lines.append(f"        purpose : {row.get('purpose', '')}")
        lines.append(f"        status  : {row.get('status', 'unknown')}")
    return "\n".join(lines)


def announce_adapter_activity(adapter: str, action: str) -> str:
    """One-line activity announcement during execution (item 9).

    Example: announce_adapter_activity("DeepEval", "Running hallucination evaluation")
    -> "[DeepEval] Running hallucination evaluation…"
    """
    return f"[{adapter}] {action}…"
