"""Agent roster + adapter activity announcer (v2.2.0 items 1 & 9).

Provides the canonical list of review agents with one-line purposes for the
startup display, and a small helper to render adapter activity announcements so
the user can see the AI-engineering ecosystem working in real time.

Pure presentation/metadata — no model logic here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentRole:
    name: str
    purpose: str


# The committee. Order = the order the user meets them in a review.
AGENT_ROSTER: tuple[AgentRole, ...] = (
    AgentRole("DatasetDiscoveryAgent", "Dataset understanding"),
    AgentRole("FeatureEngineeringAgent", "Data preparation"),
    AgentRole("ArchitectureReviewAgent", "Model selection"),
    AgentRole("HyperparameterTuningAgent", "Optimization strategy"),
    AgentRole("ModelExecutionAgent", "Training and evaluation"),
    AgentRole("ValidationAgent", "Sensitivity and robustness"),
    AgentRole("GovernanceSignoffAgent", "MRM signoff"),
    AgentRole("EvidenceCriticAgent", "Citation and evidence enforcement"),
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
