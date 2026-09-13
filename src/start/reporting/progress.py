"""Progress reporting and agentic action logging for the reviewer assistant.

Two cross-platform primitives (pure Python, no OS-specific calls):

  * ``ProgressReporter`` renders horizontal bars + percentages for the visible
    phases of a review, in the terminal and (as a table) in notebooks.
  * ``ActionLog`` records what each agent did — input reviewed, action taken,
    recommendation, evidence used, user decision, output artifact — and renders
    it to terminal/markdown/dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Canonical review phases for the progress bar.
PROGRESS_PHASES = (
    "Data loading",
    "Discovery",
    "Feature engineering",
    "Preprocessing",
    "Model fitting",
    "Tuning",
    "Evaluation",
    "Explainability",
    "Sensitivity",
    "Governance",
    "Reporting",
)


def render_bar(pct: float, width: int = 16) -> str:
    """ASCII progress bar. Uses block + dash characters (render everywhere)."""
    pct = max(0.0, min(100.0, pct))
    filled = int(round(width * pct / 100.0))
    return "[" + "#" * filled + "-" * (width - filled) + f"] {pct:5.1f}%"


@dataclass
class ProgressReporter:
    phases: tuple[str, ...] = PROGRESS_PHASES
    width: int = 16
    quiet: bool = False
    _current: int = 0
    history: list[tuple[str, float]] = field(default_factory=list)

    def update(self, phase: str, pct: float) -> str:
        line = f"  {phase:22s} {render_bar(pct, self.width)}"
        self.history.append((phase, pct))
        if not self.quiet:
            print(line)
        return line

    def advance(self, phase: str) -> str:
        """Mark a phase complete and report cumulative progress through phases."""
        if phase in self.phases:
            self._current = self.phases.index(phase) + 1
        pct = 100.0 * self._current / len(self.phases)
        return self.update(phase, pct)

    def table_rows(self) -> list[dict[str, Any]]:
        return [{"phase": p, "percent": pct} for p, pct in self.history]


@dataclass
class AgentAction:
    agent: str
    input_reviewed: str
    action: str
    recommendation: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    user_decision: str | None = None
    output_artifact: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "input_reviewed": self.input_reviewed,
            "action": self.action,
            "recommendation": self.recommendation,
            "evidence_ids": self.evidence_ids,
            "user_decision": self.user_decision,
            "output_artifact": self.output_artifact,
        }


@dataclass
class ActionLog:
    actions: list[AgentAction] = field(default_factory=list)

    def record(
        self,
        agent: str,
        input_reviewed: str,
        action: str,
        *,
        recommendation: str = "",
        evidence_ids: list[str] | None = None,
        user_decision: str | None = None,
        output_artifact: str | None = None,
    ) -> AgentAction:
        entry = AgentAction(
            agent=agent, input_reviewed=input_reviewed, action=action,
            recommendation=recommendation, evidence_ids=evidence_ids or [],
            user_decision=user_decision, output_artifact=output_artifact,
        )
        self.actions.append(entry)
        return entry

    def to_list(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.actions]

    def agents(self) -> list[str]:
        return [a.agent for a in self.actions]


def render_action_log_markdown(log: ActionLog) -> str:
    if not log.actions:
        return "### Agentic action log\n\n_No agent actions recorded._\n"
    lines = [
        "### Agentic action log",
        "",
        "| Agent | Input reviewed | Action | Recommendation | Evidence | User decision |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for a in log.actions:
        lines.append(
            f"| {a.agent} | {a.input_reviewed} | {a.action} | {a.recommendation or '—'} "
            f"| {', '.join(a.evidence_ids) or '—'} | {a.user_decision or '—'} |"
        )
    return "\n".join(lines) + "\n"
