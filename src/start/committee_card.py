"""Committee review cards (v2.3.0 #1, #4, #9).

Every agent interaction renders as a committee review card following the
mandatory evidence-first structure:

    Evidence -> Recommendation -> Alternatives -> Risks -> Questions -> Decision

For the terminal, cards render as Rich panels and tables (no markdown tables —
#9). For the dashboard, transcript, and notebook, the same card serializes to a
dict. If a recommendation has no supporting evidence, the card states
"The recommendation is not evidence-backed." explicitly (#1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CommitteeCard:
    agent: str
    purpose: str
    evidence: list[str] = field(default_factory=list)  # evidence lines
    recommendation: str = ""
    alternatives: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    artifacts_used: list[str] = field(default_factory=list)
    decision: str = ""

    @property
    def evidence_backed(self) -> bool:
        return bool(self.evidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "purpose": self.purpose,
            "evidence": self.evidence,
            "evidence_backed": self.evidence_backed,
            "recommendation": self.recommendation,
            "alternatives": self.alternatives,
            "risks": self.risks,
            "open_questions": self.open_questions,
            "artifacts_used": self.artifacts_used,
            "decision": self.decision,
        }


def render_card_rich(card: CommitteeCard) -> Any:
    """Render a committee card as a Rich renderable (panel of sections)."""
    from rich.panel import Panel
    from rich.table import Table

    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="bold cyan", justify="right", no_wrap=True)
    grid.add_column()

    grid.add_row("Purpose", card.purpose)

    if card.evidence_backed:
        ev = "\n".join(f"• {e}" for e in card.evidence)
    else:
        ev = "[yellow]The recommendation is not evidence-backed.[/yellow]"
    grid.add_row("Evidence", ev)
    grid.add_row("Recommendation", card.recommendation or "—")
    if card.alternatives:
        grid.add_row("Alternatives", "\n".join(f"{i}. {a}" for i, a in enumerate(card.alternatives, 1)))
    if card.risks:
        grid.add_row("Risks", "\n".join(f"• {r}" for r in card.risks))
    if card.artifacts_used:
        grid.add_row("Artifacts", ", ".join(card.artifacts_used))
    if card.open_questions:
        grid.add_row("Open questions", "\n".join(f"• {q}" for q in card.open_questions))
    if card.decision:
        grid.add_row("Decision", card.decision)

    return Panel(grid, title=f"[bold]{card.agent}[/bold]", border_style="cyan", title_align="left")


def render_card_markdown(card: CommitteeCard) -> str:
    """Markdown rendering for dashboard/transcript (#12)."""
    lines = [f"### {card.agent}", "", f"**Purpose:** {card.purpose}", "", "**Evidence:**"]
    if card.evidence_backed:
        lines += [f"- {e}" for e in card.evidence]
    else:
        lines.append("- _The recommendation is not evidence-backed._")
    lines += ["", f"**Recommendation:** {card.recommendation or '—'}"]
    if card.alternatives:
        lines += ["", "**Alternatives:**"]
        lines += [f"{i}. {a}" for i, a in enumerate(card.alternatives, 1)]
    if card.risks:
        lines += ["", "**Risks:**"] + [f"- {r}" for r in card.risks]
    if card.artifacts_used:
        lines += ["", f"**Artifacts used:** {', '.join(card.artifacts_used)}"]
    if card.open_questions:
        lines += ["", "**Open questions:**"] + [f"- {q}" for q in card.open_questions]
    if card.decision:
        lines += ["", f"**Decision:** {card.decision}"]
    return "\n".join(lines) + "\n"
