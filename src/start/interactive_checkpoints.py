"""Interactive decision checkpoints (v2.1.1 Section B).

Turns each agent recommendation into a visible checkpoint the user can act on:

    [A] Accept recommendation
    [K] Keep my choice
    [E] Explain further

The checkpoint is deterministic and testable: the prompt function is injected
(defaults to ``input``), and a non-interactive / auto-accept path is supported.
Every resolution returns a ``CheckpointDecision`` that records what was offered,
what the user chose, and the effective value — for the evidence ledger and the
action log. No silent overrides: the decision is always explicit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class CheckpointDecision:
    name: str
    user_value: str
    recommended_value: str
    reason: str
    evidence_id: str
    choice: str  # "accept" | "keep" | "auto_accept" | "non_interactive_keep"
    effective_value: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.name,
            "user_value": self.user_value,
            "recommended_value": self.recommended_value,
            "reason": self.reason,
            "evidence_id": self.evidence_id,
            "choice": self.choice,
            "effective_value": self.effective_value,
        }


def render_checkpoint(
    name: str, user_value: str, recommended_value: str, reason: str,
    evidence_id: str = "", extra: str = "",
) -> str:
    agree = user_value == recommended_value
    lines = [
        f"  Checkpoint: {name}",
        f"    Your choice : {user_value}",
        f"    Agent rec.  : {recommended_value}",
        f"    Reason      : {reason}",
    ]
    if evidence_id:
        lines.append(f"    Evidence    : {evidence_id}")
    if extra:
        lines.append(f"    Detail      : {extra}")
    if agree:
        lines.append("    (agent agrees with your choice)")
    else:
        lines.append("    [A] Accept recommendation   [K] Keep my choice   [E] Explain further")
    return "\n".join(lines)


def resolve_checkpoint(
    name: str,
    user_value: str,
    recommended_value: str,
    reason: str,
    *,
    evidence_id: str = "",
    explanation: str = "",
    interactive: bool = False,
    auto_accept: bool = False,
    ask: Callable[[str], str] = input,
    emit: Callable[[str], None] | None = None,
    on_ask: Callable[[str], str] | None = None,
) -> CheckpointDecision:
    """Resolve a single decision checkpoint.

    - If the agent agrees with the user, no prompt is needed.
    - auto_accept: take the recommendation, recorded as 'auto_accept'.
    - non-interactive (and not auto_accept): keep the user's choice, recorded
      as 'non_interactive_keep' (never a silent override).
    - interactive: prompt [A]/[K]/[E]; [E] prints the explanation and re-asks.
      If ``on_ask`` is provided, [Q] lets the user ask the agent a freeform
      question; the answer is printed and the prompt repeats (item 2).
    """
    say = emit or (lambda _msg: None)
    say(render_checkpoint(name, user_value, recommended_value, reason, evidence_id, explanation))

    if user_value == recommended_value:
        return CheckpointDecision(
            name, user_value, recommended_value, reason, evidence_id,
            choice="accept", effective_value=user_value,
        )

    if auto_accept:
        return CheckpointDecision(
            name, user_value, recommended_value, reason, evidence_id,
            choice="auto_accept", effective_value=recommended_value,
        )

    if not interactive:
        # Non-interactive default keeps the user's explicit choice; visible, not silent.
        return CheckpointDecision(
            name, user_value, recommended_value, reason, evidence_id,
            choice="non_interactive_keep", effective_value=user_value,
        )

    ask_hint = " / Ask agent (Q)" if on_ask else ""
    while True:
        answer = (ask(
            f"[{name}] Accept (A) / Keep (K) / Explain (E){ask_hint}? "
        ) or "").strip().lower()
        if answer in ("a", "accept"):
            return CheckpointDecision(
                name, user_value, recommended_value, reason, evidence_id,
                choice="accept", effective_value=recommended_value,
            )
        if answer in ("k", "keep", ""):
            return CheckpointDecision(
                name, user_value, recommended_value, reason, evidence_id,
                choice="keep", effective_value=user_value,
            )
        if answer in ("e", "explain"):
            say(f"    {explanation or reason}")
            continue
        if on_ask and answer in ("q", "ask", "?"):
            question = (ask("    Ask the agent: ") or "").strip()
            if question:
                say(f"    {on_ask(question)}")
            continue
        say(f"    Please answer A, K, or E{', or Q to ask' if on_ask else ''}.")
