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

import select
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


def read_multiline_paste_normalized(prompt: str) -> str:
    """Read a line of input. If additional lines are queued in stdin (indicating a paste),

    read them all with a short timeout and join them with spaces into a single string.
    This prevents pasted newlines from remaining in the stdin buffer and polluting later prompts.
    If stdin is not a TTY or not select-supported, fallback to standard readline/input.
    """
    if prompt:
        print(prompt, end="", flush=True)
    first_line = sys.stdin.readline()
    if not first_line:
        return ""
    
    lines = [first_line.strip()]
    try:
        if sys.stdin.isatty():
            while True:
                r, _, _ = select.select([sys.stdin], [], [], 0.05)
                if r:
                    next_line = sys.stdin.readline()
                    if not next_line:
                        break
                    lines.append(next_line.strip())
                else:
                    break
    except (OSError, AttributeError, ValueError):
        pass

    return " ".join([line for line in lines if line]).strip()


@dataclass
class CheckpointDecision:
    name: str
    user_value: str
    recommended_value: str
    reason: str
    evidence_id: str
    choice: str  # "accept" | "keep" | "override" | "auto_accept" | "non_interactive_keep"
    effective_value: str
    rationale: str = ""
    agent_rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.name,
            "user_value": self.user_value,
            "recommended_value": self.recommended_value,
            "reason": self.reason,
            "evidence_id": self.evidence_id,
            "choice": self.choice,
            "effective_value": self.effective_value,
            "rationale": self.rationale,
            "agent_rationale": self.agent_rationale,
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
    # v3.1.1: always show all options, even when values agree
    lines.append("    [A] Accept   [O] Override   [C] Challenge   [Q] Ask agent")
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
    llm: Any = None,
    session: Any = None,
    ctx: Any = None,
) -> CheckpointDecision:
    """Resolve a single decision checkpoint.

    v3.1.1: All checkpoints show [A]/[O]/[C]/[Q] even if user and recommended
    values agree. LLM response telemetry is printed on every agent answer.
    Q&A exchanges are persisted with checkpoint ID, agent name, question,
    answer, response ID, and evidence context.
    """
    say = emit or (lambda _msg: None)
    say(render_checkpoint(name, user_value, recommended_value, reason, evidence_id, explanation))

    def _fallback_log(q_text, ans_text):
        if llm is not None and getattr(llm, "last_response_id", None):
            if getattr(llm, "_telemetry_printed", False) is not True:
                _total_tokens = llm.last_input_tokens + llm.last_output_tokens
                say(
                    f"    [LLM Call] Response ID: {llm.last_response_id} | "
                    f"Latency: {llm.last_latency_seconds:.3f}s | "
                    f"Tokens: {llm.last_input_tokens}/{llm.last_output_tokens} "
                    f"(total: {_total_tokens})"
                )
            try:
                if hasattr(llm, "_telemetry_printed"):
                    delattr(llm, "_telemetry_printed")
            except AttributeError:
                pass

        if session is not None and hasattr(session, "record_qa"):
            if getattr(session, "_qa_recorded", False) is not True:
                session.record_qa(
                    checkpoint_id=name,
                    agent_name=current_agent,
                    question=q_text,
                    answer=ans_text,
                    response_id=getattr(llm, "last_response_id", "") if llm else "",
                    evidence_context=evidence_id,
                )
            try:
                if hasattr(session, "_qa_recorded"):
                    delattr(session, "_qa_recorded")
            except AttributeError:
                pass

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

    checkpoint_agent_map = {
        "architecture": "ArchitectureReviewAgent",
        "metric_priority": "HyperparameterTuningAgent",
        "target": "DatasetDiscoveryAgent",
    }
    current_agent = checkpoint_agent_map.get(name, "System Coordinator")

    while True:
        if ask is input:
            from rich.console import Console

            from start.cli.view import get_styled_agent_name
            c = Console()
            c.print(f"[{name}] Accept (A) / Override (O) / Challenge (C)", end="")
            if on_ask:
                c.print(" / [Q] ask ", end="")
                c.print(get_styled_agent_name(current_agent), end="")
            c.print("? ", end="")
            answer_raw = read_multiline_paste_normalized("")
        else:
            from start.cli.view import get_ansi_agent_name
            ask_hint = f" / [Q] ask {get_ansi_agent_name(current_agent)}" if on_ask else ""
            answer_raw = ask(
                f"[{name}] Accept (A) / Override (O) / Challenge (C){ask_hint}? "
            )

        stripped = (answer_raw or "").strip()
        answer = stripped.lower()

        # Check if a freeform question was prompted
        is_q = False
        question = ""
        if answer in ("q", "ask", "?"):
            is_q = True
        elif answer.startswith("q ") or answer.startswith("q\n") or answer.startswith("q:"):
            is_q = True
            question = stripped[1:].strip()
            if question.startswith(":") or question.startswith("\n"):
                question = question[1:].strip()

        if on_ask and is_q:
            if not question:
                prompt_q = f"    Ask {current_agent}: "
                if ask is input:
                    question = read_multiline_paste_normalized(prompt_q)
                else:
                    question = ask(prompt_q)

            question = (question or "").strip()
            # Do not treat the literal Q or empty as the question
            if question.lower() in ("q", "ask", "?") or not question:
                continue

            agent_answer = on_ask(question)
            say(f"    {agent_answer}")
            _fallback_log(question, agent_answer)
            continue

        # Challenge — prompts for a challenge question and calls the provider
        if answer in ("c", "challenge"):
            if on_ask:
                prompt_c = f"    Enter challenge to {current_agent}: "
                if ask is input:
                    challenge_q = read_multiline_paste_normalized(prompt_c)
                else:
                    challenge_q = ask(prompt_c)
                challenge_q = (challenge_q or "").strip()
                if not challenge_q:
                    challenge_q = (
                        f"Why is the recommended value '{recommended_value}' "
                        f"preferable to my choice '{user_value}'? "
                        f"Please justify this choice and address the alternative."
                    )
                agent_answer = on_ask(challenge_q)
                say(f"    {agent_answer}")
                _fallback_log(challenge_q, agent_answer)
            else:
                say("    Challenge accepted. Agent reasoning:")
                say(f"    {explanation or reason}")
                if evidence_id:
                    say(f"    Evidence basis: {evidence_id}")

            if session and getattr(session, "challenges", None):
                from start.governance.challenge_disposition import CONCESSION_PROMPT

                if ask is input:
                    ans_raw = read_multiline_paste_normalized(CONCESSION_PROMPT)
                else:
                    try:
                        ans_raw = ask(CONCESSION_PROMPT)
                    except (StopIteration, Exception):
                        ans_raw = ""
                ans_conceded = (ans_raw or "").strip().lower() in {"y", "yes"}
                last_ch = session.challenges[-1]
                last_ch.conceded = ans_conceded
                last_ch.changes_disposition = ans_conceded
            elif ask is input:
                from start.governance.challenge_disposition import CONCESSION_PROMPT

                read_multiline_paste_normalized(CONCESSION_PROMPT)
            continue

        if answer in ("a", "accept"):
            return CheckpointDecision(
                name,
                user_value=recommended_value,
                recommended_value=recommended_value,
                reason=reason,
                evidence_id=evidence_id,
                choice="accept",
                effective_value=recommended_value,
                rationale="Accepted agent recommendation",
                agent_rationale=reason,
            )
        # v3.1.1 / v4.0.1: Override prompts for a new value and reviewer rationale
        if answer in ("o", "override"):
            if ask is input:
                new_val = read_multiline_paste_normalized("    Override value: ")
                rev_rationale = read_multiline_paste_normalized("    Reviewer rationale for override: ")
            else:
                new_val = ask("    Override value: ")
                try:
                    rev_rationale = ask("    Reviewer rationale for override: ")
                except (StopIteration, Exception):
                    rev_rationale = ""
            new_val = (new_val or "").strip()
            rev_rationale = (rev_rationale or "").strip()
            if not new_val:
                new_val = user_value
            is_noop = (new_val == recommended_value)
            if not rev_rationale:
                rev_rationale = "Accepted agent recommendation" if is_noop else f"Reviewer overridden to {new_val}"
            return CheckpointDecision(
                name,
                user_value=new_val,
                recommended_value=recommended_value,
                reason=reason,
                evidence_id=evidence_id,
                choice="accept" if is_noop else "override",
                effective_value=new_val,
                rationale=rev_rationale,
                agent_rationale=reason,
            )
        if answer in ("k", "keep", ""):
            is_noop = (user_value == recommended_value)
            return CheckpointDecision(
                name,
                user_value=user_value,
                recommended_value=recommended_value,
                reason=reason,
                evidence_id=evidence_id,
                choice="accept" if is_noop else "keep",
                effective_value=user_value,
                rationale="Accepted agent recommendation" if is_noop else "Kept original reviewer choice",
                agent_rationale=reason,
            )
        if answer in ("e", "explain"):
            say(f"    {explanation or reason}")
            continue

        if not answer:
            continue

        say(f"    Please answer A, O, C{', or Q to ask' if on_ask else ''}.")

