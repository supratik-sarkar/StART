"""Persistent review session state (v2.2.0 items 3 & 8).

A single ``ReviewSession`` is the one workflow engine that both front-ends (the
terminal and notebook 05) drive. It records, in order:

- **decisions** — what each checkpoint offered, what the reviewer selected, and the
  effective value (accept / keep / modify / reject), so downstream agents can
  see e.g. "correlation pruning rejected by reviewer".
- **conversations** — every freeform question the reviewer prompted to an agent and the
  agent's reply (the committee transcript).
- **overrides** — explicit reviewer overrides of an agent recommendation.

State is additive and queryable: an agent calls ``session.decision_for(key)``
or ``session.rejected("correlation_pruning")`` to adapt its behavior. Nothing is
stateless; nothing is silent. The session serializes to a dict for the
dashboard transcript and for evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Decision:
    """A resolved checkpoint: what was offered and what the user decided."""

    key: str
    prompt: str
    recommended: str
    user_value: str
    effective: str
    choice: str  # accept | keep | modify | reject | auto_accept | non_interactive_keep
    rationale: str = ""
    agent_rationale: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "prompt": self.prompt,
            "recommended": self.recommended,
            "user_value": self.user_value,
            "effective": self.effective,
            "choice": self.choice,
            "rationale": self.rationale,
            "agent_rationale": self.agent_rationale,
            "evidence_ids": self.evidence_ids,
        }


@dataclass
class Exchange:
    """A single turn of user<->agent conversation at a checkpoint."""

    agent: str
    question: str
    answer: str
    checkpoint: str = ""
    backend: str = "deterministic"  # deterministic | <provider>
    provider: str = "none"
    model: str = "none"
    response_id: str = ""
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    evidence_used: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "question": self.question,
            "answer": self.answer,
            "checkpoint": self.checkpoint,
            "backend": self.backend,
            "provider": self.provider,
            "model": self.model,
            "response_id": self.response_id,
            "latency_seconds": round(self.latency_seconds, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "evidence_used": self.evidence_used,
        }


@dataclass
class Challenge:
    """A persistent reviewer challenge (v2.3.0 #3).

    Tracks a reviewer's objection or probe through the review,
    including the model provider, response id, token counts, and evidence used.
    """

    text: str
    agent: str
    response: str = ""
    evidence_used: list[str] = field(default_factory=list)
    status: str = "open"  # open | closed | unresolved
    provider: str = "none"
    model: str = "none"
    response_id: str = ""
    latency_seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    timestamp: str = ""
    conceded: bool = False
    changes_disposition: bool = False
    challenge_id: str = ""
    answer: str = ""
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "agent": self.agent,
            "response": self.response,
            "evidence_used": self.evidence_used,
            "status": self.status,
            "conceded": self.conceded,
            "changes_disposition": self.changes_disposition,
            "provider": self.provider,
            "model": self.model,
            "response_id": self.response_id,
            "latency_seconds": round(self.latency_seconds, 4),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "timestamp": self.timestamp,
        }


@dataclass
class ReviewSession:
    """The shared, persistent state of one interactive review."""

    run_id: str = "RUN"
    decisions: list[Decision] = field(default_factory=list)
    conversations: list[Exchange] = field(default_factory=list)
    # free-form clarifications the user supplied (e.g. cost priority text)
    clarifications: list[str] = field(default_factory=list)
    challenges: list[Challenge] = field(default_factory=list)
    mrm_signoff: dict[str, Any] | None = None
    validation_review: dict[str, Any] | None = None

    # -- recording -------------------------------------------------------- #
    def record_decision(self, decision: Decision) -> Decision:
        self.decisions.append(decision)
        return decision

    def record_exchange(self, exchange: Exchange) -> Exchange:
        self.conversations.append(exchange)
        return exchange

    def record_challenge(self, challenge: Challenge) -> Challenge:
        self.challenges.append(challenge)
        return challenge

    def close_challenge(
        self,
        text: str,
        response: str = "",
        evidence_used: list[str] | None = None,
        provider: str = "none",
        model: str = "none",
        response_id: str = "",
        latency_seconds: float = 0.0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        timestamp: str = "",
    ) -> Challenge | None:
        """Mark the most recent matching open challenge as closed with full model metadata."""
        for ch in reversed(self.challenges):
            if ch.text == text and ch.status == "open":
                ch.status = "closed"
                if response:
                    ch.response = response
                if evidence_used:
                    ch.evidence_used = evidence_used
                if provider:
                    ch.provider = provider
                if model:
                    ch.model = model
                if response_id:
                    ch.response_id = response_id
                if latency_seconds:
                    ch.latency_seconds = latency_seconds
                if input_tokens:
                    ch.input_tokens = input_tokens
                if output_tokens:
                    ch.output_tokens = output_tokens
                if timestamp:
                    ch.timestamp = timestamp
                return ch
        return None

    def open_challenges(self) -> list[Challenge]:
        return [c for c in self.challenges if c.status == "open"]

    def closed_challenges(self) -> list[Challenge]:
        return [c for c in self.challenges if c.status == "closed"]

    def unresolved_challenges(self) -> list[Challenge]:
        return [c for c in self.challenges if c.status == "unresolved"]

    def add_clarification(self, text: str) -> None:
        if text and text.strip():
            self.clarifications.append(text.strip())

    # -- querying (so downstream agents adapt to prior choices) ----------- #
    def decision_for(self, key: str) -> Decision | None:
        for d in reversed(self.decisions):
            if d.key == key:
                return d
        return None

    def rejected(self, key: str) -> bool:
        d = self.decision_for(key)
        return bool(d and d.choice == "reject")

    def accepted(self, key: str) -> bool:
        d = self.decision_for(key)
        return bool(d and d.choice in ("accept", "auto_accept"))

    def effective(self, key: str, default: str = "") -> str:
        d = self.decision_for(key)
        return d.effective if d else default

    def overrides(self) -> list[Decision]:
        """Decisions where the user did not take the recommendation."""
        return [d for d in self.decisions if d.effective != d.recommended]

    def context_banner(self) -> list[str]:
        """Short lines summarizing prior user decisions, for agents to honor."""
        lines = []
        for d in self.decisions:
            if d.choice == "reject":
                lines.append(f"{d.key}: REJECTED by user")
            elif d.effective != d.recommended:
                lines.append(f"{d.key}: user chose {d.effective} (rec was {d.recommended})")
        for c in self.clarifications:
            lines.append(f"clarification: {c}")
        return lines

    # -- serialization ---------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "decisions": [d.to_dict() for d in self.decisions],
            "conversations": [e.to_dict() for e in self.conversations],
            "clarifications": list(self.clarifications),
            "overrides": [d.to_dict() for d in self.overrides()],
            "challenges": [c.to_dict() for c in self.challenges],
            "challenge_summary": {
                "open": len(self.open_challenges()),
                "closed": len(self.closed_challenges()),
                "unresolved": len(self.unresolved_challenges()),
            },
            "mrm_signoff": self.mrm_signoff,
            "validation_review": self.validation_review,
        }

    def to_canonical_dict(self) -> dict[str, Any]:
        """Canonical dictionary for adjudications Merkle leaf hashing."""
        return {
            "decisions": [d.to_dict() for d in self.decisions],
            "overrides": [d.to_dict() for d in self.overrides()],
            "challenges": [c.to_dict() for c in self.challenges],
            "clarifications": sorted(self.clarifications),
            "conversations": [e.to_dict() for e in self.conversations],
            "challenge_summary": {
                "open": len(self.open_challenges()),
                "closed": len(self.closed_challenges()),
                "unresolved": len(self.unresolved_challenges()),
            },
            "validation_review": self.validation_review or {},
            "mrm_signoff": self.mrm_signoff or {},
        }

