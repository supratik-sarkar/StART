"""Challenge and override disposition.

The defect this exists to fix
-----------------------------

In a live review the reviewer challenged ValidationAgent on age as a protected
characteristic under ECOA. The agent conceded, citing evidence:

    "This model cannot be signed off without a fair-lending disparity analysis...
     a dedicated fair-lending disparity analysis is required to evaluate whether
     these features lead to discriminatory effects before signoff."

Governance then reported::

    Reviewer challenges | ok | no outstanding reviewer challenges
    READY WITH CONDITIONS: 0 blocker(s)

An agent stated that sign-off was not possible, and the disposition recorded zero
blockers. The most consequential finding in the review did not reach the decision.

The cause is a missing state. ``Challenge.status`` was ``open`` or ``closed``, and
``closed`` meant only "a response came back". A challenge the agent satisfied and a
challenge the agent conceded were indistinguishable, so both read as "no outstanding
challenges" — the phrase that appeared while the review's central objection sat
unaddressed.

A review platform whose sign-off can ignore its own findings is worse than no platform,
because it manufactures assurance. So concession becomes a first-class state, and a
conceded challenge is a blocker.

The second defect
-----------------

Both reviewer overrides in that run were counted as governance *concerns* — including
one that replaced an unjustifiable LSTM after the agent itself conceded the point, and
one that applied the dataset's own published 5:1 cost matrix.

Penalising a well-reasoned override teaches reviewers not to override. That is the
opposite of what a challenge culture needs, and it is a false positive in exactly the
register a reviewer is supposed to trust. An override is judged on whether it is
*reasoned* and whether the agent *agreed*, not on its existence.

Who decides
-----------

Concession is not inferred from prose. Keyword-matching an LLM response for "cannot be
signed off" would fire on "this cannot be signed off without evidence, which is
present" and miss "the objection stands". The reviewer is asked directly, once, after
the response:

    Does this response change the disposition? [y/N]

The machine detects that a challenge occurred and presents the exchange. The reviewer
decides what it means. The seal remembers both.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "ChallengeDisposition",
    "OverrideClass",
    "ChallengeVerdict",
    "OverrideVerdict",
    "classify_challenge",
    "classify_override",
    "challenge_factor",
    "override_factor",
    "CONCESSION_PROMPT",
]

CONCESSION_PROMPT = "    Does this response change the disposition? [y/N]: "


class ChallengeDisposition(StrEnum):
    """What became of a reviewer challenge."""

    #: Raised, answered, and the reviewer was satisfied the position holds.
    RESOLVED = "raised_and_resolved"
    #: Raised, and the agent's own answer undermined its prior position or stated
    #: that sign-off cannot proceed. Blocks.
    CONCEDED = "raised_and_conceded"
    #: Raised and never answered, or answered without resolution.
    OUTSTANDING = "outstanding"


class OverrideClass(StrEnum):
    """How a reviewer override bears on the disposition."""

    #: The agent conceded the point on challenge, then the reviewer overrode.
    #: The override implements the agent's own reasoning.
    AGENT_ENDORSED = "agent_endorsed"
    #: Reasoned, but not preceded by a concession — the reviewer exercised
    #: independent judgement.
    REASONED = "reasoned"
    #: No rationale recorded. This is the one that should worry a reader.
    UNEXPLAINED = "unexplained"
    #: Recommended and effective values are identical. Not an override at all.
    NO_OP = "no_op"


@dataclass(frozen=True)
class ChallengeVerdict:
    challenge_id: str
    agent: str
    disposition: ChallengeDisposition
    text: str = ""
    response: str = ""
    reviewer_confirmed_concession: bool = False
    evidence_used: tuple[str, ...] = field(default=())

    @property
    def blocks(self) -> bool:
        return self.disposition in {
            ChallengeDisposition.CONCEDED,
            ChallengeDisposition.OUTSTANDING,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "agent": self.agent,
            "disposition": self.disposition.value,
            "blocks": self.blocks,
            "reviewer_confirmed_concession": self.reviewer_confirmed_concession,
            "evidence_used": list(self.evidence_used),
            "text": self.text,
            "response": self.response,
        }


@dataclass(frozen=True)
class OverrideVerdict:
    key: str
    classification: OverrideClass
    recommended: str = ""
    effective: str = ""
    rationale: str = ""

    @property
    def severity(self) -> str:
        """`concern` only where the override is genuinely questionable."""
        if self.classification is OverrideClass.UNEXPLAINED:
            return "concern"
        if self.classification is OverrideClass.NO_OP:
            return "ok"
        return "informational"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "classification": self.classification.value,
            "severity": self.severity,
            "recommended": self.recommended,
            "effective": self.effective,
            "rationale": self.rationale,
        }


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #
def classify_challenge(challenge: Any) -> ChallengeVerdict:
    """Determine what became of one challenge.

    Reads whatever the session object exposes, tolerating both the current
    ``Challenge`` dataclass and a plain dict, so this can be wired in before the
    session schema is extended.

    Concession is taken from an explicit reviewer confirmation
    (``conceded`` / ``changes_disposition`` / status ``conceded``). It is never
    inferred from the wording of the response — see the module docstring.
    """

    def get(name: str, default: Any = None) -> Any:
        if isinstance(challenge, dict):
            return challenge.get(name, default)
        return getattr(challenge, name, default)

    status = str(get("status", "open") or "open").lower()
    response = str(get("response", "") or "")
    conceded_flag = bool(get("conceded", False) or get("changes_disposition", False) or status == "conceded")

    if conceded_flag:
        disposition = ChallengeDisposition.CONCEDED
    elif status in {"open", "unresolved"} or not response.strip():
        disposition = ChallengeDisposition.OUTSTANDING
    else:
        disposition = ChallengeDisposition.RESOLVED

    evidence = get("evidence_used", ()) or ()
    if isinstance(evidence, str):
        evidence = (evidence,)

    return ChallengeVerdict(
        challenge_id=str(get("challenge_id", "") or get("id", "") or ""),
        agent=str(get("agent", "") or ""),
        disposition=disposition,
        text=str(get("text", "") or ""),
        response=response,
        reviewer_confirmed_concession=conceded_flag,
        evidence_used=tuple(str(e) for e in evidence),
    )


def classify_override(decision: Any, conceded_keys: set[str] | None = None) -> OverrideVerdict:
    """Classify one reviewer override.

    ``conceded_keys`` names the checkpoints where the agent conceded on challenge.
    An override at such a checkpoint implements the agent's own reasoning and is
    informational, not a concern.
    """

    def get(name: str, default: Any = None) -> Any:
        if isinstance(decision, dict):
            return decision.get(name, default)
        return getattr(decision, name, default)

    key = str(get("key", "") or "")
    recommended = str(get("recommended", "") or "")
    effective = str(get("effective", "") or get("user_value", "") or "")
    rationale = str(get("rationale", "") or "").strip()
    agent_rationale = str(get("agent_rationale", "") or "").strip()
    choice = str(get("choice", "") or "").strip().lower()
    status = str(get("status", "") or "").strip().lower()

    # A rationale identical to the agent's own text is not the reviewer's reasoning;
    # it is the agent's boilerplate carried over. Treat it as absent.
    reviewer_reasoned = bool(rationale) and rationale != agent_rationale

    # The session's own record of what the reviewer did outranks a string diff.
    # If the decision records choice/status == "accepted", return OverrideClass.NO_OP.
    if choice in ("accepted", "accept", "auto_accept") or status in ("accepted", "accept", "auto_accept"):
        classification = OverrideClass.NO_OP
    elif recommended and effective and recommended == effective:
        classification = OverrideClass.NO_OP
    elif (conceded_keys or set()) and key in (conceded_keys or set()):
        classification = OverrideClass.AGENT_ENDORSED
    elif reviewer_reasoned:
        classification = OverrideClass.REASONED
    else:
        classification = OverrideClass.UNEXPLAINED

    return OverrideVerdict(
        key=key,
        classification=classification,
        recommended=recommended,
        effective=effective,
        rationale=rationale,
    )


# --------------------------------------------------------------------------- #
# Governance factors
# --------------------------------------------------------------------------- #
def challenge_factor(verdicts: list[ChallengeVerdict]) -> tuple[str, str, str]:
    """Build the `Reviewer challenges` factor as ``(status, detail, evidence)``.

    The phrase "no outstanding reviewer challenges" may only appear when no challenge
    was raised at all. When challenges were raised, the factor says how many and what
    became of them — the wording that would have surfaced the fair-lending concession.
    """
    if not verdicts:
        return "ok", "no reviewer challenges raised", "review_session"

    conceded = [v for v in verdicts if v.disposition is ChallengeDisposition.CONCEDED]
    outstanding = [v for v in verdicts if v.disposition is ChallengeDisposition.OUTSTANDING]
    resolved = [v for v in verdicts if v.disposition is ChallengeDisposition.RESOLVED]

    if conceded:
        agents = ", ".join(sorted({v.agent for v in conceded if v.agent})) or "an agent"
        return (
            "blocker",
            f"{len(conceded)} challenge(s) conceded by {agents} — the agent's own response "
            f"undermines its prior position and sign-off cannot proceed until addressed "
            f"({len(resolved)} resolved, {len(outstanding)} outstanding)",
            "review_session",
        )

    if outstanding:
        return (
            "blocker",
            f"{len(outstanding)} reviewer challenge(s) raised and unresolved ({len(resolved)} resolved)",
            "review_session",
        )

    return (
        "ok",
        f"{len(resolved)} reviewer challenge(s) raised and resolved",
        "review_session",
    )


def override_factor(verdicts: list[OverrideVerdict]) -> tuple[str, str, str]:
    """Build the `Reviewer overrides` factor as ``(status, detail, evidence)``."""
    real = [v for v in verdicts if v.classification is not OverrideClass.NO_OP]
    if not real:
        return "ok", "no overrides; reviewer accepted recommendations", "review_session"

    unexplained = [v for v in real if v.classification is OverrideClass.UNEXPLAINED]
    endorsed = [v for v in real if v.classification is OverrideClass.AGENT_ENDORSED]
    reasoned = [v for v in real if v.classification is OverrideClass.REASONED]

    if unexplained:
        keys = ", ".join(v.key for v in unexplained)
        return (
            "concern",
            f"{len(unexplained)} override(s) recorded without a reviewer rationale "
            f"({keys}) — an unexplained override is not a governance record",
            "review_session",
        )

    parts = []
    if endorsed:
        parts.append(f"{len(endorsed)} following an agent concession on challenge")
    if reasoned:
        parts.append(f"{len(reasoned)} with reviewer rationale")
    keys = ", ".join(v.key for v in real)

    return (
        "informational",
        f"{len(real)} reviewer override(s) ({keys}): " + "; ".join(parts),
        "review_session",
    )
