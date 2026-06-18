"""Interactive checkpoint flow (v2.2.0 items 1, 2, 5).

Runs the pre-execution review checkpoints — target confirmation, feature
engineering negotiation, and metric priority — each with the [Q] Ask-Agent
capability and full ReviewSession recording. Returns the resolved choices that
the orchestrator then executes, so user decisions actually drive the run
(not just the transcript).

The single architecture checkpoint already lives in ``interactive_review`` for
historical reasons; this module adds the rest of the committee touchpoints and
shares the same ``resolve_checkpoint`` + ``ask_agent`` primitives.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from start.agent_dialogue import AgentContext, ask_agent
from start.interactive_checkpoints import resolve_checkpoint
from start.review_session import Decision, ReviewSession


def _ask_factory(agent: str, ctx: AgentContext, session: ReviewSession,
                 llm: Any, llm_connected: bool) -> Callable[[str], str]:
    def _ask(question: str) -> str:
        return ask_agent(agent, question, ctx, session,
                         llm=llm, llm_connected=llm_connected).answer
    return _ask


def run_feature_engineering_checkpoints(
    fe_set: Any,
    session: ReviewSession,
    *,
    interactive: bool,
    auto_accept: bool,
    llm: Any = None,
    llm_connected: bool = False,
    ask: Callable[[str], str] = input,
    emit: Callable[[str], None] | None = None,
    evidence: Any = None,
) -> dict[str, str]:
    """Negotiate each FE recommendation (item 5). Returns step -> action map
    ('accept' keeps the default action; 'skip' rejects it). Decisions are
    recorded so downstream FE execution respects rejections (#2)."""
    say = emit or (lambda _m: None)
    overrides: dict[str, str] = {}
    for rec in fe_set.applicable():
        ctx = AgentContext(
            agent="FeatureEngineeringAgent",
            recommendation=rec.recommendation,
            reason=rec.reason, risk_if_ignored=rec.risk_if_ignored,
            alternatives=None,
            dataset_summary="",
            checkpoint=f"fe:{rec.step}", evidence=evidence,
        )
        on_ask = _ask_factory("FeatureEngineeringAgent", ctx, session, llm, llm_connected)
        action = "apply"
        choice = "accept"
        if auto_accept:
            choice = "auto_accept"
        elif interactive:
            # Single negotiable prompt per FE action: apply / skip / ask.
            say(f"  FeatureEngineeringAgent recommends: {rec.recommendation}")
            say(f"    reason: {rec.reason}")
            while True:
                resp = (ask(
                    f"    Apply {rec.step}? [Y]es / [n]o (skip) / [Q] ask agent: "
                ) or "").strip().lower()
                if resp in ("q", "ask", "?"):
                    q = (ask("      Ask the agent: ") or "").strip()
                    if q:
                        say(f"      {on_ask(q)}")
                    continue
                if resp in ("n", "no", "skip", "reject"):
                    action, choice = "skip", "reject"
                break
        overrides[rec.step] = action
        session.record_decision(Decision(
            key=f"fe:{rec.step}", prompt=f"Apply {rec.step}?",
            recommended="apply", user_value=action, effective=action,
            choice=choice, rationale=rec.reason, evidence_ids=[rec.evidence_id],
        ))
    return overrides


def run_metric_checkpoint(
    user_cost: str,
    recommended_cost: str,
    reason: str,
    session: ReviewSession,
    *,
    interactive: bool,
    auto_accept: bool,
    llm: Any = None,
    llm_connected: bool = False,
    ask: Callable[[str], str] = input,
    emit: Callable[[str], None] | None = None,
    evidence: Any = None,
) -> str:
    """Confirm cost priority / primary metric (items 1, 2). Returns effective
    cost priority that routes tuning + validation downstream (#2)."""
    say = emit or (lambda _m: None)
    ctx = AgentContext(
        agent="HyperparameterTuningAgent",
        recommendation=recommended_cost, reason=reason,
        risk_if_ignored="Wrong metric can optimize for the wrong error type.",
        alternatives=[{"family": "false_negatives"}, {"family": "false_positives"},
                      {"family": "balanced"}],
        dataset_summary="", checkpoint="metric_priority", evidence=evidence,
    )
    on_ask = _ask_factory("HyperparameterTuningAgent", ctx, session, llm, llm_connected)
    dec = resolve_checkpoint(
        "metric_priority", user_cost, recommended_cost, reason,
        explanation=reason, interactive=interactive, auto_accept=auto_accept,
        ask=ask, emit=say, on_ask=on_ask,
    )
    session.record_decision(Decision(
        key="metric_priority", prompt="Cost priority?",
        recommended=recommended_cost, user_value=user_cost,
        effective=dec.effective_value, choice=dec.choice, rationale=reason,
    ))
    return dec.effective_value


def run_target_checkpoint(
    candidate_target: str,
    recommended_target: str,
    reason: str,
    session: ReviewSession,
    *,
    interactive: bool,
    auto_accept: bool,
    llm: Any = None,
    llm_connected: bool = False,
    ask: Callable[[str], str] = input,
    emit: Callable[[str], None] | None = None,
    evidence: Any = None,
) -> str:
    """Confirm / override the target (item 4). Returns the effective target."""
    say = emit or (lambda _m: None)
    ctx = AgentContext(
        agent="DatasetDiscoveryAgent", recommendation=recommended_target,
        reason=reason, risk_if_ignored="Wrong target invalidates the entire review.",
        alternatives=None, dataset_summary="", checkpoint="target", evidence=evidence,
    )
    on_ask = _ask_factory("DatasetDiscoveryAgent", ctx, session, llm, llm_connected)
    dec = resolve_checkpoint(
        "target", candidate_target, recommended_target, reason,
        explanation=reason, interactive=interactive, auto_accept=auto_accept,
        ask=ask, emit=say, on_ask=on_ask,
    )
    session.record_decision(Decision(
        key="target", prompt="Target column?",
        recommended=recommended_target, user_value=candidate_target,
        effective=dec.effective_value, choice=dec.choice, rationale=reason,
    ))
    return dec.effective_value
