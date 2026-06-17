"""Ask-Agent dialogue (v2.2.0 items 2, 5, 6).

Lets the user ask a review agent freeform questions at a checkpoint —
"Why MLP?", "Show alternatives", "Why not XGBoost?", "What if I keep all
features?" — and get a live answer that is recorded into the review session
transcript.

Two backends, chosen automatically and transparently:

- **LLM** (when a provider is genuinely connected): the agent answers in natural
  language, grounded in the structured decision context we pass it. No raw user
  data is sent — only the agent's own recommendation, reasons, and the dataset
  shape summary.
- **Deterministic** (default / fallback): we answer from the structured
  reasoning the agents already produce (recommendation, reason, risk-if-ignored,
  ranked alternatives). Never fabricated — if we cannot answer a specific
  question deterministically, we say so and show what we do know.

Every reply is returned as an ``Exchange`` and appended to the session, so the
conversation becomes part of the committee transcript.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from start.review_session import Exchange, ReviewSession


@dataclass
class AgentContext:
    """The structured, data-free context an agent reasons over for Q&A."""

    agent: str
    recommendation: str
    reason: str
    risk_if_ignored: str = ""
    alternatives: list[dict[str, Any]] | None = None
    dataset_summary: str = ""
    checkpoint: str = ""
    evidence: Any = None  # EvidenceStore: diagnostic facts for grounded answers (#1)


# Canonical model-family comparison (item 6): performance / interpretability /
# maintenance / governance. Deterministic, public, model-agnostic guidance.
_FAMILY_PROFILES: dict[str, dict[str, str]] = {
    "mlp": {
        "performance": "Strong on small/medium tabular data; low variance.",
        "interpretability": "Moderate (needs attribution methods).",
        "maintenance": "Low — few moving parts.",
        "governance": "Low complexity; well understood by MRM.",
    },
    "wide_deep": {
        "performance": "Can edge out MLP on large data with strong cross features.",
        "interpretability": "Lower; wide+deep interaction is harder to explain.",
        "maintenance": "Higher — two pathways to maintain.",
        "governance": "Higher complexity; more to document.",
    },
    "residual_mlp": {
        "performance": "Helps very deep nets; little gain on small tabular data.",
        "interpretability": "Lower than plain MLP.",
        "maintenance": "Moderate.",
        "governance": "Moderate complexity.",
    },
    "xgboost": {
        "performance": "Very strong on tabular; robust to feature scaling.",
        "interpretability": "Moderate (SHAP/gain); non-differentiable.",
        "maintenance": "Low–moderate; mature tooling.",
        "governance": "Well accepted, but outside this DL harness's scope.",
    },
    "lstm": {
        "performance": "Strong default for sequential/temporal data.",
        "interpretability": "Low; sequence attributions are noisy.",
        "maintenance": "Moderate.",
        "governance": "Higher complexity for temporal models.",
    },
}


def compare_model_families(families: list[str]) -> list[dict[str, str]]:
    """Item 6: a ranked comparison table for the candidate families."""
    rows = []
    for fam in families:
        prof = _FAMILY_PROFILES.get(fam, {
            "performance": "Not profiled in the public guidance table.",
            "interpretability": "—", "maintenance": "—", "governance": "—",
        })
        rows.append({"family": fam, **prof})
    return rows


def _deterministic_answer(question: str, ctx: AgentContext) -> str:
    """Answer from the agent's structured reasoning, never fabricated."""
    q = question.lower().strip()

    # "show alternatives" / comparison
    if any(k in q for k in ("alternativ", "compare", "options", "other model")):
        fams = [r["family"] for r in (ctx.alternatives or [])] or [ctx.recommendation]
        rows = compare_model_families(fams)
        lines = [f"Alternatives considered for {ctx.checkpoint or 'this decision'}:"]
        for r in rows:
            lines.append(
                f"  - {r['family']}: perf={r['performance']} "
                f"interpretability={r['interpretability']} "
                f"maintenance={r['maintenance']} governance={r['governance']}"
            )
        return "\n".join(lines)

    # "why not X"
    if q.startswith("why not") or "instead of" in q:
        alt = q.replace("why not", "").replace("?", "").strip()
        prof = _FAMILY_PROFILES.get(alt.split()[0] if alt else "", None)
        if prof:
            return (
                f"{alt} is a reasonable choice ({prof['performance']}), but the "
                f"recommendation is {ctx.recommendation} because: {ctx.reason} "
                f"Trade-off if you switch: interpretability={prof['interpretability']}, "
                f"governance={prof['governance']}."
            )
        return (
            f"The recommendation is {ctx.recommendation} because: {ctx.reason} "
            f"I don't have a public profile for '{alt}', so I can't compare it "
            "directly here."
        )

    # overfitting / risk
    if any(k in q for k in ("overfit", "risk", "danger", "downside")):
        return (
            f"Risk if the recommendation ({ctx.recommendation}) is not followed: "
            f"{ctx.risk_if_ignored or 'not quantified for this decision.'} "
            f"Rationale: {ctx.reason}"
        )

    # "what if I keep all features" / impact of rejecting
    if "keep all" in q or ("keep" in q and "feature" in q):
        return (
            "Keeping all features is allowed. Effect: nothing is pruned, so any "
            "redundant or low-variance columns remain. That can add mild variance "
            "and slow training, but does not block the review — downstream agents "
            "will record that pruning was declined and proceed accordingly."
        )

    # default: explain the recommendation
    return (
        f"Recommendation: {ctx.recommendation}. Reason: {ctx.reason} "
        f"{('Risk if ignored: ' + ctx.risk_if_ignored) if ctx.risk_if_ignored else ''}"
    ).strip()


def _llm_answer(question: str, ctx: AgentContext, llm: Any) -> str | None:
    """Ask a connected LLM, grounded only in data-free context. None on failure."""
    alt_text = ""
    if ctx.alternatives:
        alt_text = "; ".join(
            f"{r.get('family', '?')}" for r in ctx.alternatives
        )
    prompt = (
        f"You are {ctx.agent}, a model-risk review agent. A reviewer asked: "
        f"\"{question}\".\n"
        f"Your recommendation: {ctx.recommendation}.\n"
        f"Your reason: {ctx.reason}.\n"
        f"Risk if ignored: {ctx.risk_if_ignored or 'n/a'}.\n"
        f"Alternatives on the table: {alt_text or 'n/a'}.\n"
        f"Dataset (no raw rows): {ctx.dataset_summary or 'n/a'}.\n"
        "Answer the reviewer concisely and specifically. Do not invent metrics."
    )
    try:
        text = llm.generate(prompt, system="StART review agent. Be concise and honest.",
                            metadata={"max_tokens": 256})
        return text.strip() if text else None
    except Exception:
        return None


def _is_challenge(question: str) -> bool:
    """Heuristic: does this question constitute a reviewer challenge?"""
    q = question.lower().strip()
    return (
        q.startswith("why not")
        or q.startswith("i disagree")
        or q.startswith("i don't agree")
        or "challenge" in q
        or ("show" in q and "evidence" in q)
        or "prove" in q
        or "justify" in q
    )


def ask_agent(
    agent: str,
    question: str,
    ctx: AgentContext,
    session: ReviewSession,
    *,
    llm: Any = None,
    llm_connected: bool = False,
) -> Exchange:
    """Answer a freeform question and record it in the session transcript.

    v2.3.0 #1: diagnostic questions (outliers, correlations, missingness,
    importance, sensitivity, metrics, tuning) are answered STRICTLY from the
    evidence store, or explicitly refused — before any LLM is consulted, so the
    model can never fabricate diagnostic values. Only non-diagnostic questions
    (e.g. "why MLP?") fall through to the LLM / deterministic explanation path.

    v2.3.0 #3: questions that constitute reviewer challenges ("why not X?",
    "I disagree...", "show ... evidence") are recorded as persistent challenges
    and closed with the answer + the evidence used.
    """
    from start.review_session import Challenge

    backend = "deterministic"
    answer: str | None = None
    evidence_used: list[str] = []
    is_challenge = _is_challenge(question)
    if is_challenge:
        session.record_challenge(Challenge(text=question, agent=agent))

    # --- evidence-constrained path (anti-hallucination) ---
    if ctx.evidence is not None:
        from start.evidence_dialogue import answer_from_evidence

        ea = answer_from_evidence(question, ctx.evidence)
        if ea is not None:  # it was a diagnostic question
            backend = "evidence" if ea.grounded else "evidence-refusal"
            evidence_used = list(dict.fromkeys(e.get("source", "") for e in ea.evidence))
            exchange = Exchange(
                agent=agent, question=question, answer=ea.answer,
                checkpoint=ctx.checkpoint, backend=backend,
            )
            session.record_exchange(exchange)
            if is_challenge:
                session.close_challenge(question, response=ea.answer,
                                        evidence_used=evidence_used or ["evidence_store"])
            return exchange

    # --- non-diagnostic: LLM (if connected) then deterministic explanation ---
    if llm is not None and llm_connected:
        answer = _llm_answer(question, ctx, llm)
        if answer:
            backend = getattr(llm, "name", "llm")
    if not answer:
        answer = _deterministic_answer(question, ctx)
    exchange = Exchange(
        agent=agent, question=question, answer=answer,
        checkpoint=ctx.checkpoint, backend=backend,
    )
    session.record_exchange(exchange)
    if is_challenge:
        # challenge answered from agent reasoning (not raw evidence)
        session.close_challenge(question, response=answer,
                                evidence_used=["agent_reasoning"])
    return exchange
