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

from start.agents.prompts import HUMAN_QUERY_REASONING_DIRECTIVE
from start.review_session import Exchange, ReviewSession


def _is_conceptual_question(question: str) -> bool:
    q = question.lower()
    # Check if the question is conceptual, mathematical, or structural
    conceptual_keywords = [
        "what is", "how does", "why does", "explain", "concept", "theory", "theoretical",
        "mathematical", "algorithmic", "design", "gradient", "behavior", "lookahead",
        "imputation", "critique", "difference between", "definition", "architectural",
        "structural choice", "neural structural", "design-pattern", "why did you", "why use",
        "pros and cons", "trade-off", "why not", "i disagree", "challenge"
    ]
    return any(kw in q for kw in conceptual_keywords)


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
    business_context: str = ""
    reviewer_clarification: str = ""
    task_type: str = ""
    model_name: str = ""


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
        f"Business Context: {getattr(ctx, 'business_context', '') or 'n/a'}.\n"
        f"Reviewer Clarification: {getattr(ctx, 'reviewer_clarification', '') or 'n/a'}.\n"
        f"Task Type: {getattr(ctx, 'task_type', '') or 'n/a'}.\n"
        f"Model Name: {getattr(ctx, 'model_name', '') or 'n/a'}.\n"
        "Answer the reviewer concisely and specifically. Do not invent metrics.\n\n"
        f"{HUMAN_QUERY_REASONING_DIRECTIVE}"
    )
    try:
        text = llm.generate(prompt, system=f"StART review agent. Be concise and honest.\n\n{HUMAN_QUERY_REASONING_DIRECTIVE}",
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

    If LLM is connected, always queries the LLM provider.
    """
    import time

    from rich.console import Console

    from start.review_session import Challenge

    console = Console()
    is_challenge = _is_challenge(question)
    if is_challenge:
        session.record_challenge(Challenge(text=question, agent=agent))

    # Determine if LLM is connected
    is_connected = llm_connected or (
        llm is not None 
        and getattr(llm, "name", "none") != "none" 
        and type(llm).__name__ != "NoLLMProvider"
    )

    # 1. Deterministic Evidence Lookup (Enriches prompt context, does not short-circuit)
    evidence_content = ""
    evidence_ids = []
    if ctx.evidence is not None:
        from start.evidence_dialogue import answer_from_evidence
        ea = answer_from_evidence(question, ctx.evidence)
        if ea is not None:
            evidence_content = ea.answer
            evidence_ids = list(dict.fromkeys(e.get("source", "") for e in ea.evidence))
        else:
            evidence_content = "No specific diagnostic evidence matches this question."
    else:
        evidence_content = "No evidence store available."

    backend = "deterministic"
    provider = "none"
    selected_model = "none"
    response_id = ""
    latency = 0.0
    in_tokens = 0
    out_tokens = 0
    error_reason = None
    answer = None

    if is_connected:
        # Construct context-rich prompt
        alt_text = ""
        if ctx.alternatives:
            alt_text = "; ".join(f"{r.get('family', '?')}" for r in ctx.alternatives)

        prompt = (
            f"You are {agent}, a model-risk review agent.\n"
            f"Selected Architecture: {ctx.model_name or 'n/a'}\n"
            f"Recommendation: {ctx.recommendation or 'n/a'}\n"
            f"Rationale: {ctx.reason or 'n/a'}\n"
            f"Alternatives: {alt_text or 'n/a'}\n"
            f"Dataset Summary: {ctx.dataset_summary or 'n/a'}\n"
            f"Task Type: {ctx.task_type or 'n/a'}\n"
            f"Business Context: {ctx.business_context or 'n/a'}\n"
            f"Reviewer Clarification: {ctx.reviewer_clarification or 'n/a'}\n"
            f"Evidence IDs: {', '.join(evidence_ids) or 'n/a'}\n"
            f"Evidence Content: {evidence_content or 'n/a'}\n\n"
            f"Reviewer Question/Challenge: \"{question}\"\n\n"
            f"Please answer the reviewer's question/challenge directly, concisely, and specifically. "
            f"Support your answer with the provided dataset/model context and evidence. "
            f"Do not invent metrics."
        )

        try:
            start_time = time.perf_counter()
            text = llm.generate(
                prompt,
                system="StART review agent. Be concise and honest.",
                metadata={"max_tokens": 512}
            )
            latency = time.perf_counter() - start_time
            provider = getattr(llm, "name", "unknown")
            selected_model = getattr(llm, "model", "unknown")
            response_id = getattr(llm, "last_response_id", "")
            in_tokens = getattr(llm, "last_input_tokens", 0)
            out_tokens = getattr(llm, "last_output_tokens", 0)

            if text and text.strip():
                answer = text.strip()
                backend = "llm"
            else:
                raise ValueError("LLM provider returned empty response.")
        except Exception as e:
            backend = "fallback"
            error_reason = f"Exception: {type(e).__name__} - {str(e)}"
            # Fall back to deterministic answer
            answer = _deterministic_answer(question, ctx)
    else:
        # LLM not connected
        backend = "deterministic"
        # If it was a diagnostic question, use the evidence answer directly
        if ctx.evidence is not None:
            from start.evidence_dialogue import answer_from_evidence
            ea = answer_from_evidence(question, ctx.evidence)
            if ea is not None:
                if not (ea.refused and _is_conceptual_question(question)):
                    answer = ea.answer
        if not answer:
            answer = _deterministic_answer(question, ctx)

    # 5 & 6. Print immediate output to the terminal
    if backend == "llm":
        console.print("\n[bold green]backend = llm[/bold green]")
        console.print(f"  provider            : {provider}")
        console.print(f"  selected model      : {selected_model}")
        console.print(f"  provider response ID: {response_id}")
        console.print(f"  latency             : {latency:.4f}s")
        console.print(f"  input/output tokens : {in_tokens} / {out_tokens}")
        console.print(f"  evidence IDs used   : {', '.join(evidence_ids) or 'none'}")
    elif backend == "fallback":
        console.print("\n[bold red]backend = fallback[/bold red]")
        console.print(f"  safe failure reason : {error_reason}")
        console.print(f"  evidence IDs used   : {', '.join(evidence_ids) or 'none'}")
    else:
        console.print("\n[bold yellow]backend = deterministic[/bold yellow]")
        console.print(f"  evidence IDs used   : {', '.join(evidence_ids) or 'none'}")

    # Set last response info on llm object if connected so caller can access it
    if is_connected and llm is not None:
        object.__setattr__(llm, "last_response_id", response_id)
        object.__setattr__(llm, "last_latency_seconds", latency)
        object.__setattr__(llm, "last_input_tokens", in_tokens)
        object.__setattr__(llm, "last_output_tokens", out_tokens)
        try:
            object.__setattr__(llm, "_telemetry_printed", True)
        except Exception:
            pass

    # 7. Persist the same telemetry and full exchange in the transcript / session
    exchange_backend = "deterministic" if backend == "fallback" else (provider if backend == "llm" else backend)
    exchange = Exchange(
        agent=agent,
        question=question,
        answer=answer,
        checkpoint=ctx.checkpoint,
        backend=exchange_backend,
    )
    session.record_exchange(exchange)

    if session is not None and hasattr(session, "record_qa"):
        session.record_qa(
            checkpoint_id=ctx.checkpoint,
            agent_name=agent,
            question=question,
            answer=answer,
            response_id=response_id,
            evidence_context=", ".join(evidence_ids) if evidence_ids else "",
        )
        try:
            object.__setattr__(session, "_qa_recorded", True)
        except Exception:
            pass

    if is_challenge:
        session.close_challenge(question, response=answer, evidence_used=evidence_ids or ["agent_reasoning"])

    return exchange
