"""Evidence-constrained answering for Ask-Agent (v2.3.0 #1).

Implements the pipeline:

    Question -> Artifact Retrieval -> Evidence Assembly -> Answer -> Critic

For any question that asks for *diagnostic values* (outlier burden, correlation
pairs, missingness, feature importance, sensitivity drift, cohort metrics,
tuning results), we answer ONLY from the EvidenceStore. If the relevant
evidence is absent, we return the explicit refusal:

    "I do not have sufficient evidence to answer this question."

This intercepts diagnostic questions BEFORE any LLM call, so the model can never
invent feature names, percentages, counts, thresholds, or drift values. The
evidence critic then verifies the assembled answer only cites retrieved facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from start.evidence_store import EvidenceItem, EvidenceStore

INSUFFICIENT = "I do not have sufficient evidence to answer this question."


@dataclass
class EvidenceAnswer:
    answer: str
    grounded: bool          # True if backed by retrieved evidence
    evidence: list[dict[str, Any]]
    refused: bool = False   # True if we declined for lack of evidence


# Keyword -> retrieval intent. If a question matches one of these intents we
# MUST answer from evidence or refuse; we never let an LLM free-form it.
_DIAGNOSTIC_INTENTS = {
    "outlier": "outliers",
    "missing": "missingness",
    "null": "missingness",
    "nan": "missingness",
    "correlat": "correlation",
    "collinear": "correlation",
    "importance": "importance",
    "important feature": "importance",
    "feature importance": "importance",
    "sensitiv": "sensitivity",
    "drift": "sensitivity",
    "shock": "sensitivity",
    "metric": "metrics",
    "auc": "metrics",
    "pr-auc": "metrics",
    "recall": "metrics",
    "precision": "metrics",
    "accuracy": "metrics",
    "tuning": "tuning",
    "trial": "tuning",
    "hyperparam": "tuning",
    "leakage": "leakage",
    "leak": "leakage",
}


def classify_intent(question: str) -> str | None:
    """Return the diagnostic intent of a question, or None if non-diagnostic."""
    q = question.lower()
    # longest keyword first so "feature importance" beats "importance"
    for kw in sorted(_DIAGNOSTIC_INTENTS, key=len, reverse=True):
        if kw in q:
            return _DIAGNOSTIC_INTENTS[kw]
    return None


def _extract_n(question: str, default: int = 10) -> int:
    import re

    m = re.search(r"\btop\s+(\d{1,3})\b", question.lower())
    if m:
        return max(1, min(int(m.group(1)), 100))
    return default


def retrieve(intent: str, store: EvidenceStore, question: str) -> list[EvidenceItem]:
    n = _extract_n(question)
    if intent == "outliers":
        return store.top_outliers(n)
    if intent == "missingness":
        return store.top_missing(n)
    if intent == "correlation":
        return store.top_correlations(n)
    if intent == "importance":
        return store.top_importance(n)
    if intent == "sensitivity":
        return store.sensitivity_evidence(n)
    if intent == "metrics":
        return store.metrics_evidence()
    if intent == "tuning":
        return [
            EvidenceItem("tuning", f"trial {t.get('trial')}: {t.get('params')} "
                         f"-> {t.get('validation_metric')}", t, "tuning_run.trials")
            for t in store.tuning_trials[:n]
        ]
    if intent == "leakage":
        return [
            EvidenceItem("leakage", f"leakage candidate: {c}", c,
                         "data_statistics.leakage_candidates")
            for c in store.leakage_candidates
        ]
    return []


def evidence_critic(answer: str, evidence: list[EvidenceItem]) -> bool:
    """Verify the answer is grounded: a diagnostic answer must either be the
    explicit refusal or reference at least one retrieved evidence item."""
    if answer.strip() == INSUFFICIENT:
        return True
    if not evidence:
        return False
    # at least one evidence label fragment must appear in the answer
    return any(
        str(item.value) in answer or item.label.split(":")[0] in answer
        for item in evidence
    )


def answer_from_evidence(question: str, store: EvidenceStore) -> EvidenceAnswer | None:
    """Answer a diagnostic question strictly from evidence, or refuse.

    Returns None if the question is NOT diagnostic (caller may then use the
    normal recommendation-explanation path). For diagnostic questions it always
    returns an EvidenceAnswer — grounded values or an explicit refusal.
    """
    intent = classify_intent(question)
    if intent is None:
        return None  # not a diagnostic question; let the caller handle it

    items = retrieve(intent, store, question)
    if not items:
        # The reviewer asked for a diagnostic we do not have. Refuse explicitly;
        # never fabricate. Name the category so the refusal is informative.
        return EvidenceAnswer(
            answer=f"{INSUFFICIENT} (No {intent} evidence is available for this "
            "review.)",
            grounded=False, evidence=[], refused=True,
        )

    lines = [f"Based on the {intent} evidence on record:"]
    for item in items:
        lines.append(f"  - {item.label}")
    lines.append(f"(source: {items[0].source})")
    answer = "\n".join(lines)
    ok = evidence_critic(answer, items)
    if not ok:
        # critic failed -> do not emit a possibly-ungrounded answer
        return EvidenceAnswer(answer=INSUFFICIENT, grounded=False,
                              evidence=[i.to_dict() for i in items], refused=True)
    return EvidenceAnswer(answer=answer, grounded=True,
                          evidence=[i.to_dict() for i in items], refused=False)
