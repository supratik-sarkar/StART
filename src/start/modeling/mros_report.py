"""Proof-carrying report for the Model-Risk Operating System review.

Explicitly states task type, target column(s), modality, recommended family,
cohort metrics, the evidence table, the full visible stage timeline, the
agentic review (challenge/governance/sign-off), the AI-engineering stage
surface (with honest availability), assumptions, and validation
recommendations.
"""

from __future__ import annotations

from typing import Any


def render_mros_report(
    run_id: str,
    inference: Any,
    target: Any,
    modality: str,
    recommended: str,
    cohort_metrics: dict[str, dict[str, float]],
    evidence: list[Any],
    agent_review: Any,
    ai_stages: list[Any],
    stage_events: list[Any],
) -> str:
    lines: list[str] = [
        f"# StART Model-Risk Review — `{run_id}`",
        "",
        "## Review summary",
        f"- Task type: **{inference.task_type}**",
        f"- Target column(s): `{target}`",
        f"- Target type: {inference.target_type}"
        + (f" ({inference.n_classes} classes)" if inference.n_classes else ""),
        f"- Modality: {modality}",
        f"- Recommended model family: `{recommended}`",
        f"- Agent mode: {'llm-assisted' if agent_review.mode == 'llm' else 'deterministic'}",
        f"- Evidence critique: {'PASSED' if agent_review.critique_ok else 'FAILED'}",
        "",
        "## Pipeline stages (visible execution)",
        "",
        "| Stage | Status | Detail |",
        "| --- | --- | --- |",
    ]
    for ev in stage_events:
        if ev.status in {"complete", "skipped"}:
            lines.append(f"| {ev.stage} | {ev.status} | {ev.detail} |")

    if cohort_metrics:
        lines += [
            "", "## Cohort metrics", "",
            "| Cohort | AUC-ROC | Accuracy | F1 |", "| --- | --- | --- | --- |",
        ]
        for cohort, m in cohort_metrics.items():
            lines.append(
                f"| {cohort} | {m.get('auc_roc', float('nan')):.4f} "
                f"| {m.get('accuracy', float('nan')):.4f} | {m.get('f1', float('nan')):.4f} |"
            )

    lines += [
        "", "## Evidence ledger", "",
        "| Test ID | Name | Status |", "| --- | --- | --- |",
    ]
    for rec in evidence:
        lines.append(f"| {rec.test_id} | {rec.test_name} | {rec.status.value} |")

    lines += [
        "", "## AI-engineering stage surface", "",
        "| Stage | Category | Status |", "| --- | --- | --- |",
    ]
    for s in ai_stages:
        lines.append(f"| {s.name} | {s.category} | {s.status} |")

    lines += ["", "## Agentic review"]
    for title, items in (
        ("Reviewer plan", agent_review.review_plan),
        ("Challenge memo", agent_review.challenge_memo),
        ("Governance assessment", agent_review.governance),
    ):
        if items:
            lines += ["", f"### {title}"]
            lines += [f"- {i}" for i in items]
    lines += ["", "### Sign-off recommendation", agent_review.signoff]

    lines += [
        "",
        "## Assumptions",
        f"- Task inferred as {inference.task_type}; {inference.note}",
        "- Split holds out an explicit OOS cohort for generalization estimates.",
        "- Deterministic diagnostics compute metrics; the LLM (if used) reasons only over evidence.",
        "",
        "## Validation recommendations",
        "- Confirm the inferred task type and target with a domain owner before sign-off.",
        "- Review any warn/fail evidence and the challenge memo before production use.",
        "- For deep-learning model families, run the dedicated DL review workflow for full "
        "explainability, sensitivity, and robustness suites.",
    ]
    return "\n".join(lines) + "\n"
