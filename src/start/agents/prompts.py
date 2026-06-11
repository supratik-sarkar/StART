"""Evidence-grounded prompt builders for LLM-assisted review.

The LLM never computes metrics and never sees raw data. It receives only the
structured, validated evidence bundle below — evidence records, statuses,
metric tables, policy hash, skipped tests, dataset/run metadata — and must
cite an evidence ID for every factual or quantitative claim. Anything it
produces is re-checked by EvidenceCriticAgent before it can enter a report.
"""

from __future__ import annotations

from typing import Any

from start.core.schemas import EvidenceRecord, Status

SYSTEM_PROMPT = """You are a model-risk review agent inside the StART framework.
You may only use the evidence records provided below.
Every factual or quantitative claim must cite evidence IDs in the form [EV-...].
If evidence is missing for something, say the evidence is missing.
Do not invent metrics, thresholds, test names, datasets, or model behavior.
Do not compute or recompute any numbers; only restate values that appear in the evidence.
Write plainly for a model-risk reviewer. Output one bullet per line starting with '- '."""

_MAX_METRICS_PER_RECORD = 12


def build_evidence_bundle(
    records: list[EvidenceRecord],
    *,
    policy_hash: str | None = None,
    dataset_profile: str | None = None,
    demo_meta: dict[str, Any] | None = None,
) -> str:
    """Serialize validated evidence into the only context the LLM receives.

    No raw rows, no source data, no credentials — record metadata and metrics
    only."""
    lines: list[str] = ["## Run metadata"]
    if records:
        lines.append(f"run_id: {records[0].run_id}")
        lines.append(f"model_id: {records[0].model_id} | dataset_id: {records[0].dataset_id}")
    if policy_hash:
        lines.append(f"policy_hash: {policy_hash}")
    if dataset_profile:
        lines.append(f"dataset_profile: {dataset_profile}")
    if demo_meta:
        meta = {k: v for k, v in demo_meta.items() if isinstance(v, (str, int, float, bool))}
        if meta:
            lines.append(f"model_meta: {meta}")

    lines.append("\n## Evidence records")
    for rec in records:
        lines.append(f"\n[{rec.evidence_id}] {rec.test_id} — {rec.test_name}")
        lines.append(f"  status: {rec.status.value}")
        if rec.interpretation:
            lines.append(f"  interpretation: {rec.interpretation}")
        metrics = {
            k: v
            for k, v in list(rec.metrics.items())[:_MAX_METRICS_PER_RECORD]
            if isinstance(v, (int, float, str, bool))
        }
        if metrics:
            lines.append(f"  metrics: {metrics}")
        breached = [t.metric for t in rec.thresholds if getattr(t, "breached", None)]
        if breached:
            lines.append(f"  breached_thresholds: {breached}")

    warns = [r.evidence_id for r in records if r.status == Status.WARN]
    fails = [r.evidence_id for r in records if r.status in (Status.FAIL, Status.ERROR)]
    skipped = [f"{r.evidence_id} ({r.test_name})" for r in records if r.status == Status.SKIPPED]
    lines.append("\n## Status summary")
    lines.append(f"warnings: {warns or 'none'}")
    lines.append(f"fails_or_errors: {fails or 'none'}")
    lines.append(f"skipped_tests: {skipped or 'none'}")
    return "\n".join(lines)


SECTION_INSTRUCTIONS: dict[str, str] = {
    "review_plan": (
        "Produce a short review plan: what this run validated, in what order a "
        "reviewer should read the evidence, and which records deserve the most "
        "scrutiny. Cite the evidence IDs you reference."
    ),
    "findings": (
        "Produce model-risk findings: overfitting or instability patterns across "
        "cohorts, notable warnings or failures, and explainability/sensitivity "
        "observations. Every claim that uses a number or a status must cite its "
        "evidence ID."
    ),
    "challenge_memo": (
        "Act as an adversarial second reviewer. Try to weaken or invalidate the "
        "run's conclusions using only the evidence: memorization risks, drift "
        "undermining cohort choices, sampling caveats, or gaps in coverage. Cite "
        "evidence IDs for every challenge. If you cannot mount a challenge from "
        "the evidence, say so."
    ),
    "missing_evidence": (
        "List missing or skipped evidence: planned tests that did not execute, "
        "artifacts that were unavailable, and validation areas with no coverage "
        "in this run. Cite the evidence IDs of skipped records where they exist."
    ),
    "suggested_tests": (
        "Recommend the next validation checks a reviewer should request, grounded "
        "in what the evidence shows or fails to show. Cite evidence IDs when a "
        "suggestion follows from a specific record."
    ),
    "signoff_rationale": (
        "State whether this run appears ready for reviewer sign-off and why, "
        "based strictly on the statuses and governance signals in the evidence. "
        "Do not declare readiness if any fail or error status exists. Cite the "
        "evidence IDs that drive your conclusion."
    ),
}


def build_section_prompt(section: str, bundle: str) -> str:
    if section not in SECTION_INSTRUCTIONS:
        raise ValueError(f"Unknown review section '{section}'. Known: {list(SECTION_INSTRUCTIONS)}")
    return f"{SECTION_INSTRUCTIONS[section]}\n\n{bundle}"


CORRECTION_SUFFIX = (
    "\n\nYour previous answer was rejected by the evidence critic for the issues "
    "listed below. Rewrite it so that every factual or quantitative claim cites a "
    "valid evidence ID from the bundle, no unknown test names or evidence IDs "
    "appear, and no readiness claim contradicts fail/error statuses.\nIssues:\n{issues}"
)
