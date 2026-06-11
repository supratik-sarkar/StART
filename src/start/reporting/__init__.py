"""Reviewer-ready Markdown report generation from a RunResult."""

from __future__ import annotations

from start.core.schemas import RunResult


def render_markdown(result: RunResult) -> str:
    lines: list[str] = [
        f"# StART Validation Report — `{result.run_id}`",
        "",
        f"- **Model:** `{result.plan.model_id}`",
        f"- **Dataset:** `{result.plan.dataset_id}`",
        f"- **Task type:** {result.plan.task_type.value}",
        f"- **Materiality:** {result.plan.materiality.value}",
        f"- **Tests executed:** {len(result.evidence)}",
    ]
    if result.policy:
        lines += [
            f"- **Policy hash:** `{result.policy.policy_hash}`",
            f"- **Policy decision:** {'allowed' if result.policy.allowed else 'blocked'} "
            f"({'; '.join(result.policy.reasons)})",
        ]
    lines.append("")

    if result.narrative:
        lines += ["## Reviewer narrative", "", result.narrative.summary, ""]
        if result.narrative.findings:
            lines.append("### Findings")
            lines += [f"- {f}" for f in result.narrative.findings] + [""]
        if result.narrative.next_steps:
            lines.append("### Recommended next steps")
            lines += [f"- {s}" for s in result.narrative.next_steps] + [""]
        if result.narrative.limitations:
            lines.append("### Limitations")
            lines += [f"- {lim}" for lim in result.narrative.limitations] + [""]
        if result.narrative.signoff:
            lines.append("### Reviewer sign-off recommendation")
            lines += [result.narrative.signoff, ""]

    if result.agent_review is not None:
        ar = result.agent_review
        lines.append("## Agent review")
        if ar.mode == "llm":
            lines.append("Agent mode: llm-assisted")
            lines.append(f"LLM provider: {ar.llm_provider}")
        else:
            lines.append("Agent mode: deterministic")
        lines.append(f"Evidence critique status: {'PASSED' if ar.critique_ok else 'FAILED'}")
        if ar.rejected_sections:
            lines.append(
                "Rejected LLM sections (deterministic fallback shown): "
                + ", ".join(ar.rejected_sections)
            )
        for note in ar.notes:
            lines.append(f"> {note}")
        lines.append("")
        for title, items in (
            ("Review plan", ar.review_plan),
            ("Suggested next tests", ar.suggested_tests),
            ("Model-risk findings", ar.findings),
            ("Challenge memo", ar.challenge_memo),
            ("Missing evidence", ar.missing_evidence),
            ("Governance assessment", ar.governance),
        ):
            if items:
                lines.append(f"### {title}")
                lines += [f"- {item}" for item in items] + [""]
        if ar.signoff:
            lines.append("### Sign-off recommendation")
            lines += [ar.signoff, ""]

    lines.append("## Evidence table")
    lines.append("")
    lines.append("| Evidence ID | Test | Status | Key metrics |")
    lines.append("|---|---|---|---|")
    for rec in result.evidence:
        metric_str = "; ".join(f"{k}={v}" for k, v in list(rec.metrics.items())[:4])
        lines.append(f"| {rec.evidence_id} | {rec.test_name} | {rec.status.value} | {metric_str} |")
    lines.append("")

    if result.critique:
        lines.append("## Critique")
        lines.append("")
        lines.append(f"Evidence/narrative critique: {'OK' if result.critique.ok else 'ISSUES FOUND'}")
        for issue in result.critique.issues:
            lines.append(f"- [{issue.severity}] {issue.code}: {issue.message}")
        lines.append("")

    lines.append("## Reproducibility")
    lines.append("")
    if result.evidence:
        repro = result.evidence[0].repro
        lines += [
            f"- Device: {repro.device.value}",
            f"- Runtime: {repro.runtime}",
            f"- Python: {repro.python_version}",
            f"- Git SHA: {repro.git_sha or 'n/a'}",
            f"- Seed: {repro.seed}",
            f"- Input data hash: `{result.evidence[0].input_artifact_hash or 'n/a'}`",
        ]
    return "\n".join(lines) + "\n"
