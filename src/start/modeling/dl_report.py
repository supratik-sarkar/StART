"""Proof-carrying Markdown report for the deep-learning review.

Includes run metadata, architecture, device, cohort metrics, the evidence
table (with EV-DL-000x labels), explainability, sensitivity, robustness,
generated figure paths, the deterministic/LLM agent review, sign-off, and
reproducibility metadata.
"""

from __future__ import annotations

from typing import Any

from start.core.schemas import EvidenceRecord


def _label(rec: EvidenceRecord) -> str:
    return rec.artifacts.get("dl_evidence_label", rec.evidence_id)


def render_dl_report(
    run_id: str,
    opts: Any,
    device: str,
    cohort_metrics: dict[str, dict[str, float]],
    evidence: list[EvidenceRecord],
    figures: dict[str, str],
    agent_review: Any,
    perf_extras: dict[str, dict[str, float]],
) -> str:
    lines: list[str] = [
        f"# StART Deep Learning Model Review — `{run_id}`",
        "",
        "## Run metadata",
        f"- Architecture: `{opts.architecture}`",
        f"- Device used: {device}",
        f"- Epochs (max): {opts.epochs} | batch size: {opts.batch_size} | lr: {opts.learning_rate}",
        f"- Agent mode: {'llm-assisted' if agent_review.mode == 'llm' else 'deterministic'}"
        + (f" | LLM provider: {agent_review.llm_provider}" if agent_review.mode == "llm" else ""),
        f"- Evidence critique status: {'PASSED' if agent_review.critique_ok else 'FAILED'}",
        "",
        "## Cohort metrics",
        "",
        "| Cohort | AUC-ROC | Accuracy | Precision | Recall | F1 | Top 10% Lift | Brier | ECE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for cohort in ("train", "test", "oos"):
        if cohort in cohort_metrics:
            m = cohort_metrics[cohort]
            lines.append(
                f"| {cohort} | {m['auc_roc']:.4f} | {m['accuracy']:.4f} | {m['precision']:.4f} "
                f"| {m['recall']:.4f} | {m['f1']:.4f} | {m['top_decile_lift']:.4f} "
                f"| {m['brier_score']:.4f} | {m['ece']:.4f} |"
            )

    lines += ["", "## Evidence table", "", "| Evidence ID | Diagnostic | Status |", "| --- | --- | --- |"]
    for rec in evidence:
        lines.append(f"| {_label(rec)} | {rec.test_name} | {rec.status.value} |")

    # explainability
    explain = _find(evidence, "deep_learning.explainability_diagnostics")
    if explain:
        lines += [
            "",
            "## Explainability",
            f"- Method used: **{explain.metrics.get('method')}** "
            f"(available: {explain.metrics.get('available_methods')})",
            f"- Top features: {explain.metrics.get('top_features')}",
            f"- {explain.interpretation} [{_label(explain)}]",
        ]

    # sensitivity
    sens = _find(evidence, "deep_learning.sensitivity_diagnostics")
    if sens:
        lines += [
            "",
            "## Sensitivity — top-feature shocks",
            "",
            "| Shock | AUC-ROC | AUC drift |",
            "| --- | --- | --- |",
        ]
        for pct in (-30, -20, -10, 0, 10, 20, 30):
            label = f"{pct:+d}pct"
            auc = sens.metrics.get(f"auc_{label}")
            drift = sens.metrics.get(f"drift_{label}")
            if auc is not None:
                lines.append(f"| {pct:+d}% | {auc:.4f} | {drift:+.4f} |")
        lines.append(f"\nShocked features: {sens.metrics.get('shocked_features')} [{_label(sens)}]")

    # robustness
    robust = _find(evidence, "deep_learning.robustness_diagnostics")
    if robust:
        lines += [
            "",
            "## Robustness",
            "",
            "### Input noise",
            "",
            "| Noise | AUC-ROC | AUC drift |",
            "| --- | --- | --- |",
        ]
        for level in (0.0, 0.01, 0.03, 0.05, 0.10):
            auc = robust.metrics.get(f"noise_{level:.2f}_auc")
            drift = robust.metrics.get(f"noise_{level:.2f}_drift")
            if auc is not None:
                lines.append(f"| {level:.2f} | {auc:.4f} | {drift:+.4f} |")
        lines += [
            "",
            "### Feature masking",
            "",
            "| Masked top-k | AUC-ROC | AUC drift |",
            "| --- | --- | --- |",
        ]
        for k in (1, 3, 5):
            auc = robust.metrics.get(f"mask_top{k}_auc")
            drift = robust.metrics.get(f"mask_top{k}_drift")
            if auc is not None:
                lines.append(f"| top {k} | {auc:.4f} | {drift:+.4f} |")
        lines.append(f"\n[{_label(robust)}]")

    # figures
    if figures:
        lines += ["", "## Generated figures"]
        lines += [f"- `{name}`: `{path}`" for name, path in sorted(figures.items())]

    # agent review
    lines += ["", "## Agentic review"]
    mode = "llm-assisted" if agent_review.mode == "llm" else "deterministic"
    lines.append(f"Agent mode: {mode}")
    if agent_review.mode == "llm":
        lines.append(f"LLM provider: {agent_review.llm_provider}")
    if agent_review.rejected_sections:
        lines.append(
            "Rejected LLM sections (deterministic fallback shown): "
            + ", ".join(agent_review.rejected_sections)
        )
    for note in agent_review.notes:
        lines.append(f"> {note}")
    for title, items in (
        ("Reviewer summary / plan", agent_review.review_plan),
        ("Suggested next tests", agent_review.suggested_tests),
        ("Model-risk findings", agent_review.findings),
        ("Challenger memo", agent_review.challenge_memo),
        ("Missing evidence", agent_review.missing_evidence),
        ("Governance assessment", agent_review.governance),
    ):
        if items:
            lines += ["", f"### {title}"]
            lines += [f"- {item}" for item in items]
    lines += ["", "### Sign-off recommendation", agent_review.signoff]

    # limitations + repro
    lines += ["", "## Limitations"]
    seen: set[str] = set()
    for rec in evidence:
        for lim in rec.limitations:
            if lim not in seen:
                seen.add(lim)
                lines.append(f"- {lim}")
    repro = evidence[0].repro if evidence else None
    lines += ["", "## Reproducibility metadata"]
    if repro is not None:
        lines.append(f"- seed: {opts.seed}")
        lines.append(f"- git commit: {getattr(repro, 'git_commit', 'n/a')}")
        lines.append(f"- packages: {getattr(repro, 'package_versions', {})}")
    lines.append(f"- policy hash: {evidence[0].policy_hash if evidence else 'n/a'}")

    return "\n".join(lines) + "\n"


def _find(evidence: list[EvidenceRecord], test_id: str) -> EvidenceRecord | None:
    return next((r for r in evidence if r.test_id == test_id), None)
