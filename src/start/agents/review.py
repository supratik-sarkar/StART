"""Dual-mode agent review flow.

    Validation outputs -> evidence bundle -> ReviewPlanner -> TestSuggestion
    -> Finding -> Challenge -> Governance -> Signoff -> EvidenceCritic
    -> proof-carrying report

Both modes run the same flow over the same evidence:

  * deterministic — rules/templates only; zero LLM access, no API keys, no
    network. The public-safe offline mode.
  * llm — the configured provider reasons over the structured evidence
    bundle (never raw data, never computing metrics). Every section is gated
    by EvidenceCriticAgent.critique_section: on failure the section is
    retried once with correction instructions; if it fails again it is
    marked rejected and replaced by the deterministic fallback — explicitly,
    never silently.
"""

from __future__ import annotations

from typing import Any

from start.agents import (
    ChallengeAgent,
    EvidenceCriticAgent,
    GovernanceAgent,
    ModelRiskFindingAgent,
    SignoffAgent,
    TestSuggestionAgent,
)
from start.agents.prompts import (
    CORRECTION_SUFFIX,
    SYSTEM_PROMPT,
    build_evidence_bundle,
    build_section_prompt,
)
from start.core.schemas import AgentReview, EvidenceRecord, Status


def _deterministic_sections(
    records: list[EvidenceRecord], ctx: Any = None
) -> dict[str, Any]:
    """Mode 1 — deterministic governance fallback (no LLM, no network)."""
    governance_ok, governance_items = GovernanceAgent().review(records)
    skipped = [
        f"'{r.test_name}' did not execute — {r.interpretation.rstrip('.')}. [{r.evidence_id}]"
        for r in records
        if r.status == Status.SKIPPED
    ]
    by_status = [
        f"Read '{rec.test_name}' (status: {rec.status.value}). [{rec.evidence_id}]"
        for rec in sorted(
            records,
            key=lambda r: {"fail": 0, "error": 0, "warn": 1, "pass": 2, "skipped": 3}.get(
                r.status.value, 4
            ),
        )
    ]
    return {
        "review_plan": [
            "Review the evidence in breach-first order; warnings next, passes last.",
            *by_status[:8],
        ],
        "suggested_tests": TestSuggestionAgent().suggest(records, ctx),
        "findings": ModelRiskFindingAgent().findings(records),
        "challenge_memo": ChallengeAgent().challenge(records),
        "missing_evidence": skipped or ["No planned tests were skipped in this run."],
        "governance": governance_items
        or ["Governance checks passed: no breaches, no skipped tests, all evidence policy-stamped."],
        "signoff": SignoffAgent().conclude(records, governance_ok, governance_items),
    }


_LLM_SECTIONS = (
    "review_plan",
    "findings",
    "challenge_memo",
    "missing_evidence",
    "suggested_tests",
    "signoff_rationale",
)


def _llm_section(
    llm: Any,
    critic: EvidenceCriticAgent,
    section: str,
    bundle: str,
    records: list[EvidenceRecord],
) -> tuple[list[str] | None, bool]:
    """One evidence-grounded section with a single corrective retry.

    Returns (lines, accepted). lines is None when both attempts failed."""
    prompt = build_section_prompt(section, bundle)
    text = llm.generate(prompt, system=SYSTEM_PROMPT)
    critique = critic.critique_section(text, records)
    if not critique.ok:
        issues = "\n".join(f"- {i.code}: {i.message}" for i in critique.issues)
        text = llm.generate(
            prompt + CORRECTION_SUFFIX.format(issues=issues), system=SYSTEM_PROMPT
        )
        critique = critic.critique_section(text, records)
    if not critique.ok:
        return None, False
    lines = [ln.strip().lstrip("- ").strip() for ln in text.splitlines() if ln.strip()]
    return [ln for ln in lines if ln], True


def run_agent_review(
    records: list[EvidenceRecord],
    *,
    mode: str = "deterministic",
    llm: Any = None,
    ctx: Any = None,
    policy_hash: str | None = None,
    dataset_profile: str | None = None,
    demo_meta: dict[str, Any] | None = None,
) -> AgentReview:
    """Run the full agent review flow over validated evidence.

    The deterministic sections are always computed (they are the fallback);
    in llm mode each section is replaced by accepted LLM output, and rejected
    sections are disclosed in `rejected_sections` — never silently swapped."""
    critic = EvidenceCriticAgent()
    deterministic = _deterministic_sections(records, ctx)
    review = AgentReview(mode="deterministic", llm_provider="none", **deterministic)

    llm_unavailable = llm is None or not getattr(llm, "available", False)
    if mode == "llm" and llm_unavailable:
        review.notes.append(
            "WARNING: agent mode 'llm' was requested but no usable LLM provider is "
            "configured; fell back to deterministic mode explicitly."
        )
        mode = "deterministic"

    if mode == "llm":
        review.mode = "llm"
        review.llm_provider = getattr(llm, "name", "unknown")
        bundle = build_evidence_bundle(
            records,
            policy_hash=policy_hash,
            dataset_profile=dataset_profile,
            demo_meta=demo_meta,
        )
        for section in _LLM_SECTIONS:
            target = "signoff" if section == "signoff_rationale" else section
            try:
                lines, accepted = _llm_section(llm, critic, section, bundle, records)
            except Exception as exc:  # provider failure is never silent
                lines, accepted = None, False
                review.notes.append(
                    f"Section '{target}': provider error ({type(exc).__name__}); "
                    "deterministic fallback used."
                )
            if accepted and lines:
                if target == "signoff":
                    review.signoff = " ".join(lines)
                else:
                    setattr(review, target, lines)
            else:
                review.rejected_sections.append(target)
                review.notes.append(
                    f"Section '{target}': LLM output rejected by EvidenceCriticAgent "
                    "after one corrective retry; deterministic fallback shown."
                )
        # governance stays deterministic by design: it is the gate, not prose.

    section_texts = [
        *review.review_plan,
        *review.suggested_tests,
        *review.findings,
        *review.challenge_memo,
        *review.missing_evidence,
        *review.governance,
        review.signoff,
    ]
    review.critique_ok = all(
        critic.critique_section(text, records).ok for text in section_texts if text.strip()
    )
    return review


def load_run_records(
    output_root: str = "start_output",
    ledger_file: str = "ledger.jsonl",
    run_id: str = "latest",
) -> tuple[str, list[EvidenceRecord]]:
    """Reload evidence for a past run from the tamper-evident ledger, so the
    agent review flow can run post-hoc on stored evidence."""
    from pathlib import Path

    from start.evidence.ledger import EvidenceLedger

    root = Path(output_root)
    ledger = EvidenceLedger(root / ledger_file, root / "evidence_store")
    if not ledger.verify():
        raise ValueError(f"Ledger integrity check FAILED for {root / ledger_file}.")
    all_records = ledger.records()
    if not all_records:
        raise ValueError(f"No evidence records found in {root / ledger_file}.")
    if run_id == "latest":
        run_id = all_records[-1].run_id
    records = [r for r in all_records if r.run_id == run_id]
    if not records:
        known = sorted({r.run_id for r in all_records})
        raise ValueError(f"Run '{run_id}' not found in ledger. Known runs: {known}")
    return run_id, records
