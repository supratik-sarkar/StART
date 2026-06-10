"""GenAI starter checks: grounding/citation coverage.

These are deliberately simple, deterministic starters. Full grounding (NLI
entailment), prompt-injection probes, and retrieval-faithfulness checks are
roadmap items behind the start[genai] extra.
"""

from __future__ import annotations

import re

from start.core.schemas import Status, TestResult, ThresholdSpec
from start.registry import TestContext, register_test

_CITATION_RE = re.compile(r"\[(?:EV|SRC|DOC)-[A-Za-z0-9\-]+\]")
_NUMERIC_SENTENCE_RE = re.compile(r"\d")


@register_test(
    "genai.citation_coverage",
    family="genai",
    name="Citation coverage of quantitative claims",
    requires=(),
    default_params={"warn_uncited": 0.0, "fail_uncited": 0.25},
)
def citation_coverage(
    ctx: TestContext, warn_uncited: float = 0.0, fail_uncited: float = 0.25
) -> TestResult:
    """Fraction of numeric sentences in generated text lacking a citation tag."""
    text: str = str(ctx.extra.get("generated_text", ""))
    if not text:
        return TestResult(
            test_id="genai.citation_coverage",
            test_name="Citation coverage of quantitative claims",
            status=Status.SKIPPED,
            interpretation="No generated_text provided in context.extra; skipped.",
        )
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    numeric = [s for s in sentences if _NUMERIC_SENTENCE_RE.search(s)]
    uncited = [s for s in numeric if not _CITATION_RE.search(s)]
    rate = len(uncited) / max(len(numeric), 1)
    result = TestResult(
        test_id="genai.citation_coverage",
        test_name="Citation coverage of quantitative claims",
        params={"warn_uncited": warn_uncited, "fail_uncited": fail_uncited},
        metrics={
            "numeric_sentences": len(numeric),
            "uncited_numeric_sentences": len(uncited),
            "uncited_rate": round(rate, 6),
        },
        thresholds=[ThresholdSpec(metric="uncited_rate", warn=warn_uncited, fail=fail_uncited)],
        interpretation=(
            f"{len(uncited)} of {len(numeric)} numeric sentences lack an evidence citation."
        ),
        limitations=["Surface-level check; does not verify the cited evidence supports the claim."],
    )
    return result.apply_thresholds()
