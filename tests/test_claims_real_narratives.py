"""Regression corpus for narrative invariance on real-model verbatim narratives.

Contains the six verbatim narratives produced during live sweeps against OpenAI
and DeepSeek, plus adversarial test cases verifying that evasion channels (such as
hiding decimal figures inside brackets like [0.99]) are strictly caught.
"""

from __future__ import annotations

import pytest

from start.attestation import attest_narrative_invariance
from start.attestation.claims import extract_claims

EVIDENCE = [
    {
        "evidence_id": "EV-7f3a1c8b0d21",
        "test_id": "supervised.cohort_metrics_comparison",
        "test_name": "Cohort metric comparison",
        "status": "warn",
        "metrics": {"train_auc": 0.8421, "test_auc": 0.7714, "oos_auc": 0.7602, "gap": 0.0707},
        "thresholds": [{"metric": "gap", "warn": 0.05, "fail": 0.10, "direction": "upper"}],
        "interpretation": "Holdout degradation exceeds the warn threshold.",
    },
    {
        "evidence_id": "EV-2b9e4d7a6c05",
        "test_id": "supervised.calibration",
        "test_name": "Calibration",
        "status": "fail",
        "metrics": {"ece": 0.1382, "brier": 0.1041, "slope": 0.83},
        "thresholds": [{"metric": "ece", "warn": 0.05, "fail": 0.10, "direction": "upper"}],
        "interpretation": "Expected calibration error breaches the fail threshold.",
    },
    {
        "evidence_id": "EV-c04f81a2e937",
        "test_id": "preprocessing.population_stability",
        "test_name": "Population stability",
        "status": "pass",
        "metrics": {"max_psi": 0.0912, "features_above_010": 0},
        "thresholds": [{"metric": "max_psi", "warn": 0.10, "fail": 0.25, "direction": "upper"}],
        "interpretation": "No feature exceeds the PSI warning level.",
    },
]

DETERMINISTIC_NARRATIVE = (
    "Discriminatory power degrades across cohorts: train AUC 0.8421 against test AUC 0.7714 "
    "and out-of-sample 0.7602 [EV-7f3a1c8b0d21]. The resulting gap of 0.0707 exceeds the warn "
    "threshold of 0.05 but remains below the fail threshold of 0.1. Calibration fails: expected "
    "calibration error is 0.1382 against a warn threshold of 0.05 and a fail threshold of 0.1, "
    "alongside a Brier score of 0.1041 and a calibration slope of 0.83 [EV-2b9e4d7a6c05]. "
    "Population stability passes, with a maximum PSI of 0.0912 below the warn threshold of 0.1 "
    "and fail threshold of 0.25, with 0 features above the 0.1 level [EV-c04f81a2e937]."
)

OPENAI_RUN_1 = (
    "The validation report presents three key pieces of evidence regarding model performance and stability. "
    "The first evidence (EV-7f3a1c8b0d21) indicates that holdout degradation exceeds the warning threshold, "
    "with a gap of 0.0707, an out-of-sample AUC of 0.7602, a test AUC of 0.7714, and a train AUC of 0.8421, "
    "resulting in a status of \"warn\" for the cohort metric comparison test [1]. The second evidence (EV-2b9e4d7a6c05) "
    "shows that the expected calibration error (ECE) breaches the fail threshold, with an ECE of 0.1382 and a Brier "
    "score of 0.1041, leading to a status of \"fail\" for the calibration test [2]. In contrast, the third evidence "
    "(EV-c04f81a2e937) confirms that no feature exceeds the PSI warning level, with a maximum PSI of 0.0912 and a "
    "status of \"pass\" for the population stability test [3]."
)

OPENAI_RUN_2 = (
    "The validation report presents three key pieces of evidence regarding model performance and stability. "
    "The first evidence (EV-7f3a1c8b0d21) indicates that holdout degradation exceeds the warning threshold, "
    "with a gap of 0.0707, an out-of-sample AUC of 0.7602, a test AUC of 0.7714, and a train AUC of 0.8421, "
    "resulting in a status of \"warn\" for the cohort metric comparison test [rA]. The second evidence (EV-2b9e4d7a6c05) "
    "shows that the expected calibration error (ECE) breaches the fail threshold, with an ECE of 0.1382, a Brier "
    "score of 0.1041, and a slope of 0.83, leading to a status of \"fail\" for the calibration test [rB]. In contrast, "
    "the third evidence (EV-c04f81a2e937) confirms that no feature exceeds the PSI warning level, with a maximum PSI "
    "of 0.0912 and a status of \"pass\" for the population stability test [rC]."
)

OPENAI_RUN_3 = (
    "The validation report presents three key pieces of evidence regarding model performance and stability. "
    "The first evidence (EV-7f3a1c8b0d21) indicates that holdout degradation exceeds the warning threshold, "
    "with a gap of 0.0707, an out-of-sample AUC of 0.7602, a test AUC of 0.7714, and a train AUC of 0.8421, "
    "resulting in a status of \"warn\" for the cohort metric comparison test [1]. The second evidence (EV-2b9e4d7a6c05) "
    "shows that the expected calibration error (ECE) breaches the fail threshold, with an ECE of 0.1382, a Brier "
    "score of 0.1041, and a slope of 0.83, leading to a status of \"fail\" for the calibration test [2]. In contrast, "
    "the third evidence (EV-c04f81a2e937) confirms that no feature exceeds the PSI warning level, with a maximum PSI "
    "of 0.0912 and a status of \"pass\" for the population stability test [3]."
)

DEEPSEEK_RUN_1 = (
    "The validation evidence shows mixed performance across three tests. The cohort metric comparison test "
    "(EV-7f3a1c8b0d21) returned a status of \"warn\" with a gap of 0.0707, which exceeds the warn threshold of 0.05 "
    "but remains below the fail threshold of 0.1; associated AUC values are 0.8421 (train), 0.7714 (test), and 0.7602 "
    "(out-of-sample). The calibration test (EV-2b9e4d7a6c05) failed, with an expected calibration error (ECE) of 0.1382 "
    "breaching the fail threshold of 0.1, alongside a Brier score of 0.1041 and a slope of 0.83. The population stability "
    "test (EV-c04f81a2e937) passed, with a maximum PSI of 0.0912 below the warn threshold of 0.1 and zero features above "
    "the 0.10 PSI level."
)

DEEPSEEK_RUN_2 = (
    "The cohort metric comparison test (EV-7f3a1c8b0d21) returned a status of \"warn\" because the holdout degradation "
    "gap of 0.0707 exceeds the warn threshold of 0.05 (fail threshold 0.1), with an out-of-sample AUC of 0.7602, test "
    "AUC of 0.7714, and train AUC of 0.8421. The calibration test (EV-2b9e4d7a6c05) failed, as the expected calibration "
    "error (ECE) of 0.1382 breaches the fail threshold of 0.1 (warn threshold 0.05), alongside a Brier score of 0.1041 "
    "and a slope of 0.83. The population stability test (EV-c04f81a2e937) passed, with a maximum PSI of 0.0912 below "
    "the warn threshold of 0.1 (fail threshold 0.25) and zero features above the 0.10 PSI level."
)

DEEPSEEK_RUN_3 = (
    "The validation evidence shows mixed performance across three tests. The cohort metric comparison test "
    "(EV-7f3a1c8b0d21) returned a status of \"warn\" with a gap of 0.0707, which exceeds the warn threshold of 0.05 "
    "but remains below the fail threshold of 0.1; associated AUC values were 0.8421 (train), 0.7714 (test), and 0.7602 "
    "(out-of-sample). The calibration test (EV-2b9e4d7a6c05) failed, with an expected calibration error (ECE) of 0.1382 "
    "breaching the fail threshold of 0.1, alongside a Brier score of 0.1041 and a slope of 0.83. The population stability "
    "test (EV-c04f81a2e937) passed, with a maximum PSI of 0.0912 below the warn threshold of 0.1 and zero features above "
    "the 0.10 PSI level."
)


@pytest.mark.parametrize(
    "narrative, provider_label",
    [
        (OPENAI_RUN_1, "openai / gpt-4o-mini (run 1)"),
        (OPENAI_RUN_2, "openai / gpt-4o-mini (run 2)"),
        (OPENAI_RUN_3, "openai / gpt-4o-mini (run 3)"),
        (DEEPSEEK_RUN_1, "deepseek / deepseek-chat (run 1)"),
        (DEEPSEEK_RUN_2, "deepseek / deepseek-chat (run 2)"),
        (DEEPSEEK_RUN_3, "deepseek / deepseek-chat (run 3)"),
    ],
)
def test_real_sweep_narratives_pass_invariance(narrative: str, provider_label: str) -> None:
    att = attest_narrative_invariance(
        section="model_performance",
        deterministic_narrative=DETERMINISTIC_NARRATIVE,
        model_narrative=narrative,
        evidence=EVIDENCE,
        provider_name=provider_label,
    )
    assert att.invariant is True, f"Failed for {provider_label}: {att.divergences}"
    assert len(att.blocking_divergences()) == 0
    # Zero unbound claims and zero contradictions
    unbound_or_contradiction = [d for d in att.divergences if d.kind in ("unbound", "contradiction")]
    assert len(unbound_or_contradiction) == 0


def test_soft_claim_determiner_phrase_produces_zero_divergences() -> None:
    """Prose determiners like 'three key pieces of evidence' must never produce unbound claims."""
    phrase = "The report presents three key pieces of evidence."
    claims = extract_claims(phrase)
    assert len(claims) == 1
    assert claims[0].soft is True
    assert claims[0].value == 3.0

    att = attest_narrative_invariance(
        section="test",
        deterministic_narrative=phrase,
        model_narrative=phrase,
        evidence=EVIDENCE,
    )
    assert att.invariant is True
    assert len(att.divergences) == 0


def test_adversarial_invented_decimal_fails() -> None:
    """An invented figure not in evidence must trigger a blocking divergence."""
    invented_narrative = (
        "Discriminatory power degrades across cohorts: train AUC 0.8421 against test AUC 0.7714 "
        "and accuracy was 0.913 [EV-7f3a1c8b0d21]."
    )
    att = attest_narrative_invariance(
        section="model_performance",
        deterministic_narrative=DETERMINISTIC_NARRATIVE,
        model_narrative=invented_narrative,
        evidence=EVIDENCE,
    )
    assert att.invariant is False
    assert any(d.kind == "unbound" and d.severity == "high" for d in att.divergences)


def test_adversarial_corrupted_transcription_fails() -> None:
    """A corrupted figure near a real figure must trigger contradiction."""
    corrupted_narrative = (
        "Discriminatory power degrades across cohorts: train AUC 0.8421 against test AUC 0.7914 "
        "and out-of-sample 0.7602 [EV-7f3a1c8b0d21]."
    )
    att = attest_narrative_invariance(
        section="model_performance",
        deterministic_narrative=DETERMINISTIC_NARRATIVE,
        model_narrative=corrupted_narrative,
        evidence=EVIDENCE,
    )
    assert att.invariant is False
    assert any(d.kind == "contradiction" and d.severity == "high" for d in att.divergences)


def test_adversarial_bracketed_decimal_evasion_caught() -> None:
    """A decimal inside brackets [0.99] is NOT a footnote and must be caught as an unbound claim."""
    bracket_evasion = (
        "Cohort metric comparison test had a gap of 0.0707 and score [0.99] [EV-7f3a1c8b0d21]."
    )
    claims = extract_claims(bracket_evasion)
    values = [c.value for c in claims if not c.soft]
    assert 0.99 in values, "0.99 inside brackets must not be masked as a footnote"

    att = attest_narrative_invariance(
        section="model_performance",
        deterministic_narrative=DETERMINISTIC_NARRATIVE,
        model_narrative=bracket_evasion,
        evidence=EVIDENCE,
    )
    assert att.invariant is False
    assert any(d.kind == "unbound" and d.model_value == 0.99 for d in att.divergences)
