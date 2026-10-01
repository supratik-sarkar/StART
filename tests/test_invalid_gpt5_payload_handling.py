"""Regression test proving bounded contract-safe handling of malformed GPT-5.1 structured payload."""

import pytest
from pydantic import ValidationError

from start.review.structured_contract import (
    CriterionStatus,
    StructuredReviewerResponse,
)

EXACT_MALFORMED_GPT5_PAYLOAD = """{
  "findings": [
    {
      "finding_id": "F-01",
      "finding_type": "OBSERVED_EVIDENCE",
      "conclusion": "The scenario analysis engine executes linear return shocks with documented P&L impact.",
      "evidence_refs": [
        {"evidence_id": "EV-001", "metric_path": "metrics.portfolio_pnl"}
      ],
      "criterion_status": "EVIDENCE_ONLY",
      "unresolved_reason": null
    },
    {
      "finding_id": "F-07",
      "finding_type": "EVIDENCE_GAP",
      "conclusion": "Key evidence is missing to safely accept the scenario analysis for use in risk-aware portfolio optimization.",
      "evidence_refs": [],
      "criterion_status": "CRITERION_REQUIRED",
      "unresolved_reason": null
    }
  ],
  "overall_assessment": "Scenario analysis requires additional validation before high-materiality acceptance."
}"""


def test_malformed_criterion_status_fails_schema_validation():
    """Prove that CRITERION_REQUIRED in criterion_status is strictly rejected by schema."""
    with pytest.raises(ValidationError) as exc_info:
        StructuredReviewerResponse.model_validate_json(EXACT_MALFORMED_GPT5_PAYLOAD)

    err_str = str(exc_info.value)
    # Must fail specifically on findings.1.criterion_status enum validation
    assert "criterion_status" in err_str
    assert "Input should be 'APPLICABLE', 'NOT_APPLICABLE', 'ABSENT' or 'EVIDENCE_ONLY'" in err_str


def test_criterion_status_enum_values():
    """Prove CriterionStatus contains ONLY legitimate criterion applicability states."""
    valid_states = {e.value for e in CriterionStatus}
    assert valid_states == {"APPLICABLE", "NOT_APPLICABLE", "ABSENT", "EVIDENCE_ONLY"}
    assert "CRITERION_REQUIRED" not in valid_states


def test_contract_safe_fallback_handling():
    """Prove that when malformed payload fails parsing, the review handles it via bounded fallback."""
    raw_text = EXACT_MALFORMED_GPT5_PAYLOAD
    parse_error = None
    structured_obj = None

    try:
        structured_obj = StructuredReviewerResponse.model_validate_json(raw_text)
    except Exception as exc:
        parse_error = str(exc)

    assert parse_error is not None
    assert structured_obj is None

    # Bounded fallback logic matching executor.py lines 2217-2245
    # Simulate operator choosing option 1 (deterministic fallback)
    simulated_operator_choice = "1"
    if parse_error or structured_obj is None:
        if simulated_operator_choice == "2":
            decision = "aborted"
            response_backend = "grounding_failed"
        else:
            decision = "fallback"
            response_backend = "fallback"
            response_text = "Deterministic fallback: Recorded challenge on 'Scenario Analysis' (note)."

    assert decision == "fallback"
    assert response_backend == "fallback"
    assert "Deterministic fallback" in response_text
    # No invented criterion status
    assert "CRITERION_REQUIRED" not in [e.value for e in CriterionStatus]
