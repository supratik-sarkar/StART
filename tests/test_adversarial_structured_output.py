"""Adversarial Structured Output Test Suite.

Verifies strict schema and grounding behavior across adversarial, malformed,
and edge-case LLM outputs:
1. Markdown code fence stripping (with or without language tag).
2. Conversational wrapper text extraction.
3. Invalid JSON syntax rejection without crashing.
4. Missing required top-level or nested fields.
5. Wrong types (e.g. object instead of array).
6. Invalid enum values.
7. Extra unexpected fields.
8. Hallucinated evidence ID rejection.
9. Hallucinated metric path rejection.
10. Prohibited raw numeric prose detection in qualitative text.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from start.core.schemas import EvidenceRecord, Status, TestResult
from start.review.evidence_view import build_checkpoint_evidence_view
from start.review.structured_contract import (
    EvidenceMetricRef,
    FindingType,
    ReviewerFinding,
    StructuredReviewContext,
    StructuredReviewerResponse,
    normalize_structured_json_text,
    validate_and_hydrate_structured_response,
    validate_qualitative_text_cleanliness,
)


def _make_dummy_record(
    evidence_id: str,
    test_id: str,
    metrics: dict[str, Any],
    run_id: str = "RUN-ADV-TEST",
) -> EvidenceRecord:
    rec = EvidenceRecord.from_result(
        TestResult(
            test_id=test_id,
            test_name=test_id,
            status=Status.PASS,
            metrics=metrics,
            params={},
        ),
        run_id=run_id,
    )
    rec.evidence_id = evidence_id
    return rec


@pytest.fixture
def base_context() -> StructuredReviewContext:
    rec = _make_dummy_record(
        evidence_id="EV-111122223333",
        test_id="risk.var",
        metrics={
            "n_exceptions": 2,
            "p_value": 0.45,
            "portfolio_volatility": 0.015,
        },
    )
    view = build_checkpoint_evidence_view(
        checkpoint_title="Market Risk Checkpoint",
        checkpoint_description="VaR and volatility testing",
        domains=(),
        records=[rec],
    )
    return StructuredReviewContext(
        run_id="RUN-ADV-TEST",
        checkpoint_id="Market Risk Checkpoint",
        evidence_view_hash=view.compute_evidence_view_hash(),
        allowed_evidence_ids=("EV-111122223333",),
        records_by_id={"EV-111122223333": rec},
    )


# 1. Markdown code fences
def test_normalize_markdown_fences() -> None:
    raw_with_json_fence = '```json\n{"findings": [], "overall_assessment": "Clear."}\n```'
    normalized = normalize_structured_json_text(raw_with_json_fence)
    assert normalized == '{"findings": [], "overall_assessment": "Clear."}'
    obj = StructuredReviewerResponse.model_validate_json(normalized)
    assert obj.overall_assessment == "Clear."

    raw_with_bare_fence = '```\n{"findings": [], "overall_assessment": "Clean."}\n```'
    assert normalize_structured_json_text(raw_with_bare_fence) == '{"findings": [], "overall_assessment": "Clean."}'


# 2. Conversational wrapper text
def test_normalize_conversational_wrapper() -> None:
    raw_conversational = (
        "Here is my independent model risk review evaluation:\n\n"
        '{"findings": [], "overall_assessment": "Grounded assessment without issues."}\n\n'
        "Please let me know if you need additional metrics analyzed."
    )
    normalized = normalize_structured_json_text(raw_conversational)
    assert normalized.startswith("{") and normalized.endswith("}")
    obj = StructuredReviewerResponse.model_validate_json(normalized)
    assert obj.overall_assessment == "Grounded assessment without issues."


# 3. Invalid JSON syntax
def test_invalid_json_fails_gracefully() -> None:
    malformed_json = '{"findings": [ {"finding_id": "F-01", ... truncated'
    normalized = normalize_structured_json_text(malformed_json)
    with pytest.raises((ValueError, ValidationError, json.JSONDecodeError)):
        StructuredReviewerResponse.model_validate_json(normalized)


# 4. Missing required fields
def test_missing_required_fields() -> None:
    # Missing overall_assessment
    no_overall = '{"findings": []}'
    with pytest.raises(ValidationError):
        StructuredReviewerResponse.model_validate_json(no_overall)

    # Missing findings
    no_findings = '{"overall_assessment": "Summary"}'
    with pytest.raises(ValidationError):
        StructuredReviewerResponse.model_validate_json(no_findings)

    # Finding missing conclusion
    bad_finding = {
        "findings": [
            {
                "finding_id": "F-01",
                "finding_type": "OBSERVED_EVIDENCE",
                "evidence_refs": [],
                "criterion_status": "EVIDENCE_ONLY",
            }
        ],
        "overall_assessment": "Incomplete finding",
    }
    with pytest.raises(ValidationError):
        StructuredReviewerResponse.model_validate_json(json.dumps(bad_finding))


# 5. Wrong types
def test_wrong_types() -> None:
    # findings is dict instead of list
    bad_type = {
        "findings": {"finding_id": "F-01"},
        "overall_assessment": "Wrong type for findings",
    }
    with pytest.raises(ValidationError):
        StructuredReviewerResponse.model_validate_json(json.dumps(bad_type))


# 6. Invalid enum values
def test_invalid_enum_values() -> None:
    # invalid finding_type
    bad_enum = {
        "findings": [
            {
                "finding_id": "F-01",
                "finding_type": "TOTALLY_INVENTED_TYPE",
                "conclusion": "Some qualitative conclusion",
                "evidence_refs": [],
                "criterion_status": "EVIDENCE_ONLY",
            }
        ],
        "overall_assessment": "Enum test",
    }
    with pytest.raises(ValidationError):
        StructuredReviewerResponse.model_validate_json(json.dumps(bad_enum))


# 7. Extra unexpected fields
def test_extra_unexpected_fields() -> None:
    payload = {
        "findings": [],
        "overall_assessment": "Extra fields test",
        "unexpected_top_level_field": "some_value",
    }
    # Should validate and ignore or allow extra field without crash
    resp = StructuredReviewerResponse.model_validate_json(json.dumps(payload))
    assert resp.overall_assessment == "Extra fields test"


# 8. Hallucinated evidence ID
def test_hallucinated_evidence_id(base_context: StructuredReviewContext) -> None:
    resp = StructuredReviewerResponse(
        findings=[
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Claim referencing phantom evidence",
                evidence_refs=[
                    EvidenceMetricRef(
                        evidence_id="EV-PHANTOM-9999",
                        metric_path="metrics.p_value",
                    )
                ],
                criterion_status="EVIDENCE_ONLY",
            )
        ],
        overall_assessment="Phantom evidence reference",
    )
    val_res = validate_and_hydrate_structured_response(resp, base_context)
    assert not val_res.valid
    assert any("EV-PHANTOM-9999" in str(d) for d in val_res.invalid_refs_details)


# 9. Hallucinated metric path
def test_hallucinated_metric_path(base_context: StructuredReviewContext) -> None:
    resp = StructuredReviewerResponse(
        findings=[
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Claim referencing valid EV but non-existent metric",
                evidence_refs=[
                    EvidenceMetricRef(
                        evidence_id="EV-111122223333",
                        metric_path="metrics.hallucinated_metric_never_computed",
                    )
                ],
                criterion_status="EVIDENCE_ONLY",
            )
        ],
        overall_assessment="Hallucinated metric path",
    )
    val_res = validate_and_hydrate_structured_response(resp, base_context)
    assert not val_res.valid
    assert any("hallucinated_metric_never_computed" in str(d) for d in val_res.invalid_refs_details)


# 10. Prohibited raw numeric prose
def test_prohibited_raw_numeric_prose() -> None:
    clean_text = "The Kupiec test demonstrates statistical non-rejection of the VaR model."
    is_clean, violations = validate_qualitative_text_cleanliness(clean_text)
    assert is_clean
    assert violations is None

    leaky_text = "The Kupiec p-value of 0.852109 confirms the model is acceptable."
    is_clean, violations = validate_qualitative_text_cleanliness(leaky_text)
    assert not is_clean
    assert violations is not None
    assert "0.852109" in violations
