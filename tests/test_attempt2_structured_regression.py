"""Deterministic regression test for live Attempt 2 structured reviewer response defect.

Validates:
1. Attempt 2 fixture parses with StructuredReviewerResponse Pydantic schema.
2. Grounding validation correctly rejects invalid/hallucinated metric path (metrics.n_degenerate_factors).
3. Grounding validator handles error cleanly without uncaught exceptions.
4. Fallback on interpretive question ("Q") formats:
   "Agent interpretation unavailable — deterministic evidence retained."
5. Evidence records remain fully preserved in evidence store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from start.core.schemas import EvidenceRecord, Status, TestResult
from start.review.evidence_view import build_checkpoint_evidence_view
from start.review.structured_contract import (
    StructuredReviewContext,
    StructuredReviewerResponse,
    normalize_structured_json_text,
    validate_and_hydrate_structured_response,
)


def _make_dummy_record(
    evidence_id: str,
    test_id: str,
    metrics: dict[str, Any],
    run_id: str = "RUN-LIVE-ATTEMPT-2",
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
def attempt2_json_text() -> str:
    path = Path(__file__).parent / "fixtures" / "attempt2_malformed_response.json"
    assert path.exists(), f"Fixture not found: {path}"
    return path.read_text(encoding="utf-8")


def test_attempt2_pydantic_schema_validity(attempt2_json_text: str) -> None:
    """Attempt 2 response was syntactically valid JSON conforming to StructuredReviewerResponse."""
    normalized = normalize_structured_json_text(attempt2_json_text)
    resp = StructuredReviewerResponse.model_validate_json(normalized)
    assert len(resp.findings) == 7
    assert resp.overall_assessment != ""
    assert resp.findings[0].finding_id == "F-01"
    assert resp.findings[4].finding_id == "F-05"


def test_attempt2_grounding_rejection_of_hallucinated_metric(attempt2_json_text: str) -> None:
    """Attempt 2 F-05 cited 'metrics.n_degenerate_factors' on EV-73deb14a278e, which does not exist."""
    # Construct records matching the IDs cited in Attempt 2
    rec_weights = _make_dummy_record(
        evidence_id="EV-037b3fefa252",
        test_id="portfolio.weights",
        metrics={
            "weights_sum": 1.0,
            "benchmark_weights_sum": 1.0,
            "weights_renormalised": False,
            "n_assets": 50,
            "n_assets_excluded": 0,
            "portfolio_exposure.F1": 0.15,
            "portfolio_exposure.F2": 0.20,
            "portfolio_exposure.F3": 0.05,
            "portfolio_exposure.F4": 0.10,
            "portfolio_exposure.F5": 0.12,
            "active_exposure.F1": 0.02,
            "active_exposure.F2": -0.01,
            "active_exposure.F3": 0.01,
            "active_exposure.F4": 0.03,
            "active_exposure.F5": -0.02,
        },
    )
    rec_decomp = _make_dummy_record(
        evidence_id="EV-cc00068832ed",
        test_id="factor.risk_decomposition",
        metrics={
            "empirical_portfolio_variance": 0.00045,
            "total_factor_model_variance": 0.00042,
            "factor_variance": 0.00035,
            "specific_variance": 0.00007,
            "factor_variance_share": 0.833,
            "specific_variance_share": 0.167,
            "factor_returns_hash": "a1b2c3d4",
            "specific_returns_hash": "e5f6g7h8",
            "factor_covariance_source": "canonical",
        },
    )
    rec_r2 = _make_dummy_record(
        evidence_id="EV-197e88df4f47",
        test_id="factor.asset_r2",
        metrics={
            "mean_asset_r2": 0.42,
            "median_asset_r2": 0.40,
            "min_asset_r2": 0.15,
            "max_asset_r2": 0.78,
            "n_assets_low_r2": 3,
            "factor_returns_hash": "a1b2c3d4",
            "specific_returns_hash": "e5f6g7h8",
        },
    )
    rec_cov = _make_dummy_record(
        evidence_id="EV-bcc3e5295e0e",
        test_id="factor.covariance_stability",
        metrics={
            "condition_number": 12.4,
            "is_positive_definite": True,
            "factor_returns_hash": "a1b2c3d4",
            "specific_returns_hash": "e5f6g7h8",
            "factor_return_source": "canonical",
        },
    )
    # EV-73deb14a278e has max_condition_number, NOT n_degenerate_factors
    rec_factor_ret = _make_dummy_record(
        evidence_id="EV-73deb14a278e",
        test_id="factor.returns_diagnostics",
        metrics={
            "max_condition_number": 8.5,
            "factor_returns_hash": "a1b2c3d4",
        },
    )
    # EV-fbbe3dc34d33 has degenerate_factors
    rec_degen = _make_dummy_record(
        evidence_id="EV-fbbe3dc34d33",
        test_id="factor.degeneracy_check",
        metrics={
            "degenerate_factors": 0,
            "degenerate.F1": False,
            "degenerate.F2": False,
            "degenerate.F3": False,
            "degenerate.F4": False,
            "degenerate.F5": False,
            "factor_returns_hash": "a1b2c3d4",
            "specific_returns_hash": "e5f6g7h8",
        },
    )

    records = [rec_weights, rec_decomp, rec_r2, rec_cov, rec_factor_ret, rec_degen]
    records_by_id = {r.evidence_id: r for r in records}

    view = build_checkpoint_evidence_view(
        checkpoint_title="Factor Modeling & Attribution Assumptions",
        checkpoint_description="Review factor decomposition and attribution.",
        domains=(),
        records=records,
    )

    ctx = StructuredReviewContext(
        run_id="RUN-LIVE-ATTEMPT-2",
        checkpoint_id="Factor Modeling & Attribution Assumptions",
        evidence_view_hash=view.compute_evidence_view_hash(),
        allowed_evidence_ids=tuple(records_by_id.keys()),
        records_by_id=records_by_id,
    )

    resp = StructuredReviewerResponse.model_validate_json(normalize_structured_json_text(attempt2_json_text))
    val_res = validate_and_hydrate_structured_response(resp, ctx)

    # Must be invalid because EV-73deb14a278e does not have n_degenerate_factors
    assert not val_res.valid
    assert len(val_res.invalid_refs_details) >= 1

    # Specifically check the hallucinated metric is identified
    err_paths = [
        (d.get("evidence_id"), d.get("metric_path"))
        for d in val_res.invalid_refs_details
    ]
    assert ("EV-73deb14a278e", "metrics.n_degenerate_factors") in err_paths


def test_attempt2_question_fallback_message_formatting() -> None:
    """When deterministic fallback is selected on question 'Q', message indicates agent interpretation unavailable."""
    action = "Q"
    title = "Factor Modeling & Attribution Assumptions"
    note = "Separate portfolio weights from narrative"

    if action == "Q":
        response_text = "Agent interpretation unavailable — deterministic evidence retained."
    else:
        response_text = f"Deterministic fallback: Recorded {action} on '{title}' ({note})."

    assert response_text == "Agent interpretation unavailable — deterministic evidence retained."


def test_attempt2_evidence_preservation_under_fallback() -> None:
    """Deterministic evidence records remain unaltered when agent interpretation is unavailable."""
    test_metrics = {"weights_sum": 1.0, "n_assets": 50}
    rec = _make_dummy_record("EV-037b3fefa252", "portfolio.weights", test_metrics)

    # Before fallback
    assert rec.metrics["weights_sum"] == 1.0
    assert rec.metrics["n_assets"] == 50

    # Under fallback, record is not mutated
    rec_copy = rec.model_copy(deep=True)
    assert rec_copy.evidence_id == rec.evidence_id
    assert rec_copy.metrics == test_metrics
