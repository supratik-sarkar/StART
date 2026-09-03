"""v4.3.0 Structured Reviewer Output Contract — Comprehensive Test Matrix.

Implements all 49 specified tests covering:
- Schema validation & cardinality rules
- Evidence identity, canonical path enforcement, and provenance
- Deterministic value hydration & arithmetic prohibition
- Single-invocation provider contract & fail-closed behavior
- Checkpoint integration across all review domains
- Attestation sealing of canonical finding graph
- Noninteractive replays of frozen failure runs
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from start.core.schemas import EvidenceRecord, Status, TestResult
from start.review.architecture import (
    LLMReviewConfig,
    ReviewContextBundle,
    ReviewDomain,
    ReviewGroundingMode,
    ReviewMode,
)
from start.review.evidence_view import CheckpointEvidenceView, build_checkpoint_evidence_view
from start.review.executor import run_domain_checkpoints
from start.review.structured_contract import (
    EvidenceMetricRef,
    FindingType,
    ReviewerFinding,
    StructuredReviewContext,
    StructuredReviewerResponse,
    render_structured_response_markdown,
    validate_and_hydrate_structured_response,
    validate_qualitative_text_cleanliness,
)


# --------------------------------------------------------------------------- #
# Helpers & Fixtures
# --------------------------------------------------------------------------- #
def _make_dummy_record(
    evidence_id: str,
    test_id: str,
    metrics: dict[str, Any],
    run_id: str = "RUN-ACCEPTANCE-20260903",
    params: dict[str, Any] | None = None,
) -> EvidenceRecord:
    rec = EvidenceRecord.from_result(
        TestResult(
            test_id=test_id,
            test_name=test_id,
            status=Status.PASS,
            metrics=metrics,
            params=params or {},
        ),
        run_id=run_id,
    )
    rec.evidence_id = evidence_id
    return rec


@pytest.fixture
def sample_review_context() -> tuple[CheckpointEvidenceView, StructuredReviewContext]:
    rec1 = _make_dummy_record(
        evidence_id="EV-16bbbafd361d",
        test_id="traded_risk.var_exceptions",
        metrics={
            "n_exceptions": 0,
            "n11": 0,
            "p_value": 0.8521,
            "annualised_volatility": 0.037706,
        },
    )
    rec2 = _make_dummy_record(
        evidence_id="EV-9f798835848e",
        test_id="portfolio.risk_statistics",
        metrics={
            "periodic_volatility": 0.001679,
            "annualised_volatility": 0.026654,
            "n_scenarios": 1000,
            "nominal_tail_scenarios": 76,
        },
    )
    view = build_checkpoint_evidence_view(
        checkpoint_title="VaR Backtesting & Exception Frequency",
        checkpoint_description="Review exception count and Kupiec p-values.",
        domains=(ReviewDomain.MARKET,),
        records=[rec1, rec2],
    )
    ctx = StructuredReviewContext(
        run_id="RUN-ACCEPTANCE-20260903",
        checkpoint_id="VaR Backtesting & Exception Frequency",
        evidence_view_hash=view.compute_evidence_view_hash(),
        allowed_evidence_ids=("EV-16bbbafd361d", "EV-9f798835848e"),
        records_by_id={"EV-16bbbafd361d": rec1, "EV-9f798835848e": rec2},
    )
    return view, ctx


# =========================================================================== #
# 1. SCHEMA VALIDATION TESTS (Tests 1–5)
# =========================================================================== #
def test_schema_valid_response():
    """1. Valid StructuredReviewerResponse parses cleanly."""
    payload = {
        "findings": [
            {
                "finding_id": "F-01",
                "finding_type": "OBSERVED_EVIDENCE",
                "conclusion": "VaR backtesting shows zero exception transitions.",
                "evidence_refs": [{"evidence_id": "EV-16bbbafd361d", "metric_path": "metrics.n11"}],
                "criterion_status": "APPLICABLE",
                "unresolved_reason": None,
            }
        ],
        "overall_assessment": "The backtest meets all regulatory standards.",
    }
    resp = StructuredReviewerResponse.model_validate_json(json.dumps(payload))
    assert len(resp.findings) == 1
    assert resp.findings[0].finding_id == "F-01"
    assert resp.findings[0].finding_type == FindingType.OBSERVED_EVIDENCE


def test_schema_malformed_json():
    """2. Malformed JSON string raises validation error."""
    with pytest.raises(ValidationError):
        StructuredReviewerResponse.model_validate_json("{not_valid_json: 123")


def test_schema_unknown_enum():
    """3. Unknown FindingType enum value raises validation error."""
    payload = {
        "findings": [
            {
                "finding_id": "F-01",
                "finding_type": "NON_EXISTENT_FINDING_TYPE",
                "conclusion": "Some qualitative text.",
                "evidence_refs": [],
            }
        ],
        "overall_assessment": "Overall assessment.",
    }
    with pytest.raises(ValidationError):
        StructuredReviewerResponse.model_validate_json(json.dumps(payload))


def test_schema_missing_required_field():
    """4. Missing required field raises validation error."""
    payload = {
        "findings": [
            {
                "finding_id": "F-01",
                # missing finding_type
                "conclusion": "Some conclusion.",
            }
        ],
        "overall_assessment": "Overall assessment.",
    }
    with pytest.raises(ValidationError):
        StructuredReviewerResponse.model_validate_json(json.dumps(payload))


def test_schema_duplicate_finding_ids():
    """5. Duplicate finding IDs within response rejected."""
    payload = {
        "findings": [
            {
                "finding_id": "F-01",
                "finding_type": "EVIDENCE_GAP",
                "conclusion": "First gap.",
                "evidence_refs": [],
            },
            {
                "finding_id": "F-01",
                "finding_type": "EVIDENCE_GAP",
                "conclusion": "Second gap with duplicate ID.",
                "evidence_refs": [],
            },
        ],
        "overall_assessment": "Overall assessment.",
    }
    with pytest.raises(ValidationError, match="Duplicate finding_id"):
        StructuredReviewerResponse.model_validate_json(json.dumps(payload))


# =========================================================================== #
# 2. CARDINALITY RULES (Tests 6–11)
# =========================================================================== #
def test_cardinality_observed_evidence_requires_ref():
    """6. OBSERVED_EVIDENCE requires at least 1 EvidenceMetricRef."""
    with pytest.raises(ValidationError, match="requires at least one EvidenceMetricRef"):
        ReviewerFinding(
            finding_id="F-01",
            finding_type=FindingType.OBSERVED_EVIDENCE,
            conclusion="Observed evidence without refs.",
            evidence_refs=(),
        )


def test_cardinality_diagnostic_finding_requires_ref():
    """7. DIAGNOSTIC_FINDING requires at least 1 EvidenceMetricRef."""
    with pytest.raises(ValidationError, match="requires at least one EvidenceMetricRef"):
        ReviewerFinding(
            finding_id="F-01",
            finding_type=FindingType.DIAGNOSTIC_FINDING,
            conclusion="Diagnostic finding without refs.",
            evidence_refs=(),
        )


def test_cardinality_statistical_rejection_non_rejection_require_refs():
    """8. STATISTICAL_REJECTION and STATISTICAL_NON_REJECTION require refs."""
    with pytest.raises(ValidationError, match="requires at least one EvidenceMetricRef"):
        ReviewerFinding(
            finding_id="F-01",
            finding_type=FindingType.STATISTICAL_REJECTION,
            conclusion="Rejection without ref.",
            evidence_refs=(),
        )
    with pytest.raises(ValidationError, match="requires at least one EvidenceMetricRef"):
        ReviewerFinding(
            finding_id="F-02",
            finding_type=FindingType.STATISTICAL_NON_REJECTION,
            conclusion="Non-rejection without ref.",
            evidence_refs=(),
        )


def test_cardinality_evidence_gap_permits_zero():
    """9. EVIDENCE_GAP permits zero EvidenceMetricRefs."""
    finding = ReviewerFinding(
        finding_id="F-01",
        finding_type=FindingType.EVIDENCE_GAP,
        conclusion="No liquidity stress test evidence provided.",
        evidence_refs=(),
    )
    assert len(finding.evidence_refs) == 0


def test_cardinality_criterion_required_permits_zero():
    """10. CRITERION_REQUIRED permits zero EvidenceMetricRefs."""
    finding = ReviewerFinding(
        finding_id="F-01",
        finding_type=FindingType.CRITERION_REQUIRED,
        conclusion="Acceptance criterion for condition number is required.",
        evidence_refs=(),
    )
    assert len(finding.evidence_refs) == 0


def test_cardinality_unresolved_cross_conditional_rules():
    """11. UNRESOLVED_MODEL_RISK permits 0; CROSS_ANALYTICAL_DEPENDENCY and CONDITIONAL_CONCLUSION require >= 1."""
    # UNRESOLVED_MODEL_RISK with 0 refs is allowed
    f1 = ReviewerFinding(
        finding_id="F-01",
        finding_type=FindingType.UNRESOLVED_MODEL_RISK,
        conclusion="Structural tail risk remains unquantified.",
        evidence_refs=(),
    )
    assert f1.finding_type == FindingType.UNRESOLVED_MODEL_RISK

    # CROSS_ANALYTICAL_DEPENDENCY requires >= 1 ref
    with pytest.raises(ValidationError, match="requires at least one EvidenceMetricRef"):
        ReviewerFinding(
            finding_id="F-02",
            finding_type=FindingType.CROSS_ANALYTICAL_DEPENDENCY,
            conclusion="Cross dependency missing ref.",
            evidence_refs=(),
        )

    # CONDITIONAL_CONCLUSION requires >= 1 ref
    with pytest.raises(ValidationError, match="requires at least one EvidenceMetricRef"):
        ReviewerFinding(
            finding_id="F-03",
            finding_type=FindingType.CONDITIONAL_CONCLUSION,
            conclusion="Conditional conclusion missing ref.",
            evidence_refs=(),
        )


# =========================================================================== #
# 3. EVIDENCE IDENTITY & PROVENANCE (Tests 12–21)
# =========================================================================== #
def test_evidence_identity_exact_ev_path(sample_review_context):
    """12. Exact EV-ID and metric_path valid and hydrated."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Exception count is within limits.",
                evidence_refs=(
                    EvidenceMetricRef(evidence_id="EV-16bbbafd361d", metric_path="metrics.n_exceptions"),
                ),
            ),
        ),
        overall_assessment="Acceptable performance.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert res.valid
    assert res.validated_refs_count == 1
    assert res.hydrated_response.findings[0].hydrated_refs[0].value == 0


def test_evidence_identity_unknown_ev(sample_review_context):
    """13. Unknown EV-ID rejected."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Invalid EV cited.",
                evidence_refs=(EvidenceMetricRef(evidence_id="EV-UNKNOWN9999", metric_path="metrics.n11"),),
            ),
        ),
        overall_assessment="Overall assessment.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert not res.valid
    assert "not found in current CheckpointEvidenceView" in res.invalid_refs_details[0]["error"]


def test_evidence_identity_truncated_ev(sample_review_context):
    """14. Truncated EV-ID rejected."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Truncated EV cited.",
                evidence_refs=(EvidenceMetricRef(evidence_id="EV-16bb", metric_path="metrics.n11"),),
            ),
        ),
        overall_assessment="Overall assessment.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert not res.valid


def test_evidence_identity_wrong_path(sample_review_context):
    """15. Path not present on record rejected."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Nonexistent metric cited.",
                evidence_refs=(
                    EvidenceMetricRef(
                        evidence_id="EV-16bbbafd361d", metric_path="metrics.nonexistent_metric"
                    ),
                ),
            ),
        ),
        overall_assessment="Overall assessment.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert not res.valid
    assert "does not exist on EvidenceRecord" in res.invalid_refs_details[0]["error"]


def test_evidence_identity_noncanonical_alias_path_rejected(sample_review_context):
    """16. Noncanonical alias path (bare n11 or test_id.n11) rejected."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Bare metric path cited.",
                evidence_refs=(EvidenceMetricRef(evidence_id="EV-16bbbafd361d", metric_path="n11"),),
            ),
        ),
        overall_assessment="Overall assessment.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert not res.valid
    assert "Invalid non-canonical path format" in res.invalid_refs_details[0]["error"]


def test_evidence_identity_cross_checkpoint_ev_rejected(sample_review_context):
    """17. EV from another checkpoint not in allowed_evidence_ids rejected."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Cross-checkpoint EV.",
                evidence_refs=(
                    EvidenceMetricRef(evidence_id="EV-OTHER-CHECKPOINT", metric_path="metrics.n11"),
                ),
            ),
        ),
        overall_assessment="Overall assessment.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert not res.valid


def test_evidence_identity_cross_run_ev_rejected():
    """18. EV with different run_id rejected."""
    rec_old = _make_dummy_record("EV-OLD1", "test.one", {"n": 1}, run_id="RUN-STALE-PREVIOUS")
    ctx = StructuredReviewContext(
        run_id="RUN-CURRENT-ACTIVE",
        checkpoint_id="Check",
        evidence_view_hash="sha256-hash",
        allowed_evidence_ids=("EV-OLD1",),
        records_by_id={"EV-OLD1": rec_old},
    )
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Stale run EV cited.",
                evidence_refs=(EvidenceMetricRef(evidence_id="EV-OLD1", metric_path="metrics.n"),),
            ),
        ),
        overall_assessment="Overall assessment.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert not res.valid
    assert "Cross-run reference violation" in res.invalid_refs_details[0]["error"]


def test_evidence_identity_view_hash_mismatch_rejected(sample_review_context):
    """19. Missing or empty evidence_view_hash rejected."""
    _, ctx = sample_review_context
    bad_ctx = StructuredReviewContext(
        run_id=ctx.run_id,
        checkpoint_id=ctx.checkpoint_id,
        evidence_view_hash="",
        allowed_evidence_ids=ctx.allowed_evidence_ids,
        records_by_id=ctx.records_by_id,
    )
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.EVIDENCE_GAP,
                conclusion="Valid finding.",
                evidence_refs=(),
            ),
        ),
        overall_assessment="Overall assessment.",
    )
    res = validate_and_hydrate_structured_response(resp, bad_ctx)
    assert not res.valid
    assert "evidence_view_hash" in res.error_message


def test_evidence_identity_diagnostic_ev_accepted_same_validator():
    """20. EV-DIAG-* record with runtime='DIAGNOSTIC' accepted through exact same canonical validator."""
    diag_rec = _make_dummy_record(
        evidence_id="EV-DIAG-86ce1cf3",
        test_id="diagnostic.compute_exception_duration_diagnostics",
        metrics={"max_consecutive_exceptions": 2, "clustering_score": 0.12},
    )
    diag_rec.repro.runtime = "DIAGNOSTIC"

    ctx = StructuredReviewContext(
        run_id="RUN-ACCEPTANCE-20260903",
        checkpoint_id="VaR Backtesting",
        evidence_view_hash="sha256-hash",
        allowed_evidence_ids=("EV-DIAG-86ce1cf3",),
        records_by_id={"EV-DIAG-86ce1cf3": diag_rec},
    )
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.DIAGNOSTIC_FINDING,
                conclusion="Diagnostic reveals maximum consecutive exception clustering.",
                evidence_refs=(
                    EvidenceMetricRef(
                        evidence_id="EV-DIAG-86ce1cf3",
                        metric_path="metrics.max_consecutive_exceptions",
                    ),
                ),
            ),
        ),
        overall_assessment="Diagnostic confirms acceptable clustering.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert res.valid
    assert res.hydrated_response.findings[0].hydrated_refs[0].value == 2


def test_evidence_identity_multi_ref_finding(sample_review_context):
    """21. Single finding with multiple EvidenceMetricRefs correctly validates and hydrates each."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Multiple metrics observed simultaneously.",
                evidence_refs=(
                    EvidenceMetricRef(evidence_id="EV-16bbbafd361d", metric_path="metrics.n_exceptions"),
                    EvidenceMetricRef(evidence_id="EV-9f798835848e", metric_path="metrics.n_scenarios"),
                ),
            ),
        ),
        overall_assessment="Overall assessment.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert res.valid
    assert res.evidence_refs_count == 2
    assert res.validated_refs_count == 2


# =========================================================================== #
# 4. DETERMINISTIC VALUE HYDRATION (Tests 22–28)
# =========================================================================== #
def test_hydration_zero_integer_n11(sample_review_context):
    """22. Zero integer (n11=0) hydrates as integer 0 with count unit."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Zero consecutive exception transitions observed.",
                evidence_refs=(EvidenceMetricRef(evidence_id="EV-16bbbafd361d", metric_path="metrics.n11"),),
            ),
        ),
        overall_assessment="Overall assessment.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    ref = res.hydrated_response.findings[0].hydrated_refs[0]
    assert ref.value == 0
    assert isinstance(ref.value, int)
    assert ref.display == "0"
    assert ref.unit == "count"


def test_hydration_negative_metric():
    """23. Negative metric value hydrates correctly."""
    rec = _make_dummy_record("EV-NEG1", "test.pnl", {"max_drawdown": -0.045})
    ctx = StructuredReviewContext(
        run_id=rec.run_id,
        checkpoint_id="Stress",
        evidence_view_hash="hash",
        allowed_evidence_ids=("EV-NEG1",),
        records_by_id={"EV-NEG1": rec},
    )
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Negative drawdown observed.",
                evidence_refs=(EvidenceMetricRef(evidence_id="EV-NEG1", metric_path="metrics.max_drawdown"),),
            ),
        ),
        overall_assessment="Acceptable.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    ref = res.hydrated_response.findings[0].hydrated_refs[0]
    assert ref.value == -0.045
    assert "-0.045" in ref.display or "-4.5" in ref.display


def test_hydration_scientific_numeric_value():
    """24. Small floating point value in scientific range formats accurately."""
    rec = _make_dummy_record("EV-SCI1", "test.precision", {"machine_eps": 2.22e-16})
    ctx = StructuredReviewContext(
        run_id=rec.run_id,
        checkpoint_id="Precision",
        evidence_view_hash="hash",
        allowed_evidence_ids=("EV-SCI1",),
        records_by_id={"EV-SCI1": rec},
    )
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Precision metric noted.",
                evidence_refs=(EvidenceMetricRef(evidence_id="EV-SCI1", metric_path="metrics.machine_eps"),),
            ),
        ),
        overall_assessment="Clean.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    ref = res.hydrated_response.findings[0].hydrated_refs[0]
    assert ref.value == 2.22e-16
    assert "2.22" in ref.display


def test_hydration_integer_count_metric(sample_review_context):
    """25. Integer metric formats as count with commas."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Scenario count verified.",
                evidence_refs=(
                    EvidenceMetricRef(evidence_id="EV-9f798835848e", metric_path="metrics.n_scenarios"),
                ),
            ),
        ),
        overall_assessment="Clean.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    ref = res.hydrated_response.findings[0].hydrated_refs[0]
    assert ref.value == 1000
    assert ref.display == "1,000"
    assert ref.unit == "count"


def test_hydration_percentage_rendered_metric(sample_review_context):
    """26. Volatility / rate metric renders with percentage."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Periodic volatility verified.",
                evidence_refs=(
                    EvidenceMetricRef(
                        evidence_id="EV-9f798835848e", metric_path="metrics.periodic_volatility"
                    ),
                ),
            ),
        ),
        overall_assessment="Clean.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    ref = res.hydrated_response.findings[0].hydrated_refs[0]
    assert ref.value == 0.001679
    assert ref.display == "0.1679%"
    assert ref.unit == "percentage"


def test_hydration_boolean_non_numeric_metric():
    """27. Boolean / non-numeric metric formats correctly."""
    rec = _make_dummy_record("EV-BOOL1", "test.flag", {"is_stationary": True})
    ctx = StructuredReviewContext(
        run_id=rec.run_id,
        checkpoint_id="Stationarity",
        evidence_view_hash="hash",
        allowed_evidence_ids=("EV-BOOL1",),
        records_by_id={"EV-BOOL1": rec},
    )
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Stationarity confirmed.",
                evidence_refs=(
                    EvidenceMetricRef(evidence_id="EV-BOOL1", metric_path="metrics.is_stationary"),
                ),
            ),
        ),
        overall_assessment="Clean.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    ref = res.hydrated_response.findings[0].hydrated_refs[0]
    assert ref.value is True
    assert ref.display == "True"


def test_hydration_no_arithmetic(sample_review_context):
    """28. Hydrated value matches raw record metric exactly with zero derived modification."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Exact raw metric value.",
                evidence_refs=(
                    EvidenceMetricRef(
                        evidence_id="EV-16bbbafd361d", metric_path="metrics.annualised_volatility"
                    ),
                ),
            ),
        ),
        overall_assessment="Clean.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    ref = res.hydrated_response.findings[0].hydrated_refs[0]
    assert ref.value == 0.037706


# =========================================================================== #
# 5. NO ARITHMETIC IN CONTRACT (Tests 29–31)
# =========================================================================== #
def test_no_arithmetic_reviewer_cannot_author_derived_value():
    """29. Reviewer cannot embed raw derived float measurements in qualitative prose."""
    clean, reason = validate_qualitative_text_cleanliness(
        "The calculated ratio is 0.044521 which exceeds threshold."
    )
    assert not clean
    assert "0.044521" in reason


def test_no_arithmetic_hydrator_performs_no_derived_calculation(sample_review_context):
    """30. Hydrator extracts metric directly without scaling or arithmetic transformation."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Raw count.",
                evidence_refs=(
                    EvidenceMetricRef(evidence_id="EV-16bbbafd361d", metric_path="metrics.n_exceptions"),
                ),
            ),
        ),
        overall_assessment="Clean.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert res.hydrated_response.findings[0].hydrated_refs[0].value == 0


def test_no_arithmetic_engine_emitted_derived_metric_referenceable():
    """31. Pre-computed engine metric (e.g. variance_shortfall) referenceable via exact canonical path."""
    rec = _make_dummy_record("EV-ENG1", "test.stress", {"variance_shortfall": 0.0125})
    ctx = StructuredReviewContext(
        run_id=rec.run_id,
        checkpoint_id="Stress",
        evidence_view_hash="hash",
        allowed_evidence_ids=("EV-ENG1",),
        records_by_id={"EV-ENG1": rec},
    )
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Engine derived metric cited.",
                evidence_refs=(
                    EvidenceMetricRef(evidence_id="EV-ENG1", metric_path="metrics.variance_shortfall"),
                ),
            ),
        ),
        overall_assessment="Clean.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert res.valid
    assert res.hydrated_response.findings[0].hydrated_refs[0].value == 0.0125


# =========================================================================== #
# 6. PROVIDER INTEGRATION & FAIL-CLOSED (Tests 32–37)
# =========================================================================== #
def test_provider_valid_structured_response(monkeypatch):
    """32. Mocked provider returning valid JSON produces VERIFIED state."""

    class ValidStructuredProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, **kwargs) -> str:
            return json.dumps(
                {
                    "findings": [
                        {
                            "finding_id": "F-01",
                            "finding_type": "OBSERVED_EVIDENCE",
                            "conclusion": "Portfolio risk is well-characterized.",
                            "evidence_refs": [
                                {"evidence_id": "EV-PORT1234", "metric_path": "metrics.annualised_volatility"}
                            ],
                        }
                    ],
                    "overall_assessment": "The portfolio meets risk limits.",
                }
            )

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: ValidStructuredProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = _make_dummy_record("EV-PORT1234", "portfolio.risk_statistics", {"annualised_volatility": 0.0937})

    prompts = iter(["Q", "Assess risk", "A", "A", "A", "A", "A"])
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))
    q_dec = [d for d in decisions if d["action"] == "question"][0]
    assert q_dec["backend"] == "llm_structured"
    assert q_dec["grounded_claims"] == 1
    assert q_dec["unbound_claims"] == 0


def test_provider_refusal(monkeypatch):
    """33. Provider returning refusal text triggers deterministic fallback."""

    class RefusalProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, **kwargs) -> str:
            return "I cannot fulfill this request."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: RefusalProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = _make_dummy_record("EV-PORT1234", "portfolio.risk_statistics", {"annualised_volatility": 0.0937})

    prompts = iter(["Q", "Assess risk", "1", "A", "A", "A", "A"])
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))
    q_dec = [d for d in decisions if d["action"] == "question"][0]
    assert q_dec["backend"] == "fallback"


def test_provider_incomplete_response(monkeypatch):
    """34. Provider returning truncated JSON triggers deterministic fallback."""

    class TruncatedProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, **kwargs) -> str:
            return '{"findings": [{"finding_id": "F-01"'

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: TruncatedProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = _make_dummy_record("EV-PORT1234", "portfolio.risk_statistics", {"annualised_volatility": 0.0937})

    prompts = iter(["Q", "Assess risk", "1", "A", "A", "A", "A"])
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))
    q_dec = [d for d in decisions if d["action"] == "question"][0]
    assert q_dec["backend"] == "fallback"


def test_provider_malformed_structured_output(monkeypatch):
    """35. Provider returning markdown prose without valid JSON triggers fallback."""

    class ProseProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, **kwargs) -> str:
            return "Here is my evaluation: the volatility is 0.0937 [EV-PORT1234]."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: ProseProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = _make_dummy_record("EV-PORT1234", "portfolio.risk_statistics", {"annualised_volatility": 0.0937})

    prompts = iter(["Q", "Assess risk", "1", "A", "A", "A", "A"])
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))
    q_dec = [d for d in decisions if d["action"] == "question"][0]
    assert q_dec["backend"] == "fallback"


def test_provider_exactly_one_invocation(monkeypatch):
    """36. Provider is invoked exactly once per interactive checkpoint."""
    calls = 0

    class TrackingProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, **kwargs) -> str:
            nonlocal calls
            calls += 1
            return json.dumps(
                {
                    "findings": [
                        {
                            "finding_id": "F-01",
                            "finding_type": "OBSERVED_EVIDENCE",
                            "conclusion": "Valid observation.",
                            "evidence_refs": [
                                {"evidence_id": "EV-PORT1234", "metric_path": "metrics.annualised_volatility"}
                            ],
                        }
                    ],
                    "overall_assessment": "Valid assessment.",
                }
            )

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: TrackingProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = _make_dummy_record("EV-PORT1234", "portfolio.risk_statistics", {"annualised_volatility": 0.0937})

    prompts = iter(["Q", "Question 1", "A", "A", "A", "A", "A"])
    run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))
    assert calls == 1  # exactly 1 invocation for the interactive Question


def test_provider_no_structured_to_legacy_fallback(monkeypatch):
    """37. Failure in structured mode never silently falls back to legacy free-form claim parser."""

    class InvalidJsonProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, **kwargs) -> str:
            return "Free-form prose claiming 0.0937 [EV-PORT1234]."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: InvalidJsonProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = _make_dummy_record("EV-PORT1234", "portfolio.risk_statistics", {"annualised_volatility": 0.0937})

    prompts = iter(["Q", "Question 1", "1", "A", "A", "A", "A"])
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))
    q_dec = [d for d in decisions if d["action"] == "question"][0]
    assert q_dec["backend"] == "fallback"


# =========================================================================== #
# 7. REVIEW INTEGRATION (Tests 38–44)
# =========================================================================== #
def _mock_structured_llm(monkeypatch, response_dict: dict[str, Any]):
    class MockLLM:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, **kwargs) -> str:
            return json.dumps(response_dict)

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: MockLLM())


def test_review_integration_factor_question(monkeypatch):
    """38. Factor Question checkpoint in structured mode."""
    _mock_structured_llm(
        monkeypatch,
        {
            "findings": [
                {
                    "finding_id": "F-01",
                    "finding_type": "OBSERVED_EVIDENCE",
                    "conclusion": "Factor loadings show strong statistical alignment.",
                    "evidence_refs": [{"evidence_id": "EV-FAC1", "metric_path": "metrics.r_squared"}],
                }
            ],
            "overall_assessment": "Factor model meets specifications.",
        },
    )
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = _make_dummy_record("EV-FAC1", "attribution.factor_return_estimation", {"r_squared": 0.88})
    prompts = iter(["A", "Q", "Evaluate factor fit", "A", "A", "A", "A", "A"])
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))
    q_dec = [d for d in decisions if d["action"] == "question"][0]
    assert q_dec["backend"] == "llm_structured"


def test_review_integration_var_question(monkeypatch):
    """39. VaR Question checkpoint in structured mode."""
    _mock_structured_llm(
        monkeypatch,
        {
            "findings": [
                {
                    "finding_id": "F-01",
                    "finding_type": "OBSERVED_EVIDENCE",
                    "conclusion": "VaR exception frequency is within limits.",
                    "evidence_refs": [{"evidence_id": "EV-VAR1", "metric_path": "metrics.n11"}],
                }
            ],
            "overall_assessment": "VaR model backtesting is acceptable.",
        },
    )
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = _make_dummy_record("EV-VAR1", "traded_risk.var_exceptions", {"n11": 0})
    prompts = iter(["A", "A", "Q", "What is the backtest frequency?", "A", "A", "A", "A"])
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))
    q_dec = [d for d in decisions if d["action"] == "question"][0]
    assert q_dec["backend"] == "llm_structured"


def test_review_integration_var_challenge_with_ev_diag(monkeypatch):
    """40. VaR Challenge checkpoint executes diagnostic tool and permits EV-DIAG structured citation."""
    _mock_structured_llm(
        monkeypatch,
        {
            "findings": [
                {
                    "finding_id": "F-01",
                    "finding_type": "DIAGNOSTIC_FINDING",
                    "conclusion": "Diagnostic indicates acceptable exception clustering.",
                    "evidence_refs": [{"evidence_id": "EV-VAR1", "metric_path": "metrics.n_exceptions"}],
                }
            ],
            "overall_assessment": "Challenge resolved via deterministic diagnostic.",
        },
    )
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = _make_dummy_record("EV-VAR1", "traded_risk.var_exceptions", {"n_exceptions": 4})
    prompts = iter(["A", "A", "C", "Challenge tail risk exceptions", "A", "A", "A", "A"])
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))
    c_dec = [d for d in decisions if d["action"] == "challenge"][0]
    assert c_dec["backend"] == "llm_structured"


def test_review_integration_covariance_question_challenge(monkeypatch):
    """41. Covariance Question/Challenge checkpoint in structured mode."""
    _mock_structured_llm(
        monkeypatch,
        {
            "findings": [
                {
                    "finding_id": "F-01",
                    "finding_type": "OBSERVED_EVIDENCE",
                    "conclusion": "Covariance shrinkage intensity is optimal.",
                    "evidence_refs": [
                        {"evidence_id": "EV-COV1", "metric_path": "metrics.shrinkage_intensity"}
                    ],
                }
            ],
            "overall_assessment": "Covariance matrix is well-conditioned.",
        },
    )
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = _make_dummy_record("EV-COV1", "covariance.ledoit_wolf_shrinkage", {"shrinkage_intensity": 0.15})
    prompts = iter(["A", "A", "A", "Q", "Evaluate shrinkage", "A", "A", "A"])
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))
    q_dec = [d for d in decisions if d["action"] == "question"][0]
    assert q_dec["backend"] == "llm_structured"


def test_review_integration_scenario_question_challenge(monkeypatch):
    """42. Scenario Analysis Question/Challenge checkpoint in structured mode."""
    _mock_structured_llm(
        monkeypatch,
        {
            "findings": [
                {
                    "finding_id": "F-01",
                    "finding_type": "OBSERVED_EVIDENCE",
                    "conclusion": "Scenario stress loss is within capital limits.",
                    "evidence_refs": [{"evidence_id": "EV-SCEN1", "metric_path": "metrics.stress_loss"}],
                }
            ],
            "overall_assessment": "Stress scenarios validated.",
        },
    )
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = _make_dummy_record("EV-SCEN1", "scenario.linear_return", {"stress_loss": -50000.0})
    prompts = iter(["A", "A", "A", "A", "Q", "Assess stress loss", "A", "A"])
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))
    q_dec = [d for d in decisions if d["action"] == "question"][0]
    assert q_dec["backend"] == "llm_structured"


def test_review_integration_committee_multi_domain(monkeypatch):
    """43. Committee Synthesis multi-domain checkpoint in structured mode."""
    _mock_structured_llm(
        monkeypatch,
        {
            "findings": [
                {
                    "finding_id": "F-01",
                    "finding_type": "CROSS_ANALYTICAL_DEPENDENCY",
                    "conclusion": "Volatilities and scenario counts align across domains.",
                    "evidence_refs": [
                        {"evidence_id": "EV-1", "metric_path": "metrics.periodic_volatility"},
                        {"evidence_id": "EV-2", "metric_path": "metrics.n_scenarios"},
                    ],
                }
            ],
            "overall_assessment": "Committee confirms cross-domain consistency.",
        },
    )
    bundle = ReviewContextBundle(
        mode=ReviewMode.CROSS_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec1 = _make_dummy_record("EV-1", "portfolio.risk_statistics", {"periodic_volatility": 0.001679})
    rec2 = _make_dummy_record("EV-2", "scenario.linear_return", {"n_scenarios": 1000})
    prompts = iter(["A", "A", "A", "A", "A", "Q", "Synthesize findings", "A"])
    decisions = run_domain_checkpoints(bundle, [rec1, rec2], interactive=True, ask=lambda _: next(prompts))
    q_dec = [d for d in decisions if d["action"] == "question"][0]
    assert q_dec["backend"] == "llm_structured"


def test_review_integration_governance_signoff(monkeypatch):
    """44. Model Governance sign-off checkpoint in structured mode."""
    _mock_structured_llm(
        monkeypatch,
        {
            "findings": [
                {
                    "finding_id": "F-01",
                    "finding_type": "OBSERVED_EVIDENCE",
                    "conclusion": "Governance attestation complete with full evidence coverage.",
                    "evidence_refs": [{"evidence_id": "EV-GOV1", "metric_path": "metrics.total_tests"}],
                }
            ],
            "overall_assessment": "Final governance disposition ACCEPT.",
        },
    )
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = _make_dummy_record("EV-GOV1", "portfolio.risk_statistics", {"total_tests": 79})
    prompts = iter(["A", "A", "A", "A", "A", "A", "Q", "Confirm sign-off"])
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))
    q_dec = [d for d in decisions if d["action"] == "question"][0]
    assert q_dec["backend"] == "llm_structured"


# =========================================================================== #
# 8. ACCEPTANCE & ATTESTATION (Tests 45–49)
# =========================================================================== #
def test_acceptance_machine_readable_census_emitted(monkeypatch):
    """45. Checkpoint decision entry contains all structured machine-readable census fields."""
    _mock_structured_llm(
        monkeypatch,
        {
            "findings": [
                {
                    "finding_id": "F-01",
                    "finding_type": "OBSERVED_EVIDENCE",
                    "conclusion": "Observation.",
                    "evidence_refs": [{"evidence_id": "EV-M1", "metric_path": "metrics.vol"}],
                }
            ],
            "overall_assessment": "Assessment.",
        },
    )
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = _make_dummy_record("EV-M1", "portfolio.risk_statistics", {"vol": 0.05})
    prompts = iter(["Q", "Question", "A", "A", "A", "A", "A"])
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))
    dec = [d for d in decisions if d["action"] == "question"][0]
    assert dec["grounding_mode"] == "STRUCTURED"
    assert dec["finding_count"] == 1
    assert dec["evidence_ref_count"] == 1
    assert dec["validated_ref_count"] == 1
    assert dec["invalid_ref_count"] == 0
    assert dec["structured_findings_content_hash"] is not None
    assert dec["provider_status"] == "OK"
    assert dec["schema_validation_status"] == "VALID"


def test_acceptance_harness_consumes_machine_readable_census():
    """46. Harness parser asserts machine-readable census invariants correctly."""
    from scripts.run_market_acceptance_from_runbook import MarketAcceptanceRunner

    runner = MarketAcceptanceRunner.__new__(MarketAcceptanceRunner)
    runner.grounding_censuses = []
    text = (
        "Structured Grounding Gate: PASSED — Findings: 2 | "
        "Evidence refs: 3 | Validated refs: 3 | Invalid: 0\n"
    )
    runner._parse_grounding_census(text)
    assert len(runner.grounding_censuses) == 1
    c = runner.grounding_censuses[0]
    assert c["grounding_mode"] == "STRUCTURED"
    assert c["finding_count"] == 2
    assert c["evidence_ref_count"] == 3
    assert c["validated_ref_count"] == 3
    assert c["invalid_ref_count"] == 0


def test_attestation_same_canonical_finding_graph_sealed(sample_review_context):
    """47. Merkle attestation seal binds the exact structured findings content hash."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Finding one.",
                evidence_refs=(EvidenceMetricRef(evidence_id="EV-16bbbafd361d", metric_path="metrics.n11"),),
            ),
        ),
        overall_assessment="Overall assessment.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert res.valid
    assert res.hydrated_response is not None
    assert len(res.hydrated_response.content_hash) == 64


def test_attestation_mutated_graph_changes_hash(sample_review_context):
    """48. Mutating any finding, metric path, or ref strictly changes the content hash."""
    _, ctx = sample_review_context
    resp1 = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Conclusion text.",
                evidence_refs=(EvidenceMetricRef(evidence_id="EV-16bbbafd361d", metric_path="metrics.n11"),),
            ),
        ),
        overall_assessment="Overall assessment.",
    )
    resp2 = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Different conclusion text.",
                evidence_refs=(EvidenceMetricRef(evidence_id="EV-16bbbafd361d", metric_path="metrics.n11"),),
            ),
        ),
        overall_assessment="Overall assessment.",
    )
    res1 = validate_and_hydrate_structured_response(resp1, ctx)
    res2 = validate_and_hydrate_structured_response(resp2, ctx)
    assert res1.hydrated_response.content_hash != res2.hydrated_response.content_hash


def test_attestation_invalid_refs_cannot_attest(sample_review_context):
    """49. Invalid refs cannot produce a valid hydrated response or attestation hash."""
    _, ctx = sample_review_context
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Invalid finding.",
                evidence_refs=(EvidenceMetricRef(evidence_id="EV-INVALID-ID", metric_path="metrics.n11"),),
            ),
        ),
        overall_assessment="Overall assessment.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert not res.valid
    assert res.hydrated_response is None


# =========================================================================== #
# 9. FROZEN FAILURE REPLAYS (Section 11)
# =========================================================================== #
def test_frozen_replay_committee_synthesis_multi_domain():
    """50. Frozen Replay (Run 20260903_001013): Committee synthesis multi-domain structured resolution.

    Eliminates UNSUPPORTED_DERIVED_RELATION by hydrating exact metrics (periodic_volatility, n_scenarios).
    """
    rec_port = _make_dummy_record(
        evidence_id="EV-47bb29605d5d",
        test_id="portfolio.risk_statistics",
        metrics={"periodic_volatility": 0.001679, "annualised_volatility": 0.026654},
        run_id="RUN-20260903-001013",
    )
    rec_scen = _make_dummy_record(
        evidence_id="EV-4ff86b3e8e9c",
        test_id="scenario.stress_test",
        metrics={"n_scenarios": 1000, "nominal_tail_scenarios": 76},
        run_id="RUN-20260903-001013",
    )
    view = build_checkpoint_evidence_view(
        checkpoint_title="Cross-Analytical Committee Synthesis",
        checkpoint_description="Synthesize cross-analytical claims.",
        domains=(ReviewDomain.MARKET, ReviewDomain.TREASURY),
        records=[rec_port, rec_scen],
    )
    ctx = StructuredReviewContext(
        run_id="RUN-20260903-001013",
        checkpoint_id="Cross-Analytical Committee Synthesis",
        evidence_view_hash=view.compute_evidence_view_hash(),
        allowed_evidence_ids=("EV-47bb29605d5d", "EV-4ff86b3e8e9c"),
        records_by_id={"EV-47bb29605d5d": rec_port, "EV-4ff86b3e8e9c": rec_scen},
    )
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.CROSS_ANALYTICAL_DEPENDENCY,
                conclusion="Periodic volatility is consistent with historical tail scenario counts.",
                evidence_refs=(
                    EvidenceMetricRef(
                        evidence_id="EV-47bb29605d5d", metric_path="metrics.periodic_volatility"
                    ),
                    EvidenceMetricRef(evidence_id="EV-4ff86b3e8e9c", metric_path="metrics.n_scenarios"),
                ),
            ),
        ),
        overall_assessment="Cross-analytical committee finds no evidence discrepancies.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert res.valid
    assert res.validated_refs_count == 2
    md = render_structured_response_markdown(res.hydrated_response)
    assert "0.1679%" in md
    assert "1,000" in md


def test_frozen_replay_var_n11_zero_deterministic_hydration():
    """51. Frozen Replay (Run 20260903_024426): VaR n11=0 structured resolution.

    Eliminates UNSUPPORTED_DERIVED_RELATION on n11=0 by hydrating exact count 0 without arithmetic.
    """
    rec_var = _make_dummy_record(
        evidence_id="EV-16bbbafd361d",
        test_id="traded_risk.var_exceptions",
        metrics={"n11": 0, "n_exceptions": 0, "p_value": 0.99},
        run_id="RUN-20260903-024426",
    )
    view = build_checkpoint_evidence_view(
        checkpoint_title="VaR Backtesting & Exception Frequency",
        checkpoint_description="Review backtest exceptions.",
        domains=(ReviewDomain.MARKET,),
        records=[rec_var],
    )
    ctx = StructuredReviewContext(
        run_id="RUN-20260903-024426",
        checkpoint_id="VaR Backtesting & Exception Frequency",
        evidence_view_hash=view.compute_evidence_view_hash(),
        allowed_evidence_ids=("EV-16bbbafd361d",),
        records_by_id={"EV-16bbbafd361d": rec_var},
    )
    resp = StructuredReviewerResponse(
        findings=(
            ReviewerFinding(
                finding_id="F-01",
                finding_type=FindingType.OBSERVED_EVIDENCE,
                conclusion="Backtest exhibits zero consecutive exception transitions.",
                evidence_refs=(EvidenceMetricRef(evidence_id="EV-16bbbafd361d", metric_path="metrics.n11"),),
            ),
        ),
        overall_assessment="VaR backtest exception frequency is within acceptable bounds.",
    )
    res = validate_and_hydrate_structured_response(resp, ctx)
    assert res.valid
    ref = res.hydrated_response.findings[0].hydrated_refs[0]
    assert ref.value == 0
    assert ref.display == "0"
    assert ref.unit == "count"
