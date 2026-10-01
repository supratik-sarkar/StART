"""Deterministic regression test for live Attempt 3 structured reviewer response defect.

Validates:
1. Attempt 3 fixture parses with StructuredReviewerResponse Pydantic schema.
2. Grounding validation strictly rejects invalid metric paths ("Interpretation").
3. Grounding validator does NOT invent replacement paths or loosen canonical evidence rules.
4. Checkpoint degradation policy for 'market.cross_analytical_synthesis' on action 'Q'
   resolves to ALLOW_EVIDENCE_ONLY_DEGRADATION.
5. StructuredDegradationEvent is properly instantiated and preserved in audit trail.
6. Fallback panel and response text preserve deterministic evidence.
7. Final governance disposition evaluates to ACCEPT_WITH_CONDITIONS.
8. Governance table correctly reports degraded interpretation count.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from start.core.schemas import EvidenceRecord, Status, TestResult
from start.review.architecture import (
    ReviewContextBundle,
    ReviewDomain,
    ReviewLifecycle,
    ReviewMode,
)
from start.review.evidence_view import build_checkpoint_evidence_view
from start.review.executor import evaluate_deterministic_governance_disposition
from start.review.structured_contract import (
    CheckpointDegradationPolicy,
    StructuredDegradationEvent,
    StructuredReviewContext,
    StructuredReviewerResponse,
    get_checkpoint_degradation_policy,
    normalize_structured_json_text,
    render_evidence_only_fallback_panel,
    validate_and_hydrate_structured_response,
)
from start.review.tables import build_governance_table


def _make_dummy_record(
    evidence_id: str,
    test_id: str,
    metrics: dict[str, Any],
    run_id: str = "RUN-LIVE-ATTEMPT-3",
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
def attempt3_json_text() -> str:
    path = Path(__file__).parent / "fixtures" / "attempt3_malformed_response.json"
    assert path.exists(), f"Fixture not found: {path}"
    return path.read_text(encoding="utf-8")


def test_attempt3_pydantic_schema_validity(attempt3_json_text: str) -> None:
    """Attempt 3 response parses with StructuredReviewerResponse schema."""
    normalized = normalize_structured_json_text(attempt3_json_text)
    resp = StructuredReviewerResponse.model_validate_json(normalized)
    assert len(resp.findings) == 3
    assert resp.overall_assessment != ""
    assert resp.findings[0].finding_id == "F-01"
    assert resp.findings[2].finding_id == "F-09"


def test_attempt3_grounding_rejection_of_interpretation_path(attempt3_json_text: str) -> None:
    """Attempt 3 F-09 cited 'Interpretation', which is strictly rejected by the grounding gate."""
    rec1 = _make_dummy_record(
        evidence_id="EV-7322962e3fa7",
        test_id="portfolio.risk_statistics",
        metrics={
            "annualised_volatility": 0.1425,
            "sharpe_ratio": 1.45,
        },
    )
    rec2 = _make_dummy_record(
        evidence_id="EV-72493e5f4cfd",
        test_id="var.historical_simulation",
        metrics={
            "var_historical_99": 0.0215,
        },
    )

    records = [rec1, rec2]
    view = build_checkpoint_evidence_view(
        checkpoint_title="Cross-Analytical Committee Synthesis",
        checkpoint_description="Cross-analytical risk synthesis and governance committee review.",
        domains=(ReviewDomain.MARKET,),
        records=records,
    )

    st_ctx = StructuredReviewContext(
        run_id="RUN-LIVE-ATTEMPT-3",
        checkpoint_id="Cross-Analytical Committee Synthesis",
        evidence_view_hash=view.compute_evidence_view_hash(),
        allowed_evidence_ids=tuple(r.evidence_id for r in records),
        records_by_id={r.evidence_id: r for r in records},
    )

    resp = StructuredReviewerResponse.model_validate_json(attempt3_json_text)
    val_res = validate_and_hydrate_structured_response(resp, st_ctx)

    assert not val_res.valid
    assert val_res.invalid_refs_count == 2
    assert val_res.validated_refs_count == 3  # F-01 (2 refs) + F-02 (1 ref)
    assert val_res.evidence_refs_count == 5

    # Verify both rejected references are exactly the 'Interpretation' paths
    paths_rejected = [d["metric_path"] for d in val_res.invalid_refs_details]
    assert paths_rejected == ["Interpretation", "Interpretation"]
    assert all("Invalid non-canonical path format" in d["error"] for d in val_res.invalid_refs_details)


def test_attempt3_checkpoint_degradation_policy_resolution() -> None:
    """Informational question on cross_analytical_synthesis permits evidence-only degradation."""
    policy_q = get_checkpoint_degradation_policy(
        "Cross-Analytical Committee Synthesis",
        action="Q",
    )
    assert policy_q == CheckpointDegradationPolicy.ALLOW_EVIDENCE_ONLY_DEGRADATION

    # Challenge on the same checkpoint MUST NOT degrade
    policy_c = get_checkpoint_degradation_policy(
        "Cross-Analytical Committee Synthesis",
        action="C",
    )
    assert policy_c == CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE

    # Override on the same checkpoint MUST NOT degrade
    policy_o = get_checkpoint_degradation_policy(
        "Cross-Analytical Committee Synthesis",
        action="O",
    )
    assert policy_o == CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE

    # Formal sign-off checkpoint MUST NOT degrade under any action
    policy_signoff_q = get_checkpoint_degradation_policy(
        "Model Governance & Attestation Sign-off",
        action="Q",
    )
    assert policy_signoff_q == CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE


def test_attempt3_structured_degradation_event_and_governance_disposition() -> None:
    """Degradation event is captured and deterministically yields ACCEPT_WITH_CONDITIONS."""
    event = StructuredDegradationEvent(
        checkpoint_id="market.cross_analytical_synthesis",
        checkpoint_title="Cross-Analytical Committee Synthesis",
        provider="OpenAI (gpt-5.1)",
        model="gpt-5.1",
        structured_response_status="INVALID",
        reason_class="UNGROUNDED_EVIDENCE_REFS",
        invalid_refs_count=2,
        invalid_refs_details=(
            {"finding_id": "F-09", "evidence_id": "EV-7322962e3fa7", "metric_path": "Interpretation"},
            {"finding_id": "F-09", "evidence_id": "EV-72493e5f4cfd", "metric_path": "Interpretation"},
        ),
        policy_applied="ALLOW_EVIDENCE_ONLY_DEGRADATION",
        deterministic_evidence_retained=True,
        workflow_continued=True,
        timestamp=time.time(),
    )

    rec1 = _make_dummy_record("EV-7322962e3fa7", "portfolio.risk_statistics", {"annualised_volatility": 0.14})
    dec_entry = {
        "checkpoint": "Cross-Analytical Committee Synthesis",
        "action": "question",
        "note": "Give me a final evidence-oriented summary",
        "response": "Agent interpretation unavailable — deterministic evidence retained.",
        "backend": "fallback",
        "degradation_event": asdict(event),
        "degradation_policy_applied": "ALLOW_EVIDENCE_ONLY_DEGRADATION",
        "live_reviewer_validated": False,
        "live_reviewer_status": "EVIDENCE_ONLY_DEGRADATION",
        "details": "evidence_only_degraded",
    }

    bundle = ReviewContextBundle(
        domains=(ReviewDomain.MARKET,),
        mode=ReviewMode.SINGLE_DOMAIN,
        materiality="high",
        lifecycle=ReviewLifecycle.PRE_IMPLEMENTATION,
    )

    disp = evaluate_deterministic_governance_disposition(
        bundle=bundle,
        records=[rec1],
        decisions=[dec_entry],
    )
    assert disp == "ACCEPT_WITH_CONDITIONS"

    # Governance table reflects the degradation
    table = build_governance_table(
        metadata={
            "mode": bundle.mode,
            "domains": [str(d) for d in bundle.domains],
            "materiality": bundle.materiality,
            "lifecycle": bundle.lifecycle,
            "n_evidence_records": 1,
            "disposition": disp,
        },
        decisions=[dec_entry],
    )
    rows_text = " ".join(str(cell) for col in table.columns for cell in col.cells)
    assert "Evidence-Only Degraded: 1" in rows_text


def test_attempt3_fallback_panel_rendering() -> None:
    """Fallback panel renders without error and displays retained evidence records."""
    rec1 = _make_dummy_record("EV-7322962e3fa7", "portfolio.risk_statistics", {"sharpe_ratio": 1.45})
    view = build_checkpoint_evidence_view(
        checkpoint_title="Cross-Analytical Committee Synthesis",
        checkpoint_description="Cross-analytical risk synthesis and governance committee review.",
        domains=(ReviewDomain.MARKET,),
        records=[rec1],
    )
    panel = render_evidence_only_fallback_panel(
        view=view,
        checkpoint_title="Cross-Analytical Committee Synthesis",
        note="Give me a final evidence-oriented summary",
        reason="Ungrounded metric references: Interpretation",
        invalid_count=2,
    )
    assert panel is not None
    assert "AGENT INTERPRETATION UNAVAILABLE" in str(panel.title)
