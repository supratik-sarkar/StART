"""Exact semantic routing regressions for the zero-egress demo twin."""

from __future__ import annotations

import pytest

import start.providers.llm as llm_mod
from start.core.schemas import EvidenceRecord, Status, TestResult
from start.providers.offline_demo_twin import OfflineDemoTwinProvider
from start.review.architecture import (
    LLMReviewConfig,
    ReviewContextBundle,
    ReviewDomain,
    ReviewGroundingMode,
)
from start.review.evidence_view import build_checkpoint_evidence_view
from start.review.executor import run_domain_checkpoints
from start.review.structured_contract import (
    StructuredReviewContext,
    StructuredReviewerResponse,
    validate_and_hydrate_structured_response,
)

SYSTEM = "Return a strict StructuredReviewerResponse JSON object."

CHECKPOINTS = (
    ("market.portfolio_allocation", "Portfolio Risk & Volatility Assumptions"),
    ("market.factor_attribution", "Factor Modeling & Attribution Assumptions"),
    ("market.var_backtesting", "VaR Backtesting & Exception Frequency"),
    ("market.covariance_structure", "Covariance Structure & Missing Data Treatment"),
    ("market.scenario_stress", "Scenario Analysis & Stress Testing"),
    ("market.governance_signoff", "Model Governance & Attestation Sign-off"),
)


def _record(
    evidence_id: str,
    metrics: dict[str, object],
    *,
    test_id: str = "fixture.semantic_checkpoint",
) -> EvidenceRecord:
    record = EvidenceRecord.from_result(
        TestResult(
            test_id=test_id,
            test_name="Semantic checkpoint fixture",
            status=Status.PASS,
            metrics=metrics,
        ),
        run_id="RUN-OFFLINE-TWIN",
    )
    record.evidence_id = evidence_id
    return record


def _prompt(
    semantic_checkpoint_id: str,
    title: str,
    records: list[EvidenceRecord],
    *,
    action: str = "QUESTION",
    question: str = "Explain the checkpoint evidence",
) -> tuple[str, StructuredReviewContext]:
    view = build_checkpoint_evidence_view(
        checkpoint_title=title,
        checkpoint_description="Focused semantic routing fixture.",
        domains=(ReviewDomain.MARKET,),
        records=records,
    )
    prompt = (
        f"Checkpoint: {title}\n"
        f"Semantic Checkpoint ID: {semantic_checkpoint_id}\n"
        f"Interaction Action: {action}\n"
        f'Question: "{question}"\n'
        "Permitted EvidenceRecords for this Checkpoint:\n"
        f"{view.format_llm_payload()}\n"
    )
    context = StructuredReviewContext(
        run_id="RUN-OFFLINE-TWIN",
        checkpoint_id=semantic_checkpoint_id,
        evidence_view_hash=view.compute_evidence_view_hash(),
        allowed_evidence_ids=tuple(record.evidence_id for record in records),
        records_by_id={record.evidence_id: record for record in records},
    )
    return prompt, context


@pytest.mark.parametrize(("semantic_checkpoint_id", "title"), CHECKPOINTS)
def test_mandatory_market_responses_validate_against_actual_records(
    semantic_checkpoint_id: str,
    title: str,
) -> None:
    records = [
        _record("EV-NO-METRIC", {}),
        _record("EV-ACTUAL-METRIC", {"is_valid": True, "observed_metric": 0.25}),
    ]
    prompt, context = _prompt(semantic_checkpoint_id, title, records)
    response = StructuredReviewerResponse.model_validate_json(
        OfflineDemoTwinProvider().complete(SYSTEM, prompt)
    )
    result = validate_and_hydrate_structured_response(response, context)

    assert result.valid is True
    assert result.validated_refs_count == 1
    ref = response.findings[0].evidence_refs[0]
    assert ref.evidence_id == "EV-ACTUAL-METRIC"
    assert ref.metric_path in {"metrics.is_valid", "metrics.observed_metric"}


def test_cross_synthesis_rehearsal_query_is_intentionally_invalid() -> None:
    records = [_record("EV-CROSS-ACTUAL", {"committee_complete": True})]
    prompt, context = _prompt(
        "market.cross_analytical_synthesis",
        "Cross-Analytical Committee Synthesis",
        records,
        question="Give me a final evidence-oriented summary",
    )
    response = StructuredReviewerResponse.model_validate_json(
        OfflineDemoTwinProvider().complete(SYSTEM, prompt)
    )
    result = validate_and_hydrate_structured_response(response, context)

    assert response.findings[0].evidence_refs[0].evidence_id == "EV-NOT-IN-CHECKPOINT"
    assert result.valid is False
    assert result.invalid_refs_count == 1


def test_runtime_prompt_supplies_stable_checkpoint_and_action(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    class CapturingTwin(OfflineDemoTwinProvider):
        def _response(self, system: str, user: str) -> str:
            captured.append(user)
            return super()._response(system, user)

    monkeypatch.setattr(llm_mod, "get_llm_provider", lambda cfg: CapturingTwin())
    bundle = ReviewContextBundle(
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="offline",
            provider="offline_demo_twin",
            model="deterministic-semantic-fixture-v1",
            status="OFFLINE_REHEARSAL",
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    prompts = iter(["Q", "Explain the registered portfolio evidence", "A", "A", "A", "A", "A", "A"])
    decisions = run_domain_checkpoints(
        bundle,
        [
            _record(
                "EV-PROMPT-CONTRACT",
                {"is_valid": True},
                test_id="portfolio.risk_statistics",
            )
        ],
        interactive=True,
        ask=lambda _: next(prompts),
    )

    assert decisions[0]["backend"] == "llm_structured"
    assert "Semantic Checkpoint ID: market.portfolio_allocation" in captured[0]
    assert "Interaction Action: QUESTION" in captured[0]


@pytest.mark.parametrize(
    ("semantic_checkpoint_id", "action", "question"),
    (
        ("market.portfolio_allocation", "QUESTION", "Give me a final evidence-oriented summary"),
        ("", "QUESTION", "Give me a final evidence-oriented summary"),
        ("market.cross_analytical_synthesis", "CHALLENGE", "Give me a final evidence-oriented summary"),
        ("market.cross_analytical_synthesis", "QUESTION", "Explain committee dependencies"),
    ),
)
def test_intentional_invalid_fixture_cannot_leak_outside_exact_target(
    semantic_checkpoint_id: str,
    action: str,
    question: str,
) -> None:
    records = [_record("EV-TARGET-GUARD", {"is_valid": True})]
    prompt, context = _prompt(
        semantic_checkpoint_id,
        "Cross-Analytical Committee Synthesis",
        records,
        action=action,
        question=question,
    )
    response = StructuredReviewerResponse.model_validate_json(
        OfflineDemoTwinProvider().complete(SYSTEM, prompt)
    )
    result = validate_and_hydrate_structured_response(response, context)

    assert result.valid is True
    assert response.findings[0].evidence_refs[0].evidence_id == "EV-TARGET-GUARD"
