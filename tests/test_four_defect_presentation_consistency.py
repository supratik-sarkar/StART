"""Focused regressions for the four capture-derived presentation defects."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from demo_controller import PTYController
from rich.console import Console

import start.providers.llm as llm_mod
import start.review.executor as executor_mod
from start.core.schemas import EvidenceRecord, Status, TestResult
from start.providers.offline_demo_twin import OfflineDemoTwinProvider
from start.review.architecture import (
    LLMReviewConfig,
    ReviewContextBundle,
    ReviewDomain,
    ReviewGroundingMode,
)
from start.review.executor import _render_rejected_raw_payload, run_domain_checkpoints
from start.review.terminal_observability import (
    PresentationEventKind,
    TerminalReviewObserver,
    build_flight_a_presentation_context,
    resolve_authority_presentation,
)


def _sequence_bundle() -> SimpleNamespace:
    return SimpleNamespace(
        X_train=np.zeros((480, 24, 3)),
        y_train=np.zeros(480),
        X_test=np.zeros((160, 24, 3)),
        y_test=np.zeros(160),
        X_oos=np.zeros((160, 24, 3)),
        y_oos=np.zeros(160),
        timesteps=24,
        n_features=3,
    )


def _record() -> EvidenceRecord:
    record = EvidenceRecord.from_result(
        TestResult(
            test_id="portfolio.risk_statistics",
            test_name="Portfolio risk statistics",
            status=Status.RECORDED,
            metrics={"annualised_volatility": 0.12},
        ),
        run_id="RUN-FOUR-DEFECTS",
    )
    record.evidence_id = "EV-FOUR-DEFECTS"
    return record


def _offline_bundle() -> ReviewContextBundle:
    return ReviewContextBundle(
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="offline",
            provider="offline_demo_twin",
            model="deterministic-semantic-fixture-v1",
            status="OFFLINE_REHEARSAL",
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )


def test_d1_temporal_surfaces_share_one_canonical_presentation_context() -> None:
    frame = pd.DataFrame(np.zeros((800, 7)), columns=[f"field_{i}" for i in range(7)])
    context = build_flight_a_presentation_context(
        df=frame,
        sequence_bundle=_sequence_bundle(),
        selected_architecture="lstm",
        configured_split="stratified",
    )

    assert context.review_metadata_display == (
        "REVIEW METADATA — NOT MODEL INPUT: 800 rows × 7 fields"
    )
    assert context.model_sample_count == 800
    assert context.model_feature_count == 3
    assert context.model_input_contract == "(800,24,3) (N,T,F)"
    assert "800 independent sequences" in context.model_input_display
    assert "3 model features" in context.model_input_display
    assert "rows" not in context.model_input_display
    assert "order-preserving contiguous sequence holdout" in context.split_description
    assert "stratified" not in context.split_description
    assert len(context.architecture_alternatives) == len(set(context.architecture_alternatives))
    assert "xgboost" not in context.architecture_alternatives

    evidence = context.architecture_evidence("binary_classification")
    trace = context.architecture_trace_input("lstm", "relu")
    assert context.model_input_display in evidence
    assert context.split_description in evidence[1]
    assert context.model_input_display in trace
    assert context.split_description in trace

    source = (Path(__file__).parents[1] / "src/start/interactive_review.py").read_text()
    assert "evidence=flight_a_context.architecture_evidence(task_type)" in source
    assert "alternatives=flight_a_context.architecture_alternatives" in source
    assert "presentation_context=flight_a_context" in source

    split_record = EvidenceRecord.from_result(
        TestResult(
            test_id="split.plan",
            test_name="Metadata split plan",
            status=Status.PASS,
            metrics={"strategy": "stratified"},
        ),
        run_id="RUN-TEMPORAL-CONTEXT",
    )
    stream = io.StringIO()
    observer = TerminalReviewObserver(
        "RUN-TEMPORAL-CONTEXT",
        console=Console(file=stream, width=132, color_system=None),
        session_kind="A",
    )
    observer.bind_flight_a_context(context)
    observer.evidence_committed(split_record)
    split_card = stream.getvalue()
    assert "REVIEW METADATA — NOT MODEL INPUT" in split_card
    assert "Metadata-only plan" in split_card
    assert "Actual model split" in split_card
    assert "order-preserving contiguous sequence holdout" in split_card
    assert split_record.metrics["strategy"] == "stratified"


def test_d2_genuine_grounding_acceptance_stays_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    test_console = Console(file=stream, width=132, color_system=None)
    monkeypatch.setattr(executor_mod, "console", test_console)
    monkeypatch.setattr(llm_mod, "get_llm_provider", lambda cfg: OfflineDemoTwinProvider())
    observer = TerminalReviewObserver(
        "RUN-GROUNDING-ACCEPTED",
        console=test_console,
        session_kind="B",
    )
    prompts = iter(
        ["Q", "Explain the registered portfolio evidence", "A", "A", "A", "A", "A", "A"]
    )

    run_domain_checkpoints(
        _offline_bundle(),
        [_record()],
        interactive=True,
        ask=lambda _: next(prompts),
        observer=observer,
    )

    grounding = [
        event
        for event in observer.events
        if event.kind
        in {PresentationEventKind.GROUNDING_ACCEPTED, PresentationEventKind.GROUNDING_REJECTED}
    ]
    assert len(grounding) == 1
    assert grounding[0].kind is PresentationEventKind.GROUNDING_ACCEPTED
    assert grounding[0].status == "PASSED"
    assert grounding[0].payload["accepted"] is True


@pytest.mark.parametrize("presentation_mode", [False, True])
def test_d2_d4_rejected_grounding_is_typed_truthfully_and_raw_json_is_viewer_only(
    presentation_mode: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = io.StringIO()
    test_console = Console(file=stream, width=132, color_system=None)
    monkeypatch.setattr(executor_mod, "console", test_console)
    monkeypatch.setattr(llm_mod, "get_llm_provider", lambda cfg: OfflineDemoTwinProvider())
    if presentation_mode:
        monkeypatch.setenv("START_PRESENTATION_MODE", "1")
    else:
        monkeypatch.delenv("START_PRESENTATION_MODE", raising=False)
    observer = TerminalReviewObserver(
        "RUN-GROUNDING-REJECTED",
        console=test_console,
        session_kind="B",
    )
    prompts = iter(
        [
            "A",
            "A",
            "A",
            "A",
            "A",
            "Q",
            "Give me a final evidence-oriented summary",
            "1",
            "A",
        ]
    )

    run_domain_checkpoints(
        _offline_bundle(),
        [_record()],
        interactive=True,
        ask=lambda _: next(prompts),
        observer=observer,
    )

    grounding = [
        event
        for event in observer.events
        if event.kind
        in {PresentationEventKind.GROUNDING_ACCEPTED, PresentationEventKind.GROUNDING_REJECTED}
    ]
    assert len(grounding) == 1
    event = grounding[0]
    assert event.kind is PresentationEventKind.GROUNDING_REJECTED
    assert event.status == "REJECTED"
    assert event.payload["accepted"] is False
    assert event.payload["continuation"] == "EVIDENCE_ONLY"
    assert event.payload["deterministic_evidence_retained"] is True
    assert event.payload["invalid_details"] == [
        {
            "finding_id": "F-DEMO-REJECTION",
            "evidence_id": "EV-NOT-IN-CHECKPOINT",
            "metric_path": "metrics.not_admissible",
            "error": (
                "Evidence ID 'EV-NOT-IN-CHECKPOINT' not found in current "
                "CheckpointEvidenceView."
            ),
        }
    ]

    output = stream.getvalue()
    if presentation_mode:
        viewer = io.BytesIO()
        controller = PTYController(presentation_mode=True)
        controller._viewer_stream = viewer
        controller.consume_data(output.encode())
        controller.release_viewer_hold()
        viewer_output = viewer.getvalue().decode()
        assert "Raw JSON:" not in viewer_output
        assert "Raw JSON:" in controller.transcript
    else:
        viewer_output = output
        assert "Raw JSON:" in viewer_output
    assert "Structured Grounding Diagnostics" in viewer_output
    assert "AGENT INTERPRETATION UNAVAILABLE" in viewer_output
    assert "EV-NOT-IN-CHECKPOINT" in viewer_output


def test_d3_authority_semantics_are_provider_aware_and_flight_invariant() -> None:
    offline_a = resolve_authority_presentation(
        reviewer_mode="llm",
        backend_mode="offline",
        provider="offline_demo_twin",
    )
    offline_b = resolve_authority_presentation(
        reviewer_mode="llm",
        backend_mode="offline",
        provider="offline_demo_twin",
    )
    deterministic = resolve_authority_presentation(
        reviewer_mode="deterministic",
        backend_mode="none",
        provider="none",
        deterministic_only=True,
    )
    hosted = resolve_authority_presentation(
        reviewer_mode="llm",
        backend_mode="public",
        provider="openai",
        provider_status="CONNECTED",
    )
    hosted_fallback = resolve_authority_presentation(
        reviewer_mode="llm",
        backend_mode="public",
        provider="openai",
        provider_status="FAILED",
    )

    assert offline_a == offline_b
    assert offline_a.reviewer_reasoning == "OFFLINE_DEMO_TWIN"
    assert offline_a.hosted_execution == "DISABLED"
    assert deterministic.reviewer_reasoning == "DETERMINISTIC_ONLY"
    assert deterministic.hosted_execution == "DISABLED"
    assert hosted.reviewer_reasoning == "HOSTED_PROVIDER"
    assert hosted.hosted_execution == "ENABLED"
    assert hosted.provider == "openai"
    assert hosted_fallback.reviewer_reasoning == "DETERMINISTIC_FALLBACK"
    assert hosted_fallback.hosted_execution == "UNAVAILABLE"

    rendered: list[str] = []
    for run_id in ("RUN-FLIGHT-A", "RUN-FLIGHT-B"):
        stream = io.StringIO()
        observer = TerminalReviewObserver(
            run_id,
            console=Console(file=stream, width=132, color_system=None),
        )
        observer.show_authority_boundary(
            reviewer_mode="llm",
            backend_mode="offline",
            provider="offline_demo_twin",
        )
        rendered.append(stream.getvalue())
    assert rendered[0] == rendered[1]
    assert "Reviewer reasoning" in rendered[0]
    assert "OFFLINE_DEMO_TWIN" in rendered[0]
    assert "Hosted LLM execution" in rendered[0]
    assert "Hosted LLM calls" in rendered[0]
    assert "LLM numeric authority" in rendered[0]
    assert "Deterministic engines" in rendered[0]


def test_d4_raw_rejection_helper_preserves_nonpresentation_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordinary_stream = io.StringIO()
    monkeypatch.delenv("START_PRESENTATION_MODE", raising=False)
    assert _render_rejected_raw_payload(
        Console(file=ordinary_stream, color_system=None),
        label="Raw JSON",
        payload='{"rejected": true}',
    )
    assert "Raw JSON" in ordinary_stream.getvalue()
    assert '"rejected": true' in ordinary_stream.getvalue()

    presentation_stream = io.StringIO()
    monkeypatch.setenv("START_PRESENTATION_MODE", "1")
    assert not _render_rejected_raw_payload(
        Console(file=presentation_stream, color_system=None),
        label="Raw JSON",
        payload='{"rejected": true}',
    )
    viewer = io.BytesIO()
    controller = PTYController(presentation_mode=True)
    controller._viewer_stream = viewer
    controller.consume_data(presentation_stream.getvalue().encode())
    assert "Raw JSON" in controller.transcript
    assert '"rejected": true' in controller.transcript
    assert viewer.getvalue() == b""
