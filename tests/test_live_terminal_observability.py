"""Focused acceptance regressions for the live terminal presentation pipeline."""

from __future__ import annotations

from io import StringIO
from types import SimpleNamespace

import pytest
from demo_controller import (
    DemoProgress,
    SessionResult,
    SessionStatus,
    configure_presentation_mode,
    log,
)
from rich.console import Console

from start.core.schemas import EvidenceRecord, Status, TestResult
from start.interactive_checkpoints import resolve_checkpoint
from start.orchestration.state_graph import build_live_review_graph
from start.providers.offline_demo_twin import OfflineDemoTwinProvider
from start.review.terminal_observability import (
    PresentationEventKind,
    RuntimePresentationSink,
    TerminalReviewObserver,
)
from start.runtime.events import ListEventSink, RuntimeEvent
from start.runtime.execution import CanonicalExecutionService
from start.telemetry.engineering_trace import TraceRecord


def _console(width: int = 268) -> tuple[Console, StringIO]:
    stream = StringIO()
    return Console(file=stream, width=width, color_system=None), stream


def _record(test_id: str, evidence_id: str = "EV-TEST") -> EvidenceRecord:
    record = EvidenceRecord.from_result(
        TestResult(
            test_id=test_id,
            test_name=test_id,
            status=Status.PASS,
            metrics={"value": 0.25},
        ),
        run_id="RUN-PRESENTATION",
    )
    record.evidence_id = evidence_id
    return record


def _trace() -> TraceRecord:
    return TraceRecord(
        trace_id="a" * 32,
        span_id="b" * 16,
        parent_span_id=None,
        name="start.run",
        start_time=1.0,
        end_time=1.1,
        duration_ms=100.0,
        status="OK",
        run_id="RUN-PRESENTATION",
        attributes={},
    )


def _policy() -> SimpleNamespace:
    return SimpleNamespace(
        decision_id="POL-DEC-TEST",
        policy_package="start.governance.attestation_rules",
        rule_name="allow_governance_attestation",
        decision="ALLOW",
        reason="accepted",
        evidence_ids=["EV-TEST"],
        input_fingerprint="abc123",
        engine="OPA_LOCAL",
    )


def test_typed_event_sequence_state_and_width_are_canonical() -> None:
    console, stream = _console()
    observer = TerminalReviewObserver(
        "RUN-PRESENTATION",
        console=console,
        session_kind="A",
        max_width=132,
    )
    observer.session_started(source_component="fixture")
    observer.human_action("A", "architecture", "x" * 400)
    observer.agent_card(
        "ArchitectureReviewAgent",
        role="architecture",
        deterministic_context="temporal contract / dataset facts",
        checkpoint="architecture",
    )
    observer.engine_card("SequenceClassifier", operation="fit", checkpoint="execution")
    observer.engine_completed("SequenceClassifier", checkpoint="execution")

    assert [event.sequence for event in observer.events] == list(
        range(1, len(observer.events) + 1)
    )
    assert all(event.run_id == "RUN-PRESENTATION" for event in observer.events)
    assert observer.state.active_stage_history == ["HUMAN", "AGENT", "ENGINE"]
    assert observer.state.last_human_action == "ACCEPT"
    assert observer.state.current_checkpoint == "execution"
    assert max(len(line) for line in stream.getvalue().splitlines()) <= 132


def test_live_langgraph_phase_gate_emits_before_destination_work() -> None:
    console, _ = _console()
    observer = TerminalReviewObserver("RUN-GRAPH", console=console, session_kind="B")
    execution_log: list[str] = []
    graph = build_live_review_graph(
        run_id="RUN-GRAPH",
        phases=(
            "plan",
            "execute_tools",
            "review_evidence",
            "generate_artifacts",
            "governance_signoff",
        ),
        on_transition=lambda source, target, trigger, state_hash, thread_id: (
            execution_log.append(f"transition:{target}"),
            observer.graph_transition(
                source,
                target,
                trigger=trigger,
                state_hash=state_hash,
                thread_id=thread_id,
            ),
        ),
    )

    for phase in graph.phases:
        graph.enter_phase(phase)
        execution_log.append(f"work:{phase}")

    for phase in graph.phases:
        assert execution_log.index(f"transition:{phase}") < execution_log.index(f"work:{phase}")
    assert graph.complete is True
    assert graph.checkpoint.next == ()
    transitions = [
        event for event in observer.events if event.kind is PresentationEventKind.GRAPH_TRANSITION
    ]
    assert [event.payload["target"] for event in transitions] == list(graph.phases)
    assert all(event.payload["state_hash"] for event in transitions)


def test_session_a_human_c_c_q_a_actions_emit_exactly_once() -> None:
    console, _ = _console()
    observer = TerminalReviewObserver("RUN-A", console=console, session_kind="A")
    answers = iter(
        [
            "C",
            "Challenge recurrent architecture",
            "C",
            "Verify genuine rank-3 sequence structure",
            "Q",
            "How is leakage prevented?",
            "A",
        ]
    )

    def observe(action: str, note: str) -> None:
        observer.human_action(action, "architecture", note)
        if action in {"C", "Q"}:
            observer.agent_card(
                "ArchitectureReviewAgent",
                role="architecture challenge",
                deterministic_context="temporal contract / dataset facts",
                checkpoint="architecture",
            )

    decision = resolve_checkpoint(
        "architecture",
        "lstm",
        "lstm",
        "temporal dependence",
        interactive=True,
        ask=lambda _: next(answers),
        on_ask=lambda question: f"bounded response: {question}",
        on_action=observe,
    )

    actions = [
        event.payload["action"]
        for event in observer.events
        if event.kind is PresentationEventKind.HUMAN_ACTION
    ]
    assert actions == ["CHALLENGE", "CHALLENGE", "QUESTION", "ACCEPT"]
    assert decision.choice == "accept"
    agent_events = [
        event for event in observer.events if event.kind is PresentationEventKind.AGENT_ACTIVATED
    ]
    assert len(agent_events) == 3


def test_session_a_event_journal_matches_live_acceptance_order() -> None:
    console, _ = _console()
    observer = TerminalReviewObserver("RUN-A-ORDER", console=console, session_kind="A")
    observer.session_started(source_component="fixture")
    observer.human_action("SETUP", "review.setup")
    graph = build_live_review_graph(
        run_id="RUN-A-ORDER",
        phases=("plan", "execute_tools", "review_evidence", "governance_signoff"),
        on_transition=lambda source, target, trigger, state_hash, thread_id: observer.graph_transition(
            source, target, trigger=trigger, state_hash=state_hash, thread_id=thread_id
        ),
    )
    graph.enter_phase("plan")
    observer.agent_card("ArchitectureReviewAgent", role="architecture", checkpoint="architecture")
    for action in ("C", "C", "Q", "A"):
        observer.human_action(action, "architecture", action)
        if action in {"C", "Q"}:
            observer.agent_card(
                "ArchitectureReviewAgent", role="architecture", checkpoint="architecture"
            )
    graph.enter_phase("execute_tools")
    observer.engine_card("SequenceClassifier", operation="fit and evaluate")
    observer.engine_completed("SequenceClassifier")
    observer.evidence_committed(_record("deep_learning.performance_diagnostics", "EV-PERF"))
    observer.engine_card("TemporalInputGradient", operation="input-gradient saliency")
    observer.engine_completed("TemporalInputGradient")
    observer.evidence_committed(
        _record("deep_learning.explainability_diagnostics", "EV-ATTR")
    )
    graph.enter_phase("review_evidence")
    graph.enter_phase("governance_signoff")
    observer.grounding_result(
        accepted=True,
        quantitative_claims=1,
        grounded_claims=1,
        checkpoint="governance.signoff",
    )
    observer.governance_card(
        disposition="ACCEPT",
        evidence_count=2,
        unresolved_count=0,
        validation_failures=0,
    )
    observer.policy_gate(_policy())
    observer.trace_waterfall([_trace()])
    observer.attestation_sealed(merkle_root="c" * 64, leaf_count=2)
    observer.session_completed(source_component="fixture")

    kinds = [event.kind for event in observer.events]
    required = [
        PresentationEventKind.SESSION_STARTED,
        PresentationEventKind.HUMAN_ACTION,
        PresentationEventKind.GRAPH_TRANSITION,
        PresentationEventKind.AGENT_ACTIVATED,
        PresentationEventKind.GRAPH_TRANSITION,
        PresentationEventKind.ENGINE_STARTED,
        PresentationEventKind.ENGINE_COMPLETED,
        PresentationEventKind.EVIDENCE_COMMITTED,
        PresentationEventKind.ENGINE_STARTED,
        PresentationEventKind.ENGINE_COMPLETED,
        PresentationEventKind.EVIDENCE_COMMITTED,
        PresentationEventKind.GOVERNANCE_EVALUATED,
        PresentationEventKind.OPA_EVALUATED,
        PresentationEventKind.TRACE_READY,
        PresentationEventKind.ATTESTATION_SEALED,
        PresentationEventKind.SESSION_COMPLETED,
    ]
    cursor = -1
    for kind in required:
        cursor = kinds.index(kind, cursor + 1)
    assert set(observer.state.active_stage_history) >= {
        "HUMAN",
        "LANGGRAPH",
        "AGENT",
        "ENGINE",
        "EVIDENCE",
        "GROUNDING",
        "GOVERNANCE",
        "OPA",
        "TRACE",
        "ATTESTATION",
    }


def test_session_b_graph_and_grounding_events_precede_governance_and_attestation() -> None:
    console, _ = _console()
    observer = TerminalReviewObserver("RUN-B", console=console, session_kind="B")
    graph = build_live_review_graph(
        run_id="RUN-B",
        phases=("plan", "execute_tools", "review_evidence", "generate_artifacts", "governance_signoff"),
        on_transition=lambda source, target, trigger, state_hash, thread_id: observer.graph_transition(
            source, target, trigger=trigger, state_hash=state_hash, thread_id=thread_id
        ),
    )
    graph.enter_phase("plan")
    graph.enter_phase("execute_tools")
    observer.human_action("Q", "market.cross_analytical_synthesis", "final evidence summary")
    observer.agent_card(
        "EvidenceReviewAgent",
        role="evidence interpretation",
        checkpoint="market.cross_analytical_synthesis",
    )
    observer.grounding_result(
        accepted=False,
        quantitative_claims=1,
        grounded_claims=0,
        continuation="EVIDENCE_ONLY",
        checkpoint="market.cross_analytical_synthesis",
    )
    observer.engine_card(
        "RegisteredDeterministicDispatcher",
        operation="continue from retained canonical evidence",
    )
    observer.engine_completed("RegisteredDeterministicDispatcher")
    observer.evidence_committed(_record("diagnostic.market_evidence", "EV-B-DIAG"))
    graph.enter_phase("review_evidence")
    graph.enter_phase("generate_artifacts")
    graph.enter_phase("governance_signoff")
    observer.governance_card(
        disposition="ACCEPT_WITH_CONDITIONS",
        evidence_count=1,
        unresolved_count=0,
        validation_failures=0,
    )
    observer.policy_gate(_policy())
    observer.trace_waterfall([_trace()])
    observer.attestation_sealed(merkle_root="c" * 64, leaf_count=1)

    kinds = [event.kind for event in observer.events]
    last_graph = max(i for i, kind in enumerate(kinds) if kind is PresentationEventKind.GRAPH_TRANSITION)
    assert last_graph < kinds.index(PresentationEventKind.GOVERNANCE_EVALUATED)
    assert last_graph < kinds.index(PresentationEventKind.ATTESTATION_SEALED)
    assert kinds.index(PresentationEventKind.HUMAN_ACTION) < kinds.index(
        PresentationEventKind.AGENT_ACTIVATED
    ) < kinds.index(PresentationEventKind.GROUNDING_REJECTED)
    assert kinds.index(PresentationEventKind.GROUNDING_REJECTED) < kinds.index(
        PresentationEventKind.ENGINE_STARTED
    ) < kinds.index(PresentationEventKind.GOVERNANCE_EVALUATED)
    assert set(observer.state.active_stage_history) >= {
        "HUMAN",
        "LANGGRAPH",
        "AGENT",
        "ENGINE",
        "EVIDENCE",
        "GROUNDING",
        "GOVERNANCE",
        "OPA",
        "TRACE",
        "ATTESTATION",
    }


def test_challenge_lineage_has_created_diagnostic_evidence_and_resolution_events() -> None:
    console, _ = _console()
    observer = TerminalReviewObserver("RUN-CHALLENGE", console=console, session_kind="B")
    observer.human_action("C", "scenario", "challenge scenario integrity")
    observer.challenge_created(
        checkpoint="scenario",
        note="challenge scenario integrity",
        source_evidence_ids=["EV-SOURCE"],
    )
    observer.diagnostic_started(
        "RegisteredDiagnosticDispatcher",
        operation="validate scenario data integrity",
        checkpoint="scenario",
    )
    observer.engine_completed("RegisteredDiagnosticDispatcher", checkpoint="scenario")
    observer.evidence_committed(_record("diagnostic.validate_scenario", "EV-DIAG"))
    observer.challenge_lineage(
        source_evidence_ids=["EV-SOURCE"],
        diagnostic="validate_scenario_data_integrity",
        generated_evidence_ids=["EV-DIAG"],
        resolution_status="EVIDENCE_GENERATED",
        checkpoint="scenario",
    )

    kinds = [event.kind for event in observer.events]
    assert kinds == [
        PresentationEventKind.HUMAN_ACTION,
        PresentationEventKind.CHALLENGE_CREATED,
        PresentationEventKind.DIAGNOSTIC_STARTED,
        PresentationEventKind.ENGINE_COMPLETED,
        PresentationEventKind.EVIDENCE_COMMITTED,
        PresentationEventKind.CHALLENGE_RESOLVED,
    ]


def test_runtime_event_sink_drives_session_c_control_plane_live() -> None:
    console, stream = _console()
    observer = TerminalReviewObserver(
        "RUN-C",
        console=console,
        session_kind="C",
        control_plane=True,
    )
    sink = RuntimePresentationSink(observer, workflow_id="quantitative_finance")
    sink.emit(RuntimeEvent(run_id="RUN-C", event_type="context_ready", node_id="step-context"))
    sink.emit(
        RuntimeEvent(
            run_id="RUN-C",
            event_type="workflow_resolved",
            node_id="step-context",
            metadata={"applicable_test_ids": ["portfolio.hierarchical_risk_parity"]},
        )
    )
    sink.emit(
        RuntimeEvent(
            run_id="RUN-C",
            event_type="test_completed",
            status="PASS",
            test_id="portfolio.hierarchical_risk_parity",
            evidence_refs=["EV-C"],
            metadata={"metrics": {"effective_n_positions": 3.0}},
        )
    )
    sink.emit(
        RuntimeEvent(
            run_id="RUN-C",
            event_type="evidence_committed",
            test_id="portfolio.hierarchical_risk_parity",
            evidence_refs=["EV-C"],
        )
    )
    sink.emit(
        RuntimeEvent(
            run_id="RUN-C",
            event_type="checkpoint_committed",
            checkpoint_id="CP-005",
        )
    )
    policy = vars(_policy())
    sink.emit(
        RuntimeEvent(
            run_id="RUN-C",
            event_type="governance_decided",
            metadata={
                "governance_disposition": "ACCEPT",
                "policy_decision": policy,
                "validation_failures": 0,
            },
        )
    )
    sink.emit(
        RuntimeEvent(
            run_id="RUN-C",
            event_type="trace_ready",
            metadata={"records": [_trace().to_dict()], "span_count": 1},
        )
    )
    sink.emit(
        RuntimeEvent(
            run_id="RUN-C",
            event_type="attestation_created",
            metadata={"merkle_root": "d" * 64, "leaf_count": 1},
        )
    )
    sink.emit(RuntimeEvent(run_id="RUN-C", event_type="workflow_completed"))
    observer.session_completed(source_component="fixture")

    text = stream.getvalue()
    assert "CONTROL-PLANE RIBBON" in text
    assert "ARCHITECTURE SPINE" not in text
    assert observer.state.stage_status["HUMAN"] == "NOT_APPLICABLE"
    assert observer.state.stage_status["LANGGRAPH"] == "NOT_APPLICABLE"
    assert [event.sequence for event in observer.events] == list(range(1, len(observer.events) + 1))
    kinds = [event.kind for event in observer.events]
    assert kinds.index(PresentationEventKind.OPA_EVALUATED) < kinds.index(
        PresentationEventKind.TRACE_READY
    ) < kinds.index(PresentationEventKind.ATTESTATION_SEALED)


def test_canonical_runtime_emits_trace_before_attestation(tmp_path) -> None:
    sink = ListEventSink()
    result = CanonicalExecutionService.execute(
        workflow_id="data_diagnostics",
        context_id="institutional_credit_v1",
        seed=42,
        output_root=str(tmp_path),
        trace_mode="engineering",
        event_sink=sink,
        run_id="RUN-C-LIVE-ORDER",
    )

    event_types = [event.event_type for event in sink.events]
    assert event_types.index("governance_decided") < event_types.index(
        "trace_ready"
    ) < event_types.index("attestation_created") < event_types.index("workflow_completed")
    trace_event = next(event for event in sink.events if event.event_type == "trace_ready")
    assert trace_event.metadata["span_count"] > 0
    assert trace_event.metadata["records"]
    assert result.merkle_root


def test_curated_evidence_policy_keeps_full_count_and_compact_summary() -> None:
    console, stream = _console()
    observer = TerminalReviewObserver("RUN-E", console=console, session_kind="B")
    observer.evidence_committed(_record("portfolio.hierarchical_risk_parity", "EV-HRP"))
    observer.evidence_committed(_record("portfolio.black_litterman", "EV-BL"))
    observer.evidence_committed(_record("scenario.reverse_stress", "EV-RS"))
    observer.evidence_batch_summary()

    text = stream.getvalue()
    assert observer.state.evidence_count == 3
    assert text.count("EVIDENCE COMMITTED") == 2
    assert "+ 1 additional canonical EvidenceRecords committed" in text
    assert "portfolio=2" in text and "scenario=1" in text


def test_demo_progress_is_result_derived_and_session_scoped() -> None:
    console, stream = _console()
    single = DemoProgress(["SESSION_B_MARKET"])
    single.start("SESSION_B_MARKET")
    console.print(single.render())
    assert "FLIGHT B" in stream.getvalue()
    assert "FLIGHT C" not in stream.getvalue()

    combined = DemoProgress(
        ["SESSION_A_TEMPORAL", "SESSION_B_MARKET", "SESSION_C_POLICY"]
    )
    combined.start("SESSION_A_TEMPORAL")
    combined.finish(
        SessionResult(
            session_id="SESSION_A_TEMPORAL",
            controller_status=SessionStatus.COMPLETE,
        )
    )
    assert combined.states == {
        "SESSION_A_TEMPORAL": "COMPLETE",
        "SESSION_B_MARKET": "NOT_STARTED",
        "SESSION_C_POLICY": "NOT_STARTED",
    }
    combined.start("SESSION_B_MARKET")
    assert combined.states["SESSION_B_MARKET"] == "ACTIVE"
    assert combined.states["SESSION_C_POLICY"] == "NOT_STARTED"
    combined.finish(
        SessionResult(
            session_id="SESSION_B_MARKET",
            controller_status=SessionStatus.COMPLETE,
            opa_decision="OPA_LOCAL:DENY",
            attestation_merkle_state="WITHHELD",
        )
    )
    assert combined.opa == "DENY"
    assert combined.attestation == "WITHHELD"


def test_presentation_mode_suppresses_controller_debug_chatter(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_presentation_mode(True)
    try:
        log("[SESSION_A_TEMPORAL] State A_MODE: awaiting prompt boundary...")
        captured = capsys.readouterr()
        assert captured.err == ""
        assert captured.out == ""
    finally:
        configure_presentation_mode(False)


def test_offline_temporal_twin_responses_are_semantically_distinct() -> None:
    provider = OfflineDemoTwinProvider()

    def response(question: str) -> str:
        return provider.complete(
            "StART review agent.",
            (
                "You are ArchitectureReviewAgent, a model-risk review agent. "
                f'A reviewer asked: "{question}".\n'
                "Your recommendation: lstm."
            ),
        )

    architecture = response("When is recurrent LSTM architecture justified?")
    modality = response("How do we rule out a reshaped tabular fake sequence?")
    leakage = response("How is train/test/OOS leakage prevented during preprocessing?")
    assert len({architecture, modality, leakage}) == 3
    assert "invalidated" in architecture
    assert "rank-3" in modality and "fake sequence" in modality
    assert "training cohort only" in leakage and "metadata" in leakage


def test_normalized_event_render_forensics_fixture() -> None:
    console, stream = _console()
    observer = TerminalReviewObserver("RUN-PTY", console=console, session_kind="A")
    observer.session_started(source_component="fixture")
    observer.human_action("C", "architecture", "challenge")
    observer.graph_transition(
        "plan",
        "execute_tools",
        trigger="LANGGRAPH_PHASE_GATE",
        state_hash="e" * 64,
        thread_id="thread-RUN-PTY",
    )
    observer.agent_card("ArchitectureReviewAgent", role="architecture", checkpoint="architecture")
    observer.engine_card("SequenceClassifier", operation="fit and evaluate")
    observer.engine_completed("SequenceClassifier")
    observer.evidence_committed(_record("deep_learning.performance_diagnostics"))
    observer.evidence_batch_summary()
    observer.governance_card(
        disposition="ACCEPT",
        evidence_count=1,
        unresolved_count=0,
        validation_failures=0,
    )
    observer.policy_gate(_policy())
    observer.trace_waterfall([_trace()])
    observer.attestation_sealed(merkle_root="f" * 64, leaf_count=1)
    observer.session_completed(source_component="fixture")

    normalized = stream.getvalue()
    assert normalized.count("ARCHITECTURE SPINE") >= 8
    assert "◉ HUMAN" in normalized
    assert "◉ LANGGRAPH" in normalized
    assert "◉ ENGINE" in normalized
    assert "EVIDENCE LEDGER" in normalized
    assert "StART — REVIEW COMPLETE" in normalized
    assert "[CONTROLLER]" not in normalized
    assert max(len(line) for line in normalized.splitlines()) <= 132
