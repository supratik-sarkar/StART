import os
import pathlib
import shutil
import sys

import pytest
from pydantic import ValidationError

# Enforce explicit src/ routing boundaries for pytest collection frames
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from start.engine.state import StepCheckpointer
from start.telemetry.bus import AgentMessageTrace, TelemetryBus, TelemetryEvent


def test_agent_message_trace_citation_enforcement():
    """Enforces that any message trace without a proper 'EV-' prefix raises a validation exception."""
    valid_trace = AgentMessageTrace(
        reasoning_step="Analyzing feature drift via population stability indexing.",
        evidence_citations=["EV-DQ-001"],
        confidence_score=0.98,
    )
    assert valid_trace.confidence_score == 0.98

    with pytest.raises(ValidationError) as exc_info:
        AgentMessageTrace(
            reasoning_step="Invalid trace containing unstructured citation strings.",
            evidence_citations=["BAD-CITATION-123"],
            confidence_score=0.5,
        )
    assert "Meticulous Communication Guardrail Violation" in str(exc_info.value)


def test_step_checkpointer_atomic_lifecycle():
    """Verifies atomic transactional state storage and structural resilience options."""
    test_cache_dir = "~/.state_cache_test_v240"
    resolved_path = pathlib.Path(os.path.expanduser(test_cache_dir))

    if resolved_path.exists():
        shutil.rmtree(resolved_path)

    checkpointer = StepCheckpointer(storage_dir=test_cache_dir)
    workflow_id = "WF-TEST-RECOVERY"
    stage_name = "Hyperparameter Optimization"
    payload = {"learning_rate": 0.005, "batch_size": 32}

    saved_path = checkpointer.save_checkpoint(workflow_id, stage_name, payload)
    assert os.path.exists(saved_path)

    loaded = checkpointer.load_checkpoint(workflow_id)
    assert loaded is not None
    assert loaded["workflow_id"] == workflow_id
    assert loaded["last_completed_stage"] == stage_name
    assert loaded["payload"]["learning_rate"] == 0.005

    checkpointer.clear_checkpoint(workflow_id)
    if resolved_path.exists():
        shutil.rmtree(resolved_path)


def test_telemetry_bus_broadcast_loop():
    """Verifies thread-safe registration and non-blocking event streaming loops."""
    bus = TelemetryBus()
    received_events = []

    def dummy_subscriber(event: TelemetryEvent) -> None:
        received_events.append(event)

    bus.subscribe(dummy_subscriber)

    test_event = TelemetryEvent(
        agent_name="Feature Engineering Agent",
        stage="Encoding Checks",
        progress_percentage=75.0,
        status_msg="Categorical feature cardinality optimization complete.",
    )

    bus.publish(test_event)

    assert len(received_events) == 1
    assert received_events[0].agent_name == "Feature Engineering Agent"
    assert received_events[0].progress_percentage == 75.0

    snapshot = bus.fetch_agent_snapshot("Feature Engineering Agent")
    assert snapshot is not None
    assert snapshot.status_msg == "Categorical feature cardinality optimization complete."


def test_cost_optimization_agent_routing():
    """Verifies that the Cost Optimization Agent generates precision adjustment directives under load."""
    from start.agents.cost_optimization import CostOptimizationAgent
    from start.telemetry.bus import TelemetryBus

    bus = TelemetryBus()
    agent = CostOptimizationAgent(telemetry_bus=bus)

    context = {
        "evidence_id": "EV-COST-002",
        "budget_limit_seconds": 2.0,
        "metrics": {"runtime_seconds": 4.82, "peak_memory_mb": 512.0},
    }

    result = agent.execute(context)
    assert result["status"] == "optimized"
    assert result["action_directive"]["strategy"] == "optimize_compute_footprint"
    assert result["action_directive"]["parameters"]["apply_quantization"] is True


def test_fairness_agent_violation_guardrails():
    """Verifies that the Fairness Agent flags bias profiles when disparate impact trends drop below 80%."""
    from start.agents.fairness import FairnessAgent
    from start.telemetry.bus import TelemetryBus

    bus = TelemetryBus()
    agent = FairnessAgent(telemetry_bus=bus)

    context = {
        "evidence_id": "EV-FAIR-09",
        "fairness_threshold": 0.80,
        "protected_attribute": "zip_code",
        "metrics": {"disparate_impact_ratio": 0.712},
    }

    result = agent.execute(context)
    assert result["status"] == "evaluated"
    assert result["action_directive"]["strategy"] == "calibrate_decision_boundaries"
    assert result["action_directive"]["parameters"]["shift_threshold_delta"] == 0.05
