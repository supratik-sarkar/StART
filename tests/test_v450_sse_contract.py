"""Targeted Test Suite for StART v4.5 Typed SSE Envelope Contract & Reconnection."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from start.web.app import create_app
from start.web.queue import GLOBAL_QUEUE
from start.web.schemas import START_SCHEMA_VERSION, RunRequest


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


@pytest.mark.asyncio
async def test_sse_envelope_fields_and_monotonic_sequence() -> None:
    from start.web.sse import sse_event_generator

    run_id = "RUN-TEST-SSE-CONTRACT-01"
    session_id = "SES-SSE-01"
    req = RunRequest(session_id=session_id, domain="market")
    GLOBAL_QUEUE.submit_run(run_id, req)

    # Append 3 distinct events
    GLOBAL_QUEUE.append_event(
        run_id,
        {
            "event_id": "EVT-001",
            "timestamp": 1725370001.0,
            "event_type": "agent_transition",
            "source_agent": "Director",
            "target_agent": "MarketSpecialist",
            "stage": "PLANNING",
            "action": "discover_tests",
            "status": "SUCCESS",
            "metadata": {"param": 1},
        },
    )
    GLOBAL_QUEUE.append_event(
        run_id,
        {
            "event_id": "EVT-002",
            "timestamp": 1725370002.0,
            "event_type": "tool_execution",
            "source_agent": "MarketSpecialist",
            "target_agent": "DeterministicEngine",
            "stage": "EXECUTION",
            "action": "portfolio.hrp_weights",
            "status": "SUCCESS",
            "metadata": {"param": 2},
        },
    )

    GLOBAL_QUEUE.mark_completed(run_id, [], {})

    envelopes = []
    async for item in sse_event_generator(run_id, session_id, poll_interval=0.01):
        if item["event"] != "complete":
            parsed = json.loads(item["data"])
            envelopes.append(parsed)

    assert len(envelopes) == 2

    # Verify Monotonic sequence
    assert envelopes[0]["sequence"] == 1
    assert envelopes[1]["sequence"] == 2

    # Verify All Mandatory Fields
    for env in envelopes:
        assert "event_id" in env
        assert "sequence" in env
        assert "run_id" in env
        assert "timestamp" in env
        assert "event_type" in env
        assert "schema_version" in env
        assert env["schema_version"] == START_SCHEMA_VERSION
        assert "payload" in env


@pytest.mark.asyncio
async def test_sse_reconnection_with_last_event_id() -> None:
    from start.web.sse import sse_event_generator

    run_id = "RUN-TEST-SSE-RECONNECT-01"
    session_id = "SES-SSE-02"
    req = RunRequest(session_id=session_id, domain="market")
    GLOBAL_QUEUE.submit_run(run_id, req)

    for i in range(4):
        GLOBAL_QUEUE.append_event(
            run_id,
            {
                "event_id": f"EVT-00{i + 1}",
                "timestamp": 1725370000.0 + i,
                "event_type": "tool_execution",
                "source_agent": "Specialist",
                "target_agent": "Engine",
                "stage": "EXECUTION",
                "action": f"test_{i}",
            },
        )
    GLOBAL_QUEUE.mark_completed(run_id, [], {})

    # Client reconnects with Last-Event-ID: EVT-SEQ-2
    reconnected_envelopes = []
    async for item in sse_event_generator(run_id, session_id, last_event_id="EVT-SEQ-2", poll_interval=0.01):
        if item["event"] != "complete":
            reconnected_envelopes.append(json.loads(item["data"]))

    # Should only deliver events 3 and 4
    assert len(reconnected_envelopes) == 2
    assert reconnected_envelopes[0]["sequence"] == 3
    assert reconnected_envelopes[1]["sequence"] == 4
