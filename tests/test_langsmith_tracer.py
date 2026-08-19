"""Tests for LangSmith tracer containment, envelope projection, and fail-safe behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

from start.ai_engineering.langsmith_tracer import LangSmithTracer, is_langsmith_enabled
from start.attestation.disclosure import DisclosureEnvelope


def test_langsmith_disabled_in_enterprise_by_default() -> None:
    env = {
        "START_PROFILE": "enterprise",
        "LANGSMITH_API_KEY": "ls_test_key_presence_only",
    }
    assert is_langsmith_enabled(env) is False


def test_langsmith_enabled_in_enterprise_with_override() -> None:
    env = {
        "START_PROFILE": "enterprise",
        "START_ALLOW_TELEMETRY_EGRESS": "true",
        "LANGSMITH_API_KEY": "ls_test_key_presence_only",
    }
    assert is_langsmith_enabled(env) is True


def test_langsmith_disabled_by_explicit_flag() -> None:
    env = {
        "START_PROFILE": "public_demo",
        "LANGSMITH_API_KEY": "ls_test_key_presence_only",
        "START_LANGSMITH_ENABLED": "false",
    }
    assert is_langsmith_enabled(env) is False


def test_tracer_refuses_payload_without_envelope() -> None:
    tracer = LangSmithTracer()
    tracer.client = MagicMock()

    # Attempt to trace with raw string / None envelope
    run_id = tracer.trace_llm_call(
        envelope=None,  # No envelope
        completion="Some completion",
        provider="openai",
    )
    assert run_id is None
    tracer.client.create_run.assert_not_called()


def test_tracer_sends_rendered_prompt_byte_identically() -> None:
    envelope = DisclosureEnvelope(
        policy_id="test-policy",
        projected={"supervised.auc": 0.85},
        withheld_paths=(),
        numeric_surface=frozenset({"0.85"}),
    )
    tracer = LangSmithTracer()
    mock_client = MagicMock()
    tracer.client = mock_client
    tracer.root_run_id = "root-123"

    tracer.trace_llm_call(
        envelope=envelope,
        completion="AUC is 0.85",
        latency_ms=120.5,
        provider="openai",
        model="gpt-4o-mini",
    )

    mock_client.create_run.assert_called_once()
    call_args = mock_client.create_run.call_args[1]
    assert call_args["inputs"]["prompt"] == envelope.render()
    assert call_args["outputs"]["completion"] == "AUC is 0.85"
    assert call_args["extra"]["metadata"]["envelope_hash"] == envelope.envelope_hash()
    assert call_args["extra"]["metadata"]["provider"] == "openai"


def test_tracer_fail_safe_on_client_exception() -> None:
    tracer = LangSmithTracer()
    mock_client = MagicMock()
    mock_client.create_run.side_effect = RuntimeError("Network error")
    tracer.client = mock_client
    tracer.root_run_id = "root-123"

    envelope = DisclosureEnvelope(
        policy_id="test-policy",
        projected={"supervised.auc": 0.85},
        withheld_paths=(),
        numeric_surface=frozenset({"0.85"}),
    )

    # Must not raise
    run_id = tracer.trace_llm_call(
        envelope=envelope,
        completion="Test completion",
    )
    assert run_id is None
