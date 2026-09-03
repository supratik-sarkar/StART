"""Tests for non-interactive provider probe utility and CLI command."""

from __future__ import annotations

from unittest.mock import MagicMock

from typer.testing import CliRunner

from start.cli.main import app
from start.cli.provider_cli import (
    EXIT_MISSING_CREDENTIAL,
    EXIT_PROVIDER_FAILURE,
    EXIT_SUCCESS,
    run_provider_probe,
)

runner = CliRunner()


def test_provider_probe_missing_credential_exits_distinct_code(monkeypatch):
    """When credential is missing, probe must NOT prompt, must NOT attempt live call, and must return code 2."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("start.providers.keys.keychain_has_key", lambda p: False)
    monkeypatch.setattr("start.providers.keys.keychain_get_key", lambda p: None)

    code = run_provider_probe("openai", model="gpt-5-mini")
    assert code == EXIT_MISSING_CREDENTIAL


def test_provider_probe_live_success_mock(monkeypatch):
    """When credential is present and provider succeeds, returns 0 and prints metadata."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")
    monkeypatch.setattr("start.providers.keys.keychain_has_key", lambda p: True)

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.id = "resp_probe_123"
    mock_resp.output_text = "START_PROVIDER_OK"
    mock_resp.usage.input_tokens = 10
    mock_resp.usage.output_tokens = 5
    mock_client.responses.create.return_value = mock_resp
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)

    code = run_provider_probe("openai", model="gpt-5-mini")
    assert code == EXIT_SUCCESS


def test_provider_probe_failure_returns_exit_1(monkeypatch):
    """When provider API call fails with an exception, returns 1."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")
    monkeypatch.setattr("start.providers.keys.keychain_has_key", lambda p: True)

    mock_client = MagicMock()
    mock_client.responses.create.side_effect = RuntimeError("API rate limit exceeded")
    mock_client.chat.completions.create.side_effect = RuntimeError("API rate limit exceeded")
    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)

    code = run_provider_probe("openai", model="gpt-5-mini")
    assert code == EXIT_PROVIDER_FAILURE


def test_cli_provider_probe_invocation_missing_gemini(monkeypatch):
    """CLI start provider probe --provider gemini exits 2 when key is missing."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("start.providers.keys.keychain_has_key", lambda p: False)
    monkeypatch.setattr("start.providers.keys.keychain_get_key", lambda p: None)

    res = runner.invoke(app, ["provider", "probe", "--provider", "gemini"])
    assert res.exit_code == EXIT_MISSING_CREDENTIAL
    assert "MISSING_CREDENTIAL" in res.stdout
