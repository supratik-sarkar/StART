from __future__ import annotations

from pathlib import Path

import pytest

from start.providers.keys import (
    PROVIDER_KEY_ENV,
    ensure_provider_key,
    key_required,
    resolve_key_databricks,
    run_llm_check,
)

FAKE_KEY = "sk-test-FAKE-KEY-do-not-leak-1234567890"


@pytest.fixture(autouse=True)
def _clean_key_env(monkeypatch):
    for env_var in PROVIDER_KEY_ENV.values():
        if env_var:
            monkeypatch.delenv(env_var, raising=False)


# --------------------------------------------------------------------------- #
# Mapping + policy
# --------------------------------------------------------------------------- #
def test_provider_key_mapping_matches_spec():
    assert PROVIDER_KEY_ENV["openai"] == "OPENAI_API_KEY"
    assert PROVIDER_KEY_ENV["anthropic"] == "ANTHROPIC_API_KEY"
    assert PROVIDER_KEY_ENV["grok"] == "GROK_API_KEY"  # never XAI_API_KEY
    assert PROVIDER_KEY_ENV["huggingface"] == "HF_TOKEN"
    assert PROVIDER_KEY_ENV["hf_local"] is None
    assert PROVIDER_KEY_ENV["enterprise_llm_gateway"] is None
    assert PROVIDER_KEY_ENV["none"] is None
    assert not key_required("hf_local") and key_required("openai")
    # XAI_API_KEY must not appear anywhere in the provider layer
    llm_src = Path("src/start/providers/llm.py").read_text()
    assert "XAI_API_KEY" not in llm_src


def test_unknown_provider_fails_clearly():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        ensure_provider_key("magicllm")
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        resolve_key_databricks("magicllm")


def test_missing_key_noninteractive_refuses(monkeypatch):
    status = ensure_provider_key("openai", prompt_for_key=False)
    assert status.source == "missing" and not status.ok
    # auto mode on a non-tty also refuses without prompting
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    auto = ensure_provider_key("anthropic", prompt_for_key=None)
    assert auto.source == "missing"


def test_prompt_mode_uses_hidden_getpass(monkeypatch):
    import os

    prompts: list[str] = []

    def fake_getpass(prompt: str) -> str:
        prompts.append(prompt)
        return FAKE_KEY

    monkeypatch.setattr("getpass.getpass", fake_getpass)
    status = ensure_provider_key("openai", prompt_for_key=True)
    assert status.source == "hidden prompt/session env"
    assert prompts == ["Enter OPENAI_API_KEY: "]
    assert os.environ["OPENAI_API_KEY"] == FAKE_KEY  # session env only


def test_hf_local_never_prompts(monkeypatch):
    def explode(prompt: str) -> str:
        raise AssertionError("getpass must not be called for hf_local")

    monkeypatch.setattr("getpass.getpass", explode)
    status = ensure_provider_key("hf_local", prompt_for_key=True)
    assert status.source == "not required"
    assert ensure_provider_key("enterprise_llm_gateway", prompt_for_key=True).source == "not required"


# --------------------------------------------------------------------------- #
# Databricks resolution order: secret scope -> env -> missing
# --------------------------------------------------------------------------- #
class _FakeSecrets:
    def __init__(self, store: dict[str, str]) -> None:
        self._store = store

    def get(self, scope: str, key: str) -> str:
        return self._store[f"{scope}/{key}"]


class _FakeDbutils:
    def __init__(self, store: dict[str, str]) -> None:
        self.secrets = _FakeSecrets(store)


def test_databricks_secret_scope_preferred(monkeypatch):
    import os

    dbu = _FakeDbutils({"start/ANTHROPIC_API_KEY": FAKE_KEY})
    status = resolve_key_databricks("anthropic", dbutils=dbu, scope="start")
    assert status.source == "secret scope"
    assert os.environ["ANTHROPIC_API_KEY"] == FAKE_KEY

    # absent from scope -> env wins
    monkeypatch.setenv("GROK_API_KEY", FAKE_KEY)
    env_status = resolve_key_databricks("grok", dbutils=_FakeDbutils({}), scope="start")
    assert env_status.source == "env"

    missing = resolve_key_databricks("openai", dbutils=_FakeDbutils({}), scope="start")
    assert missing.source == "missing"


# --------------------------------------------------------------------------- #
# llm-check
# --------------------------------------------------------------------------- #
def test_llm_check_with_fake_provider_passes_citation_gate():
    class CitingFake:
        available = True

        def __init__(self):
            self.calls = []

        def generate(self, prompt, *, system=None, metadata=None):
            max_tokens = (metadata or {}).get("max_tokens", 1024)
            return self.complete(system or "", prompt, max_tokens=max_tokens)

        def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
            self.calls.append((system, user))
            import re

            ev = re.search(r"\[EV-[A-Za-z0-9]+\]", user).group(0)
            return f"The synthetic discrimination check passed cleanly. {ev}"

    fake = CitingFake()
    result = run_llm_check("openai", llm=fake)

    assert result["provider"] == "openai"
    assert result["mode"] == "llm-assisted"
    assert result["synthetic_evidence_sent"] == "yes"
    assert result["raw_dataset_sent"] == "no"
    assert result["critique"] == "passed"


def test_llm_check_uncited_output_fails_gate():
    class UncitedFake:
        available = True

        def generate(self, prompt, *, system=None, metadata=None):
            return "The AUC is 0.9 and everything is great."

    fake = UncitedFake()
    result = run_llm_check("openai", llm=fake)

    assert result["provider"] == "openai"
    assert result["mode"] == "llm-assisted"
    assert result["synthetic_evidence_sent"] == "yes"
    assert result["raw_dataset_sent"] == "no"
    assert result["critique"] == "failed"
