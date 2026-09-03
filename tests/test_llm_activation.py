from __future__ import annotations

from start.core.config import LLMConfig
from start.providers.llm import get_llm_provider
from start.providers.llm_activation import (
    preflight_llm,
    render_activation_markdown,
)


def test_none_provider_is_deterministic():
    r = preflight_llm("none", None)
    assert r.status == "DETERMINISTIC"
    assert r.trust_domain == "none"


def test_unavailable_public_provider_is_explicit_fallback(monkeypatch):
    monkeypatch.setenv("START_PROFILE", "public_demo")
    for var in ("OPENAI_API_KEY", "OPENAI_KEY", "AZURE_OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    llm = get_llm_provider(LLMConfig(provider="openai"), expected_domain="public")
    r = preflight_llm("openai", llm)
    assert r.status == "FALLBACK"  # never silent
    assert r.provider == "openai"
    assert r.trust_domain == "public"
    assert "key" in r.detail.lower()


def test_activation_report_shows_endpoint_and_model(monkeypatch):
    monkeypatch.setenv("START_PROFILE", "public_demo")
    for var in ("OPENAI_API_KEY",):
        monkeypatch.delenv(var, raising=False)
    llm = get_llm_provider(LLMConfig(provider="openai"), expected_domain="public")
    r = preflight_llm("openai", llm)
    assert r.endpoint.startswith("https://")
    assert r.model  # default model shown even when unavailable


def test_enterprise_gateway_trust_domain():
    llm = get_llm_provider(LLMConfig(provider="enterprise_llm_gateway"))
    r = preflight_llm("enterprise_llm_gateway", llm)
    assert r.trust_domain == "private"


def test_terminal_render_contains_status():
    r = preflight_llm("none", None)
    out = r.render_terminal()
    assert "Status" in out and "DETERMINISTIC" in out
    assert "Provider" in out and "Trust domain" in out


def test_markdown_render():
    r = preflight_llm("none", None)
    md = render_activation_markdown(r)
    assert "### LLM activation" in md
    assert "| Provider |" in md


def test_to_dict_complete():
    r = preflight_llm("none", None)
    d = r.to_dict()
    for key in ("provider", "model", "trust_domain", "endpoint", "status"):
        assert key in d


class _FakeConnected:
    available = True
    model = "fake-model"

    def complete(self, system, user, *, output_token_budget=5, **kwargs):
        return "ok"


class _FakeBroken:
    available = True
    model = "fake-model"

    def complete(self, system, user, *, output_token_budget=5, **kwargs):
        raise RuntimeError("network down")


def test_probe_connected():
    r = preflight_llm("openai", _FakeConnected(), probe=True)
    assert r.status == "CONNECTED"


def test_probe_failure_is_surfaced_not_hidden():
    r = preflight_llm("openai", _FakeBroken(), probe=True)
    assert r.status == "FAILED"
    assert "failed" in r.detail.lower()


def test_gemini_and_deepseek_trust_domains_and_endpoints(monkeypatch):
    monkeypatch.setenv("START_PROFILE", "public_demo")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    gemini_llm = get_llm_provider(LLMConfig(provider="gemini"), expected_domain="public")
    r_gemini = preflight_llm("gemini", gemini_llm)
    assert r_gemini.status == "FALLBACK"
    assert r_gemini.trust_domain == "public"
    assert "generativelanguage" in r_gemini.endpoint
    assert r_gemini.model == "gemini-1.5-flash"

    deepseek_llm = get_llm_provider(LLMConfig(provider="deepseek"), expected_domain="public")
    r_deepseek = preflight_llm("deepseek", deepseek_llm)
    assert r_deepseek.status == "FALLBACK"
    assert r_deepseek.trust_domain == "public"
    assert "deepseek.com" in r_deepseek.endpoint
    assert r_deepseek.model == "deepseek-chat"
