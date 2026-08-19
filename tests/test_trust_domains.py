from __future__ import annotations

import pytest

from start.core.config import LLMConfig
from start.providers.llm import NoLLMProvider, get_llm_provider
from start.providers.trust_domains import (
    PRIVATE_PROVIDERS,
    PUBLIC_PROVIDERS,
    TrustDomain,
    TrustDomainViolation,
    assert_no_crossover,
    is_private,
    is_public,
    trust_domain,
)


def test_domain_mapping():
    for p in PUBLIC_PROVIDERS:
        assert trust_domain(p) == TrustDomain.PUBLIC and is_public(p)
    for p in PRIVATE_PROVIDERS:
        assert trust_domain(p) == TrustDomain.PRIVATE and is_private(p)
    assert trust_domain("none") == TrustDomain.NONE


def test_public_providers_are_exactly_the_five():
    assert set(PUBLIC_PROVIDERS) == {"openai", "anthropic", "grok", "gemini", "deepseek"}
    assert set(PRIVATE_PROVIDERS) == {"enterprise_llm_gateway"}


def test_crossover_blocked_both_ways():
    with pytest.raises(TrustDomainViolation):
        assert_no_crossover("openai", TrustDomain.PRIVATE)
    with pytest.raises(TrustDomainViolation):
        assert_no_crossover("enterprise_llm_gateway", TrustDomain.PUBLIC)


def test_same_domain_allowed():
    assert_no_crossover("openai", TrustDomain.PUBLIC)
    assert_no_crossover("anthropic", TrustDomain.PUBLIC)
    assert_no_crossover("enterprise_llm_gateway", TrustDomain.PRIVATE)


def test_none_is_domain_agnostic():
    # deterministic / no-LLM is allowed in either domain
    assert_no_crossover("none", TrustDomain.PUBLIC)
    assert_no_crossover("none", TrustDomain.PRIVATE)
    assert_no_crossover("", TrustDomain.PUBLIC)


def test_public_unavailable_degrades_within_public_domain(monkeypatch):
    # With no usable key, openai is unavailable and must degrade to the
    # domain-neutral NoLLM path (public domain), never the enterprise gateway.
    # Clear every public-provider key so this is hermetic regardless of the
    # ambient environment (item 5).
    monkeypatch.setenv("START_PROFILE", "public_demo")
    for var in ("OPENAI_API_KEY", "OPENAI_KEY", "AZURE_OPENAI_API_KEY",
                "ANTHROPIC_API_KEY", "GROK_API_KEY", "XAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    prov = get_llm_provider(LLMConfig(provider="openai"), expected_domain="public")
    assert isinstance(prov, NoLLMProvider)


def test_requesting_wrong_domain_raises(monkeypatch):
    monkeypatch.setenv("START_PROFILE", "public_demo")
    with pytest.raises(TrustDomainViolation):
        get_llm_provider(LLMConfig(provider="enterprise_llm_gateway"), expected_domain="public")
    with pytest.raises(TrustDomainViolation):
        get_llm_provider(LLMConfig(provider="openai"), expected_domain="private")


def test_no_expected_domain_still_resolves():
    # Backward compatible: without expected_domain, resolution still works.
    prov = get_llm_provider(LLMConfig(provider="none"))
    assert isinstance(prov, NoLLMProvider)
