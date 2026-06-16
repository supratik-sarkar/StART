"""Provider trust-domain separation.

Two trust domains that must never cross:

    PUBLIC   - openai, anthropic, grok  (public SaaS LLM APIs)
    PRIVATE  - enterprise_llm_gateway   (a firm's internal gateway, isolated in
               src/start/enterprise/)

Enforced invariants:
  * No routing crossover: a request bound to one domain never resolves to a
    provider in the other.
  * No key sharing: public providers read public env vars; the enterprise
    gateway reads only its own configuration. Helpers here never copy a key
    from one domain into the other.
  * No fallback across domains: if a public provider is unavailable it degrades
    to the deterministic no-LLM path (same domain), NEVER to the enterprise
    gateway, and vice versa.

This module is the single place that maps a provider name to its trust domain
and validates that a requested provider is allowed for a given mode.
"""

from __future__ import annotations

from enum import Enum


class TrustDomain(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    NONE = "none"


PUBLIC_PROVIDERS = ("openai", "anthropic", "grok")
PRIVATE_PROVIDERS = ("enterprise_llm_gateway",)


def trust_domain(provider: str) -> TrustDomain:
    if provider in PUBLIC_PROVIDERS:
        return TrustDomain.PUBLIC
    if provider in PRIVATE_PROVIDERS:
        return TrustDomain.PRIVATE
    return TrustDomain.NONE


def is_public(provider: str) -> bool:
    return trust_domain(provider) == TrustDomain.PUBLIC


def is_private(provider: str) -> bool:
    return trust_domain(provider) == TrustDomain.PRIVATE


class TrustDomainViolation(RuntimeError):
    """Raised when a provider would cross trust domains."""


def assert_no_crossover(provider: str, expected_domain: TrustDomain) -> None:
    """Guard: a provider must belong to the expected trust domain. Used at the
    routing boundary so a public request can never resolve a private provider
    (or vice versa)."""
    actual = trust_domain(provider)
    if provider in ("none", "") or actual == TrustDomain.NONE:
        return  # deterministic / no-LLM is domain-agnostic and always allowed
    if actual != expected_domain:
        raise TrustDomainViolation(
            f"Provider '{provider}' is in the {actual.value} trust domain but the "
            f"{expected_domain.value} domain was requested. Crossover is not permitted: "
            "public providers (OpenAI/Anthropic/Grok) and the enterprise gateway are "
            "isolated, share no keys, and never fall back to one another."
        )


def resolve_within_domain(provider: str) -> TrustDomain:
    """Return the trust domain a provider resolves within. The caller uses this
    to ensure any degradation stays inside the same domain (public -> no-LLM,
    private -> enterprise-unavailable), never crossing over."""
    return trust_domain(provider)
