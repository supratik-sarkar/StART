"""LLM activation preflight (v2.1.1 Section A).

Before any agent runs in LLM mode, this produces a visible activation report so
the user can tell whether LLM execution will actually occur:

  Provider / Model / Trust Domain / Endpoint / Status

Status is one of CONNECTED, FAILED, FALLBACK, NOT_CONFIGURED — never a silent
degradation. The key is never echoed or stored here; this module only reads
availability (which checks the environment) and, optionally, performs a single
lightweight connectivity probe.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from start.providers.trust_domains import trust_domain

# Public, well-known endpoints shown for transparency (no secrets).
_ENDPOINTS = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "grok": "https://api.x.ai/v1",
    "enterprise_llm_gateway": "configured by the firm (src/start/enterprise/)",
    "none": "—",
}

_DEFAULT_MODELS = {
    "openai": "gpt-4.1",
    "anthropic": "claude-sonnet-4-6",
    "grok": "grok-3",
    "enterprise_llm_gateway": "gateway-managed",
}


@dataclass
class ActivationReport:
    provider: str
    model: str
    trust_domain: str
    endpoint: str
    status: str  # CONNECTED | FAILED | FALLBACK | NOT_CONFIGURED | DETERMINISTIC
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "trust_domain": self.trust_domain,
            "endpoint": self.endpoint,
            "status": self.status,
            "detail": self.detail,
        }

    def render_terminal(self) -> str:
        return (
            "  LLM activation\n"
            f"    Provider     : {self.provider}\n"
            f"    Model        : {self.model}\n"
            f"    Trust domain : {self.trust_domain}\n"
            f"    Endpoint     : {self.endpoint}\n"
            f"    Status       : {self.status}"
            + (f"\n    Detail       : {self.detail}" if self.detail else "")
        )


def preflight_llm(provider_name: str, llm: Any = None, *, probe: bool = False) -> ActivationReport:
    """Build an activation report for the chosen provider.

    ``llm`` is the resolved provider object (may be a degraded NoLLMProvider).
    ``probe`` optionally attempts a single tiny completion to confirm
    connectivity; off by default to avoid spending tokens.
    """
    domain = trust_domain(provider_name).value
    endpoint = _ENDPOINTS.get(provider_name, "—")

    # v2.3.1 #5: for the enterprise gateway, surface the configured private
    # package name and a safe endpoint ONLY if the private package/config
    # exposes one explicitly; otherwise show the route with the endpoint hidden.
    # Never read or display secrets.
    if provider_name == "enterprise_llm_gateway":
        # Resolve the configured private-package name from the environment
        # directly (mirrors the gateway adapter's precedence) so this module
        # has no dependency on the gateway internals.
        pkg = (
            os.environ.get("START_ENTERPRISE_LLM_PACKAGE")
            or os.environ.get("START_ENTERPRISE_PACKAGE")
            or "enterprise_package"
        )
        safe_endpoint = os.environ.get("START_ENTERPRISE_LLM_ENDPOINT_PUBLIC")
        if safe_endpoint:
            endpoint = f"{pkg} -> {safe_endpoint}"
        else:
            endpoint = f"private-package route ({pkg}); endpoint hidden"

    if provider_name in ("none", "") or provider_name is None:
        return ActivationReport(
            provider="none", model="—", trust_domain="none", endpoint="—",
            status="DETERMINISTIC", detail="No LLM selected; deterministic engines only.",
        )

    model = getattr(llm, "model", None) or _DEFAULT_MODELS.get(provider_name, "unknown")

    # Detect degradation: the resolver returns a NoLLMProvider when a public
    # provider is unavailable. That is an explicit FALLBACK, not silent.
    llm_class = type(llm).__name__ if llm is not None else ""
    available = bool(getattr(llm, "available", False))

    if llm is None:
        return ActivationReport(
            provider=provider_name, model=model, trust_domain=domain, endpoint=endpoint,
            status="NOT_CONFIGURED", detail="Provider not resolved.",
        )
    if llm_class == "NoLLMProvider" or not available:
        return ActivationReport(
            provider=provider_name, model=model, trust_domain=domain, endpoint=endpoint,
            status="FALLBACK",
            detail="Provider unavailable (no key or unreachable); using deterministic "
            "fallback. Set the API key to enable LLM execution.",
        )

    if probe:
        try:
            llm.complete("You are a connectivity probe.", "Reply with: ok", max_tokens=5)
            return ActivationReport(
                provider=provider_name, model=model, trust_domain=domain, endpoint=endpoint,
                status="CONNECTED", detail="Connectivity probe succeeded.",
            )
        except Exception as exc:  # noqa: BLE001 - surface, do not hide
            return ActivationReport(
                provider=provider_name, model=model, trust_domain=domain, endpoint=endpoint,
                status="FAILED", detail=f"Connectivity probe failed: {type(exc).__name__}.",
            )

    return ActivationReport(
        provider=provider_name, model=model, trust_domain=domain, endpoint=endpoint,
        status="CONNECTED", detail="Provider available (key present).",
    )


def render_activation_markdown(report: ActivationReport) -> str:
    return (
        "### LLM activation\n\n"
        "| Field | Value |\n| --- | --- |\n"
        f"| Provider | {report.provider} |\n"
        f"| Model | {report.model} |\n"
        f"| Trust domain | {report.trust_domain} |\n"
        f"| Endpoint | {report.endpoint} |\n"
        f"| Status | {report.status} |\n"
        + (f"\n{report.detail}\n" if report.detail else "")
    )
