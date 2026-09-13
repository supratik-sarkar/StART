"""Operator-supplied LLM gateway providers.

Two ways to point StART at an internal inference endpoint, neither of which
requires editing a single file in this repository.

**1. Configuration only — for OpenAI-compatible gateways.**

Most internal gateways speak the OpenAI chat-completions wire format, because
that is what every client library already talks. If yours does, you need no
code at all::

    export START_GATEWAY_BASE_URL=https://<host>/<path>/v1
    export START_GATEWAY_API_KEY_ENV=MY_ORG_TOKEN     # name of the var, not the value
    export START_GATEWAY_MODEL=<model name>
    start review --provider gateway

Note the indirection on the credential: StART is told *which environment
variable holds the token*, never the token itself and never a hard-coded
variable name. That keeps organisation-specific identifiers — including the
naming convention of the credential — out of this repository and out of every
config file committed to it.

**2. Entry point — for gateways that are not OpenAI-compatible.**

Ship a private wheel, installed alongside StART, that declares::

    [project.entry-points."start.llm_gateways"]
    my_gateway = "my_private_pkg.start_gateway:Gateway"

The target must be a class or factory returning an object with::

    available() -> bool
    generate(prompt: str, *, system: str | None, metadata: dict | None) -> str

StART discovers it by entry point, so the public repository never imports,
vendors, names, or depends on the private package. ``start doctor`` reports it
as available; ``--provider my_gateway`` routes to it. Nothing else changes:
agents, evidence, attestation and sealing are all unaware of which
implementation answered.

This module deliberately contains no endpoints, credentials, hostnames,
organisation names, or authentication schemes.
"""

from __future__ import annotations

import os
import time
from typing import Any

from start.providers.base import LLMProvider
from start.providers.gateway_discovery import (
    ENTRY_POINT_GROUP,
    ENV_API_KEY_ENV,
    ENV_API_KEY_FALLBACK,
    ENV_BASE_URL,
    ENV_HEADERS,
    ENV_MODEL,
    ENV_TIMEOUT,
    GatewayConfigurationError,
    gateway_settings,
    load_registered_gateway,
    redacted_settings,
    registered_gateway_names,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "ENV_API_KEY_ENV",
    "ENV_API_KEY_FALLBACK",
    "ENV_BASE_URL",
    "ENV_COMPLETIONS_PATH",
    "ENV_HEADERS",
    "ENV_MODEL",
    "ENV_TIMEOUT",
    "GatewayConfigurationError",
    "OpenAICompatibleGatewayProvider",
    "PluginGatewayProvider",
    "gateway_settings",
    "registered_gateway_names",
    "load_registered_gateway",
    "gateway_diagnostics",
]

ENV_COMPLETIONS_PATH = "START_GATEWAY_COMPLETIONS_PATH"


# --------------------------------------------------------------------------- #
# OpenAI-compatible gateway
# --------------------------------------------------------------------------- #
class OpenAICompatibleGatewayProvider(LLMProvider):
    """Any inference endpoint that speaks the OpenAI chat-completions format.

    The ``openai`` client library is used purely as an HTTP transport pointed at
    ``START_GATEWAY_BASE_URL``. Installing that library says nothing about which
    endpoints may be reached — that is decided by
    :mod:`start.runtime_profile`.
    """

    name = "gateway"

    def __init__(self, model: str = "", temperature: float = 0.0) -> None:
        self._settings = gateway_settings()
        self.model = model or self._settings["model"]
        self.temperature = temperature

    # -- availability -------------------------------------------------------
    @property
    def available(self) -> bool:
        s = self._settings
        if not s["base_url"]:
            return False
        if s["header_error"]:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        # A credential is usually but not always required: some internal
        # gateways authenticate by mTLS or by network position alone. Treat a
        # reachable base URL as sufficient, and surface the credential state in
        # diagnostics rather than silently refusing.
        return True

    def explain_unavailability(self) -> str:
        s = self._settings
        if not s["base_url"]:
            return f"{ENV_BASE_URL} is not set."
        if s["header_error"]:
            return s["header_error"]
        try:
            import openai  # noqa: F401
        except ImportError:
            return "The 'openai' client library is not installed (pip install 'start-mrt[llm]')."
        return ""

    # -- inference ----------------------------------------------------------
    def _client(self) -> Any:
        from openai import OpenAI

        s = self._settings
        if not s["base_url"]:
            raise GatewayConfigurationError(
                f"{ENV_BASE_URL} is not set. StART will not guess a gateway address."
            )
        api_key = os.environ.get(s["credential_env_var"], "") or "not-required"
        kwargs: dict[str, Any] = {
            "api_key": api_key,
            "base_url": s["base_url"],
            "timeout": s["timeout_seconds"],
        }
        if s["_extra_headers"]:
            kwargs["default_headers"] = s["_extra_headers"]
        return OpenAI(**kwargs)

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        if not self.model:
            raise GatewayConfigurationError(
                f"No model selected. Set {ENV_MODEL} or pass --model. StART will not "
                "guess a model name on an operator-supplied gateway."
            )
        client = self._client()
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        self.last_latency_seconds = time.perf_counter() - t0
        self.last_response_id = getattr(resp, "id", "") or ""
        usage = getattr(resp, "usage", None)
        if usage:
            self.last_input_tokens = getattr(usage, "prompt_tokens", 0)
            self.last_output_tokens = getattr(usage, "completion_tokens", 0)
        return resp.choices[0].message.content or ""

    def generate(
        self, prompt: str, *, system: str | None = None, metadata: dict | None = None
    ) -> str:
        max_tokens = int((metadata or {}).get("max_tokens", 1024))
        return self.complete(system or "", prompt, max_tokens=max_tokens)


# --------------------------------------------------------------------------- #
# Entry-point gateways
# --------------------------------------------------------------------------- #
class PluginGatewayProvider(LLMProvider):
    """Adapts an entry-point-registered gateway to the provider interface."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._impl: Any | None = None
        self._load_error = ""

    def _get(self) -> Any:
        if self._impl is None and not self._load_error:
            try:
                self._impl = load_registered_gateway(self.name)
            except Exception as exc:
                self._load_error = str(exc)
        return self._impl

    @property
    def available(self) -> bool:
        impl = self._get()
        if impl is None:
            return False
        try:
            return bool(impl.available())
        except Exception:
            return False

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        impl = self._get()
        if impl is None:
            raise GatewayConfigurationError(
                f"Gateway '{self.name}' could not be loaded: {self._load_error}"
            )
        return impl.generate(user, system=system, metadata={"max_tokens": max_tokens})

    def generate(
        self, prompt: str, *, system: str | None = None, metadata: dict | None = None
    ) -> str:
        impl = self._get()
        if impl is None:
            raise GatewayConfigurationError(
                f"Gateway '{self.name}' could not be loaded: {self._load_error}"
            )
        return impl.generate(prompt, system=system, metadata=metadata)


# --------------------------------------------------------------------------- #
# Diagnostics — consumed by `start doctor`
# --------------------------------------------------------------------------- #
def gateway_diagnostics(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Everything a user needs to debug gateway wiring, with no secrets in it."""
    settings = gateway_settings(env)
    provider = OpenAICompatibleGatewayProvider()
    return {
        "openai_compatible": {
            "configured": bool(settings["base_url"]),
            "available": provider.available,
            "reason_unavailable": provider.explain_unavailability(),
            "settings": redacted_settings(settings),
        },
        "registered_plugins": registered_gateway_names(),
        "entry_point_group": ENTRY_POINT_GROUP,
    }
