"""Gateway discovery and configuration — standard library only.

Split out from :mod:`start.providers.gateway` on purpose. Runtime profile
detection has to work in an environment where *nothing* has been installed —
no pydantic, no numpy, no SDKs — because the very first thing StART must get
right in a locked-down environment is which egress regime it is under. If that
determination depended on the scientific stack being importable, a partially
installed environment could silently fall back to the permissive default.

So everything needed to answer "is a private gateway configured or registered
here?" lives in this module, which imports nothing outside the standard
library.
"""

from __future__ import annotations

import json
import os
from typing import Any

__all__ = [
    "ENTRY_POINT_GROUP",
    "ENV_BASE_URL",
    "ENV_API_KEY_ENV",
    "ENV_API_KEY_FALLBACK",
    "ENV_MODEL",
    "ENV_HEADERS",
    "ENV_TIMEOUT",
    "GatewayConfigurationError",
    "gateway_settings",
    "redacted_settings",
    "registered_gateway_names",
    "load_registered_gateway",
]

ENTRY_POINT_GROUP = "start.llm_gateways"

ENV_BASE_URL = "START_GATEWAY_BASE_URL"
ENV_API_KEY_ENV = "START_GATEWAY_API_KEY_ENV"
ENV_API_KEY_FALLBACK = "START_GATEWAY_API_KEY"
ENV_MODEL = "START_GATEWAY_MODEL"
ENV_HEADERS = "START_GATEWAY_EXTRA_HEADERS"
ENV_TIMEOUT = "START_GATEWAY_TIMEOUT"


class GatewayConfigurationError(RuntimeError):
    """The gateway is selected but its configuration is incomplete."""


def gateway_settings(env: dict[str, str] | None = None) -> dict[str, Any]:
    """Read gateway configuration from the environment.

    The credential is handled by *indirection*: StART is told which environment
    variable holds the token (``START_GATEWAY_API_KEY_ENV``), never the token,
    and never a fixed variable name. An organisation's credential naming
    convention is itself an internal detail, so it does not belong in this
    repository or in any config committed to it.

    The returned mapping is safe to log except for the ``_extra_headers`` key,
    which is prefixed with an underscore and stripped by
    :func:`redacted_settings`.
    """
    env = os.environ if env is None else env  # type: ignore[assignment]

    key_var = (env.get(ENV_API_KEY_ENV) or ENV_API_KEY_FALLBACK).strip()
    raw_headers = (env.get(ENV_HEADERS) or "").strip()
    headers: dict[str, str] = {}
    header_error = ""
    if raw_headers:
        try:
            parsed = json.loads(raw_headers)
            if not isinstance(parsed, dict):
                raise ValueError("must be a JSON object")
            headers = {str(k): str(v) for k, v in parsed.items()}
        except Exception as exc:
            header_error = f"{ENV_HEADERS} is not a JSON object: {exc}"

    try:
        timeout = float(env.get(ENV_TIMEOUT) or "60")
    except ValueError:
        timeout = 60.0

    return {
        "base_url": (env.get(ENV_BASE_URL) or "").strip(),
        "credential_env_var": key_var,
        "credential_present": bool((env.get(key_var) or "").strip()),
        "model": (env.get(ENV_MODEL) or "").strip(),
        "extra_header_keys": sorted(headers),  # keys only; values may be sensitive
        "_extra_headers": headers,
        "header_error": header_error,
        "timeout_seconds": timeout,
    }


def redacted_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Drop private keys so the result can be logged or put in evidence."""
    return {k: v for k, v in settings.items() if not k.startswith("_")}


def registered_gateway_names() -> list[str]:
    """Names of gateway implementations registered by installed packages.

    Entry-point *metadata* only — nothing is imported. A private package that
    fails to import, or that raises on construction, can therefore never
    prevent StART from starting or from reporting an accurate profile.
    """
    try:
        from importlib import metadata as importlib_metadata

        return sorted({ep.name for ep in importlib_metadata.entry_points().select(group=ENTRY_POINT_GROUP)})
    except Exception:  # pragma: no cover - defensive
        return []


def load_registered_gateway(name: str) -> Any:
    """Import and instantiate a registered gateway implementation.

    The target must be a class or zero-argument factory producing an object
    exposing ``available()`` and
    ``generate(prompt, *, system=None, metadata=None) -> str``.
    """
    from importlib import metadata as importlib_metadata

    for ep in importlib_metadata.entry_points().select(group=ENTRY_POINT_GROUP, name=name):
        target = ep.load()
        return target() if callable(target) else target

    raise GatewayConfigurationError(
        f"No gateway named {name!r} is registered under the {ENTRY_POINT_GROUP!r} entry-point "
        f"group. Installed gateways: {', '.join(registered_gateway_names()) or '(none)'}"
    )
