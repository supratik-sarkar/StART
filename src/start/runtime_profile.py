"""Runtime profile and egress policy.

One repository, two very different environments:

* **public_demo** — a personal laptop or a public CI runner. Public SaaS LLM
  APIs (OpenAI, Anthropic, DeepSeek, Gemini, Grok, hosted HF) are reachable and
  are the point of the demo. Keys come from the user's own shell.
* **enterprise** — the same clone, dropped inside an organisation that mandates
  an internal gateway. Public SaaS providers must be *unreachable by
  construction*, not merely "not configured".
* **airgapped** — no outbound egress at all. Only the deterministic path and
  fully local models are permitted.

The problem this module solves is a specific one. A repository that "supports"
both environments usually does so by convention: someone remembers to change a
config value. Convention fails silently and expensively — a forgotten
`OPENAI_API_KEY` in a shell profile is all it takes for an internal artefact to
leave the building.

So the profile is not advice. It is a *precondition* checked at the routing
boundary. In the ``enterprise`` profile, ``OpenAIProvider`` cannot be resolved
at all: the attempt raises :class:`ProfileViolation` with an explanatory
message, regardless of which SDKs are installed or which keys are exported.
Installing the ``openai`` client library is explicitly NOT a statement about
which endpoints may be reached — that same library is the standard transport
for any OpenAI-compatible internal gateway.

Nothing here names, encodes, or assumes any particular organisation. The
enterprise profile is described entirely in terms of *capabilities*
(base URL supplied by the operator, credential read from an operator-named
environment variable, or an implementation registered by a separately
distributed private package).

Standard library only — this module must import successfully in an environment
where nothing at all has been pip-installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import StrEnum
from hashlib import sha256
from typing import Any

__all__ = [
    "RuntimeProfile",
    "ProfileViolation",
    "EgressPolicy",
    "active_profile",
    "egress_policy",
    "assert_provider_allowed",
    "assert_sink_allowed",
    "profile_manifest",
    "profile_banner",
    "PUBLIC_SAAS_PROVIDERS",
    "PRIVATE_GATEWAY_PROVIDERS",
    "LOCAL_PROVIDERS",
    "TELEMETRY_EGRESS_SINKS",
    "ENV_ALLOW_TELEMETRY_EGRESS",
]


# --------------------------------------------------------------------------- #
# Provider classes by egress character (not by vendor)
# --------------------------------------------------------------------------- #

#: Public SaaS inference providers whose hostnames belong to third parties.
PUBLIC_SAAS_PROVIDERS: frozenset[str] = frozenset(
    {"openai", "anthropic", "deepseek", "gemini", "grok", "huggingface"}
)

#: Providers that reach an operator-supplied endpoint whose address, auth and
#: ownership are configured at runtime and never appear in this repository.
PRIVATE_GATEWAY_PROVIDERS: frozenset[str] = frozenset(
    {"gateway", "enterprise_llm_gateway"}
)

#: Providers that perform no network egress whatsoever.
LOCAL_PROVIDERS: frozenset[str] = frozenset({"none", "hf_local", "replay"})

#: Telemetry / observability sinks whose endpoints belong to third parties.
TELEMETRY_EGRESS_SINKS: frozenset[str] = frozenset({"langsmith"})


class RuntimeProfile(StrEnum):
    """Where StART believes it is running."""

    PUBLIC_DEMO = "public_demo"
    ENTERPRISE = "enterprise"
    AIRGAPPED = "airgapped"


class ProfileViolation(RuntimeError):
    """Raised when a provider or sink is not permitted under the active profile.

    This is deliberately a hard error rather than a silent downgrade. A silent
    downgrade would let a reviewer believe an LLM narrated a review when in fact
    it did not — or, far worse in the other direction, let a request leave an
    environment that was supposed to contain it.
    """


ENV_PROFILE = "START_PROFILE"
ENV_GATEWAY_BASE_URL = "START_GATEWAY_BASE_URL"
ENV_ALLOW_PUBLIC_EGRESS = "START_ALLOW_PUBLIC_EGRESS"
ENV_ALLOW_TELEMETRY_EGRESS = "START_ALLOW_TELEMETRY_EGRESS"


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def _explicit_profile(env: dict[str, str]) -> RuntimeProfile | None:
    raw = (env.get(ENV_PROFILE) or "").strip().lower()
    if not raw:
        return None
    try:
        return RuntimeProfile(raw)
    except ValueError as exc:
        valid = ", ".join(p.value for p in RuntimeProfile)
        raise ProfileViolation(
            f"{ENV_PROFILE}={raw!r} is not a known runtime profile. Valid values: {valid}."
        ) from exc


def _private_gateway_registered() -> bool:
    """True if a private gateway implementation is installed in this env.

    Detection is by entry point only — no import of the private package, and no
    knowledge of what it is called. See ``start.providers.gateway``.
    """
    try:
        from start.providers.gateway_discovery import registered_gateway_names

        return bool(registered_gateway_names())
    except Exception:  # pragma: no cover - defensive: never break profile detection
        return False


def active_profile(env: dict[str, str] | None = None) -> RuntimeProfile:
    """Resolve the active profile.

    Resolution order, most explicit first:

    1. ``START_PROFILE`` — the operator said so.
    2. A private gateway implementation is registered via entry point, or
       ``START_GATEWAY_BASE_URL`` is set. Either signals a managed environment,
       so the safe reading is ``enterprise``.
    3. Otherwise ``public_demo``.

    The asymmetry is intentional. Guessing ``enterprise`` when the truth is
    ``public_demo`` costs a demo an error message. Guessing ``public_demo``
    when the truth is ``enterprise`` costs an organisation a disclosure
    incident. So ambiguity resolves toward containment.
    """
    env = os.environ if env is None else env  # type: ignore[assignment]

    explicit = _explicit_profile(env)  # type: ignore[arg-type]
    if explicit is not None:
        return explicit

    if env.get(ENV_GATEWAY_BASE_URL, "").strip():
        return RuntimeProfile.ENTERPRISE
    if _private_gateway_registered():
        return RuntimeProfile.ENTERPRISE

    return RuntimeProfile.PUBLIC_DEMO


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EgressPolicy:
    """The set of providers permitted under a profile, and why."""

    profile: RuntimeProfile
    allowed: frozenset[str]
    denied: frozenset[str]
    rationale: str
    overrides: tuple[str, ...] = field(default=())

    def permits(self, provider: str) -> bool:
        return provider in self.allowed

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.value,
            "allowed_providers": sorted(self.allowed),
            "denied_providers": sorted(self.denied),
            "rationale": self.rationale,
            "overrides_in_effect": list(self.overrides),
        }


def egress_policy(env: dict[str, str] | None = None) -> EgressPolicy:
    """Build the policy for the active profile."""
    env = os.environ if env is None else env  # type: ignore[assignment]
    profile = active_profile(env)  # type: ignore[arg-type]
    everything = PUBLIC_SAAS_PROVIDERS | PRIVATE_GATEWAY_PROVIDERS | LOCAL_PROVIDERS
    overrides: list[str] = []

    if profile is RuntimeProfile.PUBLIC_DEMO:
        allowed = PUBLIC_SAAS_PROVIDERS | LOCAL_PROVIDERS
        rationale = (
            "public_demo: third-party SaaS inference endpoints are permitted and are the "
            "purpose of the demo. Credentials are read from the operator's own shell. No "
            "internal gateway is assumed to exist."
        )

    elif profile is RuntimeProfile.ENTERPRISE:
        allowed = PRIVATE_GATEWAY_PROVIDERS | LOCAL_PROVIDERS
        rationale = (
            "enterprise: inference is confined to the operator-supplied gateway. Public "
            "SaaS providers are refused at the routing boundary regardless of installed "
            "SDKs or exported credentials."
        )
        # A narrow, deliberately loud escape hatch. It exists because a platform
        # team may need to A/B a public model during onboarding; it is recorded
        # in the manifest and therefore in the evidence chain, so nobody can use
        # it quietly.
        if env.get(ENV_ALLOW_PUBLIC_EGRESS, "").strip().lower() in {"1", "true", "yes"}:
            allowed = allowed | PUBLIC_SAAS_PROVIDERS
            overrides.append(ENV_ALLOW_PUBLIC_EGRESS)
            rationale += (
                f" OVERRIDE ACTIVE: {ENV_ALLOW_PUBLIC_EGRESS} re-admits public SaaS "
                "providers; this override is recorded in the profile manifest and is "
                "therefore visible in every sealed review produced while it is set."
            )

    else:  # AIRGAPPED
        allowed = LOCAL_PROVIDERS
        rationale = (
            "airgapped: no outbound inference of any kind. Only the deterministic path "
            "and fully local models are available."
        )

    return EgressPolicy(
        profile=profile,
        allowed=frozenset(allowed),
        denied=frozenset(everything - allowed),
        rationale=rationale,
        overrides=tuple(overrides),
    )


def assert_provider_allowed(provider: str, env: dict[str, str] | None = None) -> None:
    """Guard at the routing boundary. Raises :class:`ProfileViolation` if denied."""
    policy = egress_policy(env)
    if policy.permits(provider):
        return

    if provider in PUBLIC_SAAS_PROVIDERS and policy.profile is RuntimeProfile.ENTERPRISE:
        detail = (
            f"Provider '{provider}' reaches a third-party inference endpoint, which the "
            "'enterprise' runtime profile does not permit. Use the operator-supplied "
            "gateway instead:\n"
            f"    export {ENV_GATEWAY_BASE_URL}=<your gateway base URL>\n"
            "    export START_GATEWAY_API_KEY_ENV=<name of the env var holding the token>\n"
            "    start review --provider gateway\n"
            "If this environment genuinely is a public demo machine, set "
            f"{ENV_PROFILE}=public_demo."
        )
    elif provider in PRIVATE_GATEWAY_PROVIDERS and policy.profile is RuntimeProfile.PUBLIC_DEMO:
        detail = (
            f"Provider '{provider}' expects an operator-supplied gateway, but the active "
            "profile is 'public_demo' and no gateway is configured. Set "
            f"{ENV_GATEWAY_BASE_URL}, or install a private gateway package that registers "
            "a 'start.llm_gateways' entry point."
        )
    else:
        detail = (
            f"Provider '{provider}' is not permitted under the "
            f"'{policy.profile.value}' runtime profile."
        )

    raise ProfileViolation(f"{detail}\n\nActive policy: {policy.rationale}")


def assert_sink_allowed(sink: str, env: dict[str, str] | None = None) -> None:
    """Guard against uncontained telemetry / observability egress.

    Raises :class:`ProfileViolation` if the sink is not permitted under the active profile.
    """
    env = os.environ if env is None else env  # type: ignore[assignment]
    prof = active_profile(env)  # type: ignore[arg-type]

    if prof is RuntimeProfile.PUBLIC_DEMO:
        return

    # In enterprise or airgapped profiles, third-party SaaS telemetry egress is forbidden
    # unless START_ALLOW_TELEMETRY_EGRESS is explicitly enabled.
    allowed_override = env.get(ENV_ALLOW_TELEMETRY_EGRESS, "").strip().lower() in {"1", "true", "yes"}
    if allowed_override:
        return

    detail = (
        f"Telemetry sink '{sink}' reaches an external SaaS service, which the "
        f"'{prof.value}' runtime profile does not permit without explicit override.\n"
        f"To permit telemetry egress under '{prof.value}', export {ENV_ALLOW_TELEMETRY_EGRESS}=true."
    )
    raise ProfileViolation(detail)


# --------------------------------------------------------------------------- #
# Manifest — stamped into evidence so a reader can see the containment regime
# --------------------------------------------------------------------------- #
def profile_manifest(env: dict[str, str] | None = None) -> dict[str, Any]:
    """A hashable description of the containment regime a review ran under.

    This goes into the evidence chain. Six months later, a reader can tell not
    just what the review concluded but what the review was *allowed to touch*
    while concluding it — including whether the public-egress override was in
    effect.
    """
    env = os.environ if env is None else env  # type: ignore[assignment]
    policy = egress_policy(env)
    body = policy.as_dict()
    body["gateway_configured"] = bool(env.get(ENV_GATEWAY_BASE_URL, "").strip())
    try:
        from start.providers.gateway_discovery import registered_gateway_names

        body["registered_gateways"] = sorted(registered_gateway_names())
    except Exception:  # pragma: no cover
        body["registered_gateways"] = []

    telemetry_override = env.get(ENV_ALLOW_TELEMETRY_EGRESS, "").strip().lower() in {"1", "true", "yes"}
    body["telemetry_egress_permitted"] = (policy.profile is RuntimeProfile.PUBLIC_DEMO) or telemetry_override
    body["telemetry_overrides"] = [ENV_ALLOW_TELEMETRY_EGRESS] if telemetry_override else []

    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    body["manifest_hash"] = sha256(payload).hexdigest()
    return body


def profile_banner(env: dict[str, str] | None = None) -> str:
    """One-line human summary, used by ``start doctor`` and the demo script."""
    policy = egress_policy(env)
    override = " [PUBLIC EGRESS OVERRIDE ACTIVE]" if policy.overrides else ""
    return (
        f"runtime profile: {policy.profile.value}{override} | "
        f"permitted providers: {', '.join(sorted(policy.allowed))}"
    )
