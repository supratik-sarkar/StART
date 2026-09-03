"""Runtime profile, egress enforcement, and gateway wiring.

The property under test is the one that makes a single public repository safe to
clone into a managed environment: the containment regime is enforced at the
routing boundary, not assumed from configuration hygiene.
"""

from __future__ import annotations

import pytest

from start.providers.gateway_discovery import (
    ENV_API_KEY_ENV,
    ENV_BASE_URL,
    gateway_settings,
    redacted_settings,
    registered_gateway_names,
)
from start.runtime_profile import (
    LOCAL_PROVIDERS,
    PUBLIC_SAAS_PROVIDERS,
    ProfileViolation,
    RuntimeProfile,
    active_profile,
    assert_provider_allowed,
    egress_policy,
    profile_banner,
    profile_manifest,
)


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def test_default_profile_is_public_demo() -> None:
    assert active_profile({}) is RuntimeProfile.PUBLIC_DEMO


def test_explicit_profile_wins() -> None:
    for value, expected in [
        ("enterprise", RuntimeProfile.ENTERPRISE),
        ("public_demo", RuntimeProfile.PUBLIC_DEMO),
        ("airgapped", RuntimeProfile.AIRGAPPED),
    ]:
        assert active_profile({"START_PROFILE": value}) is expected


def test_unknown_profile_is_an_error_not_a_default() -> None:
    """Silently defaulting a typo to the permissive profile is the wrong failure."""
    with pytest.raises(ProfileViolation, match="not a known runtime profile"):
        active_profile({"START_PROFILE": "prod"})


def test_configured_gateway_implies_enterprise() -> None:
    """Ambiguity resolves toward containment, not toward convenience."""
    env = {ENV_BASE_URL: "https://internal.example/v1"}
    assert active_profile(env) is RuntimeProfile.ENTERPRISE


# --------------------------------------------------------------------------- #
# Enforcement
# --------------------------------------------------------------------------- #
def test_enterprise_profile_refuses_every_public_saas_provider() -> None:
    env = {"START_PROFILE": "enterprise"}
    for provider in sorted(PUBLIC_SAAS_PROVIDERS):
        with pytest.raises(ProfileViolation):
            assert_provider_allowed(provider, env)


def test_enterprise_refusal_survives_installed_sdks_and_exported_keys() -> None:
    """The refusal is structural, not a check for missing credentials."""
    env = {
        "START_PROFILE": "enterprise",
        "OPENAI_API_KEY": "sk-not-a-real-key",
        "ANTHROPIC_API_KEY": "not-a-real-key",
        "DEEPSEEK_API_KEY": "not-a-real-key",
    }
    with pytest.raises(ProfileViolation, match="third-party inference endpoint"):
        assert_provider_allowed("openai", env)


def test_enterprise_error_message_tells_you_what_to_do_instead() -> None:
    env = {"START_PROFILE": "enterprise"}
    with pytest.raises(ProfileViolation) as excinfo:
        assert_provider_allowed("anthropic", env)
    message = str(excinfo.value)
    assert ENV_BASE_URL in message
    assert "--provider gateway" in message


def test_local_providers_are_permitted_under_every_profile() -> None:
    for profile in ("public_demo", "enterprise", "airgapped"):
        for provider in sorted(LOCAL_PROVIDERS):
            assert_provider_allowed(provider, {"START_PROFILE": profile})


def test_airgapped_permits_nothing_that_egresses() -> None:
    env = {"START_PROFILE": "airgapped"}
    for provider in sorted(PUBLIC_SAAS_PROVIDERS | {"gateway"}):
        with pytest.raises(ProfileViolation):
            assert_provider_allowed(provider, env)


def test_public_egress_override_is_recorded_not_silent() -> None:
    """The escape hatch exists, and using it is visible in every sealed review."""
    env = {"START_PROFILE": "enterprise", "START_ALLOW_PUBLIC_EGRESS": "true"}
    assert_provider_allowed("openai", env)  # permitted

    policy = egress_policy(env)
    assert policy.overrides == ("START_ALLOW_PUBLIC_EGRESS",)
    assert "OVERRIDE ACTIVE" in policy.rationale

    manifest = profile_manifest(env)
    assert manifest["overrides_in_effect"] == ["START_ALLOW_PUBLIC_EGRESS"]


def test_manifest_hash_changes_when_the_regime_changes() -> None:
    """The seal must distinguish a contained review from an uncontained one."""
    contained = profile_manifest({"START_PROFILE": "enterprise"})
    overridden = profile_manifest({"START_PROFILE": "enterprise", "START_ALLOW_PUBLIC_EGRESS": "1"})
    assert contained["manifest_hash"] != overridden["manifest_hash"]


def test_profile_banner_is_a_single_line() -> None:
    banner = profile_banner({"START_PROFILE": "enterprise"})
    assert "\n" not in banner
    assert "enterprise" in banner


# --------------------------------------------------------------------------- #
# Gateway configuration
# --------------------------------------------------------------------------- #
def test_credential_is_referenced_by_variable_name_never_by_value() -> None:
    """An organisation's credential naming convention stays out of the repo."""
    env = {
        ENV_BASE_URL: "https://internal.example/v1",
        ENV_API_KEY_ENV: "SOME_ORG_TOKEN",
        "SOME_ORG_TOKEN": "super-secret-value",
    }
    settings = gateway_settings(env)
    assert settings["credential_env_var"] == "SOME_ORG_TOKEN"
    assert settings["credential_present"] is True

    flat = repr(redacted_settings(settings))
    assert "super-secret-value" not in flat, "the credential value must never be surfaced"


def test_redacted_settings_drop_header_values() -> None:
    """Header values can carry tokens; only the key names are loggable."""
    env = {
        ENV_BASE_URL: "https://internal.example/v1",
        "START_GATEWAY_EXTRA_HEADERS": '{"X-Auth": "secret-header-value"}',
    }
    settings = gateway_settings(env)
    assert settings["extra_header_keys"] == ["X-Auth"]
    assert "secret-header-value" not in repr(redacted_settings(settings))


def test_malformed_headers_are_reported_not_swallowed() -> None:
    env = {ENV_BASE_URL: "https://internal.example/v1", "START_GATEWAY_EXTRA_HEADERS": "not json"}
    assert gateway_settings(env)["header_error"]


def test_gateway_discovery_never_raises() -> None:
    """A broken private package must not stop StART from starting."""
    assert isinstance(registered_gateway_names(), list)


# --------------------------------------------------------------------------- #
# The repository must not name any organisation
# --------------------------------------------------------------------------- #
def test_gateway_modules_contain_no_endpoints_or_organisation_names() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "start"
    for name in ("providers/gateway.py", "providers/gateway_discovery.py", "runtime_profile.py"):
        text = (root / name).read_text(encoding="utf-8")
        # Example URLs in docstrings use placeholders; a real host would not.
        for forbidden in ("api_key=", "Bearer sk-", "password="):
            assert forbidden not in text, f"{name} contains {forbidden!r}"
