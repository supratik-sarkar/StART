"""Gateway UX tests (v2.3.0): yes/no-first selection, key-skip, fallback,
env-var alias, trust-domain isolation, no secret leakage."""

from __future__ import annotations

from pathlib import Path

from start.interactive_review import ReviewConfig, prompt_review_config

FAKE_KEY = "sk-fake-do-not-log-1234567890ABCDEF"


def _drive(answers_map):
    """Build an ask() that answers by substring match on the prompt."""

    def ask(prompt=""):
        for needle, val in answers_map.items():
            if needle.lower() in prompt.lower():
                return val
        return ""  # default everything else

    return ask


def test_choosing_enterprise_gateway_sets_domain_and_skips_public_menu():
    seen = {"provider_menu": False}

    def ask(prompt=""):
        if "backend" in prompt.lower():
            return "2"  # Option 2: Enterprise LLM Gateway
        if "provider" in prompt.lower():
            seen["provider_menu"] = True
            return "openai"
        return ""

    cfg = prompt_review_config(ReviewConfig(agent_mode="llm", run_dl=False), ask=ask)
    assert cfg.llm_provider == "enterprise_llm_gateway"
    assert cfg.trust_domain == "enterprise"
    assert seen["provider_menu"] is False  # public menu was skipped


def test_declining_gateway_preserves_public_provider_menu():
    seen = {"provider_menu": False}

    def ask(prompt=""):
        if "backend" in prompt.lower():
            return "3"  # Option 3: Public LLM Providers
        if "provider" in prompt.lower():
            seen["provider_menu"] = True
            return "anthropic"
        return ""

    cfg = prompt_review_config(ReviewConfig(agent_mode="llm", run_dl=False), ask=ask)
    assert seen["provider_menu"] is True
    assert cfg.llm_provider == "anthropic"
    assert cfg.trust_domain == "public"


def test_gateway_default_is_no():
    # empty answer to the prompt must NOT select the gateway, should fallback to None/deterministic
    cfg = prompt_review_config(ReviewConfig(agent_mode="llm", run_dl=False), ask=lambda p: "")
    assert cfg.agent_mode == "deterministic"
    assert cfg.llm_provider == "none"


def test_enterprise_gateway_skips_public_key_prompt():
    from start.providers.keys import key_required

    # the gateway requires no public key
    assert key_required("enterprise_llm_gateway") is False
    assert key_required("openai") is True


def test_missing_enterprise_impl_falls_back_explicitly(monkeypatch, tmp_path):
    monkeypatch.delenv("START_ENTERPRISE_LLM_PACKAGE", raising=False)
    monkeypatch.delenv("START_ENTERPRISE_PACKAGE", raising=False)
    from start.enterprise import EnterpriseLLMGatewayAdapter

    ad = EnterpriseLLMGatewayAdapter()
    assert ad.available() is False
    try:
        ad.generate("evidence-grounded prompt")
        raise AssertionError("should have raised explicit fallback")
    except NotImplementedError as e:
        assert "unavailable" in str(e).lower()


def test_gateway_selected_resolves_to_deterministic_not_public(monkeypatch):
    monkeypatch.delenv("START_ENTERPRISE_LLM_PACKAGE", raising=False)
    monkeypatch.delenv("START_ENTERPRISE_PACKAGE", raising=False)
    from start.core.config import LLMConfig
    from start.providers.llm import NoLLMProvider, get_llm_provider

    llm = get_llm_provider(LLMConfig(provider="enterprise_llm_gateway"), expected_domain="private")
    # unavailable enterprise gateway degrades to the domain-neutral no-LLM path,
    # never to a public provider
    assert isinstance(llm, NoLLMProvider)


def test_env_var_alias_prefers_llm_package(monkeypatch):
    from start.enterprise.llm_gateway import _enterprise_package_name

    monkeypatch.delenv("START_ENTERPRISE_LLM_PACKAGE", raising=False)
    monkeypatch.delenv("START_ENTERPRISE_PACKAGE", raising=False)
    assert _enterprise_package_name() == "enterprise_package"  # default
    monkeypatch.setenv("START_ENTERPRISE_PACKAGE", "legacy_name")
    assert _enterprise_package_name() == "legacy_name"  # back-compat
    monkeypatch.setenv("START_ENTERPRISE_LLM_PACKAGE", "preferred_name")
    assert _enterprise_package_name() == "preferred_name"  # preferred wins


def test_fake_private_package_injection_flips_available(monkeypatch, tmp_path):
    # simulate a firm environment by putting a fake package on sys.path
    pkg = tmp_path / "fake_firm_gw"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("# fake firm gateway package\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setenv("START_ENTERPRISE_LLM_PACKAGE", "fake_firm_gw")
    from start.enterprise.llm_gateway import EnterpriseLLMGatewayAdapter

    assert EnterpriseLLMGatewayAdapter().available() is True


def test_trust_domain_crossover_blocked_both_ways():
    from start.providers.trust_domains import (
        TrustDomain,
        TrustDomainViolation,
        assert_no_crossover,
    )

    for prov, dom in [("openai", TrustDomain.PRIVATE), ("enterprise_llm_gateway", TrustDomain.PUBLIC)]:
        try:
            assert_no_crossover(prov, dom)
            raise AssertionError(f"crossover not blocked for {prov}")
        except TrustDomainViolation:
            pass


def test_no_public_key_leaks_when_gateway_selected(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("START_NB_ENTERPRISE_GATEWAY", "yes")  # notebook path env
    from start.modeling.data import load_attrition_dataset
    from start.orchestration import review_dataframes

    result = review_dataframes(load_attrition_dataset(seed=3), target_column="attrition", seed=3)
    assert result.run_id
    for art in Path(tmp_path).rglob("*"):
        if art.is_file() and art.suffix in {".md", ".json", ".jsonl", ".txt", ".html"}:
            assert FAKE_KEY not in art.read_text(errors="ignore"), art
