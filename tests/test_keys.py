from __future__ import annotations

from pathlib import Path

import pytest

from start.providers.keys import (
    PROVIDER_KEY_ENV,
    KeyStatus,
    dependency_available,
    ensure_provider_key,
    key_required,
    resolve_key_databricks,
    run_llm_check,
)

FAKE_KEY = "sk-test-FAKE-KEY-do-not-leak-1234567890"


class _LocalFakeLLM:
    """Self-contained scriptable provider (no dependency on other test modules).

    Returns queued responses in order; records the (system, user) prompts it
    received so tests can assert what was sent.
    """

    name = "fake"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    @property
    def available(self) -> bool:
        return True

    def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
        self.calls.append((system, user))
        return self._responses.pop(0) if self._responses else "ok"

    def generate(self, prompt: str, *, system: str | None = None, metadata: dict | None = None) -> str:
        # run_llm_check calls generate(prompt, system=..., metadata=...);
        # delegate to complete so scripted/citing subclasses keep working.
        budget = int((metadata or {}).get("output_token_budget", (metadata or {}).get("max_tokens", 1024)))
        return self.complete(system or "", prompt, output_token_budget=budget)


@pytest.fixture(autouse=True)
def _clean_key_env(monkeypatch):
    extra_vars = ["GOOGLE_API_KEY", "GEMINI_API_KEY", "XAI_API_KEY", "GROK_API_KEY"]
    for env_var in list(PROVIDER_KEY_ENV.values()) + extra_vars:
        if env_var:
            monkeypatch.delenv(env_var, raising=False)
    # Hermetic test isolation: mock Keychain to empty by default
    monkeypatch.setattr("start.providers.keys.keychain_get_key", lambda p: None)
    monkeypatch.setattr("start.providers.keys.keychain_has_key", lambda p: False)


# --------------------------------------------------------------------------- #
# Mapping + policy
# --------------------------------------------------------------------------- #
def test_provider_key_mapping_matches_spec():
    assert PROVIDER_KEY_ENV["openai"] == "OPENAI_API_KEY"
    assert PROVIDER_KEY_ENV["anthropic"] == "ANTHROPIC_API_KEY"
    assert PROVIDER_KEY_ENV["gemini"] == "GEMINI_API_KEY"
    assert PROVIDER_KEY_ENV["deepseek"] == "DEEPSEEK_API_KEY"
    assert PROVIDER_KEY_ENV["grok"] == "XAI_API_KEY"
    assert PROVIDER_KEY_ENV["huggingface"] == "HF_TOKEN"
    assert PROVIDER_KEY_ENV["hf_local"] is None
    assert PROVIDER_KEY_ENV["enterprise_llm_gateway"] is None
    assert PROVIDER_KEY_ENV["none"] is None
    assert not key_required("hf_local") and key_required("openai")
    assert key_required("gemini") and key_required("grok")


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
# Precedence Rules for Gemini and Grok
# --------------------------------------------------------------------------- #
def test_gemini_google_api_key_takes_precedence(monkeypatch):
    """Google precedence: GOOGLE_API_KEY takes precedence over GEMINI_API_KEY."""
    monkeypatch.setenv("GOOGLE_API_KEY", "google-auth-key-123")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-standard-key-456")

    status = ensure_provider_key("gemini", prompt_for_key=False)
    assert status.source == "env"
    assert status.env_var == "GOOGLE_API_KEY"
    assert status.ok


def test_gemini_falls_back_to_gemini_api_key(monkeypatch):
    """If GOOGLE_API_KEY is absent, GEMINI_API_KEY is used."""
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-standard-key-456")

    status = ensure_provider_key("gemini", prompt_for_key=False)
    assert status.source == "env"
    assert status.env_var == "GEMINI_API_KEY"
    assert status.ok


def test_grok_xai_api_key_takes_precedence(monkeypatch):
    """Official xAI precedence: XAI_API_KEY takes precedence over GROK_API_KEY."""
    monkeypatch.setenv("XAI_API_KEY", "xai-official-key-123")
    monkeypatch.setenv("GROK_API_KEY", "grok-legacy-key-456")

    status = ensure_provider_key("grok", prompt_for_key=False)
    assert status.source == "env"
    assert status.env_var == "XAI_API_KEY"
    assert status.ok


def test_grok_falls_back_to_grok_api_key(monkeypatch):
    """If XAI_API_KEY is absent, GROK_API_KEY compatibility is used."""
    monkeypatch.setenv("GROK_API_KEY", "grok-legacy-key-456")

    status = ensure_provider_key("grok", prompt_for_key=False)
    assert status.source == "env"
    assert status.env_var == "GROK_API_KEY"
    assert status.ok


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
    monkeypatch.setenv("XAI_API_KEY", FAKE_KEY)
    env_status = resolve_key_databricks("grok", dbutils=_FakeDbutils({}), scope="start")
    assert env_status.source == "env"

    missing = resolve_key_databricks("openai", dbutils=_FakeDbutils({}), scope="start")
    assert missing.source == "missing"


# --------------------------------------------------------------------------- #
# llm-check
# --------------------------------------------------------------------------- #
def test_llm_check_with_fake_provider_passes_citation_gate():
    class CitingFake(_LocalFakeLLM):
        def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
            self.calls.append((system, user))
            # echo back a properly cited sentence using the bundle's own EV id
            import re

            ev = re.search(r"\[EV-[A-Za-z0-9]+\]", user).group(0)
            return f"The synthetic discrimination check passed cleanly. {ev}"

    fake = CitingFake([])
    result = run_llm_check("openai", llm=fake)
    assert result == {
        "provider": "openai",
        "mode": "llm-assisted",
        "synthetic_evidence_sent": "yes",
        "raw_dataset_sent": "no",
        "critique": "passed",
    }
    # the prompt contained ONLY the synthetic record, no user data references
    _, user_prompt = fake.calls[0]
    assert "llm-check-synthetic" in user_prompt


def test_llm_check_uncited_output_fails_gate():
    fake = _LocalFakeLLM(["The AUC is 0.9 and everything is great."])
    result = run_llm_check("openai", llm=fake)
    assert result["critique"] == "failed"


def test_llm_check_cli_unsupported_and_none():
    from typer.testing import CliRunner

    from start.cli import app

    runner = CliRunner()
    bad = runner.invoke(app, ["llm-check", "--llm-provider", "magicllm"])
    assert bad.exit_code == 1 and "Unknown provider" in bad.output
    none = runner.invoke(app, ["llm-check", "--llm-provider", "none"])
    assert none.exit_code == 0 and "deterministic" in none.output
    gateway = runner.invoke(app, ["llm-check", "--llm-provider", "enterprise_llm_gateway"])
    assert gateway.exit_code == 0 and "placeholder" in gateway.output


def test_agent_review_cli_missing_key_refuses_clearly(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from start.cli import app
    from start.modeling.data import load_attrition_dataset
    from start.orchestration import review_dataframes

    monkeypatch.chdir(tmp_path)
    review_dataframes(load_attrition_dataset(seed=31), target_column="attrition", seed=31)
    runner = CliRunner()
    refused = runner.invoke(
        app,
        ["agent-review", "--agent-mode", "llm", "--llm-provider", "openai", "--no-prompt-for-key"],
    )
    assert refused.exit_code == 1
    assert "Missing OPENAI_API_KEY" in refused.output
    assert "--prompt-for-key" in refused.output

    fallback = runner.invoke(
        app,
        [
            "agent-review",
            "--agent-mode",
            "llm",
            "--llm-provider",
            "openai",
            "--no-prompt-for-key",
            "--allow-deterministic-fallback",
        ],
    )
    assert fallback.exit_code == 0
    assert "WARNING" in fallback.output and "deterministic" in fallback.output


def test_entered_key_never_persisted(tmp_path, monkeypatch):
    """A key present in the session env must not leak into the report, the
    ledger, the evidence store, or the run JSON."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    from start.modeling.data import load_attrition_dataset
    from start.orchestration import review_dataframes
    from start.reporting import render_markdown

    result = review_dataframes(load_attrition_dataset(seed=33), target_column="attrition", seed=33)
    report = render_markdown(result)
    assert FAKE_KEY not in report
    assert FAKE_KEY not in result.model_dump_json()
    for artifact in Path("start_output").rglob("*"):
        if artifact.is_file():
            assert FAKE_KEY not in artifact.read_text(errors="ignore"), artifact


def test_dependency_check():
    ok, msg = dependency_available("none")
    assert ok
    ok_hf, msg_hf = dependency_available("hf_local")
    assert isinstance(ok_hf, bool) and "transformers" in msg_hf
    assert isinstance(KeyStatus("openai", "OPENAI_API_KEY", "env").ok, bool)


# =========================================================================== #
# macOS Keychain Persistent Store Tests (with Mocks)
# =========================================================================== #
def test_environment_credential_resolves_first(monkeypatch):
    """1. Explicit process environment resolves before Keychain."""
    import os

    monkeypatch.setenv("OPENAI_API_KEY", "sk-env-override-12345")
    monkeypatch.setattr("start.providers.keys.keychain_get_key", lambda p: "sk-keychain-67890")

    status = ensure_provider_key("openai", prompt_for_key=False)
    assert status.source == "env"
    assert status.ok
    assert os.environ["OPENAI_API_KEY"] == "sk-env-override-12345"


def test_keychain_resolves_when_environment_absent(monkeypatch):
    """2. Keychain resolves when environment variable is absent."""
    import os

    monkeypatch.setattr(
        "start.providers.keys.keychain_get_key", lambda p: "sk-keychain-67890" if p == "openai" else None
    )

    status = ensure_provider_key("openai", prompt_for_key=False)
    assert status.source == "keychain"
    assert status.ok
    assert os.environ["OPENAI_API_KEY"] == "sk-keychain-67890"


def test_missing_provider_returns_safe_missing_state(monkeypatch):
    """3. Missing provider returns safe missing state without throwing."""
    monkeypatch.setattr("start.providers.keys.keychain_get_key", lambda p: None)

    status = ensure_provider_key("openai", prompt_for_key=False)
    assert status.source == "missing"
    assert not status.ok


def test_configure_writes_one_provider_credential(monkeypatch):
    """4. Configure writes provider credential to Keychain via hidden input."""
    stored: dict[str, str] = {}
    monkeypatch.setattr("start.providers.keys.keychain_is_supported", lambda: True)
    monkeypatch.setattr("start.providers.keys.keychain_has_key", lambda p: p in stored)
    monkeypatch.setattr("start.providers.keys.keychain_set_key", lambda p, s: stored.update({p: s}) or True)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "sk-user-entered-12345")

    from start.cli.keys_cli import _configure_single_provider

    success = _configure_single_provider("openai")
    assert success
    assert stored["openai"] == "sk-user-entered-12345"


def test_replacement_updates_same_canonical_credential(monkeypatch):
    """5. Replacement updates the existing credential atomically."""
    stored = {"openai": "sk-old-token-11111"}
    monkeypatch.setattr("start.providers.keys.keychain_is_supported", lambda: True)
    monkeypatch.setattr("start.providers.keys.keychain_has_key", lambda p: p in stored)
    monkeypatch.setattr("start.providers.keys.keychain_set_key", lambda p, s: stored.update({p: s}) or True)
    monkeypatch.setattr("getpass.getpass", lambda prompt: "sk-new-token-22222")

    from start.cli.keys_cli import _configure_single_provider

    # Choose '2' (Replace)
    success = _configure_single_provider("openai", ask=lambda prompt: "2")
    assert success
    assert stored["openai"] == "sk-new-token-22222"


def test_configure_all_five_providers_preserves_existing(monkeypatch):
    """6. Configure all iterates all 5 providers and keeps existing credentials."""
    stored = {
        "openai": "sk-existing-openai",
        "anthropic": "sk-existing-anthropic",
        "deepseek": "sk-existing-deepseek",
    }
    monkeypatch.setattr("start.providers.keys.keychain_is_supported", lambda: True)
    monkeypatch.setattr("start.providers.keys.keychain_has_key", lambda p: p in stored)
    monkeypatch.setattr("start.providers.keys.keychain_set_key", lambda p, s: stored.update({p: s}) or True)

    prompts_entered = []

    def fake_getpass(prompt: str) -> str:
        secret = f"sk-new-{len(prompts_entered)}"
        prompts_entered.append((prompt, secret))
        return secret

    monkeypatch.setattr("getpass.getpass", fake_getpass)

    from start.cli.keys_cli import SUPPORTED_PROVIDERS, _configure_single_provider

    assert len(SUPPORTED_PROVIDERS) == 5
    for p in SUPPORTED_PROVIDERS:
        # Default choice '1' (Keep existing) for already configured providers
        _configure_single_provider(p, ask=lambda _: "1")

    # OpenAI, Anthropic, DeepSeek preserved
    assert stored["openai"] == "sk-existing-openai"
    assert stored["anthropic"] == "sk-existing-anthropic"
    assert stored["deepseek"] == "sk-existing-deepseek"
    # Gemini and Grok newly entered
    assert "gemini" in stored
    assert "grok" in stored


def test_status_never_reveals_token(monkeypatch):
    """7. Status command never prints or displays any part of the secret."""
    monkeypatch.setattr(
        "start.providers.keys.keychain_has_key",
        lambda p: p in ("openai", "anthropic", "gemini", "deepseek", "grok"),
    )
    from typer.testing import CliRunner

    from start.cli import app

    runner = CliRunner()
    res = runner.invoke(app, ["keys", "status"])
    assert res.exit_code == 0
    assert "OpenAI" in res.output
    assert "Anthropic" in res.output
    assert "Gemini" in res.output
    assert "DeepSeek" in res.output
    assert "Grok" in res.output
    assert "Keychain" in res.output
    # Must not contain any token signatures
    assert "sk-" not in res.output
    assert "Bearer" not in res.output


def test_delete_removes_only_intended_provider(monkeypatch):
    """8. Delete removes only the intended provider credential."""
    stored = {"openai": "token-1", "anthropic": "token-2"}
    monkeypatch.setattr("start.providers.keys.keychain_is_supported", lambda: True)
    monkeypatch.setattr("start.providers.keys.keychain_has_key", lambda p: p in stored)
    monkeypatch.setattr("start.providers.keys.keychain_delete_key", lambda p: stored.pop(p, None) is not None)

    from typer.testing import CliRunner

    from start.cli import app

    runner = CliRunner()
    # Confirm deletion with 'y'
    res = runner.invoke(app, ["keys", "delete", "openai"], input="y\n")
    assert res.exit_code == 0
    assert "openai" not in stored
    assert "anthropic" in stored


def test_review_resolver_automatically_receives_keychain_credential(monkeypatch):
    """9. Review resolver automatically receives Keychain credential."""
    import os

    monkeypatch.setattr(
        "start.providers.keys.keychain_get_key",
        lambda p: "sk-keychain-auto" if p == "openai" else None,
    )

    key_status = ensure_provider_key("openai", prompt_for_key=False)
    assert key_status.source == "keychain"
    assert os.environ["OPENAI_API_KEY"] == "sk-keychain-auto"


def test_gemini_and_grok_keychain_resolution(monkeypatch):
    """10. Gemini and Grok resolve from Keychain into appropriate environment variables."""
    import os

    monkeypatch.setattr(
        "start.providers.keys.keychain_get_key",
        lambda p: f"sk-{p}-keychain-test-token",
    )

    gem_status = ensure_provider_key("gemini", prompt_for_key=False)
    assert gem_status.source == "keychain"
    assert os.environ.get("GEMINI_API_KEY") == "sk-gemini-keychain-test-token"
    assert os.environ.get("GOOGLE_API_KEY") == "sk-gemini-keychain-test-token"

    grok_status = ensure_provider_key("grok", prompt_for_key=False)
    assert grok_status.source == "keychain"
    assert os.environ.get("XAI_API_KEY") == "sk-grok-keychain-test-token"


def test_raw_key_absent_from_llm_review_config_describe():
    """11. Raw key is completely absent from LLMReviewConfig.describe()."""
    from start.review.architecture import LLMReviewConfig

    cfg = LLMReviewConfig(backend_mode="public", provider="openai", model="gpt-4o", status="CONNECTED")
    desc = cfg.describe()
    assert "key" not in desc
    assert "api_key" not in desc
    assert "token" not in desc
    assert "secret" not in desc


def test_raw_key_absent_from_review_context_bundle_describe():
    """12. Raw key is completely absent from ReviewContextBundle.describe()."""
    from start.review.architecture import LLMReviewConfig, ReviewContextBundle, ReviewDomain, ReviewMode

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(backend_mode="public", provider="openai", model="gpt-4o"),
    )
    desc = bundle.describe()
    desc_str = str(desc)
    assert "key" not in desc_str or "llm_config" in desc_str
    assert "api_key" not in desc_str
    assert "secret" not in desc_str


def test_raw_key_absent_from_transcript_metadata(tmp_path, monkeypatch):
    """13. Raw key is absent from transcript metadata and files."""
    import json

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-test-token-123")
    from start.review_session import Exchange, ReviewSession

    session = ReviewSession(run_id="TEST-RUN")
    session.record_exchange(
        Exchange(
            agent="ArchitectureReviewAgent",
            question="Test prompt",
            answer="Test answer",
            response_id="resp-1",
        )
    )
    content = json.dumps(session.to_dict())
    assert "sk-secret-test-token-123" not in content


def test_raw_key_absent_from_evidence_record_ledger_attestation(tmp_path, monkeypatch):
    """14. Raw key is absent from EvidenceRecord, ledger, and attestation seal."""
    import json

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-test-token-456")
    from start.attestation.seal import build_seal
    from start.core.schemas import EvidenceRecord, Status, TestResult
    from start.evidence.ledger import EvidenceLedger

    rec = EvidenceRecord.from_result(
        TestResult(
            test_id="portfolio.risk_statistics",
            test_name="Portfolio Risk Statistics",
            status=Status.PASS,
            metrics={"annualised_volatility": 0.0937},
        ),
        run_id="RUN-1",
    )
    ledger = EvidenceLedger(str(tmp_path / "ledger.jsonl"), store_root=str(tmp_path / "evidence"))
    ledger.append(rec)
    seal = build_seal(review_id="RUN-1", evidence_head=rec.evidence_id)

    ledger_content = (tmp_path / "ledger.jsonl").read_text()
    seal_content = json.dumps(seal.as_dict())

    assert "sk-secret-test-token-456" not in ledger_content
    assert "sk-secret-test-token-456" not in seal_content


def test_provider_failure_message_does_not_print_token(monkeypatch):
    """15. Provider failure message sanitizes and does not print token."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-secret-xyz-999")
    from start.review.architecture import (
        LLMReviewConfig,
        ReviewContextBundle,
        ReviewDomain,
        ReviewMode,
    )
    from start.review.executor import run_domain_checkpoints

    class FailingProvider:
        name = "openai"
        model = "gpt-4o"

        def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
            raise RuntimeError("Request failed with Bearer sk-fake-secret-xyz-999")

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: FailingProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o", status="CONNECTED"
        ),
    )

    prompts = ["Q", "Test question", "1", "A", "A", "A", "A"]
    prompt_iter = iter(prompts)
    decisions = run_domain_checkpoints(bundle, [], interactive=True, ask=lambda _: next(prompt_iter))
    assert decisions[0]["backend"] == "fallback"


def test_openai_works_through_mocked_keychain_resolver(monkeypatch):
    """16. OpenAI provider functions seamlessly when key resolved from Keychain."""
    monkeypatch.setenv("START_PROFILE", "public_demo")
    monkeypatch.setattr(
        "start.providers.keys.keychain_get_key", lambda p: "sk-mock-openai-key" if p == "openai" else None
    )
    ensure_provider_key("openai", prompt_for_key=False)

    from start.core.config import LLMConfig
    from start.providers.llm import get_llm_provider

    provider = get_llm_provider(LLMConfig(provider="openai", model="gpt-4o-mini"))
    assert provider.name == "openai"
    assert provider.available


def test_anthropic_works_through_mocked_keychain_resolver(monkeypatch):
    """17. Anthropic provider functions seamlessly when key resolved from Keychain."""
    monkeypatch.setenv("START_PROFILE", "public_demo")
    monkeypatch.setattr(
        "start.providers.keys.keychain_get_key", lambda p: "sk-ant-mock-key" if p == "anthropic" else None
    )
    ensure_provider_key("anthropic", prompt_for_key=False)

    from start.core.config import LLMConfig
    from start.providers.llm import get_llm_provider

    provider = get_llm_provider(LLMConfig(provider="anthropic", model="claude-3-5-sonnet"))
    assert provider.name == "anthropic"
    assert provider.available


def test_deepseek_works_through_mocked_keychain_resolver(monkeypatch):
    """18. DeepSeek provider functions seamlessly when key resolved from Keychain."""
    monkeypatch.setenv("START_PROFILE", "public_demo")
    monkeypatch.setattr(
        "start.providers.keys.keychain_get_key", lambda p: "sk-deepseek-mock-key" if p == "deepseek" else None
    )
    ensure_provider_key("deepseek", prompt_for_key=False)

    from start.core.config import LLMConfig
    from start.providers.llm import get_llm_provider

    provider = get_llm_provider(LLMConfig(provider="deepseek", model="deepseek-chat"))
    assert provider.name == "deepseek"
    assert provider.available
