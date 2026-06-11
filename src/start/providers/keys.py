"""Secure, session-only LLM API key handling.

Keys are never written to disk, YAML, logs, reports, evidence records, or the
ledger. The only mutation this module performs is setting an environment
variable for the *current process*, either from an existing environment value
or from a hidden ``getpass`` prompt. Default behavior across StART remains
no-LLM deterministic verification: nothing here runs unless the user opts
into LLM mode.

Provider → key mapping (public providers only):

    openai                 -> OPENAI_API_KEY
    anthropic              -> ANTHROPIC_API_KEY
    grok                   -> GROK_API_KEY
    huggingface            -> HF_TOKEN
    hf_local               -> no API key (local transformers dependencies)
    enterprise_llm_gateway -> no public key handling (private implementation)
    none                   -> no key (deterministic mode only)

On Databricks, prefer secret scopes over any visible widget:
``dbutils.secrets.get(scope, key)`` first, environment second, deterministic
fallback with an explicit warning last.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

PROVIDER_KEY_ENV: dict[str, str | None] = {
    "none": None,
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "grok": "GROK_API_KEY",
    "huggingface": "HF_TOKEN",
    "hf_local": None,
    "enterprise_llm_gateway": None,
}

# python packages required per provider (dependency check for llm-check)
PROVIDER_DEPENDENCY: dict[str, str | None] = {
    "none": None,
    "openai": "openai",
    "anthropic": "anthropic",
    "grok": "openai",  # Grok speaks the OpenAI wire protocol
    "huggingface": "huggingface_hub",
    "hf_local": "transformers",
    "enterprise_llm_gateway": None,
}


@dataclass
class KeyStatus:
    provider: str
    env_var: str | None
    source: str  # "env" | "hidden prompt/session env" | "not required" | "missing"

    @property
    def ok(self) -> bool:
        return self.source != "missing"


def key_required(provider: str) -> bool:
    return PROVIDER_KEY_ENV.get(provider) is not None


def ensure_provider_key(
    provider: str,
    *,
    prompt_for_key: bool | None = None,
    interactive: bool | None = None,
) -> KeyStatus:
    """Make sure the provider's API key is available for this session.

    prompt_for_key: True forces prompting when missing; False forbids it;
    None means auto — prompt only on an interactive terminal, never in CI.
    The entered key is set via ``os.environ`` for the current process only
    and is never echoed, logged, or persisted.
    """
    if provider not in PROVIDER_KEY_ENV:
        raise ValueError(
            f"Unknown LLM provider '{provider}'. Known: {sorted(PROVIDER_KEY_ENV)}"
        )
    env_var = PROVIDER_KEY_ENV[provider]
    if env_var is None:
        return KeyStatus(provider=provider, env_var=None, source="not required")
    if os.environ.get(env_var):
        return KeyStatus(provider=provider, env_var=env_var, source="env")

    if interactive is None:
        interactive = sys.stdin.isatty()
    should_prompt = interactive if prompt_for_key is None else prompt_for_key
    if not should_prompt:
        return KeyStatus(provider=provider, env_var=env_var, source="missing")

    import getpass

    entered = getpass.getpass(f"Enter {env_var}: ").strip()
    if not entered:
        return KeyStatus(provider=provider, env_var=env_var, source="missing")
    os.environ[env_var] = entered  # session-only; never written anywhere else
    return KeyStatus(provider=provider, env_var=env_var, source="hidden prompt/session env")


def resolve_key_databricks(
    provider: str, dbutils: object = None, scope: str = "start"
) -> KeyStatus:
    """Databricks key resolution: secret scope -> environment -> missing.

    Never uses visible widgets for secrets and never returns the key itself —
    only sets the session environment variable. Callers must not print keys
    to notebook output."""
    env_var = PROVIDER_KEY_ENV.get(provider)
    if provider not in PROVIDER_KEY_ENV:
        raise ValueError(f"Unknown LLM provider '{provider}'.")
    if env_var is None:
        return KeyStatus(provider=provider, env_var=None, source="not required")
    if dbutils is not None:
        try:
            secret = dbutils.secrets.get(scope=scope, key=env_var)  # type: ignore[attr-defined]
            if secret:
                os.environ[env_var] = secret
                return KeyStatus(provider=provider, env_var=env_var, source="secret scope")
        except Exception:
            pass  # scope/key absent; fall through to environment
    if os.environ.get(env_var):
        return KeyStatus(provider=provider, env_var=env_var, source="env")
    return KeyStatus(provider=provider, env_var=env_var, source="missing")


def dependency_available(provider: str) -> tuple[bool, str]:
    """Check the provider's python dependency without importing keys/data."""
    package = PROVIDER_DEPENDENCY.get(provider)
    if package is None:
        return True, "no external dependency"
    import importlib.util

    if importlib.util.find_spec(package) is not None:
        return True, f"'{package}' installed"
    return False, f"missing python package '{package}' (pip install {package})"


def run_llm_check(provider_name: str, llm: object | None = None) -> dict[str, str]:
    """Evidence-grounded provider health check.

    Sends ONE synthetic evidence record — never any user data — asks the
    provider for a single cited sentence, and verifies the output passes the
    evidence citation gate. ``llm`` is injectable for testing."""
    from start.agents import EvidenceCriticAgent
    from start.agents.prompts import SYSTEM_PROMPT, build_evidence_bundle
    from start.core.schemas import EvidenceRecord, Status, TestResult

    synthetic = EvidenceRecord.from_result(
        TestResult(
            test_id="supervised.discrimination",
            test_name="Synthetic discrimination check (llm-check)",
            status=Status.PASS,
            metrics={"auc_roc": 0.9},
            interpretation="Synthetic record used only to verify provider connectivity.",
        ),
        model_id="llm-check-model",
        dataset_id="llm-check-synthetic",
        run_id="RUN-llmcheck",
    )
    if llm is None:
        from start.core.config import LLMConfig
        from start.providers.llm import get_llm_provider

        llm = get_llm_provider(LLMConfig(provider=provider_name))  # type: ignore[arg-type]
    if not getattr(llm, "available", False):
        return {
            "provider": provider_name,
            "mode": "deterministic (provider unavailable)",
            "synthetic_evidence_sent": "no",
            "raw_dataset_sent": "no",
            "critique": "not applicable",
        }
    bundle = build_evidence_bundle([synthetic])
    prompt = (
        "Summarize the single evidence record in one sentence, citing its "
        "evidence ID.\n\n" + bundle
    )
    text = llm.generate(prompt, system=SYSTEM_PROMPT, metadata={"max_tokens": 128})  # type: ignore[attr-defined]
    critique = EvidenceCriticAgent().critique_section(text, [synthetic])
    return {
        "provider": provider_name,
        "mode": "llm-assisted",
        "synthetic_evidence_sent": "yes",
        "raw_dataset_sent": "no",
        "critique": "passed" if critique.ok else "failed",
    }
