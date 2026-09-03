"""Secure, persistent LLM API key handling.

Keys are managed securely with a strict precedence model:
  1. Explicit process environment (highest priority, e.g. for CI/containers)
  2. macOS Keychain (preferred persistent local-user store on macOS)
  3. Interactive hidden session prompt (fallback during interactive review)
  4. Missing (degrades safely to deterministic fallback)

Keys are never written to repository files, git, logs, reports, evidence records,
transcripts, or ledgers. On macOS, credentials persist in the system Keychain
under the canonical 'StART' service namespace.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

KEYCHAIN_SERVICE: str = "StART"


def load_private_env_if_present() -> None:
    """Load credentials from external private runtime if available."""
    env_override = os.environ.get("START_ENV_FILE")
    candidate_paths = [
        Path(env_override) if env_override else None,
        Path(__file__).resolve().parent.parent.parent.parent / "StART_Private_Runtime" / ".env",
    ]
    for p in candidate_paths:
        if p and p.is_file():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
            except Exception:
                pass
            break


PROVIDER_KEY_ENV: dict[str, str | None] = {
    "none": None,
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "grok": "XAI_API_KEY",
    "huggingface": "HF_TOKEN",
    "langsmith": "LANGSMITH_API_KEY",
    "hf_local": None,
    "enterprise_llm_gateway": None,
}

PROVIDER_DISPLAY_NAMES: dict[str, str] = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "gemini": "Gemini",
    "grok": "Grok",
    "huggingface": "Hugging Face",
    "langsmith": "LangSmith",
    "enterprise_llm_gateway": "Enterprise LLM Gateway",
    "none": "None",
}

# python packages required per provider (dependency check for llm-check)
PROVIDER_DEPENDENCY: dict[str, str | None] = {
    "none": None,
    "openai": "openai",
    "anthropic": "anthropic",
    "grok": "openai",  # Grok speaks the OpenAI wire protocol
    "gemini": "openai",  # Gemini OpenAI compatibility endpoint
    "deepseek": "openai",  # DeepSeek OpenAI compatibility endpoint
    "huggingface": "huggingface_hub",
    "langsmith": "langsmith",
    "hf_local": "transformers",
    "enterprise_llm_gateway": None,
}


@dataclass
class KeyStatus:
    provider: str
    env_var: str | None
    source: str  # "env" | "keychain" | "hidden prompt/session env" | "not required" | "missing"

    @property
    def ok(self) -> bool:
        return self.source != "missing"


def key_required(provider: str) -> bool:
    return PROVIDER_KEY_ENV.get(provider) is not None


# --------------------------------------------------------------------------- #
# macOS Keychain Persistent Store
# --------------------------------------------------------------------------- #
def keychain_is_supported() -> bool:
    """Return True if running on macOS with the security CLI available."""
    return sys.platform == "darwin" and os.path.exists("/usr/bin/security")


def keychain_get_key(provider: str, service: str = KEYCHAIN_SERVICE) -> str | None:
    """Retrieve an API key for the given provider from macOS Keychain."""
    if not keychain_is_supported():
        return None
    try:
        res = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                service,
                "-a",
                provider.lower(),
                "-w",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            secret = res.stdout.strip()
            return secret if secret else None
        return None
    except Exception:
        return None


def keychain_set_key(provider: str, secret: str, service: str = KEYCHAIN_SERVICE) -> bool:
    """Store or update an API key for the given provider in macOS Keychain."""
    if not keychain_is_supported() or not secret:
        return False
    try:
        res = subprocess.run(
            [
                "/usr/bin/security",
                "add-generic-password",
                "-U",
                "-s",
                service,
                "-a",
                provider.lower(),
                "-w",
                secret,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def keychain_delete_key(provider: str, service: str = KEYCHAIN_SERVICE) -> bool:
    """Delete an API key for the given provider from macOS Keychain."""
    if not keychain_is_supported():
        return False
    try:
        res = subprocess.run(
            [
                "/usr/bin/security",
                "delete-generic-password",
                "-s",
                service,
                "-a",
                provider.lower(),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


def keychain_has_key(provider: str, service: str = KEYCHAIN_SERVICE) -> bool:
    """Check if an API key exists in macOS Keychain for the provider without exposing it."""
    if not keychain_is_supported():
        return False
    try:
        res = subprocess.run(
            [
                "/usr/bin/security",
                "find-generic-password",
                "-s",
                service,
                "-a",
                provider.lower(),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        return res.returncode == 0
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Key Resolution
# --------------------------------------------------------------------------- #
def ensure_provider_key(
    provider: str,
    *,
    prompt_for_key: bool | None = None,
    interactive: bool | None = None,
) -> KeyStatus:
    """Make sure the provider's API key is available for this session.

    Resolution Precedence:
      1. Explicit process environment (e.g. os.environ['OPENAI_API_KEY'])
      2. macOS Keychain (persisted across sessions via start keys configure)
      3. Interactive hidden prompt (if prompt_for_key=True or interactive TTY)
      4. Missing

    When resolved from Keychain or prompt, the key is set via ``os.environ``
    for the current process only and is never written to project files or logs.
    """
    if provider not in PROVIDER_KEY_ENV:
        raise ValueError(f"Unknown LLM provider '{provider}'. Known: {sorted(PROVIDER_KEY_ENV)}")
    env_var = PROVIDER_KEY_ENV[provider]
    if env_var is None:
        return KeyStatus(provider=provider, env_var=None, source="not required")

    load_private_env_if_present()

    # 1. Explicit process environment (with official precedence rules)
    if provider == "gemini":
        if os.environ.get("GOOGLE_API_KEY"):
            return KeyStatus(provider="gemini", env_var="GOOGLE_API_KEY", source="env")
        if os.environ.get("GEMINI_API_KEY"):
            return KeyStatus(provider="gemini", env_var="GEMINI_API_KEY", source="env")
    elif provider == "grok":
        if os.environ.get("XAI_API_KEY"):
            return KeyStatus(provider="grok", env_var="XAI_API_KEY", source="env")
        if os.environ.get("GROK_API_KEY"):
            return KeyStatus(provider="grok", env_var="GROK_API_KEY", source="env")
    else:
        if os.environ.get(env_var):
            return KeyStatus(provider=provider, env_var=env_var, source="env")

    # 2. macOS Keychain
    keychain_secret = keychain_get_key(provider)
    if keychain_secret:
        if provider == "gemini":
            if not os.environ.get("GOOGLE_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = keychain_secret
            os.environ["GEMINI_API_KEY"] = keychain_secret
        elif provider == "grok":
            os.environ["XAI_API_KEY"] = keychain_secret
            os.environ["GROK_API_KEY"] = keychain_secret
        else:
            os.environ[env_var] = keychain_secret
        return KeyStatus(provider=provider, env_var=env_var, source="keychain")

    # 3. Interactive hidden prompt
    if interactive is None:
        interactive = sys.stdin.isatty()
    should_prompt = interactive if prompt_for_key is None else prompt_for_key
    if not should_prompt:
        return KeyStatus(provider=provider, env_var=env_var, source="missing")

    import getpass

    entered = getpass.getpass(f"Enter {env_var}: ").strip()
    if not entered:
        return KeyStatus(provider=provider, env_var=env_var, source="missing")

    if provider == "gemini":
        os.environ["GEMINI_API_KEY"] = entered
        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = entered
    elif provider == "grok":
        os.environ["XAI_API_KEY"] = entered
        os.environ["GROK_API_KEY"] = entered
    else:
        os.environ[env_var] = entered
    return KeyStatus(provider=provider, env_var=env_var, source="hidden prompt/session env")


def resolve_key_databricks(provider: str, dbutils: object = None, scope: str = "start") -> KeyStatus:
    """Databricks key resolution: secret scope -> environment -> missing."""
    if provider not in PROVIDER_KEY_ENV:
        raise ValueError(f"Unknown LLM provider '{provider}'.")
    env_var = PROVIDER_KEY_ENV.get(provider)
    if env_var is None:
        return KeyStatus(provider=provider, env_var=None, source="not required")

    vars_to_check = [env_var]
    if provider == "gemini":
        vars_to_check = ["GOOGLE_API_KEY", "GEMINI_API_KEY"]
    elif provider == "grok":
        vars_to_check = ["XAI_API_KEY", "GROK_API_KEY"]

    if dbutils is not None:
        for var_name in vars_to_check:
            try:
                secret = dbutils.secrets.get(scope=scope, key=var_name)  # type: ignore[attr-defined]
                if secret:
                    os.environ[var_name] = secret
                    return KeyStatus(provider=provider, env_var=var_name, source="secret scope")
            except Exception:
                pass

    for var_name in vars_to_check:
        if os.environ.get(var_name):
            return KeyStatus(provider=provider, env_var=var_name, source="env")

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
    """Evidence-grounded provider health check."""
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
    prompt = "Summarize the single evidence record in one sentence, citing its evidence ID.\n\n" + bundle
    text = llm.generate(prompt, system=SYSTEM_PROMPT, metadata={"output_token_budget": 128})  # type: ignore[attr-defined]
    critique = EvidenceCriticAgent().critique_section(text, [synthetic])
    return {
        "provider": provider_name,
        "mode": "llm-assisted",
        "synthetic_evidence_sent": "yes",
        "raw_dataset_sent": "no",
        "critique": "passed" if critique.ok else "failed",
    }
