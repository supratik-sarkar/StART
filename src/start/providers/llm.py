"""LLM provider layer.

Backend-agnostic chat completion. All SDK imports are lazy so the core
package installs and runs with zero LLM dependencies. ``NoLLMProvider`` is a
first-class mode: every agent has a deterministic fallback, so StART produces
complete, audit-grade output with no LLM at all.

``EnterpriseLLMGatewayProvider`` is a neutral placeholder for an internal LLM
gateway. It contains no proprietary code, endpoints, or names; map it to a
real internal module via START_LLM__PROVIDER and private configuration kept
outside this repository.
"""

from __future__ import annotations

import os

from start.core.config import LLMConfig
from start.providers.base import LLMProvider


class NoLLMProvider(LLMProvider):
    """Explicit no-LLM mode. Agents must use deterministic fallbacks."""

    name = "none"

    @property
    def available(self) -> bool:
        return False

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        raise RuntimeError("NoLLMProvider cannot complete; use deterministic fallbacks.")


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    @property
    def available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("OPENAI_API_KEY"))

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        from openai import OpenAI

        client = OpenAI()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-5", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    @property
    def available(self) -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class GrokProvider(OpenAIProvider):
    """xAI Grok via OpenAI-compatible API."""

    name = "grok"

    def __init__(self, model: str = "grok-3-mini", temperature: float = 0.0) -> None:
        super().__init__(model=model, temperature=temperature)

    @property
    def available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("GROK_API_KEY"))

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["GROK_API_KEY"], base_url="https://api.x.ai/v1")
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


class HuggingFaceProvider(LLMProvider):
    """Hosted HF Inference API."""

    name = "huggingface"

    def __init__(self, model: str = "meta-llama/Llama-3.1-8B-Instruct") -> None:
        self.model = model

    @property
    def available(self) -> bool:
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("HF_TOKEN"))

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=self.model)
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


class HFLocalProvider(LLMProvider):
    """Local transformers pipeline; runs on detected device (CUDA/MPS/CPU)."""

    name = "hf_local"

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        self.model = model
        self._pipe = None

    @property
    def available(self) -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        from transformers import pipeline

        if self._pipe is None:
            self._pipe = pipeline("text-generation", model=self.model, device_map="auto")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        out = self._pipe(messages, max_new_tokens=max_tokens)
        return out[0]["generated_text"][-1]["content"]


class EnterpriseLLMGatewayProvider(LLMProvider):
    """Neutral placeholder for an internal enterprise LLM gateway.

    This class intentionally contains no endpoints, credentials, or
    firm-specific logic. To integrate a real gateway, implement a private
    package exposing the same ``complete`` interface and select it via
    private configuration kept outside this repository.
    """

    name = "enterprise_llm_gateway"

    @property
    def available(self) -> bool:
        return False

    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str:
        raise NotImplementedError(
            "enterprise_llm_gateway is a placeholder. Provide a private "
            "implementation outside this public repository."
        )


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "none": NoLLMProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "grok": GrokProvider,
    "huggingface": HuggingFaceProvider,
    "hf_local": HFLocalProvider,
    "enterprise_llm_gateway": EnterpriseLLMGatewayProvider,
}


def get_llm_provider(config: LLMConfig) -> LLMProvider:
    cls = _PROVIDERS.get(config.provider, NoLLMProvider)
    try:
        if config.model and config.provider in {"openai", "anthropic", "grok", "huggingface", "hf_local"}:
            provider = cls(model=config.model)  # type: ignore[call-arg]
        else:
            provider = cls()
    except TypeError:
        provider = cls()  # type: ignore[call-arg]
    if not provider.available and not isinstance(provider, NoLLMProvider):
        # Safe degradation: never block a run because an LLM is unreachable.
        return NoLLMProvider()
    return provider
