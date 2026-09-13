"""Model discovery client for dynamically querying provider model availability (v3.1.1 #1)."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class ModelDiscoveryClient(ABC):
    """Abstract base class for querying available models from LLM providers."""

    @abstractmethod
    def list_models(self, provider: str) -> list[str]:
        """Query the provider's API at runtime and return the list of available model IDs."""
        ...


class RealProviderModelDiscovery(ModelDiscoveryClient):
    """Production client that queries real OpenAI/Anthropic APIs and filters by capability."""

    def list_models(self, provider: str) -> list[str]:
        if provider == "openai":
            return self._list_openai()
        elif provider == "anthropic":
            return self._list_anthropic()
        return []

    # ------------------------------------------------------------------
    def _list_openai(self) -> list[str]:
        if not os.environ.get("OPENAI_API_KEY"):
            return []
        try:
            from openai import OpenAI

            client = OpenAI()
            res = client.models.list()
            raw_ids = [m.id for m in res]
            # Curated capability filter: GPT / O-series chat models
            return sorted(
                mid for mid in raw_ids
                if mid.lower().startswith("gpt-") or mid.lower().startswith("o1-") or mid.lower().startswith("o3-")
            )
        except Exception:
            return []

    # ------------------------------------------------------------------
    def _list_anthropic(self) -> list[str]:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return []
        try:
            import anthropic

            client = anthropic.Anthropic()
            res = client.models.list()
            raw_ids = [m.id for m in res.data]
            # Curated capability filter: Claude models
            return sorted(
                mid for mid in raw_ids
                if "claude" in mid.lower()
            )
        except Exception:
            return []


class FakeModelDiscovery(ModelDiscoveryClient):
    """Testing mock client returning deterministic model lists.

    Injected by test fixtures; never used in production.
    """

    def __init__(
        self,
        mock_data: dict[str, list[str]] | None = None,
        models: list[str] | None = None,
    ) -> None:
        if models is not None:
            self.mock_data = {
                "openai": models,
                "anthropic": models,
                "gemini": models,
                "deepseek": models,
                "grok": models,
            }
        else:
            self.mock_data = mock_data or {
                "openai": ["fake-gpt-test-1", "fake-gpt-test-2"],
                "anthropic": ["fake-claude-test-1", "fake-claude-test-2"],
            }

    def list_models(self, provider: str) -> list[str]:
        return list(self.mock_data.get(provider, []))
