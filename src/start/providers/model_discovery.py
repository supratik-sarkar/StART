"""Model discovery and capability classification for LLM providers."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from enum import StrEnum


class ModelCapability(StrEnum):
    """Execution and modalities capability classification for LLM models."""

    TEXT_REVIEW = "text_review"
    IMAGE = "image"
    AUDIO = "audio"
    TRANSCRIPTION = "transcription"
    REALTIME = "realtime"
    EMBEDDING = "embedding"
    SEARCH_SPECIALIZED = "search_specialized"
    OTHER = "other"


def classify_model_capability(model_id: str, provider: str = "openai") -> ModelCapability:
    """Classify a model into its functional capability tier."""
    mid = model_id.strip().lower()
    prov = provider.strip().lower()

    # 1. Specialized Non-Review Surfaces (Checked across all providers)
    if "realtime" in mid:
        return ModelCapability.REALTIME
    if any(k in mid for k in ("-audio", "audio-", "gpt-audio", "tts", "voice")):
        return ModelCapability.AUDIO
    if any(k in mid for k in ("transcribe", "whisper")):
        return ModelCapability.TRANSCRIPTION
    if any(k in mid for k in ("image", "dall-e", "dall_e", "sora")):
        return ModelCapability.IMAGE
    if any(k in mid for k in ("embedding", "similarity")):
        return ModelCapability.EMBEDDING
    if any(k in mid for k in ("search", "moderation", "edit")):
        return ModelCapability.SEARCH_SPECIALIZED
    if any(k in mid for k in ("babbage", "davinci", "curie", "ada")) and not any(
        c in mid for c in ("gpt-", "instruct")
    ):
        return ModelCapability.OTHER

    # 2. Provider-specific review compatibility
    if prov == "openai":
        # Text-in / Text-out review-compatible OpenAI models
        is_text_model = any(
            mid.startswith(prefix)
            for prefix in ("gpt-5", "gpt-4.5", "gpt-4.1", "gpt-4o", "gpt-4", "gpt-3.5", "o1", "o3", "o4", "chatgpt-")
        )
        if is_text_model:
            return ModelCapability.TEXT_REVIEW
        return ModelCapability.OTHER

    if prov == "anthropic":
        if "claude" in mid:
            return ModelCapability.TEXT_REVIEW
        return ModelCapability.OTHER

    if prov == "gemini":
        if "gemini" in mid:
            return ModelCapability.TEXT_REVIEW
        return ModelCapability.OTHER

    if prov == "deepseek":
        if "deepseek" in mid:
            return ModelCapability.TEXT_REVIEW
        return ModelCapability.OTHER

    if prov == "grok":
        if "grok" in mid:
            return ModelCapability.TEXT_REVIEW
        return ModelCapability.OTHER

    # Default fallback check for generic / test models
    if any(mid.startswith(p) for p in ("gpt-", "claude-", "gemini-", "deepseek-", "grok-", "fake-", "test-")):
        return ModelCapability.TEXT_REVIEW

    return ModelCapability.OTHER


def is_reviewer_compatible(model_id: str, provider: str = "openai") -> bool:
    """Return True if model is text-in/text-out review compatible for StART."""
    return classify_model_capability(model_id, provider) == ModelCapability.TEXT_REVIEW


def sort_reviewer_models(models: list[str], provider: str = "openai") -> list[str]:
    """Sort compatible review models with flagship / canonical aliases first."""
    prov = provider.lower()
    priority_order: list[str] = []

    if prov == "openai":
        priority_order = [
            "gpt-5-mini",
            "gpt-5",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4o-mini",
            "o3-mini",
            "o1",
            "o1-mini",
            "o1-preview",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
        ]
    elif prov == "anthropic":
        priority_order = [
            "claude-sonnet-4-6",
            "claude-3-7-sonnet-latest",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ]
    elif prov == "gemini":
        priority_order = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro",
        ]
    elif prov == "deepseek":
        priority_order = [
            "deepseek-chat",
            "deepseek-reasoner",
        ]
    elif prov == "grok":
        priority_order = [
            "grok-2-latest",
            "grok-3-mini",
            "grok-3",
        ]

    # Partition into priority items present and remaining items
    priority_present = [m for m in priority_order if m in models]
    remaining = [m for m in models if m not in priority_present]

    # Secondary sort on remaining items: non-dated snapshots before dated snapshots
    def _snapshot_key(m: str) -> tuple[int, str]:
        import re

        has_date = bool(re.search(r"\d{4}-\d{2}-\d{2}|\d{8}", m))
        return (1 if has_date else 0, m)

    remaining_sorted = sorted(remaining, key=_snapshot_key)
    return priority_present + remaining_sorted


class ModelDiscoveryClient(ABC):
    """Abstract base class for querying available models from LLM providers."""

    @abstractmethod
    def list_models(self, provider: str) -> list[str]:
        """Query the provider's API at runtime and return review-compatible model IDs."""
        ...


class RealProviderModelDiscovery(ModelDiscoveryClient):
    """Production client that queries real APIs and filters by capability."""

    def list_models(self, provider: str) -> list[str]:
        prov = provider.lower()
        if prov == "openai":
            return self._list_openai()
        elif prov == "anthropic":
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
            # Intersect live discovery with StART capability policy
            compatible = [mid for mid in raw_ids if is_reviewer_compatible(mid, "openai")]
            return sort_reviewer_models(compatible, "openai")
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
            compatible = [mid for mid in raw_ids if is_reviewer_compatible(mid, "anthropic")]
            return sort_reviewer_models(compatible, "anthropic")
        except Exception:
            return []


class FakeModelDiscovery(ModelDiscoveryClient):
    """Testing mock client returning deterministic model lists."""

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
                "gemini": ["fake-gemini-test-1"],
                "deepseek": ["fake-deepseek-test-1"],
                "grok": ["fake-grok-test-1"],
            }

    def list_models(self, provider: str) -> list[str]:
        raw = list(self.mock_data.get(provider, []))
        # Apply review capability filter
        return [m for m in raw if is_reviewer_compatible(m, provider)]
