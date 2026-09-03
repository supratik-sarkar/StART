"""Unit and regression tests for provider-neutral generation contracts,

OpenAI Responses API integration, 5-provider translation matrix, and
model capability filtering.
"""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock

from start.providers.base import GenerationRequest, LLMProvider
from start.providers.llm import (
    AnthropicProvider,
    DeepSeekProvider,
    GeminiProvider,
    GrokProvider,
    OpenAIProvider,
)
from start.providers.model_discovery import (
    ModelCapability,
    classify_model_capability,
    is_reviewer_compatible,
    sort_reviewer_models,
)


# =========================================================================== #
# 1. Base GenerationRequest and LLMProvider Abstraction
# =========================================================================== #
def test_generation_request_is_provider_neutral():
    """Verify GenerationRequest uses semantic output_token_budget and no provider keywords."""
    req = GenerationRequest(
        prompt="Analyze model risk",
        system="You are a reviewer",
        output_token_budget=2048,
        temperature=0.0,
    )
    assert req.prompt == "Analyze model risk"
    assert req.system == "You are a reviewer"
    assert req.output_token_budget == 2048
    assert req.temperature == 0.0

    # Ensure no provider-specific keyword attributes exist on GenerationRequest
    req_fields = {f.name for f in inspect.signature(GenerationRequest).parameters.values()}
    assert "output_token_budget" in req_fields
    assert "max_tokens" not in req_fields
    assert "max_output_tokens" not in req_fields
    assert "max_completion_tokens" not in req_fields


def test_base_llm_provider_interface_has_no_provider_specific_token_keywords():
    """Verify base LLMProvider.complete uses output_token_budget and not provider keywords."""
    sig = inspect.signature(LLMProvider.complete)
    params = list(sig.parameters.keys())
    assert "output_token_budget" in params
    assert "max_tokens" not in params
    assert "max_output_tokens" not in params
    assert "max_completion_tokens" not in params


# =========================================================================== #
# 2. Exact OpenAI gpt-5-mini Regression & Responses API Contract
# =========================================================================== #
def test_openai_gpt_5_mini_uses_responses_api_with_no_max_tokens(monkeypatch):
    """Exact regression test reproducing the live 400 error and proving fix.

    The old contract passed max_tokens to gpt-5-mini which OpenAI rejected with:
      BadRequestError: 400 Unsupported parameter: 'max_tokens' is not supported.
    The new contract uses Responses API with max_output_tokens and store=False.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.id = "resp_live_12345"
    mock_response.output_text = "Verified model risk evidence [EV-TEST-1]."
    mock_response.usage.input_tokens = 50
    mock_response.usage.output_tokens = 20

    # Setup responses.create mock
    mock_client.responses.create.return_value = mock_response

    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)

    provider = OpenAIProvider(model="gpt-5-mini")
    result = provider.complete(
        system="System prompt",
        user="User query about portfolio risk",
        output_token_budget=1024,
    )

    assert result == "Verified model risk evidence [EV-TEST-1]."
    assert provider.last_response_id == "resp_live_12345"
    assert provider.last_input_tokens == 50
    assert provider.last_output_tokens == 20

    # Verify call kwargs on Responses API
    mock_client.responses.create.assert_called_once()
    called_kwargs = mock_client.responses.create.call_args.kwargs
    assert called_kwargs["model"] == "gpt-5-mini"
    assert called_kwargs["instructions"] == "System prompt"
    assert called_kwargs["input"] == "User query about portfolio risk"
    assert called_kwargs["max_output_tokens"] == 1024
    assert called_kwargs["store"] is False

    # Assert HARD rule: NO max_tokens sent
    assert "max_tokens" not in called_kwargs
    assert "max_completion_tokens" not in called_kwargs


def test_openai_chat_fallback_route_uses_max_completion_tokens_for_modern_models(monkeypatch):
    """When falling back to Chat Completions, modern models use max_completion_tokens."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")

    mock_client = MagicMock()
    # Simulate an endpoint where responses API is absent or not supported
    del mock_client.responses

    mock_chat_response = MagicMock()
    mock_chat_response.id = "chatcmpl_98765"
    mock_chat_response.choices = [MagicMock()]
    mock_chat_response.choices[0].message.content = "Fallback chat completion response"
    mock_chat_response.usage.prompt_tokens = 40
    mock_chat_response.usage.completion_tokens = 15
    mock_client.chat.completions.create.return_value = mock_chat_response

    monkeypatch.setattr("openai.OpenAI", lambda **kwargs: mock_client)

    provider = OpenAIProvider(model="gpt-5-mini")
    result = provider.complete(
        system="System prompt",
        user="User query",
        output_token_budget=512,
    )

    assert result == "Fallback chat completion response"
    assert provider.last_response_id == "chatcmpl_98765"

    mock_client.chat.completions.create.assert_called_once()
    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5-mini"
    assert kwargs["max_completion_tokens"] == 512
    assert "max_tokens" not in kwargs


# =========================================================================== #
# 3. Provider Adapter Translation Matrix (Anthropic, Gemini, DeepSeek, Grok)
# =========================================================================== #
def test_anthropic_adapter_translates_budget_to_native_max_tokens(monkeypatch):
    """Anthropic Messages API uses native max_tokens."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    mock_anthropic = MagicMock()
    mock_client = MagicMock()
    mock_resp = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Anthropic review answer"
    mock_resp.content = [text_block]
    mock_resp.id = "msg_12345"
    mock_resp.usage.input_tokens = 30
    mock_resp.usage.output_tokens = 12

    mock_client.messages.create.return_value = mock_resp
    mock_anthropic.Anthropic.return_value = mock_client
    monkeypatch.setattr("anthropic.Anthropic", lambda **kw: mock_client)

    provider = AnthropicProvider(model="claude-sonnet-4-5")
    result = provider.complete(system="System instructions", user="User question", output_token_budget=1024)

    assert result == "Anthropic review answer"
    mock_client.messages.create.assert_called_once_with(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        temperature=0.0,
        system="System instructions",
        messages=[{"role": "user", "content": "User question"}],
    )


def test_gemini_adapter_translation_and_google_api_key_precedence(monkeypatch):
    """Gemini adapter respects GOOGLE_API_KEY > GEMINI_API_KEY precedence and translates token budget."""
    monkeypatch.setenv("GOOGLE_API_KEY", "google-priority-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-secondary-key")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "Gemini answer"
    mock_resp.id = "gemini_resp_123"
    mock_resp.usage.prompt_tokens = 25
    mock_resp.usage.completion_tokens = 10
    mock_client.chat.completions.create.return_value = mock_resp

    captured_init_kwargs = {}

    def fake_openai(**kwargs):
        captured_init_kwargs.update(kwargs)
        return mock_client

    monkeypatch.setattr("openai.OpenAI", fake_openai)

    provider = GeminiProvider(model="gemini-2.0-flash")
    result = provider.complete(system="Sys", user="User", output_token_budget=800)

    assert result == "Gemini answer"
    assert captured_init_kwargs["api_key"] == "google-priority-key"
    assert "googleapis.com" in captured_init_kwargs["base_url"]
    mock_client.chat.completions.create.assert_called_once_with(
        model="gemini-2.0-flash",
        max_tokens=800,
        temperature=0.0,
        messages=[{"role": "system", "content": "Sys"}, {"role": "user", "content": "User"}],
    )


def test_deepseek_adapter_translation(monkeypatch):
    """DeepSeek adapter translates token budget to max_tokens."""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "DeepSeek answer"
    mock_resp.id = "deepseek_resp_123"
    mock_client.chat.completions.create.return_value = mock_resp

    captured_init_kwargs = {}

    def fake_openai(**kwargs):
        captured_init_kwargs.update(kwargs)
        return mock_client

    monkeypatch.setattr("openai.OpenAI", fake_openai)

    provider = DeepSeekProvider(model="deepseek-chat")
    result = provider.complete(system="Sys", user="User", output_token_budget=600)

    assert result == "DeepSeek answer"
    assert captured_init_kwargs["api_key"] == "sk-deepseek-test"
    assert "api.deepseek.com" in captured_init_kwargs["base_url"]
    mock_client.chat.completions.create.assert_called_once_with(
        model="deepseek-chat",
        max_tokens=600,
        temperature=0.0,
        messages=[{"role": "system", "content": "Sys"}, {"role": "user", "content": "User"}],
    )


def test_grok_adapter_translation_and_xai_api_key_precedence(monkeypatch):
    """Grok adapter respects XAI_API_KEY > GROK_API_KEY precedence and translates token budget."""
    monkeypatch.setenv("XAI_API_KEY", "xai-official-key")
    monkeypatch.setenv("GROK_API_KEY", "grok-compat-key")

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "Grok answer"
    mock_resp.id = "grok_resp_123"
    mock_client.chat.completions.create.return_value = mock_resp

    captured_init_kwargs = {}

    def fake_openai(**kwargs):
        captured_init_kwargs.update(kwargs)
        return mock_client

    monkeypatch.setattr("openai.OpenAI", fake_openai)

    provider = GrokProvider(model="grok-2-latest")
    result = provider.complete(system="Sys", user="User", output_token_budget=700)

    assert result == "Grok answer"
    assert captured_init_kwargs["api_key"] == "xai-official-key"
    assert "api.x.ai" in captured_init_kwargs["base_url"]
    mock_client.chat.completions.create.assert_called_once_with(
        model="grok-2-latest",
        max_tokens=700,
        temperature=0.0,
        messages=[{"role": "system", "content": "Sys"}, {"role": "user", "content": "User"}],
    )


# =========================================================================== #
# 4. Model Capability Classification & Filtering
# =========================================================================== #
def test_model_capability_classification_filters_specialized_models():
    """Verify non-reviewer models (audio, image, realtime, transcribe, TTS) are classified and excluded."""
    # Review compatible
    assert classify_model_capability("gpt-5-mini", "openai") == ModelCapability.TEXT_REVIEW
    assert classify_model_capability("gpt-5", "openai") == ModelCapability.TEXT_REVIEW
    assert classify_model_capability("gpt-4.1", "openai") == ModelCapability.TEXT_REVIEW
    assert classify_model_capability("gpt-4o", "openai") == ModelCapability.TEXT_REVIEW
    assert classify_model_capability("gpt-4o-mini", "openai") == ModelCapability.TEXT_REVIEW
    assert classify_model_capability("o3-mini", "openai") == ModelCapability.TEXT_REVIEW
    assert classify_model_capability("o1", "openai") == ModelCapability.TEXT_REVIEW

    # Specialized Non-Review surfaces
    assert classify_model_capability("gpt-image-1", "openai") == ModelCapability.IMAGE
    assert classify_model_capability("dall-e-3", "openai") == ModelCapability.IMAGE
    assert classify_model_capability("gpt-audio", "openai") == ModelCapability.AUDIO
    assert classify_model_capability("gpt-4o-mini-tts", "openai") == ModelCapability.AUDIO
    assert classify_model_capability("gpt-realtime", "openai") == ModelCapability.REALTIME
    assert classify_model_capability("gpt-4o-realtime-preview", "openai") == ModelCapability.REALTIME
    assert classify_model_capability("gpt-transcribe", "openai") == ModelCapability.TRANSCRIPTION
    assert classify_model_capability("whisper-1", "openai") == ModelCapability.TRANSCRIPTION
    assert classify_model_capability("text-embedding-3-large", "openai") == ModelCapability.EMBEDDING


def test_is_reviewer_compatible_filters_discovery_list():
    """Given discovery containing 104 raw models, reviewer filter retains only text-review models."""
    raw_models = [
        "gpt-5-mini",
        "gpt-5",
        "gpt-4.1",
        "gpt-image-1",
        "gpt-audio",
        "gpt-realtime",
        "gpt-transcribe",
        "gpt-4o-mini-tts",
        "text-embedding-3-small",
        "babbage-002",
        "davinci-002",
        "gpt-4o",
        "o3-mini",
    ]
    filtered = [m for m in raw_models if is_reviewer_compatible(m, "openai")]
    assert filtered == ["gpt-5-mini", "gpt-5", "gpt-4.1", "gpt-4o", "o3-mini"]
    assert "gpt-image-1" not in filtered
    assert "gpt-audio" not in filtered
    assert "gpt-realtime" not in filtered
    assert "gpt-transcribe" not in filtered
    assert "gpt-4o-mini-tts" not in filtered
    assert "text-embedding-3-small" not in filtered


def test_sort_reviewer_models_prioritizes_flagship_and_aliases():
    """Curated sort puts flagship/canonical aliases first."""
    models = ["gpt-4o-2024-08-06", "gpt-4o", "gpt-5-mini", "gpt-5", "o3-mini"]
    sorted_models = sort_reviewer_models(models, "openai")
    assert sorted_models[0] == "gpt-5-mini"
    assert sorted_models[1] == "gpt-5"
    assert sorted_models[2] == "gpt-4o"
    assert sorted_models[3] == "o3-mini"
    assert sorted_models[4] == "gpt-4o-2024-08-06"
