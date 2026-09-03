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
import time
from typing import Any, cast

from start.core.config import LLMConfig
from start.providers.base import LLMProvider, ProviderResult, ProviderUsage

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "gemini": "Gemini",
    "deepseek": "DeepSeek",
    "grok": "Grok",
    "enterprise_llm_gateway": "Enterprise LLM Gateway",
    "none": "None",
}


def format_safe_provider_diagnostic(res: ProviderResult) -> str:
    """Format safe live-diagnostic metadata block for terminal display without exposing secrets."""
    prov_str = (
        PROVIDER_DISPLAY_NAMES.get(res.provider.lower(), res.provider.title())
        if res.provider
        else "Unknown"
    )
    lines = [
        "  [bold red]Provider Diagnostic[/bold red]",
        f"  Provider: {prov_str}",
        f"  Model: {res.model or 'Unknown'}",
        f"  API Surface: {res.api_surface}",
        f"  Response ID: {res.response_id or '—'}",
        f"  Status: {res.status}",
        f"  Incomplete Reason: {res.incomplete_reason or '—'}",
        f"  Provider Error Code/Type: {res.error_type or '—'}",
        f"  Max Output Tokens Configured: {res.max_output_tokens}",
        f"  Input Tokens: {res.usage.input_tokens}",
        f"  Output Tokens: {res.usage.output_tokens}",
        f"  Reasoning Tokens: {res.usage.reasoning_tokens}",
        f"  Output Item Types: {res.output_item_types if res.output_item_types else '[]'}",
        f"  Content Part Types: {res.content_part_types if res.content_part_types else '[]'}",
        f"  Refusal Present: {'Yes' if res.refusal else 'No'}",
        f"  Aggregated Output-Text Length: {len(res.text)}",
    ]
    return "\n".join(lines)


class NoLLMProvider(LLMProvider):
    """Explicit no-LLM mode. Agents must use deterministic fallbacks."""

    name = "none"

    @property
    def available(self) -> bool:
        return False

    def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
        raise RuntimeError("NoLLMProvider cannot complete; use deterministic fallbacks.")


def extract_response_text(resp: Any) -> str:
    """Universal text extraction across OpenAI Responses API, Chat, Anthropic, Gemini, Grok.

    Primary Responses API contract: response.output_text.
    Explicitly ignores reasoning summaries, reasoning tokens, and tool calls.
    """
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp.strip()

    # 1. Primary route: direct output_text property (OpenAI Responses API aggregated text)
    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    text_prop = getattr(resp, "text", None)
    if isinstance(text_prop, str) and text_prop.strip():
        return text_prop.strip()

    # 2. OpenAI Responses API output items (message items only; ignore reasoning / tools)
    if hasattr(resp, "output") and resp.output:
        chunks: list[str] = []
        for item in resp.output:
            item_type = getattr(item, "type", None)
            if item_type in ("reasoning", "thought", "summary", "tool_call", "function_call"):
                continue

            content = getattr(item, "content", None)
            if isinstance(content, str) and content.strip():
                chunks.append(content.strip())
            elif isinstance(content, list):
                for part in content:
                    part_type = getattr(part, "type", None)
                    if part_type in ("reasoning", "thought", "summary"):
                        continue
                    pt = getattr(part, "text", None)
                    pot = getattr(part, "output_text", None)
                    pv = getattr(part, "value", None)
                    if isinstance(pt, str) and pt:
                        chunks.append(pt)
                    elif isinstance(pot, str) and pot:
                        chunks.append(pot)
                    elif isinstance(pv, str) and pv:
                        chunks.append(pv)
            elif item_type == "message":
                it = getattr(item, "text", None)
                if isinstance(it, str) and it:
                    chunks.append(it)
        if chunks:
            combined = "".join(chunks).strip()
            if combined:
                return combined

    # 3. Chat completion choices
    if hasattr(resp, "choices") and resp.choices:
        first = resp.choices[0]
        msg = getattr(first, "message", None)
        if msg is not None:
            content = getattr(msg, "content", None)
            if isinstance(content, str) and content:
                return content.strip()
            if isinstance(content, list):
                chunks = []
                for c in content:
                    if isinstance(c, str) and c:
                        chunks.append(c)
                    elif hasattr(c, "text") and isinstance(c.text, str) and c.text:
                        chunks.append(c.text)
                if chunks:
                    return "".join(chunks).strip()
        delta = getattr(first, "delta", None)
        if delta is not None:
            content = getattr(delta, "content", None)
            if isinstance(content, str) and content:
                return content.strip()

    # 4. Anthropic content blocks
    if hasattr(resp, "content") and isinstance(resp.content, list):
        chunks = []
        for block in resp.content:
            if hasattr(block, "text") and isinstance(block.text, str) and block.text:
                chunks.append(block.text)
        if chunks:
            return "".join(chunks).strip()

    # 5. Dict fallback
    if isinstance(resp, dict):
        if "output_text" in resp and resp["output_text"]:
            return str(resp["output_text"]).strip()
        if "text" in resp and resp["text"]:
            return str(resp["text"]).strip()
        if "content" in resp and isinstance(resp["content"], str):
            return resp["content"].strip()
        if "choices" in resp and resp["choices"]:
            choice = resp["choices"][0]
            if isinstance(choice, dict):
                msg = choice.get("message", {})
                if isinstance(msg, dict) and "content" in msg and msg["content"]:
                    return str(msg["content"]).strip()

    return ""


class OpenAIProvider(LLMProvider):
    """OpenAI provider using Responses API for zero-retention audit-safe review."""

    name = "openai"

    def __init__(
        self,
        model: str = "gpt-5-mini",
        temperature: float = 0.0,
        reasoning_effort: str = "low",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    @property
    def available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("OPENAI_API_KEY"))

    def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
        res = self.complete_result(system, user, output_token_budget=output_token_budget)
        if res.status == "error":
            raise RuntimeError(f"OpenAI error: {res.error_message or res.error_type}")
        return res.text

    def complete_result(
        self, system: str, user: str, *, output_token_budget: int = 1024
    ) -> ProviderResult:
        from openai import OpenAI

        client = OpenAI()
        t0 = time.perf_counter()
        model_lower = self.model.lower()
        is_reasoning = any(k in model_lower for k in ("gpt-5", "o1", "o3", "o4"))

        req_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": user,
            "max_output_tokens": output_token_budget,
            "store": False,
        }
        if system:
            req_kwargs["instructions"] = system
        if is_reasoning:
            req_kwargs["reasoning"] = {"effort": self.reasoning_effort}
        elif self.temperature > 0.0:
            req_kwargs["temperature"] = self.temperature

        # Primary route: modern Responses API with explicit store=False
        try:
            resp = client.responses.create(**req_kwargs)
            latency = time.perf_counter() - t0
            self.last_latency_seconds = latency
            return self._normalize_responses_api_result(
                resp, latency=latency, configured_max_tokens=output_token_budget
            )
        except Exception as exc:
            latency = time.perf_counter() - t0
            self.last_latency_seconds = latency
            exc_str = str(exc).lower()
            # Only fallback to chat completions if responses endpoint or method is absent
            if (
                isinstance(exc, (AttributeError, TypeError))
                or "has no attribute 'responses'" in exc_str
                or "responses" in str(type(exc)).lower()
            ):
                return self._complete_chat_result(
                    client, system, user, output_token_budget=output_token_budget, t0=t0
                )
            # Typed request error (do NOT silently swallow or fall back)
            return ProviderResult(
                text="",
                provider=self.name,
                model=self.model,
                status="error",
                error_type=type(exc).__name__,
                error_message=str(exc),
                latency_seconds=latency,
                api_surface="responses",
                max_output_tokens=output_token_budget,
            )

    def _normalize_responses_api_result(
        self, resp: Any, latency: float, configured_max_tokens: int
    ) -> ProviderResult:
        resp_id = getattr(resp, "id", "")
        if not isinstance(resp_id, str):
            resp_id = ""
        self.last_response_id = resp_id

        usage_obj = getattr(resp, "usage", None)
        in_tokens = (
            usage_obj.input_tokens
            if usage_obj and isinstance(getattr(usage_obj, "input_tokens", None), int)
            else 0
        )
        out_tokens = (
            usage_obj.output_tokens
            if usage_obj and isinstance(getattr(usage_obj, "output_tokens", None), int)
            else 0
        )
        details = getattr(usage_obj, "output_tokens_details", None) if usage_obj else None
        reasoning_tokens = (
            details.reasoning_tokens
            if details and isinstance(getattr(details, "reasoning_tokens", None), int)
            else 0
        )

        self.last_input_tokens = in_tokens
        self.last_output_tokens = out_tokens
        self.last_reasoning_tokens = reasoning_tokens

        usage = ProviderUsage(
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            reasoning_tokens=reasoning_tokens,
        )

        raw_output_items = getattr(resp, "output", None)
        output_items = raw_output_items if isinstance(raw_output_items, (list, tuple)) else []
        output_item_types: list[str] = []
        content_part_types: list[str] = []
        refusal_val: str | None = None

        raw_refusal = getattr(resp, "refusal", None)
        if isinstance(raw_refusal, str) and raw_refusal.strip():
            refusal_val = raw_refusal.strip()

        for item in output_items:
            itype = getattr(item, "type", None)
            if isinstance(itype, str):
                output_item_types.append(itype)
            item_refusal = getattr(item, "refusal", None)
            if isinstance(item_refusal, str) and item_refusal.strip():
                refusal_val = item_refusal.strip()
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    ptype = getattr(part, "type", None)
                    if isinstance(ptype, str):
                        content_part_types.append(ptype)
                    part_refusal = getattr(part, "refusal", None)
                    if isinstance(part_refusal, str) and part_refusal.strip():
                        refusal_val = part_refusal.strip()

        status_val = getattr(resp, "status", None)
        raw_status = status_val.lower() if isinstance(status_val, str) else "completed"

        inc_details = getattr(resp, "incomplete_details", None)
        inc_reason: str | None = None
        if inc_details is not None:
            raw_inc_reason = getattr(inc_details, "reason", None)
            if raw_inc_reason is None and isinstance(inc_details, dict):
                raw_inc_reason = inc_details.get("reason")
            if isinstance(raw_inc_reason, str):
                inc_reason = raw_inc_reason

        err_obj = getattr(resp, "error", None)
        err_type: str | None = None
        err_msg: str | None = None
        has_real_error = False
        if err_obj is not None:
            raw_err_msg = getattr(err_obj, "message", None)
            raw_err_type = getattr(err_obj, "code", None) or getattr(err_obj, "type", None)
            if isinstance(raw_err_msg, str) and raw_err_msg:
                err_msg = raw_err_msg
                has_real_error = True
            if isinstance(raw_err_type, str) and raw_err_type:
                err_type = raw_err_type
                has_real_error = True

        extracted_text = extract_response_text(resp)

        # Status Classification (Section 4 Contract)
        if raw_status in ("failed", "error") or has_real_error:
            final_status = "error"
            err_type = err_type or "PROVIDER_REQUEST_ERROR"
            err_msg = err_msg or "OpenAI Responses API returned failure status"
        elif refusal_val:
            final_status = "refusal"
        elif raw_status == "incomplete":
            final_status = "incomplete"
            inc_reason = inc_reason or "max_output_tokens"
        elif raw_status == "completed":
            if extracted_text:
                final_status = "completed"
            else:
                final_status = "empty"
        else:
            final_status = raw_status

        return ProviderResult(
            text=extracted_text,
            provider=self.name,
            model=self.model,
            response_id=resp_id,
            status=final_status,
            incomplete_reason=inc_reason,
            refusal=refusal_val,
            error_type=err_type,
            error_message=err_msg,
            usage=usage,
            latency_seconds=latency,
            api_surface="responses",
            max_output_tokens=configured_max_tokens,
            output_item_types=output_item_types,
            content_part_types=content_part_types,
        )

    def _complete_chat_result(
        self, client: Any, system: str, user: str, *, output_token_budget: int = 1024, t0: float
    ) -> ProviderResult:
        model_lower = self.model.lower()
        is_reasoning_or_modern = any(k in model_lower for k in ("o1", "o3", "o4", "gpt-5", "gpt-4.1"))
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if is_reasoning_or_modern:
            kwargs["max_completion_tokens"] = output_token_budget
        else:
            kwargs["max_tokens"] = output_token_budget
        if not is_reasoning_or_modern and self.temperature > 0.0:
            kwargs["temperature"] = self.temperature

        try:
            resp = client.chat.completions.create(**kwargs)
            latency = time.perf_counter() - t0
            self.last_latency_seconds = latency
            resp_id = getattr(resp, "id", "") or ""
            self.last_response_id = resp_id
            usage_obj = getattr(resp, "usage", None)
            in_tokens = getattr(usage_obj, "prompt_tokens", 0) if usage_obj else 0
            out_tokens = getattr(usage_obj, "completion_tokens", 0) if usage_obj else 0
            details = getattr(usage_obj, "completion_tokens_details", None) if usage_obj else None
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) if details else 0
            self.last_input_tokens = in_tokens
            self.last_output_tokens = out_tokens
            self.last_reasoning_tokens = reasoning_tokens

            text = extract_response_text(resp)
            first_choice = resp.choices[0] if getattr(resp, "choices", None) else None
            finish_reason = getattr(first_choice, "finish_reason", None) if first_choice else None

            if finish_reason == "length":
                status = "incomplete"
                inc_reason = "max_tokens"
            elif not text.strip():
                status = "empty"
                inc_reason = None
            else:
                status = "completed"
                inc_reason = None

            return ProviderResult(
                text=text,
                provider=self.name,
                model=self.model,
                response_id=resp_id,
                status=status,
                incomplete_reason=inc_reason,
                usage=ProviderUsage(
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                    reasoning_tokens=reasoning_tokens,
                ),
                latency_seconds=latency,
                api_surface="chat_completions",
                max_output_tokens=output_token_budget,
            )
        except Exception as exc:
            latency = time.perf_counter() - t0
            return ProviderResult(
                text="",
                provider=self.name,
                model=self.model,
                status="error",
                error_type=type(exc).__name__,
                error_message=str(exc),
                latency_seconds=latency,
                api_surface="chat_completions",
                max_output_tokens=output_token_budget,
            )


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

    def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
        import anthropic

        client = anthropic.Anthropic()
        t0 = time.perf_counter()
        resp = cast(Any, client.messages).create(
            model=self.model,
            max_tokens=output_token_budget,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        self.last_latency_seconds = time.perf_counter() - t0
        self.last_response_id = getattr(resp, "id", "") or ""
        usage = getattr(resp, "usage", None)
        if usage:
            self.last_input_tokens = getattr(usage, "input_tokens", 0)
            self.last_output_tokens = getattr(usage, "output_tokens", 0)
        return extract_response_text(resp)


class GrokProvider(LLMProvider):
    """xAI Grok via provider-native OpenAI-compatible endpoint."""

    name = "grok"

    def __init__(self, model: str = "grok-2-latest", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    @property
    def available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY"))

    def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
        from openai import OpenAI

        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY", "")
        client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=output_token_budget,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        self.last_latency_seconds = time.perf_counter() - t0
        self.last_response_id = getattr(resp, "id", "") or ""
        usage = getattr(resp, "usage", None)
        if usage:
            self.last_input_tokens = getattr(usage, "prompt_tokens", 0)
            self.last_output_tokens = getattr(usage, "completion_tokens", 0)
        return extract_response_text(resp)


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

    def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
        from huggingface_hub import InferenceClient

        client = InferenceClient(model=self.model)
        resp = client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=output_token_budget,
        )
        return extract_response_text(resp)


class HFLocalProvider(LLMProvider):
    """Local transformers pipeline; runs on detected device (CUDA/MPS/CPU)."""

    name = "hf_local"

    def __init__(self, model: str = "Qwen/Qwen2.5-0.5B-Instruct") -> None:
        self.model = model
        self._pipe: Any = None

    @property
    def available(self) -> bool:
        try:
            import transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
        from transformers import pipeline

        if self._pipe is None:
            self._pipe = pipeline("text-generation", model=self.model, device_map="auto")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        out = self._pipe(messages, max_new_tokens=output_token_budget)
        return out[0]["generated_text"][-1]["content"]


class GeminiProvider(LLMProvider):
    """Google Gemini via OpenAI-compatible API."""

    name = "gemini"

    def __init__(self, model: str = "gemini-2.0-flash", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    @property
    def available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        # Official Google precedence: GOOGLE_API_KEY takes precedence over GEMINI_API_KEY
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))

    def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
        from openai import OpenAI

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
        client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/",
        )
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=output_token_budget,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        self.last_latency_seconds = time.perf_counter() - t0
        self.last_response_id = getattr(resp, "id", "") or ""
        usage = getattr(resp, "usage", None)
        if usage:
            self.last_input_tokens = getattr(usage, "prompt_tokens", 0)
            self.last_output_tokens = getattr(usage, "completion_tokens", 0)
        return extract_response_text(resp)


class DeepSeekProvider(LLMProvider):
    """DeepSeek via OpenAI-compatible API."""

    name = "deepseek"

    def __init__(self, model: str = "deepseek-chat", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    @property
    def available(self) -> bool:
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("DEEPSEEK_API_KEY"))

    def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com/v1",
        )
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=output_token_budget,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        self.last_latency_seconds = time.perf_counter() - t0
        self.last_response_id = getattr(resp, "id", "") or ""
        usage = getattr(resp, "usage", None)
        if usage:
            self.last_input_tokens = getattr(usage, "prompt_tokens", 0)
            self.last_output_tokens = getattr(usage, "completion_tokens", 0)
        return extract_response_text(resp)


class EnterpriseLLMGatewayProvider(LLMProvider):
    """Neutral placeholder for an internal enterprise LLM gateway.

    This class contains no endpoints, credentials, or firm-specific logic. It
    delegates entirely to ``start.enterprise.EnterpriseLLMGatewayAdapter`` —
    the single isolated file a firm implements. The adapter reports
    unavailable (and this provider therefore degrades to deterministic review)
    until a private implementation is supplied inside the firm environment.
    """

    name = "enterprise_llm_gateway"

    def __init__(self) -> None:
        from start.enterprise import EnterpriseLLMGatewayAdapter

        self._adapter = EnterpriseLLMGatewayAdapter()

    @property
    def available(self) -> bool:
        return self._adapter.available()

    def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
        # Bridge the provider interface (system/user) onto the adapter's
        # generate(prompt, *, system, metadata) contract.
        return self._adapter.generate(
            user,
            system=system,
            metadata={"output_token_budget": output_token_budget, "max_tokens": output_token_budget},
        )

    def generate(
        self, prompt: str, *, system: str | None = None, metadata: dict | None = None
    ) -> str:
        return self._adapter.generate(prompt, system=system, metadata=metadata)


_PROVIDERS: dict[str, type[LLMProvider]] = {
    "none": NoLLMProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "grok": GrokProvider,
    "gemini": GeminiProvider,
    "deepseek": DeepSeekProvider,
    "huggingface": HuggingFaceProvider,
    "hf_local": HFLocalProvider,
    "enterprise_llm_gateway": EnterpriseLLMGatewayProvider,
}


def get_llm_provider(config: LLMConfig, expected_domain: str | None = None) -> LLMProvider:
    """Resolve a provider, enforcing trust-domain separation and runtime profile egress containment.

    If ``expected_domain`` ('public' or 'private') is given, the requested
    provider must belong to it — crossover raises TrustDomainViolation. A
    provider that is unavailable degrades only WITHIN its own domain: public
    providers fall back to the deterministic no-LLM path; the enterprise
    gateway never falls back to a public provider (and public never falls back
    to the gateway)."""
    from start.providers.trust_domains import (
        TrustDomain,
        assert_no_crossover,
    )
    from start.runtime_profile import (
        PUBLIC_SAAS_PROVIDERS,
        RuntimeProfile,
        active_profile,
        assert_provider_allowed,
    )

    if expected_domain:
        assert_no_crossover(config.provider, TrustDomain(expected_domain))

    current_profile = active_profile()
    is_restricted = current_profile in (RuntimeProfile.ENTERPRISE, RuntimeProfile.AIRGAPPED)
    if is_restricted and config.provider in PUBLIC_SAAS_PROVIDERS:
        assert_provider_allowed(config.provider)

    provider_inst: LLMProvider
    if config.provider == "gateway":
        from start.providers.gateway import OpenAICompatibleGatewayProvider

        provider_inst = OpenAICompatibleGatewayProvider(model=config.model or "")
        if not provider_inst.available:
            return NoLLMProvider()
        return provider_inst

    from start.providers.gateway_discovery import registered_gateway_names

    if config.provider in registered_gateway_names():
        from start.providers.gateway import PluginGatewayProvider

        provider_inst = PluginGatewayProvider(config.provider)
        if not provider_inst.available:
            return NoLLMProvider()
        return provider_inst

    cls = _PROVIDERS.get(config.provider, NoLLMProvider)
    known_models = {"openai", "anthropic", "grok", "gemini", "deepseek", "huggingface", "hf_local"}
    try:
        if config.model and config.provider in known_models:
            provider_inst = cls(model=config.model)  # type: ignore[call-arg]
        else:
            provider_inst = cls()
    except TypeError:
        provider_inst = cls()  # type: ignore[call-arg]
    if not provider_inst.available and not isinstance(provider_inst, NoLLMProvider):
        # Within-domain degradation: an unavailable provider falls back to the
        # domain-NEUTRAL deterministic path (NoLLMProvider makes no external
        # calls and belongs to no trust domain), never to a provider in the
        # OTHER trust domain. Explicit public<->private crossover is blocked
        # above by assert_no_crossover; here we simply degrade safely.
        return NoLLMProvider()
    return provider_inst
