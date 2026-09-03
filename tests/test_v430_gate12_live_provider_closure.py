"""Gate 12: Live Provider Contract Closure & Full Question Path Tests.

Tests:
1. All 12 Response states (completed, incomplete, refusal, error, reasoning extraction, usage).
2. Safe live-diagnostic formatting without secret or reasoning leakage.
3. Full production Question [Q] action path with real OpenAIProvider method and mocked transport.
4. Full production Question [Q] action path with incomplete GPT-5 response due to max_output_tokens.
5. Deterministic fallback with LIVE_REVIEWER_NOT_VALIDATED provenance recording.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

from start.core.schemas import EvidenceRecord, Status
from start.data.synthetic_market import generate_market_world
from start.providers.base import ProviderResult, ProviderUsage
from start.providers.llm import (
    OpenAIProvider,
    extract_response_text,
    format_safe_provider_diagnostic,
)
from start.review.architecture import (
    LLMReviewConfig,
    ReviewContextBundle,
    ReviewDomain,
    ReviewExecutionProducts,
    ReviewMode,
)
from start.review.executor import run_domain_checkpoints


# =========================================================================== #
# Helper to construct realistic Responses API mock objects
# =========================================================================== #
def make_mock_response(
    *,
    response_id: str = "resp_12345",
    status: str = "completed",
    output_text: str | None = None,
    output: list | None = None,
    incomplete_reason: str | None = None,
    refusal: str | None = None,
    error: dict | None = None,
    input_tokens: int = 250,
    output_tokens: int = 150,
    reasoning_tokens: int = 0,
):
    resp = MagicMock()
    resp.id = response_id
    resp.status = status
    resp.output_text = output_text
    resp.refusal = refusal

    if output is not None:
        resp.output = output
    else:
        resp.output = []

    if incomplete_reason:
        resp.incomplete_details = types.SimpleNamespace(reason=incomplete_reason)
    else:
        resp.incomplete_details = None

    if error:
        resp.error = types.SimpleNamespace(
            code=error.get("code", "error"),
            type=error.get("type", "api_error"),
            message=error.get("message", "Error"),
        )
    else:
        resp.error = None

    usage = types.SimpleNamespace()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.output_tokens_details = types.SimpleNamespace(reasoning_tokens=reasoning_tokens)
    resp.usage = usage

    return resp


# =========================================================================== #
# 1. Bounded Mock Responses API States
# =========================================================================== #
def test_responses_api_completed_with_output_text(monkeypatch):
    """State 1: Completed response with primary output_text."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock")
    mock_resp = make_mock_response(
        status="completed",
        output_text="Portfolio risk is well within limits [EV-1].",
        input_tokens=100,
        output_tokens=50,
    )
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_resp
    monkeypatch.setattr("openai.OpenAI", lambda **kw: mock_client)

    provider = OpenAIProvider(model="gpt-5")
    res = provider.complete_result("System prompt", "User query", output_token_budget=4096)

    assert res.status == "completed"
    assert res.ok is True
    assert res.text == "Portfolio risk is well within limits [EV-1]."
    assert res.response_id == "resp_12345"
    assert res.usage.input_tokens == 100
    assert res.usage.output_tokens == 50


def test_responses_api_completed_with_nested_output_text(monkeypatch):
    """State 2: Completed response with nested message output items."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock")
    msg_item = types.SimpleNamespace(
        type="message",
        content=[types.SimpleNamespace(type="output_text", text="Nested answer text [EV-2].")],
    )
    mock_resp = make_mock_response(
        status="completed",
        output_text="",  # Empty aggregated
        output=[msg_item],
    )
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_resp
    monkeypatch.setattr("openai.OpenAI", lambda **kw: mock_client)

    provider = OpenAIProvider(model="gpt-5")
    res = provider.complete_result("System prompt", "User query", output_token_budget=4096)

    assert res.status == "completed"
    assert res.text == "Nested answer text [EV-2]."


def test_responses_api_completed_qualitative_visible_answer(monkeypatch):
    """State 3: Completed response with purely qualitative answer."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock")
    mock_resp = make_mock_response(
        status="completed",
        output_text="The portfolio governance aligns with SR 11-7 expectations.",
    )
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_resp
    monkeypatch.setattr("openai.OpenAI", lambda **kw: mock_client)

    provider = OpenAIProvider(model="gpt-5")
    res = provider.complete_result("System", "User", output_token_budget=4096)

    assert res.status == "completed"
    assert "SR 11-7" in res.text


def test_responses_api_completed_with_empty_output(monkeypatch):
    """State 4: Completed response with truly zero output text."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock")
    mock_resp = make_mock_response(
        status="completed",
        output_text="",
        output=[],
    )
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_resp
    monkeypatch.setattr("openai.OpenAI", lambda **kw: mock_client)

    provider = OpenAIProvider(model="gpt-5")
    res = provider.complete_result("System", "User", output_token_budget=4096)

    assert res.status == "empty"
    assert res.ok is False
    assert res.text == ""


def test_responses_api_incomplete_max_output_tokens(monkeypatch):
    """State 5: Incomplete response due to max_output_tokens exhaustion."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock")
    mock_resp = make_mock_response(
        status="incomplete",
        incomplete_reason="max_output_tokens",
        output_text="",
        input_tokens=300,
        output_tokens=1024,
        reasoning_tokens=1024,
    )
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_resp
    monkeypatch.setattr("openai.OpenAI", lambda **kw: mock_client)

    provider = OpenAIProvider(model="gpt-5")
    res = provider.complete_result("System", "User", output_token_budget=1024)

    assert res.status == "incomplete"
    assert res.incomplete_reason == "max_output_tokens"
    assert res.usage.reasoning_tokens == 1024
    assert res.text == ""


def test_responses_api_incomplete_other_reason(monkeypatch):
    """State 6: Incomplete response due to content_filter."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock")
    mock_resp = make_mock_response(
        status="incomplete",
        incomplete_reason="content_filter",
        output_text="",
    )
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_resp
    monkeypatch.setattr("openai.OpenAI", lambda **kw: mock_client)

    provider = OpenAIProvider(model="gpt-5")
    res = provider.complete_result("System", "User", output_token_budget=4096)

    assert res.status == "incomplete"
    assert res.incomplete_reason == "content_filter"


def test_responses_api_refusal(monkeypatch):
    """State 7: Model refusal."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock")
    mock_resp = make_mock_response(
        status="completed",
        refusal="I cannot perform review on this input.",
    )
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_resp
    monkeypatch.setattr("openai.OpenAI", lambda **kw: mock_client)

    provider = OpenAIProvider(model="gpt-5")
    res = provider.complete_result("System", "User", output_token_budget=4096)

    assert res.status == "refusal"
    assert res.refusal == "I cannot perform review on this input."


def test_responses_api_error_handling(monkeypatch):
    """State 8: Typed API error."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock")
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = RuntimeError("OpenAI connection timeout")
    monkeypatch.setattr("openai.OpenAI", lambda **kw: mock_client)

    provider = OpenAIProvider(model="gpt-5")
    res = provider.complete_result("System", "User", output_token_budget=4096)

    assert res.status == "error"
    assert res.error_type == "RuntimeError"
    assert "OpenAI connection timeout" in (res.error_message or "")


def test_responses_api_reasoning_item_plus_message_output():
    """State 9: Ignore reasoning items; extract visible message text only."""
    reasoning_item = types.SimpleNamespace(
        type="reasoning",
        summary="Thinking through the VaR backtest parameters...",
        text="Internal CoT text that must never be presented to user",
    )
    msg_item = types.SimpleNamespace(
        type="message",
        content=[types.SimpleNamespace(type="output_text", text="Visible answer: Volatility is 0.0937.")],
    )
    resp = types.SimpleNamespace(
        output_text=None,
        output=[reasoning_item, msg_item],
    )
    text = extract_response_text(resp)
    assert text == "Visible answer: Volatility is 0.0937."
    assert "CoT" not in text
    assert "Thinking" not in text


def test_responses_api_reasoning_only_output_no_message():
    """State 10: Reasoning-only output yields empty text."""
    reasoning_item = types.SimpleNamespace(
        type="reasoning",
        summary="Thinking only...",
        text="Internal thoughts",
    )
    resp = types.SimpleNamespace(
        output_text=None,
        output=[reasoning_item],
    )
    text = extract_response_text(resp)
    assert text == ""


def test_responses_api_usage_with_nonzero_reasoning_zero_visible_text(monkeypatch):
    """State 11: Usage details with 100% reasoning tokens consumed."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock")
    mock_resp = make_mock_response(
        status="incomplete",
        incomplete_reason="max_output_tokens",
        output_text="",
        input_tokens=450,
        output_tokens=1024,
        reasoning_tokens=1024,
    )
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_resp
    monkeypatch.setattr("openai.OpenAI", lambda **kw: mock_client)

    provider = OpenAIProvider(model="gpt-5")
    res = provider.complete_result("System", "User", output_token_budget=1024)

    assert res.usage.reasoning_tokens == 1024
    assert res.usage.output_tokens == 1024
    assert len(res.text) == 0


def test_format_safe_provider_diagnostic_no_secrets():
    """Verify format_safe_provider_diagnostic contains necessary metadata and zero secrets."""
    res = ProviderResult(
        text="",
        provider="openai",
        model="gpt-5",
        api_surface="responses",
        response_id="resp_safe_999",
        status="incomplete",
        incomplete_reason="max_output_tokens",
        max_output_tokens=4096,
        usage=ProviderUsage(input_tokens=350, output_tokens=1024, reasoning_tokens=1024),
        output_item_types=["reasoning"],
        content_part_types=[],
    )
    diag = format_safe_provider_diagnostic(res)

    assert "Provider: OpenAI" in diag
    assert "Model: gpt-5" in diag
    assert "API Surface: responses" in diag
    assert "Response ID: resp_safe_999" in diag
    assert "Status: incomplete" in diag
    assert "Incomplete Reason: max_output_tokens" in diag
    assert "Max Output Tokens Configured: 4096" in diag
    assert "Reasoning Tokens: 1024" in diag
    assert "Output Tokens: 1024" in diag
    assert "Aggregated Output-Text Length: 0" in diag

    # Hard rules: no secrets, no headers, no raw prompts
    assert "sk-" not in diag
    assert "Bearer" not in diag
    assert "authorization" not in diag.lower()


# =========================================================================== #
# 2. Full Production Question Path Tests
# =========================================================================== #
def test_full_production_question_path_success(monkeypatch):
    """Full question path: checkpoint -> [Q] -> real OpenAIProvider method -> mock SDK -> grounding -> completed."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-production-key")

    mock_resp = make_mock_response(
        response_id="resp_prod_ok_1",
        status="completed",
        output_text="The portfolio's annualised volatility was 0.0937 [EV-VOL]. Return attribution is verified.",
        input_tokens=300,
        output_tokens=80,
    )
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_resp
    monkeypatch.setattr("openai.OpenAI", lambda **kw: mock_client)

    world = generate_market_world(n_assets=5, n_periods=100, seed=42)
    market = world.market_context()
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        market=market,
        llm_config=LLMReviewConfig(backend_mode="public", provider="openai", model="gpt-5"),
    )

    ev_vol = EvidenceRecord(
        evidence_id="EV-VOL",
        test_id="portfolio.risk_statistics",
        test_name="Portfolio Risk Statistics",
        model_id="MKT-MODEL",
        dataset_id="MKT-DATA",
        run_id="RUN-TEST",
        status=Status.PASS,
        metrics={"annualised_volatility": 0.0937},
    )
    records = [ev_vol]

    # Scripted ask answers: Question, question note, Accept
    prompts: list[str] = []
    answers = iter(["Q", "Please explain the observed volatility metric.", "A"])

    def mock_ask(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    products = ReviewExecutionProducts()
    decisions = run_domain_checkpoints(
        bundle,
        records,
        artifacts_by_checkpoint={},
        products=products,
        interactive=True,
        ask=mock_ask,
    )

    # Verify real OpenAIProvider method was called with appropriate request parameters
    mock_client.responses.create.assert_called_once()
    called_kwargs = mock_client.responses.create.call_args.kwargs
    assert called_kwargs["model"] == "gpt-5"
    assert called_kwargs["max_output_tokens"] == 4096
    assert called_kwargs["reasoning"] == {"effort": "low"}
    assert called_kwargs["store"] is False

    # Check that decision recorded successful LLM verification and subsequent acceptance
    assert len(decisions) >= 2
    assert decisions[0]["action"] == "question"
    assert decisions[0]["backend"] == "llm"
    assert decisions[0]["live_provider_call"] is True
    assert "0.0937 [EV-VOL]" in decisions[0]["response"]
    assert decisions[1]["action"] == "accept"


def test_full_production_question_path_incomplete_gpt5_diagnostic(monkeypatch):
    """Full question path with incomplete GPT-5: surfaces INCOMPLETE_PROVIDER_RESPONSE and diagnostic metadata."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-production-key")

    mock_resp = make_mock_response(
        response_id="resp_incomplete_gpt5",
        status="incomplete",
        incomplete_reason="max_output_tokens",
        output_text="",
        input_tokens=500,
        output_tokens=1024,
        reasoning_tokens=1024,
    )
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_resp
    monkeypatch.setattr("openai.OpenAI", lambda **kw: mock_client)

    world = generate_market_world(n_assets=5, n_periods=100, seed=42)
    market = world.market_context()
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        market=market,
        llm_config=LLMReviewConfig(backend_mode="public", provider="openai", model="gpt-5"),
    )

    ev_vol = EvidenceRecord(
        evidence_id="EV-VOL",
        test_id="portfolio.risk_statistics",
        test_name="Portfolio Risk Statistics",
        model_id="MKT-MODEL",
        dataset_id="MKT-DATA",
        run_id="RUN-TEST",
        status=Status.PASS,
        metrics={"annualised_volatility": 0.0937},
    )
    records = [ev_vol]

    # Scripted ask answers: Question, question note, fallback choice "1" (deterministic), then Accept
    answers = iter(["Q", "Explain volatility under stress.", "1", "A"])

    def mock_ask(prompt: str) -> str:
        return next(answers)

    products = ReviewExecutionProducts()
    decisions = run_domain_checkpoints(
        bundle,
        records,
        artifacts_by_checkpoint={},
        products=products,
        interactive=True,
        ask=mock_ask,
    )

    # Assert decision explicitly records that live reviewer was NOT validated
    assert len(decisions) >= 1
    fallback_decisions = [d for d in decisions if d.get("backend") == "fallback"]
    assert len(fallback_decisions) >= 1
    fb = fallback_decisions[0]
    details = fb.get("details", {})
    assert details.get("live_reviewer_validated") is False
    assert details.get("live_reviewer_status") == "LIVE_REVIEWER_NOT_VALIDATED"
    assert "INCOMPLETE_PROVIDER_RESPONSE: max_output_tokens" in details.get("provider_error", "")
    assert details.get("diagnostic", {}).get("reasoning_tokens") == 1024
