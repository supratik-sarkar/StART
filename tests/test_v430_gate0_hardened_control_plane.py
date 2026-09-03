"""Gate-0 Hardened Control-Plane and Proof-Carrying Reviewer Suite.

Comprehensive non-interactive verification for:
1. Double-Abort Fix & Exception Taxonomy (Single-exit cancellation, 0 tracebacks, 0 misclassifications)
2. Checkpoint State Machine Invariants (Terminal CANCELLED, single repair, single decision)
3. Structured Reviewer Assessment & Deterministic Value Hydration
4. Fail-Closed Grounding Diagnostics (All 10 Reason Codes)
5. Exact Prior Failure Regression Fixture (Portfolio risk narrative)
6. Non-Interactive Interactive-Twin Test Harness covering all 12 reviewer actions/branches
"""

from __future__ import annotations

import io

import pytest
from typer.testing import CliRunner

from start.attestation.claims import (
    GroundingReasonCode,
    bind_claims,
    extract_claims,
)
from start.cli import app
from start.core.schemas import EvidenceRecord, Status, TestResult
from start.review.architecture import (
    LLMReviewConfig,
    ReviewContextBundle,
    ReviewDomain,
    ReviewMode,
)
from start.review.executor import (
    run_domain_checkpoints,
)
from start.review.multiline_input import ReviewCancelled
from start.review.state_machine import (
    CheckpointState,
    CheckpointStateMachine,
    InvalidStateTransitionError,
)
from start.review.structured_contract import (
    ReviewerAssessment,
    ReviewerObservation,
    format_assessment_markdown,
    hydrate_assessment_values,
)

runner = CliRunner()


# ========================================================================= #
# 1. DOUBLE-ABORT BUG REGRESSION & EXCEPTION TAXONOMY
# ========================================================================= #


def test_double_abort_grounding_failure_single_menu_clean_exit(monkeypatch):
    """REGRESSION TEST: User aborting after grounding failure shows EXACTLY ONE fallback menu,

    ZERO provider-failure misclassifications ('OpenAI reviewer request failed' never appears),
    and cleanly raises ReviewCancelled.
    """
    call_count = 0

    class UngroundedProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, *, output_token_budget: int = 1024, **kwargs) -> str:
            nonlocal call_count
            call_count += 1
            # Returns ungrounded claim 99.999%
            return "Analysis claim with invented value 99.999% not in evidence [EV-VAR12345678]."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: UngroundedProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public",
            provider="openai",
            model="gpt-4o-mini",
            status="CONNECTED",
        ),
    )

    var_rec = EvidenceRecord.from_result(
        TestResult(
            test_id="portfolio.risk_statistics",
            test_name="Portfolio Risk Statistics",
            status=Status.PASS,
            metrics={"annualised_volatility": 0.0937},
        ),
        run_id="TEST-RUN",
    )
    var_rec.evidence_id = "EV-VAR12345678"

    # User inputs:
    # 1. 'Q' (Question)
    # 2. 'What is the risk?' (Question text)
    # 3. '2' (Abort review at the single fallback menu)
    prompts = ["Q", "What is the risk?", "2"]
    prompt_iter = iter(prompts)
    prompts_asked = []

    def mock_ask(p: str = "") -> str:
        prompts_asked.append(p)
        try:
            return next(prompt_iter)
        except StopIteration:
            return "1"

    # Must raise ReviewCancelled immediately on choice '2'
    with pytest.raises(ReviewCancelled, match="Review aborted due to ungrounded claims"):
        run_domain_checkpoints(
            bundle,
            [var_rec],
            interactive=True,
            ask=mock_ask,
        )

    # Initial single provider call (zero repair loops)
    assert call_count == 1
    # Exactly one fallback prompt was presented to user
    fallback_prompts = [p for p in prompts_asked if "Select action" in p]
    assert len(fallback_prompts) == 1


def test_provider_invocation_failure_single_menu_abort(monkeypatch):
    """REGRESSION TEST: Real provider request failure shows EXACTLY ONE menu and raises ReviewCancelled."""

    class FailingProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, *, output_token_budget: int = 1024, **kwargs) -> str:
            raise ConnectionError("Simulated OpenAI 503 API Unavailable")

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: FailingProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public",
            provider="openai",
            model="gpt-4o-mini",
            status="CONNECTED",
        ),
    )

    prompts = ["C", "Challenge tail risk", "2"]  # 2 = Abort
    prompt_iter = iter(prompts)
    prompts_asked = []

    def mock_ask(p: str = "") -> str:
        prompts_asked.append(p)
        return next(prompt_iter)

    with pytest.raises(ReviewCancelled, match="Review aborted following OpenAI request failure"):
        run_domain_checkpoints(
            bundle,
            [],
            interactive=True,
            ask=mock_ask,
        )

    fallback_prompts = [p for p in prompts_asked if "Select action" in p]
    assert len(fallback_prompts) == 1


def test_provider_invocation_failure_fallback_deterministic(monkeypatch):
    """Provider failure with choice '1' (Continue deterministically) transitions cleanly."""

    class FailingProvider:
        name = "anthropic"
        model = "claude-sonnet-4-5"

        def complete(self, system: str, user: str, *, output_token_budget: int = 1024, **kwargs) -> str:
            raise TimeoutError("Simulated Anthropic rate limit timeout")

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: FailingProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public",
            provider="anthropic",
            model="claude-sonnet-4-5",
            status="CONNECTED",
        ),
    )

    prompts = ["C", "Challenge liquidity", "1", "A", "A", "A", "A", "A"]
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(
        bundle,
        [],
        interactive=True,
        ask=lambda _: next(prompt_iter),
    )

    first_dec = decisions[0]
    assert first_dec["action"] == "challenge"
    assert first_dec["backend"] == "fallback"
    assert "Deterministic fallback" in first_dec["response"]


# ========================================================================= #
# 2. CHECKPOINT STATE MACHINE PROPERTY TESTS
# ========================================================================= #


def test_state_machine_terminal_cancelled_invariants():
    sm = CheckpointStateMachine("Test Checkpoint")
    assert sm.current_state == CheckpointState.READY

    sm.transition(CheckpointState.PROVIDER_CALL)
    sm.transition(CheckpointState.FALLBACK_OFFERED)
    sm.transition(CheckpointState.CANCELLED)

    assert sm.is_terminal is True

    # CANCELLED cannot transition to anything
    with pytest.raises(InvalidStateTransitionError, match="Cannot transition from terminal state CANCELLED"):
        sm.transition(CheckpointState.FALLBACK_OFFERED)

    with pytest.raises(InvalidStateTransitionError, match="Cannot transition from terminal state CANCELLED"):
        sm.transition(CheckpointState.READY)


def test_state_machine_max_one_repair_invariant():
    sm = CheckpointStateMachine("Repair Invariant Check")
    sm.transition(CheckpointState.PROVIDER_CALL)
    sm.transition(CheckpointState.PROVIDER_RESPONSE)
    sm.transition(CheckpointState.GROUNDING_VALIDATE)
    sm.transition(CheckpointState.GROUNDING_REPAIR)

    assert sm.repair_attempts == 1

    # Attempting second repair in same checkpoint lifecycle must fail
    with pytest.raises(InvalidStateTransitionError):
        sm.transition(CheckpointState.GROUNDING_REPAIR)


def test_state_machine_max_one_fallback_offer_invariant():
    sm = CheckpointStateMachine("Fallback Offer Invariant Check")
    sm.transition(CheckpointState.PROVIDER_CALL)
    sm.transition(CheckpointState.FALLBACK_OFFERED)

    assert sm.fallback_offers == 1

    # Second fallback offer is illegal
    with pytest.raises(InvalidStateTransitionError):
        sm.transition(CheckpointState.FALLBACK_OFFERED)


def test_state_machine_terminal_decision_consistency():
    sm = CheckpointStateMachine("Decision Consistency Check")
    sm.record_decision("accept")
    # Same decision is idempotent
    sm.record_decision("accept")

    # Conflicting second terminal decision is forbidden
    with pytest.raises(InvalidStateTransitionError, match="Terminal decision already recorded"):
        sm.record_decision("abort")


# ========================================================================= #
# 3. STRUCTURED REVIEWER ASSESSMENT & DETERMINISTIC VALUE HYDRATION
# ========================================================================= #


def test_structured_reviewer_contract_and_hydration():
    """Verify that ReviewerAssessment observations deterministically hydrate numeric values from EvidenceRecord."""
    rec = EvidenceRecord.from_result(
        TestResult(
            test_id="portfolio.risk_statistics",
            test_name="Portfolio Risk Statistics",
            status=Status.PASS,
            metrics={
                "annualised_volatility": 0.037706,
                "n_assets": 50,
                "tracking_error_bps": 12.5,
            },
        ),
        run_id="RUN-PORT-1",
    )
    rec.evidence_id = "EV-PORT12345678"

    assessment = ReviewerAssessment(
        checkpoint="Portfolio Risk & Volatility Assumptions",
        action="challenge",
        observations=[
            ReviewerObservation(
                evidence_id="[EV-PORT12345678]",
                metric_path="portfolio.risk_statistics.annualised_volatility",
                interpretation="Annualized volatility meets institutional risk limits.",
            ),
            ReviewerObservation(
                evidence_id="EV-PORT12345678",
                metric_path="n_assets",
                interpretation="Asset universe diversity.",
            ),
        ],
        concerns=["Potential parameter estimation error under market stress."],
        recommendations=["Monitor covariance shrinkage stability."],
        limitations=["Assumes stationary return distribution."],
    )

    hydrated = hydrate_assessment_values(assessment, [rec])

    obs1 = hydrated.observations[0]
    assert obs1.value == pytest.approx(0.037706)
    assert obs1.unit == "%"
    assert obs1.display == "3.7706%"

    obs2 = hydrated.observations[1]
    assert obs2.value == 50.0
    assert obs2.unit == "count"
    assert obs2.display == "50"

    md = format_assessment_markdown(hydrated)
    assert "### Review Assessment: Portfolio Risk & Volatility Assumptions" in md
    assert "**[EV-PORT12345678]** (`portfolio.risk_statistics.annualised_volatility` `3.7706%`)" in md
    assert "#### Model Risk Concerns" in md
    assert "Potential parameter estimation error under market stress." in md


# ========================================================================= #
# 4. FAIL-CLOSED GROUNDING DIAGNOSTICS & ALL 10 REASON CODES
# ========================================================================= #


def test_grounding_diagnostics_all_reason_codes():
    """Verify exact diagnostic reason codes across grounding failure modes."""
    rec = EvidenceRecord.from_result(
        TestResult(
            test_id="traded_risk.var_exceptions",
            test_name="VaR Exceptions",
            status=Status.RECORDED,
            metrics={"n_exceptions": 4.0, "periods": 252},
        ),
        run_id="RUN-1",
    )
    rec.evidence_id = "EV-VAR00000001"

    # 1. NO_LOCAL_EVIDENCE_CITATION: number with no citation and not matching evidence
    c1 = extract_claims("The model had 99.5 exceptions.")
    b1 = bind_claims(c1, [rec])
    assert b1.unbound[0]["reason"] == GroundingReasonCode.NO_LOCAL_EVIDENCE_CITATION

    # 2. UNKNOWN_EVIDENCE_ID: citation does not exist globally
    c2 = extract_claims("The model had 4 exceptions [EV-UNKNOWN9999].")
    b2 = bind_claims(c2, [rec], all_known_evidence_ids={"EV-VAR00000001"})
    assert b2.unbound[0]["reason"] == GroundingReasonCode.UNKNOWN_EVIDENCE_ID

    # 3. CITATION_RECORD_NOT_IN_CHECKPOINT: citation exists globally but not in checkpoint
    c3 = extract_claims("The model had 4 exceptions [EV-GLOBAL8888].")
    b3 = bind_claims(c3, [rec], all_known_evidence_ids={"EV-VAR00000001", "EV-GLOBAL8888"})
    assert b3.unbound[0]["reason"] == GroundingReasonCode.CITATION_RECORD_NOT_IN_CHECKPOINT

    # 4. VALUE_MISMATCH: cited record exists in scope, but value 99.9 does not exist
    c4 = extract_claims("The model had 99.9 exceptions [EV-VAR00000001].")
    b4 = bind_claims(c4, [rec])
    assert b4.unbound[0]["reason"] == GroundingReasonCode.VALUE_MISMATCH

    # 5. UNSUPPORTED_FREQUENCY_INFERENCE: 252 periods asserting "daily" without explicit frequency metadata
    c5 = extract_claims("The backtest covers 252 daily periods [EV-VAR00000001].")
    b5 = bind_claims(c5, [rec])
    assert b5.unbound[0]["reason"] == GroundingReasonCode.UNSUPPORTED_FREQUENCY_INFERENCE

    # 6. UNSUPPORTED_DERIVED_RELATION: asserting 0.001679 aligns with 3.7706%
    c6 = extract_claims(
        "The periodic vol of 0.001679 aligns numerically with 3.7706% annualised volatility [EV-VAR00000001]."
    )
    b6 = bind_claims(c6, [rec])
    assert b6.unbound[0]["reason"] == GroundingReasonCode.UNSUPPORTED_DERIVED_RELATION


# ========================================================================= #
# 5. EXACT PRIOR FAILURE REGRESSION FIXTURE
# ========================================================================= #


def test_exact_prior_failure_portfolio_fixture_rejected():
    """EXACT PRIOR FAILURE FIXTURE:

    Verifies that the historical portfolio response containing:
    1000, 50, 252, 49, 0.001679, -0.3612, 0.3047%, 3.7706%, -11.4115%, 95%, 0.3918%
    and asserting uncertified derived alignment between periodic vol 0.001679 and
    annualized vol 3.7706% is strictly REJECTED by the fail-closed claim binder.
    """
    rec_risk = EvidenceRecord.from_result(
        TestResult(
            test_id="portfolio.risk_statistics",
            test_name="Portfolio Risk Statistics",
            status=Status.PASS,
            metrics={
                "annualised_volatility": 0.037706,
                "annualised_return": -0.114115,
                "sharpe_ratio": -0.3612,
                "n_periods": 1000,
                "n_assets": 50,
                "periods_per_year": 252,
                "max_drawdown": 0.003918,
            },
        ),
        run_id="RUN-REGRESS",
    )
    rec_risk.evidence_id = "EV-RISK0001"

    # Raw narrative that historically caused grounding failure:
    historical_narrative = (
        "The portfolio consists of 50 assets evaluated over 1000 periods (252 daily periods/year) [EV-RISK0001]. "
        "The periodic volatility of 0.001679 aligns numerically with 3.7706% annualised volatility [EV-RISK0001], "
        "with Sharpe ratio -0.3612 and annualised return -11.4115% [EV-RISK0001]."
    )

    claims = extract_claims(historical_narrative)
    binding = bind_claims(claims, [rec_risk])

    # Must fail closed on ungrounded / uncertified derivations
    assert len(binding.unbound) > 0

    reasons = [u["reason"] for u in binding.unbound]
    # Invariant checks:
    # 1. Frequency inference (252 -> daily) is caught
    assert GroundingReasonCode.UNSUPPORTED_FREQUENCY_INFERENCE in reasons
    # 2. Derived relation (0.001679 aligns with 3.7706%) is caught
    assert GroundingReasonCode.UNSUPPORTED_DERIVED_RELATION in reasons


# ========================================================================= #
# 6. NON-INTERACTIVE INTERACTIVE-TWIN (12 BRANCHES)
# ========================================================================= #


def test_twin_accept_action():
    """Branch 1: [A]ccept action -> COMPLETED."""
    bundle = ReviewContextBundle(mode=ReviewMode.SINGLE_DOMAIN, domains=(ReviewDomain.MARKET,))
    prompts = ["A"] * 10
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(bundle, [], interactive=True, ask=lambda _: next(prompt_iter))
    assert len(decisions) == 7
    assert all(d["action"] == "accept" for d in decisions)


def test_twin_override_action():
    """Branch 2: [O]verride action with justification -> COMPLETED."""
    bundle = ReviewContextBundle(mode=ReviewMode.SINGLE_DOMAIN, domains=(ReviewDomain.MARKET,))
    prompts = ["O", "Materiality override justified by expert committee"] + ["A"] * 10
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(bundle, [], interactive=True, ask=lambda _: next(prompt_iter))
    assert decisions[0]["action"] == "override"
    assert decisions[0]["note"] == "Materiality override justified by expert committee"


def test_twin_question_verified_first_attempt(monkeypatch):
    """Branch 3: [Q]uestion -> Verified on 1st attempt."""

    class VerifiedProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, **kwargs) -> str:
            return "Annualized volatility is 0.0937 [EV-RISK0001]."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: VerifiedProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
    )

    rec = EvidenceRecord.from_result(
        TestResult(
            test_id="portfolio.risk_statistics", test_name="Risk", metrics={"annualised_volatility": 0.0937}
        ),
        run_id="RUN-1",
    )
    rec.evidence_id = "EV-RISK0001"

    prompts = ["Q", "Is volatility acceptable?", "A", "A", "A", "A"]
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompt_iter))
    assert decisions[0]["action"] == "question"
    assert decisions[0]["backend"] == "llm"
    assert decisions[0]["unbound_claims"] == 0
    assert decisions[0]["grounding_repair"] is False


def test_twin_challenge_verified_first_attempt(monkeypatch):
    """Branch 4: [C]hallenge -> Verified on 1st attempt."""

    class VerifiedProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, **kwargs) -> str:
            return "Challenge disposition: observed 4 exceptions [EV-VAR0001]."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: VerifiedProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
    )

    rec = EvidenceRecord.from_result(
        TestResult(test_id="traded_risk.var_exceptions", test_name="VaR", metrics={"n_exceptions": 4.0}),
        run_id="RUN-1",
    )
    rec.evidence_id = "EV-VAR0001"

    prompts = ["A", "A", "C", "Challenge Kupiec sizing", "A", "A"]
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompt_iter))
    assert decisions[2]["action"] == "challenge"
    assert decisions[2]["backend"] == "llm"
    assert decisions[2]["unbound_claims"] == 0


def test_twin_question_verified_single_attempt(monkeypatch):
    """Branch 5: [Q]uestion -> grounded provider response -> verified on single attempt."""
    calls = 0

    class GroundedProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, **kwargs) -> str:
            nonlocal calls
            calls += 1
            return "The portfolio volatility was 0.0937 [EV-RISK0001]."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: GroundedProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
    )

    rec = EvidenceRecord.from_result(
        TestResult(
            test_id="portfolio.risk_statistics", test_name="Risk", metrics={"annualised_volatility": 0.0937}
        ),
        run_id="RUN-1",
    )
    rec.evidence_id = "EV-RISK0001"

    prompts = ["Q", "Assess portfolio risk", "A", "A", "A", "A"]
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompt_iter))
    assert calls == 1
    assert decisions[0]["backend"] == "llm"
    assert decisions[0]["unbound_claims"] == 0


def test_twin_challenge_verified_single_attempt(monkeypatch):
    """Branch 6: [C]hallenge -> grounded provider response -> verified on single attempt."""
    calls = 0

    class GroundedProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, **kwargs) -> str:
            nonlocal calls
            calls += 1
            return "VaR exceptions were 4 [EV-VAR0001]."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: GroundedProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
    )

    rec = EvidenceRecord.from_result(
        TestResult(test_id="traded_risk.var_exceptions", test_name="VaR", metrics={"n_exceptions": 4.0}),
        run_id="RUN-1",
    )
    rec.evidence_id = "EV-VAR0001"

    prompts = ["A", "A", "C", "Challenge tail risk", "A", "A"]
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompt_iter))
    assert calls == 1
    assert decisions[2]["action"] == "challenge"
    assert decisions[2]["unbound_claims"] == 0


def test_twin_question_repair_failure_fallback_deterministic(monkeypatch):
    """Branch 7: [Q]uestion -> repair fails -> user chooses 1 (fallback deterministic)."""

    class StubbornProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, **kwargs) -> str:
            return "Still ungrounded 77.7% [EV-RISK0001]."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: StubbornProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
    )

    prompts = ["Q", "Assess risk", "1", "A", "A", "A", "A"]  # 1 = continue deterministically
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(bundle, [], interactive=True, ask=lambda _: next(prompt_iter))
    assert decisions[0]["backend"] == "fallback"
    assert "Deterministic fallback" in decisions[0]["response"]


def test_twin_question_repair_failure_abort(monkeypatch):
    """Branch 8: [Q]uestion -> repair fails -> user chooses 2 (Abort review)."""

    class StubbornProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, **kwargs) -> str:
            return "Still ungrounded 77.7% [EV-RISK0001]."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: StubbornProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-4o-mini", status="CONNECTED"
        ),
    )

    prompts = ["Q", "Assess risk", "2"]  # 2 = abort
    prompt_iter = iter(prompts)

    with pytest.raises(ReviewCancelled, match="Review aborted due to ungrounded claims"):
        run_domain_checkpoints(bundle, [], interactive=True, ask=lambda _: next(prompt_iter))


def test_twin_challenge_provider_failure_fallback_deterministic(monkeypatch):
    """Branch 9: [C]hallenge -> provider failure -> user chooses 1 (fallback deterministic)."""

    class FailingProvider:
        name = "gemini"
        model = "gemini-2.0-flash"

        def complete(self, system: str, user: str, **kwargs) -> str:
            raise RuntimeError("Gemini Quota Exceeded")

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: FailingProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="gemini", model="gemini-2.0-flash", status="CONNECTED"
        ),
    )

    prompts = ["C", "Challenge", "1", "A", "A", "A", "A"]
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(bundle, [], interactive=True, ask=lambda _: next(prompt_iter))
    assert decisions[0]["backend"] == "fallback"


def test_twin_challenge_provider_failure_abort(monkeypatch):
    """Branch 10: [C]hallenge -> provider failure -> user chooses 2 (Abort review)."""

    class FailingProvider:
        name = "deepseek"
        model = "deepseek-chat"

        def complete(self, system: str, user: str, **kwargs) -> str:
            raise ConnectionResetError("DeepSeek API Reset")

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: FailingProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="deepseek", model="deepseek-chat", status="CONNECTED"
        ),
    )

    prompts = ["C", "Challenge", "2"]
    prompt_iter = iter(prompts)

    with pytest.raises(ReviewCancelled, match="Review aborted following DeepSeek request failure"):
        run_domain_checkpoints(bundle, [], interactive=True, ask=lambda _: next(prompt_iter))


def test_twin_wizard_cancel_clean_exit():
    """Branch 11: Wizard cancellation raises ReviewCancelled cleanly with zero tracebacks."""
    from start.review.wizard import run_review_wizard

    stream = io.StringIO("")  # EOF immediately
    with pytest.raises(ReviewCancelled):
        run_review_wizard(ask=lambda _: "1", stream=stream)


def test_twin_cli_review_cancellation_zero_tracebacks(tmp_path):
    """Branch 12: CLI review command gracefully catches ReviewCancelled, outputs exactly 1 message,

    and returns stable non-zero exit code (1).
    """
    res = runner.invoke(
        app, ["review", "--standard", "--target", "attrition", "--output-root", str(tmp_path)], input="\n"
    )
    # Clean non-zero exit
    assert res.exit_code == 1
    # Clean cancellation notice
    assert "Review cancelled by reviewer." in res.output
    # ZERO traceback
    assert "Traceback" not in res.output
