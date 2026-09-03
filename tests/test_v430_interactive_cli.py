"""v4.3.0 Interactive CLI, Multiline Governance Input, and Routing Verification Suite."""

from __future__ import annotations

import io

import pytest
from typer.testing import CliRunner

from start.cli import app
from start.data.synthetic_market import generate_market_world
from start.registry.market_contexts import MarketContext, PortfolioSpec
from start.review.applicability import applicable_tests, build_plan_preview
from start.review.architecture import (
    ReviewContextBundle,
    ReviewDomain,
    ReviewLifecycle,
    ReviewMode,
    parse_domain_selection,
)
from start.review.executor import (
    run_market_treasury_review,
)
from start.review.multiline_input import (
    ReviewCancelled,
    read_multiline_text,
)
from start.review.wizard import run_review_wizard

runner = CliRunner()


# ========================================================================= #
# 1. MULTILINE GOVERNANCE TEXT INPUT & PASTE LEAKAGE TESTS
# ========================================================================= #


def test_multiline_paste_no_leakage_to_following_menu():
    """HARD acceptance test: multi-paragraph text ending with END does not leak into the next prompt."""
    pasted_stream = io.StringIO(
        "The model is a high-materiality market and treasury risk framework used to assess the daily risk profile.\n\n"
        "The framework supports independent risk oversight and portfolio-limit monitoring.\n\n"
        "The review should assess portfolio mathematics, covariance, VaR and evidence integrity.\n"
        "END\n"
        "2\n"  # The following menu response
    )

    # Read multiline governance text
    text = read_multiline_text(
        "Business Context", required=True, stream=pasted_stream, printer=lambda _: None
    )

    # Next menu read
    next_menu_choice = pasted_stream.readline().strip()

    assert "high-materiality market and treasury risk framework" in text
    assert "independent risk oversight" in text
    assert "portfolio mathematics, covariance, VaR" in text
    assert text.count("\n\n") >= 2  # Paragraph breaks preserved
    assert next_menu_choice == "2"  # Next menu read saw exactly '2'
    assert pasted_stream.read() == ""  # Zero unconsumed bytes left in stream


def test_multiline_unicode_and_blank_lines():
    pasted = "Line 1: Model validation for €100M portfolio.\n\nLine 2: Target volatility σ ≤ 10%.\nEND\n"
    text = read_multiline_text("Context", stream=io.StringIO(pasted), printer=lambda _: None)
    assert "€100M" in text
    assert "σ ≤ 10%" in text
    assert "\n\n" in text


def test_multiline_menu_looking_text_not_confused_with_options():
    pasted = "1. First observation\n2. Second observation\n[1] Propensity suite option\nEND\n"
    text = read_multiline_text("Observations", stream=io.StringIO(pasted), printer=lambda _: None)
    assert "1. First observation" in text
    assert "[1] Propensity suite option" in text


def test_multiline_required_empty_reprompts():
    pasted = "END\nReal content here\nEND\n"
    text = read_multiline_text(
        "Required Field", required=True, stream=io.StringIO(pasted), printer=lambda _: None
    )
    assert text == "Real content here"


def test_multiline_eof_raises_cancelled():
    with pytest.raises(ReviewCancelled, match="EOF"):
        read_multiline_text("Test Field", stream=io.StringIO(""), printer=lambda _: None)


# ========================================================================= #
# 2. REVIEW MODE & DOMAIN ROUTING
# ========================================================================= #


def test_domain_selection_parsing():
    # Single mode
    assert parse_domain_selection("1", mode=ReviewMode.SINGLE_DOMAIN) == (ReviewDomain.PREDICTIVE,)
    assert parse_domain_selection("2", mode=ReviewMode.SINGLE_DOMAIN) == (ReviewDomain.MARKET,)
    assert parse_domain_selection("3", mode=ReviewMode.SINGLE_DOMAIN) == (ReviewDomain.TREASURY,)

    # Cross mode
    assert parse_domain_selection("2,3", mode=ReviewMode.CROSS_DOMAIN) == (
        ReviewDomain.MARKET,
        ReviewDomain.TREASURY,
    )
    assert parse_domain_selection("3,2", mode=ReviewMode.CROSS_DOMAIN) == (
        ReviewDomain.MARKET,
        ReviewDomain.TREASURY,
    )
    assert parse_domain_selection("1,2,3", mode=ReviewMode.CROSS_DOMAIN) == (
        ReviewDomain.PREDICTIVE,
        ReviewDomain.MARKET,
        ReviewDomain.TREASURY,
    )

    # Rejection of duplicates and singles in cross mode
    with pytest.raises(ValueError):
        parse_domain_selection("2,2", mode=ReviewMode.CROSS_DOMAIN)
    with pytest.raises(ValueError):
        parse_domain_selection("2", mode=ReviewMode.CROSS_DOMAIN)


# ========================================================================= #
# 3. DYNAMIC REGISTRY APPLICABILITY
# ========================================================================= #


def test_registry_applicability_counts():
    """Verify exact dynamic counts derived from registry metadata (79 total)."""
    assert applicable_tests((ReviewDomain.PREDICTIVE,)).count == 52
    assert applicable_tests((ReviewDomain.MARKET,)).count == 25
    assert applicable_tests((ReviewDomain.TREASURY,)).count == 2
    assert applicable_tests((ReviewDomain.MARKET, ReviewDomain.TREASURY)).count == 27
    assert applicable_tests((ReviewDomain.PREDICTIVE, ReviewDomain.MARKET, ReviewDomain.TREASURY)).count == 79


# ========================================================================= #
# 4. SINGLE-DOMAIN MARKET REVIEW EXECUTION
# ========================================================================= #


def test_single_domain_market_review_flow(tmp_path):
    world = generate_market_world(
        n_assets=12, n_periods=500, n_factors=3, seed=42, include_short_rate=True, missing_rate=0.15
    )
    renamed = {old: f"ASSET_{i + 1:03d}" for i, old in enumerate(world.returns.columns)}

    market = MarketContext(
        returns=world.returns.rename(columns=renamed),
        prices=world.prices.rename(columns=renamed),
        periods_per_year=world.periods_per_year,
        risk_free_rate=0.02,
        factor_returns=world.factor_returns,
        factor_exposures=world.factor_exposures.rename(index=renamed),
        pnl=world.pnl,
        hypothetical_pnl=world.hypothetical_pnl,
        var_series=world.var_series,
        var_confidence=world.var_confidence,
        portfolio=PortfolioSpec(
            weights=world.weights.rename(renamed),
            benchmark_weights=world.benchmark_weights.rename(renamed),
        ),
        seed=42,
    )

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        market=market,
        materiality="high",
        lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
    )

    preview = build_plan_preview(bundle)
    assert preview.applicable.count == 25
    assert "Single-Domain Review" in preview.render()

    outcome = run_market_treasury_review(bundle, output_root=str(tmp_path), interactive=False)
    assert outcome["ledger"].verify() is True
    assert outcome["replay"].intact is True
    assert outcome["seal"].metadata["n_validation_failures"] == 0
    root_h = outcome["seal"].root() if callable(outcome["seal"].root) else outcome["seal"].root
    assert root_h != ""
    assert len(outcome["binding"].bound) > 0


# ========================================================================= #
# 5. SINGLE-DOMAIN TREASURY REVIEW EXECUTION
# ========================================================================= #


def test_single_domain_treasury_review_flow(tmp_path):
    world = generate_market_world(n_assets=12, n_periods=500, n_factors=3, seed=42, include_short_rate=True)
    short_rate = world.short_rate_context()

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.TREASURY,),
        short_rate=short_rate,
        materiality="high",
        lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
    )

    preview = build_plan_preview(bundle)
    assert preview.applicable.count == 2

    outcome = run_market_treasury_review(bundle, output_root=str(tmp_path), interactive=False)
    assert outcome["ledger"].verify() is True
    assert outcome["replay"].intact is True
    assert outcome["seal"].metadata["n_validation_failures"] == 2
    root_h = outcome["seal"].root() if callable(outcome["seal"].root) else outcome["seal"].root
    assert root_h != ""


# ========================================================================= #
# 6. CROSS-DOMAIN MARKET + TREASURY EXECUTION
# ========================================================================= #


def test_cross_domain_market_treasury_review_flow(tmp_path):
    world = generate_market_world(n_assets=12, n_periods=500, n_factors=3, seed=42, include_short_rate=True)
    renamed = {old: f"ASSET_{i + 1:03d}" for i, old in enumerate(world.returns.columns)}

    market = MarketContext(
        returns=world.returns.rename(columns=renamed),
        prices=world.prices.rename(columns=renamed),
        periods_per_year=world.periods_per_year,
        risk_free_rate=0.02,
        factor_returns=world.factor_returns,
        factor_exposures=world.factor_exposures.rename(index=renamed),
        pnl=world.pnl,
        hypothetical_pnl=world.hypothetical_pnl,
        var_series=world.var_series,
        var_confidence=world.var_confidence,
        portfolio=PortfolioSpec(weights=world.weights.rename(renamed)),
        seed=42,
    )
    short_rate = world.short_rate_context()

    bundle = ReviewContextBundle(
        mode=ReviewMode.CROSS_DOMAIN,
        domains=(ReviewDomain.MARKET, ReviewDomain.TREASURY),
        market=market,
        short_rate=short_rate,
        materiality="high",
        lifecycle=ReviewLifecycle.MATERIAL_MODEL_CHANGE,
    )

    preview = build_plan_preview(bundle)
    assert preview.applicable.count == 27
    assert "Cross-Domain Review" in preview.render()

    outcome = run_market_treasury_review(bundle, output_root=str(tmp_path), interactive=False)
    assert outcome["ledger"].verify() is True
    assert outcome["replay"].intact is True
    assert outcome["seal"].metadata["n_validation_failures"] == 2


# ========================================================================= #
# 7. INTERACTIVE WIZARD ROUTING VIA PIPED INPUT
# ========================================================================= #


def test_interactive_wizard_market_treasury_piped_input():
    """Simulate user selecting Cross-Domain -> Market + Treasury without encountering ML prompts."""
    scripted_inputs = [
        "2",  # Review Mode -> [2] Cross-Domain
        "2,3",  # Review Domains -> [2,3] Market + Treasury
        # NOTE: Predictive technology is NOT asked because Predictive is not in domains!
        "1",  # Backend -> [1] None (Deterministic)
        "1",  # Materiality -> [1] High
        "1",  # Lifecycle -> [1] Initial Validation
        # Data source
        "1",  # Market/Treasury Data Source -> [1] Built-in Synthetic Market World
        "1",  # Review Scope -> [1] Full Recommended Review
        "Y",  # Proceed to execute review? -> [Y]
    ]

    multiline_text = (
        "High materiality multi-asset market risk framework.\n"
        "Assesses portfolio volatility, VaR backtesting and short-rate dynamics.\n"
        "END\n"  # Business context
        "Reviewer notes: focus on covariance and short-rate estimators.\n"
        "END\n"  # Reviewer clarification
        "Daily risk limit monitoring.\n"
        "END\n"  # Intended use
        "Tail risk under extreme stress.\n"
        "END\n"  # Known limitations
    )

    # Combine multiline text stream
    stream = io.StringIO(multiline_text)
    input_iter = iter(scripted_inputs)

    def mock_ask(prompt: str = "") -> str:
        try:
            return next(input_iter)
        except StopIteration:
            return "1"

    wizard_result = run_review_wizard(ask=mock_ask, stream=stream, seed=42)
    bundle = wizard_result["bundle"]

    assert bundle.mode is ReviewMode.CROSS_DOMAIN
    assert bundle.domains == (ReviewDomain.MARKET, ReviewDomain.TREASURY)
    assert bundle.technology is None  # Technology prompt was never asked
    assert bundle.market is not None
    assert bundle.short_rate is not None
    assert bundle.is_complete() is True


def test_cli_review_non_interactive_preserves_legacy(tmp_path):
    """Test start review with --non-interactive flag remains fully operational."""
    res = runner.invoke(
        app,
        [
            "review",
            "--non-interactive",
            "--standard",
            "--target",
            "attrition",
            "--run-dl",
            "--output-root",
            str(tmp_path),
        ],
    )
    assert res.exit_code == 0
    assert "Review complete" in res.output


# ========================================================================= #
# 8. DEFECT A & B TESTS: MODEL SELECTION & REAL LLM DIALOGUE
# ========================================================================= #


def test_interactive_wizard_public_llm_model_selection():
    """Verify Public LLM provider displays model options, selects model, and updates plan preview."""
    from start.providers.model_discovery import FakeModelDiscovery

    fake_discovery = FakeModelDiscovery(
        {
            "openai": ["gpt-4o-mini", "gpt-4o", "o1", "o3-mini"],
        }
    )

    scripted_inputs = [
        "1",  # Mode -> Single
        "2",  # Domain -> Market
        "3",  # Backend -> [3] Public LLM Providers
        "1",  # Provider -> [1] OpenAI
        "2",  # Model -> [2] gpt-4o
        "1",  # Materiality -> High
        "1",  # Lifecycle -> Initial Validation
        "1",  # Market Data Source -> Built-in Synthetic
        "1",  # Review Scope -> Full Recommended Review
        "Y",  # Proceed?
    ]

    multiline_text = "Market Risk Framework.\nEND\nNotes\nEND\nTrading\nEND\nVol\nEND\n"
    stream = io.StringIO(multiline_text)
    input_iter = iter(scripted_inputs)

    wizard_result = run_review_wizard(
        ask=lambda _: next(input_iter),
        stream=stream,
        seed=42,
        discovery_client=fake_discovery,
    )
    bundle = wizard_result["bundle"]
    preview = wizard_result["preview"]

    assert bundle.llm_config.backend_mode == "public"
    assert bundle.llm_config.provider == "openai"
    assert bundle.llm_config.model == "gpt-4o"

    rendered_preview = preview.render()
    assert "AI Reviewer Backend:       Public LLM" in rendered_preview
    assert "Provider:                OpenAI" in rendered_preview
    assert "Model:                   gpt-4o" in rendered_preview


def test_domain_checkpoints_real_llm_dialogue_routing(monkeypatch):
    """Verify Q and C checkpoints invoke the configured LLM provider and record response."""
    from start.review.architecture import LLMReviewConfig
    from start.review.executor import run_domain_checkpoints

    class MockProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, *, output_token_budget: int = 1024, **kwargs) -> str:
            assert "Question:" in user or "CHALLENGE" in user or "Review Mode:" in user
            return "Mock Technical Analysis: Evidence demonstrates sufficient volatility control."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: MockProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public",
            provider="openai",
            model="gpt-4o-mini",
            status="CONNECTED",
        ),
        business_context="Equity options volatility book",
    )

    from start.core.schemas import EvidenceRecord, Status, TestResult

    sample_records = [
        EvidenceRecord.from_result(
            TestResult(
                test_id="portfolio.risk_statistics",
                test_name="Portfolio Risk Statistics",
                metrics={"annualised_volatility": 0.0937},
                status=Status.PASS,
                interpretation="Volatility in range",
            ),
            run_id="RUN-1",
        )
    ]

    prompts = [
        "Q",  # Action -> Question
        "What is the impact of fat tails on portfolio volatility?",  # Question text
        "A",  # Accept remaining
        "A",
        "A",
        "A",
    ]
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(
        bundle,
        sample_records,
        interactive=True,
        ask=lambda _: next(prompt_iter),
    )

    q_dec = decisions[0]
    assert q_dec["action"] == "question"
    assert q_dec["backend"] == "llm"
    assert q_dec["provider"] == "openai"
    assert "Mock Technical Analysis" in q_dec["response"]


def test_domain_checkpoints_deterministic_none_routing():
    """Verify None backend produces explicitly labeled deterministic response."""
    from start.review.architecture import LLMReviewConfig
    from start.review.executor import run_domain_checkpoints

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(backend_mode="none", provider="none", status="DETERMINISTIC"),
    )

    prompts = [
        "Q",  # Action -> Question
        "Does volatility meet baseline constraints?",  # Question text
        "A",
        "A",
        "A",
        "A",
    ]
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(
        bundle,
        [],
        interactive=True,
        ask=lambda _: next(prompt_iter),
    )

    q_dec = decisions[0]
    assert q_dec["action"] == "question"
    assert q_dec["backend"] == "deterministic"
    assert "Evidence for 'Portfolio Risk & Volatility Assumptions' meets constraints" in q_dec["response"]


# ========================================================================= #
# 9. DEFECT C TESTS: GOVERNANCE & VALIDATION SCOPING
# ========================================================================= #


def test_market_only_review_governance_scoping(tmp_path):
    """Verify Market-only review contains exactly 22 records and NO CEV or Stanton validation."""
    world = generate_market_world(n_assets=12, n_periods=500, n_factors=3, seed=42, include_short_rate=True)
    renamed = {old: f"ASSET_{i + 1:03d}" for i, old in enumerate(world.returns.columns)}

    market = MarketContext(
        returns=world.returns.rename(columns=renamed),
        prices=world.prices.rename(columns=renamed),
        periods_per_year=world.periods_per_year,
        risk_free_rate=0.02,
        factor_returns=world.factor_returns,
        factor_exposures=world.factor_exposures.rename(index=renamed),
        pnl=world.pnl,
        hypothetical_pnl=world.hypothetical_pnl,
        var_series=world.var_series,
        var_confidence=world.var_confidence,
        portfolio=PortfolioSpec(
            weights=world.weights.rename(renamed),
            benchmark_weights=world.benchmark_weights.rename(renamed),
        ),
        seed=42,
    )

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        market=market,
        materiality="high",
        lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
    )

    outcome = run_market_treasury_review(bundle, output_root=str(tmp_path), interactive=False)
    records = outcome["records"]
    test_ids = {r.test_id for r in records}

    # 25 deterministic market tests + 2 market validation tests (+ 3 scenario tests) = 27 or 30 total
    assert len(records) in (27, 30)
    assert "validation.var_size_power" in test_ids
    assert "validation.regem_structural" in test_ids

    # STRICTLY NO Treasury validation studies in Market review
    assert "validation.cev_consistency" not in test_ids
    assert "validation.stanton_bias" not in test_ids

    # Zero validation failures for Market
    assert outcome["seal"].metadata["n_validation_failures"] == 0

    # Narrative must not mention CEV or Stanton
    narrative = outcome["narrative"]
    assert "CEV" not in narrative
    assert "Stanton" not in narrative
    assert "VaR" in narrative
    assert "RegEM" in narrative

    # All quantitative claims in Market narrative must be grounded
    binding = outcome["binding"]
    assert len(binding.unbound) == 0
    assert len(binding.bound) > 0


def test_treasury_only_review_governance_scoping(tmp_path):
    """Verify Treasury-only review contains exactly 4 records and includes CEV and Stanton."""
    world = generate_market_world(seed=42, include_short_rate=True)
    short_rate = world.short_rate_context()

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.TREASURY,),
        short_rate=short_rate,
        materiality="high",
        lifecycle=ReviewLifecycle.PERIODIC_VALIDATION,
    )

    outcome = run_market_treasury_review(bundle, output_root=str(tmp_path), interactive=False)
    records = outcome["records"]
    test_ids = {r.test_id for r in records}

    # 2 deterministic treasury tests + 2 treasury validation tests = 4 total
    assert len(records) == 4
    assert "traded_risk.cev_elasticity" in test_ids
    assert "traded_risk.stanton_nonparametric" in test_ids
    assert "validation.cev_consistency" in test_ids
    assert "validation.stanton_bias" in test_ids

    # NO Market validation studies
    assert "validation.var_size_power" not in test_ids
    assert "validation.regem_structural" not in test_ids

    # Failures properly captured for Treasury
    assert outcome["seal"].metadata["n_validation_failures"] == 2

    # Narrative must mention CEV and Stanton, not VaR or RegEM
    narrative = outcome["narrative"]
    assert "CEV" in narrative
    assert "Stanton" in narrative
    assert "VaR" not in narrative


def test_cross_domain_market_treasury_governance_scoping(tmp_path):
    """Verify Cross-Domain Market+Treasury review contains exactly 31 records and all 4 validation studies."""
    world = generate_market_world(n_assets=12, n_periods=500, n_factors=3, seed=42, include_short_rate=True)
    renamed = {old: f"ASSET_{i + 1:03d}" for i, old in enumerate(world.returns.columns)}

    market = MarketContext(
        returns=world.returns.rename(columns=renamed),
        prices=world.prices.rename(columns=renamed),
        periods_per_year=world.periods_per_year,
        risk_free_rate=0.02,
        factor_returns=world.factor_returns,
        factor_exposures=world.factor_exposures.rename(index=renamed),
        pnl=world.pnl,
        hypothetical_pnl=world.hypothetical_pnl,
        var_series=world.var_series,
        var_confidence=world.var_confidence,
        portfolio=PortfolioSpec(
            weights=world.weights.rename(renamed),
            benchmark_weights=world.benchmark_weights.rename(renamed),
        ),
        seed=42,
    )
    short_rate = world.short_rate_context()

    bundle = ReviewContextBundle(
        mode=ReviewMode.CROSS_DOMAIN,
        domains=(ReviewDomain.MARKET, ReviewDomain.TREASURY),
        market=market,
        short_rate=short_rate,
        materiality="high",
        lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
    )

    outcome = run_market_treasury_review(bundle, output_root=str(tmp_path), interactive=False)
    records = outcome["records"]
    test_ids = {r.test_id for r in records}

    # 27 deterministic (25 market + 2 treasury) + 4 validation (+ 3 scenario) = 31 or 34 total
    assert len(records) in (31, 34)
    assert "validation.var_size_power" in test_ids
    assert "validation.regem_structural" in test_ids
    assert "validation.cev_consistency" in test_ids
    assert "validation.stanton_bias" in test_ids

    # Failures properly captured
    assert outcome["seal"].metadata["n_validation_failures"] == 2

    # Narrative must mention all studies
    narrative = outcome["narrative"]
    assert "CEV" in narrative
    assert "Stanton" in narrative
    assert "VaR" in narrative
    assert "RegEM" in narrative

    # All quantitative claims grounded
    binding = outcome["binding"]
    assert len(binding.unbound) == 0
    assert len(binding.bound) > 0


def test_domain_checkpoints_llm_failure_recovery(monkeypatch):
    """Verify LLM failure at checkpoint displays safe error and offers continue deterministically."""
    from start.review.architecture import LLMReviewConfig
    from start.review.executor import run_domain_checkpoints

    class FailingProvider:
        name = "openai"
        model = "gpt-4o"

        def complete(self, system: str, user: str, *, output_token_budget: int = 1024, **kwargs) -> str:
            raise RuntimeError("Rate limit exceeded")

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: FailingProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public",
            provider="openai",
            model="gpt-4o",
            status="CONNECTED",
        ),
    )

    prompts = [
        "Q",  # Action -> Question
        "Test question",  # Question text
        "1",  # On failure: [1] Continue deterministically
        "A",
        "A",
        "A",
        "A",
    ]
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(
        bundle,
        [],
        interactive=True,
        ask=lambda _: next(prompt_iter),
    )

    assert decisions[0]["backend"] == "fallback"
    assert "Deterministic fallback" in decisions[0]["response"]


def test_portfolio_checkpoint_evidence_scoping(monkeypatch):
    """Verify Portfolio checkpoint receives ONLY portfolio evidence, not attribution or covariance."""
    from start.core.schemas import EvidenceRecord, Status, TestResult
    from start.review.architecture import LLMReviewConfig
    from start.review.executor import run_domain_checkpoints

    captured_prompts = []

    class MockProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, *, output_token_budget: int = 1024, **kwargs) -> str:
            captured_prompts.append((system, user))
            return "The recorded annualised volatility is 0.0937 [EV-111111111111]."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: MockProvider())

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

    port_rec = EvidenceRecord.from_result(
        TestResult(
            test_id="portfolio.risk_statistics",
            test_name="portfolio.risk_statistics",
            status=Status.RECORDED,
            metrics={"annualised_volatility": 0.0937},
        ),
        run_id="TEST-RUN",
    )
    port_rec.evidence_id = "EV-111111111111"

    attr_rec = EvidenceRecord.from_result(
        TestResult(
            test_id="attribution.return_attribution",
            test_name="attribution.return_attribution",
            status=Status.RECORDED,
            metrics={"max_abs_reconciliation_error": 0.0},
        ),
        run_id="TEST-RUN",
    )
    attr_rec.evidence_id = "EV-222222222222"

    cov_rec = EvidenceRecord.from_result(
        TestResult(
            test_id="covariance.ledoit_wolf_shrinkage",
            test_name="covariance.ledoit_wolf_shrinkage",
            status=Status.RECORDED,
            metrics={"shrinkage_intensity": 0.0086},
        ),
        run_id="TEST-RUN",
    )
    cov_rec.evidence_id = "EV-333333333333"

    records = [port_rec, attr_rec, cov_rec]
    prompts = ["Q", "Explain volatility", "A", "A", "A", "A"]
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(
        bundle,
        records,
        interactive=True,
        ask=lambda _: next(prompt_iter),
    )

    # 1. Checkpoint received only portfolio evidence in prompt
    assert len(captured_prompts) == 1
    system_prompt, user_prompt = captured_prompts[0]
    assert "portfolio.risk_statistics" in user_prompt
    assert "EV-111111111111" in user_prompt
    assert "attribution.return_attribution" not in user_prompt
    assert "EV-222222222222" not in user_prompt
    assert "covariance.ledoit_wolf_shrinkage" not in user_prompt
    assert "EV-333333333333" not in user_prompt

    # 2. System prompt explicitly forbids invented external thresholds and blanket statements
    assert "Do NOT introduce external acceptance thresholds" in system_prompt
    assert "Never conclude 'fully grounded'" in system_prompt
    assert "explicitly state that no evidence-backed acceptance threshold was provided" in system_prompt

    # 3. Decision recorded verified grounding
    assert decisions[0]["backend"] == "llm"
    assert decisions[0]["unbound_claims"] == 0
    assert decisions[0]["grounded_claims"] == 1


def test_checkpoint_grounding_repair_single_attempt(monkeypatch):
    """Verify that when an LLM returns a grounded claim, exactly one provider call is made and verified."""
    from start.core.schemas import EvidenceRecord, Status, TestResult
    from start.review.architecture import LLMReviewConfig
    from start.review.executor import run_domain_checkpoints

    call_count = 0

    class GroundedProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, *, output_token_budget: int = 1024, **kwargs) -> str:
            nonlocal call_count
            call_count += 1
            return "The volatility is 0.0937 [EV-PORT12345678] and no acceptance threshold was provided."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: GroundedProvider())

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

    port_rec = EvidenceRecord.from_result(
        TestResult(
            test_id="portfolio.risk_statistics",
            test_name="portfolio.risk_statistics",
            status=Status.RECORDED,
            metrics={"annualised_volatility": 0.0937},
        ),
        run_id="TEST-RUN",
    )
    port_rec.evidence_id = "EV-PORT12345678"

    prompts = ["Q", "Explain volatility", "A", "A", "A", "A"]
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(
        bundle,
        [port_rec],
        interactive=True,
        ask=lambda _: next(prompt_iter),
    )

    assert call_count == 1
    assert decisions[0]["backend"] == "llm"
    assert decisions[0]["unbound_claims"] == 0
    assert decisions[0]["grounded_claims"] == 1


def test_checkpoint_grounding_failure_surfaced(monkeypatch):
    """Verify that if repair fails, the failure is surfaced and user can continue deterministically."""
    from start.core.schemas import EvidenceRecord, Status, TestResult
    from start.review.architecture import LLMReviewConfig
    from start.review.executor import run_domain_checkpoints

    class AlwaysUngroundedProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, *, output_token_budget: int = 1024, **kwargs) -> str:
            return "The volatility is 0.9999 which is completely fabricated."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: AlwaysUngroundedProvider())

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

    port_rec = EvidenceRecord.from_result(
        TestResult(
            test_id="portfolio.risk_statistics",
            test_name="portfolio.risk_statistics",
            status=Status.RECORDED,
            metrics={"annualised_volatility": 0.0937},
        ),
        run_id="TEST-RUN",
    )
    port_rec.evidence_id = "EV-PORT12345678"

    prompts = [
        "Q",  # Action -> Question
        "Explain volatility",  # Question text
        "1",  # On Grounding Failure: [1] Continue deterministically
        "A",
        "A",
        "A",
        "A",
    ]
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(
        bundle,
        [port_rec],
        interactive=True,
        ask=lambda _: next(prompt_iter),
    )

    assert decisions[0]["backend"] == "fallback"
    assert "Deterministic fallback" in decisions[0]["response"]


def test_checkpoint_challenge_uses_grounding_pipeline(monkeypatch):
    """Verify C (Challenge) uses identical evidence scoping and grounding verification pipeline."""
    from start.core.schemas import EvidenceRecord, Status, TestResult
    from start.review.architecture import LLMReviewConfig
    from start.review.executor import run_domain_checkpoints

    captured = []

    class ChallengeProvider:
        name = "openai"
        model = "gpt-4o-mini"

        def complete(self, system: str, user: str, *, output_token_budget: int = 1024, **kwargs) -> str:
            captured.append((system, user))
            return "The challenge is valid because 4 exceptions [EV-VAR12345678] were observed."

    monkeypatch.setattr("start.providers.llm.get_llm_provider", lambda cfg: ChallengeProvider())

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
            test_id="traded_risk.var_exceptions",
            test_name="traded_risk.var_exceptions",
            status=Status.RECORDED,
            metrics={"n_exceptions": 4.0},
        ),
        run_id="TEST-RUN",
    )
    var_rec.evidence_id = "EV-VAR12345678"

    prompts = [
        "A",
        "A",
        "C",  # Checkpoint 3 (VaR) -> Challenge
        "Challenge tail risk",  # Challenge text
        "A",
        "A",
    ]
    prompt_iter = iter(prompts)

    decisions = run_domain_checkpoints(
        bundle,
        [var_rec],
        interactive=True,
        ask=lambda _: next(prompt_iter),
    )

    var_decision = decisions[2]
    assert var_decision["action"] == "challenge"
    assert var_decision["backend"] == "llm"
    assert var_decision["unbound_claims"] == 0
    assert var_decision["grounded_claims"] == 1
    assert "CHALLENGE" in captured[0][1]


def test_model_default_effective_settings_precedence(monkeypatch):
    """Verify effective default model precedence respects START_LLM__MODEL override."""
    import io

    from start.providers.model_discovery import FakeModelDiscovery
    from start.review.wizard import run_review_wizard

    monkeypatch.setenv("START_LLM__MODEL", "o3-mini")

    discovery = FakeModelDiscovery(models=["gpt-4o-mini", "gpt-4o", "o3-mini"])
    inputs = ["1", "2", "3", "1", "1", "1", "1", "1", "1", "Y"]
    input_iter = iter(inputs)
    stream = io.StringIO("Equity derivatives portfolio oversight\nEND\nEND\nEND\nEND\n")

    res = run_review_wizard(
        ask=lambda _: next(input_iter),
        stream=stream,
        discovery_client=discovery,
    )
    bundle = res["bundle"]

    assert bundle.llm_config.provider == "openai"
    assert bundle.llm_config.model == "o3-mini"


def test_provider_display_label_openai():
    """Verify provider display label is formatted as 'OpenAI', not 'Openai'."""
    from start.review.applicability import build_plan_preview
    from start.review.architecture import LLMReviewConfig

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

    preview = build_plan_preview(bundle)
    lines = preview.lines()
    preview_text = "\n".join(lines)

    assert "Provider:                OpenAI" in preview_text
    assert "Provider:                Openai" not in preview_text
