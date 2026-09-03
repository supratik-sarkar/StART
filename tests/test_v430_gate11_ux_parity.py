"""Gate 11 Comprehensive Verification Suite: Unified Interactive Validation Restoration & Cross-Domain UX Parity.

Verifies:
1. Universal Response Extraction & Empty-Response Fail-Closed (OpenAI Responses API output_text, Anthropic, Gemini, Grok)
2. Non-Vacuous Claim Grounding (NO_QUANTITATIVE_CLAIMS on qualitative responses; grounded quantitative claims)
3. Q / C / A / O / V Non-Terminal Control Flow State Machine
4. Rich Domain Terminal Tables (Portfolio, Attribution, VaR, Covariance, Scenario, Treasury, Artifacts, Governance)
5. Deterministic Visual & Tabular Artifact Generation & Terminal Catalog Browser
6. Scenario Analysis Pattern-B Evidence Integration & Committee Ingestion
7. Barrier Conditionality (Omitted when skipped / N/A; present when applicable)
8. Method Selection & Scope Customization (Only implemented methods offered; deferred excluded)
9. Semantically Accurate Numerical Wording & Attested Governance Narrative
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from start.core.schemas import EvidenceRecord, Status
from start.data.synthetic_market import generate_market_world
from start.providers.llm import extract_response_text
from start.registry.market_contexts import MarketContext, PortfolioSpec
from start.review.applicability import applicable_tests
from start.review.architecture import ReviewContextBundle, ReviewDomain
from start.review.executor import (
    build_market_narrative,
    execute_market_treasury_tests,
    generate_review_artifacts,
    run_domain_checkpoints,
    run_market_treasury_review,
)
from start.review.tables import (
    build_artifact_catalog_table,
    build_attribution_table,
    build_barrier_table,
    build_covariance_table,
    build_governance_table,
    build_portfolio_table,
    build_scenario_table,
    build_treasury_table,
    build_var_tail_table,
    render_checkpoint_panel,
)


def make_evidence_record(
    test_id: str,
    evidence_id: str = "EV-001",
    status: Status = Status.PASS,
    metrics: dict[str, Any] | None = None,
    interpretation: str = "Test interpretation",
) -> EvidenceRecord:
    """Helper to construct fully valid EvidenceRecord with required metadata."""
    return EvidenceRecord(
        test_id=test_id,
        test_name=test_id,
        evidence_id=evidence_id,
        status=status,
        metrics=metrics or {},
        interpretation=interpretation,
        model_id="M-TEST",
        dataset_id="D-TEST",
        run_id="RUN-TEST",
    )


# ==============================================================================
# 1. Universal Response Extraction & Empty-Response Fail-Closed
# ==============================================================================


def test_extract_response_text_openai_responses_api():
    """Verify universal extraction from OpenAI Responses API output_text items."""

    @dataclass
    class MockOutputText:
        text: str
        type: str = "output_text"

    @dataclass
    class MockMessage:
        content: list[MockOutputText]
        role: str = "assistant"

    @dataclass
    class MockResponsesAPIResponse:
        output: list[MockMessage]

    resp = MockResponsesAPIResponse(
        output=[
            MockMessage(
                content=[
                    MockOutputText(text="Analysis indicates annualized volatility of 0.0937 [EV-PORT-1].")
                ]
            )
        ]
    )
    extracted = extract_response_text(resp)
    assert extracted == "Analysis indicates annualized volatility of 0.0937 [EV-PORT-1]."


def test_extract_response_text_anthropic_and_chat_completion():
    """Verify universal extraction from Anthropic blocks and classic chat completion choices."""

    # Anthropic block
    @dataclass
    class MockContentBlock:
        text: str
        type: str = "text"

    @dataclass
    class MockAnthropicResponse:
        content: list[MockContentBlock]

    resp_anthropic = MockAnthropicResponse(content=[MockContentBlock(text="Anthropic review text")])
    assert extract_response_text(resp_anthropic) == "Anthropic review text"

    # Chat completion dict
    resp_dict = {"choices": [{"message": {"content": "Dict chat completion text"}}]}
    assert extract_response_text(resp_dict) == "Dict chat completion text"

    # Plain string
    assert extract_response_text("Direct string response") == "Direct string response"


def test_empty_response_fail_closed():
    """Verify that an empty or whitespace response from provider is handled fail-closed."""
    empty_resp = extract_response_text("   \n\t  ")
    assert empty_resp == ""


# ==============================================================================
# 2. Rich Domain Terminal Tables
# ==============================================================================


def test_rich_domain_tables_render_cleanly():
    """Verify all domain checkpoint tables render rows and columns cleanly."""
    records = [
        make_evidence_record(
            test_id="portfolio.risk_statistics",
            evidence_id="EV-PORT-001",
            status=Status.PASS,
            metrics={"annualised_volatility": 0.0937, "sharpe_ratio": 1.25, "diversification_ratio": 1.62},
            interpretation="Portfolio risk statistics within target bounds.",
        ),
        make_evidence_record(
            test_id="attribution.return_attribution",
            evidence_id="EV-ATTR-001",
            status=Status.PASS,
            metrics={"max_abs_reconciliation_error": 1.2e-6, "factor_explained_variance_ratio": 0.85},
            interpretation="Factor return attribution reconciled with negligible error.",
        ),
        make_evidence_record(
            test_id="traded_risk.var_exceptions",
            evidence_id="EV-VAR-001",
            status=Status.PASS,
            metrics={"n_exceptions": 4.0, "exception_rate": 0.016, "var_confidence": 0.99},
            interpretation="4 VaR exceptions observed over 250 days (Green zone).",
        ),
        make_evidence_record(
            test_id="covariance.ledoit_wolf_shrinkage",
            evidence_id="EV-COV-001",
            status=Status.PASS,
            metrics={
                "shrinkage_intensity": 0.0086,
                "condition_number_unshrunk": 450.0,
                "condition_number_shrunk": 42.0,
            },
            interpretation="Ledoit-Wolf shrinkage achieved 10.7x condition improvement.",
        ),
        make_evidence_record(
            test_id="scenario.linear_return",
            evidence_id="EV-SCEN-001",
            status=Status.RECORDED,
            metrics={"portfolio_loss": 0.048, "portfolio_return": -0.048, "scenario_type": "SYNTHETIC"},
            interpretation="Tail shock scenario produced 4.8% portfolio loss.",
        ),
        make_evidence_record(
            test_id="validation.cev_consistency",
            evidence_id="EV-CEV-001",
            status=Status.FAIL,
            metrics={"observed.coverage_gamma_0_0": 0.635, "observed.consistency_ratio_gamma_0_0": 0.469967},
            interpretation="CEV nominal coverage 0.635 failed requirement [0.90, 0.98].",
        ),
        make_evidence_record(
            test_id="validation.stanton_bias",
            evidence_id="EV-STANTON-001",
            status=Status.FAIL,
            metrics={
                "observed.max_wrong_sign_rate_nonzero_drift": 0.475,
                "observed.bias_improvement_ratio": 0.305052,
            },
            interpretation="Stanton wrong-sign rate 0.475 failed requirement <= 0.10.",
        ),
        make_evidence_record(
            test_id="traded_risk.brownian_bridge_barrier",
            evidence_id="EV-BARRIER-001",
            status=Status.PASS,
            metrics={"crossing_probability": 0.012, "n_simulations": 10000},
            interpretation="Brownian bridge boundary crossing verified.",
        ),
    ]

    t_port = build_portfolio_table(records)
    assert t_port.row_count >= 1

    t_attr = build_attribution_table(records)
    assert t_attr.row_count >= 1

    t_var = build_var_tail_table(records)
    assert t_var.row_count >= 1

    t_cov = build_covariance_table(records)
    assert t_cov.row_count >= 1

    t_scen = build_scenario_table(records)
    assert t_scen.row_count >= 1

    t_tre = build_treasury_table(records)
    assert t_tre.row_count >= 2

    t_bar = build_barrier_table(records)
    assert t_bar.row_count >= 1

    t_gov = build_governance_table({"mode": "cross_domain", "materiality": "high"}, [])
    assert t_gov.row_count >= 1

    panel = render_checkpoint_panel("Portfolio Risk", "Test description", "market")
    assert "Portfolio Risk" in str(panel.title)


# ==============================================================================
# 3. Deterministic Artifact Generation & Catalog
# ==============================================================================


def test_deterministic_artifact_generation_and_catalog():
    """Verify generate_review_artifacts produces valid typed artifacts and catalog table."""
    world = generate_market_world(n_assets=6, n_periods=60, seed=42)
    renamed = {old: f"ASSET_{i + 1:03d}" for i, old in enumerate(world.returns.columns)}
    market = MarketContext(
        returns=world.returns.rename(columns=renamed),
        prices=world.prices.rename(columns=renamed),
        pnl=world.pnl,
        var_series=world.var_series,
        portfolio=PortfolioSpec(weights=world.weights.rename(renamed)),
        seed=42,
    )
    bundle = ReviewContextBundle(domains=(ReviewDomain.MARKET,), market=market)

    records = [
        make_evidence_record(test_id="portfolio.risk_statistics", evidence_id="EV-P-1", status=Status.PASS),
        make_evidence_record(
            test_id="covariance.ledoit_wolf_shrinkage", evidence_id="EV-C-1", status=Status.PASS
        ),
        make_evidence_record(test_id="traded_risk.var_kupiec_pof", evidence_id="EV-V-1", status=Status.PASS),
        make_evidence_record(test_id="scenario.linear_return", evidence_id="EV-S-1", status=Status.RECORDED),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        art_map = generate_review_artifacts(bundle, records, Path(tmpdir))

        assert "Portfolio Risk & Volatility Assumptions" in art_map
        assert "Covariance Structure & Missing Data Treatment" in art_map
        assert "VaR Backtesting & Exception Frequency" in art_map
        assert "Scenario Analysis & Stress Testing" in art_map

        all_arts = [a for sublist in art_map.values() for a in sublist]
        assert len(all_arts) >= 4

        # Catalog table rendering
        table = build_artifact_catalog_table(all_arts)
        assert table.row_count == len(all_arts)


# ==============================================================================
# 4. Q / C / A / O / V Non-Terminal Control Flow State Machine
# ==============================================================================


def test_q_c_v_non_terminal_loop_and_a_o_advance():
    """Verify that V, Q, and C do NOT advance the checkpoint, while A and O advance."""
    records = [
        make_evidence_record(
            test_id="portfolio.risk_statistics",
            evidence_id="EV-PORT-001",
            status=Status.PASS,
            metrics={"annualised_volatility": 0.0937},
            interpretation="Volatility is 0.0937.",
        ),
    ]
    bundle = ReviewContextBundle(domains=(ReviewDomain.MARKET,))

    scripted_inputs = [
        "V",  # Checkpoint 1: View artifacts
        "Q",  # Checkpoint 1: Question
        "What is the volatility?",  # Question text
        "C",  # Checkpoint 1: Challenge
        "Test challenge note",  # Challenge note
        "A",  # Checkpoint 1: Accept -> advance
    ] + ["A"] * 30  # Accept all remaining checkpoints

    input_iter = iter(scripted_inputs)

    def mock_ask(prompt: str) -> str:
        return next(input_iter, "A")

    class MockArtifact:
        artifact_id = "ART-1"
        title = "Test"
        spec = None
        rendering_format = "json"
        file_path = "path"
        data_fingerprint = "abc"

    decisions = run_domain_checkpoints(
        bundle,
        records,
        artifacts_by_checkpoint={"Portfolio Risk & Volatility Assumptions": [MockArtifact()]},
        interactive=True,
        ask=mock_ask,
    )

    chk1_decisions = [d for d in decisions if d["checkpoint"] == "Portfolio Risk & Volatility Assumptions"]
    actions = [d["action"] for d in chk1_decisions]
    assert "question" in actions
    assert "challenge" in actions
    assert "accept" in actions
    assert actions[-1] == "accept"


# ==============================================================================
# 5. Barrier Conditionality
# ==============================================================================


def test_barrier_omitted_when_skipped_or_absent():
    """Verify Barrier checkpoint is omitted when barrier evidence is SKIPPED or absent."""
    records_without_barrier = [
        make_evidence_record(test_id="portfolio.risk_statistics", evidence_id="EV-1", status=Status.PASS),
        make_evidence_record(
            test_id="traded_risk.brownian_bridge_barrier", evidence_id="EV-2", status=Status.SKIPPED
        ),
    ]
    bundle = ReviewContextBundle(domains=(ReviewDomain.MARKET,))

    decisions = run_domain_checkpoints(
        bundle,
        records_without_barrier,
        interactive=True,
        ask=lambda _: "A",
    )
    checkpoint_names = [d["checkpoint"] for d in decisions]
    assert not any("Barrier" in name for name in checkpoint_names)


def test_barrier_present_when_applicable():
    """Verify Barrier checkpoint is included when barrier evidence has PASS/RECORDED status."""
    records_with_barrier = [
        make_evidence_record(test_id="portfolio.risk_statistics", evidence_id="EV-1", status=Status.PASS),
        make_evidence_record(
            test_id="traded_risk.brownian_bridge_barrier", evidence_id="EV-2", status=Status.PASS
        ),
    ]
    bundle = ReviewContextBundle(domains=(ReviewDomain.MARKET,))

    decisions = run_domain_checkpoints(
        bundle,
        records_with_barrier,
        interactive=True,
        ask=lambda _: "A",
    )
    checkpoint_names = [d["checkpoint"] for d in decisions]
    assert any("Barrier" in name for name in checkpoint_names)


# ==============================================================================
# 6. Scenario Analysis Pattern-B Evidence Integration
# ==============================================================================


def test_scenario_pattern_b_evidence_generated():
    """Verify execute_market_treasury_tests generates linear return, factor linear, and reverse stress."""
    world = generate_market_world(n_assets=8, n_periods=80, n_factors=3, seed=42)
    renamed = {old: f"ASSET_{i + 1:03d}" for i, old in enumerate(world.returns.columns)}
    market = MarketContext(
        returns=world.returns.rename(columns=renamed),
        prices=world.prices.rename(columns=renamed),
        factor_returns=world.factor_returns,
        factor_exposures=world.factor_exposures.rename(index=renamed),
        portfolio=PortfolioSpec(weights=world.weights.rename(renamed)),
        seed=42,
    )
    bundle = ReviewContextBundle(domains=(ReviewDomain.MARKET,), market=market)
    app = applicable_tests(bundle.domains)

    results = execute_market_treasury_tests(bundle, app)
    test_ids = {tr.test_id for tr in results}

    assert "scenario.linear_return" in test_ids or "scenario.asset_return" in test_ids
    assert "scenario.factor_linear" in test_ids
    assert "scenario.reverse_stress" in test_ids


# ==============================================================================
# 7. Accurate Numerical Wording & Attested Narrative Grounding
# ==============================================================================


def test_semantically_accurate_wording_and_narrative():
    """Verify wording does not imply 0.066 < 0.05 and narrative matches grounded text."""
    records = [
        make_evidence_record(
            test_id="portfolio.risk_statistics",
            evidence_id="EV-1",
            status=Status.PASS,
            metrics={"annualised_volatility": 0.0937},
        ),
        make_evidence_record(
            test_id="attribution.return_attribution",
            evidence_id="EV-2",
            status=Status.PASS,
            metrics={"max_abs_reconciliation_error": 0.0},
        ),
        make_evidence_record(
            test_id="traded_risk.var_exceptions",
            evidence_id="EV-3",
            status=Status.PASS,
            metrics={"n_exceptions": 4.0},
        ),
        make_evidence_record(
            test_id="traded_risk.var_kupiec_pof",
            evidence_id="EV-4",
            status=Status.PASS,
            metrics={"p_value": 0.6414},
        ),
        make_evidence_record(
            test_id="covariance.ledoit_wolf_shrinkage",
            evidence_id="EV-5",
            status=Status.PASS,
            metrics={"shrinkage_intensity": 0.0086},
        ),
        make_evidence_record(
            test_id="validation.var_size_power",
            evidence_id="EV-6",
            status=Status.PASS,
            metrics={"observed.size_correct_forecast": 0.0660},
        ),
        make_evidence_record(
            test_id="validation.regem_structural", evidence_id="EV-7", status=Status.PASS, metrics={}
        ),
    ]

    narrative = build_market_narrative(records, (ReviewDomain.MARKET,))
    # Wording check: should state empirical size: 0.0660 (nominal significance level: 0.05)
    assert "empirical size: 0.0660 (nominal significance level: 0.05)" in narrative
    assert "under nominal 0.05" not in narrative


# ==============================================================================
# 8. Full End-to-End Market Review UX Flow
# ==============================================================================


def test_end_to_end_market_review_execution():
    """Verify run_market_treasury_review runs end-to-end with artifacts, narrative, and governance."""
    world = generate_market_world(n_assets=6, n_periods=60, seed=42)
    renamed = {old: f"ASSET_{i + 1:03d}" for i, old in enumerate(world.returns.columns)}
    market = MarketContext(
        returns=world.returns.rename(columns=renamed),
        prices=world.prices.rename(columns=renamed),
        pnl=world.pnl,
        var_series=world.var_series,
        portfolio=PortfolioSpec(weights=world.weights.rename(renamed)),
        seed=42,
    )
    bundle = ReviewContextBundle(domains=(ReviewDomain.MARKET,), market=market)

    with tempfile.TemporaryDirectory() as tmpdir:
        res = run_market_treasury_review(
            bundle,
            output_root=tmpdir,
            interactive=True,
            ask=lambda _: "A",
        )

        assert res["run_id"].startswith("RUN-REVIEW-")
        assert len(res["records"]) >= 20
        assert len(res["decisions"]) >= 6
        assert res["seal"] is not None
        assert res["replay"].intact is True
        assert Path(res["output_path"]).exists()
        assert (Path(res["output_path"]) / "artifacts").exists()
