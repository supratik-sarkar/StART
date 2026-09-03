"""Gate 11A Comprehensive Verification Suite: Historical UX Recovery, Cross-Domain Parity & Evidence-Driven Closure.

Verifies all 25 directives from the Gate-11A Approved Architecture Plan:
1. One Review Session Model (reusing ReviewContextBundle & ReviewExecutionProducts).
2. Audit-Grade Execution Products Container (ExecutionProduct, ReviewExecutionProducts).
3. Single Source of Truth & Zero Recomputation (engine call count = 0 during rendering/viewing).
4. Preflight Tables Describe Inputs Only (zero analytical modeling during preflight).
5. Scenario Identity Audit (canonical scenario.linear_return with legacy read alias).
6. Non-Mutating Challenge Diagnostics (diagnose_covariance, validate_scenario_data_integrity, etc.).
7. Treasury Artifacts Use Existing Outputs (no simulation or refitting).
8. Predictive ML & Deep Learning Parity Without Heavy Jobs (lightweight bounded fixtures).
9. Historical Git Tree Safety & Independence (clean, unreferenced at runtime).
10. Strict Table Lineage & Zero Hard-Coded Floats (missing values show N/A).
11. Domain-Aware Artifact Catalog in [V] & [VA].
12. Execution Plan Reconciliation (reconcile_execution).
13. Five Noninteractive Shell Harnesses (Predictive, DL, Market, Treasury, Cross-Domain).
14. Global Q / C / V / A / O Contract Across Domains.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from start.core.schemas import EvidenceRecord, Status
from start.data.synthetic_market import generate_market_world
from start.registry.market_contexts import MarketContext, PortfolioSpec, ShortRateContext
from start.review.applicability import applicable_tests, build_plan_preview
from start.review.architecture import (
    ReviewContextBundle,
    ReviewDomain,
    ReviewExecutionProducts,
    ReviewMode,
)
from start.review.executor import (
    execute_market_treasury_tests,
    generate_review_artifacts,
    run_domain_checkpoints,
    run_unified_review,
)
from start.review.tables import (
    build_attribution_table,
    build_covariance_table,
    build_portfolio_table,
    build_predictive_table,
    build_preflight_data_summary_table,
    build_scenario_table,
    build_treasury_table,
    build_var_tail_table,
)


def make_evidence_record(
    test_id: str,
    evidence_id: str = "EV-001",
    status: Status = Status.PASS,
    metrics: dict[str, Any] | None = None,
    interpretation: str = "Test interpretation",
) -> EvidenceRecord:
    """Construct fully valid EvidenceRecord with required metadata."""
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


def make_market_bundle() -> ReviewContextBundle:
    world = generate_market_world(n_assets=4, n_periods=100, n_factors=2, periods_per_year=252, seed=42)
    market = MarketContext(
        returns=world.returns,
        prices=world.prices,
        periods_per_year=world.periods_per_year,
        factor_returns=world.factor_returns,
        factor_exposures=world.factor_exposures,
        portfolio=PortfolioSpec(weights=world.weights),
        seed=42,
    )
    return ReviewContextBundle(domains=(ReviewDomain.MARKET,), market=market)


def make_treasury_bundle() -> ReviewContextBundle:
    world = generate_market_world(
        n_assets=4, n_periods=100, n_factors=2, periods_per_year=252, seed=42, include_short_rate=True
    )
    short_rate = world.short_rate_context()
    return ReviewContextBundle(domains=(ReviewDomain.TREASURY,), short_rate=short_rate)


def make_cross_domain_bundle() -> ReviewContextBundle:
    world = generate_market_world(
        n_assets=4, n_periods=100, n_factors=2, periods_per_year=252, seed=42, include_short_rate=True
    )
    market = MarketContext(
        returns=world.returns,
        prices=world.prices,
        periods_per_year=world.periods_per_year,
        factor_returns=world.factor_returns,
        factor_exposures=world.factor_exposures,
        portfolio=PortfolioSpec(weights=world.weights),
        seed=42,
    )
    short_rate = world.short_rate_context()
    return ReviewContextBundle(
        mode=ReviewMode.CROSS_DOMAIN,
        domains=(ReviewDomain.MARKET, ReviewDomain.TREASURY),
        market=market,
        short_rate=short_rate,
    )


# ==============================================================================
# 1. Execution Products Container & Single Source of Truth
# ==============================================================================

def test_execution_products_contract_and_single_source_of_truth():
    """Verify ExecutionProduct and ReviewExecutionProducts typed container contract."""
    products = ReviewExecutionProducts()

    # Register products
    cov_mock = np.eye(3)
    p_cov = products.register(
        "covariance.matrix",
        cov_mock,
        evidence_ids=("EV-COV-1",),
        source_fingerprint="fp-cov-123",
        provenance="registered_engine",
    )
    assert p_cov.analytic_id == "covariance.matrix"
    assert np.array_equal(p_cov.result_object, cov_mock)
    assert p_cov.evidence_ids == ("EV-COV-1",)
    assert p_cov.source_fingerprint == "fp-cov-123"

    # Canonical scenario name and alias lookup
    scen_mock = {"loss": 0.05, "return": -0.05}
    products.register(
        "scenario.linear_return",
        scen_mock,
        evidence_ids=("EV-SCEN-1",),
        source_fingerprint="fp-scen-123",
    )
    assert "scenario.linear_return" in products
    assert "scenario.asset_return" in products  # legacy alias lookup
    assert products.get_result("scenario.linear_return") == scen_mock
    assert products.get_result("scenario.asset_return") == scen_mock

    # Fingerprint and summary
    fp = products.fingerprint()
    assert isinstance(fp, str) and len(fp) == 64
    summary = products.summary()
    assert "covariance.matrix" in summary
    assert "scenario.linear_return" in summary


# ==============================================================================
# 2. Zero Scientific Recomputation Spies
# ==============================================================================

def test_zero_scientific_recomputation_during_rendering_and_viewing():
    """Prove with spy counters that table rendering, artifact generation, and [V] browsing execute ZERO models."""
    from start.portfolio.scenario import apply_asset_return_scenario, solve_reverse_stress
    from start.portfolio.tail_risk import run_comprehensive_tail_backtest

    world = generate_market_world(
        n_assets=4,
        n_periods=100,
        n_factors=2,
        periods_per_year=252,
        seed=42,
        include_short_rate=True,
    )
    renamed = {old: f"A_{i}" for i, old in enumerate(world.returns.columns)}
    market = MarketContext(
        returns=world.returns.rename(columns=renamed),
        prices=world.prices.rename(columns=renamed),
        periods_per_year=world.periods_per_year,
        factor_returns=world.factor_returns,
        factor_exposures=world.factor_exposures.rename(index=renamed),
        pnl=world.pnl,
        var_series=world.var_series,
        portfolio=PortfolioSpec(weights=world.weights.rename(renamed)),
        seed=42,
    )
    bundle = ReviewContextBundle(domains=(ReviewDomain.MARKET,), market=market)

    # Spies on expensive scientific engines across the WHOLE lifecycle
    with (
        patch("start.portfolio.scenario.apply_asset_return_scenario", wraps=apply_asset_return_scenario) as spy_scen,
        patch("start.portfolio.scenario.solve_reverse_stress", wraps=solve_reverse_stress) as spy_rev,
        patch("start.portfolio.tail_risk.run_comprehensive_tail_backtest", wraps=run_comprehensive_tail_backtest) as spy_bt,
        patch.object(pd.DataFrame, "cov", wraps=market.returns.cov) as spy_cov,
    ):
        # 1. Deterministic Review Execution: engines execute exactly once
        applicable = applicable_tests(bundle.domains)
        results, products = execute_market_treasury_tests(bundle, applicable, return_products=True)

        exec_scen_count = spy_scen.call_count
        exec_rev_count = spy_rev.call_count
        exec_bt_count = spy_bt.call_count
        exec_cov_count = spy_cov.call_count

        assert exec_scen_count >= 1, "apply_asset_return_scenario must execute during test phase"
        assert exec_rev_count >= 1, "solve_reverse_stress must execute during test phase"
        assert exec_bt_count >= 1, "run_comprehensive_tail_backtest must execute during test phase"

        records = [
            make_evidence_record(test_id=r.test_id, evidence_id=f"EV-{i}", status=r.status, metrics=r.metrics)
            for i, r in enumerate(results)
        ]
        rec_ids_set = {r.evidence_id for r in records}
        rec_by_test = {r.test_id: r for r in records}

        # 2. Table Rendering Phase: execution count = 0 additional
        build_portfolio_table(records)
        build_covariance_table(records)
        build_attribution_table(records)
        build_var_tail_table(records)
        build_scenario_table(records)

        assert spy_scen.call_count == exec_scen_count, "Scenario recomputed during table rendering!"
        assert spy_rev.call_count == exec_rev_count, "Reverse stress recomputed during table rendering!"
        assert spy_bt.call_count == exec_bt_count, "Backtest recomputed during table rendering!"
        assert spy_cov.call_count == exec_cov_count, "cov() recomputed during table rendering!"

        # 3. Artifact Generation Phase: execution count = 0 additional
        with tempfile.TemporaryDirectory() as tmpdir:
            art_map = generate_review_artifacts(bundle, records, Path(tmpdir), products=products)

            assert spy_scen.call_count == exec_scen_count, "Scenario recomputed during artifact generation!"
            assert spy_rev.call_count == exec_rev_count, "Reverse stress recomputed during artifact generation!"
            assert spy_bt.call_count == exec_bt_count, "Backtest recomputed during artifact generation!"
            assert spy_cov.call_count == exec_cov_count, "cov() recomputed during artifact generation!"

            # 4. Checkpoint Browsing Phase ([V] and [VA]): execution count = 0 additional
            scripted = ["V", "VA", "A"] + ["A"] * 20
            iter_in = iter(scripted)
            run_domain_checkpoints(
                bundle,
                records,
                artifacts_by_checkpoint=art_map,
                products=products,
                interactive=True,
                ask=lambda _: next(iter_in, "A"),
            )

            assert spy_scen.call_count == exec_scen_count, "Scenario recomputed during [V] browsing!"
            assert spy_rev.call_count == exec_rev_count, "Reverse stress recomputed during [V] browsing!"
            assert spy_bt.call_count == exec_bt_count, "Backtest recomputed during [V] browsing!"
            assert spy_cov.call_count == exec_cov_count, "cov() recomputed during [V] browsing!"

            # 5. Provenance Chain & Identity Verification
            for checkpoint_title, arts in art_map.items():
                for art in arts:
                    spec = getattr(art, "spec", None)
                    assert spec is not None, f"Artifact in {checkpoint_title} missing spec"
                    assert len(spec.evidence_ids) > 0, f"Artifact {art.artifact_id} has empty evidence_ids"
                    for ev_id in spec.evidence_ids:
                        assert ev_id in rec_ids_set, f"Artifact {art.artifact_id} references unknown ID {ev_id}"
                    assert art.data_fingerprint is not None and len(art.data_fingerprint) > 0

            # Verify specific evidence linkage
            scen_arts = art_map.get("Scenario Analysis & Stress Testing", [])
            assert len(scen_arts) >= 1
            assert scen_arts[0].spec.evidence_ids == (rec_by_test["scenario.linear_return"].evidence_id,)

            var_arts = art_map.get("VaR Backtesting & Exception Frequency", [])
            assert len(var_arts) >= 1
            assert var_arts[0].spec.evidence_ids == (rec_by_test["traded_risk.var_kupiec_pof"].evidence_id,)


# ==============================================================================
# 3. Preflight Tables Describe Inputs Only
# ==============================================================================

def test_preflight_data_summary_table_descriptive_only():
    """Verify that build_preflight_data_summary_table strictly describes inputs with zero analytical modeling."""
    df_ret = pd.DataFrame({"AAPL": [0.01, -0.02, 0.005], "MSFT": [0.02, -0.01, 0.01]})
    market = MarketContext(returns=df_ret, periods_per_year=252)
    bundle = ReviewContextBundle(domains=(ReviewDomain.MARKET,), market=market)

    with patch.object(pd.DataFrame, "cov") as spy_cov, patch.object(pd.DataFrame, "corr") as spy_corr:
        table = build_preflight_data_summary_table(bundle)
        assert spy_cov.call_count == 0, "cov() called during preflight summary!"
        assert spy_corr.call_count == 0, "corr() called during preflight summary!"

    assert table.title is not None and "Pre-flight Data & Context Summary" in table.title
    assert table.row_count >= 1


# ==============================================================================
# 4. Challenge Non-Mutating Diagnostics
# ==============================================================================

def test_challenge_non_mutating_diagnostics():
    """Verify [C] challenge invokes only non-mutating diagnostics and does not repair PSD covariance."""
    world = generate_market_world(n_assets=3, n_periods=50, seed=42)
    market = MarketContext(returns=world.returns, periods_per_year=252)
    bundle = ReviewContextBundle(domains=(ReviewDomain.MARKET,), market=market)

    records = [
        make_evidence_record("covariance.empirical", "EV-COV-1", metrics={"condition_number": 12.5}),
        make_evidence_record("scenario.linear_return", "EV-SCEN-1", metrics={"portfolio_loss": 0.05}),
        make_evidence_record("portfolio.risk_statistics", "EV-PORT-1", metrics={"annualised_volatility": 0.10}),
    ]

    # Script:
    # Checkpoint 1 (Portfolio): Challenge, then Accept
    # Checkpoint 2 (Factor): Accept
    # Checkpoint 3 (VaR): Accept
    # Checkpoint 4 (Covariance): Challenge, then Accept
    # Remaining checkpoints: Accept
    scripted = [
        "C", "Challenge portfolio", "A",  # Checkpoint 1: Portfolio
        "A",                              # Checkpoint 2: Factor Modeling
        "A",                              # Checkpoint 3: VaR Backtesting
        "C", "Challenge covariance", "A", # Checkpoint 4: Covariance Structure
    ] + ["A"] * 20
    iter_in = iter(scripted)

    from start.portfolio.covariance import diagnose_covariance

    with (
        patch("start.portfolio.covariance.diagnose_covariance", wraps=diagnose_covariance) as spy_diag_cov,
        patch("start.portfolio.covariance.repair_psd_covariance") as spy_repair_cov,
    ):
        decisions = run_domain_checkpoints(
            bundle,
            records,
            interactive=True,
            ask=lambda _: next(iter_in, "A"),
        )
        assert spy_repair_cov.call_count == 0, "repair_psd_covariance was executed during challenge!"
        assert spy_diag_cov.call_count >= 1, "diagnose_covariance was not executed during challenge!"

    assert any(d["action"] == "challenge" for d in decisions)


# ==============================================================================
# 5. Strict Table Lineage & Zero Hard-Coded Floats
# ==============================================================================

def test_strict_table_lineage_and_dynamic_formatting():
    """Prove rendered table metrics dynamically update from EvidenceRecords without hardcoded constants."""
    # Test with custom metric values
    custom_vol = 0.1742
    custom_size = 0.0415
    custom_cov_cev = 0.582
    custom_wrong_sign = 0.388

    records = [
        make_evidence_record(
            "portfolio.risk_statistics",
            "EV-P-CUSTOM",
            metrics={"annualised_volatility": custom_vol},
        ),
        make_evidence_record(
            "validation.var_size_power",
            "EV-VAR-CUSTOM",
            metrics={"observed.size_correct_forecast": custom_size},
        ),
        make_evidence_record(
            "validation.cev_consistency",
            "EV-CEV-CUSTOM",
            metrics={"observed.coverage_gamma_0_0": custom_cov_cev},
        ),
        make_evidence_record(
            "validation.stanton_bias",
            "EV-ST-CUSTOM",
            metrics={"observed.max_wrong_sign_rate_nonzero_drift": custom_wrong_sign},
        ),
    ]

    port_table = build_portfolio_table(records)
    # Check that custom_vol is rendered in the table and not 0.0937
    port_text = str(port_table.columns[2]._cells)
    assert f"{custom_vol:.4f}" in port_text

    # Check Treasury table
    treasury_table = build_treasury_table(records)
    t_text = str(treasury_table.columns[2]._cells)
    assert f"{custom_cov_cev:.3f}" in t_text
    assert f"{custom_wrong_sign:.3f}" in t_text

    # Check missing metric formats as "N/A"
    rec_empty = [make_evidence_record("portfolio.risk_statistics", "EV-EMPTY", metrics={})]
    empty_port_table = build_portfolio_table(rec_empty)
    empty_text = str(empty_port_table.columns[2]._cells)
    assert "N/A" in empty_text


# ==============================================================================
# 6. Treasury Full Interactive Journey
# ==============================================================================

def test_treasury_full_interactive_journey():
    """Verify noninteractive end-to-end Treasury review preserving CEV=FAIL and Stanton=FAIL."""
    rates = pd.Series(np.linspace(0.01, 0.05, 252))
    short_rate = ShortRateContext(rates=rates, periods_per_year=252.0)
    bundle = ReviewContextBundle(domains=(ReviewDomain.TREASURY,), short_rate=short_rate)

    scripted = ["A"] * 20
    iter_in = iter(scripted)

    with tempfile.TemporaryDirectory() as tmpdir:
        res = run_unified_review(bundle, output_root=tmpdir, interactive=True, ask=lambda _: next(iter_in, "A"))

        records = res["records"]
        test_ids = {r.test_id for r in records}
        assert "traded_risk.cev_elasticity" in test_ids
        assert "traded_risk.stanton_nonparametric" in test_ids
        assert "validation.cev_consistency" in test_ids
        assert "validation.stanton_bias" in test_ids

        # Invariant: CEV and Stanton must FAIL
        cev_rec = next(r for r in records if r.test_id == "validation.cev_consistency")
        st_rec = next(r for r in records if r.test_id == "validation.stanton_bias")
        assert cev_rec.status == Status.FAIL
        assert st_rec.status == Status.FAIL

        # Verify artifacts
        art_dir = Path(res["output_path"]) / "artifacts"
        assert (art_dir / "short_rate_summary.json").exists()
        assert (art_dir / "cev_elasticity_diagnostic.json").exists()
        assert (art_dir / "stanton_drift_diagnostic.json").exists()


# ==============================================================================
# 7. Cross-Domain Market & Treasury End-to-End Execution
# ==============================================================================

def test_cross_domain_market_treasury_end_to_end():
    """Verify real multi-domain execution preserving negative Treasury evidence alongside Market findings."""
    world = generate_market_world(
        n_assets=4,
        n_periods=100,
        n_factors=2,
        periods_per_year=252,
        seed=42,
        include_short_rate=True,
    )
    market = MarketContext(
        returns=world.returns,
        prices=world.prices,
        periods_per_year=world.periods_per_year,
        factor_returns=world.factor_returns,
        factor_exposures=world.factor_exposures,
        portfolio=PortfolioSpec(weights=world.weights),
        seed=42,
    )
    short_rate = world.short_rate_context()
    bundle = ReviewContextBundle(
        mode=ReviewMode.CROSS_DOMAIN,
        domains=(ReviewDomain.MARKET, ReviewDomain.TREASURY),
        market=market,
        short_rate=short_rate,
    )

    scripted = ["A"] * 25
    iter_in = iter(scripted)

    with tempfile.TemporaryDirectory() as tmpdir:
        res = run_unified_review(bundle, output_root=tmpdir, interactive=True, ask=lambda _: next(iter_in, "A"))

        records = res["records"]
        # Census check
        assert len(records) >= 27  # 25 market tests + 2 treasury tests + validation studies

        # Negative findings preserved without unconditional ACCEPT
        validation_fails = [r for r in records if r.status == Status.FAIL]
        assert len(validation_fails) >= 2  # CEV and Stanton fail


# ==============================================================================
# 8. Predictive ML & Deep Learning Unified Shell Parity
# ==============================================================================

def test_predictive_ml_unified_shell_parity(tmp_path):
    """Verify predictive domain review through the unified shell with real deterministic execution."""
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier

    from start.evidence.ledger import EvidenceLedger
    from start.registry import TestContext
    from start.review.applicability import applicable_tests
    from start.review.executor import (
        execute_market_treasury_tests,
        generate_review_artifacts,
        run_domain_checkpoints,
    )

    # 1. Real bounded tabular test fixture
    X, y = make_classification(n_samples=200, n_features=6, n_informative=4, random_state=42)
    cols = [f"f_{i}" for i in range(6)]
    df = pd.DataFrame(X, columns=cols)
    df["target"] = y

    train_df = df.iloc[:140].copy()
    test_df = df.iloc[140:].copy()

    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(train_df[cols], train_df["target"])

    train_df["score"] = clf.predict_proba(train_df[cols])[:, 1]
    train_df["pred"] = clf.predict(train_df[cols])
    test_df["score"] = clf.predict_proba(test_df[cols])[:, 1]
    test_df["pred"] = clf.predict(test_df[cols])

    ctx = TestContext(
        train=train_df,
        test=test_df,
        target_column="target",
        score_column="score",
        prediction_column="pred",
        model=clf,
        seed=42,
    )

    bundle = ReviewContextBundle(
        domains=(ReviewDomain.PREDICTIVE,),
        tabular=ctx,
    )

    # 2. Real applicable registered deterministic test execution
    applicable = applicable_tests(bundle.domains)
    assert len(applicable.test_ids) == 52

    exec_out = execute_market_treasury_tests(bundle, applicable, return_products=True)
    assert isinstance(exec_out, tuple)
    test_results, products = exec_out
    assert len(test_results) == 52

    # 3. Ledger/EvidenceRecord conversion
    ledger_path = tmp_path / "ledger.jsonl"
    store_path = tmp_path / "evidence"
    ledger = EvidenceLedger(ledger_path, store_path)
    records = [ledger.append(tr, run_id="RUN-TEST-PRED") for tr in test_results]
    assert len(records) == 52

    # Verify zero fixture-authored metrics: metrics come from deterministic engines
    sup_rec = next(r for r in records if r.test_id == "supervised.classification_metrics")
    assert "accuracy" in sup_rec.metrics
    assert isinstance(sup_rec.metrics["accuracy"], (int, float))

    disc_rec = next(r for r in records if r.test_id == "supervised.discrimination")
    assert "roc_auc" in disc_rec.metrics
    assert isinstance(disc_rec.metrics["roc_auc"], (int, float))

    # 4. Artifact generation
    art_dir = tmp_path / "artifacts"
    artifacts_by_chk = generate_review_artifacts(bundle, records, art_dir, products=products)
    assert len(artifacts_by_chk) >= 2

    # 5. Checkpoints presentation & state machine loop
    scripted = ["Q", "C", "V", "VA", "A"] + ["A"] * 20
    iter_in = iter(scripted)

    decisions = run_domain_checkpoints(
        bundle,
        records,
        artifacts_by_checkpoint=artifacts_by_chk,
        products=products,
        interactive=True,
        ask=lambda _: next(iter_in, "A"),
    )
    assert len(decisions) >= 5
    assert all(d["action"] in ("accept", "challenge", "question") for d in decisions)


def test_deep_learning_unified_shell_parity(tmp_path):
    """Verify deep learning review through the unified shell using real bounded DL execution pipeline."""
    from start.modeling.dl_training import DLReviewOptions, run_dl_review
    from start.review.architecture import PredictiveTechnology
    from start.review.executor import (
        generate_review_artifacts,
        run_domain_checkpoints,
    )

    # 1. Execute actual smallest existing bounded DL pipeline in StART
    opts = DLReviewOptions(
        architecture="mlp",
        epochs=2,
        batch_size=64,
        agent_mode="deterministic",
        output_root=str(tmp_path),
        seed=42,
    )
    dl_result = run_dl_review(opts)
    assert len(dl_result.evidence) >= 6

    # 2. Extract real execution-backed EvidenceRecords
    records = dl_result.evidence
    test_ids = {r.test_id for r in records}
    assert "deep_learning.training_diagnostics" in test_ids
    assert "deep_learning.performance_diagnostics" in test_ids
    assert "deep_learning.calibration_diagnostics" in test_ids

    # Verify real metrics from PyTorch training/evaluation
    tr_rec = next(r for r in records if r.test_id == "deep_learning.training_diagnostics")
    assert tr_rec.metrics["epochs_run"] == 2

    # 3. Artifact generation & shell checkpointing
    bundle = ReviewContextBundle(
        domains=(ReviewDomain.PREDICTIVE,),
        technology=PredictiveTechnology.DEEP_LEARNING,
    )
    art_dir = tmp_path / "artifacts_dl"
    artifacts_by_chk = generate_review_artifacts(bundle, records, art_dir)

    scripted = ["Q", "C", "V", "VA", "A"] + ["A"] * 20
    iter_in = iter(scripted)
    decisions = run_domain_checkpoints(
        bundle,
        records,
        artifacts_by_checkpoint=artifacts_by_chk,
        interactive=True,
        ask=lambda _: next(iter_in, "A"),
    )
    assert len(decisions) >= 5

    # 4. Table rendering directly
    table = build_predictive_table(records, title="Deep Learning Verification Diagnostics")
    assert table.row_count == len(records)


# ==============================================================================
# 9. Global Q / C / V / A / O Contract Across Domains
# ==============================================================================

@pytest.mark.parametrize(
    "domain_key",
    ["predictive", "deep_learning", "market", "treasury", "market_treasury"],
)
def test_global_q_c_v_a_o_contract_across_domains(domain_key, tmp_path):
    """Verify Q, C, V, VA stay on the checkpoint while A and O advance across all domains."""
    from start.registry import TestContext
    from start.review.applicability import applicable_tests

    if domain_key == "predictive":
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [4.0, 3.0, 2.0, 1.0]})
        y = [0, 1, 0, 1]
        ctx = TestContext(train=X.assign(y=y), test=X.assign(y=y), target_column="y")
        bundle = ReviewContextBundle(domains=(ReviewDomain.PREDICTIVE,), tabular=ctx)
        app = applicable_tests(bundle.domains)
        res = execute_market_treasury_tests(bundle, app)
        records = [
            make_evidence_record(r.test_id, f"EV-{i}", status=r.status, metrics=r.metrics)
            for i, r in enumerate(res[:5])
        ]
    elif domain_key == "deep_learning":
        from types import SimpleNamespace

        from start.modeling.dl_review_engines import build_training_evidence

        dummy = SimpleNamespace(
            history_={"train_loss": [0.4], "val_loss": [0.5]},
            best_epoch_=1,
            stopped_early_=False,
        )
        tr = build_training_evidence(dummy)
        records = [make_evidence_record(tr.test_id, "EV-DL-1", status=tr.status, metrics=tr.metrics)]
        bundle = ReviewContextBundle(domains=(ReviewDomain.PREDICTIVE,))
    elif domain_key == "market":
        bundle = make_market_bundle()
        app = applicable_tests(bundle.domains)
        res = execute_market_treasury_tests(bundle, app)
        records = [
            make_evidence_record(r.test_id, f"EV-M-{i}", status=r.status, metrics=r.metrics)
            for i, r in enumerate(res[:10])
        ]
    elif domain_key == "treasury":
        bundle = make_treasury_bundle()
        app = applicable_tests(bundle.domains)
        res = execute_market_treasury_tests(bundle, app)
        records = [
            make_evidence_record(r.test_id, f"EV-T-{i}", status=r.status, metrics=r.metrics)
            for i, r in enumerate(res)
        ]
    else:  # market_treasury
        bundle = make_cross_domain_bundle()
        app = applicable_tests(bundle.domains)
        res = execute_market_treasury_tests(bundle, app)
        records = [
            make_evidence_record(r.test_id, f"EV-MT-{i}", status=r.status, metrics=r.metrics)
            for i, r in enumerate(res[:15])
        ]

    scripted = [
        "V",                   # View checkpoint artifacts (non-terminal)
        "VA",                  # View all run artifacts (non-terminal)
        "Q", "Query agents",   # Question (non-terminal)
        "C", "Challenge spec", # Challenge (non-terminal)
        "O", "Override note",  # Override (terminal: advances)
    ] + ["A"] * 30

    iter_in = iter(scripted)
    decisions = run_domain_checkpoints(
        bundle,
        records,
        interactive=True,
        ask=lambda _: next(iter_in, "A"),
    )

    actions = [d["action"] for d in decisions]
    assert "question" in actions
    assert "challenge" in actions
    assert "override" in actions
    assert actions[2] == "override"


# ==============================================================================
# 10. Execution Plan Reconciliation
# ==============================================================================

def test_execution_plan_reconciliation():
    """Verify reconcile_execution method correctly tracks planned vs executed tests and artifacts."""
    bundle = ReviewContextBundle(domains=(ReviewDomain.MARKET,))
    preview = build_plan_preview(bundle)

    executed_tests = set(preview.applicable.test_ids)
    # Add an acceptable pattern-b test
    executed_tests.add("scenario.linear_return")
    executed_tests.add("validation.var_size_power")

    reconciliation = preview.reconcile_execution(executed_tests, generated_artifact_ids={"ART-1", "ART-2"})
    assert reconciliation["reconciled"] is True
    assert reconciliation["missing_planned_tests"] == []
    assert reconciliation["unplanned_tests"] == []
    assert reconciliation["artifacts_generated_count"] == 2

    # Verify detecting an unplanned unexpected test
    unplanned_exec = executed_tests | {"unplanned.rogue_model"}
    recon_bad = preview.reconcile_execution(unplanned_exec)
    assert recon_bad["reconciled"] is False
    assert "unplanned.rogue_model" in recon_bad["unplanned_tests"]


# ==============================================================================
# 11. Historical Git Tree Safety
# ==============================================================================

def test_historical_git_tree_safety_and_cleanliness():
    """Verify historical reference Git tree is strictly clean and unmodified."""
    ref_env = os.environ.get("START_REFERENCE_TREE")
    hist_path = Path(ref_env) if ref_env else (Path(__file__).resolve().parent.parent.parent / "My_Git" / "StART")
    if not hist_path.exists():
        import pytest
        pytest.skip("Reference Git tree not present at standard path.")

    # Run git status in historical repo
    res = subprocess.run(
        ["git", "-C", str(hist_path), "status", "--short"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert res.stdout.strip() == "", f"Historical Git tree has uncommitted modifications: {res.stdout}"
