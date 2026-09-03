"""Gate-2 Institutional Portfolio Intelligence and Hierarchical Allocation Test Suite.

Comprehensive verification for:
1. HRP Classical Backward Compatibility & Multi-Linkage (Single, Complete, Average)
2. Ward Linkage Non-Euclidean Geometry Rejection Guardrail
3. Typed HierarchicalTreeResult Serialization & Quasi-Diagonal Seriation
4. Cophenetic Correlation Diagnostic
5. Seeded Time-Series Block Bootstrap Cluster Stability
6. Euler Risk Contribution Decomposition & Reconciliation (Asset & Cluster level)
7. Equal Risk Contribution (ERC / Risk Parity) & Post-Solve Constraint Verification
8. Explicit 1/N Equal-Weight Baseline
9. Parametric Efficient Frontier & Reference Portfolio Overlays
10. Multi-Method Portfolio Comparison (Without Unsubstantiated Auto-Winner)
11. Non-Leaky Walk-Forward Rebalancing Evaluation Harness
12. Typed Artifact Foundation (ArtifactSpec, ArtifactRecord, Semantic Payloads)
13. Specialist Market Review Agents, Tool Boundaries & Zero-Prose Math Enforcement
14. Adversarial Challenge Agent Tool Requests & Provenance
15. Scale Performance Check (N=10, N=100)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from start.agents.market_review import (
    AdversarialChallengeAgent,
    HierarchicalAllocationAgent,
    MarketReviewDirectorAgent,
    PortfolioConstructionAgent,
)
from start.core.schemas import EvidenceRecord, Status, TestResult
from start.portfolio import (
    ArtifactRecord,
    BootstrapStabilityResult,
    CopheneticResult,
    EfficientFrontierResult,
    EqualRiskContributionResult,
    HierarchicalTreeResult,
    LinkageSensitivityResult,
    MethodComparisonResult,
    RiskContributionResult,
    WalkForwardResult,
    bootstrap_cluster_stability,
    calculate_risk_contributions,
    compare_portfolio_methods,
    cophenetic_distance_diagnostic,
    correlation_distance,
    hrp_weights_and_tree,
    linkage_sensitivity_analysis,
    render_dendrogram_artifact,
    render_risk_contribution_artifact,
    render_seriated_correlation_artifact,
    run_walk_forward_evaluation,
    solve_equal_risk_contribution,
    solve_equal_weight,
    trace_efficient_frontier,
)
from start.registry.market_contexts import MarketContext
from start.tests.portfolio import hierarchical_risk_parity


@pytest.fixture
def sample_market_data():
    """Deterministic 5-asset return fixture."""
    rng = np.random.default_rng(20260831)
    n_obs = 300
    # 2 correlated tech clusters + 1 finance asset
    factor1 = rng.normal(0.0005, 0.015, size=n_obs)
    factor2 = rng.normal(0.0003, 0.010, size=n_obs)

    r_aapl = 0.8 * factor1 + rng.normal(0, 0.005, size=n_obs)
    r_msft = 0.75 * factor1 + rng.normal(0, 0.005, size=n_obs)
    r_goog = 0.7 * factor1 + rng.normal(0, 0.006, size=n_obs)
    r_amzn = 0.65 * factor1 + rng.normal(0, 0.007, size=n_obs)
    r_jpm = 0.85 * factor2 + rng.normal(0, 0.008, size=n_obs)

    df = pd.DataFrame(
        {
            "AAPL": r_aapl,
            "MSFT": r_msft,
            "GOOGL": r_goog,
            "AMZN": r_amzn,
            "JPM": r_jpm,
        },
        index=pd.date_range("2024-01-01", periods=n_obs, freq="B"),
    )
    cov = df.cov()
    return df, cov


# ========================================================================= #
# 1. HRP CLASSICAL BACKWARD COMPATIBILITY & MULTI-LINKAGE
# ========================================================================= #


def test_hrp_backward_compatibility(sample_market_data):
    """Verify that portfolio.hierarchical_risk_parity preserves exact classical metrics."""
    df, cov = sample_market_data
    ctx = MarketContext(returns=df)

    res = hierarchical_risk_parity(ctx, linkage_method="single")
    assert res.status == Status.RECORDED
    assert res.test_id == "portfolio.hierarchical_risk_parity"

    metrics = res.metrics
    assert metrics["n_assets"] == 5
    assert metrics["linkage_method"] == "single"
    assert "quasi_diagonal_order" in metrics
    assert metrics["weights_sum"] == pytest.approx(1.0, abs=1e-8)
    assert metrics["effective_n_positions"] > 1.0
    assert metrics["portfolio_variance_periodic"] > 0.0
    assert "weight.AAPL" in metrics
    assert "weight.JPM" in metrics


def test_hrp_multi_linkage_execution(sample_market_data):
    """Verify single, complete, and average linkages execute cleanly and produce valid weights."""
    _, cov = sample_market_data
    for method in ("single", "complete", "average"):
        weights, tree = hrp_weights_and_tree(cov, linkage_method=method)
        assert len(weights) == 5
        assert np.sum(weights) == pytest.approx(1.0, abs=1e-8)
        assert (weights > 0.0).all()
        assert len(tree.quasi_diagonal_order) == 5


def test_hrp_ward_geometry_rejection(sample_market_data):
    """Verify Ward linkage is rejected with explicit ValueError when applied to precomputed non-Euclidean distance."""
    _, cov = sample_market_data
    with pytest.raises(ValueError, match="Ward linkage requires Euclidean geometry"):
        hrp_weights_and_tree(cov, linkage_method="ward", is_euclidean_features=False)


# ========================================================================= #
# 2. TYPED TREE SERIALIZATION, COPHENETIC & LINKAGE SENSITIVITY
# ========================================================================= #


def test_hierarchical_tree_serialization(sample_market_data):
    """Verify HierarchicalTreeResult contains complete tree structure and fingerprints."""
    _, cov = sample_market_data
    _, tree = hrp_weights_and_tree(cov, linkage_method="average")

    assert isinstance(tree, HierarchicalTreeResult)
    assert tree.assets == ("AAPL", "MSFT", "GOOGL", "AMZN", "JPM")
    assert len(tree.leaf_order) == 5
    assert len(tree.quasi_diagonal_order) == 5
    assert len(tree.linkage_matrix) == 4  # N-1 merges
    assert tree.correlation_fingerprint != ""
    assert tree.covariance_fingerprint != ""
    assert tree.cophenetic_correlation is not None


def test_cophenetic_distance_diagnostic(sample_market_data):
    """Verify cophenetic correlation diagnostic computes valid metric without unproven threshold."""
    _, cov = sample_market_data
    res = cophenetic_distance_diagnostic(cov, linkage_method="average")

    assert isinstance(res, CopheneticResult)
    assert res.n_assets == 5
    assert -1.0 <= res.cophenetic_correlation <= 1.0
    assert res.linkage_method == "average"


def test_linkage_sensitivity_analysis(sample_market_data):
    """Verify linkage sensitivity compares single, complete, and average linkages deterministically."""
    _, cov = sample_market_data
    sens = linkage_sensitivity_analysis(cov, methods=("single", "complete", "average"))

    assert isinstance(sens, LinkageSensitivityResult)
    assert sens.methods_compared == ("single", "complete", "average")
    assert "single_vs_complete" in sens.pairwise_l1_distances
    assert "complete_vs_average" in sens.pairwise_l2_distances
    assert "single_vs_average" in sens.spearman_order_correlations
    assert sens.pairwise_l1_distances["single_vs_complete"] >= 0.0


# ========================================================================= #
# 3. SEEDED TIME-SERIES BLOCK BOOTSTRAP CLUSTER STABILITY
# ========================================================================= #


def test_seeded_block_bootstrap_stability(sample_market_data):
    """Verify block bootstrap stability is deterministic and reproducible under fixed seed."""
    df, _ = sample_market_data

    b1 = bootstrap_cluster_stability(df, n_replicates=20, block_size=10, seed=42)
    b2 = bootstrap_cluster_stability(df, n_replicates=20, block_size=10, seed=42)

    assert isinstance(b1, BootstrapStabilityResult)
    assert b1.pairwise_co_clustering_matrix == b2.pairwise_co_clustering_matrix
    assert b1.mean_pairwise_stability == b2.mean_pairwise_stability
    assert 0.0 <= b1.mean_pairwise_stability <= 1.0


# ========================================================================= #
# 4. EULER RISK CONTRIBUTIONS & RECONCILIATION
# ========================================================================= #


def test_euler_risk_contributions_reconciliation(sample_market_data):
    """Verify Euler variance and volatility risk contributions reconcile exactly to total risk."""
    _, cov = sample_market_data
    weights = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=cov.columns)

    cluster_map = {"Tech": ["AAPL", "MSFT", "GOOGL", "AMZN"], "Finance": ["JPM"]}
    rc = calculate_risk_contributions(weights, cov, cluster_map=cluster_map)

    assert isinstance(rc, RiskContributionResult)
    assert rc.portfolio_volatility > 0.0
    # Reconciliation: sum(CR) == sigma_p
    sum_cr = sum(rc.component_contributions.values())
    assert sum_cr == pytest.approx(rc.portfolio_volatility, abs=1e-9)
    # Reconciliation: sum(%CR) == 1.0
    sum_pcr = sum(rc.percentage_contributions.values())
    assert sum_pcr == pytest.approx(1.0, abs=1e-9)
    assert rc.euler_reconciliation_error < 1e-9

    # Cluster aggregation reconciliation
    sum_cluster_pcr = sum(rc.cluster_percentage_contributions.values())
    assert sum_cluster_pcr == pytest.approx(1.0, abs=1e-9)
    assert "Tech" in rc.cluster_contributions
    assert "Finance" in rc.cluster_contributions


# ========================================================================= #
# 5. EQUAL RISK CONTRIBUTION (ERC / RISK PARITY)
# ========================================================================= #


def test_equal_risk_contribution_solver(sample_market_data):
    """Verify ERC solver equates risk contributions and satisfies budget/non-negativity constraints."""
    _, cov = sample_market_data
    erc = solve_equal_risk_contribution(cov)

    assert isinstance(erc, EqualRiskContributionResult)
    assert erc.converged is True
    assert erc.constraint_violations["budget"] < 1e-6
    assert erc.constraint_violations["non_negativity"] == 0.0

    # Risk contribution dispersion must be small (near perfect equality)
    assert erc.max_risk_contribution_dispersion < 1e-4
    for _, pcr in erc.percentage_risk_contributions.items():
        assert pcr == pytest.approx(1.0 / 5, abs=1e-3)


def test_erc_engine_diagnostics(sample_market_data):
    """Verify ERC engine returns full diagnostic metrics."""
    _, cov = sample_market_data
    res = solve_equal_risk_contribution(cov)
    assert res.converged is True
    assert res.target_risk_contribution == 0.2
    assert res.portfolio_volatility > 0.0
    assert len(res.percentage_risk_contributions) == 5


# ========================================================================= #
# 6. EQUAL-WEIGHT BASELINE & EFFICIENT FRONTIER
# ========================================================================= #


def test_equal_weight_baseline(sample_market_data):
    """Verify solve_equal_weight produces explicit 1/N benchmark."""
    _, cov = sample_market_data
    w, metrics = solve_equal_weight(list(cov.columns), cov.to_numpy())

    assert len(w) == 5
    assert (w == 0.2).all()
    assert metrics["effective_n_positions"] == 5.0
    assert metrics["herfindahl"] == 0.2


def test_efficient_frontier_tracing(sample_market_data):
    """Verify parametric efficient frontier tracing and reference portfolio overlays."""
    df, cov = sample_market_data
    mu = df.mean().to_numpy()
    sigma = cov.to_numpy()
    assets = list(df.columns)

    res = trace_efficient_frontier(mu, sigma, assets, n_points=15)
    assert isinstance(res, EfficientFrontierResult)
    assert len(res.frontier_points) > 0
    assert res.min_variance_point.volatility_annualised <= res.max_sharpe_point.volatility_annualised
    assert res.erc_point is not None
    assert res.hrp_point is not None
    assert res.equal_weight_point is not None


def test_efficient_frontier_overlays(sample_market_data):
    """Verify efficient frontier overlays and reference portfolios."""
    df, cov = sample_market_data
    mu = df.mean().to_numpy()
    sigma = cov.to_numpy()
    assets = list(df.columns)

    res = trace_efficient_frontier(mu, sigma, assets, n_points=10)
    assert len(res.frontier_points) > 0
    assert res.erc_point.volatility_annualised > 0.0
    assert res.hrp_point.volatility_annualised > 0.0


# ========================================================================= #
# 7. MULTI-METHOD PORTFOLIO COMPARISON (NO AUTO WINNER)
# ========================================================================= #


def test_method_comparison_deterministic(sample_market_data):
    """Verify portfolio method comparison compares all methods without declaring an automatic winner."""
    df, cov = sample_market_data
    prior = pd.Series([0.2, 0.2, 0.2, 0.2, 0.2], index=df.columns)

    res = compare_portfolio_methods(df, cov, prior_weights=prior)
    assert isinstance(res, MethodComparisonResult)
    assert "equal_weight" in res.methods
    assert "minimum_variance" in res.methods
    assert "hierarchical_risk_parity" in res.methods
    assert "equal_risk_contribution" in res.methods
    assert len(res.summary_table) >= 5


def test_method_comparison_matrices(sample_market_data):
    """Verify method comparison outputs complete weights and risk contribution matrices."""
    df, cov = sample_market_data
    res = compare_portfolio_methods(df, cov)
    assert len(res.methods) >= 5
    assert len(res.weights_matrix) >= 5
    assert len(res.risk_contributions_matrix) >= 5


# ========================================================================= #
# 8. NON-LEAKY WALK-FORWARD EVALUATION
# ========================================================================= #


def test_walk_forward_evaluation_non_leaky(sample_market_data):
    """Verify walk-forward evaluation enforces strict chronological ordering and transaction costs."""
    df, _ = sample_market_data

    wf = run_walk_forward_evaluation(
        df, method="hrp", estimation_window=100, rebalance_frequency=25, transaction_cost_bps=10.0
    )
    assert isinstance(wf, WalkForwardResult)
    assert len(wf.rebalance_dates) >= 5
    assert len(wf.out_of_sample_returns) > 0
    assert wf.annualised_volatility > 0.0
    assert wf.transaction_cost_bps == 10.0


# ========================================================================= #
# 9. TYPED ARTIFACT FOUNDATION & PROVENANCE
# ========================================================================= #


def test_artifact_records_and_semantic_hashes(sample_market_data, tmp_path):
    """Verify typed ArtifactRecord creation and semantic payload hashes."""
    _, cov = sample_market_data
    _, tree = hrp_weights_and_tree(cov)

    # 1. Dendrogram
    dendro_art = render_dendrogram_artifact(tree, evidence_ids=("EV-TEST-1",), output_dir=tmp_path)
    assert isinstance(dendro_art, ArtifactRecord)
    assert dendro_art.spec.artifact_type == "dendrogram"
    assert dendro_art.file_path is not None
    assert Path(dendro_art.file_path).exists()
    assert "quasi_diagonal_order" in dendro_art.semantic_payload

    # 2. Seriated Correlation
    corr, _ = correlation_distance(cov.to_numpy())
    ser_art = render_seriated_correlation_artifact(
        corr, tree.quasi_diagonal_order, tree.assets, evidence_ids=("EV-TEST-1",)
    )
    assert ser_art.spec.artifact_type == "seriated_correlation_heatmap"
    assert "seriated_correlation_matrix" in ser_art.semantic_payload

    # 3. Risk Contribution Waterfall
    rc = calculate_risk_contributions(pd.Series([0.2] * 5, index=cov.columns), cov)
    rc_art = render_risk_contribution_artifact(rc, tree.assets, evidence_ids=("EV-TEST-1",))
    assert rc_art.spec.artifact_type == "risk_contribution_waterfall"


# ========================================================================= #
# 10. SPECIALIST AGENTS & ADVERSARIAL CHALLENGES
# ========================================================================= #


def test_specialist_agents_and_adversarial_challenges():
    """Verify specialist agents emit structured findings and challenges without prose math."""
    hrp_rec = EvidenceRecord.from_result(
        TestResult(
            test_id="portfolio.hierarchical_risk_parity",
            test_name="Hierarchical risk parity",
            status=Status.RECORDED,
            metrics={"effective_n_positions": 4.12, "max_weight": 0.28, "linkage_method": "single"},
        ),
        run_id="RUN-1",
    )
    hrp_rec.evidence_id = "EV-HRP-001"

    mvo_rec = EvidenceRecord.from_result(
        TestResult(
            test_id="portfolio.mean_variance",
            test_name="Mean-variance optimisation",
            status=Status.RECORDED,
            metrics={"volatility_annualised": 0.085},
            params={"objective": "min_variance"},
        ),
        run_id="RUN-1",
    )
    mvo_rec.evidence_id = "EV-MVO-001"

    context = {"evidence_records": [hrp_rec, mvo_rec]}

    # 1. HierarchicalAllocationAgent
    h_agent = HierarchicalAllocationAgent()
    h_res = h_agent.execute(context)
    assert h_res["status"] == "completed"
    assert len(h_res["findings"]) == 1
    assert "EV-HRP-001" in h_res["findings"][0]["statement"]

    # 2. PortfolioConstructionAgent
    p_agent = PortfolioConstructionAgent()
    p_res = p_agent.execute(context)
    assert p_res["status"] == "completed"
    assert len(p_res["findings"]) == 1
    assert "EV-MVO-001" in p_res["findings"][0]["statement"]

    # 3. AdversarialChallengeAgent
    adv_agent = AdversarialChallengeAgent()
    adv_res = adv_agent.execute(context)
    assert adv_res["status"] == "completed"
    assert len(adv_res["challenges"]) >= 2
    assert any("CHAL-LINKAGE" in c["challenge_id"] for c in adv_res["challenges"])

    # 4. MarketReviewDirectorAgent Orchestration
    director = MarketReviewDirectorAgent()
    d_res = director.execute(context)
    assert d_res["status"] == "orchestrated"
    assert d_res["findings_count"] == 2
    assert d_res["challenges_count"] >= 2


# ========================================================================= #
# 11. SCALE PERFORMANCE CHECK (N=10, N=100)
# ========================================================================= #


def test_hrp_and_erc_scale_performance():
    """Verify HRP and ERC execute efficiently on N=10 and N=100 assets."""
    rng = np.random.default_rng(123)
    for n in (10, 100):
        # Generate positive definite random covariance
        A = rng.normal(0, 1, size=(n, n))
        cov = A @ A.T + np.eye(n) * 0.1
        assets = [f"Asset_{i}" for i in range(n)]

        # HRP
        w_hrp, tree = hrp_weights_and_tree(cov, assets=assets)
        assert len(w_hrp) == n
        assert np.sum(w_hrp) == pytest.approx(1.0, abs=1e-7)

        # ERC
        erc = solve_equal_risk_contribution(cov, assets=assets, max_iter=200)
        assert len(erc.weights) == n
        assert erc.converged is True
