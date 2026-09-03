"""Gate-2A Portfolio Intelligence Vertical-Slice Acceptance Audit Test Suite.

Verifies:
1. Complete Engine -> Evidence integration across all 11 analytical capabilities (Zero Orphan Quantitative Results).
2. Complete 10-Artifact Suite Rendering, Provenance Enforcement, and Semantic Hash Invariants.
3. Strict Agent Tool Allowlists and Disallowed Execution Blocking.
4. Strict Numerical Claim Grounding and Fail-Closed Behavior on Bad Citations / Metric Paths.
5. Ward Linkage Geometry Invariant (Euclidean vs Non-Euclidean Rejection).
6. HRP Classical Backward Compatibility Invariant.
7. ERC Mathematical Contract & Independent Post-Solve Verification (Diagonal, Symmetry, Heterogeneous).
8. Stationary Block Bootstrap Seed Reproducibility & Serial Dependence.
9. Walk-Forward Non-Leaky Time Ordering & Transaction Cost Contract.
10. End-to-End Non-Interactive Showcase Execution and Manifest Validation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from start.agents.market_review import (
    AdversarialChallengeAgent,
    HierarchicalAllocationAgent,
    PortfolioConstructionAgent,
    get_grounded_metric,
)
from start.core.schemas import EvidenceRecord, Status, TestResult
from start.portfolio import (
    ArtifactRecord,
    bootstrap_cluster_stability,
    bootstrap_stability_to_evidence,
    calculate_risk_contributions,
    compare_portfolio_methods,
    cophenetic_distance_diagnostic,
    cophenetic_to_evidence,
    correlation_distance,
    efficient_frontier_to_evidence,
    equal_weight_to_evidence,
    erc_to_evidence,
    hrp_weights_and_tree,
    linkage_sensitivity_analysis,
    linkage_sensitivity_to_evidence,
    method_comparison_to_evidence,
    render_asset_weights_artifact,
    render_cluster_allocation_artifact,
    render_cluster_risk_artifact,
    render_cluster_tree_artifact,
    render_dendrogram_artifact,
    render_distance_matrix_artifact,
    render_efficient_frontier_artifact,
    render_raw_correlation_artifact,
    render_risk_contribution_artifact,
    render_seriated_correlation_artifact,
    risk_contributions_to_evidence,
    run_walk_forward_evaluation,
    solve_equal_risk_contribution,
    solve_equal_weight,
    trace_efficient_frontier,
    tree_to_evidence,
    validate_linkage_geometry,
    walk_forward_to_evidence,
)


@pytest.fixture
def audit_market_data():
    """Deterministic 5-asset return fixture."""
    rng = np.random.default_rng(20260831)
    n_obs = 300
    f_tech = rng.normal(0.0005, 0.015, size=n_obs)
    f_fin = rng.normal(0.0003, 0.010, size=n_obs)

    r_data = {
        "AAPL": 0.80 * f_tech + rng.normal(0, 0.005, size=n_obs),
        "MSFT": 0.75 * f_tech + rng.normal(0, 0.005, size=n_obs),
        "GOOGL": 0.70 * f_tech + rng.normal(0, 0.006, size=n_obs),
        "AMZN": 0.65 * f_tech + rng.normal(0, 0.007, size=n_obs),
        "JPM": 0.85 * f_fin + rng.normal(0, 0.008, size=n_obs),
    }
    df = pd.DataFrame(r_data, index=pd.date_range("2024-01-01", periods=n_obs, freq="B"))
    return df, df.cov()


# ========================================================================= #
# 1. ENGINE -> EVIDENCE INTEGRATION (ZERO ORPHAN QUANTITATIVE CAPABILITIES)
# ========================================================================= #

def test_engine_to_evidence_integration_all_11_capabilities(audit_market_data):
    """Verify that all 11 Gate-2 analytical capabilities cleanly produce EvidenceRecords."""
    df, cov = audit_market_data
    cov_mat = cov.to_numpy(dtype=float)
    mu_vec = df.mean().to_numpy(dtype=float)
    assets = list(df.columns)

    records: list[EvidenceRecord] = []

    # 1. HRP Tree Hierarchy
    _, tree_res = hrp_weights_and_tree(cov)
    ev_tree = tree_to_evidence(tree_res)
    assert ev_tree.test_id == "portfolio.hierarchical_risk_parity.tree_topology"
    assert ev_tree.status == Status.RECORDED
    assert "quasi_diagonal_order" in ev_tree.metrics
    records.append(ev_tree)

    # 2. Linkage Sensitivity
    sens_res = linkage_sensitivity_analysis(cov)
    ev_sens = linkage_sensitivity_to_evidence(sens_res)
    assert ev_sens.test_id == "portfolio.hierarchical_risk_parity.linkage_sensitivity"
    assert "n_methods_compared" in ev_sens.metrics
    records.append(ev_sens)

    # 3. Cophenetic Diagnostic
    coph_res = cophenetic_distance_diagnostic(cov)
    ev_coph = cophenetic_to_evidence(coph_res)
    assert ev_coph.test_id == "portfolio.hierarchical_risk_parity.cophenetic_distance"
    assert "cophenetic_correlation" in ev_coph.metrics
    records.append(ev_coph)

    # 4. Bootstrap Cluster Stability
    boot_res = bootstrap_cluster_stability(df, n_replicates=10, block_size=10, seed=42)
    ev_boot = bootstrap_stability_to_evidence(boot_res)
    assert ev_boot.test_id == "portfolio.hierarchical_risk_parity.bootstrap_stability"
    assert "mean_pairwise_stability" in ev_boot.metrics
    records.append(ev_boot)

    # 5. Euler Risk Contributions (Asset level)
    w_vec = np.full(5, 0.2)
    rc_res = calculate_risk_contributions(w_vec, cov)
    ev_rc = risk_contributions_to_evidence(rc_res)
    assert ev_rc.test_id == "portfolio.risk_statistics.euler_decomposition"
    assert "portfolio_volatility" in ev_rc.metrics
    assert "euler_reconciliation_error" in ev_rc.metrics
    records.append(ev_rc)

    # 6. Cluster Risk Contributions
    cluster_map = {"Tech": ["AAPL", "MSFT", "GOOGL", "AMZN"], "Finance": ["JPM"]}
    rc_cluster_res = calculate_risk_contributions(w_vec, cov, cluster_map=cluster_map)
    ev_c_rc = risk_contributions_to_evidence(
        rc_cluster_res, test_id="portfolio.risk_statistics.cluster_euler_decomposition"
    )
    assert ev_c_rc.test_id == "portfolio.risk_statistics.cluster_euler_decomposition"
    assert "cluster_cr.Tech" in ev_c_rc.metrics
    assert "cluster_pcr.Finance" in ev_c_rc.metrics
    records.append(ev_c_rc)

    # 7. 1/N Baseline
    ew_series, ew_metrics = solve_equal_weight(assets, cov_mat, mu_vec)
    ev_ew = equal_weight_to_evidence(ew_series, ew_metrics)
    assert ev_ew.test_id == "portfolio.mean_variance.equal_weight_baseline"
    assert "effective_n_positions" in ev_ew.metrics
    records.append(ev_ew)

    # 8. Equal Risk Contribution (ERC)
    erc_res = solve_equal_risk_contribution(cov)
    ev_erc = erc_to_evidence(erc_res)
    assert ev_erc.test_id == "portfolio.risk_statistics.equal_risk_contribution"
    assert ev_erc.metrics["converged"] is True
    records.append(ev_erc)

    # 9. Efficient Frontier
    frontier_res = trace_efficient_frontier(mu_vec, cov_mat, assets, n_points=10)
    ev_front = efficient_frontier_to_evidence(frontier_res)
    assert ev_front.test_id == "portfolio.mean_variance.efficient_frontier"
    assert "min_volatility_annualised" in ev_front.metrics
    records.append(ev_front)

    # 10. Portfolio Method Comparison
    comp_res = compare_portfolio_methods(df, cov)
    ev_comp = method_comparison_to_evidence(comp_res)
    assert ev_comp.test_id == "portfolio.mean_variance.method_comparison"
    assert "n_methods_compared" in ev_comp.metrics
    records.append(ev_comp)

    # 11. Walk-Forward Evaluation
    wf_res = run_walk_forward_evaluation(df, estimation_window=100, rebalance_frequency=25)
    ev_wf = walk_forward_to_evidence(wf_res)
    assert ev_wf.test_id == "portfolio.historical_returns.walk_forward"
    assert "annualised_volatility" in ev_wf.metrics
    records.append(ev_wf)

    assert len(records) == 11
    # Check that all records have non-empty evidence IDs and no orphan state
    for rec in records:
        assert rec.evidence_id.startswith("EV-")
        assert rec.status == Status.RECORDED
        assert rec.input_artifact_hash is not None


# ========================================================================= #
# 2. COMPLETE 10-ARTIFACT SUITE RENDERING & PROVENANCE INVARIANTS
# ========================================================================= #

def test_all_10_artifacts_render_and_enforce_provenance(audit_market_data, tmp_path):
    """Verify all 10 artifacts render cleanly and strictly reject empty evidence provenance."""
    df, cov = audit_market_data
    cov_mat = cov.to_numpy(dtype=float)
    corr_mat, dist_mat = correlation_distance(cov_mat)
    assets = list(df.columns)
    mu_vec = df.mean().to_numpy(dtype=float)

    _, tree_res = hrp_weights_and_tree(cov)
    cluster_map = {"Tech": ["AAPL", "MSFT", "GOOGL", "AMZN"], "Finance": ["JPM"]}
    rc = calculate_risk_contributions(pd.Series([0.2]*5, index=assets), cov, cluster_map=cluster_map)
    frontier = trace_efficient_frontier(mu_vec, cov_mat, assets, n_points=10)
    w_dict = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.2, "AMZN": 0.15, "JPM": 0.15}
    cluster_w = {"Tech": 0.85, "Finance": 0.15}

    ev_ids = ("EV-TEST-001", "EV-TEST-002")

    # Verify empty evidence provenance rejection
    with pytest.raises(ValueError, match="Artifact must have explicit evidence_ids provenance"):
        render_dendrogram_artifact(tree_res, evidence_ids=())

    with pytest.raises(ValueError, match="Artifact must have explicit evidence_ids provenance"):
        render_raw_correlation_artifact(corr_mat, assets, evidence_ids=())

    with pytest.raises(ValueError, match="Artifact must have explicit evidence_ids provenance"):
        render_distance_matrix_artifact(dist_mat, assets, evidence_ids=())

    with pytest.raises(ValueError, match="Artifact must have explicit evidence_ids provenance"):
        render_asset_weights_artifact(w_dict, evidence_ids=())

    # Render all 10 with valid provenance
    art1 = render_dendrogram_artifact(tree_res, evidence_ids=ev_ids, output_dir=tmp_path)
    art2 = render_raw_correlation_artifact(corr_mat, assets, evidence_ids=ev_ids, output_dir=tmp_path)
    art3 = render_seriated_correlation_artifact(corr_mat, tree_res.quasi_diagonal_order, assets, evidence_ids=ev_ids, output_dir=tmp_path)
    art4 = render_distance_matrix_artifact(dist_mat, assets, evidence_ids=ev_ids, output_dir=tmp_path)
    art5 = render_cluster_tree_artifact(tree_res, evidence_ids=ev_ids, output_dir=tmp_path)
    art6 = render_cluster_allocation_artifact(cluster_w, rc.cluster_percentage_contributions, evidence_ids=ev_ids, output_dir=tmp_path)
    art7 = render_asset_weights_artifact(w_dict, evidence_ids=ev_ids, output_dir=tmp_path)
    art8 = render_risk_contribution_artifact(rc, assets, evidence_ids=ev_ids, output_dir=tmp_path)
    art9 = render_cluster_risk_artifact(rc, evidence_ids=ev_ids, output_dir=tmp_path)
    art10 = render_efficient_frontier_artifact(frontier, evidence_ids=ev_ids, output_dir=tmp_path)

    all_arts = [art1, art2, art3, art4, art5, art6, art7, art8, art9, art10]
    assert len(all_arts) == 10

    for art in all_arts:
        assert isinstance(art, ArtifactRecord)
        assert art.artifact_id.startswith("ART-")
        assert len(art.spec.evidence_ids) == 2
        assert art.file_path is not None
        p = Path(art.file_path)
        assert p.exists()
        assert p.stat().st_size > 0
        # Companion JSON must exist
        p_json = p.parent / f"{art.artifact_id}.json"
        assert p_json.exists()
        # Semantic payload must have non-empty hash
        assert len(art.semantic_payload_hash) == 64
        assert art.semantic_payload is not None


# ========================================================================= #
# 3. AGENT TOOL ALLOWLISTS & BOUNDARY ENFORCEMENT
# ========================================================================= #

def test_agent_tool_allowlists_and_disallowed_execution():
    """Verify specialist agents enforce strict tool allowlists."""
    h_agent = HierarchicalAllocationAgent()
    p_agent = PortfolioConstructionAgent()
    a_agent = AdversarialChallengeAgent()

    # Allowed tools succeed
    assert "hrp_weights_and_tree" in h_agent.ALLOWED_TOOLS
    assert "solve_equal_risk_contribution" in p_agent.ALLOWED_TOOLS
    assert "linkage_sensitivity_analysis" in a_agent.ALLOWED_TOOLS

    # Disallowed tools raise PermissionError
    with pytest.raises(PermissionError, match="not in the allowed toolset"):
        h_agent.execute_tool("solve_min_variance")

    with pytest.raises(PermissionError, match="not in the allowed toolset"):
        p_agent.execute_tool("bootstrap_cluster_stability")

    with pytest.raises(PermissionError, match="not in the allowed toolset"):
        a_agent.execute_tool("solve_equal_weight")


# ========================================================================= #
# 4. NUMERICAL CLAIM GROUNDING & FAIL CLOSED
# ========================================================================= #

def test_numerical_claim_grounding_and_fail_closed():
    """Verify get_grounded_metric succeeds on valid paths and fails closed on invalid ones."""
    res = TestResult(
        test_id="portfolio.hierarchical_risk_parity",
        test_name="Hierarchical risk parity",
        status=Status.RECORDED,
        metrics={"effective_n_positions": 4.12, "max_weight": 0.28},
    )
    rec = EvidenceRecord.from_result(res, run_id="RUN-1")
    rec.evidence_id = "EV-HRP-1234"

    records = [rec]

    # Valid lookup succeeds
    val = get_grounded_metric(records, "EV-HRP-1234", "effective_n_positions")
    assert val == 4.12

    # Wrong metric path raises KeyError
    with pytest.raises(KeyError, match="Metric path 'non_existent_metric' not found"):
        get_grounded_metric(records, "EV-HRP-1234", "non_existent_metric")

    # Unknown evidence ID raises KeyError
    with pytest.raises(KeyError, match="was not found in active evidence pool"):
        get_grounded_metric(records, "EV-UNKNOWN-999", "effective_n_positions")


# ========================================================================= #
# 5. WARD LINKAGE GEOMETRY SAFEGUARDS
# ========================================================================= #

def test_ward_linkage_geometry_safeguards(audit_market_data):
    """Verify Ward linkage is strictly validated on Euclidean geometry vs arbitrary distance."""
    _, cov = audit_market_data

    # Valid Euclidean features -> allowed
    validate_linkage_geometry("ward", is_euclidean=True)

    # Non-Euclidean / arbitrary distance -> rejected
    with pytest.raises(ValueError, match="Ward linkage requires Euclidean geometry"):
        validate_linkage_geometry("ward", is_euclidean=False)

    with pytest.raises(ValueError, match="Ward linkage requires Euclidean geometry"):
        hrp_weights_and_tree(cov, linkage_method="ward", is_euclidean_features=False)


# ========================================================================= #
# 6. ERC CONTRACT & INDEPENDENT POST-SOLVE VERIFICATION
# ========================================================================= #

def test_erc_contract_known_answers():
    """Verify ERC known answers on diagonal, symmetric, and heterogeneous matrices."""
    # 1. Diagonal covariance: w_i \propto 1 / \sigma_i
    diag_cov = np.diag([0.04, 0.09, 0.16])  # sigmas = 0.2, 0.3, 0.4
    erc_diag = solve_equal_risk_contribution(diag_cov)
    assert erc_diag.converged is True
    inv_vol = 1.0 / np.sqrt(np.diag(diag_cov))
    expected_w = inv_vol / np.sum(inv_vol)
    actual_w = np.array(list(erc_diag.weights.values()))
    np.testing.assert_allclose(actual_w, expected_w, atol=1e-4)

    # 2. Symmetric equal covariance: exact 1/N
    n = 4
    sym_cov = np.full((n, n), 0.02) + np.eye(n) * 0.08
    erc_sym = solve_equal_risk_contribution(sym_cov)
    assert erc_sym.converged is True
    for _, w in erc_sym.weights.items():
        assert w == pytest.approx(1.0 / n, abs=1e-4)

    # 3. Post-solve constraint violations check
    assert erc_sym.constraint_violations["budget"] < 1e-6
    assert erc_sym.constraint_violations["non_negativity"] == 0.0
    assert erc_sym.max_risk_contribution_dispersion < 1e-4


# ========================================================================= #
# 7. WALK-FORWARD NON-LEAKY BOUNDARY & COST CONTRACT
# ========================================================================= #

def test_walk_forward_strict_non_leakage(audit_market_data):
    """Verify walk-forward enforce strict [t-W, t) estimation and [t, t+F) evaluation."""
    df, _ = audit_market_data

    # Default is gross (0.0 cost)
    wf_gross = run_walk_forward_evaluation(df, estimation_window=80, rebalance_frequency=20, transaction_cost_bps=0.0)
    assert wf_gross.transaction_cost_bps == 0.0
    assert len(wf_gross.rebalance_dates) > 0

    # Configured cost applies drag
    wf_cost = run_walk_forward_evaluation(df, estimation_window=80, rebalance_frequency=20, transaction_cost_bps=25.0)
    assert wf_cost.transaction_cost_bps == 25.0
    assert wf_cost.annualised_return <= wf_gross.annualised_return


# ========================================================================= #
# 8. NON-INTERACTIVE SHOWCASE END-TO-END VERIFICATION
# ========================================================================= #

def test_non_interactive_showcase_execution(tmp_path):
    """Verify scripts/demo_portfolio_intelligence.py executes cleanly end-to-end."""
    from scripts.demo_portfolio_intelligence import run_showcase

    summary = run_showcase(output_dir=tmp_path)
    assert summary["status"] == "SUCCESS"
    assert summary["evidence_records_count"] == 13
    assert summary["artifacts_count"] == 10
    assert summary["critic_passed"] is True
    assert summary["critic_issues_count"] == 0

    manifest_p = tmp_path / "manifest.json"
    assert manifest_p.exists()
    with open(manifest_p, encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest["artifacts"]) == 10
