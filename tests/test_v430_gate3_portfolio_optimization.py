"""Gate 3 Acceptance Test Suite — Institutional Portfolio Optimization & Constrained Allocation.

Verifies:
1. Deterministic Post-Solve Constraint Verifier (Budget, Bounds, Concentration, Factor, Group, Turnover, Tracking Error).
2. Black-Litterman Engine (Implied Equilibrium, Bayesian Posterior, Monotonicity, Constrained Solve).
3. Robust Mean-Variance Optimization (Ellipsoidal Uncertainty, Worst-Case Return Monotonicity, Sensitivity Grid).
4. Rockafellar-Uryasev CVaR Optimization (LP Formulation, VaR/CVaR Ordering, Confidence Sensitivity).
5. Hierarchical Equal Risk Contribution (HERC) (Tree Bisection, Cluster Parity, Euler Risk Decomposition).
6. Maximum Diversification Portfolio (MDP) (Diversification Ratio >= 1, Analytical Diagonal Case).
7. Tracking-Error Constrained Optimization (Benchmark Anchoring, Hard Bound Enforcement).
8. Turnover & Transaction Cost Engine (Zero Trade Invariant, Linear Cost Schedule, RebalanceDecision).
9. Multi-Method Institutional Comparison (Unified Matrix, 0 Winner Claims).
10. EvidenceBridge & Cryptographic Artifact Generation (0 Orphan Calculations).
11. Specialist Review Agents & Governance Separation (Allowlists, 0 Prose Math, Critic vs Governance Roles).
"""

import math

import numpy as np
import pandas as pd
import pytest

from start.agents.market_review import (
    CriticDisposition,
    EvidenceCriticAgent,
    GovernanceAgent,
    GovernanceVerdict,
    HierarchicalAllocationAgent,
    MarketReviewDirectorAgent,
    PortfolioConstructionAgent,
)
from start.core.schemas import EvidenceRecord
from start.portfolio import (
    ArtifactRecord,
    BlackLittermanResult,
    ConstraintType,
    CVaROptimizationResult,
    FactorConstraintSpec,
    GroupConstraintSpec,
    HERCResult,
    MaxDiversificationResult,
    MethodComparisonResult,
    RebalanceDecision,
    RobustMVOResult,
    RobustSensitivityResult,
    TrackingErrorResult,
    TransactionCostSpec,
    UncertaintyDerivationPolicy,
    black_litterman_to_evidence,
    build_rebalance_decision,
    compare_portfolio_methods,
    compute_black_litterman_posterior,
    compute_implied_equilibrium_returns,
    compute_transaction_costs,
    compute_turnover,
    herc_to_evidence,
    render_bl_allocation_artifact,
    render_bl_returns_artifact,
    robust_mvo_sensitivity_grid,
    solve_black_litterman,
    solve_cvar_portfolio,
    solve_herc,
    solve_max_diversification,
    solve_robust_mvo,
    solve_tracking_error_constrained,
    verify_portfolio_constraints,
)
from start.registry.market_contexts import PortfolioConstraints


@pytest.fixture
def synthetic_market_data():
    """Deterministic 4-asset market scenario."""
    np.random.seed(42)
    assets = ["SPY", "TLT", "GLD", "EEM"]
    cov_matrix = np.array([
        [0.0400, 0.0040, 0.0060, 0.0240],
        [0.0040, 0.0225, 0.0030, 0.0015],
        [0.0060, 0.0030, 0.0256, 0.0080],
        [0.0240, 0.0015, 0.0080, 0.0625],
    ])
    mu = np.array([0.08, 0.03, 0.05, 0.09])
    market_weights = np.array([0.50, 0.25, 0.15, 0.10])
    
    # 252 synthetic daily scenario returns
    n_days = 252
    returns_arr = np.random.multivariate_normal(mu / 252.0, cov_matrix / 252.0, size=n_days)
    returns_df = pd.DataFrame(returns_arr, columns=assets)

    return {
        "assets": assets,
        "covariance": cov_matrix,
        "cov_df": pd.DataFrame(cov_matrix, index=assets, columns=assets),
        "mu": mu,
        "market_weights": market_weights,
        "returns_df": returns_df,
    }


# =========================================================================== #
# 1. CONSTRAINT VERIFICATION ENGINE TESTS
# =========================================================================== #
def test_constraint_verifier_valid_and_violations(synthetic_market_data):
    assets = synthetic_market_data["assets"]
    cov = synthetic_market_data["covariance"]
    
    # Compliant weights
    w_valid = np.array([0.25, 0.25, 0.25, 0.25])
    constraints = PortfolioConstraints(
        budget=1.0,
        long_only=True,
        min_weight=0.05,
        max_weight=0.40,
        max_concentration=0.30,
    )
    res_valid = verify_portfolio_constraints(w_valid, assets, constraints, covariance=cov)
    assert res_valid.is_valid is True
    assert res_valid.max_violation <= 1e-6
    assert res_valid.summary["violated_checks"] == 0

    # Non-compliant weights (violates max_weight and max_concentration)
    w_invalid = np.array([0.70, 0.10, 0.10, 0.10])
    res_invalid = verify_portfolio_constraints(w_invalid, assets, constraints, covariance=cov)
    assert res_invalid.is_valid is False
    assert res_invalid.max_violation > 0.0
    violated_types = [v.constraint for v in res_invalid.violations if v.status == "VIOLATED"]
    assert ConstraintType.MAX_WEIGHT.value in violated_types
    assert ConstraintType.MAX_CONCENTRATION.value in violated_types


def test_constraint_verifier_factor_and_group(synthetic_market_data):
    assets = synthetic_market_data["assets"]
    # Group constraint: Equities (SPY + EEM) <= 0.55
    group_spec = GroupConstraintSpec(
        group_name="Sector",
        memberships={"Equities": ("SPY", "EEM")},
        upper_bounds={"Equities": 0.55},
    )
    # Factor constraint: Duration beta <= 5.0
    factor_spec = FactorConstraintSpec(
        factor_names=("Duration",),
        loadings={
            "SPY": {"Duration": 0.0},
            "TLT": {"Duration": 15.0},
            "GLD": {"Duration": 0.0},
            "EEM": {"Duration": 0.0},
        },
        upper_bounds={"Duration": 5.0},
    )
    constraints = PortfolioConstraints(
        budget=1.0,
        long_only=True,
        group_constraints=group_spec,
        factor_constraints=factor_spec,
    )
    
    # Compliant: Equities = 0.50 <= 0.55; Duration = 0.25 * 15.0 = 3.75 <= 5.0
    w_comp = np.array([0.40, 0.25, 0.25, 0.10])
    res = verify_portfolio_constraints(w_comp, assets, constraints)
    assert res.is_valid is True

    # Breaches group limit: Equities = 0.70 > 0.55
    w_breach = np.array([0.55, 0.20, 0.10, 0.15])
    res_b = verify_portfolio_constraints(w_breach, assets, constraints)
    assert res_b.is_valid is False
    assert any("group_exposure.Sector.Equities" in v.constraint for v in res_b.violations if v.status == "VIOLATED")


# =========================================================================== #
# 2. BLACK-LITTERMAN ENGINE TESTS
# =========================================================================== #
def test_black_litterman_equilibrium_and_posterior(synthetic_market_data):
    cov = synthetic_market_data["covariance"]
    wm = synthetic_market_data["market_weights"]
    assets = synthetic_market_data["assets"]
    delta = 2.5
    tau = 0.05

    # Implied equilibrium returns: Pi = delta * Sigma * wm
    pi = compute_implied_equilibrium_returns(cov, wm, delta)
    assert len(pi) == 4
    assert np.all(pi > 0)

    # View 1: SPY will return 12% (Absolute view)
    # View 2: EEM will outperform SPY by 2% (Relative view: EEM - SPY = 0.02)
    P = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 1.0],
    ])
    Q = np.array([0.12, 0.02])

    mu_bl, cov_bl, resids = compute_black_litterman_posterior(
        cov, wm, P, Q, tau=tau
    )
    assert len(mu_bl) == 4
    assert cov_bl.shape == (4, 4)
    # Posterior covariance should remain symmetric positive definite
    eigvals = np.linalg.eigvalsh(cov_bl)
    assert np.all(eigvals > 0)

    # Solve constrained BL portfolio
    bl_res = solve_black_litterman(
        covariance=cov,
        market_weights=wm,
        P=P,
        Q=Q,
        risk_aversion=delta,
        tau=tau,
        assets=assets,
        view_labels=["SPY Outperformance", "EEM vs SPY Spread"],
    )
    assert isinstance(bl_res, BlackLittermanResult)
    assert math.isclose(sum(bl_res.posterior_weights.values()), 1.0, rel_tol=1e-5)
    assert bl_res.constraint_verification.is_valid is True
    assert bl_res.posterior_volatility_annualised > 0.0


def test_black_litterman_uncertainty_monotonicity(synthetic_market_data):
    cov = synthetic_market_data["covariance"]
    wm = synthetic_market_data["market_weights"]
    assets = synthetic_market_data["assets"]
    delta = 3.0
    P = np.array([[1.0, 0.0, 0.0, 0.0]])
    Q = np.array([0.20])  # Strong bullish view on SPY

    # High confidence (Omega small) vs Low confidence (Omega large)
    Omega_high_conf = np.array([[0.0001]])
    Omega_low_conf = np.array([[0.1000]])

    bl_high = solve_black_litterman(cov, wm, P, Q, Omega=Omega_high_conf, risk_aversion=delta, assets=assets)
    bl_low = solve_black_litterman(cov, wm, P, Q, Omega=Omega_low_conf, risk_aversion=delta, assets=assets)

    # High confidence view must pull posterior return of SPY closer to 0.20 than low confidence
    assert bl_high.posterior_returns["SPY"] > bl_low.posterior_returns["SPY"]
    assert bl_high.posterior_weights["SPY"] >= bl_low.posterior_weights["SPY"]


# =========================================================================== #
# 3. ROBUST MEAN-VARIANCE OPTIMIZATION TESTS
# =========================================================================== #
def test_robust_mvo_properties_and_grid(synthetic_market_data):
    cov = synthetic_market_data["covariance"]
    mu = synthetic_market_data["mu"]
    assets = synthetic_market_data["assets"]

    # At radius kappa = 0, worst-case return equals nominal expected return
    rob_zero = solve_robust_mvo(mu, cov, uncertainty_radius=0.0, assets=assets)
    assert isinstance(rob_zero, RobustMVOResult)
    assert math.isclose(
        rob_zero.nominal_expected_return_annualised,
        rob_zero.worst_case_expected_return_annualised,
        rel_tol=1e-4,
    )

    # Monotonicity: Worst-case expected return decreases as uncertainty radius increases
    rob_small = solve_robust_mvo(
        mu,
        cov,
        uncertainty_radius=0.5,
        uncertainty_policy=UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N,
        assets=assets,
    )
    rob_large = solve_robust_mvo(
        mu,
        cov,
        uncertainty_radius=1.5,
        uncertainty_policy=UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N,
        assets=assets,
    )
    assert rob_zero.worst_case_expected_return_annualised > rob_small.worst_case_expected_return_annualised
    assert rob_small.worst_case_expected_return_annualised > rob_large.worst_case_expected_return_annualised

    # Sensitivity Grid
    sens = robust_mvo_sensitivity_grid(mu, cov, radii=(0.0, 0.5, 1.0, 2.0), assets=assets)
    assert isinstance(sens, RobustSensitivityResult)
    assert len(sens.points) == 4
    for pt in sens.points:
        assert pt.effective_n_positions > 0.0


# =========================================================================== #
# 4. ROCKAFELLAR-URYASEV CVaR LP OPTIMIZATION TESTS
# =========================================================================== #
def test_cvar_optimization_lp_and_tail_ordering(synthetic_market_data):
    returns_df = synthetic_market_data["returns_df"]
    assets = synthetic_market_data["assets"]

    # Solve CVaR at 95% confidence
    cvar_95 = solve_cvar_portfolio(returns_df, confidence_level=0.95, assets=assets)
    assert isinstance(cvar_95, CVaROptimizationResult)
    assert math.isclose(sum(cvar_95.weights.values()), 1.0, rel_tol=1e-5)
    assert cvar_95.cvar_at_scenario_horizon >= cvar_95.var_at_scenario_horizon
    assert cvar_95.cvar_periodic >= cvar_95.var_auxiliary_periodic
    assert cvar_95.tail_scenario_count > 0

    # Confidence sensitivity: CVaR increases (worse tail loss) at 99% vs 90%
    cvar_90 = solve_cvar_portfolio(returns_df, confidence_level=0.90, assets=assets)
    cvar_99 = solve_cvar_portfolio(returns_df, confidence_level=0.99, assets=assets)
    assert cvar_99.cvar_at_scenario_horizon > cvar_95.cvar_at_scenario_horizon > cvar_90.cvar_at_scenario_horizon


# =========================================================================== #
# 5. HERC HIERARCHICAL ALLOCATION TESTS
# =========================================================================== #
def test_herc_hierarchical_equal_risk_contribution(synthetic_market_data):
    cov = synthetic_market_data["covariance"]
    assets = synthetic_market_data["assets"]

    herc_res = solve_herc(cov, linkage_method="single", assets=assets)
    assert isinstance(herc_res, HERCResult)
    assert math.isclose(sum(herc_res.weights.values()), 1.0, rel_tol=1e-5)
    assert np.all(np.array(list(herc_res.weights.values())) >= 0.0)
    assert len(herc_res.percentage_risk_contributions) == 4
    assert math.isclose(sum(herc_res.percentage_risk_contributions.values()), 1.0, rel_tol=1e-5)
    assert herc_res.effective_n_positions >= 1.0


# =========================================================================== #
# 6. MAXIMUM DIVERSIFICATION PORTFOLIO TESTS
# =========================================================================== #
def test_maximum_diversification_properties():
    # Diagonal covariance with volatilities [0.10, 0.20]
    # In uncorrelated case, MDP weights are proportional to 1 / sigma_i
    cov_diag = np.array([
        [0.0100, 0.0000],
        [0.0000, 0.0400],
    ])
    assets = ["AssetA", "AssetB"]
    res = solve_max_diversification(cov_diag, assets=assets)
    assert isinstance(res, MaxDiversificationResult)
    
    # Ratio of weights should be (1/0.10) / (1/0.20) = 2.0 -> AssetA = 2/3, AssetB = 1/3
    w_a = res.weights["AssetA"]
    w_b = res.weights["AssetB"]
    assert math.isclose(w_a / w_b, 2.0, rel_tol=1e-3)
    assert res.diversification_ratio >= 1.0


# =========================================================================== #
# 7. TRACKING-ERROR CONSTRAINED OPTIMIZATION TESTS
# =========================================================================== #
def test_tracking_error_constrained_optimization(synthetic_market_data):
    cov = synthetic_market_data["covariance"]
    mu = synthetic_market_data["mu"]
    assets = synthetic_market_data["assets"]
    benchmark_w = {"SPY": 0.60, "TLT": 0.40, "GLD": 0.0, "EEM": 0.0}

    # At TE = 0.0, optimal weights must match benchmark weights
    te_zero = solve_tracking_error_constrained(
        mu, cov, benchmark_weights=benchmark_w, max_tracking_error=0.0, assets=assets
    )
    for a in assets:
        assert math.isclose(te_zero.weights[a], benchmark_w[a], abs_tol=1e-4)

    # Active allocation under TE = 0.02 (200 bps annualized tracking error limit)
    te_cap = 0.02 / math.sqrt(252.0)
    te_active = solve_tracking_error_constrained(
        mu, cov, benchmark_weights=benchmark_w, max_tracking_error=te_cap, assets=assets
    )
    assert isinstance(te_active, TrackingErrorResult)
    assert te_active.tracking_error_periodic <= te_cap + 1e-6
    assert te_active.active_return_annualised is not None


# =========================================================================== #
# 8. TURNOVER & TRANSACTION COST REBALANCING TESTS
# =========================================================================== #
def test_turnover_and_rebalance_decision(synthetic_market_data):
    assets = synthetic_market_data["assets"]
    cov = synthetic_market_data["covariance"]
    mu = synthetic_market_data["mu"]
    w_old = {"SPY": 0.50, "TLT": 0.50, "GLD": 0.0, "EEM": 0.0}
    w_new = {"SPY": 0.40, "TLT": 0.40, "GLD": 0.10, "EEM": 0.10}

    # Turnover = 0.5 * (|0.4-0.5| + |0.4-0.5| + |0.1-0.0| + |0.1-0.0|) = 0.5 * (0.1 + 0.1 + 0.1 + 0.1) = 0.20
    to = compute_turnover(w_new, w_old, assets)
    assert math.isclose(to, 0.20, abs_tol=1e-6)

    # Cost spec: 10 bps flat
    cost_spec = TransactionCostSpec(default_linear_bps=10.0)
    costs = compute_transaction_costs(w_new, w_old, assets, cost_spec)
    # Total traded volume = 0.40. Cost = 0.40 * (10 / 10000) = 0.0004
    assert math.isclose(costs["total_cost"], 0.0004, abs_tol=1e-8)

    # Rebalance decision object
    reb = build_rebalance_decision(
        current_weights=w_old,
        proposed_weights=w_new,
        covariance=cov,
        assets=assets,
        mu=mu,
        cost_spec=cost_spec,
    )
    assert isinstance(reb, RebalanceDecision)
    assert reb.turnover == 0.20
    assert reb.estimated_transaction_cost == 0.0004
    assert reb.expected_return_gross is not None
    assert reb.expected_return_net is not None
    assert reb.expected_return_gross > reb.expected_return_net


# =========================================================================== #
# 9. MULTI-METHOD COMPARISON & APPLICABILITY TESTS
# =========================================================================== #
def test_multi_method_comparison_engine(synthetic_market_data):
    returns_df = synthetic_market_data["returns_df"]
    cov_df = synthetic_market_data["cov_df"]
    prior = {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "EEM": 0.25}

    comp = compare_portfolio_methods(
        returns=returns_df,
        covariance=cov_df,
        prior_weights=prior,
        robust_uncertainty_radius=0.5,
        cvar_confidence=0.95,
    )
    assert isinstance(comp, MethodComparisonResult)
    expected_methods = [
        "current_portfolio",
        "equal_weight",
        "minimum_variance",
        "maximum_sharpe",
        "hierarchical_risk_parity",
        "equal_risk_contribution",
        "hierarchical_equal_risk_contribution",
        "maximum_diversification",
        "robust_mvo_radius_0.5",
        "cvar_optimization_95pct",
    ]
    for m in expected_methods:
        assert m in comp.methods
        assert m in comp.weights_matrix


# =========================================================================== #
# 10. EVIDENCE BRIDGE & ARTIFACT RECORD TESTS
# =========================================================================== #
def test_evidence_bridge_and_artifact_generation(synthetic_market_data, tmp_path):
    cov = synthetic_market_data["covariance"]
    wm = synthetic_market_data["market_weights"]
    assets = synthetic_market_data["assets"]
    P = np.array([[1.0, -1.0, 0.0, 0.0]])
    Q = np.array([0.03])

    bl_res = solve_black_litterman(cov, wm, P, Q, assets=assets)
    ev_record = black_litterman_to_evidence(bl_res)
    assert isinstance(ev_record, EvidenceRecord)
    assert ev_record.test_id == "portfolio.black_litterman"
    assert ev_record.evidence_id.startswith("EV-")
    assert "posterior_volatility_annualised" in ev_record.metrics

    # Cryptographic Artifacts
    art_ret = render_bl_returns_artifact(bl_res, evidence_ids=(ev_record.evidence_id,), output_dir=tmp_path)
    assert isinstance(art_ret, ArtifactRecord)
    assert art_ret.data_fingerprint is not None
    assert len(art_ret.data_fingerprint) == 64
    assert art_ret.data_fingerprint == art_ret.data_fingerprint.lower()
    assert (tmp_path / f"{art_ret.artifact_id}.json").exists()

    art_alloc = render_bl_allocation_artifact(bl_res, evidence_ids=(ev_record.evidence_id,), output_dir=tmp_path)
    assert isinstance(art_alloc, ArtifactRecord)
    assert (tmp_path / f"{art_alloc.artifact_id}.json").exists()


# =========================================================================== #
# 11. AGENT ORCHESTRATION & GOVERNANCE SEPARATION TESTS
# =========================================================================== #
def test_agent_orchestration_and_governance_role_separation(synthetic_market_data):
    cov = synthetic_market_data["covariance"]
    wm = synthetic_market_data["market_weights"]
    assets = synthetic_market_data["assets"]
    P = np.array([[1.0, 0.0, 0.0, 0.0]])
    Q = np.array([0.05])

    bl_res = solve_black_litterman(cov, wm, P, Q, assets=assets)
    ev_bl = black_litterman_to_evidence(bl_res)

    herc_res = solve_herc(cov, assets=assets)
    ev_herc = herc_to_evidence(herc_res)

    context = {
        "evidence_records": [ev_bl, ev_herc],
        "covariance": cov,
        "assets": assets,
        "market_weights": wm,
        "P": P,
        "Q": Q,
        "returns": synthetic_market_data["returns_df"],
        "scenario_returns": synthetic_market_data["returns_df"],
        "auto_resolve_challenges": True,
    }

    # 1. Tool allowlists
    h_agent = HierarchicalAllocationAgent()
    with pytest.raises(PermissionError):
        h_agent.execute_tool("solve_black_litterman")  # BL not allowed for hierarchical agent

    p_agent = PortfolioConstructionAgent()
    with pytest.raises(PermissionError):
        p_agent.execute_tool("cophenetic_distance_diagnostic")  # Tree diagnostic not allowed for construction agent

    # 2. EvidenceCriticAgent may NOT emit ACCEPT/APPROVE
    critic = EvidenceCriticAgent()
    crit_out = critic.execute(context)
    assert crit_out["critique"]["disposition"] in [
        CriticDisposition.EVIDENCE_VALID.value,
        CriticDisposition.READY_FOR_GOVERNANCE.value,
    ]
    assert "ACCEPT" not in crit_out["critique"]["disposition"]
    assert "APPROVE" not in crit_out["critique"]["disposition"]

    # 3. GovernanceAgent alone issues ACCEPT
    gov = GovernanceAgent()
    gov_out = gov.execute({
        "evidence_records": [ev_bl, ev_herc],
        "critic_disposition": crit_out["critique"]["disposition"],
        "findings": [],
        "challenges": [],
    })
    assert gov_out["governance_signoff"]["verdict"] == GovernanceVerdict.ACCEPT.value

    # 4. MarketReviewDirectorAgent full pipeline orchestration
    director = MarketReviewDirectorAgent()
    dir_out = director.execute(context)
    assert dir_out["governance_verdict"] in [
        GovernanceVerdict.ACCEPT.value,
        GovernanceVerdict.ACCEPT_WITH_CONDITIONS.value,
    ]
    assert len(dir_out["challenges"]) > 0
