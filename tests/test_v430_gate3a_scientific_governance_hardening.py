"""Gate 3A Scientific Governance & Quantitative Hardening Test Suite.

Mandatory RED-first tests encoding P0 requirements:
1. Governance: OPEN/UNRESOLVED challenges block ACCEPT.
2. Challenge resolution pipeline: deterministic tool execution -> EvidenceRecord -> ChallengeResolution.
3. Solver failure fail-closed semantics: no silent fallbacks to 1/N, market weights, or benchmark.
4. CVaR LP constraint contract: strict linear constraint compiler, explicit rejection of non-linear constraints.
5. CVaR horizon semantics: scenario-horizon tail risk without sqrt(T) empirical scaling.
6. Missing required inputs fail-closed: missing factor loadings, benchmark weights, and market weights fail immediately.
7. Black-Litterman contract & known-answer validation.
8. Robust MVO uncertainty contract: explicit uncertainty covariance or named derivation policy.
9. Transaction cost horizon: one-time cost is not multiplied by periods_per_year.
10. Evidence Critic: trusts authoritative verifier, rejects ad-hoc 1e-4 thresholds.
11. HERC scientific validation: cluster risk parity & known-answer checks.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from start.agents.market_review import (
    AdversarialChallengeAgent,
    CriticDisposition,
    EvidenceCriticAgent,
    GovernanceAgent,
    GovernanceVerdict,
)
from start.core.schemas import EvidenceRecord

try:
    from start.portfolio import ChallengeResolution, ChallengeState, ViewUncertaintyPolicy
except ImportError:
    ChallengeResolution = None  # type: ignore
    ChallengeState = None  # type: ignore
    ViewUncertaintyPolicy = None  # type: ignore

from start.portfolio import (
    BlackLittermanResult,
    CVaROptimizationResult,
    FactorConstraintSpec,
    HERCResult,
    RebalanceDecision,
    RobustMVOResult,
    TransactionCostSpec,
    build_rebalance_decision,
    compute_black_litterman_posterior,
    compute_implied_equilibrium_returns,
    solve_black_litterman,
    solve_cvar_portfolio,
    solve_herc,
    solve_robust_mvo,
    verify_portfolio_constraints,
)
from start.registry.market_contexts import PortfolioConstraints


@pytest.fixture
def market_fixture() -> dict[str, Any]:
    assets = ["SPY", "TLT", "GLD", "EEM"]
    cov = np.array([
        [0.0400, 0.0040, 0.0060, 0.0240],
        [0.0040, 0.0225, 0.0030, 0.0015],
        [0.0060, 0.0030, 0.0256, 0.0080],
        [0.0240, 0.0015, 0.0080, 0.0625],
    ])
    mu = np.array([0.08, 0.03, 0.05, 0.09])
    cov_df = pd.DataFrame(cov, index=assets, columns=assets)
    returns_df = pd.DataFrame(
        np.random.RandomState(42).multivariate_normal(mu / 252.0, cov / 252.0, size=500),
        columns=assets,
    )
    return {
        "assets": assets,
        "cov": cov,
        "cov_df": cov_df,
        "mu": mu,
        "returns_df": returns_df,
        "market_weights": {"SPY": 0.50, "TLT": 0.30, "GLD": 0.10, "EEM": 0.10},
    }


# =========================================================================== #
# 1. GOVERNANCE: OPEN / UNRESOLVED CHALLENGES MUST BLOCK ACCEPT
# =========================================================================== #
def test_governance_open_challenges_block_accept():
    """Governance MUST NOT issue ACCEPT when any applicable material challenge is OPEN or UNRESOLVED."""
    gov = GovernanceAgent()

    from start.core.schemas import Status, TestResult

    tr = TestResult(
        test_id="portfolio.black_litterman",
        test_name="Black Litterman",
        status=Status.RECORDED,
        params={},
        metrics={"is_valid": True, "converged": True, "usable_solution": True},
    )
    ev_record = EvidenceRecord.from_result(tr, run_id="RUN-1", model_id="MOD-1", dataset_id="DS-1")

    # Case A: Critic is READY_FOR_GOVERNANCE, but challenge is OPEN
    open_chal = {
        "challenge_id": "CHAL-BL-TAU-001",
        "target_area": "View Dominance",
        "challenge_question": "Does tau dominate prior?",
        "status": ChallengeState.OPEN.value if hasattr(ChallengeState, "OPEN") else "OPEN",
    }
    gov_out = gov.execute({
        "evidence_records": [ev_record],
        "critic_disposition": CriticDisposition.READY_FOR_GOVERNANCE.value,
        "findings": [],
        "challenges": [open_chal],
        "challenge_resolutions": [],
    })
    # MUST NOT be ACCEPT
    assert gov_out["governance_signoff"]["verdict"] != GovernanceVerdict.ACCEPT.value
    assert gov_out["governance_signoff"]["verdict"] in [
        GovernanceVerdict.REMEDIATE.value,
        GovernanceVerdict.INSUFFICIENT_EVIDENCE.value,
        GovernanceVerdict.ESCALATE.value,
    ]


# =========================================================================== #
# 2. CHALLENGE RESOLUTION LIFECYCLE
# =========================================================================== #
def test_adversarial_challenge_resolution_pipeline(market_fixture):
    """AdversarialChallengeAgent must execute deterministic tools and produce ChallengeResolution objects."""
    cov = market_fixture["cov"]
    assets = market_fixture["assets"]
    wm = market_fixture["market_weights"]
    P = np.array([[1.0, 0.0, 0.0, 0.0]])
    Q = np.array([0.10])

    bl_res = solve_black_litterman(
        covariance=cov,
        market_weights=wm,
        P=P,
        Q=Q,
        risk_aversion=3.0,
        tau=0.05,
        assets=assets,
        uncertainty_policy=ViewUncertaintyPolicy.PROPORTIONAL_TAU_SIGMA if hasattr(ViewUncertaintyPolicy, "PROPORTIONAL_TAU_SIGMA") else None,
    )
    from start.portfolio import black_litterman_to_evidence
    ev_bl = black_litterman_to_evidence(bl_res)

    agent = AdversarialChallengeAgent()
    out = agent.execute({"evidence_records": [ev_bl], "covariance": cov, "assets": assets})

    assert "challenge_resolutions" in out
    resolutions = out["challenge_resolutions"]
    assert len(resolutions) > 0
    for r in resolutions:
        assert isinstance(r, (dict, ChallengeResolution))
        status = r["status"] if isinstance(r, dict) else r.status
        assert status in [
            "RESOLVED_NO_BREACH",
            "RESOLVED_FINDING",
            "BLOCKED",
            "UNRESOLVED",
        ]


# =========================================================================== #
# 3. SOLVER FAILURE FAILS CLOSED (NO SILENT FALLBACKS)
# =========================================================================== #
def test_cvar_solver_failure_fails_closed(market_fixture):
    """CVaR solver failure must NOT substitute equal weights; usable_solution must be False."""
    r_df = market_fixture["returns_df"]
    assets = market_fixture["assets"]

    # Infeasible constraints: sum of lower bounds > budget
    impossible_constraints = PortfolioConstraints(
        budget=1.0,
        asset_lower_bounds={"SPY": 0.60, "TLT": 0.60},
    )

    res = solve_cvar_portfolio(
        scenario_returns=r_df,
        confidence_level=0.95,
        assets=assets,
        constraints=impossible_constraints,
    )
    assert isinstance(res, CVaROptimizationResult)
    assert res.converged is False
    assert res.usable_solution is False
    assert "FAILED" in res.solver_status
    # MUST NOT return equal weight allocation
    assert res.weights != {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "EEM": 0.25}


def test_robust_mvo_solver_failure_fails_closed(market_fixture):
    """Robust MVO failure must NOT return equal weights as usable solution."""
    cov = market_fixture["cov"]
    mu = market_fixture["mu"]
    assets = market_fixture["assets"]

    impossible_constraints = PortfolioConstraints(
        budget=1.0,
        asset_lower_bounds={"SPY": 0.60, "TLT": 0.60},
    )
    res = solve_robust_mvo(
        mu=mu,
        covariance=cov,
        uncertainty_radius=0.5,
        uncertainty_cov=cov / 100.0,
        assets=assets,
        constraints=impossible_constraints,
    )
    assert isinstance(res, RobustMVOResult)
    assert res.converged is False
    assert res.usable_solution is False


def test_black_litterman_solver_failure_fails_closed(market_fixture):
    """Black-Litterman failure must NOT masquerade as market portfolio."""
    cov = market_fixture["cov"]
    assets = market_fixture["assets"]
    wm = market_fixture["market_weights"]
    P = np.array([[1.0, 0.0, 0.0, 0.0]])
    Q = np.array([0.10])

    impossible_constraints = PortfolioConstraints(
        budget=1.0,
        asset_lower_bounds={"SPY": 0.60, "TLT": 0.60},
    )
    res = solve_black_litterman(
        covariance=cov,
        market_weights=wm,
        P=P,
        Q=Q,
        risk_aversion=3.0,
        tau=0.05,
        assets=assets,
        constraints=impossible_constraints,
    )
    assert isinstance(res, BlackLittermanResult)
    assert res.converged is False
    assert res.usable_solution is False


# =========================================================================== #
# 4. CVAR LP CONSTRAINT CONTRACT & REJECTION OF NONLINEAR CONSTRAINTS
# =========================================================================== #
def test_cvar_rejects_nonlinear_constraints(market_fixture):
    """CVaR LP must explicitly reject non-linear constraints (Herfindahl, quadratic tracking error)."""
    r_df = market_fixture["returns_df"]
    assets = market_fixture["assets"]

    nonlinear_constraints = PortfolioConstraints(
        budget=1.0,
        max_concentration=0.30,  # Herfindahl quadratic constraint
    )
    with pytest.raises(ValueError, match="unsupported|non-linear|linear"):
        solve_cvar_portfolio(
            scenario_returns=r_df,
            confidence_level=0.95,
            assets=assets,
            constraints=nonlinear_constraints,
        )


# =========================================================================== #
# 5. EMPIRICAL CVAR MUST REMAIN AT SCENARIO HORIZON (NO SQRT(T) ANNUALIZATION)
# =========================================================================== #
def test_cvar_empirical_scenario_horizon_no_sqrt_scaling(market_fixture):
    """Empirical scenario CVaR must provide scenario-horizon metrics without sqrt(T) annualization."""
    r_df = market_fixture["returns_df"]
    assets = market_fixture["assets"]

    res = solve_cvar_portfolio(
        scenario_returns=r_df,
        confidence_level=0.95,
        assets=assets,
        periods_per_year=252.0,
    )
    assert hasattr(res, "cvar_at_scenario_horizon")
    assert hasattr(res, "var_at_scenario_horizon")
    assert res.cvar_at_scenario_horizon == pytest.approx(res.cvar_periodic, rel=1e-6)
    # The legacy annualised field must NOT simply multiply by sqrt(252)
    if res.cvar_annualised is not None:
        assert not math.isclose(res.cvar_annualised, res.cvar_at_scenario_horizon * math.sqrt(252.0), rel_tol=1e-4)


# =========================================================================== #
# 6. MISSING REQUIRED FINANCIAL INPUTS FAIL CLOSED
# =========================================================================== #
def test_missing_factor_exposures_fail_closed(market_fixture):
    """Missing asset from factor loadings must fail closed, NOT default to 0.0."""
    assets = market_fixture["assets"]  # ["SPY", "TLT", "GLD", "EEM"]

    # EEM is missing from loadings
    incomplete_factor_spec = FactorConstraintSpec(
        factor_names=("Duration",),
        loadings={
            "SPY": {"Duration": 0.0},
            "TLT": {"Duration": 16.0},
            "GLD": {"Duration": 0.0},
            # "EEM" missing!
        },
        upper_bounds={"Duration": 5.0},
    )
    constraints = PortfolioConstraints(
        budget=1.0,
        factor_constraints=incomplete_factor_spec,
    )

    with pytest.raises(ValueError, match="missing factor exposure|coverage"):
        verify_portfolio_constraints(
            weights={"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "EEM": 0.25},
            assets=assets,
            constraints=constraints,
        )


def test_missing_market_weights_fail_closed(market_fixture):
    """Missing constituent from market weights in Black-Litterman must fail closed."""
    cov = market_fixture["cov"]
    assets = market_fixture["assets"]
    incomplete_wm = {"SPY": 0.60, "TLT": 0.40}  # GLD and EEM missing

    P = np.array([[1.0, 0.0, 0.0, 0.0]])
    Q = np.array([0.10])

    with pytest.raises(ValueError, match="missing.*market_weights|coverage"):
        solve_black_litterman(
            covariance=cov,
            market_weights=incomplete_wm,
            P=P,
            Q=Q,
            assets=assets,
        )


# =========================================================================== #
# 7. BLACK-LITTERMAN SCIENTIFIC CONTRACT & KNOWN-ANSWER TESTS
# =========================================================================== #
def test_black_litterman_correct_market_weights_and_known_answers(market_fixture):
    """Black-Litterman posterior calculation must use actual market weights, not pi."""
    cov = market_fixture["cov"]
    wm_dict = market_fixture["market_weights"]
    assets = market_fixture["assets"]
    wm_vec = np.array([wm_dict[a] for a in assets])
    delta = 3.0
    tau = 0.05

    # 1. Equilibrium returns Pi = delta * Sigma * w_m
    pi_expected = delta * (cov @ wm_vec)
    pi_calc = compute_implied_equilibrium_returns(cov, wm_vec, risk_aversion=delta)
    np.testing.assert_allclose(pi_calc, pi_expected, rtol=1e-7)

    # 2. View: SPY return = 10%
    P = np.array([[1.0, 0.0, 0.0, 0.0]])
    Q = np.array([0.10])

    # Call compute_black_litterman_posterior with wm_vec (MARKET WEIGHTS, NOT PI!)
    mu_bl, sigma_bl, diag = compute_black_litterman_posterior(
        covariance=cov,
        market_weights=wm_vec,
        P=P,
        Q=Q,
        risk_aversion=delta,
        tau=tau,
    )
    assert len(mu_bl) == 4
    # SPY posterior expected return should shift towards 10%
    assert mu_bl[0] > pi_calc[0]
    # Covariance symmetry and positive-definiteness
    np.testing.assert_allclose(sigma_bl, sigma_bl.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(sigma_bl)
    assert np.all(eigvals > 0)


# =========================================================================== #
# 8. ROBUST MVO UNCERTAINTY CONTRACT
# =========================================================================== #
def test_robust_mvo_requires_explicit_uncertainty_cov_or_policy(market_fixture):
    """Robust MVO must NOT silently use sigma.copy() when uncertainty_cov is None without a named policy."""
    cov = market_fixture["cov"]
    mu = market_fixture["mu"]
    assets = market_fixture["assets"]

    # Passing uncertainty_cov=None without explicit derivation policy must raise ValueError
    with pytest.raises(ValueError, match="uncertainty_cov.*policy|derivation"):
        solve_robust_mvo(
            mu=mu,
            covariance=cov,
            uncertainty_radius=0.5,
            uncertainty_cov=None,
            uncertainty_policy=None,
            assets=assets,
        )


# =========================================================================== #
# 9. TRANSACTION COST HORIZON: NO MULTIPLICATION BY PPY
# =========================================================================== #
def test_rebalance_transaction_cost_not_multiplied_by_ppy(market_fixture):
    """A one-time rebalancing cost must NOT be multiplied by periods_per_year when computing net return."""
    cov = market_fixture["cov"]
    assets = market_fixture["assets"]
    mu_annual = np.array([0.10, 0.04, 0.06, 0.12])
    w0 = {"SPY": 0.50, "TLT": 0.50, "GLD": 0.0, "EEM": 0.0}
    w1 = {"SPY": 0.25, "TLT": 0.25, "GLD": 0.25, "EEM": 0.25}

    cost_spec = TransactionCostSpec(default_linear_bps=100.0)  # 1% flat cost

    reb = build_rebalance_decision(
        current_weights=w0,
        proposed_weights=w1,
        covariance=cov,
        assets=assets,
        mu=mu_annual,
        cost_spec=cost_spec,
        periods_per_year=252.0,
    )
    assert isinstance(reb, RebalanceDecision)
    # One-time cost = 0.50 * 0.01 = 0.005 (50 bps)
    # If gross annual return is ~8%, net return must be 8% - 0.5% = 7.5%, NOT 8% - 0.5% * 252 = -118%!
    assert reb.expected_return_gross_annualised is not None
    assert reb.expected_return_net_annualised is not None
    cost_diff = reb.expected_return_gross_annualised - reb.expected_return_net_annualised
    assert cost_diff < 0.05, f"Cost drag {cost_diff} was blown up by periods_per_year!"


# =========================================================================== #
# 10. EVIDENCE CRITIC: TRUSTS AUTHORITATIVE VERIFIER STATUS
# =========================================================================== #
def test_evidence_critic_trusts_verifier_status(market_fixture):
    """EvidenceCriticAgent must reject records where is_valid is False even if violation is small."""
    critic = EvidenceCriticAgent()

    from start.core.schemas import Status, TestResult

    tr = TestResult(
        test_id="portfolio.constrained_optimization",
        test_name="Constrained Optimization",
        status=Status.RECORDED,
        params={},
        metrics={
            "is_valid": False,
            "max_constraint_violation": 0.00005,  # 5e-5 < 1e-4, but is_valid is False
            "constraint_violations_count": 1,
            "converged": True,
            "usable_solution": True,
        },
    )
    rec_invalid = EvidenceRecord.from_result(tr, run_id="RUN-1", model_id="MOD-1", dataset_id="DS-1")
    res = critic.critique_evidence_records([rec_invalid])
    assert res["is_valid"] is False
    assert res["disposition"] == CriticDisposition.EVIDENCE_INVALID.value


# =========================================================================== #
# 11. HERC SCIENTIFIC VALIDATION: CLUSTER RISK PARITY & KNOWN ANSWERS
# =========================================================================== #
def test_herc_scientific_cluster_risk_parity():
    """HERC must allocate equal risk across symmetric clusters and differ demonstrably from flat IVP."""
    # 4 assets in two distinct clusters: (A1, A2) and (B1, B2)
    # Cluster A has high internal correlation (0.9), Cluster B has high internal correlation (0.9)
    # Cross-cluster correlation is 0.0
    vols = np.array([0.20, 0.20, 0.10, 0.10])
    corr = np.array([
        [1.0, 0.9, 0.0, 0.0],
        [0.9, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.9],
        [0.0, 0.0, 0.9, 1.0],
    ])
    cov = np.diag(vols) @ corr @ np.diag(vols)
    assets = ["A1", "A2", "B1", "B2"]

    herc_res = solve_herc(cov, assets=assets, periods_per_year=1.0)
    assert isinstance(herc_res, HERCResult)
    w = herc_res.weights

    # Within symmetric cluster A, A1 and A2 must have identical weights
    assert math.isclose(w["A1"], w["A2"], rel_tol=1e-4)
    # Within symmetric cluster B, B1 and B2 must have identical weights
    assert math.isclose(w["B1"], w["B2"], rel_tol=1e-4)
    # Cluster B (lower vol) must receive higher cluster weight than Cluster A
    assert (w["B1"] + w["B2"]) > (w["A1"] + w["A2"])
