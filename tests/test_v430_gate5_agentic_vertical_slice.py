"""Focused unit tests for Gate 5 Multi-Agent Vertical Slice & Pre-Flight Warning Sanity.

Verifies:
1. TailRiskAgent execution and evidence-constrained qualitative synthesis.
2. AdversarialChallengeAgent formulation and deterministic resolution of CHAL-TAIL-* challenges.
3. Complete MarketReviewDirectorAgent multi-specialist orchestration.
4. Pre-Flight Warning Debt Resolution: Deliberately indefinite covariance fixture does NOT trigger
   unexpected RuntimeWarning: covariance is not symmetric positive-semidefinite during simulation.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from start.agents.market_review import (
    AdversarialChallengeAgent,
    MarketReviewDirectorAgent,
    TailRiskAgent,
)
from start.portfolio.evidence_bridge import (
    tail_backtest_to_evidence,
    tail_risk_estimate_to_evidence,
)
from start.portfolio.tail_risk import (
    compute_historical_var_es,
    run_comprehensive_tail_backtest,
)


def test_tail_risk_agent_execution_and_findings() -> None:
    """TailRiskAgent must inspect evidence records and emit deterministic, evidence-backed findings."""
    # Create evidence records: 1 backtest with clustering, 1 thin-tail ES estimate
    pnl = np.zeros(250)
    pnl[50:54] = -2.0  # 4 clustered exceptions
    var = np.ones(250)
    backtest = run_comprehensive_tail_backtest(pnl, var, var_confidence=0.99, test_significance=0.05)
    ev_backtest = tail_backtest_to_evidence(backtest)

    # Thin tail ES estimate (10 observations at 0.99 confidence)
    thin_losses = np.arange(1.0, 11.0)
    thin_es = compute_historical_var_es(thin_losses, confidence=0.99)
    ev_thin = tail_risk_estimate_to_evidence(thin_es)

    agent = TailRiskAgent()
    out = agent.execute({"evidence_records": [ev_backtest, ev_thin]})

    assert out["status"] == "completed"
    assessment = out["assessment"]
    assert assessment["has_independence_rejection"] is True
    assert assessment["has_thin_tail_support"] is True

    # Verify findings contain exact citations and correct failure-to-reject phrasing
    findings = out["findings"]
    statements = [f["statement"] for f in findings]
    assert any("does not reject unconditional coverage" in s for s in statements)
    assert any("rejects exception independence" in s for s in statements)
    assert any("thin tail support" in s for s in statements)


def test_adversarial_challenge_formulation_and_resolution() -> None:
    """AdversarialChallengeAgent must formulate CHAL-TAIL-* challenges and resolve them via allowed tools."""
    pnl = np.zeros(250)
    pnl[10] = -2.0
    pnl[20] = -2.0
    var = np.ones(250)
    backtest = run_comprehensive_tail_backtest(pnl, var, var_confidence=0.99, test_significance=0.05)
    ev_backtest = tail_backtest_to_evidence(backtest)

    challenger = AdversarialChallengeAgent()
    challenges = challenger.formulate_portfolio_challenges([ev_backtest])

    tail_chal_ids = [c.challenge_id for c in challenges if "TAIL" in c.challenge_id]
    assert len(tail_chal_ids) >= 3
    assert any("UNCONDITIONAL-COVERAGE" in cid for cid in tail_chal_ids)
    assert any("INDEPENDENCE" in cid for cid in tail_chal_ids)
    assert any("SEVERITY" in cid for cid in tail_chal_ids)

    # Resolve challenges
    context = {
        "pnl": pnl,
        "var_series": var,
        "var_confidence": 0.99,
        "test_significance": 0.05,
    }
    resolutions = challenger.resolve_challenges(challenges, context)
    assert all(r.status != "BLOCKED" for r in resolutions)


def test_market_review_director_full_orchestration() -> None:
    """MarketReviewDirectorAgent orchestrates specialists, challenger, critic, and governance."""
    assets = ["A", "B", "C"]
    cov = np.array(
        [
            [0.04, 0.01, 0.01],
            [0.01, 0.09, 0.02],
            [0.01, 0.02, 0.16],
        ]
    )
    rng = np.random.RandomState(42)
    rets = rng.multivariate_normal(np.zeros(3), cov, size=250)
    rets_df = pd.DataFrame(rets, columns=assets)
    weights = {"A": 0.4, "B": 0.3, "C": 0.3}

    pnl = rng.normal(0, 0.01, 250)
    var = np.full(250, 0.02)
    backtest = run_comprehensive_tail_backtest(pnl, var, var_confidence=0.95, test_significance=0.05)
    ev_backtest = tail_backtest_to_evidence(backtest)

    context = {
        "evidence_records": [ev_backtest],
        "returns": rets_df,
        "covariance": cov,
        "weights": weights,
        "pnl": pnl,
        "var_series": var,
        "var_confidence": 0.95,
        "test_significance": 0.05,
        "auto_resolve_challenges": True,
    }

    director = MarketReviewDirectorAgent()
    out = director.execute(context)

    assert out["status"] == "orchestrated"
    assert out["findings_count"] > 0
    assert out["challenges_count"] > 0
    assert out["governance_verdict"] in ("ACCEPT", "ACCEPT_WITH_CONDITIONS", "REMEDIATE")


def test_pre_flight_non_psd_covariance_simulation_warning_fix() -> None:
    """Pre-flight warning debt fix: Indefinite covariance must not raise RuntimeWarning in multivariate_normal."""
    # Indefinite 3x3 covariance matrix
    raw_indefinite_cov = np.array(
        [
            [1.00, 0.90, 0.90],
            [0.90, 1.00, 0.90],
            [0.90, 0.90, 0.10],
        ]
    )

    challenger = AdversarialChallengeAgent()
    # Mock challenge requiring returns where context only provides raw indefinite covariance
    challenge_dict = {
        "challenge_id": "CHAL-COV-SIM-TEST",
        "required_tool": "compare_covariance_estimators",
        "parameters": {"estimators": ("empirical", "ledoit_wolf")},
    }
    context = {
        "covariance": raw_indefinite_cov,
        "assets": ["A", "B", "C"],
    }

    # Catch any RuntimeWarning during challenge resolution
    with warnings.catch_warnings(record=True) as recorded_warnings:
        warnings.simplefilter("always", RuntimeWarning)
        res = challenger.resolve_challenge(challenge_dict, context)

    # Assert no RuntimeWarning regarding covariance not being symmetric positive-semidefinite was emitted
    psd_warnings = [
        w for w in recorded_warnings if "covariance is not symmetric positive-semidefinite" in str(w.message)
    ]
    assert len(psd_warnings) == 0, f"Unexpected RuntimeWarning encountered: {psd_warnings}"
    assert res.status != "BLOCKED"
