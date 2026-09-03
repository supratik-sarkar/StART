"""Gate 4 Test Suite: Agentic Multi-Agent Vertical Slice & Adversarial Governance.

Comprehensive verification of:
1. CovarianceRiskAgent, FactorRiskAttributionAgent, and FactorDataIntegrityChecker.
2. Strict agent tool allowlist enforcement.
3. Adversarial challenge formulation and deterministic tool resolution.
4. Pattern B subordinate diagnostic EvidenceRecord generation without orphan metrics.
5. GovernanceAgent sign-off determination and MarketReviewDirectorAgent orchestration.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from start.agents.market_review import (
    AdversarialChallengeAgent,
    CovarianceRiskAgent,
    CriticDisposition,
    FactorDataIntegrityChecker,
    FactorRiskAttributionAgent,
    GovernanceAgent,
    GovernanceVerdict,
    MarketReviewDirectorAgent,
)
from start.portfolio.contracts import (
    ChallengeState,
    PSDRepairMethod,
)
from start.portfolio.covariance import (
    diagnose_covariance,
    repair_psd_covariance,
)
from start.portfolio.evidence_bridge import (
    active_risk_decomp_to_evidence,
    covariance_diagnostics_to_evidence,
    factor_risk_decomp_to_evidence,
    factor_risk_model_to_evidence,
    psd_repair_to_evidence,
)
from start.portfolio.factor_risk import (
    build_linear_factor_model,
    decompose_active_risk,
    decompose_factor_risk,
)


@pytest.fixture
def agentic_market_context() -> dict[str, Any]:
    """Hydrated multi-agent market context with covariance, factor model, and attribution evidence."""
    assets = ["AAPL", "MSFT", "XOM", "JNJ"]
    factors = ["Market", "Value"]

    # 1. Clean covariance
    stds = np.array([0.15, 0.20, 0.25, 0.30])
    corr = np.array([
        [1.00, 0.50, 0.30, 0.10],
        [0.50, 1.00, 0.40, 0.20],
        [0.30, 0.40, 1.00, 0.35],
        [0.10, 0.20, 0.35, 1.00],
    ])
    cov = np.outer(stds, stds) * corr
    diag = diagnose_covariance(cov, assets=assets)
    ev_cov = covariance_diagnostics_to_evidence(diag)

    # 2. Indefinite covariance requiring explicit repair
    indef_mat = np.array([
        [1.00, 0.90, 0.90],
        [0.90, 1.00, 0.90],
        [0.90, 0.90, 0.10],
    ])
    diag_indef = diagnose_covariance(indef_mat, assets=["A0", "A1", "A2"])
    ev_indef = covariance_diagnostics_to_evidence(diag_indef)

    repair = repair_psd_covariance(indef_mat, method=PSDRepairMethod.HIGHAM_NEAREST_CORRELATION)
    ev_repair = psd_repair_to_evidence(repair)

    # 3. Factor Model & Decompositions
    B_df = pd.DataFrame(
        [
            [1.20, -0.30],
            [1.10, -0.20],
            [0.70, 0.80],
            [0.60, 0.40],
        ],
        index=assets,
        columns=factors,
    )
    F_df = pd.DataFrame([[0.04, -0.005], [-0.005, 0.02]], index=factors, columns=factors)
    D_dict = {"AAPL": 0.03, "MSFT": 0.025, "XOM": 0.02, "JNJ": 0.015}
    pw = {"AAPL": 0.30, "MSFT": 0.30, "XOM": 0.20, "JNJ": 0.20}
    bw = {"AAPL": 0.25, "MSFT": 0.25, "XOM": 0.25, "JNJ": 0.25}

    frm = build_linear_factor_model(B_df, F_df, D_dict)
    frd = decompose_factor_risk(pw, frm)
    ard = decompose_active_risk(pw, bw, frm)

    ev_frm = factor_risk_model_to_evidence(frm)
    ev_frd = factor_risk_decomp_to_evidence(frd)
    ev_ard = active_risk_decomp_to_evidence(ard)

    evidence_records = [ev_cov, ev_indef, ev_repair, ev_frm, ev_frd, ev_ard]

    return {
        "assets": assets,
        "factors": factors,
        "covariance": cov,
        "exposures": B_df,
        "factor_cov": F_df,
        "specific_var": D_dict,
        "weights": pw,
        "benchmark_weights": bw,
        "factor_model": frm,
        "evidence_records": evidence_records,
    }


def test_covariance_risk_agent_execution(agentic_market_context):
    """CovarianceRiskAgent audits covariance records and identifies non-PSD input."""
    agent = CovarianceRiskAgent()
    out = agent.execute(agentic_market_context)

    assert out["status"] == "completed"
    assert out["agent"] == "Covariance Risk Agent"
    assert len(out["findings"]) >= 2

    # Check for non-PSD material warning finding
    severities = [f["severity"] for f in out["findings"]]
    assert "material_warning" in severities
    assert "info" in severities


def test_covariance_risk_agent_tool_allowlist():
    """CovarianceRiskAgent enforces strict tool allowlist."""
    agent = CovarianceRiskAgent()
    assert "diagnose_covariance" in agent.ALLOWED_TOOLS
    assert "repair_psd_covariance" in agent.ALLOWED_TOOLS

    with pytest.raises(PermissionError, match="not in the allowed toolset"):
        agent.execute_tool("solve_black_litterman")


def test_factor_risk_attribution_agent_execution(agentic_market_context):
    """FactorRiskAttributionAgent audits factor risk and tracking error evidence."""
    agent = FactorRiskAttributionAgent()
    out = agent.execute(agentic_market_context)

    assert out["status"] == "completed"
    assert out["agent"] == "Factor Risk & Attribution Agent"
    assert len(out["findings"]) >= 2

    titles = [f["title"] for f in out["findings"]]
    assert "Factor Risk Variance Decomposition Review" in titles
    assert "Active Risk & Tracking Error Review" in titles


def test_factor_data_integrity_checker(agentic_market_context):
    """FactorDataIntegrityChecker runs deterministic validation and emits EvidenceRecord."""
    checker = FactorDataIntegrityChecker()
    out = checker.execute(agentic_market_context)

    assert out["status"] == "completed"
    assert out["integrity_result"]["is_valid"] is True
    assert out["evidence_record"].test_id == "factor_risk.data_integrity"


def test_adversarial_challenge_agent_gate4_formulation_and_resolution(agentic_market_context):
    """AdversarialChallengeAgent issues and resolves Gate 4 challenges."""
    agent = AdversarialChallengeAgent()
    out = agent.execute(agentic_market_context)

    assert out["status"] == "completed"
    challenges = out["challenges"]
    resolutions = out["resolutions"]

    assert len(challenges) >= 3
    assert len(resolutions) == len(challenges)

    chal_types = [c["challenge_id"] for c in challenges]
    assert any("COV" in cid for cid in chal_types)
    assert any("FACTOR" in cid for cid in chal_types)

    # Verify subordinate diagnostic EvidenceRecords were generated
    for res in resolutions:
        assert res["status"] in (
            ChallengeState.RESOLVED_EVIDENCE_ONLY.value,
            ChallengeState.RESOLVED_NO_BREACH.value,
            ChallengeState.RESOLVED_FINDING.value,
        )
        assert len(res["generated_evidence_ids"]) == 1


def test_governance_agent_evaluates_gate4_signoff(agentic_market_context):
    """GovernanceAgent issues ACCEPT_WITH_CONDITIONS when evidence-only challenges are present."""
    gov = GovernanceAgent()
    adv_agent = AdversarialChallengeAgent()
    adv_out = adv_agent.execute(agentic_market_context)

    cov_agent = CovarianceRiskAgent()
    cov_out = cov_agent.execute(agentic_market_context)

    signoff = gov.evaluate_signoff(
        critic_disposition=CriticDisposition.READY_FOR_GOVERNANCE.value,
        challenges=adv_out["challenges"],
        findings=cov_out["findings"],
        records=agentic_market_context["evidence_records"],
        resolutions=adv_out["resolutions"],
    )

    # Because challenges have decision_criterion=NONE, disposition is ACCEPT_WITH_CONDITIONS
    assert signoff["verdict"] == GovernanceVerdict.ACCEPT_WITH_CONDITIONS.value
    assert len(signoff["conditions"]) > 0


def test_market_review_director_end_to_end(agentic_market_context):
    """MarketReviewDirectorAgent coordinates full multi-agent review across all specialists."""
    director = MarketReviewDirectorAgent()
    out = director.execute(agentic_market_context)

    assert out["status"] == "orchestrated"
    assert out["director"] == "Market Review Director Agent"
    assert out["findings_count"] > 0
    assert out["challenges_count"] > 0
    assert out["resolutions_count"] > 0
    assert out["critic_disposition"] == CriticDisposition.READY_FOR_GOVERNANCE.value
    assert out["governance_verdict"] in (
        GovernanceVerdict.ACCEPT.value,
        GovernanceVerdict.ACCEPT_WITH_CONDITIONS.value,
    )
