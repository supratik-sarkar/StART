"""Known-answer and integration tests for Cross-Analytical Committee (Combined Gate 7-9 Slice A)."""

from __future__ import annotations

import pytest

from start.agents.committee import CrossAnalyticalCommittee
from start.consensus.cross_analytical import (
    eval_factor_exposure_vs_scenario_alignment,
    eval_optimization_covariance_sensitivity,
    eval_reconciliation_identity_contradiction,
    eval_var_frequency_vs_independence,
)
from start.core.schemas import EvidenceRecord, Status
from start.evidence.claims import ClaimStatus, ClaimType
from start.evidence.graph import RelationshipType


def _make_evidence(
    test_id: str,
    name: str,
    metrics: dict,
    params: dict | None = None,
    status: Status = Status.PASS,
    artifact_hash: str = "HASH-SYNTH",
) -> EvidenceRecord:
    return EvidenceRecord(
        test_id=test_id,
        test_name=name,
        model_id="MOD-SYNTH",
        dataset_id="DS-SYNTH",
        run_id="RUN-SYNTH",
        metrics=metrics,
        params=params or {},
        status=status,
        interpretation="Synthetic test evidence.",
        limitations=["Deterministic test fixture."],
        input_artifact_hash=artifact_hash,
    )


def test_var_frequency_vs_independence_known_answer():
    """Verify Kupiec non-rejection vs Christoffersen rejection generates UNRESOLVED_RISK and CHALLENGES edge."""
    # Canonical showcase: Kupiec non-rejection (p > gamma_test) + Christoffersen rejection (p < gamma_test)
    kupiec_rec = _make_evidence(
        "traded_risk.kupiec_pof",
        "Kupiec POF",
        {"reject_unconditional_coverage": False, "p_value": 0.35, "gamma_test": 0.05},
        params={"gamma_test": 0.05},
    )
    christoffersen_rec = _make_evidence(
        "traded_risk.christoffersen_independence",
        "Christoffersen Independence",
        {"reject_independence": True, "p_value": 0.008, "gamma_test": 0.05},
        params={"gamma_test": 0.05},
    )

    claim, edges = eval_var_frequency_vs_independence(kupiec_rec, christoffersen_rec)

    assert claim.claim_type == ClaimType.UNRESOLVED_RISK
    assert claim.status == ClaimStatus.EVIDENCE_ONLY
    assert (
        "Unconditional coverage frequency adequacy does not imply exception independence" in claim.statement
    )
    assert len(edges) == 1
    assert edges[0].relation == RelationshipType.CHALLENGES
    assert edges[0].source_id == christoffersen_rec.evidence_id
    assert edges[0].target_id == kupiec_rec.evidence_id


def test_attribution_vs_factor_risk_cannot_be_reconciled_numerically():
    """RED Scientific Test: Verify return-valued attribution metric and variance-valued risk metric cannot be numerically reconciled."""
    from start.consensus.cross_analytical import eval_attribution_vs_factor_risk

    # Return attribution in decimal return units
    attr_rec = _make_evidence(
        "attribution.brinson",
        "Brinson Performance Attribution",
        {"active_return": 0.025, "reconciliation_error": 0.0},
        params={"factor_labels": ["MKT", "SMB", "HML"]},
    )
    # Factor risk decomposition in variance units (w^T \Sigma w)
    risk_rec = _make_evidence(
        "factor_risk.decomposition",
        "Factor Risk Decomposition",
        {"total_factor_variance": 0.040, "specific_variance": 0.010},
        params={"factor_labels": ["MKT", "SMB", "HML"]},
    )

    claim, edges = eval_attribution_vs_factor_risk(attr_rec, risk_rec)

    # Must be typed as DEPENDENCY or SUPPORTS, NOT algebraic RECONCILIATION between return and variance
    assert claim.claim_type == ClaimType.DEPENDENCY
    assert claim.status == ClaimStatus.VERIFIED
    assert edges[0].relation == RelationshipType.DEPENDS_ON
    # Explicit limitation noting dimensional incompatibility
    assert any("dimensionally distinct" in lim for lim in claim.limitations)


def test_reverse_stress_distance_is_non_normative_without_criterion():
    """Verify changing numerical reverse-stress distance alone does NOT create a normative severity classification without criterion."""
    from start.consensus.cross_analytical import eval_var_vs_reverse_stress

    var_rec = _make_evidence("traded_risk.var", "Parametric VaR", {"var_estimate": 0.08})
    # Small distance scenario
    rev_small = _make_evidence(
        "scenario.reverse_stress",
        "Reverse Stress Small",
        {"target_loss": 0.20, "minimum_distance": 0.05},
    )
    # Large distance scenario
    rev_large = _make_evidence(
        "scenario.reverse_stress",
        "Reverse Stress Large",
        {"target_loss": 0.20, "minimum_distance": 5.50},
    )

    claim_small, _ = eval_var_vs_reverse_stress(var_rec, rev_small)
    claim_large, _ = eval_var_vs_reverse_stress(var_rec, rev_large)

    # Both must remain EVIDENCE_ONLY and non-normative when criterion source is NONE
    assert claim_small.status == ClaimStatus.EVIDENCE_ONLY
    assert claim_large.status == ClaimStatus.EVIDENCE_ONLY
    assert claim_small.claim_type == ClaimType.UNRESOLVED_RISK
    assert claim_large.claim_type == ClaimType.UNRESOLVED_RISK

    # Neither statement should contain un-anchored normative labels ("severe", "unacceptable", "fragile")
    assert "severe tail vulnerability" not in claim_small.statement
    assert "severe tail vulnerability" not in claim_large.statement
    assert "unacceptable" not in claim_small.statement
    assert "fragile" not in claim_small.statement


def test_optimization_covariance_sensitivity_non_normative():
    """Verify optimization turnover across covariance models is typed as SENSITIVITY without fabricated threshold."""
    sample_rec = _make_evidence("covariance.empirical", "Sample Covariance", {"is_psd": True})
    lw_rec = _make_evidence("covariance.ledoit_wolf", "Ledoit-Wolf", {"shrinkage": 0.25})

    w_sample = {"AAPL": 0.70, "MSFT": 0.30}
    w_lw = {"AAPL": 0.50, "MSFT": 0.50}

    claim, edges = eval_optimization_covariance_sensitivity(sample_rec, lw_rec, w_sample, w_lw)

    assert claim.claim_type == ClaimType.SENSITIVITY
    assert claim.status == ClaimStatus.EVIDENCE_ONLY
    assert pytest.approx(claim.payload["turnover"], abs=1e-5) == 0.20
    assert "without external policy threshold" in claim.statement
    assert len(edges) == 1
    assert edges[0].relation == RelationshipType.ALTERNATIVE_METHOD


def test_factor_exposure_vs_scenario_dominant_factor_algebra():
    """Verify algebraic dominant factor matching yields SUPPORTS while divergence yields METHOD_DISAGREEMENT (not contradiction)."""
    # Aligned dominant factor (MKT is largest in both)
    f_rec_aligned = _make_evidence(
        "factor_risk.decomposition",
        "Factor Risk",
        {"beta_MKT": 1.20, "beta_SMB": 0.30, "beta_HML": -0.10},
    )
    s_rec_aligned = _make_evidence(
        "scenario.factor_linear",
        "Scenario Stress",
        {"contrib_MKT": -0.06, "contrib_SMB": -0.01, "contrib_HML": 0.005},
    )

    claim_supp, edges_supp = eval_factor_exposure_vs_scenario_alignment(f_rec_aligned, s_rec_aligned)
    assert claim_supp.claim_type == ClaimType.OBSERVATION
    assert claim_supp.status == ClaimStatus.VERIFIED
    assert edges_supp[0].relation == RelationshipType.SUPPORTS

    # Divergent dominant factor (MKT largest in beta, but massive shock on OIL makes OIL dominant in scenario)
    s_rec_divergent = _make_evidence(
        "scenario.factor_linear",
        "Scenario Stress Oil Shock",
        {"contrib_MKT": -0.01, "contrib_OIL": -0.15},
    )
    claim_div, edges_div = eval_factor_exposure_vs_scenario_alignment(f_rec_aligned, s_rec_divergent)
    assert claim_div.claim_type == ClaimType.METHOD_DISAGREEMENT
    assert claim_div.status == ClaimStatus.EVIDENCE_ONLY
    assert edges_div[0].relation == RelationshipType.ALTERNATIVE_METHOD
    assert edges_div[0].relation != RelationshipType.CONTRADICTS


def test_true_contradiction_detection_and_negative_test():
    """Verify TRUE contradiction is emitted on matching contract identities and NOT on differing identities."""
    # True contradiction: Same contract identity and fingerprint, but conflicting residual assertions
    rec_a = _make_evidence(
        "attribution.brinson",
        "Brinson Reconciled",
        {"reconciliation_error": 0.0},
        params={"reconciliation_identity": "brinson_active_return_identity"},
        artifact_hash="HASH-SAME",
    )
    rec_b = _make_evidence(
        "attribution.factor",
        "Factor Attribution Conflicting",
        {"reconciliation_error": 0.05},
        params={"reconciliation_identity": "brinson_active_return_identity"},
        artifact_hash="HASH-SAME",
    )

    claim_contra, edges_contra = eval_reconciliation_identity_contradiction(rec_a, rec_b)
    assert claim_contra is not None
    assert claim_contra.claim_type == ClaimType.CONTRADICTION
    assert claim_contra.status == ClaimStatus.CONTRADICTED
    assert edges_contra[0].relation == RelationshipType.CONTRADICTS

    # Negative test: Different identities do NOT trigger contradiction
    rec_c = _make_evidence(
        "attribution.carino",
        "Carino Multi Period",
        {"reconciliation_error": 0.05},
        params={"reconciliation_identity": "carino_multi_period_geometric"},
        artifact_hash="HASH-SAME",
    )
    claim_none, edges_none = eval_reconciliation_identity_contradiction(rec_a, rec_c)
    assert claim_none is None
    assert edges_none == []


def test_cross_analytical_committee_negative_showcase():
    """Verify committee review on individually valid modules creates distinct diagnostic evidence, passes critic, and achieves conditional governance."""
    # Synthetic multi-domain evidence set
    opt_rec = _make_evidence("portfolio.mvo", "MVO Portfolio", {"converged": True, "sharpe": 1.2})
    sample_cov = _make_evidence(
        "covariance.empirical", "Sample Cov", {"is_psd": True}, params={"weights": {"A": 0.7, "B": 0.3}}
    )
    lw_cov = _make_evidence(
        "covariance.ledoit_wolf", "LW Cov", {"is_psd": True}, params={"weights": {"A": 0.5, "B": 0.5}}
    )
    kupiec = _make_evidence(
        "traded_risk.kupiec_pof", "Kupiec", {"reject_unconditional_coverage": False, "gamma_test": 0.05}
    )
    christ = _make_evidence(
        "traded_risk.christoffersen_independence",
        "Christoffersen",
        {"reject_independence": True, "gamma_test": 0.05},
    )
    scen = _make_evidence(
        "scenario.linear_return", "Macro Stress", {"scenario_loss": 0.12, "scenario_id": "SCEN-MACRO"}
    )
    rev_stress = _make_evidence(
        "scenario.reverse_stress", "Reverse Stress", {"target_loss": 0.25, "minimum_distance": 0.18}
    )

    evidence_set = [opt_rec, sample_cov, lw_cov, kupiec, christ, scen, rev_stress]

    committee = CrossAnalyticalCommittee()
    result = committee.conduct_committee_review(evidence_set)

    # 1. Graph & Claims populated deterministically
    assert result.graph.node_count >= len(evidence_set)
    assert len(result.claims) > 0

    # 2. Distinct diagnostic EvidenceRecord IDs generated
    assert len(result.diagnostic_evidence) > 0
    all_source_ids = {r.evidence_id for r in evidence_set}
    for diag in result.diagnostic_evidence:
        assert diag.evidence_id not in all_source_ids

    # 3. Separation of Powers
    assert result.critic_disposition == "READY_FOR_GOVERNANCE"
    assert result.governance_decision == "ACCEPT_WITH_CONDITIONS"
    assert len(result.governance_conditions) > 0
