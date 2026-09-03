"""Deterministic Showcase for StART Combined Gate 7-9 / 7-9A.

Exercises full cross-analytical committee, evidence graph, adversarial diagnostic challenges,
and governance evaluation using 100% synthetic public-safe fixtures without network or provider calls.
"""

from __future__ import annotations

import json
from collections import Counter

from start.agents.committee import CrossAnalyticalCommittee
from start.core.schemas import EvidenceRecord, Status
from start.evidence.claims import ClaimStatus, ClaimType


def create_synthetic_showcase_evidence() -> list[EvidenceRecord]:
    """Build deterministic synthetic multi-domain evidence set."""
    # 1. Portfolio MVO
    opt_rec = EvidenceRecord(
        test_id="portfolio.mean_variance",
        test_name="Mean-Variance Portfolio Optimization",
        model_id="MOD-SYNTH-MVO",
        dataset_id="DS-SYNTH-RETURNS",
        run_id="RUN-SHOWCASE-79",
        metrics={
            "converged": True,
            "sharpe_ratio": 1.45,
            "portfolio_volatility": 0.125,
        },
        params={"objective": "MAX_SHARPE", "weights": {"ASSET_A": 0.65, "ASSET_B": 0.35}},
        status=Status.PASS,
        interpretation="Optimization converged to unique optimal weights.",
        limitations=["Subject to covariance estimation risk."],
        input_artifact_hash="8a1f2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a",
    )

    # 2. Covariance Sample vs Ledoit-Wolf
    cov_sample = EvidenceRecord(
        test_id="covariance.empirical",
        test_name="Sample Covariance Matrix",
        model_id="MOD-SYNTH-COV",
        dataset_id="DS-SYNTH-RETURNS",
        run_id="RUN-SHOWCASE-79",
        metrics={
            "is_psd": True,
            "condition_number": 42.5,
        },
        params={"method": "EMPIRICAL", "weights": {"ASSET_A": 0.65, "ASSET_B": 0.35}},
        status=Status.PASS,
        interpretation="Empirical sample covariance is positive semi-definite.",
        limitations=["Prone to in-sample noise."],
        input_artifact_hash="8a1f2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a",
    )
    cov_lw = EvidenceRecord(
        test_id="covariance.ledoit_wolf_shrinkage",
        test_name="Ledoit-Wolf Shrinkage Covariance",
        model_id="MOD-SYNTH-COV-LW",
        dataset_id="DS-SYNTH-RETURNS",
        run_id="RUN-SHOWCASE-79",
        metrics={
            "is_psd": True,
            "shrinkage_intensity": 0.28,
            "condition_number": 18.2,
        },
        params={"method": "LEDOIT_WOLF", "weights": {"ASSET_A": 0.50, "ASSET_B": 0.50}},
        status=Status.PASS,
        interpretation="Ledoit-Wolf shrinkage reduces condition number.",
        limitations=["Assumes linear shrinkage target."],
        input_artifact_hash="8a1f2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a",
    )

    # 3. Factor Attribution vs Risk
    attr_rec = EvidenceRecord(
        test_id="attribution.return_attribution",
        test_name="Factor Return Attribution",
        model_id="MOD-SYNTH-ATTR",
        dataset_id="DS-SYNTH-RETURNS",
        run_id="RUN-SHOWCASE-79",
        metrics={
            "active_return": 0.035,
            "reconciliation_error": 0.0,
            "factor_contribution_MKT": 0.025,
            "factor_contribution_SMB": 0.010,
        },
        params={"factors": ["MKT", "SMB"]},
        status=Status.PASS,
        interpretation="Attribution algebra reconciles with 0 residual.",
        limitations=["Single-period model."],
        input_artifact_hash="8a1f2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a",
    )
    risk_rec = EvidenceRecord(
        test_id="attribution.risk_attribution",
        test_name="Factor Risk Decomposition",
        model_id="MOD-SYNTH-RISK",
        dataset_id="DS-SYNTH-RETURNS",
        run_id="RUN-SHOWCASE-79",
        metrics={
            "total_factor_variance": 0.015,
            "specific_variance": 0.004,
            "beta_MKT": 1.10,
            "beta_SMB": 0.40,
        },
        params={"factors": ["MKT", "SMB"]},
        status=Status.PASS,
        interpretation="Factor risk decomposition computed.",
        limitations=["Static factor loadings."],
        input_artifact_hash="8a1f2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a",
    )

    # 4. Gate-5 Tail Risk: Kupiec POF (passed) vs Christoffersen Independence (failed at gamma=0.05)
    kupiec_rec = EvidenceRecord(
        test_id="traded_risk.var_kupiec_pof",
        test_name="Kupiec POF Coverage Test",
        model_id="MOD-SYNTH-VAR",
        dataset_id="DS-SYNTH-VAR",
        run_id="RUN-SHOWCASE-79",
        metrics={
            "var_estimate": 0.075,
            "exception_count": 12,
            "expected_exceptions": 12.5,
            "p_value": 0.88,
            "reject_unconditional_coverage": False,
            "gamma_test": 0.05,
        },
        params={"confidence_level": 0.95, "gamma_test": 0.05},
        status=Status.PASS,
        interpretation="Unconditional coverage is not rejected at gamma=0.05.",
        limitations=["Unconditional test blind to exception clustering."],
        input_artifact_hash="8a1f2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a",
    )
    christ_rec = EvidenceRecord(
        test_id="traded_risk.var_christoffersen",
        test_name="Christoffersen Independence Test",
        model_id="MOD-SYNTH-VAR",
        dataset_id="DS-SYNTH-VAR",
        run_id="RUN-SHOWCASE-79",
        metrics={
            "p_value": 0.008,
            "reject_independence": True,
            "gamma_test": 0.05,
        },
        params={"confidence_level": 0.95, "gamma_test": 0.05},
        status=Status.WARN,
        interpretation="Exception independence is rejected at gamma=0.05 due to temporal clustering.",
        limitations=["First-order Markov assumption."],
        input_artifact_hash="8a1f2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a",
    )

    # 5. Gate-6 Scenario & Reverse Stress
    scen_rec = EvidenceRecord(
        test_id="scenario.factor_linear",
        test_name="Macro Factor Scenario Stress",
        model_id="MOD-SYNTH-SCEN",
        dataset_id="DS-SYNTH-RETURNS",
        run_id="RUN-SHOWCASE-79",
        metrics={
            "portfolio_loss": 0.145,
            "dominant_factor": "MKT",
            "scenario_id": "SCEN-MACRO-SHOCK",
        },
        params={"scenario_id": "SCEN-MACRO-SHOCK"},
        status=Status.RECORDED,
        interpretation="Linear factor scenario shock evaluated.",
        limitations=["First-order Taylor approximation."],
        input_artifact_hash="8a1f2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a",
    )
    rev_stress_rec = EvidenceRecord(
        test_id="scenario.reverse_stress",
        test_name="Minimum-Distance Reverse Stress",
        model_id="MOD-SYNTH-REV-STRESS",
        dataset_id="DS-SYNTH-RETURNS",
        run_id="RUN-SHOWCASE-79",
        metrics={
            "target_loss": 0.25,
            "achieved_loss": 0.25,
            "minimum_distance": 0.185,
            "geometry": "MAHALANOBIS",
            "solver_converged": True,
        },
        params={"target_loss": 0.25, "geometry": "MAHALANOBIS"},
        status=Status.PASS,
        interpretation="Found minimum Mahalanobis distance shock vector achieving target loss 0.25.",
        limitations=["Subject to covariance stability."],
        input_artifact_hash="8a1f2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a",
    )

    return [
        opt_rec,
        cov_sample,
        cov_lw,
        attr_rec,
        risk_rec,
        kupiec_rec,
        christ_rec,
        scen_rec,
        rev_stress_rec,
    ]


def run_showcase() -> dict:
    """Execute complete deterministic showcase and return structured proof dictionary."""
    evidence_set = create_synthetic_showcase_evidence()

    # Pre-execution input evidence census by family
    family_census: dict[str, int] = {}
    for r in evidence_set:
        fam = r.test_id.split(".")[0]
        family_census[fam] = family_census.get(fam, 0) + 1

    committee = CrossAnalyticalCommittee()
    result = committee.conduct_committee_review(evidence_set)

    # Edge census
    edge_types = Counter(e.relation.value for e in result.graph.get_edges())
    # Claim census
    claim_types = Counter(c.claim_type.value for c in result.claims)

    contradictions = [c for c in result.claims if c.claim_type == ClaimType.CONTRADICTION]
    unresolved_risk = [c for c in result.claims if c.claim_type == ClaimType.UNRESOLVED_RISK]
    evidence_only = [c for c in result.claims if c.status == ClaimStatus.EVIDENCE_ONLY]

    source_ids = [r.evidence_id for r in evidence_set]
    diag_ids = [r.evidence_id for r in result.diagnostic_evidence]

    # Verification: proof source ID != diagnostic ID
    all_source_set = set(source_ids)
    all_diag_set = set(diag_ids)
    proof_disjoint = len(all_source_set.intersection(all_diag_set)) == 0

    first_chal = result.challenges[0] if result.challenges else None
    first_res = result.resolutions[0] if result.resolutions else None

    # First cross-analytical claim with statistical criteria
    tail_claim = next(
        (c for c in result.claims if c.statistical_criterion_source is not None),
        result.claims[0] if result.claims else None,
    )

    summary = {
        "input_evidence_count": len(evidence_set),
        "input_evidence_by_family": dict(sorted(family_census.items())),
        "graph_node_count": result.graph.node_count,
        "graph_edge_count": result.graph.edge_count,
        "edge_type_census": dict(edge_types),
        "claim_count": len(result.claims),
        "claim_type_census": dict(claim_types),
        "contradiction_count": len(contradictions),
        "unresolved_risk_count": len(unresolved_risk),
        "evidence_only_claim_count": len(evidence_only),
        "challenge_count": len(result.challenges),
        "diagnostic_evidence_count": len(result.diagnostic_evidence),
        "source_evidence_ids": source_ids,
        "diagnostic_evidence_ids": diag_ids,
        "proof_source_id_ne_diagnostic_id": proof_disjoint,
        "primary_diagnostic_tool": first_chal.required_tool if first_chal else "N/A",
        "primary_challenge_status": first_res.status.value if first_res else "N/A",
        "statistical_criterion_source": (
            tail_claim.statistical_criterion_source if tail_claim else "PRE_REGISTERED_VALIDATION"
        ),
        "statistical_gamma_test": tail_claim.statistical_gamma_test if tail_claim else 0.05,
        "materiality_criterion_source": tail_claim.materiality_criterion_source if tail_claim else "NONE",
        "critic_disposition": result.critic_disposition,
        "governance_decision": result.governance_decision,
        "governance_conditions": result.governance_conditions,
        "graph_canonical_sha256": result.graph.canonical_fingerprint(),
    }

    return summary


if __name__ == "__main__":
    proof = run_showcase()
    print("================ StART Combined Gate 7-9 Deterministic Showcase ================")
    print(json.dumps(proof, indent=2))
    print("================================================================================")
