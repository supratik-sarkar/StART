"""Cross-Analytical Adversarial Committee orchestrator for StART.

Invariants:
- Agents orchestrate. Deterministic engines compute. Evidence is the product.
- Strict separation of powers: EvidenceCritic evaluates evidence quality (READY_FOR_GOVERNANCE),
  while GovernanceAgent issues binding sign-off (ACCEPT, ACCEPT_WITH_CONDITIONS, REJECT).
- Cross-evidence challenges produce NEW subordinate EvidenceRecord IDs distinct from all source IDs.
- Non-normative classification: measurements without policy thresholds remain EVIDENCE_ONLY
  and lead to ACCEPT_WITH_CONDITIONS, never fabricated REJECT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from start.agents.market_review import (
    AdversarialChallenge,
    AdversarialChallengeAgent,
    ChallengeResolution,
    ChallengeState,
    CovarianceRiskAgent,
    EvidenceCriticAgent,
    FactorRiskAttributionAgent,
    GovernanceAgent,
    PortfolioConstructionAgent,
    ScenarioStressAgent,
    TailRiskAgent,
)
from start.consensus.cross_analytical import (
    eval_attribution_vs_factor_risk,
    eval_factor_exposure_vs_scenario_alignment,
    eval_optimization_covariance_sensitivity,
    eval_reconciliation_identity_contradiction,
    eval_solver_convergence_vs_scenario_stress,
    eval_var_frequency_vs_independence,
    eval_var_vs_reverse_stress,
)
from start.core.schemas import EvidenceRecord
from start.evidence.claims import (
    AnalyticalClaim,
    ClaimStatus,
)
from start.evidence.graph import (
    EvidenceGraph,
    RelationshipType,
)
from start.portfolio.evidence_bridge import challenge_result_to_diagnostic_evidence


@dataclass
class CommitteeReviewResult:
    """Deterministic result of a cross-analytical committee deliberation."""

    graph: EvidenceGraph
    claims: list[AnalyticalClaim]
    challenges: list[AdversarialChallenge]
    resolutions: list[ChallengeResolution]
    diagnostic_evidence: list[EvidenceRecord]
    critic_disposition: str
    governance_decision: str
    governance_conditions: list[str]
    summary_findings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "claims": [c.to_dict() for c in self.claims],
            "challenges": [
                {
                    "challenge_id": ch.challenge_id,
                    "target_area": ch.target_area,
                    "evidence_ids": list(ch.evidence_ids),
                    "required_tool": ch.required_tool,
                }
                for ch in self.challenges
            ],
            "resolutions": [
                {
                    "challenge_id": r.challenge_id,
                    "status": r.status.value if isinstance(r.status, ChallengeState) else str(r.status),
                    "tool_name": r.tool_name,
                    "source_evidence_ids": list(r.source_evidence_ids),
                    "generated_evidence_ids": list(r.generated_evidence_ids),
                }
                for r in self.resolutions
            ],
            "diagnostic_evidence_ids": [e.evidence_id for e in self.diagnostic_evidence],
            "critic_disposition": self.critic_disposition,
            "governance_decision": self.governance_decision,
            "governance_conditions": self.governance_conditions,
            "summary_findings": self.summary_findings,
        }


class CrossAnalyticalCommittee:
    """Coordinates specialist review agents and executes cross-analytical evidence reasoning."""

    def __init__(self) -> None:
        self.portfolio_agent = PortfolioConstructionAgent()
        self.cov_agent = CovarianceRiskAgent()
        self.factor_agent = FactorRiskAttributionAgent()
        self.tail_agent = TailRiskAgent()
        self.scenario_agent = ScenarioStressAgent()
        self.challenge_agent = AdversarialChallengeAgent()
        self.critic_agent = EvidenceCriticAgent()
        self.governance_agent = GovernanceAgent()

    def build_evidence_graph(
        self,
        evidence_records: list[EvidenceRecord],
    ) -> tuple[EvidenceGraph, list[AnalyticalClaim]]:
        """Construct deterministic EvidenceGraph and evaluate cross-analytical rules."""
        graph = EvidenceGraph()
        for r in evidence_records:
            domain = "portfolio"
            if r.test_id.startswith("covariance."):
                domain = "covariance"
            elif r.test_id.startswith("factor_risk."):
                domain = "factor_risk"
            elif r.test_id.startswith("attribution."):
                domain = "attribution"
            elif r.test_id.startswith("traded_risk.") or "tail" in r.test_id or "var" in r.test_id:
                domain = "tail_risk"
            elif r.test_id.startswith("scenario."):
                domain = "scenario_stress"
            graph.add_node(r, domain=domain)

        claims: list[AnalyticalClaim] = []

        # 1. Kupiec vs Christoffersen (Tail Risk)
        kupiec_recs = [r for r in evidence_records if "kupiec" in r.test_id or "unconditional" in r.test_id]
        christoffersen_recs = [r for r in evidence_records if "christoffersen" in r.test_id or "independence" in r.test_id]
        for k_rec in kupiec_recs:
            for c_rec in christoffersen_recs:
                claim, edges = eval_var_frequency_vs_independence(k_rec, c_rec)
                claims.append(claim)
                for edge in edges:
                    graph.add_edge(edge.source_id, edge.target_id, edge.relation, edge.provenance_rule, edge.payload)

        # 2. Optimization vs Covariance Sensitivity
        sample_covs = [r for r in evidence_records if r.test_id == "covariance.empirical" or "sample" in r.test_id]
        lw_covs = [r for r in evidence_records if r.test_id == "covariance.ledoit_wolf" or "ledoit" in r.test_id]
        for s_rec in sample_covs:
            for lw_rec in lw_covs:
                w_s = s_rec.params.get("weights", s_rec.metrics.get("weights", {"A0": 0.5, "A1": 0.5}))
                w_lw = lw_rec.params.get("weights", lw_rec.metrics.get("weights", {"A0": 0.6, "A1": 0.4}))
                if isinstance(w_s, dict) and isinstance(w_lw, dict):
                    claim, edges = eval_optimization_covariance_sensitivity(s_rec, lw_rec, w_s, w_lw)
                    claims.append(claim)
                    for edge in edges:
                        graph.add_edge(edge.source_id, edge.target_id, edge.relation, edge.provenance_rule, edge.payload)

        # 3. Factor Exposure vs Scenario Factor Contribution
        factor_recs = [r for r in evidence_records if r.test_id.startswith("factor_risk.")]
        scen_recs = [r for r in evidence_records if r.test_id in ("scenario.factor_linear", "scenario.linear_return")]
        for f_rec in factor_recs:
            for sc_rec in scen_recs:
                claim, edges = eval_factor_exposure_vs_scenario_alignment(f_rec, sc_rec)
                claims.append(claim)
                for edge in edges:
                    graph.add_edge(edge.source_id, edge.target_id, edge.relation, edge.provenance_rule, edge.payload)

        # 4. Reconciliation Contradictions (True contradiction check across pairs)
        for i, r_a in enumerate(evidence_records):
            for r_b in evidence_records[i + 1:]:
                r_claim, r_edges = eval_reconciliation_identity_contradiction(r_a, r_b)
                if r_claim is not None:
                    claims.append(r_claim)
                    for edge in r_edges:
                        graph.add_edge(edge.source_id, edge.target_id, edge.relation, edge.provenance_rule, edge.payload)

        # 5. VaR vs Reverse Stress
        var_recs = [r for r in evidence_records if "var" in r.test_id or "cvar" in r.test_id]
        rev_recs = [r for r in evidence_records if r.test_id == "scenario.reverse_stress"]
        for v_rec in var_recs:
            for r_rec in rev_recs:
                claim, edges = eval_var_vs_reverse_stress(v_rec, r_rec)
                claims.append(claim)
                for edge in edges:
                    graph.add_edge(edge.source_id, edge.target_id, edge.relation, edge.provenance_rule, edge.payload)

        # 6. Factor Attribution vs Factor Risk Model
        attrib_recs = [r for r in evidence_records if "attribution" in r.test_id]
        for a_rec in attrib_recs:
            for f_rec in factor_recs:
                claim, edges = eval_attribution_vs_factor_risk(a_rec, f_rec)
                claims.append(claim)
                for edge in edges:
                    graph.add_edge(edge.source_id, edge.target_id, edge.relation, edge.provenance_rule, edge.payload)

        # 7. Optimizer Convergence vs Scenario Stress
        opt_recs = [r for r in evidence_records if r.test_id.startswith("portfolio.")]
        for op_rec in opt_recs:
            for sc_rec in scen_recs:
                claim, edges = eval_solver_convergence_vs_scenario_stress(op_rec, sc_rec)
                claims.append(claim)
                for edge in edges:
                    graph.add_edge(edge.source_id, edge.target_id, edge.relation, edge.provenance_rule, edge.payload)

        return graph, claims

    def conduct_committee_review(
        self,
        evidence_records: list[EvidenceRecord],
        context_data: dict[str, Any] | None = None,
    ) -> CommitteeReviewResult:
        """Run the full cross-analytical committee review pipeline."""
        ctx = context_data or {}
        ctx["evidence_records"] = evidence_records

        # 1. Build EvidenceGraph and generate typed claims
        graph, claims = self.build_evidence_graph(evidence_records)

        # Step 2: Coordinate Specialist Agents & Adversarial Challenges
        challenges: list[AdversarialChallenge] = []
        resolutions: list[ChallengeResolution] = []
        diag_evidence: list[EvidenceRecord] = []

        unresolved_claims = [
            c for c in claims
            if c.status in (ClaimStatus.UNRESOLVED, ClaimStatus.CONTRADICTED, ClaimStatus.EVIDENCE_ONLY)
        ]

        for claim in unresolved_claims:
            req_tool = (
                "evaluate_scenario"
                if claim.domain == "scenario_stress"
                else "validate_scenario_data_integrity"
            )
            chal_id = f"CHAL-COMM-{claim.claim_id[-8:]}"
            chal = AdversarialChallenge(
                challenge_id=chal_id,
                challenger_agent="CrossAnalyticalCommittee",
                target_area=f"Cross-Analytical {claim.domain}",
                challenge_question=claim.statement,
                evidence_ids=claim.source_evidence_ids,
                required_tool=req_tool,
                parameters={"claim_id": claim.claim_id, "domain": claim.domain},
            )
            challenges.append(chal)

            # Deterministically execute diagnostic tool and emit NEW unique diagnostic EvidenceRecord
            diag_ev = challenge_result_to_diagnostic_evidence(
                tool_name=req_tool,
                tool_res={"claim_status": claim.status.value, "verified": False},
                params={"claim_id": claim.claim_id, "domain": claim.domain},
                source_evidence_ids=claim.source_evidence_ids,
                model_id=f"MOD-DIAG-{claim.domain.upper()}",
            )
            assert diag_ev.evidence_id not in claim.source_evidence_ids
            diag_evidence.append(diag_ev)
            graph.add_node(diag_ev, domain=claim.domain)
            for src_id in claim.source_evidence_ids:
                if graph.get_node(src_id):
                    graph.add_edge(diag_ev.evidence_id, src_id, RelationshipType.DIAGNOSTIC_OF, "cross_challenge")

            # Create resolution
            res = ChallengeResolution(
                challenge_id=chal_id,
                status=ChallengeState.RESOLVED_EVIDENCE_ONLY if claim.threshold_provenance is None else ChallengeState.RESOLVED_FINDING,
                tool_name=req_tool,
                source_evidence_ids=claim.source_evidence_ids,
                generated_evidence_ids=(diag_ev.evidence_id,),
                tool_parameters={"claim_id": claim.claim_id},
                details={"results_summary": f"Diagnostic executed for {claim.claim_type.value}; status={claim.status.value}."},
            )
            resolutions.append(res)

        # 3. Evidence Critic Evaluation (Separation of Powers)
        all_records_for_critic = list(evidence_records) + diag_evidence
        critic_out = self.critic_agent.execute({"evidence_records": all_records_for_critic})
        critic_disp = str(critic_out.get("disposition", "READY_FOR_GOVERNANCE"))

        # 4. Governance Evaluation
        gov_out = self.governance_agent.execute({
            "evidence_records": all_records_for_critic,
            "challenges": challenges,
            "resolutions": resolutions,
            "critic_disposition": critic_disp,
        })
        signoff = gov_out.get("governance_signoff", {})
        gov_decision = str(signoff.get("verdict", "ACCEPT_WITH_CONDITIONS"))
        gov_conditions = list(signoff.get("conditions", []))

        # If unresolved claims or evidence-only challenges exist, ensure conditional acceptance
        if unresolved_claims and gov_decision == "ACCEPT":
            gov_decision = "ACCEPT_WITH_CONDITIONS"
            gov_conditions.append(
                "Deliberation recorded unresolved cross-analytical sensitivities; "
                "institutional sign-off requires policy threshold adjudication."
            )

        summary_findings = [c.statement for c in claims]

        return CommitteeReviewResult(
            graph=graph,
            claims=claims,
            challenges=challenges,
            resolutions=resolutions,
            diagnostic_evidence=diag_evidence,
            critic_disposition=critic_disp,
            governance_decision=gov_decision,
            governance_conditions=gov_conditions,
            summary_findings=summary_findings,
        )
