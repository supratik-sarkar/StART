"""StART — Gate 6 Agentic Vertical Slice Test Suite.

Verifies:
1. ScenarioStressAgent Specialist Review & Typed Assessment Synthesis
2. Zero-Prose Computation Invariant (Agents Orchestrate, Engines Compute)
3. AdversarialChallengeAgent Gate 6 Challenge Formulation & Tool Binding
4. Deterministic Resolution of Methodological Challenges
5. Subordinate Pattern-B Diagnostic EvidenceRecord Emission
6. Proof-Carrying Negative Evidence: Repricing Discrepancy without Materiality Threshold -> ACCEPT_WITH_CONDITIONS
7. Complete Multi-Agent Orchestration via MarketReviewDirectorAgent
"""

from __future__ import annotations

import numpy as np

from start.agents.market_review import (
    AdversarialChallengeAgent,
    GovernanceVerdict,
    MarketReviewDirectorAgent,
    ScenarioStressAgent,
)
from start.portfolio.contracts import (
    RepricingMethod,
    ReverseStressNorm,
    ReverseStressSpec,
    ScenarioSpec,
    ScenarioType,
    SensitivitySpec,
    ShockUnit,
)
from start.portfolio.evidence_bridge import (
    reverse_stress_to_evidence,
    scenario_result_to_evidence,
)
from start.portfolio.scenario import (
    apply_asset_return_scenario,
    apply_delta_gamma_scenario,
    create_scenario_shock,
    solve_reverse_stress,
)


def test_scenario_stress_agent_synthesis():
    """Verify ScenarioStressAgent audits evidence records and emits typed ScenarioStressAssessment without prose math."""
    weights = {"A1": 0.60, "A2": 0.40}
    shocks = (
        create_scenario_shock("A1", raw_value=-10.0, shock_unit=ShockUnit.RELATIVE_PERCENT),
        create_scenario_shock("A2", raw_value=-5.0, shock_unit=ShockUnit.RELATIVE_PERCENT),
    )
    spec = ScenarioSpec(
        scenario_id="SCEN-TEST-1",
        scenario_name="Test Scenario 1",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=shocks,
        repricing_method=RepricingMethod.LINEAR_RETURN,
    )
    res1 = apply_asset_return_scenario(weights=weights, scenario_spec_or_shocks=spec)
    ev1 = scenario_result_to_evidence(res1)

    # Reverse stress
    rev_spec = ReverseStressSpec(target_loss=0.05, distance_norm=ReverseStressNorm.L2)
    rev_res = solve_reverse_stress(
        spec=rev_spec, sensitivities_or_weights=np.array([0.60, 0.40]), factors=["A1", "A2"]
    )
    ev_rev = reverse_stress_to_evidence(rev_res)

    agent = ScenarioStressAgent()
    out = agent.execute({"evidence_records": [ev1, ev_rev]})

    assert out["status"] == "completed"
    assert "assessment" in out
    ass = out["assessment"]
    assert "SCEN-TEST-1" in ass["scenarios_evaluated"]
    assert ass["reverse_stress_achieved"] is True
    assert len(out["findings"]) >= 2
    # Verify citations
    for f in out["findings"]:
        assert len(f["evidence_ids"]) > 0


def test_adversarial_challenge_formulation_and_resolution():
    """Verify AdversarialChallengeAgent formulates Gate 6 challenges and resolves them via deterministic portfolio tools."""
    sens = {
        "EQ": SensitivitySpec("EQ", delta=100.0, gamma=-50.0),
        "VOL": SensitivitySpec("VOL", delta=-50.0, gamma=20.0),
    }
    shocks = (
        create_scenario_shock("EQ", raw_value=-0.10, shock_unit=ShockUnit.RETURN_DECIMAL),
        create_scenario_shock("VOL", raw_value=0.05, shock_unit=ShockUnit.RETURN_DECIMAL),
    )
    spec_dg = ScenarioSpec(
        scenario_id="SCEN-DG-TEST",
        scenario_name="Delta Gamma Test",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=shocks,
        repricing_method=RepricingMethod.DELTA_GAMMA,
    )
    res_dg = apply_delta_gamma_scenario(sensitivities=sens, scenario_spec_or_shocks=spec_dg)
    ev_dg = scenario_result_to_evidence(res_dg)

    agent = AdversarialChallengeAgent()
    challenges = agent.formulate_portfolio_challenges([ev_dg])

    # Should include method sensitivity challenge
    meth_chal = [c for c in challenges if "METHOD-SENSITIVITY" in c.challenge_id]
    assert len(meth_chal) == 1

    # Resolve challenge deterministically
    context = {
        "evidence_records": [ev_dg],
        "sensitivities": sens,
        "scenario_spec_or_shocks": spec_dg,
    }
    resolution = agent.resolve_challenge(meth_chal[0], context)
    assert resolution.tool_name == "apply_delta_gamma_scenario"
    assert len(context["evidence_records"]) == 2  # Subordinate diagnostic record added


def test_proof_carrying_negative_evidence_and_governance_signoff():
    """Verify proof-carrying negative evidence (un-thresholded sensitivity challenge) yields ACCEPT_WITH_CONDITIONS."""
    weights = {"A1": 0.50, "A2": 0.50}
    shocks = (
        create_scenario_shock("A1", raw_value=-0.10, shock_unit=ShockUnit.RETURN_DECIMAL),
        create_scenario_shock("A2", raw_value=-0.05, shock_unit=ShockUnit.RETURN_DECIMAL),
    )
    spec = ScenarioSpec(
        scenario_id="SCEN-BASE",
        scenario_name="Base Scenario",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=shocks,
        repricing_method=RepricingMethod.LINEAR_RETURN,
    )
    res = apply_asset_return_scenario(weights=weights, scenario_spec_or_shocks=spec)
    ev = scenario_result_to_evidence(res)

    sens = {
        "A1": SensitivitySpec("A1", delta=50.0, gamma=-20.0),
        "A2": SensitivitySpec("A2", delta=50.0, gamma=-10.0),
    }
    res_dg = apply_delta_gamma_scenario(sensitivities=sens, scenario_spec_or_shocks=spec)
    ev_dg = scenario_result_to_evidence(res_dg)

    context = {
        "evidence_records": [ev, ev_dg],
        "weights": weights,
        "sensitivities": sens,
        "scenario_spec": spec,
    }

    director = MarketReviewDirectorAgent()
    out = director.execute(context)

    # Governance verdict must be ACCEPT_WITH_CONDITIONS due to evidence-only challenge resolution without explicit materiality threshold
    assert out["status"] == "orchestrated"
    assert out["governance_verdict"] == GovernanceVerdict.ACCEPT_WITH_CONDITIONS.value
    assert len(out["governance_signoff"]["conditions"]) > 0
