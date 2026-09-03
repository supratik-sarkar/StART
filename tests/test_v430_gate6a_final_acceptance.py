# =========================================================================== #
# StART Gate 6A — Final Acceptance Audit: Scenario, Stress & Reverse-Stress
# =========================================================================== #

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from start.agents.market_review import (
    AdversarialChallenge,
    ChallengeResolution,
    ChallengeState,
    GovernanceAgent,
    ScenarioStressAgent,
)
from start.portfolio.artifacts import (
    _hash_payload,
    render_reverse_stress_profile_artifact,
    render_scenario_active_comparison_artifact,
    render_scenario_asset_contribution_artifact,
    render_scenario_factor_contribution_artifact,
    render_scenario_group_heatmap_artifact,
    render_scenario_pnl_waterfall_artifact,
    render_scenario_sensitivity_curve_artifact,
    render_scenario_set_ranking_artifact,
)
from start.portfolio.contracts import (
    MetricHorizon,
    PartitionContract,
    RepricingMethod,
    ReverseStressNorm,
    ReverseStressSpec,
    ScenarioShock,
    ScenarioSpec,
    ScenarioType,
    SensitivitySpec,
    ShockSpace,
    ShockUnit,
)
from start.portfolio.evidence_bridge import (
    challenge_result_to_diagnostic_evidence,
    scenario_data_integrity_to_evidence,
    scenario_result_to_evidence,
)
from start.portfolio.scenario import (
    apply_active_scenario,
    apply_asset_return_scenario,
    apply_delta_gamma_scenario,
    apply_factor_scenario,
    apply_group_scenario_decomposition,
    compare_scenario_set,
    create_scenario_shock,
    evaluate_scenario,
    evaluate_scenario_sensitivity_grid,
    normalize_shock,
    replay_historical_scenario,
    solve_reverse_stress,
    validate_repricing_shock_compatibility,
    validate_scenario_data_integrity,
)
from start.registry import list_tests

# =========================================================================== #
# 5.A PRE-FLIGHT SCENARIO DATA INTEGRITY
# =========================================================================== #

def test_scenario_data_integrity_executes_before_repricing():
    """Verify deterministic pre-flight audit for ID validity, units, coverage, and execution ordering before repricing."""
    # 1. Valid ScenarioSpec
    valid_spec = ScenarioSpec(
        scenario_id="SCEN-VALID-01",
        scenario_name="Valid Institutional Spec",
        scenario_type=ScenarioType.SYNTHETIC.value,
        shocks=(
            create_scenario_shock("AAPL", -0.10, ShockUnit.RETURN_DECIMAL, ShockSpace.ASSET_RETURN),
            create_scenario_shock("MSFT", -0.05, ShockUnit.RETURN_DECIMAL, ShockSpace.ASSET_RETURN),
        ),
        repricing_method=RepricingMethod.LINEAR_RETURN.value,
    )
    diag_valid = validate_scenario_data_integrity(valid_spec, portfolio_assets=["AAPL", "MSFT"])
    assert diag_valid.valid is True
    assert len(diag_valid.issues) == 0

    # Emit subordinate EvidenceRecord
    ev = scenario_data_integrity_to_evidence(diag_valid)
    assert ev.test_id == "scenario.data_integrity"
    assert ev.metrics["valid"] is True
    assert ev.metrics["n_shocks"] == 2

    # 2. Invalid ScenarioSpec: Missing shocks, duplicate shock IDs, incompatible repricing method
    dup_shock = create_scenario_shock("AAPL", -0.10, ShockUnit.RETURN_DECIMAL, ShockSpace.ASSET_RETURN)
    incompat_spec = ScenarioSpec(
        scenario_id="",
        scenario_name="Invalid Spec",
        scenario_type=ScenarioType.HISTORICAL_REPLAY.value,
        shocks=(dup_shock, dup_shock),
        repricing_method=RepricingMethod.LINEAR_RETURN.value,
        source_reference="",  # Missing historical source
    )
    diag_bad = validate_scenario_data_integrity(incompat_spec, portfolio_assets=["AAPL", "MSFT"])
    assert diag_bad.valid is False
    assert len(diag_bad.issues) >= 3
    assert any("scenario_id" in iss for iss in diag_bad.issues)
    assert any("Duplicate" in iss for iss in diag_bad.issues)
    assert any("source_reference" in iss for iss in diag_bad.issues)

    # 3. Production Orchestration Execution Ordering Proof:
    # evaluate_scenario validates integrity strictly before dispatching to repricing
    agent = ScenarioStressAgent()
    weights = {"AAPL": 0.5, "MSFT": 0.5}

    with patch("start.portfolio.scenario.apply_asset_return_scenario", wraps=apply_asset_return_scenario) as mock_repricer:
        # A) Production call with invalid spec -> integrity fails -> ValueError raised -> repricing call count remains 0
        with pytest.raises(ValueError, match="Scenario data integrity validation failed"):
            evaluate_scenario(incompat_spec, weights, assets=list(weights.keys()))
        assert mock_repricer.call_count == 0

        # Also via ScenarioStressAgent.execute_tool
        with pytest.raises(ValueError, match="Scenario data integrity validation failed"):
            agent.execute_tool("evaluate_scenario", spec=incompat_spec, weights=weights, assets=list(weights.keys()))
        assert mock_repricer.call_count == 0

        # B) Production call with valid spec -> integrity passes -> repricing is invoked -> call count = 1
        res_good = evaluate_scenario(valid_spec, weights, assets=list(weights.keys()))
        assert res_good is not None
        assert mock_repricer.call_count == 1
        assert math.isclose(res_good.scenario_return, -0.075, rel_tol=1e-9)

        # Also via ScenarioStressAgent.execute_tool -> call count becomes 2
        res_agent = agent.execute_tool("evaluate_scenario", spec=valid_spec, weights=weights, assets=list(weights.keys()))
        assert res_agent is not None
        assert mock_repricer.call_count == 2

    # 4. NaN / Inf non-finite shock values check
    nan_shock = ScenarioShock(
        risk_factor_id="AAPL",
        shock_space=ShockSpace.ASSET_RETURN.value,
        shock_unit=ShockUnit.RETURN_DECIMAL.value,
        raw_value=float("nan"),
        normalized_value=float("nan"),
        normalization_rule="invalid",
    )
    nan_spec = ScenarioSpec("SCEN-NAN", "NaN Spec", ScenarioType.SYNTHETIC.value, (nan_shock,), RepricingMethod.LINEAR_RETURN.value)
    diag_nan = validate_scenario_data_integrity(nan_spec, portfolio_assets=["AAPL"])
    assert diag_nan.valid is False
    assert any("Non-finite" in iss for iss in diag_nan.issues)

    # 5. Missing sensitivity check under DELTA_GAMMA
    dg_spec = ScenarioSpec(
        scenario_id="SCEN-DG-01",
        scenario_name="DG Spec",
        scenario_type=ScenarioType.SYNTHETIC.value,
        shocks=(create_scenario_shock("AAPL", -0.10, ShockUnit.RETURN_DECIMAL, ShockSpace.ASSET_RETURN),),
        repricing_method=RepricingMethod.DELTA_GAMMA.value,
    )
    diag_dg_no_sens = validate_scenario_data_integrity(dg_spec)
    assert diag_dg_no_sens.valid is False
    assert any("requires explicit sensitivities" in iss for iss in diag_dg_no_sens.issues)


# =========================================================================== #
# 5.B MIXED-LEG NORMALIZATION KNOWN ANSWER
# =========================================================================== #

def test_mixed_leg_normalization_known_answer():
    """Verify exact raw-value and computational normalization rules across mixed legs."""
    # Leg 1: Equity AAPL (-10% -> -0.10)
    s1 = create_scenario_shock("AAPL", -10.0, ShockUnit.RELATIVE_PERCENT, ShockSpace.ASSET_RETURN)
    assert s1.raw_value == -10.0
    assert s1.normalized_value == -0.10
    assert "RELATIVE_PERCENT" in s1.normalization_rule
    assert s1.computational_unit == ShockUnit.RETURN_DECIMAL.value

    # Leg 2: Rate US10Y (+100 bps -> +0.0100)
    s2 = create_scenario_shock("US10Y", 100.0, ShockUnit.BASIS_POINTS, ShockSpace.RATE)
    assert s2.raw_value == 100.0
    assert s2.normalized_value == 0.0100
    assert "BASIS_POINTS" in s2.normalization_rule
    assert s2.computational_unit == ShockUnit.ABSOLUTE.value

    # Leg 3: Volatility VIX (+8 vol points -> +0.0800)
    s3 = create_scenario_shock("VIX", 8.0, ShockUnit.VOLATILITY_POINTS, ShockSpace.VOLATILITY)
    assert s3.raw_value == 8.0
    assert s3.normalized_value == 0.0800
    assert "VOLATILITY_POINTS" in s3.normalization_rule
    assert s3.computational_unit == ShockUnit.ABSOLUTE.value


# =========================================================================== #
# 5.C LOG_RETURN SEMANTICS
# =========================================================================== #

def test_log_return_semantics_known_answer():
    """Verify LOG_RETURN conversion to simple return exp(x) - 1 and computational unit labeling."""
    raw_log = math.log(0.90)  # ~ -0.1053605
    norm_val, rule, comp_unit = normalize_shock(raw_log, ShockUnit.LOG_RETURN)

    assert math.isclose(norm_val, -0.10, rel_tol=1e-9)
    assert "LOG_RETURN" in rule
    assert comp_unit == ShockUnit.RETURN_DECIMAL.value
    assert comp_unit != ShockUnit.LOG_RETURN.value

    s = create_scenario_shock("ASSET_1", raw_log, ShockUnit.LOG_RETURN, ShockSpace.ASSET_RETURN)
    assert math.isclose(s.normalized_value, -0.10, rel_tol=1e-9)
    assert s.computational_unit == ShockUnit.RETURN_DECIMAL.value


# =========================================================================== #
# 5.D FACTOR SPECIFIC-SHOCK SEMANTICS
# =========================================================================== #

def test_factor_specific_shock_semantics():
    """Verify distinct semantics: missing != explicit none != explicit zero vector."""
    weights = {"AAPL": 0.6, "MSFT": 0.4}
    exposures = pd.DataFrame(
        {"MKT": [1.2, 0.8], "TECH": [0.5, 0.6]},
        index=["AAPL", "MSFT"],
    )
    shocks = {"MKT": -0.05, "TECH": -0.02}

    # 1. Explicit NONE policy -> valid, specific_contribution is None
    res_none = apply_factor_scenario(
        weights=weights,
        exposures=exposures,
        scenario_spec_or_shocks=shocks,
        specific_shock_policy="NONE",
    )
    assert res_none.specific_contribution is None
    # factor_return = (0.6*1.2 + 0.4*0.8)*(-0.05) + (0.6*0.5 + 0.4*0.6)*(-0.02)
    #               = (0.72 + 0.32)*(-0.05) + (0.30 + 0.24)*(-0.02) = 1.04*(-0.05) + 0.54*(-0.02)
    #               = -0.0520 - 0.0108 = -0.0628
    assert math.isclose(res_none.scenario_return, -0.0628, rel_tol=1e-9)

    # 2. Explicit EXPLICIT_ZERO policy -> valid, specific_contribution == 0.0
    res_zero = apply_factor_scenario(
        weights=weights,
        exposures=exposures,
        scenario_spec_or_shocks=shocks,
        specific_shock_policy="EXPLICIT_ZERO",
    )
    assert res_zero.specific_contribution == 0.0
    assert math.isclose(res_zero.scenario_return, -0.0628, rel_tol=1e-9)

    # 3. Explicit SUPPLIED policy with valid vector -> valid, specific_contribution is float
    spec_shocks = {"AAPL": -0.01, "MSFT": 0.02}
    res_supp = apply_factor_scenario(
        weights=weights,
        exposures=exposures,
        scenario_spec_or_shocks=shocks,
        specific_shocks=spec_shocks,
        specific_shock_policy="SUPPLIED",
    )
    # specific_ret = 0.6*(-0.01) + 0.4*(0.02) = -0.006 + 0.008 = +0.002
    assert res_supp.specific_contribution is not None
    assert math.isclose(res_supp.specific_contribution, 0.002, rel_tol=1e-9)
    assert math.isclose(res_supp.scenario_return, -0.0628 + 0.002, rel_tol=1e-9)

    # 4. Fail closed on SUPPLIED/REQUIRED without vector
    with pytest.raises(ValueError, match="specific_shock_policy is 'SUPPLIED'"):
        apply_factor_scenario(
            weights=weights,
            exposures=exposures,
            scenario_spec_or_shocks=shocks,
            specific_shocks=None,
            specific_shock_policy="SUPPLIED",
        )


# =========================================================================== #
# 5.E REPRICING-METHOD COMPATIBILITY MATRIX
# =========================================================================== #

def test_repricing_method_compatibility_matrix():
    """Verify matrix validation accepting valid pairs and rejecting incompatible pairs."""
    # Valid
    ok1, _ = validate_repricing_shock_compatibility(RepricingMethod.LINEAR_RETURN, ShockSpace.ASSET_RETURN)
    assert ok1 is True
    ok2, _ = validate_repricing_shock_compatibility(RepricingMethod.FACTOR_LINEAR, ShockSpace.FACTOR_RETURN)
    assert ok2 is True
    ok3, _ = validate_repricing_shock_compatibility(RepricingMethod.DELTA, ShockSpace.RATE)
    assert ok3 is True
    ok4, _ = validate_repricing_shock_compatibility(RepricingMethod.DELTA_GAMMA, ShockSpace.VOLATILITY)
    assert ok4 is True

    # Invalid
    bad1, msg1 = validate_repricing_shock_compatibility(RepricingMethod.LINEAR_RETURN, ShockSpace.RATE)
    assert bad1 is False
    assert "Incompatible" in msg1

    bad2, msg2 = validate_repricing_shock_compatibility(RepricingMethod.FACTOR_LINEAR, ShockSpace.ASSET_RETURN)
    assert bad2 is False
    assert "Incompatible" in msg2

    bad3, msg3 = validate_repricing_shock_compatibility(RepricingMethod.FULL_REVALUATION_ADAPTER, ShockSpace.ASSET_RETURN)
    assert bad3 is False
    assert "deferred" in msg3.lower()


# =========================================================================== #
# 6. DELTA-GAMMA SCIENTIFIC KNOWN ANSWER
# =========================================================================== #

def test_delta_gamma_scientific_known_answer():
    """Verify 2-factor Delta-Gamma formula: P&L = delta' dx + 0.5 dx' Gamma dx with full symmetric Hessian."""
    sens = {
        "F1": SensitivitySpec(risk_factor_id="F1", delta=100.0, gamma=-50.0),
        "F2": SensitivitySpec(risk_factor_id="F2", delta=-200.0, gamma=150.0),
    }
    dx = {"F1": 0.10, "F2": -0.05}
    Gamma_mat = np.array([
        [-50.0, 20.0],
        [20.0, 150.0],
    ])

    # Delta term: 100*(0.10) + (-200)*(-0.05) = 10.0 + 10.0 = 20.0
    # Quadratic term: 0.5 * [ -50*(0.10)^2 + 150*(-0.05)^2 + 2*20*(0.10)*(-0.05) ]
    #               = 0.5 * [ -0.50 + 0.375 - 0.20 ] = 0.5 * [-0.325] = -0.1625
    # Total P&L = 20.0 - 0.1625 = 19.8375
    res = apply_delta_gamma_scenario(
        sensitivities=sens,
        scenario_spec_or_shocks=dx,
        gamma_matrix=Gamma_mat,
    )
    assert math.isclose(res.scenario_return, 19.8375, rel_tol=1e-9)
    assert math.isclose(res.scenario_loss, -19.8375, rel_tol=1e-9)
    assert math.isclose(res.factor_contributions["F1"] + res.factor_contributions["F2"], 19.8375, rel_tol=1e-9)
    assert res.reconciliation_error < 1e-12

    # Asymmetric Gamma matrix must fail closed
    asym_gamma = np.array([
        [-50.0, 20.0],
        [10.0, 150.0],  # 10 != 20
    ])
    with pytest.raises(ValueError, match="Gamma matrix must be symmetric"):
        apply_delta_gamma_scenario(sensitivities=sens, scenario_spec_or_shocks=dx, gamma_matrix=asym_gamma)


# =========================================================================== #
# 7. REVERSE-STRESS SUPPORTED SCOPE
# =========================================================================== #

def test_reverse_stress_supported_scope():
    """Verify that supported scopes run while deferred scopes fail closed with explicit disclosure."""
    spec_lin = ReverseStressSpec(target_loss=0.05, repricing_method=RepricingMethod.LINEAR_RETURN)
    res_lin = solve_reverse_stress(spec_lin, {"F1": 0.5, "F2": 0.5})
    assert res_lin.converged is True

    spec_fact = ReverseStressSpec(target_loss=0.05, repricing_method=RepricingMethod.FACTOR_LINEAR)
    res_fact = solve_reverse_stress(spec_fact, {"F1": 0.5, "F2": 0.5})
    assert res_fact.converged is True

    spec_quad = ReverseStressSpec(target_loss=0.05, repricing_method=RepricingMethod.DELTA_GAMMA)
    with pytest.raises(NotImplementedError, match="DELTA_GAMMA reverse stress is deferred"):
        solve_reverse_stress(spec_quad, {"F1": 0.5, "F2": 0.5})


# =========================================================================== #
# 8. UNCONSTRAINED L2 REVERSE STRESS KNOWN ANSWER
# =========================================================================== #

def test_unconstrained_l2_reverse_stress_known_answer():
    """Verify exact analytical solution x* = -L* c / (c' c) for unconstrained L2 reverse stress."""
    c = {"F1": 0.4, "F2": 0.3}
    L_star = 0.08
    spec = ReverseStressSpec(target_loss=L_star, distance_norm=ReverseStressNorm.L2)

    # c' c = 0.4^2 + 0.3^2 = 0.16 + 0.09 = 0.25
    # x* = -0.08 * [0.4, 0.3]' / 0.25 = [-0.128, -0.096]'
    # Loss(x*) = -(0.4*(-0.128) + 0.3*(-0.096)) = -(-0.0512 - 0.0288) = 0.0800
    # Distance = sqrt((-0.128)^2 + (-0.096)^2) = sqrt(0.016384 + 0.009216) = sqrt(0.0256) = 0.1600
    res = solve_reverse_stress(spec, c)

    assert res.converged is True
    assert res.is_closed_form is True
    assert res.solver_status == "OPTIMAL_CLOSED_FORM"
    assert math.isclose(res.shock_vector["F1"], -0.128, rel_tol=1e-9)
    assert math.isclose(res.shock_vector["F2"], -0.096, rel_tol=1e-9)
    assert math.isclose(res.achieved_loss, 0.0800, rel_tol=1e-9)
    assert math.isclose(res.distance, 0.1600, rel_tol=1e-9)
    assert res.loss_gap < 1e-12


# =========================================================================== #
# 9. ZERO EXPOSURE REVERSE STRESS FAILS CLOSED
# =========================================================================== #

def test_zero_exposure_reverse_stress_fails_closed():
    """Verify that zero sensitivities with positive target loss fails closed without div-by-zero or NaN."""
    c_zero = {"F1": 0.0, "F2": 0.0}
    spec = ReverseStressSpec(target_loss=0.05, distance_norm=ReverseStressNorm.L2)

    with pytest.raises(ValueError, match="All sensitivities in reverse stress problem are zero"):
        solve_reverse_stress(spec, c_zero)


# =========================================================================== #
# 10. BOUNDED REVERSE STRESS: ACTIVE BOUNDS & INFEASIBLE TARGET
# =========================================================================== #

def test_bounded_reverse_stress_active_bounds_and_infeasible():
    """Verify bounded QP solver when analytical L2 violates bounds, and fail-closed on infeasible target."""
    c = {"F1": 0.4, "F2": 0.3}
    L_star = 0.08

    # Bound: F1 >= -0.05 (unconstrained wants -0.128)
    bounds = {"F1": (-0.05, 0.10), "F2": (-0.50, 0.50)}
    spec_bounded = ReverseStressSpec(target_loss=L_star, distance_norm=ReverseStressNorm.L2, shock_bounds=bounds)

    # Constrained optimum: x_1 = -0.05, x_2 = (-0.08 - 0.4*(-0.05)) / (-0.3) = (-0.08 + 0.02)/(-0.3) = -0.20
    res_b = solve_reverse_stress(spec_bounded, c)
    assert res_b.converged is True
    assert res_b.is_closed_form is False
    assert math.isclose(res_b.shock_vector["F1"], -0.05, abs_tol=1e-4)
    assert math.isclose(res_b.shock_vector["F2"], -0.20, abs_tol=1e-4)
    assert math.isclose(res_b.achieved_loss, 0.0800, rel_tol=1e-4)
    assert res_b.distance > 0.1600  # Constrained distance is strictly larger than unconstrained 0.1600

    # Infeasible target: bounds [-0.01, 0.01] -> max achievable loss = -(0.4*(-0.01) + 0.3*(-0.01)) = 0.007 < 0.08
    tight_bounds = {"F1": (-0.01, 0.01), "F2": (-0.01, 0.01)}
    spec_infeasible = ReverseStressSpec(target_loss=L_star, distance_norm=ReverseStressNorm.L2, shock_bounds=tight_bounds)
    res_inf = solve_reverse_stress(spec_infeasible, c)
    assert res_inf.converged is False
    assert res_inf.solver_status in ("INFEASIBLE", "BOUNDS_BREACHED")


# =========================================================================== #
# 11. WEIGHTED L2 REVERSE STRESS
# =========================================================================== #

def test_weighted_l2_reverse_stress_known_answer_and_rejection():
    """Verify weighted L2 reverse stress known answer in direction W^-1 c and non-positive weight rejection."""
    c = {"F1": 0.4, "F2": 0.3}
    L_star = 0.08
    W = np.diag([4.0, 1.0])

    spec_w = ReverseStressSpec(
        target_loss=L_star,
        distance_norm=ReverseStressNorm.WEIGHTED_L2,
        weight_matrix=W,
    )
    # Direction W^-1 c = [0.4/4.0, 0.3/1.0]' = [0.1, 0.3]'
    # Normalizer c' W^-1 c = 0.4(0.1) + 0.3(0.3) = 0.04 + 0.09 = 0.13
    # x* = -0.08 * [0.1, 0.3]' / 0.13 = [-0.06153846, -0.18461538]'
    res_w = solve_reverse_stress(spec_w, c)
    assert res_w.converged is True
    assert math.isclose(res_w.shock_vector["F1"], -0.08 * 0.1 / 0.13, abs_tol=1e-4)
    assert math.isclose(res_w.shock_vector["F2"], -0.08 * 0.3 / 0.13, abs_tol=1e-4)
    assert math.isclose(res_w.achieved_loss, 0.0800, abs_tol=1e-4)

    # Rejection of non-positive weights
    spec_bad_w = ReverseStressSpec(
        target_loss=L_star,
        distance_norm=ReverseStressNorm.WEIGHTED_L2,
        scaling_factors={"F1": -1.0, "F2": 1.0},
    )
    with pytest.raises(ValueError, match="strictly positive"):
        solve_reverse_stress(spec_bad_w, c)


# =========================================================================== #
# 12. HETEROGENEOUS-UNIT L2 REJECTION
# =========================================================================== #

def test_heterogeneous_unit_reverse_stress_rejection():
    """Verify rejection of unscaled Euclidean distance mixing raw heterogeneous financial coordinates."""
    c = {"EQUITY": 0.5, "RATE_BPS": 10.0, "VOL_PTS": 2.0}
    units = {"EQUITY": "RETURN_DECIMAL", "RATE_BPS": "BASIS_POINTS", "VOL_PTS": "VOLATILITY_POINTS"}

    spec_raw = ReverseStressSpec(
        target_loss=0.10,
        distance_norm=ReverseStressNorm.L2,
        risk_factor_units=units,
        is_heterogeneous_unscaled=True,
    )
    with pytest.raises(ValueError, match="Unscaled L2 distance is undefined across heterogeneous"):
        solve_reverse_stress(spec_raw, c)


# =========================================================================== #
# 13. MAHALANOBIS REVERSE STRESS
# =========================================================================== #

def test_mahalanobis_reverse_stress_metadata_and_fail_closed():
    """Verify Mahalanobis metadata alignment, label checking, and fail-closed on singular/indefinite matrix."""
    c = {"F1": 0.4, "F2": 0.3}
    L_star = 0.08
    cov_valid = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["F1", "F2"], columns=["F1", "F2"])

    spec_mah = ReverseStressSpec(
        target_loss=L_star,
        distance_norm=ReverseStressNorm.MAHALANOBIS,
        reference_covariance=cov_valid.to_numpy().tolist(),
        covariance=cov_valid,
    )
    res_m = solve_reverse_stress(spec_mah, c)
    assert res_m.converged is True
    assert math.isclose(res_m.achieved_loss, 0.0800, abs_tol=1e-4)

    # Label mismatch rejection
    cov_mismatch = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["WRONG1", "WRONG2"], columns=["WRONG1", "WRONG2"])
    spec_bad_labels = ReverseStressSpec(
        target_loss=L_star,
        distance_norm=ReverseStressNorm.MAHALANOBIS,
        covariance=cov_mismatch,
    )
    with pytest.raises(ValueError, match="Reference covariance labels or order do not match"):
        solve_reverse_stress(spec_bad_labels, c)

    # Singular / Indefinite matrix rejection (no silent pseudoinverse / repair)
    cov_singular = [[1.0, 1.0], [1.0, 1.0]]  # Rank 1 singular
    spec_singular = ReverseStressSpec(
        target_loss=L_star,
        distance_norm=ReverseStressNorm.MAHALANOBIS,
        reference_covariance=cov_singular,
    )
    with pytest.raises(ValueError, match="strictly positive-definite"):
        solve_reverse_stress(spec_singular, c)

    # Unit mismatch rejection (covariance in RETURN_DECIMAL, factor in BASIS_POINTS)
    spec_bad_unit = ReverseStressSpec(
        target_loss=L_star,
        distance_norm=ReverseStressNorm.MAHALANOBIS,
        reference_covariance=cov_valid.to_numpy().tolist(),
        risk_factor_units={"F1": "BASIS_POINTS", "F2": "RETURN_DECIMAL"},
        provenance={"covariance_unit": "RETURN_DECIMAL"},
    )
    with pytest.raises(ValueError, match="Mahalanobis geometry unit mismatch"):
        solve_reverse_stress(spec_bad_unit, c)


# =========================================================================== #
# 14. HISTORICAL REPLAY PROVENANCE
# =========================================================================== #

def test_historical_replay_provenance():
    """Verify required provenance fields (source reference, observation date, fingerprint)."""
    weights = {"AAPL": 0.5, "MSFT": 0.5}
    hist_shocks = {"AAPL": -0.08, "MSFT": -0.04}

    # Missing source reference fails closed
    with pytest.raises(ValueError, match="Historical replay requires explicit source_reference"):
        replay_historical_scenario(hist_shocks, weights, source_reference="", observation_date="2008-09-15")

    # Missing observation date fails closed
    with pytest.raises(ValueError, match="Historical replay requires explicit observation_date"):
        replay_historical_scenario(hist_shocks, weights, source_reference="CRSP_DAILY", observation_date="")

    # Valid execution
    res = replay_historical_scenario(hist_shocks, weights, source_reference="CRSP_DAILY", observation_date="2008-09-15")
    assert res.scenario_id == "HIST-2008-09-15"
    assert math.isclose(res.scenario_return, -0.06, rel_tol=1e-9)


# =========================================================================== #
# 15. HISTORICAL ASSET COVERAGE & PROXY MAPPING
# =========================================================================== #

def test_historical_replay_asset_coverage_and_proxy_mapping():
    """Verify missing asset failure without proxy, and explicit proxy mapping provenance."""
    weights = {"AAPL": 0.5, "NEW_IPO": 0.5}
    hist_shocks = {"AAPL": -0.08, "TECH_PROXY": -0.06}

    # Missing asset without proxy fails closed
    with pytest.raises(ValueError, match="Historical shocks missing for assets"):
        replay_historical_scenario(hist_shocks, weights, source_reference="CRSP", observation_date="2020-03-16")

    # Explicit proxy mapping succeeds with disclosure
    res_proxy = replay_historical_scenario(
        hist_shocks,
        weights,
        source_reference="CRSP",
        observation_date="2020-03-16",
        proxy_mappings={"NEW_IPO": "TECH_PROXY"},
    )
    assert res_proxy.scenario_type == "HISTORICAL_REPLAY_WITH_MAPPING"
    assert math.isclose(res_proxy.scenario_return, 0.5 * (-0.08) + 0.5 * (-0.06), rel_tol=1e-9)


# =========================================================================== #
# 16. HISTORICAL CURRENCY SEMANTICS
# =========================================================================== #

def test_historical_replay_currency_mismatch_rejection():
    """Verify rejection of mismatched currencies without explicit FX conversion policy."""
    weights = {"AAPL": 0.5, "SAP": 0.5}
    hist_shocks = {"AAPL": -0.05, "SAP": -0.04}

    with pytest.raises(ValueError, match="Currency mismatch between historical scenario source"):
        replay_historical_scenario(
            hist_shocks,
            weights,
            source_reference="CRSP",
            observation_date="2020-03-16",
            source_currency="EUR",
            portfolio_currency="USD",
            fx_policy=None,
        )


# =========================================================================== #
# 17. ACTIVE SCENARIO EXACT IDENTITY
# =========================================================================== #

def test_active_scenario_exact_identity():
    """Verify exact mathematical identity: R_port - R_bmk == R_active."""
    port_weights = {"AAPL": 0.6, "MSFT": 0.4}
    bmk_weights = {"AAPL": 0.3, "MSFT": 0.7}
    shocks = {"AAPL": -0.10, "MSFT": 0.05}

    res_act = apply_active_scenario(
        portfolio_weights=port_weights,
        benchmark_weights=bmk_weights,
        scenario_spec_or_shocks=shocks,
    )
    # R_port = 0.6*(-0.10) + 0.4*(0.05) = -0.06 + 0.02 = -0.04
    # R_bmk  = 0.3*(-0.10) + 0.7*(0.05) = -0.03 + 0.035 = +0.005
    # R_act  = -0.04 - 0.005 = -0.045
    assert math.isclose(res_act.portfolio_return, -0.04, rel_tol=1e-12)
    assert math.isclose(res_act.benchmark_return, 0.005, rel_tol=1e-12)
    assert math.isclose(res_act.active_return, -0.045, rel_tol=1e-12)
    assert math.isclose(res_act.portfolio_return - res_act.benchmark_return, res_act.active_return, rel_tol=1e-14)
    assert res_act.reconciliation_error < 1e-14


# =========================================================================== #
# 18. GROUP SCENARIO EXHAUSTIVE VS OVERLAPPING SEMANTICS
# =========================================================================== #

def test_group_scenario_exhaustive_vs_overlapping_semantics():
    """Verify exhaustive partition additive reconciliation vs overlapping analytical disclosure."""
    asset_contribs = {"AAPL": -0.06, "MSFT": -0.04, "JPM": -0.02}

    # 1. Exhaustive disjoint mapping
    disjoint_map = {"AAPL": "TECH", "MSFT": "TECH", "JPM": "FIN"}
    grp_disjoint = apply_group_scenario_decomposition(
        asset_contribs,
        disjoint_map,
        partition_contract=PartitionContract.EXHAUSTIVE_PARTITION,
    )
    assert math.isclose(grp_disjoint["TECH"], -0.10, rel_tol=1e-12)
    assert math.isclose(grp_disjoint["FIN"], -0.02, rel_tol=1e-12)
    assert math.isclose(sum(grp_disjoint.values()), sum(asset_contribs.values()), rel_tol=1e-12)

    # 2. Overlapping non-additive mapping
    overlap_map = {"AAPL": ["TECH", "MEGA_CAP"], "MSFT": ["TECH", "MEGA_CAP"], "JPM": ["FIN", "VALUE"]}
    grp_overlap = apply_group_scenario_decomposition(
        asset_contribs,
        overlap_map,
        partition_contract=PartitionContract.OVERLAPPING_ANALYTICAL,
    )
    assert math.isclose(grp_overlap["MEGA_CAP"], -0.10, rel_tol=1e-12)
    assert sum(grp_overlap.values()) != sum(asset_contribs.values())


# =========================================================================== #
# 19. MULTI-SCENARIO RANKING & COMPARABILITY
# =========================================================================== #

def test_multi_scenario_ranking_and_comparability():
    """Verify canonical ranking key scenario_loss (worst = max loss, best = min loss)."""
    weights = {"AAPL": 0.5, "MSFT": 0.5}
    r1 = apply_asset_return_scenario(weights, {"AAPL": -0.15, "MSFT": -0.10}, scenario_id="SCEN_CRISIS")  # loss = +0.125
    r2 = apply_asset_return_scenario(weights, {"AAPL": -0.02, "MSFT": -0.04}, scenario_id="SCEN_MODERATE")  # loss = +0.030
    r3 = apply_asset_return_scenario(weights, {"AAPL": 0.05, "MSFT": 0.03}, scenario_id="SCEN_RALLY")  # loss = -0.040

    set_res = compare_scenario_set([r1, r2, r3])
    assert set_res.worst_scenario_id == "SCEN_CRISIS"
    assert math.isclose(set_res.worst_scenario_loss, 0.125, rel_tol=1e-9)
    assert set_res.best_scenario_id == "SCEN_RALLY"
    assert math.isclose(set_res.best_scenario_loss, -0.040, rel_tol=1e-9)
    assert set_res.loss_rankings == ("SCEN_CRISIS", "SCEN_MODERATE", "SCEN_RALLY")

    # Empty set rejection
    with pytest.raises(ValueError, match="Cannot compare an empty set"):
        compare_scenario_set([])

    # Cross-method comparability with disclosure (same portfolio state weights)
    rf1 = apply_factor_scenario(
        weights,
        pd.DataFrame({"MKT": [1.0, 1.0]}, index=["AAPL", "MSFT"]),
        {"MKT": -0.08},
        scenario_id="SCEN_FACTOR",
    )
    set_cross = compare_scenario_set([r1, rf1], allow_cross_method=True)
    assert set_cross.comparability_valid is True
    assert set_cross.method_disclosures["SCEN_CRISIS"] == RepricingMethod.LINEAR_RETURN.value
    assert set_cross.method_disclosures["SCEN_FACTOR"] == RepricingMethod.FACTOR_LINEAR.value
    assert any("Cross-method comparison" in lim for lim in set_cross.limitations)

    # Cross-method rejection when allow_cross_method is False
    set_incompat = compare_scenario_set([r1, rf1], allow_cross_method=False)
    assert set_incompat.comparability_valid is False
    assert any("Incompatible repricing methods" in lim for lim in set_incompat.limitations)


def test_scenario_set_horizon_mismatch_rejection():
    """Verify that scenarios evaluated over different horizons fail comparability."""
    weights = {"AAPL": 0.5, "MSFT": 0.5}
    r1 = apply_asset_return_scenario(weights, {"AAPL": -0.10, "MSFT": -0.05}, horizon=MetricHorizon.PERIODIC, scenario_id="S_1D")
    r2 = apply_asset_return_scenario(weights, {"AAPL": -0.20, "MSFT": -0.15}, horizon=MetricHorizon.ANNUAL, scenario_id="S_1Y")

    set_res = compare_scenario_set([r1, r2])
    assert set_res.comparability_valid is False
    assert any("Incompatible scenario horizons" in lim for lim in set_res.limitations)


def test_scenario_set_currency_mismatch_rejection():
    """Verify that scenarios evaluated with different base currencies fail comparability without silent FX conversion."""
    spec_usd = ScenarioSpec(
        scenario_id="S_USD",
        scenario_name="USD Spec",
        scenario_type=ScenarioType.SYNTHETIC.value,
        shocks=(create_scenario_shock("AAPL", -0.10),),
        repricing_method=RepricingMethod.LINEAR_RETURN.value,
        currency="USD",
    )
    spec_eur = ScenarioSpec(
        scenario_id="S_EUR",
        scenario_name="EUR Spec",
        scenario_type=ScenarioType.SYNTHETIC.value,
        shocks=(create_scenario_shock("AAPL", -0.10),),
        repricing_method=RepricingMethod.LINEAR_RETURN.value,
        currency="EUR",
    )
    r_usd = apply_asset_return_scenario({"AAPL": 1.0}, spec_usd)
    r_eur = apply_asset_return_scenario({"AAPL": 1.0}, spec_eur)

    set_res = compare_scenario_set([r_usd, r_eur])
    assert set_res.comparability_valid is False
    assert any("Incompatible scenario base currencies" in lim for lim in set_res.limitations)


def test_scenario_set_portfolio_state_mismatch_rejection():
    """Verify that scenarios evaluated on different portfolio states fail comparability."""
    r_state1 = apply_asset_return_scenario({"AAPL": 1.0}, {"AAPL": -0.10}, scenario_id="S_PORT1")
    r_state2 = apply_asset_return_scenario({"MSFT": 1.0}, {"MSFT": -0.10}, scenario_id="S_PORT2")

    set_res = compare_scenario_set([r_state1, r_state2])
    assert set_res.comparability_valid is False
    assert any("Incompatible portfolio states" in lim for lim in set_res.limitations)


def test_scenario_set_loss_pnl_basis_mismatch_rejection():
    """Verify that scenarios ranked on monetary loss fail comparability if portfolio values differ or are missing."""
    r_pv1 = apply_asset_return_scenario({"AAPL": 1.0}, {"AAPL": -0.10}, portfolio_value=1_000_000.0, scenario_id="S_1M")
    r_pv2 = apply_asset_return_scenario({"AAPL": 1.0}, {"AAPL": -0.10}, portfolio_value=5_000_000.0, scenario_id="S_5M")
    r_nopv = apply_asset_return_scenario({"AAPL": 1.0}, {"AAPL": -0.10}, portfolio_value=None, scenario_id="S_NOPV")

    # A) Monetary ranking with differing portfolio values
    set_diff_pv = compare_scenario_set([r_pv1, r_pv2], ranking_metric="scenario_pnl")
    assert set_diff_pv.comparability_valid is False
    assert any("Incompatible portfolio values" in lim for lim in set_diff_pv.limitations)

    # B) Monetary ranking with missing portfolio value
    set_nopv = compare_scenario_set([r_pv1, r_nopv], ranking_metric="scenario_monetary_loss")
    assert set_nopv.comparability_valid is False
    assert any("requires portfolio_value" in lim for lim in set_nopv.limitations)


def test_scenario_set_cross_method_opt_in_does_not_override_incompatibilities():
    """Verify combined invariant: allow_cross_method=True does NOT override horizon, currency, state, or basis mismatches."""
    spec_usd = ScenarioSpec(
        scenario_id="S_LIN_USD",
        scenario_name="Linear USD",
        scenario_type=ScenarioType.SYNTHETIC.value,
        shocks=(create_scenario_shock("AAPL", -0.10),),
        repricing_method=RepricingMethod.LINEAR_RETURN.value,
        currency="USD",
        horizon=MetricHorizon.PERIODIC,
    )
    spec_eur_ann = ScenarioSpec(
        scenario_id="S_FACT_EUR",
        scenario_name="Factor EUR Annual",
        scenario_type=ScenarioType.SYNTHETIC.value,
        shocks=(create_scenario_shock("MKT", -0.08, shock_space=ShockSpace.FACTOR_RETURN),),
        repricing_method=RepricingMethod.FACTOR_LINEAR.value,
        currency="EUR",
        horizon=MetricHorizon.ANNUAL,
    )

    r_linear = apply_asset_return_scenario({"AAPL": 1.0}, spec_usd, portfolio_value=1_000_000.0)
    r_factor = apply_factor_scenario({"AAPL": 1.0}, pd.DataFrame({"MKT": [1.0]}, index=["AAPL"]), spec_eur_ann, portfolio_value=5_000_000.0)

    # Cross-method opt-in is True, but horizons, currencies, and portfolio values all differ
    set_res = compare_scenario_set([r_linear, r_factor], ranking_metric="scenario_loss", allow_cross_method=True)
    assert set_res.comparability_valid is False
    assert any("Incompatible scenario horizons" in lim for lim in set_res.limitations)
    # Method disclosure is present
    assert any("Cross-method comparison" in lim for lim in set_res.limitations)


# =========================================================================== #
# 20. DEFERRED-SCOPE TRUTH TABLE
# =========================================================================== #

def test_deferred_scope_truth_table():
    """Verify explicit deferred status for pricing adapters and nonlinear reverse stress."""
    # Pricing adapters deferred
    is_ok, msg = validate_repricing_shock_compatibility(RepricingMethod.FULL_REVALUATION_ADAPTER, ShockSpace.ASSET_RETURN)
    assert is_ok is False
    assert "deferred" in msg.lower()

    # Delta-gamma reverse stress deferred
    spec_dg = ReverseStressSpec(target_loss=0.05, repricing_method=RepricingMethod.DELTA_GAMMA)
    with pytest.raises(NotImplementedError, match="DELTA_GAMMA reverse stress is deferred"):
        solve_reverse_stress(spec_dg, {"F1": 0.5})


# =========================================================================== #
# 21. CHALLENGE PROVENANCE CLOSURE
# =========================================================================== #

def test_challenge_provenance_closure():
    """Verify adversarial challenge provenance: source ID -> diagnostic tool -> NEW evidence ID -> resolution."""
    # 1. Base Evidence Record
    base_res = apply_asset_return_scenario({"AAPL": 1.0}, {"AAPL": -0.10}, scenario_id="SCEN-TEST")
    base_ev = scenario_result_to_evidence(base_res)

    # 2. Challenge Agent creates targeted challenge citing base_ev.evidence_id
    challenge = AdversarialChallenge(
        challenge_id="CHAL-SCENARIO-METHOD-SENSITIVITY",
        challenger_agent="AdversarialChallengeAgent",
        target_area="SCENARIO_REPRICING",
        challenge_question="Linear repricing may omit convex tail exposure.",
        evidence_ids=(base_ev.evidence_id,),
        required_tool="apply_delta_gamma_scenario",
        parameters={"delta": 1.0, "gamma": 2.0},
    )

    # 3. Diagnostic tool runs and produces NEW distinct EvidenceRecord
    diag_res = apply_delta_gamma_scenario(
        sensitivities={"AAPL": SensitivitySpec(risk_factor_id="AAPL", delta=1.0, gamma=2.0)},
        scenario_spec_or_shocks={"AAPL": -0.10},
    )
    diag_ev = scenario_result_to_evidence(diag_res)
    assert diag_ev.evidence_id != base_ev.evidence_id

    # 4. Wrap into ChallengeResolution and diagnostic EvidenceRecord
    diag_wrapped_ev = challenge_result_to_diagnostic_evidence(
        tool_name=challenge.required_tool,
        tool_res=diag_res,
        params=challenge.parameters,
        source_evidence_ids=challenge.evidence_ids,
    )
    assert "adversarial" in diag_wrapped_ev.test_id or "scenario" in diag_wrapped_ev.test_id
    assert diag_wrapped_ev.evidence_id != base_ev.evidence_id

    resolution = ChallengeResolution(
        challenge_id=challenge.challenge_id,
        status=ChallengeState.RESOLVED_EVIDENCE_ONLY,
        tool_name=challenge.required_tool,
        source_evidence_ids=challenge.evidence_ids,
        generated_evidence_ids=(diag_wrapped_ev.evidence_id,),
        decision_criterion="NONE",
    )
    assert resolution.status == ChallengeState.RESOLVED_EVIDENCE_ONLY

    # 5. Governance Agent ingests resolution and signs off with ACCEPT_WITH_CONDITIONS
    gov_agent = GovernanceAgent()
    verdict = gov_agent.evaluate_signoff(
        critic_disposition="READY_FOR_GOVERNANCE",
        challenges=[{"challenge_id": challenge.challenge_id, "title": challenge.challenge_question}],
        findings=[],
        records=[base_ev, diag_wrapped_ev],
        resolutions=[{"challenge_id": challenge.challenge_id, "status": "RESOLVED_EVIDENCE_ONLY"}],
    )
    assert verdict["verdict"] in ("ACCEPT_WITH_CONDITIONS", "ACCEPT")


# =========================================================================== #
# 22. ARTIFACT PROVENANCE CLOSURE
# =========================================================================== #

def test_artifact_provenance_closure(tmp_path: Path):
    """Verify all 8 dual-plane SVG visual artifacts and JSON machine companions have non-empty evidence links."""
    out_dir = tmp_path / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)

    ev_id = "EV-20260901-000000-TEST"

    # 1. Scenario PnL Waterfall
    spec = ScenarioSpec("S1", "Spec1", ScenarioType.SYNTHETIC.value, (create_scenario_shock("A", -0.05),), RepricingMethod.LINEAR_RETURN.value)
    scen_res = apply_asset_return_scenario({"A": 1.0}, {"A": -0.05}, portfolio_value=1_000_000.0, scenario_id="S1")
    art1 = render_scenario_pnl_waterfall_artifact(scen_res, (ev_id,), output_dir=out_dir)

    # 2. Scenario Asset Contribution Table
    art2 = render_scenario_asset_contribution_artifact(scen_res, (ev_id,), output_dir=out_dir)

    # 3. Scenario Factor Contribution Table
    fact_res = apply_factor_scenario({"A": 1.0}, pd.DataFrame({"MKT": [1.0]}, index=["A"]), {"MKT": -0.05}, scenario_id="S1")
    art3 = render_scenario_factor_contribution_artifact(fact_res, (ev_id,), output_dir=out_dir)

    # 4. Scenario Active Comparison
    act_res = apply_active_scenario({"A": 0.6}, {"A": 0.4}, {"A": -0.10})
    art4 = render_scenario_active_comparison_artifact(act_res, (ev_id,), output_dir=out_dir)

    # 5. Group Stress Heatmap
    art5 = render_scenario_group_heatmap_artifact("S1", {"TECH": -0.05, "FIN": -0.02}, PartitionContract.EXHAUSTIVE_PARTITION.value, (ev_id,), output_dir=out_dir)

    # 6. Scenario Set Ranking
    set_res = compare_scenario_set([scen_res])
    art6 = render_scenario_set_ranking_artifact(set_res, (ev_id,), output_dir=out_dir)

    # 7. Scenario Sensitivity Curve
    points = evaluate_scenario_sensitivity_grid(spec, "A", [0.5, 1.0, 1.5], {"A": 1.0})
    art7 = render_scenario_sensitivity_curve_artifact(points, (ev_id,), output_dir=out_dir)

    # 8. Reverse Stress Profile
    rev_res = solve_reverse_stress(ReverseStressSpec(target_loss=0.05), {"F1": 0.4, "F2": 0.3})
    art8 = render_reverse_stress_profile_artifact(rev_res, (ev_id,), output_dir=out_dir)

    artifacts = [art1, art2, art3, art4, art5, art6, art7, art8]
    assert len(artifacts) == 8

    for art in artifacts:
        # 1. Non-empty Artifact ID
        assert isinstance(art.artifact_id, str) and len(art.artifact_id) > 0
        assert art.artifact_id.startswith("ART-")

        # 2. Non-empty EvidenceRecord ID(s)
        assert art.spec.evidence_ids == (ev_id,)
        assert len(art.spec.evidence_ids) > 0

        # 3. Non-empty Semantic Payload
        assert isinstance(art.semantic_payload, dict) and len(art.semantic_payload) > 0

        # 4. Valid Semantic Payload Hash matching canonical payload
        assert isinstance(art.semantic_payload_hash, str) and len(art.semantic_payload_hash) == 64
        expected_hash = _hash_payload(art.semantic_payload)
        assert art.semantic_payload_hash == expected_hash

        # 5. Non-empty Data Fingerprint
        assert isinstance(art.data_fingerprint, str) and len(art.data_fingerprint) > 0

        # 6. Renderer Metadata
        assert art.spec.artifact_type is not None
        assert art.rendering_format == "svg"

        # 7. Valid SVG vector file
        assert art.file_path is not None
        svg_p = Path(art.file_path)
        assert svg_p.exists()
        assert svg_p.stat().st_size > 0

        # Validate well-formed XML structure
        tree = ET.parse(svg_p)
        root = tree.getroot()
        assert root.tag.endswith("svg")

        # 8. Valid Companion JSON file matching payload exactly
        json_p = svg_p.with_suffix(".json")
        assert json_p.exists()
        assert json_p.stat().st_size > 0
        with open(json_p, encoding="utf-8") as f_json:
            loaded_json = json.load(f_json)
        assert _hash_payload(loaded_json) == art.semantic_payload_hash


# =========================================================================== #
# 23. REGISTRY CENSUS INVARIANT
# =========================================================================== #

def test_registry_census_remains_79():
    """Verify registry census maintains exactly 79 total, 79 unique, 0 duplicate tests."""
    tests = list_tests()
    test_ids = [s.test_id for s in tests]
    total_count = len(test_ids)
    unique_count = len(set(test_ids))

    assert total_count == 79, f"Expected exactly 79 registered root tests, got {total_count}."
    assert unique_count == 79, f"Expected 79 unique tests, got {unique_count}."
    assert total_count == unique_count, "Found duplicate test registrations in registry census."
