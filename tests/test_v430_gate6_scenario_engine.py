"""StART — Gate 6 Scenario & Deterministic Repricing Engines Test Suite.

Verifies:
1. Heterogeneous Shock Normalization & Unit Semantics
2. Repricing Method x Shock Space Compatibility Validation (Fail Closed)
3. Asset-Return Scenario Repricing & Exact Contribution Sum Reconciliation
4. Factor Scenario Repricing & Specific Risk Separation
5. Delta vs Delta-Gamma Repricing with Full Symmetric Hessian
6. Gamma Matrix Symmetry Enforcement
7. Portfolio vs Benchmark Active Stress Exact Additive Decomposition
8. Group / Sector Partition & Overlap Contracts
9. Multi-Scenario Set Ranking by Canonical Loss
10. Deterministic Sensitivity Grid Sweeps
11. Historical Replay Provenance & Proxy Mapping Contracts
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from start.portfolio.contracts import (
    PartitionContract,
    RepricingMethod,
    ScenarioResult,
    ScenarioSpec,
    ScenarioType,
    SensitivitySpec,
    ShockSpace,
    ShockUnit,
)
from start.portfolio.scenario import (
    apply_asset_return_scenario,
    apply_benchmark_active_scenario,
    apply_delta_gamma_scenario,
    apply_factor_scenario,
    apply_group_scenario_decomposition,
    compare_scenario_set,
    create_scenario_shock,
    normalize_shock,
    replay_historical_scenario,
    validate_repricing_shock_compatibility,
)

# =========================================================================== #
# 1. SHOCK NORMALIZATION & COMPATIBILITY
# =========================================================================== #


def test_heterogeneous_shock_normalization_conventions():
    """Verify exact raw-value and computational normalization rules across all units."""
    # 1. BASIS_POINTS (+100 bps -> +0.0100)
    norm_val, rule, comp_unit = normalize_shock(100.0, ShockUnit.BASIS_POINTS)
    assert math.isclose(norm_val, 0.0100, rel_tol=1e-9)
    assert "BASIS_POINTS" in rule
    assert comp_unit == ShockUnit.ABSOLUTE.value

    # -75 bps -> -0.0075
    norm_val2, _, _ = normalize_shock(-75.0, ShockUnit.BASIS_POINTS)
    assert math.isclose(norm_val2, -0.0075, rel_tol=1e-9)

    # 2. RELATIVE_PERCENT (-10% -> -0.10)
    norm_val3, rule3, comp_unit3 = normalize_shock(-10.0, ShockUnit.RELATIVE_PERCENT)
    assert math.isclose(norm_val3, -0.1000, rel_tol=1e-9)
    assert "RELATIVE_PERCENT" in rule3
    assert comp_unit3 == ShockUnit.RETURN_DECIMAL.value

    # 3. VOLATILITY_POINTS (+5.0 vol points -> +0.0500)
    norm_val4, rule4, comp_unit4 = normalize_shock(5.0, ShockUnit.VOLATILITY_POINTS)
    assert math.isclose(norm_val4, 0.0500, rel_tol=1e-9)
    assert "VOLATILITY_POINTS" in rule4
    assert comp_unit4 == ShockUnit.ABSOLUTE.value

    # 4. RETURN_DECIMAL (-0.10 -> -0.10)
    norm_val5, rule5, comp_unit5 = normalize_shock(-0.10, ShockUnit.RETURN_DECIMAL)
    assert math.isclose(norm_val5, -0.10, rel_tol=1e-9)
    assert comp_unit5 == ShockUnit.RETURN_DECIMAL.value

    # 5. ABSOLUTE & LOG_RETURN
    norm_val6, _, comp_unit6 = normalize_shock(0.025, ShockUnit.ABSOLUTE)
    assert math.isclose(norm_val6, 0.025, rel_tol=1e-9)
    assert comp_unit6 == ShockUnit.ABSOLUTE.value

    # 6. LOG_RETURN: log(0.90) -> -0.10
    raw_log = math.log(0.90)
    norm_val7, rule7, comp_unit7 = normalize_shock(raw_log, ShockUnit.LOG_RETURN)
    assert math.isclose(norm_val7, -0.10, rel_tol=1e-9)
    assert "LOG_RETURN" in rule7
    assert comp_unit7 == ShockUnit.RETURN_DECIMAL.value


def test_repricing_method_shock_space_compatibility():
    """Verify that repricing methods strictly reject incompatible shock spaces."""
    # LINEAR_RETURN accepts ASSET_RETURN, rejects YIELD / RATE
    is_ok, msg = validate_repricing_shock_compatibility(
        RepricingMethod.LINEAR_RETURN, ShockSpace.ASSET_RETURN
    )
    assert is_ok is True

    is_ok, msg = validate_repricing_shock_compatibility(RepricingMethod.LINEAR_RETURN, ShockSpace.YIELD)
    assert is_ok is False
    assert "Incompatible" in msg

    # FACTOR_LINEAR accepts FACTOR_RETURN, rejects ASSET_RETURN
    is_ok, msg = validate_repricing_shock_compatibility(
        RepricingMethod.FACTOR_LINEAR, ShockSpace.FACTOR_RETURN
    )
    assert is_ok is True

    is_ok, msg = validate_repricing_shock_compatibility(
        RepricingMethod.FACTOR_LINEAR, ShockSpace.ASSET_RETURN
    )
    assert is_ok is False

    # DELTA & DELTA_GAMMA support sensitivity-aligned spaces
    is_ok, _ = validate_repricing_shock_compatibility(RepricingMethod.DELTA, ShockSpace.YIELD)
    assert is_ok is True
    is_ok, _ = validate_repricing_shock_compatibility(RepricingMethod.DELTA_GAMMA, ShockSpace.VOLATILITY)
    assert is_ok is True

    # FULL_REVALUATION_ADAPTER is deferred in Gate 6
    is_ok, msg = validate_repricing_shock_compatibility(
        RepricingMethod.FULL_REVALUATION_ADAPTER, ShockSpace.PRICE
    )
    assert is_ok is False
    assert "unavailable" in msg.lower() or "deferred" in msg.lower()


# =========================================================================== #
# 2. LINEAR ASSET REPRICING & RECONCILIATION
# =========================================================================== #


def test_asset_return_scenario_known_answer():
    """Verify asset-return repricing on a 3-asset analytical known answer."""
    weights = {"A1": 0.50, "A2": 0.30, "A3": 0.20}
    shocks = (
        create_scenario_shock("A1", raw_value=-10.0, shock_unit=ShockUnit.RELATIVE_PERCENT),
        create_scenario_shock("A2", raw_value=-5.0, shock_unit=ShockUnit.RELATIVE_PERCENT),
        create_scenario_shock("A3", raw_value=+10.0, shock_unit=ShockUnit.RELATIVE_PERCENT),
    )
    spec = ScenarioSpec(
        scenario_id="SCEN-TEST-ASSET",
        scenario_name="Test Asset Scenario",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=shocks,
        repricing_method=RepricingMethod.LINEAR_RETURN,
    )

    res = apply_asset_return_scenario(
        weights=weights, scenario_spec_or_shocks=spec, portfolio_value=1_000_000.0
    )

    # Expected return: 0.50*(-0.10) + 0.30*(-0.05) + 0.20*(+0.10) = -0.050 - 0.015 + 0.020 = -0.045
    assert math.isclose(res.scenario_return, -0.045, rel_tol=1e-9)
    assert math.isclose(res.scenario_loss, +0.045, rel_tol=1e-9)  # Canonical positive loss
    assert math.isclose(res.scenario_pnl, -45_000.0, rel_tol=1e-9)
    assert math.isclose(res.scenario_monetary_loss, +45_000.0, rel_tol=1e-9)

    # Asset contributions
    assert math.isclose(res.asset_contributions["A1"], -0.050, rel_tol=1e-9)
    assert math.isclose(res.asset_contributions["A2"], -0.015, rel_tol=1e-9)
    assert math.isclose(res.asset_contributions["A3"], +0.020, rel_tol=1e-9)
    assert math.isclose(res.reconciliation_error, 0.0, abs_tol=1e-12)


# =========================================================================== #
# 3. FACTOR REPRICING & SPECIFIC RISK
# =========================================================================== #


def test_factor_scenario_known_answer():
    """Verify factor repricing and specific risk separation on known analytical fixtures."""
    weights = {"A1": 0.60, "A2": 0.40}
    exposures = pd.DataFrame(
        [
            [1.2, 0.5],  # A1
            [0.8, -0.5],  # A2
        ],
        index=["A1", "A2"],
        columns=["F1", "F2"],
    )
    # Portfolio exposure: b_p = [0.6*1.2 + 0.4*0.8, 0.6*0.5 + 0.4*(-0.5)] = [0.72 + 0.32, 0.30 - 0.20] = [1.04, 0.10]
    shocks = (
        create_scenario_shock(
            "F1", raw_value=-0.05, shock_unit=ShockUnit.RETURN_DECIMAL, shock_space=ShockSpace.FACTOR_RETURN
        ),
        create_scenario_shock(
            "F2", raw_value=+0.10, shock_unit=ShockUnit.RETURN_DECIMAL, shock_space=ShockSpace.FACTOR_RETURN
        ),
    )
    spec = ScenarioSpec(
        scenario_id="SCEN-TEST-FACTOR",
        scenario_name="Test Factor Scenario",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=shocks,
        repricing_method=RepricingMethod.FACTOR_LINEAR,
    )

    res = apply_factor_scenario(weights=weights, exposures=exposures, scenario_spec_or_shocks=spec)

    # Expected factor return: 1.04 * (-0.05) + 0.10 * (0.10) = -0.052 + 0.010 = -0.042
    assert math.isclose(res.scenario_return, -0.042, rel_tol=1e-9)
    assert math.isclose(res.factor_contributions["F1"], -0.052, rel_tol=1e-9)
    assert math.isclose(res.factor_contributions["F2"], +0.010, rel_tol=1e-9)
    assert math.isclose(res.reconciliation_error, 0.0, abs_tol=1e-12)

    # Test with specific shocks
    specific_shocks = {"A1": -0.02, "A2": +0.01}
    res_spec = apply_factor_scenario(
        weights=weights,
        exposures=exposures,
        scenario_spec_or_shocks=spec,
        specific_shocks=specific_shocks,
        specific_shock_policy="SUPPLIED",
    )
    # Expected specific return: 0.60*(-0.02) + 0.40*(+0.01) = -0.012 + 0.004 = -0.008
    # Total return: -0.042 + (-0.008) = -0.050
    assert math.isclose(res_spec.scenario_return, -0.050, rel_tol=1e-9)
    assert math.isclose(res_spec.specific_contribution, -0.008, rel_tol=1e-9)


# =========================================================================== #
# 4. DELTA VS DELTA-GAMMA NONLINEAR REPRICING & HESSIAN SYMMETRY
# =========================================================================== #


def test_delta_gamma_repricing_and_symmetry():
    """Verify Delta vs Delta-Gamma second-order approximation and Hessian symmetry checks."""
    sens = {
        "F1": SensitivitySpec("F1", delta=100.0, gamma=-50.0),
        "F2": SensitivitySpec("F2", delta=-200.0, gamma=150.0),
    }
    shocks = (
        create_scenario_shock("F1", raw_value=0.10, shock_unit=ShockUnit.ABSOLUTE),
        create_scenario_shock("F2", raw_value=-0.05, shock_unit=ShockUnit.ABSOLUTE),
    )
    spec = ScenarioSpec(
        scenario_id="SCEN-DG",
        scenario_name="Delta-Gamma Test",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=shocks,
        repricing_method=RepricingMethod.DELTA_GAMMA,
    )

    # 1. Diagonal-only Delta-Gamma
    # Delta term: 100*0.10 + (-200)*(-0.05) = 10.0 + 10.0 = 20.0
    # Gamma term: 0.5 * (-50 * 0.10^2) + 0.5 * (150 * (-0.05)^2) = 0.5*(-0.50) + 0.5*(0.375) = -0.25 + 0.1875 = -0.0625
    # Total P&L: 20.0 - 0.0625 = 19.9375
    res_dg = apply_delta_gamma_scenario(
        sensitivities=sens, scenario_spec_or_shocks=spec, method=RepricingMethod.DELTA_GAMMA
    )
    assert math.isclose(res_dg.scenario_return, 19.9375, rel_tol=1e-9)

    # Delta-only baseline
    res_d = apply_delta_gamma_scenario(
        sensitivities=sens, scenario_spec_or_shocks=spec, method=RepricingMethod.DELTA
    )
    assert math.isclose(res_d.scenario_return, 20.0, rel_tol=1e-9)

    # 2. Full Symmetric Hessian Matrix with cross-gamma
    gamma_mat = np.array(
        [
            [-50.0, 20.0],
            [20.0, 150.0],
        ]
    )
    # Cross-gamma term: 0.5 * 2 * (0.10 * -0.05 * 20.0) = -0.10
    # Total with cross-gamma: 19.9375 - 0.10 = 19.8375
    res_full = apply_delta_gamma_scenario(
        sensitivities=sens, scenario_spec_or_shocks=spec, gamma_matrix=gamma_mat
    )
    assert math.isclose(res_full.scenario_return, 19.8375, rel_tol=1e-9)

    # 3. Asymmetric Gamma Matrix must fail closed
    asym_gamma = np.array(
        [
            [-50.0, 25.0],
            [15.0, 150.0],
        ]
    )
    with pytest.raises(ValueError, match="must be symmetric"):
        apply_delta_gamma_scenario(sensitivities=sens, scenario_spec_or_shocks=spec, gamma_matrix=asym_gamma)


# =========================================================================== #
# 5. BENCHMARK & ACTIVE STRESS DECOMPOSITION
# =========================================================================== #


def test_benchmark_active_stress_exact_identity():
    """Verify exact active return identity: R_port - R_bmk == R_active."""
    weights = {"A1": 0.40, "A2": 0.35, "A3": 0.25}
    bmk_weights = {"A1": 0.333333, "A2": 0.333333, "A3": 0.333334}
    shocks = (
        create_scenario_shock("A1", raw_value=-0.08, shock_unit=ShockUnit.RETURN_DECIMAL),
        create_scenario_shock("A2", raw_value=+0.04, shock_unit=ShockUnit.RETURN_DECIMAL),
        create_scenario_shock("A3", raw_value=-0.02, shock_unit=ShockUnit.RETURN_DECIMAL),
    )
    spec = ScenarioSpec(
        scenario_id="SCEN-ACTIVE-TEST",
        scenario_name="Active Stress Test",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=shocks,
        repricing_method=RepricingMethod.LINEAR_RETURN,
    )

    act_res = apply_benchmark_active_scenario(
        weights=weights, benchmark_weights=bmk_weights, scenario_spec_or_shocks=spec
    )

    # Portfolio return: 0.40*(-0.08) + 0.35*(0.04) + 0.25*(-0.02) = -0.032 + 0.014 - 0.005 = -0.023
    assert math.isclose(act_res.portfolio_return, -0.023, rel_tol=1e-9)
    # Active return reconciliation
    assert math.isclose(
        act_res.portfolio_return - act_res.benchmark_return, act_res.active_return, abs_tol=1e-12
    )
    assert math.isclose(act_res.reconciliation_error, 0.0, abs_tol=1e-12)


# =========================================================================== #
# 6. GROUP PARTITION & OVERLAP CONTRACTS
# =========================================================================== #


def test_group_partition_and_overlap():
    """Verify group stress decomposition under EXHAUSTIVE_PARTITION vs OVERLAPPING_ANALYTICAL."""
    asset_contribs = {"A1": -0.030, "A2": -0.015, "A3": +0.010}

    # 1. Disjoint Groups
    disjoint_mapping = {"A1": "TECH", "A2": "TECH", "A3": "ENERGY"}
    grp_res = apply_group_scenario_decomposition(
        asset_contribs, disjoint_mapping, partition_contract=PartitionContract.EXHAUSTIVE_PARTITION
    )
    assert math.isclose(grp_res["TECH"], -0.045, rel_tol=1e-9)
    assert math.isclose(grp_res["ENERGY"], +0.010, rel_tol=1e-9)
    assert math.isclose(sum(grp_res.values()), sum(asset_contribs.values()), abs_tol=1e-12)

    # 2. Overlapping Groups
    overlap_mapping = {"A1": ["TECH", "ESG"], "A2": ["TECH"], "A3": ["ENERGY", "ESG"]}
    grp_overlap = apply_group_scenario_decomposition(
        asset_contribs, overlap_mapping, partition_contract=PartitionContract.OVERLAPPING_ANALYTICAL
    )
    assert math.isclose(grp_overlap["TECH"], -0.045, rel_tol=1e-9)
    assert math.isclose(grp_overlap["ENERGY"], +0.010, rel_tol=1e-9)
    assert math.isclose(grp_overlap["ESG"], -0.020, rel_tol=1e-9)
    # Sum over overlapping groups exceeds total (non-additive analytical view)
    assert not math.isclose(sum(grp_overlap.values()), sum(asset_contribs.values()))


# =========================================================================== #
# 7. MULTI-SCENARIO SET COMPARATIVE LOSS RANKING
# =========================================================================== #


def test_multi_scenario_set_ranking():
    """Verify multi-scenario loss ranking (worst scenario is maximum canonical loss)."""
    r1 = ScenarioResult(
        "SCEN-1",
        "SYNTHETIC",
        "LINEAR_RETURN",
        -0.05,
        0.05,
        None,
        None,
        None,
        {},
        {},
        None,
        {},
        "EXHAUSTIVE_PARTITION",
        0.0,
        True,
    )
    r2 = ScenarioResult(
        "SCEN-2",
        "SYNTHETIC",
        "FACTOR_LINEAR",
        -0.12,
        0.12,
        None,
        None,
        None,
        {},
        {},
        None,
        {},
        "EXHAUSTIVE_PARTITION",
        0.0,
        True,
    )
    r3 = ScenarioResult(
        "SCEN-3",
        "SYNTHETIC",
        "DELTA_GAMMA",
        +0.02,
        -0.02,
        None,
        None,
        None,
        {},
        {},
        None,
        {},
        "EXHAUSTIVE_PARTITION",
        0.0,
        True,
    )

    set_res = compare_scenario_set([r1, r2, r3], ranking_metric="scenario_loss")

    # Sorted descending by loss: SCEN-2 (0.12), SCEN-1 (0.05), SCEN-3 (-0.02)
    assert set_res.loss_rankings == ("SCEN-2", "SCEN-1", "SCEN-3")
    assert set_res.worst_scenario_id == "SCEN-2"
    assert math.isclose(set_res.worst_scenario_loss, 0.12, rel_tol=1e-9)
    assert set_res.best_scenario_id == "SCEN-3"
    assert math.isclose(set_res.best_scenario_loss, -0.02, rel_tol=1e-9)


# =========================================================================== #
# 8. HISTORICAL REPLAY PROVENANCE & PROXY MAPPING
# =========================================================================== #


def test_historical_replay_provenance_and_proxy():
    """Verify historical replay enforces source reference, observation dates, and explicit proxy mappings."""
    weights = {"AAPL": 0.60, "NEW_IPO": 0.40}
    hist_shocks = {"AAPL": -0.08, "TECH_ETF": -0.07}

    # 1. Missing source reference must fail
    with pytest.raises(ValueError, match="source_reference"):
        replay_historical_scenario(hist_shocks, weights, source_reference="", observation_date="2020-03-16")

    # 2. Unmapped asset without proxy must fail closed
    with pytest.raises(ValueError, match="no valid proxy"):
        replay_historical_scenario(
            hist_shocks, weights, source_reference="CRISIS_DATA_V1", observation_date="2020-03-16"
        )

    # 3. Explicit proxy mapping succeeds with provenanced scenario type
    res_proxy = replay_historical_scenario(
        hist_shocks,
        weights,
        source_reference="CRISIS_DATA_V1",
        observation_date="2020-03-16",
        proxy_mappings={"NEW_IPO": "TECH_ETF"},
    )
    assert res_proxy.scenario_type == "HISTORICAL_REPLAY_WITH_MAPPING"
    # Return: 0.60*(-0.08) + 0.40*(-0.07) = -0.048 - 0.028 = -0.076
    assert math.isclose(res_proxy.scenario_return, -0.076, rel_tol=1e-9)
