"""StART — GATE 6 SHOWCASE: INSTITUTIONAL SCENARIO, STRESS & REVERSE-STRESS INTELLIGENCE.

Demonstrates:
1. Heterogeneous Scenario Shock Normalization & Semantic Contracts
2. Linear Asset-Return Repricing & Asset Contribution Waterfall
3. Factor Linear Repricing & Factor Shock Decomposition
4. Delta vs Delta-Gamma Second-Order Sensitivity Repricing with Full Symmetric Hessian
5. Portfolio vs Benchmark Active Stress Decomposition
6. Analytical Group / Sector Stress Heatmap
7. Multi-Scenario Comparative Loss Ranking (Worst / Best Identification)
8. Deterministic Risk Factor Sensitivity Sweep Grid
9. Closed-Form L2 and Bounded Mahalanobis Reverse Stress Testing
10. Specialist ScenarioStressAgent Synthesis & Deterministic Challenge Resolution
11. Proof-Carrying Negative Evidence: Repricing Method Discrepancy without Materiality Threshold -> ACCEPT_WITH_CONDITIONS

Emits dual-plane vector SVG visuals, JSON machine companions, and manifest to start_output/gate6_showcase/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from start.agents.market_review import (
    MarketReviewDirectorAgent,
)
from start.core.schemas import EvidenceRecord
from start.portfolio import (
    MetricHorizon,
    PartitionContract,
    RepricingMethod,
    ReverseStressNorm,
    ReverseStressSpec,
    ScenarioSpec,
    ScenarioType,
    SensitivitySpec,
    ShockSpace,
    ShockUnit,
    active_scenario_to_evidence,
    apply_asset_return_scenario,
    apply_benchmark_active_scenario,
    apply_delta_gamma_scenario,
    apply_factor_scenario,
    apply_group_scenario_decomposition,
    compare_scenario_set,
    create_scenario_shock,
    evaluate_scenario_sensitivity_grid,
    group_scenario_to_evidence,
    render_reverse_stress_profile_artifact,
    render_scenario_active_comparison_artifact,
    render_scenario_asset_contribution_artifact,
    render_scenario_factor_contribution_artifact,
    render_scenario_group_heatmap_artifact,
    render_scenario_pnl_waterfall_artifact,
    render_scenario_sensitivity_curve_artifact,
    render_scenario_set_ranking_artifact,
    reverse_stress_to_evidence,
    scenario_data_integrity_to_evidence,
    scenario_result_to_evidence,
    scenario_sensitivity_to_evidence,
    scenario_set_to_evidence,
    solve_reverse_stress,
    validate_scenario_data_integrity,
)


def run_gate6_showcase() -> dict[str, Any]:
    print("================================================================================")
    print("StART — GATE 6 SHOWCASE: INSTITUTIONAL SCENARIO & REVERSE-STRESS INTELLIGENCE")
    print("================================================================================\n")

    out_dir = Path(__file__).resolve().parent.parent / "start_output" / "gate6_showcase"
    out_dir.mkdir(parents=True, exist_ok=True)

    evidence_records: list[EvidenceRecord] = []
    human_artifacts: list[dict[str, Any]] = []
    semantic_artifacts: list[dict[str, Any]] = []

    # ----------------------------------------------------------------------- #
    # 1. Base Market State & Portfolio Specification
    # ----------------------------------------------------------------------- #
    assets = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM"]
    weights = {"AAPL": 0.25, "MSFT": 0.25, "GOOGL": 0.20, "AMZN": 0.15, "JPM": 0.15}
    benchmark_weights = {"AAPL": 0.20, "MSFT": 0.20, "GOOGL": 0.20, "AMZN": 0.20, "JPM": 0.20}
    portfolio_value = 100_000_000.0  # $100M institutional portfolio

    factors = ["MARKET", "TECH_GROWTH", "VALUE", "RATES"]
    B_matrix = pd.DataFrame(
        [
            [1.10, 0.85, -0.20, -0.10],  # AAPL
            [1.05, 0.90, -0.25, -0.15],  # MSFT
            [1.00, 0.75, -0.10, -0.05],  # GOOGL
            [1.15, 0.80, -0.30, -0.20],  # AMZN
            [0.95, -0.40, 0.80, 0.60],  # JPM
        ],
        index=assets,
        columns=factors,
    )

    sectors = {
        "AAPL": "Information Technology",
        "MSFT": "Information Technology",
        "GOOGL": "Communication Services",
        "AMZN": "Consumer Discretionary",
        "JPM": "Financials",
    }

    # ----------------------------------------------------------------------- #
    # 2. Scenario 1: Heterogeneous Asset Return Shock (Linear Repricing)
    # ----------------------------------------------------------------------- #
    print("--- 1. Evaluating Heterogeneous Asset Return Stress Scenario ---")
    shocks_scen1 = (
        create_scenario_shock(
            "AAPL",
            raw_value=-12.5,
            shock_unit=ShockUnit.RELATIVE_PERCENT,
            shock_space=ShockSpace.ASSET_RETURN,
        ),
        create_scenario_shock(
            "MSFT",
            raw_value=-10.0,
            shock_unit=ShockUnit.RELATIVE_PERCENT,
            shock_space=ShockSpace.ASSET_RETURN,
        ),
        create_scenario_shock(
            "GOOGL",
            raw_value=-8.0,
            shock_unit=ShockUnit.RELATIVE_PERCENT,
            shock_space=ShockSpace.ASSET_RETURN,
        ),
        create_scenario_shock(
            "AMZN",
            raw_value=-15.0,
            shock_unit=ShockUnit.RELATIVE_PERCENT,
            shock_space=ShockSpace.ASSET_RETURN,
        ),
        create_scenario_shock(
            "JPM", raw_value=-5.0, shock_unit=ShockUnit.RELATIVE_PERCENT, shock_space=ShockSpace.ASSET_RETURN
        ),
    )
    spec1 = ScenarioSpec(
        scenario_id="SCEN-TECH-PULLBACK",
        scenario_name="Tech Sector Pullback & Market Correction",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=shocks_scen1,
        repricing_method=RepricingMethod.LINEAR_RETURN,
        horizon=MetricHorizon.PERIODIC,
        as_of_date="2026-08-31",
    )

    integ1 = validate_scenario_data_integrity(spec1, assets=assets)
    ev_integ1 = scenario_data_integrity_to_evidence(integ1)
    evidence_records.append(ev_integ1)

    res1 = apply_asset_return_scenario(
        weights=weights, scenario_spec_or_shocks=spec1, portfolio_value=portfolio_value
    )
    ev_scen1 = scenario_result_to_evidence(res1)
    evidence_records.append(ev_scen1)

    art_waterfall1 = render_scenario_pnl_waterfall_artifact(
        res1, evidence_ids=(ev_scen1.evidence_id,), output_dir=out_dir
    )
    art_asset1 = render_scenario_asset_contribution_artifact(
        res1, evidence_ids=(ev_scen1.evidence_id,), output_dir=out_dir
    )
    human_artifacts.extend([art_waterfall1.to_dict(), art_asset1.to_dict()])
    semantic_artifacts.extend([art_waterfall1.semantic_payload, art_asset1.semantic_payload])

    print(f"Scenario Return: {res1.scenario_return:.4f} (-{res1.scenario_loss:.4f} canonical loss)")
    print(f"Monetary P&L: ${res1.scenario_pnl:,.2f}")
    print(f"Reconciliation Error: {res1.reconciliation_error:.2e}\n")

    # ----------------------------------------------------------------------- #
    # 3. Scenario 2: Factor Shock Decomposition
    # ----------------------------------------------------------------------- #
    print("--- 2. Evaluating Factor Shock Scenario & Specific Risk ---")
    factor_shocks = (
        create_scenario_shock(
            "MARKET",
            raw_value=-0.060,
            shock_unit=ShockUnit.RETURN_DECIMAL,
            shock_space=ShockSpace.FACTOR_RETURN,
        ),
        create_scenario_shock(
            "TECH_GROWTH",
            raw_value=-0.080,
            shock_unit=ShockUnit.RETURN_DECIMAL,
            shock_space=ShockSpace.FACTOR_RETURN,
        ),
        create_scenario_shock(
            "VALUE",
            raw_value=+0.020,
            shock_unit=ShockUnit.RETURN_DECIMAL,
            shock_space=ShockSpace.FACTOR_RETURN,
        ),
        create_scenario_shock(
            "RATES", raw_value=+75.0, shock_unit=ShockUnit.BASIS_POINTS, shock_space=ShockSpace.FACTOR_RETURN
        ),
    )
    spec2 = ScenarioSpec(
        scenario_id="SCEN-RATE-SURGE-FACTOR",
        scenario_name="Rates Surge & Growth Factor Compression",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=factor_shocks,
        repricing_method=RepricingMethod.FACTOR_LINEAR,
        horizon=MetricHorizon.PERIODIC,
        specific_shock_policy="EXPLICIT_ZERO",
    )

    res2 = apply_factor_scenario(
        weights=weights,
        exposures=B_matrix,
        scenario_spec_or_shocks=spec2,
        specific_shock_policy="EXPLICIT_ZERO",
        portfolio_value=portfolio_value,
    )
    ev_scen2 = scenario_result_to_evidence(res2)
    evidence_records.append(ev_scen2)

    art_factor2 = render_scenario_factor_contribution_artifact(
        res2, evidence_ids=(ev_scen2.evidence_id,), output_dir=out_dir
    )
    human_artifacts.append(art_factor2.to_dict())
    semantic_artifacts.append(art_factor2.semantic_payload)

    print(f"Factor Stress Return: {res2.scenario_return:.4f}")
    print(f"Factor Contributions: {res2.factor_contributions}")
    print(f"Factor Reconciliation Error: {res2.reconciliation_error:.2e}\n")

    # ----------------------------------------------------------------------- #
    # 4. Scenario 3: Delta vs Delta-Gamma Sensitivity Repricing (Proof-Carrying)
    # ----------------------------------------------------------------------- #
    print("--- 3. Evaluating Delta vs Delta-Gamma Nonlinear Repricing ---")
    rf_sens = {
        "EQUITY_INDEX": SensitivitySpec("EQUITY_INDEX", delta=85_000_000.0, gamma=-120_000_000.0),
        "RATE_10Y": SensitivitySpec("RATE_10Y", delta=-450_000_000.0, gamma=3_500_000_000.0),
        "VOLATILITY_VIX": SensitivitySpec("VOLATILITY_VIX", delta=-15_000_000.0, gamma=-25_000_000.0),
    }
    rf_shocks = (
        create_scenario_shock(
            "EQUITY_INDEX",
            raw_value=-15.0,
            shock_unit=ShockUnit.RELATIVE_PERCENT,
            shock_space=ShockSpace.PRICE,
        ),
        create_scenario_shock(
            "RATE_10Y", raw_value=+120.0, shock_unit=ShockUnit.BASIS_POINTS, shock_space=ShockSpace.YIELD
        ),
        create_scenario_shock(
            "VOLATILITY_VIX",
            raw_value=+8.0,
            shock_unit=ShockUnit.VOLATILITY_POINTS,
            shock_space=ShockSpace.VOLATILITY,
        ),
    )
    spec_dg = ScenarioSpec(
        scenario_id="SCEN-EQUITY-VOL-SPIKE",
        scenario_name="Equity Drawdown & Volatility Spike",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=rf_shocks,
        repricing_method=RepricingMethod.DELTA_GAMMA,
    )

    # Full symmetric Hessian matrix
    gamma_mat = np.array(
        [
            [-120_000_000.0, -15_000_000.0, -5_000_000.0],
            [-15_000_000.0, 3_500_000_000.0, 0.0],
            [-5_000_000.0, 0.0, -25_000_000.0],
        ]
    )

    res_dg = apply_delta_gamma_scenario(
        sensitivities=rf_sens,
        scenario_spec_or_shocks=spec_dg,
        gamma_matrix=gamma_mat,
        portfolio_value=portfolio_value,
        method=RepricingMethod.DELTA_GAMMA,
    )
    ev_dg = scenario_result_to_evidence(res_dg)
    evidence_records.append(ev_dg)

    # Linear delta-only baseline for comparison
    res_delta_only = apply_delta_gamma_scenario(
        sensitivities=rf_sens,
        scenario_spec_or_shocks=spec_dg,
        portfolio_value=portfolio_value,
        method=RepricingMethod.DELTA,
    )
    ev_delta = scenario_result_to_evidence(res_delta_only)
    evidence_records.append(ev_delta)

    delta_gamma_gap = abs(res_dg.scenario_pnl - res_delta_only.scenario_pnl)
    print(f"Delta-Only Approx P&L: ${res_delta_only.scenario_pnl:,.2f}")
    print(f"Delta-Gamma Approx P&L: ${res_dg.scenario_pnl:,.2f}")
    print(f"Nonlinear Gamma Impact Gap: ${delta_gamma_gap:,.2f}\n")

    # ----------------------------------------------------------------------- #
    # 5. Benchmark & Active Stress Decomposition
    # ----------------------------------------------------------------------- #
    print("--- 4. Evaluating Portfolio vs Benchmark Active Stress ---")
    act_res = apply_benchmark_active_scenario(
        weights=weights,
        benchmark_weights=benchmark_weights,
        scenario_spec_or_shocks=spec1,
        exposures=B_matrix,
    )
    ev_act = active_scenario_to_evidence(act_res)
    evidence_records.append(ev_act)

    art_act = render_scenario_active_comparison_artifact(
        act_res, evidence_ids=(ev_act.evidence_id,), output_dir=out_dir
    )
    human_artifacts.append(art_act.to_dict())
    semantic_artifacts.append(art_act.semantic_payload)

    print(f"Portfolio Return: {act_res.portfolio_return:.4f}")
    print(f"Benchmark Return: {act_res.benchmark_return:.4f}")
    print(f"Active Return: {act_res.active_return:.4f}")
    print(f"Active Reconciliation Error: {act_res.reconciliation_error:.2e}\n")

    # ----------------------------------------------------------------------- #
    # 6. Group / Sector Stress Heatmap
    # ----------------------------------------------------------------------- #
    print("--- 5. Evaluating Sector / Group Stress Decomposition ---")
    group_decomp = apply_group_scenario_decomposition(
        asset_contributions=res1.asset_contributions,
        group_mapping=sectors,
        partition_contract=PartitionContract.EXHAUSTIVE_PARTITION,
    )
    ev_grp = group_scenario_to_evidence(
        "SCEN-TECH-PULLBACK", group_decomp, partition_contract=PartitionContract.EXHAUSTIVE_PARTITION.value
    )
    evidence_records.append(ev_grp)

    art_grp = render_scenario_group_heatmap_artifact(
        "SCEN-TECH-PULLBACK",
        group_decomp,
        partition_contract=PartitionContract.EXHAUSTIVE_PARTITION.value,
        evidence_ids=(ev_grp.evidence_id,),
        output_dir=out_dir,
    )
    human_artifacts.append(art_grp.to_dict())
    semantic_artifacts.append(art_grp.semantic_payload)
    print(f"Group Stress Contributions: {group_decomp}\n")

    # ----------------------------------------------------------------------- #
    # 7. Multi-Scenario Comparative Loss Ranking
    # ----------------------------------------------------------------------- #
    print("--- 6. Evaluating Multi-Scenario Set Loss Rankings ---")
    scen_set_res = compare_scenario_set([res1, res2, res_dg], ranking_metric="scenario_loss")
    ev_set = scenario_set_to_evidence(scen_set_res)
    evidence_records.append(ev_set)

    art_rank = render_scenario_set_ranking_artifact(
        scen_set_res, evidence_ids=(ev_set.evidence_id,), output_dir=out_dir
    )
    human_artifacts.append(art_rank.to_dict())
    semantic_artifacts.append(art_rank.semantic_payload)

    print(f"Scenarios Ranked by Loss: {scen_set_res.loss_rankings}")
    print(
        f"Worst Scenario: '{scen_set_res.worst_scenario_id}' (Loss = {scen_set_res.worst_scenario_loss:.4f})"
    )
    print(
        f"Best Scenario: '{scen_set_res.best_scenario_id}' (Loss = {scen_set_res.best_scenario_loss:.4f})\n"
    )

    # ----------------------------------------------------------------------- #
    # 8. Deterministic Risk Factor Sensitivity Sweep Grid
    # ----------------------------------------------------------------------- #
    print("--- 7. Evaluating Risk Factor Sensitivity Sweep Curve ---")
    sens_sweep = evaluate_scenario_sensitivity_grid(
        base_spec=spec1,
        risk_factor_id="AAPL",
        shock_multipliers=[0.0, 0.5, 1.0, 1.5, 2.0],
        weights=weights,
        portfolio_value=portfolio_value,
    )
    ev_sens = scenario_sensitivity_to_evidence(sens_sweep)
    evidence_records.append(ev_sens)

    art_sens = render_scenario_sensitivity_curve_artifact(
        sens_sweep, evidence_ids=(ev_sens.evidence_id,), output_dir=out_dir
    )
    human_artifacts.append(art_sens.to_dict())
    semantic_artifacts.append(art_sens.semantic_payload)
    print(
        f"Sensitivity Sweep for AAPL: base loss={sens_sweep.base_loss:.4f}, max loss={sens_sweep.max_loss:.4f}\n"
    )

    # ----------------------------------------------------------------------- #
    # 9. Minimum Shock Reverse Stress Testing
    # ----------------------------------------------------------------------- #
    print("--- 8. Evaluating Minimum Shock Reverse Stress Optimization ---")
    rev_spec = ReverseStressSpec(
        target_loss=0.080,  # 8% portfolio target loss
        shock_space=ShockSpace.FACTOR_RETURN,
        distance_norm=ReverseStressNorm.L2,
    )
    portfolio_factor_exposures = B_matrix.T.to_numpy() @ np.array([weights[a] for a in assets])
    rev_res = solve_reverse_stress(
        spec=rev_spec,
        sensitivities_or_weights=portfolio_factor_exposures,
        factors=factors,
    )
    ev_rev = reverse_stress_to_evidence(rev_res)
    evidence_records.append(ev_rev)

    art_rev = render_reverse_stress_profile_artifact(
        rev_res, evidence_ids=(ev_rev.evidence_id,), output_dir=out_dir
    )
    human_artifacts.append(art_rev.to_dict())
    semantic_artifacts.append(art_rev.semantic_payload)

    print(f"Reverse Stress Target Loss: {rev_res.target_loss:.4f}")
    print(f"Achieved Loss: {rev_res.achieved_loss:.4f} (Gap = {rev_res.loss_gap:.2e})")
    print(f"Minimum L2 Distance: {rev_res.distance:.4f}")
    print(f"Minimum Factor Shock Vector: {rev_res.shock_vector}")
    print(f"Solver Status: {rev_res.solver_status} (is_closed_form={rev_res.is_closed_form})\n")

    # ----------------------------------------------------------------------- #
    # 10. Agentic Orchestration & Proof-Carrying Governance Adjudication
    # ----------------------------------------------------------------------- #
    factor_cov = pd.DataFrame(np.diag([0.04, 0.06, 0.03, 0.02]), index=factors, columns=factors)
    specific_var = pd.Series([0.02, 0.02, 0.015, 0.025, 0.018], index=assets)
    synth_returns = pd.DataFrame(
        np.random.RandomState(42).normal(0.0005, 0.015, size=(250, len(assets))), columns=assets
    )

    context = {
        "evidence_records": evidence_records,
        "weights": weights,
        "portfolio_weights": weights,
        "benchmark_weights": benchmark_weights,
        "portfolio_value": portfolio_value,
        "exposures": B_matrix,
        "factor_exposures": B_matrix,
        "factor_cov": factor_cov,
        "factor_covariance": factor_cov,
        "specific_var": specific_var,
        "specific_variances": specific_var,
        "returns": synth_returns,
        "returns_df": synth_returns,
        "sensitivities": rf_sens,
        "gamma_matrix": gamma_mat,
        "scenario_spec": spec_dg,
        "factor_scenario_spec": spec2,
        "scenario_specs": {
            "SCEN-TECH-PULLBACK": spec1,
            "SCEN-RATE-SURGE-FACTOR": spec2,
            "SCEN-EQUITY-VOL-SPIKE": spec_dg,
        },
        "reverse_stress_spec": rev_spec,
        "sensitivities_or_weights": portfolio_factor_exposures,
    }

    director = MarketReviewDirectorAgent()
    director_out = director.execute(context)

    # ----------------------------------------------------------------------- #
    # 11. Write Manifest and Governance Summary
    # ----------------------------------------------------------------------- #
    manifest_data = {
        "showcase_gate": "GATE_6",
        "title": "Institutional Scenario, Stress & Reverse-Stress Intelligence Showcase",
        "portfolio_value": portfolio_value,
        "n_evidence_records": len(evidence_records),
        "evidence_ids": [r.evidence_id for r in evidence_records],
        "human_review_artifacts": human_artifacts,
        "semantic_artifacts": semantic_artifacts,
        "governance_verdict": director_out.get("governance_verdict"),
        "governance_signoff": director_out.get("governance_signoff"),
    }

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, default=str)

    summary_path = out_dir / "gate6_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(director_out, f, indent=2, default=str)

    print("================================================================================")
    print(f"Showcase completed successfully. Emitted {len(human_artifacts)} artifacts to:")
    print(f"{out_dir}")
    print(f"Governance Verdict: {director_out.get('governance_verdict')}")
    print("================================================================================\n")

    return director_out


if __name__ == "__main__":
    run_gate6_showcase()
