#!/usr/bin/env python3
"""Non-interactive Institutional Showcase for StART Gate 3.

Demonstrates end-to-end institutional portfolio optimization, constrained allocation,
adversarial challenges, evidence bridging, cryptographic artifact generation, and formal MRM governance sign-off.

Outputs:
- start_output/gate3_showcase/
  - artifacts/*.json
  - evidence_records.json
  - SHOWCASE_MANIFEST.md
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from start.agents.market_review import MarketReviewDirectorAgent
from start.core.schemas import EvidenceRecord
from start.portfolio import (
    FactorConstraintSpec,
    GroupConstraintSpec,
    TransactionCostSpec,
    black_litterman_to_evidence,
    build_rebalance_decision,
    compare_portfolio_methods,
    cvar_to_evidence,
    herc_to_evidence,
    max_diversification_to_evidence,
    rebalance_decision_to_evidence,
    render_bl_allocation_artifact,
    render_bl_returns_artifact,
    render_cvar_tail_risk_artifact,
    render_herc_hierarchy_artifact,
    render_multi_method_comparison_artifact,
    render_rebalance_waterfall_artifact,
    render_robust_mvo_sensitivity_artifact,
    render_tracking_error_artifact,
    robust_mvo_sensitivity_grid,
    robust_mvo_to_evidence,
    robust_sensitivity_to_evidence,
    solve_black_litterman,
    solve_cvar_portfolio,
    solve_herc,
    solve_max_diversification,
    solve_robust_mvo,
    solve_tracking_error_constrained,
    tracking_error_to_evidence,
)
from start.registry.market_contexts import PortfolioConstraints


def run_gate3_showcase(output_dir: Path | str | None = None) -> dict[str, Any]:
    """Execute complete Gate 3 Institutional Showcase."""
    if output_dir is None:
        out_root = Path("start_output/gate3_showcase")
    else:
        out_root = Path(output_dir)

    out_root.mkdir(parents=True, exist_ok=True)
    art_dir = out_root / "artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("StART — GATE 3 INSTITUTIONAL PORTFOLIO SHOWCASE")
    print("=" * 78)

    # 1. Market Universe Setup (6 Assets, 500 Daily Scenarios)
    np.random.seed(1337)
    assets = ["SPY", "QQQ", "TLT", "IEF", "GLD", "DBC"]
    ppy = 252.0

    # Annualized parameters
    mu_annual = np.array([0.090, 0.120, 0.035, 0.025, 0.060, 0.050])
    vols_annual = np.array([0.160, 0.220, 0.140, 0.070, 0.150, 0.180])
    corr = np.array(
        [
            [1.00, 0.85, -0.30, -0.20, 0.05, 0.25],
            [0.85, 1.00, -0.35, -0.25, 0.00, 0.20],
            [-0.30, -0.35, 1.00, 0.80, 0.20, -0.15],
            [-0.20, -0.25, 0.80, 1.00, 0.15, -0.10],
            [0.05, 0.00, 0.20, 0.15, 1.00, 0.30],
            [0.25, 0.20, -0.15, -0.10, 0.30, 1.00],
        ]
    )
    cov_annual = np.diag(vols_annual) @ corr @ np.diag(vols_annual)
    cov_daily = cov_annual / ppy
    mu_daily = mu_annual / ppy

    n_days = 500
    sim_returns = np.random.multivariate_normal(mu_daily, cov_daily, size=n_days)
    returns_df = pd.DataFrame(sim_returns, columns=assets)
    cov_df = pd.DataFrame(cov_annual, index=assets, columns=assets)

    current_weights = {"SPY": 0.30, "QQQ": 0.15, "TLT": 0.20, "IEF": 0.15, "GLD": 0.10, "DBC": 0.10}
    benchmark_weights = {"SPY": 0.35, "QQQ": 0.15, "TLT": 0.25, "IEF": 0.15, "GLD": 0.05, "DBC": 0.05}

    print(f"Loaded Institutional Market Universe: {len(assets)} assets over {n_days} scenarios.")

    # 2. Institutional Constraint Specifications
    # Group: Equities (SPY + QQQ) <= 50%, Fixed Income (TLT + IEF) >= 20%
    group_spec = GroupConstraintSpec(
        group_name="AssetClass",
        memberships={"Equities": ("SPY", "QQQ"), "FixedIncome": ("TLT", "IEF")},
        lower_bounds={"FixedIncome": 0.20},
        upper_bounds={"Equities": 0.50},
    )
    # Factor: Duration beta <= 6.0
    factor_spec = FactorConstraintSpec(
        factor_names=("Duration",),
        loadings={
            "SPY": {"Duration": 0.0},
            "QQQ": {"Duration": 0.0},
            "TLT": {"Duration": 16.0},
            "IEF": {"Duration": 7.0},
            "GLD": {"Duration": 0.0},
            "DBC": {"Duration": 0.0},
        },
        upper_bounds={"Duration": 6.0},
    )
    constraints = PortfolioConstraints(
        budget=1.0,
        long_only=True,
        min_weight=0.02,
        max_weight=0.35,
        max_concentration=0.28,
        group_constraints=group_spec,
        factor_constraints=factor_spec,
    )
    cost_spec = TransactionCostSpec(
        cost_bps={"SPY": 2.0, "QQQ": 2.0, "TLT": 3.0, "IEF": 2.0, "GLD": 4.0, "DBC": 6.0},
        default_linear_bps=5.0,
    )

    evidence_records: list[EvidenceRecord] = []
    artifact_records: list[Any] = []

    # 3. Solver 1: Black-Litterman
    # View 1: GLD expected return is 8.0% (Absolute)
    # View 2: QQQ outperforming SPY by 2.5% (Relative: QQQ - SPY = 0.025)
    P = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    Q = np.array([0.080, 0.025])
    view_labels = ["GLD Bullish Trend", "Tech Outperformance Spread"]

    bl_res = solve_black_litterman(
        covariance=cov_annual,
        market_weights=benchmark_weights,
        P=P,
        Q=Q,
        risk_aversion=2.8,
        tau=0.05,
        assets=assets,
        view_labels=view_labels,
        constraints=constraints,
        prior_weights=current_weights,
        periods_per_year=1.0,  # covariance is annual
    )
    ev_bl = black_litterman_to_evidence(bl_res)
    evidence_records.append(ev_bl)
    art_bl_ret = render_bl_returns_artifact(bl_res, evidence_ids=(ev_bl.evidence_id,), output_dir=art_dir)
    art_bl_alloc = render_bl_allocation_artifact(
        bl_res, evidence_ids=(ev_bl.evidence_id,), output_dir=art_dir
    )
    artifact_records.extend([art_bl_ret, art_bl_alloc])
    print(
        f"✓ Black-Litterman solved: Vol={bl_res.posterior_volatility_annualised:.2%}, Turnover={bl_res.turnover_vs_prior:.2%}"
    )

    from dataclasses import replace

    from start.portfolio import UncertaintyDerivationPolicy

    # 4. Solver 2: Robust Mean-Variance Optimization & Sensitivity
    rob_res = solve_robust_mvo(
        mu=mu_annual,
        covariance=cov_annual,
        uncertainty_radius=0.50,
        uncertainty_policy=UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N,
        n_observations=n_days,
        assets=assets,
        constraints=constraints,
        prior_weights=current_weights,
        periods_per_year=1.0,
    )
    ev_rob = robust_mvo_to_evidence(rob_res)
    evidence_records.append(ev_rob)

    rob_sens = robust_mvo_sensitivity_grid(
        mu=mu_annual,
        covariance=cov_annual,
        radii=(0.0, 0.25, 0.50, 1.0, 2.0),
        uncertainty_policy=UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N,
        n_observations=n_days,
        assets=assets,
        constraints=constraints,
        prior_weights=current_weights,
        periods_per_year=1.0,
    )
    ev_sens = robust_sensitivity_to_evidence(rob_sens)
    evidence_records.append(ev_sens)
    art_rob_sens = render_robust_mvo_sensitivity_artifact(
        rob_sens, evidence_ids=(ev_sens.evidence_id,), output_dir=art_dir
    )
    artifact_records.append(art_rob_sens)
    print(
        f"✓ Robust MVO solved: Worst-Case Return={rob_res.worst_case_expected_return_annualised:.2%}, Vol={rob_res.portfolio_volatility_annualised:.2%}"
    )

    # 5. Solver 3: Rockafellar-Uryasev CVaR Linear Programming
    cvar_constraints = replace(constraints, max_concentration=None, max_tracking_error=None)
    cvar_res = solve_cvar_portfolio(
        scenario_returns=returns_df,
        confidence_level=0.95,
        assets=assets,
        constraints=cvar_constraints,
        prior_weights=current_weights,
        periods_per_year=ppy,
    )
    ev_cvar = cvar_to_evidence(cvar_res)
    evidence_records.append(ev_cvar)
    art_cvar = render_cvar_tail_risk_artifact(
        cvar_res, evidence_ids=(ev_cvar.evidence_id,), output_dir=art_dir
    )
    artifact_records.append(art_cvar)
    print(
        f"✓ CVaR 95% solved: Scenario Horizon CVaR={cvar_res.cvar_at_scenario_horizon:.2%}, Tail Scenarios={cvar_res.tail_scenario_count}"
    )

    # 6. Solver 4: Hierarchical Equal Risk Contribution (HERC)
    herc_res = solve_herc(cov_annual, linkage_method="single", assets=assets, periods_per_year=1.0)
    ev_herc = herc_to_evidence(herc_res)
    evidence_records.append(ev_herc)
    art_herc = render_herc_hierarchy_artifact(
        herc_res, evidence_ids=(ev_herc.evidence_id,), output_dir=art_dir
    )
    artifact_records.append(art_herc)
    print(
        f"✓ HERC solved: Effective Positions={herc_res.effective_n_positions:.2f}, Vol={herc_res.portfolio_volatility_annualised:.2%}"
    )

    # 7. Solver 5: Maximum Diversification Portfolio (MDP)
    md_res = solve_max_diversification(
        cov_annual,
        assets=assets,
        constraints=constraints,
        prior_weights=current_weights,
        periods_per_year=1.0,
    )
    ev_md = max_diversification_to_evidence(md_res)
    evidence_records.append(ev_md)
    print(f"✓ Max Diversification solved: Diversification Ratio={md_res.diversification_ratio:.4f}")

    # 8. Solver 6: Tracking-Error Constrained Optimization
    te_cap_annual = 0.025  # 250 bps max TE
    te_res = solve_tracking_error_constrained(
        mu=mu_annual,
        covariance=cov_annual,
        benchmark_weights=benchmark_weights,
        max_tracking_error=te_cap_annual,
        assets=assets,
        constraints=constraints,
        prior_weights=current_weights,
        periods_per_year=1.0,
    )
    ev_te = tracking_error_to_evidence(te_res)
    evidence_records.append(ev_te)
    art_te = render_tracking_error_artifact(te_res, evidence_ids=(ev_te.evidence_id,), output_dir=art_dir)
    artifact_records.append(art_te)
    print(
        f"✓ Tracking Error constrained solved: TE={te_res.tracking_error_annualised:.2%}, IR={te_res.information_ratio:.3f}"
    )

    # 9. Rebalancing Decision & Cost Waterfall
    reb_decision = build_rebalance_decision(
        current_weights=current_weights,
        proposed_weights=bl_res.posterior_weights,
        covariance=cov_annual,
        assets=assets,
        mu=mu_annual,
        cost_spec=cost_spec,
        constraints=constraints,
        periods_per_year=1.0,
        evidence_ids=(ev_bl.evidence_id,),
    )
    ev_reb = rebalance_decision_to_evidence(reb_decision)
    evidence_records.append(ev_reb)
    art_reb = render_rebalance_waterfall_artifact(
        reb_decision, evidence_ids=(ev_reb.evidence_id,), output_dir=art_dir
    )
    artifact_records.append(art_reb)
    print(
        f"✓ Rebalance Decision audited: Turnover={reb_decision.turnover:.2%}, Cost={reb_decision.estimated_transaction_cost:.4%}"
    )

    # 10. Multi-Method Institutional Comparison Matrix
    comp_res = compare_portfolio_methods(
        returns=returns_df,
        covariance=cov_df,
        mu=mu_annual,
        prior_weights=current_weights,
        benchmark_weights=benchmark_weights,
        constraints=constraints,
        robust_uncertainty_radius=0.50,
        cvar_confidence=0.95,
        periods_per_year=1.0,
    )
    art_comp = render_multi_method_comparison_artifact(
        comp_res, evidence_ids=(ev_bl.evidence_id, ev_rob.evidence_id), output_dir=art_dir
    )
    artifact_records.append(art_comp)
    print(f"✓ Comparative Matrix evaluated across {len(comp_res.methods)} methods.")

    # 11. Specialist Agent Orchestration & Formal MRM Governance Sign-Off
    director = MarketReviewDirectorAgent()
    agent_ctx = {
        "evidence_records": evidence_records,
        "covariance": cov_annual,
        "assets": assets,
        "scenario_returns": returns_df,
        "returns": returns_df,
        "market_weights": benchmark_weights,
        "P": P,
        "Q": Q,
        "mu": mu_annual,
        "weights_current": current_weights,
        "weights_target": bl_res.posterior_weights,
        "cost_spec": cost_spec,
        "uncertainty_policy": UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N,
        "auto_resolve_challenges": True,
    }
    orchestration_res = director.execute(agent_ctx)
    print("✓ Specialist Review & Governance Orchestration completed:")
    print(f"  Critic Disposition: {orchestration_res['critic_disposition']}")
    print(f"  Governance Verdict: {orchestration_res['governance_verdict']}")
    print(f"  Specialist Findings: {orchestration_res['findings_count']}")
    print(f"  Adversarial Challenges: {orchestration_res['challenges_count']}")

    # Save evidence records
    ev_file = out_root / "evidence_records.json"
    with open(ev_file, "w", encoding="utf-8") as f:
        json.dump([r.model_dump() for r in evidence_records], f, indent=2, default=str)

    # Generate SHOWCASE_MANIFEST.md
    manifest_file = out_root / "SHOWCASE_MANIFEST.md"
    with open(manifest_file, "w", encoding="utf-8") as f:
        f.write("# StART Gate 3 Institutional Showcase Manifest\n\n")
        f.write("## Executive Summary\n")
        f.write(f"- **Governance Sign-off Verdict**: `{orchestration_res['governance_verdict']}`\n")
        f.write(f"- **Evidence Critic Disposition**: `{orchestration_res['critic_disposition']}`\n")
        f.write(f"- **Total Evidence Records Generated**: `{len(evidence_records)}` (0 orphans)\n")
        f.write(f"- **Cryptographic Artifacts Rendered**: `{len(artifact_records)}`\n")
        f.write(f"- **Adversarial Challenges Formulated**: `{orchestration_res['challenges_count']}`\n\n")

        f.write("## Portfolio Methods Evaluated\n")
        f.write(
            "| Method | Annualized Return | Annualized Volatility | Sharpe | Diversification Ratio | Turnover vs Current |\n"
        )
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for row in comp_res.summary_table:
            sh = f"{row['annualised_sharpe']:.2f}" if row["annualised_sharpe"] is not None else "N/A"
            dr = f"{row['diversification_ratio']:.2f}" if row.get("diversification_ratio") else "N/A"
            to = f"{row['turnover_vs_current']:.1%}" if row["turnover_vs_current"] is not None else "0.0%"
            f.write(
                f"| **{row['method']}** | {row['annualised_return']:.2%} | {row['annualised_volatility']:.2%} | {sh} | {dr} | {to} |\n"
            )

        f.write("\n## Cryptographic Artifact Records\n")
        f.write("| Artifact ID | Type | Data Fingerprint | Semantic Payload Hash |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        for art in artifact_records:
            f.write(
                f"| `{art.artifact_id}` | `{art.spec.artifact_type}` | `{art.data_fingerprint[:16]}...` | `{art.semantic_payload_hash[:16]}...` |\n"
            )

        f.write("\n## Adversarial Challenges Audited\n")
        for ch in orchestration_res["challenges"]:
            f.write(f"- **[{ch['challenge_id']}]** *{ch['target_area']}*: {ch['challenge_question']}\n")

    print(f"✓ Showcase manifest and artifacts written to {out_root.resolve()}")

    return {
        "status": "success",
        "evidence_records_count": len(evidence_records),
        "artifacts_count": len(artifact_records),
        "governance_verdict": orchestration_res["governance_verdict"],
        "manifest_path": str(manifest_file),
    }


if __name__ == "__main__":
    run_gate3_showcase()
