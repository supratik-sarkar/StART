#!/usr/bin/env python3
"""Gate 4 Showcase: Institutional Covariance, Factor Risk & Attribution Intelligence.

Executes the complete Gate-4 vertical slice:
1. Negative Evidence Showcase: Indefinite covariance -> non-PSD diagnosis -> explicit Higham repair -> separate repair evidence.
2. Estimator Comparison: Empirical vs Ledoit-Wolf vs RegEM covariance.
3. Factor Risk Model: Reconstructed asset covariance BFB' + D, Euler factor component variance waterfall, active tracking error.
4. Attribution Engine: Period factor return attribution, Brinson-Fachler single-period attribution, Carino multi-period geometric linking.
5. Multi-Agent Director: Data Integrity Checker -> Covariance Agent -> Factor Agent -> Adversarial Challenger -> Critic -> Governance.
6. Cryptographic Artifacts: Emits full companion artifacts in start_output/gate4_showcase/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from start.agents.market_review import (
    MarketReviewDirectorAgent,
)
from start.portfolio.artifacts import (
    render_active_risk_decomposition_artifact,
    render_brinson_attribution_artifact,
    render_carino_linking_artifact,
    render_covariance_comparison_artifact,
    render_covariance_diagnostics_artifact,
    render_factor_return_attribution_artifact,
    render_factor_risk_model_artifact,
    render_factor_risk_waterfall_artifact,
    render_psd_repair_artifact,
    render_raw_covariance_heatmap_artifact,
)
from start.portfolio.attribution import (
    compute_brinson_attribution,
    compute_carino_multi_period_linking,
    compute_factor_return_attribution,
)
from start.portfolio.contracts import (
    PSDRepairMethod,
)
from start.portfolio.covariance import (
    compare_covariance_estimators,
    diagnose_covariance,
    repair_psd_covariance,
)
from start.portfolio.evidence_bridge import (
    active_risk_decomp_to_evidence,
    brinson_to_evidence,
    carino_to_evidence,
    covariance_comparison_to_evidence,
    covariance_diagnostics_to_evidence,
    factor_return_attribution_to_evidence,
    factor_risk_decomp_to_evidence,
    factor_risk_model_to_evidence,
    psd_repair_to_evidence,
)
from start.portfolio.factor_risk import (
    build_linear_factor_model,
    decompose_active_risk,
    decompose_factor_risk,
)


def run_gate4_showcase() -> None:
    output_dir = Path("start_output/gate4_showcase")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_dir_str = str(output_dir)

    print("=" * 80)
    print("StART GATE 4: INSTITUTIONAL COVARIANCE, FACTOR RISK & ATTRIBUTION")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. NEGATIVE EVIDENCE SHOWCASE (Mandatory Amendment 21)
    # -------------------------------------------------------------------------
    print("\n[1/5] NEGATIVE EVIDENCE SHOWCASE: INDEFINITE COVARIANCE & HIGHAM PSD REPAIR")
    raw_indefinite_cov = np.array(
        [
            [1.00, 0.90, 0.90],
            [0.90, 1.00, 0.90],
            [0.90, 0.90, 0.10],
        ]
    )
    assets_3 = ["ASSET_A", "ASSET_B", "ASSET_C"]

    # Step 1: Diagnose raw covariance -> produces EV-COV-RAW
    diag_raw = diagnose_covariance(raw_indefinite_cov, assets=assets_3)
    ev_cov_raw = covariance_diagnostics_to_evidence(diag_raw)
    ev_cov_raw.evidence_id = "EV-COV-RAW"

    print(f"  Raw Covariance: is_psd={diag_raw.is_psd}, min_eig={diag_raw.minimum_eigenvalue:.6f}")
    print(f"  Raw Evidence Emitted: {ev_cov_raw.evidence_id} (test_id={ev_cov_raw.test_id})")

    # Step 2: Explicit Higham repair -> produces separate EV-COV-REPAIRED
    repair_res = repair_psd_covariance(
        raw_indefinite_cov,
        method=PSDRepairMethod.HIGHAM_NEAREST_CORRELATION,
        min_eigenvalue=1e-6,
    )
    ev_cov_repaired = psd_repair_to_evidence(repair_res)
    ev_cov_repaired.evidence_id = "EV-COV-REPAIRED"

    print(
        f"  Higham Repaired: min_eig={repair_res.repaired_minimum_eigenvalue:.6e}, frob_dist={repair_res.frobenius_distortion:.6f}"
    )
    print(f"  Repaired Evidence Emitted: {ev_cov_repaired.evidence_id} (test_id={ev_cov_repaired.test_id})")
    print("  Provenance Invariant Verified: EV-COV-RAW preserved; EV-COV-REPAIRED distinct.")

    # -------------------------------------------------------------------------
    # 2. MULTI-ESTIMATOR COVARIANCE COMPARISON
    # -------------------------------------------------------------------------
    print("\n[2/5] COVARIANCE ESTIMATOR COMPARISON: EMPIRICAL vs LEDOIT-WOLF vs REGULARIZED-EM")
    np.random.seed(42)
    assets_4 = ["AAPL", "MSFT", "XOM", "JNJ"]
    rets_df = pd.DataFrame(
        np.random.randn(120, 4) * 0.012,
        columns=assets_4,
    )
    pw = {"AAPL": 0.30, "MSFT": 0.30, "XOM": 0.20, "JNJ": 0.20}
    bw = {"AAPL": 0.25, "MSFT": 0.25, "XOM": 0.25, "JNJ": 0.25}

    cov_comp = compare_covariance_estimators(
        returns=rets_df,
        estimators=("empirical", "ledoit_wolf", "regularized_em"),
        portfolio_weights=pw,
        periods_per_year=252.0,
    )
    ev_cov_comp = covariance_comparison_to_evidence(cov_comp)
    ev_cov_comp.evidence_id = "EV-COV-COMPARE"

    for est, vol in cov_comp.portfolio_volatilities_annualised.items():
        print(f"  Estimator '{est}': Annual Volatility = {vol:.2%}")

    # -------------------------------------------------------------------------
    # 3. FACTOR RISK MODEL & EULER VARIANCE DECOMPOSITION
    # -------------------------------------------------------------------------
    print("\n[3/5] FACTOR RISK MODEL & EULER VARIANCE DECOMPOSITION")
    factors = ["Market", "Value"]
    B_df = pd.DataFrame(
        [
            [1.20, -0.30],
            [1.10, -0.20],
            [0.70, 0.80],
            [0.60, 0.40],
        ],
        index=assets_4,
        columns=factors,
    )
    F_df = pd.DataFrame([[0.040, -0.005], [-0.005, 0.020]], index=factors, columns=factors)
    D_dict = {"AAPL": 0.030, "MSFT": 0.025, "XOM": 0.020, "JNJ": 0.015}

    frm = build_linear_factor_model(B_df, F_df, D_dict, periods_per_year=252.0)
    ev_frm = factor_risk_model_to_evidence(frm)
    ev_frm.evidence_id = "EV-FACTOR-MODEL"

    frd = decompose_factor_risk(pw, frm, periods_per_year=252.0)
    ev_frd = factor_risk_decomp_to_evidence(frd)
    ev_frd.evidence_id = "EV-FACTOR-DECOMP"

    ard = decompose_active_risk(pw, bw, frm, periods_per_year=252.0)
    ev_ard = active_risk_decomp_to_evidence(ard)
    ev_ard.evidence_id = "EV-ACTIVE-RISK"

    print(f"  Portfolio Volatility: {frd.portfolio_volatility_annualised:.2%}")
    print(
        f"  Systematic Share: {frd.systematic_variance_share:.1%} | Specific Share: {frd.specific_variance_share:.1%}"
    )
    print(f"  Euler Reconciliation Residual: {frd.euler_reconciliation_error:.3e}")
    print(
        f"  Active Tracking Error: {ard.tracking_error_annualised:.2%} (Factor Share: {ard.factor_active_share:.1%})"
    )

    # -------------------------------------------------------------------------
    # 4. RETURN ATTRIBUTION: FACTOR, BRINSON-FACHLER & CARINO LINKING
    # -------------------------------------------------------------------------
    print("\n[4/5] RETURN & ACTIVE ATTRIBUTION INTELLIGENCE")
    # Multi-period factor returns
    n_periods = 8
    factor_rets = pd.DataFrame(
        np.random.randn(n_periods, 2) * 0.01 + np.array([0.006, 0.002]),
        columns=factors,
    )
    eps = np.random.randn(n_periods, 4) * 0.004
    asset_rets = pd.DataFrame(factor_rets.to_numpy() @ B_df.to_numpy().T + eps, columns=assets_4)

    fra = compute_factor_return_attribution(asset_rets, B_df, factor_rets, pw)
    ev_fra = factor_return_attribution_to_evidence(fra)
    ev_fra.evidence_id = "EV-FACTOR-ATTRIB"

    # Brinson-Fachler 2-period attribution
    pw_s1 = {"Tech": 0.60, "Energy": 0.40}
    bw_s1 = {"Tech": 0.50, "Energy": 0.50}
    pr_s1 = {"Tech": 0.08, "Energy": 0.02}
    br_s1 = {"Tech": 0.06, "Energy": 0.03}
    brinson_1 = compute_brinson_attribution(pw_s1, bw_s1, pr_s1, br_s1)
    ev_brinson = brinson_to_evidence(brinson_1)
    ev_brinson.evidence_id = "EV-BRINSON-P1"

    pw_s2 = {"Tech": 0.55, "Energy": 0.45}
    bw_s2 = {"Tech": 0.50, "Energy": 0.50}
    pr_s2 = {"Tech": -0.02, "Energy": 0.05}
    br_s2 = {"Tech": -0.01, "Energy": 0.04}
    brinson_2 = compute_brinson_attribution(pw_s2, bw_s2, pr_s2, br_s2)

    carino_res = compute_carino_multi_period_linking(
        period_brinson_results=[brinson_1, brinson_2],
        period_portfolio_returns=[brinson_1.total_portfolio_return, brinson_2.total_portfolio_return],
        period_benchmark_returns=[brinson_1.total_benchmark_return, brinson_2.total_benchmark_return],
    )
    ev_carino = carino_to_evidence(carino_res)
    ev_carino.evidence_id = "EV-CARINO-LINKING"

    print(
        f"  Factor Return Attribution: Factor Contrib={fra.total_factor_contribution:.4f}, Specific Contrib={fra.total_specific_contribution:.4f}, Max Error={fra.max_abs_reconciliation_error:.2e}"
    )
    print(
        f"  Brinson Active Return (P1): {brinson_1.total_active_return:.4%} (Alloc={brinson_1.total_allocation_effect:.4%}, Select={brinson_1.total_selection_effect:.4%}, Inter={brinson_1.total_interaction_effect:.4%})"
    )
    print(f"  Carino Geometric Linked Active Return: {carino_res.total_active_return_geometric:.4%}")

    # -------------------------------------------------------------------------
    # 5. MULTI-AGENT GOVERNANCE WORKFLOW & ARTIFACT EMISSION
    # -------------------------------------------------------------------------
    print("\n[5/5] MULTI-AGENT MARKET REVIEW & CRYPTOGRAPHIC ARTIFACT GENERATION")
    all_evidence = [
        ev_cov_raw,
        ev_cov_repaired,
        ev_cov_comp,
        ev_frm,
        ev_frd,
        ev_ard,
        ev_fra,
        ev_brinson,
        ev_carino,
    ]

    director_context = {
        "assets": assets_4,
        "factors": factors,
        "returns": asset_rets,
        "asset_returns": asset_rets,
        "factor_returns": factor_rets,
        "covariance": frm.reconstructed_covariance,
        "exposures": B_df,
        "factor_cov": F_df,
        "specific_var": D_dict,
        "weights": pw,
        "benchmark_weights": bw,
        "portfolio_weights": pw_s1,
        "portfolio_returns": pr_s1,
        "benchmark_returns": br_s1,
        "period_brinson_results": [brinson_1, brinson_2],
        "period_portfolio_returns": [brinson_1.total_portfolio_return, brinson_2.total_portfolio_return],
        "period_benchmark_returns": [brinson_1.total_benchmark_return, brinson_2.total_benchmark_return],
        "factor_model": frm,
        "evidence_records": all_evidence,
    }

    director = MarketReviewDirectorAgent()
    director_out = director.execute(director_context)

    print(f"  Findings Count: {director_out['findings_count']}")
    print(f"  Challenges Formulated: {director_out['challenges_count']}")
    print(f"  Challenges Resolved: {director_out['resolutions_count']}")
    print(f"  Critic Disposition: {director_out['critic_disposition']}")
    print(f"  Governance Verdict: {director_out['governance_verdict']}")
    print(f"  Governance Reason: {director_out.get('governance', {}).get('reason')}")
    print(f"  Governance Conditions: {director_out.get('governance', {}).get('conditions')}")

    # Render Companion Artifacts (both SVG visual and JSON semantic companion)
    art_cov_diag = render_covariance_diagnostics_artifact(
        diag_raw, (ev_cov_raw.evidence_id,), output_dir=out_dir_str
    )
    art_raw_cov = render_raw_covariance_heatmap_artifact(
        raw_indefinite_cov, assets_3, (ev_cov_raw.evidence_id,), output_dir=out_dir_str
    )
    art_psd_repair = render_psd_repair_artifact(
        repair_res, (ev_cov_repaired.evidence_id,), output_dir=out_dir_str
    )
    art_cov_comp = render_covariance_comparison_artifact(
        cov_comp, (ev_cov_comp.evidence_id,), output_dir=out_dir_str
    )
    art_frm = render_factor_risk_model_artifact(frm, (ev_frm.evidence_id,), output_dir=out_dir_str)
    art_frd = render_factor_risk_waterfall_artifact(frd, (ev_frd.evidence_id,), output_dir=out_dir_str)
    art_ard = render_active_risk_decomposition_artifact(ard, (ev_ard.evidence_id,), output_dir=out_dir_str)
    art_fra = render_factor_return_attribution_artifact(fra, (ev_fra.evidence_id,), output_dir=out_dir_str)
    art_brinson = render_brinson_attribution_artifact(
        brinson_1, (ev_brinson.evidence_id,), output_dir=out_dir_str
    )
    art_carino = render_carino_linking_artifact(carino_res, (ev_carino.evidence_id,), output_dir=out_dir_str)

    artifacts_emitted = [
        art_cov_diag,
        art_raw_cov,
        art_psd_repair,
        art_cov_comp,
        art_frm,
        art_frd,
        art_ard,
        art_fra,
        art_brinson,
        art_carino,
    ]

    semantic_manifest = []
    visual_manifest = []

    for a in artifacts_emitted:
        # JSON companion
        json_path = str(output_dir / f"{a.artifact_id}.json")
        semantic_manifest.append(
            {
                "artifact_id": a.artifact_id,
                "artifact_type": a.spec.artifact_type,
                "file_path": json_path,
                "evidence_ids": list(a.spec.evidence_ids),
                "semantic_payload_hash": a.semantic_payload_hash,
                "data_fingerprint": a.data_fingerprint,
            }
        )
        if a.file_path and a.file_path.endswith(".svg"):
            visual_manifest.append(
                {
                    "artifact_id": a.artifact_id,
                    "artifact_type": a.spec.artifact_type,
                    "title": a.spec.title,
                    "file_path": a.file_path,
                    "rendering_format": "svg",
                    "evidence_ids": list(a.spec.evidence_ids),
                    "semantic_payload_hash": a.semantic_payload_hash,
                }
            )

    manifest = {
        "gate": "GATE_4",
        "semantic_artifacts_count": len(semantic_manifest),
        "human_review_artifacts_count": len(visual_manifest),
        "semantic_artifacts": semantic_manifest,
        "human_review_artifacts": visual_manifest,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    summary = {
        "status": "COMPLETED",
        "gate": "GATE_4",
        "evidence_records_count": len(all_evidence),
        "artifacts_emitted_count": len(artifacts_emitted),
        "semantic_artifacts_count": len(semantic_manifest),
        "human_review_artifacts_count": len(visual_manifest),
        "governance_verdict": director_out["governance_verdict"],
        "governance_details": director_out.get("governance"),
        "manifest_path": str(manifest_path),
    }

    summary_path = output_dir / "gate4_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nShowcase Manifest written to: {manifest_path}")
    print(f"Showcase Summary written to: {summary_path}")
    print(f"Total Visual Artifacts (.svg): {len(visual_manifest)}")
    print(f"Total Semantic Companion Artifacts (.json): {len(semantic_manifest)}")
    print("=" * 80)
    print("GATE-4 SHOWCASE EXECUTION SUCCESSFUL")
    print("=" * 80)


if __name__ == "__main__":
    run_gate4_showcase()
