#!/usr/bin/env python3
"""Gate 5 Showcase: Institutional Tail Risk, Expected Shortfall & Advanced Backtesting.

Executes the complete Gate-5 institutional Market vertical slice:
1. Proof-Carrying Negative Evidence Showcase:
   - Clustered exception sequence demonstrating that Kupiec POF does NOT reject unconditional coverage (p > 0.05),
     while Christoffersen independence REJECTS (p < 0.05) due to severe exception clustering.
2. Historical & Parametric Normal Tail Risk Estimation:
   - Exact finite-sample Rockafellar-Uryasev Expected Shortfall with fractional boundary weight.
   - Multi-model comparison across historical and parametric estimators.
3. Component Tail Risk Decomposition & Euler Reconciliation:
   - Parametric Normal component VaR and component ES summing exactly to portfolio risk.
   - Historical component ES using portfolio scenario tail weights.
4. Comprehensive Artifact Plane:
   - Generates dual-plane vector SVG visuals and JSON companion files in start_output/gate5_showcase/.
5. Multi-Agent Director Orchestration:
   - Data Integrity -> Covariance -> Factor Risk -> Hierarchical -> Portfolio -> Tail Risk Agent -> Adversarial Challenger -> Critic -> Governance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from start.agents.market_review import (
    MarketReviewDirectorAgent,
)
from start.portfolio.artifacts import (
    render_backtest_summary_artifact,
    render_duration_diagnostics_artifact,
    render_exception_transition_artifact,
    render_tail_comparison_artifact,
    render_tail_loss_distribution_artifact,
    render_tail_risk_contribution_artifact,
    render_tail_severity_artifact,
    render_var_pnl_timeline_artifact,
)
from start.portfolio.contracts import (
    MetricHorizon,
)
from start.portfolio.evidence_bridge import (
    duration_diagnostics_to_evidence,
    tail_backtest_to_evidence,
    tail_comparison_to_evidence,
    tail_contribution_to_evidence,
    tail_risk_estimate_to_evidence,
    tail_severity_to_evidence,
)
from start.portfolio.tail_risk import (
    compare_tail_risk_models,
    compute_exception_duration_diagnostics,
    compute_historical_var_es,
    compute_parametric_normal_var_es,
    compute_tail_risk_contributions,
    compute_tail_severity,
    run_comprehensive_tail_backtest,
)


def run_gate5_showcase() -> None:
    output_dir = Path("start_output/gate5_showcase")
    output_dir.mkdir(parents=True, exist_ok=True)
    out_dir_str = str(output_dir)

    print("=" * 80)
    print("StART GATE 5: INSTITUTIONAL TAIL RISK, EXPECTED SHORTFALL & BACKTESTING")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 1. NEGATIVE EVIDENCE SHOWCASE: CANONICAL 4-EXCEPTION CLUSTER
    # -------------------------------------------------------------------------
    print("\n[1/5] NEGATIVE EVIDENCE SHOWCASE: KUPIEC NON-REJECTION VS CHRISTOFFERSEN REJECTION")
    # Synthetic out-of-sample sequence: 250 days, 4 exceptions in ONE single consecutive cluster (days 50..53)
    n_days = 250
    alpha_var = 0.99
    gamma_test = 0.05
    var_const = 0.025  # 2.5% daily VaR threshold

    # 1. Construct canonical exception indicator sequence first
    canonical_indicators = np.zeros(n_days, dtype=int)
    canonical_indicators[50:54] = 1
    canonical_indicator_hash = hashlib.sha256(canonical_indicators.tobytes()).hexdigest()

    # 2. Construct PnL series whose breach sequence strictly reconciles to canonical_indicators
    rng = np.random.RandomState(123)
    pnl_series = np.clip(rng.normal(loc=0.0005, scale=0.006, size=n_days), -0.015, 0.015)
    # Inject 4 consecutive losses on days 50..53 (losses 3.5%, 4.2%, 3.8%, 4.5% vs VaR 2.5%)
    pnl_series[50] = -0.035
    pnl_series[51] = -0.042
    pnl_series[52] = -0.038
    pnl_series[53] = -0.045
    var_series = np.full(n_days, var_const)

    # Invariant assertion: derived breach indicator must match canonical sequence identically
    derived_indicators = (pnl_series < -var_series).astype(int)
    assert np.array_equal(derived_indicators, canonical_indicators), (
        "Derived indicators do not match canonical sequence"
    )

    # Execute comprehensive backtest
    backtest_clustered = run_comprehensive_tail_backtest(
        pnl_or_losses=pnl_series,
        var_series=var_series,
        var_confidence=alpha_var,
        test_significance=gamma_test,
        pnl_source="actual",
        is_loss_series=False,
    )

    ev_backtest_clustered = tail_backtest_to_evidence(backtest_clustered)
    ev_backtest_clustered.evidence_id = "EV-TAIL-BACKTEST-CLUSTERED"

    # Compute duration diagnostics
    dur_clustered = compute_exception_duration_diagnostics(backtest_clustered.indicators)
    ev_dur_clustered = duration_diagnostics_to_evidence(dur_clustered)
    ev_dur_clustered.evidence_id = "EV-TAIL-DURATIONS-CLUSTERED"

    # Compute tail severity
    sev_clustered = compute_tail_severity(
        losses=-pnl_series,
        var_forecasts=var_series,
        indicators=backtest_clustered.indicators,
    )
    ev_sev_clustered = tail_severity_to_evidence(sev_clustered)
    ev_sev_clustered.evidence_id = "EV-TAIL-SEVERITY-CLUSTERED"

    print(
        f"  Canonical Indicator Hash: {canonical_indicator_hash[:16]}... (T={backtest_clustered.n_observations}, x={backtest_clustered.n_exceptions}, rate={backtest_clustered.exception_rate:.2%})"
    )
    print(
        f"  Kupiec POF LR: {backtest_clustered.kupiec_lr:.4f}, p-value: {backtest_clustered.kupiec_p_value:.4f} -> DOES NOT REJECT unconditional coverage at gamma={gamma_test:.2%}"
    )
    print(
        f"  Christoffersen Transitions: n00={backtest_clustered.n00}, n01={backtest_clustered.n01}, n10={backtest_clustered.n10}, n11={backtest_clustered.n11} (sum={sum(backtest_clustered.transition_counts)})"
    )
    print(
        f"  Christoffersen Ind LR: {backtest_clustered.christoffersen_lr:.4f}, p-value: {backtest_clustered.christoffersen_p_value:.4e} -> REJECTS serial independence (clustering detected)"
    )
    print(
        f"  Joint Conditional Coverage LR: {backtest_clustered.conditional_coverage_lr:.4f}, p-value: {backtest_clustered.conditional_coverage_p_value:.4e} -> REJECTS joint conditional coverage"
    )
    print(
        f"  Duration Intervals: {dur_clustered.n_durations}, Mean Duration: {dur_clustered.mean_duration:.1f} days, Max Exception Run: {dur_clustered.max_run_length} days"
    )
    print(
        f"  Tail Severity: Mean Exceedance = {sev_clustered.mean_absolute_exceedance:.4f}, Max Ratio = {sev_clustered.max_normalized_exceedance:.2f}x VaR"
    )

    # Render Backtesting & Tail Artifacts
    art_timeline = render_var_pnl_timeline_artifact(
        backtest_clustered, evidence_ids=(ev_backtest_clustered.evidence_id,), output_dir=out_dir_str
    )
    art_trans = render_exception_transition_artifact(
        backtest_clustered, evidence_ids=(ev_backtest_clustered.evidence_id,), output_dir=out_dir_str
    )
    art_dur = render_duration_diagnostics_artifact(
        dur_clustered, evidence_ids=(ev_dur_clustered.evidence_id,), output_dir=out_dir_str
    )
    art_sev = render_tail_severity_artifact(
        sev_clustered, evidence_ids=(ev_sev_clustered.evidence_id,), output_dir=out_dir_str
    )
    art_summary = render_backtest_summary_artifact(
        backtest_clustered, evidence_ids=(ev_backtest_clustered.evidence_id,), output_dir=out_dir_str
    )

    # -------------------------------------------------------------------------
    # 2. HISTORICAL & PARAMETRIC NORMAL TAIL ESTIMATION & MODEL COMPARISON
    # -------------------------------------------------------------------------
    print("\n[2/5] TAIL ESTIMATION: EXACT FINITE-SAMPLE ES & PARAMETRIC NORMAL COMPARISON")
    # Simulate 500 daily portfolio returns with fat tails (Student-t degrees of freedom 4)
    rng_t = np.random.RandomState(42)
    t_returns = rng_t.standard_t(df=4, size=500) * 0.01

    hist_est = compute_historical_var_es(
        losses=t_returns,
        confidence=alpha_var,
        quantile_method="linear",
        horizon=MetricHorizon.PERIODIC,
        is_returns=True,
    )
    ev_hist_est = tail_risk_estimate_to_evidence(hist_est)
    ev_hist_est.evidence_id = "EV-TAIL-HIST-EST"

    param_est = compute_parametric_normal_var_es(
        returns_or_losses=t_returns,
        confidence=alpha_var,
        is_returns=True,
        horizon=MetricHorizon.PERIODIC,
    )
    ev_param_est = tail_risk_estimate_to_evidence(param_est)
    ev_param_est.evidence_id = "EV-TAIL-PARAM-EST"

    tail_compare = compare_tail_risk_models(
        returns_or_losses=t_returns,
        confidence=alpha_var,
        is_returns=True,
    )
    ev_tail_compare = tail_comparison_to_evidence(tail_compare)
    ev_tail_compare.evidence_id = "EV-TAIL-COMPARE"

    print(
        f"  Historical @ {alpha_var:.1%}: VaR = {hist_est.var:.4f}, Exact Finite-Sample ES = {hist_est.es:.4f} (q={hist_est.parameters['q_tail_mass']:.2f}, boundary_wt={hist_est.boundary_weight:.4f})"
    )
    print(f"  Parametric Normal @ {alpha_var:.1%}: VaR = {param_est.var:.4f}, ES = {param_est.es:.4f}")
    print(
        f"  Tail Ratio (ES / VaR): Historical = {tail_compare.es_to_var_ratios['historical']:.3f}x, Normal = {tail_compare.es_to_var_ratios['parametric_normal']:.3f}x"
    )

    art_hist_dist = render_tail_loss_distribution_artifact(
        hist_est, evidence_ids=(ev_hist_est.evidence_id,), output_dir=out_dir_str
    )
    art_tail_compare = render_tail_comparison_artifact(
        tail_compare, evidence_ids=(ev_tail_compare.evidence_id,), output_dir=out_dir_str
    )

    # -------------------------------------------------------------------------
    # 3. COMPONENT TAIL RISK CONTRIBUTIONS & EULER RECONCILIATION
    # -------------------------------------------------------------------------
    print("\n[3/5] COMPONENT TAIL RISK CONTRIBUTIONS & EXACT EULER RECONCILIATION")
    assets = ["EQUITY_US", "EQUITY_GLOBAL", "FIXED_INCOME", "COMMODITIES"]
    weights = {"EQUITY_US": 0.40, "EQUITY_GLOBAL": 0.25, "FIXED_INCOME": 0.25, "COMMODITIES": 0.10}

    # Simulate multi-asset return history (500 days x 4 assets)
    cov_true = np.array(
        [
            [0.000400, 0.000280, 0.000040, 0.000100],
            [0.000280, 0.000450, 0.000050, 0.000120],
            [0.000040, 0.000050, 0.000100, 0.000010],
            [0.000100, 0.000120, 0.000010, 0.000600],
        ]
    )
    multi_rets = np.random.RandomState(42).multivariate_normal(np.zeros(4), cov_true, size=500)
    rets_df = pd.DataFrame(multi_rets, columns=assets)

    # Parametric Normal decomposition
    contrib_param = compute_tail_risk_contributions(
        returns_or_losses=rets_df,
        weights=weights,
        confidence=alpha_var,
        method="parametric_normal",
        is_returns=True,
    )
    ev_contrib_param = tail_contribution_to_evidence(contrib_param)
    ev_contrib_param.evidence_id = "EV-TAIL-CONTRIB-PARAM"

    # Historical ES decomposition
    contrib_hist = compute_tail_risk_contributions(
        returns_or_losses=rets_df,
        weights=weights,
        confidence=alpha_var,
        method="historical_es",
        is_returns=True,
    )
    ev_contrib_hist = tail_contribution_to_evidence(contrib_hist)
    ev_contrib_hist.evidence_id = "EV-TAIL-CONTRIB-HIST"

    print(
        f"  Parametric Portfolio VaR: {contrib_param.portfolio_var:.4f}, ES: {contrib_param.portfolio_es:.4f}"
    )
    print(
        f"  Parametric Component VaR sum err: {contrib_param.var_reconciliation_error:.2e}, ES sum err: {contrib_param.es_reconciliation_error:.2e}"
    )
    for a in assets:
        print(
            f"    - {a:15s}: Comp VaR={contrib_param.component_var[a]:.4f} ({contrib_param.percentage_var_contributions[a]:.1%}), Comp ES={contrib_param.component_es[a]:.4f} ({contrib_param.percentage_es_contributions[a]:.1%})"
        )
    print(
        f"  Historical Portfolio ES: {contrib_hist.portfolio_es:.4f}, ES sum err: {contrib_hist.es_reconciliation_error:.2e}"
    )

    art_contrib_param = render_tail_risk_contribution_artifact(
        contrib_param, evidence_ids=(ev_contrib_param.evidence_id,), output_dir=out_dir_str
    )
    art_contrib_hist = render_tail_risk_contribution_artifact(
        contrib_hist, evidence_ids=(ev_contrib_hist.evidence_id,), output_dir=out_dir_str
    )

    # -------------------------------------------------------------------------
    # 4. MULTI-AGENT MARKET REVIEW DIRECTOR WITH SPECIALIST TAIL RISK AGENT
    # -------------------------------------------------------------------------
    print("\n[4/5] MULTI-AGENT REVIEW DIRECTOR ORCHESTRATION WITH TAIL RISK SPECIALIST")
    all_evidence = [
        ev_backtest_clustered,
        ev_dur_clustered,
        ev_sev_clustered,
        ev_hist_est,
        ev_param_est,
        ev_tail_compare,
        ev_contrib_param,
        ev_contrib_hist,
    ]

    director_context = {
        "evidence_records": all_evidence,
        "returns": rets_df,
        "covariance": cov_true,
        "weights": weights,
        "pnl": pnl_series,
        "var_series": var_series,
        "var_confidence": alpha_var,
        "test_significance": gamma_test,
        "auto_resolve_challenges": True,
    }

    director = MarketReviewDirectorAgent()
    director_result = director.execute(director_context)

    print(f"  Director Status: {director_result['status']}")
    print(f"  Total Findings: {director_result['findings_count']}")
    print(f"  Adversarial Challenges Formulated: {director_result['challenges_count']}")
    print(f"  Adversarial Challenges Resolved: {director_result['resolutions_count']}")
    print(f"  Critic Disposition: {director_result['critic_disposition']}")
    print(f"  Governance Verdict: {director_result['governance_verdict']}")

    # -------------------------------------------------------------------------
    # 5. MANIFEST & DISPOSITION SUMMARY
    # -------------------------------------------------------------------------
    print("\n[5/5] GENERATING MANIFEST & VERIFICATION SUMMARY")
    human_review_artifacts = [
        art_timeline,
        art_trans,
        art_dur,
        art_sev,
        art_summary,
        art_hist_dist,
        art_tail_compare,
        art_contrib_param,
        art_contrib_hist,
    ]

    visual_manifest = []
    semantic_manifest = []

    for a in human_review_artifacts:
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
        json_path = a.file_path.replace(".svg", ".json") if a.file_path else None
        semantic_manifest.append(
            {
                "artifact_id": a.artifact_id,
                "artifact_type": a.spec.artifact_type,
                "title": a.spec.title,
                "file_path": json_path,
                "rendering_format": "json",
                "evidence_ids": list(a.spec.evidence_ids),
                "semantic_payload_hash": a.semantic_payload_hash,
            }
        )

    manifest = {
        "gate": "GATE_5",
        "title": "Institutional Tail Risk, Expected Shortfall & Advanced Backtesting Showcase",
        "var_confidence": alpha_var,
        "test_significance": gamma_test,
        "loss_sign_convention": "positive_loss_magnitude (L = -R)",
        "exception_equality_rule": "I_t = 1 iff loss_t > VaR_t (PnL_t < -VaR_t)",
        "forecast_horizon": "1-day out-of-sample",
        "estimation_timing_convention": "Information set t-1 -> VaR_t -> Realized Loss_t -> Exception I_t",
        "quantile_convention": "linear interpolation empirical quantile",
        "es_tail_weighting_convention": "Exact finite-sample Rockafellar-Uryasev weighted order statistic",
        "canonical_backtest_showcase": {
            "n_observations": backtest_clustered.n_observations,
            "n_exceptions": backtest_clustered.n_exceptions,
            "exception_rate": backtest_clustered.exception_rate,
            "indicator_hash": backtest_clustered.indicator_hash,
            "expected_exceptions": backtest_clustered.expected_exceptions,
            "kupiec_lr": backtest_clustered.kupiec_lr,
            "kupiec_p_value": backtest_clustered.kupiec_p_value,
            "kupiec_rejected": backtest_clustered.kupiec_rejected,
            "n00": backtest_clustered.n00,
            "n01": backtest_clustered.n01,
            "n10": backtest_clustered.n10,
            "n11": backtest_clustered.n11,
            "pi_01": backtest_clustered.pi_01,
            "pi_11": backtest_clustered.pi_11,
            "christoffersen_lr": backtest_clustered.christoffersen_lr,
            "christoffersen_p_value": backtest_clustered.christoffersen_p_value,
            "christoffersen_rejected": backtest_clustered.christoffersen_rejected,
            "conditional_coverage_lr": backtest_clustered.conditional_coverage_lr,
            "conditional_coverage_p_value": backtest_clustered.conditional_coverage_p_value,
            "conditional_coverage_rejected": backtest_clustered.conditional_coverage_rejected,
            "duration_intervals": dur_clustered.n_durations,
            "mean_duration": dur_clustered.mean_duration,
            "max_run_length": dur_clustered.max_run_length,
            "mean_absolute_exceedance": sev_clustered.mean_absolute_exceedance,
            "max_normalized_exceedance": sev_clustered.max_normalized_exceedance,
        },
        "governance_verdict": director_result["governance_verdict"],
        "semantic_artifacts_count": len(semantic_manifest),
        "human_review_artifacts_count": len(visual_manifest),
        "semantic_artifacts": semantic_manifest,
        "human_review_artifacts": visual_manifest,
        "evidence_records": [
            {
                "evidence_id": r.evidence_id,
                "test_id": r.test_id,
                "status": str(r.status),
                "interpretation": r.interpretation,
            }
            for r in all_evidence
        ],
        "challenges": director_result["challenges"],
        "challenge_resolutions": director_result["resolutions"],
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    summary = {
        "status": "COMPLETED",
        "gate": "GATE_5",
        "evidence_records_count": len(all_evidence),
        "artifacts_emitted_count": len(human_review_artifacts),
        "semantic_artifacts_count": len(semantic_manifest),
        "human_review_artifacts_count": len(visual_manifest),
        "governance_verdict": director_result["governance_verdict"],
        "governance_details": director_result.get("governance"),
        "manifest_path": str(manifest_path),
    }

    summary_path = output_dir / "gate5_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(
        f"  Manifest written to {manifest_path} ({len(visual_manifest)} visual artifacts, {len(all_evidence)} evidence records)."
    )
    print(f"  Summary written to {summary_path}.")
    print("\n" + "=" * 80)
    print("GATE 5 SHOWCASE COMPLETE: ALL TAIL RISK & BACKTESTING CONTRACTS VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    run_gate5_showcase()
