"""Evidence Bridge for Institutional Portfolio Intelligence.

Bridges typed portfolio analytical results to StART's TestResult and EvidenceRecord infrastructure.
Ensures zero orphan quantitative results: every analytical calculation consumed by an agent
is wrapped in a deterministic TestResult and audit-grade EvidenceRecord.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from start.core.schemas import EvidenceRecord, Status, TestResult
from start.portfolio.contracts import (
    ActiveRiskDecompositionResult,
    ActiveScenarioResult,
    BlackLittermanResult,
    BootstrapStabilityResult,
    BrinsonAttributionResult,
    CarinoLinkedAttributionResult,
    ConstraintVerificationResult,
    CopheneticResult,
    CovarianceComparisonResult,
    CovarianceDiagnostics,
    CVaROptimizationResult,
    DurationDiagnosticsResult,
    EfficientFrontierResult,
    EqualRiskContributionResult,
    FactorDataIntegrityResult,
    FactorReturnAttributionResult,
    FactorRiskDecompositionResult,
    FactorRiskModelResult,
    HERCResult,
    HierarchicalTreeResult,
    LinkageSensitivityResult,
    MaxDiversificationResult,
    MethodComparisonResult,
    PSDRepairResult,
    RebalanceDecision,
    ReverseStressResult,
    RiskContributionResult,
    RobustMVOResult,
    RobustSensitivityResult,
    ScenarioDataIntegrityResult,
    ScenarioResult,
    ScenarioSensitivityResult,
    ScenarioSetResult,
    TailBacktestResult,
    TailModelComparisonResult,
    TailRiskContributionResult,
    TailRiskEstimate,
    TailSeverityResult,
    TrackingErrorResult,
    WalkForwardResult,
)


def _hash_dict(d: dict[str, Any]) -> str:
    serialized = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()[:32]


# --------------------------------------------------------------------------- #
# 1. HRP & Tree Hierarchy Bridge
# --------------------------------------------------------------------------- #
def tree_to_evidence(
    tree: HierarchicalTreeResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-HRP",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert HierarchicalTreeResult into a subordinate EvidenceRecord."""
    metrics: dict[str, Any] = {
        "n_assets": len(tree.assets),
        "linkage_method": tree.linkage_method,
        "distance_method": tree.distance_method,
        "quasi_diagonal_order": ", ".join(tree.quasi_diagonal_order),
        "correlation_fingerprint": tree.correlation_fingerprint,
        "covariance_fingerprint": tree.covariance_fingerprint,
    }
    if tree.cophenetic_correlation is not None:
        metrics["cophenetic_correlation"] = tree.cophenetic_correlation

    result = TestResult(
        test_id="portfolio.hierarchical_risk_parity.tree_topology",
        test_name="HRP Tree Hierarchy and Seriation",
        status=Status.RECORDED,
        params={"linkage_method": tree.linkage_method, "distance_method": tree.distance_method},
        metrics=metrics,
        interpretation=(
            f"Hierarchical tree constructed with {tree.linkage_method} linkage "
            f"over {len(tree.assets)} assets; quasi-diagonal leaf order established."
        ),
        limitations=[
            "Quasi-diagonalization ordering reflects scipy's deterministic tie-break.",
            "Distance matrix is derived from angular correlation distance.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=tree.covariance_fingerprint,
    )


# --------------------------------------------------------------------------- #
# 2. Linkage Sensitivity Bridge
# --------------------------------------------------------------------------- #
def linkage_sensitivity_to_evidence(
    sens: LinkageSensitivityResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-HRP",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert LinkageSensitivityResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "n_methods_compared": len(sens.methods_compared),
        "methods_compared": ", ".join(sens.methods_compared),
    }
    for m in sens.methods_compared:
        metrics[f"effective_positions.{m}"] = sens.effective_positions_by_linkage.get(m)
        metrics[f"portfolio_variance.{m}"] = sens.portfolio_variance_by_linkage.get(m)

    for pair, l1 in sens.pairwise_l1_distances.items():
        metrics[f"pairwise_l1.{pair}"] = l1
    for pair, l2 in sens.pairwise_l2_distances.items():
        metrics[f"pairwise_l2.{pair}"] = l2
    for pair, md in sens.max_asset_weight_diffs.items():
        metrics[f"max_weight_diff.{pair}"] = md
    for pair, sp in sens.spearman_order_correlations.items():
        metrics[f"spearman_rank_corr.{pair}"] = sp

    result = TestResult(
        test_id="portfolio.hierarchical_risk_parity.linkage_sensitivity",
        test_name="HRP Linkage Sensitivity Analysis",
        status=Status.RECORDED,
        params={"methods": list(sens.methods_compared)},
        metrics=metrics,
        interpretation=(
            f"Evaluated linkage sensitivity across {len(sens.methods_compared)} methods. "
            f"Pairwise L1 and Spearman rank correlations recorded descriptively."
        ),
        limitations=[
            "Linkage alternatives represent structural sensitivities; no universal superiority exists.",
            "Descriptive diagnostic; no pass/fail threshold is imposed without user policy.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(sens.weights_by_linkage),
    )


# --------------------------------------------------------------------------- #
# 3. Cophenetic Diagnostic Bridge
# --------------------------------------------------------------------------- #
def cophenetic_to_evidence(
    coph: CopheneticResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-HRP",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert CopheneticResult into an EvidenceRecord."""
    result = TestResult(
        test_id="portfolio.hierarchical_risk_parity.cophenetic_distance",
        test_name="Cophenetic Correlation Distance Diagnostic",
        status=Status.RECORDED,
        params={"linkage_method": coph.linkage_method, "distance_method": coph.distance_method},
        metrics={
            "cophenetic_correlation": coph.cophenetic_correlation,
            "linkage_method": coph.linkage_method,
            "distance_method": coph.distance_method,
            "n_assets": coph.n_assets,
        },
        interpretation=(
            f"Cophenetic correlation coefficient is {coph.cophenetic_correlation:.4f} "
            f"for {coph.linkage_method} linkage over {coph.n_assets} assets."
        ),
        limitations=[
            "Cophenetic correlation measures how faithfully the dendrogram preserves pairwise distances.",
            "Reported descriptively without arbitrary pass/fail thresholds.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(
            {
                "cophenetic": coph.cophenetic_correlation,
                "linkage": coph.linkage_method,
            }
        ),
    )


# --------------------------------------------------------------------------- #
# 4. Bootstrap Cluster Stability Bridge
# --------------------------------------------------------------------------- #
def bootstrap_stability_to_evidence(
    boot: BootstrapStabilityResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-HRP",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert BootstrapStabilityResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "bootstrap_method": boot.bootstrap_method,
        "block_size": boot.block_size,
        "n_replicates": boot.n_replicates,
        "seed": boot.seed,
        "n_assets": len(boot.assets),
        "mean_pairwise_stability": boot.mean_pairwise_stability,
        "min_pairwise_stability": boot.min_pairwise_stability,
        "cophenetic_stability_mean": boot.cophenetic_stability_mean,
    }
    result = TestResult(
        test_id="portfolio.hierarchical_risk_parity.bootstrap_stability",
        test_name="Time-Series Block Bootstrap Cluster Stability",
        status=Status.RECORDED,
        params={
            "bootstrap_method": boot.bootstrap_method,
            "block_size": boot.block_size,
            "n_replicates": boot.n_replicates,
            "seed": boot.seed,
        },
        metrics=metrics,
        interpretation=(
            f"Stationary block bootstrap ({boot.n_replicates} replicates, block {boot.block_size}) "
            f"yielded mean pairwise cluster co-occurrence stability of {boot.mean_pairwise_stability:.4f}."
        ),
        limitations=[
            "Block bootstrap preserves short-term serial dependence.",
            "Descriptive diagnostic; no arbitrary pass/fail threshold is imposed without user policy.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict({"matrix": boot.pairwise_co_clustering_matrix}),
    )


# --------------------------------------------------------------------------- #
# 5. Euler Risk Contributions Bridge
# --------------------------------------------------------------------------- #
def risk_contributions_to_evidence(
    rc: RiskContributionResult,
    test_id: str = "portfolio.risk_statistics.euler_decomposition",
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-RISK",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert RiskContributionResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "portfolio_variance": rc.portfolio_variance,
        "portfolio_volatility": rc.portfolio_volatility,
        "euler_reconciliation_error": rc.euler_reconciliation_error,
        "n_assets": len(rc.marginal_contributions),
    }
    for asset, mcr in rc.marginal_contributions.items():
        metrics[f"mcr.{asset}"] = mcr
    for asset, cr in rc.component_contributions.items():
        metrics[f"cr.{asset}"] = cr
    for asset, pcr in rc.percentage_contributions.items():
        metrics[f"pcr.{asset}"] = pcr

    for cname, c_cr in rc.cluster_contributions.items():
        metrics[f"cluster_cr.{cname}"] = c_cr
    for cname, c_pcr in rc.cluster_percentage_contributions.items():
        metrics[f"cluster_pcr.{cname}"] = c_pcr

    result = TestResult(
        test_id=test_id,
        test_name="Euler Risk Contribution Decomposition",
        status=Status.RECORDED,
        params={},
        metrics=metrics,
        interpretation=(
            f"Euler risk decomposition reconciled with error {rc.euler_reconciliation_error:.2e}. "
            f"Portfolio volatility: {rc.portfolio_volatility:.6f}."
        ),
        limitations=[
            "Euler decomposition assumes homogeneous degree 1 risk measure.",
            "Component risk contributions sum exactly to total portfolio volatility.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(rc.percentage_contributions),
    )


# --------------------------------------------------------------------------- #
# 6. Equal-Weight (1/N) Baseline Bridge
# --------------------------------------------------------------------------- #
def equal_weight_to_evidence(
    weights_series: Any,
    metrics_dict: dict[str, Any],
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-BASELINE",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert 1/N baseline allocation into an EvidenceRecord."""
    flat_metrics: dict[str, Any] = {}
    for k, v in metrics_dict.items():
        if isinstance(v, dict):
            for sub_k, sub_v in v.items():
                flat_metrics[f"{k}.{sub_k}"] = sub_v
        else:
            flat_metrics[k] = v

    result = TestResult(
        test_id="portfolio.mean_variance.equal_weight_baseline",
        test_name="1/N Equal-Weight Portfolio Baseline",
        status=Status.RECORDED,
        params={"method": "1_over_n"},
        metrics=flat_metrics,
        interpretation=(
            f"Explicit 1/N equal-weight benchmark computed over {metrics_dict.get('n_assets')} assets; "
            f"effective positions: {metrics_dict.get('effective_n_positions')}."
        ),
        limitations=[
            "DeMiguel et al. (2009) 1/N benchmark ignores expected return and covariance structure.",
            "Serves as an objective, non-optimized comparative standard.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(metrics_dict),
    )


# --------------------------------------------------------------------------- #
# 7. Equal Risk Contribution (ERC) Bridge
# --------------------------------------------------------------------------- #
def erc_to_evidence(
    erc: EqualRiskContributionResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-ERC",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert EqualRiskContributionResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "n_assets": len(erc.weights),
        "target_risk_contribution": erc.target_risk_contribution,
        "max_risk_contribution_dispersion": erc.max_risk_contribution_dispersion,
        "portfolio_volatility": erc.portfolio_volatility,
        "portfolio_variance": erc.portfolio_variance,
        "objective_value": erc.objective_value,
        "solver_iterations": erc.solver_iterations,
        "converged": erc.converged,
        "budget_violation": erc.constraint_violations.get("budget", 0.0),
        "non_negativity_violation": erc.constraint_violations.get("non_negativity", 0.0),
    }
    for asset, w in erc.weights.items():
        metrics[f"weight.{asset}"] = w
        metrics[f"risk_contribution.{asset}"] = erc.percentage_risk_contributions.get(asset, 0.0)

    result = TestResult(
        test_id="portfolio.risk_statistics.equal_risk_contribution",
        test_name="Equal Risk Contribution Allocation",
        status=Status.RECORDED,
        params={"target": "equal_risk_budget"},
        metrics=metrics,
        interpretation=(
            f"Equal Risk Contribution solved over {len(erc.weights)} assets; "
            f"max dispersion: {erc.max_risk_contribution_dispersion:.2e}, converged: {erc.converged}."
        ),
        limitations=[
            "ERC equates percentage volatility contributions across assets without expected return inputs.",
            "Solution verified independently against budget and non-negativity constraints.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(erc.weights),
    )


# --------------------------------------------------------------------------- #
# 8. Efficient Frontier Bridge
# --------------------------------------------------------------------------- #
def efficient_frontier_to_evidence(
    frontier: EfficientFrontierResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-FRONTIER",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert EfficientFrontierResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "n_frontier_points": len(frontier.frontier_points),
        "min_volatility_annualised": frontier.min_variance_point.volatility_annualised,
        "min_vol_expected_return_annualised": frontier.min_variance_point.expected_return_annualised,
        "max_sharpe_annualised": frontier.max_sharpe_point.sharpe_annualised,
        "max_sharpe_volatility_annualised": frontier.max_sharpe_point.volatility_annualised,
        "max_sharpe_expected_return_annualised": frontier.max_sharpe_point.expected_return_annualised,
    }
    if frontier.equal_weight_point:
        metrics["equal_weight_volatility_annualised"] = frontier.equal_weight_point.volatility_annualised
    if frontier.erc_point:
        metrics["erc_volatility_annualised"] = frontier.erc_point.volatility_annualised
    if frontier.hrp_point:
        metrics["hrp_volatility_annualised"] = frontier.hrp_point.volatility_annualised

    result = TestResult(
        test_id="portfolio.mean_variance.efficient_frontier",
        test_name="Parametric Efficient Frontier Curve",
        status=Status.RECORDED,
        params={"n_points": len(frontier.frontier_points)},
        metrics=metrics,
        interpretation=(
            f"Parametric efficient frontier traced with {len(frontier.frontier_points)} points. "
            f"Min volatility: {frontier.min_variance_point.volatility_annualised:.2%}, "
            f"Max Sharpe: {frontier.max_sharpe_point.sharpe_annualised}."
        ),
        limitations=[
            "Frontier assumes input expected return and covariance parameters are known with certainty.",
            "Solves under exact active constraint set.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(metrics),
    )


# --------------------------------------------------------------------------- #
# 9. Portfolio Method Comparison Bridge
# --------------------------------------------------------------------------- #
def method_comparison_to_evidence(
    comp: MethodComparisonResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-COMPARISON",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert MethodComparisonResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "n_methods_compared": len(comp.methods),
        "methods_evaluated": ", ".join(comp.methods),
    }
    for row in comp.summary_table:
        mname = row["method"]
        metrics[f"{mname}.annualised_return"] = row.get("annualised_return")
        metrics[f"{mname}.annualised_volatility"] = row.get("annualised_volatility")
        metrics[f"{mname}.annualised_sharpe"] = row.get("annualised_sharpe")
        metrics[f"{mname}.herfindahl"] = row.get("herfindahl")
        metrics[f"{mname}.effective_n"] = row.get("effective_n_positions")
        metrics[f"{mname}.max_weight"] = row.get("max_weight")
        metrics[f"{mname}.turnover_vs_current"] = row.get("turnover_vs_current")

    result = TestResult(
        test_id="portfolio.mean_variance.method_comparison",
        test_name="Portfolio Construction Method Comparison",
        status=Status.RECORDED,
        params={"methods": list(comp.methods)},
        metrics=metrics,
        interpretation=(
            f"Evaluated {len(comp.methods)} portfolio allocation methods side-by-side. "
            "No automatic winner is declared; full comparative evidence recorded."
        ),
        limitations=[
            "Evaluates sample empirical performance; out-of-sample stability requires walk-forward analysis.",
            "Descriptive comparative analysis with zero unproven marketing claims.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(comp.weights_matrix),
    )


# --------------------------------------------------------------------------- #
# 10. Walk-Forward Evaluation Bridge
# --------------------------------------------------------------------------- #
def walk_forward_to_evidence(
    wf: WalkForwardResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-WALKFORWARD",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert WalkForwardResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "method": wf.method,
        "n_rebalance_dates": len(wf.rebalance_dates),
        "n_oos_periods": len(wf.out_of_sample_returns),
        "annualised_return": wf.annualised_return,
        "annualised_volatility": wf.annualised_volatility,
        "realized_sharpe": wf.realized_sharpe,
        "max_drawdown": wf.max_drawdown,
        "mean_one_way_turnover": wf.mean_one_way_turnover,
        "transaction_cost_bps": wf.transaction_cost_bps,
    }
    result = TestResult(
        test_id="portfolio.historical_returns.walk_forward",
        test_name="Out-of-Sample Walk-Forward Portfolio Simulation",
        status=Status.RECORDED,
        params={
            "method": wf.method,
            "transaction_cost_bps": wf.transaction_cost_bps,
            "rebalance_count": len(wf.rebalance_dates),
        },
        metrics=metrics,
        interpretation=(
            f"Non-leaky walk-forward simulation ({wf.method}) across {len(wf.rebalance_dates)} rebalances "
            f"yielded annualized return {wf.annualised_return:.2%}, "
            f"volatility {wf.annualised_volatility:.2%}, "
            f"max drawdown {wf.max_drawdown:.2%} at {wf.transaction_cost_bps} bps cost."
        ),
        limitations=[
            "Strict chronological non-leakage enforced: estimation uses [t-W, t) only.",
            "Transaction drag deducted on rebalance decisions.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict({"rebalances": list(wf.rebalance_dates)}),
    )


# =========================================================================== #
# GATE-3 INSTITUTIONAL EVIDENCE BRIDGES
# =========================================================================== #
def black_litterman_to_evidence(
    bl: BlackLittermanResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-BL",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert BlackLittermanResult into an audit-grade EvidenceRecord."""
    metrics: dict[str, Any] = {
        "risk_aversion": bl.risk_aversion,
        "tau": bl.tau,
        "turnover_vs_prior": bl.turnover_vs_prior,
        "posterior_volatility_annualised": bl.posterior_volatility_annualised,
        "posterior_sharpe_annualised": bl.posterior_sharpe_annualised,
        "constraint_violations_count": len(
            [v for v in bl.constraint_verification.violations if v.status == "VIOLATED"]
        ),
        "max_constraint_violation": bl.constraint_verification.max_violation,
    }
    for a, ret in bl.implied_returns.items():
        metrics[f"implied_returns.{a}"] = ret
    for a, ret in bl.posterior_returns.items():
        metrics[f"posterior_returns.{a}"] = ret
    for a, w in bl.posterior_weights.items():
        metrics[f"posterior_weight.{a}"] = w
    for v, res in bl.view_residuals.items():
        metrics[f"view_residual.{v}"] = res
    for v, unc in bl.view_uncertainties.items():
        metrics[f"view_uncertainty.{v}"] = unc

    result = TestResult(
        test_id="portfolio.black_litterman",
        test_name="Black-Litterman Bayesian Portfolio Optimization",
        status=Status.RECORDED,
        params={
            "risk_aversion": bl.risk_aversion,
            "tau": bl.tau,
            "n_views": len(bl.view_labels),
            "view_labels": list(bl.view_labels),
        },
        metrics=metrics,
        interpretation=(
            f"Black-Litterman optimization with {len(bl.view_labels)} view(s) (tau={bl.tau}, delta={bl.risk_aversion}). "
            f"Posterior volatility: {bl.posterior_volatility_annualised:.2%}, "
            f"Turnover vs prior: {bl.turnover_vs_prior:.2%}."
        ),
        limitations=[
            "Prior equilibrium assumes CAPM market portfolio optimality.",
            "Posterior distribution depends on view uncertainty specification Omega.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=bl.posterior_covariance_fingerprint,
    )


def robust_mvo_to_evidence(
    rob: RobustMVOResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-ROBUST-MVO",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert RobustMVOResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "uncertainty_radius": rob.uncertainty_radius,
        "nominal_expected_return_annualised": rob.nominal_expected_return_annualised,
        "worst_case_expected_return_annualised": rob.worst_case_expected_return_annualised,
        "portfolio_volatility_annualised": rob.portfolio_volatility_annualised,
        "nominal_sharpe_annualised": rob.nominal_sharpe_annualised,
        "worst_case_sharpe_annualised": rob.worst_case_sharpe_annualised,
        "effective_n_positions": rob.effective_n_positions,
        "turnover_vs_prior": rob.turnover_vs_prior,
        "max_constraint_violation": rob.constraint_verification.max_violation,
    }
    for a, w in rob.weights.items():
        metrics[f"weights.{a}"] = w

    result = TestResult(
        test_id="portfolio.mean_variance.robust",
        test_name="Robust Mean-Variance Optimization (Ellipsoidal Uncertainty)",
        status=Status.RECORDED,
        params={"uncertainty_radius": rob.uncertainty_radius, "uncertainty_set": rob.uncertainty_set_type},
        metrics=metrics,
        interpretation=(
            f"Robust MVO solved at uncertainty radius kappa={rob.uncertainty_radius}. "
            f"Worst-case return: {rob.worst_case_expected_return_annualised:.2%}, "
            f"Volatility: {rob.portfolio_volatility_annualised:.2%}, "
            f"Effective positions: {rob.effective_n_positions:.2f}."
        ),
        limitations=[
            "Uncertainty set is ellipsoidal around sample expected return.",
            "Worst-case expected return is evaluated at the boundary of the uncertainty ellipsoid.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(rob.weights),
    )


def robust_sensitivity_to_evidence(
    sens: RobustSensitivityResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-ROBUST-MVO",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert RobustSensitivityResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "n_radii_evaluated": len(sens.radii_evaluated),
        "baseline_radius": sens.baseline_radius,
    }
    for pt in sens.points:
        r_str = f"{pt.uncertainty_radius:.2f}"
        metrics[f"volatility_annualised.r_{r_str}"] = pt.portfolio_volatility_annualised
        metrics[f"worst_case_return_annualised.r_{r_str}"] = pt.worst_case_expected_return_annualised
        metrics[f"effective_n.r_{r_str}"] = pt.effective_n_positions

    result = TestResult(
        test_id="portfolio.mean_variance.robust_sensitivity",
        test_name="Robust MVO Parameter Sensitivity Grid",
        status=Status.RECORDED,
        params={"radii": list(sens.radii_evaluated)},
        metrics=metrics,
        interpretation=(
            f"Evaluated robust optimization across {len(sens.radii_evaluated)} uncertainty radii. "
            f"Sensitivity path traces trade-off between robustness, conservatism, and concentration."
        ),
        limitations=[
            "Grid sensitivity is deterministic; selection of operating radius requires institutional policy.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict({"radii": list(sens.radii_evaluated)}),
    )


def cvar_to_evidence(
    cvar_res: CVaROptimizationResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-CVAR",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert CVaROptimizationResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "confidence_level": cvar_res.confidence_level,
        "cvar_at_scenario_horizon": cvar_res.cvar_at_scenario_horizon,
        "var_at_scenario_horizon": cvar_res.var_at_scenario_horizon,
        "cvar_periodic": cvar_res.cvar_at_scenario_horizon,
        "cvar_annualised": cvar_res.cvar_annualised,
        "var_auxiliary_periodic": cvar_res.var_at_scenario_horizon,
        "var_auxiliary_annualised": cvar_res.var_auxiliary_annualised,
        "scenario_horizon": cvar_res.scenario_horizon,
        "tail_scenario_count": cvar_res.tail_scenario_count,
        "n_scenarios": cvar_res.n_scenarios,
        "expected_return_periodic": cvar_res.expected_return_periodic,
        "expected_return_annualised": cvar_res.expected_return_annualised,
        "effective_n_positions": cvar_res.effective_n_positions,
        "turnover_vs_prior": cvar_res.turnover_vs_prior,
        "max_constraint_violation": cvar_res.constraint_verification.max_violation,
    }
    for a, w in cvar_res.weights.items():
        metrics[f"weights.{a}"] = w

    result = TestResult(
        test_id="portfolio.cvar_optimization",
        test_name="Rockafellar-Uryasev CVaR Linear Programming Optimization",
        status=Status.RECORDED,
        params={"confidence_level": cvar_res.confidence_level, "n_scenarios": cvar_res.n_scenarios},
        metrics=metrics,
        interpretation=(
            f"Minimum CVaR allocation at {cvar_res.confidence_level:.1%} confidence over {cvar_res.n_scenarios} scenarios. "
            f"Scenario-horizon CVaR: {cvar_res.cvar_at_scenario_horizon:.2%}, VaR: {cvar_res.var_at_scenario_horizon:.2%}, "
            f"Tail scenarios: {cvar_res.tail_scenario_count}."
        ),
        limitations=[
            "Non-parametric empirical scenarios; tail fidelity depends on historical crisis sample support.",
            "Scenario CVaR is evaluated directly at empirical scenario horizon without unverified time-scaling.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(cvar_res.weights),
    )


def herc_to_evidence(
    herc_res: HERCResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-HERC",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert HERCResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "n_assets": len(herc_res.weights),
        "effective_n_positions": herc_res.effective_n_positions,
        "portfolio_volatility_annualised": herc_res.portfolio_volatility_annualised,
        "portfolio_variance": herc_res.portfolio_variance,
        "risk_measure": herc_res.risk_measure,
        "max_constraint_violation": herc_res.constraint_verification.max_violation,
    }
    for a, w in herc_res.weights.items():
        metrics[f"weights.{a}"] = w
    for a, pcr in herc_res.percentage_risk_contributions.items():
        metrics[f"percentage_risk_contribution.{a}"] = pcr

    result = TestResult(
        test_id="portfolio.hierarchical_equal_risk_contribution",
        test_name="Hierarchical Equal Risk Contribution (HERC)",
        status=Status.RECORDED,
        params={"linkage_method": herc_res.tree_result.linkage_method, "risk_measure": herc_res.risk_measure},
        metrics=metrics,
        interpretation=(
            f"HERC allocation computed using {herc_res.tree_result.linkage_method} linkage across {len(herc_res.weights)} assets. "
            f"Annualized volatility: {herc_res.portfolio_volatility_annualised:.2%}, "
            f"Effective positions: {herc_res.effective_n_positions:.2f}."
        ),
        limitations=[
            "Hierarchical clustering relies on angular correlation distance.",
            "Equal risk contribution applies at dendrogram cluster partition level.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=herc_res.tree_result.covariance_fingerprint,
    )


def max_diversification_to_evidence(
    md: MaxDiversificationResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-MAXDIV",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert MaxDiversificationResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "diversification_ratio": md.diversification_ratio,
        "weighted_asset_volatility_annualised": md.weighted_asset_volatility_annualised,
        "portfolio_volatility_annualised": md.portfolio_volatility_annualised,
        "effective_n_positions": md.effective_n_positions,
        "max_constraint_violation": md.constraint_verification.max_violation,
    }
    for a, w in md.weights.items():
        metrics[f"weights.{a}"] = w

    result = TestResult(
        test_id="portfolio.maximum_diversification",
        test_name="Maximum Diversification Portfolio (Choueifaty & Coignard, 2008)",
        status=Status.RECORDED,
        params={},
        metrics=metrics,
        interpretation=(
            f"Maximum Diversification Portfolio solved. Diversification Ratio: {md.diversification_ratio:.4f}, "
            f"Portfolio volatility: {md.portfolio_volatility_annualised:.2%}, "
            f"Effective positions: {md.effective_n_positions:.2f}."
        ),
        limitations=[
            "Maximizes ratio of weighted individual asset volatilities to total portfolio volatility.",
            "Does not incorporate expected return estimates; sensitive to covariance estimation error.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(md.weights),
    )


def tracking_error_to_evidence(
    te: TrackingErrorResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-TRACKING-ERROR",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert TrackingErrorResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "tracking_error_periodic": te.tracking_error_periodic,
        "tracking_error_annualised": te.tracking_error_annualised,
        "active_return_annualised": te.active_return_annualised,
        "information_ratio": te.information_ratio,
        "portfolio_volatility_annualised": te.portfolio_volatility_annualised,
        "max_constraint_violation": te.constraint_verification.max_violation,
    }
    for a, w in te.weights.items():
        metrics[f"weights.{a}"] = w
    for a, bw in te.benchmark_weights.items():
        metrics[f"benchmark_weight.{a}"] = bw
    for a, aw in te.active_weights.items():
        metrics[f"active_weight.{a}"] = aw

    result = TestResult(
        test_id="portfolio.tracking_error_constrained",
        test_name="Tracking-Error Constrained Benchmark-Relative Portfolio Optimization",
        status=Status.RECORDED,
        params={},
        metrics=metrics,
        interpretation=(
            f"Benchmark-relative portfolio optimized under tracking error constraint. "
            f"Annualized tracking error: {te.tracking_error_annualised:.2%}, "
            f"Active return: {te.active_return_annualised if te.active_return_annualised else 0.0:.2%}, "
            f"Information Ratio: {te.information_ratio if te.information_ratio else 0.0:.4f}."
        ),
        limitations=[
            "Tracking error constraint is defined relative to the supplied benchmark weights vector.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(te.weights),
    )


def rebalance_decision_to_evidence(
    reb: RebalanceDecision,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-REBALANCE",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert RebalanceDecision into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "turnover": reb.turnover,
        "estimated_transaction_cost": reb.estimated_transaction_cost,
        "cost_provenance": reb.cost_provenance,
        "pre_trade_volatility_annualised": reb.pre_trade_risk.get("volatility_annualised"),
        "post_trade_volatility_annualised": reb.post_trade_risk.get("volatility_annualised"),
        "expected_return_gross_periodic": reb.expected_return_gross_periodic,
        "expected_return_gross_annualised": reb.expected_return_gross_annualised,
        "expected_return_net_periodic": reb.expected_return_net_periodic,
        "expected_return_net_annualised": reb.expected_return_net_annualised,
        "expected_return_gross": reb.expected_return_gross,
        "expected_return_net": reb.expected_return_net,
        "max_constraint_violation": reb.constraint_verification.max_violation,
    }
    for a, tw in reb.trade_weights.items():
        metrics[f"trade_weight.{a}"] = tw

    result = TestResult(
        test_id="portfolio.rebalance.decision",
        test_name="Institutional Rebalance Decision & Transaction Cost Analysis",
        status=Status.RECORDED,
        params={"turnover": reb.turnover, "estimated_cost": reb.estimated_transaction_cost},
        metrics=metrics,
        interpretation=(
            f"Rebalance decision analysis: one-way turnover {reb.turnover:.2%}, "
            f"estimated transaction cost {reb.estimated_transaction_cost:.4%}. "
            f"Post-trade volatility: {reb.post_trade_risk.get('volatility_annualised', 0.0):.2%}."
        ),
        limitations=[
            "Costs are evaluated against user-supplied cost schedule; zero broker connectivity.",
            "Decision support output only.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(reb.trade_weights),
    )


def constraint_verification_to_evidence(
    ver: ConstraintVerificationResult,
    run_id: str = "RUN-PORTFOLIO",
    model_id: str = "MOD-CONSTRAINTS",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert ConstraintVerificationResult into an EvidenceRecord."""
    metrics: dict[str, Any] = {
        "is_valid": ver.is_valid,
        "max_violation": ver.max_violation,
        "tolerance": ver.tolerance,
        "total_checks": ver.summary.get("total_checks", len(ver.violations)),
        "satisfied_checks": ver.summary.get("satisfied_checks", 0),
        "violated_checks": ver.summary.get("violated_checks", 0),
    }
    for v in ver.violations:
        metrics[f"violation.{v.constraint}"] = v.violation
        metrics[f"status.{v.constraint}"] = v.status

    result = TestResult(
        test_id="portfolio.constraints.verification",
        test_name="Independent Deterministic Constraint Verification",
        status=Status.RECORDED,
        params={"tolerance": ver.tolerance},
        metrics=metrics,
        interpretation=(
            f"Constraint verification audit: {ver.summary.get('satisfied_checks', 0)} of "
            f"{ver.summary.get('total_checks', 0)} checks satisfied. Max violation: {ver.max_violation:.8f}."
        ),
        limitations=[
            "Numerical tolerance is a solver precision parameter, not a governance threshold.",
        ],
    )
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(ver.summary),
    )


# --------------------------------------------------------------------------- #
# 11. Adversarial Challenge Diagnostic Evidence Bridge
# --------------------------------------------------------------------------- #
def challenge_result_to_diagnostic_evidence(
    tool_name: str,
    tool_res: Any,
    params: dict[str, Any],
    source_evidence_ids: tuple[str, ...],
    run_id: str = "RUN-CHALLENGE",
    model_id: str = "MOD-ADVERSARIAL",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Convert a deterministic challenge diagnostic tool execution into a new, unique EvidenceRecord."""
    diag_params = dict(params)
    diag_params["source_evidence_ids"] = list(source_evidence_ids)

    if isinstance(tool_res, BlackLittermanResult):
        test_id = "portfolio.adversarial.bl_tau_sensitivity"
        test_name = "Adversarial Black-Litterman Tau & Prior Sensitivity Diagnostic"
        metrics: dict[str, Any] = {
            "tau": tool_res.tau,
            "risk_aversion": tool_res.risk_aversion,
            "turnover_vs_prior": tool_res.turnover_vs_prior,
            "posterior_volatility_annualised": tool_res.posterior_volatility_annualised,
            "converged": tool_res.converged,
            "usable_solution": tool_res.usable_solution,
            "max_constraint_violation": tool_res.constraint_verification.max_violation,
        }
        for a, w in tool_res.posterior_weights.items():
            metrics[f"posterior_weight.{a}"] = w
        interpretation = (
            f"Adversarial BL diagnostic at tau={tool_res.tau}: posterior vol={tool_res.posterior_volatility_annualised:.2%}, "
            f"turnover={tool_res.turnover_vs_prior:.2%}."
        )
    elif isinstance(tool_res, RobustSensitivityResult):
        test_id = "portfolio.adversarial.robust_mvo_sensitivity"
        test_name = "Adversarial Robust MVO Radius Sensitivity & Concentration Diagnostic"
        metrics = {
            "radii_count": len(tool_res.points),
        }
        for i, pt in enumerate(tool_res.points):
            r_tag = f"r{i}"
            metrics[f"wc_return.{r_tag}"] = pt.worst_case_expected_return_annualised
            metrics[f"vol.{r_tag}"] = pt.portfolio_volatility_annualised
            metrics[f"eff_n.{r_tag}"] = pt.effective_n_positions
        interpretation = f"Adversarial Robust MVO sensitivity across {len(tool_res.points)} radii evaluated for concentration."
    elif isinstance(tool_res, CVaROptimizationResult):
        test_id = "portfolio.adversarial.cvar_tail_sensitivity"
        test_name = "Adversarial CVaR Scenario Tail Support Diagnostic"
        metrics = {
            "confidence_level": tool_res.confidence_level,
            "tail_scenario_count": tool_res.tail_scenario_count,
            "n_scenarios": tool_res.n_scenarios,
            "cvar_at_scenario_horizon": tool_res.cvar_at_scenario_horizon,
            "var_at_scenario_horizon": tool_res.var_at_scenario_horizon,
            "converged": tool_res.converged,
            "usable_solution": tool_res.usable_solution,
        }
        interpretation = (
            f"Adversarial CVaR tail diagnostic at {tool_res.confidence_level:.1%} confidence: "
            f"tail scenarios={tool_res.tail_scenario_count}/{tool_res.n_scenarios}."
        )
    elif isinstance(tool_res, dict):
        test_id = "portfolio.adversarial.cost_drag"
        test_name = "Adversarial Rebalance Turnover & Cost Drag Diagnostic"
        metrics = {
            str(k): float(v) if isinstance(v, (int, float, np.number)) else str(v)
            for k, v in tool_res.items()
            if isinstance(v, (int, float, str, bool, np.number))
        }
        interpretation = (
            f"Adversarial turnover & cost drag diagnostic: turnover={tool_res.get('turnover', 0.0):.2%}, "
            f"cost={tool_res.get('estimated_transaction_cost', 0.0):.4%}."
        )
    elif isinstance(tool_res, LinkageSensitivityResult):
        test_id = "portfolio.adversarial.linkage_sensitivity"
        test_name = "Adversarial Hierarchical Linkage Sensitivity Diagnostic"
        metrics = {
            "n_methods": len(tool_res.methods_compared),
            "methods_compared": ", ".join(tool_res.methods_compared),
        }
        for pair_k, max_d in tool_res.max_asset_weight_diffs.items():
            metrics[f"max_weight_diff.{pair_k}"] = max_d
        interpretation = (
            f"Adversarial linkage sensitivity across {len(tool_res.methods_compared)} linkage methods."
        )
    elif isinstance(tool_res, CovarianceDiagnostics):
        test_id = "portfolio.adversarial.cov_diagnostics"
        test_name = "Adversarial Covariance Condition & PSD Diagnostic"
        metrics = {
            "n_assets": tool_res.n_assets,
            "is_symmetric": tool_res.is_symmetric,
            "is_psd": tool_res.is_psd,
            "minimum_eigenvalue": tool_res.minimum_eigenvalue,
            "condition_number": tool_res.condition_number,
            "effective_rank": tool_res.effective_rank,
            "matrix_fingerprint": tool_res.matrix_fingerprint,
        }
        interpretation = (
            f"Adversarial Covariance Diagnostic: is_psd={tool_res.is_psd}, "
            f"min_eig={tool_res.minimum_eigenvalue:.6g}, condition={tool_res.condition_number:.4g}."
        )
    elif isinstance(tool_res, PSDRepairResult):
        test_id = "portfolio.adversarial.psd_repair_distortion"
        test_name = "Adversarial Covariance PSD Repair & Distortion Diagnostic"
        metrics = {
            "repair_method": str(tool_res.repair_method),
            "original_minimum_eigenvalue": tool_res.original_minimum_eigenvalue,
            "repaired_minimum_eigenvalue": tool_res.repaired_minimum_eigenvalue,
            "frobenius_distortion": tool_res.frobenius_distortion,
            "relative_frobenius_distortion": tool_res.relative_frobenius_distortion,
            "diagonal_preserved": tool_res.diagonal_preserved,
            "converged": tool_res.converged,
            "matrix_fingerprint_before": tool_res.matrix_fingerprint_before,
            "matrix_fingerprint_after": tool_res.matrix_fingerprint_after,
        }
        interpretation = (
            f"Adversarial PSD Repair: method={tool_res.repair_method}, rel_distortion={tool_res.relative_frobenius_distortion:.4%}, "
            f"repaired min eig={tool_res.repaired_minimum_eigenvalue:.6g}."
        )
    elif isinstance(tool_res, CovarianceComparisonResult):
        test_id = "portfolio.adversarial.cov_estimator_sensitivity"
        test_name = "Adversarial Covariance Estimator Sensitivity Diagnostic"
        metrics = {
            "estimators_count": len(tool_res.estimators_compared),
            "estimators_compared": ", ".join(tool_res.estimators_compared),
        }
        for pair, dist in tool_res.pairwise_frobenius_distances.items():
            metrics[f"frobenius_dist.{pair}"] = dist
        for est, vol in tool_res.portfolio_volatilities_annualised.items():
            metrics[f"portfolio_vol.{est}"] = vol
        interpretation = (
            f"Adversarial Estimator Sensitivity across {len(tool_res.estimators_compared)} estimators."
        )
    elif isinstance(tool_res, FactorRiskModelResult):
        test_id = "portfolio.adversarial.factor_model"
        test_name = "Adversarial Factor Risk Model Specification Diagnostic"
        metrics = {
            "n_assets": len(tool_res.asset_order),
            "n_factors": len(tool_res.factor_order),
            "is_psd": tool_res.diagnostics.is_psd,
            "condition_number": tool_res.diagnostics.condition_number,
            "reconstructed_covariance_fingerprint": tool_res.reconstructed_covariance_fingerprint,
        }
        interpretation = (
            f"Adversarial Factor Risk Model: {len(tool_res.asset_order)} assets, {len(tool_res.factor_order)} factors, "
            f"reconstructed is_psd={tool_res.diagnostics.is_psd}."
        )
    elif isinstance(tool_res, FactorRiskDecompositionResult):
        test_id = "portfolio.adversarial.factor_residual"
        test_name = "Adversarial Factor vs Specific Risk Residual Diagnostic"
        metrics = {
            "systematic_variance_share": tool_res.systematic_variance_share,
            "specific_variance_share": tool_res.specific_variance_share,
            "portfolio_volatility_annualised": tool_res.portfolio_volatility_annualised,
            "euler_reconciliation_error": tool_res.euler_reconciliation_error,
            "total_reconciliation_error": tool_res.total_reconciliation_error,
        }
        for f, share in tool_res.factor_variance_shares.items():
            metrics[f"factor_share.{f}"] = share
        interpretation = (
            f"Adversarial Factor Risk Decomposition: systematic share={tool_res.systematic_variance_share:.1%}, "
            f"specific share={tool_res.specific_variance_share:.1%}, vol={tool_res.portfolio_volatility_annualised:.2%}."
        )
    elif isinstance(tool_res, ActiveRiskDecompositionResult):
        test_id = "portfolio.adversarial.active_risk_driver"
        test_name = "Adversarial Active Risk (Tracking Error) Component Diagnostic"
        metrics = {
            "tracking_error_annualised": tool_res.tracking_error_annualised,
            "factor_active_share": tool_res.factor_active_share,
            "specific_active_share": tool_res.specific_active_share,
            "reconciliation_error": tool_res.reconciliation_error,
        }
        for f, exp in tool_res.active_factor_exposures.items():
            metrics[f"active_exposure.{f}"] = exp
        interpretation = (
            f"Adversarial Active Risk: TE={tool_res.tracking_error_annualised:.2%}, "
            f"factor active share={tool_res.factor_active_share:.1%}, specific active share={tool_res.specific_active_share:.1%}."
        )
    elif isinstance(tool_res, FactorReturnAttributionResult):
        test_id = "portfolio.adversarial.attribution_reconciliation"
        test_name = "Adversarial Factor Return Attribution Reconciliation Diagnostic"
        metrics = {
            "n_periods": tool_res.n_periods,
            "total_portfolio_return": tool_res.total_portfolio_return,
            "total_factor_contribution": tool_res.total_factor_contribution,
            "total_specific_contribution": tool_res.total_specific_contribution,
            "max_abs_reconciliation_error": tool_res.max_abs_reconciliation_error,
            "is_reconciled": tool_res.is_reconciled,
        }
        interpretation = (
            f"Adversarial Return Attribution over {tool_res.n_periods} periods: total return={tool_res.total_portfolio_return:.4f}, "
            f"factor={tool_res.total_factor_contribution:.4f}, specific={tool_res.total_specific_contribution:.4f}, max error={tool_res.max_abs_reconciliation_error:.3g}."
        )
    elif isinstance(tool_res, BrinsonAttributionResult):
        test_id = "portfolio.adversarial.brinson_stability"
        test_name = "Adversarial Brinson-Fachler Performance Attribution Diagnostic"
        metrics = {
            "total_active_return": tool_res.total_active_return,
            "total_allocation_effect": tool_res.total_allocation_effect,
            "total_selection_effect": tool_res.total_selection_effect,
            "total_interaction_effect": tool_res.total_interaction_effect,
            "reconciliation_error": tool_res.reconciliation_error,
            "is_reconciled": tool_res.is_reconciled,
        }
        interpretation = (
            f"Adversarial Brinson Attribution: active return={tool_res.total_active_return:.4f}, "
            f"allocation={tool_res.total_allocation_effect:.4f}, selection={tool_res.total_selection_effect:.4f}, interaction={tool_res.total_interaction_effect:.4f}."
        )
    elif isinstance(tool_res, CarinoLinkedAttributionResult):
        test_id = "portfolio.adversarial.carino_linking"
        test_name = "Adversarial Carino Multi-Period Linking Diagnostic"
        metrics = {
            "n_periods": tool_res.n_periods,
            "total_active_return_geometric": tool_res.total_active_return_geometric,
            "total_linked_allocation": tool_res.total_linked_allocation,
            "total_linked_selection": tool_res.total_linked_selection,
            "total_linked_interaction": tool_res.total_linked_interaction,
            "reconciliation_error": tool_res.reconciliation_error,
            "is_reconciled": tool_res.is_reconciled,
        }
        interpretation = (
            f"Adversarial Carino Linking over {tool_res.n_periods} periods: linked active return={tool_res.total_active_return_geometric:.4f}, "
            f"reconciliation error={tool_res.reconciliation_error:.3g}."
        )
    elif isinstance(tool_res, FactorDataIntegrityResult):
        test_id = "portfolio.adversarial.factor_data_integrity"
        test_name = "Adversarial Factor Model Data Integrity Diagnostic"
        metrics = {
            "is_valid": tool_res.is_valid,
            "n_assets": tool_res.n_assets,
            "n_factors": tool_res.n_factors,
            "missing_exposure_count": tool_res.missing_exposure_count,
            "missing_factor_return_count": tool_res.missing_factor_return_count,
            "missing_specific_variance_count": tool_res.missing_specific_variance_count,
            "issues_count": len(tool_res.issues),
        }
        interpretation = (
            f"Adversarial Factor Data Integrity: is_valid={tool_res.is_valid}, issues={len(tool_res.issues)}."
        )
    elif isinstance(tool_res, ScenarioResult):
        test_id = f"scenario.adversarial.{tool_res.repricing_method.lower()}"
        test_name = f"Adversarial Scenario Repricing Diagnostic: {tool_res.repricing_method}"
        metrics = {
            "scenario_id": tool_res.scenario_id,
            "scenario_type": tool_res.scenario_type,
            "repricing_method": tool_res.repricing_method,
            "scenario_return": tool_res.scenario_return,
            "scenario_loss": tool_res.scenario_loss,
            "scenario_pnl": tool_res.scenario_pnl,
            "reconciliation_error": tool_res.reconciliation_error,
        }
        interpretation = (
            f"Adversarial Scenario Repricing ({tool_res.repricing_method}): return={tool_res.scenario_return:.4f}, "
            f"canonical loss={tool_res.scenario_loss:.4f}, recon error={tool_res.reconciliation_error:.3g}."
        )
    elif isinstance(tool_res, ActiveScenarioResult):
        test_id = "scenario.adversarial.active_stress"
        test_name = "Adversarial Active Stress Diagnostic"
        metrics = {
            "portfolio_return": tool_res.portfolio_return,
            "benchmark_return": tool_res.benchmark_return,
            "active_return": tool_res.active_return,
            "reconciliation_error": tool_res.reconciliation_error,
        }
        interpretation = (
            f"Adversarial Active Stress: portfolio={tool_res.portfolio_return:.4f}, "
            f"benchmark={tool_res.benchmark_return:.4f}, active={tool_res.active_return:.4f}."
        )
    elif isinstance(tool_res, ScenarioSetResult):
        test_id = "scenario.adversarial.set_comparison"
        test_name = "Adversarial Multi-Scenario Set Ranking Diagnostic"
        metrics = {
            "n_scenarios": len(tool_res.scenarios_evaluated),
            "worst_scenario_id": tool_res.worst_scenario_id,
            "worst_scenario_loss": tool_res.worst_scenario_loss,
            "best_scenario_id": tool_res.best_scenario_id,
            "best_scenario_loss": tool_res.best_scenario_loss,
            "comparability_valid": tool_res.comparability_valid,
        }
        interpretation = (
            f"Adversarial Multi-Scenario Ranking: worst='{tool_res.worst_scenario_id}' (loss={tool_res.worst_scenario_loss:.4f}), "
            f"best='{tool_res.best_scenario_id}' (loss={tool_res.best_scenario_loss:.4f})."
        )
    elif isinstance(tool_res, ReverseStressResult):
        test_id = "scenario.adversarial.reverse_stress"
        test_name = "Adversarial Reverse Stress Diagnostic"
        metrics = {
            "target_loss": tool_res.target_loss,
            "achieved_loss": tool_res.achieved_loss,
            "loss_gap": tool_res.loss_gap,
            "distance": tool_res.distance,
            "distance_norm": tool_res.distance_norm,
            "bounds_satisfied": tool_res.bounds_satisfied,
            "solver_status": tool_res.solver_status,
            "converged": tool_res.converged,
        }
        interpretation = (
            f"Adversarial Reverse Stress ({tool_res.distance_norm}): target={tool_res.target_loss:.4f}, "
            f"achieved={tool_res.achieved_loss:.4f}, distance={tool_res.distance:.4f}, status={tool_res.solver_status}."
        )
    elif isinstance(tool_res, ScenarioDataIntegrityResult):
        test_id = "scenario.adversarial.data_integrity"
        test_name = "Adversarial Scenario Data Integrity Diagnostic"
        metrics = {
            "valid": tool_res.valid,
            "n_shocks": tool_res.n_shocks,
            "repricing_compatible": tool_res.repricing_compatible,
            "sensitivities_complete": tool_res.sensitivities_complete,
            "coverage_complete": tool_res.coverage_complete,
            "provenance_valid": tool_res.provenance_valid,
            "n_issues": len(tool_res.issues),
        }
        interpretation = (
            f"Adversarial Scenario Data Integrity: valid={tool_res.valid}, issues={len(tool_res.issues)}."
        )
    else:
        test_id = f"portfolio.adversarial.{tool_name}"
        test_name = f"Adversarial Diagnostic: {tool_name}"
        metrics = {"result_type": type(tool_res).__name__}
        interpretation = f"Adversarial diagnostic executed via {tool_name}."

    result = TestResult(
        test_id=test_id,
        test_name=test_name,
        status=Status.RECORDED,
        params=diag_params,
        metrics=metrics,
        interpretation=interpretation,
        limitations=[
            "Deterministic diagnostic executed in response to adversarial challenge.",
            "Materiality requires explicit threshold adjudication under governance policy.",
        ],
    )
    artifact_hash = _hash_dict({"tool": tool_name, "params": diag_params, "sources": source_evidence_ids})
    return EvidenceRecord.from_result(
        result,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=artifact_hash,
    )


# =========================================================================== #
# GATE 4: SUBORDINATE EVIDENCE RECORD ADAPTERS
# =========================================================================== #
def covariance_diagnostics_to_evidence(
    diag: CovarianceDiagnostics,
    run_id: str = "RUN-COV",
    model_id: str = "MOD-COV-DIAG",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap CovarianceDiagnostics in an EvidenceRecord."""
    params = {"n_assets": diag.n_assets}
    metrics: dict[str, bool | float | int | str | None] = {
        "n_assets": diag.n_assets,
        "is_symmetric": diag.is_symmetric,
        "symmetry_error": diag.symmetry_error,
        "is_psd": diag.is_psd,
        "minimum_eigenvalue": diag.minimum_eigenvalue,
        "maximum_eigenvalue": diag.maximum_eigenvalue,
        "rank": diag.rank,
        "numerical_rank": diag.numerical_rank,
        "condition_number": diag.condition_number,
        "trace": diag.trace,
        "effective_rank": diag.effective_rank,
        "largest_eigenvalue_share": diag.largest_eigenvalue_share,
        "diagonal_positive": diag.diagonal_positive,
        "valid_correlation_conversion": diag.valid_correlation_conversion,
        "matrix_fingerprint": diag.matrix_fingerprint,
    }
    if diag.log_determinant is not None:
        metrics["log_determinant"] = diag.log_determinant

    res = TestResult(
        test_id="covariance.diagnostics",
        test_name="Covariance Structural & Spectral Diagnostics",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Covariance Matrix ({diag.n_assets}x{diag.n_assets}): is_psd={diag.is_psd}, "
            f"min_eig={diag.minimum_eigenvalue:.6g}, condition={diag.condition_number:.4g}, "
            f"effective_rank={diag.effective_rank:.2f}."
        ),
        limitations=[
            "Descriptive spectral diagnostics; no automatic acceptance threshold applied.",
            "Effective rank computed via entropy of normalized positive eigenvalues.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=diag.matrix_fingerprint,
    )


def psd_repair_to_evidence(
    repair: PSDRepairResult,
    run_id: str = "RUN-COV",
    model_id: str = "MOD-PSD-REPAIR",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap PSDRepairResult in an EvidenceRecord preserving original vs repaired matrix fingerprints and distortion."""
    params = {
        "repair_method": str(repair.repair_method),
        "intervention_reason": repair.intervention_reason,
    }
    metrics: dict[str, bool | float | int | str | None] = {
        "repair_method": str(repair.repair_method),
        "original_minimum_eigenvalue": repair.original_minimum_eigenvalue,
        "repaired_minimum_eigenvalue": repair.repaired_minimum_eigenvalue,
        "frobenius_distortion": repair.frobenius_distortion,
        "relative_frobenius_distortion": repair.relative_frobenius_distortion,
        "maximum_element_change": repair.maximum_element_change,
        "diagonal_preserved": repair.diagonal_preserved,
        "iterations_used": repair.iterations_used,
        "converged": repair.converged,
        "matrix_fingerprint_before": repair.matrix_fingerprint_before,
        "matrix_fingerprint_after": repair.matrix_fingerprint_after,
    }
    res = TestResult(
        test_id="covariance.psd_repair",
        test_name="Covariance Numerical PSD Repair Intervention",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Numerical PSD Repair via {repair.repair_method}: original min eig={repair.original_minimum_eigenvalue:.6g} -> "
            f"repaired min eig={repair.repaired_minimum_eigenvalue:.6g} with relative Frobenius distortion {repair.relative_frobenius_distortion:.4%}."
        ),
        limitations=[
            "PSD_REPAIRED != MODEL_VALID: Numerical intervention applied to satisfy positive semi-definiteness.",
            "Material distortion must be audited by institutional model governance.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(
            {"before": repair.matrix_fingerprint_before, "after": repair.matrix_fingerprint_after}
        ),
    )


def covariance_comparison_to_evidence(
    comp: CovarianceComparisonResult,
    run_id: str = "RUN-COV-COMP",
    model_id: str = "MOD-COV-MULTI",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap CovarianceComparisonResult into a structured EvidenceRecord."""
    params = {
        "estimators_compared": list(comp.estimators_compared),
        "asset_order": list(comp.asset_order),
    }
    metrics: dict[str, bool | float | int | str | None] = {
        "n_estimators": len(comp.estimators_compared),
        "n_assets": len(comp.asset_order),
    }
    for k, v in comp.pairwise_frobenius_distances.items():
        metrics[f"frob_dist_{k}"] = v
    for k, v in comp.pairwise_spectral_distances.items():
        metrics[f"spec_dist_{k}"] = v
    for est, vol in comp.portfolio_volatilities_annualised.items():
        metrics[f"port_vol_{est}"] = vol

    res = TestResult(
        test_id="covariance.model_comparison",
        test_name="Multi-Estimator Covariance Comparison & Volatility Impact",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Compared {len(comp.estimators_compared)} covariance estimators across {len(comp.asset_order)} assets."
        ),
        limitations=[
            "Comparative metric evaluation only; no single estimator is declared optimal without an explicit model risk policy.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(params),
    )


def factor_risk_model_to_evidence(
    frm: FactorRiskModelResult,
    run_id: str = "RUN-FACTOR",
    model_id: str = "MOD-FACTOR-RISK",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap FactorRiskModelResult in an EvidenceRecord."""
    params = {
        "n_assets": len(frm.asset_order),
        "n_factors": len(frm.factor_order),
        "time_alignment": frm.time_alignment,
        "periods_per_year": frm.periods_per_year,
    }
    metrics: dict[str, bool | float | int | str | None] = {
        "n_assets": len(frm.asset_order),
        "n_factors": len(frm.factor_order),
        "is_psd": frm.diagnostics.is_psd,
        "condition_number": frm.diagnostics.condition_number,
        "effective_rank": frm.diagnostics.effective_rank,
        "exposure_fingerprint": frm.exposure_fingerprint,
        "factor_covariance_fingerprint": frm.factor_covariance_fingerprint,
        "specific_variance_fingerprint": frm.specific_variance_fingerprint,
        "reconstructed_covariance_fingerprint": frm.reconstructed_covariance_fingerprint,
    }
    res = TestResult(
        test_id="factor_risk.model",
        test_name="Linear Factor Risk Model Specification & Reconstructed Covariance",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Linear Factor Model: {len(frm.asset_order)} assets x {len(frm.factor_order)} factors; "
            f"reconstructed asset covariance is_psd={frm.diagnostics.is_psd}, condition={frm.diagnostics.condition_number:.4g}."
        ),
        limitations=[
            "Linear factor model formulation Sigma = B F B' + D with diagonal specific variance.",
            "All exposures and factor returns are aligned fail-closed.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=frm.reconstructed_covariance_fingerprint,
    )


def factor_risk_decomp_to_evidence(
    frd: FactorRiskDecompositionResult,
    run_id: str = "RUN-FACTOR",
    model_id: str = "MOD-FACTOR-DECOMP",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap FactorRiskDecompositionResult in an EvidenceRecord."""
    params = {"periods_per_year": frd.periods_per_year}
    metrics: dict[str, Any] = {
        "portfolio_volatility_annualised": frd.portfolio_volatility_annualised,
        "systematic_volatility_annualised": frd.systematic_volatility_annualised,
        "specific_volatility_annualised": frd.specific_volatility_annualised,
        "systematic_variance_share": frd.systematic_variance_share,
        "specific_variance_share": frd.specific_variance_share,
        "euler_reconciliation_error": frd.euler_reconciliation_error,
        "total_reconciliation_error": frd.total_reconciliation_error,
    }
    for f, exp in frd.portfolio_factor_exposures.items():
        metrics[f"factor_exposure.{f}"] = exp
    for f, share in frd.factor_variance_shares.items():
        metrics[f"factor_variance_share.{f}"] = share

    res = TestResult(
        test_id="factor_risk.decomposition",
        test_name="Euler Factor Risk Variance Decomposition",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Factor Risk Decomposition: Total Vol={frd.portfolio_volatility_annualised:.2%}, "
            f"Systematic Share={frd.systematic_variance_share:.1%}, Specific Share={frd.specific_variance_share:.1%}."
        ),
        limitations=[
            "Euler-consistent variance decomposition: component variances sum exactly to systematic variance.",
            "Specific risk assumes diagonal idiosyncratic structure.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(metrics),
    )


def active_risk_decomp_to_evidence(
    ard: ActiveRiskDecompositionResult,
    run_id: str = "RUN-FACTOR",
    model_id: str = "MOD-ACTIVE-RISK",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap ActiveRiskDecompositionResult in an EvidenceRecord."""
    params = {"periods_per_year": ard.periods_per_year}
    metrics: dict[str, Any] = {
        "tracking_error_annualised": ard.tracking_error_annualised,
        "factor_active_share": ard.factor_active_share,
        "specific_active_share": ard.specific_active_share,
        "reconciliation_error": ard.reconciliation_error,
    }
    for f, exp in ard.active_factor_exposures.items():
        metrics[f"active_factor_exposure.{f}"] = exp

    res = TestResult(
        test_id="factor_risk.active_decomposition",
        test_name="Benchmark-Relative Active Risk & Tracking Error Decomposition",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Active Risk Decomposition: Tracking Error={ard.tracking_error_annualised:.2%}, "
            f"Factor Active Share={ard.factor_active_share:.1%}, Specific Active Share={ard.specific_active_share:.1%}."
        ),
        limitations=[
            "Benchmark-relative tracking error decomposed into factor active and specific active variances.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(metrics),
    )


def factor_return_attribution_to_evidence(
    fra: FactorReturnAttributionResult,
    run_id: str = "RUN-ATTRIB",
    model_id: str = "MOD-FACTOR-ATTRIB",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap FactorReturnAttributionResult in an EvidenceRecord."""
    params = {
        "n_periods": fra.n_periods,
        "time_alignment_convention": fra.time_alignment_convention,
    }
    metrics: dict[str, Any] = {
        "n_periods": fra.n_periods,
        "total_portfolio_return": fra.total_portfolio_return,
        "total_factor_contribution": fra.total_factor_contribution,
        "total_specific_contribution": fra.total_specific_contribution,
        "max_abs_reconciliation_error": fra.max_abs_reconciliation_error,
        "mean_abs_reconciliation_error": fra.mean_abs_reconciliation_error,
        "is_reconciled": fra.is_reconciled,
    }
    for f, contrib in fra.cumulative_factor_contributions.items():
        metrics[f"cumulative_factor_contrib.{f}"] = contrib

    res = TestResult(
        test_id="attribution.factor_performance",
        test_name="Factor Return Performance Attribution",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Factor Return Attribution over {fra.n_periods} periods: Portfolio Return={fra.total_portfolio_return:.4f}, "
            f"Factor Contribution={fra.total_factor_contribution:.4f}, Specific Contribution={fra.total_specific_contribution:.4f}."
        ),
        limitations=[
            "Period factor returns attributed using beginning-of-period exposures.",
            "Reconciliation error reported without hidden residual adjustment.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(metrics),
    )


def brinson_to_evidence(
    ba: BrinsonAttributionResult,
    run_id: str = "RUN-ATTRIB",
    model_id: str = "MOD-BRINSON",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap BrinsonAttributionResult in an EvidenceRecord."""
    params = {"convention": ba.convention, "group_count": len(ba.group_names)}
    metrics: dict[str, Any] = {
        "total_portfolio_return": ba.total_portfolio_return,
        "total_benchmark_return": ba.total_benchmark_return,
        "total_active_return": ba.total_active_return,
        "total_allocation_effect": ba.total_allocation_effect,
        "total_selection_effect": ba.total_selection_effect,
        "total_interaction_effect": ba.total_interaction_effect,
        "reconciliation_error": ba.reconciliation_error,
        "is_reconciled": ba.is_reconciled,
    }
    for g in ba.group_names:
        metrics[f"allocation.{g}"] = ba.allocation_effects.get(g, 0.0)
        metrics[f"selection.{g}"] = ba.selection_effects.get(g, 0.0)
        metrics[f"interaction.{g}"] = ba.interaction_effects.get(g, 0.0)

    res = TestResult(
        test_id="attribution.brinson",
        test_name="Brinson-Fachler Active Return Performance Attribution",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Brinson-Fachler Attribution: Active Return={ba.total_active_return:.4%}, "
            f"Allocation={ba.total_allocation_effect:.4%}, Selection={ba.total_selection_effect:.4%}, Interaction={ba.total_interaction_effect:.4%}."
        ),
        limitations=[
            "Brinson-Fachler single-period active attribution formulation.",
            "Reconciliation error verified algebraically.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(metrics),
    )


def carino_to_evidence(
    ca: CarinoLinkedAttributionResult,
    run_id: str = "RUN-ATTRIB",
    model_id: str = "MOD-CARINO",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap CarinoLinkedAttributionResult in an EvidenceRecord."""
    params = {"n_periods": ca.n_periods, "linking_method": "Carino_logarithmic"}
    metrics: dict[str, Any] = {
        "n_periods": ca.n_periods,
        "total_portfolio_return_geometric": ca.total_portfolio_return_geometric,
        "total_benchmark_return_geometric": ca.total_benchmark_return_geometric,
        "total_active_return_geometric": ca.total_active_return_geometric,
        "total_linked_allocation": ca.total_linked_allocation,
        "total_linked_selection": ca.total_linked_selection,
        "total_linked_interaction": ca.total_linked_interaction,
        "reconciliation_error": ca.reconciliation_error,
        "is_reconciled": ca.is_reconciled,
    }
    for g in ca.group_names:
        metrics[f"linked_allocation.{g}"] = ca.linked_allocation_effects.get(g, 0.0)
        metrics[f"linked_selection.{g}"] = ca.linked_selection_effects.get(g, 0.0)
        metrics[f"linked_interaction.{g}"] = ca.linked_interaction_effects.get(g, 0.0)

    res = TestResult(
        test_id="attribution.multi_period_linking",
        test_name="Carino Logarithmic Multi-Period Attribution Linking",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Carino Multi-Period Linking ({ca.n_periods} periods): Total Active Geometric Return={ca.total_active_return_geometric:.4%}, "
            f"Linked Allocation={ca.total_linked_allocation:.4%}, Linked Selection={ca.total_linked_selection:.4%}, Linked Interaction={ca.total_linked_interaction:.4%}."
        ),
        limitations=[
            "Exact Carino (1999) logarithmic smoothing geometric active return linking.",
            "Analytically handles R_p -> R_b limit without heuristic approximations.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(metrics),
    )


def factor_data_integrity_to_evidence(
    fdi: FactorDataIntegrityResult,
    run_id: str = "RUN-DATA",
    model_id: str = "MOD-FACTOR-INTEGRITY",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap FactorDataIntegrityResult in an EvidenceRecord."""
    params = {"n_assets": fdi.n_assets, "n_factors": fdi.n_factors}
    metrics: dict[str, bool | float | int | str | None] = {
        "is_valid": fdi.is_valid,
        "n_assets": fdi.n_assets,
        "n_factors": fdi.n_factors,
        "has_duplicate_assets": fdi.has_duplicate_assets,
        "has_duplicate_factors": fdi.has_duplicate_factors,
        "missing_exposure_count": fdi.missing_exposure_count,
        "missing_factor_return_count": fdi.missing_factor_return_count,
        "missing_specific_variance_count": fdi.missing_specific_variance_count,
        "has_lookahead_violation": fdi.has_lookahead_violation,
        "issues_count": len(fdi.issues),
    }
    res = TestResult(
        test_id="factor_risk.data_integrity",
        test_name="Factor Model Data Coverage & Alignment Integrity",
        status=Status.RECORDED if fdi.is_valid else Status.FAIL,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Factor Data Integrity: is_valid={fdi.is_valid}, {len(fdi.issues)} issue(s) detected."
        ),
        limitations=[
            "Deterministic pre-flight integrity check on factor exposures, returns, and variances.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(metrics),
    )


# =========================================================================== #
# GATE 5: TAIL RISK, EXPECTED SHORTFALL & BACKTESTING EVIDENCE BRIDGES
# =========================================================================== #
def tail_risk_estimate_to_evidence(
    estimate: TailRiskEstimate,
    run_id: str = "RUN-TAIL",
    model_id: str = "MOD-TAIL-RISK",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap TailRiskEstimate into a structured EvidenceRecord."""
    params = {
        "method": str(estimate.method),
        "confidence": estimate.confidence,
        "sign_convention": str(estimate.sign_convention),
        "quantile_method": estimate.quantile_method,
        "horizon": str(estimate.horizon),
    }
    metrics: dict[str, bool | float | int | str | None] = {
        "method": str(estimate.method),
        "confidence": estimate.confidence,
        "var": estimate.var,
        "es": estimate.es,
        "n_observations": estimate.n_observations,
        "tail_observations_count": estimate.tail_observations_count,
        "tail_fraction": estimate.tail_fraction,
        "boundary_weight": estimate.boundary_weight,
        "converged": estimate.converged,
        "data_fingerprint": estimate.data_fingerprint,
    }
    res = TestResult(
        test_id="traded_risk.expected_shortfall",
        test_name="Expected Shortfall & VaR Tail Risk Estimation",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"{estimate.method.upper()} @ {estimate.confidence:.1%} confidence: "
            f"VaR = {estimate.var:.4f}, Expected Shortfall = {estimate.es:.4f} "
            f"over {estimate.n_observations} observations."
        ),
        limitations=list(estimate.limitations),
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=estimate.data_fingerprint,
    )


def tail_backtest_to_evidence(
    backtest: TailBacktestResult,
    run_id: str = "RUN-BACKTEST",
    model_id: str = "MOD-TAIL-BACKTEST",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap TailBacktestResult into a structured EvidenceRecord."""
    params = {
        "pnl_source": backtest.pnl_source,
        "var_confidence": backtest.var_confidence,
        "test_significance": backtest.test_significance,
        "gamma_test": backtest.test_significance,
        "alpha_var": backtest.expected_probability,
    }
    metrics: dict[str, bool | float | int | str | None] = {
        "var_confidence": backtest.var_confidence,
        "test_significance": backtest.test_significance,
        "gamma_test": backtest.test_significance,
        "statistical_gamma_test": backtest.test_significance,
        "statistical_criterion_source": "PRE_REGISTERED_VALIDATION",
        "alpha_var": backtest.expected_probability,
        "n_observations": backtest.n_observations,
        "n_exceptions": backtest.n_exceptions,
        "exception_rate": backtest.exception_rate,
        "expected_probability": backtest.expected_probability,
        "expected_exceptions": backtest.expected_exceptions,
        "kupiec_lr": backtest.kupiec_lr,
        "kupiec_p_value": backtest.kupiec_p_value,
        "kupiec_rejected": backtest.kupiec_rejected,
        "kupiec_estimable": backtest.kupiec_estimable,
        "christoffersen_lr": backtest.christoffersen_lr,
        "christoffersen_p_value": backtest.christoffersen_p_value,
        "christoffersen_rejected": backtest.christoffersen_rejected,
        "christoffersen_estimable": backtest.christoffersen_estimable,
        "conditional_coverage_lr": backtest.conditional_coverage_lr,
        "conditional_coverage_p_value": backtest.conditional_coverage_p_value,
        "conditional_coverage_rejected": backtest.conditional_coverage_rejected,
        "conditional_coverage_estimable": backtest.conditional_coverage_estimable,
        "n00": backtest.transition_counts[0],
        "n01": backtest.transition_counts[1],
        "n10": backtest.transition_counts[2],
        "n11": backtest.transition_counts[3],
        "pi_01": backtest.pi_01,
        "pi_11": backtest.pi_11,
        "has_zero_transition_cell": backtest.has_zero_transition_cell,
        "indicator_hash": backtest.indicator_hash,
    }
    status = (
        Status.FAIL
        if (
            backtest.kupiec_rejected
            or backtest.christoffersen_rejected
            or backtest.conditional_coverage_rejected
        )
        else Status.RECORDED
    )
    res = TestResult(
        test_id="traded_risk.var_conditional_coverage",
        test_name="Out-of-Sample VaR Coverage & Independence Backtest",
        status=status,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Backtest @ {backtest.var_confidence:.1%} VaR confidence, {backtest.test_significance:.1%} significance: "
            f"{backtest.n_exceptions} exceptions in {backtest.n_observations} obs. "
            f"Kupiec: {'REJECT' if backtest.kupiec_rejected else 'DO NOT REJECT'} (p={backtest.kupiec_p_value:.4f}); "
            f"Christoffersen Ind: {'REJECT' if backtest.christoffersen_rejected else 'DO NOT REJECT'} (p={backtest.christoffersen_p_value:.4f}); "
            f"Joint CC: {'REJECT' if backtest.conditional_coverage_rejected else 'DO NOT REJECT'} (p={backtest.conditional_coverage_p_value:.4f})."
        ),
        limitations=list(backtest.limitations),
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=backtest.data_fingerprint,
    )


def duration_diagnostics_to_evidence(
    durations: DurationDiagnosticsResult,
    run_id: str = "RUN-DUR",
    model_id: str = "MOD-DURATION",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap DurationDiagnosticsResult into an EvidenceRecord."""
    params = {"n_durations": durations.n_durations}
    metrics: dict[str, bool | float | int | str | None] = {
        "n_durations": durations.n_durations,
        "mean_duration": durations.mean_duration,
        "median_duration": durations.median_duration,
        "min_duration": durations.min_duration,
        "max_duration": durations.max_duration,
        "duration_std": durations.duration_std,
        "max_run_length": durations.max_run_length,
    }
    res = TestResult(
        test_id="traded_risk.exception_durations",
        test_name="Inter-Exception Duration & Clustering Diagnostics",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Duration Diagnostics: {durations.n_durations} inter-exception intervals. "
            f"Mean duration = {durations.mean_duration:.1f} days, max streak = {durations.max_run_length} day(s)."
        ),
        limitations=[
            "Descriptive inter-exception interval analysis.",
            "Formal Haas duration hypothesis testing is formally deferred.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=durations.data_fingerprint,
    )


def tail_severity_to_evidence(
    severity: TailSeverityResult,
    run_id: str = "RUN-SEV",
    model_id: str = "MOD-SEVERITY",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap TailSeverityResult into an EvidenceRecord."""
    params = {"n_exceptions": severity.n_exceptions}
    metrics: dict[str, bool | float | int | str | None] = {
        "n_exceptions": severity.n_exceptions,
        "mean_absolute_exceedance": severity.mean_absolute_exceedance,
        "median_absolute_exceedance": severity.median_absolute_exceedance,
        "max_absolute_exceedance": severity.max_absolute_exceedance,
        "total_tail_exceedance_loss": severity.total_tail_exceedance_loss,
        "mean_normalized_exceedance": severity.mean_normalized_exceedance,
        "max_normalized_exceedance": severity.max_normalized_exceedance,
        "mean_relative_exceedance": severity.mean_relative_exceedance,
        "max_relative_exceedance": severity.max_relative_exceedance,
    }
    res = TestResult(
        test_id="traded_risk.tail_severity",
        test_name="Tail Exceedance Loss Severity Diagnostics",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Tail Severity: {severity.n_exceptions} exceptions. Mean exceedance = {severity.mean_absolute_exceedance:.4f}, "
            f"max exceedance = {severity.max_absolute_exceedance:.4f}, total excess loss = {severity.total_tail_exceedance_loss:.4f}."
        ),
        limitations=[
            "Descriptive tail exceedance magnitude analysis.",
            "No arbitrary pass/fail severity threshold applied.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=severity.data_fingerprint,
    )


def tail_contribution_to_evidence(
    contrib: TailRiskContributionResult,
    run_id: str = "RUN-CONTRIB",
    model_id: str = "MOD-TAIL-CONTRIB",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap TailRiskContributionResult into an EvidenceRecord."""
    params = {
        "method": contrib.method,
        "confidence": contrib.confidence,
    }
    metrics: dict[str, bool | float | int | str | None] = {
        "method": contrib.method,
        "confidence": contrib.confidence,
        "portfolio_var": contrib.portfolio_var,
        "portfolio_es": contrib.portfolio_es,
        "var_reconciliation_error": contrib.var_reconciliation_error,
        "es_reconciliation_error": contrib.es_reconciliation_error,
        "n_assets": len(contrib.component_es),
    }
    for a, c_es in contrib.component_es.items():
        metrics[f"component_es_{a}"] = c_es
    for a, c_var in contrib.component_var.items():
        metrics[f"component_var_{a}"] = c_var

    res = TestResult(
        test_id="traded_risk.es_contribution",
        test_name="Component Tail Risk & Expected Shortfall Contribution",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Tail Risk Contribution ({contrib.method}): Portfolio ES = {contrib.portfolio_es:.4f}, "
            f"reconciliation residual = {contrib.es_reconciliation_error:.2e} across {len(contrib.component_es)} assets."
        ),
        limitations=[
            "Euler component risk decomposition reconciling to portfolio risk.",
            "Historical VaR component decomposition is non-smooth and formally deferred.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=contrib.data_fingerprint,
    )


def tail_comparison_to_evidence(
    comp: TailModelComparisonResult,
    run_id: str = "RUN-COMPARE",
    model_id: str = "MOD-TAIL-COMPARE",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap TailModelComparisonResult into an EvidenceRecord."""
    params = {
        "confidence": comp.confidence,
        "models_compared": list(comp.models_compared),
    }
    metrics: dict[str, bool | float | int | str | None] = {
        "confidence": comp.confidence,
        "n_models": len(comp.models_compared),
    }
    for m in comp.models_compared:
        metrics[f"var_{m}"] = comp.var_values.get(m)
        metrics[f"es_{m}"] = comp.es_values.get(m)
        metrics[f"es_to_var_{m}"] = comp.es_to_var_ratios.get(m)

    res = TestResult(
        test_id="traded_risk.var_es_comparison",
        test_name="Multi-Method VaR & Expected Shortfall Model Comparison",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Compared {len(comp.models_compared)} tail risk models @ {comp.confidence:.1%} confidence."
        ),
        limitations=[
            "Comparative metric evaluation only; no single estimator is declared universally optimal without an explicit model risk policy.",
        ],
    )
    return EvidenceRecord.from_result(
        res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=comp.data_fingerprint,
    )


# =========================================================================== #
# GATE 6: SCENARIO, STRESS & REVERSE-STRESS EVIDENCE RECORD ADAPTERS
# =========================================================================== #


def scenario_result_to_evidence(
    result: ScenarioResult,
    run_id: str = "RUN-SCEN",
    model_id: str = "MOD-SCENARIO",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap ScenarioResult into an audit-grade EvidenceRecord."""
    params = {
        "scenario_id": result.scenario_id,
        "scenario_type": result.scenario_type,
        "repricing_method": result.repricing_method,
        "has_portfolio_value": result.portfolio_value is not None,
    }
    metrics: dict[str, bool | float | int | str | None] = {
        "scenario_id": result.scenario_id,
        "scenario_type": result.scenario_type,
        "repricing_method": result.repricing_method,
        "scenario_return": result.scenario_return,
        "scenario_loss": result.scenario_loss,
        "scenario_pnl": result.scenario_pnl,
        "scenario_monetary_loss": result.scenario_monetary_loss,
        "reconciliation_error": result.reconciliation_error,
        "converged": result.converged,
        "n_asset_contributions": len(result.asset_contributions),
        "n_factor_contributions": len(result.factor_contributions),
    }
    for a, c in result.asset_contributions.items():
        metrics[f"asset_contrib.{a}"] = c
    for f, c in result.factor_contributions.items():
        metrics[f"factor_contrib.{f}"] = c

    test_id = f"scenario.{result.repricing_method.lower()}"
    test_res = TestResult(
        test_id=test_id,
        test_name=f"Portfolio Scenario Stress: {result.scenario_id} ({result.repricing_method})",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Scenario '{result.scenario_id}' ({result.repricing_method}): return={result.scenario_return:.4f}, "
            f"canonical loss={result.scenario_loss:.4f}, recon_error={result.reconciliation_error:.3g}."
        ),
        limitations=list(result.limitations),
    )
    return EvidenceRecord.from_result(
        test_res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=result.data_fingerprint,
    )


def active_scenario_to_evidence(
    active_res: ActiveScenarioResult,
    run_id: str = "RUN-ACT-SCEN",
    model_id: str = "MOD-ACTIVE-SCENARIO",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap ActiveScenarioResult into an EvidenceRecord."""
    params = {"scenario_id": active_res.scenario_id}
    metrics: dict[str, bool | float | int | str | None] = {
        "scenario_id": active_res.scenario_id,
        "portfolio_return": active_res.portfolio_return,
        "benchmark_return": active_res.benchmark_return,
        "active_return": active_res.active_return,
        "portfolio_loss": active_res.portfolio_loss,
        "benchmark_loss": active_res.benchmark_loss,
        "active_loss": active_res.active_loss,
        "reconciliation_error": active_res.reconciliation_error,
    }
    for a, c in active_res.active_asset_contributions.items():
        metrics[f"active_asset_contrib.{a}"] = c
    for f, c in active_res.active_factor_contributions.items():
        metrics[f"active_factor_contrib.{f}"] = c

    test_res = TestResult(
        test_id="scenario.active_stress",
        test_name=f"Active Portfolio Scenario Stress: {active_res.scenario_id}",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Active stress under '{active_res.scenario_id}': port={active_res.portfolio_return:.4f}, "
            f"bmk={active_res.benchmark_return:.4f}, active={active_res.active_return:.4f}."
        ),
        limitations=[
            "Exact active return decomposition: R_active = R_port - R_bmk.",
        ],
    )
    return EvidenceRecord.from_result(
        test_res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=active_res.data_fingerprint,
    )


def group_scenario_to_evidence(
    scenario_id: str,
    group_contributions: dict[str, float],
    partition_contract: str = "EXHAUSTIVE_PARTITION",
    run_id: str = "RUN-GRP-SCEN",
    model_id: str = "MOD-GROUP-SCENARIO",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap group/sector scenario contributions into an EvidenceRecord."""
    params = {
        "scenario_id": scenario_id,
        "partition_contract": partition_contract,
        "n_groups": len(group_contributions),
    }
    metrics: dict[str, bool | float | int | str | None] = {
        "scenario_id": scenario_id,
        "partition_contract": partition_contract,
        "n_groups": len(group_contributions),
        "total_group_contribution": sum(group_contributions.values()),
    }
    for g, c in group_contributions.items():
        metrics[f"group_contrib.{g}"] = c

    test_res = TestResult(
        test_id="scenario.group_decomposition",
        test_name=f"Group Scenario Stress Decomposition: {scenario_id}",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Group stress decomposition for '{scenario_id}' across {len(group_contributions)} groups "
            f"under {partition_contract}."
        ),
        limitations=[
            f"Group additivity contract: {partition_contract}.",
        ],
    )
    return EvidenceRecord.from_result(
        test_res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=_hash_dict(group_contributions),
    )


def scenario_set_to_evidence(
    set_res: ScenarioSetResult,
    run_id: str = "RUN-SET-SCEN",
    model_id: str = "MOD-SCENARIO-SET",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap ScenarioSetResult into an EvidenceRecord."""
    params = {
        "ranking_metric": set_res.ranking_metric,
        "n_scenarios": len(set_res.scenarios_evaluated),
        "scenarios": list(set_res.scenarios_evaluated),
    }
    metrics: dict[str, bool | float | int | str | None] = {
        "n_scenarios": len(set_res.scenarios_evaluated),
        "ranking_metric": set_res.ranking_metric,
        "worst_scenario_id": set_res.worst_scenario_id,
        "best_scenario_id": set_res.best_scenario_id,
        "worst_scenario_loss": set_res.worst_scenario_loss,
        "best_scenario_loss": set_res.best_scenario_loss,
        "comparability_valid": set_res.comparability_valid,
    }
    for s_id in set_res.scenarios_evaluated:
        metrics[f"loss.{s_id}"] = set_res.scenario_losses[s_id]
        metrics[f"method.{s_id}"] = set_res.method_disclosures.get(s_id, "")

    test_res = TestResult(
        test_id="scenario.set_comparison",
        test_name="Multi-Scenario Set Loss Ranking & Comparison",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Multi-scenario comparison of {len(set_res.scenarios_evaluated)} scenarios: "
            f"worst='{set_res.worst_scenario_id}' (loss={set_res.worst_scenario_loss:.4f}), "
            f"best='{set_res.best_scenario_id}' (loss={set_res.best_scenario_loss:.4f})."
        ),
        limitations=list(set_res.limitations),
    )
    return EvidenceRecord.from_result(
        test_res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=set_res.data_fingerprint,
    )


def scenario_sensitivity_to_evidence(
    sens_res: ScenarioSensitivityResult,
    run_id: str = "RUN-SENS-SCEN",
    model_id: str = "MOD-SENSITIVITY",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap ScenarioSensitivityResult into an EvidenceRecord."""
    params = {
        "risk_factor_id": sens_res.risk_factor_id,
        "n_points": len(sens_res.grid_points),
    }
    metrics: dict[str, bool | float | int | str | None] = {
        "risk_factor_id": sens_res.risk_factor_id,
        "n_points": len(sens_res.grid_points),
        "base_loss": sens_res.base_loss,
        "max_loss": sens_res.max_loss,
        "min_loss": sens_res.min_loss,
    }
    for pt in sens_res.grid_points:
        metrics[f"loss_m_{pt.shock_multiplier:.2f}"] = pt.portfolio_loss

    test_res = TestResult(
        test_id="scenario.sensitivity_grid",
        test_name=f"Scenario Sensitivity Grid: {sens_res.risk_factor_id}",
        status=Status.RECORDED,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Sensitivity sweep for '{sens_res.risk_factor_id}' over {len(sens_res.grid_points)} points: "
            f"base loss={sens_res.base_loss:.4f}, max loss={sens_res.max_loss:.4f}."
        ),
        limitations=[
            "Deterministic parameter sweep across caller-specified shock multipliers.",
        ],
    )
    return EvidenceRecord.from_result(
        test_res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=sens_res.data_fingerprint,
    )


def reverse_stress_to_evidence(
    rev_res: ReverseStressResult,
    run_id: str = "RUN-REV-STRESS",
    model_id: str = "MOD-REVERSE-STRESS",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap ReverseStressResult into an EvidenceRecord."""
    params = {
        "target_loss": rev_res.target_loss,
        "distance_norm": rev_res.distance_norm,
        "is_closed_form": rev_res.is_closed_form,
    }
    metrics: dict[str, bool | float | int | str | None] = {
        "target_loss": rev_res.target_loss,
        "achieved_loss": rev_res.achieved_loss,
        "achieved_return": rev_res.achieved_return,
        "loss_gap": rev_res.loss_gap,
        "distance": rev_res.distance,
        "distance_norm": rev_res.distance_norm,
        "bounds_satisfied": rev_res.bounds_satisfied,
        "solver_status": rev_res.solver_status,
        "converged": rev_res.converged,
        "is_closed_form": rev_res.is_closed_form,
    }
    for rf, s in rev_res.shock_vector.items():
        metrics[f"shock.{rf}"] = s

    test_res = TestResult(
        test_id="scenario.reverse_stress",
        test_name=f"Minimum Shock Reverse Stress ({rev_res.distance_norm})",
        status=Status.RECORDED if rev_res.converged else Status.FAIL,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Reverse stress under {rev_res.distance_norm} norm: target={rev_res.target_loss:.4f}, "
            f"achieved={rev_res.achieved_loss:.4f}, distance={rev_res.distance:.4f}, status={rev_res.solver_status}."
        ),
        limitations=list(rev_res.limitations),
    )
    return EvidenceRecord.from_result(
        test_res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=rev_res.data_fingerprint,
    )


def scenario_data_integrity_to_evidence(
    diag: ScenarioDataIntegrityResult,
    run_id: str = "RUN-SCEN-INTEG",
    model_id: str = "MOD-SCEN-INTEG",
    dataset_id: str = "DS-MARKET",
) -> EvidenceRecord:
    """Wrap ScenarioDataIntegrityResult into an EvidenceRecord."""
    params = {"scenario_id": diag.scenario_id, "n_shocks": diag.n_shocks}
    metrics: dict[str, bool | float | int | str | None] = {
        "scenario_id": diag.scenario_id,
        "valid": diag.valid,
        "n_shocks": diag.n_shocks,
        "repricing_compatible": diag.repricing_compatible,
        "sensitivities_complete": diag.sensitivities_complete,
        "coverage_complete": diag.coverage_complete,
        "provenance_valid": diag.provenance_valid,
        "n_issues": len(diag.issues),
    }

    test_res = TestResult(
        test_id="scenario.data_integrity",
        test_name=f"Scenario Data Integrity Audit: {diag.scenario_id}",
        status=Status.RECORDED if diag.valid else Status.FAIL,
        params=params,
        metrics=metrics,
        interpretation=(
            f"Scenario data integrity audit for '{diag.scenario_id}': valid={diag.valid}, "
            f"n_shocks={diag.n_shocks}, issues={len(diag.issues)}."
        ),
        limitations=list(diag.issues)
        if diag.issues
        else ["Scenario specification passed all semantic and provenance audits."],
    )
    return EvidenceRecord.from_result(
        test_res,
        run_id=run_id,
        model_id=model_id,
        dataset_id=dataset_id,
        input_artifact_hash=diag.data_fingerprint,
    )
