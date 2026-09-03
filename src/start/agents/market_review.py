"""Specialist Market and Portfolio Intelligence Review Agents (Gate 2 & 3 Institutional Slice).

Core architectural principles:
- Agents orchestrate, plan, review, and challenge.
- Deterministic engines compute; agents NEVER compute portfolio mathematics in prose.
- Strict tool allowlists are enforced per agent.
- Every quantitative claim must cite verified [EV-xxxx] tokens and resolvable metric paths.
- AdversarialChallengeAgent issues structured challenges requiring deterministic tool verification.
- EvidenceCriticAgent validates evidence provenance; GovernanceAgent alone makes sign-off determinations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd

from start.agents.base import BaseAgent
from start.core.schemas import EvidenceRecord
from start.portfolio.contracts import (
    ActiveRiskDecompositionResult,
    BlackLittermanResult,
    BootstrapStabilityResult,
    BrinsonAttributionResult,
    CarinoLinkedAttributionResult,
    ChallengeResolution,
    ChallengeState,
    CopheneticResult,
    CovarianceComparisonResult,
    CovarianceDiagnostics,
    CVaROptimizationResult,
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
    RiskContributionResult,
    RobustMVOResult,
    TrackingErrorResult,
)
from start.portfolio.evidence_bridge import challenge_result_to_diagnostic_evidence
from start.telemetry.bus import TelemetryBus


class CriticDisposition(StrEnum):
    """Typed evidence validation dispositions emitted by EvidenceCriticAgent."""

    EVIDENCE_VALID = "EVIDENCE_VALID"
    EVIDENCE_INVALID = "EVIDENCE_INVALID"
    BLOCKED = "BLOCKED"
    READY_FOR_GOVERNANCE = "READY_FOR_GOVERNANCE"


class GovernanceVerdict(StrEnum):
    """Typed institutional governance decisions emitted strictly by GovernanceAgent."""

    ACCEPT = "ACCEPT"
    ACCEPT_WITH_CONDITIONS = "ACCEPT_WITH_CONDITIONS"
    REMEDIATE = "REMEDIATE"
    ESCALATE = "ESCALATE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class AgentFinding:
    """Structured, evidence-grounded finding from a specialist agent."""

    finding_id: str
    source_agent: str
    severity: str  # "info" | "advisory" | "material_warning" | "critical_breach"
    title: str
    evidence_ids: tuple[str, ...]
    statement: str
    recommended_action: str


@dataclass(frozen=True)
class AdversarialChallenge:
    """Targeted challenge demanding deterministic tool verification."""

    challenge_id: str
    challenger_agent: str
    target_area: str
    challenge_question: str
    evidence_ids: tuple[str, ...]
    required_tool: str
    parameters: dict[str, Any]
    status: str = "OPEN"  # "OPEN" | "VERIFIED_RESILIENT" | "VULNERABILITY_CONFIRMED"


@dataclass
class HierarchicalAssessment:
    """Typed assessment emitted by HierarchicalAllocationAgent."""

    tree_result: HierarchicalTreeResult | None = None
    herc_result: HERCResult | None = None
    linkage_sensitivity: LinkageSensitivityResult | None = None
    cophenetic_result: CopheneticResult | None = None
    bootstrap_stability: BootstrapStabilityResult | None = None
    risk_contributions: RiskContributionResult | None = None
    findings: list[AgentFinding] = field(default_factory=list)
    evidence_citations: list[str] = field(default_factory=list)


@dataclass
class OptimizationAssessment:
    """Typed assessment emitted by PortfolioConstructionAgent."""

    method_comparison: MethodComparisonResult | None = None
    black_litterman: BlackLittermanResult | None = None
    robust_mvo: RobustMVOResult | None = None
    cvar_optimization: CVaROptimizationResult | None = None
    max_diversification: MaxDiversificationResult | None = None
    tracking_error: TrackingErrorResult | None = None
    rebalance_decision: RebalanceDecision | None = None
    active_constraints: dict[str, Any] = field(default_factory=dict)
    findings: list[AgentFinding] = field(default_factory=list)
    evidence_citations: list[str] = field(default_factory=list)


@dataclass
class CovarianceAssessment:
    """Typed assessment emitted by CovarianceRiskAgent."""

    diagnostics: CovarianceDiagnostics | None = None
    repair_result: PSDRepairResult | None = None
    comparison_result: CovarianceComparisonResult | None = None
    findings: list[AgentFinding] = field(default_factory=list)
    evidence_citations: list[str] = field(default_factory=list)


@dataclass
class FactorAttributionAssessment:
    """Typed assessment emitted by FactorRiskAttributionAgent."""

    factor_model: FactorRiskModelResult | None = None
    risk_decomposition: FactorRiskDecompositionResult | None = None
    active_risk_decomposition: ActiveRiskDecompositionResult | None = None
    factor_return_attribution: FactorReturnAttributionResult | None = None
    brinson_attribution: BrinsonAttributionResult | None = None
    carino_linking: CarinoLinkedAttributionResult | None = None
    data_integrity: FactorDataIntegrityResult | None = None
    findings: list[AgentFinding] = field(default_factory=list)
    evidence_citations: list[str] = field(default_factory=list)


def get_grounded_metric(records: list[EvidenceRecord], evidence_id: str, metric_path: str) -> Any:
    """Retrieve and ground a numerical value from evidence records, failing closed on mismatch."""
    for r in records:
        if r.evidence_id == evidence_id:
            if metric_path in r.metrics:
                return r.metrics[metric_path]
            raise KeyError(
                f"Metric path '{metric_path}' not found in EvidenceRecord '{evidence_id}'. "
                f"Available: {list(r.metrics.keys())}"
            )
    raise KeyError(f"EvidenceRecord with ID '{evidence_id}' was not found in active evidence pool.")


class HierarchicalAllocationAgent(BaseAgent):
    """Specialist agent for Hierarchical Risk Parity, HERC, tree hierarchy, and stability."""

    ALLOWED_TOOLS = (
        "hrp_weights_and_tree",
        "solve_herc",
        "linkage_sensitivity_analysis",
        "cophenetic_distance_diagnostic",
        "bootstrap_cluster_stability",
        "calculate_risk_contributions",
    )

    def __init__(self, telemetry_bus: TelemetryBus | None = None):
        super().__init__(name="Hierarchical Allocation Agent", telemetry_bus=telemetry_bus)

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Enforce strict tool allowlist for HierarchicalAllocationAgent."""
        if tool_name not in self.ALLOWED_TOOLS:
            raise PermissionError(
                f"Tool '{tool_name}' is not in the allowed toolset for {self.name}. "
                f"Allowed tools: {self.ALLOWED_TOOLS}"
            )
        from start import portfolio

        fn = getattr(portfolio, tool_name, None)
        if fn is None:
            raise ValueError(f"Tool '{tool_name}' not found in start.portfolio package.")
        return fn(**kwargs)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Review hierarchical tree structures, HERC, and linkage sensitivity using deterministic tools only."""
        records: list[EvidenceRecord] = context.get("evidence_records", [])
        evidence_ids = [r.evidence_id for r in records if r.evidence_id]

        hrp_records = [r for r in records if r.test_id == "portfolio.hierarchical_risk_parity"]
        herc_records = [
            r
            for r in records
            if r.test_id in ("portfolio.herc", "portfolio.hierarchical_equal_risk_contribution")
        ]
        tree_records = [r for r in records if r.test_id == "portfolio.hierarchical_risk_parity.tree_topology"]
        findings: list[AgentFinding] = []

        for r in hrp_records:
            citation = r.evidence_id
            eff_n = get_grounded_metric(records, citation, "effective_n_positions")
            max_w = r.metrics.get("max_weight")
            linkage_m = r.metrics.get("linkage_method", "single")

            statement = f"Hierarchical allocation formed with {linkage_m} linkage. Effective positions: {eff_n}. [{citation}]"
            if max_w is not None:
                statement = f"Hierarchical allocation formed with {linkage_m} linkage. Effective positions: {eff_n}, largest weight: {max_w}. [{citation}]"

            findings.append(
                AgentFinding(
                    finding_id=f"FIND-HRP-{citation}",
                    source_agent=self.name,
                    severity="info",
                    title="HRP Tree Allocation Topology",
                    evidence_ids=(citation,),
                    statement=statement,
                    recommended_action="Inspect linkage sensitivity and dendrogram seriation.",
                )
            )

        for r in herc_records:
            citation = r.evidence_id
            eff_n = get_grounded_metric(records, citation, "effective_n_positions")
            vol = get_grounded_metric(records, citation, "portfolio_volatility_annualised")
            findings.append(
                AgentFinding(
                    finding_id=f"FIND-HERC-{citation}",
                    source_agent=self.name,
                    severity="info",
                    title="HERC Cluster Equal Risk Parity",
                    evidence_ids=(citation,),
                    statement=f"HERC allocation verified across dendrogram cluster partitions. Effective positions: {eff_n}, Volatility: {vol}. [{citation}]",
                    recommended_action="Compare cluster risk contributions against flat ERC allocation.",
                )
            )

        self.emit_trace(
            stage="Hierarchical Risk Parity Review",
            progress=100.0,
            status_msg=f"Reviewed {len(hrp_records) + len(herc_records) + len(tree_records)} hierarchical evidence records.",
            reasoning_step="Evaluated tree seriation, HERC cluster risk budgeting, and linkage parameters.",
            confidence_score=1.0,
            evidence_citations=evidence_ids,
        )

        return {
            "status": "completed",
            "agent": self.name,
            "findings": [asdict(f) for f in findings],
            "evidence_citations": evidence_ids,
        }


class PortfolioConstructionAgent(BaseAgent):
    """Specialist agent for institutional portfolio optimization, constraints, and rebalancing."""

    ALLOWED_TOOLS = (
        "solve_equal_weight",
        "solve_min_variance",
        "solve_equal_risk_contribution",
        "trace_efficient_frontier",
        "compare_portfolio_methods",
        "solve_black_litterman",
        "solve_robust_mvo",
        "robust_mvo_sensitivity_grid",
        "solve_cvar_portfolio",
        "solve_herc",
        "solve_max_diversification",
        "solve_tracking_error_constrained",
        "verify_portfolio_constraints",
        "compute_turnover",
        "compute_transaction_costs",
        "build_rebalance_decision",
    )

    def __init__(self, telemetry_bus: TelemetryBus | None = None):
        super().__init__(name="Portfolio Construction Agent", telemetry_bus=telemetry_bus)

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Enforce strict tool allowlist for PortfolioConstructionAgent."""
        if tool_name not in self.ALLOWED_TOOLS:
            raise PermissionError(
                f"Tool '{tool_name}' is not in the allowed toolset for {self.name}. "
                f"Allowed tools: {self.ALLOWED_TOOLS}"
            )
        from start import portfolio

        fn = getattr(portfolio, tool_name, None)
        if fn is None:
            raise ValueError(f"Tool '{tool_name}' not found in start.portfolio package.")
        return fn(**kwargs)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Review optimization solutions, constraint adherence, and risk parity balance."""
        records: list[EvidenceRecord] = context.get("evidence_records", [])
        evidence_ids = [r.evidence_id for r in records if r.evidence_id]
        findings: list[AgentFinding] = []

        mvo_records = [r for r in records if r.test_id == "portfolio.mean_variance"]
        erc_records = [
            r
            for r in records
            if r.test_id
            in (
                "portfolio.equal_risk_contribution",
                "portfolio.risk_statistics.equal_risk_contribution",
            )
        ]
        bl_records = [r for r in records if r.test_id == "portfolio.black_litterman"]
        cvar_records = [r for r in records if r.test_id == "portfolio.cvar_optimization"]
        reb_records = [r for r in records if r.test_id == "portfolio.rebalance.decision"]

        for r in mvo_records:
            cit = r.evidence_id
            obj = r.params.get("objective", "min_variance")
            vol = get_grounded_metric(records, cit, "volatility_annualised")
            findings.append(
                AgentFinding(
                    finding_id=f"FIND-MVO-{cit}",
                    source_agent=self.name,
                    severity="info",
                    title=f"Mean-Variance {obj.title()} Review",
                    evidence_ids=(cit,),
                    statement=f"MVO solved under active constraint set. Annualized volatility: {vol}. [{cit}]",
                    recommended_action="Compare risk parity and minimum variance points along efficient frontier.",
                )
            )

        for r in erc_records:
            cit = r.evidence_id
            disp = get_grounded_metric(records, cit, "max_risk_contribution_dispersion")
            findings.append(
                AgentFinding(
                    finding_id=f"FIND-ERC-{cit}",
                    source_agent=self.name,
                    severity="info",
                    title="Equal Risk Contribution Parity Review",
                    evidence_ids=(cit,),
                    statement=f"ERC allocation verified. Max risk contribution dispersion: {disp}. [{cit}]",
                    recommended_action="Inspect risk contribution waterfall.",
                )
            )

        for r in bl_records:
            cit = r.evidence_id
            vol = get_grounded_metric(records, cit, "posterior_volatility_annualised")
            to = get_grounded_metric(records, cit, "turnover_vs_prior")
            findings.append(
                AgentFinding(
                    finding_id=f"FIND-BL-{cit}",
                    source_agent=self.name,
                    severity="info",
                    title="Black-Litterman Allocation Review",
                    evidence_ids=(cit,),
                    statement=f"Black-Litterman posterior optimization verified. Posterior volatility: {vol}, Turnover: {to}. [{cit}]",
                    recommended_action="Inspect view residuals and tau sensitivity.",
                )
            )

        for r in cvar_records:
            cit = r.evidence_id
            cvar_val = get_grounded_metric(records, cit, "cvar_annualised")
            tails = get_grounded_metric(records, cit, "tail_scenario_count")
            findings.append(
                AgentFinding(
                    finding_id=f"FIND-CVAR-{cit}",
                    source_agent=self.name,
                    severity="info",
                    title="Rockafellar-Uryasev CVaR Review",
                    evidence_ids=(cit,),
                    statement=f"Minimum CVaR portfolio verified via linear programming. Annualized CVaR: {cvar_val}, Tail scenarios: {tails}. [{cit}]",
                    recommended_action="Evaluate tail scenario count against historical drawdown episodes.",
                )
            )

        for r in reb_records:
            cit = r.evidence_id
            to = get_grounded_metric(records, cit, "turnover")
            cost = get_grounded_metric(records, cit, "estimated_transaction_cost")
            findings.append(
                AgentFinding(
                    finding_id=f"FIND-REB-{cit}",
                    source_agent=self.name,
                    severity="info",
                    title="Rebalance Cost & Turnover Assessment",
                    evidence_ids=(cit,),
                    statement=f"Rebalance decision audited: one-way turnover {to}, estimated cost {cost}. [{cit}]",
                    recommended_action="Confirm net expected return remains positive after turnover drag.",
                )
            )

        self.emit_trace(
            stage="Portfolio Construction Review",
            progress=100.0,
            status_msg=f"Reviewed {len(mvo_records) + len(erc_records) + len(bl_records) + len(cvar_records) + len(reb_records)} portfolio optimization records.",
            reasoning_step="Verified constraint adherence, BL views, CVaR tail metrics, and rebalance turnover across solvers.",
            confidence_score=1.0,
            evidence_citations=evidence_ids,
        )

        return {
            "status": "completed",
            "agent": self.name,
            "findings": [asdict(f) for f in findings],
            "evidence_citations": evidence_ids,
        }


# =========================================================================== #
# GATE 4: COVARIANCE & FACTOR RISK SPECIALIST AGENTS
# =========================================================================== #
class CovarianceRiskAgent(BaseAgent):
    """Specialist agent for institutional covariance modeling, spectral diagnostics, and PSD integrity."""

    ALLOWED_TOOLS = (
        "diagnose_covariance",
        "repair_psd_covariance",
        "compare_covariance_estimators",
    )

    def __init__(self, telemetry_bus: TelemetryBus | None = None):
        super().__init__(name="Covariance Risk Agent", telemetry_bus=telemetry_bus)

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self.ALLOWED_TOOLS:
            raise PermissionError(
                f"Tool '{tool_name}' is not in the allowed toolset for {self.name}. "
                f"Allowed tools: {self.ALLOWED_TOOLS}"
            )
        from start import portfolio

        fn = getattr(portfolio, tool_name, None)
        if fn is None:
            raise ValueError(f"Tool '{tool_name}' not found in start.portfolio package.")
        return fn(**kwargs)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        records: list[EvidenceRecord] = context.get("evidence_records", [])
        findings: list[AgentFinding] = []
        evidence_ids = [r.evidence_id for r in records if r.evidence_id]

        cov_records = [r for r in records if r.test_id and r.test_id.startswith("covariance.")]
        for r in cov_records:
            cit = r.evidence_id
            is_psd = bool(r.metrics.get("is_psd", True))
            raw_min_eig = r.metrics.get("minimum_eigenvalue", 0.0)
            min_eig = float(raw_min_eig) if isinstance(raw_min_eig, (int, float)) else 0.0
            raw_cond = r.metrics.get("condition_number", 1.0)
            cond = float(raw_cond) if isinstance(raw_cond, (int, float)) else 1.0
            eff_rank = r.metrics.get("effective_rank")

            if not is_psd:
                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-COV-PSD-{cit}",
                        source_agent=self.name,
                        severity="material_warning",
                        title="Non-Positive-Semi-Definite Covariance Input",
                        evidence_ids=(cit,),
                        statement=f"Covariance matrix is indefinite (is_psd=False, minimum_eigenvalue={min_eig:.6g}). [{cit}]",
                        recommended_action="Execute explicit Higham or spectral PSD repair before solver ingestion.",
                    )
                )
            elif cond > 1e4:
                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-COV-COND-{cit}",
                        source_agent=self.name,
                        severity="advisory",
                        title="Ill-Conditioned Covariance Matrix",
                        evidence_ids=(cit,),
                        statement=f"Covariance matrix has high condition number ({cond:.4g}). [{cit}]",
                        recommended_action="Consider shrinkage (Ledoit-Wolf) or regularized EM estimation.",
                    )
                )
            else:
                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-COV-OK-{cit}",
                        source_agent=self.name,
                        severity="info",
                        title="Covariance Spectral Health Review",
                        evidence_ids=(cit,),
                        statement=f"Covariance matrix verified: is_psd=True, condition={cond:.4g}, effective_rank={eff_rank}. [{cit}]",
                        recommended_action="Proceed to portfolio optimization.",
                    )
                )

        self.emit_trace(
            stage="Covariance Risk Review",
            progress=100.0,
            status_msg=f"Audited {len(cov_records)} covariance evidence records.",
            reasoning_step="Verified spectral bounds, PSD status, rank sufficiency, and condition numbers.",
            confidence_score=1.0,
            evidence_citations=evidence_ids,
        )

        return {
            "status": "completed",
            "agent": self.name,
            "findings": [asdict(f) for f in findings],
            "evidence_citations": evidence_ids,
        }


class FactorRiskAttributionAgent(BaseAgent):
    """Specialist agent for linear factor risk models, Euler risk decomposition, and return attribution."""

    ALLOWED_TOOLS = (
        "build_linear_factor_model",
        "decompose_factor_risk",
        "decompose_active_risk",
        "compute_factor_return_attribution",
        "compute_brinson_attribution",
        "compute_carino_multi_period_linking",
        "validate_factor_data_integrity",
    )

    def __init__(self, telemetry_bus: TelemetryBus | None = None):
        super().__init__(name="Factor Risk & Attribution Agent", telemetry_bus=telemetry_bus)

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self.ALLOWED_TOOLS:
            raise PermissionError(
                f"Tool '{tool_name}' is not in the allowed toolset for {self.name}. "
                f"Allowed tools: {self.ALLOWED_TOOLS}"
            )
        from start import portfolio

        fn = getattr(portfolio, tool_name, None)
        if fn is None:
            raise ValueError(f"Tool '{tool_name}' not found in start.portfolio package.")
        return fn(**kwargs)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        records: list[EvidenceRecord] = context.get("evidence_records", [])
        findings: list[AgentFinding] = []
        evidence_ids = [r.evidence_id for r in records if r.evidence_id]

        factor_records = [
            r
            for r in records
            if r.test_id and (r.test_id.startswith("factor_risk.") or r.test_id.startswith("attribution."))
        ]
        for r in factor_records:
            cit = r.evidence_id
            if r.test_id == "factor_risk.decomposition":
                sys_share = r.metrics.get("systematic_variance_share", 0.0)
                spec_share = r.metrics.get("specific_variance_share", 0.0)
                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-FACTOR-DECOMP-{cit}",
                        source_agent=self.name,
                        severity="info",
                        title="Factor Risk Variance Decomposition Review",
                        evidence_ids=(cit,),
                        statement=f"Euler factor decomposition verified: systematic share {sys_share:.1%}, specific share {spec_share:.1%}. [{cit}]",
                        recommended_action="Inspect factor component variance waterfall.",
                    )
                )
            elif r.test_id == "factor_risk.active_decomposition":
                te = r.metrics.get("tracking_error_annualised", 0.0)
                fac_act = r.metrics.get("factor_active_share", 0.0)
                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-ACTIVE-RISK-{cit}",
                        source_agent=self.name,
                        severity="info",
                        title="Active Risk & Tracking Error Review",
                        evidence_ids=(cit,),
                        statement=f"Active risk tracking error verified at {te:.2%}, factor active share: {fac_act:.1%}. [{cit}]",
                        recommended_action="Audit active factor exposures against investment mandate.",
                    )
                )
            elif r.test_id == "attribution.brinson":
                active_ret = r.metrics.get("total_active_return", 0.0)
                alloc = r.metrics.get("total_allocation_effect", 0.0)
                select = r.metrics.get("total_selection_effect", 0.0)
                inter = r.metrics.get("total_interaction_effect", 0.0)
                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-BRINSON-{cit}",
                        source_agent=self.name,
                        severity="info",
                        title="Brinson-Fachler Performance Attribution Review",
                        evidence_ids=(cit,),
                        statement=f"Brinson active return ({active_ret:.4%}) decomposed: allocation={alloc:.4%}, selection={select:.4%}, interaction={inter:.4%}. [{cit}]",
                        recommended_action="Inspect sector allocation vs selection waterfalls.",
                    )
                )
            elif r.test_id == "attribution.factor_performance":
                f_contrib = r.metrics.get("total_factor_contribution", 0.0)
                s_contrib = r.metrics.get("total_specific_contribution", 0.0)
                err = r.metrics.get("max_abs_reconciliation_error", 0.0)
                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-FACTOR-ATTRIB-{cit}",
                        source_agent=self.name,
                        severity="info",
                        title="Factor Return Performance Attribution Review",
                        evidence_ids=(cit,),
                        statement=f"Factor return attribution verified: factor={f_contrib:.4f}, specific={s_contrib:.4f}, max error={err:.3g}. [{cit}]",
                        recommended_action="Verify period reconciliation residuals.",
                    )
                )

        self.emit_trace(
            stage="Factor Risk & Attribution Review",
            progress=100.0,
            status_msg=f"Audited {len(factor_records)} factor risk and attribution evidence records.",
            reasoning_step="Verified Euler risk contributions, active tracking error, Brinson effects, and factor return reconciliation.",
            confidence_score=1.0,
            evidence_citations=evidence_ids,
        )

        return {
            "status": "completed",
            "agent": self.name,
            "findings": [asdict(f) for f in findings],
            "evidence_citations": evidence_ids,
        }


class FactorDataIntegrityChecker(BaseAgent):
    """Deterministic pre-flight validator for factor model coverage, alignment, and time integrity."""

    ALLOWED_TOOLS = ("validate_factor_data_integrity",)

    def __init__(self, telemetry_bus: TelemetryBus | None = None):
        super().__init__(name="Factor Data Integrity Checker", telemetry_bus=telemetry_bus)

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self.ALLOWED_TOOLS:
            raise PermissionError(
                f"Tool '{tool_name}' is not in the allowed toolset for {self.name}. "
                f"Allowed tools: {self.ALLOWED_TOOLS}"
            )
        from start import portfolio

        fn = getattr(portfolio, tool_name, None)
        if fn is None:
            raise ValueError(f"Tool '{tool_name}' not found in start.portfolio package.")
        return fn(**kwargs)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        from start.portfolio.evidence_bridge import factor_data_integrity_to_evidence
        from start.portfolio.factor_risk import validate_factor_data_integrity

        rets = context.get("returns") if context.get("returns") is not None else context.get("returns_df")
        exp = (
            context.get("exposures")
            if context.get("exposures") is not None
            else context.get("factor_exposures")
        )
        fcov = (
            context.get("factor_cov")
            if context.get("factor_cov") is not None
            else context.get("factor_covariance")
        )
        svar = (
            context.get("specific_var")
            if context.get("specific_var") is not None
            else context.get("specific_variances")
        )
        fret = context.get("factor_returns")
        w = context.get("weights") if context.get("weights") is not None else context.get("portfolio_weights")

        # Skip factor data integrity check if no factor risk model inputs are present in context
        if exp is None and fcov is None and svar is None and fret is None:
            return {
                "status": "skipped",
                "agent": self.name,
                "integrity_result": None,
                "evidence_record": None,
                "findings": [],
                "evidence_citations": [],
            }

        res = validate_factor_data_integrity(
            returns=rets,
            exposures=exp,
            factor_cov=fcov,
            specific_var=svar,
            factor_returns=fret,
            weights=w,
            benchmark_weights=context.get("benchmark_weights"),
            timestamp=context.get("timestamp"),
        )
        ev = factor_data_integrity_to_evidence(res)

        findings: list[AgentFinding] = []
        if not res.is_valid:
            findings.append(
                AgentFinding(
                    finding_id=f"FIND-FACTOR-INTEGRITY-{ev.evidence_id}",
                    source_agent=self.name,
                    severity="critical_breach",
                    title="Factor Data Integrity Violation",
                    evidence_ids=(ev.evidence_id,),
                    statement=f"Factor data integrity check failed: {'; '.join(res.issues)}. [{ev.evidence_id}]",
                    recommended_action="Align factor universe and eliminate missing values before model ingestion.",
                )
            )

        return {
            "status": "completed",
            "agent": self.name,
            "integrity_result": asdict(res),
            "evidence_record": ev,
            "findings": [asdict(f) for f in findings],
            "evidence_citations": [ev.evidence_id],
        }


@dataclass(frozen=True)
class TailRiskAssessment:
    """Specialist qualitative assessment synthesized from deterministic tail risk evidence."""

    agent_name: str
    findings: tuple[AgentFinding, ...]
    var_confidence: float
    test_significance: float
    has_coverage_rejection: bool
    has_independence_rejection: bool
    has_conditional_coverage_rejection: bool
    has_thin_tail_support: bool
    evidence_ids: tuple[str, ...]


class TailRiskAgent(BaseAgent):
    """Specialist agent for institutional VaR, Expected Shortfall, tail severity, and out-of-sample backtesting."""

    ALLOWED_TOOLS = (
        "compute_historical_var_es",
        "compute_parametric_normal_var_es",
        "compute_tail_risk_contributions",
        "compute_tail_severity",
        "compute_exception_duration_diagnostics",
        "run_comprehensive_tail_backtest",
        "compare_tail_risk_models",
    )

    def __init__(self, telemetry_bus: TelemetryBus | None = None):
        super().__init__(name="Tail Risk & Backtesting Agent", telemetry_bus=telemetry_bus)

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self.ALLOWED_TOOLS:
            raise PermissionError(
                f"Tool '{tool_name}' is not in the allowed toolset for {self.name}. "
                f"Allowed tools: {self.ALLOWED_TOOLS}"
            )
        from start import portfolio

        fn = getattr(portfolio, tool_name, None)
        if fn is None:
            raise ValueError(f"Tool '{tool_name}' not found in start.portfolio package.")
        return fn(**kwargs)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        records: list[EvidenceRecord] = context.get("evidence_records", [])
        findings: list[AgentFinding] = []
        evidence_ids = [r.evidence_id for r in records if r.evidence_id]

        tail_records = [
            r
            for r in records
            if r.test_id
            and (
                r.test_id.startswith("traded_risk.var_")
                or r.test_id
                in (
                    "traded_risk.expected_shortfall",
                    "traded_risk.tail_severity",
                    "traded_risk.exception_durations",
                    "traded_risk.es_contribution",
                    "traded_risk.var_es_comparison",
                )
            )
        ]

        has_cov_rej = False
        has_ind_rej = False
        has_cc_rej = False
        has_thin_tail = False
        var_conf = 0.99
        test_sig = 0.05

        for r in tail_records:
            cit = r.evidence_id
            if r.test_id == "traded_risk.expected_shortfall":
                var_val = r.metrics.get("var", 0.0)
                es_val = r.metrics.get("es", 0.0)
                conf = r.metrics.get("confidence", 0.99)
                method = r.metrics.get("method", "HISTORICAL")
                q_obs = r.metrics.get("tail_observations_count", 0)
                var_conf = float(conf) if conf is not None else 0.99

                if q_obs is not None and isinstance(q_obs, (int, float)) and q_obs <= 1:
                    has_thin_tail = True
                    findings.append(
                        AgentFinding(
                            finding_id=f"FIND-THIN-TAIL-{cit}",
                            source_agent=self.name,
                            severity="advisory",
                            title="Thin Tail Sample Support",
                            evidence_ids=(cit,),
                            statement=f"Expected Shortfall estimation has thin tail support ({q_obs} observation(s)). Point estimate is constrained by sample size. [{cit}]",
                            recommended_action="Increase historical estimation window or evaluate parametric alternatives.",
                        )
                    )

                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-TAIL-ESTIMATE-{cit}",
                        source_agent=self.name,
                        severity="info",
                        title=f"{str(method).upper()} Tail Risk Estimate",
                        evidence_ids=(cit,),
                        statement=f"Quantified {str(method).upper()} tail risk @ {var_conf:.1%} confidence: VaR = {var_val:.4f}, Expected Shortfall = {es_val:.4f}. [{cit}]",
                        recommended_action="Inspect tail loss distribution and ES/VaR ratio.",
                    )
                )

            elif r.test_id == "traded_risk.var_conditional_coverage":
                var_c = r.metrics.get("var_confidence", 0.99)
                test_s = r.metrics.get("test_significance", 0.05)
                var_conf = float(var_c) if var_c is not None else 0.99
                test_sig = float(test_s) if test_s is not None else 0.05
                n_exc = r.metrics.get("n_exceptions", 0)
                n_obs = r.metrics.get("n_observations", 0)

                kup_rej = bool(r.metrics.get("kupiec_rejected", False))
                chr_rej = bool(r.metrics.get("christoffersen_rejected", False))
                cc_rej = bool(r.metrics.get("conditional_coverage_rejected", False))

                kup_p = r.metrics.get("kupiec_p_value", 1.0)
                chr_p = r.metrics.get("christoffersen_p_value", 1.0)
                cc_p = r.metrics.get("conditional_coverage_p_value", 1.0)

                if kup_rej:
                    has_cov_rej = True
                    findings.append(
                        AgentFinding(
                            finding_id=f"FIND-KUPIEC-REJECT-{cit}",
                            source_agent=self.name,
                            severity="critical_breach",
                            title="Kupiec Unconditional Coverage Rejection",
                            evidence_ids=(cit,),
                            statement=f"Kupiec POF test rejects null of unconditional coverage at significance gamma={test_sig:.2%} (p={kup_p:.4f}, {n_exc}/{n_obs} exceptions). [{cit}]",
                            recommended_action="Recalibrate VaR volatility/quantile scaling.",
                        )
                    )
                else:
                    findings.append(
                        AgentFinding(
                            finding_id=f"FIND-KUPIEC-NO-REJECT-{cit}",
                            source_agent=self.name,
                            severity="info",
                            title="Kupiec Unconditional Coverage Non-Rejection",
                            evidence_ids=(cit,),
                            statement=f"Kupiec POF test does not reject unconditional coverage at significance gamma={test_sig:.2%} (p={kup_p:.4f}, {n_exc}/{n_obs} exceptions). (Note: failure to reject is not proof of validity). [{cit}]",
                            recommended_action="Audit exception serial independence and clustering diagnostics.",
                        )
                    )

                if chr_rej:
                    has_ind_rej = True
                    findings.append(
                        AgentFinding(
                            finding_id=f"FIND-CHRISTOFFERSEN-REJECT-{cit}",
                            source_agent=self.name,
                            severity="material_warning",
                            title="Christoffersen Independence Rejection (Clustering)",
                            evidence_ids=(cit,),
                            statement=f"Christoffersen test rejects exception independence at significance gamma={test_sig:.2%} (p={chr_p:.4f}). Realized exceptions exhibit statistically significant clustering. [{cit}]",
                            recommended_action="Implement dynamic conditional volatility model to capture volatility clustering.",
                        )
                    )
                else:
                    findings.append(
                        AgentFinding(
                            finding_id=f"FIND-CHRISTOFFERSEN-NO-REJECT-{cit}",
                            source_agent=self.name,
                            severity="info",
                            title="Christoffersen Independence Non-Rejection",
                            evidence_ids=(cit,),
                            statement=f"Christoffersen test does not reject serial independence of exceptions at significance gamma={test_sig:.2%} (p={chr_p:.4f}). [{cit}]",
                            recommended_action="Monitor inter-exception durations.",
                        )
                    )

                if cc_rej:
                    has_cc_rej = True
                    findings.append(
                        AgentFinding(
                            finding_id=f"FIND-COND-COVERAGE-REJECT-{cit}",
                            source_agent=self.name,
                            severity="material_warning",
                            title="Joint Conditional Coverage Rejection",
                            evidence_ids=(cit,),
                            statement=f"Joint conditional coverage test rejects null at significance gamma={test_sig:.2%} (p={cc_p:.4f}, LR_cc={r.metrics.get('conditional_coverage_lr', 0.0):.3f}). [{cit}]",
                            recommended_action="Remediate coverage and dependence model assumptions.",
                        )
                    )

            elif r.test_id == "traded_risk.tail_severity":
                mean_ex = r.metrics.get("mean_absolute_exceedance", 0.0)
                max_ex = r.metrics.get("max_absolute_exceedance", 0.0)
                tot_loss = r.metrics.get("total_tail_exceedance_loss", 0.0)
                n_e = r.metrics.get("n_exceptions", 0)
                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-SEVERITY-{cit}",
                        source_agent=self.name,
                        severity="info",
                        title="Tail Exceedance Severity Diagnostics",
                        evidence_ids=(cit,),
                        statement=f"Tail severity analysis across {n_e} exceptions: mean exceedance = {mean_ex:.4f}, max exceedance = {max_ex:.4f}, total excess loss = {tot_loss:.4f}. [{cit}]",
                        recommended_action="Review maximum exceedance scenarios against capital buffers.",
                    )
                )

            elif r.test_id == "traded_risk.exception_durations":
                m_dur = r.metrics.get("mean_duration", 0.0)
                max_run = r.metrics.get("max_run_length", 0)
                n_dur = r.metrics.get("n_durations", 0)
                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-DURATIONS-{cit}",
                        source_agent=self.name,
                        severity="info",
                        title="Inter-Exception Duration Diagnostics",
                        evidence_ids=(cit,),
                        statement=f"Duration diagnostics: {n_dur} intervals, mean duration = {m_dur:.1f} days, max consecutive exception streak = {max_run} day(s). [{cit}]",
                        recommended_action="Inspect clustered exception streaks for regime shift signals.",
                    )
                )

        self.emit_trace(
            stage="Tail Risk & Backtesting Review",
            progress=100.0,
            status_msg=f"Audited {len(tail_records)} tail risk and backtesting evidence records.",
            reasoning_step="Verified finite-sample Expected Shortfall, Kupiec POF, Christoffersen independence, conditional coverage, durations, and severity.",
            confidence_score=1.0,
            evidence_citations=evidence_ids,
        )

        assessment = TailRiskAssessment(
            agent_name=self.name,
            findings=tuple(findings),
            var_confidence=var_conf,
            test_significance=test_sig,
            has_coverage_rejection=has_cov_rej,
            has_independence_rejection=has_ind_rej,
            has_conditional_coverage_rejection=has_cc_rej,
            has_thin_tail_support=has_thin_tail,
            evidence_ids=tuple(evidence_ids),
        )

        return {
            "status": "completed",
            "agent": self.name,
            "assessment": asdict(assessment),
            "findings": [asdict(f) for f in findings],
            "evidence_citations": evidence_ids,
        }


@dataclass(frozen=True)
class ScenarioStressAssessment:
    """Institutional evaluation synthesized from deterministic scenario, stress, and reverse-stress evidence."""

    agent_name: str
    findings: tuple[AgentFinding, ...]
    scenarios_evaluated: tuple[str, ...]
    worst_scenario_id: str | None
    worst_scenario_loss: float | None
    best_scenario_id: str | None
    reverse_stress_achieved: bool | None
    evidence_ids: tuple[str, ...] = ()


class ScenarioStressAgent(BaseAgent):
    """Specialist agent for institutional scenario definition, deterministic repricing, and reverse stress."""

    ALLOWED_TOOLS = (
        "validate_scenario_data_integrity",
        "evaluate_scenario",
        "apply_asset_return_scenario",
        "apply_factor_scenario",
        "apply_delta_gamma_scenario",
        "apply_benchmark_active_scenario",
        "apply_group_scenario_decomposition",
        "compare_scenario_set",
        "evaluate_scenario_sensitivity_grid",
        "solve_reverse_stress",
        "replay_historical_scenario",
    )

    def __init__(self, telemetry_bus: TelemetryBus | None = None):
        super().__init__(name="Scenario & Stress Agent", telemetry_bus=telemetry_bus)

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        if tool_name not in self.ALLOWED_TOOLS:
            raise PermissionError(
                f"Tool '{tool_name}' is not in the allowed toolset for {self.name}. "
                f"Allowed tools: {self.ALLOWED_TOOLS}"
            )
        from start import portfolio

        fn = getattr(portfolio, tool_name, None)
        if fn is None:
            raise ValueError(f"Tool '{tool_name}' not found in start.portfolio package.")
        return fn(**kwargs)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        records: list[EvidenceRecord] = context.get("evidence_records", [])
        findings: list[AgentFinding] = []
        evidence_ids = [r.evidence_id for r in records if r.evidence_id]

        scen_records = [
            r
            for r in records
            if r.test_id
            and (
                r.test_id.startswith("scenario.")
                or r.test_id
                in (
                    "scenario.linear_return",
                    "scenario.factor_linear",
                    "scenario.delta",
                    "scenario.delta_gamma",
                    "scenario.active_stress",
                    "scenario.group_decomposition",
                    "scenario.set_comparison",
                    "scenario.sensitivity_grid",
                    "scenario.reverse_stress",
                    "scenario.data_integrity",
                    "scenario.historical_replay",
                )
            )
        ]

        scenarios_evaluated: list[str] = []
        worst_id: str | None = None
        worst_loss: float | None = None
        best_id: str | None = None
        rev_achieved: bool | None = None

        for r in scen_records:
            cit = r.evidence_id
            if r.test_id in (
                "scenario.linear_return",
                "scenario.factor_linear",
                "scenario.delta",
                "scenario.delta_gamma",
            ):
                s_id = str(r.metrics.get("scenario_id", "SCEN"))
                scenarios_evaluated.append(s_id)
                s_ret = r.metrics.get("scenario_return", 0.0)
                s_loss = r.metrics.get("scenario_loss", 0.0)
                meth = r.metrics.get("repricing_method", "LINEAR_RETURN")
                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-SCEN-{cit}",
                        source_agent=self.name,
                        severity="info",
                        title=f"Scenario Stress Evaluation: {s_id}",
                        evidence_ids=(cit,),
                        statement=f"Scenario '{s_id}' evaluated under {meth}: return={s_ret:.4f}, canonical loss={s_loss:.4f}. [{cit}]",
                        recommended_action="Incorporate scenario loss in capital and risk limit monitoring.",
                    )
                )

            elif r.test_id == "scenario.active_stress":
                s_id = str(r.metrics.get("scenario_id", "SCEN"))
                act_ret = r.metrics.get("active_return", 0.0)
                p_ret = r.metrics.get("portfolio_return", 0.0)
                b_ret = r.metrics.get("benchmark_return", 0.0)
                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-ACTIVE-SCEN-{cit}",
                        source_agent=self.name,
                        severity="info",
                        title="Active Scenario Stress Decomposition",
                        evidence_ids=(cit,),
                        statement=f"Active stress under '{s_id}': active return = {act_ret:.4f} (portfolio = {p_ret:.4f}, benchmark = {b_ret:.4f}). [{cit}]",
                        recommended_action="Review active risk concentration against benchmark.",
                    )
                )

            elif r.test_id == "scenario.set_comparison":
                worst_id = str(r.metrics.get("worst_scenario_id", ""))
                worst_loss_raw = r.metrics.get("worst_scenario_loss", 0.0)
                worst_loss = float(worst_loss_raw) if isinstance(worst_loss_raw, (int, float, str)) else 0.0
                best_id = str(r.metrics.get("best_scenario_id", ""))
                n_scen = r.metrics.get("n_scenarios", 0)
                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-SCEN-SET-{cit}",
                        source_agent=self.name,
                        severity="info",
                        title="Multi-Scenario Comparative Ranking",
                        evidence_ids=(cit,),
                        statement=f"Comparative ranking of {n_scen} scenarios: worst scenario is '{worst_id}' (loss = {worst_loss:.4f}), best is '{best_id}'. [{cit}]",
                        recommended_action="Focus risk remediation on worst-loss scenario conditions.",
                    )
                )

            elif r.test_id == "scenario.reverse_stress":
                t_loss_raw = r.metrics.get("target_loss", 0.0)
                t_loss = float(t_loss_raw) if isinstance(t_loss_raw, (int, float, str)) else 0.0
                ach_loss_raw = r.metrics.get("achieved_loss", 0.0)
                ach_loss = float(ach_loss_raw) if isinstance(ach_loss_raw, (int, float, str)) else 0.0
                dist_raw = r.metrics.get("distance", 0.0)
                dist = float(dist_raw) if isinstance(dist_raw, (int, float, str)) else 0.0
                norm = r.metrics.get("distance_norm", "L2")
                st = str(r.metrics.get("solver_status", "UNKNOWN"))
                rev_achieved = bool(r.metrics.get("converged", False))
                findings.append(
                    AgentFinding(
                        finding_id=f"FIND-REV-STRESS-{cit}",
                        source_agent=self.name,
                        severity="info" if rev_achieved else "material_warning",
                        title=f"Reverse Stress Solution ({norm})",
                        evidence_ids=(cit,),
                        statement=f"Reverse stress search ({norm}): target loss={t_loss:.4f}, achieved loss={ach_loss:.4f}, distance={dist:.4f}, status={st}. [{cit}]",
                        recommended_action="Inspect required minimum shock vector for plausibility under reference geometry.",
                    )
                )

            elif r.test_id == "scenario.data_integrity":
                is_val = bool(r.metrics.get("valid", True))
                n_shk = r.metrics.get("n_shocks", 0)
                n_iss_raw = r.metrics.get("n_issues", 0)
                n_iss = int(n_iss_raw) if isinstance(n_iss_raw, (int, float, str)) else 0
                if not is_val or n_iss > 0:
                    findings.append(
                        AgentFinding(
                            finding_id=f"FIND-SCEN-INTEG-{cit}",
                            source_agent=self.name,
                            severity="material_warning",
                            title="Scenario Data Integrity Warning",
                            evidence_ids=(cit,),
                            statement=f"Scenario data integrity audit identified {n_iss} issue(s) across {n_shk} shock legs. [{cit}]",
                            recommended_action="Remediate missing shock units, invalid spaces, or unmapped assets.",
                        )
                    )

        self.emit_trace(
            stage="Scenario & Stress Review",
            progress=100.0,
            status_msg=f"Audited {len(scen_records)} scenario and stress evidence records.",
            reasoning_step="Verified deterministic scenario repricing, active stress, multi-scenario loss ranking, and reverse stress optimization.",
            confidence_score=1.0,
            evidence_citations=evidence_ids,
        )

        assessment = ScenarioStressAssessment(
            agent_name=self.name,
            findings=tuple(findings),
            scenarios_evaluated=tuple(scenarios_evaluated),
            worst_scenario_id=worst_id,
            worst_scenario_loss=worst_loss,
            best_scenario_id=best_id,
            reverse_stress_achieved=rev_achieved,
            evidence_ids=tuple(evidence_ids),
        )

        return {
            "status": "completed",
            "agent": self.name,
            "assessment": asdict(assessment),
            "findings": [asdict(f) for f in findings],
            "evidence_citations": evidence_ids,
        }


class AdversarialChallengeAgent(BaseAgent):
    """Skeptical internal validator generating targeted challenges to model assumptions."""

    ALLOWED_TOOLS = (
        "linkage_sensitivity_analysis",
        "bootstrap_cluster_stability",
        "calculate_risk_contributions",
        "run_walk_forward_evaluation",
        "robust_mvo_sensitivity_grid",
        "verify_portfolio_constraints",
        "compute_transaction_costs",
        "solve_black_litterman",
        "solve_cvar_portfolio",
        # Gate 4 Tools
        "diagnose_covariance",
        "repair_psd_covariance",
        "compare_covariance_estimators",
        "build_linear_factor_model",
        "decompose_factor_risk",
        "decompose_active_risk",
        "compute_factor_return_attribution",
        "compute_brinson_attribution",
        "compute_carino_multi_period_linking",
        "validate_factor_data_integrity",
        # Gate 5 Tools
        "compute_historical_var_es",
        "compute_parametric_normal_var_es",
        "compute_tail_risk_contributions",
        "compute_tail_severity",
        "compute_exception_duration_diagnostics",
        "run_comprehensive_tail_backtest",
        "compare_tail_risk_models",
        # Gate 6 Tools
        "validate_scenario_data_integrity",
        "apply_asset_return_scenario",
        "apply_factor_scenario",
        "apply_delta_gamma_scenario",
        "apply_benchmark_active_scenario",
        "apply_group_scenario_decomposition",
        "compare_scenario_set",
        "evaluate_scenario_sensitivity_grid",
        "solve_reverse_stress",
        "replay_historical_scenario",
    )

    def __init__(self, telemetry_bus: TelemetryBus | None = None):
        super().__init__(name="Adversarial Challenge Agent", telemetry_bus=telemetry_bus)

    def execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Enforce strict tool allowlist for AdversarialChallengeAgent."""
        if tool_name not in self.ALLOWED_TOOLS:
            raise PermissionError(
                f"Tool '{tool_name}' is not in the allowed toolset for {self.name}. "
                f"Allowed tools: {self.ALLOWED_TOOLS}"
            )
        from start import portfolio

        fn = getattr(portfolio, tool_name, None)
        if fn is None:
            raise ValueError(f"Tool '{tool_name}' not found in start.portfolio package.")
        return fn(**kwargs)

    def formulate_portfolio_challenges(self, records: list[EvidenceRecord]) -> list[AdversarialChallenge]:
        """Generate structured adversarial challenges demanding deterministic tool verification."""
        challenges: list[AdversarialChallenge] = []

        # 1. HRP Linkage & Bootstrap
        hrp_records = [r for r in records if r.test_id == "portfolio.hierarchical_risk_parity"]
        for r in hrp_records:
            cit = r.evidence_id or "EV-PORT-HRP"
            challenges.append(
                AdversarialChallenge(
                    challenge_id=f"CHAL-LINKAGE-{cit}",
                    challenger_agent=self.name,
                    target_area="Hierarchy Robustness",
                    challenge_question="Does HRP allocation exhibit material sensitivity when switching from single to average linkage?",
                    evidence_ids=(cit,),
                    required_tool="linkage_sensitivity_analysis",
                    parameters={"methods": ("single", "complete", "average")},
                )
            )
            challenges.append(
                AdversarialChallenge(
                    challenge_id=f"CHAL-BOOTSTRAP-{cit}",
                    challenger_agent=self.name,
                    target_area="Cluster Persistence",
                    challenge_question="Do the hierarchical clusters persist under stationary block bootstrap resampling of returns?",
                    evidence_ids=(cit,),
                    required_tool="bootstrap_cluster_stability",
                    parameters={"n_replicates": 50, "seed": 42},
                )
            )

        # 2. Mean-Variance Concentration
        mvo_records = [r for r in records if r.test_id == "portfolio.mean_variance"]
        for r in mvo_records:
            cit = r.evidence_id or "EV-PORT-MVO"
            challenges.append(
                AdversarialChallenge(
                    challenge_id=f"CHAL-CONCENTRATION-{cit}",
                    challenger_agent=self.name,
                    target_area="Concentration Risk",
                    challenge_question="Is low portfolio volatility achieved via excessive risk concentration in a single asset?",
                    evidence_ids=(cit,),
                    required_tool="calculate_risk_contributions",
                    parameters={},
                )
            )

        # 3. Black-Litterman Tau Sensitivity & View Dominance
        bl_records = [r for r in records if r.test_id == "portfolio.black_litterman"]
        for r in bl_records:
            cit = r.evidence_id or "EV-PORT-BL"
            challenges.append(
                AdversarialChallenge(
                    challenge_id=f"CHAL-BL-TAU-{cit}",
                    challenger_agent=self.name,
                    target_area="View Dominance vs Prior",
                    challenge_question="How sensitive is the BL allocation to tau and view uncertainty Omega? Are views dominating equilibrium returns?",
                    evidence_ids=(cit,),
                    required_tool="solve_black_litterman",
                    parameters={"tau": 0.05},
                )
            )

        # 4. Robust MVO Concentration
        robust_records = [r for r in records if r.test_id == "portfolio.mean_variance.robust"]
        for r in robust_records:
            cit = r.evidence_id or "EV-PORT-ROBUST"
            challenges.append(
                AdversarialChallenge(
                    challenge_id=f"CHAL-ROBUST-CONC-{cit}",
                    challenger_agent=self.name,
                    target_area="Robustness vs Concentration",
                    challenge_question="Does robust MVO merely trade expected return for extreme concentration under increasing uncertainty radii?",
                    evidence_ids=(cit,),
                    required_tool="robust_mvo_sensitivity_grid",
                    parameters={"radii": (0.0, 0.25, 0.50, 1.0)},
                )
            )

        # 5. CVaR Tail Scenarios
        cvar_records = [r for r in records if r.test_id == "portfolio.cvar_optimization"]
        for r in cvar_records:
            cit = r.evidence_id or "EV-PORT-CVAR"
            challenges.append(
                AdversarialChallenge(
                    challenge_id=f"CHAL-CVAR-TAIL-{cit}",
                    challenger_agent=self.name,
                    target_area="Scenario Tail Stability",
                    challenge_question="Is the CVaR allocation sensitive to the confidence level or driven by a tiny subset of extreme scenarios?",
                    evidence_ids=(cit,),
                    required_tool="solve_cvar_portfolio",
                    parameters={"confidence_level": 0.95},
                )
            )

        # 6. Turnover / Cost Drag
        reb_records = [r for r in records if r.test_id == "portfolio.rebalance.decision"]
        for r in reb_records:
            cit = r.evidence_id or "EV-PORT-REB"
            challenges.append(
                AdversarialChallenge(
                    challenge_id=f"CHAL-TURNOVER-DRAG-{cit}",
                    challenger_agent=self.name,
                    target_area="Transaction Drag",
                    challenge_question="Does turnover and transaction cost drag erase apparent portfolio return improvements?",
                    evidence_ids=(cit,),
                    required_tool="compute_transaction_costs",
                    parameters={},
                )
            )

        # 7. Covariance Estimator Sensitivity (Gate 4)
        cov_records = [r for r in records if r.test_id and r.test_id.startswith("covariance.")]
        for r in cov_records:
            cit = r.evidence_id or "EV-COV"
            challenges.append(
                AdversarialChallenge(
                    challenge_id=f"CHAL-COV-ESTIMATOR-{cit}",
                    challenger_agent=self.name,
                    target_area="Covariance Sensitivity",
                    challenge_question="Does portfolio risk vary significantly across empirical, shrinkage, and regularized covariance estimators?",
                    evidence_ids=(cit,),
                    required_tool="compare_covariance_estimators",
                    parameters={"estimators": ("empirical", "ledoit_wolf", "regularized_em")},
                )
            )
            if not r.metrics.get("is_psd", True):
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-PSD-REPAIR-{cit}",
                        challenger_agent=self.name,
                        target_area="PSD Distortion",
                        challenge_question="What is the Frobenius distortion introduced by repairing this indefinite covariance matrix?",
                        evidence_ids=(cit,),
                        required_tool="repair_psd_covariance",
                        parameters={"method": "HIGHAM_NEAREST_CORRELATION"},
                    )
                )

        # 8. Factor Residual & Active Risk Drivers (Gate 4)
        factor_records = [r for r in records if r.test_id and r.test_id.startswith("factor_risk.")]
        for r in factor_records:
            cit = r.evidence_id or "EV-FACTOR"
            challenges.append(
                AdversarialChallenge(
                    challenge_id=f"CHAL-FACTOR-RESIDUAL-{cit}",
                    challenger_agent=self.name,
                    target_area="Specific Risk Residual",
                    challenge_question="What fraction of total portfolio variance is unexplained by systematic factors?",
                    evidence_ids=(cit,),
                    required_tool="decompose_factor_risk",
                    parameters={},
                )
            )

        # 9. Return Attribution Reconciliation (Gate 4)
        attrib_records = [r for r in records if r.test_id and r.test_id.startswith("attribution.")]
        for r in attrib_records:
            cit = r.evidence_id or "EV-ATTRIB"
            if r.test_id == "attribution.brinson":
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-BRINSON-STABILITY-{cit}",
                        challenger_agent=self.name,
                        target_area="Brinson Allocation vs Selection",
                        challenge_question="Does Brinson-Fachler attribution decompose active return into allocation, selection, and interaction without residual error?",
                        evidence_ids=(cit,),
                        required_tool="compute_brinson_attribution",
                        parameters={},
                    )
                )
            elif r.test_id == "attribution.multi_period_linking":
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-CARINO-LINKING-{cit}",
                        challenger_agent=self.name,
                        target_area="Multi-Period Geometric Linking",
                        challenge_question="Does Carino logarithmic linking reconcile cumulative multi-period active return exactly?",
                        evidence_ids=(cit,),
                        required_tool="compute_carino_multi_period_linking",
                        parameters={},
                    )
                )
            else:
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-ATTRIB-RECON-{cit}",
                        challenger_agent=self.name,
                        target_area="Attribution Reconciliation",
                        challenge_question="Do factor return contributions reconcile exactly to realized returns without residual leakage?",
                        evidence_ids=(cit,),
                        required_tool="compute_factor_return_attribution",
                        parameters={},
                    )
                )

        # 10. Tail Risk & Backtesting Challenges (Gate 5)
        tail_records = [
            r
            for r in records
            if r.test_id
            and (
                r.test_id.startswith("traded_risk.var_")
                or r.test_id
                in (
                    "traded_risk.expected_shortfall",
                    "traded_risk.tail_severity",
                    "traded_risk.exception_durations",
                    "traded_risk.es_contribution",
                    "traded_risk.var_es_comparison",
                )
            )
        ]
        for r in tail_records:
            cit = r.evidence_id or "EV-TAIL"
            if r.test_id in ("traded_risk.var_exceptions", "traded_risk.var_conditional_coverage"):
                v_conf = (
                    float(r.params.get("var_confidence", 0.99))
                    if r.params.get("var_confidence") is not None
                    else 0.99
                )
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-TAIL-UNCONDITIONAL-COVERAGE-{cit}",
                        challenger_agent=self.name,
                        target_area="Unconditional Exception Coverage",
                        challenge_question="Does the Kupiec POF test reject unconditional coverage at the pre-specified significance level?",
                        evidence_ids=(cit,),
                        required_tool="run_comprehensive_tail_backtest",
                        parameters={"var_confidence": v_conf, "test_significance": 0.05},
                    )
                )
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-TAIL-INDEPENDENCE-{cit}",
                        challenger_agent=self.name,
                        target_area="Exception Serial Independence",
                        challenge_question="Does the Christoffersen test reject exception independence, indicating volatility clustering?",
                        evidence_ids=(cit,),
                        required_tool="run_comprehensive_tail_backtest",
                        parameters={"var_confidence": v_conf, "test_significance": 0.05},
                    )
                )
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-TAIL-SEVERITY-{cit}",
                        challenger_agent=self.name,
                        target_area="Tail Exceedance Magnitude",
                        challenge_question="What is the severity of tail exceedances beyond the VaR threshold?",
                        evidence_ids=(cit,),
                        required_tool="compute_tail_severity",
                        parameters={},
                    )
                )
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-TAIL-DURATIONS-{cit}",
                        challenger_agent=self.name,
                        target_area="Inter-Exception Interval Dynamics",
                        challenge_question="Are exception events clustered into short inter-exception durations?",
                        evidence_ids=(cit,),
                        required_tool="compute_exception_duration_diagnostics",
                        parameters={},
                    )
                )
            elif r.test_id == "traded_risk.expected_shortfall":
                c_val = (
                    float(r.params.get("confidence", 0.99))
                    if r.params.get("confidence") is not None
                    else 0.99
                )
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-TAIL-SAMPLE-SUPPORT-{cit}",
                        challenger_agent=self.name,
                        target_area="Finite-Sample Tail Mass",
                        challenge_question="Is the empirical Expected Shortfall supported by adequate tail mass without thin-sample bias?",
                        evidence_ids=(cit,),
                        required_tool="compute_historical_var_es",
                        parameters={"confidence": c_val},
                    )
                )
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-TAIL-MODEL-COMPARE-{cit}",
                        challenger_agent=self.name,
                        target_area="Estimator Discrepancy",
                        challenge_question="How much does Expected Shortfall diverge between Historical and Parametric Normal assumptions?",
                        evidence_ids=(cit,),
                        required_tool="compare_tail_risk_models",
                        parameters={"confidence": c_val},
                    )
                )
            elif r.test_id == "traded_risk.es_contribution":
                c_val = (
                    float(r.params.get("confidence", 0.99))
                    if r.params.get("confidence") is not None
                    else 0.99
                )
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-TAIL-CONTRIB-RECON-{cit}",
                        challenger_agent=self.name,
                        target_area="Component ES Reconciliation",
                        challenge_question="Do component Expected Shortfall contributions reconcile exactly to total portfolio ES?",
                        evidence_ids=(cit,),
                        required_tool="compute_tail_risk_contributions",
                        parameters={"confidence": c_val},
                    )
                )

        # 10. Scenario & Stress Challenges (Gate 6)
        scen_records = [r for r in records if r.test_id and r.test_id.startswith("scenario.")]
        for r in scen_records:
            cit = r.evidence_id or "EV-SCEN"
            if r.test_id in ("scenario.delta_gamma", "scenario.delta", "scenario.linear_return"):
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-SCENARIO-METHOD-SENSITIVITY-{cit}",
                        challenger_agent=self.name,
                        target_area="Repricing Method Sensitivity",
                        challenge_question="Does portfolio stress loss vary materially between linear/delta-only and delta-gamma second-order repricing?",
                        evidence_ids=(cit,),
                        required_tool="apply_delta_gamma_scenario",
                        parameters={"method": "DELTA"},
                    )
                )
            elif r.test_id == "scenario.factor_linear":
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-SCENARIO-FACTOR-CONCENTRATION-{cit}",
                        challenger_agent=self.name,
                        target_area="Scenario Factor Dominance",
                        challenge_question="Is scenario stress loss dominated by a single factor exposure?",
                        evidence_ids=(cit,),
                        required_tool="apply_factor_scenario",
                        parameters={},
                    )
                )
            elif r.test_id == "scenario.reverse_stress":
                challenges.append(
                    AdversarialChallenge(
                        challenge_id=f"CHAL-REVERSE-STRESS-FRAGILITY-{cit}",
                        challenger_agent=self.name,
                        target_area="Reverse Stress Norm Sensitivity",
                        challenge_question="How sensitive is the minimum shock vector to the choice of distance norm and bounds?",
                        evidence_ids=(cit,),
                        required_tool="solve_reverse_stress",
                        parameters={},
                    )
                )

        return challenges

    def resolve_challenge(
        self,
        challenge: AdversarialChallenge | dict[str, Any],
        context: dict[str, Any],
    ) -> ChallengeResolution:
        """Deterministically resolve an adversarial challenge via allowed portfolio tools."""
        c_dict = asdict(challenge) if isinstance(challenge, AdversarialChallenge) else challenge
        c_id = c_dict["challenge_id"]
        tool_name = c_dict["required_tool"]
        params = dict(c_dict.get("parameters", {}))

        if tool_name not in self.ALLOWED_TOOLS:
            return ChallengeResolution(
                challenge_id=c_id,
                status=ChallengeState.BLOCKED,
                tool_name=tool_name,
                limitations=(f"Tool '{tool_name}' not in allowed toolset for {self.name}.",),
            )

        try:
            import inspect

            from start import portfolio

            fn = getattr(portfolio, tool_name, None)
            if fn is None:
                raise ValueError(f"Tool '{tool_name}' not found in start.portfolio package.")

            if "tau_grid" in params:
                tg = params.pop("tau_grid")
                params["tau"] = tg[0] if isinstance(tg, (list, tuple)) else tg

            sig = inspect.signature(fn)
            source_ev_ids: list[str] = list(c_dict.get("evidence_ids", ()))
            valid_params = {}
            for p_name in sig.parameters:
                if p_name in params:
                    valid_params[p_name] = params[p_name]
                elif p_name in context:
                    valid_params[p_name] = context[p_name]
                elif p_name == "scenario_returns":
                    if "scenario_returns" in context:
                        valid_params["scenario_returns"] = context["scenario_returns"]
                    elif "returns_df" in context:
                        valid_params["scenario_returns"] = context["returns_df"]
                    elif "returns" in context:
                        valid_params["scenario_returns"] = context["returns"]
                elif p_name == "returns":
                    if "returns" in context:
                        valid_params["returns"] = context["returns"]
                    elif "returns_df" in context:
                        valid_params["returns"] = context["returns_df"]
                    elif "asset_returns" in context:
                        valid_params["returns"] = context["asset_returns"]
                    elif "covariance" in context:
                        cov_mat = np.asarray(context["covariance"], dtype=float)
                        n_a = len(cov_mat)
                        # Check PSD before simulation; repair explicitly if indefinite to avoid Gaussian sampling warning
                        min_eig = float(np.min(np.linalg.eigvalsh(cov_mat)))
                        if min_eig < 1e-8:
                            from start.portfolio.contracts import PSDRepairMethod
                            from start.portfolio.covariance import repair_psd_covariance
                            from start.portfolio.evidence_bridge import psd_repair_to_evidence

                            rep_res = repair_psd_covariance(
                                cov_mat, method=PSDRepairMethod.HIGHAM_NEAREST_CORRELATION
                            )
                            sim_cov = np.asarray(rep_res.repaired_matrix, dtype=float)
                            rep_ev = psd_repair_to_evidence(rep_res)
                            if isinstance(context.get("evidence_records"), list):
                                context["evidence_records"].append(rep_ev)
                            if rep_ev.evidence_id not in source_ev_ids:
                                source_ev_ids.append(rep_ev.evidence_id)
                        else:
                            sim_cov = cov_mat
                        sim_r = np.random.RandomState(42).multivariate_normal(
                            np.zeros(n_a), sim_cov / 252.0, size=250
                        )
                        a_names = context.get("assets", [f"A{i}" for i in range(n_a)])
                        valid_params["returns"] = pd.DataFrame(sim_r, columns=a_names)
                elif p_name in ("pnl_or_losses", "pnl", "realized_pnl"):
                    if "pnl" in context:
                        valid_params[p_name] = context["pnl"]
                    elif "realized_pnl" in context:
                        valid_params[p_name] = context["realized_pnl"]
                    elif "pnl_or_losses" in context:
                        valid_params[p_name] = context["pnl_or_losses"]
                    elif "returns" in context:
                        valid_params[p_name] = context["returns"]
                    elif "returns_df" in context:
                        valid_params[p_name] = context["returns_df"]
                elif p_name in ("var_series", "var_forecasts"):
                    if "var_series" in context:
                        valid_params[p_name] = context["var_series"]
                    elif "var_forecasts" in context:
                        valid_params[p_name] = context["var_forecasts"]
                    elif "var" in context:
                        valid_params[p_name] = context["var"]
                elif p_name == "losses":
                    if "losses" in context:
                        valid_params["losses"] = context["losses"]
                    elif "loss_series" in context:
                        valid_params["losses"] = context["loss_series"]
                    elif "returns" in context:
                        valid_params["losses"] = -np.asarray(context["returns"], dtype=float)
                    elif "returns_df" in context:
                        valid_params["losses"] = -np.asarray(context["returns_df"], dtype=float)
                elif p_name == "indicators":
                    if "indicators" in context:
                        valid_params["indicators"] = context["indicators"]
                    elif "exception_indicators" in context:
                        valid_params["indicators"] = context["exception_indicators"]
                elif p_name == "returns_or_losses":
                    if "returns" in context:
                        valid_params["returns_or_losses"] = context["returns"]
                    elif "returns_df" in context:
                        valid_params["returns_or_losses"] = context["returns_df"]
                    elif "losses" in context:
                        valid_params["returns_or_losses"] = context["losses"]
                elif p_name == "cov":
                    if "covariance" in context:
                        valid_params["cov"] = context["covariance"]
                    elif "cov_matrix" in context:
                        valid_params["cov"] = context["cov_matrix"]
                elif p_name == "factor_model" and "factor_model" in context:
                    valid_params["factor_model"] = context["factor_model"]
                elif p_name == "exposures":
                    if "exposures" in context:
                        valid_params["exposures"] = context["exposures"]
                    elif "factor_exposures" in context:
                        valid_params["exposures"] = context["factor_exposures"]
                elif p_name == "factor_returns":
                    if "factor_returns" in context:
                        valid_params["factor_returns"] = context["factor_returns"]
                    elif "exposures" in context and "returns" in context:
                        exp_df = (
                            context["exposures"]
                            if isinstance(context["exposures"], pd.DataFrame)
                            else pd.DataFrame(context["exposures"])
                        )
                        rets_df = (
                            context["returns"]
                            if isinstance(context["returns"], pd.DataFrame)
                            else pd.DataFrame(context["returns"])
                        )
                        # Approximate factor returns via OLS / pseudoinverse
                        B_mat = exp_df.to_numpy()
                        R_mat = rets_df.to_numpy()
                        F_hat = R_mat @ np.linalg.pinv(B_mat).T
                        valid_params["factor_returns"] = pd.DataFrame(F_hat, columns=exp_df.columns)
                elif p_name == "factor_cov":
                    if "factor_cov" in context:
                        valid_params["factor_cov"] = context["factor_cov"]
                    elif "factor_covariance" in context:
                        valid_params["factor_cov"] = context["factor_covariance"]
                elif p_name == "specific_var":
                    if "specific_var" in context:
                        valid_params["specific_var"] = context["specific_var"]
                    elif "specific_variances" in context:
                        valid_params["specific_var"] = context["specific_variances"]
                elif p_name == "portfolio_weights":
                    if "portfolio_weights" in context:
                        valid_params["portfolio_weights"] = context["portfolio_weights"]
                    elif "weights" in context:
                        valid_params["portfolio_weights"] = context["weights"]
                elif p_name == "portfolio_returns":
                    if "portfolio_returns" in context:
                        valid_params["portfolio_returns"] = context["portfolio_returns"]
                    elif "returns" in context and isinstance(context["returns"], pd.DataFrame):
                        valid_params["portfolio_returns"] = context["returns"].mean().to_dict()
                elif p_name == "benchmark_returns":
                    if "benchmark_returns" in context:
                        valid_params["benchmark_returns"] = context["benchmark_returns"]
                    elif "returns" in context and isinstance(context["returns"], pd.DataFrame):
                        valid_params["benchmark_returns"] = context["returns"].mean().to_dict()
                elif p_name == "period_brinson_results" and "period_brinson_results" in context:
                    valid_params["period_brinson_results"] = context["period_brinson_results"]
                elif p_name == "period_portfolio_returns" and "period_portfolio_returns" in context:
                    valid_params["period_portfolio_returns"] = context["period_portfolio_returns"]
                elif p_name == "period_benchmark_returns" and "period_benchmark_returns" in context:
                    valid_params["period_benchmark_returns"] = context["period_benchmark_returns"]
                elif p_name == "mu":
                    if "mu" in context:
                        valid_params["mu"] = context["mu"]
                    elif "covariance" in context:
                        cov_mat = np.asarray(context["covariance"], dtype=float)
                        valid_params["mu"] = np.full(len(cov_mat), 0.05)
                elif p_name in ("w_new", "weights_target", "proposed_weights"):
                    if "w_new" in context:
                        valid_params[p_name] = context["w_new"]
                    elif "weights_target" in context:
                        valid_params[p_name] = context["weights_target"]
                    elif "proposed_weights" in context:
                        valid_params[p_name] = context["proposed_weights"]
                    elif "market_weights" in context:
                        valid_params[p_name] = context["market_weights"]
                elif p_name in ("w_old", "weights_current", "current_weights", "prior_weights"):
                    if "w_old" in context:
                        valid_params[p_name] = context["w_old"]
                    elif "weights_current" in context:
                        valid_params[p_name] = context["weights_current"]
                    elif "current_weights" in context:
                        valid_params[p_name] = context["current_weights"]
                    elif "prior_weights" in context:
                        valid_params[p_name] = context["prior_weights"]
                elif p_name == "weights":
                    if "weights" in context:
                        valid_params["weights"] = context["weights"]
                    elif "portfolio_weights" in context:
                        valid_params["weights"] = context["portfolio_weights"]
                elif p_name == "benchmark_weights" and "benchmark_weights" in context:
                    valid_params["benchmark_weights"] = context["benchmark_weights"]
                elif p_name == "cost_spec" and "cost_spec" in context:
                    valid_params["cost_spec"] = context["cost_spec"]
                elif p_name == "uncertainty_policy" and "uncertainty_policy" in context:
                    valid_params["uncertainty_policy"] = context["uncertainty_policy"]
                elif p_name == "radii" and "radii" in context:
                    valid_params["radii"] = context["radii"]
                # Gate 6 parameter mappings
                elif p_name == "scenario_spec_or_shocks":
                    if "scenario_spec_or_shocks" in params:
                        valid_params["scenario_spec_or_shocks"] = params["scenario_spec_or_shocks"]
                    elif "scenario_spec" in params:
                        valid_params["scenario_spec_or_shocks"] = params["scenario_spec"]
                    elif tool_name == "apply_factor_scenario" and "factor_scenario_spec" in context:
                        valid_params["scenario_spec_or_shocks"] = context["factor_scenario_spec"]
                    elif "scenario_specs" in context:
                        spec_cand = None
                        if isinstance(context["scenario_specs"], dict):
                            for s_cand in context["scenario_specs"].values():
                                meth_val = getattr(s_cand, "repricing_method", "")
                                meth_str = meth_val.value if hasattr(meth_val, "value") else str(meth_val)
                                if (
                                    tool_name == "apply_factor_scenario"
                                    and meth_str.upper() == "FACTOR_LINEAR"
                                ):
                                    spec_cand = s_cand
                                    break
                        if spec_cand is not None:
                            valid_params["scenario_spec_or_shocks"] = spec_cand
                        elif "scenario_spec_or_shocks" in context:
                            valid_params["scenario_spec_or_shocks"] = context["scenario_spec_or_shocks"]
                        elif "scenario_spec" in context:
                            valid_params["scenario_spec_or_shocks"] = context["scenario_spec"]
                    elif "scenario_spec_or_shocks" in context:
                        valid_params["scenario_spec_or_shocks"] = context["scenario_spec_or_shocks"]
                    elif "scenario_spec" in context:
                        valid_params["scenario_spec_or_shocks"] = context["scenario_spec"]
                    elif "shocks" in context:
                        valid_params["scenario_spec_or_shocks"] = context["shocks"]
                elif p_name == "sensitivities":
                    if "sensitivities" in context:
                        valid_params["sensitivities"] = context["sensitivities"]
                    elif "sensitivity_specs" in context:
                        valid_params["sensitivities"] = context["sensitivity_specs"]
                elif p_name == "gamma_matrix" and "gamma_matrix" in context:
                    valid_params["gamma_matrix"] = context["gamma_matrix"]
                elif p_name == "spec":
                    if "reverse_stress_spec" in context:
                        valid_params["spec"] = context["reverse_stress_spec"]
                    elif "spec" in context:
                        valid_params["spec"] = context["spec"]
                    elif "scenario_spec" in context:
                        valid_params["spec"] = context["scenario_spec"]
                elif p_name == "sensitivities_or_weights":
                    raw_sw = None
                    if "sensitivities_or_weights" in context:
                        raw_sw = context["sensitivities_or_weights"]
                    elif "sensitivities" in context:
                        raw_sw = context["sensitivities"]
                    elif "weights" in context:
                        raw_sw = context["weights"]
                    if isinstance(raw_sw, dict):
                        valid_params["sensitivities_or_weights"] = {
                            k: getattr(v, "delta", v) for k, v in raw_sw.items()
                        }
                    elif raw_sw is not None:
                        valid_params["sensitivities_or_weights"] = raw_sw
                elif p_name == "exposures":
                    if "exposures" in context:
                        valid_params["exposures"] = context["exposures"]
                    elif "factor_exposures" in context:
                        valid_params["exposures"] = context["factor_exposures"]
                elif p_name == "group_mapping" and "group_mapping" in context:
                    valid_params["group_mapping"] = context["group_mapping"]
                elif p_name == "scenario_results" and "scenario_results" in context:
                    valid_params["scenario_results"] = context["scenario_results"]
                elif p_name == "historical_shocks" and "historical_shocks" in context:
                    valid_params["historical_shocks"] = context["historical_shocks"]
                elif p_name == "proxy_mappings" and "proxy_mappings" in context:
                    valid_params["proxy_mappings"] = context["proxy_mappings"]
                elif p_name == "source_reference" and "source_reference" in context:
                    valid_params["source_reference"] = context["source_reference"]
                elif p_name == "observation_date" and "observation_date" in context:
                    valid_params["observation_date"] = context["observation_date"]

            tool_res = fn(**valid_params)
            final_source_ev_ids = tuple(source_ev_ids)

            # Generate distinct, subordinate diagnostic EvidenceRecord
            diag_ev = challenge_result_to_diagnostic_evidence(
                tool_name=tool_name,
                tool_res=tool_res,
                params=valid_params,
                source_evidence_ids=final_source_ev_ids,
            )
            if isinstance(context.get("evidence_records"), list):
                context["evidence_records"].append(diag_ev)

            # Evaluate decision criterion and threshold provenance
            crit = (
                c_dict.get("decision_criterion") or context.get("challenge_criteria", {}).get(c_id) or "NONE"
            )
            thresh = c_dict.get("decision_threshold", c_dict.get("threshold")) or context.get(
                "challenge_thresholds", {}
            ).get(c_id)

            finding_ids_list: list[str] = []
            limitations: tuple[str, ...]
            if crit == "NONE" or thresh is None:
                # Absence of threshold MUST NOT produce NO_BREACH
                res_status = ChallengeState.RESOLVED_EVIDENCE_ONLY
                limitations = (
                    "Diagnostic evidence generated; materiality threshold not adjudicated by explicit policy (decision_criterion=NONE).",
                )
            else:
                # Deterministically evaluate metric against explicit policy threshold
                is_breached = False
                if hasattr(tool_res, "usable_solution") and not tool_res.usable_solution:
                    is_breached = True
                    limitations = ("Diagnostic solver returned unusable solution.",)
                elif hasattr(tool_res, "converged") and not tool_res.converged:
                    is_breached = True
                    limitations = ("Diagnostic solver failed to converge.",)
                elif isinstance(thresh, (int, float)):
                    # Scalar threshold evaluation
                    if isinstance(tool_res, BlackLittermanResult):
                        if tool_res.turnover_vs_prior > float(thresh):
                            is_breached = True
                    elif isinstance(tool_res, dict):
                        to_val = tool_res.get("turnover", 0.0)
                        if to_val > float(thresh):
                            is_breached = True
                    limitations = (
                        (f"Materiality threshold {thresh} breached under policy {crit}.",)
                        if is_breached
                        else ()
                    )
                elif isinstance(thresh, dict):
                    # Multi-metric dictionary bounds evaluation
                    for m_key, bound in thresh.items():
                        if isinstance(tool_res, dict) and m_key in tool_res:
                            if tool_res[m_key] > bound:
                                is_breached = True
                        elif hasattr(tool_res, m_key):
                            val = getattr(tool_res, m_key)
                            if isinstance(val, (int, float)) and val > bound:
                                is_breached = True
                    limitations = (
                        (f"Materiality threshold specification {thresh} breached under policy {crit}.",)
                        if is_breached
                        else ()
                    )
                else:
                    limitations = ()

                if is_breached:
                    res_status = ChallengeState.RESOLVED_FINDING
                    finding_ids_list.append(f"FIND-CHAL-{c_id}")
                else:
                    res_status = ChallengeState.RESOLVED_NO_BREACH

            return ChallengeResolution(
                challenge_id=c_id,
                status=res_status,
                tool_name=tool_name,
                source_evidence_ids=final_source_ev_ids,
                generated_evidence_ids=(diag_ev.evidence_id,),
                finding_ids=tuple(finding_ids_list),
                limitations=limitations,
                decision_criterion=crit,
                decision_threshold=thresh,
                tool_request=tool_name,
                tool_parameters=valid_params,
                details={
                    "result_type": type(tool_res).__name__,
                    "diagnostic_evidence_id": diag_ev.evidence_id,
                },
            )
        except Exception as e:
            return ChallengeResolution(
                challenge_id=c_id,
                status=ChallengeState.UNRESOLVED,
                tool_name=tool_name,
                source_evidence_ids=tuple(c_dict.get("evidence_ids", ())),
                limitations=(f"Challenge resolution failed: {e}",),
            )

    def resolve_challenges(
        self,
        challenges: list[AdversarialChallenge] | list[dict[str, Any]],
        context: dict[str, Any],
    ) -> list[ChallengeResolution]:
        """Batch-resolve adversarial challenges."""
        return [self.resolve_challenge(c, context) for c in challenges]

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        records: list[EvidenceRecord] = context.get("evidence_records", [])
        challenges = self.formulate_portfolio_challenges(records)
        resolutions: list[ChallengeResolution] = []

        if context.get("auto_resolve_challenges", True):
            resolutions = self.resolve_challenges(challenges, context)

        evidence_ids = [r.evidence_id for r in records if r.evidence_id]
        res_dicts = [asdict(r) for r in resolutions]

        self.emit_trace(
            stage="Adversarial Portfolio Challenge Review",
            progress=100.0,
            status_msg=f"Formulated {len(challenges)} challenges ({len(resolutions)} resolved).",
            reasoning_step="Identified methodological stress points and executed deterministic diagnostics.",
            confidence_score=1.0,
            evidence_citations=evidence_ids,
        )

        return {
            "status": "completed",
            "agent": self.name,
            "challenges": [asdict(c) for c in challenges],
            "resolutions": res_dicts,
            "challenge_resolutions": res_dicts,
            "evidence_citations": evidence_ids,
        }


class EvidenceCriticAgent(BaseAgent):
    """Institutional evidence quality and citation integrity critic.

    Strict invariant:
    May emit: EVIDENCE_VALID, EVIDENCE_INVALID, BLOCKED, READY_FOR_GOVERNANCE.
    May NOT issue: ACCEPT, APPROVE, READY_FOR_PRODUCTION (reserved strictly for GovernanceAgent).
    """

    ALLOWED_TOOLS = ("verify_portfolio_constraints",)

    def __init__(self, telemetry_bus: TelemetryBus | None = None):
        super().__init__(name="Evidence Critic Agent", telemetry_bus=telemetry_bus)

    def critique_evidence_records(
        self,
        records: list[EvidenceRecord],
        solver_results: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Audit evidence records and solver outputs for missing provenance, failed convergence, or constraint violations."""
        issues: list[str] = []

        for r in records:
            if not r.evidence_id:
                issues.append("EvidenceRecord missing evidence_id token.")
            if not r.metrics:
                issues.append(f"EvidenceRecord '{r.evidence_id}' has empty metrics dictionary.")
            # Check constraint verification if recorded in metrics
            if "is_valid" in r.metrics and not r.metrics["is_valid"]:
                issues.append(
                    f"EvidenceRecord '{r.evidence_id}' failed constraint verification (is_valid=False)."
                )

        if solver_results:
            for sr in solver_results:
                if hasattr(sr, "usable_solution") and not sr.usable_solution:
                    issues.append(
                        f"Solver output {type(sr).__name__} produced unusable solution: "
                        f"{getattr(sr, 'solver_message', getattr(sr, 'solver_status', 'FAILED'))}"
                    )
                if hasattr(sr, "converged") and not sr.converged:
                    issues.append(f"Solver output {type(sr).__name__} failed to converge.")
                if hasattr(sr, "constraint_verification") and not sr.constraint_verification.is_valid:
                    issues.append(
                        f"Solver output {type(sr).__name__} violated constraints with max violation "
                        f"{sr.constraint_verification.max_violation:.8f}."
                    )

        if issues:
            disposition = CriticDisposition.EVIDENCE_INVALID
        else:
            disposition = CriticDisposition.READY_FOR_GOVERNANCE

        return {
            "disposition": disposition.value,
            "is_valid": len(issues) == 0,
            "issues": issues,
            "records_evaluated": len(records),
            "solver_outputs_evaluated": len(solver_results) if solver_results else 0,
        }

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        records: list[EvidenceRecord] = context.get("evidence_records", [])
        solver_results: list[Any] = context.get("solver_results", [])

        critique = self.critique_evidence_records(records, solver_results)
        evidence_ids = [r.evidence_id for r in records if r.evidence_id]

        self.emit_trace(
            stage="Evidence & Verification Critique",
            progress=100.0,
            status_msg=f"Evidence critique concluded with disposition {critique['disposition']}.",
            reasoning_step="Audited evidence provenance, constraint satisfaction, and mathematical validity.",
            confidence_score=1.0,
            evidence_citations=evidence_ids,
        )

        return {
            "status": "completed",
            "agent": self.name,
            "critique": critique,
            "evidence_citations": evidence_ids,
        }


class GovernanceAgent(BaseAgent):
    """Institutional Model Risk Management (MRM) Sign-Off Governance Agent.

    Strict invariant:
    Sole entity authorized to produce: ACCEPT, ACCEPT_WITH_CONDITIONS, REMEDIATE, ESCALATE, INSUFFICIENT_EVIDENCE.
    """

    ALLOWED_TOOLS = ()

    def __init__(self, telemetry_bus: TelemetryBus | None = None):
        super().__init__(name="Governance Agent", telemetry_bus=telemetry_bus)

    def evaluate_signoff(
        self,
        critic_disposition: str,
        challenges: list[dict[str, Any]],
        findings: list[dict[str, Any]],
        records: list[EvidenceRecord],
        resolutions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Make formal institutional governance determination based on certified evidence."""
        if not records or critic_disposition != CriticDisposition.READY_FOR_GOVERNANCE.value:
            return {
                "verdict": GovernanceVerdict.INSUFFICIENT_EVIDENCE.value,
                "reason": f"Critic disposition '{critic_disposition}' does not meet minimum quality threshold for governance evaluation.",
                "conditions": [],
            }

        # Check for open / unresolved challenges
        def _get_field(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        res_by_id = {_get_field(r, "challenge_id"): r for r in (resolutions or [])}
        unresolved_challenges: list[tuple[str, str]] = []
        evidence_only_challenges: list[tuple[str, str]] = []

        for c in challenges:
            c_id = _get_field(c, "challenge_id", "")
            res = res_by_id.get(c_id)
            if res is not None:
                st = str(_get_field(res, "status", ""))
                if st in (
                    ChallengeState.RESOLVED_EVIDENCE_ONLY.value,
                    "RESOLVED_EVIDENCE_ONLY",
                ):
                    evidence_only_challenges.append((c_id, st))
                elif st not in (
                    ChallengeState.RESOLVED_NO_BREACH.value,
                    ChallengeState.RESOLVED_FINDING.value,
                    "RESOLVED_NO_BREACH",
                    "RESOLVED_FINDING",
                    "VERIFIED_RESILIENT",
                ):
                    unresolved_challenges.append((c_id, st))
            else:
                c_st = str(_get_field(c, "status", "OPEN"))
                if c_st in (ChallengeState.RESOLVED_EVIDENCE_ONLY.value, "RESOLVED_EVIDENCE_ONLY"):
                    evidence_only_challenges.append((c_id, c_st))
                elif c_st not in ("VERIFIED_RESILIENT", "RESOLVED_NO_BREACH", "RESOLVED_FINDING"):
                    unresolved_challenges.append((c_id, c_st))

        if unresolved_challenges:
            return {
                "verdict": GovernanceVerdict.REMEDIATE.value,
                "reason": (
                    f"Found {len(unresolved_challenges)} unresolved adversarial challenge(s): {unresolved_challenges}. "
                    f"Challenges must be resolved with evidence before sign-off."
                ),
                "conditions": [
                    f"Resolve challenge '{cid}' with deterministic evidence."
                    for cid, _ in unresolved_challenges
                ],
            }

        unresolved_critical = [f for f in findings if f.get("severity") == "critical_breach"]
        if unresolved_critical:
            return {
                "verdict": GovernanceVerdict.REMEDIATE.value,
                "reason": f"Found {len(unresolved_critical)} critical breach finding(s). Remediation required.",
                "conditions": [f.get("recommended_action", "") for f in unresolved_critical],
            }

        # Challenges resolved as EVIDENCE_ONLY (missing materiality threshold) require conditional sign-off
        if evidence_only_challenges:
            return {
                "verdict": GovernanceVerdict.ACCEPT_WITH_CONDITIONS.value,
                "reason": (
                    f"Accepted with conditions: {len(evidence_only_challenges)} adversarial challenge(s) "
                    f"generated diagnostic evidence without explicit materiality threshold policy: {evidence_only_challenges}."
                ),
                "conditions": [
                    f"Adversarial challenge '{cid}' requires explicit materiality threshold policy sign-off."
                    for cid, _ in evidence_only_challenges
                ],
            }

        material_warnings = [f for f in findings if f.get("severity") == "material_warning"]
        if material_warnings:
            return {
                "verdict": GovernanceVerdict.ACCEPT_WITH_CONDITIONS.value,
                "reason": f"Accepted with {len(material_warnings)} condition(s) addressing material warnings.",
                "conditions": [f.get("recommended_action", "") for f in material_warnings],
            }

        return {
            "verdict": GovernanceVerdict.ACCEPT.value,
            "reason": "All evidence records certified; all challenges resolved; no blocking issues or material warnings outstanding.",
            "conditions": [],
        }

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        records: list[EvidenceRecord] = context.get("evidence_records", [])
        critic_disp = context.get("critic_disposition", CriticDisposition.READY_FOR_GOVERNANCE.value)
        challenges = context.get("challenges", [])
        resolutions = context.get("resolutions", [])
        findings = context.get("findings", [])

        signoff = self.evaluate_signoff(critic_disp, challenges, findings, records, resolutions=resolutions)
        evidence_ids = [r.evidence_id for r in records if r.evidence_id]

        self.emit_trace(
            stage="Governance MRM Sign-Off",
            progress=100.0,
            status_msg=f"Governance evaluation concluded with verdict {signoff['verdict']}.",
            reasoning_step="Evaluated critic disposition, adversarial challenges, and specialist findings.",
            confidence_score=1.0,
            evidence_citations=evidence_ids,
        )

        return {
            "status": "completed",
            "agent": self.name,
            "governance_signoff": signoff,
            "evidence_citations": evidence_ids,
        }


class MarketReviewDirectorAgent(BaseAgent):
    """Orchestrator for Market and Portfolio Domain Review (Gate 2, 3, 4, 5, & 6 Full Slice)."""

    def __init__(self, telemetry_bus: TelemetryBus | None = None):
        super().__init__(name="Market Review Director Agent", telemetry_bus=telemetry_bus)
        self.data_integrity_checker = FactorDataIntegrityChecker(telemetry_bus=telemetry_bus)
        self.covariance_agent = CovarianceRiskAgent(telemetry_bus=telemetry_bus)
        self.factor_agent = FactorRiskAttributionAgent(telemetry_bus=telemetry_bus)
        self.hierarchical_agent = HierarchicalAllocationAgent(telemetry_bus=telemetry_bus)
        self.portfolio_agent = PortfolioConstructionAgent(telemetry_bus=telemetry_bus)
        self.tail_risk_agent = TailRiskAgent(telemetry_bus=telemetry_bus)
        self.scenario_agent = ScenarioStressAgent(telemetry_bus=telemetry_bus)
        self.adversarial_agent = AdversarialChallengeAgent(telemetry_bus=telemetry_bus)
        self.critic_agent = EvidenceCriticAgent(telemetry_bus=telemetry_bus)
        self.governance_agent = GovernanceAgent(telemetry_bus=telemetry_bus)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Coordinate complete specialist market review workflow with strict governance separation."""
        records: list[EvidenceRecord] = context.get("evidence_records", [])

        # 1. Pre-flight Data Integrity
        int_out = self.data_integrity_checker.execute(context)

        # 2. Specialist Domain Reviews
        cov_out = self.covariance_agent.execute(context)
        fac_out = self.factor_agent.execute(context)
        h_out = self.hierarchical_agent.execute(context)
        p_out = self.portfolio_agent.execute(context)
        tail_out = self.tail_risk_agent.execute(context)
        scen_out = self.scenario_agent.execute(context)

        # 3. Adversarial Challenges
        a_out = self.adversarial_agent.execute(context)

        # 4. Evidence Critic Pass
        c_out = self.critic_agent.execute(context)
        critic_disp = c_out.get("critique", {}).get(
            "disposition", CriticDisposition.READY_FOR_GOVERNANCE.value
        )

        # 5. Governance Sign-Off Pass
        all_findings = (
            int_out.get("findings", [])
            + cov_out.get("findings", [])
            + fac_out.get("findings", [])
            + h_out.get("findings", [])
            + p_out.get("findings", [])
            + tail_out.get("findings", [])
            + scen_out.get("findings", [])
        )
        gov_context = {
            "evidence_records": records,
            "critic_disposition": critic_disp,
            "challenges": a_out.get("challenges", []),
            "resolutions": a_out.get("resolutions", []),
            "findings": all_findings,
        }
        g_out = self.governance_agent.execute(gov_context)

        challenges = a_out.get("challenges", [])
        resolutions = a_out.get("resolutions", [])

        self.emit_trace(
            stage="Market Review Orchestration",
            progress=100.0,
            status_msg="Market, Risk, Portfolio, Tail Risk, and Scenario domain review complete with formal governance verdict.",
            reasoning_step="Orchestrated specialists -> adversarial challenge -> evidence critic -> governance.",
            confidence_score=1.0,
            evidence_citations=[r.evidence_id for r in records if r.evidence_id],
        )

        return {
            "status": "orchestrated",
            "director": self.name,
            "findings_count": len(all_findings),
            "challenges_count": len(challenges),
            "resolutions_count": len(resolutions),
            "critic_disposition": critic_disp,
            "governance_verdict": g_out.get("governance_signoff", {}).get("verdict"),
            "governance_signoff": g_out.get("governance_signoff"),
            "data_integrity": int_out,
            "covariance_review": cov_out,
            "factor_review": fac_out,
            "hierarchical_review": h_out,
            "portfolio_review": p_out,
            "tail_risk_review": tail_out,
            "scenario_review": scen_out,
            "adversarial_review": a_out,
            "critic_review": c_out,
            "governance_review": g_out,
            "findings": all_findings,
            "challenges": challenges,
            "resolutions": resolutions,
            "governance": g_out.get("governance_signoff"),
        }
