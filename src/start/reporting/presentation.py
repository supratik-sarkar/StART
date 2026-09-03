"""Typed Presentation Layer for Institutional Risk & Portfolio Reviews.

Core Architectural Invariants:
1. Pure Presentation: never recomputes analytical truth. Consumes already-generated
   EvidenceRecord, ArtifactRecord, and TraceEvent objects.
2. Provenance Preservation: every quantitative metric row retains test_id, metric,
   value, unit/basis, status, evidence_id, and optional artifact_id.
3. Stable Web/CLI Contract: ReviewPresentationModel.to_dict() is the shared contract
   for Rich CLI, future Ollama UI, and WebLLM browser frontend.
4. Structured Domain Blocks:
   - DATA
   - PREPROCESSING
   - MODEL_CONSTRUCTION / PORTFOLIO_CONSTRUCTION
   - TUNING_OPTIMIZATION / SENSITIVITY
   - PERFORMANCE
   - SENSITIVITY
   - EXPLAINABILITY
   - STRESS_TAIL
   - GOVERNANCE
   - AGENT_ORCHESTRATION
   (Blocks are omitted when not applicable to the active review domains).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from start.core.schemas import EvidenceRecord


@dataclass
class QuantitativeMetricRow:
    """A single audit-grounded quantitative row in a presentation block."""

    test_id: str
    metric: str
    value: Any
    unit: str = ""
    status: str = "RECORDED"
    evidence_id: str = ""
    artifact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "status": self.status,
            "evidence_id": self.evidence_id,
            "artifact_id": self.artifact_id,
        }


@dataclass
class PresentationBlock:
    """A typed presentation section scoped to a domain workflow dimension."""

    block_id: str
    title: str
    domain: str
    rows: list[QuantitativeMetricRow] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": self.block_id,
            "title": self.title,
            "domain": self.domain,
            "rows": [r.to_dict() for r in self.rows],
            "summary": self.summary,
            "artifacts": self.artifacts,
        }


@dataclass
class ReviewPresentationModel:
    """Aggregate presentation container shared across Rich CLI and future Web UI."""

    run_id: str
    mode: str
    domains: list[str]
    materiality: str
    lifecycle: str
    blocks: dict[str, PresentationBlock] = field(default_factory=dict)
    governance_disposition: str = "ACCEPT"
    attestation_seal_merkle_root: str = ""
    orchestration_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "domains": self.domains,
            "materiality": self.materiality,
            "lifecycle": self.lifecycle,
            "governance_disposition": self.governance_disposition,
            "attestation_seal_merkle_root": self.attestation_seal_merkle_root,
            "blocks": {k: b.to_dict() for k, b in self.blocks.items()},
            "orchestration_events": self.orchestration_events,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)


def build_presentation_model(
    run_id: str,
    mode: str,
    domains: tuple[Any, ...],
    materiality: str,
    lifecycle: str,
    records: list[EvidenceRecord],
    artifacts_by_checkpoint: dict[str, list[Any]] | None = None,
    governance_disposition: str = "ACCEPT",
    attestation_seal_merkle_root: str = "",
    orchestration_events: list[dict[str, Any]] | None = None,
) -> ReviewPresentationModel:
    """Construct a typed ReviewPresentationModel from deterministic evidence and artifacts."""
    domains_str = [str(getattr(d, "value", d)) for d in domains]
    model = ReviewPresentationModel(
        run_id=run_id,
        mode=mode,
        domains=domains_str,
        materiality=materiality,
        lifecycle=lifecycle,
        governance_disposition=governance_disposition,
        attestation_seal_merkle_root=attestation_seal_merkle_root,
        orchestration_events=orchestration_events or [],
    )

    all_artifacts = [art for arts in (artifacts_by_checkpoint or {}).values() for art in arts]

    has_market = "market" in domains_str
    has_predictive = "predictive" in domains_str

    # ----------------------------------------------------------------------- #
    # MARKET PRESENTATION BLOCKS
    # ----------------------------------------------------------------------- #
    if has_market:
        # 1. PORTFOLIO_CONSTRUCTION Block
        port_rows: list[QuantitativeMetricRow] = []
        port_summary: dict[str, Any] = {}
        port_arts: list[dict[str, Any]] = []

        # Equal Weight / Risk statistics
        if "portfolio.risk_statistics" in rec_map(records):
            r = rec_map(records)["portfolio.risk_statistics"]
            m = r.metrics
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Portfolio Annualised Volatility",
                    value=m.get("annualised_volatility"),
                    unit="annualized_vol",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Periodic Volatility",
                    value=m.get("volatility"),
                    unit="periodic_vol",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        # Mean-Variance / MinVar
        if "portfolio.mean_variance" in rec_map(records):
            r = rec_map(records)["portfolio.mean_variance"]
            m = r.metrics
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Mean-Variance Annualised Sharpe",
                    value=m.get("sharpe_annualised"),
                    unit="sharpe",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="MVO Active Positions",
                    value=m.get("n_active_positions"),
                    unit="count",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="MVO Solver Converged",
                    value=m.get("converged"),
                    unit="boolean",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        # HRP
        if "portfolio.hierarchical_risk_parity" in rec_map(records):
            r = rec_map(records)["portfolio.hierarchical_risk_parity"]
            m = r.metrics
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="HRP Max Weight",
                    value=m.get("max_weight"),
                    unit="weight",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="HRP Effective N Positions",
                    value=m.get("effective_n_positions"),
                    unit="effective_n",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="HRP Herfindahl Concentration",
                    value=m.get("herfindahl"),
                    unit="hhi",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            port_summary["hrp_linkage"] = m.get("linkage_method")
            port_summary["hrp_quasi_diagonal_order"] = m.get("quasi_diagonal_order")

        # HERC
        if "portfolio.herc" in rec_map(records):
            r = rec_map(records)["portfolio.herc"]
            m = r.metrics
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="HERC Effective N Positions",
                    value=m.get("effective_n_positions"),
                    unit="effective_n",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="HERC Annualised Volatility",
                    value=m.get("portfolio_volatility_annualised"),
                    unit="annualized_vol",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        # Black-Litterman
        if "portfolio.black_litterman" in rec_map(records):
            r = rec_map(records)["portfolio.black_litterman"]
            m = r.metrics
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Black-Litterman Posterior Volatility",
                    value=m.get("posterior_volatility_annualised"),
                    unit="annualized_vol",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Black-Litterman Turnover vs Prior",
                    value=m.get("turnover_vs_prior"),
                    unit="turnover",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        # CVaR
        if "portfolio.cvar_optimization" in rec_map(records):
            r = rec_map(records)["portfolio.cvar_optimization"]
            m = r.metrics
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="CVaR (Expected Shortfall)",
                    value=m.get("cvar_annualised"),
                    unit="annualized_loss",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="CVaR Tail Scenarios",
                    value=m.get("tail_scenario_count"),
                    unit="count",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        # Constrained Optimization
        if "portfolio.constrained_optimization" in rec_map(records):
            r = rec_map(records)["portfolio.constrained_optimization"]
            m = r.metrics
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Constraint Audit Is Valid",
                    value=m.get("is_valid"),
                    unit="boolean",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            port_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Max Constraint Violation",
                    value=m.get("max_violation"),
                    unit="violation_magnitude",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        # Collect portfolio artifacts
        for art in all_artifacts:
            spec = getattr(art, "spec", None)
            tid = getattr(spec, "test_id", "") if spec else ""
            if "portfolio." in tid or "asset_weights" in getattr(spec, "artifact_type", ""):
                port_arts.append(art_summary(art))

        model.blocks["PORTFOLIO_CONSTRUCTION"] = PresentationBlock(
            block_id="PORTFOLIO_CONSTRUCTION",
            title="Portfolio Construction & Method Comparison",
            domain="market",
            rows=port_rows,
            summary=port_summary,
            artifacts=port_arts,
        )

        # 2. HRP_SHOWCASE Block
        hrp_rows: list[QuantitativeMetricRow] = []
        hrp_summary: dict[str, Any] = {}
        hrp_arts: list[dict[str, Any]] = []

        if "portfolio.hierarchical_risk_parity" in rec_map(records):
            r = rec_map(records)["portfolio.hierarchical_risk_parity"]
            m = r.metrics
            hrp_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Linkage Method",
                    value=m.get("linkage_method", "single"),
                    unit="algorithm",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            hrp_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Quasi-Diagonal Order",
                    value=str(m.get("quasi_diagonal_order", "—"))[:40] + "...",
                    unit="ordering",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            hrp_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Portfolio Variance (Periodic)",
                    value=m.get("portfolio_variance_periodic"),
                    unit="variance",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            if "cophenetic_correlation" in m:
                hrp_rows.append(
                    QuantitativeMetricRow(
                        test_id=r.test_id,
                        metric="Cophenetic Correlation",
                        value=m.get("cophenetic_correlation"),
                        unit="correlation",
                        status=str(r.status).upper(),
                        evidence_id=r.evidence_id,
                    )
                )

        for art in all_artifacts:
            spec = getattr(art, "spec", None)
            atype = getattr(spec, "artifact_type", "") if spec else ""
            if "dendrogram" in atype or "seriated" in atype or "hrp" in atype:
                hrp_arts.append(art_summary(art))

        model.blocks["HRP_SHOWCASE"] = PresentationBlock(
            block_id="HRP_SHOWCASE",
            title="Hierarchical Risk Parity (HRP) Architecture & Topology",
            domain="market",
            rows=hrp_rows,
            summary=hrp_summary,
            artifacts=hrp_arts,
        )

        # 3. FACTOR_ATTRIBUTION Block
        attr_rows: list[QuantitativeMetricRow] = []
        attr_summary: dict[str, Any] = {}
        attr_arts: list[dict[str, Any]] = []

        if "attribution.factor_return_estimation" in rec_map(records):
            r = rec_map(records)["attribution.factor_return_estimation"]
            m = r.metrics
            attr_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Estimated Factors Count",
                    value=m.get("n_factors"),
                    unit="count",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            attr_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Estimation Periods",
                    value=f"{m.get('n_periods_estimated')}/{m.get('n_periods_total')}",
                    unit="periods",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        if "attribution.return_attribution" in rec_map(records):
            r = rec_map(records)["attribution.return_attribution"]
            m = r.metrics
            attr_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Max Return Reconciliation Error",
                    value=m.get("max_abs_reconciliation_error"),
                    unit="tolerance",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            attr_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Factor Return Source",
                    value=m.get("factor_return_source", "supplied"),
                    unit="provenance",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        if "attribution.risk_attribution" in rec_map(records):
            r = rec_map(records)["attribution.risk_attribution"]
            m = r.metrics
            attr_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Factor Model to Empirical Ratio",
                    value=m.get("factor_model_to_empirical_ratio"),
                    unit="ratio",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            attr_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Attribution Variance Shortfall",
                    value=m.get("variance_shortfall"),
                    unit="percentage",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        if "attribution.risk_change_decomposition" in rec_map(records):
            r = rec_map(records)["attribution.risk_change_decomposition"]
            m = r.metrics
            attr_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Risk Change Total Delta Variance",
                    value=m.get("delta_total_variance_periodic"),
                    unit="variance_delta",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        for art in all_artifacts:
            spec = getattr(art, "spec", None)
            atype = getattr(spec, "artifact_type", "") if spec else ""
            if "factor" in atype or "attribution" in atype:
                attr_arts.append(art_summary(art))

        model.blocks["FACTOR_ATTRIBUTION"] = PresentationBlock(
            block_id="FACTOR_ATTRIBUTION",
            title="Factor Risk Modeling & Return Attribution",
            domain="market",
            rows=attr_rows,
            summary=attr_summary,
            artifacts=attr_arts,
        )

        # 4. COVARIANCE_STRUCTURE Block
        cov_rows: list[QuantitativeMetricRow] = []
        cov_summary: dict[str, Any] = {}
        cov_arts: list[dict[str, Any]] = []

        if "covariance.empirical" in rec_map(records):
            r = rec_map(records)["covariance.empirical"]
            m = r.metrics
            cov_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Empirical Covariance PSD",
                    value=m.get("is_psd", True),
                    unit="boolean",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        if "covariance.ledoit_wolf_shrinkage" in rec_map(records):
            r = rec_map(records)["covariance.ledoit_wolf_shrinkage"]
            m = r.metrics
            cov_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Ledoit-Wolf Shrinkage Intensity",
                    value=m.get("shrinkage_intensity"),
                    unit="intensity",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            cov_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Condition Number Before",
                    value=m.get("condition_number_before"),
                    unit="kappa",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            cov_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Condition Number After",
                    value=m.get("condition_number_after"),
                    unit="kappa",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        if "portfolio.covariance_conditioning" in rec_map(records):
            r = rec_map(records)["portfolio.covariance_conditioning"]
            m = r.metrics
            cov_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Covariance Condition Number",
                    value=m.get("condition_number"),
                    unit="kappa",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            cov_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Full Rank Status",
                    value=m.get("full_rank"),
                    unit="boolean",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        if "validation.regem_structural" in rec_map(records):
            r = rec_map(records)["validation.regem_structural"]
            m = r.metrics
            cov_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="RegEM Structural Criteria Met",
                    value=m.get("classification", "all cells met"),
                    unit="validation",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        for art in all_artifacts:
            spec = getattr(art, "spec", None)
            atype = getattr(spec, "artifact_type", "") if spec else ""
            if "covariance" in atype or "correlation" in atype:
                cov_arts.append(art_summary(art))

        model.blocks["COVARIANCE_STRUCTURE"] = PresentationBlock(
            block_id="COVARIANCE_STRUCTURE",
            title="Covariance Conditioning, Shrinkage & Missing Data",
            domain="market",
            rows=cov_rows,
            summary=cov_summary,
            artifacts=cov_arts,
        )

        # 5. STRESS_TAIL Block
        tail_rows: list[QuantitativeMetricRow] = []
        tail_summary: dict[str, Any] = {}
        tail_arts: list[dict[str, Any]] = []

        if "traded_risk.var_exceptions" in rec_map(records):
            r = rec_map(records)["traded_risk.var_exceptions"]
            m = r.metrics
            tail_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="VaR Exception Count",
                    value=m.get("n_exceptions"),
                    unit="count",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            tail_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Expected Exceptions",
                    value=m.get("expected_exceptions"),
                    unit="count",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            tail_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Exception Rate",
                    value=m.get("exception_rate"),
                    unit="rate",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        if "traded_risk.var_kupiec_pof" in rec_map(records):
            r = rec_map(records)["traded_risk.var_kupiec_pof"]
            m = r.metrics
            tail_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Kupiec POF p-value",
                    value=m.get("p_value"),
                    unit="p_value",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            tail_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Kupiec Test Rejected",
                    value=m.get("rejected"),
                    unit="boolean",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        if "traded_risk.var_christoffersen_independence" in rec_map(records):
            r = rec_map(records)["traded_risk.var_christoffersen_independence"]
            m = r.metrics
            tail_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Christoffersen Independence Rejected",
                    value=m.get("rejected"),
                    unit="boolean",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        if "scenario.linear_return" in rec_map(records):
            r = rec_map(records)["scenario.linear_return"]
            m = r.metrics
            tail_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Scenario Shock P&L",
                    value=m.get("portfolio_pnl_periodic"),
                    unit="pnl",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        if "scenario.reverse_stress" in rec_map(records):
            r = rec_map(records)["scenario.reverse_stress"]
            m = r.metrics
            tail_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Reverse Stress Minimal Distance",
                    value=m.get("distance"),
                    unit="mahalanobis_distance",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )
            tail_rows.append(
                QuantitativeMetricRow(
                    test_id=r.test_id,
                    metric="Reverse Stress Target Loss Gap",
                    value=m.get("target_loss_gap"),
                    unit="gap",
                    status=str(r.status).upper(),
                    evidence_id=r.evidence_id,
                )
            )

        for art in all_artifacts:
            spec = getattr(art, "spec", None)
            atype = getattr(spec, "artifact_type", "") if spec else ""
            if "backtest" in atype or "scenario" in atype or "stress" in atype:
                tail_arts.append(art_summary(art))

        model.blocks["STRESS_TAIL"] = PresentationBlock(
            block_id="STRESS_TAIL",
            title="VaR Backtesting, Scenario Repricing & Reverse Stress",
            domain="market",
            rows=tail_rows,
            summary=tail_summary,
            artifacts=tail_arts,
        )

    # ----------------------------------------------------------------------- #
    # PREDICTIVE PRESENTATION BLOCKS
    # ----------------------------------------------------------------------- #
    if has_predictive:
        # DATA & PREPROCESSING
        pred_data_rows: list[QuantitativeMetricRow] = []
        for tid in ("data.quality", "data.imbalance", "data.leakage", "preprocessing.missingness"):
            if tid in rec_map(records):
                r = rec_map(records)[tid]
                for k, v in list(r.metrics.items())[:3]:
                    pred_data_rows.append(
                        QuantitativeMetricRow(
                            test_id=r.test_id,
                            metric=k,
                            value=v,
                            status=str(r.status).upper(),
                            evidence_id=r.evidence_id,
                        )
                    )

        if pred_data_rows:
            model.blocks["DATA_PREPROCESSING"] = PresentationBlock(
                block_id="DATA_PREPROCESSING",
                title="Data Quality, Missingness & Preprocessing Diagnostics",
                domain="predictive",
                rows=pred_data_rows,
            )

        # MODEL ARCHITECTURE & PARAMETERS
        arch_rows: list[QuantitativeMetricRow] = []
        for tid in ("model.architecture", "dl.architecture", "supervised.discrimination"):
            if tid in rec_map(records):
                r = rec_map(records)[tid]
                for k, v in list(r.metrics.items())[:3]:
                    arch_rows.append(
                        QuantitativeMetricRow(
                            test_id=r.test_id,
                            metric=k,
                            value=v,
                            status=str(r.status).upper(),
                            evidence_id=r.evidence_id,
                        )
                    )
        if arch_rows:
            model.blocks["MODEL_ARCHITECTURE"] = PresentationBlock(
                block_id="MODEL_ARCHITECTURE",
                title="Model Architecture, Parameters & Device Placement",
                domain="predictive",
                rows=arch_rows,
            )

        # TRAINING & HYPERPARAMETER TUNING
        tune_rows: list[QuantitativeMetricRow] = []
        for tid in ("model.tuning", "dl.training", "training.loss_history", "supervised.discrimination"):
            if tid in rec_map(records):
                r = rec_map(records)[tid]
                for k, v in list(r.metrics.items())[:3]:
                    tune_rows.append(
                        QuantitativeMetricRow(
                            test_id=r.test_id,
                            metric=k,
                            value=v,
                            status=str(r.status).upper(),
                            evidence_id=r.evidence_id,
                        )
                    )
        if tune_rows:
            model.blocks["TRAINING_TUNING"] = PresentationBlock(
                block_id="TRAINING_TUNING",
                title="Training History, Tuning & Overfitting Diagnostics",
                domain="predictive",
                rows=tune_rows,
            )

        # PERFORMANCE & DISCRIMINATION
        perf_rows: list[QuantitativeMetricRow] = []
        for tid in (
            "metrics.performance",
            "supervised.classification_metrics",
            "metrics.calibration",
            "supervised.discrimination",
        ):
            if tid in rec_map(records):
                r = rec_map(records)[tid]
                for k, v in list(r.metrics.items())[:4]:
                    perf_rows.append(
                        QuantitativeMetricRow(
                            test_id=r.test_id,
                            metric=k,
                            value=v,
                            status=str(r.status).upper(),
                            evidence_id=r.evidence_id,
                        )
                    )

        if perf_rows:
            model.blocks["PERFORMANCE"] = PresentationBlock(
                block_id="PERFORMANCE",
                title="Model Discrimination, Calibration & Decision Metrics",
                domain="predictive",
                rows=perf_rows,
            )

        # SENSITIVITY & ROBUSTNESS
        sens_rows: list[QuantitativeMetricRow] = []
        for tid in (
            "sensitivity.perturbation",
            "robustness.missingness",
            "stability.seed_dispersion",
            "supervised.discrimination",
        ):
            if tid in rec_map(records):
                r = rec_map(records)[tid]
                for k, v in list(r.metrics.items())[:3]:
                    sens_rows.append(
                        QuantitativeMetricRow(
                            test_id=r.test_id,
                            metric=k,
                            value=v,
                            status=str(r.status).upper(),
                            evidence_id=r.evidence_id,
                        )
                    )
        if sens_rows:
            model.blocks["SENSITIVITY"] = PresentationBlock(
                block_id="SENSITIVITY",
                title="Model Sensitivity, Perturbation & Robustness Verification",
                domain="predictive",
                rows=sens_rows,
            )

        # EXPLAINABILITY (XAI)
        xai_rows: list[QuantitativeMetricRow] = []
        for tid in (
            "explainability.importance",
            "explainability.shap",
            "xai.global_importance",
            "supervised.discrimination",
        ):
            if tid in rec_map(records):
                r = rec_map(records)[tid]
                for k, v in list(r.metrics.items())[:3]:
                    xai_rows.append(
                        QuantitativeMetricRow(
                            test_id=r.test_id,
                            metric=k,
                            value=v,
                            status=str(r.status).upper(),
                            evidence_id=r.evidence_id,
                        )
                    )

        if xai_rows:
            model.blocks["EXPLAINABILITY"] = PresentationBlock(
                block_id="EXPLAINABILITY",
                title="Feature Attribution & Model Explainability (XAI)",
                domain="predictive",
                rows=xai_rows,
            )

    # ----------------------------------------------------------------------- #
    # GOVERNANCE BLOCK
    # ----------------------------------------------------------------------- #
    gov_rows: list[QuantitativeMetricRow] = [
        QuantitativeMetricRow(
            test_id="governance.disposition",
            metric="Final Governance Disposition",
            value=governance_disposition,
            status="FINAL",
            evidence_id="GOV-FINAL",
        ),
        QuantitativeMetricRow(
            test_id="governance.attestation_seal",
            metric="Attestation Seal Merkle Root",
            value=attestation_seal_merkle_root,
            status="SIGNED",
            evidence_id="SEAL-ROOT",
        ),
    ]
    model.blocks["GOVERNANCE"] = PresentationBlock(
        block_id="GOVERNANCE",
        title="Model Governance, Lifecycle & Cryptographic Attestation",
        domain="governance",
        rows=gov_rows,
        summary={
            "disposition": governance_disposition,
            "merkle_root": attestation_seal_merkle_root,
            "evidence_count": len(records),
        },
    )

    return model


def rec_map(records: list[EvidenceRecord]) -> dict[str, EvidenceRecord]:
    """Helper to map test_id to EvidenceRecord."""
    return {r.test_id: r for r in records}


def art_summary(art: Any) -> dict[str, Any]:
    """Helper to extract clean summary from an ArtifactRecord."""
    spec = getattr(art, "spec", None)
    return {
        "artifact_id": getattr(art, "artifact_id", "ART"),
        "title": getattr(spec, "title", getattr(art, "title", "Artifact"))
        if spec
        else getattr(art, "title", "Artifact"),
        "artifact_type": getattr(spec, "artifact_type", getattr(art, "artifact_type", "unknown"))
        if spec
        else "unknown",
        "format": getattr(art, "rendering_format", getattr(art, "format", "svg")),
        "file_path": getattr(art, "file_path", None),
        "evidence_ids": list(getattr(art, "evidence_ids", ())),
    }
