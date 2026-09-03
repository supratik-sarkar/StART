"""Deterministic cross-analytical rules for StART committee reasoning.

Invariants:
- Zero LLM arithmetic: all numerical metrics are extracted from deterministic EvidenceRecords.
- Zero hard-coded materiality thresholds (e.g., no hard-coded p < 0.05 logic; rules consume
  pre-registered gamma_test / decision fields).
- Non-normative classification: measurements without policy thresholds remain EVIDENCE_ONLY.
- CONTRADICTS is strictly reserved for genuine logical/algebraic incompatibilities on the SAME contract.
- Algebraic dominance is defined via argmax(|metric|) with deterministic tie-breaking.
"""

from __future__ import annotations

from typing import Any, cast

from start.core.schemas import EvidenceRecord
from start.evidence.claims import (
    AnalyticalClaim,
    ClaimStatus,
    ClaimType,
    new_claim_id,
)
from start.evidence.graph import (
    EvidenceEdge,
    RelationshipType,
)


def _argmax_abs(values: dict[str, float]) -> tuple[str, float]:
    """Deterministically find the key with maximum absolute value, sorting ties alphabetically."""
    if not values:
        return "", 0.0
    sorted_items = sorted(values.items(), key=lambda kv: (-abs(float(kv[1])), kv[0]))
    return sorted_items[0][0], float(sorted_items[0][1])


def eval_var_frequency_vs_independence(
    kupiec_record: EvidenceRecord,
    christoffersen_record: EvidenceRecord,
) -> tuple[AnalyticalClaim, list[EvidenceEdge]]:
    """Evaluate Kupiec unconditional coverage vs Christoffersen independence backtest evidence."""
    gamma_test: float = float(
        christoffersen_record.params.get("gamma_test")
        or christoffersen_record.metrics.get("gamma_test")
        or 0.05
    )
    reject_unc = bool(
        kupiec_record.metrics.get(
            "reject_unconditional_coverage",
            kupiec_record.metrics.get("reject", False),
        )
    )
    reject_ind = bool(
        christoffersen_record.metrics.get(
            "reject_independence",
            christoffersen_record.metrics.get("reject", False),
        )
    )

    metric_paths = {
        "kupiec_reject": f"{kupiec_record.evidence_id}.metrics.reject_unconditional_coverage",
        "christoffersen_reject": f"{christoffersen_record.evidence_id}.metrics.reject_independence",
        "gamma_test": f"{christoffersen_record.evidence_id}.metrics.gamma_test",
    }

    if not reject_unc and reject_ind:
        statement = (
            f"Unconditional coverage frequency adequacy does not imply exception independence "
            f"under recorded gamma_test={gamma_test} (Kupiec reject={reject_unc}, "
            f"Christoffersen independence reject={reject_ind})."
        )
        claim = AnalyticalClaim(
            claim_id=new_claim_id(),
            source_evidence_ids=(kupiec_record.evidence_id, christoffersen_record.evidence_id),
            claim_type=ClaimType.UNRESOLVED_RISK,
            domain="tail_risk",
            metric_paths=metric_paths,
            status=ClaimStatus.EVIDENCE_ONLY,
            statement=statement,
            limitations=(
                "Tail clustering challenges a benign unconditional frequency summary.",
                "Requires risk committee review of conditional tail dependence.",
            ),
            statistical_criterion_source="PRE_REGISTERED_VALIDATION",
            statistical_gamma_test=float(gamma_test),
            materiality_criterion_source="NONE",
            threshold_provenance=None,
            payload={
                "statistical_criterion_source": "PRE_REGISTERED_VALIDATION",
                "statistical_gamma_test": float(gamma_test),
                "materiality_criterion_source": "NONE",
                "gamma_test": gamma_test,
                "reject_unconditional_coverage": reject_unc,
                "reject_independence": reject_ind,
            },
            data_fingerprint=christoffersen_record.input_artifact_hash or "",
        )
        edges = [
            EvidenceEdge(
                source_id=christoffersen_record.evidence_id,
                target_id=kupiec_record.evidence_id,
                relation=RelationshipType.CHALLENGES,
                provenance_rule="var_frequency_vs_independence",
                payload={"gamma_test": gamma_test},
            )
        ]
        return claim, edges

    statement = (
        f"VaR backtests show consistent coverage and independence results under gamma_test={gamma_test} "
        f"(Kupiec reject={reject_unc}, Christoffersen independence reject={reject_ind})."
    )
    claim = AnalyticalClaim(
        claim_id=new_claim_id(),
        source_evidence_ids=(kupiec_record.evidence_id, christoffersen_record.evidence_id),
        claim_type=ClaimType.OBSERVATION,
        domain="tail_risk",
        metric_paths=metric_paths,
        status=ClaimStatus.VERIFIED,
        statement=statement,
        limitations=(),
        statistical_criterion_source="PRE_REGISTERED_VALIDATION",
        statistical_gamma_test=float(gamma_test),
        materiality_criterion_source="NONE",
        threshold_provenance=None,
        payload={
            "statistical_criterion_source": "PRE_REGISTERED_VALIDATION",
            "statistical_gamma_test": float(gamma_test),
            "materiality_criterion_source": "NONE",
            "gamma_test": gamma_test,
            "reject_unconditional_coverage": reject_unc,
            "reject_independence": reject_ind,
        },
        data_fingerprint=christoffersen_record.input_artifact_hash or "",
    )
    edges = [
        EvidenceEdge(
            source_id=christoffersen_record.evidence_id,
            target_id=kupiec_record.evidence_id,
            relation=RelationshipType.SUPPORTS,
            provenance_rule="var_frequency_vs_independence",
            payload={"gamma_test": gamma_test},
        )
    ]
    return claim, edges


def eval_optimization_covariance_sensitivity(
    sample_cov_record: EvidenceRecord,
    lw_cov_record: EvidenceRecord,
    weights_sample: dict[str, float],
    weights_lw: dict[str, float],
) -> tuple[AnalyticalClaim, list[EvidenceEdge]]:
    """Evaluate portfolio weight sensitivity across covariance estimation methodologies."""
    all_assets = sorted(set(weights_sample.keys()) | set(weights_lw.keys()))
    turnover = 0.5 * sum(abs(weights_sample.get(a, 0.0) - weights_lw.get(a, 0.0)) for a in all_assets)

    statement = (
        f"Deterministically measured allocation turnover between sample covariance and "
        f"Ledoit-Wolf shrinkage covariance is {turnover:.4f} without external policy threshold."
    )
    metric_paths = {
        "sample_evidence": f"{sample_cov_record.evidence_id}",
        "lw_evidence": f"{lw_cov_record.evidence_id}",
        "turnover": f"computed_l1_turnover({sample_cov_record.evidence_id}, {lw_cov_record.evidence_id})",
    }
    claim = AnalyticalClaim(
        claim_id=new_claim_id(),
        source_evidence_ids=(sample_cov_record.evidence_id, lw_cov_record.evidence_id),
        claim_type=ClaimType.SENSITIVITY,
        domain="portfolio_construction",
        metric_paths=metric_paths,
        status=ClaimStatus.EVIDENCE_ONLY,
        statement=statement,
        limitations=(
            "Measured weight divergence across covariance models indicates estimator sensitivity.",
            "Materiality requires an explicit institutional policy threshold.",
        ),
        payload={"turnover": turnover, "weights_sample": weights_sample, "weights_lw": weights_lw},
        data_fingerprint=sample_cov_record.input_artifact_hash or "",
    )
    edges = [
        EvidenceEdge(
            source_id=lw_cov_record.evidence_id,
            target_id=sample_cov_record.evidence_id,
            relation=RelationshipType.ALTERNATIVE_METHOD,
            provenance_rule="optimization_covariance_sensitivity",
            payload={"turnover": turnover},
        )
    ]
    return claim, edges


def eval_factor_exposure_vs_scenario_alignment(
    factor_exposure_record: EvidenceRecord,
    scenario_record: EvidenceRecord,
) -> tuple[AnalyticalClaim, list[EvidenceEdge]]:
    """Evaluate alignment between factor exposures and scenario stress factor contributions."""
    exposures: dict[str, float] = {}
    for k, v in factor_exposure_record.metrics.items():
        if k.startswith("beta_") or k.startswith("exposure_"):
            factor_name = k.split("_", 1)[1]
            try:
                exposures[factor_name] = float(v)  # type: ignore
            except (ValueError, TypeError):
                pass

    contributions: dict[str, float] = {}
    for k, v in scenario_record.metrics.items():
        if k.startswith("contrib_") or k.startswith("factor_"):
            factor_name = k.split("_", 1)[1]
            try:
                contributions[factor_name] = float(v)  # type: ignore
            except (ValueError, TypeError):
                pass

    dom_exp, exp_val = _argmax_abs(exposures)
    dom_scen, scen_val = _argmax_abs(contributions)

    metric_paths = {
        "dominant_factor_exposure": f"{factor_exposure_record.evidence_id}.metrics.dominant_factor",
        "dominant_scenario_contribution": f"{scenario_record.evidence_id}.metrics.dominant_contribution",
    }

    if dom_exp and dom_scen and dom_exp == dom_scen:
        statement = (
            f"Factor exposure analysis and scenario stress decomposition identify the same "
            f"dominant risk factor: '{dom_exp}' (exposure={exp_val:.4f}, contribution={scen_val:.4f})."
        )
        claim = AnalyticalClaim(
            claim_id=new_claim_id(),
            source_evidence_ids=(factor_exposure_record.evidence_id, scenario_record.evidence_id),
            claim_type=ClaimType.OBSERVATION,
            domain="factor_risk",
            metric_paths=metric_paths,
            status=ClaimStatus.VERIFIED,
            statement=statement,
            limitations=("Reflects alignment between static factor beta and scenario shock sensitivity.",),
            payload={"dominant_factor": dom_exp, "exposure": exp_val, "contribution": scen_val},
            data_fingerprint=scenario_record.input_artifact_hash or "",
        )
        edges = [
            EvidenceEdge(
                source_id=scenario_record.evidence_id,
                target_id=factor_exposure_record.evidence_id,
                relation=RelationshipType.SUPPORTS,
                provenance_rule="factor_exposure_vs_scenario_alignment",
                payload={"dominant_factor": dom_exp},
            )
        ]
        return claim, edges

    statement = (
        f"Factor exposure identifies dominant factor '{dom_exp}' while scenario factor contribution "
        f"identifies dominant factor '{dom_scen}' (reflecting shock vector interaction); "
        f"distinct analytical lenses do not constitute logical contradiction."
    )
    claim = AnalyticalClaim(
        claim_id=new_claim_id(),
        source_evidence_ids=(factor_exposure_record.evidence_id, scenario_record.evidence_id),
        claim_type=ClaimType.METHOD_DISAGREEMENT,
        domain="factor_risk",
        metric_paths=metric_paths,
        status=ClaimStatus.EVIDENCE_ONLY,
        statement=statement,
        limitations=(
            "Scenario contribution depends jointly on factor beta and the applied macro shock vector.",
            "Ranking divergence across distinct analytical lenses is not a logical contradiction.",
        ),
        payload={"dominant_exposure": dom_exp, "dominant_scenario": dom_scen},
        data_fingerprint=scenario_record.input_artifact_hash or "",
    )
    edges = [
        EvidenceEdge(
            source_id=scenario_record.evidence_id,
            target_id=factor_exposure_record.evidence_id,
            relation=RelationshipType.ALTERNATIVE_METHOD,
            provenance_rule="factor_exposure_vs_scenario_alignment",
            payload={"dominant_exposure": dom_exp, "dominant_scenario": dom_scen},
        )
    ]
    return claim, edges


def eval_reconciliation_identity_contradiction(
    claim_a_record: EvidenceRecord,
    claim_b_record: EvidenceRecord,
) -> tuple[AnalyticalClaim | None, list[EvidenceEdge]]:
    """Evaluate true logical/algebraic contradiction on identical contract reconciliation identities."""
    # Check if both records reference the same reconciliation contract identity
    identity_a = claim_a_record.params.get("reconciliation_identity") or claim_a_record.metrics.get(
        "identity"
    )
    identity_b = claim_b_record.params.get("reconciliation_identity") or claim_b_record.metrics.get(
        "identity"
    )

    if not identity_a or not identity_b or identity_a != identity_b:
        return None, []

    fp_a = claim_a_record.input_artifact_hash or ""
    fp_b = claim_b_record.input_artifact_hash or ""
    if fp_a and fp_b and fp_a != fp_b:
        return None, []

    res_a = float(claim_a_record.metrics.get("reconciliation_error", 0.0))  # type: ignore
    res_b = float(claim_b_record.metrics.get("reconciliation_error", 0.0))  # type: ignore

    # If one claims exact zero residual (<= 1e-8) and the other claims nonzero (> 1e-6)
    if (abs(res_a) <= 1e-8 and abs(res_b) > 1e-6) or (abs(res_b) <= 1e-8 and abs(res_a) > 1e-6):
        statement = (
            f"Mutually contradictory reconciliation claims on contract identity '{identity_a}': "
            f"Record '{claim_a_record.evidence_id}' asserts residual={res_a:.6f}, while "
            f"Record '{claim_b_record.evidence_id}' asserts residual={res_b:.6f}."
        )
        claim = AnalyticalClaim(
            claim_id=new_claim_id(),
            source_evidence_ids=(claim_a_record.evidence_id, claim_b_record.evidence_id),
            claim_type=ClaimType.CONTRADICTION,
            domain="reconciliation",
            metric_paths={
                "residual_a": f"{claim_a_record.evidence_id}.metrics.reconciliation_error",
                "residual_b": f"{claim_b_record.evidence_id}.metrics.reconciliation_error",
            },
            status=ClaimStatus.CONTRADICTED,
            statement=statement,
            limitations=(
                "Deterministic algebraic contradiction detected across identical contract identity.",
            ),
            payload={"identity": identity_a, "residual_a": res_a, "residual_b": res_b},
            data_fingerprint=fp_a,
        )
        edges = [
            EvidenceEdge(
                source_id=claim_b_record.evidence_id,
                target_id=claim_a_record.evidence_id,
                relation=RelationshipType.CONTRADICTS,
                provenance_rule="reconciliation_identity_contradiction",
                payload={"identity": identity_a},
            )
        ]
        return claim, edges

    return None, []


def eval_var_vs_reverse_stress(
    var_record: EvidenceRecord,
    reverse_stress_record: EvidenceRecord,
) -> tuple[AnalyticalClaim, list[EvidenceEdge]]:
    """Evaluate historical VaR model summary against reverse-stress tail geometry."""
    var_loss = float(
        cast(
            Any,
            var_record.metrics.get("var_estimate", var_record.metrics.get("portfolio_loss", 0.0)),
        )
    )
    target_loss = float(cast(Any, reverse_stress_record.metrics.get("target_loss", 0.0)))
    min_dist = float(
        cast(
            Any,
            reverse_stress_record.metrics.get(
                "minimum_distance", reverse_stress_record.metrics.get("shock_norm", 0.0)
            ),
        )
    )

    statement = (
        f"Reverse stress evaluates portfolio loss geometry (target loss={target_loss:.4f}, "
        f"minimum shock distance={min_dist:.4f}) outside historical VaR distribution "
        f"(var_estimate={var_loss:.4f})."
    )
    claim = AnalyticalClaim(
        claim_id=new_claim_id(),
        source_evidence_ids=(var_record.evidence_id, reverse_stress_record.evidence_id),
        claim_type=ClaimType.UNRESOLVED_RISK,
        domain="scenario_stress",
        metric_paths={
            "var_loss": f"{var_record.evidence_id}.metrics.var_estimate",
            "reverse_stress_distance": f"{reverse_stress_record.evidence_id}.metrics.minimum_distance",
        },
        status=ClaimStatus.EVIDENCE_ONLY,
        statement=statement,
        limitations=(
            "Reverse stress identifies minimal plausible shock geometry reaching designated target loss.",
            "Absence of historical VaR breaches does not preclude plausible reverse-stress tail scenarios.",
            "Without explicit policy threshold, numerical shock distance is non-normative.",
        ),
        payload={"var_loss": var_loss, "target_loss": target_loss, "minimum_distance": min_dist},
        data_fingerprint=reverse_stress_record.input_artifact_hash or "",
    )
    edges = [
        EvidenceEdge(
            source_id=reverse_stress_record.evidence_id,
            target_id=var_record.evidence_id,
            relation=RelationshipType.DIAGNOSTIC_OF,
            provenance_rule="var_vs_reverse_stress",
            payload={"target_loss": target_loss, "minimum_distance": min_dist},
        )
    ]
    return claim, edges


def eval_attribution_vs_factor_risk(
    attribution_record: EvidenceRecord,
    factor_risk_record: EvidenceRecord,
) -> tuple[AnalyticalClaim, list[EvidenceEdge]]:
    """Evaluate performance attribution and factor risk shared factor specification dependency.

    Scientific Invariant:
    Performance attribution is denominated in return units (decimal return).
    Factor risk decomposition is denominated in variance/volatility units.
    They represent distinct dimensional projections and MUST NOT be numerically reconciled.
    """
    recon_err = float(cast(Any, attribution_record.metrics.get("reconciliation_error", 0.0)))

    statement = (
        f"Performance attribution (return units) and factor risk decomposition (variance units) "
        f"share common factor specification. Attribution internal residual={recon_err:.6f}."
    )
    claim = AnalyticalClaim(
        claim_id=new_claim_id(),
        source_evidence_ids=(attribution_record.evidence_id, factor_risk_record.evidence_id),
        claim_type=ClaimType.DEPENDENCY,
        domain="attribution",
        metric_paths={
            "attribution_residual": f"{attribution_record.evidence_id}.metrics.reconciliation_error",
        },
        status=ClaimStatus.VERIFIED if recon_err <= 1e-6 else ClaimStatus.UNRESOLVED,
        statement=statement,
        limitations=(
            "Return attribution and factor variance decomposition are dimensionally distinct "
            "and cannot be numerically reconciled against each other.",
        ),
        payload={"reconciliation_error": recon_err},
        data_fingerprint=attribution_record.input_artifact_hash or "",
    )
    edges = [
        EvidenceEdge(
            source_id=attribution_record.evidence_id,
            target_id=factor_risk_record.evidence_id,
            relation=RelationshipType.DEPENDS_ON,
            provenance_rule="attribution_vs_factor_risk",
            payload={"reconciliation_error": recon_err},
        )
    ]
    return claim, edges


def eval_solver_convergence_vs_scenario_stress(
    opt_record: EvidenceRecord,
    scenario_record: EvidenceRecord,
) -> tuple[AnalyticalClaim, list[EvidenceEdge]]:
    """Evaluate portfolio optimization convergence against scenario stress sensitivity."""
    converged = bool(opt_record.metrics.get("converged", True))
    scen_loss = float(
        cast(
            Any,
            scenario_record.metrics.get("scenario_loss", scenario_record.metrics.get("portfolio_loss", 0.0)),
        )
    )

    statement = (
        f"Portfolio optimizer converged (converged={converged}); "
        f"stress scenario evaluates portfolio loss under shock: {scen_loss:.4f} without policy threshold."
    )
    claim = AnalyticalClaim(
        claim_id=new_claim_id(),
        source_evidence_ids=(opt_record.evidence_id, scenario_record.evidence_id),
        claim_type=ClaimType.UNRESOLVED_RISK,
        domain="portfolio_construction",
        metric_paths={
            "solver_converged": f"{opt_record.evidence_id}.metrics.converged",
            "scenario_loss": f"{scenario_record.evidence_id}.metrics.scenario_loss",
        },
        status=ClaimStatus.EVIDENCE_ONLY,
        statement=statement,
        limitations=(
            "Numerical solver convergence does not imply portfolio robustness under adverse market shocks.",
        ),
        payload={"converged": converged, "scenario_loss": scen_loss},
        data_fingerprint=scenario_record.input_artifact_hash or "",
    )
    edges = [
        EvidenceEdge(
            source_id=scenario_record.evidence_id,
            target_id=opt_record.evidence_id,
            relation=RelationshipType.DIAGNOSTIC_OF,
            provenance_rule="solver_convergence_vs_scenario_stress",
            payload={"scenario_loss": scen_loss},
        )
    ]
    return claim, edges
