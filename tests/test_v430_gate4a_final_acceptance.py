"""StART — Gate 4A Final Acceptance Audit Test Suite.

Audits:
1. Human-review visual artifacts (.svg) + semantic companions (.json) completeness and provenance.
2. Financial horizon contract integration (periodic vs annual, double-annualization rejection, frequency contradiction).
3. Attribution lookahead-free beginning-of-period alignment.
4. Higham nearest correlation contract, Dykstra alternating projections, symmetry, diagonal preservation, idempotence, and PD floor.
5. Factor data integrity partial-presence fail-closed behavior.
6. Adversarial challenge diagnostic evidence provenance and ACCEPT_WITH_CONDITIONS governance semantics.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from start.agents.market_review import (
    AdversarialChallengeAgent,
    FactorDataIntegrityChecker,
    GovernanceAgent,
    GovernanceVerdict,
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
    MetricHorizon,
    PSDRepairMethod,
)
from start.portfolio.covariance import (
    compare_covariance_estimators,
    diagnose_covariance,
    repair_psd_covariance,
)
from start.portfolio.evidence_bridge import (
    covariance_diagnostics_to_evidence,
)
from start.portfolio.factor_risk import (
    build_linear_factor_model,
    decompose_active_risk,
    decompose_factor_risk,
)


# =========================================================================== #
# 1. HUMAN-REVIEW ARTIFACTS & PROVENANCE TESTS
# =========================================================================== #
def test_human_review_visual_artifacts_and_semantic_companions() -> None:
    """Verify that all Gate 4 artifact renderers produce BOTH SVG visual files and JSON semantic companion files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Covariance Diagnostics
        diag = diagnose_covariance(np.eye(3), assets=["A", "B", "C"])
        art_cov_diag = render_covariance_diagnostics_artifact(diag, ("EV-001",), output_dir=tmp_path)
        assert art_cov_diag.rendering_format == "svg"
        assert art_cov_diag.file_path is not None
        assert Path(art_cov_diag.file_path).exists()
        assert Path(tmp_path / f"{art_cov_diag.artifact_id}.json").exists()
        assert art_cov_diag.spec.evidence_ids == ("EV-001",)

        # 2. Raw Covariance Heatmap
        art_raw_cov = render_raw_covariance_heatmap_artifact(
            np.eye(3), ["A", "B", "C"], ("EV-001",), output_dir=tmp_path
        )
        assert art_raw_cov.rendering_format == "svg"
        assert Path(art_raw_cov.file_path).exists()
        assert Path(tmp_path / f"{art_raw_cov.artifact_id}.json").exists()

        # 3. PSD Repair
        rep = repair_psd_covariance(
            np.array([[1.0, 0.9], [0.9, 0.1]]), method=PSDRepairMethod.HIGHAM_NEAREST_CORRELATION
        )
        art_rep = render_psd_repair_artifact(rep, ("EV-002",), output_dir=tmp_path)
        assert art_rep.rendering_format == "svg"
        assert Path(art_rep.file_path).exists()
        assert Path(tmp_path / f"{art_rep.artifact_id}.json").exists()

        # 4. Covariance Comparison
        ret_df = pd.DataFrame(np.random.randn(20, 2), columns=["A", "B"])
        comp = compare_covariance_estimators(ret_df, estimators=["empirical", "ledoit_wolf"])
        art_comp = render_covariance_comparison_artifact(comp, ("EV-003",), output_dir=tmp_path)
        assert art_comp.rendering_format == "svg"
        assert Path(art_comp.file_path).exists()
        assert Path(tmp_path / f"{art_comp.artifact_id}.json").exists()

        # 5. Factor Risk Model
        B = pd.DataFrame([[1.0], [0.8]], index=["A", "B"], columns=["Mkt"])
        F = pd.DataFrame([[0.04]], index=["Mkt"], columns=["Mkt"])
        D = {"A": 0.01, "B": 0.02}
        frm = build_linear_factor_model(B, F, D)
        art_frm = render_factor_risk_model_artifact(frm, ("EV-004",), output_dir=tmp_path)
        assert art_frm.rendering_format == "svg"
        assert Path(art_frm.file_path).exists()
        assert Path(tmp_path / f"{art_frm.artifact_id}.json").exists()

        # 6. Factor Risk Waterfall
        frd = decompose_factor_risk({"A": 0.5, "B": 0.5}, frm)
        art_frd = render_factor_risk_waterfall_artifact(frd, ("EV-005",), output_dir=tmp_path)
        assert art_frd.rendering_format == "svg"
        assert Path(art_frd.file_path).exists()
        assert Path(tmp_path / f"{art_frd.artifact_id}.json").exists()

        # 7. Active Risk Decomposition
        ard = decompose_active_risk({"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5}, frm)
        art_ard = render_active_risk_decomposition_artifact(ard, ("EV-006",), output_dir=tmp_path)
        assert art_ard.rendering_format == "svg"
        assert Path(art_ard.file_path).exists()
        assert Path(tmp_path / f"{art_ard.artifact_id}.json").exists()

        # 8. Factor Return Attribution
        rets = pd.DataFrame([[0.01, 0.02], [0.03, -0.01]], columns=["A", "B"])
        frets = pd.DataFrame([[0.015], [0.010]], columns=["Mkt"])
        fra = compute_factor_return_attribution(rets, B, frets, {"A": 0.5, "B": 0.5})
        art_fra = render_factor_return_attribution_artifact(fra, ("EV-007",), output_dir=tmp_path)
        assert art_fra.rendering_format == "svg"
        assert Path(art_fra.file_path).exists()
        assert Path(tmp_path / f"{art_fra.artifact_id}.json").exists()

        # 9. Brinson Attribution
        br = compute_brinson_attribution(
            {"G1": 0.6, "G2": 0.4}, {"G1": 0.5, "G2": 0.5}, {"G1": 0.05, "G2": 0.02}, {"G1": 0.04, "G2": 0.03}
        )
        art_br = render_brinson_attribution_artifact(br, ("EV-008",), output_dir=tmp_path)
        assert art_br.rendering_format == "svg"
        assert Path(art_br.file_path).exists()
        assert Path(tmp_path / f"{art_br.artifact_id}.json").exists()

        # 10. Carino Linking
        car = compute_carino_multi_period_linking([br, br], [0.038, 0.038], [0.035, 0.035])
        art_car = render_carino_linking_artifact(car, ("EV-009",), output_dir=tmp_path)
        assert art_car.rendering_format == "svg"
        assert Path(art_car.file_path).exists()
        assert Path(tmp_path / f"{art_car.artifact_id}.json").exists()

        # Empty evidence validation (fail closed)
        with pytest.raises(ValueError, match="Artifact must have explicit evidence_ids"):
            render_covariance_diagnostics_artifact(diag, ())


# =========================================================================== #
# 2. FINANCIAL HORIZON CONTRACT INTEGRATION TESTS
# =========================================================================== #
def test_financial_horizon_active_risk_and_factor_model() -> None:
    """Verify explicit financial horizon contracts across covariance, factor risk, and tracking error."""
    B = pd.DataFrame([[1.0], [0.8]], index=["A", "B"], columns=["Mkt"])
    F = pd.DataFrame([[0.04]], index=["Mkt"], columns=["Mkt"])
    D = {"A": 0.01, "B": 0.02}

    # A. PERIODIC covariance + daily + ppy=252 -> valid annualized tracking error
    frm_periodic = build_linear_factor_model(
        B,
        F,
        D,
        factor_cov_horizon=MetricHorizon.PERIODIC,
        specific_var_horizon=MetricHorizon.PERIODIC,
        frequency="daily",
        periods_per_year=252.0,
    )
    ard_periodic = decompose_active_risk(
        {"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5}, frm_periodic, periods_per_year=252.0
    )
    assert ard_periodic.tracking_error_annualised > 0.0
    assert ard_periodic.horizon == MetricHorizon.PERIODIC

    # B. ANNUAL covariance + ppy=1 -> valid annual tracking error without additional scaling
    frm_annual = build_linear_factor_model(
        B,
        F,
        D,
        factor_cov_horizon=MetricHorizon.ANNUAL,
        specific_var_horizon=MetricHorizon.ANNUAL,
        periods_per_year=1.0,
    )
    ard_annual = decompose_active_risk(
        {"A": 0.6, "B": 0.4}, {"A": 0.5, "B": 0.5}, frm_annual, periods_per_year=1.0
    )
    assert ard_annual.tracking_error_annualised == pytest.approx(
        np.sqrt(ard_annual.total_active_variance_periodic), rel=1e-6
    )
    assert ard_annual.horizon == MetricHorizon.ANNUAL

    # C. ANNUAL covariance + ppy=252 -> REJECT double annualization
    with pytest.raises(ValueError, match="Double annualization error"):
        build_linear_factor_model(
            B,
            F,
            D,
            factor_cov_horizon=MetricHorizon.ANNUAL,
            specific_var_horizon=MetricHorizon.ANNUAL,
            periods_per_year=252.0,
        )

    # D. PERIODIC covariance + monthly + ppy=252 -> REJECT frequency contradiction
    with pytest.raises(ValueError, match="Frequency contradiction"):
        build_linear_factor_model(
            B,
            F,
            D,
            factor_cov_horizon=MetricHorizon.PERIODIC,
            specific_var_horizon=MetricHorizon.PERIODIC,
            frequency="monthly",
            periods_per_year=252.0,
        )

    # E. Factor covariance F (ANNUAL) + Specific risk D (PERIODIC) mismatch -> REJECT
    with pytest.raises(ValueError, match="Financial horizon mismatch"):
        build_linear_factor_model(
            B,
            F,
            D,
            factor_cov_horizon=MetricHorizon.ANNUAL,
            specific_var_horizon=MetricHorizon.PERIODIC,
        )


# =========================================================================== #
# 3. ATTRIBUTION HORIZON / LOOKAHEAD BOUNDARY TEST
# =========================================================================== #
def test_attribution_timing_and_temporal_alignment() -> None:
    """Verify that factor return attribution enforces lookahead-free period-by-period alignment."""
    assets = ["A", "B", "C"]
    factors = ["Mkt", "Val"]
    n_periods = 5

    B = pd.DataFrame(
        [[1.1, 0.2], [0.9, -0.3], [1.0, 0.5]],
        index=assets,
        columns=factors,
    )
    factor_rets = pd.DataFrame(
        np.random.randn(n_periods, 2) * 0.01,
        columns=factors,
    )
    asset_rets = pd.DataFrame(
        factor_rets.to_numpy() @ B.to_numpy().T,
        columns=assets,
    )
    weights = {"A": 0.4, "B": 0.3, "C": 0.3}

    fra = compute_factor_return_attribution(
        returns=asset_rets,
        exposures=B,
        factor_returns=factor_rets,
        weights=weights,
        time_alignment_convention="beginning_of_period_exposures",
    )

    assert fra.n_periods == n_periods
    assert fra.is_reconciled is True
    assert fra.max_abs_reconciliation_error < 1e-12

    # Mismatched period counts fail closed
    with pytest.raises(ValueError, match="Factor returns shape"):
        compute_factor_return_attribution(
            returns=asset_rets,
            exposures=B,
            factor_returns=factor_rets.iloc[:3],
            weights=weights,
        )


# =========================================================================== #
# 4. HIGHAM METHOD CONTRACT, NOMENCLATURE & IDEMPOTENCE
# =========================================================================== #
def test_higham_nearest_correlation_algorithm_contract() -> None:
    """Audit Higham alternating projections algorithm, symmetry, diagonal preservation, and idempotence."""
    # 1. Already valid positive definite correlation matrix -> unchanged within tolerance
    valid_corr = np.array(
        [
            [1.0, 0.3, 0.2],
            [0.3, 1.0, 0.1],
            [0.2, 0.1, 1.0],
        ]
    )
    res_valid = repair_psd_covariance(
        valid_corr, method=PSDRepairMethod.HIGHAM_NEAREST_CORRELATION, min_eigenvalue=0.0
    )
    assert np.allclose(res_valid.repaired_matrix, valid_corr, atol=1e-5)
    assert res_valid.frobenius_distortion < 1e-4

    # 2. Indefinite correlation matrix -> repaired to PSD
    indef_mat = np.array(
        [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.9],
            [0.9, 0.9, 0.1],  # diagonal altered to 0.1 causing negative eigenvalues
        ]
    )
    res_indef = repair_psd_covariance(
        indef_mat, method=PSDRepairMethod.HIGHAM_NEAREST_CORRELATION, min_eigenvalue=1e-8
    )
    rep_arr = np.asarray(res_indef.repaired_matrix)

    assert res_indef.original_minimum_eigenvalue < 0.0
    assert res_indef.repaired_minimum_eigenvalue >= 0.0
    assert res_indef.converged is True
    assert res_indef.iterations_used >= 1

    # Symmetry preserved
    assert np.allclose(rep_arr, rep_arr.T, atol=1e-12)

    # Covariance diagonal preserved after rescaling
    assert res_indef.diagonal_preserved is True
    assert np.allclose(np.diag(rep_arr), np.diag(indef_mat), atol=1e-5)

    # Idempotence: re-running repair on already repaired matrix produces negligible distortion
    res_repaired_again = repair_psd_covariance(
        rep_arr, method=PSDRepairMethod.HIGHAM_NEAREST_CORRELATION, min_eigenvalue=1e-8
    )
    assert res_repaired_again.frobenius_distortion < 1e-6
    assert np.allclose(res_repaired_again.repaired_matrix, rep_arr, atol=1e-6)

    # PD floor provenance recorded
    assert res_indef.pd_floor == 1e-8


# =========================================================================== #
# 5. FACTOR DATA INTEGRITY PARTIAL PRESENCE (FAIL CLOSED)
# =========================================================================== #
def test_factor_data_integrity_partial_presence() -> None:
    """Verify that FactorDataIntegrityChecker skips on absent model but fails closed on partial presence."""
    checker = FactorDataIntegrityChecker()

    # 1. All factor inputs absent -> SKIP
    res_skip = checker.execute({})
    assert res_skip["status"] == "skipped"
    assert res_skip["findings"] == []
    assert res_skip["evidence_record"] is None

    # 2. Partial input: B supplied, but F and D missing -> FAIL CLOSED
    B = pd.DataFrame([[1.0, 0.5]], index=["AAPL"], columns=["Mkt", "Val"])
    res_partial_b = checker.execute({"exposures": B})
    assert res_partial_b["status"] != "skipped"
    assert len(res_partial_b["findings"]) >= 1
    assert any("critical_breach" == f["severity"] for f in res_partial_b["findings"])

    # 3. Partial input: F supplied, but B missing -> FAIL CLOSED
    F = pd.DataFrame([[0.04]], index=["Mkt"], columns=["Mkt"])
    res_partial_f = checker.execute({"factor_cov": F})
    assert len(res_partial_f["findings"]) >= 1

    # 4. Partial input: Factor returns supplied without exposures -> FAIL CLOSED
    fret = pd.DataFrame([[0.01]], columns=["Mkt"])
    res_partial_fret = checker.execute({"factor_returns": fret})
    assert len(res_partial_fret["findings"]) >= 1


# =========================================================================== #
# 6. ADVERSARIAL CHALLENGE PROVENANCE & GOVERNANCE SEMANTICS
# =========================================================================== #
def test_challenge_diagnostic_evidence_and_governance_verdict() -> None:
    """Verify distinct generated diagnostic EvidenceRecords and ACCEPT_WITH_CONDITIONS sign-off."""
    diag = diagnose_covariance(np.array([[1.0, 0.9], [0.9, 0.1]]))
    ev_raw = covariance_diagnostics_to_evidence(diag)
    ev_raw.evidence_id = "EV-COV-SRC-001"

    challenger = AdversarialChallengeAgent()
    challenge_context = {
        "covariance": np.array([[1.0, 0.9], [0.9, 0.1]]),
        "evidence_records": [ev_raw],
    }
    chal_out = challenger.execute(challenge_context)
    resolutions = chal_out["resolutions"]

    assert len(resolutions) >= 1
    # Check distinct generated IDs
    for r in resolutions:
        for g_id in r["generated_evidence_ids"]:
            assert g_id != ev_raw.evidence_id
            assert g_id.startswith("EV-")

    # Check GovernanceAgent handles evidence-only challenges with ACCEPT_WITH_CONDITIONS
    gov_agent = GovernanceAgent()
    gov_res = gov_agent.evaluate_signoff(
        critic_disposition="READY_FOR_GOVERNANCE",
        challenges=chal_out["challenges"],
        findings=[],
        records=[ev_raw],
        resolutions=resolutions,
    )
    assert gov_res["verdict"] == GovernanceVerdict.ACCEPT_WITH_CONDITIONS.value
    assert len(gov_res["conditions"]) == len(resolutions)
