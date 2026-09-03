"""Gate 4 Test Suite: Institutional Factor Risk Model & Euler Variance Decomposition.

Comprehensive verification of:
1. Linear factor risk model reconstruction: Sigma_asset = B F B' + D.
2. Strict fail-closed factor/asset alignment without silent zero imputation.
3. Euler-consistent factor component variance decomposition: C_k = b_{p,k} (F b_p)_k, sum C_k = b_p' F b_p.
4. Asset-level specific risk contributions: w_i^2 * d_i, sum = w' D w.
5. Active risk (tracking error) factor vs specific decomposition: TE^2 = Delta b' F Delta b + a' D a.
6. Deterministic data integrity checking, evidence adapters, and cryptographic artifact generation.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from start.portfolio.artifacts import (
    render_active_risk_decomposition_artifact,
    render_factor_risk_model_artifact,
    render_factor_risk_waterfall_artifact,
)
from start.portfolio.contracts import (
    ActiveRiskDecompositionResult,
    FactorDataIntegrityResult,
    FactorRiskDecompositionResult,
    FactorRiskModelResult,
)
from start.portfolio.evidence_bridge import (
    active_risk_decomp_to_evidence,
    factor_risk_decomp_to_evidence,
    factor_risk_model_to_evidence,
)
from start.portfolio.factor_risk import (
    build_linear_factor_model,
    decompose_active_risk,
    decompose_factor_risk,
    validate_factor_alignment,
    validate_factor_data_integrity,
)


@pytest.fixture
def factor_universe_4x2() -> dict[str, Any]:
    """Canonical 4-asset, 2-factor risk model setup."""
    assets = ["AAPL", "MSFT", "XOM", "JNJ"]
    factors = ["Market", "Value"]

    # 4 x 2 Exposure Matrix B
    B_df = pd.DataFrame(
        [
            [1.20, -0.30],  # AAPL (high beta, growth)
            [1.10, -0.20],  # MSFT (high beta, growth)
            [0.70, 0.80],   # XOM (low beta, high value)
            [0.60, 0.40],   # JNJ (defensive, value)
        ],
        index=assets,
        columns=factors,
    )

    # 2 x 2 Factor Covariance F
    F_df = pd.DataFrame(
        [
            [0.040, -0.005],
            [-0.005, 0.020],
        ],
        index=factors,
        columns=factors,
    )

    # Specific Variances D
    D_dict = {
        "AAPL": 0.030,
        "MSFT": 0.025,
        "XOM": 0.020,
        "JNJ": 0.015,
    }

    portfolio_w = {"AAPL": 0.30, "MSFT": 0.30, "XOM": 0.20, "JNJ": 0.20}
    benchmark_w = {"AAPL": 0.25, "MSFT": 0.25, "XOM": 0.25, "JNJ": 0.25}

    return {
        "assets": assets,
        "factors": factors,
        "exposures": B_df,
        "factor_cov": F_df,
        "specific_var": D_dict,
        "portfolio_weights": portfolio_w,
        "benchmark_weights": benchmark_w,
    }


def test_validate_factor_alignment_success(factor_universe_4x2):
    """Validate properly aligned factor inputs."""
    u = factor_universe_4x2
    B, F, d_vec, assets, factors = validate_factor_alignment(
        exposures=u["exposures"],
        factor_cov=u["factor_cov"],
        specific_var=u["specific_var"],
    )
    assert B.shape == (4, 2)
    assert F.shape == (2, 2)
    assert len(d_vec) == 4
    assert assets == ("AAPL", "MSFT", "XOM", "JNJ")
    assert factors == ("Market", "Value")


def test_validate_factor_alignment_duplicate_labels_rejected(factor_universe_4x2):
    """Duplicate factor or asset labels must fail closed."""
    u = factor_universe_4x2
    bad_exp = pd.DataFrame([[1.0, 2.0]], index=["A0"], columns=["F0", "F0"])
    with pytest.raises(ValueError, match="Duplicate factor labels"):
        validate_factor_alignment(bad_exp, u["factor_cov"], u["specific_var"])

    bad_exp_asset = pd.DataFrame([[1.0, 2.0], [1.0, 2.0]], index=["A0", "A0"], columns=["F0", "F1"])
    with pytest.raises(ValueError, match="Duplicate asset labels"):
        validate_factor_alignment(bad_exp_asset, u["factor_cov"], u["specific_var"])


def test_validate_factor_alignment_missing_fail_closed(factor_universe_4x2):
    """Missing assets or factors in covariance / specific variances must fail closed."""
    u = factor_universe_4x2
    bad_svar = {"AAPL": 0.02, "MSFT": 0.02}  # Missing XOM and JNJ
    with pytest.raises(ValueError, match="Asset.*missing from specific variances"):
        validate_factor_alignment(u["exposures"], u["factor_cov"], bad_svar)


def test_build_linear_factor_model_exact_reconstruction(factor_universe_4x2):
    """Verify exact algebraic reconstruction Sigma = B F B' + D."""
    u = factor_universe_4x2
    frm = build_linear_factor_model(
        exposures=u["exposures"],
        factor_cov=u["factor_cov"],
        specific_var=u["specific_var"],
    )
    assert isinstance(frm, FactorRiskModelResult)
    assert frm.asset_order == ("AAPL", "MSFT", "XOM", "JNJ")
    assert frm.factor_order == ("Market", "Value")

    # Manually compute Sigma = B F B' + D
    B = u["exposures"].to_numpy()
    F = u["factor_cov"].to_numpy()
    D = np.diag([u["specific_var"][a] for a in u["assets"]])
    expected_sigma = B @ F @ B.T + D

    reconstructed_sigma = np.asarray(frm.reconstructed_covariance)
    np.testing.assert_allclose(reconstructed_sigma, expected_sigma, atol=1e-12)
    assert frm.diagnostics.is_psd
    assert frm.diagnostics.minimum_eigenvalue > 0.0


def test_decompose_factor_risk_euler_reconciliation(factor_universe_4x2):
    """Euler factor component variance sum must exactly equal systematic variance."""
    u = factor_universe_4x2
    frm = build_linear_factor_model(
        exposures=u["exposures"],
        factor_cov=u["factor_cov"],
        specific_var=u["specific_var"],
        periods_per_year=252.0,
    )

    frd = decompose_factor_risk(
        weights=u["portfolio_weights"],
        factor_model=frm,
        periods_per_year=252.0,
    )

    assert isinstance(frd, FactorRiskDecompositionResult)
    # Systematic variance = sum_k C_k
    sum_factor_comps = sum(frd.factor_variance_contributions_periodic.values())
    assert math.isclose(sum_factor_comps, frd.systematic_variance_periodic, abs_tol=1e-14)
    assert frd.euler_reconciliation_error < 1e-14

    # Specific variance = sum_i SC_i
    sum_asset_spec = sum(frd.asset_specific_variance_contributions.values())
    assert math.isclose(sum_asset_spec, frd.specific_variance_periodic, abs_tol=1e-14)

    # Total variance = systematic + specific
    expected_total = frd.systematic_variance_periodic + frd.specific_variance_periodic
    assert math.isclose(frd.total_variance_periodic, expected_total, abs_tol=1e-14)
    assert frd.total_reconciliation_error < 1e-14

    # Shares sum to 1.0
    assert math.isclose(frd.systematic_variance_share + frd.specific_variance_share, 1.0, abs_tol=1e-5)


def test_decompose_active_risk_tracking_error_reconciliation(factor_universe_4x2):
    """Active risk tracking error must reconcile factor active and specific active variance."""
    u = factor_universe_4x2
    frm = build_linear_factor_model(
        exposures=u["exposures"],
        factor_cov=u["factor_cov"],
        specific_var=u["specific_var"],
        periods_per_year=252.0,
    )

    ard = decompose_active_risk(
        weights=u["portfolio_weights"],
        benchmark_weights=u["benchmark_weights"],
        factor_model=frm,
        periods_per_year=252.0,
    )

    assert isinstance(ard, ActiveRiskDecompositionResult)
    # Active weights: AAPL +0.05, MSFT +0.05, XOM -0.05, JNJ -0.05
    assert math.isclose(ard.active_weights["AAPL"], 0.05, abs_tol=1e-6)
    assert math.isclose(ard.active_weights["XOM"], -0.05, abs_tol=1e-6)

    # Total active variance = factor active + specific active
    expected_active_total = ard.factor_active_variance_periodic + ard.specific_active_variance_periodic
    assert math.isclose(ard.total_active_variance_periodic, expected_active_total, abs_tol=1e-14)
    assert ard.reconciliation_error < 1e-14

    # Tracking error is positive and non-zero
    assert ard.tracking_error_annualised > 0.0
    assert math.isclose(ard.factor_active_share + ard.specific_active_share, 1.0, abs_tol=1e-5)


def test_validate_factor_data_integrity(factor_universe_4x2):
    """Pre-flight deterministic factor data integrity checker."""
    u = factor_universe_4x2
    res = validate_factor_data_integrity(
        exposures=u["exposures"],
        factor_cov=u["factor_cov"],
        specific_var=u["specific_var"],
        weights=u["portfolio_weights"],
    )
    assert isinstance(res, FactorDataIntegrityResult)
    assert res.is_valid
    assert res.n_assets == 4
    assert res.n_factors == 2
    assert not res.has_duplicate_assets
    assert not res.has_duplicate_factors
    assert res.missing_exposure_count == 0


def test_factor_risk_evidence_adapters_and_artifacts(factor_universe_4x2):
    """Convert factor risk results into EvidenceRecords and ArtifactRecords."""
    u = factor_universe_4x2
    frm = build_linear_factor_model(
        exposures=u["exposures"],
        factor_cov=u["factor_cov"],
        specific_var=u["specific_var"],
    )
    frd = decompose_factor_risk(weights=u["portfolio_weights"], factor_model=frm)
    ard = decompose_active_risk(weights=u["portfolio_weights"], benchmark_weights=u["benchmark_weights"], factor_model=frm)

    ev_frm = factor_risk_model_to_evidence(frm)
    ev_frd = factor_risk_decomp_to_evidence(frd)
    ev_ard = active_risk_decomp_to_evidence(ard)

    assert ev_frm.test_id == "factor_risk.model"
    assert ev_frd.test_id == "factor_risk.decomposition"
    assert ev_ard.test_id == "factor_risk.active_decomposition"

    with tempfile.TemporaryDirectory() as tmpdir:
        art_frm = render_factor_risk_model_artifact(frm, evidence_ids=(ev_frm.evidence_id,), output_dir=tmpdir)
        art_frd = render_factor_risk_waterfall_artifact(frd, evidence_ids=(ev_frd.evidence_id,), output_dir=tmpdir)
        art_ard = render_active_risk_decomposition_artifact(ard, evidence_ids=(ev_ard.evidence_id,), output_dir=tmpdir)

        assert Path(art_frm.file_path).exists()
        assert Path(art_frd.file_path).exists()
        assert Path(art_ard.file_path).exists()
