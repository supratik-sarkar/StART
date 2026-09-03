"""Gate 4 Test Suite: Institutional Covariance Diagnostics, PSD Repair & Estimator Comparison.

Comprehensive verification of:
1. Exact spectral, rank, condition, trace, and entropy-based effective rank diagnostics.
2. Mathematically explicit PSD repair: SPECTRAL_CLIPPING and HIGHAM_NEAREST_CORRELATION.
3. Multi-estimator comparison composing canonical empirical, Ledoit-Wolf, and RegEM implementations.
4. Subordinate EvidenceRecord adapters and cryptographic artifact generation.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from start.portfolio.artifacts import (
    render_covariance_diagnostics_artifact,
    render_psd_repair_artifact,
)
from start.portfolio.contracts import (
    PSDRepairMethod,
    PSDRepairResult,
)
from start.portfolio.covariance import (
    compare_covariance_estimators,
    diagnose_covariance,
    repair_psd_covariance,
)
from start.portfolio.evidence_bridge import (
    covariance_diagnostics_to_evidence,
    psd_repair_to_evidence,
)


@pytest.fixture
def clean_cov_4x4() -> tuple[np.ndarray, list[str]]:
    """Well-conditioned positive definite 4x4 covariance matrix."""
    stds = np.array([0.15, 0.20, 0.25, 0.30])
    corr = np.array([
        [1.00, 0.50, 0.30, 0.10],
        [0.50, 1.00, 0.40, 0.20],
        [0.30, 0.40, 1.00, 0.35],
        [0.10, 0.20, 0.35, 1.00],
    ])
    cov = np.outer(stds, stds) * corr
    assets = ["A0", "A1", "A2", "A3"]
    return cov, assets


@pytest.fixture
def indefinite_cov_3x3() -> np.ndarray:
    """Symmetric 3x3 matrix with a strictly negative eigenvalue."""
    mat = np.array([
        [1.00, 0.90, 0.90],
        [0.90, 1.00, 0.90],
        [0.90, 0.90, 0.10],  # Incompatible third asset correlation/variance structure
    ])
    return mat


def test_diagnose_covariance_positive_definite(clean_cov_4x4):
    """Diagnose clean positive definite covariance matrix."""
    cov, assets = clean_cov_4x4
    diag = diagnose_covariance(cov, assets=assets)

    assert diag.n_assets == 4
    assert diag.is_symmetric
    assert diag.symmetry_error < 1e-10
    assert diag.is_psd
    assert diag.minimum_eigenvalue > 0
    assert diag.maximum_eigenvalue > diag.minimum_eigenvalue
    assert diag.rank == 4
    assert diag.numerical_rank == 4
    assert math.isfinite(diag.condition_number)
    assert diag.log_determinant is not None
    assert diag.diagonal_positive
    assert diag.valid_correlation_conversion
    assert len(diag.matrix_fingerprint) == 32
    # Effective rank for 4 assets with moderate correlation should be between 1.5 and 4.0
    assert 1.0 <= diag.effective_rank <= 4.0


def test_diagnose_covariance_indefinite(indefinite_cov_3x3):
    """Diagnose indefinite covariance matrix with negative eigenvalue."""
    diag = diagnose_covariance(indefinite_cov_3x3)

    assert diag.n_assets == 3
    assert diag.is_symmetric
    assert not diag.is_psd
    assert diag.minimum_eigenvalue < 0.0
    assert diag.condition_number == float("inf")
    assert diag.log_determinant is None
    # Effective rank still computes safely on positive parts
    assert 0.0 <= diag.effective_rank <= 3.0


def test_diagnose_covariance_effective_rank_extremes():
    """Effective rank equals N for identity and approaches 1 for rank-1 equicorrelated matrix."""
    # Identity matrix (4x4): all eigenvalues equal -> effective rank == 4.0
    eye_4 = np.eye(4)
    diag_eye = diagnose_covariance(eye_4)
    assert math.isclose(diag_eye.effective_rank, 4.0, abs_tol=1e-5)
    assert math.isclose(diag_eye.largest_eigenvalue_share, 0.25, abs_tol=1e-5)

    # Rank-1 matrix: [1, 1, 1, 1] outer product -> effective rank == 1.0
    ones_4 = np.ones((4, 4))
    diag_ones = diagnose_covariance(ones_4)
    assert math.isclose(diag_ones.effective_rank, 1.0, abs_tol=1e-5)
    assert math.isclose(diag_ones.largest_eigenvalue_share, 1.0, abs_tol=1e-5)


def test_diagnose_covariance_fail_closed_non_finite():
    """Non-finite (NaN/Inf) covariance elements must fail closed immediately."""
    bad_mat = np.array([
        [1.0, float("nan")],
        [float("nan"), 1.0],
    ])
    with pytest.raises(ValueError, match="non-finite"):
        diagnose_covariance(bad_mat)


def test_repair_psd_spectral_clipping(indefinite_cov_3x3):
    """Spectral clipping clamps negative eigenvalues to floor while preserving symmetry."""
    res = repair_psd_covariance(
        indefinite_cov_3x3,
        method=PSDRepairMethod.SPECTRAL_CLIPPING,
        min_eigenvalue=1e-6,
    )
    assert isinstance(res, PSDRepairResult)
    assert res.repair_method == PSDRepairMethod.SPECTRAL_CLIPPING
    assert res.original_minimum_eigenvalue < 0.0
    assert res.repaired_minimum_eigenvalue >= 1e-6 - 1e-12
    assert res.frobenius_distortion > 0.0
    assert res.relative_frobenius_distortion > 0.0
    assert res.converged
    assert res.matrix_fingerprint_before != res.matrix_fingerprint_after

    repaired_arr = np.asarray(res.repaired_matrix)
    diag_repaired = diagnose_covariance(repaired_arr)
    assert diag_repaired.is_psd
    assert diag_repaired.minimum_eigenvalue >= 1e-6 - 1e-12


def test_repair_psd_higham_nearest_correlation(indefinite_cov_3x3):
    """Higham (2002) alternating projections repairs correlation and preserves original diagonal variances."""
    orig_diag = np.diag(indefinite_cov_3x3).copy()

    res = repair_psd_covariance(
        indefinite_cov_3x3,
        method=PSDRepairMethod.HIGHAM_NEAREST_CORRELATION,
        min_eigenvalue=1e-6,
        max_iter=100,
        tol=1e-7,
    )
    assert res.repair_method == PSDRepairMethod.HIGHAM_NEAREST_CORRELATION
    assert res.original_minimum_eigenvalue < 0.0
    assert res.repaired_minimum_eigenvalue >= 1e-6 - 1e-10
    assert res.diagonal_preserved
    assert res.converged
    assert res.iterations_used >= 1

    repaired_arr = np.asarray(res.repaired_matrix)
    repaired_diag = np.diag(repaired_arr)
    # Original diagonal variances preserved
    np.testing.assert_allclose(repaired_diag, orig_diag, atol=1e-5)

    diag_repaired = diagnose_covariance(repaired_arr)
    assert diag_repaired.is_psd
    assert diag_repaired.minimum_eigenvalue >= 1e-6 - 1e-10


def test_compare_covariance_estimators_canonical_composition():
    """compare_covariance_estimators directly evaluates empirical, Ledoit-Wolf, and RegEM."""
    np.random.seed(42)
    # 100 observations of 4 assets
    rets = pd.DataFrame(
        np.random.randn(100, 4) * 0.01,
        columns=["AAPL", "MSFT", "GOOG", "AMZN"],
    )
    weights = {"AAPL": 0.25, "MSFT": 0.25, "GOOG": 0.25, "AMZN": 0.25}

    comp = compare_covariance_estimators(
        returns=rets,
        estimators=("empirical", "ledoit_wolf", "regularized_em"),
        portfolio_weights=weights,
        periods_per_year=252.0,
    )

    assert comp.estimators_compared == ("empirical", "ledoit_wolf", "regularized_em")
    assert comp.asset_order == ("AAPL", "MSFT", "GOOG", "AMZN")
    assert "empirical" in comp.diagnostics_by_estimator
    assert "ledoit_wolf" in comp.diagnostics_by_estimator
    assert "regularized_em" in comp.diagnostics_by_estimator

    # Pairwise Frobenius distances exist for all pairs
    assert "empirical_vs_ledoit_wolf" in comp.pairwise_frobenius_distances
    assert "empirical_vs_regularized_em" in comp.pairwise_frobenius_distances
    assert "ledoit_wolf_vs_regularized_em" in comp.pairwise_frobenius_distances

    # Portfolio annual volatilities computed across estimators
    for est in ("empirical", "ledoit_wolf", "regularized_em"):
        vol = comp.portfolio_volatilities_annualised[est]
        assert 0.05 <= vol <= 0.30


def test_covariance_evidence_adapters_and_artifacts(clean_cov_4x4):
    """Wrap covariance diagnostics, repair, and comparisons into EvidenceRecords and ArtifactRecords."""
    cov, assets = clean_cov_4x4
    diag = diagnose_covariance(cov, assets=assets)
    ev_diag = covariance_diagnostics_to_evidence(diag)
    assert ev_diag.test_id == "covariance.diagnostics"
    assert ev_diag.metrics["is_psd"] is True
    assert ev_diag.metrics["n_assets"] == 4

    # PSD Repair evidence
    repair = repair_psd_covariance(cov, method=PSDRepairMethod.HIGHAM_NEAREST_CORRELATION)
    ev_repair = psd_repair_to_evidence(repair)
    assert ev_repair.test_id == "covariance.psd_repair"
    assert "relative_frobenius_distortion" in ev_repair.metrics

    # Render artifacts
    with tempfile.TemporaryDirectory() as tmpdir:
        art_diag = render_covariance_diagnostics_artifact(diag, evidence_ids=(ev_diag.evidence_id,), output_dir=tmpdir)
        assert Path(art_diag.file_path).exists()

        art_repair = render_psd_repair_artifact(repair, evidence_ids=(ev_repair.evidence_id,), output_dir=tmpdir)
        assert Path(art_repair.file_path).exists()
