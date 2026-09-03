"""Gate 4 Test Suite: Institutional Return & Active Performance Attribution (Factor, Brinson-Fachler, Carino Linking).

Comprehensive verification of:
1. Exact period-by-period factor return attribution: r_p = (B'w)'f + w'epsilon + residual.
2. Single-period Brinson-Fachler performance attribution: A_g + S_g + I_g = R_p - R_b.
3. Carino (1999) logarithmic multi-period geometric linking with analytical R_p -> R_b limits and fail-closed bounds.
4. Transparent residual reporting and subordinate EvidenceRecord adapters.
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
    render_brinson_attribution_artifact,
    render_carino_linking_artifact,
    render_factor_return_attribution_artifact,
)
from start.portfolio.attribution import (
    compute_brinson_attribution,
    compute_carino_multi_period_linking,
    compute_factor_return_attribution,
)
from start.portfolio.contracts import (
    BrinsonAttributionResult,
    CarinoLinkedAttributionResult,
    FactorReturnAttributionResult,
)
from start.portfolio.evidence_bridge import (
    brinson_to_evidence,
    carino_to_evidence,
    factor_return_attribution_to_evidence,
)


@pytest.fixture
def attribution_universe() -> dict[str, Any]:
    """10-period, 4-asset, 2-factor return and exposure setup."""
    np.random.seed(42)
    assets = ["AAPL", "MSFT", "XOM", "JNJ"]
    factors = ["Market", "Value"]
    n_periods = 10

    # Static exposures
    B = pd.DataFrame(
        [
            [1.20, -0.30],
            [1.10, -0.20],
            [0.70, 0.80],
            [0.60, 0.40],
        ],
        index=assets,
        columns=factors,
    )

    factor_rets = pd.DataFrame(
        np.random.randn(n_periods, 2) * 0.01 + np.array([0.005, 0.002]),
        columns=factors,
    )

    # Asset returns = B * f + epsilon
    eps = np.random.randn(n_periods, 4) * 0.005
    asset_rets_arr = factor_rets.to_numpy() @ B.to_numpy().T + eps
    asset_rets = pd.DataFrame(asset_rets_arr, columns=assets)

    weights = {"AAPL": 0.30, "MSFT": 0.30, "XOM": 0.20, "JNJ": 0.20}

    return {
        "assets": assets,
        "factors": factors,
        "exposures": B,
        "factor_returns": factor_rets,
        "asset_returns": asset_rets,
        "weights": weights,
    }


def test_factor_return_attribution_exact_reconciliation(attribution_universe):
    """Verify period factor contributions and specific contributions reconcile to total return."""
    u = attribution_universe
    res = compute_factor_return_attribution(
        returns=u["asset_returns"],
        exposures=u["exposures"],
        factor_returns=u["factor_returns"],
        weights=u["weights"],
    )

    assert isinstance(res, FactorReturnAttributionResult)
    assert res.n_periods == 10
    assert res.is_reconciled
    assert res.max_abs_reconciliation_error < 1e-14
    assert res.mean_abs_reconciliation_error < 1e-14

    # Total observed portfolio return = total factor + total specific
    expected_tot = res.total_factor_contribution + res.total_specific_contribution
    assert math.isclose(res.total_portfolio_return, expected_tot, abs_tol=1e-12)

    # Cumulative per-factor contributions sum to total factor contribution
    sum_factor_contribs = sum(res.cumulative_factor_contributions.values())
    assert math.isclose(sum_factor_contribs, res.total_factor_contribution, abs_tol=1e-12)


def test_brinson_fachler_attribution_algebraic_identity():
    """Brinson-Fachler allocation, selection, and interaction must sum exactly to active return."""
    # 3 Sectors: Tech, Energy, Healthcare
    pw = {"Tech": 0.50, "Energy": 0.30, "Healthcare": 0.20}
    bw = {"Tech": 0.30, "Energy": 0.40, "Healthcare": 0.30}

    pr = {"Tech": 0.12, "Energy": 0.04, "Healthcare": 0.06}
    br = {"Tech": 0.10, "Energy": 0.05, "Healthcare": 0.04}

    res = compute_brinson_attribution(
        portfolio_weights=pw,
        benchmark_weights=bw,
        portfolio_returns=pr,
        benchmark_returns=br,
    )

    assert isinstance(res, BrinsonAttributionResult)
    assert res.convention == "BRINSON_FACHLER"
    assert res.is_reconciled
    assert res.reconciliation_error < 1e-14

    # R_p = 0.50*0.12 + 0.30*0.04 + 0.20*0.06 = 0.060 + 0.012 + 0.012 = 0.084
    # R_b = 0.30*0.10 + 0.40*0.05 + 0.30*0.04 = 0.030 + 0.020 + 0.012 = 0.062
    # Active Return = 0.084 - 0.062 = 0.022 (2.20%)
    assert math.isclose(res.total_portfolio_return, 0.084, abs_tol=1e-10)
    assert math.isclose(res.total_benchmark_return, 0.062, abs_tol=1e-10)
    assert math.isclose(res.total_active_return, 0.022, abs_tol=1e-10)

    # Allocation for Tech: (0.50 - 0.30) * (0.10 - 0.062) = 0.20 * 0.038 = 0.0076
    assert math.isclose(res.allocation_effects["Tech"], 0.0076, abs_tol=1e-10)

    # Selection for Tech: 0.30 * (0.12 - 0.10) = 0.30 * 0.02 = 0.0060
    assert math.isclose(res.selection_effects["Tech"], 0.0060, abs_tol=1e-10)

    # Interaction for Tech: (0.50 - 0.30) * (0.12 - 0.10) = 0.20 * 0.02 = 0.0040
    assert math.isclose(res.interaction_effects["Tech"], 0.0040, abs_tol=1e-10)

    # Sum of effects = active return
    sum_effects = res.total_allocation_effect + res.total_selection_effect + res.total_interaction_effect
    assert math.isclose(sum_effects, res.total_active_return, abs_tol=1e-14)


def test_carino_multi_period_linking_two_period_fixture():
    """Verify Carino geometric active attribution linking on a two-period test fixture."""
    # Period 1
    pw1 = {"Tech": 0.60, "Energy": 0.40}
    bw1 = {"Tech": 0.50, "Energy": 0.50}
    pr1 = {"Tech": 0.10, "Energy": 0.02}
    br1 = {"Tech": 0.08, "Energy": 0.04}
    b_res1 = compute_brinson_attribution(pw1, bw1, pr1, br1)

    # Period 2
    pw2 = {"Tech": 0.55, "Energy": 0.45}
    bw2 = {"Tech": 0.50, "Energy": 0.50}
    pr2 = {"Tech": -0.04, "Energy": 0.06}
    br2 = {"Tech": -0.02, "Energy": 0.05}
    b_res2 = compute_brinson_attribution(pw2, bw2, pr2, br2)

    carino_res = compute_carino_multi_period_linking(
        period_brinson_results=[b_res1, b_res2],
        period_portfolio_returns=[b_res1.total_portfolio_return, b_res2.total_portfolio_return],
        period_benchmark_returns=[b_res1.total_benchmark_return, b_res2.total_benchmark_return],
    )

    assert isinstance(carino_res, CarinoLinkedAttributionResult)
    assert carino_res.n_periods == 2
    assert carino_res.is_reconciled
    assert carino_res.reconciliation_error < 1e-14

    # Geometric compound returns
    # R_p = (1 + R_p1)(1 + R_p2) - 1
    rp_geom = (1.0 + b_res1.total_portfolio_return) * (1.0 + b_res2.total_portfolio_return) - 1.0
    rb_geom = (1.0 + b_res1.total_benchmark_return) * (1.0 + b_res2.total_benchmark_return) - 1.0
    assert math.isclose(carino_res.total_portfolio_return_geometric, rp_geom, abs_tol=1e-12)
    assert math.isclose(carino_res.total_benchmark_return_geometric, rb_geom, abs_tol=1e-12)

    # Linked effects sum exactly to geometric active return
    sum_linked = carino_res.total_linked_allocation + carino_res.total_linked_selection + carino_res.total_linked_interaction
    assert math.isclose(sum_linked, carino_res.total_active_return_geometric, abs_tol=1e-14)


def test_carino_multi_period_linking_analytical_limit():
    """Carino coefficients evaluate smoothly when R_p equals or approaches R_b."""
    # Construct identical portfolio and benchmark returns
    pw = {"Tech": 0.50, "Energy": 0.50}
    bw = {"Tech": 0.50, "Energy": 0.50}
    pr = {"Tech": 0.05, "Energy": 0.05}
    br = {"Tech": 0.05, "Energy": 0.05}
    b_res = compute_brinson_attribution(pw, bw, pr, br)

    carino_res = compute_carino_multi_period_linking(
        period_brinson_results=[b_res],
        period_portfolio_returns=[0.05],
        period_benchmark_returns=[0.05],
    )
    assert carino_res.is_reconciled
    assert math.isclose(carino_res.total_active_return_geometric, 0.0, abs_tol=1e-14)
    # When Rp == Rb == 0.05, k_t = 1 / (1 + 0.05) = 1 / 1.05 = 0.95238095
    assert math.isclose(carino_res.period_linking_coefficients[0], 1.0 / 1.05, abs_tol=1e-6)


def test_carino_multi_period_linking_fail_closed_on_total_loss():
    """Period return <= -100% (-1.0) must fail closed as logarithm is undefined."""
    pw = {"Tech": 1.0}
    bw = {"Tech": 1.0}
    pr = {"Tech": -1.00}  # -100% total loss
    br = {"Tech": 0.00}
    b_res = compute_brinson_attribution(pw, bw, pr, br)

    with pytest.raises(ValueError, match="undefined under Carino logarithmic linking"):
        compute_carino_multi_period_linking(
            period_brinson_results=[b_res],
            period_portfolio_returns=[-1.00],
            period_benchmark_returns=[0.00],
        )


def test_attribution_evidence_adapters_and_artifacts(attribution_universe):
    """Convert factor return, Brinson, and Carino attribution results into EvidenceRecords and ArtifactRecords."""
    u = attribution_universe
    fra = compute_factor_return_attribution(
        returns=u["asset_returns"],
        exposures=u["exposures"],
        factor_returns=u["factor_returns"],
        weights=u["weights"],
    )
    ba = compute_brinson_attribution(
        portfolio_weights={"Tech": 0.60, "Energy": 0.40},
        benchmark_weights={"Tech": 0.50, "Energy": 0.50},
        portfolio_returns={"Tech": 0.10, "Energy": 0.02},
        benchmark_returns={"Tech": 0.08, "Energy": 0.04},
    )
    ca = compute_carino_multi_period_linking(
        period_brinson_results=[ba],
        period_portfolio_returns=[ba.total_portfolio_return],
        period_benchmark_returns=[ba.total_benchmark_return],
    )

    ev_fra = factor_return_attribution_to_evidence(fra)
    ev_ba = brinson_to_evidence(ba)
    ev_ca = carino_to_evidence(ca)

    assert ev_fra.test_id == "attribution.factor_performance"
    assert ev_ba.test_id == "attribution.brinson"
    assert ev_ca.test_id == "attribution.multi_period_linking"

    with tempfile.TemporaryDirectory() as tmpdir:
        art_fra = render_factor_return_attribution_artifact(fra, evidence_ids=(ev_fra.evidence_id,), output_dir=tmpdir)
        art_ba = render_brinson_attribution_artifact(ba, evidence_ids=(ev_ba.evidence_id,), output_dir=tmpdir)
        art_ca = render_carino_linking_artifact(ca, evidence_ids=(ev_ca.evidence_id,), output_dir=tmpdir)

        assert Path(art_fra.file_path).exists()
        assert Path(art_ba.file_path).exists()
        assert Path(art_ca.file_path).exists()
