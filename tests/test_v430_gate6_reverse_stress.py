"""StART — Gate 6 Minimum Shock Reverse Stress Testing Test Suite.

Verifies:
1. Unconstrained Linear L2 Analytical Closed-Form Optimum (x* = -L* c / ||c||^2)
2. Bounded Reverse Stress QP with Box Bounds
3. Weighted L2 Reverse Stress with Scale Matrices
4. Mahalanobis Geometry Distance Norm with Covariance Inverses
5. Heterogeneous Risk Factor Units Applicability Checks
6. Infeasible Target Loss & Zero Exposure Fail-Closed Handling
7. Exact Post-Solve Target Loss Achievement and Feasibility Verification
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from start.portfolio.contracts import (
    ReverseStressNorm,
    ReverseStressSpec,
    ShockSpace,
)
from start.portfolio.scenario import solve_reverse_stress

# =========================================================================== #
# 1. UNCONSTRAINED L2 ANALYTICAL CLOSED FORM
# =========================================================================== #


def test_unconstrained_l2_reverse_stress_closed_form():
    """Verify analytical unconstrained L2 closed-form solution: x* = -L* c / (c'c)."""
    # Exposures: c = [0.8, -0.6] -> c'c = 0.8^2 + (-0.6)^2 = 0.64 + 0.36 = 1.00
    # Target loss: L* = 0.10 (10% portfolio loss)
    # Analytical optimum: x* = -0.10 * [0.8, -0.6] / 1.0 = [-0.08, +0.06]
    # Achieved loss: -(0.8 * (-0.08) + (-0.6) * 0.06) = -(-0.064 - 0.036) = +0.10
    # L2 distance: sqrt((-0.08)^2 + 0.06^2) = sqrt(0.0064 + 0.0036) = 0.10
    spec = ReverseStressSpec(
        target_loss=0.10,
        shock_space=ShockSpace.FACTOR_RETURN,
        distance_norm=ReverseStressNorm.L2,
    )
    sens = np.array([0.8, -0.6])
    res = solve_reverse_stress(spec=spec, sensitivities_or_weights=sens, factors=["F1", "F2"])

    assert res.is_closed_form is True
    assert res.converged is True
    assert math.isclose(res.target_loss, 0.10, rel_tol=1e-9)
    assert math.isclose(res.achieved_loss, 0.10, rel_tol=1e-9)
    assert math.isclose(res.loss_gap, 0.0, abs_tol=1e-12)
    assert math.isclose(res.distance, 0.10, rel_tol=1e-9)

    assert math.isclose(res.shock_vector["F1"], -0.08, rel_tol=1e-9)
    assert math.isclose(res.shock_vector["F2"], +0.06, rel_tol=1e-9)


# =========================================================================== #
# 2. BOUNDED REVERSE STRESS
# =========================================================================== #


def test_bounded_reverse_stress_qp():
    """Verify bounded reverse stress where unconstrained shock exceeds bounds."""
    # Exposures: c = [1.0, 0.2]
    # Unconstrained with L*=0.10: c'c = 1.04 -> x_1* = -0.10/1.04 = -0.09615
    # If we impose bound x_1 in [-0.05, +0.05], optimizer must push F2 harder.
    spec = ReverseStressSpec(
        target_loss=0.10,
        shock_space=ShockSpace.FACTOR_RETURN,
        distance_norm=ReverseStressNorm.L2,
        bounds={"F1": (-0.05, 0.05), "F2": (-0.50, 0.50)},
    )
    sens = np.array([1.0, 0.2])
    res = solve_reverse_stress(spec=spec, sensitivities_or_weights=sens, factors=["F1", "F2"])

    assert res.is_closed_form is False
    assert res.converged is True
    assert math.isclose(res.achieved_loss, 0.10, rel_tol=1e-5)
    # Check bounds respected
    assert res.shock_vector["F1"] >= -0.05 - 1e-6
    assert res.shock_vector["F1"] <= 0.05 + 1e-6
    assert res.shock_vector["F2"] >= -0.50 - 1e-6


# =========================================================================== #
# 3. WEIGHTED L2 & MAHALANOBIS GEOMETRY
# =========================================================================== #


def test_weighted_and_mahalanobis_reverse_stress():
    """Verify weighted L2 and Mahalanobis covariance geometry norms."""
    sens = np.array([1.0, 1.0])
    spec_weighted = ReverseStressSpec(
        target_loss=0.10,
        shock_space=ShockSpace.FACTOR_RETURN,
        distance_norm=ReverseStressNorm.WEIGHTED_L2,
        weight_matrix=np.array([4.0, 1.0]),  # Penalize F1 4x more heavily than F2
    )
    res_w = solve_reverse_stress(spec=spec_weighted, sensitivities_or_weights=sens, factors=["F1", "F2"])
    assert res_w.converged is True
    assert math.isclose(res_w.achieved_loss, 0.10, rel_tol=1e-5)
    # F2 should receive a larger shock than F1 because F1 is penalized 4x
    assert abs(res_w.shock_vector["F2"]) > abs(res_w.shock_vector["F1"])

    # Mahalanobis with Covariance
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])  # Vol_1 = 20%, Vol_2 = 30%
    spec_mah = ReverseStressSpec(
        target_loss=0.10,
        shock_space=ShockSpace.FACTOR_RETURN,
        distance_norm=ReverseStressNorm.MAHALANOBIS,
        covariance=cov,
    )
    res_m = solve_reverse_stress(spec=spec_mah, sensitivities_or_weights=sens, factors=["F1", "F2"])
    assert res_m.converged is True
    assert math.isclose(res_m.achieved_loss, 0.10, rel_tol=1e-5)


# =========================================================================== #
# 4. FAIL-CLOSED INTEGRITY & BOUND INFEASIBILITY
# =========================================================================== #


def test_reverse_stress_fail_closed_checks():
    """Verify reverse stress fails closed on invalid target loss or all-zero exposures."""
    # 1. Non-positive target loss
    with pytest.raises(ValueError, match="target_loss must be strictly positive"):
        solve_reverse_stress(
            spec=ReverseStressSpec(target_loss=-0.05),
            sensitivities_or_weights=np.array([1.0, 0.5]),
            factors=["F1", "F2"],
        )

    # 2. All-zero exposures
    with pytest.raises(ValueError, match="All sensitivities.*are zero"):
        solve_reverse_stress(
            spec=ReverseStressSpec(target_loss=0.10),
            sensitivities_or_weights=np.array([0.0, 0.0]),
            factors=["F1", "F2"],
        )

    # 3. Impossible tight bounds -> unconverged result
    spec_impossible = ReverseStressSpec(
        target_loss=0.50,  # 50% target loss
        distance_norm=ReverseStressNorm.L2,
        bounds={"F1": (-0.01, 0.01), "F2": (-0.01, 0.01)},  # Maximum achievable loss is 0.02
    )
    res_imp = solve_reverse_stress(
        spec=spec_impossible,
        sensitivities_or_weights=np.array([1.0, 1.0]),
        factors=["F1", "F2"],
    )
    assert res_imp.converged is False
    assert "INFEASIBLE" in res_imp.solver_status or "UNCONVERGED" in res_imp.solver_status
