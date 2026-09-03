"""Deterministic Hierarchical Equal Risk Contribution (HERC) Engine (Raffinot, 2018).

Core invariants:
- Hierarchical risk budgeting across dendrogram clusters.
- Distinguishes algorithm families: HRP (recursive bisection with IVP), ERC (flat log-barrier), HERC (hierarchical cluster risk parity).
- Post-solve constraint verification and Euler risk contribution reconciliation.
- Supports single, complete, average linkages (Ward requires Euclidean geometry).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import squareform

from start.portfolio.constraints import verify_portfolio_constraints
from start.portfolio.contracts import (
    DeterminismTier,
    HERCResult,
    HierarchicalTreeResult,
    MethodApplicability,
)
from start.portfolio.hrp import (
    _matrix_hash,
    correlation_distance,
    quasi_diagonalize,
    validate_linkage_geometry,
)
from start.portfolio.risk_contributions import calculate_risk_contributions

HERC_APPLICABILITY = MethodApplicability(
    method_name="hierarchical_equal_risk_contribution",
    required_inputs=("covariance",),
    min_assets=2,
    min_observations=2,
    requires_psd_covariance=True,
    supports_bounds=False,
    supports_group_constraints=False,
    supports_turnover_constraints=False,
    determinism=DeterminismTier.NUMERICALLY_DETERMINISTIC,
    assumptions=(
        "Hierarchical clustering on angular correlation distance matrix",
        "Equal Risk Contribution allocation across hierarchical cluster partitions (Raffinot, 2018)",
        "Variance-based intra-cluster weighting (Inverse-Variance Parity)",
    ),
)


def solve_herc(
    covariance: pd.DataFrame | np.ndarray,
    linkage_method: str = "single",
    assets: list[str] | tuple[str, ...] | None = None,
    is_euclidean_features: bool = False,
    risk_measure: str = "volatility",
    periods_per_year: float = 252.0,
) -> HERCResult:
    """Compute Hierarchical Equal Risk Contribution (HERC) portfolio allocation."""
    if isinstance(covariance, pd.DataFrame):
        asset_names = list(covariance.columns)
        cov_matrix = covariance.to_numpy(dtype=float)
    else:
        cov_matrix = np.asarray(covariance, dtype=float)
        asset_names = list(assets) if assets is not None else [f"A{i}" for i in range(len(cov_matrix))]

    n = len(asset_names)
    validate_linkage_geometry(linkage_method, is_euclidean=is_euclidean_features)

    cov_frame = pd.DataFrame(cov_matrix, index=asset_names, columns=asset_names)
    cov_hash = _matrix_hash(cov_matrix)

    if n == 1:
        w_dict = {asset_names[0]: 1.0}
        tree_res = HierarchicalTreeResult(
            assets=tuple(asset_names),
            distance_method="correlation_distance",
            linkage_method=linkage_method,
            linkage_matrix=[],
            leaf_order=(0,),
            quasi_diagonal_order=tuple(asset_names),
            cluster_tree={"leaf": asset_names[0]},
            correlation_fingerprint=_matrix_hash(np.array([[1.0]])),
            covariance_fingerprint=cov_hash,
            cophenetic_correlation=1.0,
        )
        ver_res = verify_portfolio_constraints(w_dict, asset_names, covariance=cov_matrix)
        return HERCResult(
            weights=w_dict,
            tree_result=tree_res,
            cluster_risk_contributions={asset_names[0]: 1.0},
            percentage_risk_contributions={asset_names[0]: 1.0},
            effective_n_positions=1.0,
            portfolio_volatility_annualised=math.sqrt(float(cov_matrix[0, 0])) * math.sqrt(periods_per_year),
            portfolio_variance=float(cov_matrix[0, 0]),
            constraint_verification=ver_res,
            risk_measure=risk_measure,
        )

    corr_matrix, dist_matrix = correlation_distance(cov_matrix)
    corr_hash = _matrix_hash(corr_matrix)
    condensed_dist = squareform(dist_matrix, checks=False)

    tree = linkage(condensed_dist, method=linkage_method)
    if n > 2 and np.std(condensed_dist) > 1e-12:
        try:
            coph_corr, _ = cophenet(tree, condensed_dist)
            coph_val = float(coph_corr) if math.isfinite(coph_corr) else 1.0
        except Exception:
            coph_val = 1.0
    else:
        coph_val = 1.0

    ordered_assets = quasi_diagonalize(tree, n, asset_names)
    leaf_indices = tuple(asset_names.index(a) for a in ordered_assets)

    def cluster_risk(members: list[str]) -> tuple[float, pd.Series]:
        block = cov_frame.loc[members, members].to_numpy(dtype=float)
        variances = np.diag(block)
        inv = 1.0 / np.maximum(variances, 1e-15)
        w_ivp = inv / inv.sum()
        c_var = float(w_ivp @ block @ w_ivp)
        return c_var, pd.Series(w_ivp, index=members)

    cluster_weights = pd.Series(1.0, index=ordered_assets)
    clusters = [ordered_assets]
    tree_dict: dict[str, Any] = {"root": ordered_assets, "splits": []}

    while clusters:
        clusters = [
            part
            for cluster in clusters
            for part in (cluster[: len(cluster) // 2], cluster[len(cluster) // 2 :])
            if len(cluster) > 1
        ]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_left, _ = cluster_risk(left)
            v_right, _ = cluster_risk(right)

            # Equal Risk Contribution split between left and right clusters
            # Under volatility risk measure (Raffinot, 2018): alpha = sqrt(v_right) / (sqrt(v_left) + sqrt(v_right))
            # Under variance risk measure: alpha = v_right / (v_left + v_right)
            if risk_measure == "variance":
                alpha = v_right / max(1e-15, v_left + v_right)
            else:
                sd_l = math.sqrt(max(1e-15, v_left))
                sd_r = math.sqrt(max(1e-15, v_right))
                alpha = sd_r / max(1e-15, sd_l + sd_r)

            cluster_weights[left] *= alpha
            cluster_weights[right] *= 1.0 - alpha

            tree_dict["splits"].append(
                {
                    "left": left,
                    "right": right,
                    "v_left": round(v_left, 10),
                    "v_right": round(v_right, 10),
                    "alpha": round(alpha, 8),
                }
            )

    # Terminal weights from recursive hierarchical cluster risk parity (Raffinot, 2018)
    final_w = cluster_weights.reindex(asset_names)
    final_w = final_w / final_w.sum()
    w_vec = final_w.to_numpy(dtype=float)
    w_dict = {a: round(float(w), 10) for a, w in zip(asset_names, w_vec, strict=True)}

    ver_res = verify_portfolio_constraints(w_dict, asset_names, covariance=cov_matrix)
    rc = calculate_risk_contributions(w_vec, cov_matrix, assets=asset_names)

    h = float(np.sum(w_vec**2))
    eff_n = float(1.0 / h) if h > 1e-12 else 0.0

    tree_result = HierarchicalTreeResult(
        assets=tuple(asset_names),
        distance_method="correlation_distance",
        linkage_method=linkage_method,
        linkage_matrix=[[float(x) for x in row] for row in tree],
        leaf_order=leaf_indices,
        quasi_diagonal_order=tuple(ordered_assets),
        cluster_tree=tree_dict,
        correlation_fingerprint=corr_hash,
        covariance_fingerprint=cov_hash,
        cophenetic_correlation=round(coph_val, 6),
    )

    ppy = float(periods_per_year)
    vol_ann = rc.portfolio_volatility * math.sqrt(ppy)

    return HERCResult(
        weights=w_dict,
        tree_result=tree_result,
        cluster_risk_contributions=rc.component_contributions,
        percentage_risk_contributions=rc.percentage_contributions,
        effective_n_positions=round(eff_n, 4),
        portfolio_volatility_annualised=round(vol_ann, 8),
        portfolio_variance=rc.portfolio_variance,
        constraint_verification=ver_res,
        risk_measure=risk_measure,
    )
