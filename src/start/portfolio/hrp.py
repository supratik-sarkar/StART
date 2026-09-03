"""Institutional Hierarchical Risk Parity (HRP) Engine, Tree Builder, and Diagnostics.

Implements:
1. Classical López de Prado (2016) HRP with multi-linkage support (Single, Complete, Average).
2. Euclidean geometry validation for Ward linkage.
3. Quasi-diagonal tree traversal and recursive bisection.
4. Typed HierarchicalTreeResult construction.
5. Cophenetic correlation diagnostic.
6. Linkage sensitivity analysis.
7. Seeded time-series block bootstrap cluster stability diagnostic.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cophenet, linkage
from scipy.spatial.distance import squareform

from start.portfolio.contracts import (
    BootstrapStabilityResult,
    CopheneticResult,
    HierarchicalTreeResult,
    LinkageSensitivityResult,
)

DEGENERATE_VARIANCE_RELATIVE = 1e-20
VALID_LINKAGES = ("single", "complete", "average", "ward")


def _matrix_hash(matrix: np.ndarray) -> str:
    """Deterministic hash of a numpy float array."""
    data = np.asarray(matrix, dtype=float)
    return hashlib.sha256(data.tobytes()).hexdigest()[:32]


def correlation_distance(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute correlation matrix and continuous correlation distance matrix d_ij = sqrt(0.5*(1 - rho_ij))."""
    variances = np.diag(covariance)
    largest = float(np.max(variances))
    floor = DEGENERATE_VARIANCE_RELATIVE * largest if largest > 0 else 0.0
    if float(np.min(variances)) <= max(floor, 0.0):
        raise ValueError(
            f"Non-positive or degenerate variance: smallest variance {float(np.min(variances)):.3g} "
            f"against largest {largest:.3g}"
        )

    scale = np.sqrt(np.outer(variances, variances))
    corr = np.clip(covariance / scale, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    # Angular correlation distance metric d = sqrt(0.5 * (1 - rho)) in [0, 1]
    distance = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - corr)))
    np.fill_diagonal(distance, 0.0)
    return corr, distance


def validate_linkage_geometry(linkage_method: str, is_euclidean: bool = False) -> None:
    """Enforce mathematical applicability: Ward linkage requires Euclidean geometry."""
    if linkage_method not in VALID_LINKAGES:
        raise ValueError(
            f"linkage_method={linkage_method!r} is not supported. Supported methods: {VALID_LINKAGES}"
        )
    if linkage_method == "ward" and not is_euclidean:
        raise ValueError(
            "Ward linkage requires Euclidean geometry; cannot be applied to an arbitrary "
            "non-Euclidean distance matrix without Euclidean coordinate representation."
        )


def quasi_diagonalize(tree: np.ndarray, n: int, assets: list[str]) -> list[str]:
    """Quasi-diagonalization: expand the linkage tree into continuous leaf order."""
    order = [int(tree[-1, 0]), int(tree[-1, 1])]
    while max(order) >= n:
        expanded: list[int] = []
        for item in order:
            if item < n:
                expanded.append(item)
            else:
                row = int(item - n)
                expanded.extend([int(tree[row, 0]), int(tree[row, 1])])
        order = expanded
    return [assets[i] for i in order]


def hrp_weights_and_tree(
    covariance: pd.DataFrame | np.ndarray,
    linkage_method: str = "single",
    assets: list[str] | tuple[str, ...] | None = None,
    is_euclidean_features: bool = False,
) -> tuple[pd.Series, HierarchicalTreeResult]:
    """Execute HRP allocation and return weights plus typed HierarchicalTreeResult."""
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
        w_series = pd.Series([1.0], index=asset_names)
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
        return w_series, tree_res

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

    def cluster_variance(members: list[str]) -> float:
        block = cov_frame.loc[members, members].to_numpy(dtype=float)
        inv = 1.0 / np.diag(block)
        w = inv / inv.sum()
        return float(w @ block @ w)

    weights = pd.Series(1.0, index=ordered_assets)
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
            v_left, v_right = cluster_variance(left), cluster_variance(right)
            alpha = 1.0 - v_left / (v_left + v_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            tree_dict["splits"].append({
                "left": left,
                "right": right,
                "v_left": round(v_left, 10),
                "v_right": round(v_right, 10),
                "alpha": round(alpha, 8),
            })

    final_weights = weights.reindex(asset_names)

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
    return final_weights, tree_result


def cophenetic_distance_diagnostic(
    covariance: pd.DataFrame | np.ndarray,
    linkage_method: str = "single",
    assets: list[str] | None = None,
) -> CopheneticResult:
    """Compute the cophenetic correlation coefficient diagnostic without pass/fail thresholds."""
    _, tree_res = hrp_weights_and_tree(covariance, linkage_method=linkage_method, assets=assets)
    return CopheneticResult(
        cophenetic_correlation=tree_res.cophenetic_correlation or 0.0,
        linkage_method=linkage_method,
        distance_method="correlation_distance",
        n_assets=len(tree_res.assets),
    )


def linkage_sensitivity_analysis(
    covariance: pd.DataFrame | np.ndarray,
    methods: tuple[str, ...] = ("single", "complete", "average"),
    assets: list[str] | None = None,
) -> LinkageSensitivityResult:
    """Evaluate weight dispersion and ordering rank correlations across linkage methods."""
    weights_map: dict[str, dict[str, float]] = {}
    eff_pos_map: dict[str, float] = {}
    var_map: dict[str, float] = {}
    orders_map: dict[str, tuple[str, ...]] = {}

    if isinstance(covariance, pd.DataFrame):
        asset_names = list(covariance.columns)
        cov_mat = covariance.to_numpy(dtype=float)
    else:
        cov_mat = np.asarray(covariance, dtype=float)
        asset_names = list(assets) if assets else [f"A{i}" for i in range(len(cov_mat))]

    for m in methods:
        w_series, t_res = hrp_weights_and_tree(cov_mat, linkage_method=m, assets=asset_names)
        w_arr = w_series.to_numpy(dtype=float)
        weights_map[m] = {a: float(w) for a, w in w_series.items()}
        h = float(np.sum(w_arr**2))
        eff_pos_map[m] = round(float(1.0 / h) if h > 0 else 0.0, 4)
        var_map[m] = round(float(w_arr @ cov_mat @ w_arr), 12)
        orders_map[m] = t_res.quasi_diagonal_order

    l1_diffs: dict[str, float] = {}
    l2_diffs: dict[str, float] = {}
    max_diffs: dict[str, float] = {}
    spearman_corrs: dict[str, float] = {}

    method_list = list(methods)
    for i in range(len(method_list)):
        for j in range(i + 1, len(method_list)):
            m1, m2 = method_list[i], method_list[j]
            pair_key = f"{m1}_vs_{m2}"
            w1 = np.array([weights_map[m1][a] for a in asset_names], dtype=float)
            w2 = np.array([weights_map[m2][a] for a in asset_names], dtype=float)
            l1_diffs[pair_key] = round(float(0.5 * np.sum(np.abs(w1 - w2))), 6)
            l2_diffs[pair_key] = round(float(np.linalg.norm(w1 - w2)), 6)
            max_diffs[pair_key] = round(float(np.max(np.abs(w1 - w2))), 6)

            # Spearman correlation of weights
            r1 = pd.Series(w1).rank()
            r2 = pd.Series(w2).rank()
            spearman = float(r1.corr(r2))
            spearman_corrs[pair_key] = round(spearman if math.isfinite(spearman) else 1.0, 6)

    return LinkageSensitivityResult(
        methods_compared=methods,
        weights_by_linkage=weights_map,
        effective_positions_by_linkage=eff_pos_map,
        portfolio_variance_by_linkage=var_map,
        pairwise_l1_distances=l1_diffs,
        pairwise_l2_distances=l2_diffs,
        max_asset_weight_diffs=max_diffs,
        spearman_order_correlations=spearman_corrs,
    )


def bootstrap_cluster_stability(
    returns: pd.DataFrame,
    n_replicates: int = 50,
    block_size: int = 21,
    seed: int = 42,
    linkage_method: str = "single",
) -> BootstrapStabilityResult:
    """Seeded stationary block bootstrap cluster stability diagnostic."""
    rng = np.random.default_rng(seed)
    assets = tuple(str(c) for c in returns.columns)
    n_assets = len(assets)
    n_obs = len(returns)

    if n_obs < block_size * 2 or n_assets < 2:
        # Trivial single asset or insufficient observations
        matrix = [[1.0] * n_assets for _ in range(n_assets)]
        return BootstrapStabilityResult(
            bootstrap_method="stationary_block_bootstrap",
            block_size=block_size,
            n_replicates=n_replicates,
            seed=seed,
            assets=assets,
            pairwise_co_clustering_matrix=matrix,
            mean_pairwise_stability=1.0,
            min_pairwise_stability=1.0,
            cophenetic_stability_mean=1.0,
        )

    co_clustering = np.zeros((n_assets, n_assets), dtype=float)
    coph_values: list[float] = []

    for _ in range(n_replicates):
        # Generate block indices with geometric block length / circular wrap
        indices: list[int] = []
        while len(indices) < n_obs:
            start = int(rng.integers(0, n_obs))
            length = int(rng.geometric(1.0 / max(block_size, 1)))
            for k in range(length):
                indices.append(int((start + k) % n_obs))
                if len(indices) == n_obs:
                    break

        boot_data = returns.iloc[indices]
        boot_cov = boot_data.cov().to_numpy(dtype=float)

        try:
            _, tree_res = hrp_weights_and_tree(
                boot_cov, linkage_method=linkage_method, assets=list(assets)
            )
            if tree_res.cophenetic_correlation is not None:
                coph_values.append(tree_res.cophenetic_correlation)

            # Check co-clustering at top split level
            if tree_res.cluster_tree.get("splits"):
                left_assets = set(tree_res.cluster_tree["splits"][0]["left"])
                right_assets = set(tree_res.cluster_tree["splits"][0]["right"])
                for i, a1 in enumerate(assets):
                    for j, a2 in enumerate(assets):
                        if (a1 in left_assets and a2 in left_assets) or (
                            a1 in right_assets and a2 in right_assets
                        ):
                            co_clustering[i, j] += 1.0
            else:
                co_clustering += 1.0
        except Exception:
            continue

    co_matrix = co_clustering / max(len(coph_values), 1)
    np.fill_diagonal(co_matrix, 1.0)

    # Summary statistics (off-diagonal)
    off_diag = co_matrix[~np.eye(n_assets, dtype=bool)]
    mean_stab = float(np.mean(off_diag)) if len(off_diag) else 1.0
    min_stab = float(np.min(off_diag)) if len(off_diag) else 1.0
    mean_coph = float(np.mean(coph_values)) if coph_values else 0.0

    return BootstrapStabilityResult(
        bootstrap_method="stationary_block_bootstrap",
        block_size=block_size,
        n_replicates=n_replicates,
        seed=seed,
        assets=assets,
        pairwise_co_clustering_matrix=[[round(float(x), 4) for x in row] for row in co_matrix],
        mean_pairwise_stability=round(mean_stab, 6),
        min_pairwise_stability=round(min_stab, 6),
        cophenetic_stability_mean=round(mean_coph, 6),
    )
