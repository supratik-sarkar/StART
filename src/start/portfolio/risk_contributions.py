"""Deterministic portfolio risk contribution engine.

Euler variance and volatility decomposition:
- Marginal Contribution to Risk (MCR): MCR_i = (Sigma w)_i / sigma_p
- Component Contribution to Risk (CR): CR_i = w_i * MCR_i = w_i * (Sigma w)_i / sigma_p
- Percentage Contribution to Risk (%CR): %CR_i = CR_i / sigma_p = w_i * (Sigma w)_i / sigma_p^2
- Euler Theorem: sum(CR_i) = sigma_p, and sum(%CR_i) = 1.0 exactly for homogeneous degree 1 risk measures.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from start.portfolio.contracts import RiskContributionResult

DEGENERATE_VARIANCE_RELATIVE = 1e-20
RECONCILIATION_TOLERANCE = 1e-10


def calculate_risk_contributions(
    weights: pd.Series | np.ndarray | dict[str, float],
    covariance: pd.DataFrame | np.ndarray,
    assets: list[str] | tuple[str, ...] | None = None,
    cluster_map: dict[str, list[str]] | None = None,
) -> RiskContributionResult:
    """Compute Euler marginal, component, and percentage risk contributions."""
    if isinstance(covariance, pd.DataFrame):
        asset_names = list(covariance.columns)
        sigma = covariance.to_numpy(dtype=float)
    else:
        sigma = np.asarray(covariance, dtype=float)
        asset_names = list(assets) if assets is not None else [f"A{i}" for i in range(len(sigma))]

    n = len(asset_names)
    if isinstance(weights, pd.Series):
        aligned_w = weights.reindex(asset_names).fillna(0.0).to_numpy(dtype=float)
    elif isinstance(weights, dict):
        aligned_w = np.array([float(weights.get(a, 0.0)) for a in asset_names], dtype=float)
    else:
        aligned_w = np.asarray(weights, dtype=float)

    if len(aligned_w) != n or sigma.shape != (n, n):
        raise ValueError(f"Dimension mismatch: weights length {len(aligned_w)} vs covariance {sigma.shape}")

    # Variance and Volatility
    variance = float(aligned_w @ sigma @ aligned_w)
    scale = float(np.max(np.abs(np.diag(sigma)))) if sigma.size else 1.0
    floor = DEGENERATE_VARIANCE_RELATIVE * max(scale, 1e-300)

    if variance <= floor:
        # Degenerate zero variance case
        mcr = {a: 0.0 for a in asset_names}
        cr = {a: 0.0 for a in asset_names}
        pcr = {a: 1.0 / n for a in asset_names} if n > 0 else {}
        return RiskContributionResult(
            portfolio_variance=0.0,
            portfolio_volatility=0.0,
            marginal_contributions=mcr,
            component_contributions=cr,
            percentage_contributions=pcr,
            euler_reconciliation_error=0.0,
            cluster_contributions={},
            cluster_percentage_contributions={},
        )

    volatility = math.sqrt(variance)
    marginal_vec = (sigma @ aligned_w) / volatility
    component_vec = aligned_w * marginal_vec
    percentage_vec = component_vec / volatility

    mcr_dict = {a: float(m) for a, m in zip(asset_names, marginal_vec, strict=True)}
    cr_dict = {a: float(c) for a, c in zip(asset_names, component_vec, strict=True)}
    pcr_dict = {a: float(p) for a, p in zip(asset_names, percentage_vec, strict=True)}

    # Euler reconciliation error
    sum_cr = float(np.sum(component_vec))
    sum_pcr = float(np.sum(percentage_vec))
    reconciliation_error = abs(sum_cr - volatility) + abs(sum_pcr - 1.0)

    # Cluster-level risk aggregation
    cluster_cr: dict[str, float] = {}
    cluster_pcr: dict[str, float] = {}
    if cluster_map:
        for cname, members in cluster_map.items():
            c_cr = sum(cr_dict.get(m, 0.0) for m in members)
            c_pcr = sum(pcr_dict.get(m, 0.0) for m in members)
            cluster_cr[cname] = float(c_cr)
            cluster_pcr[cname] = float(c_pcr)

    return RiskContributionResult(
        portfolio_variance=round(variance, 14),
        portfolio_volatility=round(volatility, 10),
        marginal_contributions=mcr_dict,
        component_contributions=cr_dict,
        percentage_contributions=pcr_dict,
        euler_reconciliation_error=round(reconciliation_error, 14),
        cluster_contributions=cluster_cr,
        cluster_percentage_contributions=cluster_pcr,
    )
