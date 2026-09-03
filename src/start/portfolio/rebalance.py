"""Deterministic Portfolio Rebalancing, Turnover, and Transaction Cost Engine.

Core invariants:
- Costs are strict user INPUTS; never invent 10 bps, 15 bps, or imaginary market impact.
- Turnover convention: T = 0.5 * sum(|w_new - w_old|) in [0, 1].
- Produces typed RebalanceDecision object linking pre/post risk, costs, and constraint audits.
- Zero broker / execution connectivity: analytical decision support only.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from start.portfolio.constraints import verify_portfolio_constraints
from start.portfolio.contracts import (
    RebalanceDecision,
    TransactionCostSpec,
)
from start.registry.market_contexts import PortfolioConstraints


def compute_turnover(
    w_new: dict[str, float] | pd.Series | np.ndarray,
    w_old: dict[str, float] | pd.Series | np.ndarray,
    assets: list[str] | tuple[str, ...],
) -> float:
    """Compute deterministic one-way portfolio turnover: 0.5 * sum(|w_new - w_old|)."""
    if isinstance(w_new, pd.Series):
        w1 = w_new.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    elif isinstance(w_new, dict):
        w1 = np.array([float(w_new.get(a, 0.0)) for a in assets], dtype=float)
    else:
        w1 = np.asarray(w_new, dtype=float)

    if isinstance(w_old, pd.Series):
        w0 = w_old.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    elif isinstance(w_old, dict):
        w0 = np.array([float(w_old.get(a, 0.0)) for a in assets], dtype=float)
    else:
        w0 = np.asarray(w_old, dtype=float)

    return float(0.5 * np.sum(np.abs(w1 - w0)))


def compute_transaction_costs(
    w_new: dict[str, float] | pd.Series | np.ndarray,
    w_old: dict[str, float] | pd.Series | np.ndarray,
    assets: list[str] | tuple[str, ...],
    cost_spec: TransactionCostSpec | None = None,
) -> dict[str, Any]:
    """Calculate estimated deterministic transaction costs from explicit cost specification."""
    if isinstance(w_new, pd.Series):
        w1 = w_new.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    elif isinstance(w_new, dict):
        w1 = np.array([float(w_new.get(a, 0.0)) for a in assets], dtype=float)
    else:
        w1 = np.asarray(w_new, dtype=float)

    if isinstance(w_old, pd.Series):
        w0 = w_old.reindex(assets).fillna(0.0).to_numpy(dtype=float)
    elif isinstance(w_old, dict):
        w0 = np.array([float(w_old.get(a, 0.0)) for a in assets], dtype=float)
    else:
        w0 = np.asarray(w_old, dtype=float)

    trade_sizes = np.abs(w1 - w0)
    per_asset_costs: dict[str, float] = {}
    total_cost = 0.0

    if cost_spec is None:
        return {
            "total_cost": 0.0,
            "cost_bps_total": 0.0,
            "per_asset_costs": {a: 0.0 for a in assets},
            "provenance": "no_cost_spec_supplied_gross_only",
        }

    for i, a in enumerate(assets):
        linear_bps = cost_spec.get_asset_linear_bps(a)
        spread_bps = (
            cost_spec.bid_ask_spread_bps.get(a, 0.0)
            if isinstance(cost_spec.bid_ask_spread_bps, dict)
            else float(cost_spec.bid_ask_spread_bps)
        )
        total_asset_bps = linear_bps + 0.5 * spread_bps
        asset_cost = trade_sizes[i] * (total_asset_bps / 10000.0)
        per_asset_costs[a] = round(float(asset_cost), 8)
        total_cost += asset_cost

    return {
        "total_cost": round(float(total_cost), 8),
        "cost_bps_total": round(float(total_cost * 10000.0), 4),
        "per_asset_costs": per_asset_costs,
        "provenance": cost_spec.provenance,
    }


def build_rebalance_decision(
    current_weights: pd.Series | dict[str, float] | np.ndarray,
    proposed_weights: pd.Series | dict[str, float] | np.ndarray,
    covariance: np.ndarray | pd.DataFrame,
    assets: list[str] | tuple[str, ...],
    mu: np.ndarray | pd.Series | None = None,
    cost_spec: TransactionCostSpec | None = None,
    constraints: PortfolioConstraints | None = None,
    periods_per_year: float = 252.0,
    evidence_ids: tuple[str, ...] = (),
) -> RebalanceDecision:
    """Construct an audit-grade RebalanceDecision object."""
    if isinstance(covariance, pd.DataFrame):
        sigma = covariance.to_numpy(dtype=float)
    else:
        sigma = np.asarray(covariance, dtype=float)

    if isinstance(current_weights, (pd.Series, dict)):
        missing_c = [a for a in assets if a not in current_weights]
        if missing_c:
            raise ValueError(f"Asset(s) {missing_c} missing from current_weights (fail-closed)")
        w0_dict = {a: float(current_weights[a]) for a in assets}
    else:
        w0_arr = np.asarray(current_weights, dtype=float)
        if len(w0_arr) != len(assets):
            raise ValueError(f"current_weights length {len(w0_arr)} != assets length {len(assets)}")
        w0_dict = {a: float(w0_arr[i]) for i, a in enumerate(assets)}

    if isinstance(proposed_weights, (pd.Series, dict)):
        missing_p = [a for a in assets if a not in proposed_weights]
        if missing_p:
            raise ValueError(f"Asset(s) {missing_p} missing from proposed_weights (fail-closed)")
        w1_dict = {a: float(proposed_weights[a]) for a in assets}
    else:
        w1_arr = np.asarray(proposed_weights, dtype=float)
        if len(w1_arr) != len(assets):
            raise ValueError(f"proposed_weights length {len(w1_arr)} != assets length {len(assets)}")
        w1_dict = {a: float(w1_arr[i]) for i, a in enumerate(assets)}

    w0_vec = np.array([w0_dict[a] for a in assets], dtype=float)
    w1_vec = np.array([w1_dict[a] for a in assets], dtype=float)
    trade_vec = w1_vec - w0_vec
    trade_dict = {a: round(float(trade_vec[i]), 8) for i, a in enumerate(assets)}

    turnover = compute_turnover(w1_vec, w0_vec, assets)
    cost_info = compute_transaction_costs(w1_vec, w0_vec, assets, cost_spec)
    est_cost = float(cost_info["total_cost"])
    cost_prov = cost_spec.provenance if cost_spec is not None else "COST_NOT_SUPPLIED"

    ppy = float(periods_per_year)

    # Pre-trade risk
    pre_var = float(w0_vec @ sigma @ w0_vec)
    pre_vol_ann = math.sqrt(max(0.0, pre_var)) * math.sqrt(ppy)
    pre_risk = {
        "portfolio_variance": round(pre_var, 10),
        "volatility_annualised": round(pre_vol_ann, 8),
    }

    # Post-trade risk
    post_var = float(w1_vec @ sigma @ w1_vec)
    post_vol_ann = math.sqrt(max(0.0, post_var)) * math.sqrt(ppy)
    post_risk = {
        "portfolio_variance": round(post_var, 10),
        "volatility_annualised": round(post_vol_ann, 8),
    }

    # Expected returns: separate gross and net returns across periodic and annual horizons
    ret_gross_per = None
    ret_gross_ann = None
    ret_net_per = None
    ret_net_ann = None
    if mu is not None:
        if isinstance(mu, (pd.Series, dict)):
            missing_mu = [a for a in assets if a not in mu]
            if missing_mu:
                raise ValueError(f"Asset(s) {missing_mu} missing from expected returns mu (fail-closed)")
            mu_vec = np.array([float(mu[a]) for a in assets], dtype=float)
        else:
            mu_vec = np.asarray(mu, dtype=float)
        ret_gross_per = float(w1_vec @ mu_vec)
        ret_gross_ann = ret_gross_per * ppy
        ret_net_per = ret_gross_per - est_cost
        ret_net_ann = ret_gross_ann - est_cost

    ver_res = verify_portfolio_constraints(
        weights=w1_vec,
        assets=assets,
        constraints=constraints,
        covariance=sigma,
        prior_weights=w0_vec,
    )

    return RebalanceDecision(
        current_weights={a: round(float(w0_dict[a]), 8) for a in assets},
        proposed_weights={a: round(float(w1_dict[a]), 8) for a in assets},
        trade_weights=trade_dict,
        turnover=round(turnover, 6),
        estimated_transaction_cost=round(est_cost, 8),
        constraint_verification=ver_res,
        pre_trade_risk=pre_risk,
        post_trade_risk=post_risk,
        expected_return_gross_periodic=round(ret_gross_per, 8) if ret_gross_per is not None else None,
        expected_return_gross_annualised=round(ret_gross_ann, 8) if ret_gross_ann is not None else None,
        expected_return_net_periodic=round(ret_net_per, 8) if ret_net_per is not None else None,
        expected_return_net_annualised=round(ret_net_ann, 8) if ret_net_ann is not None else None,
        cost_provenance=cost_prov,
        evidence_ids=evidence_ids,
        expected_return_gross=round(ret_gross_ann, 8) if ret_gross_ann is not None else None,
        expected_return_net=round(ret_net_ann, 8) if ret_net_ann is not None else None,
    )
