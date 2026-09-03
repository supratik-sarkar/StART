"""Institutional Return and Active Performance Attribution Engine.

Core Invariants:
1. Factor Return Attribution: r_p = (B' w)' f + w' epsilon + residual. Exact period reconciliation.
2. Brinson-Fachler Performance Attribution: A_g = (w_pg - w_bg)(r_bg - R_b), S_g = w_bg(r_pg - r_bg), I_g = (w_pg - w_bg)(r_pg - r_bg). Exact active return identity.
3. Carino Multi-Period Geometric Linking: Logarithmic smoothing coefficients k_t and K with exact analytical limits for R_p -> R_b.
4. Transparent Residual Policy: Zero hidden residuals. All reconciliation errors are preserved as first-class evidence.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from start.portfolio.contracts import (
    BrinsonAttributionResult,
    CarinoLinkedAttributionResult,
    FactorReturnAttributionResult,
)


def compute_factor_return_attribution(
    returns: pd.DataFrame | np.ndarray,
    exposures: pd.DataFrame | np.ndarray,
    factor_returns: pd.DataFrame | np.ndarray,
    weights: dict[str, float] | pd.Series | np.ndarray | None = None,
    assets: list[str] | tuple[str, ...] | None = None,
    factors: list[str] | tuple[str, ...] | None = None,
    time_alignment_convention: str = "beginning_of_period_exposures",
) -> FactorReturnAttributionResult:
    """Decompose period portfolio returns into factor contributions (B'w)'f and specific contributions w'epsilon."""
    # Resolve assets & factors
    if isinstance(returns, pd.DataFrame):
        asset_names = tuple(str(c) for c in returns.columns)
        r_mat = returns.to_numpy(dtype=float)
    else:
        r_mat = np.asarray(returns, dtype=float)
        if assets is not None:
            asset_names = tuple(str(a) for a in assets)
        else:
            asset_names = tuple(f"A{i}" for i in range(r_mat.shape[1]))

    if isinstance(factor_returns, pd.DataFrame):
        factor_names = tuple(str(c) for c in factor_returns.columns)
        f_mat = factor_returns.to_numpy(dtype=float)
    else:
        f_mat = np.asarray(factor_returns, dtype=float)
        if factors is not None:
            factor_names = tuple(str(f) for f in factors)
        else:
            factor_names = tuple(f"F{i}" for i in range(f_mat.shape[1]))

    n_periods, n_assets = r_mat.shape
    n_factors = len(factor_names)

    if f_mat.shape != (n_periods, n_factors):
        raise ValueError(f"Factor returns shape {f_mat.shape} must match ({n_periods}, {n_factors})")

    # Weights
    if weights is not None:
        if isinstance(weights, (dict, pd.Series)):
            w = np.array([float(weights.get(a, 0.0)) for a in asset_names], dtype=float)
        else:
            w = np.asarray(weights, dtype=float)
    else:
        w = np.full(n_assets, 1.0 / n_assets, dtype=float)

    # Exposures B (can be static N x K or dynamic T x N x K)
    if isinstance(exposures, pd.DataFrame):
        B_static = exposures.reindex(index=list(asset_names), columns=list(factor_names)).to_numpy(
            dtype=float
        )
    else:
        B_static = np.asarray(exposures, dtype=float)

    if B_static.shape != (n_assets, n_factors):
        raise ValueError(f"Exposure matrix shape {B_static.shape} must match ({n_assets}, {n_factors})")

    period_port_returns: list[float] = []
    period_factor_contribs: list[dict[str, float]] = []
    period_spec_contribs: list[float] = []
    period_recon_errors: list[float] = []

    cumulative_factor_contribs: dict[str, float] = {f: 0.0 for f in factor_names}
    total_port_ret = 0.0
    total_factor_contrib = 0.0
    total_spec_contrib = 0.0

    b_p = B_static.T @ w  # factor exposures of portfolio

    for t in range(n_periods):
        r_t = r_mat[t]
        f_t = f_mat[t]

        r_pred = B_static @ f_t
        eps_t = r_t - r_pred

        obs_ret = float(w @ r_t)
        fact_contrib = float(b_p @ f_t)
        spec_contrib = float(w @ eps_t)

        recon_err = float(obs_ret - (fact_contrib + spec_contrib))

        fact_comp_t = {factor_names[k]: float(b_p[k] * f_t[k]) for k in range(n_factors)}
        for f in factor_names:
            cumulative_factor_contribs[f] += fact_comp_t[f]

        period_port_returns.append(round(obs_ret, 12))
        period_factor_contribs.append(fact_comp_t)
        period_spec_contribs.append(round(spec_contrib, 12))
        period_recon_errors.append(round(recon_err, 15))

        total_port_ret += obs_ret
        total_factor_contrib += fact_contrib
        total_spec_contrib += spec_contrib

    abs_errors = [abs(e) for e in period_recon_errors]
    max_err = float(max(abs_errors)) if abs_errors else 0.0
    mean_err = float(sum(abs_errors) / len(abs_errors)) if abs_errors else 0.0
    is_reconciled = bool(max_err <= 1e-8)

    return FactorReturnAttributionResult(
        n_periods=n_periods,
        factor_order=factor_names,
        asset_order=asset_names,
        total_portfolio_return=round(total_port_ret, 12),
        total_factor_contribution=round(total_factor_contrib, 12),
        total_specific_contribution=round(total_spec_contrib, 12),
        cumulative_factor_contributions={f: round(v, 12) for f, v in cumulative_factor_contribs.items()},
        period_portfolio_returns=tuple(period_port_returns),
        period_factor_contributions=tuple(period_factor_contribs),
        period_specific_contributions=tuple(period_spec_contribs),
        period_reconciliation_errors=tuple(period_recon_errors),
        max_abs_reconciliation_error=round(max_err, 15),
        mean_abs_reconciliation_error=round(mean_err, 15),
        time_alignment_convention=time_alignment_convention,
        is_reconciled=is_reconciled,
    )


def compute_brinson_attribution(
    portfolio_weights: dict[str, float] | pd.Series,
    benchmark_weights: dict[str, float] | pd.Series,
    portfolio_returns: dict[str, float] | pd.Series,
    benchmark_returns: dict[str, float] | pd.Series,
    groups: list[str] | tuple[str, ...] | None = None,
) -> BrinsonAttributionResult:
    """Compute exact single-period Brinson-Fachler active performance attribution.

    Formulas (Brinson-Fachler):
    - Allocation effect: A_g = (w_p^g - w_b^g) * (r_b^g - R_b)
    - Selection effect:  S_g = w_b^g * (r_p^g - r_b^g)
    - Interaction effect: I_g = (w_p^g - w_b^g) * (r_p^g - r_b^g)
    - Identity: sum_g (A_g + S_g + I_g) == R_p - R_b
    """
    if groups is None:
        group_set = sorted(set(portfolio_weights.keys()) | set(benchmark_weights.keys()))
        group_names = tuple(str(g) for g in group_set)
    else:
        group_names = tuple(str(g) for g in groups)

    pw = {g: float(portfolio_weights.get(g, 0.0)) for g in group_names}
    bw = {g: float(benchmark_weights.get(g, 0.0)) for g in group_names}
    pr = {g: float(portfolio_returns.get(g, 0.0)) for g in group_names}
    br = {g: float(benchmark_returns.get(g, 0.0)) for g in group_names}

    r_p = sum(pw[g] * pr[g] for g in group_names)
    r_b = sum(bw[g] * br[g] for g in group_names)
    active_ret = r_p - r_b

    alloc_effects: dict[str, float] = {}
    select_effects: dict[str, float] = {}
    inter_effects: dict[str, float] = {}

    tot_alloc = 0.0
    tot_select = 0.0
    tot_inter = 0.0

    for g in group_names:
        w_pg = pw[g]
        w_bg = bw[g]
        r_pg = pr[g]
        r_bg = br[g]

        a_g = (w_pg - w_bg) * (r_bg - r_b)
        s_g = w_bg * (r_pg - r_bg)
        i_g = (w_pg - w_bg) * (r_pg - r_bg)

        alloc_effects[g] = round(a_g, 12)
        select_effects[g] = round(s_g, 12)
        inter_effects[g] = round(i_g, 12)

        tot_alloc += a_g
        tot_select += s_g
        tot_inter += i_g

    recon_error = float(abs(active_ret - (tot_alloc + tot_select + tot_inter)))
    is_reconciled = bool(recon_error <= 1e-8)

    return BrinsonAttributionResult(
        group_names=group_names,
        portfolio_group_weights=pw,
        benchmark_group_weights=bw,
        portfolio_group_returns=pr,
        benchmark_group_returns=br,
        total_portfolio_return=round(r_p, 12),
        total_benchmark_return=round(r_b, 12),
        total_active_return=round(active_ret, 12),
        allocation_effects=alloc_effects,
        selection_effects=select_effects,
        interaction_effects=inter_effects,
        total_allocation_effect=round(tot_alloc, 12),
        total_selection_effect=round(tot_select, 12),
        total_interaction_effect=round(tot_inter, 12),
        reconciliation_error=round(recon_error, 16),
        convention="BRINSON_FACHLER",
        is_reconciled=is_reconciled,
    )


def compute_carino_multi_period_linking(
    period_brinson_results: list[BrinsonAttributionResult],
    period_portfolio_returns: list[float] | tuple[float, ...],
    period_benchmark_returns: list[float] | tuple[float, ...],
) -> CarinoLinkedAttributionResult:
    """Perform multi-period geometric active attribution linking using Carino logarithmic smoothing coefficients.

    For period t:
      k_t = (ln(1 + R_{p,t}) - ln(1 + R_{b,t})) / (R_{p,t} - R_{b,t}), with limit k_t -> 1 / (1 + R_{b,t}) as R_{p,t} -> R_{b,t}.
    For multi-period total:
      K = (ln(1 + R_p) - ln(1 + R_b)) / (R_p - R_b), with limit K -> 1 / (1 + R_b) as R_p -> R_b.
    Linking weight: w_t = k_t / K.
    """
    n_periods = len(period_brinson_results)
    if n_periods == 0:
        raise ValueError("Cannot perform multi-period linking on empty period results")

    if len(period_portfolio_returns) != n_periods or len(period_benchmark_returns) != n_periods:
        raise ValueError("Period returns lengths must match period results count")

    # Check for invalid returns <= -1.0 (-100%)
    for t, (rp_t, rb_t) in enumerate(zip(period_portfolio_returns, period_benchmark_returns, strict=True)):
        if rp_t <= -1.0 or rb_t <= -1.0:
            raise ValueError(
                f"Period {t} return <= -100% is undefined under Carino logarithmic linking (fail-closed)"
            )

    # 1. Geometric cumulative returns
    r_p_geom = float(np.prod([1.0 + r for r in period_portfolio_returns]) - 1.0)
    r_b_geom = float(np.prod([1.0 + r for r in period_benchmark_returns]) - 1.0)
    total_active_geom = r_p_geom - r_b_geom

    # 2. Period linking coefficients k_t
    k_t_list: list[float] = []
    for rp_t, rb_t in zip(period_portfolio_returns, period_benchmark_returns, strict=True):
        diff = rp_t - rb_t
        if abs(diff) > 1e-10:
            k_t = (math.log(1.0 + rp_t) - math.log(1.0 + rb_t)) / diff
        else:
            k_t = 1.0 / (1.0 + rb_t)
        k_t_list.append(k_t)

    # 3. Benchmark linking coefficient K
    diff_total = r_p_geom - r_b_geom
    if abs(diff_total) > 1e-10:
        K = (math.log(1.0 + r_p_geom) - math.log(1.0 + r_b_geom)) / diff_total
    else:
        K = 1.0 / (1.0 + r_b_geom)

    # 4. Link effects per group
    group_names = period_brinson_results[0].group_names
    linked_alloc: dict[str, float] = {g: 0.0 for g in group_names}
    linked_select: dict[str, float] = {g: 0.0 for g in group_names}
    linked_inter: dict[str, float] = {g: 0.0 for g in group_names}

    tot_linked_alloc = 0.0
    tot_linked_select = 0.0
    tot_linked_inter = 0.0

    for t in range(n_periods):
        w_t = k_t_list[t] / K
        bres = period_brinson_results[t]
        for g in group_names:
            a_linked_gt = w_t * bres.allocation_effects.get(g, 0.0)
            s_linked_gt = w_t * bres.selection_effects.get(g, 0.0)
            i_linked_gt = w_t * bres.interaction_effects.get(g, 0.0)

            linked_alloc[g] += a_linked_gt
            linked_select[g] += s_linked_gt
            linked_inter[g] += i_linked_gt

            tot_linked_alloc += a_linked_gt
            tot_linked_select += s_linked_gt
            tot_linked_inter += i_linked_gt

    sum_linked_effects = tot_linked_alloc + tot_linked_select + tot_linked_inter
    recon_error = float(abs(total_active_geom - sum_linked_effects))
    is_reconciled = bool(recon_error <= 1e-8)

    return CarinoLinkedAttributionResult(
        n_periods=n_periods,
        group_names=group_names,
        total_portfolio_return_geometric=round(r_p_geom, 12),
        total_benchmark_return_geometric=round(r_b_geom, 12),
        total_active_return_geometric=round(total_active_geom, 12),
        linked_allocation_effects={g: round(v, 12) for g, v in linked_alloc.items()},
        linked_selection_effects={g: round(v, 12) for g, v in linked_select.items()},
        linked_interaction_effects={g: round(v, 12) for g, v in linked_inter.items()},
        total_linked_allocation=round(tot_linked_alloc, 12),
        total_linked_selection=round(tot_linked_select, 12),
        total_linked_interaction=round(tot_linked_inter, 12),
        period_linking_coefficients=tuple(round(k, 10) for k in k_t_list),
        benchmark_linking_coefficient=round(K, 10),
        reconciliation_error=round(recon_error, 16),
        is_reconciled=is_reconciled,
    )
