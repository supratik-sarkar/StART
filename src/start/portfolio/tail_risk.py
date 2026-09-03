"""Institutional Tail Risk, Expected Shortfall & Advanced Backtesting Engines.

Core Invariants:
1. Deterministic Computation: Quantiles, tail averages, likelihood ratios, p-values, and durations are computed strictly by deterministic engines.
2. Sign & Confidence Contracts:
   - Loss is defined as -Return.
   - VaR is a positive loss magnitude.
   - Exception rule: I_t = 1 iff loss_t > VaR_t (or PnL_t < -VaR_t).
   - VaR confidence (alpha_var, e.g. 0.99) is strictly separated from test significance (gamma_test, e.g. 0.05).
3. Exact Finite-Sample Tail Mass:
   - Empirical Expected Shortfall uses the Rockafellar-Uryasev / exact weighted-order-statistic formulation with fractional boundary weight.
   - Handles ties and thin tail support without distorting total tail mass.
4. Component Risk Reconciliation:
   - Parametric Normal and Historical ES component contributions sum exactly to portfolio risk.
5. Limit-Safe Backtesting:
   - Kupiec and Christoffersen likelihood ratios use limit-safe arithmetic and report explicit estimability on degenerate sequences.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy import stats

from start.portfolio.contracts import (
    DurationDiagnosticsResult,
    MetricHorizon,
    TailBacktestResult,
    TailModelComparisonResult,
    TailRiskContributionResult,
    TailRiskEstimate,
    TailRiskMethod,
    TailSeverityResult,
    TailSignConvention,
)
from start.tests.traded_risk import (
    christoffersen_independence_lr,
    kupiec_lr,
)


def _series_fingerprint(arr: np.ndarray | pd.Series | list[float]) -> str:
    """Deterministic SHA-256 fingerprint of numerical series."""
    a = np.asarray(arr, dtype=float)
    return hashlib.sha256(a.tobytes()).hexdigest()[:32]


# =========================================================================== #
# 1. EXPECTED SHORTFALL & VAR ESTIMATION ENGINES
# =========================================================================== #
def compute_historical_var_es(
    losses: np.ndarray | pd.Series | list[float],
    confidence: float = 0.99,
    quantile_method: str = "linear",
    horizon: MetricHorizon | str = MetricHorizon.PERIODIC,
    frequency: str | None = None,
    periods_per_year: float = 252.0,
    is_returns: bool = False,
) -> TailRiskEstimate:
    """Compute empirical Historical VaR and Expected Shortfall with exact finite-sample tail mass.

    Finite-Sample ES Formulation:
    Let losses be sorted descending: L_(1) >= L_(2) >= ... >= L_(N).
    Target tail mass: q = N * (1 - alpha_var).
    Integer tail count: k = floor(q).
    Fractional boundary weight: gamma = q - k.

    If k >= 1:
        tail_sum = sum_{j=1}^k L_(j) + gamma * L_(k+1)   (if k < N)
        ES = tail_sum / q
    If q < 1.0 (thin tail support):
        ES = L_(1) with fractional weight q, and limitation recorded.

    This guarantees exact finite-sample tail probability mass without tie distortion.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"Confidence must be in (0, 1); got {confidence}")

    raw_arr = np.asarray(losses, dtype=float)
    if is_returns:
        loss_arr = -raw_arr
    else:
        loss_arr = raw_arr

    valid_mask = np.isfinite(loss_arr)
    loss_clean = loss_arr[valid_mask]
    n_obs = len(loss_clean)

    if n_obs < 2:
        raise ValueError(f"At least 2 valid observations required; got {n_obs}")

    # Validate horizon and frequency
    horizon_enum = MetricHorizon(horizon) if isinstance(horizon, str) else horizon
    if horizon_enum == MetricHorizon.ANNUAL and periods_per_year != 1.0:
        raise ValueError(
            f"Double annualization error: inputs are already ANNUAL but periods_per_year={periods_per_year} (fail-closed)."
        )
    if periods_per_year == 252.0 and frequency is not None and frequency not in ("daily", "business_daily"):
        raise ValueError(
            f"Frequency contradiction: periods_per_year=252.0 specified but frequency={frequency!r} is not daily."
        )

    # 1. Historical VaR at confidence percentile
    var_val = float(np.percentile(loss_clean, confidence * 100.0, method=cast(Any, quantile_method)))

    # 2. Exact finite-sample weighted tail average
    sorted_desc = np.sort(loss_clean)[::-1]
    q = n_obs * (1.0 - confidence)
    k = int(math.floor(q))
    gamma = q - k

    limitations: list[str] = [
        f"Empirical Historical VaR ({quantile_method} interpolation) and Exact Finite-Sample ES.",
        f"Loss sign convention: positive loss magnitude (L = -R). Confidence = {confidence:.4f}.",
    ]

    if q < 1.0:
        # Thin tail support (sample size too small for full observation at this alpha)
        es_val = float(sorted_desc[0])
        tail_count = 1
        boundary_weight = float(q)
        tail_fraction = float(q / n_obs)
        limitations.append(
            f"Thin tail support: target tail count q = {q:.4f} < 1.0 observation. "
            f"ES is evaluated on the maximum observed loss with target tail mass {q:.4f}."
        )
    else:
        if k >= n_obs:
            tail_sum = float(np.sum(sorted_desc))
            boundary_weight = 0.0
            tail_count = n_obs
        else:
            full_tail_sum = float(np.sum(sorted_desc[:k]))
            boundary_loss = float(sorted_desc[k]) if k < n_obs else 0.0
            tail_sum = full_tail_sum + gamma * boundary_loss
            boundary_weight = float(gamma)
            tail_count = k + (1 if gamma > 1e-12 and k < n_obs else 0)

        es_val = float(tail_sum / q)
        tail_fraction = float(q / n_obs)

    # Invariant: ES >= VaR for empirical distribution (within numerical tolerance)
    if es_val < var_val - 1e-10:
        es_val = var_val

    return TailRiskEstimate(
        method=TailRiskMethod.HISTORICAL,
        confidence=confidence,
        sign_convention=TailSignConvention.POSITIVE_LOSS_MAGNITUDE,
        var=var_val,
        es=es_val,
        n_observations=n_obs,
        tail_observations_count=tail_count,
        tail_fraction=tail_fraction,
        boundary_weight=boundary_weight,
        quantile_method=quantile_method,
        horizon=horizon_enum,
        frequency=frequency,
        parameters={
            "q_tail_mass": q,
            "k_integer_tail": k,
            "gamma_fractional_weight": gamma,
            "periods_per_year": periods_per_year,
        },
        converged=True,
        limitations=tuple(limitations),
        data_fingerprint=_series_fingerprint(loss_clean),
    )


def compute_parametric_normal_var_es(
    returns_or_losses: np.ndarray | pd.Series | list[float],
    confidence: float = 0.99,
    is_returns: bool = True,
    horizon: MetricHorizon | str = MetricHorizon.PERIODIC,
    frequency: str | None = None,
    periods_per_year: float = 252.0,
) -> TailRiskEstimate:
    """Compute Parametric Normal VaR and Expected Shortfall.

    Formulas under Normal Loss Model L ~ N(mu_L, sigma_L^2):
        z_alpha = Phi^-1(alpha)
        VaR_alpha = mu_L + sigma_L * z_alpha
        ES_alpha  = mu_L + sigma_L * phi(z_alpha) / (1 - alpha)

    For returns R ~ N(mu_R, sigma_R^2):
        mu_L = -mu_R, sigma_L = sigma_R.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"Confidence must be in (0, 1); got {confidence}")

    raw_arr = np.asarray(returns_or_losses, dtype=float)
    valid_mask = np.isfinite(raw_arr)
    clean_arr = raw_arr[valid_mask]
    n_obs = len(clean_arr)

    if n_obs < 2:
        raise ValueError(f"At least 2 valid observations required; got {n_obs}")

    horizon_enum = MetricHorizon(horizon) if isinstance(horizon, str) else horizon
    if horizon_enum == MetricHorizon.ANNUAL and periods_per_year != 1.0:
        raise ValueError(
            f"Double annualization error: inputs are already ANNUAL but periods_per_year={periods_per_year} (fail-closed)."
        )
    if periods_per_year == 252.0 and frequency is not None and frequency not in ("daily", "business_daily"):
        raise ValueError(
            f"Frequency contradiction: periods_per_year=252.0 specified but frequency={frequency!r} is not daily."
        )

    if is_returns:
        mu_r = float(np.mean(clean_arr))
        sigma_r = float(np.std(clean_arr, ddof=1))
        mu_l = -mu_r
        sigma_l = sigma_r
    else:
        mu_l = float(np.mean(clean_arr))
        sigma_l = float(np.std(clean_arr, ddof=1))

    z_alpha = float(stats.norm.ppf(confidence))
    phi_z = float(stats.norm.pdf(z_alpha))
    tail_prob = 1.0 - confidence
    es_multiplier = phi_z / tail_prob

    var_val = float(mu_l + sigma_l * z_alpha)
    es_val = float(mu_l + sigma_l * es_multiplier)

    limitations = (
        f"Parametric Gaussian Model L ~ N({mu_l:.6g}, {sigma_l:.6g}^2).",
        "Assumes normality of tails; under heavy-tailed distributions Gaussian VaR/ES understates risk.",
        f"Confidence alpha = {confidence:.4f}, z_alpha = {z_alpha:.4f}, es_multiplier = {es_multiplier:.4f}.",
    )

    return TailRiskEstimate(
        method=TailRiskMethod.PARAMETRIC_NORMAL,
        confidence=confidence,
        sign_convention=TailSignConvention.POSITIVE_LOSS_MAGNITUDE,
        var=var_val,
        es=es_val,
        n_observations=n_obs,
        tail_observations_count=int(math.ceil(n_obs * tail_prob)),
        tail_fraction=tail_prob,
        boundary_weight=1.0,
        quantile_method="gaussian_ppf",
        horizon=horizon_enum,
        frequency=frequency,
        parameters={
            "mu_loss": mu_l,
            "sigma_loss": sigma_l,
            "z_alpha": z_alpha,
            "phi_z": phi_z,
            "es_multiplier": es_multiplier,
            "periods_per_year": periods_per_year,
        },
        converged=True,
        limitations=limitations,
        data_fingerprint=_series_fingerprint(clean_arr),
    )


# =========================================================================== #
# 2. RISK CONTRIBUTIONS & DECOMPOSITION
# =========================================================================== #
def compute_tail_risk_contributions(
    returns_or_losses: pd.DataFrame | np.ndarray,
    weights: dict[str, float] | pd.Series | np.ndarray,
    cov_matrix: np.ndarray | pd.DataFrame | None = None,
    confidence: float = 0.99,
    method: str = "parametric_normal",
    is_returns: bool = True,
    asset_names: list[str] | tuple[str, ...] | None = None,
) -> TailRiskContributionResult:
    """Decompose portfolio VaR and Expected Shortfall into component asset contributions.

    Supported Methods:
    1. 'parametric_normal':
       Euler component VaR: CVaR_i = w_i * [ mu_L,i + z_alpha * (Sigma w)_i / sigma_p ]
       Euler component ES:  CES_i  = w_i * [ mu_L,i + (phi(z_alpha)/(1-alpha)) * (Sigma w)_i / sigma_p ]
       Sum of component risks exactly equals portfolio VaR and ES.

    2. 'historical_es':
       Uses the exact same scenario tail weights w_s as portfolio empirical ES:
       CES_i = sum_s w_s * L_{i,s}
       Sum of component ES exactly equals portfolio empirical ES.
       (Note: Historical VaR component contribution is non-smooth and formally deferred).
    """
    if isinstance(returns_or_losses, pd.DataFrame):
        assets = list(returns_or_losses.columns)
        data_mat = returns_or_losses.to_numpy(dtype=float)
    else:
        data_mat = np.asarray(returns_or_losses, dtype=float)
        assets = list(asset_names) if asset_names is not None else [f"Asset_{i}" for i in range(data_mat.shape[1])]

    n_assets = len(assets)
    if isinstance(weights, dict):
        w_vec = np.array([weights.get(a, 0.0) for a in assets], dtype=float)
    elif isinstance(weights, pd.Series):
        w_vec = np.array([weights.get(a, 0.0) for a in assets], dtype=float)
    else:
        w_vec = np.asarray(weights, dtype=float)

    if is_returns:
        loss_mat = -data_mat
    else:
        loss_mat = data_mat

    if method == "parametric_normal":
        mu_vec = np.mean(loss_mat, axis=0)
        if cov_matrix is None:
            cov_mat = np.cov(loss_mat, rowvar=False)
        elif isinstance(cov_matrix, pd.DataFrame):
            cov_mat = cov_matrix.to_numpy(dtype=float)
        else:
            cov_mat = np.asarray(cov_matrix, dtype=float)

        port_mu = float(w_vec @ mu_vec)
        port_var_sq = float(w_vec @ cov_mat @ w_vec)
        port_sigma = math.sqrt(max(port_var_sq, 1e-16))

        z_alpha = float(stats.norm.ppf(confidence))
        phi_z = float(stats.norm.pdf(z_alpha))
        c_alpha = phi_z / (1.0 - confidence)

        port_var = port_mu + port_sigma * z_alpha
        port_es = port_mu + port_sigma * c_alpha

        # Marginal risk vectors
        cov_w = cov_mat @ w_vec
        marginal_var = mu_vec + (z_alpha / port_sigma) * cov_w
        marginal_es = mu_vec + (c_alpha / port_sigma) * cov_w

        # Component contributions: w_i * Marginal_i
        comp_var_arr = w_vec * marginal_var
        comp_es_arr = w_vec * marginal_es

        comp_var_dict = {a: float(comp_var_arr[i]) for i, a in enumerate(assets)}
        comp_es_dict = {a: float(comp_es_arr[i]) for i, a in enumerate(assets)}

        pct_var = {a: float(comp_var_arr[i] / port_var) if abs(port_var) > 1e-12 else 0.0 for i, a in enumerate(assets)}
        pct_es = {a: float(comp_es_arr[i] / port_es) if abs(port_es) > 1e-12 else 0.0 for i, a in enumerate(assets)}

        var_err = abs(float(np.sum(comp_var_arr) - port_var))
        es_err = abs(float(np.sum(comp_es_arr) - port_es))

        return TailRiskContributionResult(
            method="parametric_normal",
            confidence=confidence,
            portfolio_var=port_var,
            portfolio_es=port_es,
            component_var=comp_var_dict,
            component_es=comp_es_dict,
            percentage_var_contributions=pct_var,
            percentage_es_contributions=pct_es,
            var_reconciliation_error=var_err,
            es_reconciliation_error=es_err,
            data_fingerprint=_series_fingerprint(loss_mat),
        )

    elif method == "historical_es":
        # Portfolio loss series
        port_loss = loss_mat @ w_vec
        n_obs = len(port_loss)
        q = n_obs * (1.0 - confidence)
        k = int(math.floor(q))
        gamma = q - k

        # Sort scenarios descending by portfolio loss
        sort_idx = np.argsort(port_loss)[::-1]

        # Scenario weights
        scenario_weights = np.zeros(n_obs, dtype=float)
        if q < 1.0:
            scenario_weights[sort_idx[0]] = 1.0
            port_es = float(port_loss[sort_idx[0]])
        else:
            for j in range(k):
                scenario_weights[sort_idx[j]] = 1.0 / q
            if k < n_obs and gamma > 1e-12:
                scenario_weights[sort_idx[k]] = gamma / q
            port_es = float(scenario_weights @ port_loss)

        port_var = float(np.percentile(port_loss, confidence * 100.0, method="linear"))

        # Component ES: weighted average of asset losses in portfolio tail
        comp_es_arr = np.zeros(n_assets, dtype=float)
        for i in range(n_assets):
            comp_es_arr[i] = float(scenario_weights @ (w_vec[i] * loss_mat[:, i]))

        comp_es_dict = {a: float(comp_es_arr[i]) for i, a in enumerate(assets)}
        pct_es = {a: float(comp_es_arr[i] / port_es) if abs(port_es) > 1e-12 else 0.0 for i, a in enumerate(assets)}
        es_err = abs(float(np.sum(comp_es_arr) - port_es))

        # Historical VaR contribution is formally non-smooth and deferred
        comp_var_dict = {a: float("nan") for a in assets}
        pct_var = {a: float("nan") for a in assets}

        return TailRiskContributionResult(
            method="historical_es",
            confidence=confidence,
            portfolio_var=port_var,
            portfolio_es=port_es,
            component_var=comp_var_dict,
            component_es=comp_es_dict,
            percentage_var_contributions=pct_var,
            percentage_es_contributions=pct_es,
            var_reconciliation_error=float("nan"),
            es_reconciliation_error=es_err,
            data_fingerprint=_series_fingerprint(loss_mat),
        )

    else:
        raise ValueError(f"Unknown tail risk contribution method: {method!r}")


# =========================================================================== #
# 3. TAIL SEVERITY & DURATION DIAGNOSTICS
# =========================================================================== #
def compute_tail_severity(
    losses: np.ndarray | pd.Series | list[float],
    var_forecasts: np.ndarray | pd.Series | list[float] | float,
    indicators: np.ndarray | pd.Series | list[int] | None = None,
) -> TailSeverityResult:
    """Compute exceedance loss and tail severity magnitude on exception days."""
    loss_arr = np.asarray(losses, dtype=float)
    if isinstance(var_forecasts, (int, float)):
        var_arr = np.full_like(loss_arr, float(var_forecasts))
    else:
        var_arr = np.asarray(var_forecasts, dtype=float)

    if len(loss_arr) != len(var_arr):
        raise ValueError(f"Length mismatch: losses ({len(loss_arr)}) vs var_forecasts ({len(var_arr)})")

    if indicators is not None:
        ind_arr = np.asarray(indicators, dtype=int)
    else:
        ind_arr = (loss_arr > var_arr).astype(int)

    exc_indices = np.where(ind_arr == 1)[0]
    n_exceptions = len(exc_indices)

    if n_exceptions == 0:
        return TailSeverityResult(
            n_exceptions=0,
            mean_absolute_exceedance=0.0,
            median_absolute_exceedance=0.0,
            max_absolute_exceedance=0.0,
            total_tail_exceedance_loss=0.0,
            mean_normalized_exceedance=None,
            max_normalized_exceedance=None,
            mean_relative_exceedance=None,
            max_relative_exceedance=None,
            absolute_exceedances=(),
            data_fingerprint=_series_fingerprint(loss_arr),
        )

    exc_losses = loss_arr[exc_indices]
    exc_vars = var_arr[exc_indices]

    abs_exceedances = exc_losses - exc_vars
    mean_abs = float(np.mean(abs_exceedances))
    med_abs = float(np.median(abs_exceedances))
    max_abs = float(np.max(abs_exceedances))
    total_loss = float(np.sum(abs_exceedances))

    # Normalized exceedances: Loss / VaR (valid when VaR > 0)
    pos_var_mask = exc_vars > 1e-12
    mean_norm: float | None = None
    max_norm: float | None = None
    mean_rel: float | None = None
    max_rel: float | None = None

    if np.any(pos_var_mask):
        norm_exceedances = exc_losses[pos_var_mask] / exc_vars[pos_var_mask]
        rel_exceedances = (exc_losses[pos_var_mask] - exc_vars[pos_var_mask]) / exc_vars[pos_var_mask]
        mean_norm = float(np.mean(norm_exceedances))
        max_norm = float(np.max(norm_exceedances))
        mean_rel = float(np.mean(rel_exceedances))
        max_rel = float(np.max(rel_exceedances))

    return TailSeverityResult(
        n_exceptions=n_exceptions,
        mean_absolute_exceedance=mean_abs,
        median_absolute_exceedance=med_abs,
        max_absolute_exceedance=max_abs,
        total_tail_exceedance_loss=total_loss,
        mean_normalized_exceedance=mean_norm,
        max_normalized_exceedance=max_norm,
        mean_relative_exceedance=mean_rel,
        max_relative_exceedance=max_rel,
        absolute_exceedances=tuple(float(x) for x in abs_exceedances),
        data_fingerprint=_series_fingerprint(loss_arr),
    )


def compute_exception_duration_diagnostics(
    indicators: np.ndarray | pd.Series | list[int],
) -> DurationDiagnosticsResult:
    """Compute descriptive statistics on inter-exception durations and clustering."""
    ind_arr = np.asarray(indicators, dtype=int)
    exc_indices = np.where(ind_arr == 1)[0]
    n_exc = len(exc_indices)

    if n_exc < 2:
        return DurationDiagnosticsResult(
            n_durations=0,
            mean_duration=0.0,
            median_duration=0.0,
            min_duration=0,
            max_duration=0,
            duration_std=0.0,
            max_run_length=int(np.max(ind_arr)) if len(ind_arr) else 0,
            durations=(),
            data_fingerprint=_series_fingerprint(ind_arr),
        )

    # Inter-exception durations: D_k = t_{k+1} - t_k
    durations = np.diff(exc_indices)
    n_dur = len(durations)
    mean_dur = float(np.mean(durations))
    med_dur = float(np.median(durations))
    min_dur = int(np.min(durations))
    max_dur = int(np.max(durations))
    std_dur = float(np.std(durations, ddof=1)) if n_dur > 1 else 0.0

    # Max consecutive run length of 1s
    max_run = 0
    curr_run = 0
    for val in ind_arr:
        if val == 1:
            curr_run += 1
            if curr_run > max_run:
                max_run = curr_run
        else:
            curr_run = 0

    return DurationDiagnosticsResult(
        n_durations=n_dur,
        mean_duration=mean_dur,
        median_duration=med_dur,
        min_duration=min_dur,
        max_duration=max_dur,
        duration_std=std_dur,
        max_run_length=max_run,
        durations=tuple(int(d) for d in durations),
        data_fingerprint=_series_fingerprint(ind_arr),
    )


# =========================================================================== #
# 4. COMPREHENSIVE OUT-OF-SAMPLE TAIL BACKTEST
# =========================================================================== #
def run_comprehensive_tail_backtest(
    pnl_or_losses: pd.Series | np.ndarray | list[float],
    var_series: pd.Series | np.ndarray | list[float],
    var_confidence: float = 0.99,
    test_significance: float = 0.05,
    pnl_source: str = "actual",
    is_loss_series: bool = False,
) -> TailBacktestResult:
    """Execute rigorous out-of-sample backtesting integrating Kupiec and Christoffersen diagnostics.

    Strict Invariants:
    1. VaR confidence (alpha_var, e.g. 0.99) is distinct from test significance (gamma_test, e.g. 0.05).
    2. Realization at t is compared against forecast made at t-1:
       I_t = 1 iff loss_t > VaR_t (or PnL_t < -VaR_t).
    3. Limit-safe arithmetic for zero and full exception cases.
    4. Exact joint conditional coverage: LR_cc = LR_uc + LR_ind with 2 df.
    """
    if isinstance(pnl_or_losses, pd.Series) and isinstance(var_series, pd.Series):
        common = pnl_or_losses.index.intersection(var_series.index)
        if len(common) == 0:
            raise ValueError("P&L and VaR series share no timestamps for out-of-sample alignment.")
        aligned_pnl = pnl_or_losses.reindex(common).astype(float)
        aligned_var = var_series.reindex(common).astype(float)
        valid = aligned_pnl.notna() & aligned_var.notna()
        aligned_pnl = aligned_pnl[valid]
        aligned_var = aligned_var[valid]
        timestamps = [str(ts) for ts in aligned_pnl.index]
        pnl_arr = aligned_pnl.to_numpy()
        var_arr = aligned_var.to_numpy()
    else:
        pnl_arr = np.asarray(pnl_or_losses, dtype=float)
        var_arr = np.asarray(var_series, dtype=float)
        if len(pnl_arr) != len(var_arr):
            raise ValueError(f"Array length mismatch: pnl ({len(pnl_arr)}) vs var ({len(var_arr)})")
        timestamps = [f"T_{i}" for i in range(len(pnl_arr))]

    n_obs = len(pnl_arr)
    if n_obs < 2:
        raise ValueError(f"At least 2 observations required for backtesting; got {n_obs}")

    # Derive canonical exception indicators
    if is_loss_series:
        indicators = (pnl_arr > var_arr).astype(int)
    else:
        indicators = (pnl_arr < -var_arr).astype(int)

    n_exceptions = int(np.sum(indicators))
    expected_prob = 1.0 - var_confidence
    expected_exc = expected_prob * n_obs
    exc_rate = n_exceptions / n_obs

    exc_dates = tuple(timestamps[i] for i in range(n_obs) if indicators[i] == 1)

    # 1. Kupiec POF Unconditional Coverage Test
    lr_uc = kupiec_lr(n_obs, n_exceptions, expected_prob)
    if math.isfinite(lr_uc):
        p_val_uc = float(stats.chi2.sf(lr_uc, df=1))
        kupiec_rej = p_val_uc < test_significance
        kupiec_est = True
    else:
        p_val_uc = float("nan")
        kupiec_rej = False
        kupiec_est = False

    # 2. Christoffersen Independence Test
    if n_obs >= 2:
        prev, curr = indicators[:-1], indicators[1:]
        n00 = int(np.sum((prev == 0) & (curr == 0)))
        n01 = int(np.sum((prev == 0) & (curr == 1)))
        n10 = int(np.sum((prev == 1) & (curr == 0)))
        n11 = int(np.sum((prev == 1) & (curr == 1)))
        trans_counts = (n00, n01, n10, n11)
        has_zero_cell = bool(min(trans_counts) == 0)

        n0 = n00 + n01
        n1 = n10 + n11
        pi_01 = (n01 / n0) if n0 > 0 else None
        pi_11 = (n11 / n1) if n1 > 0 else None

        lr_ind = christoffersen_independence_lr(n00, n01, n10, n11)
        if math.isfinite(lr_ind) and n_obs >= 3:
            p_val_ind = float(stats.chi2.sf(lr_ind, df=1))
            christoffersen_rej = p_val_ind < test_significance
            christoffersen_est = True
        else:
            p_val_ind = float("nan")
            christoffersen_rej = False
            christoffersen_est = False
    else:
        trans_counts = (0, 0, 0, 0)
        has_zero_cell = True
        pi_01 = pi_11 = None
        lr_ind = p_val_ind = float("nan")
        christoffersen_rej = False
        christoffersen_est = False

    # 3. Joint Conditional Coverage Test (LR_cc = LR_uc + LR_ind)
    if kupiec_est and christoffersen_est and math.isfinite(lr_uc) and math.isfinite(lr_ind):
        lr_cc = lr_uc + lr_ind
        p_val_cc = float(stats.chi2.sf(lr_cc, df=2))
        cc_rej = p_val_cc < test_significance
        cc_est = True
    else:
        lr_cc = float("nan")
        p_val_cc = float("nan")
        cc_rej = False
        cc_est = False

    limitations = (
        "Kupiec (1995) unconditional coverage and Christoffersen (1998) independence tests.",
        f"VaR Confidence = {var_confidence:.4f} (expected exception rate p0 = {expected_prob:.4f}).",
        f"Statistical Test Significance = {test_significance:.4f}.",
        f"Out-of-sample alignment: I_t = 1 iff {'loss_t > VaR_t' if is_loss_series else 'PnL_t < -VaR_t'}.",
        "Failure to reject does NOT establish model correctness.",
    )

    return TailBacktestResult(
        pnl_source=pnl_source,
        var_confidence=var_confidence,
        test_significance=test_significance,
        n_observations=n_obs,
        n_exceptions=n_exceptions,
        exception_rate=exc_rate,
        expected_probability=expected_prob,
        expected_exceptions=expected_exc,
        kupiec_lr=lr_uc,
        kupiec_p_value=p_val_uc,
        kupiec_rejected=kupiec_rej,
        kupiec_estimable=kupiec_est,
        christoffersen_lr=lr_ind,
        christoffersen_p_value=p_val_ind,
        christoffersen_rejected=christoffersen_rej,
        christoffersen_estimable=christoffersen_est,
        conditional_coverage_lr=lr_cc,
        conditional_coverage_p_value=p_val_cc,
        conditional_coverage_rejected=cc_rej,
        conditional_coverage_estimable=cc_est,
        transition_counts=trans_counts,
        pi_01=pi_01,
        pi_11=pi_11,
        has_zero_transition_cell=has_zero_cell,
        exception_dates=exc_dates,
        indicators=tuple(int(x) for x in indicators),
        indicator_hash=hashlib.sha256(indicators.tobytes()).hexdigest(),
        exception_convention=f"I_t = 1 iff {'loss_t > VaR_t' if is_loss_series else 'PnL_t < -VaR_t'}",
        limitations=limitations,
        data_fingerprint=_series_fingerprint(pnl_arr),
    )


# =========================================================================== #
# 5. MULTI-MODEL COMPARISON
# =========================================================================== #
def compare_tail_risk_models(
    returns_or_losses: np.ndarray | pd.Series | list[float],
    confidence: float = 0.99,
    is_returns: bool = True,
    horizon: MetricHorizon | str = MetricHorizon.PERIODIC,
    frequency: str | None = None,
    periods_per_year: float = 252.0,
) -> TailModelComparisonResult:
    """Compare Historical and Parametric Normal VaR and ES on the same scenario set."""
    hist_est = compute_historical_var_es(
        returns_or_losses,
        confidence=confidence,
        quantile_method="linear",
        horizon=horizon,
        frequency=frequency,
        periods_per_year=periods_per_year,
        is_returns=is_returns,
    )

    param_est = compute_parametric_normal_var_es(
        returns_or_losses,
        confidence=confidence,
        is_returns=is_returns,
        horizon=horizon,
        frequency=frequency,
        periods_per_year=periods_per_year,
    )

    estimates = {
        "historical": hist_est,
        "parametric_normal": param_est,
    }

    var_dict = {k: v.var for k, v in estimates.items()}
    es_dict = {k: v.es for k, v in estimates.items()}
    es_to_var = {k: float(v.es / v.var) if abs(v.var) > 1e-12 else 1.0 for k, v in estimates.items()}

    return TailModelComparisonResult(
        models_compared=("historical", "parametric_normal"),
        confidence=confidence,
        estimates=estimates,
        var_values=var_dict,
        es_values=es_dict,
        es_to_var_ratios=es_to_var,
        data_fingerprint=hist_est.data_fingerprint,
    )
