"""Synthetic market world generator.

Why this exists before any analytics
------------------------------------

Gate B's estimators — HRP, cross-sectional WLS, CEV, Stanton, RegEM — are the kind of
code where a subtle error survives review and surfaces two years later. Testing them
against real data would only establish that they *run*.

So the generator comes first, and it produces **ground truth that is known independently
of any estimator**. The true factor covariance is the matrix used to draw the factor
returns, not something recovered from them. The true γ is the parameter fed to the
short-rate SDE. The true barrier-crossing probability is the closed-form log-space
expression, not a count from the simulated paths.

That converts most of Gate B from "does it run" into known-answer testing: generate a
CIR path with γ = 0.5 and confirm the estimator recovers 0.5 within its interval;
mis-specify a VaR series by 1.5× and confirm Kupiec rejects; mask 20% MCAR and compare
RegEM against pairwise deletion on a covariance you already know.

**Never validate an estimator against itself.** Every ground-truth quantity below is a
generator input or a closed-form consequence of one.

Determinism
-----------

Every draw comes from a seeded ``numpy.random.default_rng``. Same seed, same world,
bit-for-bit — including under the adversarial modes. ``n_replications`` derives
independent child seeds rather than re-using one stream, so Monte Carlo replications are
genuinely independent instead of correlated slices of a single sequence.

No network. No external data. Nothing here reaches outside the process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from start.registry.market_contexts import (
    MarketContext,
    PortfolioConstraints,
    PortfolioSpec,
    ShortRateContext,
)

__all__ = [
    "MarketWorld",
    "generate_market_world",
    "generate_short_rate_path",
    "generate_barrier_paths",
    "barrier_crossing_probability",
    "ADVERSARIAL_MODES",
]

#: Adversarial modes, each deterministic under seed.
ADVERSARIAL_MODES: tuple[str, ...] = (
    "regime_shift", "fat_tails", "near_singular", "constant_asset",
    "mcar_missing", "mar_missing", "var_misspecification",
)


@dataclass
class MarketWorld:
    """A generated world plus the ground truth used to generate it.

    The ``true_*`` fields are generator **inputs** or closed-form consequences of them.
    None is recovered from the simulated data, so comparing an estimator against them is
    a genuine known-answer test.
    """

    returns: pd.DataFrame
    prices: pd.DataFrame
    factor_returns: pd.DataFrame
    factor_exposures: pd.DataFrame
    weights: pd.Series
    benchmark_weights: pd.Series
    pnl: pd.Series
    hypothetical_pnl: pd.Series
    var_series: pd.Series
    var_confidence: float

    # -- ground truth ------------------------------------------------------
    true_factor_covariance: pd.DataFrame
    true_specific_variance: pd.Series
    true_asset_covariance: pd.DataFrame
    true_portfolio_variance: float
    var_misspecification_factor: float

    # -- optional artefacts ------------------------------------------------
    incomplete_returns: pd.DataFrame | None = None
    missing_mask: pd.DataFrame | None = None
    missing_mechanism: str = ""
    short_rate: pd.Series | None = None
    true_short_rate_params: dict[str, float] = field(default_factory=dict)
    barrier_paths: pd.DataFrame | None = None
    true_barrier_probability: float | None = None
    barrier_level: float | None = None
    asset_metadata: pd.DataFrame | None = None
    prior_weights: pd.Series | None = None
    constraints: PortfolioConstraints | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    periods_per_year: float = 252.0
    seed: int = 42
    modes: tuple[str, ...] = field(default=())

    # -- contexts ----------------------------------------------------------
    def market_context(self, *, incomplete: bool = False) -> MarketContext:
        """Wrap the world as a :class:`MarketContext`."""
        extra_dict = dict(self.extra)
        return MarketContext(
            returns=self.incomplete_returns if incomplete else self.returns,
            prices=self.prices,
            periods_per_year=self.periods_per_year,
            return_basis="simple",
            risk_free_rate=0.02,
            risk_free_frequency="annual",
            factor_returns=self.factor_returns,
            factor_exposures=self.factor_exposures,
            pnl=self.pnl,
            hypothetical_pnl=self.hypothetical_pnl,
            var_series=self.var_series,
            var_confidence=self.var_confidence,
            asset_metadata=self.asset_metadata,
            portfolio=PortfolioSpec(
                weights=self.weights,
                benchmark_weights=self.benchmark_weights,
                prior_weights=self.prior_weights,
                constraints=self.constraints,
            ),
            seed=self.seed,
            extra=extra_dict,
        )

    def short_rate_context(self) -> ShortRateContext | None:
        if self.short_rate is None:
            return None
        return ShortRateContext(
            rates=self.short_rate,
            units="decimal",
            periods_per_year=self.periods_per_year,
            min_observations=min(250, int(self.short_rate.size)),
            seed=self.seed,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "n_assets": int(self.returns.shape[1]),
            "n_periods": int(self.returns.shape[0]),
            "n_factors": int(self.factor_returns.shape[1]),
            "periods_per_year": self.periods_per_year,
            "seed": self.seed,
            "modes": list(self.modes),
            "true_portfolio_variance": self.true_portfolio_variance,
            "var_confidence": self.var_confidence,
            "var_misspecification_factor": self.var_misspecification_factor,
            "missing_mechanism": self.missing_mechanism,
            "has_short_rate": self.short_rate is not None,
            "has_barrier_paths": self.barrier_paths is not None,
        }


# --------------------------------------------------------------------------- #
# Short rate
# --------------------------------------------------------------------------- #
def generate_short_rate_path(
    *,
    n_periods: int = 2500,
    gamma: float = 0.5,
    kappa: float = 0.3,
    theta: float = 0.04,
    sigma: float = 0.08,
    r0: float = 0.04,
    periods_per_year: float = 252.0,
    seed: int = 42,
    start: str = "2015-01-01",
) -> tuple[pd.Series, dict[str, float]]:
    """CEV short-rate path: ``dr = kappa(theta - r)dt + sigma r^gamma dW``.

    ``gamma`` selects the family — 0 is Vasicek, 0.5 is CIR, 1.0 is
    Brennan–Schwartz — and is returned as ground truth for the CEV estimator to recover.

    The path is reflected at a small positive floor rather than allowed to go negative.
    Euler discretisation of a CEV process can cross zero for γ < 1, and a negative rate
    makes ``r^gamma`` undefined; reflecting keeps the process in its domain without
    changing the local dynamics the estimator sees.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    sqrt_dt = np.sqrt(dt)
    floor = 1e-4

    rates = np.empty(n_periods, dtype=float)
    rates[0] = r0
    for t in range(1, n_periods):
        previous = rates[t - 1]
        drift = kappa * (theta - previous) * dt
        diffusion = sigma * (previous ** gamma) * sqrt_dt * rng.standard_normal()
        value = previous + drift + diffusion
        rates[t] = value if value > floor else floor + (floor - value)

    index = pd.date_range(start, periods=n_periods, freq="B")
    truth = {
        "gamma": float(gamma), "kappa": float(kappa), "theta": float(theta),
        "sigma": float(sigma), "r0": float(r0), "dt": float(dt),
    }
    return pd.Series(rates, index=index, name="short_rate"), truth


# --------------------------------------------------------------------------- #
# Barrier paths
# --------------------------------------------------------------------------- #
def barrier_crossing_probability(
    start_price: float, end_price: float, barrier: float, sigma: float, dt: float,
    *, direction: str = "up",
) -> float:
    """Closed-form Brownian-bridge crossing probability, in **log-price** space.

    For GBM the bridge is Brownian in ``log S``, so the standard reflection result
    applies to log distances::

        P(cross upper H) = exp( -2 * ln(H/a) * ln(H/b) / (sigma^2 * dt) )

    Applying the arithmetic-price form to GBM prices is a real and easy mistake, and it
    understates crossing probability at exactly the levels that matter.
    """
    if sigma <= 0 or dt <= 0:
        return 0.0
    if direction == "up":
        if start_price >= barrier or end_price >= barrier:
            return 1.0
        numerator = np.log(barrier / start_price) * np.log(barrier / end_price)
    else:
        if start_price <= barrier or end_price <= barrier:
            return 1.0
        numerator = np.log(start_price / barrier) * np.log(end_price / barrier)
    return float(np.exp(-2.0 * numerator / (sigma * sigma * dt)))


def generate_barrier_paths(
    *,
    n_paths: int = 500,
    n_steps: int = 50,
    s0: float = 100.0,
    mu: float = 0.0,
    sigma: float = 0.25,
    barrier: float = 115.0,
    periods_per_year: float = 252.0,
    seed: int = 42,
    fine_steps_per_step: int = 40,
) -> tuple[pd.DataFrame, float]:
    """GBM paths plus the **true** continuous crossing probability.

    Ground truth is obtained from a much finer simulation (``fine_steps_per_step``×),
    not from the coarse observed path — otherwise the "truth" would be the very
    discrete-monitoring estimate the test exists to show is biased.
    """
    dt = 1.0 / periods_per_year
    fine_dt = dt / fine_steps_per_step

    rng = np.random.default_rng(seed)
    coarse = np.empty((n_paths, n_steps + 1), dtype=float)
    coarse[:, 0] = s0
    crossed_fine = np.zeros(n_paths, dtype=bool)

    for path in range(n_paths):
        price = s0
        for step in range(n_steps):
            for _ in range(fine_steps_per_step):
                shock = rng.standard_normal()
                price *= np.exp(
                    (mu - 0.5 * sigma * sigma) * fine_dt
                    + sigma * np.sqrt(fine_dt) * shock
                )
                if price >= barrier:
                    crossed_fine[path] = True
            coarse[path, step + 1] = price

    frame = pd.DataFrame(
        coarse,
        index=pd.Index(range(n_paths), name="path"),
        columns=pd.Index(range(n_steps + 1), name="step"),
    )
    return frame, float(crossed_fine.mean())


# --------------------------------------------------------------------------- #
# The world
# --------------------------------------------------------------------------- #
def generate_market_world(
    *,
    n_assets: int = 50,
    n_periods: int = 1000,
    n_factors: int = 5,
    seed: int = 42,
    periods_per_year: float = 252.0,
    missing_rate: float = 0.0,
    missing_mechanism: str = "mcar",
    regime_shift_at: int | None = None,
    fat_tails: bool = False,
    near_singular: bool = False,
    constant_asset: bool = False,
    var_confidence: float = 0.99,
    var_misspecification_factor: float = 1.0,
    include_short_rate: bool = False,
    short_rate_gamma: float = 0.5,
    include_barrier_paths: bool = False,
    n_replications: int = 1,
    replication: int = 0,
    start: str = "2020-01-01",
) -> MarketWorld:
    """Generate one coherent market world with known ground truth.

    ``replication`` selects an independent child seed. Monte Carlo studies must use it
    rather than slicing one long stream, because consecutive slices of a single
    generator are not independent draws and the resulting size and power estimates
    would be wrong in a way that is invisible.
    """
    if n_replications > 1 or replication:
        child = np.random.SeedSequence(seed).spawn(max(n_replications, replication + 1))
        rng = np.random.default_rng(child[replication])
    else:
        rng = np.random.default_rng(seed)

    modes: list[str] = []
    asset_ids = [f"A{i:03d}" for i in range(n_assets)]
    factor_ids = [f"F{i + 1}" for i in range(n_factors)]
    index = pd.date_range(start, periods=n_periods, freq="B")

    # -- factor covariance (ground truth) ----------------------------------
    loadings = rng.normal(0.0, 1.0, (n_factors, n_factors))
    factor_cov = loadings @ loadings.T / n_factors
    if near_singular:
        # Collapse the last factor onto the first: a genuinely rank-deficient
        # structure, which is what stresses conditioning and shrinkage.
        factor_cov[-1, :] = factor_cov[0, :]
        factor_cov[:, -1] = factor_cov[:, 0]
        modes.append("near_singular")
    factor_cov *= (0.15**2) / periods_per_year
    factor_cov = (factor_cov + factor_cov.T) / 2.0

    # -- factor returns ----------------------------------------------------
    if fat_tails:
        # Student-t with 4 df, rescaled so the covariance matches the target: the
        # tails change while the second moment does not, isolating the tail effect.
        df = 4.0
        raw = rng.standard_t(df, size=(n_periods, n_factors)) * np.sqrt((df - 2.0) / df)
        chol = np.linalg.cholesky(factor_cov + np.eye(n_factors) * 1e-12)
        factor_draws = raw @ chol.T
        modes.append("fat_tails")
    else:
        factor_draws = rng.multivariate_normal(
            np.zeros(n_factors), factor_cov, size=n_periods
        )

    if regime_shift_at is not None and 0 < regime_shift_at < n_periods:
        factor_draws[regime_shift_at:] *= 2.0
        factor_draws[regime_shift_at:] += 0.0008
        modes.append("regime_shift")

    factor_returns = pd.DataFrame(factor_draws, index=index, columns=factor_ids)

    # -- exposures and specific risk (ground truth) -------------------------
    exposures = rng.normal(0.0, 1.0, (n_assets, n_factors))
    specific_sd = rng.uniform(0.10, 0.35, n_assets) / np.sqrt(periods_per_year)
    if constant_asset:
        exposures[-1, :] = 0.0
        specific_sd[-1] = 0.0
        modes.append("constant_asset")

    exposures_frame = pd.DataFrame(exposures, index=asset_ids, columns=factor_ids)
    specific_variance = pd.Series(specific_sd**2, index=asset_ids, name="specific_variance")

    # -- asset returns: r = X f + e ----------------------------------------
    specific = rng.normal(0.0, 1.0, (n_periods, n_assets)) * specific_sd
    returns = pd.DataFrame(
        factor_draws @ exposures.T + specific, index=index, columns=asset_ids
    )
    prices = (1.0 + returns).cumprod() * 100.0

    # True asset covariance = X F X' + D, from the generator's own inputs.
    true_asset_cov = pd.DataFrame(
        exposures @ factor_cov @ exposures.T + np.diag(specific_sd**2),
        index=asset_ids, columns=asset_ids,
    )

    # -- portfolio ---------------------------------------------------------
    raw_weights = rng.uniform(0.5, 1.5, n_assets)
    weights = pd.Series(raw_weights / raw_weights.sum(), index=asset_ids, name="weight")
    benchmark = pd.Series(np.full(n_assets, 1.0 / n_assets), index=asset_ids, name="benchmark")
    true_portfolio_variance = float(weights.to_numpy() @ true_asset_cov.to_numpy() @ weights.to_numpy())

    # -- P&L: hypothetical is frozen-position, actual adds trading and cost --
    hypothetical_pnl = (returns @ weights).rename("hypothetical_pnl")
    trading_noise = rng.normal(0.0, hypothetical_pnl.std() * 0.05, n_periods)
    pnl = (hypothetical_pnl + trading_noise - 0.00002).rename("pnl")

    # -- VaR: the TRUE quantile of the generating distribution --------------
    from scipy import stats as sp

    z = float(sp.norm.ppf(1.0 - var_confidence))
    true_sigma = float(np.sqrt(true_portfolio_variance))
    true_var = -z * true_sigma
    if var_misspecification_factor != 1.0:
        modes.append("var_misspecification")
    var_series = pd.Series(
        np.full(n_periods, true_var * var_misspecification_factor),
        index=index, name="var",
    )

    # -- missingness -------------------------------------------------------
    incomplete = None
    mask = None
    mechanism = ""
    if missing_rate > 0:
        mechanism = missing_mechanism.lower()
        if mechanism == "mcar":
            mask = pd.DataFrame(
                rng.random((n_periods, n_assets)) < missing_rate,
                index=index, columns=asset_ids,
            )
            modes.append("mcar_missing")
        elif mechanism == "mar":
            # Missingness depends on an OBSERVED variable (the first asset's return),
            # which is what makes it MAR rather than MNAR. Depending on the missing
            # value itself would be MNAR and would break RegEM's stated assumptions.
            driver = returns[asset_ids[0]].to_numpy()
            centred = (driver - driver.mean()) / (driver.std() + 1e-12)
            probability = np.clip(
                missing_rate * (1.0 + 1.5 * centred[:, None]), 0.0, 0.95
            )
            mask = pd.DataFrame(
                rng.random((n_periods, n_assets)) < probability,
                index=index, columns=asset_ids,
            )
            mask.iloc[:, 0] = False      # the driver stays observed
            modes.append("mar_missing")
        else:
            raise ValueError(f"missing_mechanism={missing_mechanism!r} must be mcar or mar")
        incomplete = returns.mask(mask)

    # -- short rate --------------------------------------------------------
    short_rate = None
    short_truth: dict[str, float] = {}
    if include_short_rate:
        short_rate, short_truth = generate_short_rate_path(
            n_periods=n_periods, gamma=short_rate_gamma,
            periods_per_year=periods_per_year, seed=seed + 1, start=start,
        )

    # -- barrier -----------------------------------------------------------
    barrier_paths = None
    true_barrier = None
    barrier_level = None
    if include_barrier_paths:
        barrier_level = 115.0
        barrier_paths, true_barrier = generate_barrier_paths(
            n_paths=200, n_steps=25, barrier=barrier_level,
            periods_per_year=periods_per_year, seed=seed + 2,
        )

    # -- asset metadata: synthetic sector, class, region, currency ---------
    sectors = ["Technology", "Healthcare", "Financials", "Consumer", "Industrials", "Energy", "Utilities", "Materials"]
    asset_classes = ["Equity", "Fixed_Income", "Commodities", "Real_Estate"]
    regions = ["North_America", "Europe", "Asia_Pacific", "Emerging_Markets"]
    currencies = ["USD", "EUR", "GBP", "JPY"]

    meta_rows = []
    for i, aid in enumerate(asset_ids):
        meta_rows.append({
            "asset_id": aid,
            "name": f"Synthetic Asset {i:03d}",
            "sector": sectors[i % len(sectors)],
            "asset_class": asset_classes[i % len(asset_classes)],
            "region": regions[i % len(regions)],
            "currency": currencies[i % len(currencies)],
            "is_synthetic_demo": True,
            "provenance_tag": "SYNTHETIC_DEMO / NON-PRODUCTION / FICTIONAL ASSET METADATA",
        })
    asset_metadata_df = pd.DataFrame(meta_rows).set_index("asset_id")

    # -- prior weights & constraints ---------------------------------------
    from start.portfolio.contracts import GroupConstraintSpec, GroupCoveragePolicy

    noise_w = rng.uniform(0.9, 1.1, n_assets)
    prior_unnorm = raw_weights * noise_w
    prior_weights = pd.Series(prior_unnorm / prior_unnorm.sum(), index=asset_ids, name="prior_weights")

    tech_assets = tuple(aid for i, aid in enumerate(asset_ids) if sectors[i % len(sectors)] == "Technology")
    fin_assets = tuple(aid for i, aid in enumerate(asset_ids) if sectors[i % len(sectors)] == "Financials")
    group_spec = GroupConstraintSpec(
        group_name="sector",
        memberships={"Technology": tech_assets, "Financials": fin_assets},
        lower_bounds={"Technology": 0.0, "Financials": 0.0},
        upper_bounds={"Technology": 0.40, "Financials": 0.35},
        coverage_policy=GroupCoveragePolicy.OPTIONAL_UNMAPPED_ALLOWED,
        provenance="SYNTHETIC_DEMO",
    )

    portfolio_constraints = PortfolioConstraints(
        min_weight=0.0,
        max_weight=0.15,
        long_only=True,
        max_leverage=1.0,
        group_constraints=group_spec,
    )

    # -- Black-Litterman demo views (P, Q, Omega) --------------------------
    p_mat = np.zeros((2, n_assets), dtype=float)
    p_mat[0, 0] = 1.0
    p_mat[0, min(2, n_assets - 1)] = -1.0
    p_mat[1, min(1, n_assets - 1)] = 1.0
    q_vec = np.array([0.025, 0.040], dtype=float)
    omega_mat = np.diag([0.0004, 0.0009])
    bl_views_dict = {
        "P": p_mat,
        "Q": q_vec,
        "Omega": omega_mat,
        "labels": (
            "Synthetic Tech relative outperformance vs Financials (+2.5%)",
            "Synthetic Healthcare absolute expected return (+4.0%)",
        ),
        "provenance_tag": "SYNTHETIC_DEMO / NON-PRODUCTION / FICTIONAL ASSET METADATA",
    }

    # -- Scenario specifications with real shocks --------------------------
    from start.portfolio.contracts import (
        RepricingMethod,
        ScenarioShock,
        ScenarioSpec,
        ScenarioType,
        ShockSpace,
        ShockUnit,
    )
    from start.tests.attribution import AttributionState

    scen_asset_tail = ScenarioSpec(
        scenario_id="SCEN-ASSET-TAIL",
        scenario_name="Synthetic Equity Tail Shock (-8.5%)",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=tuple(
            ScenarioShock(
                risk_factor_id=asset_ids[k],
                shock_space=ShockSpace.ASSET_RETURN,
                shock_unit=ShockUnit.RETURN_DECIMAL,
                raw_value=-0.085,
                normalized_value=-0.085,
                normalization_rule="IDENTITY_RETURN_DECIMAL",
                provenance={"tag": "SYNTHETIC_DEMO / NON-PRODUCTION / FICTIONAL ASSET METADATA"},
            )
            for k in range(min(5, n_assets))
        ),
        repricing_method=RepricingMethod.LINEAR_RETURN,
        source_reference="SYNTHETIC_DEMO_TAIL_SIMULATION",
    )
    scen_factor_stress = ScenarioSpec(
        scenario_id="SCEN-FACTOR-STRESS",
        scenario_name="Synthetic Macro Factor Stress (+2.0 Sigma)",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=tuple(
            ScenarioShock(
                risk_factor_id=factor_ids[k],
                shock_space=ShockSpace.FACTOR_RETURN,
                shock_unit=ShockUnit.RETURN_DECIMAL,
                raw_value=0.035,
                normalized_value=0.035,
                normalization_rule="VOLATILITY_SCALED_FACTOR_DECIMAL",
                provenance={"tag": "SYNTHETIC_DEMO / NON-PRODUCTION / FICTIONAL ASSET METADATA"},
            )
            for k in range(min(3, n_factors))
        ),
        repricing_method=RepricingMethod.FACTOR_LINEAR,
        source_reference="SYNTHETIC_DEMO_MACRO_SHOCK",
    )
    scenarios_tuple = (scen_asset_tail, scen_factor_stress)

    # -- Attribution prior comparison state --------------------------------
    x0 = pd.Series(exposures.T @ prior_weights.to_numpy(dtype=float), index=factor_ids)
    F0 = pd.DataFrame(factor_cov * 0.95, index=factor_ids, columns=factor_ids)
    S0 = float((prior_weights.to_numpy(dtype=float)**2) @ (specific_sd**2))
    comparison_state = AttributionState(
        exposure=x0,
        factor_covariance=F0,
        specific_variance=S0,
        label="prior_synthetic_reporting_period",
    )

    extra_dict: dict[str, Any] = {
        "provenance_tag": "SYNTHETIC_DEMO / NON-PRODUCTION / FICTIONAL ASSET METADATA",
        "bl_views": bl_views_dict,
        "scenarios": scenarios_tuple,
        "scenario_spec": scen_asset_tail,
        "comparison_state": comparison_state,
        "market_weights": benchmark,
    }

    return MarketWorld(
        returns=returns, prices=prices, factor_returns=factor_returns,
        factor_exposures=exposures_frame, weights=weights, benchmark_weights=benchmark,
        pnl=pnl, hypothetical_pnl=hypothetical_pnl, var_series=var_series,
        var_confidence=var_confidence,
        true_factor_covariance=pd.DataFrame(factor_cov, index=factor_ids, columns=factor_ids),
        true_specific_variance=specific_variance,
        true_asset_covariance=true_asset_cov,
        true_portfolio_variance=true_portfolio_variance,
        var_misspecification_factor=float(var_misspecification_factor),
        incomplete_returns=incomplete, missing_mask=mask, missing_mechanism=mechanism,
        short_rate=short_rate, true_short_rate_params=short_truth,
        barrier_paths=barrier_paths, true_barrier_probability=true_barrier,
        barrier_level=barrier_level,
        asset_metadata=asset_metadata_df,
        prior_weights=prior_weights,
        constraints=portfolio_constraints,
        extra=extra_dict,
        periods_per_year=periods_per_year, seed=seed, modes=tuple(modes),
    )
