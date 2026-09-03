"""Focused unit tests for Gate 5 Tail Risk and Expected Shortfall estimation engines.

Verifies:
1. Empirical VaR quantile convention on known small ordered loss vector.
2. Exact finite-sample Rockafellar-Uryasev Expected Shortfall with fractional boundary weighting.
3. Tie handling without distorting effective tail probability mass.
4. Thin tail support (q < 1.0) handling.
5. Parametric Normal VaR and ES known-answer benchmarks.
6. Parametric component VaR and component ES Euler reconciliation.
7. Historical ES component risk decomposition reconciliation.
8. Multi-model comparison across historical and parametric estimators.
9. Financial horizon contract validation.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from start.portfolio.contracts import (
    MetricHorizon,
    TailSignConvention,
)
from start.portfolio.tail_risk import (
    compare_tail_risk_models,
    compute_historical_var_es,
    compute_parametric_normal_var_es,
    compute_tail_risk_contributions,
)


def test_empirical_var_known_quantile_convention() -> None:
    """Historical VaR must adhere to explicit linear quantile interpolation on small ordered losses."""
    # 10 losses: 1.0, 2.0, 3.0, ..., 10.0
    losses = np.arange(1.0, 11.0)
    # At alpha = 0.90, linear interpolation percentile = 90th percentile = 9.1
    est_90 = compute_historical_var_es(losses, confidence=0.90, quantile_method="linear")
    assert est_90.sign_convention == TailSignConvention.POSITIVE_LOSS_MAGNITUDE
    assert math.isclose(est_90.var, 9.1, rel_tol=1e-5)
    assert est_90.n_observations == 10

    # At alpha = 0.95, percentile = 9.55
    est_95 = compute_historical_var_es(losses, confidence=0.95, quantile_method="linear")
    assert math.isclose(est_95.var, 9.55, rel_tol=1e-5)


def test_empirical_es_exact_finite_sample_tail_mass() -> None:
    """Empirical ES must compute exact weighted tail average matching Rockafellar-Uryasev tail mass."""
    # 100 observations: 1, 2, ..., 100
    losses = np.arange(1.0, 101.0)
    # At alpha = 0.95, q = 100 * 0.05 = 5.0 observations (exact integer k=5, gamma=0.0)
    # Top 5 losses: 100, 99, 98, 97, 96 -> mean = 98.0
    est_95 = compute_historical_var_es(losses, confidence=0.95, quantile_method="linear")
    assert math.isclose(est_95.es, 98.0, rel_tol=1e-6)
    assert math.isclose(est_95.parameters["q_tail_mass"], 5.0, rel_tol=1e-6)
    assert math.isclose(est_95.boundary_weight, 0.0, abs_tol=1e-6)

    # Fractional tail mass: N = 10 observations, alpha = 0.85 -> q = 10 * 0.15 = 1.5 observations
    # Top losses: 10 (full weight 1.0), 9 (boundary weight 0.5)
    # Tail sum = 10 * 1.0 + 9 * 0.5 = 14.5
    # ES = 14.5 / 1.5 = 9.666667
    losses_10 = np.arange(1.0, 11.0)
    est_85 = compute_historical_var_es(losses_10, confidence=0.85, quantile_method="linear")
    assert math.isclose(est_85.parameters["q_tail_mass"], 1.5, rel_tol=1e-6)
    assert math.isclose(est_85.boundary_weight, 0.5, rel_tol=1e-6)
    assert math.isclose(est_85.es, 14.5 / 1.5, rel_tol=1e-6)


def test_empirical_es_boundary_tie_handling() -> None:
    """Tied observations at the VaR boundary must not artificially inflate effective tail probability mass."""
    # 10 observations with ties at value 8.0: [1, 2, 3, 4, 5, 6, 7, 8, 8, 10]
    losses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 10.0])
    # At alpha = 0.80, q = 10 * 0.20 = 2.0 observations
    # Sorted desc: 10.0, 8.0, 8.0, 7.0, ...
    # Top 2 observations: 10.0 (full weight 1.0), 8.0 (full weight 1.0)
    # Tail sum = 18.0, ES = 9.0
    est_80 = compute_historical_var_es(losses, confidence=0.80, quantile_method="linear")
    assert math.isclose(est_80.es, 9.0, rel_tol=1e-6)
    assert math.isclose(est_80.parameters["q_tail_mass"], 2.0, rel_tol=1e-6)


def test_empirical_es_thin_tail_support() -> None:
    """When q < 1.0 (sample size too small for target confidence), ES evaluates maximum observed loss."""
    # 20 observations, alpha = 0.99 -> q = 20 * 0.01 = 0.2 < 1.0
    losses = np.arange(1.0, 21.0)
    est = compute_historical_var_es(losses, confidence=0.99, quantile_method="linear")
    assert est.parameters["q_tail_mass"] < 1.0
    assert math.isclose(est.es, 20.0, rel_tol=1e-6)
    assert any("Thin tail support" in lim for lim in est.limitations)


def test_parametric_normal_var_es_analytical_benchmarks() -> None:
    """Parametric Normal VaR and ES must match analytical Gaussian unit-normal identities."""
    # For standard normal loss L ~ N(0, 1)
    # At alpha = 0.95:
    #   z_0.95 = 1.6448536269514722
    #   phi(z_0.95) = 0.10313564037803028
    #   ES_0.95 = 0.10313564 / 0.05 = 2.0627128
    # At alpha = 0.99:
    #   z_0.99 = 2.3263478740408408
    #   phi(z_0.99) = 0.0266521424
    #   ES_0.99 = 0.02665214 / 0.01 = 2.6652142
    sample_size = 10000
    rng = np.random.RandomState(42)
    sample_normal = rng.normal(loc=0.0, scale=1.0, size=sample_size)

    est_95 = compute_parametric_normal_var_es(sample_normal, confidence=0.95, is_returns=False)
    # Verify analytical multiplier formulas
    z_95 = float(stats.norm.ppf(0.95))
    phi_95 = float(stats.norm.pdf(z_95))
    mult_95 = phi_95 / 0.05
    assert math.isclose(est_95.parameters["z_alpha"], z_95, rel_tol=1e-6)
    assert math.isclose(est_95.parameters["es_multiplier"], mult_95, rel_tol=1e-6)
    assert math.isclose(mult_95, 2.0627128, rel_tol=1e-4)

    est_99 = compute_parametric_normal_var_es(sample_normal, confidence=0.99, is_returns=False)
    z_99 = float(stats.norm.ppf(0.99))
    phi_99 = float(stats.norm.pdf(z_99))
    mult_99 = phi_99 / 0.01
    assert math.isclose(est_99.parameters["z_alpha"], z_99, rel_tol=1e-6)
    assert math.isclose(est_99.parameters["es_multiplier"], mult_99, rel_tol=1e-6)
    assert math.isclose(mult_99, 2.6652142, rel_tol=1e-4)


def test_parametric_component_var_and_es_euler_reconciliation() -> None:
    """Parametric component VaR and component ES must sum exactly to portfolio VaR and ES."""
    assets = ["A", "B", "C"]
    weights = {"A": 0.50, "B": 0.30, "C": 0.20}
    cov = np.array(
        [
            [0.0400, 0.0150, 0.0080],
            [0.0150, 0.0900, 0.0200],
            [0.0080, 0.0200, 0.1600],
        ]
    )
    rng = np.random.RandomState(42)
    rets = rng.multivariate_normal(np.array([0.0005, 0.0008, 0.0010]), cov, size=500)
    rets_df = pd.DataFrame(rets, columns=assets)

    contrib = compute_tail_risk_contributions(
        returns_or_losses=rets_df,
        weights=weights,
        confidence=0.99,
        method="parametric_normal",
        is_returns=True,
    )

    # Component VaR reconciliation
    sum_comp_var = sum(contrib.component_var.values())
    assert math.isclose(sum_comp_var, contrib.portfolio_var, rel_tol=1e-6)
    assert math.isclose(contrib.var_reconciliation_error, 0.0, abs_tol=1e-12)

    # Component ES reconciliation
    sum_comp_es = sum(contrib.component_es.values())
    assert math.isclose(sum_comp_es, contrib.portfolio_es, rel_tol=1e-6)
    assert math.isclose(contrib.es_reconciliation_error, 0.0, abs_tol=1e-12)

    # Percentage contributions sum to 100%
    assert math.isclose(sum(contrib.percentage_var_contributions.values()), 1.0, rel_tol=1e-5)
    assert math.isclose(sum(contrib.percentage_es_contributions.values()), 1.0, rel_tol=1e-5)


def test_historical_component_es_reconciliation() -> None:
    """Historical component ES using portfolio tail scenario weights must sum exactly to portfolio ES."""
    assets = ["A", "B"]
    weights = {"A": 0.60, "B": 0.40}
    rng = np.random.RandomState(42)
    rets = rng.normal(loc=0.0, scale=0.02, size=(300, 2))
    rets_df = pd.DataFrame(rets, columns=assets)

    contrib = compute_tail_risk_contributions(
        returns_or_losses=rets_df,
        weights=weights,
        confidence=0.95,
        method="historical_es",
        is_returns=True,
    )

    sum_comp_es = sum(contrib.component_es.values())
    assert math.isclose(sum_comp_es, contrib.portfolio_es, rel_tol=1e-6)
    assert math.isclose(contrib.es_reconciliation_error, 0.0, abs_tol=1e-12)
    # Historical VaR contribution is marked deferred as required
    assert math.isnan(contrib.component_var["A"])


def test_multi_model_tail_comparison() -> None:
    """Multi-model comparison computes Historical and Parametric Normal estimates and ES/VaR ratios."""
    rng = np.random.RandomState(42)
    rets = rng.standard_t(df=5, size=400) * 0.01

    comp = compare_tail_risk_models(rets, confidence=0.99, is_returns=True)
    assert comp.confidence == 0.99
    assert "historical" in comp.estimates
    assert "parametric_normal" in comp.estimates
    assert comp.es_values["historical"] > comp.var_values["historical"]
    assert comp.es_to_var_ratios["historical"] > 1.0


def test_horizon_contract_validation() -> None:
    """Tail risk functions must reject double annualization or frequency contradictions."""
    losses = np.arange(1.0, 101.0)
    # Double annualization error
    with pytest.raises(ValueError, match="Double annualization error"):
        compute_historical_var_es(
            losses,
            confidence=0.95,
            horizon=MetricHorizon.ANNUAL,
            periods_per_year=252.0,
        )

    # Frequency contradiction
    with pytest.raises(ValueError, match="Frequency contradiction"):
        compute_parametric_normal_var_es(
            losses,
            confidence=0.95,
            periods_per_year=252.0,
            frequency="monthly",
        )
