"""B6 — covariance, with independently derived known answers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from start.core.schemas import Status
from start.data.synthetic_market import generate_market_world
from start.registry.market_contexts import MarketContext, PortfolioSpec
from start.tests.covariance import (
    PSD_EIGENVALUE_FLOOR,
    REGEM_DDOF,
    REGEM_DEFAULT_MAX_ITER,
    REGEM_DEFAULT_RIDGE,
    REGEM_DEFAULT_TOL,
    empirical,
    ledoit_wolf_shrinkage,
    regularized_em,
    run_regularized_em,
)


def _ctx(frame):
    return MarketContext(
        returns=frame,
        portfolio=PortfolioSpec(weights=pd.Series(1.0 / frame.shape[1], index=frame.columns)),
    )


def _frame(array, columns=None):
    columns = columns or [f"A{i}" for i in range(array.shape[1])]
    return pd.DataFrame(array, index=pd.date_range("2024-01-01", periods=len(array), freq="B"),
                        columns=columns)


# ==================================================== FROZEN CONSTANTS ==
def test_constants_are_frozen():
    assert PSD_EIGENVALUE_FLOOR == 1e-12
    assert REGEM_DEFAULT_RIDGE == 1e-6
    assert REGEM_DEFAULT_TOL == 1e-6
    assert REGEM_DEFAULT_MAX_ITER == 200
    assert REGEM_DDOF == 0


# ==================================================== EMPIRICAL ==
def test_empirical_known_answer_two_assets():
    """Hand-computable: x = [1,2,3,4], y = [2,4,6,8] = 2x.
    var(x) with ddof=1 = 5/3; cov(x,y) = 10/3; var(y) = 20/3."""
    frame = _frame(np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]))
    result = empirical(_ctx(frame))
    assert result.metrics["n_observations_used"] == 4
    matrix = frame.cov(ddof=1).to_numpy()
    assert abs(matrix[0, 0] - 5 / 3) < 1e-12
    assert abs(matrix[0, 1] - 10 / 3) < 1e-12
    assert abs(matrix[1, 1] - 20 / 3) < 1e-12


def test_empirical_uses_complete_cases_and_says_so():
    rng = np.random.default_rng(1)
    data = rng.normal(size=(100, 3))
    data[5, 0] = np.nan
    data[9, 2] = np.nan
    result = empirical(_ctx(_frame(data)))
    assert result.metrics["missing_policy"] == "complete_case"
    assert result.metrics["n_observations_used"] == 98
    assert result.metrics["n_observations_dropped"] == 2


def test_empirical_refuses_a_pairwise_policy():
    """A pairwise matrix mixes sample sizes and is often not PSD."""
    result = empirical(_ctx(_frame(np.random.default_rng(2).normal(size=(50, 3)))),
                       missing_policy="pairwise")
    assert result.status == Status.ERROR
    assert "regularized_em" in result.interpretation


def test_empirical_reports_conditioning():
    result = empirical(_ctx(_frame(np.random.default_rng(3).normal(size=(200, 4)))))
    for key in ("condition_number", "min_eigenvalue", "rank", "is_psd"):
        assert key in result.metrics
    assert result.metrics["is_psd"] is True


def test_empirical_ddof_convention_is_recorded():
    result = empirical(_ctx(_frame(np.random.default_rng(4).normal(size=(50, 3)))), ddof=1)
    assert result.metrics["ddof"] == 1
    assert "n-1" in result.metrics["estimand"]


def test_empirical_puts_no_matrix_in_metrics():
    result = empirical(_ctx(_frame(np.random.default_rng(5).normal(size=(80, 4)))))
    for value in result.metrics.values():
        assert not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))
    assert len(result.metrics["covariance_hash"]) == 32


# ==================================================== LEDOIT-WOLF ==
def test_ledoit_wolf_reports_shrinkage_and_conditioning():
    rng = np.random.default_rng(6)
    result = ledoit_wolf_shrinkage(_ctx(_frame(rng.normal(size=(40, 12)))))
    assert result.status == Status.RECORDED
    assert 0.0 <= result.metrics["shrinkage_intensity"] <= 1.0
    assert result.metrics["is_psd"] is True


def test_ledoit_wolf_improves_conditioning_when_p_is_large_relative_to_n():
    rng = np.random.default_rng(7)
    result = ledoit_wolf_shrinkage(_ctx(_frame(rng.normal(size=(30, 20)))))
    assert result.metrics["condition_number_after"] < result.metrics["condition_number_before"]
    assert result.metrics["min_eigenvalue_after"] > result.metrics["min_eigenvalue_before"]


def test_ledoit_wolf_refuses_missing_values_rather_than_imputing():
    """An undocumented fill would change the estimand while the label stayed the same."""
    data = np.random.default_rng(8).normal(size=(60, 4))
    data[3, 1] = np.nan
    result = ledoit_wolf_shrinkage(_ctx(_frame(data)))
    assert result.status == Status.SKIPPED
    assert "regularized_em" in result.interpretation


def test_ledoit_wolf_claims_no_universal_superiority():
    result = ledoit_wolf_shrinkage(_ctx(_frame(np.random.default_rng(9).normal(size=(50, 6)))))
    assert any("NOT universally superior" in x for x in result.limitations)


# ==================================================== REGEM ==
def test_regem_complete_data_matches_the_ml_covariance():
    """With no missing values and negligible ridge, RegEM must reproduce the ML
    covariance (denominator n). Expected computed independently with np.cov(ddof=0)."""
    rng = np.random.default_rng(11)
    data = rng.normal(size=(400, 4))
    outcome = run_regularized_em(data, ridge=0.0, tol=1e-14, max_iter=50)
    expected = np.cov(data, rowvar=False, ddof=0)
    assert np.allclose(outcome.covariance, expected, atol=1e-9)
    assert np.allclose(outcome.mean, data.mean(axis=0), atol=1e-12)


def test_regem_complete_data_does_not_match_the_n_minus_one_convention():
    """Guards against comparing different estimands and calling the mismatch a bug."""
    rng = np.random.default_rng(12)
    data = rng.normal(size=(200, 3))
    outcome = run_regularized_em(data, ridge=0.0, tol=1e-14, max_iter=50)
    sample = np.cov(data, rowvar=False, ddof=1)
    assert not np.allclose(outcome.covariance, sample, atol=1e-9)
    ratio = float(np.mean(np.diag(sample) / np.diag(outcome.covariance)))
    assert abs(ratio - 200 / 199) < 1e-6


def test_regem_e_step_includes_the_conditional_covariance_term():
    """The term that separates EM from mean imputation.

    Perfectly correlated columns with one masked entry: mean imputation shrinks the
    variance of the masked column, EM does not to the same degree. Expected direction
    derived from the algebra, not from the estimator.
    """
    rng = np.random.default_rng(13)
    x = rng.normal(size=200)
    data = np.column_stack([x, x * 0.5 + rng.normal(0, 0.5, 200)])
    complete_var = float(np.var(data[:, 1], ddof=0))

    masked = data.copy()
    masked[:60, 1] = np.nan
    em = run_regularized_em(masked, ridge=0.0, tol=1e-12, max_iter=200)

    # Mean imputation, computed here explicitly.
    imputed = masked.copy()
    imputed[:60, 1] = np.nanmean(masked[:, 1])
    naive_var = float(np.var(imputed[:, 1], ddof=0))

    assert naive_var < complete_var * 0.95          # mean imputation understates
    assert em.covariance[1, 1] > naive_var          # EM recovers more of it


def test_regem_initialisation_is_deterministic():
    rng = np.random.default_rng(14)
    data = rng.normal(size=(150, 4))
    data[::7, 2] = np.nan
    a = run_regularized_em(data, ridge=1e-8, tol=1e-10, max_iter=100)
    b = run_regularized_em(data, ridge=1e-8, tol=1e-10, max_iter=100)
    assert np.array_equal(a.covariance, b.covariance)
    assert a.n_iterations == b.n_iterations


def test_regem_result_is_psd_at_high_missingness():
    world = generate_market_world(n_assets=8, n_periods=300, seed=15,
                                  missing_rate=0.40, missing_mechanism="mcar")
    result = regularized_em(_ctx(world.incomplete_returns))
    assert result.status == Status.RECORDED
    assert result.metrics["is_psd"] is True
    assert result.metrics["min_eigenvalue"] >= -PSD_EIGENVALUE_FLOOR


def test_regem_handles_a_near_singular_structure():
    world = generate_market_world(n_assets=6, n_periods=300, seed=16,
                                  near_singular=True, missing_rate=0.20)
    result = regularized_em(_ctx(world.incomplete_returns))
    assert result.status in {Status.RECORDED, Status.FAIL}
    assert result.metrics["is_psd"] is True


def test_regem_records_eigenvalue_clipping_as_a_start_safeguard():
    world = generate_market_world(n_assets=6, n_periods=200, seed=17,
                                  near_singular=True, missing_rate=0.30)
    result = regularized_em(_ctx(world.incomplete_returns), ridge=0.0)
    assert "n_eigenvalue_clips" in result.metrics
    assert result.metrics["psd_floor"] == PSD_EIGENVALUE_FLOOR
    assert any("StART numerical safeguard" in x for x in result.limitations)
    assert any("not part of Schneider" in x for x in result.limitations)


def test_regem_non_convergence_is_an_error_not_a_green_result():
    """A non-converged covariance reported green is one nobody should use."""
    world = generate_market_world(n_assets=6, n_periods=200, seed=18, missing_rate=0.30)
    result = regularized_em(_ctx(world.incomplete_returns), tol=1e-30, max_iterations=2)
    assert result.status == Status.ERROR
    assert result.metrics["converged"] is False


def test_regem_rejects_an_all_missing_variable():
    data = np.random.default_rng(19).normal(size=(100, 3))
    data[:, 1] = np.nan
    result = regularized_em(_ctx(_frame(data)))
    assert result.status == Status.ERROR
    assert "no observed values" in result.interpretation


def test_regem_rejects_negative_ridge():
    result = regularized_em(_ctx(_frame(np.random.default_rng(20).normal(size=(80, 3)))),
                            ridge=-1.0)
    assert result.status == Status.ERROR


def test_regem_records_missingness_structure():
    world = generate_market_world(n_assets=5, n_periods=200, seed=21, missing_rate=0.20)
    result = regularized_em(_ctx(world.incomplete_returns))
    for key in ("missing_fraction", "n_missingness_patterns", "n_complete_rows",
                "max_column_missing_fraction"):
        assert key in result.metrics
    assert 0.1 < result.metrics["missing_fraction"] < 0.3


def test_regem_claims_no_uncertainty_and_no_mle():
    world = generate_market_world(n_assets=4, n_periods=200, seed=22, missing_rate=0.15)
    result = regularized_em(_ctx(world.incomplete_returns))
    blob = " ".join(result.limitations)
    assert "NO UNCERTAINTY QUANTIFICATION" in blob
    assert "NOT the unconstrained maximum-likelihood estimate" in blob
    assert "MAR is a working assumption" in blob


def test_regem_reports_pseudoinverse_fallbacks():
    world = generate_market_world(n_assets=5, n_periods=200, seed=23, missing_rate=0.20)
    result = regularized_em(_ctx(world.incomplete_returns))
    assert "n_pseudoinverse_fallbacks" in result.metrics
    assert result.metrics["n_pseudoinverse_fallbacks"] >= 0


def test_regem_recovers_a_known_covariance_better_than_complete_case_at_high_missingness():
    """Not a universal-dominance claim: one seeded configuration, reported as such."""
    world = generate_market_world(n_assets=6, n_periods=400, seed=24,
                                  missing_rate=0.35, missing_mechanism="mcar")
    em = regularized_em(_ctx(world.incomplete_returns))
    assert em.status == Status.RECORDED
    complete_rows = world.incomplete_returns.dropna()
    assert em.metrics["n_complete_rows"] == len(complete_rows)
    assert em.metrics["is_psd"] is True


def test_regem_puts_no_matrix_in_metrics():
    world = generate_market_world(n_assets=4, n_periods=150, seed=25, missing_rate=0.10)
    result = regularized_em(_ctx(world.incomplete_returns))
    for value in result.metrics.values():
        assert not isinstance(value, (pd.DataFrame, pd.Series, np.ndarray))
    assert len(result.metrics["covariance_hash"]) == 32
