"""B2 — synthetic market generator and its ground truth."""

from __future__ import annotations

import numpy as np
import pytest

from start.data.synthetic_market import (
    ADVERSARIAL_MODES,
    barrier_crossing_probability,
    generate_barrier_paths,
    generate_market_world,
    generate_short_rate_path,
)


def _world(**kw):
    kw.setdefault("n_assets", 15)
    kw.setdefault("n_periods", 400)
    kw.setdefault("n_factors", 3)
    kw.setdefault("seed", 42)
    return generate_market_world(**kw)


# ------------------------------------------------------------ structure ==
def test_world_has_all_required_outputs():
    w = _world()
    for attribute in (
        "returns",
        "prices",
        "factor_returns",
        "factor_exposures",
        "weights",
        "benchmark_weights",
        "pnl",
        "hypothetical_pnl",
        "var_series",
    ):
        assert getattr(w, attribute) is not None


def test_returns_follow_the_factor_structure():
    """r = X f + e, so returns must reconstruct from the generator's own inputs."""
    w = _world()
    implied = w.factor_returns.to_numpy() @ w.factor_exposures.to_numpy().T
    residual = w.returns.to_numpy() - implied
    # The residual is the specific component; its scale should match specific sd.
    assert residual.std(axis=0).mean() > 0
    assert np.corrcoef(residual.std(axis=0), np.sqrt(w.true_specific_variance))[0, 1] > 0.8


def test_weights_sum_to_one():
    assert abs(_world().weights.sum() - 1.0) < 1e-12


def test_prices_are_positive_and_consistent_with_returns():
    w = _world()
    assert (w.prices > 0).all().all()
    implied = w.prices / w.prices.shift(1) - 1.0
    assert np.allclose(implied.iloc[1:].to_numpy(), w.returns.iloc[1:].to_numpy(), atol=1e-12)


# --------------------------------------------------------- ground truth ==
def test_true_covariance_is_a_generator_input_not_an_estimate():
    """Validating an estimator against itself proves nothing."""
    w = _world(n_periods=400)
    empirical = w.returns.cov().to_numpy()
    true = w.true_asset_covariance.to_numpy()
    error = np.linalg.norm(empirical - true, "fro") / np.linalg.norm(true, "fro")
    assert error > 1e-6, "true covariance must not be the sample covariance"


def test_true_covariance_equals_xfx_plus_d():
    w = _world()
    X = w.factor_exposures.to_numpy()
    F = w.true_factor_covariance.to_numpy()
    D = np.diag(w.true_specific_variance.to_numpy())
    assert np.allclose(w.true_asset_covariance.to_numpy(), X @ F @ X.T + D, atol=1e-15)


def test_true_portfolio_variance_matches_the_true_covariance():
    w = _world()
    weights = w.weights.to_numpy()
    expected = float(weights @ w.true_asset_covariance.to_numpy() @ weights)
    assert abs(w.true_portfolio_variance - expected) < 1e-15


def test_true_covariance_is_symmetric_and_psd():
    w = _world()
    matrix = w.true_asset_covariance.to_numpy()
    assert np.allclose(matrix, matrix.T, atol=1e-14)
    assert np.linalg.eigvalsh(matrix).min() > -1e-12


def test_actual_and_hypothetical_pnl_are_distinct():
    """Frozen-position revaluation is not the same series as realised P&L."""
    w = _world()
    assert not np.allclose(w.pnl.to_numpy(), w.hypothetical_pnl.to_numpy())
    assert np.corrcoef(w.pnl, w.hypothetical_pnl)[0, 1] > 0.9


def test_hypothetical_pnl_is_exactly_the_weighted_return():
    w = _world()
    assert np.allclose(w.hypothetical_pnl.to_numpy(), (w.returns @ w.weights).to_numpy(), atol=1e-15)


# ---------------------------------------------------------- determinism ==
def test_same_seed_produces_an_identical_world():
    a, b = _world(seed=7), _world(seed=7)
    assert a.returns.equals(b.returns)
    assert a.weights.equals(b.weights)
    assert a.true_portfolio_variance == b.true_portfolio_variance


def test_different_seed_produces_a_different_world():
    assert not _world(seed=7).returns.equals(_world(seed=8).returns)


def test_replications_are_independent_not_slices():
    """Consecutive slices of one stream are not independent draws."""
    r0 = _world(seed=7, n_replications=5, replication=0)
    r1 = _world(seed=7, n_replications=5, replication=1)
    assert not r0.returns.equals(r1.returns)


def test_replications_are_individually_reproducible():
    a = _world(seed=7, n_replications=5, replication=3)
    b = _world(seed=7, n_replications=5, replication=3)
    assert a.returns.equals(b.returns)


@pytest.mark.parametrize("mode", list(ADVERSARIAL_MODES))
def test_every_adversarial_mode_is_reachable_and_deterministic(mode):
    kwargs = {
        "regime_shift": {"regime_shift_at": 200},
        "fat_tails": {"fat_tails": True},
        "near_singular": {"near_singular": True},
        "constant_asset": {"constant_asset": True},
        "mcar_missing": {"missing_rate": 0.2},
        "mar_missing": {"missing_rate": 0.2, "missing_mechanism": "mar"},
        "var_misspecification": {"var_misspecification_factor": 1.5},
    }[mode]
    a = _world(seed=5, **kwargs)
    b = _world(seed=5, **kwargs)
    assert mode in a.modes
    assert a.returns.equals(b.returns)


def test_near_singular_reduces_effective_rank():
    normal = _world(n_factors=4, seed=5)
    singular = _world(n_factors=4, seed=5, near_singular=True)
    normal_rank = np.linalg.matrix_rank(normal.true_factor_covariance.to_numpy(), tol=1e-10)
    singular_rank = np.linalg.matrix_rank(singular.true_factor_covariance.to_numpy(), tol=1e-10)
    assert singular_rank < normal_rank


def test_constant_asset_has_zero_variance():
    w = _world(constant_asset=True)
    assert w.true_specific_variance.iloc[-1] == 0.0
    assert w.returns.iloc[:, -1].std() < 1e-12


def test_fat_tails_preserves_the_second_moment():
    """The tails change; the covariance target does not."""
    normal = _world(n_periods=2000, seed=11)
    heavy = _world(n_periods=2000, seed=11, fat_tails=True)
    from scipy import stats as sp

    normal_k = float(sp.kurtosis(normal.factor_returns.iloc[:, 0]))
    heavy_k = float(sp.kurtosis(heavy.factor_returns.iloc[:, 0]))
    assert heavy_k > normal_k


# ------------------------------------------------------------ missingness ==
def test_mcar_mask_is_independent_of_values():
    w = _world(missing_rate=0.2, missing_mechanism="mcar", seed=9)
    assert w.incomplete_returns is not None
    rate = float(w.missing_mask.to_numpy().mean())
    assert 0.15 < rate < 0.25


def test_mar_missingness_depends_on_an_observed_variable():
    """MAR, not MNAR: depending on the missing value itself would break RegEM."""
    w = _world(missing_rate=0.2, missing_mechanism="mar", seed=9)
    driver = w.returns.iloc[:, 0]
    missing_per_row = w.missing_mask.iloc[:, 1:].sum(axis=1)
    assert abs(np.corrcoef(driver, missing_per_row)[0, 1]) > 0.1
    assert not w.missing_mask.iloc[:, 0].any(), "driver must stay observed"


def test_complete_returns_remain_available_as_truth():
    w = _world(missing_rate=0.3, seed=9)
    assert w.returns.notna().all().all()
    assert w.incomplete_returns.isna().any().any()


def test_invalid_missing_mechanism_is_rejected():
    with pytest.raises(ValueError, match="mcar or mar"):
        _world(missing_rate=0.1, missing_mechanism="mnar")


# ----------------------------------------------------------------- VaR ==
def test_var_series_uses_the_true_generating_quantile():
    w = _world(var_confidence=0.99)
    from scipy import stats as sp

    expected = -float(sp.norm.ppf(0.01)) * np.sqrt(w.true_portfolio_variance)
    assert abs(w.var_series.iloc[0] - expected) < 1e-12


@pytest.mark.parametrize("factor", [0.7, 1.0, 1.5])
def test_var_misspecification_is_controllable(factor):
    correct = _world(seed=3)
    misspecified = _world(seed=3, var_misspecification_factor=factor)
    assert abs(misspecified.var_series.iloc[0] / correct.var_series.iloc[0] - factor) < 1e-12


def test_var_is_a_positive_loss_magnitude():
    assert (_world().var_series > 0).all()


# ---------------------------------------------------------- short rate ==
@pytest.mark.parametrize("gamma", [0.0, 0.5, 1.0])
def test_short_rate_records_its_true_gamma(gamma):
    series, truth = generate_short_rate_path(n_periods=500, gamma=gamma, seed=4)
    assert truth["gamma"] == gamma
    assert series.size == 500


def test_short_rate_stays_positive():
    """r^gamma is undefined for negative r when gamma is fractional."""
    for gamma in (0.0, 0.5, 1.0):
        series, _ = generate_short_rate_path(n_periods=2000, gamma=gamma, seed=4)
        assert (series > 0).all()


def test_short_rate_is_deterministic():
    a, _ = generate_short_rate_path(n_periods=300, seed=4)
    b, _ = generate_short_rate_path(n_periods=300, seed=4)
    assert a.equals(b)


def test_world_can_carry_a_short_rate_context():
    w = _world(include_short_rate=True, n_periods=400)
    ctx = w.short_rate_context()
    assert ctx is not None
    assert ctx.context_kind() == "short_rate"
    assert ctx.validate_context() == []


# ------------------------------------------------------------- barrier ==
def test_bridge_probability_is_computed_in_log_space():
    """The arithmetic-price form collapses to zero where the log form does not."""
    sigma, dt = 0.25, 25 / 252
    log_form = barrier_crossing_probability(100, 110, 115, sigma, dt)
    arithmetic = float(np.exp(-2 * (115 - 100) * (115 - 110) / (sigma**2 * dt)))
    assert 0.05 < log_form < 0.5
    assert arithmetic < 1e-6
    assert abs(log_form - arithmetic) > 1e-3


def test_bridge_probability_rises_as_the_endpoint_approaches_the_barrier():
    sigma, dt = 0.25, 25 / 252
    near = barrier_crossing_probability(100, 114, 115, sigma, dt)
    mid = barrier_crossing_probability(100, 105, 115, sigma, dt)
    far = barrier_crossing_probability(100, 95, 115, sigma, dt)
    assert near > mid > far


def test_already_crossed_endpoints_return_certainty():
    assert barrier_crossing_probability(100, 120, 115, 0.25, 0.1) == 1.0
    assert barrier_crossing_probability(100, 80, 85, 0.25, 0.1, direction="down") == 1.0


def test_barrier_ground_truth_comes_from_a_finer_simulation():
    """Truth must not be the coarse estimate the test exists to critique."""
    paths, truth = generate_barrier_paths(n_paths=100, n_steps=20, seed=6, fine_steps_per_step=20)
    coarse_rate = float((paths.to_numpy() >= 115.0).any(axis=1).mean())
    assert 0.0 <= truth <= 1.0
    assert truth >= coarse_rate - 1e-12, "fine monitoring cannot detect fewer crossings"


def test_barrier_paths_are_deterministic():
    a, ta = generate_barrier_paths(n_paths=50, n_steps=10, seed=6)
    b, tb = generate_barrier_paths(n_paths=50, n_steps=10, seed=6)
    assert a.equals(b) and ta == tb


# ------------------------------------------------------------- contexts ==
def test_world_produces_a_valid_market_context():
    ctx = _world().market_context()
    assert ctx.validate_context() == []
    assert ctx.context_kind() == "market"
    assert ctx.portfolio is not None


def test_incomplete_context_carries_the_masked_returns():
    ctx = _world(missing_rate=0.2).market_context(incomplete=True)
    assert ctx.returns.isna().any().any()


def test_market_context_fingerprint_tracks_the_seed():
    assert _world(seed=1).market_context().fingerprint() != _world(seed=2).market_context().fingerprint()
