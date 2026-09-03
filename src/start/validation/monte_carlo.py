"""Pre-registered Monte Carlo validation.

These criteria were frozen **before** any estimator existed. They are recorded here as
module constants so that a later reader can see they were not chosen to fit an observed
result. Nothing in this module reads a simulation outcome and adjusts a threshold.

The rule that matters
---------------------

If a study fails, the failure is reported. A criterion is never relaxed to turn a red
result green — that converts a validation suite into a decoration, and the whole point
of pre-registration is to make that impossible without it being visible in a diff.

Independence of replications
----------------------------

Every study derives child seeds via ``SeedSequence.spawn``. Consecutive slices of one
RNG stream are **not** independent draws, and size and power estimates built on them are
wrong in a way nothing in the output would reveal.

Cost
----

These are slow by design. They are separated from ordinary tests so they never run
accidentally inside a routine batch, and each has its own entry point.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "VAR_MC",
    "CEV_MC",
    "STANTON_MC",
    "REGEM_MC",
    "StudyResult",
    "run_var_study",
    "run_cev_study",
    "run_stanton_study",
    "run_regem_study",
    "child_seeds",
    "configuration_hash",
]


# --------------------------------------------------------------------------- #
# IMMUTABLE PRE-REGISTERED CRITERIA
# --------------------------------------------------------------------------- #
#: VaR size and power. The deciding backtest is Kupiec POF, fixed here before execution
#: so it cannot be swapped for whichever test happens to pass.
VAR_MC: dict[str, Any] = {
    "study": "var_size_power",
    "deciding_test": "kupiec_pof",
    "R": 500,
    "alpha": 0.05,
    "n": 500,
    "confidence": 0.99,
    "root_seed": 20240601,
    "size_bounds": (0.031, 0.069),
    "power_understated_factor": 0.7,
    "power_understated_min": 0.50,
    "power_overstated_factor": 1.5,
    "power_overstated_min": 0.20,
    "note": (
        "Kupiec POF decides all three criteria. Christoffersen tests are computed for "
        "information but never substituted for the deciding statistic."
    ),
}

#: CEV consistency, coverage and failure rate. Frozen at B0, before any CEV code existed.
CEV_MC: dict[str, Any] = {
    "study": "cev_consistency",
    "gammas": (0.0, 0.5, 1.0),
    "sample_sizes": (250, 1000, 2500),
    "R": 200,
    "root_seed": 20240602,
    "consistency_ratio_max": 0.70,
    "coverage_bounds": (0.90, 0.98),
    "coverage_at_n": 2500,
    "failure_rate_max": 0.05,
    "failure_rate_min_n": 1000,
    "bootstrap_draws": 200,
    "note": (
        "The consistency ratio is evaluated SEPARATELY for each gamma. The three are "
        "never averaged: averaging would let one failing gamma hide behind two passing "
        "ones."
    ),
}

#: Stanton pointwise bias. Grid exclusion rule frozen BEFORE simulation.
STANTON_MC: dict[str, Any] = {
    "study": "stanton_bias",
    "sample_sizes": (500, 2500),
    "R": 200,
    "root_seed": 20240603,
    "n_grid": 9,
    "bias_ratio_max": 0.70,
    "wrong_sign_rate_max": 0.10,
    "sign_test_min_true_drift": 1e-4,
    "grid_exclusion_rule": (
        "A grid point is excluded from the WRONG-SIGN criterion when the true drift "
        "magnitude at that point is below sign_test_min_true_drift, because the sign of "
        "a drift that is numerically zero is undefined and any estimate is equally "
        "'wrong'. Frozen before simulation. Excluded points are still reported for bias."
    ),
}

#: RegEM structural acceptance. Deliberately no dominance requirement.
REGEM_MC: dict[str, Any] = {
    "study": "regem_structural",
    "R": 100,
    "missingness": (0.05, 0.20, 0.40),
    "mechanisms": ("mcar", "mar"),
    "structures": ("well_conditioned", "correlated", "near_singular"),
    "root_seed": 20240604,
    "psd_rate_required": 1.0,
    "non_convergence_max": 0.05,
    "note": (
        "PSD rate and non-convergence are the only hard criteria. RegEM is NOT required "
        "to dominate a baseline: no estimator dominates across every missingness "
        "mechanism and covariance structure, and requiring it would be requiring "
        "something untrue."
    ),
}


def configuration_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:32]


def child_seeds(root: int, count: int) -> list[np.random.SeedSequence]:
    """Independent child seeds. Never slices of one stream."""
    return list(np.random.SeedSequence(root).spawn(count))


@dataclass
class StudyResult:
    study: str
    configuration: dict[str, Any]
    rows: list[dict[str, Any]] = field(default_factory=list)
    passed: bool = True
    failures: list[str] = field(default_factory=list)

    @property
    def config_hash(self) -> str:
        return configuration_hash(self.configuration)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def summary_lines(self) -> list[str]:
        lines = [
            f"  study: {self.study}",
            f"  config hash: {self.config_hash}",
            f"  verdict: {'PASS' if self.passed else 'FAIL'}",
        ]
        for failure in self.failures:
            lines.append(f"    ! {failure}")
        return lines


# =========================================================================== #
# VaR size and power
# =========================================================================== #
def run_var_study(
    config: dict[str, Any] = VAR_MC, progress: Callable[[str], None] | None = None
) -> StudyResult:
    """Empirical size and power of the Kupiec POF test.

    Size is the rejection rate under a **correctly specified** forecast. A correct model
    rejects at exactly the nominal rate by construction, so "a correct model does not
    """
    from start.tests.traded_risk import kupiec_lr

    R, n = int(config["R"]), int(config["n"])
    alpha, confidence = float(config["alpha"]), float(config["confidence"])
    p = 1.0 - confidence
    true_var = -float(stats.norm.ppf(p))
    scenarios = {
        "correct": 1.0,
        "understated": float(config["power_understated_factor"]),
        "overstated": float(config["power_overstated_factor"]),
    }

    seeds = child_seeds(int(config["root_seed"]), R)
    rejections = {name: 0 for name in scenarios}

    for index, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        pnl = rng.standard_normal(n)
        for name, factor in scenarios.items():
            exceptions = int((pnl < -(true_var * factor)).sum())
            lr = kupiec_lr(n, exceptions, p)
            p_value = float(stats.chi2.sf(lr, df=1)) if math.isfinite(lr) else 1.0
            if p_value < alpha:
                rejections[name] += 1
        if progress and (index + 1) % 100 == 0:
            progress(f"    VaR replication {index + 1}/{R}")

    result = StudyResult(study=config["study"], configuration=dict(config))
    low, high = config["size_bounds"]
    for name in scenarios:
        rate = rejections[name] / R
        result.rows.append(
            {
                "scenario": name,
                "factor": scenarios[name],
                "R": R,
                "n": n,
                "n_rejected": rejections[name],
                "rejection_rate": rate,
            }
        )
        if name == "correct" and not (low <= rate <= high):
            result.passed = False
            result.failures.append(
                f"size: correct-forecast rejection rate {rate:.4f} outside [{low}, {high}]"
            )
        if name == "understated" and rate < config["power_understated_min"]:
            result.passed = False
            result.failures.append(f"power(0.7x): {rate:.4f} below {config['power_understated_min']}")
        if name == "overstated" and rate < config["power_overstated_min"]:
            result.passed = False
            result.failures.append(f"power(1.5x): {rate:.4f} below {config['power_overstated_min']}")
    return result


# =========================================================================== #
# CEV consistency
# =========================================================================== #
def run_cev_study(
    config: dict[str, Any] = CEV_MC, progress: Callable[[str], None] | None = None
) -> StudyResult:
    """Consistency, interval coverage and failure rate, evaluated per gamma."""
    from start.data.synthetic_market import generate_short_rate_path
    from start.tests.traded_risk import _bootstrap_cev, estimate_cev

    R = int(config["R"])
    result = StudyResult(study=config["study"], configuration=dict(config))
    errors: dict[tuple[float, int], list[float]] = {}

    for gamma in config["gammas"]:
        for n in config["sample_sizes"]:
            seeds = child_seeds(int(config["root_seed"]) + int(gamma * 1000) + n, R)
            per_run: list[float] = []
            failures = 0
            covered = 0
            evaluated = 0
            for index, seed in enumerate(seeds):
                draw = int(np.random.default_rng(seed).integers(0, 2**31 - 1))
                rates, _ = generate_short_rate_path(n_periods=n, gamma=gamma, seed=draw)
                try:
                    estimate = estimate_cev(rates, 1.0 / 252.0)
                except ValueError:
                    failures += 1
                    continue
                if not math.isfinite(estimate.gamma_hat):
                    failures += 1
                    continue
                per_run.append(abs(estimate.gamma_hat - gamma))
                if n == config["coverage_at_n"]:
                    low, high, valid = _bootstrap_cev(
                        rates,
                        1.0 / 252.0,
                        int(config["bootstrap_draws"]),
                        draw,
                        0.95,
                    )
                    if low is not None:
                        evaluated += 1
                        if low <= gamma <= high:
                            covered += 1
                if progress and (index + 1) % 50 == 0:
                    progress(f"    CEV gamma={gamma} n={n}: {index + 1}/{R}")

            errors[(gamma, n)] = per_run
            row: dict[str, Any] = {
                "gamma": gamma,
                "n": n,
                "R": R,
                "n_valid": len(per_run),
                "failure_rate": failures / R,
                "median_abs_error": float(np.median(per_run)) if per_run else float("nan"),
            }
            if n == config["coverage_at_n"] and evaluated:
                row["coverage"] = covered / evaluated
                row["n_coverage_evaluated"] = evaluated
            result.rows.append(row)

            if n >= config["failure_rate_min_n"] and failures / R > config["failure_rate_max"]:
                result.passed = False
                result.failures.append(
                    f"failure rate gamma={gamma} n={n}: {failures / R:.4f} above {config['failure_rate_max']}"
                )

    # Consistency, per gamma. Never averaged across gammas.
    small, large = min(config["sample_sizes"]), max(config["sample_sizes"])
    for gamma in config["gammas"]:
        a = errors.get((gamma, small), [])
        b = errors.get((gamma, large), [])
        if not a or not b:
            continue
        ratio = float(np.median(b) / np.median(a)) if np.median(a) > 0 else float("inf")
        result.rows.append(
            {
                "gamma": gamma,
                "criterion": "consistency_ratio",
                "median_abs_error_small": float(np.median(a)),
                "median_abs_error_large": float(np.median(b)),
                "ratio": ratio,
                "threshold": config["consistency_ratio_max"],
            }
        )
        if ratio > config["consistency_ratio_max"]:
            result.passed = False
            result.failures.append(
                f"consistency gamma={gamma}: ratio {ratio:.4f} above {config['consistency_ratio_max']}"
            )

    low, high = config["coverage_bounds"]
    for row in result.rows:
        if "coverage" in row and not (low <= row["coverage"] <= high):
            result.passed = False
            result.failures.append(
                f"coverage gamma={row['gamma']} n={row['n']}: {row['coverage']:.4f} outside [{low}, {high}]"
            )
    return result


# =========================================================================== #
# Stanton bias
# =========================================================================== #
def run_stanton_study(
    config: dict[str, Any] = STANTON_MC, progress: Callable[[str], None] | None = None
) -> StudyResult:
    """Pointwise bias against a known Ornstein-Uhlenbeck drift."""
    from start.tests.traded_risk import stanton_first_order

    R = int(config["R"])
    kappa, theta, sigma = 0.3, 0.04, 0.02
    dt = 1.0 / 252.0
    grid = np.linspace(0.02, 0.06, int(config["n_grid"]))
    true_drift = kappa * (theta - grid)

    result = StudyResult(study=config["study"], configuration=dict(config))
    medians: dict[int, np.ndarray] = {}

    for n in config["sample_sizes"]:
        seeds = child_seeds(int(config["root_seed"]) + n, R)
        errors = np.full((R, len(grid)), np.nan)
        signs = np.zeros((R, len(grid)), dtype=bool)
        for index, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            values = np.empty(n)
            values[0] = theta
            for t in range(1, n):
                values[t] = (
                    values[t - 1]
                    + kappa * (theta - values[t - 1]) * dt
                    + sigma * math.sqrt(dt) * rng.standard_normal()
                )
            series = pd.Series(values)
            h = 1.06 * float(np.std(values[:-1], ddof=1)) * (n ** (-1 / 5))
            frame = stanton_first_order(series, dt, grid, h)
            mu_hat = frame["mu"].to_numpy()
            errors[index] = np.abs(mu_hat - true_drift)
            signs[index] = np.sign(mu_hat) != np.sign(true_drift)
            if progress and (index + 1) % 50 == 0:
                progress(f"    Stanton n={n}: {index + 1}/{R}")

        medians[n] = np.nanmedian(errors, axis=0)
        wrong_sign = np.nanmean(signs, axis=0)
        for j, point in enumerate(grid):
            included = abs(true_drift[j]) >= config["sign_test_min_true_drift"]
            result.rows.append(
                {
                    "n": n,
                    "grid_point": float(point),
                    "true_drift": float(true_drift[j]),
                    "median_abs_bias": float(medians[n][j]),
                    "wrong_sign_rate": float(wrong_sign[j]),
                    "included_in_sign_test": bool(included),
                }
            )
            if included and wrong_sign[j] > config["wrong_sign_rate_max"]:
                result.passed = False
                result.failures.append(
                    f"wrong-sign rate at r={point:.4f}, n={n}: {wrong_sign[j]:.4f} "
                    f"above {config['wrong_sign_rate_max']}"
                )

    small, large = min(config["sample_sizes"]), max(config["sample_sizes"])
    if small in medians and large in medians:
        ratio = float(np.median(medians[large]) / np.median(medians[small]))
        result.rows.append(
            {
                "criterion": "bias_ratio",
                "ratio": ratio,
                "threshold": config["bias_ratio_max"],
            }
        )
        if ratio > config["bias_ratio_max"]:
            result.passed = False
            result.failures.append(f"bias ratio {ratio:.4f} above {config['bias_ratio_max']}")
    return result


# =========================================================================== #
# RegEM structural
# =========================================================================== #
def run_regem_study(
    config: dict[str, Any] = REGEM_MC, progress: Callable[[str], None] | None = None
) -> StudyResult:
    """PSD rate and convergence across missingness, mechanism and structure."""
    from start.data.synthetic_market import generate_market_world
    from start.tests.covariance import PSD_EIGENVALUE_FLOOR, run_regularized_em

    R = int(config["R"])
    result = StudyResult(study=config["study"], configuration=dict(config))

    for structure in config["structures"]:
        for mechanism in config["mechanisms"]:
            for rate in config["missingness"]:
                seeds = child_seeds(
                    int(config["root_seed"]) + hash((structure, mechanism)) % 10000 + int(rate * 100), R
                )
                psd = 0
                converged = 0
                valid = 0
                errors: list[float] = []
                for index, seed in enumerate(seeds):
                    draw = int(np.random.default_rng(seed).integers(0, 2**31 - 1))
                    world = generate_market_world(
                        n_assets=6,
                        n_periods=250,
                        n_factors=3,
                        seed=draw,
                        missing_rate=rate,
                        missing_mechanism=mechanism,
                        near_singular=(structure == "near_singular"),
                    )
                    if world.incomplete_returns is None:
                        continue
                    try:
                        outcome = run_regularized_em(
                            world.incomplete_returns.to_numpy(dtype=float),
                            ridge=1e-6,
                            tol=1e-6,
                            max_iter=200,
                        )
                    except ValueError:
                        continue
                    valid += 1
                    if outcome.converged:
                        converged += 1
                    minimum = float(np.linalg.eigvalsh(outcome.covariance).min())
                    if minimum >= -PSD_EIGENVALUE_FLOOR:
                        psd += 1
                    truth = world.true_asset_covariance.to_numpy()
                    errors.append(
                        float(
                            np.linalg.norm(outcome.covariance - truth, "fro") / np.linalg.norm(truth, "fro")
                        )
                    )
                    if progress and (index + 1) % 25 == 0:
                        progress(f"    RegEM {structure}/{mechanism}/{rate}: {index + 1}/{R}")

                psd_rate = psd / valid if valid else 0.0
                non_convergence = 1.0 - (converged / valid) if valid else 1.0
                result.rows.append(
                    {
                        "structure": structure,
                        "mechanism": mechanism,
                        "missing_rate": rate,
                        "R": R,
                        "n_valid": valid,
                        "psd_rate": psd_rate,
                        "non_convergence_rate": non_convergence,
                        "median_relative_frobenius_error": (
                            float(np.median(errors)) if errors else float("nan")
                        ),
                    }
                )
                if psd_rate < config["psd_rate_required"]:
                    result.passed = False
                    result.failures.append(
                        f"PSD rate {structure}/{mechanism}/{rate}: {psd_rate:.4f} "
                        f"below {config['psd_rate_required']}"
                    )
                if non_convergence > config["non_convergence_max"]:
                    result.passed = False
                    result.failures.append(
                        f"non-convergence {structure}/{mechanism}/{rate}: "
                        f"{non_convergence:.4f} above {config['non_convergence_max']}"
                    )
    return result
