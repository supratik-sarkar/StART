"""Gate-B validation evidence.

The point of this module
------------------------

Two of the four pre-registered Monte Carlo studies **failed**. This module carries those
failures into the ordinary evidence spine — ``TestResult`` → ``EvidenceRecord`` → store →
ledger → narrative → critic → seal — without any of those stages converting them into
something reassuring.

That is the whole B8 acceptance requirement. A governance system that quietly rounds a
failed statistical criterion up to a warning is worse than one with no validation at all,
because it produces confident-looking evidence that the estimator was checked.

So the statuses here are fixed by the observed numbers, and the module deliberately
provides no route to override them.

Provenance
----------

These are **verified B7 validation results**, independently reproduced by the verification suite
against the frozen configurations. They are not freshly simulated demo outcomes, and
every record says so. Re-running the full studies inside a demo would take minutes and
would invite the temptation to accept whichever run looked better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from start.core.schemas import Status, TestResult

__all__ = [
    "ValidationOutcome", "VERIFIED_B7_RESULTS", "validation_results",
    "validation_results_for_domains", "PROVENANCE", "OVERALL_STATISTICAL_DISPOSITION",
]

#: Every record carries this. The distinction between reproduced validation evidence and
#: a demo-time simulation matters: only the first was run at the frozen R.
PROVENANCE = (
    "verified B7 validation results, independently reproduced against the frozen "
    "pre-registered configurations; NOT freshly simulated demo outcomes"
)

#: The honest one-line summary. Not "pass with observations".
OVERALL_STATISTICAL_DISPOSITION = (
    "PARTIAL — 2 of 4 pre-registered studies failed a frozen criterion"
)


@dataclass(frozen=True)
class ValidationOutcome:
    """One pre-registered study outcome, with its criterion and observation."""

    study_id: str
    configuration_hash: str
    root_seed: int
    criteria: tuple[dict[str, Any], ...]
    status: Status
    classification: str
    limitations: tuple[str, ...]
    summary: str
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def to_test_result(self) -> TestResult:
        """Ordinary evidence. The failing status is the status."""
        metrics: dict[str, Any] = {
            "study_id": self.study_id,
            "configuration_hash": self.configuration_hash,
            "root_seed": self.root_seed,
            "provenance": PROVENANCE,
            "n_criteria": len(self.criteria),
            "n_criteria_failed": sum(1 for c in self.criteria if not c["passed"]),
            "classification": self.classification,
        }
        metrics.update(self.extra_metrics)
        for criterion in self.criteria:
            name = criterion["name"]
            metrics[f"observed.{name}"] = criterion["observed"]
            metrics[f"required.{name}"] = criterion["required"]
            metrics[f"passed.{name}"] = criterion["passed"]

        return TestResult(
            test_id=f"validation.{self.study_id}",
            test_name=f"Pre-registered validation: {self.study_id}",
            status=self.status,
            params={
                "study_id": self.study_id,
                "configuration_hash": self.configuration_hash,
                "root_seed": self.root_seed,
                "provenance": PROVENANCE,
            },
            metrics=metrics,
            interpretation=self.summary,
            limitations=list(self.limitations),
        )


# --------------------------------------------------------------------------- #
# The four verified outcomes. Values are the independently reproduced ones.
# --------------------------------------------------------------------------- #
VERIFIED_B7_RESULTS: tuple[ValidationOutcome, ...] = (
    ValidationOutcome(
        study_id="var_size_power",
        configuration_hash="a2cb102177317466ea47dc9a4d9737d8",
        root_seed=20240601,
        criteria=(
            {"name": "size_correct_forecast", "observed": 0.066,
             "required": "in [0.031, 0.069]", "passed": True},
            {"name": "power_understated_0_7x", "observed": 1.000,
             "required": ">= 0.50", "passed": True},
            {"name": "power_overstated_1_5x", "observed": 0.992,
             "required": ">= 0.20", "passed": True},
        ),
        status=Status.PASS,
        classification="all pre-registered criteria met",
        limitations=(
            "The deciding test is Kupiec POF, fixed before execution. Christoffersen "
            "statistics were computed for information and never substituted.",
            "A PASS here means the backtest has correct size and useful power against "
            "the two stated mis-specifications. It does NOT establish that any "
            "particular VaR model is correct.",
            PROVENANCE,
        ),
        summary=(
            "VaR backtest validation PASSED. Empirical size 0.066 under a correct "
            "forecast lies inside the pre-registered band [0.031, 0.069]; power was "
            "1.000 against a 0.7x understated forecast and 0.992 against a 1.5x "
            "overstated forecast."
        ),
        extra_metrics={"nominal_significance_level": 0.05, "nominal_size": 0.05},
    ),
    ValidationOutcome(
        study_id="cev_consistency",
        configuration_hash="a9b387fb2905aa48fa3732cee79d749a",
        root_seed=20240602,
        criteria=(
            {"name": "consistency_ratio_gamma_0_0", "observed": 0.469967,
             "required": "<= 0.70", "passed": True},
            {"name": "consistency_ratio_gamma_0_5", "observed": 0.100207,
             "required": "<= 0.70", "passed": True},
            {"name": "consistency_ratio_gamma_1_0", "observed": 0.120244,
             "required": "<= 0.70", "passed": True},
            {"name": "coverage_gamma_0_0", "observed": 0.635,
             "required": "in [0.90, 0.98]", "passed": False},
            {"name": "coverage_gamma_0_5", "observed": 0.970,
             "required": "in [0.90, 0.98]", "passed": True},
            {"name": "coverage_gamma_1_0", "observed": 0.935,
             "required": "in [0.90, 0.98]", "passed": True},
            {"name": "failure_rate", "observed": 0.0,
             "required": "<= 0.05 for n >= 1000", "passed": True},
        ),
        # FAIL. Not WARN. Understanding a cause does not discharge a criterion.
        status=Status.FAIL,
        classification="estimator approximation / interval limitation",
        limitations=(
            "The pre-registered CONSISTENCY criterion passed for all three gamma "
            "values; the estimator converges as sample size grows.",
            "The pre-registered nominal-COVERAGE criterion FAILED at gamma = 0, where "
            "empirical coverage was 0.635 against a required [0.90, 0.98].",
            "Diagnostic at n = 2500: bias +0.03389 against a sampling standard "
            "deviation of 0.03218, so |bias|/sd = 1.053. The bootstrap interval is "
            "correctly SIZED but centred away from the true value, which is why "
            "coverage collapses while consistency holds.",
            "A pair bootstrap resamples sampling variability and cannot correct a "
            "systematic discretisation bias. At gamma = 0 the true slope is zero, so "
            "no signal offsets the bias; at gamma = 0.5 and 1.0 the true slopes "
            "dominate it and coverage passes.",
            "The CEV estimator is NOT fully validated. Its interval behaviour at "
            "gamma = 0 remains an open scientific question.",
            PROVENANCE,
        ),
        summary=(
            "CEV statistical validation FAILED. The pre-registered consistency "
            "criterion was satisfied for every gamma (ratios 0.469967, 0.100207, "
            "0.120244 against a required <= 0.70) and the failure rate was 0.0, but "
            "empirical coverage at gamma = 0 was 0.635 against a required interval of "
            "[0.90, 0.98]. The observed behaviour is consistent with a discretisation "
            "bias that the pair-bootstrap interval does not correct."
        ),
        extra_metrics={
            "required.gamma_0": 0.0,
            "required.coverage_interval_lower": 0.90,
            "required.coverage_interval_upper": 0.98,
        },
    ),
    ValidationOutcome(
        study_id="stanton_bias",
        configuration_hash="f9a471fc64bd7f8c6064aae9cd11f85d",
        root_seed=20240603,
        criteria=(
            {"name": "bias_improvement_ratio", "observed": 0.305052,
             "required": "<= 0.70", "passed": True},
            {"name": "max_wrong_sign_rate_nonzero_drift", "observed": 0.475,
             "required": "<= 0.10 at every included grid point", "passed": False},
        ),
        status=Status.FAIL,
        classification=(
            "process / criterion calibration + finite-sample estimator resolution "
            "limitation"
        ),
        limitations=(
            "The pre-registered BIAS-IMPROVEMENT criterion passed: the ratio was "
            "0.305052 against a required <= 0.70, so pointwise bias shrinks materially "
            "with sample size.",
            "The pre-registered WRONG-SIGN criterion FAILED. Rates at the non-zero "
            "drift grid points ranged from roughly 0.250 to 0.475 against a required "
            "<= 0.10.",
            "True drift magnitudes on the simulated OU process are 0.0015 to 0.0060, "
            "while estimator noise is roughly 0.0104 to 0.0135, giving signal-to-noise "
            "ratios of 0.137 to 0.472. Sign cannot be recovered better than chance when "
            "the noise exceeds the quantity whose sign is being asked about.",
            "Drift enters at O(dt) while diffusion noise enters at O(sqrt(dt)), which "
            "is why drift is substantially harder to estimate than diffusion.",
            "Whether to re-specify the simulated process, the grid or the criterion is "
            "a design decision for review. Nothing was changed after observing the "
            "failure.",
            PROVENANCE,
        ),
        summary=(
            "Stanton statistical validation FAILED. The pre-registered bias-improvement "
            "criterion was satisfied (ratio 0.305052 against a required <= 0.70), but "
            "wrong-sign rates at non-zero drift grid points reached 0.475 against a "
            "required <= 0.10. The observed behaviour is consistent with drift "
            "magnitudes falling below the finite-sample resolution of a first-order "
            "nonparametric estimator."
        ),
        extra_metrics={
            "required.max_wrong_sign_rate": 0.10,
        },
    ),
    ValidationOutcome(
        study_id="regem_structural",
        configuration_hash="e18f982a841276621834340256474d0f",
        root_seed=20240604,
        criteria=(
            {"name": "psd_rate_all_cells", "observed": 1.0,
             "required": "= 1.0 in all 18 cells", "passed": True},
            {"name": "non_convergence_rate_all_cells", "observed": 0.0,
             "required": "<= 0.05 in all 18 cells", "passed": True},
        ),
        status=Status.PASS,
        classification="structural criteria met in all cells",
        limitations=(
            "PSD rate and non-convergence are STRUCTURAL criteria. They establish that "
            "the estimator returns a usable covariance, not that it is accurate.",
            "No dominance criterion was imposed and none is claimed. No covariance "
            "estimator dominates across every missingness mechanism and structure, and "
            "requiring it would be requiring something untrue.",
            "The Gaussian and MAR working assumptions were satisfied by construction in "
            "the simulation. Real data need not satisfy either.",
            PROVENANCE,
        ),
        summary=(
            "RegEM statistical validation PASSED. Across all 18 cells (3 missingness "
            "levels x 2 mechanisms x 3 covariance structures) the PSD rate was 1.0 and "
            "the non-convergence rate was 0.0."
        ),
        extra_metrics={"n_cells": 18},
    ),
)


def validation_results() -> list[TestResult]:
    """The four outcomes as ordinary ``TestResult`` objects."""
    return [outcome.to_test_result() for outcome in VERIFIED_B7_RESULTS]


def validation_results_for_domains(domains: tuple[Any, ...]) -> list[TestResult]:
    """Domain-scoped validation evidence: Market gets VaR+RegEM; Treasury gets CEV+Stanton."""
    # Convert domain elements to string representations to remain decoupled/flexible
    domain_strs = {str(d).lower() for d in domains}
    include_market = any("market" in ds for ds in domain_strs)
    include_treasury = any("treasury" in ds or "short_rate" in ds for ds in domain_strs)

    results: list[TestResult] = []
    for outcome in VERIFIED_B7_RESULTS:
        if outcome.study_id in ("var_size_power", "regem_structural") and include_market:
            results.append(outcome.to_test_result())
        elif outcome.study_id in ("cev_consistency", "stanton_bias") and include_treasury:
            results.append(outcome.to_test_result())
    return results
