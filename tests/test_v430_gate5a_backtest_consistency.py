"""Gate 5A: Tail Backtest Statistical Self-Consistency & Acceptance Audit Tests.

Verifies:
1. Exception count and exception rate invariants: sum(I_t) == n_exceptions, rate == sum(I_t)/T.
2. Kupiec POF independent analytical known-answer tests (x=4/T=250 vs x=5/T=250).
3. Christoffersen independence transition count contract (n00+n01+n10+n11 == T-1) and consecutive-4 known-answer test.
4. Joint conditional coverage exact additive identity (LR_cc == LR_uc + LR_ind, df=2).
5. Negative evidence showcase verification on single canonical indicator sequence.
6. Showcase manifest metrics sourced directly from EvidenceRecords (no hardcoded prose discrepancy).
7. Traceable PSD repair provenance before Gaussian simulation in adversarial challenge resolution.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

from start.agents.market_review import AdversarialChallengeAgent
from start.portfolio.contracts import (
    TailBacktestResult,
)
from start.portfolio.covariance import diagnose_covariance
from start.portfolio.evidence_bridge import (
    covariance_diagnostics_to_evidence,
    tail_backtest_to_evidence,
)
from start.portfolio.tail_risk import (
    christoffersen_independence_lr,
    compute_exception_duration_diagnostics,
    compute_tail_severity,
    kupiec_lr,
    run_comprehensive_tail_backtest,
)

# =========================================================================== #
# 1. EXCEPTION COUNT & RATE INVARIANTS
# =========================================================================== #


def test_exception_count_and_rate_invariants() -> None:
    """Identity check: sum(I_t) == n_exceptions and rate == sum(I_t)/T across all outputs."""
    n_days = 250
    canonical_indicators = np.zeros(n_days, dtype=int)
    canonical_indicators[50:54] = 1  # 4 exceptions
    ind_hash = hashlib.sha256(canonical_indicators.tobytes()).hexdigest()

    var_series = np.full(n_days, 0.025)
    rng = np.random.RandomState(42)
    pnl_series = np.clip(rng.normal(loc=0.0005, scale=0.005, size=n_days), -0.015, 0.015)
    pnl_series[50] = -0.035
    pnl_series[51] = -0.042
    pnl_series[52] = -0.038
    pnl_series[53] = -0.045

    res: TailBacktestResult = run_comprehensive_tail_backtest(
        pnl_or_losses=pnl_series,
        var_series=var_series,
        var_confidence=0.99,
        test_significance=0.05,
        pnl_source="actual",
        is_loss_series=False,
    )

    # 1. Indicator invariants
    assert res.n_observations == 250
    assert res.n_exceptions == 4
    assert sum(res.indicators) == 4
    assert math.isclose(res.exception_rate, 4 / 250, rel_tol=1e-12)
    assert res.exception_rate == 0.016
    assert res.indicator_hash == ind_hash

    # 2. Evidence bridge invariants
    ev = tail_backtest_to_evidence(res)
    assert ev.metrics["n_observations"] == 250
    assert ev.metrics["n_exceptions"] == 4
    assert ev.metrics["exception_rate"] == 0.016
    assert ev.metrics["indicator_hash"] == ind_hash


# =========================================================================== #
# 2. KUPIEC INDEPENDENT ANALYTICAL KNOWN-ANSWER TESTS
# =========================================================================== #


def test_kupiec_independent_analytical_known_answers() -> None:
    """Independently calculate Kupiec POF from defining likelihood ratio formula."""

    def _independent_kupiec(T: int, x: int, p0: float) -> tuple[float, float]:
        pi_hat = x / T
        lr = 2.0 * (x * math.log(pi_hat / p0) + (T - x) * math.log((1.0 - pi_hat) / (1.0 - p0)))
        p_val = float(stats.chi2.sf(lr, df=1))
        return lr, p_val

    # Case 1: T=250, x=4, p0=0.01 (observed rate = 1.6%)
    expected_lr_4, expected_p_4 = _independent_kupiec(250, 4, 0.01)
    prod_lr_4 = kupiec_lr(250, 4, 0.01)
    prod_p_4 = float(stats.chi2.sf(prod_lr_4, df=1))

    assert math.isclose(prod_lr_4, expected_lr_4, rel_tol=1e-10)
    assert math.isclose(prod_p_4, expected_p_4, rel_tol=1e-10)
    assert math.isclose(prod_lr_4, 0.7691383643858458, rel_tol=1e-6)
    assert math.isclose(prod_p_4, 0.380483738238954, rel_tol=1e-6)

    # Case 2: T=250, x=5, p0=0.01 (observed rate = 2.0%)
    expected_lr_5, expected_p_5 = _independent_kupiec(250, 5, 0.01)
    prod_lr_5 = kupiec_lr(250, 5, 0.01)
    prod_p_5 = float(stats.chi2.sf(prod_lr_5, df=1))

    assert math.isclose(prod_lr_5, expected_lr_5, rel_tol=1e-10)
    assert math.isclose(prod_p_5, expected_p_5, rel_tol=1e-10)
    assert math.isclose(prod_lr_5, 1.956809788230652, rel_tol=1e-6)
    assert math.isclose(prod_p_5, 0.16185491719603548, rel_tol=1e-6)

    # Distinctness proof: x=4 (LR ~ 0.769) and x=5 (LR ~ 1.957) cannot be confused
    assert abs(prod_lr_4 - prod_lr_5) > 1.18


# =========================================================================== #
# 3. CHRISTOFFERSEN INDEPENDENCE TRANSITION CONTRACT & CONSECUTIVE-4
# =========================================================================== #


def test_christoffersen_transition_count_contract_and_consecutive_four() -> None:
    """Verify standard (T-1) transition contract and exact consecutive-4 Markov likelihoods."""
    n_days = 250
    canonical_indicators = np.zeros(n_days, dtype=int)
    canonical_indicators[50:54] = 1  # Exactly 4 consecutive exceptions at 50, 51, 52, 53

    prev, curr = canonical_indicators[:-1], canonical_indicators[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    # 1. Standard (T-1) transition identity
    assert n00 + n01 + n10 + n11 == n_days - 1 == 249
    assert n00 == 244
    assert n01 == 1
    assert n10 == 1
    assert n11 == 3

    # 2. Independent analytical transition likelihood calculation
    pi01 = 1.0 / (244 + 1)
    pi11 = 3.0 / (1 + 3)
    pi = 4.0 / 249.0

    log_LA = 244 * math.log(1 - pi01) + 1 * math.log(pi01) + 1 * math.log(1 - pi11) + 3 * math.log(pi11)
    log_L0 = (244 + 1) * math.log(1 - pi) + (1 + 3) * math.log(pi)
    expected_lr_ind = 2.0 * (log_LA - log_L0)
    expected_p_ind = float(stats.chi2.sf(expected_lr_ind, df=1))

    prod_lr_ind = christoffersen_independence_lr(244, 1, 1, 3)
    prod_p_ind = float(stats.chi2.sf(prod_lr_ind, df=1))

    assert math.isclose(prod_lr_ind, expected_lr_ind, rel_tol=1e-10)
    assert math.isclose(prod_p_ind, expected_p_ind, rel_tol=1e-10)
    assert math.isclose(prod_lr_ind, 23.48755400276484, rel_tol=1e-6)
    assert math.isclose(prod_p_ind, 1.2572445791397529e-06, rel_tol=1e-6)


# =========================================================================== #
# 4. CONDITIONAL COVERAGE EXACT ADDITIVE IDENTITY
# =========================================================================== #


def test_conditional_coverage_exact_additive_identity() -> None:
    """LR_cc == LR_uc + LR_ind must hold strictly on the same canonical sequence."""
    n_days = 250
    canonical_indicators = np.zeros(n_days, dtype=int)
    canonical_indicators[50:54] = 1

    var_series = np.full(n_days, 0.025)
    rng = np.random.RandomState(42)
    pnl_series = np.clip(rng.normal(loc=0.0005, scale=0.005, size=n_days), -0.015, 0.015)
    pnl_series[50] = -0.035
    pnl_series[51] = -0.042
    pnl_series[52] = -0.038
    pnl_series[53] = -0.045

    res = run_comprehensive_tail_backtest(
        pnl_or_losses=pnl_series,
        var_series=var_series,
        var_confidence=0.99,
        test_significance=0.05,
        pnl_source="actual",
        is_loss_series=False,
    )

    # Strict additive identity
    assert math.isclose(res.conditional_coverage_lr, res.kupiec_lr + res.christoffersen_lr, rel_tol=1e-12)
    expected_p_cc = float(stats.chi2.sf(res.conditional_coverage_lr, df=2))
    assert math.isclose(res.conditional_coverage_p_value, expected_p_cc, rel_tol=1e-12)
    assert math.isclose(res.conditional_coverage_lr, 24.25669236715067, rel_tol=1e-5)


# =========================================================================== #
# 5. NEGATIVE SHOWCASE SCIENTIFIC TARGET
# =========================================================================== #


def test_negative_showcase_scientific_target() -> None:
    """Proof-carrying negative evidence: Kupiec DOES NOT REJECT, Christoffersen REJECTS."""
    n_days = 250
    var_series = np.full(n_days, 0.025)
    rng = np.random.RandomState(123)
    pnl_series = np.clip(rng.normal(loc=0.0005, scale=0.006, size=n_days), -0.015, 0.015)
    pnl_series[50] = -0.035
    pnl_series[51] = -0.042
    pnl_series[52] = -0.038
    pnl_series[53] = -0.045

    res = run_comprehensive_tail_backtest(
        pnl_or_losses=pnl_series,
        var_series=var_series,
        var_confidence=0.99,
        test_significance=0.05,
        pnl_source="actual",
        is_loss_series=False,
    )

    # 1. Kupiec fails to reject unconditional coverage (p > 0.05)
    assert res.kupiec_p_value > 0.05
    assert res.kupiec_rejected is False

    # 2. Christoffersen independence rejects serial independence (p < 0.05)
    assert res.christoffersen_p_value < 0.05
    assert res.christoffersen_rejected is True

    # 3. Joint conditional coverage rejects (p < 0.05)
    assert res.conditional_coverage_p_value < 0.05
    assert res.conditional_coverage_rejected is True

    # 4. Duration diagnostics indicate consecutive cluster of 4
    dur = compute_exception_duration_diagnostics(res.indicators)
    assert dur.n_durations == 3
    assert dur.mean_duration == 1.0
    assert dur.max_run_length == 4

    # 5. Severity diagnostics
    sev = compute_tail_severity(losses=-pnl_series, var_forecasts=var_series, indicators=res.indicators)
    assert sev.n_exceptions == 4
    assert math.isclose(sev.mean_absolute_exceedance, 0.015, rel_tol=1e-6)
    assert math.isclose(sev.max_normalized_exceedance, 1.80, rel_tol=1e-6)


# =========================================================================== #
# 6. SHOWCASE MANIFEST VALUES SOURCED DIRECTLY FROM EVIDENCE
# =========================================================================== #


def test_showcase_manifest_values_match_evidence() -> None:
    """Verify that values in showcase manifest match the computed EvidenceRecords."""
    manifest_path = Path("start_output/gate5_showcase/manifest.json")
    summary_path = Path("start_output/gate5_showcase/gate5_summary.json")

    assert manifest_path.exists(), "manifest.json missing"
    assert summary_path.exists(), "gate5_summary.json missing"

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    showcase_metrics = manifest.get("canonical_backtest_showcase", {})
    assert showcase_metrics["n_observations"] == 250
    assert showcase_metrics["n_exceptions"] == 4
    assert showcase_metrics["exception_rate"] == 0.016
    assert math.isclose(showcase_metrics["kupiec_lr"], 0.769138, rel_tol=1e-4)
    assert math.isclose(showcase_metrics["kupiec_p_value"], 0.380484, rel_tol=1e-4)
    assert showcase_metrics["n00"] == 244
    assert showcase_metrics["n01"] == 1
    assert showcase_metrics["n10"] == 1
    assert showcase_metrics["n11"] == 3
    assert math.isclose(showcase_metrics["christoffersen_lr"], 23.487554, rel_tol=1e-4)
    assert math.isclose(showcase_metrics["conditional_coverage_lr"], 24.256692, rel_tol=1e-4)


# =========================================================================== #
# 7. TRACEABLE PSD REPAIR PROVENANCE BEFORE GAUSSIAN SIMULATION
# =========================================================================== #


def test_psd_simulation_provenance_chain_traceability() -> None:
    """Verify explicit provenance chain: raw covariance -> PSD repair -> challenge diagnostic."""
    diag = diagnose_covariance(np.array([[1.0, 0.9], [0.9, 0.1]]))
    ev_raw = covariance_diagnostics_to_evidence(diag)
    ev_raw.evidence_id = "EV-COV-SRC-001"

    challenger = AdversarialChallengeAgent()
    challenge_context = {
        "covariance": np.array([[1.0, 0.9], [0.9, 0.1]]),
        "evidence_records": [ev_raw],
    }

    chal_out = challenger.execute(challenge_context)
    resolutions = chal_out["resolutions"]
    all_evs = challenge_context["evidence_records"]

    # Verify distinct PSD repair evidence record was generated and registered in context
    psd_rep_evs = [e for e in all_evs if e.test_id == "covariance.psd_repair"]
    assert len(psd_rep_evs) >= 1
    repair_ev_id = psd_rep_evs[0].evidence_id

    # Verify raw covariance ID, repair evidence ID, and generated diagnostic IDs are distinct
    assert ev_raw.evidence_id != repair_ev_id
    for r in resolutions:
        for gen_id in r["generated_evidence_ids"]:
            assert gen_id != ev_raw.evidence_id
            assert gen_id != repair_ev_id
            assert gen_id.startswith("EV-")
