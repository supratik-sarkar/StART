"""StART v4.2.0 — synthetic market review demo.

Runs the Gate-B analytical surfaces on synthetic data and carries every result through
the full governance path: evidence store, chained ledger, proof-carrying narrative,
evidence critic, disclosure envelope and attestation seal.

**The point is not a green screen.** Two of the four pre-registered Monte Carlo studies
failed a frozen criterion, and this demo exists partly to show those failures surviving
the entire chain without being softened into something reassuring. A governance pipeline
that quietly upgrades a failed statistical criterion is worse than none, because it
produces confident evidence that the estimator was checked.

The demo calls live registered implementations. No analytical formula is reimplemented
here. Data is synthetic, the seed is fixed, and nothing touches the network.

Run::

    python scripts/demo_market.py
    python scripts/demo_market.py --output-dir /tmp/demo_market
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

DEMO_SEED = 42


def _rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def _line(label: str, value: Any) -> None:
    print(f"  {label:<44}{value}")


def build_contexts(seed: int = DEMO_SEED) -> tuple[Any, Any, Any]:
    """Synthetic world plus its market and short-rate contexts."""
    from start.data.synthetic_market import generate_market_world
    from start.registry.market_contexts import MarketContext, PortfolioSpec

    world = generate_market_world(
        n_assets=12,
        n_periods=500,
        n_factors=3,
        seed=seed,
        include_short_rate=True,
        short_rate_gamma=0.5,
        missing_rate=0.15,
    )
    # Synthetic identifiers only. Nothing proprietary anywhere in this demo.
    renamed = {old: f"ASSET_{i + 1:03d}" for i, old in enumerate(world.returns.columns)}

    market = MarketContext(
        returns=world.returns.rename(columns=renamed),
        prices=world.prices.rename(columns=renamed),
        periods_per_year=world.periods_per_year,
        risk_free_rate=0.02,
        risk_free_frequency="annual",
        factor_returns=world.factor_returns,
        factor_exposures=world.factor_exposures.rename(index=renamed),
        pnl=world.pnl,
        hypothetical_pnl=world.hypothetical_pnl,
        var_series=world.var_series,
        var_confidence=world.var_confidence,
        portfolio=PortfolioSpec(
            weights=world.weights.rename(renamed),
            benchmark_weights=world.benchmark_weights.rename(renamed),
        ),
        seed=seed,
    )
    incomplete_ret = (
        world.incomplete_returns.rename(columns=renamed)
        if world.incomplete_returns is not None
        else world.returns.rename(columns=renamed)
    )
    incomplete = MarketContext(
        returns=incomplete_ret,
        periods_per_year=world.periods_per_year,
        portfolio=PortfolioSpec(weights=world.weights.rename(renamed)),
        seed=seed,
    )
    return world, market, incomplete


def run_analytics(market: Any, incomplete: Any, short_rate: Any) -> list[Any]:
    """Execute a representative cross-section of live registered surfaces.

    At least one surface from every Gate-B family, and both market and short-rate
    contexts. Nothing here recomputes an analytical formula.
    """
    from start.registry import list_tests

    registry = {spec.test_id: spec for spec in list_tests()}
    results: list[Any] = []

    market_tests = [
        "portfolio.historical_returns",
        "portfolio.risk_statistics",
        "portfolio.mean_variance",
        "attribution.factor_return_estimation",
        "attribution.return_attribution",
        "attribution.risk_attribution",
        "traded_risk.var_exceptions",
        "traded_risk.var_kupiec_pof",
        "covariance.empirical",
        "covariance.ledoit_wolf_shrinkage",
    ]
    for test_id in market_tests:
        results.append(registry[test_id].fn(market))

    # RegEM is the surface built for incomplete data.
    results.append(registry["covariance.regularized_em"].fn(incomplete))

    # Barrier monitoring needs an explicit level; it is never inferred.
    barrier = float(market.prices.iloc[:, 0].max() * 0.97)
    results.append(registry["traded_risk.brownian_bridge_barrier"].fn(market, barrier=barrier, sigma=0.25))

    # short_rate context: the two diffusion estimators.
    results.append(
        registry["traded_risk.cev_elasticity"].fn(short_rate, stated_gamma=0.5, bootstrap_draws=150)
    )
    results.append(registry["traded_risk.stanton_nonparametric"].fn(short_rate))
    return results


def build_narrative(records: list[Any]) -> str:
    """Proof-carrying narrative. Every number comes from an EvidenceRecord."""
    by_test = {record.test_id: record for record in records}

    def metric(test_id: str, key: str, default: float = 0.0) -> float:
        record = by_test.get(test_id)
        if record is None:
            return default
        value = record.metrics.get(key, default)
        return float(value) if isinstance(value, (int, float)) else default

    def ev(test_id: str) -> str:
        record = by_test.get(test_id)
        if record is None:
            return "[EV-MISSING]"
        eid = record.evidence_id
        return f"[{eid}]" if eid.startswith("EV-") else f"[EV-{eid}]"

    volatility = metric("portfolio.risk_statistics", "annualised_volatility")
    exceptions = metric("traded_risk.var_exceptions", "n_exceptions")
    kupiec_p = metric("traded_risk.var_kupiec_pof", "p_value")
    reconciliation = metric("attribution.return_attribution", "max_abs_reconciliation_error")
    shrinkage = metric("covariance.ledoit_wolf_shrinkage", "shrinkage_intensity")

    cev_coverage = metric("validation.cev_consistency", "observed.coverage_gamma_0_0")
    cev_consistency = metric("validation.cev_consistency", "observed.consistency_ratio_gamma_0_0")
    stanton_ratio = metric("validation.stanton_bias", "observed.bias_improvement_ratio")
    stanton_sign = metric("validation.stanton_bias", "observed.max_wrong_sign_rate_nonzero_drift")
    var_size = metric("validation.var_size_power", "observed.size_correct_forecast")

    return "\n".join(
        [
            "Synthetic market review — StART v4.2.0 Gate B.",
            "",
            "Analytical results.",
            f"The portfolio's annualised volatility was {volatility:.4f} "
            f"{ev('portfolio.risk_statistics')}. Return attribution reconciled to within "
            f"{reconciliation:.2e} {ev('attribution.return_attribution')}. The VaR backtest "
            f"recorded {exceptions:.0f} exceptions {ev('traded_risk.var_exceptions')}, and "
            f"the Kupiec proportion-of-failures test returned a p-value of {kupiec_p:.4f} "
            f"{ev('traded_risk.var_kupiec_pof')}. Ledoit-Wolf shrinkage intensity was "
            f"{shrinkage:.4f} {ev('covariance.ledoit_wolf_shrinkage')}.",
            "",
            "Pre-registered statistical validation.",
            f"The VaR backtest study met every pre-registered criterion, with empirical "
            f"size {var_size:.4f} under a correct forecast "
            f"{ev('validation.var_size_power')}. The RegEM study met both structural "
            f"criteria in all eighteen cells {ev('validation.regem_structural')}.",
            "",
            f"The CEV estimator satisfied the pre-registered consistency requirement, with "
            f"a ratio of {cev_consistency:.6f} at gamma = 0, but FAILED the nominal-coverage "
            f"requirement at gamma = 0, where empirical coverage was {cev_coverage:.4f} "
            f"against a required interval of [0.90, 0.98] "
            f"{ev('validation.cev_consistency')}. The CEV estimator is not fully validated.",
            "",
            f"The Stanton estimator satisfied the bias-improvement criterion, with a ratio "
            f"of {stanton_ratio:.6f}, but FAILED the pre-registered wrong-sign criterion, "
            f"reaching {stanton_sign:.4f} against a required maximum of 0.10 "
            f"{ev('validation.stanton_bias')}. The Stanton estimator is not fully validated.",
            "",
            "CEV and Stanton failed frozen pre-registered criteria, while "
            "VaR and RegEM passed their frozen criteria. Statistical validation is therefore "
            "partial and requires scientific disposition before CEV or Stanton is relied upon.",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="StART synthetic market review demo")
    parser.add_argument(
        "--output-dir", default=None, help="where to write evidence and ledger (default: a temp dir)"
    )
    parser.add_argument("--seed", type=int, default=DEMO_SEED)
    args = parser.parse_args(argv)

    from start.attestation.claims import bind_claims, extract_claims
    from start.attestation.disclosure import (
        DisclosureViolation,
        build_envelope,
        policy_for,
    )
    from start.attestation.replay import replay_ledger
    from start.attestation.seal import build_seal
    from start.core.schemas import EvidenceRecord
    from start.evidence.ledger import EvidenceLedger
    from start.risk.coverage import coverage_for_plan
    from start.risk.objects import RiskObject
    from start.risk.plan import synthesise_plan
    from start.validation.gate_b_evidence import (
        OVERALL_STATISTICAL_DISPOSITION,
        validation_results,
    )

    root = Path(args.output_dir) if args.output_dir else Path(tempfile.mkdtemp(prefix="start_market_"))
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"RUN-MARKET-{args.seed}"

    print("\nStART v4.2.0 — Synthetic Market Review")

    # ---------------------------------------------------------------- input
    _rule("INPUT")
    world, market, incomplete = build_contexts(args.seed)
    short_rate = world.short_rate_context()
    _line("data source", "synthetic (start.data.synthetic_market)")
    _line("network access", "none")
    _line("seed", args.seed)
    _line(
        "assets / periods / factors",
        f"{market.returns.shape[1]} / {market.returns.shape[0]} / {market.factor_returns.shape[1]}",
    )
    _line("market context fingerprint", market.fingerprint()[:32])
    _line("short-rate observations", short_rate.rates.size)
    _line("identifiers", "synthetic (ASSET_001 ...)")

    # ------------------------------------------------------------ risk plan
    _rule("RISK PLAN")
    for stripe, kind, context in (
        ("market", "deterministic_calculator", "market"),
        ("treasury_irrbb", "deterministic_calculator", "short_rate"),
    ):
        obj = RiskObject(object_id="SYN-1", kind=kind, name="Synthetic book", owner="", materiality="high")
        coverage = coverage_for_plan(
            synthesise_plan(stripe_id=stripe, obj=obj, materiality="high"),
            context_type=context,
        )
        candidates = {t for d in coverage.dimensions for t in d.test_ids}
        _line(
            f"{stripe} / {context}",
            f"{len(candidates)} candidate test(s), "
            f"{sum(1 for d in coverage.dimensions if d.covered)}/"
            f"{len(coverage.dimensions)} dimensions covered",
        )

    # ------------------------------------------------------- executed tests
    _rule("EXECUTED TESTS")
    results = run_analytics(market, incomplete, short_rate)
    results.extend(validation_results())
    for result in results:
        _line(result.test_id, str(result.status).upper())

    # --------------------------------------------------------------- evidence
    _rule("EVIDENCE")
    ledger = EvidenceLedger(root / "ledger.jsonl", root / "evidence")
    records: list[EvidenceRecord] = []
    for result in results:
        records.append(ledger.append(result, run_id=run_id))
    _line("EvidenceRecords appended", len(records))
    _line("evidence IDs unique", len({r.evidence_id for r in records}) == len(records))
    _line("store root", str(root / "evidence"))
    matrix_free = all(not hasattr(value, "shape") for record in records for value in record.metrics.values())
    _line("no raw matrices in metrics", matrix_free)

    # --------------------------------------------- statistical validation
    _rule("STATISTICAL VALIDATION (verified B7 results)")
    validation_records = [r for r in records if r.test_id.startswith("validation.")]
    for record in validation_records:
        failed = record.metrics.get("n_criteria_failed", 0)
        _line(record.test_id, f"{str(record.status).upper()}  ({failed} criterion/criteria failed)")
    _line("overall disposition", OVERALL_STATISTICAL_DISPOSITION)

    # -------------------------------------------------------------- narrative
    _rule("NARRATIVE")
    narrative = build_narrative(records)
    print(narrative)

    # ------------------------------------------------------------- critic
    _rule("CRITIC")
    claims = extract_claims(narrative)
    binding = bind_claims(claims, records)
    _line("quantitative claims extracted", len(claims))
    _line("claims bound to evidence", len(binding.bound))
    _line("claims unbound", len(binding.unbound))

    # A quantitative claim whose number appears nowhere in evidence must not bind.
    bad_claim = "CEV achieved 0.9873 empirical coverage for every gamma."
    bad_binding = bind_claims(extract_claims(bad_claim), records)
    _line("fabricated-number claim rejected", len(bad_binding.bound) == 0)

    # The correctly grounded version of the same statement DOES bind.
    good_claim = "CEV empirical coverage was 0.6350 at gamma zero."
    good_binding = bind_claims(extract_claims(good_claim), records)
    _line("grounded failure claim accepted", len(good_binding.bound) > 0)

    # Honest limitation: the binder matches NUMBERS, not semantics. "95% coverage"
    # would bind to any 0.95 already in evidence (a confidence level, say), so a
    # semantically false claim can bind on a coincidental value. Numeric grounding is
    # necessary, not sufficient.
    _line("binder matches numbers not semantics", True)

    # --------------------------------------------------------- disclosure
    _rule("DISCLOSURE")
    policy = policy_for("enterprise")
    _line("policy", policy.id)
    _line("strict", policy.strict)

    permitted = {
        "run_id": run_id,
        "metrics": {"annualised_volatility": 0.12, "n_exceptions": 4},
    }
    envelope = build_envelope(permitted, policy=policy)
    _line("permitted payload projected", len(envelope.projected) > 0)
    _line("withheld paths", len(envelope.withheld_paths))

    # An entity-like field must not survive. Under a strict policy the layer refuses to
    # build an envelope at all rather than silently dropping the field, which is the
    # stronger behaviour: the caller learns they assembled something they should not have.
    restricted = dict(permitted, customer_email="someone@example.invalid")
    try:
        blocked = build_envelope(restricted, policy=policy)
        survived = any("customer_email" in path for path in blocked.projected)
        _line("restricted field survived projection", survived)
        _line("restricted field withheld", not survived)
    except DisclosureViolation:
        _line("restricted field", "REFUSED (strict policy raises rather than drops)")

    # ---------------------------------------------------------------- ledger
    _rule("LEDGER")
    _line("records", len(ledger))
    _line("chain verify()", ledger.verify())
    verdict = replay_ledger(root / "ledger.jsonl")
    _line("replay verdict", getattr(verdict, "ok", verdict))

    # ----------------------------------------------------------- attestation
    _rule("ATTESTATION")
    seal = build_seal(
        review_id=run_id,
        evidence_head=records[-1].evidence_id if records else None,
        metadata={
            "statistical_disposition": OVERALL_STATISTICAL_DISPOSITION,
            "n_validation_failures": sum(1 for r in validation_records if str(r.status).lower() == "fail"),
        },
    )
    _line("seal created", True)
    _line("merkle root", seal.merkle_root[:32] if hasattr(seal, "merkle_root") else "n/a")
    _line("seal records failures", seal.metadata.get("n_validation_failures"))

    # ------------------------------------------------------------- summary
    _rule("FINAL GOVERNANCE SUMMARY")
    n_failed = sum(1 for r in validation_records if str(r.status).lower() == "fail")
    integrity = ledger.verify() and len({r.evidence_id for r in records}) == len(records)

    _line("implementation status", "COMPLETE")
    _line("software verification", "PASS" if integrity else "FAIL")
    _line("statistical validation", f"PARTIAL — {n_failed} of 4 studies failed a frozen criterion")
    _line("release / governance disposition", "REVIEW REQUIRED")
    print()
    print("  Two pre-registered criteria failed. Those failures were preserved through")
    print("  evidence, ledger, narrative and attestation without being softened. That")
    print("  propagation is the result this demo is designed to show.")
    print(f"\n  Artefacts: {root}\n")

    # An intentional statistical FAIL is faithfully-reported evidence, not a crash.
    # Only an integrity failure is an infrastructure problem.
    return 0 if integrity else 1


if __name__ == "__main__":
    sys.exit(main())
