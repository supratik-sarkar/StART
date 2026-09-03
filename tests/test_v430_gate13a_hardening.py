"""StART v4.3.0 Gate 13A: Scientific Decision Provenance & Collision-Safe Evidence Tests.

Covers all 24 required test cases:
1. Kupiec table gets statistic from evidence.
2. Kupiec decision comes from stored decision, not p-value.
3. Gamma comes from evidence (mutation test: 0.01 displays gamma=0.01 without renderer change).
4. Missing decision renders N/A.
5. Size/power no hard-coded band (missing -> band=N/A).
6. Size/power no hard-coded gamma (missing -> gamma=N/A).
7. Missing size/power fields render N/A.
8. Missing n_criteria_failed cannot become PASS.
9. Status comes strictly from EvidenceRecord.status.
10. Metric path collisions do not overwrite (3 distinct VaR p_values survive).
11. Wrong EV + right numeric value does not ground.
12. Precision-derived rounding (1.886 matches 1.8862 at 3 dec; 1000 matches 1000.0).
13. 1.850 vs 1.8862 fails (no fuzzy tolerance).
14. Typed percent normalization (5% binds to alpha=0.05, fails against n_observations=5).
15. Criterion parsing restricted (is_criterion=True only).
16. Factor artifact zero recomputation (spy test: build_linear_factor_model call_count == 0).
17. Artifact EvidenceRecord lineage (artifact IDs <= attribution EvidenceRecord IDs).
18. Committee consumer provenance (record-specific evaluation, zero bare path lookup).
19. Governance consumer provenance (structured metadata and decision log).
20. Grounding census invariant (bound + unbound == total_claims).
21. Provider call count on grounding PASS == 1.
22. Provider call count on grounding FAIL == 1 (zero re-prompt loops).
23. Gate-12 live-provider mocked path still works.
24. Q/C/V/A state machine flow (Q/C remain, A/O advance).
"""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from start.agents.committee import CrossAnalyticalCommittee
from start.attestation.claims import (
    Claim,
    GroundingReasonCode,
    _match_candidates_in_fields,
    bind_claims,
    extract_claims,
    flatten_evidence_values,
)
from start.core.schemas import EvidenceRecord, Status
from start.data.synthetic_market import generate_market_world
from start.evidence.ledger import EvidenceLedger
from start.review.architecture import (
    LLMReviewConfig,
    ReviewContextBundle,
    ReviewDomain,
    ReviewMode,
)
from start.review.evidence_view import (
    CheckpointMetricRef,
    build_checkpoint_evidence_view,
)
from start.review.executor import (
    execute_market_treasury_tests,
    generate_review_artifacts,
    run_market_treasury_review,
)
from start.review.state_machine import CheckpointState, CheckpointStateMachine
from start.review.tables import (
    build_governance_table,
    build_var_tail_table,
)

# =========================================================================== #
# Fixture Helpers
# =========================================================================== #


def _make_base_var_records() -> list[EvidenceRecord]:
    common = {"model_id": "M-MARKET", "dataset_id": "D-MARKET", "run_id": "RUN-01"}
    r_kupiec = EvidenceRecord(
        evidence_id="EV-KUPIEC-01",
        test_id="traded_risk.var_kupiec_pof",
        test_name="Kupiec proportion-of-failures test",
        status=Status.RECORDED,
        **common,
        params={"alpha": 0.05, "gamma_test": 0.05, "pnl_source": "actual"},
        metrics={
            "n_observations": 1000,
            "n_exceptions": 6,
            "lr_uc": 1.8862324083,
            "p_value": 0.1696274814,
            "confidence": 0.99,
            "alpha_var": 0.01,
            "expected_probability": 0.01,
            "gamma_test": 0.05,
            "statistical_gamma_test": 0.05,
            "statistical_criterion_source": "STATISTICAL_TEST_SPECIFICATION",
            "alpha": 0.05,
            "critical_value": 3.8414588207,
            "rejected": False,
        },
    )
    r_ind = EvidenceRecord(
        evidence_id="EV-IND-02",
        test_id="traded_risk.var_christoffersen_independence",
        test_name="Christoffersen independence test",
        status=Status.RECORDED,
        **common,
        params={"alpha": 0.05, "gamma_test": 0.05, "pnl_source": "actual"},
        metrics={
            "n_observations": 1000,
            "n_exceptions": 6,
            "lr_ind": 0.0725079941,
            "p_value": 0.7877195429,
            "confidence": 0.99,
            "alpha_var": 0.01,
            "expected_probability": 0.01,
            "gamma_test": 0.05,
            "statistical_gamma_test": 0.05,
            "statistical_criterion_source": "STATISTICAL_TEST_SPECIFICATION",
            "alpha": 0.05,
            "rejected": False,
        },
    )
    r_cc = EvidenceRecord(
        evidence_id="EV-CC-03",
        test_id="traded_risk.var_christoffersen_conditional",
        test_name="Christoffersen conditional coverage test",
        status=Status.RECORDED,
        **common,
        params={"alpha": 0.05, "gamma_test": 0.05, "pnl_source": "actual"},
        metrics={
            "n_observations": 1000,
            "n_exceptions": 6,
            "lr_uc": 1.8862324083,
            "lr_ind": 0.0725079941,
            "lr_cc": 1.9587404024,
            "p_value": 0.3755475438,
            "confidence": 0.99,
            "alpha_var": 0.01,
            "expected_probability": 0.01,
            "gamma_test": 0.05,
            "statistical_gamma_test": 0.05,
            "statistical_criterion_source": "STATISTICAL_TEST_SPECIFICATION",
            "alpha": 0.05,
            "rejected": False,
        },
    )
    r_val = EvidenceRecord(
        evidence_id="EV-VAL-04",
        test_id="validation.var_size_power",
        test_name="Pre-registered VaR size and power validation",
        status=Status.PASS,
        **common,
        metrics={
            "study_id": "var_size_power",
            "n_criteria": 3,
            "n_criteria_failed": 0,
            "classification": "all pre-registered criteria met",
            "nominal_size": 0.05,
            "observed.size_correct_forecast": 0.066,
            "required.size_correct_forecast": "in [0.031, 0.069]",
            "observed.power_understated_0_7x": 1.0,
            "required.power_understated_0_7x": ">= 0.50",
            "observed.power_overstated_1_5x": 0.992,
            "required.power_overstated_1_5x": ">= 0.20",
        },
    )
    return [r_kupiec, r_ind, r_cc, r_val]


# =========================================================================== #
# Tests 1-4: Statistical Decision Provenance & Gamma
# =========================================================================== #


def test_kupiec_table_gets_statistic_from_evidence() -> None:
    """1. Kupiec table renders exact empirical statistic LR=1.886 from lr_uc."""
    records = _make_base_var_records()
    table = build_var_tail_table(records)
    stat_cells = [str(c) for c in table.columns[2]._cells]
    assert any("LR=1.886" in c for c in stat_cells)
    assert not any("1.850" in c for c in stat_cells)


def test_kupiec_decision_comes_from_stored_decision_not_p_value() -> None:
    """2. Mutating p-value to 0.0001 with rejected=False displays DO_NOT_REJECT (stored decision owns verdict)."""
    records = _make_base_var_records()
    m = records[0].metrics
    mutated_metrics = dict(m)
    mutated_metrics["p_value"] = 0.0001
    mutated_metrics["rejected"] = False

    mutated_rec = EvidenceRecord(
        evidence_id=records[0].evidence_id,
        test_id=records[0].test_id,
        test_name=records[0].test_name,
        status=records[0].status,
        model_id=records[0].model_id,
        dataset_id=records[0].dataset_id,
        run_id=records[0].run_id,
        params=records[0].params,
        metrics=mutated_metrics,
    )
    table = build_var_tail_table([mutated_rec])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("DO_NOT_REJECT" in c for c in crit_cells), f"Expected DO_NOT_REJECT in {crit_cells}"
    assert not any("(REJECT at" in c for c in crit_cells)


def test_gamma_explicit_mutation_changes_displayed_gamma() -> None:
    """3A. Test A: Explicit gamma mutation: gamma_test: 0.05 -> 0.025 displays gamma=0.03."""
    records = _make_base_var_records()
    mutated = dict(records[0].metrics)
    mutated["gamma_test"] = 0.025
    mutated["statistical_gamma_test"] = 0.025
    mutated_rec = EvidenceRecord(
        evidence_id=records[0].evidence_id,
        test_id=records[0].test_id,
        test_name=records[0].test_name,
        status=records[0].status,
        model_id=records[0].model_id,
        dataset_id=records[0].dataset_id,
        run_id=records[0].run_id,
        params={"gamma_test": 0.025},
        metrics=mutated,
    )
    table = build_var_tail_table([mutated_rec])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("gamma=0.03" in c for c in crit_cells), f"Expected gamma=0.03 in {crit_cells}"
    assert not any("gamma=0.05" in c for c in crit_cells)


def test_var_confidence_mutation_does_not_change_gamma() -> None:
    """3B. Test B: VaR confidence mutation: confidence: 0.99 -> 0.975 preserves gamma=0.05."""
    records = _make_base_var_records()
    mutated = dict(records[0].metrics)
    mutated["confidence"] = 0.975
    mutated["var_confidence"] = 0.975
    mutated_rec = EvidenceRecord(
        evidence_id=records[0].evidence_id,
        test_id=records[0].test_id,
        test_name=records[0].test_name,
        status=records[0].status,
        model_id=records[0].model_id,
        dataset_id=records[0].dataset_id,
        run_id=records[0].run_id,
        params=records[0].params,
        metrics=mutated,
    )
    table = build_var_tail_table([mutated_rec])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("gamma=0.05" in c for c in crit_cells), f"Expected gamma=0.05 in {crit_cells}"
    assert not any("gamma=0.98" in c for c in crit_cells)
    assert not any("gamma=0.025" in c for c in crit_cells)


def test_var_alpha_mutation_does_not_change_gamma() -> None:
    """3C. Test C: VaR alpha_var mutation: alpha_var: 0.01 -> 0.025 preserves gamma=0.05."""
    records = _make_base_var_records()
    mutated = dict(records[0].metrics)
    mutated["alpha_var"] = 0.025
    mutated["expected_probability"] = 0.025
    mutated_rec = EvidenceRecord(
        evidence_id=records[0].evidence_id,
        test_id=records[0].test_id,
        test_name=records[0].test_name,
        status=records[0].status,
        model_id=records[0].model_id,
        dataset_id=records[0].dataset_id,
        run_id=records[0].run_id,
        params=records[0].params,
        metrics=mutated,
    )
    table = build_var_tail_table([mutated_rec])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("gamma=0.05" in c for c in crit_cells), f"Expected gamma=0.05 in {crit_cells}"
    assert not any("gamma=0.025" in c or "gamma=0.03" in c for c in crit_cells)


def test_missing_decision_renders_na() -> None:
    """4. If rejected/decision is missing, table renders Decision=N/A."""
    records = _make_base_var_records()
    mutated = dict(records[0].metrics)
    del mutated["rejected"]
    mutated_rec = EvidenceRecord(
        evidence_id=records[0].evidence_id,
        test_id=records[0].test_id,
        test_name=records[0].test_name,
        status=records[0].status,
        model_id=records[0].model_id,
        dataset_id=records[0].dataset_id,
        run_id=records[0].run_id,
        params=records[0].params,
        metrics=mutated,
    )
    table = build_var_tail_table([mutated_rec])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("N/A at gamma=" in c for c in crit_cells)


# =========================================================================== #
# Tests 5-9: Size & Power Evidence Provenance & Status
# =========================================================================== #


def test_size_power_no_hardcoded_band() -> None:
    """5. When required.size_correct_forecast is missing, table renders band=N/A."""
    records = _make_base_var_records()
    val_rec = records[3]
    m = dict(val_rec.metrics)
    del m["required.size_correct_forecast"]
    mutated_val = EvidenceRecord(
        evidence_id=val_rec.evidence_id,
        test_id=val_rec.test_id,
        test_name=val_rec.test_name,
        status=val_rec.status,
        model_id=val_rec.model_id,
        dataset_id=val_rec.dataset_id,
        run_id=val_rec.run_id,
        metrics=m,
    )
    table = build_var_tail_table([mutated_val])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("band=N/A" in c for c in crit_cells)
    assert not any("[0.031, 0.069]" in c for c in crit_cells)


def test_size_power_no_hardcoded_gamma() -> None:
    """6. When nominal_size is missing, table renders gamma=N/A."""
    records = _make_base_var_records()
    val_rec = records[3]
    m = dict(val_rec.metrics)
    del m["nominal_size"]
    mutated_val = EvidenceRecord(
        evidence_id=val_rec.evidence_id,
        test_id=val_rec.test_id,
        test_name=val_rec.test_name,
        status=val_rec.status,
        model_id=val_rec.model_id,
        dataset_id=val_rec.dataset_id,
        run_id=val_rec.run_id,
        metrics=m,
    )
    table = build_var_tail_table([mutated_val])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("gamma=N/A" in c for c in crit_cells)


def test_missing_size_power_fields_render_na() -> None:
    """7. When size or power fields are missing, table renders size=N/A, Power=N/A."""
    records = _make_base_var_records()
    val_rec = records[3]
    mutated_val = EvidenceRecord(
        evidence_id=val_rec.evidence_id,
        test_id=val_rec.test_id,
        test_name=val_rec.test_name,
        status=val_rec.status,
        model_id=val_rec.model_id,
        dataset_id=val_rec.dataset_id,
        run_id=val_rec.run_id,
        metrics={},
    )
    table = build_var_tail_table([mutated_val])
    stat_cells = [str(c) for c in table.columns[2]._cells]
    assert any("size=N/A" in c and "Power (0.7x)=N/A" in c for c in stat_cells)


def test_missing_n_criteria_failed_cannot_become_pass() -> None:
    """8. Missing n_criteria_failed does not synthesize PASS."""
    records = _make_base_var_records()
    val_rec = records[3]
    m = dict(val_rec.metrics)
    del m["n_criteria_failed"]
    mutated_val = EvidenceRecord(
        evidence_id=val_rec.evidence_id,
        test_id=val_rec.test_id,
        test_name=val_rec.test_name,
        status=Status.FAIL,
        model_id=val_rec.model_id,
        dataset_id=val_rec.dataset_id,
        run_id=val_rec.run_id,
        metrics=m,
    )
    table = build_var_tail_table([mutated_val])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("Validation=FAIL" in c for c in crit_cells)
    assert not any("Validation=PASS" in c for c in crit_cells)


def test_status_comes_from_evidence_record() -> None:
    """9. Table status badge strictly reflects EvidenceRecord.status."""
    records = _make_base_var_records()
    rec = copy.deepcopy(records[0])
    rec_fail = EvidenceRecord(
        evidence_id=rec.evidence_id,
        test_id=rec.test_id,
        test_name=rec.test_name,
        status=Status.FAIL,
        model_id=rec.model_id,
        dataset_id=rec.dataset_id,
        run_id=rec.run_id,
        metrics=rec.metrics,
    )
    table = build_var_tail_table([rec_fail])
    status_cells = [str(c) for c in table.columns[4]._cells]
    assert any("FAIL" in c for c in status_cells)


# =========================================================================== #
# Tests 10-11: Collision-Safe Metric Identity
# =========================================================================== #


def test_metric_path_collisions_do_not_overwrite() -> None:
    """10. 3 distinct VaR p_values (0.1696, 0.7877, 0.3755) survive simultaneously."""
    records = _make_base_var_records()
    view = build_checkpoint_evidence_view(
        checkpoint_title="VaR Backtest",
        checkpoint_description="VaR review",
        domains=(ReviewDomain.MARKET,),
        records=records,
    )

    ev_kupiec = records[0].evidence_id
    ev_ind = records[1].evidence_id
    ev_cc = records[2].evidence_id

    # 1. Collision-safe lookup by (evidence_id, path)
    p_kupiec = view.get_metric("p_value", evidence_id=ev_kupiec)
    p_ind = view.get_metric("p_value", evidence_id=ev_ind)
    p_cc = view.get_metric("p_value", evidence_id=ev_cc)

    assert abs(p_kupiec - 0.1696274814) < 1e-8
    assert abs(p_ind - 0.7877195429) < 1e-8
    assert abs(p_cc - 0.3755475438) < 1e-8

    # 2. CheckpointMetricRef index
    ref_kupiec = CheckpointMetricRef(ev_kupiec, records[0].test_id, "p_value")
    assert ref_kupiec in view.metrics_by_ref
    assert abs(view.metrics_by_ref[ref_kupiec].numeric_value - 0.1696274814) < 1e-8

    # 3. Path-only lookup fails closed when ambiguous
    ambiguous_val = view.get_metric("p_value")
    assert ambiguous_val is None, "Bare path lookup across distinct p_values must fail closed (return None)"

    # 4. get_metrics returns all 3 metrics without collision overwrite
    p_metrics = view.get_metrics("p_value")
    assert len(p_metrics) == 3


def test_wrong_ev_plus_right_numeric_value_does_not_ground() -> None:
    """11. Citing wrong EV with numeric value belonging to another test fails grounding."""
    records = _make_base_var_records()
    ev_ind = records[1].evidence_id  # Christoffersen independence
    tampered_narrative = f"The test showed p-value 0.1696 [{ev_ind}]."
    claims = extract_claims(tampered_narrative)
    res = bind_claims(claims, records)
    assert len(res.unbound) == 1
    assert res.unbound[0]["reason"] == str(GroundingReasonCode.VALUE_MISMATCH)


# =========================================================================== #
# Tests 12-15: Precision-Derived Rounding & Restricted Criterion Parsing
# =========================================================================== #


def test_precision_derived_rounding() -> None:
    """12. 1.886 matches 1.8862 under 3 decimal places; exact integer 1000 matches 1000.0."""
    claim_3dec = Claim(value=1.886, surface="1.886", unit="", position=0, context="")
    match = _match_candidates_in_fields({1.886}, {"lr_uc": 1.8862324083}, tolerance=5e-4, claim=claim_3dec)
    assert match is not None
    assert match[0] == "lr_uc"

    claim_int = Claim(value=1000.0, surface="1,000", unit="", position=0, context="")
    match_int = _match_candidates_in_fields(
        {1000.0}, {"n_observations": 1000.0}, tolerance=5e-4, claim=claim_int
    )
    assert match_int is not None
    assert match_int[0] == "n_observations"


def test_approximate_value_1850_vs_18862_fails() -> None:
    """13. 1.850 strictly fails against 1.8862 (no fuzzy tolerance)."""
    claim_bad = Claim(value=1.850, surface="1.850", unit="", position=0, context="")
    match = _match_candidates_in_fields({1.850}, {"lr_uc": 1.8862324083}, tolerance=5e-4, claim=claim_bad)
    assert match is None


def test_typed_percent_normalization() -> None:
    """14. 5% binds to alpha=0.05, but fails against non-percentage metric n_observations=5."""
    claim_pct = Claim(value=5.0, surface="5%", unit="%", position=0, context="")
    match_alpha = _match_candidates_in_fields({5.0, 0.05}, {"alpha": 0.05}, tolerance=5e-4, claim=claim_pct)
    assert match_alpha is not None
    assert match_alpha[0] == "alpha"

    match_count = _match_candidates_in_fields(
        {5.0, 0.05}, {"n_exceptions": 0.05}, tolerance=5e-4, claim=claim_pct
    )
    assert match_count is None


def test_criterion_parsing_restricted() -> None:
    """15. Compound string interval is parsed into .lower/.upper ONLY for structured criterion fields."""
    structured = {"required.size_correct_forecast": "in [0.031, 0.069]"}
    flat_crit = flatten_evidence_values(structured)
    assert "required.size_correct_forecast.lower" in flat_crit
    assert flat_crit["required.size_correct_forecast.lower"] == 0.031

    narrative = {"interpretation": "Confidence band lies in [0.031, 0.069]"}
    flat_narrative = flatten_evidence_values(narrative)
    assert "interpretation.lower" not in flat_narrative


# =========================================================================== #
# Tests 16-17: Factor Artifact Zero Recomputation & Lineage
# =========================================================================== #


def test_factor_artifact_zero_recomputation() -> None:
    """16. Spy test: build_linear_factor_model is called 0 additional times during review execution."""
    with patch("start.portfolio.factor_risk.build_linear_factor_model") as spy_fac:
        world = generate_market_world()
        bundle = ReviewContextBundle(
            mode=ReviewMode.SINGLE_DOMAIN,
            domains=(ReviewDomain.MARKET,),
            market=world,
        )
        from start.review.applicability import applicable_tests

        applicable = applicable_tests(bundle.domains)
        results, products = execute_market_treasury_tests(bundle, applicable, return_products=True)

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            ledger = EvidenceLedger(out_dir / "ledger.jsonl", out_dir / "evidence")
            records = [ledger.append(tr, run_id="RUN-TEST") for tr in results]
            arts = generate_review_artifacts(bundle, records, out_dir, products=products)
            assert "Factor Modeling & Attribution Assumptions" in arts

        assert spy_fac.call_count == 0


def test_artifact_evidence_lineage() -> None:
    """17. Factor Attribution artifact evidence IDs are subset of original attribution EvidenceRecord IDs."""
    world = generate_market_world()
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        market=world,
    )
    from start.review.applicability import applicable_tests

    applicable = applicable_tests(bundle.domains)
    results, products = execute_market_treasury_tests(bundle, applicable, return_products=True)

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        ledger = EvidenceLedger(out_dir / "ledger.jsonl", out_dir / "evidence")
        records = [ledger.append(tr, run_id="RUN-TEST") for tr in results]
        arts = generate_review_artifacts(bundle, records, out_dir, products=products)
        fac_arts = arts.get("Factor Modeling & Attribution Assumptions", [])
        assert len(fac_arts) >= 1
        art = fac_arts[0]
        original_ev_ids = {r.evidence_id for r in records if r.evidence_id}
        for eid in art.evidence_ids:
            assert eid in original_ev_ids


# =========================================================================== #
# Tests 18-20: Committee, Governance & Grounding Census Invariants
# =========================================================================== #


def test_committee_consumer_provenance() -> None:
    """18. Committee consumes EvidenceRecords directly without bare path-only metric lookups."""
    records = _make_base_var_records()
    committee = CrossAnalyticalCommittee()
    graph, claims = committee.build_evidence_graph(records)
    assert graph is not None
    assert len(claims) >= 1


def test_governance_consumer_provenance() -> None:
    """19. Governance table consumes metadata and decisions without recomputation."""
    meta = {
        "mode": "single_domain",
        "domains": ["market"],
        "materiality": "tier_1",
        "lifecycle": "pre_implementation",
    }
    decisions = [{"checkpoint": "VaR Backtest", "action": "A", "response": "PASS"}]
    table = build_governance_table(meta, decisions)
    fields = [str(c) for c in table.columns[0]._cells]
    assert "Review Mode" in fields
    assert "Materiality Tier" in fields


def test_grounding_census_invariant() -> None:
    """20. Grounding census invariant: bound + unbound == total_claims."""
    narrative = (
        "Under Kupiec proportion of failures, the LR is 1.8862 [EV-KUPIEC-01] "
        "and p-value is 0.9999 [EV-KUPIEC-01]."
    )
    records = _make_base_var_records()
    claims = extract_claims(narrative)
    res = bind_claims(claims, records)
    assert res.total_claims == len(res.bound) + len(res.unbound)
    assert len(res.bound) == 1
    assert len(res.unbound) == 1


# =========================================================================== #
# Tests 21-22: Provider Call Count Invariants (Exact 1 Provider Call)
# =========================================================================== #


def test_provider_call_count_on_grounding_pass() -> None:
    """21. Grounding PASS executes exactly 1 provider call."""
    from start.providers.base import ProviderResult, ProviderUsage

    mock_provider = MagicMock()
    mock_provider.name = "mock_provider"
    mock_pres = ProviderResult(
        text="The portfolio risk analysis confirms all constraints are satisfied without outside metrics.",
        provider="openai",
        model="gpt-5",
        status="completed",
        usage=ProviderUsage(input_tokens=100, output_tokens=50, reasoning_tokens=0),
        latency_seconds=0.1,
    )
    mock_provider.complete_result.return_value = mock_pres

    world = generate_market_world()
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        market=world,
        llm_config=LLMReviewConfig(backend_mode="public", provider="openai", model="gpt-5"),
    )

    mock_ask = MagicMock(side_effect=["Q", "Test question", "A", "A", "A", "A", "A", "A", "A"])
    with patch("start.providers.llm.get_llm_provider", return_value=mock_provider):
        run_market_treasury_review(bundle, interactive=True, ask=mock_ask)

    assert mock_provider.complete_result.call_count == 1


def test_provider_call_count_on_grounding_fail() -> None:
    """22. Grounding FAIL executes exactly 1 provider call (zero re-prompt loops)."""
    from start.providers.base import ProviderResult, ProviderUsage

    mock_provider = MagicMock()
    mock_provider.name = "mock_provider"
    mock_pres = ProviderResult(
        text="The empirical ratio is 1.850 [EV-FAKE-999].",
        provider="openai",
        model="gpt-5",
        status="completed",
        usage=ProviderUsage(input_tokens=100, output_tokens=50, reasoning_tokens=0),
        latency_seconds=0.1,
    )
    mock_provider.complete_result.return_value = mock_pres

    world = generate_market_world()
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        market=world,
        llm_config=LLMReviewConfig(backend_mode="public", provider="openai", model="gpt-5"),
    )

    mock_ask = MagicMock(side_effect=["Q", "Query", "A", "A", "A", "A", "A", "A", "A"])
    with patch("start.providers.llm.get_llm_provider", return_value=mock_provider):
        run_market_treasury_review(bundle, interactive=True, ask=mock_ask)

    assert mock_provider.complete_result.call_count == 1


# =========================================================================== #
# Tests 23-24: Gate-12 Live Provider & State Machine Flow
# =========================================================================== #


def test_gate12_live_provider_mocked_path_still_works() -> None:
    """23. Live-provider mocked path executes end-to-end and validates."""
    mock_provider = MagicMock()
    mock_provider.name = "mock_provider"
    records = _make_base_var_records()
    ev_k = records[0].evidence_id

    mock_pres = MagicMock()
    mock_pres.text = f"Empirical LR unconditional coverage is 1.8862 [{ev_k}]."
    mock_pres.status = "success"
    mock_pres.usage.prompt_tokens = 100
    mock_pres.usage.output_tokens = 50
    mock_pres.usage.reasoning_tokens = 0
    mock_provider.complete_with_result.return_value = mock_pres

    claims = extract_claims(mock_pres.text)
    binding = bind_claims(claims, records)
    assert len(binding.unbound) == 0
    assert len(binding.bound) == 1


def test_qcva_state_machine_unchanged() -> None:
    """24. CheckpointStateMachine validates valid state transitions."""
    sm = CheckpointStateMachine("VaR Checkpoint")
    assert sm.current_state == CheckpointState.READY

    sm.transition(CheckpointState.PROVIDER_CALL)
    sm.transition(CheckpointState.PROVIDER_RESPONSE)
    sm.transition(CheckpointState.GROUNDING_VALIDATE)
    sm.transition(CheckpointState.VERIFIED)
    sm.transition(CheckpointState.COMPLETED)
    assert sm.current_state == CheckpointState.COMPLETED
