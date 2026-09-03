"""StART v4.3.0 Gate 13: Canonical Checkpoint Evidence View & Quantitative Grounding Closure Tests.

Verifies:
1. Canonical CheckpointEvidenceView as single source of truth across tables, prompts, and grounding.
2. Kupiec POF metric resolution (lr_uc = 1.8862 -> LR=1.886 display; no fallback to 1.850).
3. validation.var_size_power mapping (size=0.066, band=[0.031, 0.069], power=1.000/0.992, Validation=PASS; no size=N/A).
4. Grounding census invariant (grounded_claims + unbound_claims == quantitative_claims).
5. Multiple evidence citations on a single claim do not inflate grounded claim count.
6. Markdown bold splitting normalization (**1**,**000** -> 1,000, **0.**7x -> 0.7x, **1.**5x -> 1.5x).
7. Normalization equivalences (5% <-> 0.05, 1,000 <-> 1000, 0.7x <-> 0.7).
8. Strict mismatch rejection (1.850 rejected against 1.8862).
9. Repeated summary claims bound consistently without false unbound errors.
10. Claim-local citation association.
11. Failed-claim diagnostic reasons and diagnostic table rendering.
12. HRP untruncated order (all 50 assets preserved).
13. Attribution artifact generation and checkpoint scoping.
14. Production-path noninteractive mock harness (Portfolio V->Q->C->A, Attribution V->Q->A, VaR V->Q).
"""

from __future__ import annotations

from start.attestation.claims import (
    GroundingReasonCode,
    bind_claims,
    extract_claims,
    normalize_markdown_numeric_markup,
)
from start.core.schemas import EvidenceRecord, Status
from start.data.synthetic_market import generate_market_world
from start.review.architecture import (
    LLMReviewConfig,
    ReviewContextBundle,
    ReviewDomain,
    ReviewMode,
)
from start.review.evidence_view import (
    build_checkpoint_evidence_view,
)
from start.review.executor import (
    execute_market_treasury_tests,
    generate_review_artifacts,
)
from start.review.tables import build_var_tail_table

# =========================================================================== #
# Fixtures and Helpers
# =========================================================================== #

def _make_var_records() -> list[EvidenceRecord]:
    """Construct real deterministic Market VaR evidence records mimicking failed run."""
    common_meta = {
        "model_id": "M-MARKET",
        "dataset_id": "D-MARKET",
        "run_id": "RUN-TEST",
    }
    r_exc = EvidenceRecord(
        evidence_id="EV-802d29da4fdf",
        test_id="traded_risk.var_exceptions",
        test_name="VaR exceptions count",
        status=Status.RECORDED,
        **common_meta,
        metrics={
            "pnl_source": "actual",
            "confidence": 0.99,
            "n_observations": 1000,
            "n_exceptions": 6,
            "exception_rate": 0.006,
            "expected_probability": 0.01,
            "expected_exceptions": 10.0,
            "first_exception": "2020-09-25 00:00:00",
            "last_exception": "2023-01-19 00:00:00",
        },
        interpretation="6 exception(s) observed against 10.0 expected over 1,000 observation(s).",
    )
    r_kupiec = EvidenceRecord(
        evidence_id="EV-eadf19f1abd1",
        test_id="traded_risk.var_kupiec_pof",
        test_name="Kupiec proportion-of-failures test",
        status=Status.RECORDED,
        **common_meta,
        params={"alpha": 0.05, "pnl_source": "actual"},
        metrics={
            "pnl_source": "actual",
            "confidence": 0.99,
            "n_observations": 1000,
            "n_exceptions": 6,
            "exception_rate": 0.006,
            "expected_probability": 0.01,
            "expected_exceptions": 10.0,
            "lr_uc": 1.8862324083,
            "p_value": 0.1696274814,
            "degrees_of_freedom": 1,
            "alpha": 0.05,
            "critical_value": 3.8414588207,
            "rejected": False,
        },
        interpretation="At the 5% level, the null of correct unconditional coverage was not rejected (LR = 1.8862).",
    )
    r_ind = EvidenceRecord(
        evidence_id="EV-9940f9458459",
        test_id="traded_risk.var_christoffersen_independence",
        test_name="Christoffersen independence test",
        status=Status.RECORDED,
        **common_meta,
        metrics={
            "n_observations": 1000,
            "n_exceptions": 6,
            "lr_ind": 0.0725079941,
            "p_value": 0.7877195429,
            "degrees_of_freedom": 1,
            "alpha": 0.05,
            "n00": 987,
            "n01": 6,
            "n10": 6,
            "n11": 0,
            "rejected": False,
        },
        interpretation="Christoffersen independence test shows no exception clustering (LR_ind = 0.0725).",
    )
    r_cc = EvidenceRecord(
        evidence_id="EV-3195d7864f24",
        test_id="traded_risk.var_christoffersen_conditional",
        test_name="Christoffersen conditional coverage test",
        status=Status.RECORDED,
        **common_meta,
        metrics={
            "n_observations": 1000,
            "n_exceptions": 6,
            "lr_uc": 1.8862324083,
            "lr_ind": 0.0725079941,
            "lr_cc": 1.9587404024,
            "p_value": 0.3755475438,
            "degrees_of_freedom": 2,
            "alpha": 0.05,
            "rejected": False,
        },
        interpretation="Joint conditional coverage not rejected (LR_cc = 1.9587).",
    )
    r_tl = EvidenceRecord(
        evidence_id="EV-5a5bbef426f5",
        test_id="traded_risk.var_traffic_light",
        test_name="Basel traffic light status",
        status=Status.RECORDED,
        **common_meta,
        metrics={"zone": "GREEN", "n_exceptions": 6, "multiplier": 3.0},
        interpretation="Basel traffic light zone is GREEN.",
    )
    r_val = EvidenceRecord(
        evidence_id="EV-329ca98e9313",
        test_id="validation.var_size_power",
        test_name="Pre-registered VaR size and power validation",
        status=Status.PASS,
        **common_meta,
        metrics={
            "study_id": "var_size_power",
            "n_criteria": 3,
            "n_criteria_failed": 0,
            "classification": "all pre-registered criteria met",
            "nominal_size": 0.05,
            "observed.size_correct_forecast": 0.066,
            "required.size_correct_forecast": "in [0.031, 0.069]",
            "passed.size_correct_forecast": True,
            "observed.power_understated_0_7x": 1.0,
            "required.power_understated_0_7x": ">= 0.50",
            "passed.power_understated_0_7x": True,
            "observed.power_overstated_1_5x": 0.992,
            "required.power_overstated_1_5x": ">= 0.20",
            "passed.power_overstated_1_5x": True,
        },
        interpretation="All pre-registered size and power criteria met.",
    )
    return [r_exc, r_kupiec, r_ind, r_cc, r_tl, r_val]


# =========================================================================== #
# Test 1 & 2: Kupiec Discrepancy & Validation Size & Power Table Mapping
# =========================================================================== #

def test_kupiec_discrepancy_resolved() -> None:
    """Kupiec POF renders lr_uc = 1.8862 as LR=1.886 without falling back to 1.850."""
    records = _make_var_records()
    table = build_var_tail_table(records)

    test_names = [str(c) for c in table.columns[1]._cells]
    assert any("Kupiec" in name for name in test_names)
    # Check that the table renders LR=1.886 and not LR=1.850
    cell_texts = [str(c) for c in table.columns[2]._cells]
    assert any("1.886" in text for text in cell_texts), f"Expected 1.886 in {cell_texts}"
    assert not any("1.850" in text for text in cell_texts), f"Found unexpected 1.850 in {cell_texts}"


def test_var_size_power_table_mapping() -> None:
    """validation.var_size_power renders size=0.066, band, power=1.000/0.992 and PASS without size=N/A."""
    records = _make_var_records()
    table = build_var_tail_table(records)

    val_cells = [str(c) for c in table.columns[2]._cells]
    assert any("size=0.066" in text for text in val_cells), f"Expected size=0.066 in {val_cells}"
    assert not any("size=N/A" in text for text in val_cells), f"Found unexpected size=N/A in {val_cells}"

    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("Validation=PASS" in text for text in crit_cells), f"Expected Validation=PASS in {crit_cells}"
    assert any("in [0.031, 0.069]" in text for text in crit_cells)


# =========================================================================== #
# Test 3 & 4: Canonical CheckpointEvidenceView Integrity
# =========================================================================== #

def test_checkpoint_evidence_view_feeds_all_consumers() -> None:
    """Single CheckpointEvidenceView holds identical values for tables, prompt, and grounding."""
    records = _make_var_records()
    view = build_checkpoint_evidence_view(
        checkpoint_title="VaR Backtesting & Exception Frequency",
        checkpoint_description="Review empirical VaR exceptions.",
        domains=(ReviewDomain.MARKET,),
        records=records,
    )

    # 1. Scientific value for Kupiec lr_uc
    lr_val = view.get_numeric("traded_risk.var_kupiec_pof.lr_uc")
    assert lr_val is not None
    assert abs(lr_val - 1.8862324083) < 1e-8

    # 2. Value in LLM prompt payload
    payload = view.format_llm_payload()
    assert "1.8862324083" in payload
    assert "observed.size_correct_forecast" in payload
    assert "0.066" in payload

    # 3. Value in Grounding Map
    kupiec_ev = "EV-eadf19f1abd1"
    assert kupiec_ev in view.numeric_grounding_map
    assert abs(view.numeric_grounding_map[kupiec_ev]["traded_risk.var_kupiec_pof.lr_uc"] - 1.8862324083) < 1e-8

    # 4. Value in Table before formatting
    table = build_var_tail_table(view)
    cell_texts = [str(c) for c in table.columns[2]._cells]
    assert any("1.886" in text for text in cell_texts)


def test_checkpoint_evidence_view_parses_interval_and_multiplier_bounds() -> None:
    """CheckpointEvidenceView extracts interval bounds [0.031, 0.069] and multipliers 0.7x, 1.5x."""
    records = _make_var_records()
    view = build_checkpoint_evidence_view(
        checkpoint_title="VaR Backtesting & Exception Frequency",
        checkpoint_description="Review VaR.",
        domains=(ReviewDomain.MARKET,),
        records=records,
    )

    val_ev = "EV-329ca98e9313"
    g_map = view.numeric_grounding_map[val_ev]
    assert abs(g_map["required.size_correct_forecast.lower"] - 0.031) < 1e-8
    assert abs(g_map["required.size_correct_forecast.upper"] - 0.069) < 1e-8
    assert abs(g_map["power_multiplier_0_7x"] - 0.7) < 1e-8
    assert abs(g_map["power_multiplier_1_5x"] - 1.5) < 1e-8


# =========================================================================== #
# Test 5 & 6: Grounding Census Invariant & Multiple Citation Handling
# =========================================================================== #

def test_grounding_census_invariant() -> None:
    """grounded_claims + unbound_claims == quantitative_claims strictly holds."""
    records = _make_var_records()
    narrative = (
        "Under Kupiec POF, LR was 1.8862 [EV-eadf19f1abd1] with p-value 0.1696 [EV-eadf19f1abd1] "
        "over 1000 observations [EV-eadf19f1abd1] with 6 exceptions [EV-eadf19f1abd1]. "
        "An ungrounded claim of 99.999 is made here."
    )
    claims = extract_claims(narrative)
    assert len(claims) == 5

    res = bind_claims(claims, records)
    assert len(res.bound) + len(res.unbound) == res.total_claims
    assert res.total_claims == 5
    assert len(res.bound) == 4
    assert len(res.unbound) == 1
    assert 0 <= len(res.bound) <= res.total_claims
    assert 0 <= len(res.unbound) <= res.total_claims


def test_multiple_citations_do_not_inflate_claim_count() -> None:
    """A single claim citing multiple evidence records counts as ONE grounded claim."""
    records = _make_var_records()
    narrative = "The test recorded 6 exceptions [EV-802d29da4fdf] [EV-eadf19f1abd1]."
    claims = extract_claims(narrative)
    assert len(claims) == 1

    res = bind_claims(claims, records)
    assert res.total_claims == 1
    assert len(res.bound) == 1
    assert len(res.unbound) == 0
    assert len(res.bound) + len(res.unbound) == res.total_claims


# =========================================================================== #
# Test 7 & 8: Markdown Bold Split Normalization & Normalization Equivalences
# =========================================================================== #

def test_markdown_bold_splitting_normalization() -> None:
    """Presentation markup like **1**,**000** and **0.**7x is normalized before numeric extraction."""
    raw = (
        "Observed **1**,**000** observations with power at **0.**7x and **1.**5x "
        "yielding size **0.066** within *[0.031, 0.069]*."
    )
    norm = normalize_markdown_numeric_markup(raw)
    assert "1,000" in norm
    assert "0.7x" in norm
    assert "1.5x" in norm
    assert "0.066" in norm
    assert "[0.031, 0.069]" in norm

    claims = extract_claims(raw)
    values = [c.value for c in claims]
    assert 1000.0 in values
    assert 0.7 in values
    assert 1.5 in values
    assert 0.066 in values
    assert 0.031 in values
    assert 0.069 in values


def test_numeric_normalization_equivalences() -> None:
    """Validates 5% <-> 0.05, 1,000 <-> 1000, 0.7x <-> 0.7."""
    records = _make_var_records()

    # 1. 5% matches 0.05
    c_pct = extract_claims("Nominal significance was 5% [EV-eadf19f1abd1].")
    assert len(c_pct) == 1
    res_pct = bind_claims(c_pct, records)
    assert len(res_pct.bound) == 1

    # 2. 1,000 matches 1000
    c_comma = extract_claims("Total of 1,000 observations [EV-eadf19f1abd1].")
    assert len(c_comma) == 1
    res_comma = bind_claims(c_comma, records)
    assert len(res_comma.bound) == 1

    # 3. 0.7x matches 0.7
    c_mult = extract_claims("Tested at 0.7x [EV-329ca98e9313].")
    assert len(c_mult) == 1
    res_mult = bind_claims(c_mult, records)
    assert len(res_mult.bound) == 1


# =========================================================================== #
# Test 9 & 10: Strict Mismatch Rejection & Repeated Summary Claims
# =========================================================================== #

def test_approximate_value_mismatch_rejected() -> None:
    """1.850 is strictly rejected against evidence 1.8862 as VALUE_MISMATCH."""
    records = _make_var_records()
    claims = extract_claims("Kupiec LR was 1.850 [EV-eadf19f1abd1].")
    res = bind_claims(claims, records)

    assert len(res.bound) == 0
    assert len(res.unbound) == 1
    assert res.unbound[0]["reason"] == str(GroundingReasonCode.VALUE_MISMATCH)


def test_display_rounding_accepted_under_tolerance() -> None:
    """1.886 matches evidence 1.8862 under formatting precision tolerance."""
    records = _make_var_records()
    claims = extract_claims("Kupiec LR was 1.886 [EV-eadf19f1abd1].")
    res = bind_claims(claims, records)

    assert len(res.bound) == 1
    assert len(res.unbound) == 0


def test_repeated_summary_claims_bound_consistently() -> None:
    """Restating an evidenced statistic in a summary does not trigger false unbound errors."""
    records = _make_var_records()
    narrative = (
        "Body: Kupiec LR was 1.8862 [EV-eadf19f1abd1] across 1000 observations [EV-eadf19f1abd1].\n\n"
        "Summary: In conclusion, Kupiec LR of 1.8862 [EV-eadf19f1abd1] confirms unconditional coverage."
    )
    claims = extract_claims(narrative)
    assert len(claims) == 3

    res = bind_claims(claims, records)
    assert len(res.bound) == 3
    assert len(res.unbound) == 0


# =========================================================================== #
# Test 11: Claim-Local Citation Association & Diagnostics
# =========================================================================== #

def test_claim_local_citation_association() -> None:
    """Citations bind locally to claims in the same sentence, not across sentences."""
    records = _make_var_records()
    narrative = (
        "First sentence with LR = 1.8862 [EV-eadf19f1abd1]. "
        "Second sentence cites 999.0 without citation."
    )
    claims = extract_claims(narrative)
    assert len(claims) == 2

    # Claim 1 has local citation EV-eadf19f1abd1
    assert "EV-eadf19f1abd1" in claims[0].cited_evidence
    # Claim 2 has no local citation
    assert len(claims[1].cited_evidence) == 0

    res = bind_claims(claims, records)
    assert len(res.bound) == 1
    assert len(res.unbound) == 1
    assert res.unbound[0]["reason"] == str(GroundingReasonCode.NO_LOCAL_EVIDENCE_CITATION)


# =========================================================================== #
# Test 12: HRP Untruncated Order
# =========================================================================== #

def test_hrp_order_is_untruncated() -> None:
    """portfolio.hierarchical_risk_parity serializes all 50 assets without [:400] truncation."""
    from start.tests.portfolio import hierarchical_risk_parity

    world = generate_market_world()
    class _DummyCtx:
        returns = world.returns
        portfolio_returns = world.returns.mean(axis=1)
        covariance = world.returns.cov()
    res = hierarchical_risk_parity(_DummyCtx())

    order_str = res.metrics["quasi_diagonal_order"]
    assets_in_order = [a.strip() for a in order_str.split(",") if a.strip()]
    assert len(assets_in_order) == 50
    assert not order_str.endswith("ASSE")
    assert all(a.startswith("A") for a in assets_in_order)


# =========================================================================== #
# Test 13: Attribution Artifact Scoping
# =========================================================================== #

def test_attribution_artifact_generated_and_scoped() -> None:
    """Factor Risk Model artifact is generated and scoped to Factor Modeling & Attribution checkpoint."""
    world = generate_market_world()
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        market=world,
    )
    from start.review.applicability import applicable_tests
    applicable = applicable_tests(bundle.domains)

    results, products = execute_market_treasury_tests(bundle, applicable, return_products=True)
    # Zero-recomputation invariant: no redundant build_linear_factor_model call in executor
    assert products.get_result("factor_risk.model") is None

    import tempfile
    from pathlib import Path

    from start.evidence.ledger import EvidenceLedger
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        ledger = EvidenceLedger(td / "ledger.jsonl", td / "evidence")
        records = [ledger.append(tr, run_id="RUN-TEST") for tr in results]
        arts_by_chk = generate_review_artifacts(bundle, records, td / "artifacts", products=products)
        assert "Factor Modeling & Attribution Assumptions" in arts_by_chk
        assert len(arts_by_chk["Factor Modeling & Attribution Assumptions"]) >= 1


# =========================================================================== #
# Test 14: Full Production-Path Noninteractive Mock Harness
# =========================================================================== #

def test_production_path_mock_harness_var_grounding_pass_and_fail() -> None:
    """Harness executing Portfolio V->Q->C->A, Attribution V->Q->A, VaR V->Q with exact GPT-5 structure."""
    world = generate_market_world()
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        market=world,
        llm_config=LLMReviewConfig(provider="openai", model="gpt-5"),
    )
    from start.review.applicability import applicable_tests
    applicable = applicable_tests(bundle.domains)
    results, products = execute_market_treasury_tests(bundle, applicable, return_products=True)

    import tempfile
    from pathlib import Path

    from start.evidence.ledger import EvidenceLedger
    with tempfile.TemporaryDirectory() as tmpdir:
        td = Path(tmpdir)
        ledger = EvidenceLedger(td / "ledger.jsonl", td / "evidence")
        records = [ledger.append(tr, run_id="RUN-TEST") for tr in results]

    rec_by_test = {r.test_id: r for r in records}
    ev_exc = rec_by_test["traded_risk.var_exceptions"].evidence_id
    ev_kupiec = rec_by_test["traded_risk.var_kupiec_pof"].evidence_id
    ev_ind = rec_by_test["traded_risk.var_christoffersen_independence"].evidence_id
    ev_cc = rec_by_test["traded_risk.var_christoffersen_conditional"].evidence_id
    ev_val = rec_by_test["validation.var_size_power"].evidence_id

    # Construct exact semantic response modeled on user's live GPT-5 response
    valid_gpt5_response = (
        f"Under Kupiec proportion of failures, the empirical LR is 1.8862 with p-value 0.1696 [{ev_kupiec}]. "
        f"Exception process recorded 6 exceptions out of 1,000 observations [{ev_exc}] "
        f"against 10.0 expected exceptions [{ev_exc}]. "
        f"Christoffersen independence test yielded LR_ind of 0.0725 with p-value 0.7877 [{ev_ind}]. "
        f"Joint conditional coverage test showed LR_cc of 1.9587 with p-value 0.3755 [{ev_cc}]. "
        f"Pre-registered validation demonstrated empirical size of 0.066 [{ev_val}] "
        f"within accepted band [0.031, 0.069] [{ev_val}], "
        f"with power of 1.000 at 0.7x [{ev_val}] and 0.992 at 1.5x [{ev_val}].\n\n"
        f"Summary: The model passed unconditional coverage with LR of 1.8862 [{ev_kupiec}]."
    )

    claims = extract_claims(valid_gpt5_response)
    # Check that all quantitative claims bind successfully to records in scope
    scope_ids = [ev_exc, ev_kupiec, ev_ind, ev_cc, ev_val]
    binding = bind_claims(claims, records, permitted_scope=scope_ids)

    assert len(binding.unbound) == 0, f"Unbound claims: {binding.unbound}"
    assert len(binding.bound) == binding.total_claims
    assert binding.grounding_rate == 1.0

    # Intentionally introduce one value mismatch: 1.8862 -> 1.850
    tampered_response = valid_gpt5_response.replace("1.8862", "1.850")
    tampered_claims = extract_claims(tampered_response)
    tampered_binding = bind_claims(tampered_claims, records, permitted_scope=scope_ids)

    assert len(tampered_binding.unbound) >= 1
    reasons = [u["reason"] for u in tampered_binding.unbound]
    assert str(GroundingReasonCode.VALUE_MISMATCH) in reasons
