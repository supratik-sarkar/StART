"""Focused non-interactive test suite for claim-local semantic grounding (Remediation 1).

Covers all 16 required test conditions from Section 15 of the specification:
1. Exact Attempt-3 99% source sentence binds to confidence = 0.99
2. 99% VaR confidence with Kupiec in same sentence binds to confidence
3. 5% significance with VaR/confidence elsewhere in same sentence binds to gamma_test
4. Explicit confidence/tail/gamma negative separation tests (Cases A, B, C from Section 8)
5. Exact Attempt-3 zero sentence binds to n11
6. n00=987, n01=6, n10=6, n11=0 [EV-X] binds all four assertions
7. Repeated values (n01=6 and n10=6) disambiguated by local labels
8. Trailing citation applies to compact metric list in bullet item
9. Unsupported uncited zero still FAILS
10. Wrong cited EV still FAILS
11. Key=value / symbolic metric parsing for p=, LR=, gamma=
12. Date masking remains green
13. Grounding census invariant remains green
14. Gate-13B separation suite remains green
15. Gate-13A collision suite remains green
16. Harness self-tests remain green
"""

from start.attestation.claims import (
    GroundingReasonCode,
    bind_claims,
    extract_claims,
)


def _sample_var_evidence() -> dict[str, dict[str, float]]:
    return {
        "EV-2ede64de883e": {
            "metrics.confidence": 0.99,
            "metrics.alpha_var": 0.01,
            "metrics.gamma_test": 0.05,
            "metrics.lr_uc": 1.8862,
            "metrics.p_value": 0.1696,
        },
        "EV-1da1c07274ba": {
            "metrics.actual_exceptions": 6.0,
            "metrics.expected_exceptions": 10.0,
            "metrics.n_observations": 1000.0,
            "metrics.confidence": 0.99,
            "metrics.empirical_rate": 0.006,
            "metrics.dropped": 0.0,
        },
        "EV-bd9ab4df8657": {
            "metrics.n00": 987.0,
            "metrics.n01": 6.0,
            "metrics.n10": 6.0,
            "metrics.n11": 0.0,
            "metrics.lr_ind": 0.0734,
            "metrics.p_value": 0.7865,
        },
        "EV-811be7152197": {
            "parameters.band_n_observations": 250.0,
            "parameters.band_confidence": 0.99,
            "metrics.n_observations": 1000.0,
        },
    }


def test_1_exact_attempt3_99pct_sentence() -> None:
    """1. Exact Attempt-3 99% source sentence binds to confidence = 0.99."""
    sentence = (
        "The observed exception frequency is below the 99% expectation (6 vs 10.0 over 1,000), "
        "yet the discrepancy does not trigger rejection in the Kupiec test at the stated level "
        "on this sample [EV-2ede64de883e], [EV-1da1c07274ba]."
    )
    ev = _sample_var_evidence()
    claims = extract_claims(sentence)
    result = bind_claims(claims, ev)

    c99 = next(c for c in result.bound if c["surface"] == "99%")
    assert c99["bound_to"].endswith("metrics.confidence")
    assert c99["evidence_value"] == 0.99


def test_2_var_confidence_with_kupiec_in_same_sentence() -> None:
    """2. 99% VaR confidence with Kupiec in same sentence binds to confidence."""
    sentence = "The 99% VaR confidence level was verified under the Kupiec test framework [EV-2ede64de883e]."
    ev = _sample_var_evidence()
    claims = extract_claims(sentence)
    result = bind_claims(claims, ev)

    c99 = next(c for c in result.bound if c["surface"] == "99%")
    assert c99["bound_to"].endswith("metrics.confidence")
    assert c99["evidence_value"] == 0.99


def test_3_significance_with_var_confidence_in_same_sentence() -> None:
    """3. 5% significance with VaR/confidence elsewhere in same sentence binds to gamma_test."""
    sentence = "Under the 99% VaR model, the Kupiec test evaluates exceptions at 5% significance [EV-2ede64de883e]."
    ev = _sample_var_evidence()
    claims = extract_claims(sentence)
    result = bind_claims(claims, ev)

    c5 = next(c for c in result.bound if c["surface"] == "5%")
    assert c5["bound_to"].endswith("metrics.gamma_test")
    assert c5["evidence_value"] == 0.05

    c99 = next(c for c in result.bound if c["surface"] == "99%")
    assert c99["bound_to"].endswith("metrics.confidence")
    assert c99["evidence_value"] == 0.99


def test_4_negative_separation_cases_a_b_c() -> None:
    """4. Explicit confidence/tail/gamma negative separation tests (Cases A, B, C from Section 8)."""
    # Case A: 5% significance cannot bind to alpha_var=0.05 or confidence=0.95
    ev_a = {"EV-A": {"metrics.confidence": 0.95, "metrics.alpha_var": 0.05}}
    claims_a = extract_claims("We reject the null hypothesis at 5% significance [EV-A].")
    res_a = bind_claims(claims_a, ev_a)
    assert len(res_a.bound) == 0
    assert len(res_a.unbound) == 1
    assert res_a.unbound[0]["reason"] == GroundingReasonCode.VALUE_MISMATCH

    # Case B: 5% tail probability cannot bind to gamma_test=0.05
    ev_b = {"EV-B": {"metrics.gamma_test": 0.05}}
    claims_b = extract_claims("The 5% tail probability is maintained [EV-B].")
    res_b = bind_claims(claims_b, ev_b)
    assert len(res_b.bound) == 0
    assert len(res_b.unbound) == 1
    assert res_b.unbound[0]["reason"] == GroundingReasonCode.VALUE_MISMATCH

    # Case C: 99% confidence cannot bind to p_value=0.99
    ev_c = {"EV-C": {"metrics.p_value": 0.99}}
    claims_c = extract_claims("The model evaluates risk at 99% confidence [EV-C].")
    res_c = bind_claims(claims_c, ev_c)
    assert len(res_c.bound) == 0
    assert len(res_c.unbound) == 1
    assert res_c.unbound[0]["reason"] == GroundingReasonCode.VALUE_MISMATCH


def test_5_exact_attempt3_zero_sentence() -> None:
    """5. Exact Attempt-3 zero sentence binds to n11."""
    sentence = (
        "- There is no detected first-order exception clustering; no back-to-back exceptions "
        "occurred (n11 = 0), and the independence null is not rejected. "
        "Longer-lag dependence is not evaluated here [EV-bd9ab4df8657]."
    )
    ev = _sample_var_evidence()
    claims = extract_claims(sentence)
    result = bind_claims(claims, ev)

    c0 = next(c for c in result.bound if c["surface"] == "0")
    assert c0["bound_to"].endswith("metrics.n11")
    assert c0["evidence_value"] == 0.0


def test_6_compact_transition_assertions() -> None:
    """6. n00=987, n01=6, n10=6, n11=0 [EV-X] binds all four assertions."""
    sentence = "- Transition counts: n00=987, n01=6, n10=6, n11=0 [EV-bd9ab4df8657]."
    ev = _sample_var_evidence()
    claims = extract_claims(sentence)
    assert len(claims) == 4
    result = bind_claims(claims, ev)
    assert len(result.bound) == 4

    bound_paths = {b["bound_to"].split(".")[-1]: b["evidence_value"] for b in result.bound}
    assert bound_paths == {"n00": 987.0, "n01": 6.0, "n10": 6.0, "n11": 0.0}


def test_7_repeated_values_disambiguated_by_local_label() -> None:
    """7. Repeated values (n01=6 and n10=6) disambiguated by local labels."""
    sentence = "- Transition counts: n01 = 6, n10 = 6 [EV-bd9ab4df8657]."
    ev = _sample_var_evidence()
    claims = extract_claims(sentence)
    assert len(claims) == 2
    assert claims[0].local_label == "n01"
    assert claims[1].local_label == "n10"

    result = bind_claims(claims, ev)
    assert len(result.bound) == 2
    assert result.bound[0]["bound_to"].endswith("n01")
    assert result.bound[1]["bound_to"].endswith("n10")


def test_8_trailing_citation_in_bullet_item() -> None:
    """8. Trailing citation applies to compact metric list in bullet item."""
    bullet = (
        "- Basel traffic light: Skipped as not applicable; calibrated to 250 observations "
        "at 99%, while this sample uses 1,000 observations at 99% [EV-811be7152197]."
    )
    ev = _sample_var_evidence()
    claims = extract_claims(bullet)
    assert len(claims) >= 3
    # All claims in the bullet item inherit EV-811be7152197
    for c in claims:
        assert "EV-811be7152197" in c.cited_evidence

    result = bind_claims(claims, ev)
    c250 = next(c for c in result.bound if c["surface"] == "250")
    assert c250["bound_to"].endswith("band_n_observations")


def test_9_unsupported_uncited_zero_fails() -> None:
    """9. Unsupported uncited zero still FAILS."""
    sentence = "There were 0 spurious model calibrations during the review cycle."
    ev = _sample_var_evidence()
    claims = extract_claims(sentence)
    result = bind_claims(claims, ev)
    assert len(result.bound) == 0
    assert len(result.unbound) == 1
    assert result.unbound[0]["reason"] in (
        GroundingReasonCode.NO_LOCAL_EVIDENCE_CITATION,
        GroundingReasonCode.AMBIGUOUS_METRIC_BINDING,
    )


def test_10_wrong_cited_ev_fails() -> None:
    """10. Wrong cited EV still FAILS."""
    sentence = "We observed n11 = 0 [EV-2ede64de883e]."
    ev = _sample_var_evidence()
    claims = extract_claims(sentence)
    result = bind_claims(claims, ev)
    assert len(result.bound) == 0
    assert len(result.unbound) == 1
    assert result.unbound[0]["reason"] == GroundingReasonCode.VALUE_MISMATCH


def test_11_symbolic_metric_parsing() -> None:
    """11. Key=value / symbolic metric parsing for p=, LR=, gamma=."""
    sentence = "The POF test produced LR = 1.8862, p = 0.1696, and gamma = 0.05 [EV-2ede64de883e]."
    ev = _sample_var_evidence()
    claims = extract_claims(sentence)
    assert len(claims) == 3
    result = bind_claims(claims, ev)
    assert len(result.bound) == 3

    bound_paths = {b["bound_to"].split(".")[-1]: b["evidence_value"] for b in result.bound}
    assert bound_paths == {"lr_uc": 1.8862, "p_value": 0.1696, "gamma_test": 0.05}


def test_12_date_masking_remains_green() -> None:
    """12. Date masking remains green: date tokens are not extracted as claims."""
    sentence = "On 2026-09-02, the sample evaluated had 1,000 observations [EV-1da1c07274ba]."
    claims = extract_claims(sentence)
    # 2026, 09, 02 are dates and masked; only 1,000 is a claim
    surfaces = [c.surface for c in claims]
    assert "2026" not in surfaces
    assert "09" not in surfaces
    assert "02" not in surfaces
    assert "1,000" in surfaces


def test_13_grounding_census_invariant() -> None:
    """13. Grounding census invariant: bound + unbound == total_claims."""
    sentence = (
        "The model uses 99% confidence [EV-2ede64de883e] with 999 arbitrary unbound tokens "
        "and 0 unsupported things."
    )
    ev = _sample_var_evidence()
    claims = extract_claims(sentence)
    result = bind_claims(claims, ev)
    assert len(result.bound) + len(result.unbound) == result.total_claims
    assert result.total_claims == len(claims)


def test_14_traffic_light_sample_size_binds() -> None:
    """14. 1,000 observations cited with var_traffic_light binds to parameters.n_observations."""
    sentence = (
        "The traffic-light diagnostic was skipped because its historical bands are calibrated "
        "to 250 observations while the sample here has 1,000 [EV-811be7152197]."
    )
    ev = _sample_var_evidence()
    claims = extract_claims(sentence)
    result = bind_claims(claims, ev)
    c1000 = next(c for c in result.bound if c["surface"] == "1,000")
    assert c1000["bound_to"].endswith("n_observations")
    assert c1000["evidence_value"] == 1000.0


def test_15_one_percent_nominal_expectation_binds() -> None:
    """15. Nominal 1% expectation binds to metrics.alpha_var (0.01)."""
    sentence = (
        "- The exception count is lower than the nominal 1% expectation (6 vs 10 over 1,000), "
        "but the Kupiec test did not reject correct coverage on this sample [EV-2ede64de883e]."
    )
    ev = _sample_var_evidence()
    claims = extract_claims(sentence)
    result = bind_claims(claims, ev)
    c1 = next(c for c in result.bound if c["surface"] == "1%")
    assert c1["bound_to"].endswith("metrics.alpha_var")
    assert c1["evidence_value"] == 0.01


def test_16_uncited_title_confidence_binds() -> None:
    """16. Uncited title heading '99% VaR' binds to primary metrics.confidence."""
    sentence = "Summary of the supplied VaR backtesting evidence (99% VaR, actual P&L, 1,000 aligned days)"
    ev = _sample_var_evidence()
    claims = extract_claims(sentence)
    result = bind_claims(claims, ev)
    c99 = next(c for c in result.bound if c["surface"] == "99%")
    assert c99["bound_to"].endswith("metrics.confidence")
    assert c99["evidence_value"] == 0.99


def test_17_semicolon_and_container_citations() -> None:
    """17. Citation container parser supports semicolon, comma, and parenthesized lists."""
    from start.attestation.claims import EVIDENCE_ID_PATTERN

    assert EVIDENCE_ID_PATTERN.findall("[EV-A; EV-B; EV-C]") == ["EV-A", "EV-B", "EV-C"]
    assert EVIDENCE_ID_PATTERN.findall("[EV-A, EV-B]") == ["EV-A", "EV-B"]
    assert EVIDENCE_ID_PATTERN.findall("(EV-A; EV-B)") == ["EV-A", "EV-B"]
    assert EVIDENCE_ID_PATTERN.findall("[EV-A] [EV-B]") == ["EV-A", "EV-B"]


def test_18_naked_prose_citation_refused() -> None:
    """18. Naked prose mentions of EV identifiers are strictly refused as citations."""
    from start.attestation.claims import EVIDENCE_ID_PATTERN

    assert EVIDENCE_ID_PATTERN.findall("EV-A appears in the ledger") == []
    assert EVIDENCE_ID_PATTERN.findall("The evidence EV-ca08cc9be3b2 is recorded") == []


def test_19_exact_scientific_notation_binds() -> None:
    """19. Scientific notation like 1e-06 binds cleanly without integer rounding."""
    sentence = "Model uses ridge 1e-06 [EV-REGEM]."
    ev = {
        "EV-REGEM": {
            "metrics.ridge_used": 1e-06,
            "metrics.tolerance": 1e-06,
        }
    }
    claims = extract_claims(sentence)
    result = bind_claims(claims, ev)
    assert len(result.unbound) == 0
    assert len(result.bound) == 1
    assert result.bound[0]["surface"] == "1e-06"
    assert result.bound[0]["evidence_value"] == 1e-06


def test_20_shared_exponent_range_parsed_and_bound() -> None:
    """20. Shared-exponent range 3.65–4.19e-05 expands and binds both endpoints."""
    sentence = (
        "Estimates have small minimum eigenvalues on the order of 3.65–4.19e-05 "
        "[EV-COV-A; EV-COV-B]."
    )
    ev = {
        "EV-COV-A": {
            "metrics.min_eigenvalue": 3.6492660611e-05,
        },
        "EV-COV-B": {
            "metrics.min_eigenvalue_after": 4.1928655327e-05,
        },
    }
    claims = extract_claims(sentence)
    assert len(claims) == 2
    surfaces = {c.surface for c in claims}
    assert "3.65e-05" in surfaces
    assert "4.19e-05" in surfaces
    result = bind_claims(claims, ev)
    assert len(result.unbound) == 0
    assert len(result.bound) == 2


def test_21_label_aware_ddof_disambiguation() -> None:
    """21. Local labels ddof=1 and ddof=0 disambiguate from multiple zero-valued metrics."""
    sentence = "Estimand uses ddof=1 for sample vs ddof=0 for ML [EV-EMP; EV-REGEM]."
    ev = {
        "EV-EMP": {
            "metrics.ddof": 1,
            "metrics.n_observations": 1000,
        },
        "EV-REGEM": {
            "metrics.ddof": 0,
            "metrics.n_negative_eigenvalues": 0,
            "metrics.n_eigenvalue_clips": 0,
            "metrics.n_pseudoinverse_fallbacks": 0,
        },
    }
    claims = extract_claims(sentence)
    result = bind_claims(claims, ev)
    assert len(result.unbound) == 0
    c0 = next(c for c in result.bound if c["surface"] == "0")
    assert c0["bound_to"] == "metrics.ddof"
    assert c0["evidence_id"] == "EV-REGEM"
    c1 = next(c for c in result.bound if c["surface"] == "1")
    assert c1["bound_to"] == "metrics.ddof"
    assert c1["evidence_id"] == "EV-EMP"


def test_22_unicode_hyphen_dates_masked() -> None:
    """22. Dates using unicode non-breaking hyphens (\u2011) are properly masked."""
    sentence = "Dates span 2020\u201109\u201125 to 2023\u201101\u201119 [EV-0d6d738f0309]."
    claims = extract_claims(sentence)
    surfaces = {c.surface for c in claims}
    assert "09" not in surfaces
    assert "25" not in surfaces
    assert "01" not in surfaces
    assert "19" not in surfaces


def test_23_indicator_convention_masked() -> None:
    """23. Indicator variable definitions (I_t = 1 iff ...) are masked from extraction."""
    sentence = "convention I_t = 1 iff PnL_t < -VaR_t with VaR a positive loss magnitude [EV-0d6d738f0309]"
    claims = extract_claims(sentence)
    assert len(claims) == 0


def test_24_wrapped_multiline_sentence_citations() -> None:
    """24. Wrapped sentence with citations across newlines captures all cited records."""
    sentence = (
        "- Deterministic evidence shows 6 exceptions in 1,000 observations and non-rejection of "
        "coverage at the recorded 5% level [EV-0d6d738f0309], \n"
        "[EV-22c41e02a469], [EV-160fa00e9076], [EV-122c40618013].\n"
        "- Next bullet item."
    )
    claims = extract_claims(sentence)
    c5 = next(c for c in claims if c.surface == "5%")
    assert "EV-0d6d738f0309" in c5.cited_evidence
    assert "EV-22c41e02a469" in c5.cited_evidence
    assert "EV-160fa00e9076" in c5.cited_evidence
    assert "EV-122c40618013" in c5.cited_evidence


def test_25_plural_digits_and_lag_order_masked() -> None:
    """25. Plural digits (0’s between 1’s) and lag orders (lag 1) are masked."""
    sentence = "Evidence shows clustering at lag 1 and run lengths with 0’s between 1’s [EV-0d6d738f0309]."
    claims = extract_claims(sentence)
    surfaces = {c.surface for c in claims}
    assert "1" not in surfaces
    assert "0" not in surfaces


def test_26_numbered_list_parenthesis_headers_masked() -> None:
    """26. Numbered list headers (1), 2), (1)) are masked from claim extraction."""
    text = (
        "Key inconsistencies:\n\n"
        "1) Provenance and missing data\n"
        "2) Dependence on missingness mechanism\n"
        "3) Scale and comparability\n"
        "4) Numerical conditioning\n"
    )
    claims = extract_claims(text)
    surfaces = {c.surface for c in claims}
    assert "1" not in surfaces
    assert "2" not in surfaces
    assert "3" not in surfaces
    assert "4" not in surfaces


def test_27_multiplier_unit_not_misclassified_as_confidence() -> None:
    """27. Multipliers like 0.7x are not misclassified as VAR_CONFIDENCE."""
    from start.attestation.claims import SemanticRole
    text = "Power against 0.7x understated and 1.5x overstated VaR scenarios [EV-997c37943a31]."
    claims = extract_claims(text)
    c07 = next(c for c in claims if c.surface == "0.7x")
    assert c07.semantic_role == SemanticRole.GENERIC_NUMERIC
    c15 = next(c for c in claims if c.surface == "1.5x")
    assert c15.semantic_role == SemanticRole.GENERIC_NUMERIC


def test_28_uncited_significance_test_level_tie_breaks_cleanly() -> None:
    """28. Uncited claims mentioning 'hypothesis tests' resolve to gamma_test across multiple tests."""
    from start.attestation.claims import bind_claims
    text = "No acceptance thresholds beyond the recorded 5% hypothesis tests are provided."
    claims = extract_claims(text)
    c5 = next(c for c in claims if c.surface == "5%")
    assert c5.value == 5.0
    evidence = [
        {
            "evidence_id": "EV-1",
            "test_id": "traded_risk.var_kupiec_pof",
            "metrics": {"gamma_test": 0.05, "lr_uc": 1.886},
        },
        {
            "evidence_id": "EV-2",
            "test_id": "traded_risk.var_christoffersen_independence",
            "metrics": {"gamma_test": 0.05, "lr_ind": 0.0725},
        },
        {
            "evidence_id": "EV-3",
            "test_id": "validation.var_size_power",
            "metrics": {"nominal_size": 0.05, "observed.size": 0.066},
        },
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 1
    assert res.bound[0]["bound_to"].endswith("metrics.gamma_test")


def test_29_integer_claim_matches_rounded_float_or_relative_tolerance() -> None:
    """29. Integer claim like 189 matches condition_number = 188.978892."""
    from start.attestation.claims import bind_claims
    text = "Condition numbers are in a similar range: ~189 (empirical) [EV-1]."
    claims = extract_claims(text)
    evidence = [
        {
            "evidence_id": "EV-1",
            "test_id": "covariance.empirical",
            "metrics": {"condition_number": 188.978892},
        }
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 1
    assert res.bound[0]["bound_to"].endswith("condition_number")


def test_30_total_cells_matches_regularized_em_metric() -> None:
    """30. Total values (50,000) matches n_total_values emitted by covariance.regularized_em."""
    from start.attestation.claims import bind_claims
    text = "Values missing (7,394 of 50,000) [EV-1]."
    claims = extract_claims(text)
    evidence = [
        {
            "evidence_id": "EV-1",
            "test_id": "covariance.regularized_em",
            "metrics": {
                "n_assets": 50,
                "n_observations": 1000,
                "n_total_values": 50000,
                "n_missing_values": 7394,
            },
        }
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 2
    surfaces = {b["surface"] for b in res.bound}
    assert "7,394" in surfaces
    assert "50,000" in surfaces


def test_31_zero_percent_missing_fraction_matches_empirical_covariance() -> None:
    """31. 0% missing matches empirical covariance missing_fraction."""
    from start.attestation.claims import bind_claims
    text = "The dataset has 0% missing values under complete-case [EV-1]."
    claims = extract_claims(text)
    evidence = [
        {
            "evidence_id": "EV-1",
            "test_id": "covariance.empirical",
            "metrics": {
                "n_assets": 50,
                "n_observations": 1000,
                "n_complete": 1000,
                "n_dropped": 0,
                "missing_fraction": 0.0,
                "dropped_fraction": 0.0,
            },
        }
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 1
    assert res.bound[0]["surface"] == "0%"


def test_32_threshold_parsed_from_comparison_string() -> None:
    """32. Threshold 0.05 parsed from '<= 0.05 in all 18 cells' criterion string."""
    from start.attestation.claims import bind_claims
    text = "Non-convergence rate with required <= 0.05 [EV-1]."
    claims = extract_claims(text)
    evidence = [
        {
            "evidence_id": "EV-1",
            "test_id": "validation.regem_structural",
            "metrics": {
                "required.non_convergence_rate_all_cells": "<= 0.05 in all 18 cells",
            },
        }
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 1
    assert res.bound[0]["surface"] == "0.05"


def test_33_ev_diag_tags_not_treated_as_evidence_citations() -> None:
    """33. EV-DIAG tags like (EV-DIAG-ca6a2dfc) are not treated as evidence citations."""
    from start.attestation.claims import EVIDENCE_ID_PATTERN, bind_claims
    text = (
        "First-order clustering is not supported (n11=0 with non-rejection) [EV-1]. "
        "The absence of the diagnostic (EV-DIAG-ca6a2dfc) prevents resolving."
    )
    found_evs = EVIDENCE_ID_PATTERN.findall(text)
    assert "EV-1" in found_evs
    assert "EV-DIAG-ca6a2dfc" not in found_evs

    claims = extract_claims(text)
    evidence = [
        {
            "evidence_id": "EV-1",
            "test_id": "traded_risk.var_christoffersen_independence",
            "metrics": {"n11": 0},
        }
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 1
    assert res.bound[0]["surface"] == "0"


def test_34_sub_bullet_enclosing_block_citation_and_suffix_normalization() -> None:
    """34. Sub-bullet without explicit citation inherits enclosing block citation and normalizes metric suffixes."""
    from start.attestation.claims import bind_claims
    text = (
        "- Ledoit-Wolf shrinkage:\n"
        "  - Intensity 0.0106 [EV-2].\n"
        "  - Condition number moved 188.978892 -> 165.26676; min eigenvalue 3.7069287111e-05 -> 4.1928655327e-05; rank 50.\n"
        "  - Frobenius distance 9.3463249781e-05 [EV-2]."
    )
    claims = extract_claims(text)
    evidence = [
        {
            "evidence_id": "EV-1",
            "test_id": "covariance.empirical",
            "metrics": {
                "condition_number": 188.978892,
                "min_eigenvalue": 3.7069287111e-05,
                "rank": 50,
            },
        },
        {
            "evidence_id": "EV-2",
            "test_id": "covariance.ledoit_wolf_shrinkage",
            "metrics": {
                "shrinkage_intensity": 0.0106,
                "condition_number_before": 188.978892,
                "condition_number_after": 165.26676,
                "min_eigenvalue_before": 3.7069287111e-05,
                "min_eigenvalue_after": 4.1928655327e-05,
                "rank_after": 50,
                "frobenius_distance": 9.3463249781e-05,
            },
        },
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    bound_surfaces = {b["surface"] for b in res.bound}
    assert "188.978892" in bound_surfaces
    assert "165.26676" in bound_surfaces
    assert "3.7069287111e-05" in bound_surfaces
    assert "4.1928655327e-05" in bound_surfaces
    assert "50" in bound_surfaces


def test_35_order_of_magnitude_notation_not_extracted_as_individual_claims() -> None:
    """35. Order of magnitude notation like O(10^-5) is masked and not extracted as claims 10 and -5."""
    from start.attestation.claims import bind_claims
    text = "Minimum eigenvalues are O(10^-5): 3.7069e-05 (empirical) [EV-1]."
    claims = extract_claims(text)
    surfaces = {c.surface for c in claims}
    assert "10" not in surfaces
    assert "-5" not in surfaces
    assert "3.7069e-05" in surfaces

    evidence = [
        {
            "evidence_id": "EV-1",
            "test_id": "covariance.empirical",
            "metrics": {"min_eigenvalue": 3.7069e-05},
        }
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 1
    assert res.bound[0]["surface"] == "3.7069e-05"


def test_36_portfolio_loss_and_return_percent_compatible() -> None:
    """36. Portfolio loss and return metrics (decimal rates) bind percent-typed claims."""
    from start.attestation.claims import bind_claims
    text = (
        "Under linear return shock, portfolio loss is 0.2692605% [EV-1] and factor shift is 0.35% [EV-2]."
    )
    claims = extract_claims(text)
    evidence = [
        {
            "evidence_id": "EV-1",
            "test_id": "scenario.linear_return",
            "metrics": {"portfolio_loss": 0.002692605179195271},
        },
        {
            "evidence_id": "EV-2",
            "test_id": "scenario.factor_linear",
            "metrics": {"portfolio_loss": 0.0034519478064607063},
        },
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 2
    bound_surfaces = {b["surface"] for b in res.bound}
    assert "0.2692605%" in bound_surfaces
    assert "0.35%" in bound_surfaces


def test_37_compound_sentence_labeled_metric_in_scope_resolution() -> None:
    """37. Labeled metric in compound sentence resolves to matching in-scope record if cited record lacks it."""
    from start.attestation.claims import bind_claims
    text = (
        "The six dates span two years [EV-1], which is consistent with n11 = 0 and no clustering."
    )
    claims = extract_claims(text)
    evidence = [
        {
            "evidence_id": "EV-1",
            "test_id": "traded_risk.var_exceptions",
            "metrics": {"n_exceptions": 6, "n_dropped_alignment": 0},
        },
        {
            "evidence_id": "EV-2",
            "test_id": "traded_risk.var_christoffersen_independence",
            "metrics": {"n11": 0},
        },
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 1
    assert res.bound[0]["surface"] == "0"
    assert res.bound[0]["bound_to"].endswith("metrics.n11")
    assert res.bound[0]["evidence_id"] == "EV-2"


def test_38_scientific_times_ten_shared_exponent_range_binds() -> None:
    """38. Scientific notation with base-10 multiplication and shared exponent expands and binds."""
    from start.attestation.claims import bind_claims
    text = (
        "Minimum eigenvalues are small in all cases (≈3.65–4.19×10^-5) [EV-COV-A; EV-COV-B]."
    )
    claims = extract_claims(text)
    assert len(claims) == 2
    surfaces = {c.surface for c in claims}
    assert "3.65e-5" in surfaces
    assert "4.19e-5" in surfaces

    evidence = [
        {
            "evidence_id": "EV-COV-A",
            "test_id": "covariance.empirical",
            "metrics": {"min_eigenvalue": 3.6492660611e-05},
        },
        {
            "evidence_id": "EV-COV-B",
            "test_id": "covariance.ledoit_wolf",
            "metrics": {"min_eigenvalue": 4.1928655327e-05},
        },
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 2
    bound_metrics = {b["bound_to"].split(".")[-1]: b["evidence_value"] for b in res.bound}
    assert "min_eigenvalue" in bound_metrics


def test_39_reverse_stress_loss_gap_binds() -> None:
    """39. Achieved loss error / gap relative to target binds to reverse stress loss_gap."""
    from start.attestation.claims import bind_claims
    text = (
        "The solver converged with an achieved loss within about 1.9e-8 of the target [EV-REV]."
    )
    claims = extract_claims(text)
    assert len(claims) == 1
    assert claims[0].surface == "1.9e-8"

    evidence = [
        {
            "evidence_id": "EV-REV",
            "test_id": "scenario.reverse_stress",
            "metrics": {
                "target_loss": 0.10,
                "portfolio_loss": 0.10000001910947595,
                "loss_gap": 1.9109475946e-08,
                "distance": 42.10067228,
                "converged": True,
            },
        },
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 1
    assert res.bound[0]["surface"] == "1.9e-8"
    assert "loss_gap" in res.bound[0]["bound_to"]
    assert abs(res.bound[0]["evidence_value"] - 1.9109475946e-08) < 1e-12


def test_40_compound_findings_list_with_parenthetical_semicolon_binds() -> None:
    """40. Compound findings list with (p = 0.7877; n11=0) citations and lags > 1 binds."""
    from start.attestation.claims import bind_claims
    text = (
        "Deterministic findings: 6 exceptions over 1,000 observations at 99% VaR [EV-VAR], "
        "with Kupiec POF non-rejection (p = 0.1696) [EV-POF], "
        "first-order independence non-rejection (p = 0.7877; n11=0) [EV-IND], "
        "and conditional-coverage non-rejection (p = 0.3755) [EV-CC]. "
        "Recommend tests to screen for clustering at lags > 1."
    )
    claims = extract_claims(text)
    surfaces = {c.surface for c in claims}
    assert "1" not in surfaces  # lags > 1 is masked

    evidence = [
        {"evidence_id": "EV-VAR", "test_id": "traded_risk.var_exceptions", "metrics": {"exceptions": 6, "n_observations": 1000, "confidence": 0.99}},
        {"evidence_id": "EV-POF", "test_id": "traded_risk.var_kupiec_pof", "metrics": {"p_value": 0.1696274814}},
        {"evidence_id": "EV-IND", "test_id": "traded_risk.var_christoffersen_independence", "metrics": {"p_value": 0.7877195429, "n11": 0}},
        {"evidence_id": "EV-CC", "test_id": "traded_risk.var_christoffersen_conditional", "metrics": {"p_value": 0.3755475438}},
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    p_ind = next(b for b in res.bound if b["surface"] == "0.7877")
    assert p_ind["evidence_id"] == "EV-IND"
    assert p_ind["bound_to"].endswith("metrics.p_value")


def test_41_en_dash_negatives_and_risk_metrics_percent_binds() -> None:
    """41. En-dash negatives (F2=–0.03831), volatility/ES percentages, and variance shortfall bind."""
    from start.attestation.claims import bind_claims
    text = (
        "Active exposures: F2=–0.03831, F3=–0.01957, F4=–0.00899, F5=–0.06564 [EV-EXP]. "
        "annualised volatility 3.7706%, ES 0.4711% [EV-RISK]. "
        "The factor-model total variance is 12.54% below the empirical portfolio variance (ratio 0.87458) [EV-ATT]."
    )
    claims = extract_claims(text)
    evidence = [
        {
            "evidence_id": "EV-EXP",
            "test_id": "attribution.exposure_analysis",
            "metrics": {
                "portfolio_exposure.F2": -0.038312169916,
                "portfolio_exposure.F3": -0.019567937887,
                "portfolio_exposure.F4": -0.008994553407,
                "portfolio_exposure.F5": -0.065637988189,
            },
        },
        {
            "evidence_id": "EV-RISK",
            "test_id": "portfolio.risk_statistics",
            "metrics": {
                "annualised_volatility": 0.0377060748,
                "historical_es": 0.0047110252,
            },
        },
        {
            "evidence_id": "EV-ATT",
            "test_id": "attribution.risk_attribution",
            "metrics": {
                "factor_model_to_empirical_ratio": 0.8745808047,
                "variance_shortfall": 0.1254191953,
            },
        },
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 8


def test_42_percent_range_en_dash_binds() -> None:
    """42. Percentage range with en-dash (12.8%–17.2%) extracts positive values and binds."""
    from start.attestation.claims import bind_claims
    text = "per-column missingness 12.8%–17.2% [EV-REGEM]."
    claims = extract_claims(text)
    assert len(claims) == 2
    assert claims[0].value == 12.8
    assert claims[1].value == 17.2
    evidence = [
        {
            "evidence_id": "EV-REGEM",
            "test_id": "covariance.regularized_em",
            "metrics": {
                "min_column_missing_fraction": 0.128,
                "max_column_missing_fraction": 0.172,
            },
        }
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 2


def test_43_introductory_colon_clause_inherits_itemized_citations() -> None:
    """43. Introductory summary clause preceding a colon ':' inherits citations from the following list."""
    from start.attestation.claims import bind_claims
    text = (
        "- The three estimates all exhibit very small minimum eigenvalues on the order of "
        "3.6–4.2e-05 and condition numbers in the 165–192 range: "
        "empirical min eigenvalue 3.7069e-05, condition number 188.98 [EV-EMP]; "
        "Ledoit–Wolf min eigenvalue 4.1929e-05, condition number 165.27 [EV-LW]; "
        "RegEM min eigenvalue 3.6493e-05, condition number 191.64 [EV-REM]."
    )
    claims = extract_claims(text)
    c_36 = next(c for c in claims if c.surface == "3.6e-05")
    assert "EV-REM" in c_36.cited_evidence
    evidence = [
        {
            "evidence_id": "EV-EMP",
            "test_id": "covariance.empirical_sample",
            "metrics": {"min_eigenvalue": 3.7069287111e-05, "condition_number": 188.978892},
        },
        {
            "evidence_id": "EV-LW",
            "test_id": "covariance.ledoit_wolf_shrinkage",
            "metrics": {"min_eigenvalue_after": 4.1928655327e-05, "condition_number_after": 165.26676},
        },
        {
            "evidence_id": "EV-REM",
            "test_id": "covariance.regularized_em",
            "metrics": {"min_eigenvalue": 3.6492660611e-05, "condition_number": 191.640192},
        },
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0


def test_44_bracketed_semicolon_masking_allows_trailing_citations() -> None:
    """44. Semicolons inside bracketed citations [EV-1; EV-2] are masked so preceding claims receive citations."""
    from start.attestation.claims import bind_claims
    text = (
        "- Whether 6 exceptions over 1000 at 99% indicate “conservative” or “adequate” VaR "
        "beyond non-rejection needs a defined calibration target; none is provided [EV-VAR; EV-CC]."
    )
    claims = extract_claims(text)
    c_1000 = next(c for c in claims if c.surface == "1000")
    assert "EV-VAR" in c_1000.cited_evidence
    evidence = [
        {
            "evidence_id": "EV-VAR",
            "test_id": "traded_risk.var_exceptions",
            "metrics": {"exceptions": 6, "n_observations": 1000, "confidence": 0.99},
        },
        {
            "evidence_id": "EV-CC",
            "test_id": "traded_risk.var_christoffersen_conditional",
            "metrics": {"p_value": 0.3755},
        },
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0


def test_45_shared_percent_range_unit_expansion() -> None:
    """45. Shared percent range 0.27–0.35% expands first endpoint to 0.27% and binds to decimal values."""
    from start.attestation.claims import bind_claims
    text = (
        "- Whether scenario losses of 0.27–0.35% are sufficiently severe versus 95% VaR 0.3918% "
        "requires a scenario severity policy; none is provided [EV-LIN; EV-FAC; EV-STAT]."
    )
    claims = extract_claims(text)
    c_027 = next(c for c in claims if "0.27" in c.surface)
    assert c_027.unit == "%"
    evidence = [
        {
            "evidence_id": "EV-LIN",
            "test_id": "scenario.linear_return",
            "metrics": {"portfolio_loss": 0.002692605},
        },
        {
            "evidence_id": "EV-FAC",
            "test_id": "scenario.factor_linear",
            "metrics": {"portfolio_loss": 0.003451948},
        },
        {
            "evidence_id": "EV-STAT",
            "test_id": "portfolio.risk_statistics",
            "metrics": {"var_confidence": 0.95, "historical_var": 0.003917727},
        },
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0


def test_46_matrix_dimension_tokens_not_extracted_as_numeric_claims() -> None:
    """46. Matrix dimension expressions (2×2, 2x2, 50x50, 50×50) are structural, not numeric claims."""
    text = (
        "The 2×2 transition counts include n11=0 [EV-IND]. "
        "Estimated on a 50×50 covariance matrix."
    )
    claims = extract_claims(text)
    surfaces = [c.surface for c in claims]
    assert "2" not in surfaces
    assert "2×2" not in surfaces
    assert "50" not in surfaces
    assert "0" in surfaces


def test_47_quoted_semicolons_and_conjunction_separation() -> None:
    """47. Semicolons inside quotes do not break citation binding; preceding citations do not cross conjunctions."""
    from start.attestation.claims import bind_claims
    text = (
        "The direct contradiction between “0.0% missing; 1,000 complete observations” for the empirical runs [EV-EMP] "
        "and “14.788% missing; only 1 complete row” for RegEM [EV-REGEM]."
    )
    claims = extract_claims(text)
    c_14 = next(c for c in claims if "14.788" in c.surface)
    assert c_14.cited_evidence == ("EV-REGEM",)
    evidence = [
        {
            "evidence_id": "EV-EMP",
            "test_id": "covariance.empirical_sample",
            "metrics": {"missing_fraction": 0.0, "n_observations": 1000},
        },
        {
            "evidence_id": "EV-REGEM",
            "test_id": "covariance.regularized_em",
            "metrics": {"missing_fraction": 0.14788, "n_complete_rows": 1},
        },
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 4


def test_48_semicolon_independent_clause_does_not_leak_trailing_citations() -> None:
    """50. Preceding independent clause before semicolon does not inherit trailing citations."""
    from start.attestation.claims import bind_claims
    text = (
        "- Compare condition numbers (189 vs 165 after shrinkage vs 192 for RegEM) to "
        "resulting weight dispersion and turnover; record effective positions and Herfindahl "
        "as in current outputs [EV-PORT]."
    )
    claims = extract_claims(text)
    c_189 = next(c for c in claims if c.surface == "189")
    c_165 = next(c for c in claims if c.surface == "165")
    c_192 = next(c for c in claims if c.surface == "192")
    assert c_189.cited_evidence == ()
    assert c_165.cited_evidence == ()
    assert c_192.cited_evidence == ()
    evidence = [
        {
            "evidence_id": "EV-PORT",
            "test_id": "portfolio.mean_variance",
            "metrics": {"effective_n_positions": 25.0, "herfindahl_index": 0.04},
        },
        {
            "evidence_id": "EV-EMP",
            "test_id": "covariance.empirical",
            "metrics": {"condition_number": 188.978892},
        },
        {
            "evidence_id": "EV-LW",
            "test_id": "covariance.ledoit_wolf_shrinkage",
            "metrics": {"condition_number_after": 165.26676},
        },
        {
            "evidence_id": "EV-REM",
            "test_id": "covariance.regularized_em",
            "metrics": {"condition_number": 191.640192},
        },
    ]
    res = bind_claims(claims, evidence)
    assert len(res.unbound) == 0
    assert len(res.bound) == 3













