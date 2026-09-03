"""v4.3.0 — review routing, multiline input and registry-driven applicability."""
from __future__ import annotations

import io

import pytest

from start.registry import list_tests
from start.review.applicability import applicable_tests, build_plan_preview
from start.review.architecture import (
    DOMAIN_CONTEXT,
    TRADITIONAL_ML_MODELS,
    PredictiveTechnology,
    ReviewContextBundle,
    ReviewDomain,
    ReviewLifecycle,
    ReviewMode,
    parse_domain_selection,
    required_context_types,
    requires_predictive_technology,
)
from start.review.multiline_input import (
    MULTILINE_TERMINATOR,
    ReviewCancelled,
    read_multiline_text,
)

D = ReviewDomain


def _read(text, **kw):
    return read_multiline_text("Business Context", stream=io.StringIO(text),
                               printer=lambda _: None, **kw)


# ==================================================== MODE / DOMAIN ==
def test_review_mode_values():
    assert {str(m) for m in ReviewMode} == {"single_domain", "cross_domain"}


def test_exactly_three_atomic_domains():
    assert {str(d) for d in ReviewDomain} == {"predictive", "market", "treasury"}


def test_no_composite_or_technology_domain_values():
    """market+treasury is a composition; deep learning is a technology."""
    values = {str(d) for d in ReviewDomain}
    for forbidden in ("integrated", "market_treasury", "deep_learning",
                      "traditional_ml", "integrated_model_risk", "genai",
                      "genai_agentic"):
        assert forbidden not in values


def test_predictive_technology_values():
    assert {str(t) for t in PredictiveTechnology} == {"traditional_ml", "deep_learning"}


def test_lifecycle_has_five_options():
    assert len(list(ReviewLifecycle)) == 5
    assert str(ReviewLifecycle.INITIAL_VALIDATION) == "initial_validation"


# ==================================================== DOMAIN MAPPING ==
@pytest.mark.parametrize("domain,context", [
    (D.PREDICTIVE, "tabular"), (D.MARKET, "market"), (D.TREASURY, "short_rate"),
])
def test_domain_maps_to_exactly_one_context(domain, context):
    assert DOMAIN_CONTEXT[domain] == context


def test_market_plus_treasury_is_a_union_not_a_new_type():
    assert required_context_types((D.MARKET, D.TREASURY)) == ("market", "short_rate")


def test_all_three_domains_union():
    assert required_context_types((D.PREDICTIVE, D.MARKET, D.TREASURY)) == (
        "tabular", "market", "short_rate")


def test_no_synthetic_context_type_is_ever_produced():
    for combo in [(D.MARKET,), (D.TREASURY,), (D.MARKET, D.TREASURY),
                  (D.PREDICTIVE, D.MARKET, D.TREASURY)]:
        for context in required_context_types(combo):
            assert context in {"tabular", "market", "short_rate"}


# ==================================================== SELECTION PARSING ==
def test_selection_order_is_canonical_not_input_order():
    """Two reviewers describing the same scope must get the same review."""
    assert parse_domain_selection("3,2") == parse_domain_selection("2,3")
    assert parse_domain_selection("3,2") == (D.MARKET, D.TREASURY)


def test_duplicate_selection_is_rejected_not_deduplicated():
    """'2,2' is far more likely a typo for '2,3' than a deliberate request."""
    with pytest.raises(ValueError, match="duplicate"):
        parse_domain_selection("2,2")


def test_cross_domain_requires_two_distinct_domains():
    with pytest.raises(ValueError, match="at least two"):
        parse_domain_selection("1", mode=ReviewMode.CROSS_DOMAIN)


def test_single_domain_takes_exactly_one():
    assert parse_domain_selection("2", mode=ReviewMode.SINGLE_DOMAIN) == (D.MARKET,)
    with pytest.raises(ValueError, match="exactly one"):
        parse_domain_selection("2,3", mode=ReviewMode.SINGLE_DOMAIN)


@pytest.mark.parametrize("bad", ["9", "0", "-1", "x", "", "   ", "1,x"])
def test_invalid_selections_are_rejected(bad):
    with pytest.raises(ValueError):
        parse_domain_selection(bad, mode=ReviewMode.CROSS_DOMAIN)


def test_whitespace_separated_selection_is_accepted():
    assert parse_domain_selection("2 3") == (D.MARKET, D.TREASURY)


# ==================================================== TECHNOLOGY GATING ==
@pytest.mark.parametrize("domains,expected", [
    ((D.MARKET,), False),
    ((D.TREASURY,), False),
    ((D.MARKET, D.TREASURY), False),
    ((D.PREDICTIVE,), True),
    ((D.PREDICTIVE, D.MARKET), True),
    ((D.PREDICTIVE, D.MARKET, D.TREASURY), True),
])
def test_technology_offered_only_when_predictive_selected(domains, expected):
    assert requires_predictive_technology(domains) is expected


def test_legacy_tree_model_menu_survives_intact():
    """The functionality moves under Predictive -> Traditional ML; it is not deleted."""
    for model in ("Random Forest", "CatBoost", "XGBoost", "LightGBM",
                  "Distributed Random Forest", "Extra Trees",
                  "Random Rotation Forest"):
        assert model in TRADITIONAL_ML_MODELS


# ==================================================== APPLICABILITY ==
def test_market_applicability_is_twenty_five():
    """Verify Market domain applicability is exactly 25 registered root tests."""
    result = applicable_tests((D.MARKET,))
    assert result.count == 25
    assert result.context_types == ("market",)


def test_market_family_breakdown():
    """Verify Market family breakdown: portfolio=10 (5 basic + 5 optimization), attribution=6, traded_risk=6, covariance=3."""
    by_family = applicable_tests((D.MARKET,)).by_family
    assert by_family == {"portfolio": 10, "attribution": 6,
                         "traded_risk": 6, "covariance": 3}


def test_treasury_applicability_is_exactly_cev_and_stanton():
    """Verify Treasury domain applicability is exactly 2 registered root tests (CEV and Stanton)."""
    result = applicable_tests((D.TREASURY,))
    assert result.count == 2
    assert set(result.test_ids) == {"traded_risk.cev_elasticity",
                                    "traded_risk.stanton_nonparametric"}


def test_market_plus_treasury_is_twenty_seven():
    """Verify Market + Treasury domain union applicability is exactly 27 registered root tests (25 Market + 2 Treasury)."""
    assert applicable_tests((D.MARKET, D.TREASURY)).count == 27


def test_short_rate_tests_never_leak_into_market_only():
    ids = set(applicable_tests((D.MARKET,)).test_ids)
    assert "traded_risk.cev_elasticity" not in ids
    assert "traded_risk.stanton_nonparametric" not in ids


def test_market_var_tests_never_leak_into_treasury_only():
    ids = set(applicable_tests((D.TREASURY,)).test_ids)
    assert not any("var_" in i for i in ids)
    assert not any(i.startswith(("portfolio.", "attribution.", "covariance.")) for i in ids)


def test_applicability_is_derived_from_the_registry_not_a_constant():
    """Every applicable ID must be a live TestSpec with a matching context_type."""
    specs = {s.test_id: s for s in list_tests()}
    for domains in [(D.MARKET,), (D.TREASURY,), (D.MARKET, D.TREASURY),
                    (D.PREDICTIVE,)]:
        result = applicable_tests(domains)
        wanted = set(result.context_types)
        for test_id in result.test_ids:
            assert test_id in specs
            assert specs[test_id].context_type in wanted


def test_family_filter_narrows_but_never_adds():
    full = set(applicable_tests((D.MARKET,)).test_ids)
    narrowed = applicable_tests((D.MARKET,), families=("portfolio",))
    assert set(narrowed.test_ids) <= full
    assert narrowed.by_family == {"portfolio": 10}


def test_predictive_applicability_is_tabular_only():
    result = applicable_tests((D.PREDICTIVE,))
    assert result.context_types == ("tabular",)
    assert result.count == 52


def test_all_domains_covers_the_whole_registry():
    assert applicable_tests((D.PREDICTIVE, D.MARKET, D.TREASURY)).count == len(list_tests())


# ==================================================== CONTEXT BUNDLE ==
def test_bundle_is_not_an_analytical_context_type():
    """It must never be dispatchable as a registered context."""
    bundle = ReviewContextBundle(domains=(D.MARKET,))
    assert not hasattr(bundle, "context_type")
    assert {s.context_type for s in list_tests()} <= {"tabular", "market", "short_rate"}


def test_bundle_reports_missing_contexts():
    bundle = ReviewContextBundle(domains=(D.MARKET, D.TREASURY))
    assert bundle.missing_context_types() == ("market", "short_rate")
    assert bundle.is_complete() is False


def test_bundle_complete_when_populated():
    bundle = ReviewContextBundle(domains=(D.MARKET,), market=object())
    assert bundle.is_complete() is True
    assert bundle.available_context_types() == ("market",)


def test_bundle_describe_carries_no_objects():
    bundle = ReviewContextBundle(domains=(D.MARKET,), market=object(),
                                 mode=ReviewMode.SINGLE_DOMAIN)
    for value in bundle.describe().values():
        assert isinstance(value, (str, list, bool, type(None)))


# ==================================================== PLAN PREVIEW ==
def test_plan_preview_counts_are_derived():
    bundle = ReviewContextBundle(domains=(D.MARKET, D.TREASURY),
                                 mode=ReviewMode.CROSS_DOMAIN,
                                 market=object(), short_rate=object())
    text = build_plan_preview(bundle).render()
    assert "Applicable Registered Tests: 27" in text
    assert "Traded Risk" in text and "Portfolio" in text


def test_plan_preview_flags_missing_context():
    bundle = ReviewContextBundle(domains=(D.MARKET, D.TREASURY),
                                 mode=ReviewMode.CROSS_DOMAIN)
    text = build_plan_preview(bundle).render()
    assert "Missing required context" in text


def test_plan_preview_shows_technology_only_when_predictive():
    market = ReviewContextBundle(domains=(D.MARKET,), market=object())
    assert "Technology:" not in build_plan_preview(market).render()
    predictive = ReviewContextBundle(
        domains=(D.PREDICTIVE,), tabular=object(),
        technology=PredictiveTechnology.TRADITIONAL_ML)
    assert "Technology:" in build_plan_preview(predictive).render()


# ==================================================== MULTILINE INPUT ==
def test_single_line_then_end():
    assert _read("Hello world\nEND\n") == "Hello world"


def test_multi_paragraph_preserved():
    text = _read("Para one line A\nPara one line B\n\nPara two\nEND\n")
    assert text == "Para one line A\nPara one line B\n\nPara two"
    assert "\n\n" in text


def test_blank_lines_inside_are_preserved():
    assert _read("A\n\n\nB\nEND\n") == "A\n\n\nB"


def test_unicode_preserved():
    assert _read("Résumé — naïve café 日本語\nEND\n") == "Résumé — naïve café 日本語"


def test_numeric_only_lines_are_content_not_menu_answers():
    """The exact bug: '2' used to select menu option 2."""
    assert _read("1\n2\n3\nEND\n") == "1\n2\n3"


def test_menu_looking_strings_are_content():
    text = _read("Random Forest\n[1] Predictive Modeling\n2\nEND\n")
    assert "Random Forest" in text
    assert "[1] Predictive Modeling" in text
    assert text.endswith("2")


def test_terminator_is_case_sensitive():
    """'the end of the sample' must not terminate a reviewer's paragraph."""
    assert _read("end\nEnd\nthe end of the sample\nEND\n") == (
        "end\nEnd\nthe end of the sample")


def test_optional_field_accepts_immediate_end():
    assert _read("END\n", required=False) == ""


def test_required_field_reprompts_then_accepts():
    assert _read("END\nActual content\nEND\n", required=True) == "Actual content"


def test_eof_without_terminator_raises_cancelled():
    with pytest.raises(ReviewCancelled, match="EOF|ended"):
        _read("some text without terminator\n")


def test_eof_on_empty_input_raises_cancelled():
    with pytest.raises(ReviewCancelled):
        _read("")


def test_terminator_constant():
    assert MULTILINE_TERMINATOR == "END"


# ============================================ PASTE-LEAK REGRESSION ==
def test_paste_leak_regression_mandatory():
    """The exact mandated regression. Every pasted line belongs to business_context
    and ZERO characters reach the following menu."""
    pasted = (
        "The model is a high-materiality market and treasury risk framework...\n"
        "The framework supports independent risk oversight...\n"
        "The review should assess portfolio mathematics, covariance, VaR and "
        "evidence integrity...\n"
        "END\n"
        "2\n"
    )
    stream = io.StringIO(pasted)
    business_context = read_multiline_text("Business Context", stream=stream,
                                           printer=lambda _: None, required=True)

    assert "high-materiality market and treasury" in business_context
    assert "independent risk oversight" in business_context
    assert "portfolio mathematics, covariance, VaR" in business_context
    assert len(business_context.splitlines()) == 3

    # The very next read is the menu, and it sees ONLY the menu answer.
    next_menu = stream.readline().strip()
    assert next_menu == "2"
    assert stream.read() == ""


def test_pasted_menu_digits_do_not_leak_into_later_prompts():
    stream = io.StringIO("Context line\n1\n2\n3\nEND\n1\n")
    text = read_multiline_text("Business Context", stream=stream, printer=lambda _: None)
    assert text == "Context line\n1\n2\n3"
    assert stream.readline().strip() == "1"


def test_four_governance_fields_read_sequentially_without_leakage():
    stream = io.StringIO(
        "Business ctx line 1\nBusiness ctx line 2\nEND\n"
        "Clarification text\nEND\n"
        "Intended use\nEND\n"
        "END\n"                      # optional limitations, skipped
        "3\n"                        # the menu that follows
    )
    fields = [
        read_multiline_text(label, stream=stream, printer=lambda _: None)
        for label in ("Business Context", "Reviewer Clarification",
                      "Intended Use", "Known Limitations")
    ]
    assert fields[0] == "Business ctx line 1\nBusiness ctx line 2"
    assert fields[1] == "Clarification text"
    assert fields[2] == "Intended use"
    assert fields[3] == ""
    assert stream.readline().strip() == "3"


# ==================================================== REGISTRY GUARD ==
def test_registry_unchanged_at_79():
    assert len(list_tests()) == 79


def test_no_review_or_interactive_family_was_registered():
    families = {s.family for s in list_tests()}
    for forbidden in ("review", "profile", "interactive", "domain", "mode"):
        assert forbidden not in families


def test_no_duplicate_test_ids():
    ids = [s.test_id for s in list_tests()]
    assert len(ids) == len(set(ids))
