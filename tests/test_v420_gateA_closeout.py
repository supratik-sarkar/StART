"""Gate A closeout — registry census, F1 metadata, coverage, analytical invariance.

The census test exists because a narrative summary once reported
``preprocessing 22 / feature_engineering 14`` against a live registry of 52 with zero
duplicates. Those cannot both be right. A count that is asserted rather than transcribed
cannot drift silently, and a missing registration cannot hide behind arithmetic.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from start.registry import TestContext, list_tests
from start.registry.legacy_metadata import (
    DECLINED,
    LEGACY_METADATA,
    LEGACY_TEST_IDS,
    apply_legacy_metadata,
)
from start.risk.coverage import (
    CONTEXT_ALIASES,
    coverage_for_plan,
    normalise_context_type,
    tests_for_plan,
)
from start.risk.objects import RiskObject
from start.risk.plan import synthesise_plan

#: The 52 surfaces that existed at Gate-A closeout. Gate B ADDS to this; it must never
#: remove from it, which is what test_no_gate_a_id_was_lost enforces.
GATE_A_FAMILY_COUNTS = {
    "genai": 1, "preprocessing": 21, "supervised": 5,
    "xai": 4, "eda": 6, "feature_engineering": 15,
}
GATE_A_TOTAL = 52

#: Gate-B families as they land. The census is updated slice by slice rather than
#: deleted: a locked count is the thing that stops a missing registration hiding behind
#: arithmetic, so it is kept and moved deliberately.
GATE_B_FAMILY_COUNTS = {"portfolio": 10, "attribution": 6,
                        "traded_risk": 8, "covariance": 3}

EXPECTED_FAMILY_COUNTS = {**GATE_A_FAMILY_COUNTS, **GATE_B_FAMILY_COUNTS}
EXPECTED_TOTAL = sum(EXPECTED_FAMILY_COUNTS.values())

EDA_IDS = {
    "eda.descriptive_statistics", "eda.correlation", "eda.multicollinearity",
    "eda.numeric_distribution", "eda.categorical_distribution", "eda.class_imbalance",
}
A3_IDS = {
    "preprocessing.leakage_target_reconstruction", "preprocessing.leakage_high_correlation",
    "preprocessing.leakage_temporal", "preprocessing.leakage_entity_overlap",
    "preprocessing.leakage_row_overlap", "preprocessing.leakage_suspicious_predictivity",
    "preprocessing.leakage_name_heuristic", "preprocessing.target_analysis",
    "preprocessing.feature_target_relationship", "preprocessing.redundancy",
    "preprocessing.dimensionality_diagnostic", "preprocessing.categorical_drift",
}
LEGACY_PREPROCESSING_IDS = {
    "preprocessing.constant_features", "preprocessing.duplicates",
    "preprocessing.feature_drift", "preprocessing.feature_ranges",
    "preprocessing.high_cardinality", "preprocessing.missingness",
    "preprocessing.outliers", "preprocessing.split_diagnostics",
    "preprocessing.target_leakage",
}
A4_IDS = {
    "feature_engineering.plan", "feature_engineering.imputation",
    "feature_engineering.scaling", "feature_engineering.numeric_transform",
    "feature_engineering.winsorization", "feature_engineering.categorical_encoding",
    "feature_engineering.rare_category_grouping", "feature_engineering.woe_iv",
    "feature_engineering.monotonic_binning", "feature_engineering.interactions",
    "feature_engineering.temporal_features", "feature_engineering.aggregation_features",
    "feature_engineering.pca_transform", "feature_engineering.selection",
    "feature_engineering.fitting_scope_audit",
}


def _ids():
    return [s.test_id for s in list_tests()]


# ================================================================== census ==
def test_registry_total_is_locked():
    """Normal discovery only — no manual family imports."""
    assert len(list_tests()) == EXPECTED_TOTAL


def test_family_counts_are_locked():
    counts = dict(Counter(s.family for s in list_tests()))
    assert counts == EXPECTED_FAMILY_COUNTS


def test_no_duplicate_registrations():
    ids = _ids()
    assert len(ids) == len(set(ids))


def test_the_six_eda_ids_are_the_verified_a2_contract():
    """Guards against registry ID drift; a rename here is a blocker, not a fixup."""
    assert {i for i in _ids() if i.startswith("eda.")} == EDA_IDS


def test_all_twelve_a3_ids_present():
    assert A3_IDS <= set(_ids())


def test_a3_and_legacy_preprocessing_are_disjoint():
    """categorical_drift is A3 and must not also be counted as legacy."""
    assert not (A3_IDS & LEGACY_PREPROCESSING_IDS)
    assert len(A3_IDS) + len(LEGACY_PREPROCESSING_IDS) == EXPECTED_FAMILY_COUNTS["preprocessing"]


def test_all_fifteen_a4_ids_present_including_the_audit():
    ids = set(_ids())
    assert A4_IDS <= ids
    assert "feature_engineering.fitting_scope_audit" in ids


def test_preprocessing_family_is_exactly_legacy_plus_a3():
    actual = {i for i in _ids() if i.startswith("preprocessing.")}
    assert actual == LEGACY_PREPROCESSING_IDS | A3_IDS


# ==================================================================== F1 ==
def test_all_nineteen_legacy_ids_are_mapped():
    assert set(LEGACY_METADATA) == set(LEGACY_TEST_IDS)
    assert len(LEGACY_TEST_IDS) == 19


def test_no_legacy_test_lacks_risk_metadata():
    unmapped = [s.test_id for s in list_tests() if not s.risk_dimensions]
    assert unmapped == []


def test_every_mapping_uses_live_taxonomy_ids_only():
    """No invented taxonomy values."""
    from start.risk import dimension_ids, object_kind_ids, stripe_ids
    stripes, dimensions, objects = set(stripe_ids()), set(dimension_ids()), set(object_kind_ids())
    for test_id, (_ctx, s, d, o, _why) in LEGACY_METADATA.items():
        assert set(s) <= stripes, (test_id, s)
        assert set(d) <= dimensions, (test_id, d)
        assert set(o) <= objects, (test_id, o)


def test_every_mapping_carries_a_justification():
    """No unexplained mappings."""
    for test_id, (_c, _s, _d, _o, why) in LEGACY_METADATA.items():
        assert len(why) > 60, test_id


def test_declined_mappings_are_recorded_with_reasons():
    """A decision not to map is part of the record."""
    assert DECLINED
    for _test, _dimension, reason in DECLINED:
        assert len(reason) > 30


def test_calibration_maps_to_calibration_not_discrimination():
    """A well-calibrated model can rank badly."""
    _c, _s, dimensions, _o, _w = LEGACY_METADATA["supervised.calibration"]
    assert dimensions == ("accuracy_calibration",)


def test_discrimination_does_not_claim_calibration():
    """AUC is invariant to monotone rescaling."""
    _c, _s, dimensions, _o, _w = LEGACY_METADATA["supervised.discrimination"]
    assert "accuracy_calibration" not in dimensions


def test_no_legacy_test_claims_outcomes_analysis():
    """No legacy engine produces post-deployment outcome evidence."""
    for test_id, (_c, _s, dimensions, _o, _w) in LEGACY_METADATA.items():
        assert "outcomes_analysis" not in dimensions, test_id


def test_no_legacy_test_claims_bias_fairness_or_benchmarking():
    for test_id, (_c, _s, dimensions, _o, _w) in LEGACY_METADATA.items():
        assert "bias_fairness" not in dimensions, test_id
        assert "benchmarking" not in dimensions, test_id


def test_integrated_gradients_is_scoped_to_differentiable_models():
    """Claiming it for a rules engine would be a false capability claim."""
    _c, _s, _d, objects, _w = LEGACY_METADATA["xai.integrated_gradients"]
    assert "deep_learning_model" in objects
    assert "rules_engine" not in objects and "scorecard" not in objects


def test_genai_uses_the_genai_stripe_not_model():
    _c, stripes, dimensions, _o, _w = LEGACY_METADATA["genai.citation_coverage"]
    assert stripes == ("ai_genai",)
    assert "explainability" not in dimensions


def test_backfill_is_idempotent_and_never_overwrites_a_declaration():
    first = apply_legacy_metadata()
    second = apply_legacy_metadata()
    assert second["applied"] == 0
    assert second["already_present"] >= first["already_present"]


def test_backfill_does_not_change_the_registry_size():
    before = len(list_tests())
    apply_legacy_metadata()
    assert len(list_tests()) == before == EXPECTED_TOTAL


def test_gate_b_family_counts_are_exact():
    """Each Gate-B family is pinned individually, so a missing registration in one
    family cannot be masked by an extra one in another."""
    counts = dict(Counter(s.family for s in list_tests()))
    for family, expected in GATE_B_FAMILY_COUNTS.items():
        assert counts.get(family) == expected, (family, counts.get(family), expected)
    assert counts.get("portfolio") == 10
    assert counts.get("attribution") == 6
    assert counts.get("traded_risk") == 8
    assert counts.get("covariance") == 3


def test_all_gate_b_tests_use_a_market_family_context():
    """market for everything except the two short-rate diffusion estimators."""
    short_rate_ids = {"traded_risk.cev_elasticity", "traded_risk.stanton_nonparametric"}
    for spec in list_tests():
        if spec.family in GATE_B_FAMILY_COUNTS:
            expected = "short_rate" if spec.test_id in short_rate_ids else "market"
            assert spec.context_type == expected, (spec.test_id, spec.context_type)


def test_exactly_two_short_rate_surfaces_exist():
    short_rate = {s.test_id for s in list_tests() if s.context_type == "short_rate"}
    assert short_rate == {"traded_risk.cev_elasticity",
                          "traded_risk.stanton_nonparametric"}


def test_gate_b_families_are_complete():
    """All four Gate-B families are now present; the registry is at its final size."""
    families = {s.family for s in list_tests()}
    assert {"portfolio", "attribution", "traded_risk", "covariance"} <= families


def test_no_gate_a_id_was_lost():
    """Gate B may only ADD. Every Gate-A family keeps exactly its closeout count."""
    counts = dict(Counter(s.family for s in list_tests()))
    for family, expected in GATE_A_FAMILY_COUNTS.items():
        assert counts.get(family) == expected, (family, counts.get(family), expected)
    assert sum(counts.get(f, 0) for f in GATE_A_FAMILY_COUNTS) == GATE_A_TOTAL


# ============================================================== coverage ==
def _plan(stripe="credit", kind="ml_model"):
    obj = RiskObject(object_id="d", kind=kind, name="D", owner="", materiality="high")
    return synthesise_plan(stripe_id=stripe, obj=obj, materiality="high")


@pytest.mark.parametrize(
    "dimension,expected",
    [
        ("accuracy_calibration", "supervised.calibration"),
        ("discriminatory_power", "supervised.discrimination"),
        ("overfitting_generalisation", "supervised.cohort_metrics_comparison"),
        ("monitoring", "preprocessing.feature_drift"),
    ],
)
def test_false_gaps_are_now_covered(dimension, expected):
    mapping = tests_for_plan(_plan(), context_type="tabular")
    assert expected in mapping.get(dimension, []), (dimension, mapping.get(dimension))


def test_outcomes_analysis_remains_a_genuine_gap():
    """F1 must remove FALSE gaps without suppressing TRUE ones."""
    coverage = coverage_for_plan(_plan(), context_type="tabular")
    outcomes = next(d for d in coverage.dimensions if d.dimension_id == "outcomes_analysis")
    assert not outcomes.covered
    assert outcomes in coverage.gaps


def test_explainability_is_covered_by_the_xai_engines():
    mapping = tests_for_plan(_plan(), context_type="tabular")
    candidates = set(mapping.get("explainability", []))
    assert {"xai.global_importance", "xai.feature_sensitivity"} <= candidates


def test_data_quality_is_covered_by_legacy_preprocessing():
    mapping = tests_for_plan(_plan(), context_type="tabular")
    candidates = set(mapping.get("data_quality_lineage", []))
    assert {"preprocessing.missingness", "preprocessing.duplicates"} <= candidates


def test_data_quality_mapping_is_not_over_broad():
    """Not every preprocessing test maps to every data-related dimension."""
    mapping = tests_for_plan(_plan(), context_type="tabular")
    assert "preprocessing.missingness" not in mapping.get("use_boundary", [])
    assert "preprocessing.target_leakage" not in mapping.get("data_quality_lineage", [])


def test_coverage_improves_but_is_not_total():
    """The point was accuracy, not making gaps disappear."""
    coverage = coverage_for_plan(_plan(), context_type="tabular")
    covered = sum(1 for d in coverage.dimensions if d.covered)
    assert covered > 9
    assert covered < len(coverage.dimensions)


# ==================================================== context vocabulary ==
def test_registered_context_vocabulary_is_canonical():
    """Only canonical values. Gate B introduces market; short_rate follows with B5."""
    assert {s.context_type for s in list_tests()} <= {"tabular", "market", "short_rate"}


def test_every_gate_a_test_remains_tabular():
    """Gate B must not have re-typed an existing surface."""
    for spec in list_tests():
        if spec.family in GATE_A_FAMILY_COUNTS:
            assert spec.context_type == "tabular", spec.test_id


def test_context_aliases_normalise_to_the_canonical_value():
    """One coherent contract; no hidden TestContext vs tabular mismatch."""
    assert normalise_context_type("TestContext") == "tabular"
    assert normalise_context_type("tabular") == "tabular"
    assert normalise_context_type(None) is None
    assert "testcontext" in CONTEXT_ALIASES


def test_a_tabular_plan_discovers_tabular_tests_under_either_spelling():
    canonical = tests_for_plan(_plan(), context_type="tabular")
    aliased = tests_for_plan(_plan(), context_type="TestContext")
    assert canonical == aliased
    assert any(v for v in canonical.values())


def test_an_unknown_context_matches_nothing_rather_than_everything():
    """A typo must be visible, not silently permissive."""
    mapping = tests_for_plan(_plan(), context_type="nonsense")
    assert all(not v for v in mapping.values())


# =========================================== legacy analytical invariance ==
def _supervised_ctx(n=400):
    rng = np.random.default_rng(7)
    y = rng.integers(0, 2, n)
    score = np.clip(y * 0.4 + rng.normal(0.3, 0.2, n), 0, 1)
    train = pd.DataFrame({"f": rng.normal(size=n), "y": y, "s": score})
    test = train.sample(150, random_state=1).reset_index(drop=True)
    return TestContext(train=train, test=test, target_column="y", score_column="s")


def _preprocessing_ctx(n=300):
    rng = np.random.default_rng(8)
    train = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n),
                          "c": list("AB") * (n // 2), "y": rng.integers(0, 2, n)})
    train.loc[:9, "a"] = np.nan
    test = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120),
                         "c": list("AB") * 60, "y": rng.integers(0, 2, 120)})
    return TestContext(train=train, test=test, target_column="y")


def _threshold_view(spec):
    """Compare thresholds by VALUE, not identity.

    ThresholdSpec has no __str__, so str() yields the object repr with its memory
    address — which differs on every call and would make this test fail for a reason
    that has nothing to do with F1.
    """
    return {
        "metric": getattr(spec, "metric", None),
        "warn": getattr(spec, "warn", None),
        "fail": getattr(spec, "fail", None),
        "direction": getattr(spec, "direction", None),
    }


def _analytical(result):
    """Only the fields F1 must not touch. TestSpec metadata is excluded by design."""
    return {
        "status": str(result.status),
        "metrics": dict(result.metrics),
        "params": dict(result.params),
        "interpretation": result.interpretation,
        "limitations": list(result.limitations),
        "artifacts": dict(result.artifacts),
        "thresholds": [_threshold_view(t) for t in result.thresholds],
    }


@pytest.mark.parametrize(
    "test_id,builder",
    [
        ("supervised.calibration", _supervised_ctx),
        ("supervised.discrimination", _supervised_ctx),
        ("supervised.classification_metrics", _supervised_ctx),
        ("supervised.cohort_metrics_comparison", _supervised_ctx),
        ("supervised.top_decile_lift", _supervised_ctx),
        ("preprocessing.missingness", _preprocessing_ctx),
        ("preprocessing.duplicates", _preprocessing_ctx),
        ("preprocessing.outliers", _preprocessing_ctx),
        ("preprocessing.feature_drift", _preprocessing_ctx),
        ("preprocessing.target_leakage", _preprocessing_ctx),
    ],
)
def test_legacy_analytical_output_is_unchanged_by_f1(test_id, builder):
    """Same test + same context + same params -> identical TestResult, before and after.

    Metadata on TestSpec may change; TestResult behaviour may not.
    """
    from dataclasses import replace

    from start.registry import _REGISTRY

    list_tests()
    spec = _REGISTRY[test_id]
    after = _analytical(spec.fn(builder()))

    stripped = replace(spec, risk_stripes=(), risk_dimensions=(), object_kinds=())
    before = _analytical(stripped.fn(builder()))
    assert before == after


def test_genai_analytical_output_is_unchanged_by_f1():
    from dataclasses import replace

    from start.registry import _REGISTRY

    list_tests()
    spec = _REGISTRY["genai.citation_coverage"]
    ctx = TestContext(train=pd.DataFrame({"a": [1.0]}),
                      extra={"generated_text": "AUC was 0.85 [EV-1]. Recall was 0.6."})
    after = _analytical(spec.fn(ctx))
    stripped = replace(spec, risk_stripes=(), risk_dimensions=(), object_kinds=())
    before = _analytical(stripped.fn(ctx))
    assert before == after


def test_xai_metadata_does_not_alter_registration_identity():
    from start.registry import _REGISTRY

    list_tests()
    for test_id in ("xai.global_importance", "xai.feature_sensitivity",
                    "xai.importance_stability", "xai.integrated_gradients"):
        spec = _REGISTRY[test_id]
        assert spec.test_id == test_id
        assert spec.family == "xai"
        assert callable(spec.fn)
