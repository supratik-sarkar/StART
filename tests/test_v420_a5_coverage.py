"""A5 — risk-plan to registered-test coverage."""
from __future__ import annotations

import pytest

from start.risk.coverage import coverage_for_plan, tests_for_plan
from start.risk.objects import RiskObject
from start.risk.plan import synthesise_plan


def _plan(stripe="credit", kind="ml_model", materiality="high"):
    obj = RiskObject(object_id="demo", kind=kind, name="Demo", owner="",
                     materiality=materiality)
    return synthesise_plan(stripe_id=stripe, obj=obj, materiality=materiality)


def test_mapping_is_dimension_to_test_ids():
    mapping = tests_for_plan(_plan())
    assert isinstance(mapping, dict)
    assert all(isinstance(v, list) for v in mapping.values())


def test_every_planned_dimension_appears():
    plan = _plan()
    coverage = coverage_for_plan(plan)
    assert len(coverage.dimensions) == len(plan.planned)
    assert {d.dimension_id for d in coverage.dimensions} == {
        p.dimension_id for p in plan.planned
    }


def test_matching_is_on_declared_metadata_not_inferred_from_names():
    """A declared mapping is a statement the author can be held to; an inferred one
    drifts silently as names change."""
    coverage = coverage_for_plan(_plan())
    covered = [d for d in coverage.dimensions if d.covered]
    assert covered
    from start.registry import list_tests
    specs = {s.test_id: s for s in list_tests()}
    for dimension in covered:
        for test_id in dimension.test_ids:
            assert dimension.dimension_id in specs[test_id].risk_dimensions


def test_uncovered_dimensions_are_reported_not_hidden():
    """Reporting only matches would make the system look complete by omission."""
    coverage = coverage_for_plan(_plan())
    assert coverage.uncovered
    assert coverage.as_dict()["n_uncovered"] == len(coverage.uncovered)


def test_required_gaps_are_distinguished_from_optional_ones():
    coverage = coverage_for_plan(_plan())
    for gap in coverage.gaps:
        assert gap.required and not gap.test_ids
    assert all(d.required for d in coverage.gaps)


def test_context_filter_excludes_and_records_unavailable():
    plan = _plan()
    unfiltered = coverage_for_plan(plan, context_type=None)
    tabular = coverage_for_plan(plan, context_type="tabular")
    assert unfiltered.context_type == "any"
    assert tabular.context_type == "tabular"
    for dimension in tabular.dimensions:
        assert all(t not in dimension.test_ids for t in dimension.unavailable_test_ids)


def test_market_filter_surfaces_only_market_tests():
    """Superseded premise: this asserted emptiness before Gate B landed. Now a market
    filter must surface market surfaces and NEVER a tabular one."""
    from start.registry import list_tests
    specs = {s.test_id: s for s in list_tests()}
    coverage = coverage_for_plan(_plan(stripe="market",
                                       kind="deterministic_calculator"),
                                 context_type="market")
    surfaced = {t for d in coverage.dimensions for t in d.test_ids}
    assert surfaced, "Gate B market tests should now be candidates"
    assert all(specs[t].context_type == "market" for t in surfaced)


def test_tabular_filter_never_surfaces_a_market_test():
    from start.registry import list_tests
    specs = {s.test_id: s for s in list_tests()}
    coverage = coverage_for_plan(_plan(stripe="credit"), context_type="tabular")
    surfaced = {t for d in coverage.dimensions for t in d.test_ids}
    assert all(specs[t].context_type == "tabular" for t in surfaced)


def test_stripe_specific_tests_are_preferred():
    """A drift test written for the market stripe and one for the model stripe answer
    the same dimension differently."""
    from start.registry import list_tests
    coverage = coverage_for_plan(_plan(stripe="credit"))
    specs = {s.test_id: s for s in list_tests()}
    for dimension in coverage.dimensions:
        if not dimension.test_ids:
            continue
        stripes = [specs[t].risk_stripes for t in dimension.test_ids]
        if any("credit" in s for s in stripes):
            assert all("credit" in s for s in stripes)


def test_coverage_hash_is_stable():
    assert coverage_for_plan(_plan()).coverage_hash() == coverage_for_plan(_plan()).coverage_hash()


def test_coverage_hash_changes_with_the_plan():
    a = coverage_for_plan(_plan(stripe="credit")).coverage_hash()
    b = coverage_for_plan(_plan(stripe="market")).coverage_hash()
    assert a != b


def test_summary_states_that_coverage_is_not_discharge():
    """Three candidate tests do not satisfy a dimension."""
    lines = "\n".join(coverage_for_plan(_plan()).summary_lines())
    assert "COULD supply evidence" in lines
    assert "not that the dimension is discharged" in lines


def test_summary_names_the_required_gaps():
    coverage = coverage_for_plan(_plan())
    lines = "\n".join(coverage.summary_lines())
    if coverage.gaps:
        assert "REQUIRED dimension(s) have no candidate test" in lines
        for gap in coverage.gaps:
            assert gap.dimension_id in lines


def test_as_dict_is_evidence_shaped():
    block = coverage_for_plan(_plan()).as_dict()
    for key in ("stripe", "object_kind", "n_dimensions", "n_covered",
                "n_required_gaps", "dimensions"):
        assert key in block


@pytest.mark.parametrize("stripe", ["credit", "model", "market", "fraud"])
def test_coverage_runs_for_every_stripe(stripe):
    coverage = coverage_for_plan(_plan(stripe=stripe))
    assert coverage.dimensions
    assert coverage.stripe_id == stripe
