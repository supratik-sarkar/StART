"""Risk core: stripes, objects, dimensions, plan synthesis, control coverage.

The central claim these tests defend is that StART is not an ML tool with a
governance wrapper. A deterministic pricing calculator, a spreadsheet and a
vendor black box must each produce a *coherent, different, non-empty* review
plan — and none of them should be handled by pretending they are models.
"""

from __future__ import annotations

import pytest

from start.risk import (
    DIMENSIONS,
    Applicability,
    RiskObject,
    applicability,
    coverage_report,
    dimension_ids,
    object_kind_ids,
    stripe,
    stripe_ids,
    synthesise_plan,
)


# --------------------------------------------------------------------------- #
# Catalogue integrity
# --------------------------------------------------------------------------- #
def test_every_stripe_references_known_dimensions() -> None:
    known = set(dimension_ids())
    for stripe_id in stripe_ids():
        spec = stripe(stripe_id)
        for field in (spec.mandatory_dimensions, spec.heightened_dimensions):
            unknown = set(field) - known
            assert not unknown, f"stripe {stripe_id!r} references unknown dimensions: {unknown}"


def test_every_object_kind_references_known_dimensions() -> None:
    from start.risk import object_kind

    known = set(dimension_ids())
    for kind_id in object_kind_ids():
        unknown = set(object_kind(kind_id).always_required) - known
        assert not unknown, f"object kind {kind_id!r} references unknown dimensions: {unknown}"


def test_dimension_substitutes_are_themselves_dimensions() -> None:
    """A burden cannot transfer to something that does not exist."""
    known = set(DIMENSIONS)
    for dim_id, dim in DIMENSIONS.items():
        unknown = set(dim.substitutes) - known
        assert not unknown, f"dimension {dim_id!r} substitutes to unknown: {unknown}"


def test_no_dimension_substitutes_to_itself() -> None:
    for dim_id, dim in DIMENSIONS.items():
        assert dim_id not in dim.substitutes, f"{dim_id} substitutes to itself"


def test_the_catalogue_covers_more_than_machine_learning() -> None:
    kinds = set(object_kind_ids())
    non_ml = {
        "deterministic_calculator",
        "rules_engine",
        "spreadsheet_euc",
        "expert_judgment_overlay",
        "vendor_model",
    }
    assert non_ml <= kinds, f"missing non-ML object kinds: {non_ml - kinds}"
    assert len(stripe_ids()) >= 10, "the stripe taxonomy should span a bank, not one desk"


# --------------------------------------------------------------------------- #
# Applicability and burden transfer
# --------------------------------------------------------------------------- #
def test_calculator_cannot_overfit_and_says_so() -> None:
    obj = RiskObject(object_id="C-1", kind="deterministic_calculator")
    verdict = applicability(obj, "overfitting_generalisation")
    assert verdict.applicability is not Applicability.APPLICABLE
    assert "is_fitted" in verdict.reason


def test_burden_transfers_rather_than_disappearing() -> None:
    """The property that separates this from writing 'N/A' in a template."""
    obj = RiskObject(object_id="V-1", kind="vendor_model")
    verdict = applicability(obj, "discriminatory_power")

    assert verdict.applicability is Applicability.SUBSTITUTED
    assert verdict.burden_transferred_to, "a substitution must name who inherits the burden"

    plan = synthesise_plan(stripe_id="financial_crime", obj=obj)
    for inheritor in verdict.burden_transferred_to:
        planned = {p.dimension_id: p for p in plan.planned}
        assert inheritor in planned, f"{inheritor} inherited burden but is not planned"
        assert planned[inheritor].required, (
            f"{inheritor} inherited burden from discriminatory_power and must be mandatory"
        )
        assert "discriminatory_power" in planned[inheritor].inherited_burden_from


def test_capability_override_changes_the_plan() -> None:
    """A vendor AML engine does affect individuals, whatever the kind default says."""
    base = RiskObject(object_id="V-2", kind="vendor_model")
    overridden = RiskObject(
        object_id="V-2", kind="vendor_model", capability_overrides={"affects_individuals": True}
    )
    assert applicability(base, "bias_fairness").applicability is Applicability.NOT_APPLICABLE
    assert applicability(overridden, "bias_fairness").applicability is Applicability.APPLICABLE


def test_unknown_capability_override_is_rejected() -> None:
    obj = RiskObject(object_id="X", kind="ml_model", capability_overrides={"is_magic": True})
    with pytest.raises(KeyError, match="Unknown capability override"):
        obj.capabilities()


# --------------------------------------------------------------------------- #
# Plan synthesis
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kind", sorted(object_kind_ids()))
@pytest.mark.parametrize("stripe_id", ["credit", "market", "financial_crime", "ai_genai"])
def test_every_combination_produces_a_usable_plan(stripe_id: str, kind: str) -> None:
    """No stripe/object pairing may produce an empty or all-optional plan."""
    plan = synthesise_plan(stripe_id=stripe_id, obj=RiskObject(object_id="T", kind=kind, materiality="high"))
    assert plan.planned, f"{stripe_id}/{kind} produced no planned dimensions"
    assert plan.required_dimension_ids(), f"{stripe_id}/{kind} produced nothing mandatory"


def test_no_dimension_is_silently_dropped() -> None:
    """Every dimension is planned, substituted, or excluded with a stated reason."""
    plan = synthesise_plan(
        stripe_id="valuation", obj=RiskObject(object_id="P-1", kind="deterministic_calculator")
    )
    accounted = set(plan.all_dimension_ids()) | {e["dimension"] for e in plan.excluded}
    assert accounted == set(dimension_ids()), f"unaccounted dimensions: {set(dimension_ids()) - accounted}"
    for excluded in plan.excluded:
        assert excluded["reason"], "an exclusion without a reason is just a gap"


def test_plan_hash_is_stable_and_discriminating() -> None:
    obj = RiskObject(object_id="M-1", kind="ml_model", materiality="medium")
    a = synthesise_plan(stripe_id="credit", obj=obj)
    b = synthesise_plan(stripe_id="credit", obj=obj)
    assert a.plan_hash() == b.plan_hash(), "identical inputs must hash identically"

    c = synthesise_plan(stripe_id="credit", obj=obj, materiality="high")
    assert a.plan_hash() != c.plan_hash(), "a scope change must change the hash"

    d = synthesise_plan(stripe_id="fraud", obj=obj, materiality="medium")
    assert a.plan_hash() != d.plan_hash(), "a different stripe must change the hash"


def test_high_materiality_promotes_heightened_dimensions() -> None:
    obj = RiskObject(object_id="M-2", kind="ml_model")
    medium = synthesise_plan(stripe_id="credit", obj=obj, materiality="medium")
    high = synthesise_plan(stripe_id="credit", obj=obj, materiality="high")
    assert set(medium.required_dimension_ids()) <= set(high.required_dimension_ids())

    # A heightened dimension that the object cannot support is excluded, not
    # promoted — bias_fairness against an ml_model with no individual-level
    # impact, for example. Promotion applies to what is actually planned.
    planned = set(high.all_dimension_ids())
    heightened_and_planned = set(stripe("credit").heightened_dimensions) & planned
    assert heightened_and_planned <= set(high.required_dimension_ids())
    assert heightened_and_planned, "high materiality should promote something"


def test_unknown_stripe_and_materiality_are_rejected() -> None:
    obj = RiskObject(object_id="M-3", kind="ml_model")
    with pytest.raises(KeyError, match="Unknown risk stripe"):
        synthesise_plan(stripe_id="not_a_stripe", obj=obj)
    with pytest.raises(ValueError, match="Unknown materiality"):
        synthesise_plan(stripe_id="credit", obj=obj, materiality="critical")


# --------------------------------------------------------------------------- #
# Control coverage
# --------------------------------------------------------------------------- #
def test_coverage_reports_unmapped_frameworks_rather_than_claiming_them() -> None:
    report = coverage_report(["sr_11_7", "bsa_aml"], {"conceptual_soundness"})
    assert "bsa_aml" in report["unmapped_frameworks"]
    assert all(f["framework_id"] != "bsa_aml" for f in report["frameworks"])


def test_requires_all_expectations_need_every_dimension() -> None:
    partial = coverage_report(["sr_11_7"], {"conceptual_soundness"})
    full = coverage_report(["sr_11_7"], {"conceptual_soundness", "assumption_validity"})

    def covered(report: dict, expectation_id: str) -> bool:
        for framework in report["frameworks"]:
            for row in framework["expectations"]:
                if row["expectation_id"] == expectation_id:
                    return bool(row["covered"])
        raise AssertionError(f"{expectation_id} not found")

    assert not covered(partial, "sr11_7.dev.conceptual")
    assert covered(full, "sr11_7.dev.conceptual")


def test_empty_examination_covers_nothing() -> None:
    report = coverage_report(["sr_11_7", "eu_ai_act"], set())
    assert report["expectations_covered"] == 0
    assert report["overall_coverage_ratio"] == 0.0


def test_coverage_rejects_unknown_dimensions() -> None:
    with pytest.raises(KeyError, match="Unknown dimension"):
        coverage_report(["sr_11_7"], {"made_up_dimension"})


def test_coverage_carries_its_own_caveat() -> None:
    """Coverage is examination, not adequacy, and must say so wherever it appears."""
    report = coverage_report(["sr_11_7"], {"monitoring"})
    assert "not an assessment" in report["caveat"]
    assert report["mapping_version"]
