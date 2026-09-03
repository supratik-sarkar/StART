from __future__ import annotations

from collections import Counter
from dataclasses import FrozenInstanceError

import pytest

from start.core.schemas import EvidenceRecord, Status
from start.registry import list_tests
from start.validation.gate_b_evidence import (
    OVERALL_STATISTICAL_DISPOSITION,
    PROVENANCE,
    VERIFIED_B7_RESULTS,
    validation_results,
)


def _by_id():
    return {o.study_id: o for o in VERIFIED_B7_RESULTS}


# ------------------------------------------------------------- registry --
def test_registry_census_is_79():
    """Verify live registry contains exactly 79 tests (including 5 Gate-3 portfolio optimization tests)."""
    assert len(list_tests()) == 79


def test_no_duplicate_ids():
    ids = [s.test_id for s in list_tests()]
    assert len(ids) == len(set(ids))


def test_b8_creates_no_registered_family():
    families = {s.family for s in list_tests()}
    assert "governance" not in families
    assert "demo" not in families
    assert "validation" not in families


def test_final_family_census():
    assert dict(Counter(s.family for s in list_tests())) == {
        "genai": 1, "preprocessing": 21, "supervised": 5, "xai": 4, "eda": 6,
        "feature_engineering": 15, "portfolio": 10, "attribution": 6,
        "traded_risk": 8, "covariance": 3,
    }


# ------------------------------------------ validation status preservation --
def test_all_four_studies_are_represented():
    assert set(_by_id()) == {"var_size_power", "cev_consistency",
                             "stanton_bias", "regem_structural"}


def test_var_remains_pass():
    assert _by_id()["var_size_power"].status == Status.PASS


def test_regem_remains_pass():
    assert _by_id()["regem_structural"].status == Status.PASS


def test_cev_remains_fail_not_warn():
    """Understanding a cause does not discharge a criterion."""
    outcome = _by_id()["cev_consistency"]
    assert outcome.status == Status.FAIL
    assert outcome.status != Status.WARN


def test_stanton_remains_fail_not_warn():
    outcome = _by_id()["stanton_bias"]
    assert outcome.status == Status.FAIL
    assert outcome.status != Status.WARN


def test_cev_coverage_value_is_the_verified_one():
    criteria = {c["name"]: c for c in _by_id()["cev_consistency"].criteria}
    assert criteria["coverage_gamma_0_0"]["observed"] == 0.635
    assert criteria["coverage_gamma_0_0"]["passed"] is False


def test_cev_consistency_passed_while_coverage_failed():
    """Both facts must survive; neither may be collapsed into the other."""
    criteria = {c["name"]: c for c in _by_id()["cev_consistency"].criteria}
    assert criteria["consistency_ratio_gamma_0_0"]["passed"] is True
    assert criteria["coverage_gamma_0_0"]["passed"] is False


def test_stanton_bias_passed_while_sign_failed():
    criteria = {c["name"]: c for c in _by_id()["stanton_bias"].criteria}
    assert criteria["bias_improvement_ratio"]["passed"] is True
    assert criteria["max_wrong_sign_rate_nonzero_drift"]["passed"] is False


def test_failures_survive_conversion_to_test_result():
    results = {r.test_id: r for r in validation_results()}
    assert results["validation.cev_consistency"].status == Status.FAIL
    assert results["validation.stanton_bias"].status == Status.FAIL
    assert results["validation.var_size_power"].status == Status.PASS
    assert results["validation.regem_structural"].status == Status.PASS


def test_failures_survive_conversion_to_evidence_record():
    for result in validation_results():
        record = EvidenceRecord.from_result(result, run_id="RUN-TEST")
        assert record.status == result.status
        assert record.metrics["n_criteria_failed"] == result.metrics["n_criteria_failed"]


def test_provenance_is_recorded_on_every_validation_result():
    """Reproduced validation evidence, not a freshly simulated demo outcome."""
    for result in validation_results():
        assert result.params["provenance"] == PROVENANCE
        assert "NOT freshly simulated" in result.params["provenance"]


def test_overall_disposition_is_not_green():
    assert "PARTIAL" in OVERALL_STATISTICAL_DISPOSITION
    assert "2 of 4" in OVERALL_STATISTICAL_DISPOSITION


def test_cev_is_never_described_as_validated():
    blob = " ".join(_by_id()["cev_consistency"].limitations).lower()
    assert "not fully validated" in blob


def test_stanton_limitations_name_the_resolution_limit():
    blob = " ".join(_by_id()["stanton_bias"].limitations)
    assert "signal-to-noise" in blob
    assert "Nothing was changed after observing the failure" in blob


def test_regem_pass_claims_no_dominance():
    blob = " ".join(_by_id()["regem_structural"].limitations)
    assert "No dominance criterion was imposed and none is claimed" in blob


def test_var_pass_claims_no_model_correctness():
    blob = " ".join(_by_id()["var_size_power"].limitations)
    assert "does NOT establish that any" in blob


def test_no_override_route_exists():
    """The outcomes are frozen dataclasses; there is no setter to soften a status."""
    outcome = _by_id()["cev_consistency"]
    with pytest.raises(FrozenInstanceError):
        outcome.status = Status.WARN


# ------------------------------------------------------------ narrative --
def test_narrative_reports_both_failures_and_claims_no_blanket_pass():
    from scripts.demo_market import build_narrative

    results = validation_results()
    records = [EvidenceRecord.from_result(r, run_id="RUN-TEST") for r in results]
    text = build_narrative(records)
    lowered = text.lower()

    assert "all validation passed" not in lowered
    assert "fully validated" not in lowered.replace("not fully validated", "")
    assert "failed" in lowered
    assert "0.6350" in text or "0.635" in text
    assert "not fully validated" in lowered


def test_narrative_cites_evidence_for_quantitative_claims():
    from scripts.demo_market import build_narrative

    records = [EvidenceRecord.from_result(r, run_id="RUN-TEST")
               for r in validation_results()]
    text = build_narrative(records)
    assert "[EV-" in text
    assert text.count("[EV-") >= 4


# ------------------------------------------------------------ evidence --
def test_evidence_record_schema_is_unchanged():
    """B8 adds no field to the core schema."""
    result = validation_results()[0]
    record = EvidenceRecord.from_result(result, run_id="RUN-TEST")
    for field in ("test_id", "test_name", "model_id", "dataset_id", "run_id",
                  "params", "metrics", "status", "evidence_id"):
        assert hasattr(record, field)


def test_no_large_payload_in_validation_metrics():
    for result in validation_results():
        for value in result.metrics.values():
            assert not hasattr(value, "shape")


def test_validation_metrics_carry_criterion_detail():
    metrics = {r.test_id: r.metrics for r in validation_results()}
    cev = metrics["validation.cev_consistency"]
    assert cev["observed.coverage_gamma_0_0"] == 0.635
    assert cev["required.coverage_gamma_0_0"] == "in [0.90, 0.98]"
    assert cev["passed.coverage_gamma_0_0"] is False
    assert cev["configuration_hash"] == "a9b387fb2905aa48fa3732cee79d749a"


# -------------------------------------------------- claim binding & citations --
def test_narrative_citations_have_single_ev_prefix():
    import re

    from scripts.demo_market import build_narrative

    records = [EvidenceRecord.from_result(r, run_id="RUN-TEST")
               for r in validation_results()]
    text = build_narrative(records)
    # Must NOT have double EV- prefix
    assert not re.search(r"\[EV-EV-", text)
    # Must match single EV- prefix pattern
    citations = re.findall(r"\[(EV-[A-Za-z0-9_-]+)\]", text)
    assert len(citations) >= 4
    for c in citations:
        assert not c.startswith("EV-EV-")


def test_citation_scoped_binding_wrong_record_same_number():
    """Evidence A has 0.10, Evidence B has 0.10. Citing B binds ONLY to B."""
    from start.attestation.claims import bind_claims, extract_claims

    rec_a = {
        "evidence_id": "EV-PORTFOLIO-01",
        "test_id": "portfolio.risk_statistics",
        "metrics": {"some_ratio": 0.10},
    }
    rec_b = {
        "evidence_id": "EV-STANTON-02",
        "test_id": "validation.stanton_bias",
        "metrics": {"required.max_wrong_sign_rate": 0.10},
    }
    claims = extract_claims("Stanton required maximum was 0.10 [EV-STANTON-02].")
    res = bind_claims(claims, [rec_a, rec_b])
    assert res.grounding_rate == 1.0
    assert len(res.bound) == 1
    assert res.bound[0]["evidence_id"] == "EV-STANTON-02"
    assert res.bound[0]["test_id"] == "validation.stanton_bias"
    assert res.bound[0]["bound_to"] == "validation.stanton_bias.metrics.required.max_wrong_sign_rate"


def test_citation_scoped_binding_zero_collision():
    """Multiple records have 0.0. Citing CEV with gamma=0 binds ONLY to CEV."""
    from start.attestation.claims import bind_claims, extract_claims

    rec_hist = {
        "evidence_id": "EV-HIST-01",
        "test_id": "portfolio.historical_returns",
        "metrics": {"mean_periodic_return": 0.0},
    }
    rec_cev = {
        "evidence_id": "EV-CEV-02",
        "test_id": "validation.cev_consistency",
        "metrics": {"required.gamma_0": 0.0, "observed.coverage_gamma_0_0": 0.635},
    }
    claims = extract_claims("at gamma = 0, where empirical coverage was 0.6350 [EV-CEV-02].")
    res = bind_claims(claims, [rec_hist, rec_cev])
    assert len(res.bound) == 2
    for b in res.bound:
        assert b["evidence_id"] == "EV-CEV-02"
        assert b["test_id"] == "validation.cev_consistency"


def test_citation_scoped_binding_scale_collision():
    """Stanton required 0.10 must not scale to match min_ess_threshold = 10.0."""
    from start.attestation.claims import bind_claims, extract_claims

    rec_stanton = {
        "evidence_id": "EV-STANTON-ANALYTIC",
        "test_id": "traded_risk.stanton_nonparametric",
        "metrics": {"min_ess_threshold": 10.0},
    }
    claims = extract_claims("reaching 0.4750 against a required maximum of 0.10 [EV-STANTON-ANALYTIC].")
    res = bind_claims(claims, [rec_stanton])
    # 0.10 is NOT in rec_stanton (only 10.0 is) -> must fail closed and remain unbound
    assert res.unbound_count >= 1
    unbound_values = [u["value"] for u in res.unbound]
    assert 0.10 in unbound_values


def test_citation_not_found_fails_closed():
    from start.attestation.claims import bind_claims, extract_claims

    rec = {
        "evidence_id": "EV-EXISTING",
        "test_id": "some.test",
        "metrics": {"value": 0.5},
    }
    claims = extract_claims("The value was 0.50 [EV-NONEXISTENT].")
    res = bind_claims(claims, [rec])
    assert res.grounding_rate == 0.0
    assert len(res.unbound) == 1
    assert res.unbound[0]["bound_to"] is None


def test_value_absent_from_cited_record_fails_closed():
    from start.attestation.claims import bind_claims, extract_claims

    rec_a = {
        "evidence_id": "EV-CITED",
        "test_id": "test.a",
        "metrics": {"val_a": 0.1},
    }
    rec_b = {
        "evidence_id": "EV-OTHER",
        "test_id": "test.b",
        "metrics": {"val_b": 0.9},
    }
    claims = extract_claims("The value was 0.90 [EV-CITED].")
    res = bind_claims(claims, [rec_a, rec_b])
    # Even though 0.9 is in rec_b, citing rec_a fails closed
    assert res.grounding_rate == 0.0
    assert len(res.unbound) == 1
    assert res.unbound[0]["bound_to"] is None


def test_all_demo_market_claims_semantically_bound():
    from scripts.demo_market import build_contexts, build_narrative, run_analytics

    from start.attestation.claims import bind_claims, extract_claims

    world, market, incomplete = build_contexts(42)
    short_rate = world.short_rate_context()
    results = run_analytics(market, incomplete, short_rate)
    results.extend(validation_results())
    records = [EvidenceRecord.from_result(r, run_id="RUN-1") for r in results]

    narrative = build_narrative(records)
    claims = extract_claims(narrative)
    res = bind_claims(claims, records)

    assert len(claims) == 16
    assert res.unbound_count == 0
    assert res.grounding_rate == 1.0

    # Verify key semantic bindings
    by_surface = {b["surface"]: b for b in res.bound}
    assert by_surface["0.00e+00"]["test_id"] == "attribution.return_attribution"
    assert "reconciliation" in by_surface["0.00e+00"]["bound_to"]

    assert by_surface["0.10"]["test_id"] == "validation.stanton_bias"
    assert "max_wrong_sign_rate" in by_surface["0.10"]["bound_to"]

    assert by_surface["0.6350"]["test_id"] == "validation.cev_consistency"
    assert "coverage_gamma_0_0" in by_surface["0.6350"]["bound_to"]

