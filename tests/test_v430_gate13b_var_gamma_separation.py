"""Tests for Gate 13B: VaR Confidence vs Statistical Significance Semantic Separation.

Frozen Gate-5 Semantic Invariants:
1. confidence = 0.99 (VaR model confidence / quantile level).
2. alpha_var = 0.01 (VaR model tail probability / null exception probability).
3. gamma_test = 0.05 (Statistical hypothesis-test significance level).
4. gamma_test must NEVER be derived from confidence, 1 - confidence, alpha_var, or generic alpha.
5. Missing gamma_test must render gamma=N/A.
6. Stored decision is authoritative (no p_value <= gamma_test in presentation).
7. Grounding enforces semantic distinction: percentage claims bind only to compatible concepts.
"""

from __future__ import annotations

from start.attestation.claims import (
    Claim,
    _match_candidates_in_fields,
    extract_claims,
)
from start.core.schemas import EvidenceRecord, Status, TestResult
from start.review.architecture import ReviewDomain
from start.review.evidence_view import (
    CheckpointMetricRef,
    build_checkpoint_evidence_view,
)
from start.review.tables import build_var_tail_table


def _make_separation_record() -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id="EV-KUPIEC-SEP",
        test_id="traded_risk.var_kupiec_pof",
        test_name="Kupiec proportion-of-failures test",
        status=Status.RECORDED,
        model_id="M-MARKET",
        dataset_id="D-MARKET",
        run_id="RUN-SEP",
        params={"pnl_source": "actual", "gamma_test": 0.05, "confidence": 0.99},
        metrics={
            "n_observations": 1000,
            "n_exceptions": 6,
            "lr_uc": 1.8862324083,
            "p_value": 0.1696274814,
            "confidence": 0.99,
            "var_confidence": 0.99,
            "alpha_var": 0.01,
            "expected_probability": 0.01,
            "gamma_test": 0.05,
            "statistical_gamma_test": 0.05,
            "statistical_criterion_source": "STATISTICAL_TEST_SPECIFICATION",
            "critical_value": 3.8414588207,
            "rejected": False,
        },
    )


def test_confidence_alpha_gamma_separation() -> None:
    """1. Coexistence of confidence=0.99, alpha_var=0.01, gamma_test=0.05 renders gamma=0.05."""
    rec = _make_separation_record()
    table = build_var_tail_table([rec])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("gamma=0.05" in c for c in crit_cells), f"Expected gamma=0.05 in {crit_cells}"
    assert not any("gamma=0.99" in c for c in crit_cells)
    assert not any("gamma=0.01" in c for c in crit_cells)


def test_missing_gamma_renders_na() -> None:
    """2. When gamma_test is missing, table renders gamma=N/A; does NOT derive from 1 - confidence."""
    rec = _make_separation_record()
    m = dict(rec.metrics)
    del m["gamma_test"]
    del m["statistical_gamma_test"]
    if "alpha" in m:
        del m["alpha"]
    rec_missing = rec.model_copy(update={"metrics": m})

    table = build_var_tail_table([rec_missing])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("gamma=N/A" in c for c in crit_cells), f"Expected gamma=N/A in {crit_cells}"
    assert not any("gamma=0.01" in c for c in crit_cells)
    assert not any("gamma=0.99" in c for c in crit_cells)
    assert not any("gamma=0.05" in c for c in crit_cells)


def test_confidence_mutation_does_not_change_gamma() -> None:
    """3. Mutating confidence: 0.99 -> 0.975 preserves displayed gamma=0.05."""
    rec = _make_separation_record()
    m = dict(rec.metrics)
    m["confidence"] = 0.975
    m["var_confidence"] = 0.975
    rec_mut = rec.model_copy(update={"metrics": m})

    table = build_var_tail_table([rec_mut])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("gamma=0.05" in c for c in crit_cells)
    assert not any("gamma=0.975" in c or "gamma=0.98" in c for c in crit_cells)
    assert not any("gamma=0.025" in c for c in crit_cells)


def test_alpha_var_mutation_does_not_change_gamma() -> None:
    """4. Mutating alpha_var: 0.01 -> 0.025 preserves displayed gamma=0.05."""
    rec = _make_separation_record()
    m = dict(rec.metrics)
    m["alpha_var"] = 0.025
    m["expected_probability"] = 0.025
    rec_mut = rec.model_copy(update={"metrics": m})

    table = build_var_tail_table([rec_mut])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("gamma=0.05" in c for c in crit_cells)
    assert not any("gamma=0.025" in c or "gamma=0.03" in c for c in crit_cells)


def test_explicit_gamma_mutation_changes_gamma() -> None:
    """5. Explicitly mutating gamma_test: 0.05 -> 0.025 displays gamma=0.03."""
    rec = _make_separation_record()
    m = dict(rec.metrics)
    m["gamma_test"] = 0.025
    m["statistical_gamma_test"] = 0.025
    rec_mut = rec.model_copy(update={"metrics": m})

    table = build_var_tail_table([rec_mut])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("gamma=0.03" in c for c in crit_cells)
    assert not any("gamma=0.05" in c for c in crit_cells)


def test_stored_decision_unchanged_by_ui() -> None:
    """6. Stored decision rejected=False renders DO_NOT_REJECT regardless of p_value vs gamma."""
    rec = _make_separation_record()
    m = dict(rec.metrics)
    m["p_value"] = 0.0001
    m["rejected"] = False
    m["gamma_test"] = 0.05
    rec_mut = rec.model_copy(update={"metrics": m})

    table = build_var_tail_table([rec_mut])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("DO_NOT_REJECT" in c for c in crit_cells)
    assert not any("(REJECT at" in c for c in crit_cells)


def test_no_p_le_gamma_recomputation() -> None:
    """7. Presentation does not recompute p <= gamma."""
    rec = _make_separation_record()
    m = dict(rec.metrics)
    m["p_value"] = 0.9999
    m["rejected"] = True
    rec_mut = rec.model_copy(update={"metrics": m})

    table = build_var_tail_table([rec_mut])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("REJECT at gamma=0.05" in c for c in crit_cells)
    assert not any("DO_NOT_REJECT" in c for c in crit_cells)


def test_grounding_semantic_identity_separation() -> None:
    """8. Grounding distinguishes test significance vs VaR confidence vs VaR tail probability."""
    fields = {
        "metrics.confidence": 0.95,
        "metrics.alpha_var": 0.05,
        "metrics.gamma_test": 0.05,
    }

    # Claim asserting test significance must bind to gamma_test, NOT alpha_var
    sig_claim = Claim(value=5.0, surface="5%", unit="%", position=0, context="Hypothesis test significance level is 5%")
    matched_sig = _match_candidates_in_fields(
        candidates={0.05, 5.0},
        fields=fields,
        tolerance=1e-4,
        context="Hypothesis test significance level is 5%",
        claim=sig_claim,
    )
    assert matched_sig is not None
    assert matched_sig[0] == "metrics.gamma_test"

    # Claim asserting tail probability must bind to alpha_var, NOT gamma_test
    tail_claim = Claim(value=5.0, surface="5%", unit="%", position=0, context="VaR tail probability is 5%")
    matched_tail = _match_candidates_in_fields(
        candidates={0.05, 5.0},
        fields=fields,
        tolerance=1e-4,
        context="VaR tail probability is 5%",
        claim=tail_claim,
    )
    assert matched_tail is not None
    assert matched_tail[0] == "metrics.alpha_var"

    # If gamma_test is missing, claim asserting test significance must NOT fall back to alpha_var
    fields_no_gamma = {
        "metrics.confidence": 0.95,
        "metrics.alpha_var": 0.05,
    }
    matched_forbidden = _match_candidates_in_fields(
        candidates={0.05, 5.0},
        fields=fields_no_gamma,
        tolerance=1e-4,
        context="Hypothesis test significance level is 5%",
        claim=sig_claim,
    )
    assert matched_forbidden is None


def test_collision_safe_refs_preserve_all_three() -> None:
    """9. CheckpointEvidenceView preserves confidence, alpha_var, gamma_test under distinct refs."""
    rec = _make_separation_record()
    view = build_checkpoint_evidence_view(
        checkpoint_title="VaR Backtesting",
        checkpoint_description="VaR review",
        domains=(ReviewDomain.MARKET,),
        records=[rec],
    )
    ref_conf = CheckpointMetricRef(rec.evidence_id, rec.test_id, "confidence")
    ref_alpha = CheckpointMetricRef(rec.evidence_id, rec.test_id, "alpha_var")
    ref_gamma = CheckpointMetricRef(rec.evidence_id, rec.test_id, "gamma_test")

    assert ref_conf in view.metrics_by_ref
    assert ref_alpha in view.metrics_by_ref
    assert ref_gamma in view.metrics_by_ref

    assert view.metrics_by_ref[ref_conf].value == 0.99
    assert view.metrics_by_ref[ref_alpha].value == 0.01
    assert view.metrics_by_ref[ref_gamma].value == 0.05

    # Direct evidence and path lookups
    assert view.metrics_by_evidence_and_path[(rec.evidence_id, "confidence")].value == 0.99
    assert view.metrics_by_evidence_and_path[(rec.evidence_id, "alpha_var")].value == 0.01
    assert view.metrics_by_evidence_and_path[(rec.evidence_id, "gamma_test")].value == 0.05


def test_christoffersen_independence_separation() -> None:
    """11. Christoffersen independence test separates confidence, alpha_var, gamma_test."""
    rec = EvidenceRecord(
        evidence_id="EV-IND-SEP",
        test_id="traded_risk.var_christoffersen_independence",
        test_name="Christoffersen independence test",
        status=Status.RECORDED,
        model_id="M-MARKET",
        dataset_id="D-MARKET",
        run_id="RUN-SEP",
        params={"pnl_source": "actual", "gamma_test": 0.05},
        metrics={
            "n_observations": 1000,
            "n_exceptions": 6,
            "lr_ind": 0.0725,
            "p_value": 0.7877,
            "confidence": 0.99,
            "alpha_var": 0.01,
            "gamma_test": 0.05,
            "statistical_gamma_test": 0.05,
            "rejected": False,
        },
    )
    table = build_var_tail_table([rec])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("gamma=0.05" in c for c in crit_cells)

    # Missing gamma renders N/A
    m = dict(rec.metrics)
    del m["gamma_test"]
    del m["statistical_gamma_test"]
    rec_na = rec.model_copy(update={"metrics": m})
    table_na = build_var_tail_table([rec_na])
    crit_cells_na = [str(c) for c in table_na.columns[3]._cells]
    assert any("gamma=N/A" in c for c in crit_cells_na)


def test_conditional_coverage_separation() -> None:
    """12. Joint conditional coverage separates confidence, alpha_var, gamma_test."""
    rec = EvidenceRecord(
        evidence_id="EV-CC-SEP",
        test_id="traded_risk.var_christoffersen_conditional",
        test_name="Christoffersen conditional coverage test",
        status=Status.RECORDED,
        model_id="M-MARKET",
        dataset_id="D-MARKET",
        run_id="RUN-SEP",
        params={"pnl_source": "actual", "gamma_test": 0.05},
        metrics={
            "n_observations": 1000,
            "n_exceptions": 6,
            "lr_cc": 1.9587,
            "p_value": 0.3755,
            "confidence": 0.99,
            "alpha_var": 0.01,
            "gamma_test": 0.05,
            "statistical_gamma_test": 0.05,
            "rejected": False,
        },
    )
    table = build_var_tail_table([rec])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("gamma=0.05" in c for c in crit_cells)

    # Missing gamma renders N/A
    m = dict(rec.metrics)
    del m["gamma_test"]
    del m["statistical_gamma_test"]
    rec_na = rec.model_copy(update={"metrics": m})
    table_na = build_var_tail_table([rec_na])
    crit_cells_na = [str(c) for c in table_na.columns[3]._cells]
    assert any("gamma=N/A" in c for c in crit_cells_na)


def test_evidence_bridge_normalizes_from_test_result() -> None:
    """10. Deterministic evidence bridge normalizes statistical alpha to gamma_test and expected_probability to alpha_var."""
    tr = TestResult(
        test_id="traded_risk.var_kupiec_pof",
        test_name="Kupiec proportion-of-failures test",
        status=Status.RECORDED,
        params={"alpha": 0.05, "pnl_source": "actual"},
        metrics={
            "confidence": 0.99,
            "expected_probability": 0.01,
            "alpha": 0.05,
            "lr_uc": 1.8862,
            "p_value": 0.1696,
            "rejected": False,
        },
    )
    ev = EvidenceRecord.from_result(tr, run_id="RUN-BRIDGE")
    assert ev.metrics["gamma_test"] == 0.05
    assert ev.metrics["statistical_gamma_test"] == 0.05
    assert ev.metrics["statistical_criterion_source"] == "STATISTICAL_TEST_SPECIFICATION"
    assert ev.metrics["alpha_var"] == 0.01
    assert ev.metrics["confidence"] == 0.99

    table = build_var_tail_table([ev])
    crit_cells = [str(c) for c in table.columns[3]._cells]
    assert any("gamma=0.05" in c for c in crit_cells)


def test_date_timestamp_tokenization_defect_fixed() -> None:
    """14. Date components (e.g. 2023-10-31) are temporal metadata and do not produce spurious quantitative claims."""
    c1 = extract_claims("As of 2023-10-31, the model was evaluated.")
    assert len(c1) == 0, f"Expected 0 claims from date 2023-10-31, got {c1}"

    c2 = extract_claims("The sample contains 2023 observations.")
    assert len(c2) == 1 and c2[0].value == 2023.0

    c3 = extract_claims("In 2023, the parameter was 0.05 [EV-01].")
    assert len(c3) == 1 and c3[0].value == 0.05

    c4 = extract_claims("Evaluated on 2026-09-02 with 6 exceptions.")
    assert len(c4) == 1 and c4[0].value == 6.0

    c5 = extract_claims("Backtest from 2023/10/31 to 2024/10/31.")
    assert len(c5) == 0

    c6 = extract_claims("Run timestamp 2023-10-31T15:30:00Z.")
    assert len(c6) == 0

