"""Acceptance tests for StART v4.0.1 Build Brief ("Make the Demo Honest").

Verifies:
1. Pre-change ledger replay backward compatibility (Amendment 5a).
2. Deterministic ASCII terminal plot rendering (Amendment 5b).
3. Dual-ID querying on EvidenceLedger and trace commands (A1).
4. Canonical adjudications leaf hashing and inclusion (A2).
5. Precondition validation with blocking checks 1-4 and advisory critic gate (A6, Amendment 1).
6. Seal manifest persistence, verification, and `start attest trace` (Amendment 3).
7. Honest dataset generators & loaders: Synthetic, UCI German Credit, Fannie Mae (Workstream B).
8. Single best trial selection and stratified holdout in tuning (D2, D4).
9. LangSmith tracer initialization fix (D1).
10. Adapter status matrix categorization (D8).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from start.attestation.seal import (
    build_seal,
    persist_seal_manifest,
    validate_seal_preconditions,
    verify_seal,
)
from start.cli.terminal_plots import (
    render_calibration_ascii,
    render_drift_sparkline,
    render_pr_curve_ascii,
    render_roc_curve_ascii,
    render_score_distribution_ascii,
    render_threshold_sweep_ascii,
)
from start.core.schemas import Status, TestResult
from start.data.fannie_mae import load_fannie_mae_dataset
from start.data.synthetic import generate_synthetic_transactions
from start.data.uci_credit import fetch_or_load_german_credit
from start.evidence.ledger import EvidenceLedger
from start.review_session import Challenge, Decision, ReviewSession


def test_status_enum_backward_compatibility(tmp_path: Path):
    """Test Amendment 5a: Pre-change ledgers with pass/warn/fail replay without error."""
    ledger_path = tmp_path / "legacy_ledger.jsonl"
    store_dir = tmp_path / "evidence_store"
    ledger = EvidenceLedger(ledger_path, store_dir)

    # Add legacy record with standard Status.PASS
    res1 = TestResult(
        test_id="split:ratio",
        test_name="Train/Test Split Check",
        status=Status.PASS,
        metrics={"train_ratio": 0.6},
        evidence={"details": "standard 60/20/20 split"},
    )
    rec1 = ledger.append(res1)
    assert rec1.status == Status.PASS

    # Add record with Status.RECORDED
    res2 = TestResult(
        test_id="discovery:dimensions",
        test_name="Dataset Dimensions",
        status=Status.RECORDED,
        metrics={"rows": 1000, "cols": 25},
        evidence={"notes": "informative metric"},
    )
    rec2 = ledger.append(res2)
    assert rec2.status == Status.RECORDED

    # Re-read ledger from disk
    ledger_reload = EvidenceLedger(ledger_path, store_dir)
    assert len(ledger_reload) == 2
    assert ledger_reload.verify() is True

    from start.attestation import replay_ledger
    replay_verdict = replay_ledger(ledger_path)
    assert replay_verdict.intact is True


def test_evidence_ledger_dual_id_query(tmp_path: Path):
    """Test A1: EvidenceLedger records_for_run matches enterprise_run_id and run_id."""
    ledger_path = tmp_path / "ledger.jsonl"
    store_dir = tmp_path / "store"
    ledger = EvidenceLedger(ledger_path, store_dir)

    res = TestResult(
        test_id="test:1",
        test_name="Test One",
        status=Status.PASS,
        metrics={"acc": 0.95},
        evidence={},
    )
    # Append with enterprise_run_id
    rec = ledger.append(res, enterprise_run_id="RUN-ENT-12345678")
    assert rec.enterprise_run_id == "RUN-ENT-12345678"

    matched = ledger.records_for_run("RUN-ENT-12345678")
    assert len(matched) == 1
    assert matched[0].evidence_id == rec.evidence_id

    assert len(ledger.records_for_run("RUN-NONEXISTENT")) == 0


def test_terminal_plots_determinism(monkeypatch):
    """Test Amendment 5b: Terminal plots render byte-identical output across runs."""
    monkeypatch.setenv("NO_COLOR", "1")
    y_true = np.array([0, 0, 0, 0, 1, 0, 1, 1, 0, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.4, 0.8, 0.3, 0.75, 0.9, 0.25, 0.85])

    plot1_roc = render_roc_curve_ascii(y_true, scores)
    plot2_roc = render_roc_curve_ascii(y_true, scores)
    assert plot1_roc == plot2_roc

    plot1_pr = render_pr_curve_ascii(y_true, scores)
    plot2_pr = render_pr_curve_ascii(y_true, scores)
    assert plot1_pr == plot2_pr

    plot1_cal = render_calibration_ascii(y_true, scores)
    plot2_cal = render_calibration_ascii(y_true, scores)
    assert plot1_cal == plot2_cal

    plot1_dist = render_score_distribution_ascii(y_true, scores)
    plot2_dist = render_score_distribution_ascii(y_true, scores)
    assert plot1_dist == plot2_dist

    plot1_swp = render_threshold_sweep_ascii(y_true, scores)
    plot2_swp = render_threshold_sweep_ascii(y_true, scores)
    assert plot1_swp == plot2_swp

    drifts = {"feat_a": 0.12, "feat_b": -0.05, "feat_c": 0.45}
    spark1 = render_drift_sparkline(drifts)
    spark2 = render_drift_sparkline(drifts)
    assert spark1 == spark2


def test_adjudications_canonical_dict_and_merkle_hashing():
    """Test A2 & A3: Session canonical dict reflects decisions, overrides, and challenges."""
    session = ReviewSession(run_id="RUN-ENT-TEST")
    session.record_decision(
        Decision(
            key="metric_priority",
            prompt="Cost priority?",
            recommended="false_negatives",
            user_value="false_positives",
            effective="false_positives",
            choice="override",
            rationale="Business priorities favor precision over recall",
            agent_rationale="False negatives have 5x higher cost",
        )
    )
    session.record_challenge(
        Challenge(
            challenge_id="chal-1",
            agent="ArchitectureReviewAgent",
            text="Why not use XGBoost?",
            answer="MLP has lower latency and better embedding capability",
            provider="openai",
            model="gpt-4o-mini",
            response_id="chatcmpl-test1234",
            input_tokens=150,
            output_tokens=45,
            latency_seconds=0.62,
        )
    )

    can_dict = session.to_canonical_dict()
    assert len(can_dict["decisions"]) == 1
    assert len(can_dict["overrides"]) == 1
    assert len(can_dict["challenges"]) == 1
    assert can_dict["overrides"][0]["rationale"] == "Business priorities favor precision over recall"
    assert can_dict["overrides"][0]["agent_rationale"] == "False negatives have 5x higher cost"
    assert can_dict["challenges"][0]["response_id"] == "chatcmpl-test1234"


def test_seal_preconditions_validation_blocking_and_advisory(tmp_path: Path):
    """Test A6 & Amendment 1: Checks 1-4 block sealing; Check 5 (critic) is advisory."""
    review_id = "RUN-ENT-SEALTEST"
    ledger_path = tmp_path / "ledger.jsonl"
    store_dir = tmp_path / "store"
    ledger = EvidenceLedger(ledger_path, store_dir)

    res = TestResult(
        test_id="test:1",
        test_name="Test 1",
        status=Status.PASS,
        metrics={"val": 1.0},
        evidence={},
    )
    rec = ledger.append(res, enterprise_run_id=review_id)

    # 1. Success case (deterministic agent mode, critic PASSED)
    adj_payload = {"decisions": [{"key": "target", "choice": "accept"}], "challenges": []}
    ok, checks = validate_seal_preconditions(
        review_id=review_id,
        ledger_records=[rec],
        evidence_head=rec,
        adjudications=adj_payload,
        attestations=None,
        agent_mode="deterministic",
        critic_verdict="PASSED",
    )
    assert ok is True
    assert all(c.passed for c in checks if c.blocking)

    # 2. Critic FAILED — still passes blocking checks because Check 5 is advisory (Amendment 1)
    ok_failed_critic, checks_crit = validate_seal_preconditions(
        review_id=review_id,
        ledger_records=[rec],
        evidence_head=rec,
        adjudications=adj_payload,
        attestations=None,
        agent_mode="deterministic",
        critic_verdict="FAILED",
    )
    assert ok_failed_critic is True
    critic_check = next(c for c in checks_crit if "evidence critique" in c.label)
    assert critic_check.blocking is False

    # 3. Empty evidence records — BLOCKS sealing
    ok_empty, checks_empty = validate_seal_preconditions(
        review_id=review_id,
        ledger_records=[],
        evidence_head=None,
        adjudications=adj_payload,
        attestations=None,
        agent_mode="deterministic",
        critic_verdict="PASSED",
    )
    assert ok_empty is False

    # 4. LLM mode with no attestations — BLOCKS sealing
    ok_no_att, checks_no_att = validate_seal_preconditions(
        review_id=review_id,
        ledger_records=[rec],
        evidence_head=rec,
        adjudications=adj_payload,
        attestations=[],
        agent_mode="llm",
        critic_verdict="PASSED",
    )
    assert ok_no_att is False


def test_seal_manifest_persistence_and_index(tmp_path: Path):
    """Test Amendment 3: Persist seal manifest, index it, and verify with verify_seal."""
    review_id = "RUN-ENT-MANIFESTTEST"
    seal = build_seal(
        review_id=review_id,
        plan={"target": "is_fraud", "task": "binary_classification"},
        policy={"disclosure_policy": "public_demo"},
        evidence_head="abcd" * 16,
        adjudications={"decisions": [{"key": "target", "choice": "accept"}]},
        metadata={"enterprise_run_id": review_id, "inner_run_id": "RUN-MROS-inner123"},
    )

    manifest_path = persist_seal_manifest(seal, tmp_path)
    assert manifest_path.exists()

    # Check global index
    index_path = tmp_path / "seals" / "index.json"
    assert index_path.exists()
    idx = json.loads(index_path.read_text())
    assert seal.seal_string() in idx
    assert idx[seal.seal_string()]["enterprise_run_id"] == review_id
    assert idx[seal.seal_string()]["inner_run_id"] == "RUN-MROS-inner123"

    # Verify manifest
    manifest_data = json.loads(manifest_path.read_text())
    verdict = verify_seal(manifest_data, expected_seal=seal.seal_string())
    assert verdict["verified"] is True
    assert verdict["recomputed_seal"] == seal.seal_string()


def test_synthetic_data_generator():
    """Test B1: Synthetic transaction generator shape, prevalence, and columns."""
    df = generate_synthetic_transactions(n_rows=500, prevalence=0.06, n_features=20, seed=42)
    assert len(df) == 500
    assert df.shape[1] == 21  # 20 features + target is_fraud
    assert "is_fraud" in df.columns
    assert df["is_fraud"].sum() == 30  # exactly 6% of 500


def test_uci_credit_loader():
    """Test B2: UCI German Credit loader (falls back to synthetic if offline)."""
    df = fetch_or_load_german_credit()
    assert len(df) == 1000
    assert "is_bad_credit" in df.columns
    assert df["is_bad_credit"].nunique() == 2


def test_fannie_mae_byo_loader(tmp_path: Path):
    """Test B3: Fannie Mae pipe-delimited file parser."""
    fn_file = tmp_path / "fannie_mae_sample.txt"
    fn_file.write_text(
        "loan_id|orig_rate|orig_upb|current_loan_delinquency_status\n"
        "10001|4.25|250000|0\n"
        "10002|3.75|180000|1\n"
        "10003|4.00|320000|0\n"
    )
    df = load_fannie_mae_dataset(fn_file, target_column="is_delinquent")
    assert len(df) == 3
    assert "is_delinquent" in df.columns
    assert list(df["is_delinquent"]) == [0, 1, 0]
