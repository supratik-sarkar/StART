from start.core.schemas import EvidenceRecord, Status, TestResult, ThresholdSpec


def test_threshold_directions():
    upper = ThresholdSpec(metric="m", warn=1.0, fail=2.0, direction="upper")
    assert upper.evaluate(0.5) == Status.PASS
    assert upper.evaluate(1.5) == Status.WARN
    assert upper.evaluate(2.5) == Status.FAIL
    lower = ThresholdSpec(metric="auc", warn=0.65, fail=0.55, direction="lower")
    assert lower.evaluate(0.8) == Status.PASS
    assert lower.evaluate(0.6) == Status.WARN
    assert lower.evaluate(0.5) == Status.FAIL


def test_apply_thresholds_takes_worst():
    result = TestResult(
        test_id="t",
        test_name="t",
        metrics={"x": 3.0, "y": 0.5},
        thresholds=[
            ThresholdSpec(metric="x", warn=1.0, fail=5.0),
            ThresholdSpec(metric="y", warn=0.1, fail=0.4),
        ],
    )
    assert result.apply_thresholds().status == Status.FAIL


def test_evidence_from_result_carries_fields():
    result = TestResult(test_id="t", test_name="T", metrics={"m": 1}, status=Status.PASS)
    rec = EvidenceRecord.from_result(
        result, model_id="m1", dataset_id="d1", run_id="r1", policy_hash="ph"
    )
    assert rec.evidence_id.startswith("EV-")
    assert rec.policy_hash == "ph"
    assert rec.metrics == {"m": 1}
