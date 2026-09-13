from start.core.schemas import Status
from start.registry import TestContext, get_test


def test_discrimination_reasonable(toy_frame):
    spec = get_test("supervised.discrimination")
    result = spec.fn(
        TestContext(test=toy_frame, target_column="target", score_column="score")
    )
    assert 0.5 < result.metrics["roc_auc"] <= 1.0
    assert result.status == Status.PASS


def test_calibration_runs(toy_frame):
    spec = get_test("supervised.calibration")
    result = spec.fn(
        TestContext(test=toy_frame, target_column="target", score_column="score")
    )
    assert "ece" in result.metrics and "brier_score" in result.metrics


def test_missing_columns_skip():
    import pandas as pd

    spec = get_test("supervised.discrimination")
    result = spec.fn(TestContext(test=pd.DataFrame({"a": [1, 2]})))
    assert result.status == Status.SKIPPED
