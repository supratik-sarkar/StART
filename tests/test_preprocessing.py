import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from start.core.hashing import hash_dataframe
from start.core.schemas import Status
from start.registry import TestContext, get_test


def test_missingness_metrics(toy_frame):
    spec = get_test("preprocessing.missingness")
    result = spec.fn(TestContext(train=toy_frame, target_column="target"))
    assert result.metrics["overall_missing_pct"] > 0
    assert result.status in (Status.PASS, Status.WARN)


def test_leakage_flags_perfect_feature(toy_frame):
    df = toy_frame.copy()
    df["leak"] = df["target"].astype(float)
    spec = get_test("preprocessing.target_leakage")
    result = spec.fn(TestContext(train=df, target_column="target"))
    assert result.status == Status.FAIL
    assert result.metrics["worst_feature"] == "leak"


def test_split_overlap_detected(toy_frame):
    spec = get_test("preprocessing.split_diagnostics")
    result = spec.fn(
        TestContext(train=toy_frame, test=toy_frame.head(50), target_column="target")
    )
    assert result.metrics["test_rows_seen_in_train_pct"] == 100.0
    assert result.status == Status.FAIL


@settings(max_examples=25, deadline=None)
@given(seed=st.integers(min_value=0, max_value=10_000))
def test_missingness_row_order_invariance(seed):
    """Determinism claim: metrics are invariant to row order."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.normal(size=200), "y": rng.normal(size=200)})
    df.loc[rng.choice(200, 20, replace=False), "x"] = np.nan
    shuffled = df.sample(frac=1.0, random_state=seed)
    spec = get_test("preprocessing.missingness")
    a = spec.fn(TestContext(train=df)).metrics
    b = spec.fn(TestContext(train=shuffled)).metrics
    assert a == b
    assert hash_dataframe(df) == hash_dataframe(shuffled)
