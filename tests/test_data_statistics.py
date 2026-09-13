from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from start.modeling.data_statistics import (
    compute_data_statistics,
    render_statistics_markdown,
)


@pytest.fixture()
def messy_frame():
    rng = np.random.default_rng(0)
    n = 300
    d = pd.DataFrame(
        {
            "id": range(n),
            "const": [1] * n,
            "cat_hi": [f"v{i}" for i in range(n)],
            "x": rng.normal(0, 1, n),
            "y": rng.integers(0, 2, n),
        }
    )
    d.loc[0:10, "x"] = 999.0
    d = pd.concat([d, d.iloc[[0, 1]]], ignore_index=True)
    d["leak"] = d["y"] * 1.0
    return d


def test_basic_counts(messy_frame):
    s = compute_data_statistics(messy_frame, "y")
    assert s.n_rows == 302  # 300 + 2 duplicates
    assert s.n_columns == 6
    assert s.target_column == "y"
    assert s.target_type == "binary"
    assert s.n_numeric >= 3


def test_duplicate_detection(messy_frame):
    assert compute_data_statistics(messy_frame, "y").n_duplicate_rows == 2


def test_low_variance_detection(messy_frame):
    assert "const" in compute_data_statistics(messy_frame, "y").low_variance_columns


def test_high_cardinality_detection(messy_frame):
    assert "cat_hi" in compute_data_statistics(messy_frame, "y").high_cardinality_columns


def test_leakage_detection(messy_frame):
    assert "leak" in compute_data_statistics(messy_frame, "y").leakage_candidates


def test_outlier_detection(messy_frame):
    s = compute_data_statistics(messy_frame, "y")
    assert "x" in s.outlier_summary and s.outlier_summary["x"] > 0


def test_class_distribution_and_imbalance():
    rng = np.random.default_rng(1)
    y = np.zeros(1000, dtype=int)
    y[:30] = 1  # 3% minority -> severe
    df = pd.DataFrame({"f": rng.normal(0, 1, 1000), "target": y})
    s = compute_data_statistics(df, "target")
    assert s.target_type == "binary"
    assert abs(s.class_distribution["1"] - 0.03) < 0.001
    assert "severe" in s.imbalance_warning


def test_suggested_split_time_based():
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2021-01-01", periods=50, freq="D"),
            "x": range(50),
            "target": [0, 1] * 25,
        }
    )
    assert compute_data_statistics(df, "target").suggested_split == "time_based"


def test_multiclass_target():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"f": rng.normal(0, 1, 300), "target": rng.integers(0, 4, 300)})
    s = compute_data_statistics(df, "target")
    assert s.target_type == "multiclass"
    assert len(s.class_distribution) == 4


def test_no_correlation_warning_on_constant_column(messy_frame):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        compute_data_statistics(messy_frame, "y")  # must not raise


def test_markdown_render(messy_frame):
    md = render_statistics_markdown(compute_data_statistics(messy_frame, "y"))
    assert "### Initial data statistics" in md
    assert "| Metric | Value |" in md
    assert "Suggested split" in md


def test_to_dict_complete(messy_frame):
    d = compute_data_statistics(messy_frame, "y").to_dict()
    for key in ("n_rows", "target_type", "leakage_candidates", "suggested_split",
                "high_cardinality_columns", "outlier_summary"):
        assert key in d
