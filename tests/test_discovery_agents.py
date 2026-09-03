from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from start.agents.discovery import (
    TASK_TYPES,
    DatasetDiscoveryAgent,
    TargetDiscoveryAgent,
    TaskInferenceAgent,
    run_discovery,
)


@pytest.fixture()
def rich_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "customer_id": range(120),
            "age": rng.integers(20, 70, 120),
            "balance": rng.normal(1000, 200, 120),
            "notes": ["a longer free-text description of the customer relationship here"] * 120,
            "signup_date": pd.date_range("2021-01-01", periods=120),
            "churned": rng.integers(0, 2, 120),
        }
    )


def test_discovery_roles_and_columns(rich_frame):
    profile = DatasetDiscoveryAgent().discover(rich_frame)
    assert profile.n_rows == 120 and profile.n_columns == 6
    assert "customer_id" in profile.entity_columns
    assert "notes" in profile.text_columns
    assert "signup_date" in profile.timestamp_columns
    assert "churned" in profile.candidate_targets
    # id and text/datetime never proposed as targets
    assert "customer_id" not in profile.candidate_targets
    assert "notes" not in profile.candidate_targets


def test_discovery_evidence_record(rich_frame):
    agent = DatasetDiscoveryAgent()
    profile = agent.discover(rich_frame)
    ev = agent.to_evidence(profile)
    assert ev.test_id == "discovery.dataset_profile"
    assert ev.metrics["n_rows"] == 120
    assert "churned" in ev.metrics["candidate_targets"]


def test_target_requires_confirmation(rich_frame):
    profile = DatasetDiscoveryAgent().discover(rich_frame)
    agent = TargetDiscoveryAgent()
    unconfirmed = agent.recommend(profile)
    assert unconfirmed.selected is None
    assert agent.to_evidence(unconfirmed).status.value == "warn"  # no training yet
    confirmed = agent.recommend(profile, user_target="churned")
    assert confirmed.selected == "churned"
    assert agent.to_evidence(confirmed).status.value == "pass"


def test_image_path_detection(tmp_path):
    df = pd.DataFrame(
        {
            "image_path": [f"/data/img_{i}.png" for i in range(20)],
            "label": (["cat", "dog"] * 10),
        }
    )
    profile = DatasetDiscoveryAgent().discover(df)
    assert "image_path" in profile.image_path_columns


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0, 1] * 30, "binary_classification"),
        (list(range(4)) * 15, "multiclass_classification"),
        (np.random.default_rng(1).normal(0, 1, 60).tolist(), "regression"),
    ],
)
def test_task_inference_single_target(values, expected):
    df = pd.DataFrame({"x": range(len(values)), "y": values})
    inference = TaskInferenceAgent().infer(df, "y")
    assert inference.task_type == expected


def test_task_inference_forecasting_with_timestamp():
    df = pd.DataFrame({"value": np.random.default_rng(2).normal(0, 1, 50)})
    inference = TaskInferenceAgent().infer(df, "value", has_timestamp=True)
    assert inference.task_type == "forecasting"


def test_task_inference_multilabel_and_override():
    df = pd.DataFrame({"x": range(40), "a": [0, 1] * 20, "b": [1, 0] * 20})
    multi = TaskInferenceAgent().infer(df, ["a", "b"])
    assert multi.task_type == "multilabel_classification"
    assert multi.target_type == "multi_output"

    overridden = TaskInferenceAgent().infer(df, "a", override="anomaly_detection")
    assert overridden.task_type == "anomaly_detection" and overridden.overridden
    with pytest.raises(ValueError, match="Unknown task"):
        TaskInferenceAgent().infer(df, "a", override="not_a_task")


def test_run_discovery_pipeline(rich_frame):
    profile, target, task, evidence = run_discovery(rich_frame, user_target="churned")
    assert task.task_type == "binary_classification"
    assert [e.test_id for e in evidence] == [
        "discovery.dataset_profile",
        "discovery.target_selection",
        "discovery.task_inference",
    ]
    assert all(t in TASK_TYPES for t in [task.task_type])
