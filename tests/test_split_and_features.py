from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from start.modeling.feature_engineering import FeatureEngineeringAgent
from start.modeling.split_planner import SPLIT_STRATEGIES, SplitPlanner


@pytest.fixture()
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "entity": rng.integers(0, 30, 300),
            "ts": pd.date_range("2021-01-01", periods=300, freq="D"),
            "x1": rng.normal(0, 1, 300),
            "x2": rng.normal(0, 1000, 300),
            "target": rng.integers(0, 2, 300),
        }
    )


def test_all_strategies_partition_completely(frame):
    sp = SplitPlanner()
    configs = {
        "random": {},
        "stratified": {"target_column": "target"},
        "time_based": {"time_column": "ts"},
        "group": {"group_column": "entity"},
    }
    for strategy, kw in configs.items():
        plan = sp.plan(frame, strategy=strategy, fractions=(0.6, 0.2, 0.2), seed=0, **kw)
        n_train, n_test, n_oos = plan.sizes
        assert n_train + n_test + n_oos == len(frame), strategy
        assert n_train > 0 and n_test > 0 and n_oos > 0, strategy
        assert plan.strategy == strategy
    assert set(SPLIT_STRATEGIES) == {"random", "stratified", "time_based", "group", "custom"}


def test_user_controlled_fractions(frame):
    plan = SplitPlanner().plan(
        frame, strategy="random", fractions=(0.5, 0.3, 0.2), seed=0
    )
    n = len(frame)
    assert abs(len(plan.train) - 0.5 * n) <= 1
    assert abs(len(plan.test) - 0.3 * n) <= 1


def test_fractions_must_sum_to_one(frame):
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitPlanner().plan(frame, strategy="random", fractions=(0.6, 0.2, 0.1))


def test_unknown_strategy_raises(frame):
    with pytest.raises(ValueError, match="Unknown split strategy"):
        SplitPlanner().plan(frame, strategy="quantum")


def test_group_split_has_no_entity_leakage(frame):
    plan = SplitPlanner().plan(frame, strategy="group", group_column="entity", seed=0)
    tr, te, oo = set(plan.train.entity), set(plan.test.entity), set(plan.oos.entity)
    assert not (tr & te) and not (tr & oo) and not (te & oo)


def test_time_based_oos_is_most_recent(frame):
    plan = SplitPlanner().plan(frame, strategy="time_based", time_column="ts")
    assert plan.train.ts.max() <= plan.oos.ts.min()


def test_time_based_requires_time_column(frame):
    with pytest.raises(ValueError, match="time_column"):
        SplitPlanner().plan(frame, strategy="time_based")


def test_custom_split_with_masks(frame):
    masks = {
        "train": frame.index < 180,
        "test": (frame.index >= 180) & (frame.index < 240),
        "oos": frame.index >= 240,
    }
    plan = SplitPlanner().plan(frame, strategy="custom", custom_masks=masks)
    assert plan.sizes == (180, 60, 60)


def test_split_evidence_record(frame):
    sp = SplitPlanner()
    plan = sp.plan(frame, strategy="stratified", target_column="target", seed=0)
    ev = sp.to_evidence(plan, "target")
    assert ev.test_id == "split.plan"
    assert ev.metrics["strategy"] == "stratified"
    assert ev.metrics["n_train"] + ev.metrics["n_test"] + ev.metrics["n_oos"] == len(frame)
    assert "train_pos_rate" in ev.metrics


def test_feature_engineering_tabular(frame):
    fe = FeatureEngineeringAgent()
    diag = fe.diagnose(
        frame.drop(columns=["ts", "entity"]),
        "target",
        test=frame.drop(columns=["ts", "entity"]),
    )
    assert diag.modality == "tabular"
    assert diag.findings["needs_scaling"] is True  # x2 scale >> x1
    assert fe.to_evidence(diag).status.value in {"pass", "warn", "fail"}


def test_feature_engineering_detects_leakage(frame):
    leaky = frame.copy()
    leaky["leak"] = leaky["target"] * 1.0
    diag = FeatureEngineeringAgent().diagnose(leaky.drop(columns=["ts", "entity"]), "target")
    assert "leak" in diag.leakage_suspects
    assert FeatureEngineeringAgent().to_evidence(diag).status.value == "fail"


def test_feature_engineering_modalities(frame):
    fe = FeatureEngineeringAgent()
    seq = fe.diagnose(frame, "target", modality="sequential", time_column="ts")
    assert seq.modality == "sequential" and seq.findings["window_feasible"]
    vis = pd.DataFrame({"image_path": ["a.png"] * 20, "target": [0, 1] * 10})
    vdiag = fe.diagnose(vis, "target", modality="vision")
    assert vdiag.modality == "vision" and vdiag.findings["n_classes"] == 2
    txt = pd.DataFrame({"notes": ["several words here please"] * 20, "target": [0, 1] * 10})
    tdiag = fe.diagnose(txt, "target", modality="text")
    assert tdiag.modality == "text" and tdiag.findings["n_text_columns"] == 1
