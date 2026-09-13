from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from start.modeling.data_statistics import compute_data_statistics
from start.modeling.fe_recommendations import (
    recommend_feature_engineering,
    render_fe_recommendations_markdown,
)


@pytest.fixture()
def stats():
    rng = np.random.default_rng(0)
    n = 300
    d = pd.DataFrame(
        {
            "const": [1] * n,
            "cat_hi": [f"v{i}" for i in range(n)],
            "city": rng.choice(["NY", "LA", "SF"], n),
            "x": rng.normal(0, 1, n),
            "amount": rng.normal(0, 1000, n),
            "y": (rng.random(n) < 0.05).astype(int),
        }
    )
    d.loc[0:15, "x"] = 999.0
    d["leak"] = d["y"] * 1.0
    d.loc[0:20, "amount"] = np.nan
    return compute_data_statistics(d, "y")


def test_each_recommendation_has_required_fields(stats):
    recs = recommend_feature_engineering(stats)
    assert recs.applicable()
    for r in recs.applicable():
        assert r.recommendation and r.reason and r.evidence_id
        assert r.risk_if_ignored and r.default_action
        d = r.to_dict()
        assert set(["step", "recommendation", "reason", "evidence_id",
                    "risk_if_ignored", "default_action", "user_override"]) <= set(d)


def test_covers_expected_steps(stats):
    steps = {r.step for r in recommend_feature_engineering(stats).applicable()}
    for expected in ("imputation", "encoding", "scaling", "outliers", "imbalance",
                     "low_variance", "leakage_exclusion", "high_cardinality"):
        assert expected in steps, f"missing recommendation: {expected}"


def test_evidence_ids_unique(stats):
    recs = recommend_feature_engineering(stats).applicable()
    ids = [r.evidence_id for r in recs]
    assert len(ids) == len(set(ids))


def test_leakage_recommendation_present(stats):
    recs = recommend_feature_engineering(stats).applicable()
    leak = next((r for r in recs if r.step == "leakage_exclusion"), None)
    assert leak is not None
    assert "EXCLUDE" in leak.recommendation
    assert leak.default_action == "exclude_leakage"


def test_user_override_changes_effective_action(stats):
    recs = recommend_feature_engineering(stats)
    recs.apply_overrides({"scaling": "skip"})
    scaling = next(r for r in recs.recommendations if r.step == "scaling")
    assert scaling.user_override == "skip"
    assert scaling.effective_action == "skip"
    # untouched recommendation falls back to default
    enc = next(r for r in recs.recommendations if r.step == "encoding")
    assert enc.effective_action == enc.default_action


def test_modality_routing():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "ts": pd.date_range("2021-01-01", periods=60, freq="D"),
            "v": rng.normal(0, 1, 60),
            "target": [0, 1] * 30,
        }
    )
    stats = compute_data_statistics(df, "target")
    seq = recommend_feature_engineering(stats, modality="sequential")
    assert any(r.step == "datetime_expansion" for r in seq.applicable())
    vis = recommend_feature_engineering(stats, modality="vision")
    assert any(r.step == "image_transforms" for r in vis.applicable())
    txt = recommend_feature_engineering(stats, modality="text")
    assert any(r.step == "text_vectorization" for r in txt.applicable())


def test_clean_dataset_minimal_recommendations():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "a": rng.normal(0, 1, 200),
            "b": rng.normal(0, 1, 200),
            "target": rng.integers(0, 2, 200),
        }
    )
    recs = recommend_feature_engineering(compute_data_statistics(df, "target"))
    steps = {r.step for r in recs.applicable()}
    # clean numeric data: scaling yes, but no imputation/leakage/low_variance
    assert "imputation" not in steps
    assert "leakage_exclusion" not in steps


def test_markdown_render(stats):
    md = render_fe_recommendations_markdown(recommend_feature_engineering(stats))
    assert "### Feature-engineering recommendations" in md
    assert "Evidence" in md and "Risk if ignored" in md
