from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from start.agents import (
    ChallengeAgent,
    GovernanceAgent,
    ModelRecommendationAgent,
    SignoffAgent,
    ValidationPlannerAgent,
)
from start.core.schemas import EvidenceRecord, Status, TestResult
from start.modeling.explain import detect_model_family, route_explainability
from start.taxonomy import profile_dataset


def _record(test_id: str, status: Status, metrics: dict | None = None) -> EvidenceRecord:
    result = TestResult(
        test_id=test_id,
        test_name=test_id,
        status=status,
        metrics=metrics or {},
        interpretation="synthetic record for agent tests.",
    )
    return EvidenceRecord.from_result(
        result, model_id="m-test", dataset_id="d-test", run_id="RUN-test", policy_hash="abc123"
    )


# --------------------------------------------------------------------------- #
# Taxonomy / profiling
# --------------------------------------------------------------------------- #
def test_profile_tabular_binary():
    df = pd.DataFrame({"a": np.arange(100.0), "b": np.arange(100.0), "y": [0, 1] * 50})
    profile = profile_dataset(df, target_column="y")
    assert profile.dataset_type == "tabular"
    assert profile.target_type == "binary"
    assert not profile.declared


def test_profile_panel_and_time_series():
    n = 60
    base = {
        "ts": pd.date_range("2024-01-01", periods=n),
        "x": np.arange(float(n)),
        "y": np.random.default_rng(0).normal(size=n),
    }
    ts = profile_dataset(pd.DataFrame(base), target_column="y", timestamp_column="ts")
    assert ts.dataset_type == "time_series"
    assert ts.target_type == "continuous"
    panel_df = pd.DataFrame({**base, "asset": ["A", "B", "C"] * 20})
    panel = profile_dataset(
        panel_df, target_column="y", timestamp_column="ts", entity_id_column="asset"
    )
    assert panel.dataset_type == "panel_time_series"
    assert panel.n_entities == 3


def test_domain_types_only_via_declaration():
    df = pd.DataFrame({"bid_1": [1.0, 2.0], "ask_1": [1.1, 2.1], "y": [0, 1]})
    inferred = profile_dataset(df, target_column="y")
    assert inferred.dataset_type == "tabular"  # never auto-claims LOB
    declared = profile_dataset(df, target_column="y", declared_type="limit_order_book")
    assert declared.dataset_type == "limit_order_book" and declared.declared
    with pytest.raises(ValueError, match="Unknown dataset_type"):
        profile_dataset(df, target_column="y", declared_type="nonsense")


# --------------------------------------------------------------------------- #
# Model recommendations / validation planner
# --------------------------------------------------------------------------- #
def test_model_recommendations_by_type():
    df = pd.DataFrame({"a": np.arange(50.0), "y": [0, 1] * 25})
    tabular = profile_dataset(df, target_column="y")
    lines = ModelRecommendationAgent().recommend(tabular)
    assert any("random_forest" in line and "available now" in line for line in lines)
    lob = profile_dataset(df, target_column="y", declared_type="limit_order_book")
    lob_lines = ModelRecommendationAgent().recommend(lob)
    assert any("deeplob" in line and "roadmap" in line for line in lob_lines)


def test_validation_planner_tree_vs_lob():
    from sklearn.ensemble import RandomForestClassifier

    df = pd.DataFrame({"a": np.arange(50.0), "y": [0, 1] * 25})
    tabular = profile_dataset(df, target_column="y")
    plan = ValidationPlannerAgent().plan_for(tabular, model=RandomForestClassifier())
    assert plan["model_family"] == "tree"
    assert any("cohort_metrics_comparison" in item for item in plan["available_now"])
    assert any("shap" in m for m in plan["explainability"]["implemented"] + plan["explainability"]["roadmap"])

    lob = profile_dataset(df, target_column="y", declared_type="limit_order_book")
    lob_plan = ValidationPlannerAgent().plan_for(lob, model_family="deep_learning")
    assert any("latency" in item for item in lob_plan["roadmap"])
    assert any("integrated_gradients" in item for item in lob_plan["roadmap"])
    assert any("feature_drift" in item for item in lob_plan["available_now"])


def test_explainability_router_families():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    assert detect_model_family(RandomForestClassifier()) == "tree"
    assert detect_model_family(LogisticRegression()) == "linear"
    assert detect_model_family(None) == "unknown"
    plan = route_explainability(model_family="deep_learning")
    assert "integrated_gradients" in plan.roadmap()
    assert "permutation_sensitivity" in plan.implemented()


# --------------------------------------------------------------------------- #
# Governance / challenge / sign-off
# --------------------------------------------------------------------------- #
def test_governance_flags_breaches_and_skips():
    records = [
        _record("supervised.discrimination", Status.PASS),
        _record("supervised.calibration", Status.FAIL),
        _record("xai.global_importance", Status.SKIPPED),
    ]
    ok, items = GovernanceAgent().review(records)
    assert not ok
    assert any("requires documented disposition" in item for item in items)
    assert any("did not execute" in item for item in items)
    clean_ok, clean_items = GovernanceAgent().review([_record("x", Status.PASS)])
    assert clean_ok and clean_items == []


def test_challenge_agent_rules():
    records = [
        _record(
            "supervised.cohort_metrics_comparison",
            Status.PASS,
            {"train_auc_roc": 1.0, "test_auc_roc": 0.95},
        ),
        _record("preprocessing.feature_drift", Status.WARN, {"n_rows": 500}),
        _record("xai.feature_sensitivity", Status.PASS, {"cohort": "test"}),
    ]
    challenges = ChallengeAgent().challenge(records)
    text = " ".join(challenges)
    assert "memorization" in text
    assert "OOS or development cohort" in text
    assert "sampling" in text
    # every quantitative/challenge claim carries a citation
    assert all("[EV-" in c for c in challenges)


def test_signoff_gating():
    clean = [_record("a", Status.PASS), _record("b", Status.WARN)]
    ready = SignoffAgent().conclude(clean, governance_ok=True, governance_items=[])
    assert "READY FOR SIGN-OFF" in ready and "NOT READY" not in ready
    assert "[EV-" in ready  # warns are cited for reviewer judgment
    breached = clean + [_record("c", Status.FAIL)]
    not_ready = SignoffAgent().conclude(breached, governance_ok=False, governance_items=["x"])
    assert "NOT READY FOR SIGN-OFF" in not_ready
    assert "[EV-" in not_ready


def test_enriched_narrative_passes_citation_gate(tmp_path, monkeypatch):
    """End-to-end: governance/challenge/signoff additions must survive the
    EvidenceCriticAgent citation gate inside a real run."""
    monkeypatch.chdir(tmp_path)
    from start.modeling.data import load_attrition_dataset
    from start.orchestration import review_dataframes

    df = load_attrition_dataset(seed=5)
    result = review_dataframes(df, target_column="attrition", seed=5)
    assert result.narrative is not None
    assert result.narrative.signoff  # sign-off present
    assert result.critique is not None and result.critique.ok


def test_recommend_cli(tmp_path):
    pytest.importorskip("pyarrow")
    from typer.testing import CliRunner

    from start.cli import app

    df = pd.DataFrame(
        {
            "ts": pd.date_range("2024-01-01", periods=40),
            "asset": ["A", "B"] * 20,
            "x": np.arange(40.0),
            "y": [0, 1] * 20,
        }
    )
    path = tmp_path / "panel.parquet"
    df.to_parquet(path)
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "recommend",
            str(path),
            "--target",
            "y",
            "--timestamp-col",
            "ts",
            "--entity-col",
            "asset",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "panel_time_series" in result.output
    assert "tft" in result.output
    assert "roadmap" in result.output
