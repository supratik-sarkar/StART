from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="model execution trains a small model")

from start.modeling.data import load_attrition_dataset  # noqa: E402
from start.modeling.model_execution import (  # noqa: E402
    render_model_execution_markdown,
    run_model_execution,
)
from start.reporting.artifacts import ArtifactRegistry  # noqa: E402


@pytest.fixture(scope="module")
def execution(tmp_path_factory):
    df = load_attrition_dataset(seed=0)
    reg = ArtifactRegistry()
    out = str(tmp_path_factory.mktemp("model_execution"))
    ex = run_model_execution(
        df, "attrition", split_props=(0.60, 0.20, 0.20), metric_name="pr_auc",
        seed=0, output_root=out, run_id="RUN-TEST", registry=reg,
    )
    return ex, reg


def test_split_table_generated(execution):
    ex, _ = execution
    assert {r["split"] for r in ex.split_table} == {"train", "test", "oos"}
    total = sum(r["percent"] for r in ex.split_table)
    assert abs(total - 100.0) < 1.0


def test_split_proportions_honored(execution):
    ex, _ = execution
    by = {r["split"]: r["percent"] for r in ex.split_table}
    assert 55 <= by["train"] <= 65
    assert 15 <= by["test"] <= 25
    assert 15 <= by["oos"] <= 25


def test_stratification_preserves_positive_rate(execution):
    ex, _ = execution
    rates = [r["positive_rate"] for r in ex.split_table]
    # stratified: positive rates should be close across splits
    assert max(rates) - min(rates) < 0.05


def test_metrics_by_split_generated(execution):
    ex, _ = execution
    assert set(ex.metrics_by_split) == {"train", "test", "oos"}
    for m in ex.metrics_by_split.values():
        for key in ("auc_roc", "pr_auc", "accuracy", "precision", "recall", "f1",
                    "brier_score", "ece"):
            assert key in m


def test_generalization_gap_computed(execution):
    ex, _ = execution
    assert ex.generalization_gap is not None


def test_training_diagnostics_present(execution):
    ex, _ = execution
    assert "device" in ex.training_diagnostics
    assert "architecture" in ex.training_diagnostics


def test_explainability_table_generated(execution):
    ex, _ = execution
    assert ex.explainability_method in ("integrated_gradients", "gradient_shap", "permutation")
    assert ex.global_importance
    top = ex.global_importance[0]
    for key in ("rank", "feature", "importance", "direction"):
        assert key in top


def test_artifacts_registered(execution):
    ex, reg = execution
    assert len(ex.artifacts) >= 5
    names = reg.names()
    assert any("split_distribution.csv" in n for n in names)
    assert any("metrics_by_split.csv" in n for n in names)
    assert any("global_feature_importance.csv" in n for n in names)


def test_markdown_render(execution):
    ex, _ = execution
    md = render_model_execution_markdown(ex)
    assert "### Train/Test/OOS split" in md
    assert "### Metrics by split" in md
    assert "Global explainability" in md


def test_to_dict_complete(execution):
    ex, _ = execution
    d = ex.to_dict()
    for key in ("split_table", "metrics_by_split", "training_diagnostics",
                "explainability_method", "global_importance", "generalization_gap"):
        assert key in d
