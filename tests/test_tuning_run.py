from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="tuning trains small models")

from start.modeling.data import load_attrition_dataset  # noqa: E402
from start.modeling.tuning_run import (  # noqa: E402
    render_tuning_run_markdown,
    run_tuning,
)
from start.reporting.artifacts import ArtifactRegistry  # noqa: E402


@pytest.fixture(scope="module")
def tuned(tmp_path_factory):
    df = load_attrition_dataset(seed=0)
    feats = [c for c in df.columns if c != "attrition"]
    reg = ArtifactRegistry()
    run = run_tuning(
        df,
        "attrition",
        feats,
        strategy="bounded_random_search",
        n_trials=5,
        primary_metric="pr_auc",
        seed=0,
        output_root=str(tmp_path_factory.mktemp("tune")),
        run_id="R",
        registry=reg,
    )
    return run, reg


def test_tuning_actually_runs_trials(tuned):
    run, _ = tuned
    assert run.ran is True
    assert len(run.trials) == 5
    for t in run.trials:
        assert 0.0 <= t.validation_metric <= 1.0
        assert "learning_rate" in t.params


def test_exactly_one_best_trial(tuned):
    run, _ = tuned
    assert sum(1 for t in run.trials if t.status == "best") == 1
    assert run.best_params
    assert run.best_metric == max(t.validation_metric for t in run.trials)


def test_rejected_params_recorded(tuned):
    run, _ = tuned
    assert len(run.rejected_params) == len(run.trials) - 1


def test_tuning_artifacts_generated(tuned):
    run, reg = tuned
    assert any("tuning_trials.csv" in a for a in run.artifacts)
    assert any("tuning_summary.json" in a for a in run.artifacts)
    assert any("tuning_trials.csv" in n for n in reg.names())


def test_tuning_metric_follows_cost():
    df = load_attrition_dataset(seed=0)
    feats = [c for c in df.columns if c != "attrition"]
    run = run_tuning(df, "attrition", feats, primary_metric="recall", n_trials=3, seed=0)
    assert run.primary_metric == "recall"


def test_disabled_tuning_explicit_note():
    df = load_attrition_dataset(seed=0)
    feats = [c for c in df.columns if c != "attrition"]
    run = run_tuning(df, "attrition", feats, strategy="none", seed=0)
    assert run.ran is False
    assert "disabled" in run.note.lower()
    assert len(run.trials) == 0


def test_grid_search_strategy_runs():
    df = load_attrition_dataset(seed=0)
    feats = [c for c in df.columns if c != "attrition"]
    run = run_tuning(df, "attrition", feats, strategy="grid_search", n_trials=4, seed=0)
    assert run.ran is True
    assert len(run.trials) == 4


def test_markdown_render(tuned):
    run, _ = tuned
    md = render_tuning_run_markdown(run)
    assert "### Hyperparameter tuning" in md
    assert "| Trial |" in md
    assert "best" in md


def test_to_dict_complete(tuned):
    run, _ = tuned
    d = run.to_dict()
    for key in (
        "strategy",
        "primary_metric",
        "n_trials",
        "ran",
        "trials",
        "best_params",
        "rejected_params",
        "search_space",
    ):
        assert key in d
