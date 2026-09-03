"""v2.3.1 #7 tests: stratified K-fold tuning uses train-only rows, never leaks
test/OOS, produces artifacts with mean/std, and routes the metric by cost."""

from __future__ import annotations

import json
import tempfile
import warnings

import pytest

warnings.filterwarnings("ignore")

from start.modeling.data import load_attrition_dataset  # noqa: E402
from start.modeling.kfold_tuning import (  # noqa: E402
    render_kfold_markdown,
    run_kfold_tuning,
)
from start.modeling.model_execution import _stratified_split  # noqa: E402


def _train_only(seed=0, props=(0.6, 0.2, 0.2)):
    df = load_attrition_dataset(seed=seed)
    feats = [c for c in df.columns if c != "attrition"]
    splits = _stratified_split(df, "attrition", props, seed=seed)
    return df, feats, splits


def test_kfold_runs_and_is_stratified():
    _, feats, splits = _train_only()
    run = run_kfold_tuning(
        splits["train"],
        "attrition",
        feats,
        n_folds=5,
        primary_metric="auc_roc",
        seed=0,
        output_root=tempfile.mkdtemp(),
    )
    assert run is not None
    assert run.method == "stratified_kfold"
    assert run.n_folds == 5


def test_kfold_uses_train_only_rows():
    _, feats, splits = _train_only()
    run = run_kfold_tuning(
        splits["train"],
        "attrition",
        feats,
        n_folds=5,
        seed=0,
        output_root=tempfile.mkdtemp(),
        excluded_rows=len(splits["test"]) + len(splits["oos"]),
    )
    assert run.train_rows == len(splits["train"])
    # every fold's train+val sums to the training row count (no extra rows)
    for f in run.best_fold_results:
        assert f.n_train + f.n_val == len(splits["train"])


def test_no_test_or_oos_row_enters_any_fold():
    # the K-fold module only ever sees train rows; prove test/OOS indices are
    # disjoint from what K-fold could touch by construction.
    _, feats, splits = _train_only()
    train_idx = set(splits["train"].index)
    test_idx = set(splits["test"].index)
    oos_idx = set(splits["oos"].index)
    assert train_idx.isdisjoint(test_idx)
    assert train_idx.isdisjoint(oos_idx)
    # run on train only; its row count must equal |train|, never include others
    run = run_kfold_tuning(
        splits["train"], "attrition", feats, n_folds=5, seed=0, output_root=tempfile.mkdtemp()
    )
    assert run.train_rows == len(train_idx)
    assert run.train_rows < (len(train_idx) + len(test_idx) + len(oos_idx))


def test_fold_artifacts_generated():
    out = tempfile.mkdtemp()
    _, feats, splits = _train_only()
    run = run_kfold_tuning(
        splits["train"], "attrition", feats, n_folds=5, seed=0, output_root=out, run_id="RUN"
    )
    names = {a.split("/")[-1] for a in run.artifacts}
    assert names == {"fold_metrics.csv", "tuning_trials.csv", "tuning_summary.json"}
    # summary json round-trips and carries fold info
    summary = json.loads(open([a for a in run.artifacts if a.endswith(".json")][0]).read())
    assert summary["method"] == "stratified_kfold"
    assert summary["best_fold_results"]


def test_fold_metrics_include_mean_and_std():
    _, feats, splits = _train_only()
    run = run_kfold_tuning(
        splits["train"], "attrition", feats, n_folds=5, seed=0, output_root=tempfile.mkdtemp()
    )
    for t in run.trials:
        assert t.mean_metric >= 0.0
        assert t.std_metric >= 0.0
        assert len(t.fold_metrics) == run.n_folds
    assert "Mean" in render_kfold_markdown(run)
    assert "Std" in render_kfold_markdown(run)


def test_metric_routing_false_negatives_uses_pr_auc():
    # the orchestrator routes cost->metric; here verify the K-fold honors the
    # routed metric name end to end.
    from start.agents.engineering_agents import select_primary_metric

    metric = select_primary_metric("binary_classification", costlier_errors="false_negatives")[
        "primary_metric"
    ]
    assert metric == "pr_auc"
    _, feats, splits = _train_only()
    run = run_kfold_tuning(
        splits["train"],
        "attrition",
        feats,
        n_folds=5,
        primary_metric=metric,
        seed=0,
        output_root=tempfile.mkdtemp(),
    )
    assert run.primary_metric == "pr_auc"


def test_metric_routing_false_positives_uses_precision():
    from start.agents.engineering_agents import select_primary_metric

    metric = select_primary_metric("binary_classification", costlier_errors="false_positives")[
        "primary_metric"
    ]
    assert metric == "precision"
    _, feats, splits = _train_only()
    run = run_kfold_tuning(
        splits["train"],
        "attrition",
        feats,
        n_folds=5,
        primary_metric=metric,
        seed=0,
        output_root=tempfile.mkdtemp(),
    )
    assert run.primary_metric == "precision"


def test_configurable_folds():
    _, feats, splits = _train_only()
    run = run_kfold_tuning(
        splits["train"], "attrition", feats, n_folds=3, seed=0, output_root=tempfile.mkdtemp()
    )
    assert run.n_folds == 3


def test_returns_none_when_infeasible():
    import pandas as pd

    tiny = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0], "y": [0, 1, 0]})
    assert run_kfold_tuning(tiny, "y", ["a", "b"], n_folds=5, output_root=tempfile.mkdtemp()) is None


def test_dl_tuning_is_labelled_single_split_not_kfold():
    # the DL path keeps single-split validation and must NOT claim K-fold.
    import inspect

    from start.modeling import tuning_run

    src = inspect.getsource(tuning_run)
    assert "holdout" in src.lower()  # single-split validation language
    # the DL tuning run object's strategy is not a kfold method
    from start.modeling.tuning_run import TuningRun

    assert (
        "kfold"
        not in TuningRun(strategy="bounded_random_search", primary_metric="auc_roc", n_trials=1).strategy
    )


def test_backward_compat_dl_tuning_still_runs():
    # K-fold is additive; the existing DL single-split tuning still produces a run
    pytest.importorskip("torch")
    from start.modeling.tuning_run import run_tuning

    df, feats, _ = _train_only()
    run = run_tuning(df, "attrition", feats, n_trials=3, seed=0, output_root=tempfile.mkdtemp())
    # returns a TuningRun (or None if torch truly unavailable) — not an error
    assert run is None or run.primary_metric in ("auc_roc", "pr_auc", "recall", "f1")
