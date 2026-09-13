from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from start.modeling.propensity import PropensityOptions, run_propensity_demo


@pytest.fixture()
def demo_result(tmp_path, monkeypatch):
    """One full non-interactive propensity run shared by assertions below."""
    monkeypatch.chdir(tmp_path)
    Path("configs/policy").mkdir(parents=True)
    Path("configs/policy/default_policy.yaml").write_text(
        yaml.safe_dump({"name": "test", "version": "0.0.1"})
    )
    opts = PropensityOptions(model="random_forest", tuning="none", seed=0)
    return run_propensity_demo(opts), tmp_path


def test_propensity_workflow_generates_evidence(demo_result):
    result, tmp_path = demo_result
    test_ids = {rec.test_id for rec in result.evidence}
    # feature-engineering checks
    assert {
        "preprocessing.missingness",
        "preprocessing.duplicates",
        "preprocessing.constant_features",
        "preprocessing.high_cardinality",
        "preprocessing.feature_ranges",
        "preprocessing.outliers",
        "preprocessing.feature_drift",
        "preprocessing.target_leakage",
        "preprocessing.split_diagnostics",
    } <= test_ids
    # model review checks
    assert {
        "supervised.cohort_metrics_comparison",
        "supervised.top_decile_lift",
        "xai.global_importance",
        "xai.feature_sensitivity",
    } <= test_ids
    assert all(rec.policy_hash for rec in result.evidence)
    assert (Path("start_output") / "ledger.jsonl").exists()
    reports = list((Path("start_output") / "reports").glob("RUN-*.md"))
    assert reports, "markdown report should be written"


def test_cohort_comparison_covers_three_cohorts(demo_result):
    result, _ = demo_result
    rec = next(r for r in result.evidence if r.test_id == "supervised.cohort_metrics_comparison")
    assert rec.status.value in {"pass", "warn", "fail"}
    for cohort in ("train", "test", "oos"):
        for metric in ("auc_roc", "accuracy", "precision", "recall", "f1", "top_decile_lift"):
            assert f"{cohort}_{metric}" in rec.metrics
    assert "auc_gap_train_test" in rec.metrics


def test_sensitivity_evidence_zero_shock_is_baseline(demo_result):
    result, _ = demo_result
    rec = next(r for r in result.evidence if r.test_id == "xai.feature_sensitivity")
    assert rec.status.value in {"pass", "warn", "fail"}
    assert rec.metrics["auc_+0pct"] == rec.metrics["baseline_auc"]
    assert rec.metrics["drift_+0pct"] == 0.0
    assert rec.metrics["importance_method"] in {"shap", "permutation"}
    assert len(rec.metrics["shocked_features"].split(", ")) == 5


def test_global_importance_states_method_honestly(demo_result):
    result, _ = demo_result
    rec = next(r for r in result.evidence if r.test_id == "xai.global_importance")
    assert rec.metrics["method"] in {"shap", "permutation"}
    if rec.metrics["method"] == "permutation":
        assert rec.metrics["n_local_examples"] == 0  # no fabricated local attributions


def test_risk_findings_and_suggestions_in_narrative(demo_result):
    result, _ = demo_result
    assert result.narrative is not None
    # TestSuggestionAgent: default model + no shap installed should yield suggestions
    assert any("tuned challenger" in step or "xai" in step for step in result.narrative.next_steps)
    # all narrative content still passes the citation gate
    assert result.critique is not None and result.critique.ok


def test_propensity_cli_non_interactive(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from start.cli import app

    monkeypatch.chdir(tmp_path)
    Path("configs/policy").mkdir(parents=True)
    Path("configs/policy/default_policy.yaml").write_text(
        yaml.safe_dump({"name": "test", "version": "0.0.1"})
    )
    runner = CliRunner()
    result = runner.invoke(
        app, ["propensity-demo", "--non-interactive", "--tuning", "none", "--seed", "1"]
    )
    assert result.exit_code == 0, result.output
    assert "Cohort metrics comparison" in result.output
    assert (Path("start_output") / "ledger.jsonl").exists()


def test_interactive_prompt_flow_with_scripted_input():
    from start.modeling.propensity import prompt_options

    answers = iter(
        [
            "random_forest",  # model
            "grid",           # tuning
            "50,100",         # n_estimators custom grid
            "",               # max_depth -> accept default
            "",               # min_samples_split
            "",               # min_samples_leaf
            "",               # max_features
            "kfold",          # validation scheme
            "5",              # K
            "oos",            # sensitivity cohort
        ]
    )
    opts = prompt_options(ask=lambda _prompt: next(answers))
    assert opts.model == "random_forest"
    assert opts.tuning == "grid"
    assert opts.cv_folds == 5
    assert opts.sensitivity_cohort == "oos"
    assert opts.custom_space is not None
    assert opts.custom_space["n_estimators"]["grid"] == [50, 100]
