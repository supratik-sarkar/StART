"""v2.3.1 polish tests: wording rename, calibration config, decision ledger."""

from __future__ import annotations

import glob

import pytest
from typer.testing import CliRunner

from start.cli import app

runner = CliRunner()
pytestmark = pytest.mark.filterwarnings("ignore")


# --- #1 wording rename ---------------------------------------------------- #
def test_default_run_uses_ai_review_committee_wording(tmp_path):
    res = runner.invoke(
        app,
        ["review", "--non-interactive", "--target", "attrition", "--run-dl", "--output-root", str(tmp_path)],
    )
    assert res.exit_code == 0, res.output
    assert "Running AI REVIEW COMMITTEE workflow" in res.output
    assert "AI review committee complete" in res.output


def test_old_enterprise_review_wording_absent(tmp_path):
    res = runner.invoke(
        app,
        ["review", "--non-interactive", "--target", "attrition", "--run-dl", "--output-root", str(tmp_path)],
    )
    assert "Running ENTERPRISE review" not in res.output
    assert "Enterprise review complete" not in res.output


def test_enterprise_wording_only_for_gateway_in_source():
    # the only public 'enterprise' that should remain is the gateway path
    import start.interactive_review as ir

    src = open(ir.__file__).read()
    assert "Running ENTERPRISE review" not in src
    assert "Enterprise review complete" not in src
    # gateway name preserved
    assert "enterprise_llm_gateway" in src


# --- #9 calibration threshold configurable -------------------------------- #
def test_calibration_threshold_in_config():
    from start.core.config import StartConfig

    assert hasattr(StartConfig().governance, "max_ece")
    assert StartConfig().governance.max_ece == 0.10


def test_calibration_threshold_is_configurable():
    from start.evidence_store import EvidenceStore
    from start.mrm_signoff import evaluate_signoff
    from start.review_session import ReviewSession

    store = EvidenceStore()
    store.cohort_metrics = {"oos": {"auc_roc": 0.9, "ece": 0.12}}
    # default 0.10 -> concern
    d = evaluate_signoff(store, ReviewSession(run_id="R"))
    cal = [f for f in d.factors if f.name == "Calibration"][0]
    assert cal.status == "concern"
    assert "configured threshold" in cal.detail  # wording explains configurability
    # raised threshold -> ok
    d2 = evaluate_signoff(store, ReviewSession(run_id="R"), max_ece=0.2)
    cal2 = [f for f in d2.factors if f.name == "Calibration"][0]
    assert cal2.status == "ok"


# --- #8 decision ledger --------------------------------------------------- #
def _decisions():
    return [
        {
            "key": "architecture",
            "recommended": "mlp",
            "effective": "wide_deep",
            "choice": "keep",
            "evidence_ids": ["ARCH-01"],
        },
        {
            "key": "fe:correlation_pruning",
            "recommended": "apply",
            "effective": "skip",
            "choice": "reject",
            "evidence_ids": [],
        },
    ]


def test_decision_ledger_table_builds():
    from start.review_tables import decision_ledger_table

    t = decision_ledger_table(_decisions())
    assert t.row_count == 2
    assert t.columns[0].header == "Checkpoint"


def test_decision_ledger_markdown_has_impact_and_status():
    from start.review_tables import decision_ledger_markdown

    md = decision_ledger_markdown(_decisions())
    assert "Execution impact" in md
    assert "overridden" in md and "rejected" in md
    assert "kept all features" in md  # rejection impact


def test_decision_ledger_in_outputs(tmp_path):
    res = runner.invoke(
        app,
        ["review", "--non-interactive", "--target", "attrition", "--run-dl", "--output-root", str(tmp_path)],
    )
    assert res.exit_code == 0, res.output
    assert "Review decision ledger" in res.output  # terminal
    dash = open(glob.glob(f"{tmp_path}/dashboards/*/dashboard.md")[0]).read()
    txt = open(glob.glob(f"{tmp_path}/transcripts/*/transcript.md")[0]).read()
    assert "Review decision ledger" in dash
    assert "Review decision ledger" in txt
