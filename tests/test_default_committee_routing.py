"""Regression tests for the v2.3.0 default routing blocker fix.

Plain `start review` must enter the evidence-driven committee path by default
(not the legacy pipeline), `--enterprise` must do the same, `--standard` must be
explicitly labelled legacy, and the default flow must show the committee roster,
ValidationAgent review, and MRM-grade signoff — never the old weak signoff.
"""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from start.cli import app
from start.interactive_review import ReviewConfig, prompt_review_config

runner = CliRunner()

pytestmark = pytest.mark.filterwarnings("ignore")


def test_plain_review_enters_committee_path_by_default(tmp_path):
    # no --enterprise / --standard flag -> committee path is the default
    res = runner.invoke(app, ["review", "--non-interactive", "--target", "attrition",
                              "--run-dl", "--output-root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "Running AI REVIEW COMMITTEE workflow" in res.output
    assert "Review committee" in res.output


def test_explicit_enterprise_enters_committee_path(tmp_path):
    res = runner.invoke(app, ["review", "--non-interactive", "--target", "attrition",
                              "--run-dl", "--enterprise", "--output-root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "Running AI REVIEW COMMITTEE workflow" in res.output


def test_standard_flag_is_labelled_legacy(tmp_path):
    res = runner.invoke(app, ["review", "--non-interactive", "--target", "attrition",
                              "--run-dl", "--standard", "--output-root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "Running legacy/basic review" in res.output
    assert "Running AI REVIEW COMMITTEE workflow" not in res.output


def test_default_flow_prints_committee_roster(tmp_path):
    res = runner.invoke(app, ["review", "--non-interactive", "--target", "attrition",
                              "--run-dl", "--output-root", str(tmp_path)])
    assert "Review committee" in res.output
    # a couple of named committee members
    assert "ValidationAgent" in res.output
    assert "GovernanceSignoffAgent" in res.output


def test_default_flow_prints_validation_review(tmp_path):
    res = runner.invoke(app, ["review", "--non-interactive", "--target", "attrition",
                              "--run-dl", "--output-root", str(tmp_path)])
    assert "ValidationAgent" in res.output and "validation review" in res.output
    assert "feature sensitivity ranking" in res.output


def test_default_flow_uses_mrm_grade_signoff(tmp_path):
    res = runner.invoke(app, ["review", "--non-interactive", "--target", "attrition",
                              "--run-dl", "--output-root", str(tmp_path)])
    assert "MRM decision" in res.output
    # verdict vocabulary of the MRM-grade signoff
    assert any(v in res.output for v in ("READY", "READY WITH CONDITIONS", "NOT READY"))


def test_default_flow_does_not_use_old_weak_signoff(tmp_path):
    res = runner.invoke(app, ["review", "--non-interactive", "--target", "attrition",
                              "--run-dl", "--output-root", str(tmp_path)])
    # the legacy phrasing must not appear in the default committee flow
    assert "all evidence records pass" not in res.output.lower()
    assert "ready for reviewer sign-off because all evidence" not in res.output.lower()


def test_interactive_committee_prompt_defaults_yes():
    # blank answer to the committee prompt -> enterprise committee path
    def ask(prompt=""):
        return ""  # default everything, including the committee y/N
    cfg = prompt_review_config(ReviewConfig(agent_mode="deterministic"), ask=ask)
    assert cfg.enterprise_mode is True


def test_interactive_committee_prompt_can_opt_into_legacy():
    def ask(prompt=""):
        if "committee workflow" in prompt.lower():
            return "n"
        return ""
    cfg = prompt_review_config(ReviewConfig(agent_mode="deterministic"), ask=ask)
    assert cfg.enterprise_mode is False
