"""CI workflow parity.

The scheduled workflows used to fail every night on commits that had passed their
pull-request gate, because the two disagreed about what to check:

* core-ci linted with a narrowed, non-blocking rule set; the scheduled job linted
  with the full pyproject configuration and blocked on it;
* core-ci skipped fifteen stale modules with ``--ignore`` flags; the scheduled jobs
  ran them;
* nothing installed with ``constraints.txt``, so tool versions drifted between runs;
* a weekly job selected ``-m integration`` when no test carries that marker, so
  pytest collected nothing and exited 5.

These tests run inside core-ci, so reintroducing any of that fails the pull request
that does it, rather than a scheduled run days later.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

#: Ruleset required status checks match these display names. Renaming a job makes
#: every pull request wait forever on a check that no longer reports. Change this
#: set only together with the repository ruleset.
REQUIRED_CORE_CI_JOBS = {
    "no-secrets": "no credentials or organisation identifiers",
    "stdlib-core": "risk + attestation core with nothing installed",
    "packaging": "packaging contract",
    "core-tests": "fast hermetic core tests (ubuntu-latest, py3.12.13)",
    "frontend": "frontend workbench (tests, typecheck, build)",
}

EXCLUDED_IN_EVERY_SCHEDULED_RUN = ("integration", "network", "release", "slow")
LINT_COMMAND = "ruff check src tests scripts"


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _triggers(doc: dict) -> dict:
    # PyYAML reads the bare key `on` as boolean True.
    value = doc.get("on", doc.get(True))
    return value if isinstance(value, dict) else {}


def _run_lines(doc: dict) -> list[str]:
    lines: list[str] = []
    for job in doc.get("jobs", {}).values():
        for step in job.get("steps", []):
            run = step.get("run")
            if run:
                lines.extend(line.strip() for line in run.splitlines() if line.strip())
    return lines


WORKFLOW_FILES = sorted(WORKFLOWS.glob("*.yml"))
SCHEDULED = [p for p in WORKFLOW_FILES if "schedule" in _triggers(_load(p))]


def test_workflows_exist():
    assert WORKFLOW_FILES, "no workflow files found"


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_every_editable_install_is_pinned(path):
    for line in _run_lines(_load(path)):
        if re.search(r"pip install .*-e\b", line):
            assert "-c constraints.txt" in line, f"{path.name}: unpinned install: {line}"


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_no_workflow_ignores_test_files(path):
    """Quarantine belongs in tests/quarantine.txt, where every environment sees it."""
    for line in _run_lines(_load(path)):
        assert "--ignore=" not in line, f"{path.name}: use tests/quarantine.txt: {line}"


@pytest.mark.parametrize("path", WORKFLOW_FILES, ids=lambda p: p.name)
def test_lint_is_identical_and_blocking(path):
    for line in _run_lines(_load(path)):
        if line.startswith("ruff "):
            assert line == LINT_COMMAND, (
                f"{path.name}: lint must be exactly `{LINT_COMMAND}` "
                f"(rules live in pyproject.toml, never `|| true`): {line}"
            )


def test_core_ci_lints():
    assert LINT_COMMAND in _run_lines(_load(WORKFLOWS / "core-ci.yml"))


@pytest.mark.parametrize("path", SCHEDULED, ids=lambda p: p.name)
def test_scheduled_pytest_selection_matches_the_pull_request_gate(path):
    commands = [line for line in _run_lines(_load(path)) if line.startswith("pytest")]
    assert commands, f"{path.name}: scheduled workflow runs no pytest command"
    for command in commands:
        selection = re.search(r'-m\s+"([^"]+)"', command)
        assert selection, f"{path.name}: unfiltered `pytest tests/` diverges from core-ci"
        expression = selection.group(1)
        for marker in EXCLUDED_IN_EVERY_SCHEDULED_RUN:
            assert f"not {marker}" in expression, (
                f"{path.name}: scheduled selection must exclude `{marker}`: {expression}"
            )


def test_core_ci_job_ids_and_names_are_stable():
    jobs = _load(WORKFLOWS / "core-ci.yml")["jobs"]
    for job_id, display_name in REQUIRED_CORE_CI_JOBS.items():
        assert job_id in jobs, f"core-ci job `{job_id}` removed; update the ruleset first"
        assert jobs[job_id].get("name") == display_name, (
            f"core-ci job `{job_id}` renamed; ruleset required checks match the old name"
        )


def test_quarantine_entries_exist():
    """A stale entry hides nothing but implies coverage that is not there."""
    listing = ROOT / "tests" / "quarantine.txt"
    for raw in listing.read_text(encoding="utf-8").splitlines():
        entry = raw.split("#", 1)[0].strip()
        if entry:
            assert (ROOT / entry).is_file(), f"quarantine.txt lists missing file: {entry}"


def test_pre_commit_ruff_matches_the_ci_pin():
    pins = dict(
        line.split("==", 1)
        for line in (ROOT / "constraints.txt").read_text().splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    )
    config = _load(ROOT / ".pre-commit-config.yaml")
    ruff_repo = next(r for r in config["repos"] if "ruff-pre-commit" in r["repo"])
    assert ruff_repo["rev"].lstrip("v") == pins["ruff"].strip(), (
        "pre-commit ruff and constraints.txt ruff must be the same version"
    )
