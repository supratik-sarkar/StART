"""Focused release-hygiene regressions for v6.0.2."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import demo_controller
from typer.testing import CliRunner

from start import __version__
from start.cli.main import app

ROOT = Path(__file__).resolve().parent.parent
WEBAPP = ROOT / "webapp"


def test_root_cli_version_and_help_surfaces() -> None:
    runner = CliRunner()

    version = runner.invoke(app, ["--version"])
    assert version.exit_code == 0
    assert version.stdout.strip() == f"StART {__version__}"
    assert __version__ == "6.0.2"

    assert runner.invoke(app, ["--help"]).exit_code == 0
    assert runner.invoke(app, ["review", "--help"]).exit_code == 0


def test_demo_controller_uses_repository_relative_paths() -> None:
    source = (ROOT / "demo_controller.py").read_text(encoding="utf-8")

    assert demo_controller.REPOSITORY_ROOT == (ROOT / "demo_controller.py").resolve().parent
    assert demo_controller.CHEATSHEET_PATH == ROOT / "StART_2.0_DEMO_CHEATSHEET_PRIVATE.md"
    assert "/Users/" not in source
    assert "PUBLIC_GIT_PATH" not in source


def test_demo_controller_default_result_root_is_canonical(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(demo_controller, "REPOSITORY_ROOT", tmp_path)
    result = demo_controller.SessionResult(session_id="SESSION_A_TEMPORAL")

    target = demo_controller._persist_session_result(result, None)

    assert target.name == "session_result.json"
    assert target.parent.parent == tmp_path / "start_output" / "demo_results"
    assert not (tmp_path / "demo_results").exists()
    assert json.loads(target.read_text(encoding="utf-8"))["session_id"] == "SESSION_A_TEMPORAL"


def test_market_runbook_has_one_version_neutral_active_path() -> None:
    from scripts.run_market_acceptance_from_runbook import RUNBOOK_PATH

    expected = ROOT / "docs" / "MARKET_MANUAL_ACCEPTANCE_RUNBOOK.md"
    assert RUNBOOK_PATH == expected
    assert expected.is_file()
    assert not (ROOT / "StART_v4.3.0_Market_Manual_Acceptance_Runbook.md").exists()


def test_webapp_versions_and_changed_manifest_entries_are_consistent() -> None:
    package = json.loads((WEBAPP / "package.json").read_text(encoding="utf-8"))
    lock = json.loads((WEBAPP / "package-lock.json").read_text(encoding="utf-8"))
    manifest = json.loads((WEBAPP / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))

    assert package["version"] == lock["version"] == lock["packages"][""]["version"] == "6.0.2"

    entries = {entry["path"]: entry for entry in manifest["files"]}
    changed = {
        "README.md",
        "docs/BACKEND_INTEGRATION_GUIDE.md",
        "docs/BACKEND_WIRING_GUIDE.md",
        "docs/ARCHITECTURE.md",
        "docs/BACKEND_ADAPTER.md",
        "docs/BACKEND_GAP_CHECKLIST.md",
        "docs/DESIGN_FREEZE.md",
        "docs/TERMINAL_TO_VISUAL_MAPPING.md",
        "package.json",
    }
    for relative in changed:
        payload = (WEBAPP / relative).read_bytes()
        assert entries[relative]["bytes"] == len(payload)
        assert entries[relative]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_release_hygiene_documents_have_no_development_provenance() -> None:
    paths = [
        ROOT / "docs" / "START_BACKEND_BINDING_CLOSURE.md",
        ROOT / "docs" / "START_AGENTIC_WORKBENCH_PRODUCT_SPEC.md",
        ROOT / "docs" / "START_CAPABILITY_CENSUS.md",
        ROOT / "docs" / "START_PRESENTATION_API_CONTRACT.md",
        WEBAPP / "README.md",
        WEBAPP / "PACKAGE_MANIFEST.json",
        *(WEBAPP / "docs").glob("*.md"),
    ]
    forbidden = ("ready for external review", "/users/")

    for path in paths:
        text = path.read_text(encoding="utf-8").lower()
        assert not any(term in text for term in forbidden), path

    assert not tuple((WEBAPP / "docs").glob("*HANDOFF*.md"))
    assert not tuple((WEBAPP / "docs").glob("*PROMPT*.md"))


def test_publication_boundaries_are_ignored() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    for boundary in ("mlflow.db", "mlruns/", "tmp/", "start_output/", ".env", ".env.*"):
        assert boundary in ignore
