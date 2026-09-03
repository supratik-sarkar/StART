"""v2.3.1 batch-2 tests: dataset/target transparency, boxed panels, colored
agent/adapter names, safe endpoint display, progress helpers."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from start.cli import app

runner = CliRunner()
pytestmark = pytest.mark.filterwarnings("ignore")


# --- #2 dataset / target transparency ------------------------------------- #
def test_demo_dataset_transparency_printed(tmp_path):
    res = runner.invoke(app, ["review", "--non-interactive", "--run-dl", "--output-root", str(tmp_path)])
    assert res.exit_code == 0, res.output
    assert "Demo dataset:" in res.output
    assert "Synthetic" in res.output or "synthetic" in res.output
    assert "rows x" in res.output and "columns" in res.output


def test_target_transparency_user_supplied(tmp_path):
    res = runner.invoke(
        app,
        ["review", "--non-interactive", "--target", "attrition", "--run-dl", "--output-root", str(tmp_path)],
    )
    assert "Target:" in res.output
    assert "supplied by user" in res.output


def test_candidate_targets_listed(tmp_path):
    res = runner.invoke(app, ["review", "--non-interactive", "--run-dl", "--output-root", str(tmp_path)])
    assert "candidate target" in res.output.lower()


# --- #3 boxed panels + colored agent names -------------------------------- #
def test_roster_panel_renders_box_and_colors():
    from rich.console import Console

    from start.agent_roster import render_agent_roster_panel

    console = Console(record=True, width=100)
    console.print(render_agent_roster_panel())
    text = console.export_text()
    assert "Review committee" in text
    # box-drawing border present
    assert "─" in text or "╭" in text


def test_agent_colors_assigned():
    from start.agent_roster import AGENT_COLORS, agent_color

    assert agent_color("ValidationAgent") != "white"
    assert agent_color("GovernanceSignoffAgent") != "white"
    # every roster agent has a non-white color
    from start.agent_roster import AGENT_ROSTER

    for r in AGENT_ROSTER:
        assert AGENT_COLORS.get(r.name, "white") != "white", r.name


def test_llm_activation_panel_renders():
    from rich.console import Console

    from start.providers.llm_activation import preflight_llm
    from start.review_tables import llm_activation_panel

    act = preflight_llm("none", llm=None)
    console = Console(record=True, width=100)
    console.print(llm_activation_panel(act))
    text = console.export_text()
    assert "LLM activation" in text
    assert "Provider" in text and "Endpoint" in text


# --- #4 colored adapter inventory ----------------------------------------- #
def test_adapter_inventory_has_colored_names_and_fallback_col():
    from rich.console import Console

    from start.review_tables import adapter_inventory_table

    cs = [
        {
            "adapter": "DeepEval",
            "status": "not_installed",
            "purpose": "quality",
            "runtime_s": 0.0,
            "artifacts": 0,
            "evidence": 0,
            "install_guidance": "pip install deepeval",
        }
    ]
    console = Console(record=True, width=140)
    console.print(adapter_inventory_table(cs))
    text = console.export_text()
    assert "AI Engineering Environment" in text
    assert "Install / fallback" in text
    assert "DeepEval" in text


def test_adapter_color_distinct():
    from start.review_tables import _adapter_color

    assert _adapter_color("OPA") != _adapter_color("DeepEval")
    assert _adapter_color("OPA") != "white"


# --- #5 safe endpoint display --------------------------------------------- #
def test_public_provider_endpoints_populated():
    from start.providers.llm_activation import preflight_llm

    assert preflight_llm("openai", llm=None).endpoint.startswith("https://api.openai.com")
    assert preflight_llm("anthropic", llm=None).endpoint.startswith("https://api.anthropic.com")
    assert preflight_llm("grok", llm=None).endpoint.startswith("https://api.x.ai")


def test_gateway_endpoint_hidden_by_default(monkeypatch):
    monkeypatch.delenv("START_ENTERPRISE_LLM_ENDPOINT_PUBLIC", raising=False)
    monkeypatch.delenv("START_ENTERPRISE_LLM_PACKAGE", raising=False)
    monkeypatch.delenv("START_ENTERPRISE_PACKAGE", raising=False)
    from start.providers.llm_activation import preflight_llm

    ep = preflight_llm("enterprise_llm_gateway", llm=None).endpoint
    assert "endpoint hidden" in ep
    assert "private-package route" in ep
    # no secret-looking content
    assert "sk-" not in ep and "AKIA" not in ep


def test_gateway_endpoint_shows_package_name(monkeypatch):
    monkeypatch.setenv("START_ENTERPRISE_LLM_PACKAGE", "firm_llm_gateway_client")
    monkeypatch.delenv("START_ENTERPRISE_LLM_ENDPOINT_PUBLIC", raising=False)
    from start.providers.llm_activation import preflight_llm

    assert "firm_llm_gateway_client" in preflight_llm("enterprise_llm_gateway", llm=None).endpoint


def test_gateway_safe_public_endpoint_shown_when_exposed(monkeypatch):
    monkeypatch.setenv("START_ENTERPRISE_LLM_ENDPOINT_PUBLIC", "https://gw.example/v1")
    from start.providers.llm_activation import preflight_llm

    ep = preflight_llm("enterprise_llm_gateway", llm=None).endpoint
    assert "gw.example" in ep


# --- #6 progress helpers -------------------------------------------------- #
def test_progress_bar_counts_and_completes():
    from start.progress import progress_bar

    seen = {"n": 0}
    with progress_bar(3, "test", enabled=True) as advance:
        for _ in range(3):
            advance(1)
            seen["n"] += 1
    assert seen["n"] == 3


def test_progress_bar_disabled_is_noop():
    from start.progress import progress_bar

    with progress_bar(5, "test", enabled=False) as advance:
        advance(1)  # must not raise


def test_progress_bar_zero_total_noop():
    from start.progress import progress_bar

    with progress_bar(0, "test") as advance:
        advance(1)


def test_spinner_runs():
    from start.progress import spinner

    with spinner("working", enabled=True):
        pass
    with spinner("working", enabled=False):
        pass


# --- no secret leakage with the new surfaces ------------------------------ #
def test_no_secret_leak_in_panels(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-leak-test-0000000000")
    res = runner.invoke(app, ["review", "--non-interactive", "--run-dl", "--output-root", str(tmp_path)])
    assert "sk-fake-leak-test" not in res.output
    from pathlib import Path

    for art in Path(tmp_path).rglob("*"):
        if art.is_file() and art.suffix in {".md", ".json", ".html", ".txt"}:
            assert "sk-fake-leak-test" not in art.read_text(errors="ignore"), art
