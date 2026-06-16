from __future__ import annotations

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from start.cli import app
from start.interactive_review import ReviewConfig, prompt_review_config

runner = CliRunner()


def test_review_command_non_interactive_demo(tmp_path):
    result = runner.invoke(
        app,
        [
            "review",
            "--non-interactive",
            "--target", "attrition",
            "--run-dl",
            "--output-root", str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Review complete" in result.output
    assert "binary_classification" in result.output
    # every stage should be visible in the console output
    for stage in ("Discovery", "Task Inference", "Split Planning", "Evidence Ledger", "Signoff"):
        assert stage in result.output
    assert (tmp_path / "ledger.jsonl").exists()


def test_review_command_on_user_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 300
    age = rng.integers(20, 70, n)
    churn = ((age > 50).astype(int) + rng.integers(0, 2, n) >= 1).astype(int)
    df = pd.DataFrame({"age": age, "income": rng.normal(50000, 10000, n), "churned": churn})
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    result = runner.invoke(
        app,
        ["review", str(csv), "--non-interactive", "--target", "churned",
         "--output-root", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.output
    assert "Review complete" in result.output


def test_review_command_diagnostics_only(tmp_path):
    result = runner.invoke(
        app,
        ["review", "--non-interactive", "--target", "attrition",
         "--no-run-dl", "--output-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "SKIPPED" in result.output and "Model Execution" in result.output


def test_interactive_config_deterministic_shows_no_llm_prompts():
    answers = iter([
        "",            # dataset path (demo)
        "attrition",   # target
        "stratified",  # split strategy
        "mlp",         # architecture
        "relu",        # activation
        "integrated_gradients",  # explainability
        "standard",    # robustness
        "deterministic",  # agent mode -> NO llm prompts follow
        "yes",         # run_dl (Section C)
        "0.60", "0.20", "0.20",  # split proportions (Section D)
        "bounded_random_search",  # tuning strategy (Section H)
        "5",           # trials
    ])
    cfg = prompt_review_config(ReviewConfig(), ask=lambda *_: next(answers))
    assert cfg.agent_mode == "deterministic"
    assert cfg.llm_provider == "none"  # untouched
    assert cfg.objective == ""  # never prompted in deterministic mode


def test_interactive_config_llm_shows_objective_prompt():
    answers = iter([
        "",            # dataset path
        "attrition",   # target
        "stratified",  # split
        "mlp",         # architecture
        "relu",        # activation
        "integrated_gradients",
        "standard",
        "llm",         # agent mode -> llm prompts follow
        "yes",         # run_dl (Section C)
        "0.60", "0.20", "0.20",  # split proportions (Section D)
        "bounded_random_search",  # tuning strategy (Section H)
        "5",           # trials
        "openai",      # provider
        "Predict customer attrition",  # objective
        "",            # clarification (none)
    ])
    cfg = prompt_review_config(ReviewConfig(), ask=lambda *_: next(answers))
    assert cfg.agent_mode == "llm"
    assert cfg.llm_provider == "openai"
    assert cfg.objective == "Predict customer attrition"
    assert cfg.run_dl is True  # Section C: training enabled by default


def test_review_help_lists_full_surface():
    result = runner.invoke(app, ["review", "--help"])
    assert result.exit_code == 0
    for flag in ("--split-strategy", "--architecture", "--activation",
                 "--explain-method", "--robustness", "--agent-mode", "--llm-provider"):
        assert flag in result.output


def test_review_notebook_py_compiles():
    import py_compile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    py_compile.compile(str(root / "notebooks" / "04_model_risk_review.py"), doraise=True)


def test_review_ipynb_is_valid_and_exposes_surface():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks" / "04_model_risk_review.ipynb").read_text())
    assert nb["nbformat"] == 4 and len(nb["cells"]) > 5
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "ReviewOrchestrator" in code
    # same surface as the CLI
    for token in ("split_strategy", "architecture", "activation", "agent_mode", "llm_provider"):
        assert token in code
    # no hardcoded secrets
    assert "sk-" not in code


def test_enterprise_review_command(tmp_path):
    from typer.testing import CliRunner

    from start.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["review", "--non-interactive", "--target", "attrition", "--enterprise",
         "--output-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "Enterprise review complete" in result.output
    # the seven layers stream visibly
    for layer in ("Data", "Model", "Validation", "Governance", "AI-Engineering",
                  "Evidence", "Reporting"):
        assert layer in result.output
    # dashboard generated
    from pathlib import Path

    dashboards = list(Path(tmp_path).glob("dashboards/*/dashboard.html"))
    assert dashboards, "enterprise mode must generate a dashboard"


def test_enterprise_help_documents_flag():
    from typer.testing import CliRunner

    from start.cli import app

    result = CliRunner().invoke(app, ["review", "--help"])
    assert result.exit_code == 0
    assert "--enterprise" in result.output


def test_enterprise_notebook_py_compiles():
    import py_compile
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    py_compile.compile(str(root / "notebooks" / "05_enterprise_review.py"), doraise=True)


def test_enterprise_ipynb_exposes_full_surface():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks" / "05_enterprise_review.ipynb").read_text())
    assert nb["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "EnterpriseReviewOrchestrator" in code
    # widget-driven full surface incl. enterprise + CNN + governance
    for token in ("enterprise_mode", "governance_mode", "cnn_preset", "provider",
                  "describe_cnn", "trust_domain"):
        assert token in code
    assert "sk-" not in code


def test_review_cost_flag_routes_metric(tmp_path):
    import glob
    import json

    from typer.testing import CliRunner

    from start.cli import app

    result = CliRunner().invoke(
        app,
        ["review", "--non-interactive", "--target", "attrition", "--run-dl", "--enterprise",
         "--cost", "false_negatives", "--output-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    dash = glob.glob(str(tmp_path / "dashboards" / "*" / "dashboard.json"))
    assert dash
    d = json.loads(open(dash[0]).read())
    assert d["metric_choice"]["primary_metric"] == "pr_auc"
    assert d["hyperparameter_tuning"]["primary_metric"] == "pr_auc"


def test_review_help_documents_new_flags():
    from typer.testing import CliRunner

    from start.cli import app

    result = CliRunner().invoke(app, ["review", "--help"])
    assert result.exit_code == 0
    for flag in ("--cost", "--accept-recommendatio", "--show-progress"):
        assert flag in result.output


# -- v2.1.1 visible co-pilot terminal output ---------------------------------- #
def test_enterprise_terminal_shows_visibility(tmp_path):
    from typer.testing import CliRunner

    from start.cli import app

    result = CliRunner().invoke(
        app,
        ["review", "--non-interactive", "--target", "attrition", "--run-dl",
         "--enterprise", "--output-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    # Section A: LLM activation visible
    assert "LLM activation" in result.output
    # Section K: agent reasoning traces visible
    assert "Agent reasoning traces" in result.output
    assert "DatasetDiscoveryAgent" in result.output
    # Section N: artifacts discoverable
    assert "Artifacts generated" in result.output


def test_enterprise_dashboard_has_v211_sections(tmp_path):
    import glob
    import json

    from typer.testing import CliRunner

    from start.cli import app

    result = CliRunner().invoke(
        app,
        ["review", "--non-interactive", "--target", "attrition", "--run-dl",
         "--enterprise", "--output-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    d = json.loads(open(glob.glob(str(tmp_path / "dashboards" / "*" / "dashboard.json"))[0]).read())
    assert d.get("llm_activation") is not None
    assert len(d.get("agent_reasoning_traces", [])) >= 5
    from start.ai_engineering.adapters import ADAPTER_CLASSES
    assert len(d.get("ai_engineering_control_surface", [])) == len(ADAPTER_CLASSES)
    # artifact catalog includes the dashboard's own files + telemetry + graph
    names = [a["name"] for a in d.get("artifact_catalog", [])]
    assert any("dashboard" in n for n in names)
    assert any("telemetry" in n for n in names)


def test_accept_recommendations_flag_shows_checkpoint(tmp_path):
    from typer.testing import CliRunner

    from start.cli import app

    result = CliRunner().invoke(
        app,
        ["review", "--non-interactive", "--target", "attrition", "--enterprise",
         "--architecture", "wide_deep", "--accept-recommendations",
         "--output-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert "Checkpoint: architecture" in result.output


def test_non_interactive_no_checkpoint_prompts(tmp_path):
    from typer.testing import CliRunner

    from start.cli import app

    result = CliRunner().invoke(
        app,
        ["review", "--non-interactive", "--target", "attrition", "--enterprise",
         "--architecture", "wide_deep", "--output-root", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    # without --accept-recommendations, no interactive checkpoint is shown
    assert "Checkpoint: architecture" not in result.output
