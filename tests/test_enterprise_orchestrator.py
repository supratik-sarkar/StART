from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from start.modeling.enterprise_orchestrator import (
    LAYER_NAMES,
    EnterpriseReviewOrchestrator,
)


@pytest.fixture()
def churn_frame():
    rng = np.random.default_rng(0)
    n = 400
    age = rng.integers(20, 70, n)
    churn = ((age > 50).astype(int) + rng.integers(0, 2, n) >= 1).astype(int)
    return pd.DataFrame(
        {
            "customer_id": range(n),
            "age": age,
            "balance": rng.normal(1000, 200, n),
            "tenure": rng.integers(0, 10, n),
            "churned": churn,
        }
    )


def test_all_seven_layers_execute(churn_frame, tmp_path):
    seen = []
    orch = EnterpriseReviewOrchestrator(on_layer=lambda lr: seen.append((lr.name, lr.status)))
    outcome = orch.run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=True, seed=0
    )
    assert [lr.name for lr in outcome.layers] == list(LAYER_NAMES)
    for lr in outcome.layers:
        assert lr.status == "complete"
        assert lr.runtime_seconds >= 0.0


def test_each_layer_emits_required_fields(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=True, seed=0
    )
    for lr in outcome.layers:
        d = lr.to_dict()
        # spec: every layer emits status/runtime/warnings/findings/artifacts/evidence_ids
        assert set(["layer", "status", "runtime_seconds", "warnings", "findings",
                    "artifacts", "evidence_ids"]) <= set(d)
    # at least one layer carries evidence IDs, one carries findings, one carries artifacts
    assert any(lr.evidence_ids for lr in outcome.layers)
    assert any(lr.findings for lr in outcome.layers)
    assert any(lr.artifacts for lr in outcome.layers)


def test_governance_findings_derived(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=True, seed=0
    )
    register = outcome.findings_register
    assert register.summary()["total"] >= 1
    # every finding is cited (EvidenceCritic requirement)
    gov_layer = next(lr for lr in outcome.layers if lr.name == "Governance")
    assert gov_layer.findings


def test_ai_engineering_layer_integrated(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=False, seed=0
    )
    ai = outcome.ai_engineering
    from start.ai_engineering.adapters import ADAPTER_CLASSES
    assert ai.total == len(ADAPTER_CLASSES)
    assert ai.available_count >= 1  # OpenTelemetry executes for real
    ai_layer = next(lr for lr in outcome.layers if lr.name == "AI-Engineering")
    # unavailable backends surface as a warning, not a silent skip
    assert ai_layer.warnings


def test_dashboard_generated_all_formats(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=True, seed=0
    )
    assert set(outcome.dashboard_paths) == {"json", "md", "html"}
    from pathlib import Path

    for p in outcome.dashboard_paths.values():
        assert Path(p).exists()
    d = json.loads(Path(outcome.dashboard_paths["json"]).read_text())
    assert "executive_summary" in d and "governance_findings" in d


def test_enterprise_graph_mode(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path),
        run_dl=False, enterprise_mode=True, seed=0,
    )
    assert outcome.graph_paths
    names = [p.split("/")[-1] for p in outcome.graph_paths]
    assert "review_graph.json" in names


def test_no_graph_without_enterprise_mode(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path),
        run_dl=False, enterprise_mode=False, seed=0,
    )
    assert outcome.graph_paths == []


def test_cnn_config_flows_to_dashboard(churn_frame, tmp_path):
    cnn = {"preset": "simple_cnn_small", "n_blocks": 2, "base_channels": 16, "param_count": 9999}
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path),
        run_dl=False, cnn_config=cnn, seed=0,
    )
    from pathlib import Path

    md = Path(outcome.dashboard_paths["md"]).read_text()
    assert "param_count" in md and "9999" in md


# --- v2.1.0 co-pilot integration --------------------------------------------- #
def test_copilot_artifacts_in_outcome(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path),
        run_dl=True, architecture="wide_deep", activation="gelu",
        costlier_errors="false_negatives", seed=0,
    )
    assert outcome.data_statistics is not None
    assert outcome.fe_recommendations is not None
    assert outcome.architecture_review is not None
    assert outcome.tuning_plan is not None
    assert outcome.action_log is not None and outcome.action_log.agents()
    # FN-costly routes the metric to PR-AUC
    assert outcome.metric_choice["primary_metric"] == "pr_auc"
    assert outcome.tuning_plan.primary_metric == "pr_auc"


def test_architecture_review_recommends_simpler_on_small_data(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path),
        run_dl=False, architecture="wide_deep", activation="gelu", seed=0,
    )
    ar = outcome.architecture_review
    assert ar.recommendation["family"] == "mlp"
    assert not ar.agrees


def test_action_log_covers_core_agents(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=True, seed=0
    )
    agents = set(outcome.action_log.agents())
    for expected in ("DatasetDiscoveryAgent", "FeatureEngineeringAgent",
                     "ArchitectureReviewAgent", "HyperparameterTuningAgent",
                     "GovernanceSignoffAgent", "EvidenceCriticAgent"):
        assert expected in agents


def test_sensitivity_runs_when_dl_trained(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=True, seed=0
    )
    assert outcome.sensitivity is not None
    assert outcome.sensitivity.most_sensitive_feature is not None
    # 0% shock equals baseline
    zero = [r for r in outcome.sensitivity.shock_rows if r.shock == 0.0]
    assert zero and all(r.drift == 0.0 for r in zero)


def test_dashboard_has_all_copilot_sections(churn_frame, tmp_path):
    import json

    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=True, seed=0
    )
    md = (tmp_path / "dashboards" / outcome.run_id / "dashboard.md").read_text()
    html = (tmp_path / "dashboards" / outcome.run_id / "dashboard.html").read_text()
    d = json.loads((tmp_path / "dashboards" / outcome.run_id / "dashboard.json").read_text())
    for section in ("Initial Data Statistics", "Feature-Engineering Recommendations",
                    "Architecture Review", "Hyperparameter Tuning", "Sensitivity Analysis",
                    "Agentic Action Log"):
        assert section in md, f"md missing {section}"
        assert section in html, f"html missing {section}"
    for key in ("initial_data_statistics", "feature_engineering_recommendations",
                "architecture_review", "hyperparameter_tuning", "sensitivity_analysis",
                "agentic_action_log", "metric_choice"):
        assert key in d, f"json missing {key}"


def test_dataset_source_in_outcome_and_dashboard(churn_frame, tmp_path):
    import json

    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=False, seed=0
    )
    assert outcome.dataset_source is not None
    d = json.loads((tmp_path / "dashboards" / outcome.run_id / "dashboard.json").read_text())
    assert d.get("dataset_source") is not None
    assert "data_hash" in d["dataset_source"]
    md = (tmp_path / "dashboards" / outcome.run_id / "dashboard.md").read_text()
    assert "## Dataset Source" in md


def test_demo_dataset_source_has_public_url(tmp_path):
    import json

    from start.modeling.data import load_attrition_dataset

    df = load_attrition_dataset(seed=0)
    outcome = EnterpriseReviewOrchestrator().run(
        df, user_target="attrition", output_root=str(tmp_path), run_dl=False, seed=0
    )
    d = json.loads((tmp_path / "dashboards" / outcome.run_id / "dashboard.json").read_text())
    src = d["dataset_source"]
    if src["kind"] == "builtin_demo":
        assert src["public_url"].startswith("https://")
        assert src["reason_selected"] and src["task_suitability"]


# --- v2.1.1 visibility (Sections A/K/L/M/N) ---------------------------------- #
def test_agent_traces_in_outcome(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=True, seed=0
    )
    assert outcome.trace_log is not None
    agents = outcome.trace_log.agents()
    for expected in ("DatasetDiscoveryAgent", "TaskInferenceAgent", "ArchitectureReviewAgent",
                     "HyperparameterTuningAgent", "GovernanceSignoffAgent", "EvidenceCriticAgent"):
        assert expected in agents
    # every trace has thinking fields populated
    for t in outcome.trace_log.traces:
        assert t.inputs and t.decision
        assert t.confidence is not None


def test_activation_report_in_outcome(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=False, seed=0
    )
    assert outcome.activation_report is not None
    # no LLM passed -> deterministic, never silent
    assert outcome.activation_report.status == "DETERMINISTIC"


def test_artifact_registry_in_outcome(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path),
        run_dl=True, enterprise_mode=True, seed=0,
    )
    assert outcome.artifact_registry is not None
    names = outcome.artifact_registry.names()
    # dashboard files self-register; graph artifacts register in enterprise mode
    assert any("dashboard" in n for n in names)


def test_control_surface_in_dashboard(churn_frame, tmp_path):
    import json

    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=False, seed=0
    )
    d = json.loads((tmp_path / "dashboards" / outcome.run_id / "dashboard.json").read_text())
    cs = d["ai_engineering_control_surface"]
    from start.ai_engineering.adapters import ADAPTER_CLASSES
    assert len(cs) == len(ADAPTER_CLASSES)
    for row in cs:
        assert "purpose" in row and "role" in row and "install_guidance" in row


def test_visibility_sections_in_all_dashboard_formats(churn_frame, tmp_path):
    import json

    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=True, seed=0
    )
    base = tmp_path / "dashboards" / outcome.run_id
    md = (base / "dashboard.md").read_text()
    html = (base / "dashboard.html").read_text()
    d = json.loads((base / "dashboard.json").read_text())
    for section in ("LLM Activation", "Agent Reasoning Traces",
                    "AI-Engineering Control Surface", "Artifact Catalog"):
        assert section in md, f"md missing {section}"
        assert section in html, f"html missing {section}"
    for key in ("llm_activation", "agent_reasoning_traces",
                "ai_engineering_control_surface", "artifact_catalog"):
        assert key in d, f"json missing {key}"


# --- v2.1.1 remediation: model execution surfacing ---------------------------- #
def test_copilot_execution_in_outcome(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path),
        run_dl=True, split_props=(0.60, 0.20, 0.20), seed=0,
    )
    ce = outcome.copilot_execution
    assert ce is not None
    assert {r["split"] for r in ce.split_table} == {"train", "test", "oos"}
    assert set(ce.metrics_by_split) == {"train", "test", "oos"}
    assert ce.global_importance  # explainability table present


def test_execution_sections_in_all_dashboard_formats(churn_frame, tmp_path):
    import json

    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path), run_dl=True, seed=0
    )
    base = tmp_path / "dashboards" / outcome.run_id
    md = (base / "dashboard.md").read_text()
    html = (base / "dashboard.html").read_text()
    d = json.loads((base / "dashboard.json").read_text())
    for section in ("Train/Test/OOS Split", "Metrics by Split", "Explainability"):
        assert section in md, f"md missing {section}"
        assert section in html, f"html missing {section}"
    assert d.get("model_execution") is not None
    assert d.get("metrics_by_split") is not None


def test_custom_split_proportions_honored(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path),
        run_dl=True, split_props=(0.70, 0.15, 0.15), seed=0,
    )
    by = {r["split"]: r["percent"] for r in outcome.copilot_execution.split_table}
    assert 65 <= by["train"] <= 75


def test_requested_provider_shows_fallback_not_none(churn_frame, tmp_path, monkeypatch):
    for var in ("OPENAI_API_KEY", "OPENAI_KEY"):
        monkeypatch.delenv(var, raising=False)
    from start.core.config import LLMConfig
    from start.providers.llm import get_llm_provider

    llm = get_llm_provider(LLMConfig(provider="openai"), expected_domain="public")
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path),
        run_dl=False, llm=llm, requested_provider="openai", seed=0,
    )
    # the activation report must name openai and report FALLBACK, never "none"
    assert outcome.activation_report.provider == "openai"
    assert outcome.activation_report.status == "FALLBACK"


def test_tuning_actually_runs_in_enterprise(churn_frame, tmp_path):
    import json

    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path),
        run_dl=True, tuning_strategy="bounded_random_search", tuning_trials=5, seed=0,
    )
    tr = outcome.tuning_run
    assert tr is not None and tr.ran is True
    assert len(tr.trials) == 5
    assert sum(1 for t in tr.trials if t.status == "best") == 1
    d = json.loads((tmp_path / "dashboards" / outcome.run_id / "dashboard.json").read_text())
    assert d["tuning_execution"]["ran"] is True
    md = (tmp_path / "dashboards" / outcome.run_id / "dashboard.md").read_text()
    assert "Tuning trials (executed)" in md


def test_tuning_disabled_records_explicit_note(churn_frame, tmp_path):
    outcome = EnterpriseReviewOrchestrator().run(
        churn_frame, user_target="churned", output_root=str(tmp_path),
        run_dl=True, tuning_strategy="none", seed=0,
    )
    assert outcome.tuning_run is not None
    assert outcome.tuning_run.ran is False
