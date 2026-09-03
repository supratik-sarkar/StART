from __future__ import annotations

import json

import pytest

from start.reporting.dashboard import (
    DASHBOARD_SECTIONS,
    DashboardModel,
    render_dashboard_html,
    render_dashboard_json,
    render_dashboard_md,
    write_dashboard,
)


@pytest.fixture()
def model():
    return DashboardModel(
        run_id="RUN-DASH",
        task_type="binary_classification",
        target="attrition",
        modality="tabular",
        recommended_family="mlp",
        cohort_metrics={"test": {"auc_roc": 0.82, "accuracy": 0.78, "f1": 0.70}},
        dataset_summary="569 rows x 31 columns.",
        model_summary="Tabular MLP.",
        explainability={"method": "integrated_gradients", "top_feature": "balance"},
        robustness={"baseline_auc": 0.82, "max_noise_drift": -0.03},
        ai_engineering_rows=[
            {
                "adapter": "OpenTelemetry",
                "category": "telemetry",
                "status": "complete",
                "runtime_s": 0.05,
                "artifacts": 1,
                "evidence": 1,
            },
            {
                "adapter": "OPA",
                "category": "policy",
                "status": "not_installed",
                "runtime_s": 0.0,
                "artifacts": 0,
                "evidence": 1,
            },
        ],
        findings=[
            {
                "title": "Target Leakage",
                "description": "post-event var",
                "severity": "High",
                "materiality": "High",
                "risk_category": "Data Quality",
                "evidence_ids": ["EV-3"],
                "recommendation": "Remove post-event variables.",
            },
            {
                "title": "Calibration drift",
                "description": "ECE elevated",
                "severity": "Medium",
                "materiality": "Medium",
                "risk_category": "Calibration",
                "evidence_ids": ["EV-7"],
                "recommendation": "Recalibrate.",
            },
        ],
        evidence_summary={"total": 8},
        evidence_rows=[
            {"evidence_id": "EV-1", "test_name": "Discovery", "status": "pass"},
            {"evidence_id": "EV-3", "test_name": "Leakage", "status": "fail"},
        ],
        signoff="NOT READY FOR SIGN-OFF. 1 blocking finding.",
        critique_ok=True,
    )


def test_all_sections_present_in_markdown(model):
    md = render_dashboard_md(model)
    for section in DASHBOARD_SECTIONS:
        assert section in md, f"missing section: {section}"


def test_all_sections_present_in_html(model):
    h = render_dashboard_html(model)
    for section in DASHBOARD_SECTIONS:
        assert section in h, f"missing section: {section}"


def test_html_is_self_contained(model):
    h = render_dashboard_html(model)
    assert h.startswith("<!doctype html>")
    # no external assets (audit/offline safe)
    assert "http://" not in h
    assert "cdn" not in h.lower()
    assert "<script src=" not in h


def test_json_structure(model):
    d = json.loads(render_dashboard_json(model))
    assert set(
        [
            "executive_summary",
            "dataset_review",
            "model_review",
            "validation_review",
            "explainability_review",
            "robustness_review",
            "ai_engineering_review",
            "governance_findings",
            "evidence_ledger_summary",
            "final_signoff",
        ]
    ) <= set(d)
    ex = d["executive_summary"]
    assert ex["total_findings"] == 2
    assert ex["blocking_findings"] == 1  # the High finding
    assert ex["ai_engineering_available"] == 1  # only OpenTelemetry complete
    assert ex["ai_engineering_total"] == 2


def test_write_dashboard_produces_three_files(model, tmp_path):
    paths = write_dashboard(model, str(tmp_path), "RUN-DASH")
    assert set(paths) == {"json", "md", "html"}
    for key, p in paths.items():
        from pathlib import Path

        assert Path(p).exists()
        assert Path(p).suffix == f".{key if key != 'md' else 'md'}"
    # findings rendered with severity in md
    md = (tmp_path / "dashboards" / "RUN-DASH" / "dashboard.md").read_text()
    assert "Target Leakage" in md and "High" in md


def test_cnn_config_rendered():
    model = DashboardModel(
        run_id="RUN-CNN",
        task_type="vision_image_classification",
        target="label",
        modality="vision",
        recommended_family="simple_cnn_small",
        cnn_config={
            "preset": "simple_cnn_small",
            "n_blocks": 2,
            "base_channels": 16,
            "kernel_size": 3,
            "pooling": "max",
            "dropout": 0.1,
            "dense": 64,
            "param_count": 12345,
        },
    )
    md = render_dashboard_md(model)
    assert "CNN configuration" in md
    assert "param_count" in md and "12345" in md
    h = render_dashboard_html(model)
    assert "param_count" in h


def test_empty_findings_handled(model):
    model.findings = []
    md = render_dashboard_md(model)
    assert "No findings raised." in md
    h = render_dashboard_html(model)
    assert "No findings raised." in h


# -- v2.1.1 visibility sections ----------------------------------------------- #
def _v211_model():
    from start.reporting.dashboard import DashboardModel

    return DashboardModel(
        run_id="RUN-X",
        task_type="binary_classification",
        target="y",
        modality="tabular",
        recommended_family="mlp",
        activation_report={
            "provider": "openai",
            "model": "gpt-4.1",
            "trust_domain": "public",
            "endpoint": "https://api.openai.com/v1",
            "status": "FALLBACK",
            "detail": "no key",
        },
        agent_traces=[
            {
                "agent": "ArchitectureReviewAgent",
                "inputs": "569 rows",
                "reasoning": "small tabular",
                "decision": "recommend MLP",
                "confidence": 0.8,
                "alternative_considered": "Wide&Deep",
                "evidence_ids": ["ARCH-01"],
            }
        ],
        control_surface=[
            {
                "adapter": "OPA",
                "category": "policy",
                "purpose": "policy as code",
                "role": "validate governance",
                "status": "not_installed",
                "would_do": "validate controls",
                "expected_outputs": ["policy_report.json"],
                "install_guidance": "install OPA",
                "artifacts": 0,
                "evidence": 1,
            }
        ],
        artifact_catalog=[
            {
                "name": "sensitivity.csv",
                "path": "/out/sensitivity.csv",
                "type": "table (CSV)",
                "category": "sensitivity",
                "description": "",
            }
        ],
    )


def test_dashboard_md_has_v211_sections():
    from start.reporting.dashboard import render_dashboard_md

    md = render_dashboard_md(_v211_model())
    for sec in (
        "## LLM Activation",
        "## Agent Reasoning Traces",
        "## AI-Engineering Control Surface",
        "## Artifact Catalog",
    ):
        assert sec in md, f"markdown missing {sec}"
    assert "ArchitectureReviewAgent" in md
    assert "policy as code" in md
    assert "sensitivity.csv" in md


def test_dashboard_html_has_v211_sections():
    from start.reporting.dashboard import render_dashboard_html

    html = render_dashboard_html(_v211_model())
    for sec in (
        "LLM Activation",
        "Agent Reasoning Traces",
        "AI-Engineering Control Surface",
        "Artifact Catalog",
    ):
        assert sec in html, f"html missing {sec}"
    assert "FALLBACK" in html


def test_dashboard_json_has_v211_keys():
    d = _v211_model().to_dict()
    for key in (
        "llm_activation",
        "agent_reasoning_traces",
        "ai_engineering_control_surface",
        "artifact_catalog",
    ):
        assert key in d
