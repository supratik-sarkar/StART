"""Enterprise dashboard generator.

Renders the audit-ready governance package in three formats from a single
``DashboardModel``:

    dashboard.json  - machine-readable, the source of truth
    dashboard.md    - human-readable Markdown
    dashboard.html  - self-contained styled HTML (no external assets)

Sections (all mandatory): Executive Summary, Dataset Review, Model Review,
Validation Review, Explainability Review, Robustness Review, AI-Engineering
Review, Governance Findings, Evidence Ledger Summary, Final Signoff.
"""

from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DASHBOARD_SECTIONS = (
    "Executive Summary",
    "Dataset Review",
    "Model Review",
    "Validation Review",
    "Explainability Review",
    "Robustness Review",
    "AI-Engineering Review",
    "Governance Findings",
    "Evidence Ledger Summary",
    "Final Signoff",
)


@dataclass
class DashboardModel:
    run_id: str
    task_type: str
    target: Any
    modality: str
    recommended_family: str
    cohort_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    dataset_summary: str = ""
    model_summary: str = ""
    cnn_config: dict[str, Any] | None = None
    validation_rows: list[dict[str, Any]] = field(default_factory=list)
    explainability: dict[str, Any] = field(default_factory=dict)
    robustness: dict[str, Any] = field(default_factory=dict)
    ai_engineering_rows: list[dict[str, Any]] = field(default_factory=list)
    findings: list[dict[str, Any]] = field(default_factory=list)
    evidence_summary: dict[str, int] = field(default_factory=dict)
    evidence_rows: list[dict[str, Any]] = field(default_factory=list)
    signoff: str = ""
    critique_ok: bool = True
    stage_timeline: list[dict[str, Any]] = field(default_factory=list)
    # v2.1.0 model execution sections
    data_statistics: dict[str, Any] | None = None
    fe_recommendations: list[dict[str, Any]] = field(default_factory=list)
    architecture_review: dict[str, Any] | None = None
    tuning_plan: dict[str, Any] | None = None
    sensitivity: dict[str, Any] | None = None
    action_log: list[dict[str, Any]] = field(default_factory=list)
    metric_choice: dict[str, Any] | None = None
    dataset_source: dict[str, Any] | None = None
    # v2.1.1 visibility
    agent_traces: list[dict[str, Any]] = field(default_factory=list)
    activation_report: dict[str, Any] | None = None
    control_surface: list[dict[str, Any]] = field(default_factory=list)
    artifact_catalog: list[dict[str, Any]] = field(default_factory=list)
    model_execution: dict[str, Any] | None = None
    tuning_run: dict[str, Any] | None = None
    kfold: dict[str, Any] | None = None
    review_journey: dict[str, Any] | None = None

    def executive_summary(self) -> dict[str, Any]:
        sev_counts: dict[str, int] = {}
        for f in self.findings:
            sev_counts[f["severity"]] = sev_counts.get(f["severity"], 0) + 1
        blocking = sum(v for k, v in sev_counts.items() if k in ("High", "Critical"))
        return {
            "run_id": self.run_id,
            "task_type": self.task_type,
            "target": self.target,
            "modality": self.modality,
            "recommended_family": self.recommended_family,
            "total_findings": len(self.findings),
            "blocking_findings": blocking,
            "severity_breakdown": sev_counts,
            "evidence_records": self.evidence_summary.get("total", len(self.evidence_rows)),
            "ai_engineering_available": sum(
                1 for r in self.ai_engineering_rows if r.get("status") == "complete"
            ),
            "ai_engineering_total": len(self.ai_engineering_rows),
            "evidence_critique": "PASSED" if self.critique_ok else "FAILED",
            "signoff": self.signoff,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "executive_summary": self.executive_summary(),
            "llm_activation": self.activation_report,
            "dataset_source": self.dataset_source,
            "dataset_review": {"summary": self.dataset_summary},
            "initial_data_statistics": self.data_statistics,
            "feature_engineering_recommendations": self.fe_recommendations,
            "model_review": {
                "summary": self.model_summary,
                "recommended_family": self.recommended_family,
                "cnn_config": self.cnn_config,
            },
            "architecture_review": self.architecture_review,
            "hyperparameter_tuning": self.tuning_plan,
            "tuning_execution": self.tuning_run,
            "kfold_tuning": self.kfold,
            "review_journey": self.review_journey,
            "metric_choice": self.metric_choice,
            "validation_review": {
                "cohort_metrics": self.cohort_metrics,
                "rows": self.validation_rows,
            },
            "model_execution": self.model_execution,
            "train_test_oos_split": (
                self.model_execution.get("split_table") if self.model_execution else None
            ),
            "metrics_by_split": (
                self.model_execution.get("metrics_by_split") if self.model_execution else None
            ),
            "explainability_review": self.explainability,
            "robustness_review": self.robustness,
            "sensitivity_analysis": self.sensitivity,
            "ai_engineering_review": {"adapters": self.ai_engineering_rows},
            "agentic_action_log": self.action_log,
            "agent_reasoning_traces": self.agent_traces,
            "ai_engineering_control_surface": self.control_surface,
            "artifact_catalog": self.artifact_catalog,
            "governance_findings": self.findings,
            "evidence_ledger_summary": {
                "summary": self.evidence_summary,
                "records": self.evidence_rows,
            },
            "final_signoff": {"decision": self.signoff, "evidence_critique": self.critique_ok},
            "stage_timeline": self.stage_timeline,
        }


def render_dashboard_json(model: DashboardModel) -> str:
    return json.dumps(model.to_dict(), indent=2, default=str)


def render_dashboard_md(model: DashboardModel) -> str:
    ex = model.executive_summary()
    lines = [
        f"# StART Enterprise Review Dashboard — `{model.run_id}`",
        "",
        "## Executive Summary",
        "",
        f"- Task: **{model.task_type}** | Target: `{model.target}` | Modality: {model.modality}",
        f"- Recommended model family: `{model.recommended_family}`",
        f"- Findings: **{ex['total_findings']}** ({ex['blocking_findings']} blocking) "
        f"— {ex['severity_breakdown']}",
        f"- AI-engineering controls available: "
        f"{ex['ai_engineering_available']}/{ex['ai_engineering_total']}",
        f"- Evidence records: {ex['evidence_records']} | "
        f"Evidence critique: **{ex['evidence_critique']}**",
        f"- Sign-off: {model.signoff}",
        "",
    ]
    # v2.1.1 Section A: LLM activation
    if model.activation_report:
        ar = model.activation_report
        lines += [
            "## LLM Activation", "",
            "| Field | Value |", "| --- | --- |",
            f"| Provider | {ar.get('provider')} |",
            f"| Model | {ar.get('model')} |",
            f"| Trust domain | {ar.get('trust_domain')} |",
            f"| Endpoint | {ar.get('endpoint')} |",
            f"| Status | **{ar.get('status')}** |",
        ]
        if ar.get("detail"):
            lines += ["", ar["detail"]]
    lines += [
        "## Dataset Review",
        "",
        model.dataset_summary or "_n/a_",
        "",
        "## Model Review",
        "",
        model.model_summary or f"Recommended family: `{model.recommended_family}`.",
    ]
    if model.cnn_config:
        lines += ["", "**CNN configuration:**", ""]
        lines += [f"- {k}: {v}" for k, v in model.cnn_config.items()]

    # v2.1.0 model execution: dataset source provenance
    # #6: the review journey — embed the committee transcript (decisions,
    # overrides, agent conversations) directly in the primary dashboard.
    if model.review_journey:
        rj = model.review_journey
        decisions = rj.get("decisions", [])
        overrides = rj.get("overrides", [])
        convos = rj.get("conversations", [])
        if decisions or convos:
            lines += ["", "## Review Journey", ""]
            if decisions:
                from start.review_tables import decision_ledger_markdown
                lines += [decision_ledger_markdown(decisions).rstrip()]
            if overrides:
                lines += ["", "**User overrides**", ""]
                for d in overrides:
                    lines.append(
                        f"- {d['key']}: chose `{d['effective']}` over "
                        f"recommended `{d['recommended']}` ({d['choice']})"
                    )
            if convos:
                lines += ["", "**Agent conversations**", ""]
                for ex in convos:
                    lines.append(f"- _{ex['agent']}_ (via {ex['backend']}): "
                                 f"Q: {ex['question']} — A: {ex['answer']}")

        # v2.3.0 #3/#12: reviewer challenge log
        challenges = rj.get("challenges", [])
        if challenges:
            lines += ["", "### Reviewer Challenges", "",
                      "| Status | Agent | Challenge | Evidence used |",
                      "| --- | --- | --- | --- |"]
            for c in challenges:
                lines.append(
                    f"| {c.get('status')} | {c.get('agent')} | {c.get('text')} "
                    f"| {', '.join(c.get('evidence_used', [])) or '—'} |"
                )
            cs_sum = rj.get("challenge_summary", {})
            lines.append("")
            lines.append(f"_Open: {cs_sum.get('open', 0)} · "
                         f"Closed: {cs_sum.get('closed', 0)} · "
                         f"Unresolved: {cs_sum.get('unresolved', 0)}_")

        # v2.3.0 #8/#12: ValidationAgent sensitivity review
        vr = rj.get("validation_review")
        if vr:
            lines += ["", "### ValidationAgent Review", "",
                      f"- Most sensitive feature: {vr.get('most_sensitive_feature')}",
                      f"- Max |drift|: {vr.get('max_abs_drift')}",
                      f"- Signoff impact: {vr.get('signoff_impact')}"]
            if vr.get("business_interpretation"):
                lines += ["", "**Business interpretation**", ""]
                lines += [f"- {b}" for b in vr["business_interpretation"]]

        # v2.3.0 #11/#12: MRM-grade signoff
        ms = rj.get("mrm_signoff")
        if ms:
            lines += ["", "### MRM Signoff Decision", "",
                      f"**Verdict: {ms.get('verdict')}**", "", ms.get("rationale", ""),
                      "", "| Factor | Status | Detail | Evidence |",
                      "| --- | --- | --- | --- |"]
            for f in ms.get("factors", []):
                lines.append(
                    f"| {f['factor']} | {f['status']} | {f['detail']} | {f['evidence']} |"
                )

    if model.dataset_source:
        src = model.dataset_source
        lines += [
            "", "## Dataset Source", "",
            "| Field | Value |", "| --- | --- |",
            f"| Name | {src.get('name')} |",
            f"| Kind | {src.get('kind')} |",
            f"| Rows / Columns | {src.get('n_rows')} / {src.get('n_columns')} |",
            f"| Target | {src.get('target_column')} |",
        ]
        if src.get("public_url"):
            lines.append(f"| Public source | {src['public_url']} |")
        if src.get("file_path"):
            lines.append(f"| File path | {src['file_path']} |")
            lines.append(f"| Detected format | {src.get('detected_format')} |")
        if src.get("loading_route"):
            lines.append(f"| Loading route | {src['loading_route']} |")
        if src.get("data_hash"):
            lines.append(f"| Data hash | {src['data_hash']} |")
        if src.get("reason_selected"):
            lines += ["", f"**Why selected:** {src['reason_selected']}"]
        if src.get("task_suitability"):
            lines += ["", f"**Task suitability:** {src['task_suitability']}"]

    # v2.1.0 model execution: initial data statistics
    if model.data_statistics:
        ds = model.data_statistics
        lines += [
            "", "## Initial Data Statistics", "",
            "| Metric | Value |", "| --- | --- |",
            f"| Rows | {ds.get('n_rows')} |",
            f"| Columns | {ds.get('n_columns')} |",
            f"| Target type | {ds.get('target_type')} |",
            f"| Numeric / Categorical | {ds.get('n_numeric')} / {ds.get('n_categorical')} |",
            f"| Duplicate rows | {ds.get('n_duplicate_rows')} |",
            f"| Leakage candidates | {len(ds.get('leakage_candidates', []))} |",
            f"| Imbalance | {ds.get('imbalance_warning')} |",
            f"| Suggested split | {ds.get('suggested_split')} |",
        ]
        if ds.get("class_distribution"):
            lines += ["", "**Class distribution:** " + ", ".join(
                f"{k}={v:.1%}" for k, v in ds["class_distribution"].items()
            )]

    # v2.1.0 model execution: feature-engineering recommendations
    if model.fe_recommendations:
        lines += [
            "", "## Feature-Engineering Recommendations", "",
            "| Step | Recommendation | Reason | Evidence | Risk if ignored | Default |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in model.fe_recommendations:
            lines.append(
                f"| {r['step']} | {r['recommendation']} | {r['reason']} "
                f"| {r['evidence_id']} | {r['risk_if_ignored']} | {r['default_action']} |"
            )

    # v2.1.0 model execution: architecture review
    if model.architecture_review:
        ar = model.architecture_review
        lines += [
            "", "## Architecture Review", "",
            f"- User selected: `{ar['user_choice']['family']} + {ar['user_choice']['activation']}`",
            f"- Agent recommends: `{ar['recommendation']['family']} + "
            f"{ar['recommendation']['activation']}`",
            f"- Reason: {ar['reason']}",
            f"- Evidence: {ar['evidence_id']}",
            f"- Risk if ignored: {ar['risk_if_ignored']}",
            f"- Agreement: {'yes' if ar['agrees'] else 'no — user decision required'}",
        ]

    # v2.1.0 model execution: hyperparameter tuning
    if model.tuning_plan:
        tp = model.tuning_plan
        lines += [
            "", "## Hyperparameter Tuning", "",
            f"- Strategy: {tp['strategy']}",
            f"- Primary metric: {tp['primary_metric']}",
            f"- Trials: {tp['n_trials']} | Early stopping: {tp['early_stopping']}",
            f"- Validation: {tp['validation']} (no test/OOS leakage)",
            f"- Evidence: {tp['evidence_id']}",
        ]
    # v2.1.1 remediation Section H: real tuning trials
    if model.tuning_run:
        tr = model.tuning_run
        if tr.get("ran"):
            param_keys = []
            for t in tr.get("trials", []):
                for k in t.get("params", {}).keys():
                    if k not in param_keys:
                        param_keys.append(k)
            lines += ["", "### Tuning trials (executed)", "",
                      f"- Best metric: {tr.get('best_metric'):.4f} | "
                      f"best params: {tr.get('best_params')}",
                      ""]
            header_cols = ["Trial"] + param_keys + ["Validation metric", "Status"]
            lines.append("| " + " | ".join(header_cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(header_cols)) + " |")
            for t in tr.get("trials", []):
                p = t["params"]
                cells = []
                for k in param_keys:
                    cells.append(str(p.get(k, "-")))
                lines.append(
                    f"| {t['trial']} | " + " | ".join(cells) +
                    f" | {t['validation_metric']:.4f} | {t['status']} |"
                )
        else:
            lines += ["", f"_{tr.get('note', 'Tuning disabled.')}_"]

    # v2.3.1 #7: K-fold tuning summary + per-fold metrics (train-only).
    if model.kfold:
        kf = model.kfold
        lines += ["", "## K-fold Tuning (train-only, stratified)", "",
                  f"- Method: {kf.get('method')} ({kf.get('n_folds')}-fold)",
                  f"- Primary metric: {kf.get('primary_metric')}",
                  f"- Train rows used: {kf.get('train_rows')} "
                  f"(test/OOS excluded from selection: {kf.get('excluded_rows')})",
                  f"- Best params: {kf.get('best_params')}",
                  f"- Best mean: {kf.get('best_mean_metric')} "
                  f"(std {kf.get('best_std_metric')})",
                  "", "| Fold | Metric | n_train | n_val |",
                  "| --- | --- | --- | --- |"]
        for f in kf.get("best_fold_results", []):
            lines.append(f"| {f['fold']} | {f['metric']:.4f} | {f['n_train']} "
                         f"| {f['n_val']} |")

    # v2.1.1 remediation: model execution — split table, metrics-by-split,
    # training diagnostics, explainability (Sections D/G/I/J/K)
    if model.model_execution:
        me = model.model_execution
        split = me.get("split_table") or []
        if split:
            lines += ["", "## Train/Test/OOS Split", "",
                      "| Split | Rows | Percent | Positive rate | Negative rate |",
                      "| --- | --- | --- | --- | --- |"]
            for r in split:
                lines.append(
                    f"| {r['split']} | {r['rows']} | {r['percent']}% "
                    f"| {r['positive_rate']} | {r['negative_rate']} |"
                )
        mbs = me.get("metrics_by_split") or {}
        if mbs:
            keys = ["auc_roc", "pr_auc", "accuracy", "precision", "recall", "f1",
                    "specificity", "brier_score", "ece"]
            lines += ["", "## Metrics by Split", "",
                      "| Split | " + " | ".join(keys) + " |",
                      "| --- " * (len(keys) + 1) + "|"]
            for split_name, m in mbs.items():
                cells = " | ".join(
                    f"{m.get(k, float('nan')):.4f}" if isinstance(m.get(k), (int, float))
                    else "—" for k in keys
                )
                lines.append(f"| {split_name} | {cells} |")
            if me.get("generalization_gap") is not None:
                lines += ["", f"Generalization gap (train - OOS): {me['generalization_gap']:.4f}"]
        gi = me.get("global_importance") or []
        if gi:
            lines += ["", f"## Explainability — {me.get('explainability_method')}", "",
                      "| Rank | Feature | Importance | Direction |",
                      "| --- | --- | --- | --- |"]
            for r in gi:
                lines.append(
                    f"| {r['rank']} | {r['feature']} | {r['importance']} | {r['direction']} |"
                )

    lines += ["", "## Validation Review", ""]
    if model.cohort_metrics:
        lines += ["| Cohort | AUC-ROC | Accuracy | F1 |", "| --- | --- | --- | --- |"]
        for cohort, m in model.cohort_metrics.items():
            lines.append(
                f"| {cohort} | {m.get('auc_roc', float('nan')):.4f} "
                f"| {m.get('accuracy', float('nan')):.4f} | {m.get('f1', float('nan')):.4f} |"
            )
    else:
        lines.append("_No cohort metrics (diagnostics-only review)._")

    lines += ["", "## Explainability Review", "", _kv_block(model.explainability)]
    lines += ["", "## Robustness Review", "", _kv_block(model.robustness)]

    # v2.1.1 Sections L/M: AI-Engineering Control Surface (purpose/role/install)
    if model.control_surface:
        lines += ["", "## AI-Engineering Control Surface", "",
                  "| Adapter | Purpose | Role | Status | Outputs | Install guidance |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in model.control_surface:
            outputs = ", ".join(r.get("expected_outputs", [])) or "—"
            lines.append(
                f"| {r.get('adapter')} | {r.get('purpose', '')} | {r.get('role', '')} "
                f"| {r.get('status')} | {outputs} | {r.get('install_guidance', '') or '—'} |"
            )
    else:
        lines += ["", "## AI-Engineering Review", "",
                  "| Adapter | Category | Status | Runtime (s) | Artifacts | Evidence |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for r in model.ai_engineering_rows:
            lines.append(
                f"| {r.get('adapter')} | {r.get('category')} | {r.get('status')} "
                f"| {r.get('runtime_s', 0)} | {r.get('artifacts', 0)} | {r.get('evidence', 0)} |"
            )

    # Sensitivity analysis (#4): per-row feature/shock/baseline/shocked/delta/risk
    if model.sensitivity:
        sa = model.sensitivity
        lines += [
            "", "## Sensitivity Analysis", "",
            f"- Metric: {sa.get('metric_name')}",
            f"- Baseline (0% shock): {sa.get('baseline')}",
            f"- Most sensitive feature: {sa.get('most_sensitive_feature')}",
            f"- Max |drift|: {sa.get('max_abs_drift')}",
        ]
        rows = sa.get("rows", [])
        if rows:
            base = sa.get("baseline", 0.0)
            lines += ["", "| Feature | Shock % | Baseline | Shocked | Delta | Risk impact |",
                      "| --- | --- | --- | --- | --- | --- |"]
            for r in rows[:35]:
                lines.append(
                    f"| {r['feature']} | {int(r['shock'] * 100):+d}% | {base:.4f} "
                    f"| {r['metric']:.4f} | {r['drift']:+.4f} | "
                    f"{r.get('risk_impact', '')} |"
                )

    # v2.1.0 model execution: agentic action log
    if model.action_log:
        lines += [
            "", "## Agentic Action Log", "",
            "| Agent | Input reviewed | Action | Recommendation | Evidence | User decision |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for a in model.action_log:
            lines.append(
                f"| {a['agent']} | {a['input_reviewed']} | {a['action']} "
                f"| {a.get('recommendation') or '—'} | {', '.join(a.get('evidence_ids', [])) or '—'} "
                f"| {a.get('user_decision') or '—'} |"
            )

    # v2.1.1 Section K: agent reasoning traces (thinking visibility)
    if model.agent_traces:
        lines += [
            "", "## Agent Reasoning Traces", "",
            "| Agent | Inputs | Reasoning | Decision | Confidence | Alternative | Evidence |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for t in model.agent_traces:
            conf = f"{t['confidence']:.0%}" if t.get("confidence") is not None else "—"
            lines.append(
                f"| {t['agent']} | {t['inputs']} | {t.get('reasoning') or '—'} "
                f"| {t['decision']} | {conf} | {t.get('alternative_considered') or '—'} "
                f"| {', '.join(t.get('evidence_ids', [])) or '—'} |"
            )

    lines += ["", "## Governance Findings", ""]
    if model.findings:
        lines += ["| Severity | Materiality | Category | Title | Evidence | Recommendation |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for f in model.findings:
            lines.append(
                f"| {f['severity']} | {f['materiality']} | {f['risk_category']} "
                f"| {f['title']} | {', '.join(f['evidence_ids']) or '—'} | {f['recommendation']} |"
            )
    else:
        lines.append("_No findings raised._")

    lines += ["", "## Evidence Ledger Summary", "",
              f"Total records: {model.evidence_summary.get('total', len(model.evidence_rows))}", ""]
    if model.evidence_rows:
        lines += ["| Evidence ID | Test | Status |", "| --- | --- | --- |"]
        for e in model.evidence_rows[:50]:
            lines.append(f"| {e.get('evidence_id', '—')} | {e.get('test_name')} | {e.get('status')} |")

    lines += ["", "## Final Signoff", "",
              f"**Evidence critique:** {'PASSED' if model.critique_ok else 'FAILED'}",
              "", model.signoff or "_pending_", ""]

    # v2.1.1 Section N/Q: artifact catalog
    if model.artifact_catalog:
        lines += ["", "## Artifact Catalog", "",
                  "| Artifact | Type | Category | Location |",
                  "| --- | --- | --- | --- |"]
        for a in model.artifact_catalog:
            lines.append(
                f"| {a.get('name')} | {a.get('type')} | {a.get('category')} | {a.get('path')} |"
            )
    return "\n".join(lines) + "\n"


def _kv_block(d: dict[str, Any]) -> str:
    if not d:
        return "_n/a_"
    return "\n".join(f"- {k}: {v}" for k, v in d.items())


_HTML_STYLE = """
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#f6f7f9;color:#1a1a1a}
 header{background:#0f172a;color:#fff;padding:24px 32px}
 header h1{margin:0;font-size:20px} header p{margin:4px 0 0;color:#94a3b8;font-size:13px}
 main{max-width:1100px;margin:0 auto;padding:24px 32px}
 section{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:18px 22px;margin:16px 0}
 .qa{background:#f6f8fa;border-left:3px solid #0969da;padding:8px 12px;margin:8px 0;border-radius:4px}
 .qa .q{font-weight:600}
 section h2{margin:0 0 12px;font-size:15px;color:#0f172a;border-bottom:2px solid #e2e8f0;padding-bottom:8px}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #eef2f6}
 th{color:#475569;font-weight:600;background:#f8fafc}
 .kpi{display:inline-block;margin:4px 18px 4px 0}
 .kpi b{display:block;font-size:22px;color:#0f172a}
 .kpi span{font-size:12px;color:#64748b}
 .sev-Critical{color:#fff;background:#991b1b;padding:2px 8px;border-radius:4px;font-size:12px}
 .sev-High{color:#fff;background:#c2410c;padding:2px 8px;border-radius:4px;font-size:12px}
 .sev-Medium{color:#fff;background:#a16207;padding:2px 8px;border-radius:4px;font-size:12px}
 .sev-Low{color:#fff;background:#475569;padding:2px 8px;border-radius:4px;font-size:12px}
 .ok{color:#15803d;font-weight:600}.bad{color:#b91c1c;font-weight:600}
 .status-complete{color:#15803d}.status-not_installed{color:#b45309}.status-error{color:#b91c1c}
</style>
"""


def render_dashboard_html(model: DashboardModel) -> str:
    ex = model.executive_summary()

    def esc(x: Any) -> str:
        return html.escape(str(x))

    def table(headers: list[str], rows: list[list[Any]]) -> str:
        head = "".join(f"<th>{esc(h)}</th>" for h in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows
        )
        return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

    crit = "ok" if model.critique_ok else "bad"
    kpis = "".join(
        f'<div class="kpi"><b>{esc(v)}</b><span>{esc(k)}</span></div>'
        for k, v in [
            ("Findings", ex["total_findings"]),
            ("Blocking", ex["blocking_findings"]),
            ("Evidence", ex["evidence_records"]),
            ("AI controls", f"{ex['ai_engineering_available']}/{ex['ai_engineering_total']}"),
        ]
    )

    sections = [
        f"<section><h2>Executive Summary</h2>{kpis}"
        f"<p>Task <b>{esc(model.task_type)}</b> · target <code>{esc(model.target)}</code> · "
        f"modality {esc(model.modality)} · recommended <code>{esc(model.recommended_family)}</code></p>"
        f"<p>Evidence critique: <span class='{crit}'>"
        f"{'PASSED' if model.critique_ok else 'FAILED'}</span></p></section>",
        f"<section><h2>Dataset Review</h2><p>{esc(model.dataset_summary or 'n/a')}</p></section>",
    ]

    # v2.1.1 Section A: LLM activation
    if model.activation_report:
        ar = model.activation_report
        status_cls = {"CONNECTED": "ok", "FAILED": "bad", "FALLBACK": "warn"}.get(
            ar.get("status"), "warn"
        )
        arows = [
            ["Provider", esc(ar.get("provider"))], ["Model", esc(ar.get("model"))],
            ["Trust domain", esc(ar.get("trust_domain"))], ["Endpoint", esc(ar.get("endpoint"))],
            ["Status", f"<span class='{status_cls}'>{esc(ar.get('status'))}</span>"],
        ]
        detail = f"<p>{esc(ar['detail'])}</p>" if ar.get("detail") else ""
        sections.append(
            "<section><h2>LLM Activation</h2>" + table(["Field", "Value"], arows)
            + detail + "</section>"
        )

    model_html = f"<p>{esc(model.model_summary or model.recommended_family)}</p>"
    if model.cnn_config:
        model_html += table(
            ["Parameter", "Value"], [[esc(k), esc(v)] for k, v in model.cnn_config.items()]
        )
    sections.append(f"<section><h2>Model Review</h2>{model_html}</section>")

    # v2.1.0 model execution HTML sections
    if model.dataset_source:
        src = model.dataset_source
        srows = [
            ["Name", esc(src.get("name"))], ["Kind", esc(src.get("kind"))],
            ["Rows / Columns", f"{esc(src.get('n_rows'))} / {esc(src.get('n_columns'))}"],
            ["Target", esc(src.get("target_column"))],
        ]
        if src.get("public_url"):
            srows.append([
                "Public source",
                f"<a href='{esc(src['public_url'])}'>{esc(src['public_url'])}</a>",
            ])
        if src.get("file_path"):
            srows.append(["File path", esc(src.get("file_path"))])
            srows.append(["Detected format", esc(src.get("detected_format"))])
        if src.get("loading_route"):
            srows.append(["Loading route", esc(src.get("loading_route"))])
        if src.get("data_hash"):
            srows.append(["Data hash", esc(src.get("data_hash"))])
        extra = ""
        if src.get("reason_selected"):
            extra += f"<p><b>Why selected:</b> {esc(src['reason_selected'])}</p>"
        if src.get("task_suitability"):
            extra += f"<p><b>Task suitability:</b> {esc(src['task_suitability'])}</p>"
        sections.append(
            "<section><h2>Dataset Source</h2>"
            + table(["Field", "Value"], srows) + extra + "</section>"
        )
    # #6: Review Journey embedded in the primary dashboard (committee transcript).
    if model.review_journey:
        rj = model.review_journey
        decisions = rj.get("decisions", [])
        overrides = rj.get("overrides", [])
        convos = rj.get("conversations", [])
        if decisions or convos:
            parts_rj = ["<section><h2>Review Journey</h2>"]
            if decisions:
                drows = [[esc(d["key"]), esc(d["recommended"]), esc(d["effective"]),
                          esc(d["choice"])] for d in decisions]
                parts_rj.append("<h3>Decisions</h3>" + table(
                    ["Checkpoint", "Recommended", "User chose", "Outcome"], drows))
            if overrides:
                parts_rj.append("<h3>User overrides</h3><ul>" + "".join(
                    f"<li><b>{esc(d['key'])}</b>: chose <code>{esc(d['effective'])}</code> "
                    f"over <code>{esc(d['recommended'])}</code> ({esc(d['choice'])})</li>"
                    for d in overrides) + "</ul>")
            if convos:
                parts_rj.append("<h3>Agent conversations</h3>" + "".join(
                    f"<div class='qa'><div>{esc(ex['agent'])} "
                    f"<small>(via {esc(ex['backend'])})</small></div>"
                    f"<div class='q'>Q: {esc(ex['question'])}</div>"
                    f"<div>A: {esc(ex['answer'])}</div></div>"
                    for ex in convos))
            parts_rj.append("</section>")
            sections.append("".join(parts_rj))
    if model.data_statistics:
        ds = model.data_statistics
        srows = [
            ["Rows", esc(ds.get("n_rows"))], ["Columns", esc(ds.get("n_columns"))],
            ["Target type", esc(ds.get("target_type"))],
            ["Numeric / Categorical", f"{esc(ds.get('n_numeric'))} / {esc(ds.get('n_categorical'))}"],
            ["Duplicate rows", esc(ds.get("n_duplicate_rows"))],
            ["Leakage candidates", esc(len(ds.get("leakage_candidates", [])))],
            ["Imbalance", esc(ds.get("imbalance_warning"))],
            ["Suggested split", esc(ds.get("suggested_split"))],
        ]
        sections.append(
            "<section><h2>Initial Data Statistics</h2>"
            + table(["Metric", "Value"], srows) + "</section>"
        )
    if model.fe_recommendations:
        frows = [
            [esc(r["step"]), esc(r["recommendation"]), esc(r["evidence_id"]),
             esc(r["risk_if_ignored"]), esc(r["default_action"])]
            for r in model.fe_recommendations
        ]
        sections.append(
            "<section><h2>Feature-Engineering Recommendations</h2>"
            + table(["Step", "Recommendation", "Evidence", "Risk if ignored", "Default"], frows)
            + "</section>"
        )
    if model.architecture_review:
        ar = model.architecture_review
        agree = "ok" if ar["agrees"] else "bad"
        sections.append(
            "<section><h2>Architecture Review</h2>"
            f"<p>User: <code>{esc(ar['user_choice']['family'])}+"
            f"{esc(ar['user_choice']['activation'])}</code> "
            f"&rarr; Recommended: <code>{esc(ar['recommendation']['family'])}+"
            f"{esc(ar['recommendation']['activation'])}</code></p>"
            f"<p>{esc(ar['reason'])}</p>"
            f"<p>Agreement: <span class='{agree}'>"
            f"{'yes' if ar['agrees'] else 'review needed'}</span> "
            f"(evidence {esc(ar['evidence_id'])})</p></section>"
        )
    if model.tuning_plan:
        tp = model.tuning_plan
        sections.append(
            "<section><h2>Hyperparameter Tuning</h2>"
            + table(["Setting", "Value"], [
                ["Strategy", esc(tp["strategy"])], ["Primary metric", esc(tp["primary_metric"])],
                ["Trials", esc(tp["n_trials"])], ["Early stopping", esc(tp["early_stopping"])],
                ["Validation", esc(tp["validation"])], ["Evidence", esc(tp["evidence_id"])],
            ]) + "</section>"
        )
    # v2.1.1 remediation Section H: real tuning trials (executed)
    if model.tuning_run:
        tr = model.tuning_run
        if tr.get("ran"):
            param_keys = []
            for t in tr.get("trials", []):
                for k in t.get("params", {}).keys():
                    if k not in param_keys:
                        param_keys.append(k)
            trows = []
            for t in tr.get("trials", []):
                p = t["params"]
                row_cells = [esc(t["trial"])]
                for k in param_keys:
                    row_cells.append(esc(p.get(k, "-")))
                row_cells += [
                    f"{t['validation_metric']:.4f}",
                    f"<b>{esc(t['status'])}</b>" if t["status"] == "best" else esc(t["status"]),
                ]
                trows.append(row_cells)
            sections.append(
                "<section><h2>Tuning Trials (executed)</h2>"
                f"<p>Best metric: <b>{esc(round(tr.get('best_metric', 0.0), 4))}</b> "
                f"| best params: <code>{esc(tr.get('best_params'))}</code></p>"
                + table(["Trial"] + param_keys + ["Validation metric", "Status"], trows)
                + "</section>"
            )
        else:
            sections.append(
                f"<section><h2>Tuning Trials</h2><p>{esc(tr.get('note', 'Disabled.'))}</p></section>"
            )

    # v2.1.1 remediation: model execution HTML (split / metrics / explainability)
    if model.model_execution:
        me = model.model_execution
        split = me.get("split_table") or []
        if split:
            srows = [
                [esc(r["split"]), esc(r["rows"]), f"{r['percent']}%",
                 esc(r["positive_rate"]), esc(r["negative_rate"])]
                for r in split
            ]
            sections.append(
                "<section><h2>Train/Test/OOS Split</h2>"
                + table(["Split", "Rows", "Percent", "Positive rate", "Negative rate"], srows)
                + "</section>"
            )
        mbs = me.get("metrics_by_split") or {}
        if mbs:
            keys = ["auc_roc", "pr_auc", "accuracy", "precision", "recall", "f1",
                    "specificity", "brier_score", "ece"]
            mrows = []
            for split_name, m in mbs.items():
                row = [esc(split_name)] + [
                    f"{m.get(k, float('nan')):.4f}" if isinstance(m.get(k), (int, float)) else "—"
                    for k in keys
                ]
                mrows.append(row)
            gap = ""
            if me.get("generalization_gap") is not None:
                gap = f"<p>Generalization gap (train - OOS): {esc(me['generalization_gap'])}</p>"
            sections.append(
                "<section><h2>Metrics by Split</h2>"
                + table(["Split"] + keys, mrows) + gap + "</section>"
            )
        gi = me.get("global_importance") or []
        if gi:
            grows = [
                [esc(r["rank"]), esc(r["feature"]), esc(r["importance"]), esc(r["direction"])]
                for r in gi
            ]
            sections.append(
                f"<section><h2>Explainability — {esc(me.get('explainability_method'))}</h2>"
                + table(["Rank", "Feature", "Importance", "Direction"], grows)
                + "</section>"
            )

    if model.cohort_metrics:
        rows = [
            [esc(c), f"{m.get('auc_roc', float('nan')):.4f}",
             f"{m.get('accuracy', float('nan')):.4f}", f"{m.get('f1', float('nan')):.4f}"]
            for c, m in model.cohort_metrics.items()
        ]
        val = table(["Cohort", "AUC-ROC", "Accuracy", "F1"], rows)
    else:
        val = "<p>No cohort metrics (diagnostics-only review).</p>"
    sections.append(f"<section><h2>Validation Review</h2>{val}</section>")

    sections.append(
        f"<section><h2>Explainability Review</h2>{_kv_html(model.explainability, esc)}</section>"
    )
    sections.append(
        f"<section><h2>Robustness Review</h2>{_kv_html(model.robustness, esc)}</section>"
    )

    ai_rows = [
        [esc(r.get("adapter")), esc(r.get("category")),
         f"<span class='status-{esc(r.get('status'))}'>{esc(r.get('status'))}</span>",
         esc(r.get("runtime_s", 0)), esc(r.get("artifacts", 0)), esc(r.get("evidence", 0))]
        for r in model.ai_engineering_rows
    ]
    if model.control_surface:
        cs_rows = [
            [esc(r.get("adapter")), esc(r.get("purpose", "")), esc(r.get("role", "")),
             f"<span class='{('ok' if r.get('status') in ('complete', 'available') else 'warn')}'>"
             f"{esc(r.get('status'))}</span>",
             esc(", ".join(r.get("expected_outputs", [])) or "—"),
             esc(r.get("install_guidance", "") or "—")]
            for r in model.control_surface
        ]
        sections.append(
            "<section><h2>AI-Engineering Control Surface</h2>"
            + table(["Adapter", "Purpose", "Role", "Status", "Outputs", "Install guidance"], cs_rows)
            + "</section>"
        )
    else:
        sections.append(
            "<section><h2>AI-Engineering Review</h2>"
            + table(["Adapter", "Category", "Status", "Runtime (s)", "Artifacts", "Evidence"], ai_rows)
            + "</section>"
        )

    # v2.1.0 model execution: sensitivity + action log HTML
    if model.sensitivity:
        sa = model.sensitivity
        shock_cols = ["-30%", "-20%", "-10%", "+0%", "+10%", "+20%", "+30%"]
        srows = []
        for feat, drifts in sa.get("drift_table", {}).items():
            srows.append([esc(feat)] + [f"{drifts.get(c, 0.0):+.4f}" for c in shock_cols])
        sens_html = (
            f"<p>Metric: <b>{esc(sa.get('metric_name'))}</b> | baseline "
            f"{esc(sa.get('baseline'))} | most sensitive: "
            f"<b>{esc(sa.get('most_sensitive_feature'))}</b> | max |drift| "
            f"{esc(sa.get('max_abs_drift'))}</p>"
            + (table(["Feature"] + shock_cols, srows) if srows else "")
        )
        sections.append(f"<section><h2>Sensitivity Analysis</h2>{sens_html}</section>")
    if model.action_log:
        arows = [
            [esc(a["agent"]), esc(a["input_reviewed"]), esc(a["action"]),
             esc(a.get("recommendation") or "—"),
             esc(", ".join(a.get("evidence_ids", [])) or "—"),
             esc(a.get("user_decision") or "—")]
            for a in model.action_log
        ]
        sections.append(
            "<section><h2>Agentic Action Log</h2>"
            + table(["Agent", "Input reviewed", "Action", "Recommendation",
                     "Evidence", "User decision"], arows)
            + "</section>"
        )

    # v2.1.1 Section K: agent reasoning traces
    if model.agent_traces:
        trows = []
        for t in model.agent_traces:
            conf = f"{t['confidence']:.0%}" if t.get("confidence") is not None else "—"
            trows.append([
                esc(t["agent"]), esc(t["inputs"]), esc(t.get("reasoning") or "—"),
                esc(t["decision"]), conf, esc(t.get("alternative_considered") or "—"),
                esc(", ".join(t.get("evidence_ids", [])) or "—"),
            ])
        sections.append(
            "<section><h2>Agent Reasoning Traces</h2>"
            + table(["Agent", "Inputs", "Reasoning", "Decision", "Confidence",
                     "Alternative", "Evidence"], trows)
            + "</section>"
        )

    if model.findings:
        fhtml = table(
            ["Severity", "Materiality", "Category", "Title", "Evidence", "Recommendation"],
            [
                [f"<span class='sev-{esc(f['severity'])}'>{esc(f['severity'])}</span>",
                 esc(f["materiality"]), esc(f["risk_category"]), esc(f["title"]),
                 esc(", ".join(f["evidence_ids"]) or "—"), esc(f["recommendation"])]
                for f in model.findings
            ],
        )
    else:
        fhtml = "<p>No findings raised.</p>"
    sections.append(f"<section><h2>Governance Findings</h2>{fhtml}</section>")

    erows = [
        [esc(e.get("evidence_id", "—")), esc(e.get("test_name")), esc(e.get("status"))]
        for e in model.evidence_rows[:80]
    ]
    sections.append(
        f"<section><h2>Evidence Ledger Summary</h2>"
        f"<p>Total records: <b>{esc(model.evidence_summary.get('total', len(model.evidence_rows)))}</b></p>"
        + table(["Evidence ID", "Test", "Status"], erows)
        + "</section>"
    )

    sections.append(
        f"<section><h2>Final Signoff</h2>"
        f"<p>Evidence critique: <span class='{crit}'>"
        f"{'PASSED' if model.critique_ok else 'FAILED'}</span></p>"
        f"<p>{esc(model.signoff or 'pending')}</p></section>"
    )

    # v2.1.1 Section N/Q: artifact catalog
    if model.artifact_catalog:
        cat_rows = [
            [esc(a.get("name")), esc(a.get("type")), esc(a.get("category")), esc(a.get("path"))]
            for a in model.artifact_catalog
        ]
        sections.append(
            "<section><h2>Artifact Catalog</h2>"
            + table(["Artifact", "Type", "Category", "Location"], cat_rows)
            + "</section>"
        )

    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>StART Review — {esc(model.run_id)}</title>{_HTML_STYLE}</head><body>"
        f"<header><h1>StART Enterprise Review Dashboard</h1>"
        f"<p>{esc(model.run_id)}</p></header><main>{''.join(sections)}</main></body></html>"
    )


def _kv_html(d: dict[str, Any], esc) -> str:
    if not d:
        return "<p>n/a</p>"
    return "<table><tbody>" + "".join(
        f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in d.items()
    ) + "</tbody></table>"


def write_dashboard(model: DashboardModel, output_root: str | Path, run_id: str) -> dict[str, str]:
    out_dir = Path(output_root) / "dashboards" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "dashboard.json",
        "md": out_dir / "dashboard.md",
        "html": out_dir / "dashboard.html",
    }
    # Self-register the dashboard's own files so the Artifact Catalog is complete
    # (v2.1.1 Section N: no hidden outputs, including the dashboard itself).
    existing = {a.get("path") for a in model.artifact_catalog}
    type_by_ext = {".json": "dashboard (JSON)", ".md": "report (Markdown)",
                   ".html": "dashboard (HTML)"}
    for p in paths.values():
        if str(p) not in existing:
            model.artifact_catalog.append({
                "name": p.name, "path": str(p),
                "type": type_by_ext.get(p.suffix.lower(), "file"),
                "category": "report", "description": "",
            })
    paths["json"].write_text(render_dashboard_json(model))
    paths["md"].write_text(render_dashboard_md(model))
    paths["html"].write_text(render_dashboard_html(model))
    return {k: str(v) for k, v in paths.items()}
