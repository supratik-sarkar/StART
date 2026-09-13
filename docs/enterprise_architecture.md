# StART v2.0.0 — Enterprise Architecture

This document describes the v2.0.0 enterprise layer added on top of the frozen
v0.5.0 foundation. It is an **upgrade, not a rewrite**: Layers 1–7 (loaders,
discovery agents, split/feature planning, architecture registry, tabular /
sequence / vision deep learning) are unchanged. No schema churn, no renaming,
no breaking changes.

## Layered execution

`EnterpriseReviewOrchestrator` composes the existing visible pipeline into the
seven explicit layers the enterprise spec requires:

```
Data → Model → Validation → Governance → AI-Engineering → Evidence → Reporting
```

Each layer emits `status`, `runtime`, `warnings`, `findings`, `artifacts`, and
`evidence_ids` (`LayerResult`). The base `ReviewOrchestrator` (and its 22 visible
stages) runs underneath and is unchanged; the enterprise orchestrator wraps it,
attributes evidence to layers, derives findings, runs the AI-engineering layer,
and generates the dashboard.

## Governance findings engine (`start.governance`)

`Finding(title, description, severity, materiality, risk_category,
evidence_ids, recommendation)` with `Severity` ∈ {Low, Medium, High, Critical}
and `Materiality` ∈ {Low, Medium, High}. `FindingsRegister` prioritizes,
filters blocking (High/Critical) findings, and flags uncited findings.
`derive_findings_from_evidence` maps warn/fail evidence records into cited
findings. The `EvidenceCritic` gate ensures no uncited governance findings,
recommendations, or sign-off decisions.

## AI-engineering adapters (`start.ai_engineering`)

Every adapter subclasses `BaseAdapter` and implements:

```
available()         -> bool
validate()          -> ValidationResult
execute(context)    -> ExecutionResult     (artifacts, findings, evidence, status, summary)
collect_artifacts() -> list[Artifact]
emit_evidence()     -> list[TestResult]
```

The eleven adapters: `OPAAdapter` (policy_report.json), `MCPServerAdapter` /
`MCPSDKAdapter` / `MCPInspectorAdapter` (mcp_inventory.json,
mcp_health_report.json), `LangfuseAdapter` (langfuse_trace.json),
`OpenTelemetryAdapter` (telemetry.json — genuinely emits a span),
`GarakAdapter` / `PromptfooAdapter` (redteam_report.json), `MoonshotAdapter`
(compliance_report.json), `NeMoGuardrailsAdapter` (guardrail_report.json),
`DeepEvalAdapter` (deepeval_report.json).

When a backend is installed the adapter runs it for real; when absent it returns
an explicit `not_installed` result with install guidance, still emits an evidence
record, and stays visible in the report. There is no fabricated success and no
silent degradation. Install backends via extras, e.g.
`pip install -e ".[telemetry,observability,mcp,evals,orchestration]"`.

## Graph execution (`start.modeling.graph_orchestrator`)

`GraphReviewOrchestrator` runs the pipeline as a DAG with checkpointing,
resume-from-failure, cycle detection, and state tracking, emitting
`review_graph.json` and `review_graph.png`. It uses LangGraph when installed and
a built-in deterministic DAG executor otherwise — the standard orchestrator stays
the default; this is the opt-in enterprise execution mode.

## Dashboard (`start.reporting.dashboard`)

`write_dashboard` renders `dashboard.json`, `dashboard.md`, and a self-contained
`dashboard.html` (no external assets) from a single `DashboardModel`. Sections:
Executive Summary, Dataset Review, Model Review, Validation Review,
Explainability Review, Robustness Review, AI-Engineering Review, Governance
Findings, Evidence Ledger Summary, Final Sign-off.

## Provider trust domains (`start.providers.trust_domains`)

Two isolated trust domains: PUBLIC (openai, anthropic, grok) and PRIVATE
(enterprise_llm_gateway, isolated in `src/start/enterprise/`). Enforced
invariants: no routing crossover, no key sharing, no fallback between domains.
`get_llm_provider(config, expected_domain=...)` raises `TrustDomainViolation` on
crossover; an unavailable provider degrades only to the domain-neutral
deterministic path.

## CNN transparency (`start.modeling.vision_models.describe_cnn`)

Returns the resolved architecture as evidence metadata: preset, conv blocks,
per-block channel progression, kernel size, pooling, dropout, dense size, image
size, and real trainable parameter count. Custom configs are user-editable and
flow into the dashboard and evidence.

## Surfaces

- CLI: `start review --enterprise`
- Notebook (Databricks): `notebooks/05_enterprise_review.py`
- Notebook (Jupyter/VS Code, widget-driven): `notebooks/05_enterprise_review.ipynb`
- Examples: `examples/enterprise_dashboard_demo.py`, `examples/governance_findings_demo.py`

---

## v2.1.0 — Live Model-Engineering Co-Pilot

v2.1.0 adds a reviewer-experience layer on top of the v2.0.0 enterprise review.
No v2.0.0 layer is modified; these modules are composed in by the
`EnterpriseReviewOrchestrator` and rendered in the dashboard.

### Modules

- `modeling/dataset_source.py` — `DatasetSource` provenance. For the demo
  dataset: name, public URL (UCI Breast Cancer Wisconsin Diagnostic), reason
  selected, task suitability, loading route, content hash. For custom data:
  path, detected format, loading route, shape, hash.
- `modeling/data_statistics.py` — `compute_data_statistics` -> `DataStatistics`
  (rows/cols, target type, class distribution, role counts, duplicates,
  high-cardinality, low-variance, IQR outliers, correlation summary, leakage
  candidates, imbalance warning, suggested split).
- `modeling/fe_recommendations.py` — `recommend_feature_engineering` ->
  `FERecommendationSet`. Each `FERecommendation` carries recommendation,
  reason, evidence ID, risk-if-ignored, default action, and a user-override
  slot. Covers imputation, encoding, scaling, outliers, imbalance,
  low-variance removal, correlation pruning, leakage exclusion, high-cardinality
  handling, and modality routing.
- `agents/engineering_agents.py` — `select_primary_metric` (metric-priority
  routing), `ArchitectureReviewAgent` (user-choice vs recommendation), and
  `HyperparameterTuningAgent` (bounded, leakage-safe plan).
- `modeling/sensitivity_analysis.py` — `run_sensitivity_analysis`. Shocks the
  top features across DEFAULT_SHOCKS (-30%..+30%) and measures metric drift;
  the 0% row equals the unshocked baseline by construction.
- `reporting/progress.py` — `ProgressReporter` (cross-platform ASCII bars) and
  `ActionLog` / `AgentAction` (per-agent input/action/recommendation/evidence/
  user-decision/output).

### Metric-priority routing

`select_primary_metric(task_type, costlier_errors=...)`:

| Task | balanced | false_negatives | false_positives |
| --- | --- | --- | --- |
| binary | auc_roc | pr_auc (+recall) | precision |
| multiclass | f1_macro | — | — |
| regression | rmse | — | — |

The chosen metric flows through the tuning plan and the sensitivity analysis.

### Dashboard sections added

Dataset Source, Initial Data Statistics, Feature-Engineering Recommendations,
Architecture Review, Hyperparameter Tuning, Sensitivity Analysis, and Agentic
Action Log — in `dashboard.md`, `dashboard.html`, and `dashboard.json`.

### CLI

`start review` gains `--cost {balanced|false_negatives|false_positives}`,
`--accept-recommendations`, and `--show-progress/--no-progress`.

---

## v2.1.1 — Visible Reviewer Co-Pilot

v2.1.1 adds a visibility layer so nothing important happens silently. All v2.1.0
engines are unchanged; these modules surface them.

### Modules

- `reporting/agent_trace.py` — `AgentTrace` / `TraceLog`. Each agent emits
  inputs, evidence, reasoning summary, decision, confidence, alternative
  considered, action taken (Section K). Rendered to terminal, markdown, and
  dashboard.
- `reporting/artifacts.py` — `ArtifactRegistry`. Every generated artifact is
  registered (name/type/category/location) and announced immediately
  (Section N). Becomes the dashboard Artifact Catalog.
- `providers/llm_activation.py` — `preflight_llm` -> `ActivationReport`
  (provider/model/trust-domain/endpoint/status). Status is CONNECTED, FAILED,
  FALLBACK, NOT_CONFIGURED, or DETERMINISTIC — never a silent degradation
  (Section A).
- `interactive_checkpoints.py` — `resolve_checkpoint` -> `CheckpointDecision`.
  The Accept / Keep / Explain pattern (Section B). Auto-accept and
  non-interactive paths are explicit and recorded, never silent overrides.

### Adapter control surface (Sections L/M)

`BaseAdapter` gains `purpose` and `role` attributes and a `describe()` method;
`AIEngineeringReport.control_surface()` returns per-adapter rows with purpose,
role, status, what it would do if installed, expected outputs, and install
guidance. Unavailable adapters remain visible and never report "pass."

### Dashboard sections added

LLM Activation, Agent Reasoning Traces, AI-Engineering Control Surface, and
Artifact Catalog — in `dashboard.md`, `dashboard.html`, and `dashboard.json`.
The dashboard's own files self-register into the Artifact Catalog.

### CLI / notebook

`start review` honors `--accept-recommendations` (auto-accept checkpoints) and
`--show-progress`. The terminal run prints the activation report, agent
reasoning traces, and the artifact list. Notebook 05 renders the same via
`activation_report`, `trace_log`, `control_surface()`, and `artifact_registry`.

---

## v2.1.1 remediation — Make the model review actually run

The remediation fixed the interactive enterprise run so it behaves like a live
review execution layer rather than silently running diagnostics-only.

### Provider activation (Sections A/B)

The root cause of "Provider: none" was that an unavailable public provider
degraded to ``NoLLMProvider`` (name "none"), losing the user's intent. Fixes:
the existing secure ``ensure_provider_key`` (hidden ``getpass``, session-only,
never persisted) is now called before resolution, and the *requested* provider
name is threaded through ``EnterpriseReviewOrchestrator.run(requested_provider=...)``
so the activation report reads e.g. ``Provider: openai -> FALLBACK`` instead of
``none``. ``llm-check`` and trust-domain isolation are unchanged.

### Interactive prompts (Sections C/D/H/M)

``prompt_review_config`` now asks: run training? (default yes), train/test/OOS
proportions (validated to sum to 1.0), tuning strategy + trials, and infers cost
priority from a free-text clarification (``_infer_cost_priority``).

### Real execution modules

- ``modeling/model_execution.py`` — trains on the user-proportioned split and
  emits the split table, metrics-by-split (AUC/PR-AUC/precision/recall/
  specificity/F1/Brier/ECE + confusion matrix), training diagnostics,
  generalization gap, and a global explainability table — as registered CSV/JSON
  artifacts (Sections D/G/I/J/K).
- ``modeling/tuning_run.py`` — runs an actual bounded randomized (or grid)
  search; each trial trains on a train-internal holdout (no test/OOS leakage)
  and is scored, producing a per-trial table and the best/rejected parameters
  (Section H). A disabled strategy records an explicit note.

### Trace backend context (Section N)

``AgentTrace`` carries ``backend`` / ``llm_used`` / ``fallback_reason``; the
orchestrator stamps every trace from the activation status (``llm_used`` is true
only when genuinely ``CONNECTED``).

### Surfacing

All of the above render in the terminal, in ``dashboard.md`` / ``dashboard.html``
/ ``dashboard.json`` (new sections: Train/Test/OOS Split, Metrics by Split,
Explainability, Tuning Trials), and in notebook 05.
