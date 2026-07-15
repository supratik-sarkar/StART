# StART Enterprise Review Dashboard — `RUN-ENT-04283359`

## Executive Summary

- Task: **regression** | Target: `target_value` | Modality: tabular
- Recommended model family: `dcn`
- Findings: **2** (0 blocking) — {'Low': 2}
- AI-engineering controls available: 12/13
- Evidence records: 6 | Evidence critique: **PASSED**
- Sign-off: Reviewer recommendation: READY FOR SIGN-OFF. [EV-bd04dc945a3d]

## LLM Activation

| Field | Value |
| --- | --- |
| Provider | none |
| Model | — |
| Trust domain | none |
| Endpoint | — |
| Status | **DETERMINISTIC** |

No LLM selected; deterministic engines only.
## Dataset Review

600 rows x 6 columns; 1 candidate target(s), 0 text, 0 image-path, 0 timestamp, 0 entity column(s).

## Model Review

Recommended family: dcn.

## Review Journey

### Review decision ledger

| Checkpoint | Recommended | User choice | Status | Evidence | Execution impact |
| --- | --- | --- | --- | --- | --- |
| validation | accept | accept | accepted | — | validation review accepted into signoff |

### ValidationAgent Review

- Most sensitive feature: feature_lag2
- Max |drift|: 0.090003
- Signoff impact: low feature dependence; no signoff concern from sensitivity.

**Business interpretation**

- The model shows a moderate dependence worth monitoring on 'feature_lag2' (max metric drift 0.0900 under +/-30% shocks). If 'feature_lag2' shifts in production, expect a proportional change in model output; monitor it for drift and data-quality issues.
- The model shows a moderate dependence worth monitoring on 'feature_lag1' (max metric drift 0.0697 under +/-30% shocks). If 'feature_lag1' shifts in production, expect a proportional change in model output; monitor it for drift and data-quality issues.
- The model shows a moderate dependence worth monitoring on 'feature_noise' (max metric drift 0.0457 under +/-30% shocks). If 'feature_noise' shifts in production, expect a proportional change in model output; monitor it for drift and data-quality issues.

### MRM Signoff Decision

**Verdict: READY**

READY: 0 blocker(s), 0 concern(s), 3 factor(s) clear across performance, generalization, calibration, feature dependence, and reviewer activity. No blocking issues or concerns identified.

| Factor | Status | Detail | Evidence |
| --- | --- | --- | --- |
| Performance | unknown | No OOS metric available. |  |
| Feature dependence | ok | max drift 0.0900 (most sensitive: feature_lag2) | sensitivity_analysis |
| Reviewer challenges | ok | no outstanding reviewer challenges | review_session |
| Reviewer overrides | ok | no overrides; reviewer accepted recommendations | review_session |

## Dataset Source

| Field | Value |
| --- | --- |
| Name | preset_b.csv |
| Kind | custom |
| Rows / Columns | 600 / 6 |
| Target | target_value |
| File path | preset_b.csv |
| Detected format | csv |
| Loading route | load_any_tabular -> read_csv |
| Data hash | 176dd47501dee065 |

## Initial Data Statistics

| Metric | Value |
| --- | --- |
| Rows | 600 |
| Columns | 6 |
| Target type | continuous |
| Numeric / Categorical | 6 / 0 |
| Duplicate rows | 0 |
| Leakage candidates | 0 |
| Imbalance | n/a |
| Suggested split | random |

## Feature-Engineering Recommendations

| Step | Recommendation | Reason | Evidence | Risk if ignored | Default |
| --- | --- | --- | --- | --- | --- |
| imputation | Impute missing values (median for numeric, mode for categorical). | 2 column(s) contain missing values (max 4.7%). | FE-01 | Rows dropped or model errors; biased estimates if missingness is informative. | impute_median_mode |
| scaling | Standardize numeric features (fit on train only). | Multiple numeric features; gradient-based and distance-based models benefit from scaling. | FE-02 | Slow/unstable convergence; features on large scales dominate. | standardize_train_only |
| outliers | Winsorize/clip extreme numeric outliers (e.g. 1st/99th percentile). | Outliers detected in 4 column(s) (worst: feature_lag1, 15 points). | FE-03 | Outliers distort scaling and can dominate the loss. | winsorize_1_99 |

## Architecture Review

- User selected: `dcn + relu`
- Agent recommends: `dcn + relu`
- Reason: User choice is appropriate for the data; no change recommended.
- Evidence: ARCH-01
- Risk if ignored: None — choice validated.
- Agreement: yes

## Hyperparameter Tuning

- Strategy: bounded_randomized_search
- Primary metric: rmse
- Trials: 5 | Early stopping: True
- Validation: train_internal_holdout (no test/OOS leakage)
- Evidence: TUNE-01

### Tuning trials (executed)

- Best metric: 25.5182 | best params: {'learning_rate': 0.01, 'hidden_dims': [64, 32], 'dropout': 0.0}

| Trial | learning_rate | hidden_dims | dropout | Validation metric | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.001 | [128, 64] | 0.1 | 34.5615 | ok |
| 2 | 0.001 | [64, 32] | 0.2 | 35.9705 | ok |
| 3 | 0.01 | [64, 32] | 0.0 | 25.5182 | best |
| 4 | 0.001 | [32] | 0.2 | 36.1892 | ok |
| 5 | 0.003 | [32] | 0.0 | 35.7252 | ok |

## Train/Test/OOS Split

| Split | Rows | Percent | Positive rate | Negative rate |
| --- | --- | --- | --- | --- |
| train | 360 | 60.0% | 0.0028 | 0.9972 |
| test | 120 | 20.0% | 0.0083 | 0.9917 |
| oos | 120 | 20.0% | 0.0083 | 0.9917 |

## Metrics by Split

| Split | auc_roc | pr_auc | accuracy | precision | recall | f1 | specificity | brier_score | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | — | — | — | — | — | — | — | — | — |
| test | — | — | — | — | — | — | — | — | — |
| oos | — | — | — | — | — | — | — | — | — |

Generalization gap (train - OOS): 0.5264

## Explainability — permutation

| Rank | Feature | Importance | Direction |
| --- | --- | --- | --- |
| 1 | feature_lag2 | 0.007565 | positive |
| 2 | feature_noise | 0.003784 | positive |
| 3 | feature_lag1 | 0.002022 | positive |
| 4 | feature_trend | -0.00185 | negative |
| 5 | feature_season | -0.010978 | negative |

## Validation Review

| Cohort | AUC-ROC | Accuracy | F1 |
| --- | --- | --- | --- |
| train | nan | nan | nan |
| test | nan | nan | nan |
| oos | nan | nan | nan |

## Explainability Review

- note: Global feature importance shown below (permutation).

## Robustness Review

- note: Feature-shock sensitivity analysis shown below.

## AI-Engineering Control Surface

| Adapter | Purpose | Role | Status | Outputs | Install guidance |
| --- | --- | --- | --- | --- | --- |
| OPA | Policy-as-code governance. | Validate governance controls and detect policy violations. | complete | policy_report.json | Install OPA (https://www.openpolicyagent.org) or `pip install opa-python-client`. |
| MCP Server | External tool/server capability validation. | Discover MCP servers and validate their health. | complete | mcp_inventory.json, mcp_health_report.json | pip install mcp to enable MCP server discovery. |
| MCP SDK | MCP integration health. | Inspect MCP SDK capabilities and validate integration. | complete | — | pip install mcp to enable the MCP SDK. |
| MCP Inspector | MCP inspection and debugging. | Interactively inspect and debug MCP capabilities. | complete | — | npm install -g @modelcontextprotocol/inspector to enable the inspector. |
| Langfuse | LLM trace and prompt lineage. | Capture LLM traces and prompt/session lineage. | complete | langfuse_trace.json | pip install langfuse to capture traces (requires keys for cloud export). |
| OpenTelemetry | Telemetry spans and run observability. | Emit spans and metrics for run observability. | complete | telemetry.json | pip install opentelemetry-sdk opentelemetry-api to enable telemetry. |
| Garak | Red-team and jailbreak testing. | Probe LLMs for jailbreaks and adversarial failures. | complete | redteam_report.json | pip install garak to run LLM red-team probes. |
| Promptfoo | Prompt evaluation and attack suite. | Run prompt evals and red-team attack suites. | complete | redteam_report.json | npm install -g promptfoo to run prompt red-teaming/evals. |
| Moonshot | Compliance evaluation (optional; intentionally excluded by default). | Score compliance and run governance benchmarks. Excluded from the default StART environment: aiverify-moonshot hard-pins pydantic==2.8.2 and huggingface-hub~=0.36, which conflict with MCP, DeepEval, Garak, LiteLLM, and Transformers. Use LangSmith or Phoenix as safer alternatives. | not_installed | compliance_report.json | Optional and NOT recommended in the primary environment: aiverify-moonshot==0.7.6 conflicts with MCP/DeepEval/Garak/LiteLLM (pins pydantic==2.8.2, huggingface-hub~=0.36). Prefer LangSmith (pip install langsmith) or Phoenix (pip install arize-phoenix) for observability/evaluation instead. |
| NeMo Guardrails | Runtime guardrails and safety checks. | Enforce runtime guardrails and conversational safety. | complete | guardrail_report.json | pip install nemoguardrails to enable runtime guardrails. |
| DeepEval | LLM quality, hallucination and faithfulness checks. | Evaluate hallucination, faithfulness, relevancy, bias, toxicity. | complete | deepeval_report.json | pip install deepeval to run MRM/LLM evaluation metrics. |
| LangSmith | LLM trace capture and evaluation (Moonshot-safe alternative). | Capture LLM run traces and evaluation datasets for review lineage. | complete | langsmith_report.json | pip install langsmith to enable trace capture and evaluation. |
| Phoenix | LLM/ML observability and evaluation (Moonshot-safe alternative). | Provide tracing, evaluation, and drift/quality observability dashboards. | complete | phoenix_report.json | pip install arize-phoenix to enable observability and evaluation. |

## Sensitivity Analysis

- Metric: rmse
- Baseline (0% shock): 35.812444
- Most sensitive feature: feature_lag2
- Max |drift|: 0.090003

| Feature | Shock % | Baseline | Shocked | Delta | Risk impact |
| --- | --- | --- | --- | --- | --- |
| feature_lag1 | -30% | 35.8124 | 35.8514 | +0.0390 | moderate |
| feature_lag1 | -20% | 35.8124 | 35.8435 | +0.0311 | moderate |
| feature_lag1 | -10% | 35.8124 | 35.8306 | +0.0181 | low |
| feature_lag1 | +0% | 35.8124 | 35.8124 | +0.0000 | negligible |
| feature_lag1 | +10% | 35.8124 | 35.7900 | -0.0225 | low |
| feature_lag1 | +20% | 35.8124 | 35.7660 | -0.0465 | moderate |
| feature_lag1 | +30% | 35.8124 | 35.7427 | -0.0697 | moderate |
| feature_lag2 | -30% | 35.8124 | 35.8736 | +0.0612 | moderate |
| feature_lag2 | -20% | 35.8124 | 35.8593 | +0.0468 | moderate |
| feature_lag2 | -10% | 35.8124 | 35.8385 | +0.0261 | low |
| feature_lag2 | +0% | 35.8124 | 35.8124 | +0.0000 | negligible |
| feature_lag2 | +10% | 35.8124 | 35.7823 | -0.0301 | moderate |
| feature_lag2 | +20% | 35.8124 | 35.7514 | -0.0611 | moderate |
| feature_lag2 | +30% | 35.8124 | 35.7224 | -0.0900 | moderate |
| feature_trend | -30% | 35.8124 | 35.8004 | -0.0120 | low |
| feature_trend | -20% | 35.8124 | 35.8140 | +0.0016 | negligible |
| feature_trend | -10% | 35.8124 | 35.8180 | +0.0055 | low |
| feature_trend | +0% | 35.8124 | 35.8124 | +0.0000 | negligible |
| feature_trend | +10% | 35.8124 | 35.8028 | -0.0097 | low |
| feature_trend | +20% | 35.8124 | 35.7941 | -0.0183 | low |
| feature_trend | +30% | 35.8124 | 35.7871 | -0.0254 | low |
| feature_season | -30% | 35.8124 | 35.8175 | +0.0051 | low |
| feature_season | -20% | 35.8124 | 35.8167 | +0.0043 | negligible |
| feature_season | -10% | 35.8124 | 35.8150 | +0.0026 | negligible |
| feature_season | +0% | 35.8124 | 35.8124 | +0.0000 | negligible |
| feature_season | +10% | 35.8124 | 35.8097 | -0.0028 | negligible |
| feature_season | +20% | 35.8124 | 35.8076 | -0.0048 | negligible |
| feature_season | +30% | 35.8124 | 35.8056 | -0.0068 | low |
| feature_noise | -30% | 35.8124 | 35.8573 | +0.0448 | moderate |
| feature_noise | -20% | 35.8124 | 35.8436 | +0.0312 | moderate |
| feature_noise | -10% | 35.8124 | 35.8283 | +0.0159 | low |
| feature_noise | +0% | 35.8124 | 35.8124 | +0.0000 | negligible |
| feature_noise | +10% | 35.8124 | 35.7966 | -0.0159 | low |
| feature_noise | +20% | 35.8124 | 35.7813 | -0.0311 | moderate |
| feature_noise | +30% | 35.8124 | 35.7668 | -0.0457 | moderate |

## Agentic Action Log

| Agent | Input reviewed | Action | Recommendation | Evidence | User decision |
| --- | --- | --- | --- | --- | --- |
| DatasetDiscoveryAgent | 600 rows x 6 cols | computed data statistics | suggested split: random | EV-bd04dc945a3d | — |
| FeatureEngineeringAgent | data statistics | recommended 3 preprocessing step(s) | imputation; scaling; outliers | FE-01, FE-02, FE-03 | — |
| ArchitectureReviewAgent | dcn+relu | reviewed architecture choice | dcn+relu | ARCH-01 | agrees |
| HyperparameterTuningAgent | search space | planned 5-trial bounded_randomized_search | metric: rmse | TUNE-01 | — |
| ValidationPlannerAgent | top features, metric=rmse | ran feature-shock sensitivity analysis | most sensitive: feature_lag2 | EV-e4ae250a0193 | — |
| GovernanceSignoffAgent | 2 findings | produced sign-off disposition | Reviewer recommendation: READY FOR SIGN-OFF | EV-bd04dc945a3d, EV-eefb201d039e | — |
| EvidenceCriticAgent | all findings + narrative | ran citation gate | PASSED | EV-bd04dc945a3d | — |

## Agent Reasoning Traces

| Agent | Inputs | Reasoning | Decision | Confidence | Alternative | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| DatasetDiscoveryAgent | 600 rows x 6 cols | 6 numeric / 0 categorical; imbalance n/a | profiled dataset; suggested split = random | 90% | random split | EV-bd04dc945a3d |
| TaskInferenceAgent | target 'target_value', type continuous | target has type continuous | regression | 95% | classification | — |
| FeatureEngineeringAgent | initial data statistics | imputation; scaling; outliers | recommended 3 preprocessing step(s) | 85% | use raw features | FE-01, FE-02, FE-03 |
| ArchitectureReviewAgent | user choice dcn+relu; 5 features, 600 rows, tabular | User choice is appropriate for the data; no change recommended. | recommend dcn+relu | 90% | dcn+relu | ARCH-01 |
| HyperparameterTuningAgent | task regression, 600 rows, cost=balanced | metric routed by cost preference 'balanced'; train_internal_holdout (no test/OOS leakage) | 5-trial bounded_randomized_search, metric rmse | 80% | exhaustive grid search | TUNE-01 |
| ModelExecutionAgent | train/test/oos split (0.6, 0.2, 0.2) | generalization gap 0.526351 | trained mlp; explainability via permutation | 85% | diagnostics-only (no training) | EV-e4ae250a0193 |
| HyperparameterTuningAgent | 5-trial bounded_random_search | train-internal holdout only (no test/OOS leakage) | best metric 25.5182 @ {'learning_rate': 0.01, 'hidden_dims': [64, 32], 'dropout': 0.0} | 85% | grid search | TUNE-01 |
| GovernanceSignoffAgent | 2 findings, 6 evidence records | 2 findings weighed against acceptance criteria | Reviewer recommendation: READY FOR SIGN-OFF | 85% | conditional sign-off | EV-bd04dc945a3d, EV-eefb201d039e |
| EvidenceCriticAgent | all findings, recommendations, sign-off narrative | every claim must cite evidence; uncited claims are flagged | PASSED | 100% | allow uncited narrative | EV-bd04dc945a3d |

## Governance Findings

| Severity | Materiality | Category | Title | Evidence | Recommendation |
| --- | --- | --- | --- | --- | --- |
| Low | Medium | Governance | Governance sign-off disposition | EV-bd04dc945a3d, EV-eefb201d039e, EV-2e0edad34d91 | Review the governance findings before approval. |
| Low | Low | Operational | Moonshot unavailable | ai_engineering.compliance.moonshot | Optional and NOT recommended in the primary environment: aiverify-moonshot==0.7.6 conflicts with MCP/DeepEval/Garak/LiteLLM (pins pydantic==2.8.2, huggingface-hub~=0.36). Prefer LangSmith (pip install langsmith) or Phoenix (pip install arize-phoenix) for observability/evaluation instead. |

## Evidence Ledger Summary

Total records: 6

| Evidence ID | Test | Status |
| --- | --- | --- |
| EV-bd04dc945a3d | Dataset discovery profile | pass |
| EV-eefb201d039e | Target selection | pass |
| EV-2e0edad34d91 | Task inference | pass |
| EV-8871694b4cda | Data split plan | pass |
| EV-0f3a3b981099 | Feature engineering diagnostics (tabular) | pass |
| EV-e4ae250a0193 | Cohort performance metrics | pass |

## Final Signoff

**Evidence critique:** PASSED

Reviewer recommendation: READY FOR SIGN-OFF. [EV-bd04dc945a3d]


## Artifact Catalog

| Artifact | Type | Category | Location |
| --- | --- | --- | --- |
| split_distribution.csv | table (CSV) | split | test_dl_output/copilot/RUN-ENT-04283359/split_distribution.csv |
| split_distribution.json | data (JSON) | split | test_dl_output/copilot/RUN-ENT-04283359/split_distribution.json |
| metrics_by_split.csv | table (CSV) | split | test_dl_output/copilot/RUN-ENT-04283359/metrics_by_split.csv |
| training_summary.json | data (JSON) | training | test_dl_output/copilot/RUN-ENT-04283359/training_summary.json |
| training_history.csv | table (CSV) | training | test_dl_output/copilot/RUN-ENT-04283359/training_history.csv |
| global_feature_importance.csv | table (CSV) | explainability | test_dl_output/copilot/RUN-ENT-04283359/global_feature_importance.csv |
| tuning_trials.csv | table (CSV) | tuning | test_dl_output/tuning/RUN-ENT-04283359/tuning_trials.csv |
| tuning_summary.json | data (JSON) | tuning | test_dl_output/tuning/RUN-ENT-04283359/tuning_summary.json |
| review_graph.json | data (JSON) | graph | test_dl_output/ai_engineering/RUN-ENT-04283359/review_graph.json |
| review_graph.png | figure (PNG) | graph | test_dl_output/ai_engineering/RUN-ENT-04283359/review_graph.png |
| policy_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-04283359/policy_report.json |
| mcp_inventory.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-04283359/mcp_inventory.json |
| mcp_health_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-04283359/mcp_health_report.json |
| langfuse_trace.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-04283359/langfuse_trace.json |
| telemetry.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-04283359/telemetry.json |
| redteam_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-04283359/redteam_report.json |
| promptfoo_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-04283359/promptfoo_report.json |
| guardrail_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-04283359/guardrail_report.json |
| deepeval_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-04283359/deepeval_report.json |
| langsmith_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-04283359/langsmith_report.json |
| phoenix_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-04283359/phoenix_report.json |
| dashboard.json | dashboard (JSON) | report | test_dl_output/dashboards/RUN-ENT-04283359/dashboard.json |
| dashboard.md | report (Markdown) | report | test_dl_output/dashboards/RUN-ENT-04283359/dashboard.md |
| dashboard.html | dashboard (HTML) | report | test_dl_output/dashboards/RUN-ENT-04283359/dashboard.html |
