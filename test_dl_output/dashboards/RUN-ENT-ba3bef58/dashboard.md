# StART Enterprise Review Dashboard — `RUN-ENT-ba3bef58`

## Executive Summary

- Task: **binary_classification** | Target: `attrition` | Modality: tabular
- Recommended model family: `mlp`
- Findings: **2** (0 blocking) — {'Low': 2}
- AI-engineering controls available: 12/13
- Evidence records: 6 | Evidence critique: **PASSED**
- Sign-off: Reviewer recommendation: READY FOR SIGN-OFF. [EV-8b88ddddccbd]

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

569 rows x 31 columns; 1 candidate target(s), 0 text, 0 image-path, 0 timestamp, 0 entity column(s).

## Model Review

Recommended family: mlp.

## Review Journey

### Review decision ledger

| Checkpoint | Recommended | User choice | Status | Evidence | Execution impact |
| --- | --- | --- | --- | --- | --- |
| validation | accept | accept | accepted | — | validation review accepted into signoff |

### ValidationAgent Review

- Most sensitive feature: worst_perimeter
- Max |drift|: 0.001506
- Signoff impact: low feature dependence; no signoff concern from sensitivity.

**Business interpretation**

- The model shows no material dependence on 'worst_perimeter' (max metric drift 0.0015 under +/-30% shocks). If 'worst_perimeter' shifts in production, expect a proportional change in model output; monitor it for drift and data-quality issues.
- The model shows no material dependence on 'worst_area' (max metric drift 0.0006 under +/-30% shocks). If 'worst_area' shifts in production, expect a proportional change in model output; monitor it for drift and data-quality issues.
- The model shows no material dependence on 'mean_perimeter' (max metric drift 0.0006 under +/-30% shocks). If 'mean_perimeter' shifts in production, expect a proportional change in model output; monitor it for drift and data-quality issues.

### MRM Signoff Decision

**Verdict: READY WITH CONDITIONS**

READY WITH CONDITIONS: 0 blocker(s), 1 concern(s), 5 factor(s) clear across performance, generalization, calibration, feature dependence, and reviewer activity. Sign-off is conditional on addressing the listed concerns.

| Factor | Status | Detail | Evidence |
| --- | --- | --- | --- |
| Performance | ok | OOS auc_roc=0.9671 | cohort_metrics.oos |
| Generalization | ok | train-OOS gap -0.0015 | cohort_metrics |
| Calibration | concern | OOS ECE=0.2570 exceeds the configured threshold 0.100 (adjustable per model/risk appetite) | cohort_metrics.oos.ece |
| Feature dependence | ok | max drift 0.0015 (most sensitive: worst_perimeter) | sensitivity_analysis |
| Reviewer challenges | ok | no outstanding reviewer challenges | review_session |
| Reviewer overrides | ok | no overrides; reviewer accepted recommendations | review_session |

## Dataset Source

| Field | Value |
| --- | --- |
| Name | scikit-learn Breast Cancer Wisconsin (Diagnostic) |
| Kind | builtin_demo |
| Rows / Columns | 569 / 31 |
| Target | attrition |
| Public source | https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic |
| Loading route | sklearn.datasets.load_breast_cancer |
| Data hash | a94ac4b7071ffa5d |

**Why selected:** A small, clean, fully public binary-classification dataset bundled with scikit-learn — ideal for a reproducible, verifiable model-review demo with no download or credentials required.

**Task suitability:** Binary classification (30 numeric features, 2 balanced classes); suitable for tabular DL, calibration, explainability, and sensitivity demonstrations.

## Initial Data Statistics

| Metric | Value |
| --- | --- |
| Rows | 569 |
| Columns | 31 |
| Target type | binary |
| Numeric / Categorical | 31 / 0 |
| Duplicate rows | 0 |
| Leakage candidates | 0 |
| Imbalance | balanced |
| Suggested split | stratified |

**Class distribution:** 0=62.7%, 1=37.3%

## Feature-Engineering Recommendations

| Step | Recommendation | Reason | Evidence | Risk if ignored | Default |
| --- | --- | --- | --- | --- | --- |
| scaling | Standardize numeric features (fit on train only). | Multiple numeric features; gradient-based and distance-based models benefit from scaling. | FE-01 | Slow/unstable convergence; features on large scales dominate. | standardize_train_only |
| outliers | Winsorize/clip extreme numeric outliers (e.g. 1st/99th percentile). | Outliers detected in 29 column(s) (worst: area_error, 65 points). | FE-02 | Outliers distort scaling and can dominate the loss. | winsorize_1_99 |
| correlation_pruning | Review highly correlated feature pairs; consider pruning redundancy. | 8 feature(s) correlate > 0.7 with the target or each other. | FE-03 | Multicollinearity; unstable coefficients and attributions. | review_only |

## Architecture Review

- User selected: `mlp + relu`
- Agent recommends: `mlp + relu`
- Reason: User choice is appropriate for the data; no change recommended.
- Evidence: ARCH-01
- Risk if ignored: None — choice validated.
- Agreement: yes

## Hyperparameter Tuning

- Strategy: bounded_randomized_search
- Primary metric: auc_roc
- Trials: 5 | Early stopping: True
- Validation: train_internal_holdout (no test/OOS leakage)
- Evidence: TUNE-01

### Tuning trials (executed)

- Best metric: 1.0000 | best params: {'learning_rate': 0.003, 'hidden_dims': [128, 64], 'dropout': 0.2}

| Trial | learning_rate | hidden_dims | dropout | Validation metric | Status |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.003 | [32] | 0.2 | 0.9884 | ok |
| 2 | 0.01 | [64, 32] | 0.0 | 0.9987 | ok |
| 3 | 0.003 | [128, 64] | 0.0 | 1.0000 | best |
| 4 | 0.001 | [64, 32] | 0.2 | 0.9726 | ok |
| 5 | 0.003 | [128, 64] | 0.2 | 1.0000 | best |

## K-fold Tuning (train-only, stratified)

- Method: stratified_kfold (5-fold)
- Primary metric: auc_roc
- Train rows used: 341 (test/OOS excluded from selection: 228)
- Best params: {'C': 3.0, 'class_weight': None}
- Best mean: 0.992486 (std 0.006888)

| Fold | Metric | n_train | n_val |
| --- | --- | --- | --- |
| 1 | 0.9937 | 272 | 69 |
| 2 | 0.9799 | 273 | 68 |
| 3 | 0.9916 | 273 | 68 |
| 4 | 0.9991 | 273 | 68 |
| 5 | 0.9981 | 273 | 68 |

## Train/Test/OOS Split

| Split | Rows | Percent | Positive rate | Negative rate |
| --- | --- | --- | --- | --- |
| train | 341 | 59.93% | 0.3724 | 0.6276 |
| test | 113 | 19.86% | 0.3717 | 0.6283 |
| oos | 115 | 20.21% | 0.3739 | 0.6261 |

## Metrics by Split

| Split | auc_roc | pr_auc | accuracy | precision | recall | f1 | specificity | brier_score | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0.9655 | 0.9601 | 0.8886 | 0.9785 | 0.7165 | 0.8273 | 0.9907 | 0.1423 | 0.2674 |
| test | 0.9480 | 0.9386 | 0.8673 | 1.0000 | 0.6429 | 0.7826 | 1.0000 | 0.1484 | 0.2360 |
| oos | 0.9671 | 0.9564 | 0.8696 | 1.0000 | 0.6512 | 0.7887 | 1.0000 | 0.1531 | 0.2570 |

Generalization gap (train - OOS): -0.0015

## Explainability — permutation

| Rank | Feature | Importance | Direction |
| --- | --- | --- | --- |
| 1 | radius_error | 0.031858 | positive |
| 2 | worst_area | 0.023009 | positive |
| 3 | mean_concavity | 0.023009 | positive |
| 4 | mean_concave_points | 0.023009 | positive |
| 5 | mean_radius | 0.021239 | positive |
| 6 | worst_compactness | 0.021239 | positive |
| 7 | fractal_dimension_error | 0.019469 | positive |
| 8 | worst_concavity | 0.012389 | positive |
| 9 | mean_compactness | 0.012389 | positive |
| 10 | worst_texture | 0.012389 | positive |
| 11 | mean_texture | 0.010619 | positive |
| 12 | compactness_error | 0.010619 | positive |
| 13 | worst_smoothness | 0.010619 | positive |
| 14 | worst_concave_points | 0.010619 | positive |
| 15 | concavity_error | 0.00885 | positive |
| 16 | mean_fractal_dimension | 0.00354 | positive |
| 17 | worst_fractal_dimension | 0.00354 | positive |
| 18 | smoothness_error | 0.00177 | positive |
| 19 | texture_error | 0.0 | positive |
| 20 | concave_points_error | 0.0 | positive |

## Validation Review

| Cohort | AUC-ROC | Accuracy | F1 |
| --- | --- | --- | --- |
| train | 0.9889 | 0.9326 | 0.9004 |
| test | 0.9974 | 0.9298 | 0.8974 |
| oos | 0.9970 | 0.9211 | 0.8800 |

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

- Metric: auc_roc
- Baseline (0% shock): 0.988003
- Most sensitive feature: worst_perimeter
- Max |drift|: 0.001506

| Feature | Shock % | Baseline | Shocked | Delta | Risk impact |
| --- | --- | --- | --- | --- | --- |
| worst_area | -30% | 0.9880 | 0.9874 | -0.0006 | negligible |
| worst_area | -20% | 0.9880 | 0.9876 | -0.0004 | negligible |
| worst_area | -10% | 0.9880 | 0.9878 | -0.0002 | negligible |
| worst_area | +0% | 0.9880 | 0.9880 | +0.0000 | negligible |
| worst_area | +10% | 0.9880 | 0.9882 | +0.0002 | negligible |
| worst_area | +20% | 0.9880 | 0.9884 | +0.0004 | negligible |
| worst_area | +30% | 0.9880 | 0.9885 | +0.0004 | negligible |
| mean_area | -30% | 0.9880 | 0.9876 | -0.0004 | negligible |
| mean_area | -20% | 0.9880 | 0.9878 | -0.0002 | negligible |
| mean_area | -10% | 0.9880 | 0.9880 | -0.0000 | negligible |
| mean_area | +0% | 0.9880 | 0.9880 | +0.0000 | negligible |
| mean_area | +10% | 0.9880 | 0.9881 | +0.0001 | negligible |
| mean_area | +20% | 0.9880 | 0.9882 | +0.0002 | negligible |
| mean_area | +30% | 0.9880 | 0.9883 | +0.0003 | negligible |
| area_error | -30% | 0.9880 | 0.9879 | -0.0001 | negligible |
| area_error | -20% | 0.9880 | 0.9879 | -0.0001 | negligible |
| area_error | -10% | 0.9880 | 0.9879 | -0.0001 | negligible |
| area_error | +0% | 0.9880 | 0.9880 | +0.0000 | negligible |
| area_error | +10% | 0.9880 | 0.9880 | +0.0000 | negligible |
| area_error | +20% | 0.9880 | 0.9881 | +0.0001 | negligible |
| area_error | +30% | 0.9880 | 0.9882 | +0.0002 | negligible |
| worst_perimeter | -30% | 0.9880 | 0.9865 | -0.0015 | negligible |
| worst_perimeter | -20% | 0.9880 | 0.9871 | -0.0009 | negligible |
| worst_perimeter | -10% | 0.9880 | 0.9875 | -0.0005 | negligible |
| worst_perimeter | +0% | 0.9880 | 0.9880 | +0.0000 | negligible |
| worst_perimeter | +10% | 0.9880 | 0.9884 | +0.0004 | negligible |
| worst_perimeter | +20% | 0.9880 | 0.9887 | +0.0007 | negligible |
| worst_perimeter | +30% | 0.9880 | 0.9888 | +0.0008 | negligible |
| mean_perimeter | -30% | 0.9880 | 0.9874 | -0.0006 | negligible |
| mean_perimeter | -20% | 0.9880 | 0.9877 | -0.0003 | negligible |
| mean_perimeter | -10% | 0.9880 | 0.9879 | -0.0001 | negligible |
| mean_perimeter | +0% | 0.9880 | 0.9880 | +0.0000 | negligible |
| mean_perimeter | +10% | 0.9880 | 0.9880 | +0.0000 | negligible |
| mean_perimeter | +20% | 0.9880 | 0.9880 | -0.0000 | negligible |
| mean_perimeter | +30% | 0.9880 | 0.9881 | +0.0001 | negligible |

## Agentic Action Log

| Agent | Input reviewed | Action | Recommendation | Evidence | User decision |
| --- | --- | --- | --- | --- | --- |
| DatasetDiscoveryAgent | 569 rows x 31 cols | computed data statistics | suggested split: stratified | EV-8b88ddddccbd | — |
| FeatureEngineeringAgent | data statistics | recommended 3 preprocessing step(s) | scaling; outliers; correlation_pruning | FE-01, FE-02, FE-03 | — |
| ArchitectureReviewAgent | mlp+relu | reviewed architecture choice | mlp+relu | ARCH-01 | agrees |
| HyperparameterTuningAgent | search space | planned 5-trial bounded_randomized_search | metric: auc_roc | TUNE-01 | — |
| ValidationPlannerAgent | top features, metric=auc_roc | ran feature-shock sensitivity analysis | most sensitive: worst_perimeter | EV-312d174e6057 | — |
| GovernanceSignoffAgent | 2 findings | produced sign-off disposition | Reviewer recommendation: READY FOR SIGN-OFF | EV-8b88ddddccbd, EV-7e423f877e0c | — |
| EvidenceCriticAgent | all findings + narrative | ran citation gate | PASSED | EV-8b88ddddccbd | — |

## Agent Reasoning Traces

| Agent | Inputs | Reasoning | Decision | Confidence | Alternative | Evidence |
| --- | --- | --- | --- | --- | --- | --- |
| DatasetDiscoveryAgent | 569 rows x 31 cols | 31 numeric / 0 categorical; imbalance balanced | profiled dataset; suggested split = stratified | 90% | random split | EV-8b88ddddccbd |
| TaskInferenceAgent | target 'attrition', type binary | target has type binary | binary_classification | 95% | regression | — |
| FeatureEngineeringAgent | initial data statistics | scaling; outliers; correlation_pruning | recommended 3 preprocessing step(s) | 85% | use raw features | FE-01, FE-02, FE-03 |
| ArchitectureReviewAgent | user choice mlp+relu; 30 features, 569 rows, tabular | User choice is appropriate for the data; no change recommended. | recommend mlp+relu | 90% | mlp+relu | ARCH-01 |
| HyperparameterTuningAgent | task binary_classification, 569 rows, cost=balanced | metric routed by cost preference 'balanced'; train_internal_holdout (no test/OOS leakage) | 5-trial bounded_randomized_search, metric auc_roc | 80% | exhaustive grid search | TUNE-01 |
| FeatureEngineeringAgent | 24 features after pruning | Correlation pruning applied (>0.95); user did not reject it. | dropped 6 highly-correlated feature(s) | 80% | keep all features | — |
| ModelExecutionAgent | train/test/oos split (0.6, 0.2, 0.2) | generalization gap -0.00153 | trained mlp; explainability via permutation | 85% | diagnostics-only (no training) | EV-312d174e6057 |
| HyperparameterTuningAgent | 5-trial bounded_random_search | train-internal holdout only (no test/OOS leakage) | best metric 1.0000 @ {'learning_rate': 0.003, 'hidden_dims': [128, 64], 'dropout': 0.2} | 85% | grid search | TUNE-01 |
| HyperparameterTuningAgent | 5-fold stratified K-fold on 341 train rows (228 test/OOS rows excluded) | folds created only within the training split; test/OOS never used for model selection | best mean auc_roc=0.9925 (std 0.0069) @ {'C': 3.0, 'class_weight': None} | 90% | single-split validation | TUNE-01 |
| GovernanceSignoffAgent | 2 findings, 6 evidence records | 2 findings weighed against acceptance criteria | Reviewer recommendation: READY FOR SIGN-OFF | 85% | conditional sign-off | EV-8b88ddddccbd, EV-7e423f877e0c |
| EvidenceCriticAgent | all findings, recommendations, sign-off narrative | every claim must cite evidence; uncited claims are flagged | PASSED | 100% | allow uncited narrative | EV-8b88ddddccbd |

## Governance Findings

| Severity | Materiality | Category | Title | Evidence | Recommendation |
| --- | --- | --- | --- | --- | --- |
| Low | Medium | Governance | Governance sign-off disposition | EV-8b88ddddccbd, EV-7e423f877e0c, EV-83c2bafb7934 | Review the governance findings before approval. |
| Low | Low | Operational | Moonshot unavailable | ai_engineering.compliance.moonshot | Optional and NOT recommended in the primary environment: aiverify-moonshot==0.7.6 conflicts with MCP/DeepEval/Garak/LiteLLM (pins pydantic==2.8.2, huggingface-hub~=0.36). Prefer LangSmith (pip install langsmith) or Phoenix (pip install arize-phoenix) for observability/evaluation instead. |

## Evidence Ledger Summary

Total records: 6

| Evidence ID | Test | Status |
| --- | --- | --- |
| EV-8b88ddddccbd | Dataset discovery profile | pass |
| EV-7e423f877e0c | Target selection | pass |
| EV-83c2bafb7934 | Task inference | pass |
| EV-8ff8d852e8b6 | Data split plan | pass |
| EV-e64bf2e4f531 | Feature engineering diagnostics (tabular) | pass |
| EV-312d174e6057 | Cohort performance metrics | pass |

## Final Signoff

**Evidence critique:** PASSED

Reviewer recommendation: READY FOR SIGN-OFF. [EV-8b88ddddccbd]


## Artifact Catalog

| Artifact | Type | Category | Location |
| --- | --- | --- | --- |
| split_distribution.csv | table (CSV) | split | test_dl_output/copilot/RUN-ENT-ba3bef58/split_distribution.csv |
| split_distribution.json | data (JSON) | split | test_dl_output/copilot/RUN-ENT-ba3bef58/split_distribution.json |
| metrics_by_split.csv | table (CSV) | split | test_dl_output/copilot/RUN-ENT-ba3bef58/metrics_by_split.csv |
| confusion_matrix.csv | table (CSV) | execution | test_dl_output/copilot/RUN-ENT-ba3bef58/confusion_matrix.csv |
| training_summary.json | data (JSON) | training | test_dl_output/copilot/RUN-ENT-ba3bef58/training_summary.json |
| training_history.csv | table (CSV) | training | test_dl_output/copilot/RUN-ENT-ba3bef58/training_history.csv |
| global_feature_importance.csv | table (CSV) | explainability | test_dl_output/copilot/RUN-ENT-ba3bef58/global_feature_importance.csv |
| tuning_trials.csv | table (CSV) | tuning | test_dl_output/tuning/RUN-ENT-ba3bef58/tuning_trials.csv |
| tuning_summary.json | data (JSON) | tuning | test_dl_output/tuning/RUN-ENT-ba3bef58/tuning_summary.json |
| fold_metrics.csv | table (CSV) | tuning | test_dl_output/tuning/RUN-ENT-ba3bef58/fold_metrics.csv |
| tuning_trials.csv | table (CSV) | tuning | test_dl_output/tuning/RUN-ENT-ba3bef58/tuning_trials.csv |
| tuning_summary.json | data (JSON) | tuning | test_dl_output/tuning/RUN-ENT-ba3bef58/tuning_summary.json |
| review_graph.json | data (JSON) | graph | test_dl_output/ai_engineering/RUN-ENT-ba3bef58/review_graph.json |
| review_graph.png | figure (PNG) | graph | test_dl_output/ai_engineering/RUN-ENT-ba3bef58/review_graph.png |
| policy_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-ba3bef58/policy_report.json |
| mcp_inventory.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-ba3bef58/mcp_inventory.json |
| mcp_health_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-ba3bef58/mcp_health_report.json |
| langfuse_trace.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-ba3bef58/langfuse_trace.json |
| telemetry.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-ba3bef58/telemetry.json |
| redteam_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-ba3bef58/redteam_report.json |
| promptfoo_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-ba3bef58/promptfoo_report.json |
| guardrail_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-ba3bef58/guardrail_report.json |
| deepeval_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-ba3bef58/deepeval_report.json |
| langsmith_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-ba3bef58/langsmith_report.json |
| phoenix_report.json | data (JSON) | ai_engineering | test_dl_output/ai_engineering/RUN-ENT-ba3bef58/phoenix_report.json |
| dashboard.json | dashboard (JSON) | report | test_dl_output/dashboards/RUN-ENT-ba3bef58/dashboard.json |
| dashboard.md | report (Markdown) | report | test_dl_output/dashboards/RUN-ENT-ba3bef58/dashboard.md |
| dashboard.html | dashboard (HTML) | report | test_dl_output/dashboards/RUN-ENT-ba3bef58/dashboard.html |
