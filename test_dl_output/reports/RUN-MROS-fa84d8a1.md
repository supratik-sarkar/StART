# StART Model-Risk Review — `RUN-MROS-fa84d8a1`

## Review summary
- Task type: **binary_classification**
- Target column(s): `attrition`
- Target type: binary (2 classes)
- Modality: tabular
- Recommended model family: `mlp`
- Agent mode: deterministic
- Evidence critique: PASSED

## Pipeline stages (visible execution)

| Stage | Status | Detail |
| --- | --- | --- |
| dataset | complete | 569 rows x 31 columns |
| discovery | complete | 569 rows x 31 columns; 1 candidate target(s), 0 text, 0 image-path, 0 timestamp, 0 entity column(s). |
| target_confirmation | complete | target = attrition |
| task_inference | complete | binary_classification |
| split_planning | complete | stratified (341, 114, 114) |
| feature_engineering | complete | tabular diagnostics |
| model_recommendation | complete | recommended: mlp |
| model_execution | complete | device=mps |
| metrics | complete |  |
| explainability | complete | computed |
| sensitivity | complete | computed |
| robustness | complete | computed |
| evidence_ledger | complete | 6 records sealed |
| review_planner | complete |  |
| test_suggestion | complete |  |
| model_risk_finding | complete |  |
| challenge | complete | 1 memo items |
| governance | complete |  |
| signoff | complete | Reviewer recommendation: READY FOR SIGN-OFF |
| evidence_critic | complete | citation gate: passed |
| ai_engineering | complete | 9/10 stages available; rest reported not installed |

## Cohort metrics

| Cohort | AUC-ROC | Accuracy | F1 |
| --- | --- | --- | --- |
| train | 0.9889 | 0.9326 | 0.9004 |
| test | 0.9974 | 0.9298 | 0.8974 |
| oos | 0.9970 | 0.9211 | 0.8800 |

## Evidence ledger

| Test ID | Name | Status |
| --- | --- | --- |
| discovery.dataset_profile | Dataset discovery profile | pass |
| discovery.target_selection | Target selection | pass |
| discovery.task_inference | Task inference | pass |
| split.plan | Data split plan | pass |
| feature_engineering.diagnostics | Feature engineering diagnostics (tabular) | pass |
| execution.cohort_metrics | Cohort performance metrics | pass |

## AI-engineering stage surface

| Stage | Category | Status |
| --- | --- | --- |
| Policy Validation | policy | complete |
| MCP Integration | mcp | complete |
| Observability Export | observability | complete |
| Telemetry | telemetry | complete |
| Red Team Evaluation (Garak) | redteam | complete |
| Red Team Evaluation (Promptfoo) | redteam | complete |
| Compliance Evaluation | compliance | not_installed |
| Guardrails | guardrails | complete |
| Evals (DeepEval) | evals | complete |
| Orchestration (LangGraph) | orchestration | complete |

## Agentic review

### Reviewer plan
- Review the evidence in breach-first order; warnings next, passes last.
- Read 'Dataset discovery profile' (status: pass). [EV-8b88ddddccbd]
- Read 'Target selection' (status: pass). [EV-7e423f877e0c]
- Read 'Task inference' (status: pass). [EV-83c2bafb7934]
- Read 'Data split plan' (status: pass). [EV-8ff8d852e8b6]
- Read 'Feature engineering diagnostics (tabular)' (status: pass). [EV-e64bf2e4f531]
- Read 'Cohort performance metrics' (status: pass). [EV-312d174e6057]

### Challenge memo
- No memorization or sampling anomalies found. [EV-8b88ddddccbd]

### Governance assessment
- Governance checks passed: no breaches, no skipped tests, all evidence policy-stamped.

### Sign-off recommendation
Reviewer recommendation: READY FOR SIGN-OFF. [EV-8b88ddddccbd]

## Assumptions
- Task inferred as binary_classification; Inferred binary_classification from target 'attrition' (2 unique values).
- Split holds out an explicit OOS cohort for generalization estimates.
- Deterministic diagnostics compute metrics; the LLM (if used) reasons only over evidence.

## Validation recommendations
- Confirm the inferred task type and target with a domain owner before sign-off.
- Review any warn/fail evidence and the challenge memo before production use.
- For deep-learning model families, run the dedicated DL review workflow for full explainability, sensitivity, and robustness suites.
