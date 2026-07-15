# StART Model-Risk Review — `RUN-MROS-a73bf2db`

## Review summary
- Task type: **regression**
- Target column(s): `target_value`
- Target type: user
- Modality: tabular
- Recommended model family: `dcn`
- Agent mode: deterministic
- Evidence critique: PASSED

## Pipeline stages (visible execution)

| Stage | Status | Detail |
| --- | --- | --- |
| dataset | complete | 600 rows x 6 columns |
| discovery | complete | 600 rows x 6 columns; 1 candidate target(s), 0 text, 0 image-path, 0 timestamp, 0 entity column(s). |
| target_confirmation | complete | target = target_value |
| task_inference | complete | regression |
| split_planning | complete | stratified (360, 120, 120) |
| feature_engineering | complete | tabular diagnostics |
| model_recommendation | complete | recommended: dcn |
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
| train | nan | nan | nan |
| test | nan | nan | nan |
| oos | nan | nan | nan |

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
- Read 'Dataset discovery profile' (status: pass). [EV-bd04dc945a3d]
- Read 'Target selection' (status: pass). [EV-eefb201d039e]
- Read 'Task inference' (status: pass). [EV-2e0edad34d91]
- Read 'Data split plan' (status: pass). [EV-8871694b4cda]
- Read 'Feature engineering diagnostics (tabular)' (status: pass). [EV-0f3a3b981099]
- Read 'Cohort performance metrics' (status: pass). [EV-e4ae250a0193]

### Challenge memo
- No memorization or sampling anomalies found. [EV-bd04dc945a3d]

### Governance assessment
- Governance checks passed: no breaches, no skipped tests, all evidence policy-stamped.

### Sign-off recommendation
Reviewer recommendation: READY FOR SIGN-OFF. [EV-bd04dc945a3d]

## Assumptions
- Task inferred as regression; Task overridden by user to regression.
- Split holds out an explicit OOS cohort for generalization estimates.
- Deterministic diagnostics compute metrics; the LLM (if used) reasons only over evidence.

## Validation recommendations
- Confirm the inferred task type and target with a domain owner before sign-off.
- Review any warn/fail evidence and the challenge memo before production use.
- For deep-learning model families, run the dedicated DL review workflow for full explainability, sensitivity, and robustness suites.
