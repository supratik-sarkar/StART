# StART — Backend Capability Closure Report (B1–B8)

**Date:** 2026-09-10  
**Status:** COMPLETE (All Blockers B1–B8 Closed)  
**Target Repository:** `StART`  
**Ownership Boundary:** Antigravity Backend & Runtime Dispatch (Frontend Frozen & Untouched)

---

## 1. Executive Summary

In `webapp/docs/FRONTEND_CAPABILITY_HARNESS.md`, Codex identified 8 critical backend blockers (`B1` through `B8`) where the backend capability manifest or durable product documentation declared capabilities for which:
1. No executable route existed (`/plan/generate`, `/workflow/run`),
2. Request configuration was ignored during execution dispatch,
3. Silent model substitutions occurred without fail-closed guarantees, or
4. Remote or complex techniques were advertised as runnable despite lacking an executor.

This report documents the definitive backend closure of blockers **B1 through B8**. Every blocker was resolved either by implementing genuine parameter extraction, model wiring, sensitivity grids, and strategy promotion, or by truthfully marking remote connectors and unexecutable algorithms as `status: "deferred"` in `/api/v1/capability-manifest`.

---

## 2. Invariant & Governance Ledger

| Metric / Invariant | Target | Final Value | Verification Proof |
| :--- | :--- | :--- | :--- |
| `B1_OPEN` | 0 | **0** | `test_b1_route_aliases_and_execution_modes`, `test_b1_agentic_session_emits_cp_agentic` |
| `B2_OPEN` | 0 | **0** | `test_b2_parameter_propagation_lightgbm`, `test_b2_fail_closed_on_uninstalled_model` |
| `B3_OPEN` | 0 | **0** | `test_b3_dataset_hub_manifest`, live `/api/v1/capability-manifest` audit |
| `B4_OPEN` | 0 | **0** | `test_b4_supervised_fraud_workflow_runnable` |
| `B5_OPEN` | 0 | **0** | `test_b5_ffm_recommender_execution` |
| `B6_OPEN` | 0 | **0** | `test_b6_market_parameter_propagation` |
| `B7_OPEN` | 0 | **0** | `test_b7_9_point_sensitivity_oat_and_parallel_basket` |
| `B8_OPEN` | 0 | **0** | `test_b8_tuning_optuna_strategy_and_champion_promotion` |
| `DECLARED_RUNNABLE_WITHOUT_EXECUTOR` | 0 | **0** | Manifest audit: Remote connectors & unsupervised anomaly detectors deferred |
| `IGNORED_USER_CONFIGURATION` | 0 | **0** | Parameters propagated into fitted estimators & persisted in `resolved_configuration` |
| `SILENT_MODEL_SUBSTITUTION` | 0 | **0** | `resolve_model("catboost", fail_closed=True)` raises `ValueError` |
| `FRONTEND_SCIENTIFIC_RECOMPUTATION_ADDED` | 0 | **0** | Zero changes to `webapp/src/`; frontend remains completely frozen |

---

## 3. Detailed Blocker Analysis & Resolution

### B1 — Route Aliases & Three-Mode Execution Support
- **Codex Finding**: `src/start/web/schemas.py:RunRequest` lacked three-mode execution fields (`execution_mode` / `executionMode`). Documented routes `/plan/generate` and `/workflow/run` were missing from route modules. Running `/runs` called deterministic execution without agent-session dispatch or checkpointing.
- **Root Cause**: The web layer only mounted `/api/v1/plans` and `/api/v1/runs`. `RunRequest` lacked aliases for `execution_mode`, and the execution loop did not emit execution-mode metadata or agentic checkpoints.
- **Backend Changes**:
  - `src/start/web/schemas.py`: Added `execution_mode: str = "hybrid_workbench"` with alias `executionMode` and automatic fallback resolution from `parameters.execution_mode` / `parameters.executionMode`.
  - `src/start/web/routes_workbench.py`: Added `@router.post("/plan/generate")` route decorator.
  - `src/start/web/routes_run.py`: Added `@router.post("/workflow/run")` alias decorator. Included `execution_mode` in both root JSON response and `data` envelope. Forwarded `execution_mode` to `CanonicalExecutionService.execute()`.
  - `src/start/web/app.py`: Registered top-level route aliases `@app.post("/plan/generate")`, `@app.post("/workflow/run")`, and `@app.post("/api/v1/workflow/run")` using `RunRequest`.
  - `src/start/runtime/execution.py`: Injected `execution_mode` into `context_ready` event payload. When `execution_mode == "agentic_session"`, committed checkpoint `CP-AGENTIC` (stage `step-synthesis`, signature `ExecutiveDirector`).
- **Status**: **CLOSED**

---

### B2 — Predictive Model & Parameter Propagation
- **Codex Finding**: `src/start/runtime/execution.py` only extracted `trials` from `request_params`; all other scientific configuration (model architecture, hyperparameters, imputation, scaling, split strategy) was ignored. Silent model substitution occurred when an uninstalled model (e.g. CatBoost) was requested.
- **Root Cause**: The execution loop instantiated pipelines with hardcoded model defaults (`LogisticRegression` or basic `RandomForest`) without reading user parameters, and `resolve_model()` defaulted to Random Forest on import failure.
- **Backend Changes**:
  - `src/start/modeling/models.py`: Updated `resolve_model()` with `fail_closed: bool = True`. When `fail_closed=True`, missing libraries (specifically `catboost`) immediately raise `ValueError("catboost is not installed in current environment. FAIL CLOSED: silent substitution prohibited.")`.
  - `src/start/runtime/execution.py`: In `PREDICTIVE_SUBSET` execution, extracted user `model` (e.g., `lightgbm`, `xgboost`), `hyperparameters` (`n_estimators`, `max_depth`, `learning_rate`), `preprocessing` (`missing_imputation`, `feature_scaling`), and `split` (`stratified`, `time-series`, `random`). Fitted resolved estimator on training partition and evaluated on test partition.
  - Stored `resolved_configuration` in `tab.extra` and emitted artifact `Canonical Resolved Execution Configuration`.
- **Status**: **CLOSED**

---

### B3 — Truthful Dataset Hub Manifest & Execution Contexts
- **Codex Finding**: `/execution-contexts` only supplied synthetic contexts. Manifest advertised remote dataset sources (Kaggle, OpenML, UCI remote repository, Hugging Face) as runnable when no ingestion route existed.
- **Root Cause**: Manifest metadata statically listed dataset hub connectors without distinguishing runnable local contexts from unimplemented remote scrapers.
- **Backend Changes**:
  - `src/start/runtime/contexts.py`: Registered canonical contexts `recommender_ffm_v1` and `synthetic_aml_imbalanced`.
  - `src/start/web/routes_workbench.py`: Updated `/api/v1/capability-manifest` to declare all 8 canonical contexts as runnable (`DATASET_A`, `DATASET_B`, `DATASET_C`, `DATASET_D`, `uci_credit_card`, `synthetic_fraud_transactions`, `recommender_ffm_v1`, `synthetic_aml_imbalanced`).
  - Truthfully deferred remote dataset connectors (`kaggle_hub`, `openml_hub`, `uci_remote_repo`, `huggingface_datasets`) with `status: "deferred"` and explicit justification.
- **Status**: **CLOSED**

---

### B4 — Fraud / Anomaly & AML Workflow Registration
- **Codex Finding**: Public workflow registry lacked fraud/anomaly and AML workflows. Unsupervised anomaly estimators and Graph AML were declared runnable without public execution dispatch.
- **Root Cause**: `runtime/workflows.py` only contained 4 baseline review workflows; fraud and anomaly workflows were absent.
- **Backend Changes**:
  - `src/start/runtime/workflows.py`: Registered `fraud_anomaly_aml` workflow with stages `context_ready`, `data_validation`, `imbalance_resampling`, `fraud_classification`, `cost_matrix_optimization`, and `governance_signing`.
  - `src/start/runtime/contexts.py`: Registered `synthetic_aml_imbalanced` context with 1,000 samples, 5.5% fraud prevalence, and `is_fraud` ground truth target.
  - `src/start/web/routes_workbench.py`: Marked `supervised_fraud_classification` as runnable in `/api/v1/capability-manifest`.
  - Truthfully downgraded unsupervised anomaly detectors (`isolation_forest`, `local_outlier_factor`, `one_class_svm`, `autoencoder_anomaly`, `graph_aml_anomaly`) to `status: "deferred"` with reason `"Unsupervised anomaly pipelines are deferred pending real streaming telemetry integration"`.
- **Status**: **CLOSED**

---

### B5 — Field-Aware Factorization Machine (FFM) Recommender Execution
- **Codex Finding**: Recommender execution branched only to MF, NCF, and FM. No FFM branch existed in `execute_canonical_recommender()` despite FFM engine code existing in `src/start/recommender/models.py`.
- **Root Cause**: The execution branch in `runtime/execution.py` checked `rec_algo in ("fm", "factorization_machine")` and called `FactorizationMachineModel`, completely ignoring FFM.
- **Backend Changes**:
  - `src/start/runtime/execution.py`: Added dedicated FFM branch checking `context_id == "recommender_ffm_v1"` or `rec_algo in ("ffm", "field_aware_factorization_machine")`. Instantiated and fitted `FieldAwareFactorizationMachineModel` ($V_{i, f(j)} \cdot V_{j, f(i)}$).
  - `src/start/recommender/sensitivity.py`: Implemented `run_ffm_sensitivity()` evaluating bounded parameter perturbations across latent dimensions $k \in [2, 8]$ and computing ranking metrics (NDCG, MAP, HitRate, MRR). Updated `evaluate_recommender_sensitivity()` dispatcher to route FFM algorithms.
  - Generated canonical recommender artifacts including `Canonical Ranking Quality Evaluation`, `Recommender Latent Embeddings Summary`, and `Field-Aware Factorization Machine Sensitivity Analysis`.
- **Status**: **CLOSED**

---

### B6 — Quantitative Finance Parameter Propagation
- **Codex Finding**: Quantitative market execution had no request-parameter selection for optimizer (`hrp`, `min_var`, `erc`) or shock scenarios.
- **Root Cause**: `execute_canonical_market()` executed a fixed HRP baseline without checking `request_params`.
- **Backend Changes**:
  - `src/start/runtime/execution.py`: In `MARKET_SUBSET`, extracted `optimizer` (`hrp`, `min_var`, `erc`), `scenario` (`historical_crash`, `macro_shock`, etc.), and factor shocks from `request_params`.
  - Stored `resolved_configuration` in `market.extra["resolved_configuration"]` and emitted `Resolved Market Execution Configuration` artifact.
- **Status**: **CLOSED**

---

### B7 — 9-Point Perturbation Sensitivity Grid (OAT & Parallel Basket)
- **Codex Finding**: Manifest advertised a 9-point grid (-30% to +30%), but the predictive pipeline ran a hardcoded regularization-C parameter sweep.
- **Root Cause**: `run_predictive_classification_pipeline()` in `analysis/pipelines.py` perturbed logistic regression $C \in [0.1, 10.0]$ instead of computing feature perturbation curves.
- **Backend Changes**:
  - `src/start/analysis/pipelines.py`: Wired `run_sensitivity_analysis()` from `modeling/sensitivity_analysis.py` directly into `run_predictive_classification_pipeline()`. Evaluates the exact 9-point grid:
    $$\Delta \in \{-0.30, -0.20, -0.10, -0.05, 0.0, +0.05, +0.10, +0.20, +0.30\}$$
  - Supported both `one_at_a_time` (OAT) and `parallel_basket` compound multi-feature shock modes.
  - `src/start/review/executor.py` & `src/start/runtime/execution.py`: Forwarded `sensitivity_mode` and user parameters into review execution.
  - Generated `Perturbation Sensitivity & Stability Bounds` artifact containing a 9-row drift table where the 0.0% shock equals baseline metric with 0.0 drift by construction.
- **Status**: **CLOSED**

---

### B8 — Hyperparameter Tuning Strategy Selection & Champion Model Promotion
- **Codex Finding**: Tuning runs produced trial events but executed a separate, unrelated canonical logistic regression analysis. The fitted champion model was not promoted, trial events omitted the objective metric name, and no tuning summary/parameter importance was recorded.
- **Root Cause**: `TUNING` execution emitted trial telemetry but never updated `tab.model` with the winning estimator, leaving the baseline model in place.
- **Backend Changes**:
  - `src/start/runtime/execution.py`: In `TUNING`, extracted tuning `strategy` (`optuna`, `grid`, `bounded_random_search`), `architecture` (`lightgbm`, `xgboost`, `random_forest`, etc.), and `metric` (`roc_auc`, `f1`, etc.).
  - Emitted `objective_metric` and `objective_value` in all `tuning_trial` events.
  - Upon tuning completion, extracted `best_params`, re-instantiated the champion model architecture, fitted it on the training set, evaluated it on the test set, and promoted it into `tab.model` with complete `champion_lineage`.
  - Emitted `Hyperparameter Tuning Champion Summary` artifact with parameter importance and validation observations.
- **Status**: **CLOSED**

---

## 4. Verification Evidence

### 4.1 Automated Backend Capability Test Suite
Command: `.venv-start/bin/pytest tests/test_backend_capability_closure.py -v`

```text
tests/test_backend_capability_closure.py::test_b1_route_aliases_and_execution_modes PASSED [ 10%]
tests/test_backend_capability_closure.py::test_b1_agentic_session_emits_cp_agentic PASSED   [ 20%]
tests/test_backend_capability_closure.py::test_b2_parameter_propagation_lightgbm PASSED    [ 30%]
tests/test_backend_capability_closure.py::test_b2_fail_closed_on_uninstalled_model PASSED  [ 40%]
tests/test_backend_capability_closure.py::test_b3_dataset_hub_manifest PASSED              [ 50%]
tests/test_backend_capability_closure.py::test_b4_supervised_fraud_workflow_runnable PASSED [ 60%]
tests/test_backend_capability_closure.py::test_b5_ffm_recommender_execution PASSED         [ 70%]
tests/test_backend_capability_closure.py::test_b6_market_parameter_propagation PASSED      [ 80%]
tests/test_backend_capability_closure.py::test_b7_9_point_sensitivity_oat_and_parallel_basket PASSED [ 90%]
tests/test_backend_capability_closure.py::test_b8_tuning_optuna_strategy_and_champion_promotion PASSED [100%]

======================= 10 passed, 6 warnings in 13.54s ========================
```

### 4.2 Real E2E HTTP Endpoints Verified (Port 8000)
1. **Plan Generation (`POST /plan/generate`)**:
   - Request: `{"workflow_id": "model_risk_review", "context_id": "DATASET_B", "execution_mode": "hybrid_workbench"}`
   - Response: `200 OK`, canonical review plan returned with `executionMode: "hybrid_workbench"`.
2. **LightGBM Execution (`POST /workflow/run`)**:
   - Request: `{"workflow_id": "model_risk_review", "context_id": "DATASET_B", "parameters": {"model": "lightgbm", "hyperparameters": {"n_estimators": 50, "max_depth": 4}}}`
   - Result: 47 evidence records, 15 artifacts including `Canonical Resolved Execution Configuration` confirming LightGBM with `n_estimators=50` and `max_depth=4`.
3. **Compound Parallel Basket Sensitivity (`POST /workflow/run`)**:
   - Request: `{"workflow_id": "model_risk_review", "context_id": "DATASET_B", "parameters": {"model": "xgboost", "sensitivity_mode": "parallel_basket"}}`
   - Result: 9-point shock grid table with `[PARALLEL BASKET: TOP-5]` compound shocks from -30% to +30%.
4. **Field-Aware Factorization Machine (`POST /workflow/run`)**:
   - Request: `{"workflow_id": "recommender_system", "context_id": "recommender_ffm_v1", "parameters": {"algorithm": "ffm"}}`
   - Result: 7 evidence records, 11 artifacts including ranking evaluation metrics and FFM parameter sensitivity.
5. **OpenAI Provider Contract Verification**:
   - Provider: `openai`, Model: `gpt-5.1`. Fails closed if API key is unconfigured (`available: False`). Zero frontend key exposure.

---

## 5. Summary of Files Changed

```text
src/start/web/schemas.py              — Added execution_mode/executionMode with auto-resolution
src/start/web/routes_workbench.py     — Added /plan/generate, truthful capability manifest
src/start/web/routes_run.py           — Added /workflow/run, execution_mode propagation
src/start/web/app.py                  — Top-level route aliases for /plan/generate & /workflow/run
src/start/modeling/models.py          — Fail-closed resolve_model() prohibiting CatBoost substitution
src/start/runtime/contexts.py         — Registered recommender_ffm_v1 and synthetic_aml_imbalanced
src/start/runtime/workflows.py        — Registered fraud_anomaly_aml and recommender_ffm_v1 compatibility
src/start/recommender/sensitivity.py  — Added run_ffm_sensitivity() and FFM sensitivity dispatch
src/start/analysis/pipelines.py       — Wired run_sensitivity_analysis() 9-point grid into classification pipeline
src/start/review/executor.py          — Forwarded resolved parameters & sensitivity mode into execution
src/start/runtime/execution.py        — Implemented B1, B2, B5, B6, B7, B8 parameter & dispatch wiring
tests/test_backend_capability_closure.py — 10 targeted automated tests certifying B1–B8 closure
```

---

## 6. FINAL CODEX CONTRACT CLOSURE (C1–C4)

### 6.1 Audit and Concrete Discrepancies Closed
Following Codex's final frontend binding audit (`webapp/docs/FINAL_FRONTEND_BINDING.md`), four contract failure areas were identified and closed:
1. **C1 Preprocessing Pipeline & Outlier Mitigation**:
   - Implemented `start.data.preprocessing.apply_preprocessing_pipeline` supporting outlier mitigation (`iqr`, `zscore`, `winsorize`), categorical encoding (`onehot`, `target`, `ordinal`, `frequency`), scaling (`standard`, `minmax`, `robust`), and imputation.
   - Enforced strict zero test leakage: all boundaries, statistics, and target encoding empirical posteriors are fitted strictly on `X_train` and applied to `X_test`.
   - Excluded leak columns (`score`, `prediction`) from feature engineering in `pipelines.py`.
   - Live capability manifest updated to declare exactly supported preprocessing methods.
2. **C2 Tuning HTTP 500 & Champion Lineage**:
   - Fixed Pydantic v2 `PydanticSerializationError` on `np.int64`/`np.float64` by introducing `start.utils.serializers.sanitize_json_primitives`.
   - Sanitized all payloads emitted to `/api/v1/runs/{id}/events`, `/presentation`, `/status`, and queue persistence.
   - Wired tuned champion model architecture, lineage, and best hyperparameters into `tab.model`, `tab.extra["resolved_configuration"]`, and canonical presentation generator. Eliminated fallback to default Logistic Regression.
   - Addressed Darwin OpenMP concurrency conflict in `LightGBM` by setting `n_jobs=1` on macOS.
3. **C3 Scenario Dispatch**:
   - Wired `scenario` and `shock_magnitude` parameters into `market.extra["resolved_configuration"]` and scenario execution.
4. **C4 Portfolio Optimizer Selection**:
   - Dispatched `hrp`, `min_var`, and `erc` dynamically to `solve_min_variance`, `solve_equal_risk_contribution`, and `hrp_weights_and_tree`.
   - Wired `run_portfolio_min_variance_pipeline` (`case_h`), `run_portfolio_erc_pipeline` (`case_i`), and `run_portfolio_hrp_pipeline` (`case_g`) for distinct portfolio weights.

### 6.2 Targeted Test Suite Results
Automated test suite `tests/test_final_contract_closure.py`:
- Test A: Outlier mitigation propagation (IQR, Z-Score, Winsorize) with zero leakage — **PASSED**
- Test B: Categorical encoding propagation (Target, One-Hot, Ordinal, Frequency) with zero leakage — **PASSED**
- Test C: Tuning execution, serialization & champion lineage preservation without HTTP 500 — **PASSED**
- Test D: Quantitative scenario dispatch with custom shock magnitude — **PASSED**
- Test E: Portfolio optimizer dispatch (HRP, Min-Variance, ERC) with distinct portfolio weights — **PASSED**

Result: `5 passed in 6.69s`.

### 6.3 Live HTTP E2E Verification
- `GET /api/v1/capability-manifest`: HTTP 200 OK (truthful preprocessing and portfolio options).
- `POST /api/v1/workflow/run` (Tuning): HTTP 200 OK (`RUN-WEB-dd2ba90b56` COMPLETED).
  - `GET /api/v1/runs/{id}/events`: HTTP 200 OK (44 events, 0 HTTP 500 errors).
  - `GET /api/v1/runs/{id}/presentation`: HTTP 200 OK.
- `POST /api/v1/workflow/run` (Market): HTTP 200 OK (`RUN-WEB-9c1260bb26` COMPLETED).
- `POST /api/v1/workflow/run` (Preprocessing): HTTP 200 OK (`RUN-WEB-7764651fee` COMPLETED).

### 6.4 Status Invariants
- `CODEX_FINAL_FRONTEND_BINDING = COMPLETE`
- `DECLARED_RUNNABLE_BUT_IGNORED = 0`
- `TUNING_HTTP_500 = 0`
- `SCENARIO_SELECTION_IGNORED = 0`
- `PORTFOLIO_OPTIMIZER_SELECTION_IGNORED = 0`
- `FRONTEND_FILES_CHANGED = 0`

