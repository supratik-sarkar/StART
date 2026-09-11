# StART — Agentic AI Engineering Workbench: E2E Acceptance Specification & Runbook

This document defines the rigorous verification and acceptance runbook for **StART — Agentic AI Engineering Workbench** (*Build · Tune · Stress · Explain · Compare · Govern*).

---

## 1. Acceptance Prerequisites

Before initiating the end-to-end acceptance sequence, verify the host environment:

1. **Virtual Environment**:
   - Python executable: `.venv-start/bin/python` (Python 3.11+)
   - Core libraries: `torch`, `xgboost`, `lightgbm`, `catboost`, `scikit-learn`, `optuna`, `shap`, `fastapi`, `uvicorn`.
2. **OpenAI Credentials & Model Verification**:
   - OpenAI API Key retrieved securely from macOS Keychain (`account: start_openai_key`, `service: StART_OpenAI`).
   - Run probe: `.venv-start/bin/python scripts/provider_probe.py`
   - Verification criteria:
     - Target model: `gpt-5.1`
     - Provider status: `ONLINE`
     - Latency: Reported in milliseconds (< 10,000ms)
     - `MODEL_SUBSTITUTION = 0` (strict enforcement)
     - `API_KEY_EXPOSED = 0` (zero key leakage to browser/client)
3. **Frontend Build & Test**:
   - Working directory: `webapp/`
   - Vitest suite: `npm test -- --run`
   - Typecheck: `npm run typecheck`
   - Production bundle: `npm run build`

---

## 2. Tested Execution Modes

StART exposes three distinct execution modes. The acceptance sequence exercises all three:

1. **`HYBRID WORKBENCH`** (Primary Default):
   - Combines generative AI for requirement synthesis, architecture recommendation, and objective structuring (`gpt-5.1`) with deterministic science execution engines (scikit-learn, XGBoost, LightGBM, PyTorch, Optuna).
2. **`AGENTIC SESSION`**:
   - Initiates multi-agent deliberation where specialized personas (Model Architect, Tuner, Stress Engineer, Fairness Auditor) deliberate on model constraints and trade-offs.
3. **`DETERMINISTIC RUN`**:
   - Direct, deterministic execution pipeline without generative LLM calls, providing 100% offline reproducible runs for CI/CD and air-gapped environments.

---

## 3. Step-by-Step E2E Acceptance Sequence

The automated browser acceptance test (`scratch/workbench_e2e.py` or Playwright script) executes the following end-to-end journey:

### Step 1: Initial Workbench Load & Capability Manifest Fetch
- The browser navigates to `http://127.0.0.1:8000/`.
- UI invokes `GET /api/v1/capability-manifest`.
- Verifies that the manifest loads dynamically with 8 domains:
  - Predictive ML (including Random Forest, XGBoost, LightGBM, CatBoost, Logistic Regression, Gradient Boosting)
  - Deep Learning (Sequence & Vision)
  - Fraud & Anomaly (IForest, OCSVM, LOF, Autoencoders)
  - Recommender Systems (MF, NCF, FM, and FFM)
  - Quant & Risk (HRP, MinVar, ERC, Scenario Shocks)
  - LLM & Agents (12-agent roster, prompt chaining)
  - Dataset Hub (UCI German Credit semantic features)
  - Governance (Evidence sealing)
- Verifies header status pill: `AI · OpenAI · gpt-5.1 · Ready` and mode `HYBRID WORKBENCH`.

### Step 2: Workspace Setup with Semantic Dataset
- Select domain: `Predictive ML`.
- Select dataset: `UCI German Credit (Semantic Features)`.
- Verify features render with semantic names (`credit_amount`, `duration_months`, `age_years`, etc.) rather than opaque `feat_0`.
- Select Estimator: `LightGBM Classifier` or `XGBoost Classifier`.
- Select Outlier Method: `Isolation Forest`.
- Select Imputation Method: `Median Imputation`.
- Select Feature Scaling: `Standard Scaler`.
- Select Categorical Encoding: `OneHot Encoding`.
- Select Tuning: `Optuna Bayesian Optimization (10 trials)`.
- Configure Sensitivity: Top 5 Global Features with full 9-point shock grid (`-30%` to `+30%`), mode: `Parallel Basket`.

### Step 3: AI Build Plan Generation (`gpt-5.1`)
- Enter Objective: *"Engineer high-precision credit underwriting model with robust tail risk bounds and non-linear sensitivity profiling under -30% to +30% macroeconomic shocks."*
- Click `Generate Plan (AI)`.
- Backend triggers `POST /api/v1/plan/generate` using `gpt-5.1`.
- Verify plan modal displays structured steps: Data Partitioning, Feature Transformation, Hyperparameter Optimization, Deterministic Evaluation, Stress Frontier Analysis.

### Step 4: Deterministic Pipeline Execution
- Click `Execute Plan` (or `Build & Run`).
- Backend executes `POST /api/v1/workflow/run`.
- Telemetry displays real-time progress across pipeline stages.
- Engines complete:
  - K-fold Cross Validation
  - Bayesian Optuna tuning iterations
  - Confusion matrix and ROC-AUC computation
  - TreeSHAP feature attribution
  - 9-point shock grid sensitivity evaluations (-30% to +30%)
- Status transitions to `RUN COMPLETE`.

### Step 5: Evidence Vault & Question Answering
- Canvas renders diagnostic cards, SHAP plots, and sensitivity curves.
- Open `Artifact & Evidence Vault`.
- Verify presence of deterministic `EvidenceRecord` with SHA-256 hash.
- In Ask Evidence drawer, submit query: *"What are the top risk drivers for elevated default probability under credit expansion shocks?"*
- Backend executes `POST /api/v1/evidence/ask` using `gpt-5.1`, grounded strictly in the generated evidence record.
- Verify answer accurately references the generated metrics without hallucination.

### Step 6: Governance Sign-Off & Receipt
- Trigger `Sign-off & Seal`.
- Enter Approver metadata (`Engineering Lead`, sign-off notes).
- Backend executes `POST /api/v1/approval/sign`.
- Verify receipt generated with cryptographic seal, timestamp, and immutable record link.

### Step 7: History & Comparison Verification
- Click `+ New Workspace`.
- Verify fresh state resets inputs cleanly.
- Re-run a Challenger Model (e.g. `Random Forest Classifier` or baseline `Logistic Regression`).
- Open `Run History` rail.
- Select both runs and click `Compare Runs`.
- Verify side-by-side metric delta display (AUC diff, Brier score diff, Latency diff).

---

## 4. Verification Evidence & Artifact Standards

Every completed acceptance run produces verifiable evidence:
1. **Screenshots**: High-resolution PNGs captured at every major transition (saved to `scratch/acceptance_pass/screenshots/`).
2. **API Interaction Log**: Full request/response traces confirming `gpt-5.1` model usage and HTTP 200 OK statuses.
3. **Vitest & Typecheck Log**: Zero TypeScript errors, zero Vitest test failures.
4. **Backend Test Log**: Passing PyTest suite for extended models, sensitivity grids, and FFM recommender.
