# StART — Agentic AI Engineering Workbench Product Specification

**Version:** 5.2.0  
**Status:** Canonical Durable Specification  
**Authority:** Product Architecture & Engineering  
**Primary Repository Location:** `StART`

---

## 0. Product Vision & Global Identity

### 0.1 Public Product Identity
StART ceases to present itself primarily as a model validation or model review tool. Its public identity on GitHub and in open engineering environments is:

> **StART — Agentic AI Engineering Workbench**  
> *Build · Tune · Stress · Explain · Compare · Govern*

A next-generation platform for engineering, inspecting, and governing machine learning, deep learning, quantitative finance, recommender, and autonomous agent systems.

### 0.2 Core Capabilities Lifecycle
The workbench provides end-to-end support across the full model and algorithm engineering lifecycle:
1. **Build**: Define pipelines, dataset schemas, feature roles, and training objectives.
2. **Configure**: Select estimators, preprocessing transforms, imputation strategies, and split protocols.
3. **Experiment**: Run candidate models, baselines, and ablation variations.
4. **Tune**: Execute Bayesian (Optuna), Grid, and Random hyperparameter searches with trial histories.
5. **Train**: Fit tree ensembles, linear baselines, PyTorch deep neural networks, and factor models.
6. **Evaluate**: Compute comprehensive performance metrics across classification, regression, ranking, and risk.
7. **Stress**: Apply controlled feature perturbations, distribution shifts, and macroeconomic scenario shocks.
8. **Explain**: Generate global and local feature attributions via Permutation Importance, Tree SHAP, PDP, and ICE.
9. **Compare**: Conduct multi-candidate and champion-versus-challenger delta analyses.
10. **Iterate**: Spawn deterministic child runs recording parameter variations.
11. **Trace**: Track full execution graphs, intermediate tensors, and runtime event streams.
12. **Challenge**: Interrogate findings, empirical conclusions, and model assumptions via reasoning agents.
13. **Evidence**: Commit immutable, cryptographically hashed `EvidenceRecord` objects.
14. **Govern**: Sign and seal attestation Merkle roots with complete provenance and decision receipts.

### 0.3 Public Scope of Domains
The workbench natively structures engineering workspaces across:
* **Predictive ML**: Tabular binary, multiclass, and regression problems.
* **Deep Learning**: Tabular MLP, Sequence (RNN/LSTM/GRU/Bi-LSTM), and Vision (CNN architectures).
* **Fraud / Anomaly / AML**: Supervised fraud with severe class imbalance, unsupervised outlier detection (Isolation Forest, LOF, One-Class SVM), and transaction graphs.
* **Recommender Systems**: Matrix Factorization, Neural Collaborative Filtering (NCF), Factorization Machines (FM), and Field-Aware Factorization Machines (FFM).
* **Quantitative Finance & Market Risk**: Return construction, covariance estimation, factor risk, volatility modeling, and historical/hypothetical scenario engineering.
* **Scenario Engineering**: Gate-6 / Gate-6A historical shocks, factor shocks, asset shocks, correlated shocks, and reverse stress testing.
* **Portfolio Optimization**: Hierarchical Risk Parity (HRP), Minimum Variance (MinVar), Equal Risk Contribution (ERC/Risk Parity), and HERC.
* **LLM / RAG / Agent Engineering**: Evaluation of autonomous AI agents, tool correctness, retrieval grounding, citation integrity, and latency/cost tracking.

---

## 1. Architectural Principles & Invariants

### 1.1 Separation of Concerns
# AGENTS PROPOSE / REASON
# DETERMINISTIC TOOLS CALCULATE
# EVIDENCE RECORDS PROVE

1. **Deterministic Scientific Sovereignty**: All calculations (metrics, ROC curves, loss values, eigenvalues, dendrogram linkages, SHAP attributions, sensitivity deltas, and risk contributions) are computed strictly by deterministic Python engines. LLMs NEVER perform arithmetic or numeric calculations.
2. **Zero Scientific Duplication in Frontend**: The React application renders and configures; it never calculates ROC points, confusion matrices, SHAP values, or portfolio allocations.
3. **Fail-Closed Security & Provenance**: Evidence records are immutable and cryptographically hashed with SHA-256 Merkle roots.
4. **Zero Credential Exposure**: API keys (including OpenAI) exist only in backend memory / secure macOS Keychain and are never exposed to Vite, client JavaScript, local storage, or network responses.

---

## 2. Dynamic Capability Manifest Architecture

### 2.1 Backend Capability Manifest Endpoint (`GET /api/v1/capability-manifest`)
The frontend must not hardcode the scientific capability surface. The backend exposes an authoritative manifest declaring all available engines, models, preprocessors, metrics, and execution modes.

### 2.2 Schema Structure (`StARTCapabilityManifest`)
```json
{
  "platform": {
    "name": "StART",
    "version": "5.2.0",
    "schema_version": "5.2.0",
    "product_profile": "PUBLIC",
    "tagline": "Agentic AI Engineering Workbench"
  },
  "execution_modes": [
    {
      "id": "agentic_session",
      "label": "Agentic Session",
      "description": "Interactive autonomous planning where AI agents propose pipeline design and hyperparameters based on stated objectives.",
      "requires_ai": true
    },
    {
      "id": "deterministic_run",
      "label": "Deterministic Run",
      "description": "Full manual control over reproducible pipeline configuration with 0 automatic LLM calls.",
      "requires_ai": false
    },
    {
      "id": "hybrid_workbench",
      "label": "Hybrid Workbench",
      "description": "Recommended default. Agents propose designs, humans inspect, edit, or challenge, and deterministic tools compute all science.",
      "requires_ai": true,
      "is_default": true
    }
  ],
  "dataset_sources": [...],
  "datasets": [...],
  "domains": [...],
  "tasks": [...],
  "models": [...],
  "preprocessing": [...],
  "split_strategies": [...],
  "tuners": [...],
  "metrics": [...],
  "explainers": [...],
  "sensitivity_methods": [...],
  "stress_methods": [...],
  "artifacts": [...],
  "agent_actions": [...],
  "provider_capabilities": [...]
}
```

---

## 3. Global Terminology & Presentation Profiles

### 3.1 Public Terminology Profile (Default)
| Legacy Validation Term | Public Engineering Workbench Term |
| :--- | :--- |
| `New Review` | `New Workspace` |
| `Review Objective` | `Objective` |
| `Build Review Plan` | `Build Plan` |
| `Execute Review` | `Run Plan` |
| `Completed Review` | `Run Complete` |
| `Investigation Canvas` | `Analysis Workspace` |
| `Review Findings` | `Findings` |
| `Review History` | `Run History` |
| `Validation Tests` | `Tests & Evaluations` |
| `Reviewer` | `Agentic Assistant` / `Reviewer Agent` |
| `Model Review` | `AI / Model Engineering` |

### 3.2 Enterprise Model Risk Terminology Profile (Configurable Switch)
For regulated institutional environments, the terminology switches gracefully to MRM governance conventions (`Review`, `Validate`, `Approve`, `Disposition`, `Attestation Seal`) without altering the underlying computational engine.

---

## 4. Execution Modes

### 4.1 AGENTIC SESSION
* User supplies an engineering objective.
* Orchestrated committee (Planner, Data, Preprocessing, Modeling, Experiment, Stress, Explainability, Critic) autonomously drafts the end-to-end plan.
* Produces structured proposals with clear rationales.

### 4.2 DETERMINISTIC RUN
* Complete reproducibility with 0 automatic LLM calls.
* User configures datasets, preprocessing, estimators, hyperparameters, and evaluations manually.
* Unaffected by external network or provider status.

### 4.3 HYBRID WORKBENCH (Preferred Default)
* The default experience for modern engineering.
* Agents propose configuration based on data characteristics.
* Humans retain full sovereignty: `Accept`, `Edit`, `Override`, `Challenge`, `Ask Why`.
* Deterministic engines execute every computation.

---

## 5. AI Provider Integration (`gpt-5.1`)

### 5.1 Strict Provider & Model Constraints
* **Provider**: OpenAI
* **Exact Model**: `gpt-5.1`
* **Zero Model Substitution**: Newer models (`gpt-5.2`, `gpt-5.4`, etc.) or older models must NOT be silently substituted. If `gpt-5.1` is unreachable, AI activation must report blocked.
* **Storage**: Local macOS Keychain or backend environment variable (`START_LLM__KEY`).
* **Transport**: Server-side only via backend FastAPI routes (`/api/v1/runs/{id}/question`).
* **Invariants**:
  - `FRONTEND_OPENAI_CALLS = 0`
  - `WEBLLM_CALLS_FOR_THIS_TEST = 0`
  - `API_KEY_EXPOSED = 0`

---

## 6. Dataset Hub & Semantic Data Contracts

### 6.1 Semantic Features
Replaces artificial `feat_0` columns with realistic semantic names:
* `credit_amount`, `duration_months`, `age_years`, `income`, `debt_to_income`, `employment_length`, `delinquencies`, `utilization`.

### 6.2 Supported Dataset Connectors
* **Built-in Curated Benchmarks**: German Credit (1,000 rows, 20 attributes), Fannie Mae Loan Performance, Credit Card Fraud, MovieLens Ratings, Multi-Asset Market Returns.
* **Connectors**: OpenML adapter, UCI Machine Learning Repository, Kaggle dataset registry (credentials/licensing managed), Hugging Face datasets adapter, Local CSV and Parquet files (`pyarrow`).

---

## 7. Predictive ML Pipeline & Champion/Challenger Framework

### 7.1 Estimators & Candidate Families
* **XGBoost**: `xgboost.XGBClassifier` with histogram tree method.
* **LightGBM**: `lightgbm.LGBMClassifier` (real optional integration wired).
* **CatBoost**: `catboost.CatBoostClassifier`.
* **Random Forest**: `sklearn.ensemble.RandomForestClassifier`.
* **Gradient Boosting**: `sklearn.ensemble.GradientBoostingClassifier`.
* **Baseline**: `sklearn.linear_model.LogisticRegression`.

### 7.2 Preprocessing & Feature Engineering
* **Missing Value Imputation**: `None`, `Mean`, `Median`, `Mode / Most Frequent`, `Constant`, `KNN Imputation`, `Iterative Imputation`.
* **Outlier Engineering**: `None`, `Winsorization (IQR 1.5x)`, `Wide Winsorization (IQR 3.0x)`, `Percentile Clipping (1/99)`, `Z-score (|z|>3)`.
* **Categorical Encoding**: `One-Hot Encoding`, `Ordinal Encoding`, `Target Encoding`, Native tree categorical handling.
* **Feature Scaling**: `None`, `StandardScaler`, `RobustScaler`, `MinMaxScaler`.
* **Data Splitting**: `Holdout`, `Stratified Holdout`, `K-Fold`, `Stratified K-Fold`, `Group Split`, `Group K-Fold`, `Time-Based Sequential Split`.

### 7.3 Hyperparameter Tuning
* **Search Algorithms**: Grid Search, Random Search, Bayesian Optimization via `optuna.study`.
* **Tuning Artifacts**: Trial table, Optimization history, Best configuration, Parameter importance, Parallel-coordinate data.

### 7.4 Explainability Laboratory
* **Global Attributions**: Permutation Importance, Tree SHAP global summary, Partial Dependence Plots (PDP), Accumulated Local Effects (ALE).
* **Local Attributions**: SHAP local waterfall, ICE curves, LIME explanations.

### 7.5 Sensitivity Analysis (Required Exact Design)
* **Feature Selection**: Top 5 global features ranked by permutation/SHAP importance.
* **Exact Shock Grid**: `-30%`, `-20%`, `-10%`, `-5%`, `0%`, `+5%`, `+10%`, `+20%`, `+30%`.
* **Primary Evaluation Metric**: `ROC-AUC`.
* **Two Operating Modes**:
  1. *One-At-A-Time (OAT)*: Shock each top feature independently while keeping others baseline.
  2. *Parallel Basket*: Shock all top five features simultaneously at the selected level.
* **Two Scientific Semantics**:
  1. *Fixed-Model Sensitivity*: Perturb evaluation inputs -> score existing fitted model -> observe metric drift.
  2. *Retraining Sensitivity*: Perturb training dataset -> retrain model pipeline -> evaluate generalization response.

---

## 8. Deep Learning Laboratory

### 8.1 Dedicated Architectures
* **Tabular DL**: Deep MLP with configurable hidden layers, activations (`ReLU`, `GELU`, `SiLU`, `LeakyReLU`, `Tanh`), dropout, and batch normalization.
* **Sequence / Time Series**: Recurrent Neural Networks (`RNN`, `LSTM`, `GRU`, `Bi-LSTM`).
* **Computer Vision**: Convolutional Neural Networks (`simple_cnn_small`, `simple_cnn_medium`, `simple_cnn_deep`).

### 8.2 Training & Optimization
* **Optimizers**: `Adam`, `AdamW`, `SGD`, `RMSprop`.
* **Schedulers**: StepLR, CosineAnnealingLR, ReduceLROnPlateau.
* **Diagnostics**: Loss curves (Train vs Validation), Learning rate schedule, Gradient norms, Checkpoints.

---

## 9. Fraud / Anomaly / AML Domain

### 9.1 Supervised Fraud
* Severe class imbalance mitigation via class weighting, SMOTE, and focal loss.
* Metrics: `PR-AUC`, `Precision@K`, `Recall@FPR<=1%`, F1-optimal and Cost-optimal threshold analysis.

### 9.2 Unsupervised Anomaly Detection
* `Isolation Forest` (`sklearn.ensemble.IsolationForest`).
* `Local Outlier Factor` (`sklearn.neighbors.LocalOutlierFactor`).
* `One-Class SVM` (`sklearn.svm.OneClassSVM`).

---

## 10. Recommender Systems Laboratory

### 10.1 Distinct Estimator Engines
* **Matrix Factorization (MF)**: Biased regularized SGD matrix factorization ($r_{ui} = \mu + b_u + b_i + P_u \cdot Q_i$).
* **Neural Collaborative Filtering (NCF)**: Dual-stream Generalized Matrix Factorization + Multi-Layer Perceptron neural network.
* **Factorization Machine (FM)**: 2-way linear-time feature interaction modeling.
* **Field-Aware Factorization Machine (FFM)**: Distinct field-aware latent factors ($V_{i, f(j)} \cdot V_{j, f(i)}$).

### 10.2 Recommender Metrics
* **Ranking**: `Precision@K`, `Recall@K`, `NDCG@K`, `HitRate@K`, `MRR`, `MAP@K`.
* **Rating**: `RMSE`, `MAE`.
* **Beyond-Accuracy**: Catalog coverage, Gini diversity, Novelty score, Cold-start degradation.

---

## 11. Quantitative Finance, Scenario Engineering & Portfolio Optimization

### 11.1 Scenario Engineering & Stress Testing
* Gate-6 & Gate-6A stress testing: Historical shocks (2008 GFC, 2020 Covid), Factor shocks, Asset shocks, Correlated shocks, Bounded Reverse Stress Testing.

### 11.2 Deferred Scope Invariant
* **Monte Carlo VaR / Expected Shortfall**: Retained strictly as `DEFERRED` per historical acceptance contracts. Never advertised as runnable.

### 11.3 Portfolio Optimization Workflows
* **Hierarchical Risk Parity (HRP)**: Returns -> Correlation -> Distance -> Linkage clustering -> Dendrogram -> Quasi-diagonal ordering -> Recursive bisection weights -> Risk contributions.
* **Minimum Variance (MinVar)**: Ledoit-Wolf / empirical covariance -> quadratic optimization under constraints -> weights -> risk contributions.
* **Equal Risk Contribution (ERC / Risk Parity)**: Equalizing risk contributions across assets -> risk-budget deviation analysis.

---

## 12. Experiment Lineage & Artifact Navigation

Every execution produces a traceable lineage graph:
`Dataset -> Preprocessing -> Model -> Configuration -> Hyperparameter Tuning -> Stress/Sensitivity -> Evidence -> Champion`

Child runs capture parameter differentials (`perturbation_rate`, `model_type`, `regularization`), enabling comprehensive delta comparison in the workbench.

---

## 13. Codex Porcelain Engineering Visual System Invariants

1. **Identity**: Porcelain Engineering Workspace.
2. **Palette**: Warm porcelain backgrounds (`#fbfbfa`), graphite typography, institutional indigo accents, subtle stone borders (`#e6e5e0`).
3. **Typography**: Inter / system sans for editorial clarity, JetBrains Mono for metrics and identities.
4. **Layout**: Staged progressive disclosure (`DEFINE` -> `EXECUTE` -> `INVESTIGATE`).
5. **Anti-Patterns Forbidden**:
   - 0 old 7-tab architecture.
   - 0 active surfaces dump.
   - 0 event wall upon run completion.
   - 0 raw Evidence ID walls.
