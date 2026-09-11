# StART — Agentic AI Engineering Workbench: Capability Census

This document serves as the exhaustive, definitive capability census for **StART — Agentic AI Engineering Workbench** (*Build · Tune · Stress · Explain · Compare · Govern*). It audits capabilities across all core engineering domains, mapping each feature to its backend implementation, current activation status, frontend exposure, test coverage, and explicit deferred/unimplemented classifications.

---

## 1. Capability Census Matrix

| Domain | Capability | Backend Source | Current Status | Frontend Status (Before) | Frontend Status (After) | Tests | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Predictive ML** | Random Forest Classifier | `src/start/modeling/models.py` (`RandomForestClassifier`) | Production Ready | Exposed | Exposed | `tests/test_modeling.py` | Full scikit-learn tree ensemble with feature importances |
| **Predictive ML** | XGBoost Classifier | `src/start/modeling/models.py` (`XGBClassifier`) | Production Ready | Exposed | Exposed | `tests/test_modeling.py` | Native gradient boosting with histogram acceleration |
| **Predictive ML** | LightGBM Classifier | `src/start/modeling/models.py` (`LGBMClassifier`) | Production Ready | Partial | Fully Exposed | `tests/test_models_extended.py` | `lightgbm 4.7.0` active; fast leaf-wise gradient boosting |
| **Predictive ML** | CatBoost Classifier | `src/start/modeling/models.py` (`CatBoostClassifier`) | Fail-Closed (Deferred if uninstalled) | Partial | Fully Exposed | `tests/test_models_extended.py`, `tests/test_backend_capability_closure.py` | Fails closed with `ValueError` if uninstalled; silent substitution strictly prohibited |
| **Predictive ML** | Extra Trees Classifier | `src/start/modeling/models.py` (`ExtraTreesClassifier`) | Production Ready | Partial | Fully Exposed | `tests/test_modeling.py` | Extremely randomized trees ensemble |
| **Predictive ML** | Multi-Layer Perceptron (MLP) | `src/start/modeling/models.py` (`MLPClassifier`) | Production Ready | Partial | Fully Exposed | `tests/test_modeling.py` | Dense feedforward neural network |
| **Predictive ML** | Logistic Regression (Baseline) | `src/start/modeling/models.py` (`LogisticRegression`) | Production Ready | Missing | Fully Exposed | `tests/test_models_extended.py` | Linear baseline with L2/L1 regularization |
| **Predictive ML** | Gradient Boosting Classifier | `src/start/modeling/models.py` (`GradientBoostingClassifier`) | Production Ready | Missing | Fully Exposed | `tests/test_models_extended.py` | Scikit-learn deviance-loss gradient boosting |
| **Predictive ML** | Distributed Random Forest | `src/start/modeling/models.py` (`DistributedRandomForest`) | Production Ready | Missing | Exposed (Experimental) | `tests/test_modeling.py` | Partition-based parallel ensemble |
| **Predictive ML** | Optuna Bayesian Tuning | `src/start/optimization/tuning.py` | Production Ready | Hidden behind toggle | Exposed as First-Class Option | `tests/test_tuning.py` | TPE Bayesian hyperparameter optimization with pruning |
| **Predictive ML** | Data Split Strategies (K-Fold, Stratified, Time-Series) | `src/start/data/splitters.py` | Production Ready | Fixed K-Fold | Configurable Split Selection | `tests/test_splitters.py` | Temporal and stratified partition preservation |
| **Predictive ML** | Preprocessing: Outlier Mitigation | `src/start/preprocessing/method_options.py` | Production Ready | Fixed IQR | Dropdown (Isolation Forest, IQR, Z-Score, Winsorize, None) | `tests/test_preprocessing.py` | Configurable outlier bounds and detection |
| **Predictive ML** | Preprocessing: Missing Value Imputation | `src/start/preprocessing/method_options.py` | Production Ready | Fixed Median | Dropdown (Median, Mean, KNN, Iterative, Drop, None) | `tests/test_preprocessing.py` | Multivariate and univariate imputation |
| **Predictive ML** | Preprocessing: Feature Scaling | `src/start/preprocessing/method_options.py` | Production Ready | Standard | Dropdown (Standard, MinMax, Robust, None) | `tests/test_preprocessing.py` | Outlier-resistant and unit-variance scalers |
| **Predictive ML** | Preprocessing: Categorical Encoding | `src/start/preprocessing/method_options.py` | Production Ready | OneHot | Dropdown (OneHot, Target, Ordinal, None) | `tests/test_preprocessing.py` | High-cardinality target and ordinal encoders |
| **Predictive ML** | Threshold Optimization | `src/start/evaluation/threshold_optimizer.py` | Production Ready | Partial | Exposed (F1, Youden's J, Cost-Sensitive, Custom) | `tests/test_evaluation.py` | Optimal decision threshold selection under cost matrices |
| **Predictive ML** | Top-5 Global Feature Sensitivity Grid | `src/start/modeling/sensitivity_analysis.py` | Production Ready | Partial 7-point | Full 9-point Grid (-30% to +30%) | `tests/test_sensitivity.py` | Grid: -30%, -20%, -10%, -5%, 0%, +5%, +10%, +20%, +30% |
| **Predictive ML** | Sensitivity Modes (One-At-A-Time vs Parallel Basket) | `src/start/modeling/sensitivity_analysis.py` | Production Ready | OAT only | Dual Mode (OAT & Basket) | `tests/test_sensitivity.py` | Compound multi-feature shock vs isolated marginals |
| **Predictive ML** | SHAP & Permutation Explainability | `src/start/explainability/` | Production Ready | Exposed | Exposed | `tests/test_explainability.py` | TreeSHAP, KernelSHAP, and Permutation Importance |
| **Deep Learning** | Sequence Models (RNN, LSTM, GRU, Bi-LSTM) | `src/start/deep_learning/sequence_models.py` | Production Ready | Schema-only | Active Catalog Option | `tests/test_deep_learning.py` | PyTorch recurrent networks for temporal data |
| **Deep Learning** | Vision Models (CNN Architectures) | `src/start/deep_learning/vision_models.py` | Production Ready | Schema-only | Active Catalog Option | `tests/test_deep_learning.py` | 2D convolutional networks with dropout & pooling |
| **Deep Learning** | Deep Training Diagnostics | `src/start/deep_learning/training.py` | Production Ready | Basic | Loss curves, LR schedules, early stopping, gradient norm | `tests/test_deep_learning.py` | Full convergence telemetry |
| **Fraud & Anomaly** | Anomaly Estimators (IForest, OCSVM, LOF) | `src/start/anomaly/detectors.py` | Production Ready | Partial | Exposed in Anomaly Domain | `tests/test_anomaly.py` | Unsupervised contamination and novelty detection |
| **Fraud & Anomaly** | Autoencoder Anomaly Scoring | `src/start/anomaly/autoencoders.py` | Production Ready | Partial | Exposed in Anomaly Domain | `tests/test_anomaly.py` | Reconstruction loss error thresholds |
| **Fraud & Anomaly** | Cost Matrix & Fraud Loss Optimization | `src/start/anomaly/cost_curves.py` | Production Ready | Hidden | Cost Tradeoff Visualizer | `tests/test_anomaly.py` | Expected financial loss vs friction curve |
| **Recommenders** | Matrix Factorization (MF / SVD) | `src/start/recommender/models.py` | Production Ready | Exposed | Exposed | `tests/test_recommender.py` | Latent factor collaborative filtering |
| **Recommenders** | Neural Collaborative Filtering (NCF) | `src/start/recommender/models.py` | Production Ready | Exposed | Exposed | `tests/test_recommender.py` | Dual MLP + GMF hybrid embedding architecture |
| **Recommenders** | Factorization Machine (FM) | `src/start/recommender/models.py` | Production Ready | Exposed | Exposed | `tests/test_recommender.py` | 2nd-order feature interaction embeddings |
| **Recommenders** | Field-Aware Factorization Machine (FFM) | `src/start/recommender/models.py` (`FieldAwareFactorizationMachineModel`) | Production Ready | Not Available | Dedicated FFM Model | `tests/test_recommender.py` | Real field-aware interactions V_{i, f(j)} * V_{j, f(i)}; distinct from FM |
| **Recommenders** | Ranking Evaluation (NDCG, MAP, HitRate, MRR) | `src/start/recommender/evaluation.py` | Production Ready | Exposed | Exposed | `tests/test_recommender.py` | Top-K cutoff ranking metrics |
| **Quant & Risk** | Scenario Shock Engine | `src/start/finance/scenario_engine.py` | Production Ready | Partial | Configurable Factor Shocks | `tests/test_finance.py` | Historical crash, hypothetical shift, and factor shocks |
| **Quant & Risk** | Hierarchical Risk Parity (HRP) | `src/start/finance/portfolio.py` | Production Ready | Exposed | Exposed | `tests/test_finance.py` | Graph dendrogram covariance clustering |
| **Quant & Risk** | Minimum Variance Portfolio (MinVar) | `src/start/finance/portfolio.py` | Production Ready | Exposed | Exposed | `tests/test_finance.py` | Quadratic programming constrained variance minimization |
| **Quant & Risk** | Equal Risk Contribution (ERC / Risk Parity) | `src/start/finance/portfolio.py` | Production Ready | Exposed | Exposed | `tests/test_finance.py` | Equal marginal risk contribution allocation |
| **Quant & Risk** | Performance Ratios (Sharpe, Sortino, MaxDD) | `src/start/finance/metrics.py` | Production Ready | Exposed | Exposed | `tests/test_finance.py` | Risk-adjusted return analytics |
| **LLM & Agents** | 12-Agent Engineering Committee | `src/start/agents/agent_roster.py` | Production Ready | Partial preview | Multi-Agent Collaborative Session | `tests/test_agents.py` | Specialized personas (Architect, Tuner, Stress, etc.) |
| **LLM & Agents** | OpenAI Backend Wiring (`gpt-5.1`) | `src/start/web/routes_workbench.py` | Production Ready | Review only | Session AI Assistant & Plan Generation | `tests/test_provider_contracts.py` | Strict zero-substitution `gpt-5.1` integration |
| **Dataset Hub** | UCI German Credit (Semantic 20 Features) | `src/start/data/uci_credit.py` | Production Ready | Opaque synthetic | Semantic Features (`credit_amount`, `age`, etc.) | `tests/test_data_hub.py` | Ground-truth dataset with real feature distributions |
| **Dataset Hub** | Anomaly & Fraud Benchmark Datasets | `src/start/data/datasets.py` | Production Ready | Basic | Catalog Selection | `tests/test_data_hub.py` | Real credit transaction anomaly profiles |
| **Governance** | Cryptographic Evidence Records & Receipts | `src/start/governance/` | Production Ready | Validation only | Signed Artifact & Evidence Vault | `tests/test_governance.py` | SHA-256 verifiable manifest sealing & approval receipts |

---

## 2. Unimplemented / Deferred Section

The following capabilities are intentionally designated as **UNIMPLEMENTED / DEFERRED**. In accordance with scientific integrity and governance standards, StART strictly refuses to fabricate or mock these capabilities.

### 2.1. Monte Carlo Value-at-Risk (VaR) & Expected Shortfall (ES) Simulation with Heavy-Tailed Copulas
* **Domain**: Quantitative Finance / Market Risk
* **Current Status**: `DEFERRED`
* **Rationale for Deferral**:
  1. Standard Gaussian Monte Carlo vastly underestimates tail risk during liquidity crises and volatility clustering, creating an illusion of precision without rigorous copula calibration (e.g., Clayton or Student-t copulas).
  2. Numerical convergence of multi-asset high-dimensional heavy-tailed simulation requires high-performance parallel execution (e.g. CUDA/OpenCL or optimized C++/Numba kernels) that is outside the single-box lightweight workbench scope.
  3. Parametric and historical simulation are already well-represented in the Scenario Shock Engine without misleading users on stochastic tail precision.
* **Activation Requirements**:
  - Implementation of a calibrated Student-t / Gumbel copula joint distribution module in `src/start/finance/copulas.py`.
  - Quasi-Monte Carlo Sobol sequence generator with empirical extreme-value tail fitting (GPD via Pickands-Balkema-de Haan theorem).
  - Validation test suite matching Basel III/IV backtesting standards (Kupiec POF test, Christoffersen independence test).

### 2.2. Distributed Cluster Training (Ray / Spark MLlib Backend)
* **Domain**: Distributed Predictive ML & Deep Learning
* **Current Status**: `DEFERRED`
* **Rationale for Deferral**:
  - The local workbench is optimized for single-machine workstation development with multi-threading. Invoking cluster orchestration introduces unmanaged infrastructure dependencies.
* **Activation Requirements**:
  - Dynamic Ray cluster client adapter in `src/start/distributed/ray_runner.py`.
  - Remote object storage bridge for evidence record persistence.

### 2.3. Online / Streaming Drift Detection & Real-Time Feature Stores
* **Domain**: MLOps & Production Monitoring
* **Current Status**: `DEFERRED`
* **Rationale for Deferral**:
  - StART focuses on the engineering, development, experimentation, stress-testing, and governance stages of models. Real-time sub-millisecond streaming telemetry requires continuous Kafka/Flink ingestion pipelines.
* **Activation Requirements**:
  - Streaming sliding-window Kolmogorov-Smirnov / Page-Hinkley test harness in `src/start/monitoring/streaming.py`.

### 2.4. Remote Dataset Ingestion Connectors (Kaggle, OpenML, UCI Remote, Hugging Face)
* **Domain**: Dataset Hub
* **Current Status**: `DEFERRED`
* **Rationale for Deferral**:
  - Workbench runtime operates in a local, single-box environment with 8 canonical local datasets and synthetic generators. Remote scraping and third-party credential management (Kaggle API keys, HuggingFace tokens) are intentionally deferred to ensure zero network failure modes during deterministic test runs.
* **Activation Requirements**:
  - Authenticated ingestion pipelines with checksum caching in `src/start/data/remote_connectors.py`.

### 2.5. Unsupervised Anomaly Detectors & Transaction Graph AML
* **Domain**: Fraud & Anomaly
* **Current Status**: `DEFERRED` (Supervised fraud classification remains fully active)
* **Rationale for Deferral**:
  - While supervised imbalance classifiers (e.g. on `synthetic_aml_imbalanced`) are fully executable via the `fraud_anomaly_aml` workflow, unsupervised anomaly estimators (Isolation Forest, LOF, One-Class SVM, autoencoders) and Graph AML require specialized streaming transaction pipelines and graph-database drivers (e.g. Neo4j / NetworkX) not wired to public execution dispatch.
* **Activation Requirements**:
  - Integration into runtime review pipelines in `src/start/anomaly/pipeline.py`.

---

## 3. Execution Modes Census

| Execution Mode | Backend Flow | AI Provider Role | Output Artifacts | Determinism |
| :--- | :--- | :--- | :--- | :--- |
| **`HYBRID WORKBENCH`** (Default) | AI Plan Generation + Deterministic Engine Execution | `gpt-5.1` synthesizes objectives and plans; engines compute all numbers | Full Evidence Records, Diagnostic Metrics, Explanations, AI Synthesis Report | 100% Deterministic numerical outputs; LLM strictly bounded to text synthesis |
| **`AGENTIC SESSION`** | Multi-Turn 12-Agent Committee Orchestration | Full multi-agent deliberation across Architect, Tuner, Stress Engineer, etc. | Agent deliberation transcript, consensus strategy, execution plan | Agent proposals with deterministic evaluation checkpoints |
| **`DETERMINISTIC RUN`** | Direct Parameterized Pipeline Execution | Bypassed / Offline | Benchmark metrics, parameter tables, evidence hashes | Bit-for-bit reproducible runs |

---

## 4. Architectural Invariant Compliance

1. **Deterministic Science Invariant**: No LLM ever invents, predicts, or interpolates a metric value, SHAP value, or risk number. All numbers are computed by scikit-learn, XGBoost, LightGBM, PyTorch, Optuna, or NumPy.
2. **Capability Manifest Ownership**: The capability manifest is dynamically computed and served by the backend at `GET /api/v1/capability-manifest`, ensuring zero drift between backend capabilities and frontend controls.
3. **Purity of Categorical Boundaries**: Factorization Machines (FM) and Field-Aware Factorization Machines (FFM) are maintained as separate, mathematically distinct model choices.
4. **Fail-Closed Execution**: If an uninstalled or unavailable architecture is requested (e.g., CatBoost), the system immediately raises a descriptive error rather than silently substituting an alternative model.
5. **Requested == Resolved == Executed**: User parameters (hyperparameters, preprocessing scalers/imputers, split strategies, tuning objectives) are directly passed into fitted estimators and recorded in `resolved_configuration`.

---

## 5. Backend Capability Closure (B1–B8 Audit)

In response to Codex's frontend capability harness (`webapp/docs/FRONTEND_CAPABILITY_HARNESS.md`), all 8 backend blockers have been definitively closed (`B1_OPEN = 0` through `B8_OPEN = 0`):

| ID | Capability Area | Final Status | Resolution Summary |
| :--- | :--- | :--- | :--- |
| **B1** | Route Aliases & Three-Mode Execution | **CLOSED** | Exposed `/plan/generate`, `/workflow/run`, and `execution_mode` parameter propagation; committed `CP-AGENTIC` checkpoint in agentic session. |
| **B2** | Predictive Parameter Propagation | **CLOSED** | Extracted and propagated model, hyperparameters, preprocessing, and split strategies into estimators; enforced fail-closed model resolution. |
| **B3** | Truthful Dataset Hub Manifest | **CLOSED** | Registered `recommender_ffm_v1` & `synthetic_aml_imbalanced`; truthfully marked remote connectors as deferred. |
| **B4** | Fraud / Anomaly Workflow Dispatch | **CLOSED** | Registered `fraud_anomaly_aml` workflow; truthfully marked unsupervised anomaly detectors as deferred. |
| **B5** | FFM Recommender Execution | **CLOSED** | Dispatched dedicated `FieldAwareFactorizationMachineModel` ($V_{i, f(j)} \cdot V_{j, f(i)}$) with ranking evaluation and FFM sensitivity. |
| **B6** | Quantitative Parameter Propagation | **CLOSED** | Propagated optimizer (`hrp`, `min_var`, `erc`) and shock scenario parameters into market execution. |
| **B7** | 9-Point Sensitivity Grid | **CLOSED** | Integrated `run_sensitivity_analysis()` 9-point grid (-30% to +30%) in both OAT and Parallel Basket modes into classification pipeline. |
| **B8** | Tuning Strategy & Champion Promotion | **CLOSED** | Supported Optuna/Grid/Random strategies, emitted `objective_metric` in trial events, and promoted fitted champion model into canonical lineage. |

---

## 6. FINAL CODEX CONTRACT CLOSURE (C1–C4)

Following the final frontend binding audit (`webapp/docs/FINAL_FRONTEND_BINDING.md`), all four concrete blockers were verified and closed:

| Code | Defect Description | Status | Concrete Backend Fix |
| :--- | :--- | :--- | :--- |
| **C1** | Preprocessing outlier/encoding missing or leaky | **CLOSED** | Created `start.data.preprocessing.apply_preprocessing_pipeline` with IQR, Z-Score, Winsorize, One-Hot, Ordinal, Target, and Frequency transformers. Fitted strictly on `X_train` with zero test leakage. Excluded `score` and `prediction` leakage columns. |
| **C2** | Tuning HTTP 500 on events/presentation & fallback to Logistic Regression | **CLOSED** | Introduced `start.utils.serializers.sanitize_json_primitives` to convert numpy scalars (`np.int64`, `np.float64`) before Pydantic serialization. Promoted champion architecture and lineage into `tab.model`, `tab.extra["resolved_configuration"]`, and canonical artifact builder. Configured `n_jobs=1` for LightGBM on Darwin to eliminate OpenMP concurrency failures. |
| **C3** | Quantitative scenario shocks ignored | **CLOSED** | Wired scenario selection (`asset_tail_stress`, `factor_macro`, `reverse_stress`) and `shock_magnitude` into `market.extra["resolved_configuration"]` and scenario pricing engine. |
| **C4** | Portfolio optimizer selection ignored | **CLOSED** | Wired `hrp`, `min_var`, and `erc` to solve `solve_min_variance`, `solve_equal_risk_contribution`, and `hrp_weights_and_tree` across `run_portfolio_min_variance_pipeline` (`case_h`), `run_portfolio_erc_pipeline` (`case_i`), and `run_portfolio_hrp_pipeline` (`case_g`). |


