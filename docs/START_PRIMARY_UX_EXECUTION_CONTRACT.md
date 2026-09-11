# StART 2.0 — Primary UX Execution Contract & Truth Census

**Status**: Authoritative Reference  
**Scope**: Primary Composer (`webapp/src/features/composer/Composer.tsx`, `EngineeringControls.tsx`), FastAPI transport schemas (`src/start/web/schemas.py`), Runtime orchestrator (`src/start/runtime/execution.py`, `contexts.py`, `workflows.py`), and Scientific engines.

---

## 1. Primary Composer Control Census

| UI Section | UI Label | Visible Options | Request Field | Backend Schema Field | Backend Consumer | Scientific Engine | Current State | Target State |
|---|---|---|---|---|---|---|---|---|
| **01 Engineering Objective** | Engineering objective | Freeform text area | `goal` | `RunRequest.goal` | `CanonicalExecutionService.execute`, `ReviewPlannerAgent` | Agent planning & metadata | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |
| **02 Execution Mode** | Execution mode | `Deterministic Run`, `Hybrid Workbench`, `Agentic Session` | `execution_mode` / `executionMode` | `RunRequest.execution_mode` | `_execute_run_in_background`, `create_agent_plan` | Orchestration dispatch | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |
| **03 Domain / Workflow** | Domain / executable workflow | 11 capability workflows | `workflow` / `workflowId` | `RunRequest.workflow` | `resolve_workflow`, `CanonicalExecutionService` | Workflows registry | VISIBLE_BUT_PARTIALLY_BOUND | BOUND_AND_EXECUTABLE |
| **04 Dataset Hub - Built-in** | Dataset Hub (Built-in) | Canonical fixtures (`institutional_credit_v1`, `synthetic_aml_imbalanced`, etc.) | `contextId` / `synthetic_profile` | `RunRequest.contextId` | `resolve_context_spec`, `instantiate_context` | Synthetic benchmark generators | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |
| **04 Dataset Hub - Live** | Dataset Hub (Live Providers) | Hugging Face, OpenML, UCI, Kaggle | `contextId`, `parameters.provider`, `parameters.dataset_id` | `RunRequest.contextId`, `RunRequest.parameters` | `resolve_context_spec`, `get_provider_adapter` | Live streaming data adapters | BROKEN (Previously blocked with "deferred connectors" notice) | BOUND_AND_EXECUTABLE |
| **04 Dataset Hub - Local** | Dataset Hub (Local File) | Local CSV, Local Parquet | `parameters.local_path`, `parameters.format` | `RunRequest.parameters` | `LocalCSVProviderAdapter` | Local Arrow/CSV reader | DISPLAY_ONLY | BOUND_AND_EXECUTABLE |
| **Execution Settings** | Seed | Integer (`0` to `4294967295`) | `seed` | `RunRequest.seed` | `instantiate_context`, `resolve_model` | Deterministic RNG | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |
| **Execution Settings** | Trial budget | Integer (`5` to `30`) | `trials` / `parameters.trials` | `RunRequest.parameters.trials` | `run_tuning` | Optuna / Random search | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |
| **Model** | Model Architecture | `xgboost`, `lightgbm`, `random_forest`, `gradient_boosting`, `logistic_regression`, `mlp`, `extra_trees`, `catboost` | `parameters.model` | `RunRequest.parameters["model"]` | `resolve_model`, `apply_preprocessing_pipeline` | Scikit-learn / LightGBM / XGBoost | VISIBLE_BUT_PARTIALLY_BOUND (CatBoost fail-closed, Extra Trees unbound) | BOUND_AND_EXECUTABLE |
| **Preprocessing - Imputation** | Missing values | `mean`, `median`, `most_frequent`, `none` | `parameters.preprocessing.imputation` | `RunRequest.parameters["preprocessing"]["imputation"]` | `apply_preprocessing_pipeline` | `SimpleImputer` (train-only fitting) | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |
| **Preprocessing - Scaling** | Scaling | `none`, `standard`, `minmax`, `robust`, `maxabs` | `parameters.preprocessing.scaler` | `RunRequest.parameters["preprocessing"]["scaler"]` | `apply_preprocessing_pipeline` | `StandardScaler` / `MinMaxScaler` / `RobustScaler` | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |
| **Preprocessing - Outliers** | Outlier mitigation | `none`, `iqr`, `zscore`, `winsorize` | `parameters.preprocessing.outlier` | `RunRequest.parameters["preprocessing"]["outlier"]` | `apply_preprocessing_pipeline` | IQR / Z-score clipping | VISIBLE_BUT_IGNORED (Backend supported, UI had "no applied binding") | BOUND_AND_EXECUTABLE |
| **Preprocessing - Encoding** | Categorical encoding | `none`, `onehot`, `ordinal`, `target`, `frequency` | `parameters.preprocessing.encoding` | `RunRequest.parameters["preprocessing"]["encoding"]` | `apply_preprocessing_pipeline` | `OneHotEncoder` / Smoothed target encoding | VISIBLE_BUT_IGNORED (Backend supported, UI had "no applied binding") | BOUND_AND_EXECUTABLE |
| **Split Design** | Split strategy | `stratified_holdout`, `random_holdout`, `time_series_split` | `parameters.split.strategy` | `RunRequest.parameters["split"]["strategy"]` | `train_test_split` | Scikit-learn model selection | VISIBLE_BUT_PARTIALLY_BOUND (Removed dead K-fold disclaimer) | BOUND_AND_EXECUTABLE |
| **Split Design** | Test fraction | Float (`0.05` to `0.50`) | `parameters.split.test_size` | `RunRequest.parameters["split"]["test_size"]` | `train_test_split` | Scikit-learn model selection | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |
| **Hyperparameters** | Model-specific params | Dynamic fields per `modelSchemas.json` | `parameters.hyperparameters` | `RunRequest.parameters["hyperparameters"]` | `resolve_model(**user_hyper)` | Underlying estimators | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |
| **Tuning** | Search strategy | `optuna_bayesian`, `grid_search`, `random_search` | `parameters.strategy` | `RunRequest.parameters["strategy"]` | `run_tuning` | Optuna / parameter grid | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |
| **Tuning** | Primary metric | `auc_roc`, `pr_auc`, `recall`, `f1` | `parameters.primary_metric` | `RunRequest.parameters["primary_metric"]` | `run_tuning` | Metric scorers | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |
| **Explainability** | Method selection / inspect | `native_gain`, `permutation`, `shap_tree_explainer`, `pdp`, `ice` | `parameters.xai_methods` | `RunRequest.parameters["xai_methods"]` | `CanonicalExecutionService.execute` | TreeSHAP / Permutation / PDP | DISPLAY_ONLY (Authoritative read-only surface) | BOUND_AND_EXECUTABLE |
| **Sensitivity Lab** | Sensitivity mode | `one_at_a_time`, `parallel_basket` | `parameters.sensitivity.mode` | `RunRequest.parameters["sensitivity"]["mode"]` | `run_sensitivity_analysis` | Fixed-model shock grid | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |
| **Recommender** | Algorithm | `matrix_factorization`, `neural_collaborative_filtering`, `factorization_machine`, `field_aware_factorization_machine` | `parameters.algorithm` | `RunRequest.parameters["algorithm"]` | `FieldAwareFactorizationMachineModel` | `start.recommender.models` | BOUND_AND_EXECUTABLE (FFM != FM verified) | BOUND_AND_EXECUTABLE |
| **Quantitative Finance** | Portfolio Optimizer | `hrp`, `min_var`, `erc` | `parameters.optimizer` | `RunRequest.parameters["optimizer"]` | `generate_market_world`, `market.extra` | HRP / Minimum Variance / ERC | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |
| **Quantitative Finance** | Scenario shock | `asset_tail_stress`, `factor_macro`, `reverse_stress`, `correlation_breakdown` | `parameters.scenario` | `RunRequest.parameters["scenario"]` | `market.extra`, Traded risk tests | Gate-6 tail shocks | BOUND_AND_EXECUTABLE | BOUND_AND_EXECUTABLE |

---

## 2. Domain & Executable Workflow Census

| Workflow ID | Display Label | Category | Execution Handler | Compatible Datasets | Supported Models | Bound Controls |
|---|---|---|---|---|---|---|
| `predictive_ml` | Predictive ML | ML | `EngineKind.PREDICTIVE_SUBSET` | `institutional_credit_v1`, Live tabular datasets (`scikit-learn/adult-census-income`, OpenML, UCI) | `xgboost`, `lightgbm`, `random_forest`, `gradient_boosting`, `logistic_regression` | Model, Preprocessing, Split, Hyperparameters, Sensitivity |
| `deep_learning` | Deep Learning Diagnostics | ML | `EngineKind.DEEP_LEARNING_REVIEW` | `deep_learning_v1`, Tabular DL datasets | `mlp` (`TabularMLPClassifier`) | Model, Preprocessing, Hyperparameters |
| `data_diagnostics` | Data Diagnostics | ML | `EngineKind.PREDICTIVE_SUBSET` | All tabular datasets | Canonical benchmark model | Preprocessing, Imbalance, Outlier tests |
| `model_diagnostics`| Model Diagnostics | ML | `EngineKind.PREDICTIVE_SUBSET` | All tabular datasets | Supervised classification candidate | Residuals, Subpopulation discrimination |
| `calibration` | Calibration Refinement | ML | `EngineKind.PREDICTIVE_SUBSET` | All tabular datasets | Probabilistic classifiers | Brier score, ECE calibration |
| `robustness` | Stress Testing & Robustness | ML | `EngineKind.PREDICTIVE_SUBSET` | All tabular datasets | Candidate classifiers | Gaussian perturbations, Feature dropout |
| `explainability` | Explainability Audit | ML | `EngineKind.PREDICTIVE_SUBSET` | All tabular datasets | Tree-based / Linear classifiers | SHAP, Permutation, PDP, ICE |
| `hyperparameter_tuning` | Hyperparameter Tuning | ML | `EngineKind.TUNING` | All tabular datasets | `lightgbm`, `xgboost`, `random_forest`, `logistic_regression` | Search strategy, Trial budget, Primary metric |
| `quantitative_finance` | Quantitative Finance | Quant | `EngineKind.MARKET_SUBSET` | `institutional_market_v1` | `hrp`, `min_var`, `erc` | Optimizer choice, Scenario type, Shock magnitude |
| `recommender_system` | Recommender Systems Validation | Recommender | `EngineKind.RECOMMENDER_REVIEW` | `recommender_ratings_v1`, `recommender_implicit_v1`, `recommender_contextual_v1`, `recommender_ffm_v1` | `mf`, `ncf`, `fm`, `ffm` | Algorithm choice, Latent dimension, Regularization |
| `fraud_anomaly_aml` | Fraud, Anomaly & AML Monitoring | ML | `EngineKind.PREDICTIVE_SUBSET` | `synthetic_aml_imbalanced`, `institutional_credit_v1` | `random_forest_balanced`, `lightgbm_balanced`, `xgboost_weighted` | Imbalance handling, Class weights, PR-AUC metric |
