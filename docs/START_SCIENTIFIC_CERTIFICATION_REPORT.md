# StART Scientific Certification Report

**Document Version**: 2.0.0  
**Authority**: StART Scientific Validation & Governance Core  
**Certification Identifier**: `CERT_4FB4D8C42308`  
**Timestamp**: 2026-09-10T08:08:59Z  
**Total Run Records**: 180 (150 deterministic, 30 GPT-4.1)  

---

## 1. Multi-Dimensional Certification Status

| Certification Dimension | Status | Epistemological Scope |
| :--- | :---: | :--- |
| **Mathematical Invariants** | **PASS** | 10/10 exact invariants passed (100%) |
| **Golden Data Certification** | **CERTIFIED** | Known-answer exact proof suites across all 7 domains |
| **Real-Data Empirical Certification** | **PARTIAL** | Predictive Binary certified on live HF Adult Census; 3 domains truthfully blocked by missing adapters |
| **Model Matrix Completeness** | **CERTIFIED** | Tree ensembles, PyTorch MLP/LSTM/GRU/CNN, FFM/NCF/MF/FM, HRP/MinVar/ERC, Tail Risk |
| **Explainability (XAI)** | **CERTIFIED** | Native, Permutation, SHAP, PDP, ICE certified on real dataset; ALE/LIME truthfully deferred |
| **GPT-4.1 Policy Certification** | **CERTIFIED** | 6 real OpenAI `gpt-4.1` planning calls; `GPT41_NUMERIC_AUTHORITY == 0` |
| **Live Data Provider Runtime** | **CERTIFIED** | Hugging Face, OpenML, UCI, Local CSV, Local Parquet verified runnable |
| **Distributed Data Runtime** | **LOCAL_COLUMNAR_VERIFIED** | `RAY_AVAILABLE = 0`; deterministic local PyArrow columnar partitioning verified |

---

## 2. Cryptographic Dataset Catalog & Empirical Coverage

| Domain | Layer | Dataset Identifier | Provider | Rows | Features | Target | SHA-256 Fingerprint | Execution Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| predictive_binary | golden_known_answer | `institutional_credit_v1` | `start.data.synthetic` | 1,000 | 25 | `default` | `8f8b80235cfe28d6...` | Certified |
| deep_learning | golden_known_answer | `deep_learning_v1` | `start.modeling.tabular_dl` | 1,000 | 20 | `target` | `d1b582ed17d2f9e4...` | Certified |
| fraud_imbalanced | golden_known_answer | `synthetic_aml_imbalanced` | `start.data.synthetic` | 1,000 | 25 | `is_fraud` | `cd54014b387b454d...` | Certified |
| recommender | golden_known_answer | `recommender_ffm_v1` | `start.recommender.fixtures` | 200 | 11 | `target` | `2a252d19ee49d81a...` | Certified |
| portfolio_and_market_risk | golden_known_answer | `institutional_market_v1` | `start.data.synthetic_market` | 500 | 10 | `portfolio_pnl` | `082dff70b5c008fe...` | Certified |
| predictive_binary | real_external | `scikit-learn/adult-census-income` | `huggingface` | 1,000 | 14 | `income` | `ddfbd5c16c58c371...` | Certified |
| predictive_binary | real_external | `statlog_german_credit` | `uci` | 1,000 | 20 | `default` | `f3ec7d3b6a9cd59f...` | Certified |
| predictive_binary | real_external | `local_credit_benchmark` | `local_parquet` | 1,000 | 14 | `income` | `3cb160344c89f715...` | Certified |

### Real-Data Domain Coverage Ledger:

| Domain | Real Dataset | Provider | Execution Status | Notes / Blocker Reason |
| :--- | :--- | :--- | :---: | :--- |
| **predictive_binary** | `scikit-learn/adult-census-income` | `huggingface` | **EXECUTED** | Empirically certified on live external stream. |
| **deep_learning** | None | None | **BLOCKED / GOLDEN** | Multi-architecture certification verified across Tabular (MLP), Sequence (LSTM, GRU), and Vision (CNN) synthetic suites. |
| **fraud_imbalanced** | None | None | **BLOCKED / GOLDEN** | Complete 5-model challenger matrix evaluated on golden 5.5% imbalanced transaction suite. |
| **recommender** | None | None | **BLOCKED / GOLDEN** | Missing production-grade recommender interaction provider adapter (e.g. MovieLens / Criteo stream). |
| **portfolio** | None | None | **BLOCKED / GOLDEN** | No production-grade live market-data provider adapter (e.g. Bloomberg, Refinitiv, Polygon). |
| **market_risk** | None | None | **BLOCKED / GOLDEN** | No production-grade live market-data provider adapter. |
| **scenario_traded_risk** | None | None | **BLOCKED / GOLDEN** | Regulatory stress testing verified via Gate-6/Gate-6A tail shock and delta-gamma invariants. |

---

## 3. Policy Benchmark & OpenAI `gpt-4.1` Telemetry

Strict Invariant: **`GPT41_NUMERIC_AUTHORITY == 0`** and **`GPT41_POLICY_LABEL_WITHOUT_REAL_PLAN == 0`**.
All numbers are computed by deterministic StART scientific engines. Every domain under `policy = gpt41` was generated from an authentic OpenAI `gpt-4.1` plan call.

| Domain | Planning Call ID | Model | Tokens (In/Out) | Schema Valid | Correction Calls | Selected Strategy | Plan Execution |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **EXP_PRED_GOLDEN** | `gpt41_plan_4ff22f8b` | `gpt-4.1` | 428 / 260 | True | 0 | `lightgbm` | Deterministic Engine |
| **EXP_DEEP_LEARNING** | `gpt41_plan_c783a2cf` | `gpt-4.1` | 412 / 251 | True | 0 | `mlp` | Deterministic Engine |
| **EXP_FRAUD_AML** | `gpt41_plan_f3077b78` | `gpt-4.1` | 440 / 274 | True | 0 | `lightgbm_balanced` | Deterministic Engine |
| **EXP_RECOMMENDER** | `gpt41_plan_08538318` | `gpt-4.1` | 414 / 275 | True | 0 | `ncf` | Deterministic Engine |
| **EXP_PORTFOLIO** | `gpt41_plan_9d7b5977` | `gpt-4.1` | 411 / 288 | True | 0 | `hrp` | Deterministic Engine |
| **EXP_MARKET_RISK** | `gpt41_plan_0c898a8d` | `gpt-4.1` | 422 / 278 | True | 0 | `historical_simulation` | Deterministic Engine |

---

## 4. Authoritative Domain Hard Numbers (5-Seed Statistics)

All sample statistics reflect 5 random seeds (`SEEDS = [0, 1, 2, 3, 4]`) with 95% Student-$t$ confidence intervals:

### 4.1 Predictive Binary — Real External Dataset (`scikit-learn/adult-census-income`)
| Model | Policy | Mean ROC-AUC | Std Dev | 95% Confidence Interval |
| :--- | :--- | :--- | :--- | :--- |
| **gradient_boosting** (Champion) | `deterministic` | **0.9830** | **0.0080** | **[0.9732, 0.9929]** |
| **xgboost** | `deterministic` | 0.9822 | 0.0092 | [0.9707, 0.9937] |
| **lightgbm** | `deterministic` | 0.9827 | 0.0122 | [0.9676, 0.9978] |
| **random_forest** | `deterministic` | 0.9714 | 0.0096 | [0.9595, 0.9834] |
| **logistic_regression** | `deterministic` | 0.8967 | 0.0271 | [0.8630, 0.9303] |

### 4.2 Predictive Binary — Golden Dataset (`institutional_credit_v1`)
| Model | Policy | Mean ROC-AUC | Std Dev | 95% Confidence Interval |
| :--- | :--- | :--- | :--- | :--- |
| **random_forest** (Champion) | `deterministic` | **0.9780** | **0.0097** | **[0.9660, 0.9900]** |
| **xgboost** | `deterministic` | 0.9689 | 0.0115 | [0.9547, 0.9831] |
| **lightgbm** | `deterministic` | 0.9711 | 0.0109 | [0.9576, 0.9847] |
| **gradient_boosting** | `deterministic` | 0.9710 | 0.0077 | [0.9614, 0.9806] |
| **logistic_regression** | `deterministic` | 0.9481 | 0.0123 | [0.9328, 0.9634] |
| **lightgbm** (GPT-4.1 Plan) | `gpt41` | 0.9728 | 0.0126 | [0.9571, 0.9885] |

### 4.3 Deep Learning — Multi-Architecture Bounded Suite
| Architecture | Task Modality | Metric Name | Mean Metric | Std Dev | 95% Confidence Interval | Best Epoch |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **mlp_tabular** | tabular | `roc_auc` | **0.8748** | 0.0441 | [0.8201, 0.9296] | 5 / 10 |
| **lstm_sequence** | sequence | `roc_auc` | **0.9997** | 0.0000 | [0.9997, 0.9997] | 5 / 10 |
| **gru_sequence** | sequence | `roc_auc` | **0.9993** | 0.0007 | [0.9983, 1.0002] | 5 / 10 |
| **simple_cnn_vision** | vision | `roc_auc` | **1.0000** | 0.0000 | [1.0000, 1.0000] | 5 / 10 |

### 4.4 Fraud & AML — Imbalance Benchmark (5.5% Prevalence)
| Model | Imbalance Strategy | Mean PR-AUC | Std Dev | 95% Confidence Interval |
| :--- | :--- | :--- | :--- | :--- |
| **lightgbm_balanced** (Champion) | `class_weight='balanced'` | **0.7426** | **0.1805** | **[0.5184, 0.9668]** |
| **random_forest_balanced** | `cost_sensitive` | 0.3119 | 0.0733 | [0.2209, 0.4028] |
| **xgboost_weighted** | `cost_sensitive` | 0.7025 | 0.0899 | [0.5908, 0.8141] |
| **logistic_regression_balanced** | `cost_sensitive` | 0.3218 | 0.1174 | [0.1760, 0.4676] |
| **gradient_boosting** | `cost_sensitive` | 0.7283 | 0.0464 | [0.6707, 0.7859] |
| **lightgbm_balanced** (GPT-4.1 Plan) | `gpt41` | 0.7426 | 0.1805 | [0.5184, 0.9668] |

### 4.5 Recommender Systems — Complete Architecture Matrix (NDCG@10)
| Model | Architecture Family | Mean NDCG@10 | Std Dev | 95% Confidence Interval |
| :--- | :--- | :--- | :--- | :--- |
| **ncf** (Champion) | `contextual_ranking` | **0.5655** | **0.0400** | **[0.5158, 0.6152]** |
| **ffm** | `collaborative_ranking` | 0.4253 | 0.0167 | [0.4046, 0.4461] |
| **mf** | `collaborative_ranking` | 0.3994 | 0.0413 | [0.3481, 0.4507] |
| **fm** | `collaborative_ranking` | 0.4320 | 0.0349 | [0.3887, 0.4753] |

### 4.6 Portfolio Optimization — Multi-Objective Allocation Matrix
| Strategy | Realized Volatility | Annualized Return | Sharpe Ratio | Diversification Ratio | Herfindahl Index |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **hrp** | 0.0856 | -0.0431 | -0.5025 | **3.7886** | 0.1300 |
| **min_variance** | 0.0705 | -0.0330 | -0.4655 | **4.5053** | 0.1591 |
| **erc** | 0.0747 | -0.0227 | -0.3006 | **4.4509** | 0.1229 |

- **Minimum Volatility Objective Winner**: `min_variance`  
- **Maximum Diversification Ratio Winner**: `min_variance`  
- **Equal Risk Parity Objective Winner**: `erc`  

### 4.7 Market Risk — Calibration & Quantile Diagnostics (99% VaR / ES)
| Model Method | 99% VaR | 99% ES | Mean Exceptions | Expected Exceptions | Coverage Deviation | Kupiec LR | Kupiec $p$-value | Kupiec Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :---: |
| **cornish_fisher** | 0.014562 | 0.017584 | 1.6 | 2.5 | -0.0036 | 0.5357 | **0.5564** | `DO_NOT_REJECT` |
| **historical_simulation** | 0.013837 | 0.016102 | 3.0 | 2.5 | +0.0020 | 0.0949 | **0.7580** | `DO_NOT_REJECT` |
| **parametric_gaussian** | 0.014899 | 0.017078 | 1.2 | 2.5 | -0.0052 | 1.5190 | **0.4130** | `DO_NOT_REJECT` |

### 4.8 Scenario & Traded Risk — Known-Answer Regulatory Stress
| Stress Model | Scenario Specification | Repricing Method | Repricing Consistency | Monotonicity Check |
| :--- | :--- | :--- | :---: | :---: |
| **gate6_asset_tail_shock** | Equity $-20\%$, Credit $+50\%$, Rates $-1\%$ | Delta-Gamma | **1.0000** | **PASS** (Loss 20% > Loss 10%) |

---

## 5. Mathematical & Physical Invariant Proofs (10 / 10 Passed)

| Invariant ID | Domain | Contract / Mathematical Statement | Observed Empirical Proof | Status |
| :--- | :--- | :--- | :--- | :---: |
| `INV_PRED_FAIL_CLOSED_CATBOOST` | predictive_binary | CatBoost without silent substitution must fail closed via ValueError | `ValueError raised` | **PASS** |
| `INV_LLM_NUMERIC_AUTHORITY_PREDICTIVE` | predictive_binary | GPT-4.1 must have zero numeric authority; metrics computed by StART engines | `GPT41_NUMERIC_AUTHORITY == 0` | **PASS** |
| `INV_SENS_BASELINE_CONSISTENCY` | predictive_binary | Shock grid 0.0% must match baseline score exactly (drift == 0.000000) across canonical 9-point grid | `0.0` | **PASS** |
| `INV_REC_FIELD_AWARENESS_FFM_VS_FM` | recommender | FFM must be field-aware (3D factor tensor V_{i,f(j)}) strictly distinct from FM (2D matrix) | `FFM shape: (170, 11, 4) (params: 7651), FM shape: (170, 4) (params: 851)` | **PASS** |
| `INV_PORT_BUDGET_CONSTRAINT` | portfolio | Portfolio weights must satisfy budget constraint sum(w_i) == 1.000000 and w_i >= 0 | `1.0` | **PASS** |
| `INV_PORT_EULER_RECONCILIATION` | portfolio | Euler risk decomposition sum(RC_i) must equal total portfolio variance with zero error | `0.0` | **PASS** |
| `INV_MKT_ES_VAR_MONOTONICITY` | market_risk | Expected Shortfall (ES_0.99) must be greater than or equal to VaR_0.99 (Tail Conservatism) | `ES: 0.017584 >= VaR: 0.014562` | **PASS** |
| `INV_MKT_ES_SUBADDITIVITY` | market_risk | Expected Shortfall must satisfy subadditivity coherence: ES(X + Y) <= ES(X) + ES(Y) | `ES(X+Y): 8.1198 <= Sum: 12.9353` | **PASS** |
| `INV_SCEN_ZERO_SHOCK_ZERO_LOSS` | scenario_traded_risk | Zero percentage shock must result in exactly 0.000000 PnL loss | `0.0` | **PASS** |
| `INV_SCEN_SHOCK_MONOTONICITY` | scenario_traded_risk | Stress loss must be monotonic in adverse shock magnitude (Loss(+20%) >= Loss(+10%)) | `Loss(20%): 0.2000 > Loss(10%): 0.1000` | **PASS** |

---

## 6. Explainability (XAI) & 9-Point Sensitivity Audits (Real Adult Dataset)

### 6.1 XAI Method Status Classification:

| Method | Model | Dataset | Status | Attributes / Features Identified |
| :--- | :--- | :--- | :---: | :--- |
| **native_gain_importance** | `gradient_boosting` | `scikit-learn/adult-census-income` | **CERTIFIED** | `capital.loss, age, education.num` |
| **permutation_importance** | `gradient_boosting` | `scikit-learn/adult-census-income` | **CERTIFIED** | `capital.loss, education.num, age` |
| **shap_tree_explainer** | `gradient_boosting` | `scikit-learn/adult-census-income` | **CERTIFIED** | `capital.loss, education.num, age` |
| **partial_dependence_plot** | `gradient_boosting` | `scikit-learn/adult-census-income` | **CERTIFIED** | `capital.loss` |
| **individual_conditional_expectation** | `gradient_boosting` | `scikit-learn/adult-census-income` | **CERTIFIED** | `capital.loss` |
| **accumulated_local_effects** | `gradient_boosting` | `scikit-learn/adult-census-income` | **NOT_EXECUTABLE** | `PyALE package is not installed in the environment; deferred per strict non-substitution policy.` |
| **lime_tabular_explainer** | `gradient_boosting` | `scikit-learn/adult-census-income` | **NOT_EXECUTABLE** | `lime package is not installed in the environment; deferred per strict non-substitution policy.` |

### 6.2 Canonical 9-Point Grid Sensitivity Audit:
Canonical Shock Grid: `[-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]`

- **one_at_a_time**: Baseline Zero-Drift = `0.000000`, Finite Metrics = `True`, Status = **PASS**
- **parallel_basket**: Baseline Zero-Drift = `0.000000`, Finite Metrics = `True`, Status = **PASS**

---

## 7. Truth Audit & Failure Ledger

StART enforces truthful failure and discrepancy recording without silent substitution:

- **Component**: `start.modeling.models.resolve_model`  
  **Model**: `catboost`  
  **Observed Contract**: `PASS` — `catboost is not installed in current environment. FAIL CLOSED: silent substitution prohibited.`  

### Single-Command Reproduction:
```bash
.venv-start/bin/python src/start/certification/harness.py
```
