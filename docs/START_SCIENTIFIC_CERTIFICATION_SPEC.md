# StART Scientific Certification Specification

**Document Version**: 1.0.0  
**Authority**: StART Scientific Validation & Governance Core  
**Scope**: Model Adapters, Architectures, Tuning Engines, Metrics, Explainability (XAI), Sensitivity, Traded/Market Risk, and Portfolio Optimization  

---

## 1. Executive Mission & Dual Data Layers

The primary mission of the StART Scientific Certification Harness is to verify that StART produces **scientifically defensible hard numbers**, reproducible across seeds, resistant to data leakage, and mathematically sound.

To achieve empirical validity without compromising mathematical rigor, all certification evaluations are executed across two complementary data layers:

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                           CERTIFICATION HARNESS                         │
├────────────────────────────────────┬────────────────────────────────────┤
│   LAYER A: GOLDEN / KNOWN-ANSWER   │      LAYER B: REAL EXTERNAL        │
├────────────────────────────────────┼────────────────────────────────────┤
│ • Proves mathematical correctness  │ • Proves real empirical behavior   │
│ • Exact known-answer solutions     │ • Complex distributions & noise    │
│ • Deterministic invariant checks   │ • Missingness, high cardinality    │
│ • Provable zero data leakage       │ • Real multi-asset tail events     │
│ • Bit-for-bit regression control   │ • Live data streaming & caching    │
└────────────────────────────────────┴────────────────────────────────────┘
```

**Zero-Substitution Rule**: Golden data and real external data serve distinct epistemological roles. A golden test cannot substitute for an empirical test on real data, and real data cannot substitute for a known-answer mathematical invariant proof.

---

## 2. Two Execution Policies: Deterministic vs. GPT-4.1

To benchmark autonomous algorithmic configuration against predeclared standards without compromising numeric determinism, every experiment is executed under two separate policies:

### 2.1 DETERMINISTIC_POLICY
- **Nature**: Fixed, predeclared engineering contract.
- **Specification**: Predefined model architecture, deterministic preprocessing pipeline (imputation, scaling, outlier mitigation, categorical encoding), holdout/split protocol, standard hyperparameter search space, and deterministic tuning strategy.
- **Role**: Baseline control representing standard production pipelines.

### 2.2 GPT41_POLICY
- **Nature**: Bounded generative AI planning policy.
- **Provider & Model**: Strict adherence to OpenAI `gpt-4.1` (`GPT41_MODEL = gpt-4.1`).
- **Bounded Planning Scope**: GPT-4.1 is granted authority **only** to choose/propose configuration parameters from within the declared runnable StART capability space:
  1. Estimator / Model selection
  2. Preprocessing pipeline (imputation, scaling, outlier clipping, categorical encoding)
  3. Split protocol (test size, holdout strategy)
  4. Hyperparameter space and bounded ranges
  5. Tuning strategy (`bounded_random_search`, `optuna`, `grid`) and trial budget
  6. Metric selection from supported metrics
  7. Explainability methods to compute
  8. Sensitivity perturbation mode (`one_at_a_time` vs `parallel_basket`)
- **Strict Numeric Invariant**:
  $$\text{GPT-4.1 Numeric Authority} \equiv 0$$
  - GPT-4.1 **never** computes metrics, calculates losses, estimates weights, computes SHAP values, or generates numbers.
  - After the structured plan is validated against Pydantic schema contracts, it is submitted to the **exact same deterministic StART scientific engines** as the deterministic policy.
- **Call Bounds**: Maximum 1 primary planning call per experiment, plus at most 1 bounded correction call if the output fails JSON schema validation. No conversational loops.
- **Application Scope**: Global application preference (`gpt-5.1`) is untouched; `gpt-4.1` is used specifically within the certification harness.

---

## 3. Seed Reproducibility & Statistical Rigor

For all stochastic modeling, training, split, and sampling procedures:
$$\text{SEEDS} = [0, 1, 2, 3, 4]$$

1. **Seed-Level Recording**: Individual run records for each of the 5 seeds must be preserved in full before any aggregation.
2. **Aggregations**:
   - Sample Mean: $\bar{x} = \frac{1}{N}\sum_{i=1}^N x_i$
   - Sample Standard Deviation: $s = \sqrt{\frac{1}{N-1}\sum_{i=1}^N (x_i - \bar{x})^2}$
   - 95% Student-$t$ Confidence Interval: $\bar{x} \pm t_{0.025, N-1} \cdot \frac{s}{\sqrt{N}}$ (for $N=5$, $t_{0.025, 4} \approx 2.7764$)
3. **Partition Parity**: In comparisons between `DETERMINISTIC_POLICY` and `GPT41_POLICY`, identical data train/test index partitions are enforced for each seed unless the experimental question explicitly concerns split generation.

---

## 4. Predeclared Domain Champion Selection

Champion selection criteria must be declared in the experiment contract **before** execution. Post-hoc metric shopping is strictly prohibited.

| Domain | Primary Champion Metric | Required Secondary Metrics | Selection Rationale |
| :--- | :--- | :--- | :--- |
| **Predictive Binary** | **ROC-AUC** | PR-AUC, $F_1$, Precision, Recall, Brier score, Calibration error | Discrimination across all decision thresholds |
| **Fraud & Imbalanced** | **PR-AUC** | ROC-AUC, Recall at fixed FDR, Precision, Matthews Correlation | Robustness to extreme class rarity (avoids true-negative inflation) |
| **Recommender Systems** | **NDCG@10** | MAP@10, MRR, HitRate@10, Item Coverage | Top-$K$ position-discounted ranking quality |
| **Deep Learning** | Task-specific (ROC-AUC / MSE) | Latency (ms/sample), Parameter count, Training loss convergence | Alignment with neural architecture task objective |
| **Portfolio Optimization** | Multi-Objective Comparison (No generic scalar) | Realized Volatility, Sharpe Ratio, Sortino Ratio, Diversification Ratio, Turnover, Max Drawdown, Risk Contributions | Risk-based asset allocation balances risk parity vs minimum variance objectives |
| **Market Risk** | **Kupiec POF $p$-value** | Christoffersen Independence, Basel Traffic Light zone, VaR exception count | Statistical validity of tail loss quantile estimation |
| **Scenario & Traded Risk** | Known-Answer Error / Invariant Satisfaction | Shock propagation fidelity, Repricing consistency, Monotonicity | Mathematical adherence to regulatory stress specifications |

---

## 5. Executable Model Matrices & Hard Boundaries

Certification is restricted exclusively to genuinely executable engines:

```text
PREDICTIVE:
  • XGBoost (XGBClassifier)
  • LightGBM (LGBMClassifier, n_jobs=1 on Darwin)
  • Random Forest (RandomForestClassifier)
  • Gradient Boosting (GradientBoostingClassifier)
  • Logistic Regression (LogisticRegression baseline)
  • CatBoost (Fail-closed ValueError if uninstalled)

DEEP LEARNING:
  • Tabular PyTorch MLP (TabularDLClassifier)
  • Sequence DL (LSTM, GRU)
  • Vision CNN (SimpleCNN)

FRAUD / AML:
  • Supervised Imbalance Classifier (Class-weighted / SMOTE)
  • Deferred: Isolation Forest, LOF, One-Class SVM, Graph AML

RECOMMENDER SYSTEMS:
  • Field-Aware Factorization Machine (FFM: V_{i,f(j)} · V_{j,f(i)})
  • Neural Collaborative Filtering (NCF)
  • Matrix Factorization (MF)
  • Factorization Machine (FM)
  [CRITICAL: FFM != FM; explicit field interaction verification]

PORTFOLIO OPTIMIZATION:
  • Hierarchical Risk Parity (HRP)
  • Minimum Variance (Min-Var)
  • Equal Risk Contribution (ERC / Risk Parity)

MARKET RISK:
  • Parametric Gaussian VaR
  • Historical Simulation VaR
  • Cornish-Fisher Modified VaR
  • GARCH Volatility Estimator

SCENARIO & TRADED RISK:
  • Gate-6/6A Asset Tail Shocks
  • Macro Factor Scenario Projection
  • Reverse Stress Testing (Mahalanobis minimum-distance shock)
```

---

## 6. Explainability (XAI) Certification Contract

Every explanation output must be audited for structural and mathematical validity:
1. **Model & Estimator Binding**: Attributions must be generated from the fitted champion estimator, not an uncalibrated surrogate.
2. **Feature Integrity**: Feature names in explanation tables must match the exact feature set from the preprocessing output in correct order.
3. **Dimensionality**: Attribution vector dimension must equal the number of active features ($d$).
4. **Finiteness**: Zero `NaN`, `Infinity`, or null attributions allowed.
5. **Methods Certified**:
   - **Native Gini / Gain Importance** (Tree models)
   - **Permutation Feature Importance** (Model-agnostic)
   - **SHAP Kernel / Tree Attributions** (Additive efficiency: $\sum \phi_i = f(x) - E[f(x)]$)
   - **Partial Dependence Profiles (PDP)** (Average marginal effects)

---

## 7. Sensitivity Certification Contract

Sensitivity engines must verify:
1. **Grid Completeness**: Exact 9-point percentage shock grid:
   $$\text{SHOCK\_GRID} = [-0.30, -0.20, -0.10, -0.05, 0.00, +0.05, +0.10, +0.20, +0.30]$$
2. **Top-5 Global Features**: Shocks applied only to the top 5 globally influential features.
3. **Perturbation Modes**:
   - `one_at_a_time`: Single feature shocked while holding others constant.
   - `parallel_basket`: Compound basket shock applied simultaneously across all top features.
4. **Baseline Invariant**: Shock $0.0$ must match the baseline model score bit-for-bit ($\Delta = 0.0000$).
5. **Type Safety**: Percentage shocks are applied only to continuous/numeric variables; categorical/signed identifiers are protected.

---

## 8. Output Certification Bundle Schema

All artifacts and execution records are emitted to `scratch/scientific_certification/`:

- `certification_manifest.json`: Top-level metadata, system version, environment, pass/fail status.
- `dataset_manifest.json`: Complete catalog of golden fixtures and external datasets with fingerprints.
- `experiment_matrix.json`: Detailed specification of all planned experiments across domains.
- `deterministic_runs.jsonl`: JSON Lines of all seed-level deterministic policy runs.
- `gpt41_runs.jsonl`: JSON Lines of all seed-level GPT-4.1 policy runs.
- `champion_challenger.json`: Champion vs challenger tables and statistical comparisons.
- `invariant_results.json`: Mathematical invariant verification results.
- `xai_results.json`: XAI structural integrity and attribution audits.
- `sensitivity_results.json`: 9-point sensitivity tables and baseline consistency results.
- `provider_trace.json`: LLM provider call traces, prompts, token counts, and latency.
- `failures.json`: Any observed scientific or runtime failures (truthful, un-sanitized).
