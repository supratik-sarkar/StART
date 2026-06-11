# StART — Standardized Agentic Reusable Tests

> **StART is a standardized agentic reusable testing framework for AI/ML evaluation. It combines deterministic quantitative validation engines with agent-assisted orchestration, evidence generation, adaptive compute routing, and reviewer-ready reporting.**

[![ci](https://github.com/supratik-sarkar/StART/actions/workflows/ci.yml/badge.svg)](https://github.com/supratik-sarkar/StART/actions)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.12-blue)
![license](https://img.shields.io/badge/license-Apache--2.0-green)
![platform](https://img.shields.io/badge/runs%20on-CPU%20%7C%20MPS%20%7C%20CUDA%20%7C%20Databricks-orange)

StART targets model review and model-risk evaluation across data preprocessing, supervised and unsupervised ML, propensity models, recommender systems, portfolio optimization, performance attribution, deep learning, global/local/deep XAI, and emerging GenAI/agentic systems.

**Who this is for:** open-source contributors, model validation practitioners, MLOps engineers, AI governance teams, Databricks users, and local Python users. The framework runs **fully offline on a laptop** — no GPU, no Databricks, no MLFlow, no API keys, no LLM provider required.

---

## Table of contents

1. [The core idea](#the-core-idea)
2. [Why this design](#why-this-design)
3. [Requirements](#requirements)
4. [Installation (step by step)](#installation-step-by-step)
5. [Environment verification](#environment-verification)
6. [Flagship demo: propensity model review](#flagship-demo-propensity-model-review)
7. [Bring your own data](#bring-your-own-data)
8. [Model & validation recommendations](#model--validation-recommendations)
9. [Agentic governance: challenge & sign-off](#agentic-governance-challenge--sign-off)
10. [Minimal offline smoke demo](#minimal-offline-smoke-demo)
11. [Generated artifacts](#generated-artifacts)
12. [Safe degradation](#safe-degradation)
13. [Running on your own data](#running-on-your-own-data)
14. [Adaptive compute routing](#adaptive-compute-routing)
15. [LLM providers](#llm-providers)
16. [Databricks](#databricks)
17. [Extending the registry](#extending-the-registry)
18. [Repository layout](#repository-layout)
19. [Development workflow](#development-workflow)
20. [Troubleshooting](#troubleshooting)
21. [Public-safety statement](#public-safety-statement)
22. [Roadmap](#roadmap)

---

## The core idea

**Agents orchestrate. Deterministic engines compute. Evidence is the product.**

```
                ┌────────────────────────────────────────────────┐
                │                 Agentic layer                  │
                │  ReviewPlanner → PolicyGuard → TestRouter      │
                │  → ExecutionAgent → EvidenceCritic → Narrative │
                └───────────────────────┬────────────────────────┘
                                        │  plans / critiques / narrates
                                        │  (never computes metrics)
                ┌───────────────────────▼────────────────────────┐
                │        Deterministic test registry             │
                │  preprocessing · supervised · xai · genai · …  │
                └───────────────────────┬────────────────────────┘
                                        │  typed TestResult
                ┌───────────────────────▼────────────────────────┐
                │   Evidence layer: content-addressed store +    │
                │   append-only SHA-256–chained JSONL ledger     │
                └────────────────────────────────────────────────┘
```

The LLM/agentic layer **never** invents a number. It plans validation work, routes it to registered deterministic Python engines, verifies evidence completeness, and writes reviewer narratives **from** the evidence.

## Why this design

**Deterministic test registry.** Every quantitative check is a pure Python function registered via `@register_test(...)`, with explicit parameters, seeds, thresholds, and declared limitations. Same data + same parameters + same policy ⇒ same numbers, same status. CI verifies determinism claims (e.g., row-order invariance) with property-based tests (Hypothesis).

**Evidence layer.** Every test produces a typed `EvidenceRecord`: evidence/test/model/dataset/run IDs, timestamp, parameters, metrics, thresholds, pass/warn/fail/error/skipped status, interpretation, limitations, **input-data hash**, **policy hash**, git SHA, and reproducibility metadata (seed, device, package versions).

**Tamper-evident ledger.** Records are canonicalized, SHA-256 hashed, written to a content-addressed store, and appended to a hash-chained JSONL ledger (`entry_hash_n = sha256(entry_hash_{n-1} + record_hash_n)`). Any retroactive edit breaks the chain; `start doctor` verifies it. Content addressing also enables result caching keyed by `(test, data hash, params, policy)`.

**Proof-carrying narratives.** Every quantitative claim in a reviewer narrative must carry an inline evidence citation like `[EV-8535b74e2121]`. The **EvidenceCriticAgent** blocks narratives containing uncited quantitative claims or citations to nonexistent evidence. In no-LLM mode, narratives come from a deterministic template that is proof-carrying *by construction* — so the guarantee holds with zero LLM access.

**Policy hashing.** Thresholds and validation regimes live in versioned YAML (`configs/policy/`). The policy file's content hash is stamped into every evidence record, so a reviewer can prove *which threshold regime* produced a verdict.

**Plugin architecture.** New tests register via the `@register_test` decorator in-repo, or ship as external pip packages exposing a `start.test_packs` entry point — no core changes required.

---

## Requirements

| Requirement | Detail |
|---|---|
| Python | **3.12 recommended** (the version this project is developed and verified on); 3.10 minimum |
| OS | macOS (Apple Silicon and Intel), Linux, Windows/WSL |
| Package manager | `pip` (editable installs via `pip install -e`) |
| Build backend | `hatchling` (handled automatically by pip) |
| GPU | **Optional.** CUDA and Apple Silicon MPS are used when present; CPU works everywhere |
| Databricks / MLFlow / LLM keys | **Optional.** Everything degrades to local, deterministic execution |

Core dependencies are intentionally light: `numpy`, `pandas`, `scipy`, `scikit-learn`, `pydantic`, `pydantic-settings`, `pyyaml`, `typer`, `rich`. Heavy stacks live in extras: `[tree-models]` (xgboost, lightgbm), `[optuna]` (Bayesian tuning), `[xai]` (shap), `[formats]` (pyarrow for Parquet/Feather), `[delta]` (local Delta tables), `[snowflake]` (warehouse connector), `[torch]` (torch + captum, for the DL roadmap), `[mlflow]`, `[openai]`, `[anthropic]`, `[huggingface]`, `[genai]`, `[dev]`, `[all]`.

### A note on the src layout

The importable package lives at **`src/start/`**, not at the repository root. This is deliberate:

- It prevents accidentally importing the package from the working directory instead of the installed (editable) version — a classic source of "works on my machine" bugs.
- It avoids any name collision between the `start` package and a virtual-environment directory created inside the repo.

You never import from `src.start`; after installation you simply `import start`.

---

## Installation (step by step)

All commands below are run from Terminal. Replace paths as needed.

```bash
# 1. Get the code
git clone https://github.com/supratik-sarkar/StART.git
cd StART

# 2. Confirm your Python version (3.12 recommended, 3.10 minimum)
python3.12 --version

# 3. Create and activate a virtual environment.
#    Recommended name: .venv-start  (kept out of git via .gitignore)
python3.12 -m venv .venv-start
source .venv-start/bin/activate          # Windows: .venv-start\Scripts\activate

# Your prompt should now show: (.venv-start)

# 4. Upgrade pip and install StART in editable mode with dev tools
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# 5. Sanity-check the import and the CLI
python -c "import start; print(start.__version__)"
start --help
```

**Why editable (`-e`)?** Your source edits under `src/start/` take effect immediately without reinstalling. This is the expected development workflow.

**Pinned installs (optional).** For a clean-room or CI-matching environment:

```bash
python -m pip install -e ".[dev]" -c constraints.txt
```

---

## Environment verification

Run these two commands after every fresh install. They are also the first thing to run when anything misbehaves.

### `start doctor`

```bash
start doctor --config configs/default.yaml
```

`doctor` validates, in one table:

- **Detected device (CUDA→MPS→CPU)** — which accelerator the compute router found. On an Apple Silicon Mac with PyTorch installed this should read `mps`; without PyTorch it reads `cpu` (that is correct, not an error — torch is optional).
- **Databricks runtime** — whether `DATABRICKS_RUNTIME_VERSION` is present. Independent of device detection.
- **MLFlow importable** — whether MLFlow-backed experiment tracking is available.
- **LLM provider availability** — for each provider (`none`, `openai`, `anthropic`, `grok`, `huggingface`, `hf_local`, `enterprise_llm_gateway`), whether its SDK is installed **and** its API key is present. All `False` is a perfectly valid, fully supported state.
- **Registered test families / tests** — confirms the registry loaded (18 deterministic tests across `preprocessing`, `supervised`, `xai`, `genai`).
- **Config valid / Policy hash** — your YAML config parses and the active policy's content hash (the one stamped into evidence) is shown.
- **Ledger integrity** — if a ledger exists, recomputes the entire hash chain and reports `True`/`False`.

### `start list-tests`

```bash
start list-tests                    # all registered tests, as plain JSON
start list-tests --family supervised
```

This validates that the deterministic test registry imported correctly and shows exactly which checks the planner can schedule (test ID, family, name, description). Output is plain JSON on stdout, so it pipes cleanly:

```bash
start list-tests | python -c "import sys,json; print(len(json.load(sys.stdin)), 'tests')"
# -> 18 tests
```

### Test suite

```bash
pytest
# -> 64 passed
```

The suite includes ledger tamper detection, citation-gate enforcement, compute-routing degradation, and Hypothesis-based determinism properties.

---

## Flagship demo: propensity model review

> The main demo is no longer a toy logistic regression. The flagship workflow is a realistic **propensity-style model review** — framed as a client-attrition case on public sklearn data — covering the full arc a model validator actually walks: data checks → model + tuning choice → cohort metrics → explainability → sensitivity → evidence.

```bash
start propensity-demo                    # interactive: prompts for every choice
start propensity-demo --non-interactive  # safe defaults, zero prompts
python examples/propensity_interactive.py            # same workflow as a script
python examples/propensity_interactive.py --non-interactive
```

Non-interactive flags:

```bash
start propensity-demo --non-interactive --model random_forest --tuning none
start propensity-demo --non-interactive --model random_forest --tuning random --cv 3
start propensity-demo --non-interactive --model xgboost --tuning optuna --cv 5 --cohort oos
```

**What the workflow does, step by step:**

1. **Dataset.** Loads sklearn's public breast-cancer dataset reframed as a generic attrition/propensity case (~37% event rate; synthetic `make_classification` fallback if needed). No client data anywhere.
2. **Split.** Stratified **60% train / 20% test / 20% out-of-sample (OOS)**.
3. **Feature engineering checks.** Missingness, duplicate rows, constant/near-constant features, high-cardinality categorical scan, numeric range summary, outlier rate, train/test drift (PSI + KS), target-leakage screen, and split diagnostics — all as evidence records.
4. **Model choice.** Random Forest (always works, sklearn-only), XGBoost, or LightGBM. The optional backends are **not** core dependencies: if `xgboost`/`lightgbm` is missing, you get a clean message and a Random Forest fallback (`pip install -e ".[tree-models]"` to enable them).
5. **Tuning choice.** `none` (default model), grid search, random search, or Bayesian optimization via **Optuna** (`pip install -e ".[optuna]"`; skips gracefully with instructions if absent). Each model exposes **five standard hyperparameters** with suggested grids/ranges — in interactive mode you can accept the suggestions or type your own values/ranges. Validation scheme: holdout or **K-fold CV with K=3 (default) or K=5**; grid/random use sklearn CV searchers, the Optuna objective uses the same CV internally.
6. **Metrics table.** The fitted model is scored on **train, test, and OOS**, and compared on **AUC-ROC, Accuracy, Precision, Recall, F1, and top-10% lift** — printed as a table and recorded as a single comparison evidence record including the train-test AUC gap (overfitting check).
7. **Explainability — honest by construction.** SHAP `TreeExplainer` when `shap` is installed (global importance + local attributions for a sample of rows); otherwise **permutation importance, and the evidence says so explicitly** — the method actually used is recorded, local attributions are never fabricated on the fallback path. (Note: Optuna/Hyperopt are tuning tools, not explainability engines; explainability stays model-specific.)
8. **Sensitivity test.** The top-5 features from global importance are shocked **in parallel at −30%, −20%, −10%, 0%, +10%, +20%, +30%**; AUC-ROC and drift vs baseline are tabulated. The 0% row equals the baseline AUC by construction (tested). You choose the evaluation cohort: `test`, `oos`, or `development` (train+test+OOS).
9. **Agentic review.** `ModelRiskFindingAgent` turns cross-evidence patterns into findings (overfitting gap, test-vs-OOS instability, sensitivity profile, threshold breaches), and `TestSuggestionAgent` recommends next steps (enable SHAP, add OOS, run a tuned challenger, explain skipped tests) — every quantitative statement citing its `[EV-…]` record, enforced by the citation gate.
10. **Outputs.** Proof-carrying Markdown report, hash-chained ledger entries, and content-addressed evidence records — same as every StART run.

## Bring your own data

> The public datasets are examples. **Your datasets are the product.** Every workflow runs unchanged on user data through one abstraction — `start.connectors` — with five modes. If you provide only a train source, a stratified 60/20/20 train/test/OOS split is applied automatically. A packaged default policy ships inside the wheel, so runs work from any directory, not just a repo checkout.

**Mode 1 — demo (default).** Public sklearn / synthetic data; exists only to demonstrate functionality.

**Mode 2 — local files** (CSV / Parquet / Feather / Delta directory):

```bash
start propensity-demo --non-interactive --train train.csv --test test.csv --oos oos.csv --target churned
start propensity-demo --non-interactive --train clients.parquet --target churned   # auto 60/20/20 split
start run --config configs/local_no_llm.yaml train.csv --test test.csv --oos oos.csv
```

**Mode 3 — pandas DataFrames (first-class Python API):**

```python
from start.orchestration import review_dataframes

result = review_dataframes(
    train_df, test_df, oos_df,          # or just train_df for an auto split
    target_column="churned",
    model=fitted_model,                  # optional: enables XAI + sensitivity engines
)
print(result.narrative.signoff)
```

**Mode 4 — Spark DataFrames / tables.** Hand over `spark.table(...)`, `spark.sql(...)`, or a table-name string; `SparkDataFrameAdapter` standardizes conversion with a row-limit guard. Used by `notebooks/02_propensity_model_review.py`, which now exposes Databricks **widgets** for data source, model family, tuning, CV strategy, and sensitivity cohort.

**Mode 5 — Snowflake (generic, config-driven).** Coordinates live in config; credentials come from standard `SNOWFLAKE_*` environment variables — never from the repo. Requires the optional driver: `pip install -e ".[snowflake]"`.

```yaml
data:
  source: snowflake          # demo | files | pandas | spark | snowflake
  snowflake:
    database: ANALYTICS
    schema: RETENTION
    table: CLIENT_FEATURES
  timestamp_column: as_of_date     # universal dataset metadata, consumed by
  entity_id_column: client_id      # agents and (future) temporal engines
  dataset_type: auto               # or declare: panel_time_series, limit_order_book, ...
```

(`dataset_id` and the target/score columns stay in their existing config blocks; `data.dataset.train/test/oos` holds file paths or table references for the `files` and `spark` modes.)

## Model & validation recommendations

`start recommend` profiles a dataset deterministically (rows, feature types, time structure, entity structure, target type), classifies its type, and produces type-aware model candidates plus a model/dataset-specific validation plan — each item honestly labeled **available now** vs **roadmap**:

```bash
start recommend clients.csv --target churned
start recommend panel.parquet --target ret_1d --timestamp-col ts --entity-col asset
start recommend book.parquet --target mid_move --dataset-type limit_order_book
```

Domain types that cannot be inferred from columns alone (limit order books, tick event streams, volatility surfaces) are **never auto-claimed** — declare them via `--dataset-type` or `data.dataset_type`. The same logic is importable: `ModelRecommendationAgent` and `ValidationPlannerAgent` consume a `DatasetProfile`, and `route_explainability()` picks model-appropriate explainability (tree → SHAP/permutation today; gradient and attention methods routed for the DL roadmap) so explainability is never hard-tied to one library.

## Agentic governance: challenge & sign-off

Every run now ends with a deterministic governance pass, written into the narrative and report:

- **ChallengeAgent** tries to invalidate the run's own conclusions from its evidence — perfect train separation (memorization risk), drift warnings undermining a test-cohort-only sensitivity, small-sample caveats — every challenge citing its `[EV-…]` records.
- **GovernanceAgent** gates the run: unresolved FAIL/ERROR evidence, skipped planned tests without justification, missing policy stamps, and citation-gate failures all block a clean verdict.
- **SignoffAgent** issues the reviewer-ready conclusion: **READY FOR SIGN-OFF** only when governance is clean; otherwise **NOT READY** with the outstanding items cited. It appears as its own section in the Markdown report.

Combined with `ModelRiskFindingAgent`, `TestSuggestionAgent`, and the `EvidenceCriticAgent` citation gate, the agentic layer is the product: a model-review copilot + validation planner + evidence-governance engine + audit assistant on top of deterministic engines — LLM-assisted only where configured, deterministic everywhere else.

## Minimal offline smoke demo

```bash
python examples/quickstart_local.py
```

A 30-second logistic-regression smoke test of the full pipeline (kept as the fastest possible installation check). Like the flagship demo it needs **no API keys, no LLM provider, no Databricks cluster, no GPU** — it prints per-test statuses, verifies ledger integrity, and writes a report. If it runs clean, your installation is fully functional.

---

## Generated artifacts

Every run writes to the output root (default `start_output/`, configurable):

```text
start_output/
├── ledger.jsonl          # append-only, hash-chained evidence ledger
├── evidence_store/       # content-addressed evidence records (<sha256>.json)
├── reports/              # proof-carrying reviewer reports (RUN-*.md, RUN-*.json)
└── experiments/          # local experiment tracking (JSONL), unless MLFlow is used
```

- **`ledger.jsonl`** — one line per evidence record: `{index, prev_hash, record_hash, entry_hash, record}`. Each `entry_hash` chains to the previous one from a zero genesis hash. Editing **any** historical line breaks verification (`start doctor` recomputes the full chain). The ledger is append-only by contract: nothing in StART ever rewrites it.
- **`evidence_store/`** — each record serialized canonically and stored under its own SHA-256 content hash, plus a cache index. Identical invocations of `(test, input-data hash, params, policy hash)` can be served from cache.
- **`reports/RUN-*.md`** — reviewer-ready Markdown: policy hash and decision, narrative with **inline `[EV-…]` citations on every quantitative claim**, evidence table, critique results, and a reproducibility block (device, runtime, Python, git SHA, seed, input-data hash).

These three artifacts together are the audit trail: the report makes claims, the citations bind claims to records, the ledger proves the records were not altered.

---

## Safe degradation

StART is built to degrade **loudly and visibly**, never silently:

| Missing capability | Behavior |
|---|---|
| No CUDA | Falls back to MPS, then CPU |
| No MPS (or no PyTorch at all) | Falls back to CPU |
| No Databricks runtime | Local execution; Databricks configs still parse |
| No MLFlow | Local JSONL experiment tracking |
| No LLM provider / no API key / unreachable API | `NoLLMProvider` → deterministic-only mode; template narratives (still proof-carrying) |
| Required columns/model missing for a test | Test emits an explicit **`skipped`** evidence record with the reason in its interpretation |
| Test engine raises an exception | Execution emits an **`error`** evidence record carrying the traceback — metrics are **never** substituted or invented |
| Ray / Dask distributed backends (roadmap) | Explicit `DistributedBackendNotImplemented` error, not a silent no-op |

Every degradation is recorded in evidence outputs: skipped and errored tests appear in the ledger, the report's evidence table, and the narrative inputs, so a reviewer always sees what *didn't* run and why.

---

## Running on your own data

The CLI operates on CSV or Parquet files. Your config tells StART which columns matter.

### 1. `start init` — scaffold a project

```bash
mkdir my-model-review && cd my-model-review
start init
```

Creates `configs/default.yaml`, `configs/policy/default_policy.yaml`, and `start_output/`. Edit `configs/default.yaml` to describe your model:

```yaml
model:
  model_id: churn-propensity-v3
  task_type: binary_classification     # drives which test families are planned
  materiality: high
  target_column: churned               # ground-truth label column
  score_column: churn_score            # model probability column (enables supervised tests)
```

### 2. `start plan` — preview, without executing

```bash
start plan --config configs/default.yaml
```

Shows exactly which registered tests the planner will schedule and why. Nothing runs, nothing is written.

### 3. `start run` — execute the review

```bash
start run --config configs/default.yaml data/train.csv --test data/holdout.csv
```

Runs the full pipeline and writes the report, ledger entries, and evidence records. Notes:

- `--test` (the holdout file) is optional but required for drift, split-diagnostics, and supervised tests.
- If your files lack a `score_column`, supervised and XAI tests are **skipped explicitly** (visible in the output table and evidence) rather than failing — score your data first, or use the Python API to pass a trained model:

```python
from start import build_context, load_config, run_review

config = load_config("configs/default.yaml")
result = run_review(config, build_context(config, train_df, test_df, model=fitted_model))
```

### 4. `start report` — re-print a report

```bash
start report --config configs/default.yaml                 # latest run
start report --config configs/default.yaml --run-id RUN-a2732245b4
```

### Tuning thresholds per test

```yaml
test_families:
  enabled: [preprocessing, supervised, xai]
  overrides:
    preprocessing.missingness: { warn_pct: 2.0, fail_pct: 10.0 }
    supervised.discrimination: { auc_warn: 0.70, auc_fail: 0.60 }
```

For governed regimes, put thresholds in the **policy YAML** instead — its content hash is stamped into every record.

---

## Adaptive compute routing

Device detection order: **CUDA → MPS (Apple Silicon) → CPU**. Detection never raises; PyTorch is optional. Databricks **runtime** detection (`DATABRICKS_RUNTIME_VERSION`) is independent of **device** detection — a Databricks GPU cluster detects both.

```
Compute Router
    ├── CUDA
    ├── MPS (Apple Silicon)
    ├── CPU
    └── Distributed backends
          ├── Databricks Spark   (runtime detection + stubs in v0.1)
          ├── Ray                (roadmap)
          └── Dask / k8s jobs    (roadmap)
```

- **Apple Silicon (M-series):** install the `[torch]` extra and `start doctor` reports `mps`. The deterministic engines in v0.1 are NumPy/scikit-learn-based (CPU); MPS/CUDA matter for the deep-learning and `hf_local` paths.
- **CUDA:** detected automatically when a CUDA-enabled PyTorch sees a GPU.
- **CPU:** always available; requesting `mode: gpu` on a CPU-only machine degrades transparently to CPU.

Force a device via YAML (`compute: { device: cpu }`) or env (`START_COMPUTE__DEVICE=cpu`).

---

## LLM providers

Backend-agnostic interface with lazy imports — the core installs with zero LLM dependencies:

| Provider | Selector | Needs |
|---|---|---|
| No LLM | `none` | Nothing. First-class mode; deterministic fallbacks everywhere |
| OpenAI | `openai` | `OPENAI_API_KEY` + `pip install -e ".[openai]"` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` + `pip install -e ".[anthropic]"` |
| xAI Grok | `grok` | `XAI_API_KEY` (OpenAI-compatible API) |
| HF Inference | `huggingface` | `HF_TOKEN` + `pip install -e ".[huggingface]"` |
| HF local | `hf_local` | `[huggingface]` extra; runs on detected device |
| Enterprise gateway | `enterprise_llm_gateway` | **Neutral placeholder** — no proprietary code/endpoints; map to a private implementation outside this repo |

Select via YAML (`llm.provider`) or env (`START_LLM__PROVIDER=anthropic`). API keys go in `.env` (copy `.env.example`; never commit `.env`). An unreachable provider degrades to `NoLLMProvider` rather than blocking a run, and LLM-drafted narratives must still pass the EvidenceCriticAgent's citation gate — a blocked draft is replaced by the deterministic template narrative.

---

## Databricks

Databricks is an **optional execution target**, never a requirement. Two notebooks show the intended pattern (`notebooks/01_databricks_quickstart.py` for the basic pipeline, `notebooks/02_propensity_model_review.py` for the flagship propensity review reading from a Spark table with a public-data fallback): notebooks are thin orchestration layers that install the package, detect the runtime, read data via Spark, convert the cohort to pandas for the deterministic engines, and run the same `run_review` pipeline with optional MLFlow logging (`experiment.provider: mlflow`, degrading to local JSONL when MLFlow is absent). The same notebook runs locally via a toy-data fallback — no cluster required. CI deliberately covers local providers only.

---

## Extending the registry

In-repo:

```python
from start import TestContext, TestResult, register_test
from start.core.schemas import ThresholdSpec

@register_test("recommender.ndcg", family="recommender", default_params={"k": 10})
def ndcg_at_k(ctx: TestContext, k: int = 10) -> TestResult:
    score = ...  # deterministic computation from ctx.train / ctx.test
    return TestResult(
        test_id="recommender.ndcg",
        test_name=f"NDCG@{k}",
        metrics={"ndcg": score},
        thresholds=[ThresholdSpec(metric="ndcg", warn=0.3, fail=0.2, direction="lower")],
        interpretation=f"NDCG@{k} is {score:.4f}.",
    ).apply_thresholds()
```

As an external pip package, expose an entry point and StART loads it automatically:

```toml
[project.entry-points."start.test_packs"]
my_pack = "my_pack.tests"
```

Contract for every engine: pure function of `(ctx, params)`, seeded, no LLM calls, no input mutation, declared limitations.

---

## Repository layout

```
src/start/              # the importable package (src layout — see Requirements)
  agents/               # planner, router, executor, critic, narrator, policy guard
  core/                 # typed schemas, config, hashing
  evidence/             # chained ledger + content-addressed store
  orchestration/        # end-to-end pipeline (run_review)
  providers/            # compute, data, experiment, llm, storage interfaces + impls
  registry/             # @register_test decorator + entry-point plugin loading
  connectors/           # universal data layer: demo | files | pandas | spark |
                        # snowflake, with auto 60/20/20 split
  taxonomy.py           # dataset profiling + type-aware model/validation maps
  policies/             # packaged default policy (runs work from any directory)
  modeling/             # demo workflows: data, model factory (RF/XGB/LGBM),
                        # tuning (grid/random/Optuna), metrics incl. lift,
                        # explainability w/ honest SHAP fallback, sensitivity,
                        # propensity workflow, DL roadmap skeleton
  reporting/            # Markdown report rendering
  tests/                # deterministic test families (preprocessing, supervised,
                        # xai, genai implemented; unsupervised, recommender,
                        # portfolio, attribution, deep_learning are roadmap stubs)
configs/                # run configs + versioned policy YAML
notebooks/              # Databricks-style thin orchestration notebooks
                        # (01: pipeline quickstart, 02: propensity model review)
examples/               # propensity_interactive.py (flagship), quickstart_local.py
                        # (smoke), deep_learning_sequence_demo.py (roadmap)
tests/                  # pytest + hypothesis suite (64 tests)
docs/architecture.md    # layer responsibilities, data flow, invariants
scripts/bootstrap.sh    # one-shot dev environment setup
```

---

## Development workflow

```bash
# 0. One-time setup (see Installation), then per change:
source .venv-start/bin/activate

# 1. Branch
git checkout -b feature/my-change

# 2. Make your changes under src/start/ and tests/

# 3. Lint — auto-fix, then verify clean
ruff check src tests --fix
ruff check src tests

# 4. Tests — the full suite must pass
pytest

# 5. (Optional, advisory in v0.1) type check
mypy

# 6. Commit and push
git add -A
git commit -m "feat: describe your change"
git push -u origin feature/my-change
```

Then open a Pull Request against `main` on GitHub. CI runs ruff, mypy (advisory), pytest with Hypothesis, and a CLI smoke test (`start doctor`, `start list-tests`) on Python 3.10 and 3.12 — the same commands you ran locally, so a green local run should mean a green PR.

Ground rules for contributions:

- Deterministic engines stay pure: no LLM calls, no network, no input mutation, seeds explicit.
- New engines ship with tests, including a determinism property where applicable.
- Agents may plan/route/critique/narrate; they never compute metrics.
- Nothing firm-specific, no credentials, no internal endpoints (see [Public-safety statement](#public-safety-statement)).

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'start'`**
The package isn't installed into the interpreter you're running. Almost always one of:

```bash
# Is the right venv active? Prompt should show (.venv-start)
which python        # should point inside .venv-start/bin
# Reinstall editable into THIS interpreter:
python -m pip install -e ".[dev]"
```

Use `python -m pip` (not bare `pip`) so the install targets the active interpreter.

**Editable install "not working" (edits don't take effect)**
Verify the install is actually editable and points at this checkout:

```bash
python -m pip show start-mrt | grep -i location
python -c "import start; print(start.__file__)"   # should be .../StART/src/start/__init__.py
```

If `start.__file__` points elsewhere, you have a second installation shadowing this one: `python -m pip uninstall start-mrt` everywhere, then reinstall in the venv.

**"Missing `src/start` package" / `error: package directory ... does not exist`**
You're not in the repository root, or the clone is incomplete. `ls src/start/__init__.py` must succeed from where you run pip. Re-clone if needed.

**Virtual environment conflicts**
Symptoms: imports resolve to unexpected paths, two Pythons fighting. Fix by being explicit:

```bash
deactivate 2>/dev/null; conda deactivate 2>/dev/null   # leave any other env
cd StART
python3.12 -m venv .venv-start --clear                  # rebuild cleanly
source .venv-start/bin/activate
python -m pip install -e ".[dev]"
```

Avoid creating the venv with a name that collides with source directories; `.venv-start` is safe and already in `.gitignore`. If Jupyter is involved, register the kernel explicitly: `python -m ipykernel install --user --name start`.

**MPS not detected on Apple Silicon (`doctor` shows `cpu`)**
Expected if PyTorch isn't installed — torch is optional and `cpu` is a fully supported answer. To enable MPS:

```bash
python -m pip install -e ".[torch]"
python -c "import torch; print(torch.backends.mps.is_available())"   # True on M-series
start doctor
```

If torch is installed but MPS is `False`: confirm you're on a native arm64 Python (`python -c "import platform; print(platform.machine())"` should print `arm64`, not `x86_64` under Rosetta) and on macOS 12.3+.

**Databricks unavailable**
Nothing to fix — Databricks is optional. `Databricks runtime: False` from `doctor` simply means you're local; the same configs and pipeline run locally. On an actual cluster, the runtime is detected via `DATABRICKS_RUNTIME_VERSION` automatically.

**LLM providers all show `available: False`**
This is the **default, fully supported state**. To enable one: install its extra, put the API key in `.env` (copied from `.env.example`), and set `llm.provider` in your config. If a configured provider is unreachable at run time, StART degrades to `NoLLMProvider` and the run still completes with template narratives.

**`pip install -e .` fails during metadata generation**
Usually a stale pip with the src layout/hatchling combination: `python -m pip install --upgrade pip` and retry. The error output names the real cause (e.g., a missing file) in its last lines.

Still stuck? Run `start doctor`, then open an issue with its output plus `python --version` and `pip --version`.

---

## Public-safety statement

This repository is a clean-room public implementation. It contains **no** proprietary code, internal endpoints, credentials, firm-specific templates/policies/thresholds, or internal schemas. `enterprise_llm_gateway` and `SnowflakePlaceholderProvider` are intentionally empty interfaces for private, out-of-repo implementations. Keep real policies in private configuration; never commit `.env`.

## Roadmap

- Quantitative-finance DL tracks by dataset type: limit order books (DeepLOB, CNN, temporal transformers), tick events (signature networks, neural point processes), multi-asset panels (TFT, temporal CNN, LSTM/GRU), volatility surfaces (CNN, neural PDEs, GNNs), and alternative text data (FinBERT variants, multimodal transformers, RAG) — the type-aware recommendation maps already ship in `start.taxonomy`
- Deep learning: torch-backed MLP/RNN/LSTM/GRU/TCN behind `[torch]` on genuinely sequential data, with Captum explainability (Integrated Gradients, DeepLIFT, Gradient SHAP) and occlusion analysis — scoped skeleton already in `src/start/modeling/deep_learning.py` and `examples/deep_learning_sequence_demo.py`; DL tuning stays laptop-safe by default, with large searches reserved for GPU clusters
- Test families: unsupervised, recommender ranking (NDCG/MAP/recall@k), portfolio optimization diagnostics, performance attribution, embedding drift, robustness
- GenAI: NLI-based grounding, prompt-injection probes, retrieval faithfulness (`start[genai]`)
- SHAP global/local consistency checks (`start[xai]`)
- Ray/Dask distributed backends; Spark-native engines for large cohorts
- HTML/PDF report rendering; signed report bundles

## License

Apache-2.0
