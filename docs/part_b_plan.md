# StART Part B — "Model Review Operating System": Phased Build Plan

**Status:** planning document for review/approval. No code in this PR beyond
this file, the README refresh, and the `requirements.txt` snapshot.

**Baseline at time of writing:** v0.4.3, 133 tests passing, ruff clean across
`src tests examples notebooks`. Enterprise adapter (Part A) merged.

---

## 1. Purpose and guardrails

The vNext spec reframes StART from "train a model + generate diagnostics" into
a **data-first, agent-driven model-review operating system**: point it at a
dataset, answer three questions (business objective, target column(s), model
family), and the framework plans, validates, challenges, evidences, and
reports everything else.

This is a **multi-PR platform evolution, not one commit.** The phasing below is
ordered lowest-risk-first so the working v0.4.x demos (which you just validated
on the Mac) never break mid-stream. Each phase ships green: full test suite
passing, ruff clean, demos working, before the next begins.

**Non-negotiable guardrails carried from every prior phase:**

- Deterministic mode stays the default and needs no key.
- The LLM never touches raw data — only schema, statistics, the dataset
  profile, and the user objective. Every claim is gated by `EvidenceCriticAgent`.
- The enterprise gateway stays isolated behind `src/start/enterprise/`.
- Public-safe: no firm names, endpoints, credentials, internal schemas.
- No silent fallbacks; degradation is always explicit and disclosed.
- Don't break existing tests or the propensity / DL / agent-review workflows.

---

## 2. What already exists (reuse, don't rebuild)

The spec lists several capabilities StART already has. These get **extended**,
not created from scratch:

| vNext item | Already present | Gap to close |
| --- | --- | --- |
| Data abstraction (CSV/Parquet/Feather, pandas, Spark, Snowflake) | `src/start/connectors/` (6 connectors) | Add TSV/TXT/JSON/JSONL/Pickle/Excel/HuggingFace/OpenML/UCI loaders |
| Dataset profiling | `taxonomy.py: profile_dataset` / `DatasetProfile` | Promote to a first-class agent; add text/image-path/entity detection |
| Model recommendation | `ModelRecommendationAgent` | Add modality + sample-size/class-balance inputs |
| Validation planning | `ValidationPlannerAgent`, `ReviewPlannerAgent` | Wire to new task taxonomy |
| Split (60/20/20, configurable, stratified) | `three_way_split(fracs=...)` | Add time-based / group / custom strategies + a planner |
| Agentic governance (Challenge/Governance/Signoff/Critic) | all present | Keep; add new agents alongside |
| Enterprise LLM abstraction | Part A adapter | None — done |
| DL architectures (MLP/Residual/Wide&Deep) | `deep_learning.py` | Add registry; add CNN/sequence as own tracks |

**Net-new** items: `DatasetDiscoveryAgent`, `TargetSelectionAgent`,
`FeatureEngineeringAgent`, `TrainingStabilityAgent`, `CalibrationAgent`,
`ExplainabilityAgent`, `SensitivityAgent`, `RobustnessAgent` (several of these
wrap logic that exists today as functions but isn't agent-shaped), task
inference, the architecture registry, prompt-guided intent, and the vision
modality.

---

## 3. The one breaking change (call it out early)

vNext item #9 wants to **separate model family from activation function** and
**eliminate architecture names like `leaky_relu_mlp`**. Today `leaky_relu_mlp`
is a first-class entry in `SUPPORTED_ARCHITECTURES` and is depended on by:
`deep_learning.py`, `dl_models.py`, the DL CLI/example flags, both notebooks,
and `tests/test_dl_suite.py`.

**Plan:** when we reach Phase 5, introduce `family="mlp"` + `activation="leaky_relu"`
as the canonical form, and keep `leaky_relu_mlp` working as a **deprecated alias**
that emits a warning and maps to `(mlp, leaky_relu)`. No hard removal until a
later major version. This keeps existing tests and demos green through the
transition.

---

## 4. Phased breakdown

Each phase is a self-contained PR with its own tests and a version bump.

### Phase B1 — Data-first foundation (low risk, additive)
**Goal:** "What dataset are we reviewing?" becomes the first interaction.
- Extend the loader to all listed formats: TSV, TXT, JSON, JSONL, Pickle (with
  a safety note — pickle executes code on load, so gate it behind an explicit
  `--allow-pickle`), Excel (`openpyxl`), plus optional `HuggingFace` / `OpenML`
  / `UCI` dataset loaders behind extras (`[datasets]`).
- `DatasetDiscoveryAgent`: wraps and upgrades `profile_dataset` — schema, dtypes,
  candidate targets, timestamps, entity IDs, categoricals, **text fields**,
  **image-path columns**, missingness. Emits a `DatasetProfile` as evidence.
- New CLI: `start discover <path>` prints the profile + evidence.
- **Ships:** new loaders, the agent, profile evidence, tests, README snippet.
- **Risk:** low — purely additive; existing code paths untouched.
- **Version:** 0.5.0.

### Phase B2 — Objective, target, task (low risk, additive)
**Goal:** the three user questions become a guided flow.
- `TargetSelectionAgent`: user explicitly confirms target column(s); supports
  single and multi-output. No training without confirmation.
- Task inference: binary / multiclass / multilabel / regression / forecasting /
  ranking / recommendation / anomaly — inferred from the profile, user-overridable,
  evidence-backed. (Extends the `target_type` already in `DatasetProfile`.)
- Prompt-guided intent (LLM mode only): a "Dataset Intent Prompt" sends only
  schema + stats + profile + objective to the LLM, which proposes target
  candidates / task type / risk considerations / a validation plan. In
  deterministic mode this prompt never appears. Critic-gated as always.
- **Ships:** two agents, task taxonomy, the intent prompt path, tests.
- **Risk:** low — additive; LLM path reuses the existing evidence-bundle plumbing.
- **Version:** 0.5.1.

### Phase B3 — Split planner + feature engineering (medium risk)
**Goal:** explicit, non-hardcoded splits and a per-modality FE layer.
- `DataSplitPlanner`: Random / Stratified / Time-based / Group / Custom, with
  user-controlled train/test/oos percentages (60/20/20 default, never
  hardcoded). OOS generation is explicit and evidenced. Builds on the existing
  configurable `three_way_split`.
- `FeatureEngineeringAgent`: tabular (scaling, encoding, leakage detection,
  missingness indicators, drift), sequential (sliding windows, lags, rolling
  stats, trend), vision (normalization, augmentation diagnostics — diagnostics
  only, no training yet), text (tokenization/embedding diagnostics). Everything
  produces evidence.
- **Risk:** medium — touches the data pipeline; time/group splits need careful
  leakage handling and their own tests.
- **Version:** 0.5.2.

### Phase B4 — Agent orchestration consolidation (medium risk)
**Goal:** the full ~14-agent roster wired into one coherent flow.
- Wrap existing diagnostic *functions* into named agents the spec calls for:
  `TrainingStabilityAgent`, `CalibrationAgent`, `ExplainabilityAgent`,
  `SensitivityAgent`, `RobustnessAgent` (these currently live as functions /
  engines; the agent shells make the orchestration legible and the report
  sections uniform).
- A single `ReviewOrchestrator` that runs: Discovery → TargetSelection →
  FeatureEngineering → ModelRecommendation → (train) → Stability → Calibration
  → Explainability → Sensitivity → Robustness → Challenge → Governance →
  Signoff → Critic.
- **Risk:** medium — refactor of orchestration; mitigated by keeping the
  existing `run_review` / `run_dl_review` entry points working as thin wrappers.
- **Version:** 0.5.3.

### Phase B5 — Architecture registry (THE breaking change, contained)
**Goal:** family ⊕ activation, per spec #9.
- `ArchitectureRegistry` separating family (`mlp`, `residual_mlp`, `wide_deep`,
  `cnn`, `rnn`, `gru`, `lstm`, `bi_lstm`, `transformer`) from activation
  (`relu`, `leaky_relu`, `gelu`, `tanh`, `selu`, `elu`).
- `leaky_relu_mlp` retained as a **deprecated alias** → `(mlp, leaky_relu)` with
  a `DeprecationWarning`. All existing tests/demos keep passing.
- **Risk:** medium-high — the one breaking item; contained by the alias shim and
  a dedicated migration test.
- **Version:** 0.6.0 (minor bump signals the registry change).

### Phase B6 — Vision / CNN modality (highest risk, own track)
**Goal:** real computer-vision review, isolated from the tabular pipeline.
- Per our earlier discussion: a **separate modality track**, not a fifth tabular
  architecture. `vision_data.py` (image-folder / small public set), preset CNNs
  (`simple_cnn`, a ResNet-style block) — **preset architectures, not free-form
  layer specs**, for governability. Grad-CAM / occlusion explainability;
  corruption / perturbation robustness; heatmap figures. Gated behind `[vision]`.
- Sequence models (RNN/GRU/LSTM/Bi-LSTM/TCN/Transformer) get the same treatment
  on genuinely sequential data — still roadmap until a real sequential dataset
  and metrics are in place; never forced onto tabular data.
- **Risk:** high — new data contract (4-D tensors), new metrics, new
  explainability. Worth its own design doc before building.
- **Version:** 0.7.0.

---

## 5. Cross-cutting acceptance criterion

By end of Phase B4, a user should be able to point StART at an arbitrary
tabular dataset and answer only:

1. What is the business objective?
2. Which column(s) are the target?
3. Which model family should we evaluate?

…with discovery, split planning, validation, challenge, evidence, and report
all handled by the framework. Vision/sequence modalities (B6) extend this to
non-tabular data.

## 6. Recommendation

Approve **B1 → B2 → B4** as the near-term sequence (data-first entry, the three
questions, and orchestration consolidation deliver the headline UX with the
least risk). Treat **B3** (split/FE depth), **B5** (registry), and **B6**
(vision) as deliberate follow-ups, with B6 getting its own design doc. I will
not start any phase until you pick one.
