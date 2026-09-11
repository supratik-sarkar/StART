# Final frontend binding pass — PARTIAL

Date: 2026-09-10. Work performed in `StART`.

## Implemented bindings

- Live capability manifest remains authoritative for model lists, modes, source declarations, provider status, sensitivity grid, and deferred capabilities.
- Plan uses `POST /plan/generate`; run uses `POST /api/v1/workflow/run`. Both submit the selected `executionMode` and nested `parameters`. The frontend preserves the returned `agentProposal` and renders it structurally. Backend proposals are correctly labeled templates, not claimed to be live LLM-generated reasoning.
- Hybrid is the fresh default. Agentic and Deterministic are executable mode selections. Replay mode reads original runtime events. No OpenAI provider/model selection was changed; displayed readiness remains manifest-derived.
- Predictive controls submit `model`, `hyperparameters`, `preprocessing.imputation`, `preprocessing.scaler`, `split.strategy`, `split.test_size`, `sensitivity.mode` and `sensitivity_mode`. Actual supported mean/median, none/standard/minmax and stratified/random/time_series holdouts are exposed. No K-fold execution is implied.
- Hyperparameter descriptors were mechanically extracted from `HYPERPARAM_SPACES` in `src/start/modeling/models.py` via Python AST into `modelSchemas.json`. No hand-authored scientific search lists were introduced. This is a checked-in descriptor snapshot because the current public API has no model-parameter schema endpoint; refresh it when that registry changes.
- Tuning manifest labels map to actual runtime values: `optuna_bayesian → optuna`, `grid_search → grid`, `random_search → bounded_random_search`. Trial budget, model and objective metric reach execution. Trial observations, backend best parameters and objective name render from runtime events. Custom search-space and validation inputs are not sent because the web executor does not forward them.
- Sensitivity Lab exposes the exact nine-point grid, top-feature count and OAT/parallel-basket modes. Returned matrices and curves use source values only. Fractional chart-axis precision corrected.
- Recommender algorithms remain distinct. Choosing MF/NCF/FM/FFM selects the matching canonical context. A mismatched context blocks plan execution, preventing the backend's context-priority branches from silently changing the requested algorithm.
- Supervised fraud uses the actual `fraud_anomaly_aml` workflow and `synthetic_aml_imbalanced` context. Deferred anomaly techniques and dataset connectors are read-only availability information.
- Deep Learning exposes the parameterized tabular MLP via the current model registry. Unverified CNN/LSTM/GRU configuration is not advertised as runnable in the dedicated DL flow.
- Existing Evidence, Provenance, Governance, History, Replay, Compare, search, splitter and porcelain styling are preserved.

## Bounded real-browser verification

| Run | Selection | Observed result |
| --- | --- | --- |
| `RUN-WEB-b39cc69dcf` | Hybrid; LightGBM; mean imputation; minmax scaler; random split; test_size 0.3; n_estimators 100; num_leaves 15; parallel_basket | Completed with 47 evidence records and 15 artifacts. Resolved configuration confirms every selection. Nine returned basket shock rows from -0.30 through +0.30, including zero baseline. Replay verified. |
| `RUN-WEB-6561cf3925` | Agentic; field_aware_factorization_machine; recommender_ffm_v1 | Completed with 7 evidence records and 11 artifacts. Canonical model and configuration report FFM. Raw events confirm agentic_session and CP-AGENTIC. No FM substitution. |
| `RUN-WEB-09b62fbf00` | Deterministic; LightGBM; bounded_random_search; auc_roc; 5 trials | Five live trial events rendered, with best metric 0.895382 and backend best parameters. Backend reports completion. Presentation and persisted-events endpoints both return HTTP 500; fallback canonical artifacts identify Logistic Regression. This is not presented as a successful champion-analysis binding. |

Mode selectors, provider status, OAT/basket selection, compatible datasets, deferred connector/anomaly safeguards and fresh DEFINE were checked. Comparison of LightGBM run against `RUN-WEB-5306ae642f` returned backend differences. History displayed 82 runs. The final fresh workspace screenshot retains the porcelain theme and has no horizontal page overflow at 774px.

No broad estimator sweep, AI-provider call, approval, override, remote dataset upload or expensive broad experiment was performed.

## Remaining BACKEND_BLOCKER findings

1. **Quant dispatch is still metadata-only.** `runtime/execution.py` MARKET_SUBSET stores optimizer/scenario/request parameters in `market.extra.resolved_configuration`, then executes the unchanged registered tests. It does not select an optimizer, apply factor shocks or reprice based on those inputs. Controls stay unavailable; existing output presentation is preserved.
2. **Manifest preprocessing/split/explainer declarations still exceed applied inputs.** Runtime reads imputation/scaler only. Outliers and encoding are ignored; robust scaling is not applied; K-fold values fall through to a holdout branch. Explainability selection is not consumed. These unsupported selections remain unavailable.
3. **Tuning serialization and canonical promotion remain inconsistent.** Real five-trial LightGBM events and best parameters were received, but the presentation and raw persisted-events endpoints return HTTP 500. Fallback artifacts identify Logistic Regression. Parameter importance/champion comparison are not supplied in usable output. Frontend labels canonical versus tuning scope separately and does not infer a champion.
4. **Availability/schema metadata remains incomplete.** The live manifest has no per-estimator installed-dependency flag or public parameter-schema endpoint. Model dependencies are checked by backend fail-closed resolution at execution. CatBoost installation was not assumed or tested; no estimator substitution is performed by the frontend.
5. **Plan semantics are bounded.** `/plan/generate` creates a template proposal; it does not call an LLM. Agentic execution adds an executive synthesis checkpoint, not an autonomous committee. UI describes that returned behavior.
6. **Canonical data feature concern observed.** The LightGBM canonical result reports `score` and `prediction` among imputed features, unlike the execution loop's feature exclusion. Backend data selection and scientific interpretation require review. The frontend does not modify or conceal the supplied configuration.

## Verification and invariants

- `npm test -- --run`: 60 passed, 6 files.
- `npm run typecheck`: PASS.
- `npm run build`: PASS.
- Backend source hash comparison against start-of-task snapshot: unchanged.
- BACKEND_FILES_CHANGED = 0.
- FRONTEND_SCIENTIFIC_RECOMPUTATION = 0.
- SILENT_MODEL_SUBSTITUTION = 0 in frontend.
- RUNNABLE_DEFERRED_CAPABILITIES = 0.
- HARDCODED_AI_READY = 0.
- FFM_MAPPED_TO_FM = 0.
- No commit, push, tag or publishing.

One implementation pass and one targeted correction pass completed. Same-origin app left running at http://127.0.0.1:8000/ with fresh DEFINE open.

## Files changed

- `webapp/src/app/App.tsx`
- `webapp/src/contracts/types.ts`
- `webapp/src/contracts/validators.ts`
- `webapp/src/features/capabilities/runtimeSupport.ts`
- `webapp/src/features/capabilities/runtimeSupport.test.tsx`
- `webapp/src/features/composer/Composer.tsx`
- `webapp/src/features/investigation/ExperimentResults.tsx`
- `webapp/src/features/investigation/Science.tsx`
- `webapp/src/features/investigation/InvestigationWorkbench.tsx`
- `webapp/src/adapters/demo/DemoBackend.ts`
- `webapp/src/adapters/public/PublicStARTBackend.ts`
- `webapp/src/components/WorkbenchHeader.tsx`
- `webapp/src/features/capabilities/modelSchemas.json`
- `webapp/src/features/capabilities/EngineeringControls.tsx`
- `webapp/docs/FINAL_FRONTEND_BINDING.md`
