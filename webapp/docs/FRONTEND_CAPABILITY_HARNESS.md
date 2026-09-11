# Frontend capability harness — PARTIAL

Work performed directly in `StART`. Porcelain Engineering Workspace retained. No backend source changes, scientific calculations, model changes, commits, pushes, tags or publishing.

## Connected capabilities

- Live `/api/v1/capability-manifest` provides execution-mode labels, provider/model/status and read-only scientific declarations. Loading/failure never implies AI readiness. OpenAI gpt-5.1 is displayed when reported by the backend. Questions continue through the backend; no browser OpenAI calls.
- Objective-first DEFINE uses current enabled workflows and compatibility from `/scenarios`, joined to `/execution-contexts`. Dataset metadata and scenario EDA remain source-backed. Noncanonical inspection scenarios cannot silently execute their mapped synthetic dataset.
- Hybrid defaults to deterministic execution with optional evidence questions, explicitly stating that AI planning is unavailable. Agentic Session is selectable for inspection but cannot execute. Deterministic Run invokes no automatic AI.
- Actual top-level seed and tuning `parameters.trials` are connected to plan/run submission. Bounds match the current executor (5–30 trials). Configuration changes invalidate the accepted plan. Absent values stay with the backend defaults.
- Existing parameter controls that never reached execution were removed. Unsupported options are declarations, not fake scientific selectors. LightGBM, XGBoost, CatBoost, Random Forest, Gradient Boosting, Logistic Regression, FM/FFM and all other lists come from the live manifest.
- Sensitivity has its own investigation section. Supplied shock rows produce exact-observation curves and supplied drift tables; missing observations remain unavailable. Explainability methods remain distinct. Training exposes supplied LR/gradient telemetry.
- Tuning renders persisted trial metadata and validation observations. No best-model inference or tuning arithmetic occurs in the browser.
- Replay uses the existing `/runs/{id}/events` endpoint rather than fabricated stage-completion events. This preserves actual tuning history.
- Existing evidence, contextual Ask Why/Challenge, human Approval/Override/Escalate, provenance, governance, history, comparison and search are retained. No human decision or AI question was submitted during this check.

## BACKEND_BLOCKER audit

The durable specification and acceptance prose exceed the actual routes and executor. These findings refer to current source and the responding same-origin server, not hypothetical missing features.

| ID | Backend evidence | Consequence |
| --- | --- | --- |
| B1 | `src/start/web/schemas.py:RunRequest` has no three-mode field. `routes_workbench.py:create_agent_plan` calls `make_canonical_plan` with no provider invocation. The run route calls deterministic `CanonicalExecutionService.execute` without agent-session dispatch. | AI-generated proposals, editable AI plan boundaries and Agentic Session execution cannot be integrated. The documented `/plan/generate` and `/workflow/run` routes are not in the actual route modules. |
| B2 | `src/start/runtime/execution.py` only reads `request_params` for `trials`; all other scientific configuration is ignored. The tuning call hardcodes `strategy="bounded_random_search"`. Manifest supplies no parameter schemas, ranges, combination constraints or dispatch bindings. | No truthful model/preprocessing/split/hyperparameter/explainer/architecture/scenario/portfolio controls can be made executable from this API. Grid/Optuna selection would be misleading. |
| B3 | `/execution-contexts` supplies six synthetic contexts. `/scenarios` includes inspection-only generators; execution instantiates the context ID, not the selected scenario ID. No remote-source ingestion or local file import route exists. | Kaggle/OpenML/UCI/Hugging Face/local CSV/Parquet are not runnable dataset choices. The manifest's `semantic_features_dataset` string alone is not an available UCI dataset. Actual synthetic feature names are preserved. |
| B4 | Public workflow registry lacks fraud/anomaly and LLM/agent execution workflows. Detectors and committee names appear only in the manifest. | No fraud/anomaly task dispatch, Graph AML or autonomous committee run can be truthfully exposed. |
| B5 | Recommender execution branches by context to MF, NCF and FM. No FFM branch exists, despite FFM engine code and manifest declaration. | FFM is distinctly named in read-only declarations; it is not executable through this public route. |
| B6 | Quantitative execution has no request-parameter selection for optimizer or shocks. | Existing supplied HRP/MinVar/ERC output renderers remain; choosing them or advanced reverse/group/correlated shocks is blocked. Deferred Monte Carlo options are not offered. |
| B7 | Manifest explainability includes only Tree SHAP, Kernel SHAP and permutation importance. Predictive canonical sensitivity is a regularization-C comparison, not the declared nine-point top-feature shock grid. | PDP/ICE/ALE/LIME and nine-point response outputs are not advertised as available without matching data. Canonical sensitivity metric and modes cannot be selected. |
| B8 | A real tuning run produced trial events but a separate canonical logistic-regression analysis. Trial events omit the objective metric name and canonical result omits a tuning summary/parameter importance. | Tuning observations and canonical analysis are separate; no champion claim is made from the unrelated canonical result. |

## Verification

Final `npm test -- --run`: **56 passed in 6 files**. `npm run typecheck`: **PASS**. `npm run build`: **PASS**, no chunk warning in the current build. No full Python regression or backend modification.

One finite browser sequence used `http://127.0.0.1:8000/`:

- Fresh public workspace, manifest fetch and backend-reported OpenAI gpt-5.1 Ready verified.
- Hybrid/Agentic/Deterministic selectable; Agentic execution blocked with explanation.
- Compatible Dataset Hub, source metadata and EDA verified.
- Manifest declarations inspected: predictive algorithms including LightGBM, preprocessing/splits/tuners/explainers, full nine-point grid, DL, fraud/anomaly, distinct MF/NCF/FM/FFM, scenario and portfolio methods, agent committee.
- Structured seed/trial controls and stale-plan invalidation verified.
- Executed **RUN-WEB-874b5f7f93**, hyperparameter tuning with **seed 0** and **5 trials**. Backend raw events confirm `actual_seed: 0`, five `tuning_trial` events, `requested_trials: 5` and `completed_trials: 5`.
- Reloaded the final build and replayed that run; all five trial observations, parameter details, original event IDs and optimization curve remained visible.
- Governance, run history (73 persisted runs) and comparison dialog verified. Existing Evidence/Provenance navigation remains present. No AI request, approval or override submitted.
- Porcelain screenshot inspected at 720px: document width equals viewport width; no horizontal page overflow. Existing splitter and independent panes retained.

Source hash comparison against the beginning-of-task snapshot confirms **BACKEND_FILES_CHANGED = 0**. The only generated scientific outputs are the authorized local verification run, stored by the unchanged backend.

Local application remains running at http://127.0.0.1:8000/ with fresh DEFINE open.

## Files changed

- `webapp/src/app/App.tsx`
- `webapp/src/contracts/types.ts`
- `webapp/src/contracts/backend.ts`
- `webapp/src/features/composer/Composer.tsx`
- `webapp/src/features/investigation/ModelAnalysis.tsx`
- `webapp/src/features/investigation/InvestigationWorkbench.tsx`
- `webapp/src/features/compare/RunCompareView.tsx`
- `webapp/src/state/useWorkbench.ts`
- `webapp/src/adapters/public/PublicReviewer.ts`
- `webapp/src/adapters/public/PublicStARTBackend.ts`
- `webapp/src/components/WorkbenchHeader.tsx`
- `webapp/src/design-system/workstation.css`
- `webapp/src/features/capabilities/runtimeSupport.ts`
- `webapp/src/features/capabilities/runtimeSupport.test.tsx`
- `webapp/src/features/capabilities/CapabilityDetails.tsx`
- `webapp/src/features/investigation/ExperimentResults.tsx`
- `webapp/docs/FRONTEND_CAPABILITY_HARNESS.md`
