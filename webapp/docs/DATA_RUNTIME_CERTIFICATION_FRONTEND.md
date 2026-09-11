> Superseded for production loading by [Final live API binding](FINAL_LIVE_API_BINDING.md). Snapshot loading described below is historical; production now uses live APIs.

# Data Runtime and Scientific Certification frontend

Implemented two first-class surfaces in the existing Porcelain Engineering Workspace. The workbench's DEFINE / EXECUTE / INVESTIGATE state, investigation selector, evidence/provenance, governance, history/replay and compare components remain intact. Opening a stored run returns to the workspace. No scientific engine code was changed, no certification matrix was executed, and nothing was committed or published.

## Data source and freshness

The frontend packages all fourteen existing JSON/JSONL artifacts from `scratch/scientific_certification`, certification `CERT_4FB4D8C42308`, timestamp `2026-09-10T08:08:59Z`. Original source file hashes are recorded. Loading verifies the packaged artifact SHA-256 and certification identity before rendering. This provides snapshot integrity, not an independent scientific attestation.

`npm run build` runs `scripts/export-certification.mjs` first. `npm run sync:certification` explicitly refreshes the development snapshot. It imports the existing backend registry through the local Python environment, calls the read-only availability probe and Ray environment probe, and packages the result with its own timestamp. It does not connect to datasets, read credential values into the payload, stream data, train models, execute planning calls, or compute scientific statistics. Python can be selected with `START_PYTHON`; if the probe cannot run, readiness is shown as unavailable, never recovered from stale data.

The backend's `is_runnable()` status is a dependency/credential readiness result, not a remote connectivity test. Registry results in this environment: Hugging Face, OpenML, UCI, Local CSV and Local Parquet RUNNABLE; Kaggle DEFERRED because its package is absent. The source also raises NotImplementedError for Kaggle streaming. Local runtime probe: `local_arrow_partitioned`, worker_count 1, distributed false, Ray absent. These environment values are explicitly separate from active ingestion telemetry.

The live `/api/v1/capability-manifest` was inspected. It remains the source for the existing workspace's OpenAI gpt-5.1 status; the certification's historical gpt-4.1 planning calls do not change that configuration.

## Scientific presentation boundaries

- Aggregate tables and CI plots only map supplied values. There is no mean/std/CI calculation, sorting by score, champion selection, sensitivity calculation, portfolio calculation or XAI computation.
- Golden and real predictive experiments are separate. The real-data record has a misleadingly named `gpt41_vs_deterministic` object containing only the real champion. A matching planning trace and GPT seed records are required to display a policy comparison; that experiment has neither.
- Deep learning preserves MLP/LSTM/GRU/CNN task context. CNN's JSON metric is accuracy despite the report's ROC-AUC label. GRU's supplied confidence bound above 1 is not clipped. Sequence and vision identities come from their actual run records, with missing dataset dimensions explicitly unavailable.
- FFM and FM remain separate, with their field-awareness invariant. The emitted NCF champion is shown.
- Portfolio presents the supplied minimum-volatility, maximum-diversification and equal-risk-contribution winners. No concentration winner is inferred. Market risk shows calibration metrics and DO_NOT_REJECT decisions without p-value ranking.
- Scenario's misleading `champion_model` field is presented as the repricing method, alongside its known-answer results and invariants.
- XAI methods retain independent status, semantic feature names, and deferred ALE/LIME reasons. Sensitivity displays the supplied nine-point grid, highlighted zero baseline, and OAT/basket audit flags, with missing response data called out.
- Exclusions are visible directly below the multidimensional status. The earlier discrepancy audit is clearly identified as historical, not treated as the current bundle's status.
- Certification seed IDs are not assumed to be replayable workbench run IDs. Empty evidence IDs and artifact hashes remain empty.

## BACKEND_PRESENTATION_GAP

1. No public HTTP route exposes the provider registry, dataset contract/loading controls, secure provider credential flow, or live telemetry. The frontend therefore provides a dated probe and inspection surface, not live connection/execution controls. Selecting a recorded dataset does not modify the current execution dataset.
2. Dataset manifest gives benchmark rows/features, target, provider, revision, fingerprint, license and a precertified flag. It does not emit rows available/consumed, column schema, feature roles, partition/shard metadata, bytes, throughput, cache events or active workers. They are shown unavailable.
3. No per-dataset pre-certification report is emitted. The boolean flag is not expanded into invented passing checks for missingness, duplicates, leakage, temporal ordering, split validity, etc.
4. No public certification API exists. The frontend uses an explicitly timestamped build snapshot with source hashes; it does not claim automatic live updates.
5. XAI records omit numerical attributions and PDP/ICE observations. Feature count 6 accompanies only five ordered feature names; all supplied values are preserved without inventing the sixth name.
6. Sensitivity records omit response values at each shock. No response curves are generated.
7. Run records contain tuning strategy and trial budgets but no completed tuning trial observations, best trial/parameters or promotion evidence. Proposed planning settings remain distinct from recorded execution settings.
8. The experiment matrix contains only EXP_PRED_GOLDEN. Other domain results are joined from their emitted aggregate/seed records; missing experiment specifications remain unavailable.
9. Portfolio lacks a distinct risk-dispersion/concentration winner and asset-level risk contributions. Scenario lacks separately emitted unit/provenance and Gate-6A integrity checks.
10. Sequence/vision dataset dimensions are absent from the dataset manifest. Dataset identities are preserved from JSONL without borrowing tabular row/feature counts.

## Bounded verification

- `npm test -- --run`: 68 tests passed, 7 files. Added eight evidence-boundary tests covering authoritative champion order/statistics, CNN accuracy, unmodified CI bounds, non-ranked domains, experiment-scoped GPT joins, five seeds, FM versus FFM, dataset identity joins and absent/deferred fields.
- `npm run typecheck`: PASS.
- `npm run build`: PASS, production bundle.
- `http://127.0.0.1:8000/`: fresh workspace, Data Runtime provider states, Hugging Face identity/fingerprint, pre-certification availability, golden/real predictive results, seed 0 exact configuration, GPT planning trace/token counts, XAI, nine-point OAT/basket grids, all seven domain views, portfolio objectives, calibration, scenario invariants and prominent exclusions verified.
- Actual persisted run `RUN-WEB-cf1c1ce554` opened through History as read-only replay. Evidence records, scientific provenance, attestation and 171 recorded execution events remained available.
- One correction pass clarified domain-wide coverage versus selected evaluation scope, explicitly labeled real-data non-execution, improved scientific acronyms, and ensured opening history returns to the workspace.
- Final browser console: no errors. Screenshot inspection retained porcelain/graphite/indigo/stone layout at the actual browser viewport. Backend source hash comparison: zero changed files. All fourteen original certification artifact hashes matched the packaged source manifest.

## Final report

```text
CODEX_DATA_CERTIFICATION_FRONTEND = PARTIAL
BACKEND_FILES_CHANGED = 0

DATA_RUNTIME
PROVIDER_REGISTRY = DATED_BACKEND_PROBE
HUGGINGFACE = RUNNABLE
OPENML = RUNNABLE
UCI = RUNNABLE
KAGGLE = DEFERRED_PACKAGE_ABSENT
LOCAL_CSV = RUNNABLE
LOCAL_PARQUET = RUNNABLE
DATASET_IDENTITY = BUNDLE_BOUND
REVISION = BUNDLE_BOUND
FINGERPRINT = BUNDLE_BOUND_COPYABLE
PRE_CERTIFICATION = FLAG_ONLY_DETAILED_REPORT_UNAVAILABLE
STREAMING_TELEMETRY = UNAVAILABLE
PARTITION_TELEMETRY = UNAVAILABLE
WORKER_TELEMETRY = UNAVAILABLE
CACHE_TELEMETRY = UNAVAILABLE
LOCAL_COLUMNAR = LOCAL_COLUMNAR_VERIFIED
DISTRIBUTED_RUNTIME = NOT_VERIFIED
FAKE_DISTRIBUTED_WORKERS = 0

SCIENTIFIC_CERTIFICATION
MULTIDIMENSIONAL_STATUS = PRESERVED
PREDICTIVE = GOLDEN_AND_REAL_BOUND
DEEP_LEARNING = FOUR_ARCHITECTURES_BOUND
FRAUD = SUPERVISED_GOLDEN_MATRIX_BOUND
RECOMMENDER = GOLDEN_MATRIX_BOUND_REAL_BLOCKED
PORTFOLIO = OBJECTIVES_BOUND_REAL_BLOCKED
MARKET_RISK = CALIBRATION_BOUND_REAL_BLOCKED
SCENARIO_TRADED_RISK = KNOWN_ANSWER_INVARIANTS_BOUND
CHAMPION_CHALLENGER = BACKEND_AUTHORITY
SEED_LEVEL_RESULTS = 180_RECORDS_INSPECTABLE
CONFIDENCE_INTERVALS = SUPPLIED_VALUES_ONLY
GPT41_VS_DETERMINISTIC = SIX_EXPERIMENTS_BOUND_REAL_PREDICTIVE_UNAVAILABLE
XAI_NATIVE = CERTIFIED_AUDIT
XAI_PERMUTATION = CERTIFIED_AUDIT
XAI_SHAP = CERTIFIED_AUDIT
XAI_PDP = CERTIFIED_AUDIT_CURVES_UNAVAILABLE
XAI_ICE = CERTIFIED_AUDIT_CURVES_UNAVAILABLE
XAI_ALE_DEFERRED = VISIBLE
XAI_LIME_DEFERRED = VISIBLE
SENSITIVITY_GRID = NINE_POINTS_ZERO_HIGHLIGHTED
SENSITIVITY_OAT = AUDIT_BOUND_RESPONSES_UNAVAILABLE
SENSITIVITY_PARALLEL = AUDIT_BOUND_RESPONSES_UNAVAILABLE
RECOMMENDER_MATRIX = NCF_FFM_MF_FM
PORTFOLIO_MULTI_OBJECTIVE = SUPPLIED_OBJECTIVE_WINNERS
MARKET_RISK_CALIBRATION = NO_PVALUE_RANKING
SCENARIO_INVARIANTS = NO_CHAMPION_RANKING
EXCLUSIONS_VISIBLE = 1
EVIDENCE = PRESERVED_BROWSER_VERIFIED
PROVENANCE = PRESERVED_BROWSER_VERIFIED
HISTORY = PRESERVED_BROWSER_VERIFIED
REPLAY = PRESERVED_BROWSER_VERIFIED
PORCELAIN_DESIGN_PRESERVED = 1
OLD_7_TAB_ARCHITECTURE = 0
ACTIVE_SURFACES = 0
EVIDENCE_ID_WALL = 0
FAKE_PROVIDER_STATUS = 0
FRONTEND_TESTS = 68_PASS
TYPECHECK = PASS
BUILD = PASS
REAL_BROWSER_CHECK = PASS_WITH_DOCUMENTED_DATA_GAPS
FRONTEND_SCIENTIFIC_RECOMPUTATION = 0
BACKEND_PRESENTATION_GAPS = 10_DOCUMENTED_ABOVE
FILES_CHANGED = LISTED_BELOW
```

## Files changed

Authored frontend files:

- `webapp/src/app/App.tsx`
- `webapp/src/features/certification/CertificationSurface.tsx`
- `webapp/src/features/certification/bundle.ts`
- `webapp/src/features/certification/certification.css`
- `webapp/scripts/export-certification.mjs`
- `webapp/scripts/probe-data-runtime.py`
- `webapp/tests/certification.test.ts`
- `webapp/package.json`
- `webapp/docs/DATA_RUNTIME_CERTIFICATION_FRONTEND.md`

Generated outputs: `webapp/public/certification/index.json`, its referenced SHA-256-named JSON artifact, and the production `webapp/dist` assets/index/certification snapshot. No `src/start/**` source file changes.
