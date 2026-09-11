# Final live API binding — 2026-09-10

The available live frontend bindings are implemented. Overall status remains PARTIAL because the backend does not expose all requested observations and credential execution support. The Porcelain workspace is preserved. No backend source files changed: SHA-256 comparison of 288 files under src/start against the pre-change baseline matched. No training, full certification harness, commits, pushes, tags, or publishing occurred.

## Verification

One bounded browser journey used Hugging Face discovery (8 matches, limit 10), resolved scikit-learn/adult-census-income at main with income target, and pre-certified a 100-row sample. The explicit connectivity probe succeeded. Resolution supplied 15 columns and 14 features; unavailable row counts and dataset fingerprint were not filled from snapshots. Pre-certification returned CERTIFIED and temporal ordering NOT_EVALUATED.

Session SES-DATA-a45dba5fb3 consumed 500 rows in 5 batches, reported 285473 bytes read, one worker, 231.6 rows/second, and train/validation/test counts 349/74/77. STREAMING, COMPLETED, then STOPPED were observed. The session was stopped before completion of this task. No second ingestion was launched. Stop now refreshes the full session telemetry instead of retaining stale terminal measurements.

Live certification CERT_4FB4D8C42308 was generated at 2026-09-10T08:08:59Z, with bundle hash 72b605d3554ba97291b395132614adc31e95ffe9ca457ee0aef45e0b3c181e1b. The UI loads all 180 seed records, joins domain details only on matching experiment identities, and shows fetched time separately. Seven domains and both predictive scopes were inspected. A real predictive seed loaded detail, tuning, XAI, and sensitivity presentations. GPT-4.1 planning trace and numeric authority 0 remained distinct from the normal GPT-5.1 product provider.

Fresh workspace, history, replay RUN-WEB-48e7e0b981, evidence, provenance, and comparison with RUN-WEB-538b1d6bc7 were exercised without execution. Comparison returned artifact fingerprints and lineage. Runtime and certification were inspected at 1440, 1280, and 1024 pixels: document width equalled viewport width, with wide tables locally scrollable. Screenshots confirmed the existing visual language. One targeted frontend correction pass was completed, followed by 73 passing tests, typecheck, build, and final browser spot checks.

## BACKEND_BLOCKER

1. GET /certification/domains/predictive_binary returns the first (golden) experiment only. The domain index exposes the real champion aggregate but not real challenger aggregates or champion policy. The UI preserves the real index and seed records, displays unavailable detail, and does not substitute golden records or reconstruct statistics.
2. Live certification APIs omit dataset manifest dimensions, feature metadata, and targets. Those fields remain unavailable. The data resolve endpoint also leaves rows_available and fingerprint empty for the verified dataset; the sample fingerprint is shown only as sample evidence.
3. Temporary provider credential sessions are stored but are not consumed by discovery/resolve/pre-certification/ingestion adapters. No supported session reference connects those operations. Secret entry is therefore deferred; no frontend secret persistence or logging was added.
4. Granular sensitivity, portfolio, and scenario endpoints can return certification-wide fallback records even without establishing a valid selected run. XAI fallback is also bundle-wide. The frontend explicitly labels this scope limitation. The backend needs validated identity and source-scope fields before those observations can be claimed as selected-run evidence.
5. Completed tuning trials, sensitivity response curves, and XAI attribution curves are not persisted in the current certification payload. Trial budgets are not displayed as completed trials. Scenario units, horizon, currency, and Gate-6A integrity remain NOT_AVAILABLE when supplied that way.
6. Dataset discovery can silently fall back to curated entries, so discovery alone is not proof of a successful remote query. Registry connectivity and the latest explicit probe are displayed separately.
7. Certification is served from a process cache; frontend refresh retrieves the server view and does not reload artifact files. Generation time and source hashes remain visible.
8. Existing scientific exclusions remain: Kaggle deferred, Ray distributed execution unverified, ALE/LIME not executable, CatBoost absent, and real recommender/portfolio/market-risk adapters blocked.

## Required result

```text
CODEX_FINAL_LIVE_API_BINDING = PARTIAL
BACKEND_FILES_CHANGED = 0

DATA_RUNTIME
LIVE_PROVIDER_REGISTRY = BOUND
LIVE_PROVIDER_PROBE = BOUND; HUGGINGFACE VERIFIED
LIVE_DATASET_DISCOVERY = BOUND; HUGGINGFACE VERIFIED
LIVE_DATASET_RESOLVE = BOUND; HUGGINGFACE VERIFIED
LIVE_PRECERTIFICATION = BOUND; 100-ROW SAMPLE VERIFIED
LIVE_DATA_SESSION = BOUND; 500-ROW INGESTION STOPPED
LIVE_DATA_TELEMETRY = BOUND; REAL MEASUREMENTS VERIFIED
LIVE_RUNTIME_DESCRIPTOR = BOUND
HUGGINGFACE = RUNNABLE; BOUNDED JOURNEY VERIFIED
OPENML = BACKEND REPORTS RUNNABLE; INGESTION NOT RETESTED
UCI = BACKEND REPORTS RUNNABLE; INGESTION NOT RETESTED
KAGGLE = DEFERRED
LOCAL_CSV = BACKEND REPORTS RUNNABLE; INGESTION NOT RETESTED
LOCAL_PARQUET = BACKEND REPORTS RUNNABLE; INGESTION NOT RETESTED
WORKER_COUNT_DYNAMIC = 1
DISTRIBUTED_STATE_DYNAMIC = 1
FAKE_DISTRIBUTED_WORKERS = 0
CREDENTIAL_FLOW = DEFERRED; BACKEND ADAPTER CONNECTION MISSING
SECRET_PERSISTENCE_FRONTEND = 0

SCIENTIFIC_CERTIFICATION
LIVE_CERTIFICATION_API = BOUND
LIVE_DOMAIN_API = BOUND; REAL PREDICTIVE DETAIL INCOMPLETE
LIVE_RUN_API = BOUND; 180 RECORDS AND ON-DEMAND SEED DETAIL
LIVE_INVARIANTS_API = BOUND
LIVE_PROVIDER_TRACES = BOUND
LIVE_EXCLUSIONS = BOUND
TUNING = BOUND; TRIAL_DETAILS_NOT_PERSISTED
XAI = BOUND; METHOD AUDITS; CURVES UNAVAILABLE
SENSITIVITY = BOUND; NINE-POINT GRID; RESPONSES_NOT_PERSISTED
PORTFOLIO = BOUND; SUPPLIED OBJECTIVE RESULTS; FALLBACK SCOPE LABELED
SCENARIO = BOUND; SUPPLIED INVARIANTS; UNAVAILABLE FIELDS PRESERVED
GPT41_CERTIFICATION_POLICY = PRESERVED; NUMERIC AUTHORITY 0
GPT51_PRODUCT_PROVIDER = PRESERVED
SNAPSHOT_FALLBACK = DISABLED; OLD FILES RETAINED AS MANUAL FIXTURES ONLY
STALE_SNAPSHOT_OVERRIDE = 0
EVIDENCE = VERIFIED
PROVENANCE = VERIFIED
HISTORY = VERIFIED
REPLAY = VERIFIED
COMPARE = VERIFIED
PORCELAIN_DESIGN_PRESERVED = 1
FRONTEND_TESTS = PASS; 73 TESTS
TYPECHECK = PASS
BUILD = PASS
REAL_BROWSER_CHECK = PASS; BOUNDED JOURNEY
RESPONSIVE_CHECK = PASS; 1440 / 1280 / 1024
FRONTEND_SCIENTIFIC_RECOMPUTATION = 0
BACKEND_BLOCKERS = EIGHT ITEMS ABOVE
FILES_CHANGED = LIST BELOW
STOP.
```

## Files changed

- webapp/src/features/certification/liveApi.ts (new live loader and fail-closed request handling)
- webapp/src/features/certification/LiveDataRuntime.tsx (new live provider, dataset, and session controls)
- webapp/src/features/certification/LiveRunPresentation.tsx (new granular presentations and live seed detail)
- webapp/src/features/certification/CertificationSurface.tsx (live certification bindings)
- webapp/src/features/certification/certification.css (controls and responsive details)
- webapp/src/features/investigation/InvestigationWorkbench.tsx (granular presentation integration)
- webapp/src/App.tsx (retain runtime state across surface navigation)
- webapp/package.json (remove automatic prebuild snapshot/probe export; manual sync remains)
- webapp/tests/liveBinding.test.ts (five meaningful API/join boundary tests)
- webapp/docs/FINAL_LIVE_API_BINDING.md (this report)
- webapp/docs/DATA_RUNTIME_CERTIFICATION_FRONTEND.md (supersession note)
- webapp/dist/** (generated production build)
