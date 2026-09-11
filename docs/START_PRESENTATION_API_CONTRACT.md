# StART — Authoritative Presentation & Live Data Runtime API Contract

**Specification Version**: 1.0.0  
**Authority**: StART Scientific Validation & Governance Core  
**Frontend Binding Target**: Codex Porcelain Engineering Workspace  
**Status**: ACTIVE CONTRACT  

---

## 1. Executive Summary & Epistemological Boundaries

This contract defines the public HTTP presentation and data runtime interfaces for the StART workbench. It directly addresses and closes the ten backend presentation gaps identified in `webapp/docs/DATA_RUNTIME_CERTIFICATION_FRONTEND.md`.

### Strict System Invariants
1. **Frontend Frozen**: `FRONTEND_FILES_CHANGED = 0` (Codex owns `webapp/src/**`).
2. **Web Layer Non-Computation**: `WEB_LAYER_SCIENTIFIC_CALCULATIONS_ADDED = 0`. The web transport layer may read, validate, paginate, filter, and serialize existing models and bundles. It must **never** compute statistical tests, confidence intervals, Shapley values, portfolio weights, or scenario losses.
3. **Zero Secret Exposure**: `SECRET_EXPOSURE = 0`. Credentials (API keys, Kaggle tokens) are handled exclusively via ephemeral server-side in-memory scopes or declared deferred. Secrets are never persisted to disk, never logged, never added to EvidenceRecords, and never echoed back in HTTP envelopes.
4. **Architectural Truth**: Ray cluster execution is reported truthfully as unavailable (`ray_available = false`, `ray_distributed_execution_verified = false`). Parallelism is reported as `local_arrow_partitioned` with 1 worker.
5. **Snapshot & Bundle Integrity**: Scientific certification endpoints serve data from the authoritative 14-artifact bundle in `scratch/scientific_certification/` (`CERT_4FB4D8C42308`, timestamp `2026-09-10T08:08:59Z`) with cryptographically verified artifact SHA-256 hashes.

---

## 2. Complete Codex Gap Mapping Matrix

| GAP_ID | Description | Source Data Exists | Backend Source | New / Extended Route | Security Requirement | Status |
| :--- | :--- | :---: | :--- | :--- | :--- | :---: |
| **GAP-1** | Missing public routes for provider registry, contract resolve, live telemetry & credential flow | **YES** | `start.data.providers.registry`, `contract.py`, `parallel.py` | `GET /api/v1/data/providers`<br>`GET /api/v1/data/providers/{provider}/probe`<br>`GET /api/v1/data/providers/{provider}/datasets`<br>`POST /api/v1/data/resolve`<br>`POST /api/v1/data/sessions`<br>`GET /api/v1/data/sessions/{id}/telemetry` | Ephemeral memory-only credentials; no secret echoes | **CLOSED** |
| **GAP-2** | Dataset manifest lacks consumed rows, column schema, feature roles, partition metadata, bytes, throughput | **YES** (on resolved contract) | `DatasetContract`, `DatasetTelemetry` | `POST /api/v1/data/resolve`<br>`GET /api/v1/data/sessions/{id}/telemetry` | Clean URI sanitation; no embedded tokens | **CLOSED** |
| **GAP-3** | Missing per-dataset pre-certification quality report (boolean flag only) | **YES** | `start.data.providers.precertification.precertify_dataset` | `POST /api/v1/data/precertify` | Fail-closed on missing target / severe leakage | **CLOSED** |
| **GAP-4** | Frontend forced to package build-time JSON snapshot due to missing certification HTTP API | **YES** | `scratch/scientific_certification/*.json[l]` | `GET /api/v1/certification`<br>`GET /api/v1/certification/domains`<br>`GET /api/v1/certification/runs`<br>`GET /api/v1/certification/invariants` | Read-only; bundle & source SHA-256 integrity checks | **CLOSED** |
| **GAP-5** | XAI records lack numerical attributions and PDP/ICE observations in static bundle | **YES** (audit metadata) / **NO** (raw arrays) | `xai_results.json`, `run.presentation` | `GET /api/v1/certification/xai`<br>`GET /api/v1/runs/{id}/xai` | No web-layer recalculation; unpersisted arrays return `null` | **CLOSED** |
| **GAP-6** | Sensitivity records omit per-shock response values | **YES** (grid & audit flags) / **NO** (response curves) | `sensitivity_results.json` | `GET /api/v1/certification/sensitivity`<br>`GET /api/v1/runs/{id}/sensitivity` | No web-layer recalculation; responses unpersisted reported truthfully | **CLOSED** |
| **GAP-7** | Run records contain strategy and budget but omit completed trial logs | **YES** (budget/strategy) / **NO** (trial trajectories) | `deterministic_runs.jsonl`, `GLOBAL_QUEUE` | `GET /api/v1/runs/{id}/tuning` | Report `TRIAL_DETAILS_NOT_PERSISTED`; do not synthesize trials | **CLOSED** |
| **GAP-8** | `experiment_matrix.json` contains only EXP_PRED_GOLDEN; other domains joined from seeds | **YES** (across seed & challenger files) | `champion_challenger.json`, `deterministic_runs.jsonl` | `GET /api/v1/certification/domains`<br>`GET /api/v1/certification/domains/{domain}` | Missing experiment specs return `experiment_spec = null`; do not invent | **CLOSED** |
| **GAP-9** | Portfolio lacks risk-dispersion winner & asset-level risk contributions; Scenario lacks units & Gate-6A | **YES** (in aggregate results) | `champion_challenger.json`, `invariant_results.json` | `GET /api/v1/runs/{id}/portfolio`<br>`GET /api/v1/runs/{id}/scenario`<br>`GET /api/v1/certification/domains/portfolio` | Return multi-objective winners; uncomputed checks return `NOT_AVAILABLE` | **CLOSED** |
| **GAP-10** | Sequence & vision dataset dimensions absent from dataset manifest | **YES** (in generator / runs) / **NO** (in static manifest) | `dataset_manifest.json`, `deterministic_runs.jsonl` | `GET /api/v1/certification/domains/deep_learning` | Expose only dimensions present in run records; do not infer | **CLOSED** |

---

## 3. Data Provider & Runtime Contracts (`/api/v1/data/*`)

### A. Provider Registry API
- **Route**: `GET /api/v1/data/providers`
- **Description**: Returns operational status of all registered data provider adapters.
- **Response Schema**:
```json
{
  "success": true,
  "schema_version": "5.0.0",
  "data": {
    "providers": [
      {
        "provider_id": "huggingface",
        "display_name": "Hugging Face Hub",
        "status": "RUNNABLE",
        "runnable": true,
        "credential_required": false,
        "credential_configured": false,
        "streaming_supported": true,
        "package_available": true,
        "connection_test_supported": true,
        "remote_connectivity_verified": false,
        "reason_if_unavailable": null
      },
      {
        "provider_id": "openml",
        "display_name": "OpenML",
        "status": "RUNNABLE",
        "runnable": true,
        "credential_required": false,
        "credential_configured": false,
        "streaming_supported": true,
        "package_available": true,
        "connection_test_supported": true,
        "remote_connectivity_verified": false,
        "reason_if_unavailable": null
      },
      {
        "provider_id": "uci",
        "display_name": "UCI Machine Learning Repository",
        "status": "RUNNABLE",
        "runnable": true,
        "credential_required": false,
        "credential_configured": false,
        "streaming_supported": true,
        "package_available": true,
        "connection_test_supported": true,
        "remote_connectivity_verified": false,
        "reason_if_unavailable": null
      },
      {
        "provider_id": "kaggle",
        "display_name": "Kaggle Datasets",
        "status": "DEFERRED",
        "runnable": false,
        "credential_required": true,
        "credential_configured": false,
        "streaming_supported": false,
        "package_available": false,
        "connection_test_supported": true,
        "remote_connectivity_verified": false,
        "reason_if_unavailable": "Kaggle package is not installed (pip install kaggle)."
      },
      {
        "provider_id": "local_csv",
        "display_name": "Local CSV Files",
        "status": "RUNNABLE",
        "runnable": true,
        "credential_required": false,
        "credential_configured": false,
        "streaming_supported": true,
        "package_available": true,
        "connection_test_supported": false,
        "remote_connectivity_verified": true,
        "reason_if_unavailable": null
      },
      {
        "provider_id": "local_parquet",
        "display_name": "Local Apache Parquet Files",
        "status": "RUNNABLE",
        "runnable": true,
        "credential_required": false,
        "credential_configured": false,
        "streaming_supported": true,
        "package_available": true,
        "connection_test_supported": false,
        "remote_connectivity_verified": true,
        "reason_if_unavailable": null
      }
    ]
  }
}
```

### B. Remote Connectivity Probe
- **Route**: `GET /api/v1/data/providers/{provider}/probe`
- **Description**: Executes a lightweight remote network connectivity test (timeout: 3.0s).
- **Response Schema**:
```json
{
  "success": true,
  "data": {
    "provider_id": "huggingface",
    "package_available": true,
    "remote_connectivity_verified": true,
    "latency_ms": 142.5,
    "probe_endpoint": "https://huggingface.co/api/datasets",
    "error": null
  }
}
```

### C. Dataset Discovery API
- **Route**: `GET /api/v1/data/providers/{provider}/datasets`
- **Query Parameters**:
  - `query`: string (search term, e.g. "credit", "adult")
  - `limit`: integer (default 20, max 100)
  - `task`: string (optional filter, e.g. "classification", "regression")
- **Response Schema**:
```json
{
  "success": true,
  "data": {
    "provider": "huggingface",
    "count": 1,
    "datasets": [
      {
        "dataset_id": "scikit-learn/adult-census-income",
        "name": "Adult Census Income",
        "description": "Extraction from 1994 Census database for income classification.",
        "revision": "main",
        "task": "binary_classification",
        "rows": 32561,
        "columns": 15,
        "target_candidates": ["income"],
        "license": "Public Domain",
        "provider": "huggingface",
        "credential_required": false,
        "streaming_supported": true
      }
    ]
  }
}
```

### D. Dataset Contract Resolution API
- **Route**: `POST /api/v1/data/resolve`
- **Request Schema**:
```json
{
  "provider": "huggingface",
  "dataset_id": "scikit-learn/adult-census-income",
  "revision": "main",
  "target": "income",
  "configuration": {}
}
```
- **Response Schema**:
```json
{
  "success": true,
  "data": {
    "contract": {
      "provider": "huggingface",
      "dataset_id": "scikit-learn/adult-census-income",
      "dataset_name": "adult-census-income",
      "revision": "main",
      "source_uri": "https://huggingface.co/datasets/scikit-learn/adult-census-income",
      "license": "Hugging Face Open Dataset (Apache 2.0 / CC-BY / Open)",
      "citation": "Hugging Face Hub repository: scikit-learn/adult-census-income",
      "schema": {
        "age": "int64",
        "workclass": "object",
        "education": "object",
        "income": "object"
      },
      "feature_names": ["age", "workclass", "education"],
      "feature_roles": {
        "age": "numeric",
        "workclass": "categorical",
        "education": "categorical",
        "income": "target"
      },
      "target": "income",
      "target_column": "income",
      "rows_available": 32561,
      "columns": 15,
      "streaming_supported": true,
      "cache_policy": "arrow_stream",
      "fingerprint": "ddfbd5c16c58c371be86359f92f2fed1184fb78540d56f9f5daeacb83243639d"
    }
  }
}
```

### E. Data Pre-Certification API
- **Route**: `POST /api/v1/data/precertify`
- **Request Schema**:
```json
{
  "provider": "huggingface",
  "dataset_id": "scikit-learn/adult-census-income",
  "revision": "main",
  "target": "income",
  "sample_rows": 500
}
```
- **Response Schema**:
```json
{
  "success": true,
  "data": {
    "source_identity": "huggingface://scikit-learn/adult-census-income@main",
    "license_status": "Hugging Face Open Dataset (Apache 2.0 / CC-BY / Open)",
    "schema_validity": true,
    "target_validity": true,
    "feature_roles": true,
    "duplicate_count": 0,
    "duplicate_rate": 0.0,
    "missingness_by_feature": {},
    "target_distribution": { "<=50K": 379, ">50K": 121 },
    "temporal_ordering": "NOT_EVALUATED",
    "leakage_findings": [],
    "split_validity": true,
    "fingerprint": "ddfbd5c16c58c371be86359f92f2fed1184fb78540d56f9f5daeacb83243639d",
    "overall_status": "CERTIFIED",
    "warnings": [],
    "failures": []
  }
}
```

### F. Ingestion Sessions & Real-Time Telemetry
- **Routes**:
  - `POST /api/v1/data/sessions`: Create and begin bounded stream ingestion.
  - `GET /api/v1/data/sessions/{session_id}`: Query session configuration.
  - `POST /api/v1/data/sessions/{session_id}/stop`: Terminate active ingestion.
  - `GET /api/v1/data/sessions/{session_id}/telemetry`: Fetch real-time metrics.
- **Telemetry Response Schema**:
```json
{
  "success": true,
  "data": {
    "session_id": "SES-DATA-7c2a10b48f",
    "state": "COMPLETED",
    "provider": "huggingface",
    "dataset_id": "scikit-learn/adult-census-income",
    "revision": "main",
    "rows_available": 32561,
    "rows_consumed": 2000,
    "bytes_read": 184520,
    "batch_count": 2,
    "batch_size": 1000,
    "partition_count": 3,
    "active_partition": "completed",
    "partition_progress": 1.0,
    "worker_count": 1,
    "execution_mode": "local_arrow_partitioned",
    "rows_per_second": 8420.5,
    "cache_state": "memory",
    "cache_hits": 0,
    "cache_misses": 1,
    "current_operation": "idle",
    "train_rows": 1400,
    "validation_rows": 300,
    "test_rows": 300,
    "device": "Darwin arm64 (Apple Silicon)",
    "started_at": 1773324000.12,
    "updated_at": 1773324000.35,
    "errors": []
  }
}
```

### G. Parallelism & Runtime Descriptor
- **Route**: `GET /api/v1/data/runtime`
- **Response Schema**:
```json
{
  "success": true,
  "data": {
    "execution_backend": "local_arrow_partitioned",
    "worker_count": 1,
    "distributed": false,
    "partition_strategy": "stratified_columnar",
    "ray_available": false,
    "ray_distributed_execution_verified": false,
    "openmp_threads_invariant": "n_jobs=1 on Darwin enforced"
  }
}
```

### H. Ephemeral Credential Security
- **Routes**:
  - `POST /api/v1/data/provider-sessions`
  - `DELETE /api/v1/data/provider-sessions/{session_id}`
- **Request Schema**:
```json
{
  "provider": "kaggle",
  "credentials": {
    "username": "researcher",
    "key": "sensitive-token-never-echoed"
  },
  "ttl_seconds": 3600
}
```
- **Response Schema**:
```json
{
  "success": true,
  "data": {
    "session_id": "SEC-PROV-48a1ef90b2",
    "provider": "kaggle",
    "credential_configured": true,
    "expires_at": 1773327600.0
  }
}
```
*(Credentials remain exclusively in server memory; never returned, never persisted, never logged).*

---

## 4. Scientific Certification Read APIs (`/api/v1/certification/*`)

### A. Root Certification Summary
- **Route**: `GET /api/v1/certification`
- **Description**: Returns certification identity, multi-dimensional status, 14 artifact SHA-256 hashes, and bundle digest.
- **Response Schema**:
```json
{
  "success": true,
  "data": {
    "certification_id": "CERT_4FB4D8C42308",
    "generated_at": "2026-09-10T08:08:59Z",
    "bundle_hash": "c86d8b...",
    "source_artifact_hashes": [
      { "file": "certification_manifest.json", "sha256": "...", "bytes": 2136 },
      { "file": "champion_challenger.json", "sha256": "...", "bytes": 10797 }
    ],
    "status_dimensions": {
      "mathematical_invariants": "PASS",
      "golden_data_certification": "CERTIFIED",
      "real_data_empirical_certification": "PARTIAL",
      "model_matrix_completeness": "CERTIFIED",
      "xai_certification": "CERTIFIED",
      "gpt41_policy_certification": "CERTIFIED",
      "live_data_provider_certification": "CERTIFIED",
      "distributed_data_runtime_certification": "LOCAL_COLUMNAR_VERIFIED"
    },
    "total_runs": 180,
    "deterministic_runs": 150,
    "gpt41_runs": 30,
    "invariants_passed": "10 / 10"
  }
}
```

### B. Domain Certification Records
- **Routes**:
  - `GET /api/v1/certification/domains`
  - `GET /api/v1/certification/domains/{domain}`
- **Description**: Exposes domain champions, challengers, 95% confidence intervals, and policy comparisons. Missing experiment specifications return `experiment_spec = null`.
- **Response Schema for Recommender**:
```json
{
  "success": true,
  "data": {
    "domain": "recommender",
    "experiment_id": "EXP_RECOMMENDER",
    "layer": "golden_known_answer",
    "dataset": "recommender_ffm_v1",
    "primary_metric": "ndcg@10",
    "champion": {
      "model": "ncf",
      "mean": 0.565466,
      "std": 0.040017,
      "ci95": [0.515779, 0.615153]
    },
    "challengers": [
      { "model": "fm", "mean": 0.432005, "std": 0.034897, "ci95": [0.388675, 0.475336] },
      { "model": "ffm", "mean": 0.425331, "std": 0.016734, "ci95": [0.404553, 0.446109] },
      { "model": "mf", "mean": 0.399385, "std": 0.041328, "ci95": [0.348069, 0.450701] }
    ],
    "real_data_status": "RECOMMENDER_REAL_DATA_CERTIFICATION = BLOCKED",
    "experiment_spec": null
  }
}
```

### C. Seed-Level Run Explorer
- **Routes**:
  - `GET /api/v1/certification/runs` (Query params: `domain`, `model`, `policy`, `seed`, `limit`, `offset`)
  - `GET /api/v1/certification/runs/{run_id}`
- **Description**: Truthful pagination over all 180 runs with zero fabricated records.

### D. Invariants, XAI & Sensitivity Endpoints
- `GET /api/v1/certification/invariants`: 10/10 invariant proofs.
- `GET /api/v1/certification/xai`: 7 methods (Native, Permutation, SHAP, PDP, ICE certified; ALE, LIME deferred).
- `GET /api/v1/certification/sensitivity`: Canonical 9-point grid, OAT, parallel basket, zero baseline delta (`0.000000`).
- `GET /api/v1/certification/provider-traces`: Authentic GPT-4.1 planning traces.
- `GET /api/v1/certification/exclusions`: Fail-closed and deferred item catalog.

---

## 5. Granular Run Presentation Endpoints (`/api/v1/runs/{id}/*`)

### A. Tuning Presentation
- **Route**: `GET /api/v1/runs/{run_id}/tuning`
- **Response Schema**:
```json
{
  "success": true,
  "data": {
    "run_id": "run_det_pred_golden_xgboost_seed_0",
    "strategy": "bounded_random_search",
    "objective_metric": "roc_auc",
    "trial_budget": 5,
    "trials": [],
    "best_trial": null,
    "best_parameters": { "seed": 0 },
    "champion_model": "xgboost",
    "champion_lineage": null,
    "status": "TRIAL_DETAILS_NOT_PERSISTED",
    "notes": "Historical certification runs evaluated fixed seeds without per-trial persistence."
  }
}
```

### B. XAI & Sensitivity Presentation
- `GET /api/v1/runs/{run_id}/xai`: Serializes existing run/certification XAI outputs. Unpersisted numerical matrices return `null`.
- `GET /api/v1/runs/{run_id}/sensitivity`: Returns the 9-point grid with zero baseline verification.

### C. Portfolio Presentation
- **Route**: `GET /api/v1/runs/{run_id}/portfolio`
- **Response Schema**:
```json
{
  "success": true,
  "data": {
    "domain": "portfolio",
    "multi_objective_evaluation": {
      "min_variance": { "vol": 0.0705, "ret": -0.0330, "sharpe": -0.4655, "div_ratio": 4.5053, "hhi": 0.1591 },
      "erc": { "vol": 0.0747, "ret": -0.0227, "sharpe": -0.3006, "div_ratio": 4.4509, "hhi": 0.1229 },
      "hrp": { "vol": 0.0856, "ret": -0.0431, "sharpe": -0.5025, "div_ratio": 3.7886, "hhi": 0.1300 }
    },
    "objective_winners": {
      "minimum_volatility": "min_variance",
      "maximum_diversification": "min_variance",
      "equal_risk_contribution": "erc"
    },
    "real_data_status": "PORTFOLIO_REAL_MARKET_DATA_CERTIFICATION = BLOCKED"
  }
}
```

### D. Scenario & Traded Risk Presentation
- **Route**: `GET /api/v1/runs/{run_id}/scenario`
- **Response Schema**:
```json
{
  "success": true,
  "data": {
    "scenario_type": "gate6_asset_tail_shock",
    "shock_definition": "relative_equity_shock",
    "units": "NOT_AVAILABLE",
    "horizon": "NOT_AVAILABLE",
    "currency": "NOT_AVAILABLE",
    "repricing_method": "gate6_asset_tail_shock",
    "loss": 0.2000,
    "zero_shock_check": "PASS",
    "monotonicity_check": "PASS",
    "Gate6_integrity": "PASS",
    "Gate6A_integrity": "NOT_AVAILABLE"
  }
}
```

---

## 6. Security & Invariant Audit Confirmation

- **Secret Exposure Check**: Verified that no route returns credentials, OpenAI keys, or Kaggle tokens.
- **Science Boundary Check**: Verified that no route performs scientific recomputation.
- **Frontend File Integrity**: Verified that `webapp/src/**` has zero changes.

---

## 7. Final Evidence & Data-Session Backend Additions (Codex G1–G7 Closure)

This section specifies the contracts for the seven blocker groups identified in `webapp/docs/FINAL_LIVE_API_BINDING.md`.

### G1 — Predictive Binary Dual Scope & Distinct Domain Indexing
- **Invariants**: `CROSS_EXPERIMENT_RESULT_SUBSTITUTION = 0`. Golden known-answer and real-world external scopes for `predictive_binary` are distinct entities and must never substitute for one another.
- `GET /api/v1/certification/domains`:
  - Returns separate entries for `EXP_PRED_GOLDEN` (institutional credit, synthetic) and `EXP_PRED_REAL_ADULT` (adult census income, scikit-learn/huggingface).
  - Exposes `challenger_summaries` for all models (XGBoost, LightGBM, Gradient Boosting, Logistic Regression).
  - Exposes `champion_policy: "deterministic"`.
  - Exposes `scopes` array providing lightweight descriptors for each certified domain/layer combination.
- `GET /api/v1/certification/domains/{domain}`:
  - Accepts `experiment_id` and `data_layer` query parameters.
  - When `data_layer=real_external` or `experiment_id=EXP_PRED_REAL_ADULT`, returns the real adult census experiment evaluation without substitution.
  - When `data_layer=golden_known_answer` or `experiment_id=EXP_PRED_GOLDEN`, returns the golden credit experiment evaluation.
  - Returns `selected_experiment_id`, `selected_data_layer`, `scopes`, and `experiments`.

### G2 — Truthful Dataset Metadata & Fingerprint Scopes
- **Invariants**: `DATASET_METADATA_SCOPE_TRUTHFUL = 1`. Dataset-level SHA-256 is strictly segregated from sample-level SHA-256 (`dataset_fingerprint != sample_fingerprint`).
- `POST /api/v1/data/resolve`:
  - Exposes full dataset SHA-256 (`ddfbd5c1...` for adult census) and sample SHA-256.
  - Exposes `fingerprint_scope: "full"` for manifested datasets, `"sample"` for dynamic streams.
  - Exposes `rows_available: 32561` for verified adult dataset.
- `POST /api/v1/data/precertify`:
  - Returns `sample_fingerprint`, `dataset_fingerprint`, and `fingerprint_scope: "sample"`.
- `GET /api/v1/certification/domains/{domain}`:
  - Exposes `dataset_metadata`, `dataset_rows`, `dataset_features`, `dataset_target`, and `dataset_fingerprint` from `dataset_manifest.json`.

### G3 — Provider Credential Session Propagation & Zero Secret Echo
- **Invariants**: `PROVIDER_SESSION_PROPAGATION = 1`, `PROVIDER_CREDENTIAL_ECHO = 0`, `SECRET_EXPOSURE = 0`.
- Client passes `X-Provider-Session-ID` header or `provider_session_id` request field.
- Routes lookup ephemeral memory-only credentials from `PROVIDER_SESSION_STORE` and wire them directly to provider adapters via `adapter.set_credentials()`.
- Endpoints (`/providers`, `/runtime`, `/sessions`) never persist, log, or echo credentials; report `credential_scope: "ephemeral_backend_session"` and `multi_tenant_isolation_verified: false`.

### G4 — Run-Scoped Presentation Identity & Zero Unscoped Fallback
- **Invariants**: `UNSCOPED_SCIENTIFIC_FALLBACK = 0`.
- All five granular endpoints (`/tuning`, `/xai`, `/sensitivity`, `/portfolio`, `/scenario`) return standardized identity fields:
  - `requested_run_id`: requested run identifier.
  - `source_run_id`: identifier of the run that generated the evidence, or `null`.
  - `experiment_id`: experiment identifier or `null`.
  - `certification_id`: certification bundle ID or `null`.
  - `dataset_id`: dataset ID or `null`.
  - `model_id`: model ID or `null`.
  - `source_scope`: `"RUN"`, `"CERTIFICATION_EXPERIMENT"`, or `"UNAVAILABLE"`.
  - `status`: `"AVAILABLE"`, `"TRIAL_DETAILS_NOT_PERSISTED"`, or `"NOT_AVAILABLE_FOR_RUN"`.
- Active runs without scientific blocks in the requested domain return `status: "NOT_AVAILABLE_FOR_RUN"` and `source_scope: "UNAVAILABLE"` without falling back to certification bundle.

### G5 — Future Run Detail Persistence Without Web Recomputation
- **Invariants**: `NEW_RUN_DETAIL_PERSISTENCE = 1`, `WEB_LAYER_SCIENTIFIC_RECOMPUTATION = 0`.
- Background run execution (`_execute_run_in_background`) extracts test evidence records and populates `scientific_presentation` in the presentation model.
- Detail endpoints read directly from the completed run presentation model without performing any runtime scientific recomputations.

### G6 — Explicit Discovery Provenance Tracking
- **Invariants**: `SILENT_DISCOVERY_FALLBACK = 0`.
- `GET /api/v1/data/providers/{provider}/datasets`:
  - Returns `discovery_source`: `"remote"`, `"curated_fallback"`, `"local_filesystem"`, or `"unavailable"`.
  - Returns `remote_query_attempted: bool`.
  - Returns `remote_query_succeeded: bool`.
  - Returns `remote_error_category: str | null`.
  - Returns `fetched_at: str` (ISO-8601 UTC timestamp).

### G7 — Automatic Certification Cache Invalidation
- **Invariants**: `STALE_CERTIFICATION_AFTER_ARTIFACT_CHANGE = 0`.
- `CertificationBundleCache` monitors disk modification times (`mtime`) and file sizes across all 14 artifacts in `scratch/scientific_certification/`.
- If any artifact changes on disk, `ensure_loaded()` invalidates the cache and automatically re-reads the artifacts and updates the bundle hash on the next request without process restart.
- Exposes `loaded_at` (ISO-8601 UTC) in `GET /api/v1/certification`.

