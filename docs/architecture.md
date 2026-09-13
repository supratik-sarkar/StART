# StART Architecture

## Layers and responsibilities

| Layer | Modules | Responsibility | What it must never do |
|---|---|---|---|
| Agents | `start.agents` | Plan, route, execute (orchestrate), critique, narrate | Compute or invent metrics |
| Test registry | `start.registry`, `start.tests.*` | Pure deterministic engines returning typed `TestResult` | Call an LLM; mutate inputs |
| Evidence | `start.evidence` | Content-addressed store + hash-chained ledger | Allow in-place mutation |
| Providers | `start.providers` | Compute / data / experiment / LLM / storage abstractions | Hard-depend on enterprise systems |
| Orchestration | `start.orchestration.pipeline` | Wire the above into `run_review` | Skip the policy guard or critic |
| Reporting | `start.reporting` | Render reviewer-facing Markdown from `RunResult` | Add numbers not in evidence |

## Run data flow

```
StartConfig + PolicyConfig (hashed)
        │
ReviewPlannerAgent ──► ValidationPlan
        │
PolicyGuardAgent ──► PolicyDecision (block ⇒ stop; hash stamped downstream)
        │
TestRouterAgent ──► registered tests only (unknown IDs dropped)
        │
ExecutionAgent ──► ComputeProvider.run(engine, ctx, **params)
        │              └─ engine exception ⇒ ERROR EvidenceRecord (traceback kept,
        │                 metrics never substituted)
        ▼
EvidenceRecord[]  (ids, metrics, thresholds, status, interpretation,
        │          input-data hash, policy hash, git SHA, repro meta)
        │
EvidenceCriticAgent ──► completeness critique
        │
NarrativeAgent ──► proof-carrying Narrative
        │
EvidenceCriticAgent ──► citation critique
        │              └─ blocked LLM narrative ⇒ replaced by template narrative
        ▼
EvidenceLedger.append (chained JSONL) + ContentAddressedStore.put + experiment logging
        ▼
RunResult ──► reporting.render_markdown
```

## Invariants (enforced by tests)

1. **Determinism** — engines are pure functions of `(ctx, params)`; Hypothesis
   verifies row-order invariance of metrics and of `hash_dataframe`.
2. **Proof-carrying narratives** — every sentence containing a number (after
   stripping identifier tokens) must carry an `[EV-…]` citation that resolves
   to a real record; the critic blocks violations, and the template narrative
   satisfies this by construction.
3. **Tamper evidence** — `entry_hash_n = sha256(entry_hash_{n-1} + record_hash_n)`
   with a zero genesis; editing any historical record breaks `verify()`.
4. **Safe degradation** — missing torch ⇒ CPU; missing MLFlow ⇒ local JSONL
   tracking; unreachable LLM ⇒ `NoLLMProvider`; missing columns ⇒ SKIPPED
   evidence, never a crash.
5. **Policy traceability** — the active policy YAML's content hash is stamped
   into every evidence record.

## Compute routing

`detect_device()` order: CUDA → MPS → CPU (torch optional; never raises).
`is_databricks_runtime()` checks `DATABRICKS_RUNTIME_VERSION` and is
independent of device detection. `get_compute_provider()` resolves
`(mode, device, distributed_backend)`:

- `databricks_spark` → Databricks stub providers (in-process in v0.1)
- `ray` / `dask` → declared, raise `DistributedBackendNotImplemented` in v0.1
- otherwise local CPU/GPU with transparent CPU fallback

## Caching

Cache identity: `(test_id, input_artifact_hash, params_hash, policy_hash)`.
A record without a data hash is never cached. Hits return the prior
`EvidenceRecord` from the content-addressed store.

## Extension points

- `@register_test(test_id, family=...)` for in-repo engines
- `start.test_packs` entry points for external pip-installed packs
- New providers: subclass the relevant ABC in `start.providers.base`
- Private enterprise integrations implement public interfaces
  (`enterprise_llm_gateway`, `SnowflakePlaceholderProvider`) outside this repo
