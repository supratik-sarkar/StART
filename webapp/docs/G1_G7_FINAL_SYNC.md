# G1–G7 frontend synchronization — 2026-09-10

Frontend synchronization is complete within the requested frozen design and truthful-unavailable policy. This supersedes the production binding limitations in FINAL_LIVE_API_BINDING.md where the updated APIs now supply fields. No backend source files changed, verified against a pre-change SHA-256 baseline. No large certification execution or redesign occurred.

G1: Fetch every exact domain/experiment/data_layer from the live scopes index. Join only matching experiment identities, validating selected_experiment_id and selected_data_layer. Golden random-forest champion 0.9780027611596871 remains separate from real gradient-boosting champion 0.9830450251502884. Real challenger aggregates now appear. Dataset metadata comes from the live domain responses and matches run dataset/revision/fingerprint.

G2: Dataset and sample fingerprint fields render separately. Adult resolve reports 32561 available rows, full dataset fingerprint ddfbd5c16c58c371be86359f92f2fed1184fb78540d56f9f5daeacb83243639d, a distinct 50-row preview fingerprint, and full scope. The 100-row pre-certification reports sample scope with its own sample fingerprint. Certification reports 1000 recorded evaluation rows; these are not relabeled as total available rows.

G3: The request helper supports the opaque X-Provider-Session-ID header without URL or storage propagation. No credential entry is offered, using the brief's explicit fallback. The existing provider panel says Local / trusted backend session and explains the backend capability; runtime renders credential_scope=ephemeral_backend_session and multi_tenant_isolation_verified=false. No secret/session was created for the public-data check. Authenticated ingestion was not tested.

G4: All five granular presentations show requested/source run IDs, experiment, certification, dataset, model, source_scope, and status. Numerical observations require requested_run_id=source_run_id=selected run and source_scope=RUN. Other-run and certification-experiment responses are suppressed as selected-run science. NOT_AVAILABLE_FOR_RUN stays explicit. Unknown-run HTTP errors do not fall back to certification.

G5: One targeted correction added reads of the main persisted presentation, whose scientific_presentation.run_id must match before its kind-specific observations render. Fresh run RUN-WEB-d912c8a550 completed deterministically on the synthetic predictive benchmark, with 47 evidence records and 15 artifacts. Tuning had matching RUN identity and TRIAL_DETAILS_NOT_PERSISTED. XAI and sensitivity had matching RUN identity and AVAILABLE; the persisted details/measurements objects were empty. Sensitivity responses remained RESPONSES_NOT_PERSISTED. Portfolio and scenario returned NOT_AVAILABLE_FOR_RUN, verified via their live endpoints. No missing historical science was reconstructed.

G6: Hugging Face discovery returned remote, remote_query_attempted=true, remote_query_succeeded=true, no error category, and fetched_at=2026-09-10T11:04:31.261617+00:00. All five provenance fields are shown unchanged for remote, curated_fallback, local_filesystem, or unavailable responses.

G7: Refresh reads live APIs with cache:no-store, verifies the bundle hash again after loading, and displays certification_id, generated_at, loaded_at, fetched_at, and bundle_hash. Verified loaded_at=2026-09-10T11:02:48.152768+00:00. No artifact files were changed to force invalidation; automatic invalidation was inspected in the backend contract/source. No snapshot fallback.

## Bounded browser evidence

Fresh backend process served port 8000. Session SES-DATA-82d5ea8944 ingested 500 rows in 5 batches and reported 285473 bytes, one worker, 217 rows/second, and train/validation/test counts 349/74/77. COMPLETED and STOPPED were observed. Golden and real scopes, refresh freshness, fresh-run tuning/XAI/sensitivity, replay after reload, Evidence, Provenance, History, and Compare against RUN-WEB-db24008cf9 were verified. Browser error log was empty. The existing Porcelain layout was preserved; no CSS changes were needed.

Tests: 76 passed in 8 files. Typecheck and production build passed after the single correction pass.

## Remaining backend limitations, not pending permissions

- Fresh tuning trial details and numerical persisted XAI/sensitivity details were absent for the tested run. The frontend exposes supplied status and empty fields, not replacement calculations.
- Certification XAI still has an experiment-wide fallback in backend source when model/dataset matching fails. The frontend blocks that entire payload from selected-run observations because source_scope is CERTIFICATION_EXPERIMENT.
- Multi-tenant credential isolation remains unverified. Credential entry is intentionally unavailable as allowed by the synchronization brief.
- Existing scientific exclusions (including missing adapters/packages) remain backend-reported, unchanged.

These do not prevent the requested synchronization under its explicit unavailable-state acceptance path. No access approval is pending.

## Files changed

- webapp/src/features/certification/liveApi.ts
- webapp/src/features/certification/LiveDataRuntime.tsx
- webapp/src/features/certification/LiveRunPresentation.tsx
- webapp/src/features/certification/CertificationSurface.tsx
- webapp/tests/liveBinding.test.ts
- webapp/docs/G1_G7_FINAL_SYNC.md
- webapp/dist/** (generated build)

STOP.
