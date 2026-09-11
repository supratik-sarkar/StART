# StART frontend implementation

Implemented directly in `webapp/src`. No commit, push, deployment, backend edit, contract edit, adapter edit, or `useWorkbench` edit.

## Presentation boundary

`features/investigation/presentation.ts` reads the existing public GET `/api/v1/runs/{runId}/presentation` endpoint. It checks run identity and aborts stale requests. The fallback projects existing artifact dictionaries into presentation sections. It supplies no configuration defaults, classifications, metrics, synthetic curves, or scientific calculations. Chart arithmetic scales source observations to screen coordinates only. Canonical model checks and execution evidence remain separately labeled because their scopes differ.

`dev/fixtures.ts` maps field names from the supplied design pack into existing UI contracts without replacing source numeric values. Only the lazy, development-gated `/dev/ux-preview` imports it. Production assets were checked for fixture identities and pack references; none were present.

## Implemented experience

- DEFINE: full-width workflow, context and objective; context EDA; review-plan handoff.
- EXECUTE: real plan stages, actual status and latest evidence; secondary trace.
- INVESTIGATE: one family-specific Analysis navigation and one Review group; completed editorial overview; analytical-object-only inspector.
- Predictive: data, exact resolved configuration, ROC, supplied confusion counts, calibration, importance and validation.
- Deep learning: supplied architecture/layers, training and validation observations, checkpoint and configuration.
- Recommenders: MF/NCF/FM-specific configuration, interaction protocol, ranking/rating, beyond-accuracy and cold-start sections.
- Portfolio: allocations, risk contributions, sensitivity and technique-specific sections; HRP supplied matrices and SVG dendrogram, source linkage table. Missing outputs remain explicit.
- Findings, evidence, artifacts, governance and provenance: linked records, readable disposition, raw details, copyable identities and hashes, contextual actions.
- History/replay/compare/search: date/status run ledger, read-only replay, grouped backend comparisons with incompatible-family rejection, grouped search and keyboard selection.
- Splitter: continuous pointer drag, persisted ratio, keyboard controls, double-click reset, collapse controls and independent scrolling. Fresh state contains no empty right pane.

## Preview states (33)

fresh, configured, eda, plan, execution, predictive, data, config, performance, explainability, validation, findings, evidence, governance, provenance, artifacts, history, replay, compare, dl, architecture, training, mf, ncf, fm, protocol, ranking, coldstart, hrp, minvar, erc, narrow, 1024.

## Verification

Final commands in `webapp`: `npm test` (47 tests across 5 files), `npm run typecheck`, and `npm run build` all passed. Build retains the large WebLLM dependency chunk warning.

Browser checks covered all 33 gallery states, absence of invalid display values and old duplicate navigation, page overflow, and splitter ratios 21/50/65/73 at widths 1024/1280/1440. Fresh, predictive, performance, deep-learning training, HRP and narrow layouts were visually inspected. Double-click reset and fresh state after collapse were checked.

Existing local backend was started unchanged. Production UI loaded persisted history and replayed RUN-WEB-67e64ecddd with canonical presentation, evidence, artifacts and governance. Live comparison of RUN-TEST-CMP-A-1788900596 and RUN-TEST-CMP-B-1788900597 showed backend parameter deltas; comparison with quantitative run RUN-TEST-CMP-C-1788900598 rejected incompatible workflows.

Hash verification against the pre-edit baseline found no changes in backend Python, contracts, adapters, `useWorkbench`, or design-pack files. No Python regression was run.

## Known limitations

- Existing backend findings can contain `Attention item: None`; existing contract normalization omits backend `testStatus`. The UI preserves source wording and explicitly reports absent severity instead of inventing it.
- Existing replay state reconstructs a short event trace; it does not supply the complete original event stream.
- Missing analytical outputs and fields discarded by immutable adapters are displayed as unavailable. No frontend inference fills them.
- Offline preview actions that would mutate runtime state are explicitly unavailable. Live approval, rerun execution, challenge submission and local browser AI generation were not invoked during verification.
- The dev gallery on port 4182 is offline. The existing backend CORS configuration does not permit that origin. Use the built same-origin application on port 8000 for live runtime interaction.
- Existing source artifact titles may contain legacy scientific terminology; titles are retained as supplied while analytical rendering uses actual provided data.

## Local preview

Development gallery: http://127.0.0.1:4182/dev/ux-preview
Built live application: http://127.0.0.1:8000/

## Files changed

- `webapp/src/main.tsx`
- `webapp/src/app/App.tsx`
- `webapp/src/features/artifacts/TypedArtifactRenderer.tsx`
- `webapp/src/features/eda/EdaCanvas.tsx`
- `webapp/src/features/composer/Composer.tsx`
- `webapp/src/features/canvas/ConfigurationCodeView.tsx`
- `webapp/src/features/search/CommandPalette.tsx`
- `webapp/src/features/execution/IterationModal.tsx`
- `webapp/src/features/history/RunHistoryModal.tsx`
- `webapp/src/features/compare/RunCompareView.tsx`
- `webapp/src/components/ResizableWorkspace.tsx`
- `webapp/src/features/investigation/ExecutionReview.tsx`
- `webapp/src/features/investigation/Science.tsx`
- `webapp/src/features/investigation/ModelAnalysis.tsx`
- `webapp/src/features/investigation/presentation.ts`
- `webapp/src/features/investigation/InvestigationWorkbench.tsx`
- `webapp/src/features/investigation/Science.test.tsx`
- `webapp/src/components/ReviewDialog.tsx`
- `webapp/src/components/WorkbenchHeader.tsx`
- `webapp/src/dev/fixtures.ts`
- `webapp/src/dev/preview.css`
- `webapp/src/dev/UxPreview.tsx`
- `webapp/src/design-system/science.css`
- `webapp/src/design-system/workstation.css`
- `webapp/docs/UX_REDESIGN_IMPLEMENTATION.md`
