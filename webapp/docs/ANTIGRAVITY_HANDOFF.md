# Antigravity wiring handoff

This package is a **greenfield frontend product**. Treat the visual implementation as frozen unless a wiring defect requires a tiny compatibility adjustment.

## Required work

1. Move/copy `webapp/` into `${START_ROOT}/webapp`.
2. Do not use the legacy `web/` as design reference.
3. Reconcile `PublicStARTBackend` against canonical StART web schemas/routes.
4. Add only thin backend transport endpoints needed for:
   - capabilities
   - execution contexts
   - plan
   - run snapshot
   - ordered events
   - execution graph/lineage
   - evidence
   - findings
   - artifacts
   - agent messages/actions
   - governance
   - attestation
5. Do not implement science in TypeScript or web routes.
6. Keep `DemoBackend` for frontend visual tests, but production must use `PublicStARTBackend`.
7. Wire production adapter through environment configuration.
8. Validate one complete real ML journey before expanding acceptance.
9. Sanitize and publish only after real backend evidence flows through the new UI.

## Never do

- reintroduce old `web/` styling/layout;
- add fake analytics to satisfy components;
- hardcode run/evidence IDs;
- invent progress;
- expose cloud topology in components;
- make WebLLM block deterministic execution.

## Mandatory pre-wiring read order

1. `TERMINAL_TO_VISUAL_MAPPING.md`
2. `BACKEND_ADAPTER.md`
3. `EVENT_MODEL.md`
4. `BROWSER_AI.md`
5. `BACKEND_GAP_CHECKLIST.md`
6. `PORTING_TO_ANOTHER_BACKEND.md`

## Wiring acceptance principle

Do not declare the new webapp wired because it renders. The representative acceptance path is:

`context → plan → run → ordered events → real tool/test nodes → real evidence → contextual question → explicit child action → child lineage → Browser AI over current evidence → server gating → governance → attestation → reverse trace from sign-off`.
