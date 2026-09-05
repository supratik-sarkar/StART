# ANTIGRAVITY — WIRE THE GREENFIELD StART WEBAPP, DO NOT REDESIGN IT

A new greenfield frontend package has been placed at:

`/Users/supratiksarkar/Desktop/StART/webapp`

It replaces the previous frontend product direction.

## HARD OWNERSHIP BOUNDARY

The new `webapp/` visual/product design is FROZEN.

Do NOT:
- use the legacy `web/` frontend as design reference;
- copy legacy components/styles/layout;
- redesign cosmetics;
- add dashboard/KPI layouts;
- reintroduce Market-first navigation;
- add fake plots, metrics, progress, findings or Evidence IDs;
- calculate analytical truth in TypeScript;
- use `manage_task`;
- launch arbitrary interactive StART CLI flows;
- enter recursive remediation loops.

Your job is **backend wiring + transport/presentation support + sanitization + publication**.

## READ FIRST

Read, in order:
1. `webapp/DESIGN_FREEZE.md` if copied there, otherwise the package-level design freeze
2. `webapp/docs/TERMINAL_TO_VISUAL_MAPPING.md`
3. `webapp/docs/ARCHITECTURE.md`
4. `webapp/docs/BACKEND_ADAPTER.md`
5. `webapp/docs/EVENT_MODEL.md`
6. `webapp/docs/BROWSER_AI.md`
7. `webapp/docs/BACKEND_GAP_CHECKLIST.md`
8. `webapp/docs/PORTING_TO_ANOTHER_BACKEND.md`
9. `webapp/docs/ANTIGRAVITY_HANDOFF.md`

## FIRST ACTION — BACKEND GAP MAP

Before editing backend code, inspect canonical StART and produce a mapping:

`webapp contract → existing backend object/route → gap if any`

Use real StART objects wherever available:
- registry / 79 deterministic surfaces
- ReviewPresentationModel
- RuntimeEvent
- EvidenceRecord
- ArtifactRecord
- parent/child lineage
- agent orchestration
- reviewer hydration/gating
- OPA / governance
- attestation
- execution contexts / synthetic public worlds

If a webapp contract lacks an endpoint, add only a THIN transport/presentation endpoint over existing canonical state.

Do NOT create new analytical mathematics merely to satisfy the UI.

## PUBLIC ADAPTER

Wire only:
`webapp/src/adapters/public/PublicStARTBackend.ts`

React feature components must remain infrastructure-agnostic.

Normalize canonical backend payloads into the contracts under:
`webapp/src/contracts/`

Do not put Cloudflare/Oracle/HMAC/IP details inside React components.

## REQUIRED REAL JOURNEY

Before publication prove one complete real Predictive ML journey through the new webapp:

context selection
→ goal
→ visible agent plan
→ create run
→ real ordered runtime events
→ real progress
→ real tool/test nodes
→ real branch formation
→ real EvidenceRecords
→ evidence drill-down
→ contextual agent question
→ explicit proposed deterministic action
→ child run
→ parent→child lineage
→ Browser AI over current real EvidenceRecords
→ server Evidence-ID validation/hydration
→ OPA
→ governance
→ attestation
→ final sign-off
→ reverse trace from sign-off to evidence/test/parent path

Then smoke-test Deep Learning and Quantitative Finance entry points without making them default.

## BROWSER AI

Preserve current accepted browser-reviewer architecture where compatible:
- asynchronous/non-blocking
- deterministic run never waits for model load
- WebGPU/WebLLM reviewer
- EvidenceRecord input
- Evidence ID citations
- numeric values remain untrusted until server hydration
- graceful failure if browser AI unavailable

Integrate Browser AI into the contextual Agent ↔ Human surface without turning it into a generic chatbot.

## ACCEPTANCE DISCIPLINE

All work finite and foreground-bounded.
For a failed gate: one diagnosis → one narrow fix → one rerun; otherwise STOP.
Do not repeatedly rerun full suites.
Run one final full regression only after focused wiring gates are green.

## PUBLICATION

Only after real wiring acceptance:
- privacy/secret scan
- ensure `webapp/` contains no developer-private paths
- manifest sync to protected Git tree
- build/typecheck/tests
- GitHub Actions green
- commit/tag release

Do not rewrite root README unless explicitly instructed separately.

Final report should distinguish:
- real backend-wired capabilities
- any remaining adapter TODOs
- exact run IDs/evidence IDs used for acceptance
- webapp build result
- privacy result
- CI result
- commit/tag

Then STOP.
