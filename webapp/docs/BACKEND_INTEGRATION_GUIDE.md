# Backend Integration Guide

The webapp is a portable frontend product. Backend integration binds canonical StART contracts while preserving the distinction between scientific evidence, governance, and policy.

## Integration boundary

The production adapter is `PublicStARTBackend`. It normalizes canonical backend payloads into the contracts under `webapp/src/contracts/`; React components remain independent of deployment infrastructure.

The integration surface includes:

- capabilities and execution contexts;
- planning and run snapshots;
- ordered runtime events and execution lineage;
- evidence, findings, and artifacts;
- contextual agent messages and explicit actions;
- governance and attestation.

Scientific calculations remain in the Python runtime and thin web transport. The frontend does not invent metrics, progress, findings, evidence identities, or analytical outcomes. `DemoBackend` remains available only for clearly labeled visual tests.

## Required reference order

1. `TERMINAL_TO_VISUAL_MAPPING.md`
2. `BACKEND_ADAPTER.md`
3. `EVENT_MODEL.md`
4. `BROWSER_AI.md`
5. `BACKEND_GAP_CHECKLIST.md`
6. `PORTING_TO_ANOTHER_BACKEND.md`

## Acceptance principle

Rendering alone does not establish a complete integration. The representative path is:

`context → plan → run → ordered events → real tool/test nodes → real evidence → contextual question → explicit child action → child lineage → browser review over current evidence → server gating → governance → attestation → reverse trace from sign-off`.

The visual baseline, runtime truth boundaries, and evidence identities remain preserved throughout this path.
