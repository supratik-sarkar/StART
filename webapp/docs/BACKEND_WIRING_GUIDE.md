# Backend Wiring Guide

This guide defines the technical wiring boundary for the StART webapp. Integration work is confined to backend transport, presentation support, sanitation, and publication readiness; scientific calculations remain backend-owned.

## Architecture boundary

The frontend does not derive analytical truth. Existing StART objects remain authoritative, including the capability registry, `ReviewPresentationModel`, `RuntimeEvent`, `EvidenceRecord`, `ArtifactRecord`, lineage, reviewer hydration and gating, governance, OPA decisions, attestation, and execution contexts.

Where a frontend contract has no route, the supported extension is a thin transport or presentation endpoint over canonical state. New analytical mathematics and frontend recomputation are outside this boundary.

## Adapter boundary

Only `webapp/src/adapters/public/PublicStARTBackend.ts` contains public transport knowledge. It normalizes backend payloads into `webapp/src/contracts/`. React feature components remain infrastructure-agnostic and contain no Cloudflare, Oracle, HMAC, origin, or deployment-address logic.

## Representative integrated journey

A release-ready integration demonstrates the following path with real identities and ordered state:

`context selection → goal → visible plan → run creation → runtime events → progress → tool/test nodes → branch formation → EvidenceRecords → evidence detail → contextual question → proposed deterministic action → child run → parent/child lineage → browser review → server evidence validation → OPA → governance → attestation → sign-off → reverse trace`.

Deep-learning and quantitative-finance entry points remain secondary smoke-test surfaces rather than default navigation.

## Browser review boundary

Browser review is asynchronous and non-blocking. Deterministic execution never waits for model loading. Evidence identifiers remain citations until validated and hydrated by the server, numerical values remain untrusted until hydration, and unavailable browser inference fails gracefully.

## Verification and publication

Verification is finite and focused: diagnose a failed gate once, make one narrow correction, and rerun the affected gate. Publication follows real integration acceptance, privacy and credential scanning, manifest consistency, build, type checking, tests, and continuous-integration success.

The release record distinguishes real backend-bound capabilities from remaining adapter gaps and records the exact run and evidence identities used for acceptance.
