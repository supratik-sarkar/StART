# Backend Adapter Contract

Implement `StartBackend` in `src/contracts/backend.ts`.

The intended firm-portability model is:

```text
React UI → StartBackend → PublicStARTBackendAdapter
                     ↘ FirmStARTBackendAdapter (later)
```

## Required capability families

- capabilities / workflow availability
- execution contexts (datasets/models/portfolios)
- plan creation
- run creation/snapshot
- ordered event stream
- execution graph
- EvidenceRecords
- Findings
- Artifacts
- contextual agent messages
- human action submission / child run creation
- governance
- attestation

## Important integration rule

Do not rewrite UI components to match backend payloads. Normalize backend payloads inside the adapter into the contracts defined in `src/contracts/types.ts`.

## Public wiring notes for Antigravity

The `PublicStARTBackend` contains intentionally generic route names. Reconcile them against actual StART route/schema contracts. If a backend surface is missing, prefer a thin transport/presentation endpoint over duplicating analytical logic.
