# Runtime Event Model

Events must be ordered and replayable enough to reconstruct the visible run state.

Minimum event identity:
- `eventId`
- `sequence`
- `runId`
- `timestamp`
- `type`
- optional `nodeId` / `parentNodeId`

Progress is authoritative only when emitted by backend work units.

Examples:
- tuning: completed trials / total trials
- DL: epoch/batch counters
- test plans: completed tests / total compatible tests
- scenario sweeps: completed cases / total cases

If the backend cannot truthfully supply a percentage, omit `percent`; the UI displays phase/state rather than fabricating one.
