# StART Greenfield Webapp Architecture

This folder is intentionally portable. UI components depend on **domain contracts**, never Cloudflare, Oracle, HMAC, deployment IPs, or public-demo routes.

## Core flow

`Composer → Agent Plan → Run → Runtime Events → Execution Graph → Evidence → Conversation/Actions → Governance → Attestation`

`src/contracts/` defines stable frontend-domain types. `src/contracts/backend.ts` defines the `StartBackend` port. Infrastructure-specific knowledge belongs only in adapters.

## Current adapters

- `DemoBackend`: deterministic visual-development adapter; it exists so the package can be reviewed before StART backend wiring. It is explicitly labeled in the UI and must not be used for public truth claims.
- `PublicStARTBackend`: wiring target for the public repo. Endpoint paths are deliberately isolated here and should be reconciled against canonical StART routes by Antigravity.

## Visual architecture

The webapp has three synchronized models:
1. **Living execution path** — what is happening now and what comes next.
2. **Parent/child lineage** — why each tool/test/evidence object exists and what it created.
3. **Contextual conversation** — evidence explanation or explicit proposed deterministic actions.

All three share selected runtime/evidence identity.

## Truth rules

- React never computes scientific truth.
- Missing progress remains missing; components must not invent percentages.
- Findings come from backend state, not frontend copy.
- Evidence IDs must be backend identities.
- Human/agent actions become lineage, not hidden chat side effects.
