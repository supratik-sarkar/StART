# Porting `webapp/` to Another Backend

The `webapp/` workspace is designed to be completely standalone and backend-portable. A firm or private enterprise can copy `webapp/` into any infrastructure and connect it to internal model evaluation, risk, or compliance engines by implementing a single TypeScript adapter satisfying `StartBackend`.

## Architecture & Separation of Concerns

```
UI Feature Layer (Composer, ExecutionPath, EvidenceExplorer, LineageGraph, Signoff)
       │
       ▼
Contracts Boundary (src/contracts/backend.ts, types.ts, validators.ts)
       │
       ▼
Adapter Implementation:
  - Public: PublicStARTBackend.ts (FastAPI / REST / SSE)
  - Firm:   FirmStARTBackendAdapter.ts (Internal GRPC / Enterprise REST / WebSocket)
```

No Cloudflare Worker, Oracle Cloud, Python internals, or public model mirrors are compiled into the React feature components.

## Implementation Steps for `FirmStARTBackendAdapter`

### 1. Implement the `StartBackend` Interface

Create `src/adapters/firm/FirmStARTBackendAdapter.ts` implementing `StartBackend` (`src/contracts/backend.ts`):

```typescript
export interface StartBackend {
  getCapabilities(): Promise<Capability[]>
  listExecutionContexts(): Promise<ExecutionContext[]>
  createPlan(request: RunRequest): Promise<AgentPlanPreview>
  createRun(request: RunRequest): Promise<RunSnapshot>
  getRun(runId: string): Promise<RunSnapshot>
  streamRun(runId: string, onEvent: (event: RuntimeEvent) => void, onError?: (error: Error) => void): StreamSubscription
  getExecutionGraph(runId: string): Promise<ExecutionGraph>
  getEvidence(runId: string): Promise<EvidenceRecord[]>
  getFindings(runId: string): Promise<Finding[]>
  getArtifacts(runId: string): Promise<ArtifactRecord[]>
  askAgent(runId: string, message: string, contextNodeId?: string): Promise<ConversationMessage>
  validateAction(runId: string, action: ProposedAction): Promise<ProposedAction>
  submitHumanAction(runId: string, action: ProposedAction): Promise<RunSnapshot>
  submitReviewerOutput(runId: string, review: ReviewerOutput): Promise<ReviewerGateResult>
  getGovernance(runId: string): Promise<GovernanceState | null>
  getAttestation(runId: string): Promise<AttestationState | null>
}
```

### 2. Runtime Schema Validation

Use `src/contracts/validators.ts` at the boundary of your adapter to validate response shapes:
- `validateCapabilities`
- `validateExecutionContexts`
- `validateAgentPlanPreview` (prevents fabricating fake run IDs during planning)
- `validateRunSnapshot`
- `validateRuntimeEvent`
- `validateEvidenceRecords`
- `validateFindings`
- `validateArtifactRecords`
- `validateExecutionGraph`
- `validateGovernanceState`
- `validateAttestationState`
- `validateProposedAction`
- `validateReviewerGateResult`

### 3. Reviewer Runtime Decoupling

The qualitative review runtime is decoupled via `ReviewerRuntime` (`src/contracts/reviewer.ts`).
- Public deployment uses `WebLLMReviewer.ts` (client-side in-browser WebGPU model).
- A private firm deployment can replace this with `FirmReviewerRuntime` (calling an internal enterprise LLM gateway or hosted inference endpoint) without altering UI components.

### 4. Wire the Adapter at the Root

In `src/main.tsx` or `src/app/App.tsx`, instantiate your adapter:

```typescript
const backend = new FirmStARTBackendAdapter({ baseUrl: "https://risk-api.firm.internal" })
```

## Core Invariants to Preserve

1. **Truthful Capabilities:** Return `enabled: false` with an explicit `disabledReason` for unsupported workflows. Do not enable surfaces that lack deterministic backing.
2. **Dedicated Plan Preview:** `createPlan()` must return `AgentPlanPreview` and must not manufacture a run ID. Real run IDs begin with `createRun()`.
3. **Deterministic Validation Boundary:** `validateAction()` must sanitize, validate, and bound any candidate parameters before human approval or execution.
4. **Parent-Child Lineage:** When `submitHumanAction()` is called, the child run must link to `parentRunId`.
5. **Sandboxed Artifacts:** Artifact endpoints must prevent path traversal and return typed previews.
