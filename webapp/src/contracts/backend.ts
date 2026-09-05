import type {
  AgentPlanPreview, ArtifactRecord, AttestationState, Capability, EvidenceRecord,
  ExecutionContext, ExecutionGraph, Finding, GovernanceState, ProposedAction,
  ReviewerGateResult, RunRequest, RunSnapshot, RuntimeEvent
} from './types'
import type { ReviewerOutput } from './reviewer'

export interface StreamSubscription { close(): void }

export interface StartBackend {
  readonly adapterName: string
  readonly adapterMode: 'demo' | 'public' | 'firm'
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
  validateAction?(runId: string, action: ProposedAction): Promise<ProposedAction>
  submitHumanAction(runId: string, action: ProposedAction): Promise<RunSnapshot>
  submitReviewerOutput?(runId: string, review: ReviewerOutput): Promise<ReviewerGateResult>
  getGovernance(runId: string): Promise<GovernanceState | null>
  getAttestation(runId: string): Promise<AttestationState | null>
}
