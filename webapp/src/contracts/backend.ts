import type {
  AgentPlanPreview, ArtifactRecord, AttestationState, Capability, CheckpointRecord,
  DecisionReceipt, EdaProfile, EvidenceRecord, ExecutionContext, ExecutionGraph,
  Finding, GovernanceState, ProposedAction, QuestionResponse, ReviewerGateResult,
  RunCompareResult, RunHistoryItem, RunLineage,
  RunRequest, RunSnapshot, RuntimeEvent, ScenarioItem, StARTCapabilityManifest, TestCatalogItem
} from './types'
import type { ReviewerOutput } from './reviewer'

export interface StreamSubscription { close(): void }

export interface StartBackend {
  readonly adapterName: string
  readonly adapterMode: 'demo' | 'public' | 'firm'
  getCapabilities(): Promise<Capability[]>
  getCapabilityManifest?(): Promise<StARTCapabilityManifest>
  listExecutionContexts(): Promise<ExecutionContext[]>
  createPlan(request: RunRequest): Promise<AgentPlanPreview>
  createRun(request: RunRequest): Promise<RunSnapshot>
  getRun(runId: string): Promise<RunSnapshot>
  getRunEvents?(runId: string): Promise<RuntimeEvent[]>
  streamRun(runId: string, onEvent: (event: RuntimeEvent) => void, onError?: (error: Error) => void): StreamSubscription
  getExecutionGraph(runId: string): Promise<ExecutionGraph>
  getEvidence(runId: string): Promise<EvidenceRecord[]>
  getFindings(runId: string): Promise<Finding[]>
  getArtifacts(runId: string): Promise<ArtifactRecord[]>
  getCheckpoints?(runId: string): Promise<CheckpointRecord[]>
  recordDecision?(runId: string, decision: Partial<DecisionReceipt>): Promise<DecisionReceipt>
  askQuestion?(runId: string, query: { question: string; targetStage?: string; evidenceId?: string }): Promise<QuestionResponse>
  validateAction?(runId: string, action: ProposedAction): Promise<ProposedAction>
  submitHumanAction(runId: string, action: ProposedAction): Promise<RunSnapshot>
  submitReviewerOutput?(runId: string, review: ReviewerOutput): Promise<ReviewerGateResult>
  getGovernance(runId: string): Promise<GovernanceState | null>
  getAttestation(runId: string): Promise<AttestationState | null>
  listScenarios?(): Promise<ScenarioItem[]>
  getScenarioEda?(scenarioId: string): Promise<EdaProfile>
  getContextEda?(contextId: string): Promise<EdaProfile>
  listTests?(): Promise<TestCatalogItem[]>
  listRuns?(query?: { workflow?: string; status?: string; domain?: string }): Promise<RunHistoryItem[]>
  compareRuns?(runA: string, runB: string): Promise<RunCompareResult>
  getRunLineage?(runId: string): Promise<RunLineage>
  searchGlobal?(query: string): Promise<Array<{ id: string; category: string; title: string; subtitle?: string; data?: any }>>
}

