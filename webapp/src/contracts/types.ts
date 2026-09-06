export type WorkflowId =
  | 'predictive_ml'
  | 'deep_learning'
  | 'data_diagnostics'
  | 'model_diagnostics'
  | 'calibration'
  | 'robustness'
  | 'explainability'
  | 'hyperparameter_tuning'
  | 'model_comparison'
  | 'quantitative_finance'

export type RunPhase =
  | 'idle' | 'configuring' | 'validating' | 'queued' | 'planning'
  | 'running' | 'waiting_human' | 'waiting_ai' | 'partial'
  | 'completed' | 'recoverable_error' | 'failed' | 'reconnecting'

export type NodeKind =
  | 'context' | 'agent' | 'tool' | 'test' | 'evidence' | 'finding'
  | 'human' | 'governance' | 'attestation' | 'artifact'

export type NodeStatus = 'future' | 'queued' | 'running' | 'completed' | 'attention' | 'failed' | 'waiting'

export interface Capability {
  id: WorkflowId
  label: string
  description: string
  category: 'ml' | 'quant'
  enabled: boolean
  disabledReason?: string
  icon?: string
}

export interface AgentPlanPreview {
  workflowId: WorkflowId
  contextId: string
  goal: string
  plan: AgentPlanStep[]
  requiredInputs?: string[]
  warnings?: string[]
}

export interface ExecutionContext {
  id: string
  label: string
  kind: 'dataset' | 'model' | 'portfolio' | 'synthetic-world'
  description: string
  provenance: string
  shape?: string
  target?: string
  seed?: number
  badges?: string[]
}

export interface AgentPlanStep {
  id: string
  label: string
  description?: string
  kind: NodeKind
  status: NodeStatus
  parentId?: string
}

export interface RunRequest {
  workflowId: WorkflowId
  contextId: string
  goal: string
  parameters: Record<string, string | number | boolean>
  parentRunId?: string
  sourceEvidenceId?: string
  intervention?: string
}

export interface RunSnapshot {
  runId: string
  workflowId: WorkflowId
  contextId: string
  goal: string
  phase: RunPhase
  statusLabel: string
  startedAt: string
  updatedAt: string
  elapsedMs: number
  progress?: ProgressState
  plan: AgentPlanStep[]
  parentRunId?: string
  sourceEvidenceId?: string
}

export interface ProgressState {
  label: string
  completed?: number
  total?: number
  percent?: number
  detail?: string
  etaSeconds?: number
}

export interface RuntimeEvent {
  eventId: string
  sequence: number
  runId: string
  timestamp: string
  type:
    | 'phase'
    | 'tool_started'
    | 'tool_completed'
    | 'test_completed'
    | 'evidence_created'
    | 'evidence_commit'
    | 'finding_created'
    | 'human_required'
    | 'governance'
    | 'governance_seal'
    | 'artifact_created'
    | 'attested'
    | 'progress'
    | 'agent_transition'
    | 'tool_execution'
    | 'run_completed'
    | 'complete'
  nodeId?: string
  parentNodeId?: string
  title: string
  message: string
  status: NodeStatus
  progress?: ProgressState
  evidenceIds?: string[]
  artifactIds?: string[]
  metadata?: Record<string, unknown>
}

export interface EvidenceRecord {
  evidenceId: string
  runId: string
  testId: string
  title: string
  status: 'RECORDED' | 'PASS' | 'FAIL' | 'ATTENTION' | 'NOT_APPLICABLE'
  metrics: Array<{ name: string; value: number | string | boolean | null; unit?: string; criterion?: string }>
  provenance: string[]
  parentNodeId?: string
  createdAt: string
  summary?: string
}

export interface Finding {
  findingId: string
  runId: string
  title: string
  summary: string
  evidenceIds: string[]
  sourceNodeId?: string
  severity?: 'info' | 'attention' | 'critical'
  limitations?: string[]
  availableActions: Array<'explain' | 'challenge' | 'deeper_test' | 'compare' | 'change_parameter' | 'rerun'>
}

export interface ArtifactRecord {
  artifactId: string
  runId: string
  label: string
  kind: 'plot' | 'table' | 'json' | 'pdf' | 'report' | 'attestation'
  mimeType: string
  createdAt: string
  description?: string
  preview?: { type: 'text' | 'key-value'; payload: unknown }
}

export interface ExecutionGraphNode {
  id: string
  runId: string
  kind: NodeKind
  label: string
  status: NodeStatus
  parentId?: string
  subtitle?: string
  evidenceIds?: string[]
  artifactIds?: string[]
  durationMs?: number
  observed?: boolean
}

export interface ExecutionGraphEdge {
  id: string
  source: string
  target: string
  relation: 'next' | 'branch' | 'creates' | 'supports' | 'challenges' | 'rerun'
  edgeKind?: 'planned' | 'observed'
}

export interface ExecutionGraph {
  nodes: ExecutionGraphNode[]
  edges: ExecutionGraphEdge[]
}

export interface ConversationMessage {
  id: string
  role: 'human' | 'agent' | 'system'
  timestamp: string
  text: string
  contextNodeId?: string
  evidenceIds?: string[]
  proposedAction?: ProposedAction
}

export interface ProposedAction {
  actionId: string
  label: string
  description: string
  kind: 'challenge' | 'deeper_test' | 'compare' | 'rerun' | 'change_parameter'
  sourceNodeId?: string
  sourceEvidenceId?: string
  parameters?: Record<string, string | number | boolean>
}

export interface GovernanceState {
  disposition: string
  policyDecision?: string
  rationale?: string
  evidenceCoverage?: number
  unresolvedItems?: string[]
}

export interface AttestationState {
  merkleRoot: string
  createdAt: string
  evidenceCount: number
  artifactCount: number
  reproducibilityId?: string
}

export interface ReviewerGateResult {
  runId: string
  modelName: string
  hydratedFindings: Array<{
    findingId: string
    title: string
    grounded: boolean
    evidenceRefs: Array<{
      evidenceId: string
      metricName: string
      status: string
      hydratedValue: unknown
      testId?: string
      recordStatus?: string
    }>
    recommendation?: string
  }>
  allGrounded: boolean
  governanceDisposition: string
  attestationSealMerkleRoot: string
}
