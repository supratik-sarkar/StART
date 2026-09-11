export type WorkflowId =
  | 'fraud_anomaly_aml'
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
  | 'recommender_system'

export type RunPhase =
  | 'idle' | 'configuring' | 'validating' | 'queued' | 'planning'
  | 'running' | 'waiting_human' | 'waiting_ai' | 'partial'
  | 'completed' | 'recoverable_error' | 'failed' | 'reconnecting'

export type NodeKind =
  | 'context' | 'agent' | 'tool' | 'test' | 'evidence' | 'finding'
  | 'human' | 'governance' | 'attestation' | 'artifact'

export type ExecutionMode = 'hybrid_workbench' | 'agentic_session' | 'deterministic_run'

export interface StARTCapabilityManifest {
  workbench: {
    name: string
    tagline: string
    version: string
    default_execution_mode: ExecutionMode
    deterministic_science_invariant: boolean
  }
  execution_modes: Array<{
    id: ExecutionMode
    label: string
    is_default: boolean
    description: string
  }>
  domains: Record<string, any>
  deferred_capabilities: Array<{
    capability: string
    domain: string
    status: string
    rationale: string
    activation_requirements: string
  }>
  ai_provider: {
    provider: string
    model: string
    status: 'ONLINE' | 'OFFLINE'
    source: string
    strict_zero_substitution: boolean
  }
}

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
  executionMode?: ExecutionMode
  agentProposal?: Record<string, unknown> | null
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
  executionMode?: ExecutionMode
  seed?: number
  workflowId: WorkflowId
  contextId: string
  goal: string
  parameters: Record<string, any>
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
    | 'checkpoint_committed'
    | 'checkpoint'
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
  title?: string
  kind: 'plot' | 'table' | 'json' | 'pdf' | 'report' | 'attestation' | 'metric' | 'evidence' | 'finding' | 'graph' | 'decision'
  artifactType?: string
  mimeType: string
  createdAt: string
  description?: string
  preview?: { type: 'text' | 'key-value'; payload: unknown }
  content?: any
  evidenceIds?: string[]
  dataFingerprint?: string
  producerNodeId?: string
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

export type ScenarioClassification =
  | 'CANONICAL_EXECUTION_CONTEXT'
  | 'SUPPORTED_DATA_SCENARIO'
  | 'GENERATOR_ONLY'
  | 'UNSUPPORTED'

export interface ScenarioItem {
  id: string
  label: string
  domain: string
  classification: ScenarioClassification
  compatible_context_id: string
  compatible_workflows: string[]
  shape: string
  rows: number
  features: number
  target: string
  categories: string[]
  registered_tests_count: number
  applicable_tests_count: number
  generator_identity: string
  seed: number
  description: string
  provenance_note: string
}

export interface TestCatalogItem {
  testId: string
  name: string
  family: string
  domain: string
  description: string
  contextType: string
  riskStripes: string[]
  riskDimensions: string[]
  requires: string[]
  objectKinds: string[]
}

export interface EdaProvenance {
  scenario_id: string
  context_id: string
  classification: string
  generator_identity: string
  seed: number
  input_shape: string
  profiling_operation: string
  sha256_fingerprint: string
  provenance_note: string
}

export interface EdaProfile {
  domain: string
  provenance: EdaProvenance
  overview?: {
    rows?: number
    columns?: number
    feature_count?: number
    target?: string
    missing_cells?: number
    duplicate_rows?: number
    memory_kb?: number
    assets?: number
    periods?: number
    frequency?: string
    factor_count?: number
    portfolio_weight_sum?: number
    max_single_holding_pct?: number
    annualized_volatility?: number
    mean_asset_correlation?: number
    observations?: number
    model?: string
    initial_rate_r0?: number
    terminal_rate?: number
    mean_rate?: number
    volatility?: number
    min_rate?: number
    max_rate?: number
    non_negative?: boolean
  }
  schema?: Array<{
    name: string
    dtype: string
    null_count: number
    null_pct: number
    distinct_count: number
    is_target: boolean
  }>
  target_distribution?: {
    target_column: string
    counts: Record<string, number>
    positive_ratio: number
    prevalence_pct: number
    total_samples: number
    is_binary: boolean
  }
  feature_moments?: Array<{
    feature: string
    mean: number
    std: number
    min: number
    q25: number
    median: number
    q75: number
    max: number
    skewness: number
    outlier_count: number
    outlier_pct: number
  }>
  correlation?: {
    columns: string[]
    matrix: number[][]
  }
  feature_target_correlations?: Array<{
    feature: string
    pearson_r: number
    abs_r: number
  }>
  data_profile?: EdaProfile
  model_fixture_context?: {
    architecture: string
    task: string
    family: string
    hidden_dims: number[]
    epochs: number
    batch_size: number
    learning_rate: number
    device: string
    splits: {
      train_samples: number
      validation_samples: number
      test_samples: number
      total_samples: number
    }
    tensor_shapes: Record<string, number[]>
    normalization: string
  }
  portfolio_composition?: Array<{
    asset: string
    portfolio_weight: number
    benchmark_weight: number
  }>
  factor_coverage?: Array<{
    factor: string
    mean_daily: number
    annualized_vol: number
    min: number
    max: number
  }>
  risk_summary?: {
    daily_mean_return: number
    daily_volatility: number
    annualized_volatility: number
    min_single_day_return: number
    max_single_day_return: number
    var_confidence_level: number
  }
  path_sample?: Array<{
    step: number
    rate: number
  }>
  classification_notice?: string
}

export interface CheckpointRecord {
  checkpoint_id: string
  checkpointId?: string
  name: string
  status: 'completed' | 'active' | 'running' | 'pending' | 'blocked' | 'human-pending'
  producing_stage: string
  producingStage?: string
  agent_signature: string
  agentSignature?: string
  evidence_ids: string[]
  evidenceIds?: string[]
  commit_hash: string
  commitHash?: string
  timestamp: string
  summary: string
}

export interface DecisionReceipt {
  receipt_id: string
  receiptId?: string
  run_id: string
  runId?: string
  action: 'ACCEPT' | 'QUESTION' | 'CHALLENGE' | 'OVERRIDE' | 'RERUN' | 'ESCALATE'
  target_stage?: string
  targetStage?: string
  target_checkpoint?: string
  targetCheckpoint?: string
  evidence_ids: string[]
  evidenceIds?: string[]
  rationale: string
  override_value?: unknown
  overrideValue?: unknown
  author: string
  decision_hash: string
  decisionHash?: string
  timestamp: string
  status: string
  immutable_evidence_preserved: boolean
}

export interface QuestionCitation {
  evidence_id: string
  evidenceId?: string
  test_id: string
  testId?: string
  status: string
  metrics: Record<string, unknown>
  snippet: string
}

export interface QuestionResponse {
  question: string
  answer: string
  citations: QuestionCitation[]
  targetStage?: string
  targetEvidenceId?: string
  receipt: DecisionReceipt
  timestamp: string
}

export interface HandoffTransition {
  eventId: string
  sourceAgent: string
  targetAgent: string
  stage: string
  action: string
  timestamp: string
  active: boolean
}

export interface RunHistoryItem {
  run_id: string
  session_id?: string
  workflow: string
  domain: string
  context_id: string
  created_at: number
  started_at?: number | null
  completed_at?: number | null
  status: 'queued' | 'running' | 'completed' | 'failed'
  evidence_count: number
  artifact_count: number
  governance_disposition: string
  parent_run_id?: string | null
  intervention?: string | null
  goal?: string
}

export interface MetricDelta {
  metric: string
  valA: unknown
  valB: unknown
  delta?: number | null
  pctChange?: number | null
}

export interface MetricComparisonItem {
  testId: string
  statusA: string
  statusB: string
  statusChanged: boolean
  evidenceIdA?: string
  evidenceIdB?: string
  metrics: MetricDelta[]
  isChanged: boolean
}

export interface ParameterDiff {
  param: string
  valA: unknown
  valB: unknown
  changed: boolean
}

export interface FindingDiff {
  title: string
  presentInA: boolean
  presentInB: boolean
  severityA?: string | null
  severityB?: string | null
  severityChanged: boolean
  evidenceIdsA?: string[]
  evidenceIdsB?: string[]
}

export interface ArtifactDiff {
  artifactId: string
  title: string
  presentInA: boolean
  presentInB: boolean
  fingerprintA?: string | null
  fingerprintB?: string | null
  fingerprintChanged: boolean
}

export interface RunCompareResult {
  compatible: boolean
  incompatibleReason?: string
  runA?: {
    runId: string
    workflow: string
    domain: string
    contextId: string
    status: string
    createdAt?: number
    completedAt?: number
    evidenceCount: number
    artifactCount: number
    governanceDisposition: string
  }
  runB?: {
    runId: string
    workflow: string
    domain: string
    contextId: string
    status: string
    createdAt?: number
    completedAt?: number
    evidenceCount: number
    artifactCount: number
    governanceDisposition: string
  }
  lineage?: {
    isLineage: boolean
    relation: string
    parentRunId?: string | null
  }
  parameters?: ParameterDiff[]
  fingerprints?: {
    fingerprintA: string
    fingerprintB: string
    matched: boolean
  }
  metricsSummary?: {
    totalCommonTests: number
    changedTestsCount: number
    unchangedTestsCount: number
    onlyInACount: number
    onlyInBCount: number
  }
  metricComparisons?: MetricComparisonItem[]
  onlyInA?: Array<{ testId: string; status: string; evidenceId?: string }>
  onlyInB?: Array<{ testId: string; status: string; evidenceId?: string }>
  findings?: FindingDiff[]
  artifacts?: ArtifactDiff[]
  governance?: {
    dispositionA: string
    dispositionB: string
    changed: boolean
  }
}

export interface RunLineage {
  runId: string
  parentRunId?: string | null
  intervention?: string | null
  parameterDelta?: Record<string, { parent: unknown; child: unknown }>
  children?: Array<{
    runId: string
    createdAt?: number
    status: string
    intervention?: string | null
    parameters?: Record<string, unknown>
  }>
}

export interface PinnedItem {
  id: string
  itemType: 'artifact' | 'evidence' | 'finding'
  itemId: string
  runId: string
  label: string
  producerProvenance?: string | null
  stage?: string | null
  timestamp: string
}
