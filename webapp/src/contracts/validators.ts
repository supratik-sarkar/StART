import type {
  AgentPlanPreview, ArtifactRecord, AttestationState, Capability, EvidenceRecord,
  ExecutionContext, ExecutionGraph, Finding, GovernanceState, ProposedAction,
  ReviewerGateResult, RunSnapshot, RuntimeEvent
} from './types'

export class SchemaValidationError extends Error {
  constructor(public readonly boundary: string, message: string) {
    super(`[SchemaValidationError: ${boundary}] ${message}`)
    this.name = 'SchemaValidationError'
  }
}

function assertObject(val: unknown, boundary: string): Record<string, unknown> {
  if (!val || typeof val !== 'object' || Array.isArray(val)) {
    throw new SchemaValidationError(boundary, `Expected object, got ${typeof val}`)
  }
  return val as Record<string, unknown>
}

function assertArray(val: unknown, boundary: string): unknown[] {
  if (!Array.isArray(val)) {
    throw new SchemaValidationError(boundary, `Expected array, got ${typeof val}`)
  }
  return val
}

export function validateCapabilities(data: unknown): Capability[] {
  const arr = assertArray(data, 'Capabilities')
  return arr.map((item, idx) => {
    const o = assertObject(item, `Capability[${idx}]`)
    if (typeof o.id !== 'string') throw new SchemaValidationError(`Capability[${idx}]`, 'Missing id')
    if (typeof o.label !== 'string') throw new SchemaValidationError(`Capability[${idx}]`, 'Missing label')
    if (typeof o.description !== 'string') throw new SchemaValidationError(`Capability[${idx}]`, 'Missing description')
    if (typeof o.enabled !== 'boolean') throw new SchemaValidationError(`Capability[${idx}]`, 'Missing enabled boolean')
    return {
      id: o.id as any,
      label: o.label,
      description: o.description,
      category: (o.category as any) || 'ml',
      enabled: o.enabled,
      disabledReason: typeof o.disabledReason === 'string' ? o.disabledReason : undefined,
      icon: typeof o.icon === 'string' ? o.icon : undefined,
    }
  })
}

export function validateExecutionContexts(data: unknown): ExecutionContext[] {
  const arr = assertArray(data, 'ExecutionContexts')
  return arr.map((item, idx) => {
    const o = assertObject(item, `ExecutionContext[${idx}]`)
    if (typeof o.id !== 'string') throw new SchemaValidationError(`ExecutionContext[${idx}]`, 'Missing id')
    if (typeof o.label !== 'string') throw new SchemaValidationError(`ExecutionContext[${idx}]`, 'Missing label')
    return {
      id: o.id,
      label: o.label,
      kind: (o.kind as any) || 'dataset',
      description: String(o.description || ''),
      provenance: String(o.provenance || ''),
      shape: typeof o.shape === 'string' ? o.shape : undefined,
      target: typeof o.target === 'string' ? o.target : undefined,
      seed: typeof o.seed === 'number' ? o.seed : undefined,
      badges: Array.isArray(o.badges) ? o.badges.map(String) : [],
    }
  })
}

export function validateAgentPlanPreview(data: unknown): AgentPlanPreview {
  const o = assertObject(data, 'AgentPlanPreview')
  if (typeof o.workflowId !== 'string') throw new SchemaValidationError('AgentPlanPreview', 'Missing workflowId')
  if (typeof o.contextId !== 'string') throw new SchemaValidationError('AgentPlanPreview', 'Missing contextId')
  if (!Array.isArray(o.plan)) throw new SchemaValidationError('AgentPlanPreview', 'Missing plan array')
  return {
    workflowId: o.workflowId as any,
    contextId: o.contextId,
    goal: String(o.goal || ''),
    plan: o.plan.map((step: any, idx: number) => {
      const s = assertObject(step, `AgentPlanStep[${idx}]`)
      return {
        id: String(s.id || `step-${idx + 1}`),
        label: String(s.label || ''),
        description: s.description ? String(s.description) : undefined,
        kind: (s.kind as any) || 'test',
        status: (s.status as any) || 'queued',
        parentId: s.parentId ? String(s.parentId) : undefined,
      }
    }),
    requiredInputs: Array.isArray(o.requiredInputs) ? o.requiredInputs.map(String) : [],
    warnings: Array.isArray(o.warnings) ? o.warnings.map(String) : [],
  }
}

export function validateRunSnapshot(data: unknown): RunSnapshot {
  const o = assertObject(data, 'RunSnapshot')
  const runId = String(o.runId || o.run_id || '')
  const workflowId = String(o.workflowId || o.workflow || 'predictive_ml')
  const phase = String(o.phase || (o.status === 'COMPLETED' ? 'completed' : o.status === 'RUNNING' ? 'running' : 'planning'))
  if (!runId) throw new SchemaValidationError('RunSnapshot', 'Missing runId')
  return {
    runId,
    workflowId: workflowId as any,
    contextId: String(o.contextId || o.synthetic_profile || ''),
    goal: String(o.goal || ''),
    phase: phase as any,
    statusLabel: String(o.statusLabel || o.phase || o.status || 'Active'),
    startedAt: String(o.startedAt || o.created_at || new Date().toISOString()),
    updatedAt: String(o.updatedAt || o.completed_at || new Date().toISOString()),
    elapsedMs: Number(o.elapsedMs || 0),
    progress: o.progress && typeof o.progress === 'object' ? (o.progress as any) : undefined,
    plan: Array.isArray(o.plan) ? (o.plan as any) : [],
    parentRunId: o.parentRunId ? String(o.parentRunId) : (o.parent_run_id ? String(o.parent_run_id) : undefined),
    sourceEvidenceId: o.sourceEvidenceId ? String(o.sourceEvidenceId) : (o.source_evidence_id ? String(o.source_evidence_id) : undefined),
  }
}

export function validateRuntimeEvent(data: unknown): RuntimeEvent {
  const o = assertObject(data, 'RuntimeEvent')
  const eventId = String(o.eventId || o.event_id || '')
  const runId = String(o.runId || o.run_id || '')
  const type = String(o.type || o.event_type || '')
  if (!eventId) throw new SchemaValidationError('RuntimeEvent', 'Missing eventId')
  if (!runId) throw new SchemaValidationError('RuntimeEvent', 'Missing runId')
  if (!type) throw new SchemaValidationError('RuntimeEvent', 'Missing type')
  return {
    eventId,
    sequence: Number(o.sequence || 0),
    runId,
    timestamp: String(o.timestamp || new Date().toISOString()),
    type: type as any,
    nodeId: o.nodeId ? String(o.nodeId) : (o.node_id ? String(o.node_id) : undefined),
    parentNodeId: o.parentNodeId ? String(o.parentNodeId) : (o.parent_node_id ? String(o.parent_node_id) : undefined),
    title: String(o.title || o.action || o.phase || 'Runtime event'),
    message: String(o.message || ''),
    status: (o.status as any) || 'running',
    progress: o.progress && typeof o.progress === 'object' ? (o.progress as any) : undefined,
    evidenceIds: Array.isArray(o.evidenceIds) ? o.evidenceIds.map(String) : (Array.isArray(o.evidence_refs) ? o.evidence_refs.map(String) : []),
    artifactIds: Array.isArray(o.artifactIds) ? o.artifactIds.map(String) : (Array.isArray(o.artifact_refs) ? o.artifact_refs.map(String) : []),
    metadata: o.metadata && typeof o.metadata === 'object' ? (o.metadata as Record<string, unknown>) : (o.payload && typeof o.payload === 'object' ? (o.payload as Record<string, unknown>) : {}),
  }
}

export function validateEvidenceRecords(data: unknown): EvidenceRecord[] {
  const arr = assertArray(data, 'EvidenceRecords')
  return arr.map((item, idx) => {
    const o = assertObject(item, `EvidenceRecord[${idx}]`)
    const evidenceId = String(o.evidenceId || o.evidence_id || '')
    const testId = String(o.testId || o.test_id || '')
    if (!evidenceId) throw new SchemaValidationError(`EvidenceRecord[${idx}]`, 'Missing evidenceId')
    if (!testId) throw new SchemaValidationError(`EvidenceRecord[${idx}]`, 'Missing testId')

    let metrics: any[] = []
    if (Array.isArray(o.metrics)) {
      metrics = o.metrics
    } else if (o.metrics && typeof o.metrics === 'object') {
      metrics = Object.entries(o.metrics).map(([k, v]) => ({
        name: k,
        value: typeof v === 'number' ? roundValue(v) : v,
        status: 'PASS',
      }))
    }

    return {
      evidenceId,
      runId: String(o.runId || o.run_id || ''),
      testId,
      title: String(o.title || testId),
      status: (o.status as any) || 'RECORDED',
      metrics,
      provenance: Array.isArray(o.provenance) ? o.provenance.map(String) : [],
      parentNodeId: o.parentNodeId ? String(o.parentNodeId) : undefined,
      createdAt: String(o.createdAt || o.created_at || new Date().toISOString()),
      summary: o.summary ? String(o.summary) : undefined,
    }
  })
}

function roundValue(v: number): number {
  return Math.round(v * 10000) / 10000
}

export function validateFindings(data: unknown): Finding[] {
  const arr = assertArray(data, 'Findings')
  return arr.map((item, idx) => {
    const o = assertObject(item, `Finding[${idx}]`)
    return {
      findingId: String(o.findingId || `F-${idx + 1}`),
      runId: String(o.runId || ''),
      title: String(o.title || 'Finding'),
      summary: String(o.summary || ''),
      evidenceIds: Array.isArray(o.evidenceIds) ? o.evidenceIds.map(String) : [],
      sourceNodeId: o.sourceNodeId ? String(o.sourceNodeId) : undefined,
      severity: (o.severity as any) || 'info',
      limitations: Array.isArray(o.limitations) ? o.limitations.map(String) : [],
      availableActions: Array.isArray(o.availableActions) ? (o.availableActions as any) : ['explain', 'rerun'],
    }
  })
}

export function validateArtifactRecords(data: unknown): ArtifactRecord[] {
  const arr = assertArray(data, 'ArtifactRecords')
  return arr.map((item, idx) => {
    const o = assertObject(item, `ArtifactRecord[${idx}]`)
    return {
      artifactId: String(o.artifactId || `ART-${idx + 1}`),
      runId: String(o.runId || ''),
      label: String(o.label || 'Artifact'),
      kind: (o.kind as any) || 'table',
      mimeType: String(o.mimeType || 'application/json'),
      createdAt: String(o.createdAt || new Date().toISOString()),
      description: o.description ? String(o.description) : undefined,
      preview: o.preview && typeof o.preview === 'object' ? (o.preview as any) : undefined,
    }
  })
}

export function validateExecutionGraph(data: unknown): ExecutionGraph {
  const o = assertObject(data, 'ExecutionGraph')
  if (!Array.isArray(o.nodes)) throw new SchemaValidationError('ExecutionGraph', 'Missing nodes array')
  if (!Array.isArray(o.edges)) throw new SchemaValidationError('ExecutionGraph', 'Missing edges array')
  return {
    nodes: o.nodes.map((n: any) => assertObject(n, 'ExecutionGraphNode') as any),
    edges: o.edges.map((e: any) => assertObject(e, 'ExecutionGraphEdge') as any),
  }
}

export function validateGovernanceState(data: unknown): GovernanceState | null {
  if (!data) return null
  const o = assertObject(data, 'GovernanceState')
  if (typeof o.disposition !== 'string') return null
  return {
    disposition: o.disposition,
    policyDecision: o.policyDecision ? String(o.policyDecision) : undefined,
    rationale: o.rationale ? String(o.rationale) : undefined,
    evidenceCoverage: typeof o.evidenceCoverage === 'number' ? o.evidenceCoverage : undefined,
    unresolvedItems: Array.isArray(o.unresolvedItems) ? o.unresolvedItems.map(String) : [],
  }
}

export function validateAttestationState(data: unknown): AttestationState | null {
  if (!data) return null
  const o = assertObject(data, 'AttestationState')
  if (typeof o.merkleRoot !== 'string') return null
  return {
    merkleRoot: o.merkleRoot,
    createdAt: String(o.createdAt || new Date().toISOString()),
    evidenceCount: Number(o.evidenceCount || 0),
    artifactCount: Number(o.artifactCount || 0),
    reproducibilityId: o.reproducibilityId ? String(o.reproducibilityId) : undefined,
  }
}

export function validateProposedAction(data: unknown): ProposedAction {
  const o = assertObject(data, 'ProposedAction')
  if (typeof o.actionId !== 'string') throw new SchemaValidationError('ProposedAction', 'Missing actionId')
  if (typeof o.label !== 'string') throw new SchemaValidationError('ProposedAction', 'Missing label')
  return {
    actionId: o.actionId,
    label: o.label,
    description: String(o.description || ''),
    kind: (o.kind as any) || 'deeper_test',
    sourceNodeId: o.sourceNodeId ? String(o.sourceNodeId) : undefined,
    sourceEvidenceId: o.sourceEvidenceId ? String(o.sourceEvidenceId) : undefined,
    parameters: o.parameters && typeof o.parameters === 'object' ? (o.parameters as any) : {},
  }
}

export function validateReviewerGateResult(data: unknown): ReviewerGateResult {
  const o = assertObject(data, 'ReviewerGateResult')
  if (typeof o.runId !== 'string') throw new SchemaValidationError('ReviewerGateResult', 'Missing runId')
  return {
    runId: o.runId,
    modelName: String(o.modelName || ''),
    hydratedFindings: Array.isArray(o.hydratedFindings) ? (o.hydratedFindings as any) : [],
    allGrounded: Boolean(o.allGrounded),
    governanceDisposition: typeof o.governanceDisposition === 'string' ? o.governanceDisposition : '',
    attestationSealMerkleRoot: String(o.attestationSealMerkleRoot || ''),
  }
}
