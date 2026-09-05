import { describe, expect, it } from 'vitest'
import type { ProgressState, RunRequest } from './types'
import {
  validateCapabilities,
  validateExecutionContexts,
  validateAgentPlanPreview,
  validateRunSnapshot,
  validateRuntimeEvent,
  validateEvidenceRecords,
  validateFindings,
  validateArtifactRecords,
  validateExecutionGraph,
  validateGovernanceState,
  validateAttestationState,
  validateProposedAction,
  validateReviewerGateResult,
  SchemaValidationError,
} from './validators'

describe('StART Contract & Validator Invariants', () => {
  it('allows phase-only progress without fabricated percent', () => {
    const p: ProgressState = {
      label: 'Running deterministic tests',
      completed: 3,
      total: 12,
    }
    expect(p.percent).toBeUndefined()
  })

  it('validates capabilities and preserves truthful disabledReason', () => {
    const raw = [
      {
        id: 'predictive_ml',
        label: 'Predictive ML',
        description: 'Supervised ML',
        category: 'ml',
        enabled: true,
      },
      {
        id: 'model_comparison',
        label: 'Compare Models',
        description: 'Multi-candidate comparison',
        category: 'ml',
        enabled: false,
        disabledReason: 'Multi-candidate protocol not yet enabled.',
      },
    ]
    const caps = validateCapabilities(raw)
    expect(caps).toHaveLength(2)
    expect(caps[0].enabled).toBe(true)
    expect(caps[1].enabled).toBe(false)
    expect(caps[1].disabledReason).toBe('Multi-candidate protocol not yet enabled.')
  })

  it('rejects invalid capability payload with SchemaValidationError', () => {
    expect(() => validateCapabilities([{ id: 'bad' }])).toThrow(SchemaValidationError)
    expect(() => validateCapabilities('not an array')).toThrow(SchemaValidationError)
  })

  it('validates execution contexts correctly', () => {
    const raw = [
      {
        id: 'institutional_credit_v1',
        label: 'Credit Context',
        kind: 'dataset',
        description: 'Credit benchmark',
        provenance: 'Built-in',
        shape: '12,000 x 31',
        target: 'default_flag',
        seed: 42,
        badges: ['public-safe', 'seeded'],
      },
    ]
    const ctxs = validateExecutionContexts(raw)
    expect(ctxs).toHaveLength(1)
    expect(ctxs[0].target).toBe('default_flag')
  })

  it('validates dedicated AgentPlanPreview without requiring runId', () => {
    const planPreview = {
      workflowId: 'predictive_ml',
      contextId: 'institutional_credit_v1',
      goal: 'Evaluate model',
      plan: [
        { id: 'step-1', label: 'Load context', kind: 'context', status: 'completed' },
        { id: 'step-2', label: 'Evaluate checks', kind: 'test', status: 'queued', parentId: 'step-1' },
      ],
      requiredInputs: [],
      warnings: [],
    }
    const validated = validateAgentPlanPreview(planPreview)
    expect(validated.workflowId).toBe('predictive_ml')
    expect(validated.plan).toHaveLength(2)
    expect((validated as any).runId).toBeUndefined()
  })

  it('validates RunSnapshot and preserves parent-child linkage', () => {
    const snap = {
      runId: 'RUN-CHILD-01',
      workflowId: 'predictive_ml',
      contextId: 'institutional_credit_v1',
      goal: 'Deeper verification',
      phase: 'running',
      statusLabel: 'Executing deeper tests',
      startedAt: '2026-09-05T12:00:00Z',
      updatedAt: '2026-09-05T12:00:01Z',
      elapsedMs: 1000,
      plan: [],
      parentRunId: 'RUN-PARENT-01',
      sourceEvidenceId: 'EV-01',
    }
    const validated = validateRunSnapshot(snap)
    expect(validated.runId).toBe('RUN-CHILD-01')
    expect(validated.parentRunId).toBe('RUN-PARENT-01')
    expect(validated.sourceEvidenceId).toBe('EV-01')
  })

  it('validates RuntimeEvent and supports monotonic sequences', () => {
    const ev = {
      eventId: 'EVT-01',
      sequence: 1,
      runId: 'RUN-01',
      timestamp: '2026-09-05T12:00:00Z',
      type: 'phase',
      nodeId: 'plan',
      title: 'Plan accepted',
      message: 'Orchestration plan ready',
      status: 'completed',
    }
    const validated = validateRuntimeEvent(ev)
    expect(validated.sequence).toBe(1)
    expect(validated.status).toBe('completed')
  })

  it('validates EvidenceRecords and preserves metric provenance', () => {
    const records = [
      {
        evidenceId: 'EV-TEST-1',
        runId: 'RUN-01',
        testId: 'supervised.calibration_curve',
        title: 'Calibration Reliability',
        status: 'PASS',
        metrics: [{ name: 'brier_score', value: 0.118, criterion: '<= 0.20' }],
        provenance: ['run:RUN-01', 'test:supervised.calibration_curve'],
      },
    ]
    const validated = validateEvidenceRecords(records)
    expect(validated).toHaveLength(1)
    expect(validated[0].metrics[0].value).toBe(0.118)
    expect(validated[0].status).toBe('PASS')
  })

  it('validates ExecutionGraph structure', () => {
    const g = {
      nodes: [
        { id: 'parent-run', runId: 'RUN-CHILD', kind: 'human', label: 'Parent RUN-PARENT', status: 'completed' },
        { id: 'context', runId: 'RUN-CHILD', kind: 'context', label: 'Context', status: 'completed', parentId: 'parent-run' },
      ],
      edges: [
        { id: 'e-1', source: 'parent-run', target: 'context', relation: 'rerun' },
      ],
    }
    const validated = validateExecutionGraph(g)
    expect(validated.nodes).toHaveLength(2)
    expect(validated.edges[0].relation).toBe('rerun')
  })

  it('validates ReviewerGateResult with server-side attestation seal', () => {
    const gateResult = {
      runId: 'RUN-01',
      modelName: 'SmolLM2-1.7B-Instruct-q4f16_1-MLC',
      hydratedFindings: [
        {
          findingId: 'F-01',
          title: 'Stress observation',
          grounded: true,
          evidenceRefs: [{ evidenceId: 'EV-01', metricName: 'brier_score', status: 'GROUNDED', hydratedValue: 0.118 }],
        },
      ],
      allGrounded: true,
      governanceDisposition: 'ACCEPT',
      attestationSealMerkleRoot: 'sha256:7fd2a8019b…',
    }
    const validated = validateReviewerGateResult(gateResult)
    expect(validated.allGrounded).toBe(true)
    expect(validated.attestationSealMerkleRoot).toContain('sha256:')
  })

  it('preserves empty governanceDisposition without defaulting to ACCEPT', () => {
    const gateResult = {
      runId: 'RUN-02',
      modelName: 'SmolLM2-1.7B-Instruct-q4f16_1-MLC',
      hydratedFindings: [],
      allGrounded: true,
      attestationSealMerkleRoot: 'sha256:abc…',
    }
    const validated = validateReviewerGateResult(gateResult)
    expect(validated.governanceDisposition).toBe('')
  })
})
