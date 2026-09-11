import { describe, expect, it } from 'vitest'
import { DemoBackend } from '../adapters/demo/DemoBackend'

describe('StART Local UI Journey Contracts & Invariants', () => {
  const backend = new DemoBackend()

  it('GOAL_INPUT_HAS_PRODUCT_EFFECT: Plan preview preserves explicit goal', async () => {
    const goalText = 'I want to evaluate a binary classification model using a real deterministic StART execution context.'
    const plan = await backend.createPlan({
      workflowId: 'predictive_ml',
      contextId: 'institutional_credit_v1',
      goal: goalText,
      parameters: {},
    })

    expect(plan.goal).toBe(goalText)
    expect(plan.workflowId).toBe('predictive_ml')
    expect(plan.contextId).toBe('institutional_credit_v1')
  })

  it('NO_PREEXECUTION_FAKE_RESULTS: Plan preview contains only queued/planned stages without fake metrics', async () => {
    const plan = await backend.createPlan({
      workflowId: 'predictive_ml',
      contextId: 'institutional_credit_v1',
      goal: 'Benchmark run',
      parameters: {},
    })

    expect(plan.plan.length).toBe(8)
    for (const step of plan.plan) {
      expect(step.status).toBe('queued')
      expect((step as any).metrics).toBeUndefined()
    }
  })

  it('BUILD_PLAN_AND_EXECUTE_ARE_DISTINCT_ACTIONS: Creating plan does not create a run or emit execution events', async () => {
    const plan = await backend.createPlan({
      workflowId: 'predictive_ml',
      contextId: 'institutional_credit_v1',
      goal: 'Pre-flight check',
      parameters: {},
    })

    expect((plan as any).runId).toBeUndefined()
  })

  it('DATA_ONBOARDING_VISIBLE: Context metadata strictly adheres to canonical benchmark specs', async () => {
    const contexts = await backend.listExecutionContexts()
    const credit = contexts.find(c => c.id === 'institutional_credit_v1')

    expect(credit).toBeDefined()
    expect(credit!.label).toBe('Synthetic Binary Classification Benchmark')
    expect(credit!.shape).toBe('500 × 8')
    expect(credit!.target).toBe('target')
    expect(credit!.provenance).toBe('Built-in deterministic synthetic generator')
  })

  it('LIVE_RUNTIME_TIMELINE: Canonical execution chronology matches required stage order', async () => {
    const run = await backend.createRun({
      workflowId: 'predictive_ml',
      contextId: 'institutional_credit_v1',
      goal: 'Deterministic verification',
      parameters: {},
    })

    expect(run.runId).toMatch(/^RUN-/)
    expect(run.plan.map(s => s.label)).toEqual([
      'Context prepared',
      'Preflight checks',
      'Feature diagnostics',
      'Supervised evaluation',
      'Explainability',
      'Evidence assembly',
      'Governance',
      'Attestation',
    ])
  })
})
