import type {
  ReviewerProgress,
  ReviewerRuntime,
  ReviewerOutput,
  ReviewerRuntimeState,
} from '../../contracts/reviewer'
import type { ConversationMessage, EvidenceRecord, ProposedAction } from '../../contracts/types'

const id = (prefix: string) => `${prefix}-${Math.random().toString(36).slice(2, 8).toUpperCase()}`

/**
 * Demo / Development Reviewer Runtime.
 *
 * Restricted strictly to local preview and development mode.
 * Provides deterministic simulation of qualitative review and engineering conversation
 * without requiring WebGPU or external model weights.
 */
export class DemoReviewerRuntime implements ReviewerRuntime {
  readonly runtimeName = 'Demo Preview Reviewer (Local Development)'
  private state: ReviewerRuntimeState = 'ready'
  private listeners: Array<(state: ReviewerRuntimeState, progress?: ReviewerProgress) => void> = []

  getState(): ReviewerRuntimeState {
    return this.state
  }

  subscribeState(
    listener: (state: ReviewerRuntimeState, progress?: ReviewerProgress) => void
  ): () => void {
    this.listeners.push(listener)
    listener(this.state)
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener)
    }
  }

  async checkWebGPUSupport(): Promise<boolean> {
    return true
  }

  async initialize(onProgress: (p: ReviewerProgress) => void): Promise<void> {
    this.state = 'ready'
    onProgress({ state: 'ready', label: 'Demo reviewer runtime ready (development simulation)' })
  }

  async ask(args: {
    runId: string
    text: string
    evidence: EvidenceRecord[]
    contextNodeId?: string
  }): Promise<ConversationMessage> {
    const evidenceIds = args.evidence.slice(0, 2).map((e) => e.evidenceId)
    const lower = args.text.toLowerCase()

    let action: ProposedAction | undefined = undefined
    if (lower.includes('rerun') || lower.includes('deeper') || lower.includes('challenge')) {
      const kind = lower.includes('challenge')
        ? 'challenge'
        : lower.includes('deeper')
        ? 'deeper_test'
        : 'rerun'

      action = {
        actionId: id('ACT'),
        label: lower.includes('deeper')
          ? 'Run deeper deterministic verification'
          : lower.includes('challenge')
          ? 'Challenge finding with stress perturbation'
          : 'Re-run from selected execution point',
        description: 'Creates a child run linked to the selected execution node and evidence.',
        kind,
        sourceNodeId: args.contextNodeId,
        sourceEvidenceId: evidenceIds[0],
        parameters: { depth: 'focused' },
      }
    }

    return {
      id: id('MSG'),
      role: 'agent',
      timestamp: new Date().toISOString(),
      contextNodeId: args.contextNodeId,
      evidenceIds,
      proposedAction: action,
      text: action
        ? 'I can turn that into a bounded deterministic follow-up. I have prepared the action below for your approval.'
        : `Deterministic review notes: ${args.contextNodeId || 'active run'} evaluated with bounded evidence ledger (${args.evidence.length} records available). All observations remain grounded in verified evidence.`,
    }
  }

  async review(
    args: {
      runId: string
      goal: string
      evidence: EvidenceRecord[]
      contextNodeId?: string
    },
    onChunk?: (chunk: string) => void
  ): Promise<ReviewerOutput> {
    const attentionEv = args.evidence.find((e) => e.status === 'FAIL')
    const targetEv = attentionEv || args.evidence[0]

    const summary = `Deterministic preview review for run ${args.runId}. Grounded across ${args.evidence.length} evidence records.`
    if (onChunk) {
      onChunk(summary)
    }

    return {
      executiveSummary: summary,
      findings: targetEv
        ? [
            {
              findingId: `F-${targetEv.evidenceId}`,
              title: `${targetEv.title} evaluation`,
              description: `Deterministic surface '${targetEv.testId}' reported status ${targetEv.status}.`,
              evidenceIds: [targetEv.evidenceId],
              metricRefs: targetEv.metrics.slice(0, 2).map((m) => ({
                evidenceId: targetEv.evidenceId,
                metricName: m.name,
                value: m.value,
              })),
              severity: targetEv.status === 'FAIL' ? 'ATTENTION' : 'INFO',
              limitations: ['Evaluated against deterministic test bounds in demo preview.'],
              suggestedActions: ['Verify evidence ledger before sign-off.'],
            },
          ]
        : [],
      limitations: ['Demo preview reviewer runtime active (development mode).'],
      evidenceIds: args.evidence.map((e) => e.evidenceId),
    }
  }

  dispose(): void {
    this.state = 'idle'
  }
}

export const demoReviewerRuntime = new DemoReviewerRuntime()
