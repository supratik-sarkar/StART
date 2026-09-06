import type {
  ReviewerContext,
  ReviewerProgress,
  ReviewerRuntime,
  ReviewerOutput,
  ReviewerRuntimeState,
} from '../../contracts/reviewer'
import type { ConversationMessage, EvidenceRecord, ProposedAction, RunSnapshot } from '../../contracts/types'

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

  async ask(
    context: ReviewerContext,
    args: {
      text: string
      evidence: EvidenceRecord[]
      runSnapshot?: RunSnapshot
    }
  ): Promise<ConversationMessage> {
    // Validate selectedEvidenceId strictly against active run evidence universe (Amendments 8, 9)
    const isContextEvidenceValid = Boolean(
      context.selectedEvidenceId &&
        args.evidence.some((e) => e.evidenceId === context.selectedEvidenceId)
    )
    const activeEvidenceId = isContextEvidenceValid ? context.selectedEvidenceId : undefined
    const evidenceIds = activeEvidenceId
      ? [activeEvidenceId]
      : args.evidence.slice(0, 2).map((e) => e.evidenceId)

    // Parse structured proposal if provided as JSON; NO keyword detection (Amendment 10)
    let action: ProposedAction | undefined = undefined
    try {
      const trimmed = args.text.trim()
      if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
        const parsed = JSON.parse(trimmed)
        if (parsed.proposedAction && typeof parsed.proposedAction === 'object') {
          const kind = parsed.proposedAction.kind
          if (['rerun', 'change_parameter', 'deeper_test', 'challenge'].includes(kind)) {
            action = {
              actionId: id('ACT'),
              label: parsed.proposedAction.rationale || `Proposed ${kind}`,
              description:
                parsed.proposedAction.rationale || 'Creates a child run linked to selected context.',
              kind,
              sourceNodeId: context.selectedNodeId,
              sourceEvidenceId: activeEvidenceId,
              parameters: parsed.proposedAction.parameters || { depth: 'focused' },
            }
          }
        }
      }
    } catch {
      // Conversational response only
    }

    return {
      id: id('MSG'),
      role: 'agent',
      timestamp: new Date().toISOString(),
      contextNodeId: context.selectedNodeId,
      evidenceIds,
      proposedAction: action,
      text: action
        ? 'I have parsed your structured follow-up proposal. The deterministic action is prepared below for approval.'
        : `Deterministic review notes: ${
            activeEvidenceId ? `Evidence [${activeEvidenceId}]` : context.selectedNodeId || 'active run'
          } evaluated with bounded evidence ledger (${args.evidence.length} records available). All observations remain grounded in verified evidence.`,
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
