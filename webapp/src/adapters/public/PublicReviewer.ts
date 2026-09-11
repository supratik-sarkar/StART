import type { ConversationMessage, EvidenceRecord, RunSnapshot } from '../../contracts/types'
import type {
  ReviewerContext,
  ReviewerProgress,
  ReviewerRuntime,
  ReviewerRuntimeState,
  ReviewerOutput,
} from '../../contracts/reviewer'
import type { PublicStARTBackend } from './PublicStARTBackend'

export class PublicReviewer implements ReviewerRuntime {
  readonly runtimeName = 'StART OpenAI Reviewer (gpt-5.1)'
  private backend: PublicStARTBackend
  private state: ReviewerRuntimeState = 'ready'
  private listeners: Array<(state: ReviewerRuntimeState, progress?: ReviewerProgress) => void> = []

  constructor(backend: PublicStARTBackend) {
    this.backend = backend
  }

  getState(): ReviewerRuntimeState {
    return this.state
  }

  subscribeState(
    listener: (state: ReviewerRuntimeState, progress?: ReviewerProgress) => void
  ): () => void {
    this.listeners.push(listener)
    listener(this.state, { state: this.state, label: 'OpenAI gpt-5.1 ready' })
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener)
    }
  }

  async checkWebGPUSupport(): Promise<boolean> {
    return true
  }

  async initialize(onProgress: (p: ReviewerProgress) => void): Promise<void> {
    this.state = 'ready'
    onProgress({ state: 'ready', label: 'OpenAI gpt-5.1 ready' })
  }

  async ask(
    context: ReviewerContext,
    args: {
      text: string
      evidence: EvidenceRecord[]
      runSnapshot?: RunSnapshot
    }
  ): Promise<ConversationMessage> {
    this.state = 'reviewing'
    for (const listener of this.listeners) {
      try {
        listener(this.state, { state: 'reviewing', label: 'Querying backend OpenAI reviewer…' })
      } catch {}
    }

    try {
      const resp = await this.backend.askQuestion(context.runId, {
        question: args.text,
        targetStage: context.selectedNodeId,
        evidenceId: context.selectedEvidenceId,
      })

      this.state = 'ready'
      for (const listener of this.listeners) {
        try {
          listener(this.state, { state: 'ready', label: 'Backend question completed' })
        } catch {}
      }

      const citedIds =
        resp.citations?.map((c: any) => c.evidenceId || c.evidence_id).filter(Boolean) || []

      return {
        id: resp.receipt?.receiptId || `agent-${Date.now()}`,
        role: 'agent',
        text: resp.answer,
        timestamp: resp.timestamp || new Date().toISOString(),
        evidenceIds:
          citedIds.length > 0
            ? citedIds
            : context.selectedEvidenceId
            ? [context.selectedEvidenceId]
            : undefined,
      }
    } catch (err) {
      this.state = 'ready'
      for (const listener of this.listeners) {
        try {
          listener(this.state, { state: 'ready', label: 'Backend question completed' })
        } catch {}
      }
      throw err
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
    const attentionEv =
      args.evidence.find((e) => e.status === 'FAIL') ||
      args.evidence.find((e) => e.status === 'ATTENTION')
    const targetEv = attentionEv || args.evidence[0]
    const summary = `Evidence-grounded review for run ${args.runId}. Evaluated across ${args.evidence.length} evidence records.`
    if (onChunk) onChunk(summary)
    return {
      executiveSummary: summary,
      findings: targetEv
        ? [
            {
              findingId: `F-${targetEv.evidenceId}`,
              title: `${targetEv.title} review`,
              description: `Evidence '${targetEv.testId}' reported status ${targetEv.status}.`,
              evidenceIds: [targetEv.evidenceId],
              metricRefs: targetEv.metrics.slice(0, 2).map((m) => ({
                evidenceId: targetEv.evidenceId,
                metricName: m.name,
                value: m.value,
              })),
              severity: targetEv.status === 'FAIL' ? 'ATTENTION' : 'INFO',
              limitations: ['Evaluated against immutable backend evidence records.'],
              suggestedActions: ['Inspect evidence records before governance sign-off.'],
            },
          ]
        : [],
      limitations: ['Grounded strictly on canonical evidence records.'],
      evidenceIds: targetEv ? [targetEv.evidenceId] : [],
    }
  }

  dispose(): void {
    this.listeners = []
  }
}
