import type { ConversationMessage, EvidenceRecord, RunSnapshot } from './types'

export type ReviewerRuntimeState = 'idle' | 'loading' | 'ready' | 'reviewing' | 'unavailable' | 'error'

export interface ReviewerContext {
  runId: string
  selectedNodeId?: string
  selectedEvidenceId?: string
  selectedFindingId?: string
  selectedArtifactId?: string
}

export interface ReviewerProgress {
  state: ReviewerRuntimeState
  label: string
  percent?: number
  downloadedBytes?: number
  totalBytes?: number
}

export interface ReviewerMetricRef {
  evidenceId: string
  metricName: string
  value?: string | number | boolean | null
}

export interface ReviewerFinding {
  findingId: string
  title: string
  description: string
  evidenceIds: string[]
  metricRefs?: ReviewerMetricRef[]
  severity?: 'INFO' | 'ATTENTION' | 'CRITICAL' | string
  limitations?: string[]
  suggestedActions?: string[]
}

export interface ReviewerOutput {
  executiveSummary: string
  findings: ReviewerFinding[]
  limitations: string[]
  evidenceIds: string[]
  rawStructuredOutput?: unknown
}

export interface ReviewerRuntime {
  readonly runtimeName: string
  getState(): ReviewerRuntimeState
  subscribeState?(listener: (state: ReviewerRuntimeState, progress?: ReviewerProgress) => void): () => void
  checkWebGPUSupport(): Promise<boolean>
  initialize(onProgress: (p: ReviewerProgress) => void): Promise<void>
  ask(
    context: ReviewerContext,
    args: {
      text: string
      evidence: EvidenceRecord[]
      runSnapshot?: RunSnapshot
    },
    onFirstToken?: () => void,
    onChunk?: (chunk: string) => void
  ): Promise<ConversationMessage>
  review(
    args: { runId: string; goal: string; evidence: EvidenceRecord[]; contextNodeId?: string },
    onChunk?: (chunk: string) => void,
    onFirstToken?: () => void
  ): Promise<ReviewerOutput>
  dispose(): Promise<void> | void
}

