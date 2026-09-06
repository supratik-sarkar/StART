import type {
  ReviewerContext,
  ReviewerProgress,
  ReviewerRuntime,
  ReviewerOutput,
  ReviewerRuntimeState,
} from '../../contracts/reviewer'
import type { ConversationMessage, EvidenceRecord, ProposedAction, RunSnapshot } from '../../contracts/types'

export const PINNED_MODEL_ID = 'SmolLM2-1.7B-Instruct-q4f16_1-MLC'

/**
 * Derives the model base URL dynamically from environment configuration.
 * Fails closed in production if VITE_START_MODEL_BASE is absent.
 * No silent Hugging Face fallback in production (Amendments 14, 15).
 */
export function getModelBaseUrl(): string {
  const envBase = import.meta.env.VITE_START_MODEL_BASE
  const isProd = import.meta.env.PROD

  if (envBase) {
    const base = String(envBase).replace(/\/+$/, '')
    return base.endsWith(PINNED_MODEL_ID) ? base : `${base}/${PINNED_MODEL_ID}`
  }

  if (isProd) {
    throw new Error(
      'WebLLM model initialization failed: VITE_START_MODEL_BASE is required in production. Silent Hugging Face fallback is prohibited.'
    )
  }

  const allowHf = import.meta.env.VITE_START_ALLOW_HF_FALLBACK === 'true'
  if (allowHf) {
    return `https://huggingface.co/mlc-ai/${PINNED_MODEL_ID}/resolve/main/`
  }

  throw new Error(
    'WebLLM model initialization failed: VITE_START_MODEL_BASE must be configured. For development fallback, set VITE_START_ALLOW_HF_FALLBACK=true.'
  )
}

/**
 * Browser-Private WebLLM Reviewer & Contextual Engineering Agent.
 *
 * Implements ReviewerRuntime using WebGPU and @mlc-ai/web-llm.
 * Reuses a single loaded instance of SmolLM2-1.7B for both:
 * 1. Qualitative structured review of empirical EvidenceRecords.
 * 2. Contextual engineering conversation grounded in active run evidence.
 */
export class WebLLMReviewer implements ReviewerRuntime {
  readonly runtimeName = 'Browser WebLLM (SmolLM2-1.7B-Instruct)'
  private engine: any = null
  private state: ReviewerRuntimeState = 'idle'
  private listeners: Array<(state: ReviewerRuntimeState, progress?: ReviewerProgress) => void> = []

  async checkWebGPUSupport(): Promise<boolean> {
    if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
      return false
    }
    try {
      const adapter = await (navigator as any).gpu.requestAdapter()
      return !!adapter
    } catch {
      return false
    }
  }

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

  private notifyProgress(progress: ReviewerProgress): void {
    this.state = progress.state
    for (const listener of this.listeners) {
      try {
        listener(this.state, progress)
      } catch {
        // Listener error suppression
      }
    }
  }

  async initialize(onProgress: (p: ReviewerProgress) => void): Promise<void> {
    const emit = (p: ReviewerProgress) => {
      this.notifyProgress(p)
      onProgress(p)
    }

    if (this.engine) {
      emit({ state: 'ready', label: 'Engine ready' })
      return
    }

    const hasWebGPU = await this.checkWebGPUSupport()
    if (!hasWebGPU) {
      emit({
        state: 'unavailable',
        label: 'WebGPU not available on this browser/hardware device.',
      })
      return
    }

    emit({ state: 'loading', label: 'Initializing WebGPU runtime...', percent: 5 })

    try {
      // Lazy load @mlc-ai/web-llm to keep initial shell bundle ultra-lean
      const webllm = await import('@mlc-ai/web-llm')
      const modelUrl = getModelBaseUrl()

      const appConfig = {
        model_list: [
          {
            model: modelUrl,
            model_id: PINNED_MODEL_ID,
            model_lib:
              webllm.modelLibURLPrefix +
              webllm.modelVersion +
              '/SmolLM2-1.7B-Instruct-q4f16_1_cs1k-webgpu.wasm',
            vram_required_MB: 1774.19,
            low_resource_required: true,
            required_features: ['shader-f16'],
            overrides: { context_window_size: 4096 },
          },
        ],
      }

      this.engine = await webllm.CreateMLCEngine(PINNED_MODEL_ID, {
        appConfig,
        initProgressCallback: (report: { text: string; progress: number }) => {
          emit({
            state: 'loading',
            label: report.text || 'Loading model weights...',
            percent: Math.round((report.progress || 0) * 100),
          })
        },
      })

      emit({ state: 'ready', label: 'SmolLM2-1.7B ready' })
    } catch (err: any) {
      emit({
        state: 'error',
        label: `Failed to initialize WebLLM engine: ${err?.message || 'Unknown error'}`,
      })
      throw err
    }
  }

  async ask(
    context: ReviewerContext,
    args: {
      text: string
      evidence: EvidenceRecord[]
      runSnapshot?: RunSnapshot
    },
    onFirstToken?: () => void,
    onChunk?: (chunk: string) => void
  ): Promise<ConversationMessage> {
    if (!this.engine) {
      throw new Error(
        'WebLLM reviewer runtime is not initialized. Initialize the runtime before requesting an analysis.'
      )
    }

    // Validate selectedEvidenceId strictly against active run evidence universe (Amendments 8, 9)
    const isContextEvidenceValid = Boolean(
      context.selectedEvidenceId &&
        args.evidence.some((e) => e.evidenceId === context.selectedEvidenceId)
    )
    const activeEvidenceId = isContextEvidenceValid ? context.selectedEvidenceId : undefined

    const evidenceUniverse = args.evidence
      .slice(0, 8)
      .map(
        (e) =>
          `[${e.evidenceId}] ${e.testId} (${e.status}): ${e.metrics
            .map((m) => `${m.name}=${m.value}`)
            .join(', ')}`
      )
      .join('\n')

    const systemPrompt = `You are the StART contextual engineering agent.
Ground your response strictly in the permitted Evidence Records below:
${evidenceUniverse}
${activeEvidenceId ? `Active user selection is focused on evidence: [${activeEvidenceId}].` : ''}

You must respond with a JSON object conforming to:
{
  "message": "string (your evidence-grounded response)",
  "proposedAction": {
    "kind": "rerun" | "change_parameter" | "deeper_test" | "challenge",
    "sourceEvidenceId": "${activeEvidenceId || args.evidence[0]?.evidenceId || ''}",
    "sourceNodeId": "${context.selectedNodeId || ''}",
    "parameters": {},
    "rationale": "string explanation"
  }
}
If no executable action is warranted, omit "proposedAction" or set it to null.
Invariants:
1. Never invent or compute numbers.
2. If proposing an action, sourceEvidenceId must be in the permitted list.
3. Output valid JSON only.`

    const chunks = await this.engine.chat.completions.create({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: args.text },
      ],
      stream: true,
      temperature: 0.2,
      max_tokens: 350,
    })

    let fullText = ''
    let firstTokenNotified = false
    for await (const chunk of chunks) {
      const delta = chunk.choices[0]?.delta?.content || ''
      if (delta && !firstTokenNotified) {
        firstTokenNotified = true
        onFirstToken?.()
      }
      fullText += delta
      onChunk?.(delta)
    }

    let cleaned = fullText.trim()
    if (cleaned.startsWith('```')) {
      const firstNl = cleaned.indexOf('\n')
      if (firstNl !== -1) cleaned = cleaned.substring(firstNl + 1)
      if (cleaned.endsWith('```')) cleaned = cleaned.substring(0, cleaned.length - 3)
      cleaned = cleaned.trim()
    }

    let replyMessage = fullText
    let action: ProposedAction | undefined = undefined

    try {
      const parsed = JSON.parse(cleaned)
      if (parsed && typeof parsed === 'object') {
        if (typeof parsed.message === 'string') {
          replyMessage = parsed.message
        }
        if (parsed.proposedAction && typeof parsed.proposedAction === 'object') {
          const kind = parsed.proposedAction.kind
          if (['rerun', 'change_parameter', 'deeper_test', 'challenge'].includes(kind)) {
            const rawEvId = parsed.proposedAction.sourceEvidenceId || activeEvidenceId
            const validEvId = args.evidence.some((e) => e.evidenceId === rawEvId)
              ? rawEvId
              : undefined
            action = {
              actionId: `ACT-${Date.now().toString(36).toUpperCase()}`,
              label: parsed.proposedAction.rationale || `Proposed ${kind}`,
              description:
                parsed.proposedAction.rationale ||
                'Creates a child run linked to the selected execution node and evidence.',
              kind,
              sourceNodeId: context.selectedNodeId || parsed.proposedAction.sourceNodeId,
              sourceEvidenceId: validEvId,
              parameters: parsed.proposedAction.parameters || { depth: 'focused' },
            }
          }
        }
      }
    } catch {
      // If structured parsing fails, show conversational response only.
      // KEYWORD_TO_EXECUTABLE_ACTION = 0 (Amendment 10)
    }

    return {
      id: `msg-${Date.now()}`,
      role: 'agent',
      timestamp: new Date().toISOString(),
      text: replyMessage,
      contextNodeId: context.selectedNodeId,
      evidenceIds: activeEvidenceId
        ? [activeEvidenceId]
        : args.evidence.slice(0, 3).map((e) => e.evidenceId),
      proposedAction: action,
    }
  }

  async review(
    args: {
      runId: string
      goal: string
      evidence: EvidenceRecord[]
      contextNodeId?: string
    },
    onChunk?: (chunk: string) => void,
    onFirstToken?: () => void
  ): Promise<ReviewerOutput> {
    if (!this.engine) {
      throw new Error(
        'WebLLM reviewer runtime is not initialized. Initialize the runtime before requesting a review.'
      )
    }

    const evidenceUniverse = args.evidence
      .slice(0, 10)
      .map(
        (e) =>
          `- [${e.evidenceId}] ${e.testId} (${e.status}): ${e.metrics
            .map((m) => `${m.name}=${m.value}`)
            .join(', ')}`
      )
      .join('\n')

    const systemPrompt = `You are an evidence-grounded engineering reviewer.
Output strictly valid JSON matching this schema:
{
  "executive_summary": "string",
  "findings": [
    {
      "finding_id": "string",
      "title": "string",
      "description": "string",
      "evidence_ids": ["string"],
      "limitations": ["string"],
      "suggested_actions": ["string"]
    }
  ],
  "limitations": ["string"],
  "evidence_ids": ["string"]
}

Cite only real bracketed Evidence IDs from the permitted list. Do not perform arithmetic.`

    const userPrompt = `Goal: ${args.goal}
Permitted Evidence Records:
${evidenceUniverse}

Please evaluate the empirical evidence and output your review strictly as JSON.`

    const chunks = await this.engine.chat.completions.create({
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      stream: true,
      temperature: 0.1,
      max_tokens: 800,
    })

    let fullText = ''
    let firstTokenNotified = false
    for await (const chunk of chunks) {
      const delta = chunk.choices[0]?.delta?.content || ''
      if (delta && !firstTokenNotified) {
        firstTokenNotified = true
        onFirstToken?.()
      }
      fullText += delta
      onChunk?.(delta)
    }

    let cleaned = fullText.trim()
    if (cleaned.startsWith('```')) {
      const firstNl = cleaned.indexOf('\n')
      if (firstNl !== -1) cleaned = cleaned.substring(firstNl + 1)
      if (cleaned.endsWith('```')) cleaned = cleaned.substring(0, cleaned.length - 3)
      cleaned = cleaned.trim()
    }

    try {
      const parsed = JSON.parse(cleaned)
      return {
        executiveSummary: parsed.executive_summary || 'Qualitative engineering review synthesized.',
        findings: Array.isArray(parsed.findings)
          ? parsed.findings.map((f: any) => ({
              findingId: f.finding_id || 'F-01',
              title: f.title || 'Observation',
              description: f.description || '',
              evidenceIds: Array.isArray(f.evidence_ids) ? f.evidence_ids : [],
              limitations: Array.isArray(f.limitations) ? f.limitations : [],
              suggestedActions: Array.isArray(f.suggested_actions) ? f.suggested_actions : [],
              metricRefs: Array.isArray(f.metric_refs)
                ? f.metric_refs.map((m: any) => ({
                    evidenceId: String(m.evidence_id || m.evidenceId || ''),
                    metricName: String(m.metric_name || m.metricName || ''),
                    value: m.value,
                  }))
                : undefined,
            }))
          : [],
        limitations: Array.isArray(parsed.limitations) ? parsed.limitations : [],
        evidenceIds: Array.isArray(parsed.evidence_ids) ? parsed.evidence_ids : [],
        rawStructuredOutput: parsed,
      }
    } catch {
      return {
        executiveSummary: 'Qualitative engineering review synthesized from evidence records.',
        findings: [],
        limitations: ['JSON parsing fallback applied.'],
        evidenceIds: args.evidence.map((e) => e.evidenceId),
      }
    }
  }

  async dispose(): Promise<void> {
    if (this.engine) {
      try {
        await this.engine.unload()
      } catch {
        // Unload suppression
      }
      this.engine = null
    }
    this.notifyProgress({ state: 'idle', label: 'Idle' })
  }
}

export const webLLMReviewer = new WebLLMReviewer()
