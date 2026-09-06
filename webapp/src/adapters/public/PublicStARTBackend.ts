import type { StartBackend, StreamSubscription } from '../../contracts/backend'
import type { ReviewerOutput } from '../../contracts/reviewer'
import type {
  AgentPlanPreview, ArtifactRecord, AttestationState, Capability, EvidenceRecord,
  ExecutionContext, ExecutionGraph, Finding, GovernanceState, ProposedAction, ReviewerGateResult,
  RunRequest, RunSnapshot, RuntimeEvent
} from '../../contracts/types'
import {
  validateAgentPlanPreview, validateArtifactRecords, validateAttestationState, validateCapabilities,
  validateEvidenceRecords, validateExecutionContexts, validateExecutionGraph, validateFindings,
  validateGovernanceState, validateProposedAction, validateReviewerGateResult, validateRunSnapshot,
  validateRuntimeEvent
} from '../../contracts/validators'

/**
 * Public StART Backend Adapter.
 *
 * Implements the StartBackend interface against canonical StART HTTP and SSE endpoints.
 * Integrates schema validation at the transport boundary to guard against API contract drift.
 */
export class PublicStARTBackend implements StartBackend {
  readonly adapterName = 'Public StART backend adapter'
  readonly adapterMode = 'public' as const
  private activeSessionId: string | null = null

  constructor(private readonly baseUrl: string = '') {}

  private async json<T>(path: string, init?: RequestInit): Promise<T> {
    const url = `${this.baseUrl}${path}`
    const r = await fetch(url, {
      ...init,
      headers: {
        'content-type': 'application/json',
        accept: 'application/json',
        ...(init?.headers || {}),
      },
    })
    if (!r.ok) {
      let errText = ''
      try {
        const errJson = await r.json()
        errText = errJson.error || errJson.detail || JSON.stringify(errJson)
      } catch {
        errText = await r.text()
      }
      throw new Error(`StART backend ${r.status}: ${errText}`)
    }
    return r.json() as Promise<T>
  }

  private unwrap<T>(res: any, key?: string): unknown {
    if (res && typeof res === 'object' && 'success' in res && 'data' in res) {
      if (key && res.data && typeof res.data === 'object') {
        if (key in res.data) return res.data[key]
        if (key === 'evidence' && 'evidence_records' in res.data) return res.data.evidence_records
        if (key === 'artifacts' && 'artifacts' in res.data) return res.data.artifacts
      }
      return res.data
    }
    return res
  }

  async getCapabilities(): Promise<Capability[]> {
    const raw = await this.json<unknown>('/api/v1/capabilities')
    const unwrapped = this.unwrap(raw, 'capabilities')
    return validateCapabilities(unwrapped)
  }

  async listExecutionContexts(): Promise<ExecutionContext[]> {
    const raw = await this.json<unknown>('/api/v1/execution-contexts')
    const unwrapped = this.unwrap(raw, 'contexts')
    return validateExecutionContexts(unwrapped)
  }

  async createPlan(request: RunRequest): Promise<AgentPlanPreview> {
    const raw = await this.json<unknown>('/api/v1/plans', {
      method: 'POST',
      body: JSON.stringify(request),
    })
    const unwrapped = this.unwrap(raw, 'plan')
    return validateAgentPlanPreview(unwrapped)
  }

  async createRun(request: RunRequest): Promise<RunSnapshot> {
    const sessionId = (request as any).sessionId || (request as any).session_id || this.activeSessionId || `SES-${Date.now().toString(36)}`
    this.activeSessionId = sessionId
    const payload = { ...request, session_id: sessionId }
    const raw = await this.json<unknown>('/api/v1/runs', {
      method: 'POST',
      body: JSON.stringify(payload),
    })
    const unwrapped = this.unwrap(raw, 'run')
    return validateRunSnapshot(unwrapped)
  }

  async getRun(runId: string): Promise<RunSnapshot> {
    const raw = await this.json<unknown>(`/api/v1/runs/${runId}`)
    const unwrapped = this.unwrap(raw, 'run')
    return validateRunSnapshot(unwrapped)
  }

  streamRun(
    runId: string,
    onEvent: (event: RuntimeEvent) => void,
    onError?: (error: Error) => void
  ): StreamSubscription {
    const url = `${this.baseUrl}/api/v1/runs/${runId}/events`
    const source = new EventSource(url)

    const handleMsg = (e: MessageEvent) => {
      try {
        const parsed = JSON.parse(e.data)
        const validated = validateRuntimeEvent(parsed)
        onEvent(validated)
      } catch (err) {
        onError?.(err as Error)
      }
    }

    source.onmessage = handleMsg
    source.addEventListener('complete', handleMsg as EventListener)

    source.onerror = () => {
      if (source.readyState === 2) {
        // EventSource.CLOSED
        return
      }
      // Transient reconnect is handled by EventSource automatically; don't flash UI errors
    }

    return {
      close() {
        source.close()
      },
    }
  }

  async getExecutionGraph(runId: string): Promise<ExecutionGraph> {
    const raw = await this.json<unknown>(`/api/v1/runs/${runId}/graph`)
    const unwrapped = this.unwrap(raw, 'graph')
    return validateExecutionGraph(unwrapped)
  }

  async getEvidence(runId: string): Promise<EvidenceRecord[]> {
    const raw = await this.json<unknown>(`/api/v1/runs/${runId}/evidence`)
    const unwrapped = this.unwrap(raw, 'evidence')
    return validateEvidenceRecords(unwrapped)
  }

  async getFindings(runId: string): Promise<Finding[]> {
    const raw = await this.json<unknown>(`/api/v1/runs/${runId}/findings`)
    const unwrapped = this.unwrap(raw, 'findings')
    return validateFindings(unwrapped)
  }

  async getArtifacts(runId: string): Promise<ArtifactRecord[]> {
    const raw = await this.json<unknown>(`/api/v1/runs/${runId}/artifacts`)
    const unwrapped = this.unwrap(raw, 'artifacts')
    return validateArtifactRecords(unwrapped)
  }

  async validateAction(runId: string, action: ProposedAction): Promise<ProposedAction> {
    const raw = await this.json<unknown>(`/api/v1/runs/${runId}/actions/validate`, {
      method: 'POST',
      body: JSON.stringify(action),
    })
    const unwrapped = this.unwrap(raw, 'action')
    return validateProposedAction(unwrapped)
  }

  async submitHumanAction(runId: string, action: ProposedAction): Promise<RunSnapshot> {
    const raw = await this.json<unknown>(`/api/v1/runs/${runId}/actions`, {
      method: 'POST',
      body: JSON.stringify(action),
    })
    const unwrapped = this.unwrap(raw, 'child_run')
    return validateRunSnapshot(unwrapped)
  }

  async submitReviewerOutput(runId: string, review: ReviewerOutput): Promise<ReviewerGateResult> {
    const sevMap: Record<string, 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'> = {
      info: 'LOW',
      low: 'LOW',
      attention: 'MEDIUM',
      warn: 'MEDIUM',
      warning: 'MEDIUM',
      medium: 'MEDIUM',
      high: 'HIGH',
      critical: 'CRITICAL',
    }

    const submission = {
      run_id: runId,
      session_id: this.activeSessionId || `SES-${runId.slice(-8)}`,
      model_name: 'SmolLM2-1.7B-Instruct-q4f16_1-MLC',
      executive_summary: review.executiveSummary,
      findings: review.findings.map((f) => {
        const sev = f.severity ? (sevMap[f.severity.toLowerCase()] || 'MEDIUM') : 'MEDIUM'
        const evidence_refs = (f.metricRefs && f.metricRefs.length > 0)
          ? f.metricRefs.map((m) => ({
              evidence_id: m.evidenceId,
              metric_name: m.metricName,
            }))
          : f.evidenceIds.map((id) => ({
              evidence_id: id,
              metric_name: '',
            }))
        return {
          finding_id: f.findingId,
          severity: sev,
          title: f.title,
          description: f.description,
          evidence_refs,
          recommendation: f.suggestedActions?.[0] || '',
        }
      }),
      limitations: review.limitations || [],
      suggested_actions: review.findings.flatMap((f) => f.suggestedActions || []),
    }

    const raw = await this.json<unknown>(`/api/v1/runs/${runId}/reviewer/hydrate-and-gate`, {
      method: 'POST',
      body: JSON.stringify(submission),
    })
    const unwrapped = this.unwrap(raw) as any
    return validateReviewerGateResult({
      runId: unwrapped.run_id || runId,
      modelName: unwrapped.model_name || 'SmolLM2-1.7B-Instruct-q4f16_1-MLC',
      hydratedFindings: unwrapped.hydrated_findings || [],
      allGrounded: Boolean(unwrapped.all_grounded),
      governanceDisposition: unwrapped.governance_disposition,
      attestationSealMerkleRoot: unwrapped.attestation_seal_merkle_root || '',
    })
  }

  async getGovernance(runId: string): Promise<GovernanceState | null> {
    const raw = await this.json<unknown>(`/api/v1/runs/${runId}/governance`)
    const unwrapped = this.unwrap(raw, 'governance')
    return validateGovernanceState(unwrapped)
  }

  async getAttestation(runId: string): Promise<AttestationState | null> {
    const raw = await this.json<unknown>(`/api/v1/runs/${runId}/attestation`)
    const unwrapped = this.unwrap(raw, 'attestation')
    return validateAttestationState(unwrapped)
  }
}
