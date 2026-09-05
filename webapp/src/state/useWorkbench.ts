import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { StartBackend } from '../contracts/backend'
import type { ReviewerRuntime } from '../contracts/reviewer'
import type {
  AgentPlanPreview, ArtifactRecord, AttestationState, Capability, ConversationMessage,
  EvidenceRecord, ExecutionContext, ExecutionGraph, Finding, GovernanceState, ProposedAction,
  RunRequest, RunSnapshot, RuntimeEvent, WorkflowId
} from '../contracts/types'

export function useWorkbench(backend: StartBackend, reviewer?: ReviewerRuntime) {
  const [capabilities, setCapabilities] = useState<Capability[]>([])
  const [contexts, setContexts] = useState<ExecutionContext[]>([])
  const [selectedWorkflow, _setSelectedWorkflow] = useState<WorkflowId | null>(null)
  const [selectedContext, _setSelectedContext] = useState<string | null>(null)
  const [goal, setGoal] = useState('')
  const [plan, setPlan] = useState<AgentPlanPreview | null>(null)

  const setSelectedWorkflow = useCallback((w: WorkflowId | null) => {
    _setSelectedWorkflow(w)
    setPlan(null)
  }, [])

  const setSelectedContext = useCallback((c: string | null) => {
    _setSelectedContext(c)
    setPlan(null)
  }, [])
  const [run, setRun] = useState<RunSnapshot | null>(null)
  const [events, setEvents] = useState<RuntimeEvent[]>([])
  const [graph, setGraph] = useState<ExecutionGraph>({ nodes: [], edges: [] })
  const [evidence, setEvidence] = useState<EvidenceRecord[]>([])
  const [findings, setFindings] = useState<Finding[]>([])
  const [artifacts, setArtifacts] = useState<ArtifactRecord[]>([])
  const [governance, setGovernance] = useState<GovernanceState | null>(null)
  const [attestation, setAttestation] = useState<AttestationState | null>(null)
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const subscription = useRef<{ close(): void } | null>(null)
  const seenEventIds = useRef<Set<string>>(new Set())
  const lastSequence = useRef<number>(0)

  useEffect(() => {
    Promise.all([backend.getCapabilities(), backend.listExecutionContexts()])
      .then(([c, x]) => {
        setCapabilities(c)
        setContexts(x)
      })
      .catch((e) => setError(e.message))
  }, [backend])

  useEffect(() => () => subscription.current?.close(), [])

  const refreshRunData = useCallback(
    async (runId: string) => {
      try {
        const [snap, g, e, f, a, gov, att] = await Promise.all([
          backend.getRun(runId),
          backend.getExecutionGraph(runId),
          backend.getEvidence(runId),
          backend.getFindings(runId),
          backend.getArtifacts(runId),
          backend.getGovernance(runId),
          backend.getAttestation(runId),
        ])
        setRun(snap)
        setGraph(g)
        setEvidence(e)
        setFindings(f)
        setArtifacts(a)
        setGovernance(gov)
        setAttestation(att)
      } catch (err) {
        setError((err as Error).message)
      }
    },
    [backend]
  )

  const handleEvent = useCallback(
    (ev: RuntimeEvent) => {
      // Deduplicate by eventId (Amendment 12)
      if (seenEventIds.current.has(ev.eventId)) {
        return
      }
      seenEventIds.current.add(ev.eventId)

      // Maintain monotonic sequence
      if (ev.sequence && ev.sequence < lastSequence.current) {
        // out of order, ignore
      } else if (ev.sequence) {
        lastSequence.current = ev.sequence
      }

      setEvents((prev) => [...prev, ev].slice(-300))

      if (ev.nodeId) {
        setSelectedNodeId((curr) => curr || ev.nodeId!)
      }

      // Granular event-driven state transitions without full polling (Amendment 11)
      if (ev.progress) {
        setRun((curr) => (curr ? { ...curr, progress: ev.progress, statusLabel: ev.title || curr.statusLabel } : curr))
      }

      if (ev.nodeId) {
        setGraph((curr) => ({
          ...curr,
          nodes: curr.nodes.map((n) => (n.id === ev.nodeId ? { ...n, status: ev.status === 'completed' ? 'completed' : 'running' } : n)),
        }))
      }

      const isCompletedEvent = ev.type === 'run_completed' || ev.type === 'complete' || (ev.status === 'completed' && ev.progress?.percent === 100)

      if (isCompletedEvent) {
        subscription.current?.close()
        setRun((curr) =>
          curr
            ? {
                ...curr,
                phase: 'completed',
                statusLabel: 'Run signed off',
                progress: ev.progress || curr.progress || (curr.plan.length > 0 ? {
                  label: 'Completed',
                  percent: 100,
                  completed: curr.plan.length,
                  total: curr.plan.length,
                  detail: 'Deterministic verification sealed',
                } : undefined),
              }
            : curr
        )
        refreshRunData(ev.runId)
      } else if (ev.type === 'phase') {
        setRun((curr) =>
          curr
            ? {
                ...curr,
                phase: ev.status === 'completed' ? 'completed' : 'running',
                statusLabel: ev.title,
                progress: ev.progress || curr.progress,
              }
            : curr
        )
      } else if (ev.type === 'evidence_created' || ev.type === 'evidence_commit') {
        backend.getEvidence(ev.runId).then(setEvidence).catch(() => {})
      } else if (ev.type === 'finding_created') {
        backend.getFindings(ev.runId).then(setFindings).catch(() => {})
      } else if (ev.type === 'artifact_created') {
        backend.getArtifacts(ev.runId).then(setArtifacts).catch(() => {})
      } else if (ev.type === 'governance' || ev.type === 'governance_seal') {
        backend.getGovernance(ev.runId).then(setGovernance).catch(() => {})
        backend.getAttestation(ev.runId).then(setAttestation).catch(() => {})
      } else if (ev.type === 'attested') {
        backend.getAttestation(ev.runId).then(setAttestation).catch(() => {})
      }
    },
    [backend, refreshRunData]
  )

  const buildRequest = useCallback(
    (overrides?: Partial<RunRequest>): RunRequest => ({
      workflowId: (overrides?.workflowId || selectedWorkflow)!,
      contextId: overrides?.contextId || selectedContext!,
      goal: overrides?.goal ?? goal,
      parameters: overrides?.parameters || {},
      ...overrides,
    }),
    [selectedWorkflow, selectedContext, goal]
  )

  const previewPlan = useCallback(async () => {
    if (!selectedWorkflow || !selectedContext) return
    setBusy(true)
    setError(null)
    try {
      setPlan(await backend.createPlan(buildRequest()))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setBusy(false)
    }
  }, [backend, buildRequest, selectedWorkflow, selectedContext])

  const startRun = useCallback(
    async (overrides?: Partial<RunRequest>) => {
      if (!selectedWorkflow || !selectedContext) return
      setBusy(true)
      setError(null)
      setEvents([])
      setEvidence([])
      setFindings([])
      setArtifacts([])
      setGovernance(null)
      setAttestation(null)
      setMessages([])
      seenEventIds.current.clear()
      lastSequence.current = 0

      try {
        const snap = await backend.createRun(buildRequest(overrides))
        setRun(snap)
        setPlan(null)
        setSelectedNodeId(snap.plan[0]?.id || null)

        subscription.current?.close()
        subscription.current = backend.streamRun(
          snap.runId,
          (ev) => handleEvent(ev),
          (err) => setError(err.message)
        )
        await refreshRunData(snap.runId)
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setBusy(false)
      }
    },
    [backend, buildRequest, selectedWorkflow, selectedContext, handleEvent, refreshRunData]
  )

  const askAgent = useCallback(
    async (text: string) => {
      if (!run) return
      const human: ConversationMessage = {
        id: `human-${Date.now()}`,
        role: 'human',
        timestamp: new Date().toISOString(),
        text,
        contextNodeId: selectedNodeId || undefined,
        evidenceIds: selectedEvidenceId ? [selectedEvidenceId] : undefined,
      }
      setMessages((m) => [...m, human])
      try {
        if (!reviewer) {
          throw new Error('Reviewer runtime is not available for engineering conversation.')
        }
        const reply = await reviewer.ask({
          runId: run.runId,
          text,
          evidence,
          contextNodeId: selectedNodeId || undefined,
        })
        setMessages((m) => [...m, reply])
      } catch (e) {
        setError((e as Error).message)
      }
    },
    [run, reviewer, evidence, selectedNodeId, selectedEvidenceId]
  )

  const attachRun = useCallback(
    async (snap: RunSnapshot) => {
      subscription.current?.close()
      seenEventIds.current.clear()
      lastSequence.current = 0

      setRun(snap)
      setPlan(null)
      setEvents([])
      setEvidence([])
      setFindings([])
      setArtifacts([])
      setGovernance(null)
      setAttestation(null)
      setMessages([])
      setSelectedEvidenceId(null)
      setSelectedNodeId(snap.plan[0]?.id || null)

      subscription.current = backend.streamRun(
        snap.runId,
        (ev) => handleEvent(ev),
        (err) => setError(err.message)
      )
      await refreshRunData(snap.runId)
    },
    [backend, handleEvent, refreshRunData]
  )

  const executeAction = useCallback(
    async (action: ProposedAction) => {
      if (!run) return
      setBusy(true)
      try {
        const validated = backend.validateAction
          ? await backend.validateAction(run.runId, action)
          : action
        const child = await backend.submitHumanAction(run.runId, validated)
        setSelectedWorkflow(child.workflowId)
        setSelectedContext(child.contextId)
        setGoal(validated.label)
        await attachRun(child)
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setBusy(false)
      }
    },
    [backend, run, attachRun]
  )

  const reset = useCallback(() => {
    subscription.current?.close()
    seenEventIds.current.clear()
    lastSequence.current = 0
    setSelectedWorkflow(null)
    setSelectedContext(null)
    setGoal('')
    setPlan(null)
    setRun(null)
    setEvents([])
    setGraph({ nodes: [], edges: [] })
    setEvidence([])
    setFindings([])
    setArtifacts([])
    setGovernance(null)
    setAttestation(null)
    setMessages([])
    setSelectedNodeId(null)
    setSelectedEvidenceId(null)
    setError(null)
  }, [])

  const selectedEvidence = useMemo(
    () => evidence.find((e) => e.evidenceId === selectedEvidenceId) || null,
    [evidence, selectedEvidenceId]
  )

  return {
    capabilities,
    contexts,
    selectedWorkflow,
    setSelectedWorkflow,
    selectedContext,
    setSelectedContext,
    goal,
    setGoal,
    plan,
    run,
    events,
    graph,
    evidence,
    findings,
    setFindings,
    artifacts,
    governance,
    setGovernance,
    attestation,
    setAttestation,
    messages,
    selectedNodeId,
    setSelectedNodeId,
    selectedEvidenceId,
    setSelectedEvidenceId,
    selectedEvidence,
    busy,
    error,
    setError,
    refreshRunData,
    previewPlan,
    startRun,
    askAgent,
    executeAction,
    reset,
  }
}
