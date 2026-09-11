import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { StartBackend } from '../contracts/backend'
import type { ReviewerContext, ReviewerRuntime } from '../contracts/reviewer'
import type {
  AgentPlanPreview, ArtifactRecord, AttestationState, Capability, CheckpointRecord,
  ConversationMessage, DecisionReceipt, EvidenceRecord, ExecutionContext, ExecutionGraph,
  Finding, GovernanceState, HandoffTransition, PinnedItem, ProposedAction, QuestionResponse,
  RunCompareResult, RunHistoryItem, RunLineage,
  RunRequest, RunSnapshot, RuntimeEvent, WorkflowId
} from '../contracts/types'
import { isViewableArtifactForCanvas } from '../app/splitterLayout'

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
  const [checkpoints, setCheckpoints] = useState<CheckpointRecord[]>([])
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<CheckpointRecord | null>(null)
  const [decisions, setDecisions] = useState<DecisionReceipt[]>([])
  const [handoffs, setHandoffs] = useState<HandoffTransition[]>([])
  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(null)
  const [activeHighlight, setActiveHighlight] = useState<{ type: 'stage' | 'evidence' | 'artifact' | 'checkpoint'; id: string } | null>(null)
  const [governance, setGovernance] = useState<GovernanceState | null>(null)
  const [attestation, setAttestation] = useState<AttestationState | null>(null)
  const [messages, setMessages] = useState<ConversationMessage[]>([])
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isReplay, setIsReplay] = useState(false)
  const [runtimeArtifactCanvasUnlocked, setRuntimeArtifactCanvasUnlocked] = useState(false)
  const activeLiveRunId = useRef<string | null>(null)
  const runStartArtifactIds = useRef<Set<string>>(new Set())

  // Pinning (localStorage)
  const [pinnedItems, setPinnedItems] = useState<PinnedItem[]>(() => {
    try {
      const saved = localStorage.getItem('start_pinned_items')
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })
  const [isPinnedDrawerOpen, setIsPinnedDrawerOpen] = useState(false)

  // History & Command Palette
  const [isHistoryOpen, setIsHistoryOpen] = useState(false)
  const [isSearchOpen, setIsSearchOpen] = useState(false)

  // Compare state
  const [isCompareOpen, setIsCompareOpen] = useState(false)
  const [compareRunA, setCompareRunA] = useState<string | null>(null)
  const [compareRunB, setCompareRunB] = useState<string | null>(null)
  const [compareResult, setCompareResult] = useState<RunCompareResult | null>(null)
  const [compareLoading, setCompareLoading] = useState(false)

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
        const [snap, g, e, f, a, gov, att, cps] = await Promise.all([
          backend.getRun(runId),
          backend.getExecutionGraph(runId),
          backend.getEvidence(runId),
          backend.getFindings(runId),
          backend.getArtifacts(runId),
          backend.getGovernance(runId),
          backend.getAttestation(runId),
          backend.getCheckpoints ? backend.getCheckpoints(runId) : Promise.resolve([]),
        ])
        setRun(snap)
        setGraph(g)
        setEvidence(e)
        setFindings(f)
        setArtifacts(a)
        setGovernance(gov)
        setAttestation(att)
        setCheckpoints(cps || [])
        if (a.length > 0) {
          setSelectedArtifactId((prev) => prev || a[0].artifactId)
        }
        return { snap, artifacts: a }
      } catch (err) {
        setError((err as Error).message)
        return null
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

      // Track genuine agent handoffs (150-250ms animation trigger)
      const sourceAgent = (ev as any).sourceAgent || (ev as any).source_agent
      const targetAgent = (ev as any).targetAgent || (ev as any).target_agent
      if (sourceAgent && targetAgent && sourceAgent !== targetAgent) {
        setHandoffs((prev) => [
          ...prev.slice(-9),
          {
            eventId: ev.eventId,
            sourceAgent,
            targetAgent,
            stage: (ev as any).stage || 'EXECUTION',
            action: (ev as any).action || ev.title,
            timestamp: ev.timestamp,
            active: true,
          },
        ])
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
        refreshRunData(ev.runId).then((res) => {
          if (res && res.artifacts) {
            const targetRunId = activeLiveRunId.current || ev.runId
            if (
              res.artifacts.some(
                (a) =>
                  (a.runId === targetRunId || (a as any).run_id === targetRunId) &&
                  !runStartArtifactIds.current.has(a.artifactId) &&
                  isViewableArtifactForCanvas(a)
              )
            ) {
              setRuntimeArtifactCanvasUnlocked(true)
            }
          }
        })
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
        const evRunId = ev.runId || (ev as any).run_id
        const targetRunId = activeLiveRunId.current || run?.runId
        backend.getArtifacts(ev.runId).then((arts) => {
          setArtifacts(arts)
          const newArtId = ev.artifactIds?.[0]
          if (newArtId) {
            setSelectedArtifactId(newArtId)
          }
          if (
            targetRunId &&
            arts.some(
              (a) =>
                (a.runId === targetRunId || (a as any).run_id === targetRunId) &&
                !runStartArtifactIds.current.has(a.artifactId) &&
                isViewableArtifactForCanvas(a)
            )
          ) {
            setRuntimeArtifactCanvasUnlocked(true)
          }
        }).catch(() => {})
      } else if (ev.type === 'checkpoint_committed' || (ev as any).checkpointId) {
        if (backend.getCheckpoints) {
          backend.getCheckpoints(ev.runId).then(setCheckpoints).catch(() => {})
        }
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

  const previewPlan = useCallback(async (overrides?: Partial<RunRequest>) => {
    if (!selectedWorkflow || !selectedContext) return
    setBusy(true)
    setError(null)
    try {
      setPlan(await backend.createPlan(buildRequest(overrides)))
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
      setCheckpoints([])
      setSelectedCheckpoint(null)
      setDecisions([])
      setHandoffs([])
      setSelectedArtifactId(null)
      setActiveHighlight(null)
      setGovernance(null)
      setAttestation(null)
      setMessages([])
      seenEventIds.current.clear()
      lastSequence.current = 0

      // Reset dynamic canvas lock for new live run
      activeLiveRunId.current = null
      runStartArtifactIds.current = new Set(artifacts.map((a) => a.artifactId))
      setRuntimeArtifactCanvasUnlocked(false)

      try {
        const snap = await backend.createRun(buildRequest(overrides))
        activeLiveRunId.current = snap.runId
        setRun(snap)
        setPlan(null)
        setSelectedNodeId(snap.plan[0]?.id || null)
        setIsReplay(false)
        setRuntimeArtifactCanvasUnlocked(false)
        if (window.location.hash !== `#run=${snap.runId}`) {
          window.location.hash = `#run=${snap.runId}`
        }

        subscription.current?.close()
        subscription.current = backend.streamRun(
          snap.runId,
          (ev) => handleEvent(ev),
          (err) => setError(err.message)
        )
        await refreshRunData(snap.runId)
        setRuntimeArtifactCanvasUnlocked(false)
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setBusy(false)
      }
    },
    [backend, buildRequest, selectedWorkflow, selectedContext, handleEvent, refreshRunData, artifacts]
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
        const reviewerContext: ReviewerContext = {
          runId: run.runId,
          selectedNodeId: selectedNodeId || undefined,
          selectedEvidenceId: selectedEvidenceId || undefined,
        }
        const reply = await reviewer.ask(reviewerContext, {
          text,
          evidence,
          runSnapshot: run,
        })
        setMessages((m) => [...m, reply])
      } catch (e) {
        setError((e as Error).message)
      }
    },
    [run, reviewer, evidence, selectedNodeId, selectedEvidenceId]
  )

  const recordDecision = useCallback(
    async (
      action: 'ACCEPT' | 'QUESTION' | 'CHALLENGE' | 'OVERRIDE' | 'RERUN' | 'ESCALATE',
      rationale: string,
      targetStage?: string,
      targetCheckpoint?: string,
      evidenceIds?: string[]
    ) => {
      if (!run || !backend.recordDecision) return null
      try {
        const receipt = await backend.recordDecision(run.runId, {
          action,
          rationale,
          target_stage: targetStage,
          target_checkpoint: targetCheckpoint,
          evidence_ids: evidenceIds || [],
        })
        setDecisions((prev) => [...prev, receipt])
        return receipt
      } catch (err) {
        setError((err as Error).message)
        throw err
      }
    },
    [run, backend]
  )

  const askQuestion = useCallback(
    async (queryText: string, targetStage?: string, evidenceId?: string) => {
      if (!run || !backend.askQuestion) return null
      try {
        const resp = await backend.askQuestion(run.runId, {
          question: queryText,
          targetStage,
          evidenceId,
        })
        if (resp.receipt) {
          setDecisions((prev) => [...prev, resp.receipt])
        }
        return resp
      } catch (err) {
        setError((err as Error).message)
        throw err
      }
    },
    [run, backend]
  )

  const attachRun = useCallback(
    async (snap: RunSnapshot) => {
      subscription.current?.close()
      seenEventIds.current.clear()
      activeLiveRunId.current = snap.runId
      runStartArtifactIds.current = new Set(artifacts.map((a) => a.artifactId))
      setRuntimeArtifactCanvasUnlocked(false)

      setRun(snap)
      setPlan(null)
      setIsReplay(false)
      if (window.location.hash !== `#run=${snap.runId}`) {
        window.location.hash = `#run=${snap.runId}`
      }
      setEvents([])
      setEvidence([])
      setFindings([])
      setArtifacts([])
      setCheckpoints([])
      setSelectedCheckpoint(null)
      setDecisions([])
      setHandoffs([])
      setSelectedArtifactId(null)
      setActiveHighlight(null)
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
      setRuntimeArtifactCanvasUnlocked(false)
    },
    [backend, handleEvent, refreshRunData, artifacts]
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

  const loadRun = useCallback(
    async (runId: string) => {
      setBusy(true)
      setError(null)
      try {
        subscription.current?.close()
        seenEventIds.current.clear()
        lastSequence.current = 0

        if (window.location.hash !== `#run=${runId}`) {
          window.location.hash = `#run=${runId}`
        }

        const result = await refreshRunData(runId)
        if (!result) return
        const { snap, artifacts: loadedArtifacts } = result

        setPlan(null)
        setSelectedWorkflow(snap.workflowId)
        setSelectedContext(snap.contextId)
        setGoal(snap.goal || '')
        setSelectedNodeId(snap.plan[0]?.id || null)

        const isCompleted = snap.phase === 'completed' || snap.phase === 'failed' || (snap as any).status === 'COMPLETED'
        setIsReplay(isCompleted)

        // Historical replay: if that persisted run already has viewable runtime artifacts:
        // runtimeArtifactCanvasUnlocked = true immediately.
        const hasPersistedRuntimeArtifacts = loadedArtifacts.some(
          (a) =>
            (a.runId === runId || (a as any).run_id === runId) &&
            isViewableArtifactForCanvas(a)
        )

        if (hasPersistedRuntimeArtifacts) {
          setRuntimeArtifactCanvasUnlocked(true)
        } else {
          setRuntimeArtifactCanvasUnlocked(false)
        }

        if (isCompleted) {
          activeLiveRunId.current = null
          runStartArtifactIds.current.clear()
        } else {
          activeLiveRunId.current = runId
          runStartArtifactIds.current = new Set(loadedArtifacts.map((a) => a.artifactId))
        }

        if (isCompleted) {
          const replayEvents = backend.getRunEvents ? await backend.getRunEvents(runId) : []
          setEvents(replayEvents)
        } else {
          subscription.current = backend.streamRun(
            snap.runId,
            (ev) => handleEvent(ev),
            (err) => setError(err.message)
          )
        }
      } catch (err) {
        setError((err as Error).message)
      } finally {
        setBusy(false)
      }
    },
    [backend, refreshRunData, handleEvent, setSelectedWorkflow, setSelectedContext]
  )

  // Pinning actions
  const pinItem = useCallback((item: Omit<PinnedItem, 'id' | 'timestamp'>) => {
    const newItem: PinnedItem = {
      ...item,
      id: `pin-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`,
      timestamp: new Date().toISOString(),
    }
    setPinnedItems((prev) => {
      const next = [newItem, ...prev.filter((p) => p.itemId !== item.itemId)]
      try {
        localStorage.setItem('start_pinned_items', JSON.stringify(next))
      } catch {}
      return next
    })
  }, [])

  const unpinItem = useCallback((itemIdOrId: string) => {
    setPinnedItems((prev) => {
      const next = prev.filter((p) => p.id !== itemIdOrId && p.itemId !== itemIdOrId)
      try {
        localStorage.setItem('start_pinned_items', JSON.stringify(next))
      } catch {}
      return next
    })
  }, [])

  const isPinned = useCallback(
    (itemId: string) => pinnedItems.some((p) => p.itemId === itemId),
    [pinnedItems]
  )

  // Compare actions
  const executeCompare = useCallback(
    async (runAId: string, runBId: string) => {
      if (!backend.compareRuns) return
      setCompareLoading(true)
      try {
        const res = await backend.compareRuns(runAId, runBId)
        setCompareResult(res)
      } catch (e) {
        setError((e as Error).message)
      } finally {
        setCompareLoading(false)
      }
    },
    [backend]
  )

  const openCompare = useCallback(
    (runAId?: string, runBId?: string) => {
      setIsCompareOpen(true)
      const a = runAId || compareRunA || (run?.parentRunId ? run.parentRunId : null)
      const b = runBId || compareRunB || (run ? run.runId : null)
      if (a) setCompareRunA(a)
      if (b) setCompareRunB(b)
      if (a && b) {
        executeCompare(a, b)
      }
    },
    [compareRunA, compareRunB, run, executeCompare]
  )

  const closeCompare = useCallback(() => {
    setIsCompareOpen(false)
  }, [])

  const openHistory = useCallback(() => setIsHistoryOpen(true), [])
  const closeHistory = useCallback(() => setIsHistoryOpen(false), [])
  const openSearch = useCallback(() => setIsSearchOpen(true), [])
  const closeSearch = useCallback(() => setIsSearchOpen(false), [])

  // Cmd+K palette shortcut
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        setIsSearchOpen((prev) => !prev)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  // Auto-restore run on page refresh or URL hash change
  useEffect(() => {
    const checkHash = () => {
      const hash = window.location.hash
      const match = hash.match(/#run=([a-zA-Z0-9_\-]+)/)
      if (match && match[1]) {
        loadRun(match[1])
        return
      }
      // Bare root navigation (no #run= in URL) MUST NOT restore any previous run
      try {
        sessionStorage.clear()
      } catch {}
    }

    checkHash()
    window.addEventListener('hashchange', checkHash)
    return () => window.removeEventListener('hashchange', checkHash)
  }, [loadRun])

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
    setCheckpoints([])
    setSelectedCheckpoint(null)
    setDecisions([])
    setHandoffs([])
    setSelectedArtifactId(null)
    setActiveHighlight(null)
    setGovernance(null)
    setAttestation(null)
    setMessages([])
    setSelectedNodeId(null)
    setSelectedEvidenceId(null)
    setError(null)
    setIsReplay(false)
    activeLiveRunId.current = null
    runStartArtifactIds.current.clear()
    setRuntimeArtifactCanvasUnlocked(false)
    window.location.hash = ''
    try {
      sessionStorage.clear()
    } catch {}
  }, [setSelectedWorkflow, setSelectedContext])

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
    isReplay,
    runtimeArtifactCanvasUnlocked,
    loadRun,
    events,
    graph,
    evidence,
    findings,
    setFindings,
    artifacts,
    checkpoints,
    selectedCheckpoint,
    setSelectedCheckpoint,
    decisions,
    handoffs,
    selectedArtifactId,
    setSelectedArtifactId,
    activeHighlight,
    setActiveHighlight,
    recordDecision,
    askQuestion,
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
    pinnedItems,
    pinItem,
    unpinItem,
    isPinned,
    isPinnedDrawerOpen,
    setIsPinnedDrawerOpen,
    isCompareOpen,
    compareRunA,
    compareRunB,
    compareResult,
    compareLoading,
    openCompare,
    closeCompare,
    setCompareRunA,
    setCompareRunB,
    executeCompare,
    isHistoryOpen,
    openHistory,
    closeHistory,
    isSearchOpen,
    openSearch,
    closeSearch,
  }
}

