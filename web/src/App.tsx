import React, { useState, useEffect, useRef } from "react";
import {
  Sparkles,
  Cpu,
  Activity,
  RotateCcw,
  ShieldCheck,
  ListOrdered,
  FileText,
  HelpCircle,
  Database,
  ExternalLink,
} from "lucide-react";
import { AgenticComposer, IterationContext } from "./components/AgenticComposer";
import { LiveExecutionWorkspace } from "./components/LiveExecutionWorkspace";
import {
  fetchRunPresentation,
  fetchSystemInfo,
  startAnalyticalRun,
  subscribeRunEvents,
} from "./services/api";
import { webLLMService } from "./services/webllm";
import {
  ReviewPresentationExport,
  RunRequest,
  SSEEnvelope,
  SystemInfo,
} from "./types/start_schema";

export function App() {
  const [currentView, setCurrentView] = useState<"COMPOSE" | "WORKSPACE">("COMPOSE");

  // Run & Execution State
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [activeDomain, setActiveDomain] = useState<string>("predictive");
  const [activeWorkflow, setActiveWorkflow] = useState<string>("predictive_ml");
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [events, setEvents] = useState<SSEEnvelope[]>([]);
  const [presentation, setPresentation] = useState<ReviewPresentationExport | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [iterationContext, setIterationContext] = useState<IterationContext | null>(null);

  // Session ID (persisted in sessionStorage for refresh reconnection)
  const [sessionId] = useState<string>(() => {
    const stored = sessionStorage.getItem("start_session_id");
    if (stored) return stored;
    const generated = `SES-${Math.random().toString(36).substring(2, 11)}`;
    sessionStorage.setItem("start_session_id", generated);
    return generated;
  });

  // System & WebLLM status
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [webGPUAvailable, setWebGPUAvailable] = useState<boolean | null>(null);
  const [aiStatus, setAiStatus] = useState<"READY" | "LOADING" | "OFFLINE" | "CHECKING">("CHECKING");
  const [aiProgress, setAiProgress] = useState<number>(0);

  const unsubscribeRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    fetchSystemInfo().then(setSystemInfo).catch(() => {});
    webLLMService.checkWebGPUSupport().then((sup) => {
      setWebGPUAvailable(sup);
      setAiStatus(sup ? "READY" : "OFFLINE");
    });

    // Check if previous active run in session can be restored
    const storedRunId = sessionStorage.getItem("start_active_run_id");
    const storedDomain = sessionStorage.getItem("start_active_domain");
    const storedWorkflow = sessionStorage.getItem("start_active_workflow");
    if (storedRunId) {
      setActiveRunId(storedRunId);
      if (storedDomain) setActiveDomain(storedDomain);
      if (storedWorkflow) setActiveWorkflow(storedWorkflow);
      setCurrentView("WORKSPACE");

      // Fetch presentation
      fetchRunPresentation(storedRunId, sessionId)
        .then((pres) => {
          if (pres) {
            setPresentation(pres);
          }
        })
        .catch(() => {});
    }
  }, [sessionId]);

  // Launch Analytical Run
  const handleLaunchRun = async (req: RunRequest) => {
    setIsRunning(true);
    setEvents([]);
    setPresentation(null);
    setErrorMessage(null);
    setIterationContext(null);
    setCurrentView("WORKSPACE");
    setActiveDomain(req.domain || "predictive");
    setActiveWorkflow(req.workflow || "predictive_ml");

    // Initialize WebLLM concurrently in background without blocking deterministic execution
    if (webGPUAvailable && !webLLMService.isReady()) {
      setAiStatus("LOADING");
      webLLMService
        .initialize((p) => {
          setAiProgress(p.progress);
        })
        .then(() => setAiStatus("READY"))
        .catch(() => setAiStatus("OFFLINE"));
    }

    try {
      const { run_id } = await startAnalyticalRun(req);
      setActiveRunId(run_id);
      sessionStorage.setItem("start_active_run_id", run_id);
      sessionStorage.setItem("start_active_domain", req.domain || "predictive");
      sessionStorage.setItem("start_active_workflow", req.workflow || "predictive_ml");

      // Subscribe to real SSE stream
      if (unsubscribeRef.current) unsubscribeRef.current();
      unsubscribeRef.current = subscribeRunEvents(
        run_id,
        sessionId,
        undefined,
        (envelope) => {
          setEvents((prev) => [...prev, envelope]);
        },
        async () => {
          // Completed
          try {
            const pres = await fetchRunPresentation(run_id, sessionId);
            if (pres) setPresentation(pres);
          } catch (e) {
            console.error("Failed to load presentation", e);
          } finally {
            setIsRunning(false);
          }
        },
        (err) => {
          console.error("SSE stream notification", err);
          setIsRunning(false);
        }
      );
    } catch (err: any) {
      setErrorMessage(err.message || String(err));
      setIsRunning(false);
    }
  };

  const handleNewRun = () => {
    if (unsubscribeRef.current) {
      unsubscribeRef.current();
      unsubscribeRef.current = null;
    }
    sessionStorage.removeItem("start_active_run_id");
    sessionStorage.removeItem("start_active_domain");
    sessionStorage.removeItem("start_active_workflow");
    setActiveRunId(null);
    setPresentation(null);
    setEvents([]);
    setErrorMessage(null);
    setIterationContext(null);
    setCurrentView("COMPOSE");
  };

  const handleTriggerIterateAction = (action: string, context: Record<string, any>) => {
    // Switch to Composer pre-filled with real iterative lineage context
    setIterationContext({
      parent_run_id: activeRunId || "RUN-PREV",
      action,
      evidenceId: context.evidenceId,
      title: context.finding || context.title,
      parameterDelta: context.parameterDelta,
    });
    setCurrentView("COMPOSE");
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-[#FBFBFA] text-stone-900 font-sans overflow-hidden antialiased select-none">
      {/* ───────────────────────────────────────────────────────────── */}
      {/* HEADER BAR (PORCELAIN WORKBENCH)                               */}
      {/* ───────────────────────────────────────────────────────────── */}
      <header className="h-13 border-b border-[#E5E5E2] bg-white px-6 flex items-center justify-between shrink-0 shadow-2xs z-20">
        {/* Left: Brand Identity */}
        <div className="flex items-center gap-4">
          <div
            onClick={handleNewRun}
            className="flex items-center gap-2.5 cursor-pointer group"
          >
            <div className="w-7 h-7 rounded-md bg-stone-900 text-white flex items-center justify-center font-bold text-xs shadow-xs group-hover:bg-indigo-600 transition-colors">
              ⚡
            </div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-sm tracking-tight text-stone-900">
                StART
              </span>
              <span className="text-[11px] font-mono text-stone-400">
                Workbench
              </span>
            </div>
          </div>

          <div className="h-4 w-px bg-stone-200" />

          <nav className="flex items-center gap-1 text-xs font-medium">
            <button
              onClick={handleNewRun}
              className={`px-2.5 py-1 rounded-md transition-colors ${
                currentView === "COMPOSE"
                  ? "bg-stone-100 text-stone-900 font-semibold"
                  : "text-stone-600 hover:text-stone-900 hover:bg-stone-50"
              }`}
            >
              New Run
            </button>
            {activeRunId && (
              <button
                onClick={() => setCurrentView("WORKSPACE")}
                className={`px-2.5 py-1 rounded-md transition-colors ${
                  currentView === "WORKSPACE"
                    ? "bg-stone-100 text-stone-900 font-semibold"
                    : "text-stone-600 hover:text-stone-900 hover:bg-stone-50"
                }`}
              >
                Active Run ({activeRunId.substring(0, 12)})
              </button>
            )}
          </nav>
        </div>

        {/* Right: Status Indicators */}
        <div className="flex items-center gap-3 text-xs font-mono">
          {/* Engine Status */}
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-stone-50 border border-stone-200/80 text-stone-700">
            <span
              className={`w-2 h-2 rounded-full ${
                isRunning ? "bg-amber-500 animate-pulse" : "bg-emerald-500"
              }`}
            />
            <span>Engine: {isRunning ? "Running" : "Ready"}</span>
          </div>

          {/* Browser AI Status */}
          <div className="flex items-center gap-1.5 px-2 py-1 rounded bg-stone-50 border border-stone-200/80 text-stone-700">
            {aiStatus === "LOADING" ? (
              <>
                <span className="w-2 h-2 rounded-full bg-indigo-500 animate-ping" />
                <span>AI: Loading {aiProgress.toFixed(0)}%</span>
              </>
            ) : aiStatus === "READY" ? (
              <>
                <span className="w-2 h-2 rounded-full bg-emerald-500" />
                <span>Browser AI: Ready</span>
              </>
            ) : (
              <>
                <span className="w-2 h-2 rounded-full bg-stone-400" />
                <span>Browser AI: Offline</span>
              </>
            )}
          </div>
        </div>
      </header>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* ERROR BANNER (IF ANY)                                          */}
      {/* ───────────────────────────────────────────────────────────── */}
      {errorMessage && (
        <div className="bg-rose-50 border-b border-rose-200 px-6 py-2.5 flex items-center justify-between text-xs text-rose-800">
          <div className="flex items-center gap-2">
            <span className="font-semibold">Run Error:</span>
            <span>{errorMessage}</span>
          </div>
          <button
            onClick={() => setErrorMessage(null)}
            className="text-rose-600 hover:text-rose-900 font-bold px-2 cursor-pointer"
          >
            ✕
          </button>
        </div>
      )}

      {/* ───────────────────────────────────────────────────────────── */}
      {/* MAIN VIEW AREA                                                */}
      {/* ───────────────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-hidden">
        {currentView === "COMPOSE" ? (
          <AgenticComposer
            onLaunchRun={handleLaunchRun}
            isRunning={isRunning}
            sessionId={sessionId}
            iterationContext={iterationContext}
            onClearIteration={() => setIterationContext(null)}
          />
        ) : (
          <LiveExecutionWorkspace
            runId={activeRunId || "RUN-WEB-PENDING"}
            sessionId={sessionId}
            domain={activeDomain}
            workflow={activeWorkflow}
            isRunning={isRunning}
            events={events}
            presentation={presentation}
            onNewRun={handleNewRun}
            onTriggerIterateAction={handleTriggerIterateAction}
          />
        )}
      </div>
    </div>
  );
}

export default App;
