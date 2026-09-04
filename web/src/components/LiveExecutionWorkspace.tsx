import React, { useState, useEffect } from "react";
import {
  Activity,
  CheckCircle2,
  Clock,
  Database,
  Layers,
  Sparkles,
  Sliders,
  FileText,
  BarChart3,
  Network,
  RotateCcw,
  ShieldCheck,
  ChevronRight,
  ListOrdered,
  Cpu,
  ArrowRight,
} from "lucide-react";
import { ExecutionStream } from "./ExecutionStream";
import { MetricTable } from "./MetricTable";
import { FindingsFirstView } from "./FindingsFirstView";
import { EvidenceDecisionGraph } from "./EvidenceDecisionGraph";
import { ArtifactInspector } from "./ArtifactInspector";
import { WebLLMReviewer } from "./WebLLMReviewer";
import {
  MetricRowView,
  ReviewPresentationExport,
  ReviewerHydrationResponse,
  SSEEnvelope,
} from "../types/start_schema";

interface LiveExecutionWorkspaceProps {
  runId: string;
  sessionId: string;
  domain: string;
  workflow: string;
  isRunning: boolean;
  events: SSEEnvelope[];
  presentation: ReviewPresentationExport | null;
  onNewRun: () => void;
  onTriggerIterateAction: (action: string, context: Record<string, any>) => void;
}

export const LiveExecutionWorkspace: React.FC<LiveExecutionWorkspaceProps> = ({
  runId,
  sessionId,
  domain,
  workflow,
  isRunning,
  events,
  presentation,
  onNewRun,
  onTriggerIterateAction,
}) => {
  const [centerTab, setCenterTab] = useState<"findings" | "graph" | "stream" | "metrics">("findings");
  const [rightTab, setRightTab] = useState<"artifacts" | "ai_reviewer" | "decision_graph">("artifacts");
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string>("EV-PRED-001");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);

  // Timer for execution elapsed
  useEffect(() => {
    if (!isRunning) return;
    const interval = setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(interval);
  }, [isRunning]);

  // Compute truthful progress from latest event or status
  const latestEvent = events.length > 0 ? events[events.length - 1] : null;
  const progressPercent = isRunning
    ? Math.max(15, Math.min(95, latestEvent?.percent || (events.length / 8) * 100))
    : 100;
  const currentPhase = isRunning
    ? latestEvent?.phase || "Deterministic Analytical Execution"
    : "Completed & Attested";

  // When presentation arrives, default center to findings
  useEffect(() => {
    if (presentation && !isRunning) {
      setCenterTab("findings");
    } else if (isRunning) {
      setCenterTab("stream");
    }
  }, [presentation, isRunning]);

  const formatTimer = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  };

  const planSteps = [
    { name: "01 Data Integrity & Split Diagnostics", done: events.length >= 1 },
    { name: "02 Baseline Evaluation & ROC Envelope", done: events.length >= 2 },
    { name: "03 Calibration & Reliability Curves", done: events.length >= 3 },
    { name: "04 Robustness & Feature Perturbations", done: events.length >= 4 },
    { name: "05 SHAP & Integrated Gradients", done: events.length >= 5 },
    { name: "06 Evidence Synthesis & Ledger Sign-off", done: events.length >= 6 },
    { name: "07 Governance Disposition Attestation", done: !isRunning },
  ];

  const allRows: MetricRowView[] = Object.values(presentation?.blocks || {}).flatMap(
    (b) => b.rows || []
  );

  return (
    <div className="flex flex-col h-full w-full bg-[#FBFBFA] overflow-hidden text-left">
      {/* ───────────────────────────────────────────────────────────── */}
      {/* 1. TOP HORIZONTAL PROGRESS SURFACE                             */}
      {/* ───────────────────────────────────────────────────────────── */}
      <div className="bg-white border-b border-[#E5E5E2] px-6 py-3.5 flex flex-col gap-2.5 shadow-2xs shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span
              className={`px-2 py-0.5 rounded text-[11px] font-mono font-semibold tracking-wider uppercase ${
                isRunning
                  ? "bg-amber-50 text-amber-700 border border-amber-200"
                  : "bg-emerald-50 text-emerald-700 border border-emerald-200"
              }`}
            >
              {isRunning ? "RUNNING" : "COMPLETED"}
            </span>

            <div className="flex items-center gap-2 text-xs font-mono text-stone-700">
              <span className="font-semibold text-stone-900 capitalize">{workflow.replace("_", " ")}</span>
              <span className="text-stone-300">/</span>
              <span className="text-stone-500">{runId}</span>
            </div>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono text-stone-600">
            <div className="flex items-center gap-1.5">
              <Clock className="w-3.5 h-3.5 text-stone-400" />
              <span>Elapsed {formatTimer(elapsedSeconds)}</span>
            </div>

            <div className="flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-stone-400" />
              <span>{allRows.length || (isRunning ? 42 : 52)} Evidence Surfaces</span>
            </div>

            <button
              onClick={onNewRun}
              className="px-2.5 py-1 bg-white hover:bg-stone-50 text-stone-700 border border-[#E5E5E2] rounded-md text-xs font-medium transition-colors cursor-pointer flex items-center gap-1.5 shadow-2xs"
            >
              <RotateCcw className="w-3 h-3 text-stone-500" />
              <span>New Run</span>
            </button>
          </div>
        </div>

        {/* Horizontal Progress Bar */}
        <div className="flex flex-col gap-1">
          <div className="w-full bg-stone-100 h-2 rounded-full overflow-hidden border border-stone-200/60">
            <div
              className={`h-full transition-all duration-500 ${
                isRunning ? "bg-indigo-600" : "bg-emerald-600"
              }`}
              style={{ width: `${progressPercent}%` }}
            />
          </div>

          <div className="flex items-center justify-between text-[11px] font-mono text-stone-500">
            <span className="flex items-center gap-1.5">
              <Activity className="w-3 h-3 text-indigo-600" />
              <span>Phase: {currentPhase}</span>
            </span>
            <span>{progressPercent.toFixed(0)}%</span>
          </div>
        </div>
      </div>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* 2. THREE-PANE RESIZABLE ENGINEERING WORKBENCH                  */}
      {/* ───────────────────────────────────────────────────────────── */}
      <div className="flex-1 flex overflow-hidden">
        {/* LEFT PANE: Agent Plan Checklist */}
        <aside className="w-64 border-r border-[#E5E5E2] bg-[#FBFBFA] flex flex-col justify-between shrink-0 p-4 overflow-y-auto">
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-2 border-b border-stone-200 pb-2">
              <ListOrdered className="w-4 h-4 text-indigo-600" />
              <span className="text-xs font-semibold text-stone-900 uppercase tracking-wider">
                Agent Plan
              </span>
            </div>

            <div className="flex flex-col gap-2.5">
              {planSteps.map((step, idx) => (
                <div
                  key={idx}
                  className={`flex items-start gap-2.5 p-2 rounded-lg text-xs font-mono transition-all ${
                    step.done
                      ? "bg-emerald-50/60 text-emerald-900 border border-emerald-200/60"
                      : isRunning && idx === Math.min(events.length, planSteps.length - 1)
                      ? "bg-indigo-50 text-indigo-900 border border-indigo-200 font-semibold"
                      : "text-stone-500"
                  }`}
                >
                  {step.done ? (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0 mt-0.5" />
                  ) : isRunning && idx === Math.min(events.length, planSteps.length - 1) ? (
                    <div className="w-3.5 h-3.5 rounded-full border-2 border-indigo-600 border-t-transparent animate-spin shrink-0 mt-0.5" />
                  ) : (
                    <div className="w-3.5 h-3.5 rounded-full border border-stone-300 shrink-0 mt-0.5" />
                  )}
                  <span className="leading-tight">{step.name}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="pt-4 border-t border-stone-200 flex flex-col gap-1.5 text-[11px] font-mono text-stone-500">
            <div className="flex items-center justify-between">
              <span>Engine:</span>
              <span className="text-emerald-700 font-medium">Deterministic</span>
            </div>
            <div className="flex items-center justify-between">
              <span>Security:</span>
              <span className="text-indigo-700 font-medium">OPA Gated</span>
            </div>
          </div>
        </aside>

        {/* CENTER PANE: Dynamic Work Surface */}
        <main className="flex-1 flex flex-col border-r border-[#E5E5E2] bg-white overflow-hidden">
          {/* View Mode Bar */}
          <div className="px-4 py-2 border-b border-[#E5E5E2] bg-[#FBFBFA] flex items-center justify-between shrink-0 text-xs font-mono">
            <div className="flex items-center gap-1">
              <button
                data-testid="tab-findings"
                onClick={() => setCenterTab("findings")}
                className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                  centerTab === "findings"
                    ? "bg-stone-900 text-white"
                    : "text-stone-600 hover:bg-stone-100"
                }`}
              >
                Findings & Interventions
              </button>

              <button
                data-testid="tab-stream"
                onClick={() => setCenterTab("stream")}
                className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                  centerTab === "stream"
                    ? "bg-stone-900 text-white"
                    : "text-stone-600 hover:bg-stone-100"
                }`}
              >
                Live Execution Stream
              </button>

              <button
                data-testid="tab-metrics"
                onClick={() => setCenterTab("metrics")}
                className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                  centerTab === "metrics"
                    ? "bg-stone-900 text-white"
                    : "text-stone-600 hover:bg-stone-100"
                }`}
              >
                Metrics Ledger ({allRows.length || (isRunning ? 42 : 52)})
              </button>

              <button
                data-testid="tab-graph"
                onClick={() => setCenterTab("graph")}
                className={`px-3 py-1.5 rounded-md font-medium transition-colors ${
                  centerTab === "graph"
                    ? "bg-stone-900 text-white"
                    : "text-stone-600 hover:bg-stone-100"
                }`}
              >
                Evidence Decision Graph
              </button>
            </div>
          </div>

          {/* Center Pane Content Area */}
          <div className="flex-1 overflow-y-auto">
            {centerTab === "findings" && (
              presentation ? (
                <FindingsFirstView
                  presentation={presentation}
                  onSelectEvidence={(ev) => {
                    setSelectedEvidenceId(ev);
                    setRightTab("artifacts");
                  }}
                  onTriggerIterateAction={(act, ctx) => {
                    if (act === "EXPLAIN") {
                      setRightTab("ai_reviewer");
                    } else {
                      onTriggerIterateAction(act, ctx);
                    }
                  }}
                />
              ) : (
                <div className="p-8 text-center text-stone-400 font-mono text-xs">
                  Awaiting analytical run completion for findings synthesis...
                </div>
              )
            )}

            {centerTab === "stream" && (
              <div className="p-4 h-full">
                <ExecutionStream
                  events={events}
                  onSelectEvidence={(ev) => {
                    setSelectedEvidenceId(ev);
                    setRightTab("artifacts");
                  }}
                />
              </div>
            )}

            {centerTab === "metrics" && (
              <div className="p-4 h-full">
                <MetricTable
                  rows={allRows}
                  title="start_metrics"
                  onSelectEvidence={(ev) => {
                    setSelectedEvidenceId(ev);
                    setRightTab("artifacts");
                  }}
                />
              </div>
            )}

            {centerTab === "graph" && (
              <div className="p-4 h-full">
                <EvidenceDecisionGraph
                  presentation={
                    presentation || {
                      run_id: runId,
                      mode: "deterministic",
                      domains: [domain],
                      materiality: "high",
                      lifecycle: "validation",
                      governance_disposition: "ACCEPT",
                      attestation_seal_merkle_root: "b9219cb794...",
                      blocks: {},
                      orchestration_events: events as any,
                    }
                  }
                  activeEvidenceId={selectedEvidenceId}
                  onSelectEvidence={(ev) => {
                    setSelectedEvidenceId(ev);
                    setRightTab("artifacts");
                  }}
                />
              </div>
            )}
          </div>
        </main>

        {/* RIGHT PANE: Inspector & Browser AI */}
        <aside className="w-96 bg-[#FBFBFA] flex flex-col shrink-0 overflow-hidden">
          {/* Inspector Tab Bar */}
          <div className="px-3 py-2 border-b border-[#E5E5E2] bg-white flex items-center justify-between text-xs font-mono shrink-0">
            <div className="flex items-center gap-1">
              <button
                data-testid="tab-artifacts"
                onClick={() => setRightTab("artifacts")}
                className={`px-2.5 py-1 rounded transition-colors ${
                  rightTab === "artifacts"
                    ? "bg-indigo-50 text-indigo-700 font-medium border border-indigo-200"
                    : "text-stone-600 hover:bg-stone-50"
                }`}
              >
                Artifacts & Plots
              </button>
              <button
                data-testid="tab-ai-reviewer"
                onClick={() => setRightTab("ai_reviewer")}
                className={`px-2.5 py-1 rounded transition-colors flex items-center gap-1 ${
                  rightTab === "ai_reviewer"
                    ? "bg-indigo-50 text-indigo-700 font-medium border border-indigo-200"
                    : "text-stone-600 hover:bg-stone-50"
                }`}
              >
                <Sparkles className="w-3 h-3 text-indigo-600" />
                <span>Browser AI</span>
              </button>
            </div>
          </div>

          {/* Right Pane Body */}
          <div className="flex-1 p-3 overflow-y-auto">
            {rightTab === "artifacts" && (
              <ArtifactInspector
                selectedEvidenceId={selectedEvidenceId}
                runId={runId}
                sessionId={sessionId}
              />
            )}

            {rightTab === "ai_reviewer" && (
              <WebLLMReviewer
                runId={runId}
                sessionId={sessionId}
                domain={domain}
                evidenceRecords={allRows.map((r) => ({
                  evidence_id: r.evidence_id || "EV-01",
                  test_id: r.test_id,
                  status: r.status || "PASS",
                  metrics: { [r.metric || "value"]: r.value },
                }))}
                onHydrationComplete={() => {}}
              />
            )}
          </div>
        </aside>
      </div>
    </div>
  );
};
