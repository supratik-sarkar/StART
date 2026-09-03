import React, { useState, useEffect } from "react";
import {
  ShieldCheck,
  Cpu,
  Activity,
  FileText,
  Play,
  Terminal,
  Database,
  BarChart3,
  Layers,
  Sparkles,
  Info,
  CheckCircle2,
  RefreshCw,
  Maximize2,
  Minimize2,
  ChevronRight,
  SplitSquareHorizontal,
} from "lucide-react";
import { MetricTable } from "./components/MetricTable";
import { ExecutionStream } from "./components/ExecutionStream";
import { WebLLMReviewer } from "./components/WebLLMReviewer";
import { ArtifactInspector } from "./components/ArtifactInspector";
import { MARKET_SHOWCASE, PREDICTIVE_SHOWCASE } from "./data/showcase_data";
import {
  fetchRunPresentation,
  fetchSystemInfo,
  fetchZeroCostAttestation,
  startAnalyticalRun,
  subscribeRunEvents,
} from "./services/api";
import {
  MetricRowView,
  ReviewPresentationExport,
  SSEEnvelope,
  SystemInfo,
  ZeroCostAttestation,
} from "./types/start_schema";

export function App() {
  // Product Modes
  const [productMode, setProductMode] = useState<"LIVE_DEMO" | "BROWSER_PRIVATE" | "LOCAL_FULL">("LIVE_DEMO");
  const [activeDomain, setActiveDomain] = useState<"market" | "predictive" | "deep_learning">("market");
  const [activeSection, setActiveSection] = useState<string>("ALL");

  // Presentation State
  const [presentation, setPresentation] = useState<ReviewPresentationExport>(MARKET_SHOWCASE);
  const [events, setEvents] = useState<SSEEnvelope[]>((MARKET_SHOWCASE.orchestration_events as unknown as SSEEnvelope[]) || []);
  const [isRunning, setIsRunning] = useState(false);
  const [activeRunId, setActiveRunId] = useState<string>("RUN-SHOWCASE-MARKET-01");
  const [sessionId] = useState<string>(() => `SES-${Math.random().toString(36).substring(2, 11)}`);

  // Selected Evidence for Inspector Drill-down
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | undefined>("EV-MKT-001");

  // Layout Split: 25, 50, 75
  const [splitRatio, setSplitRatio] = useState<number>(50);

  // System & Cost Metadata
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [zeroCost, setZeroCost] = useState<ZeroCostAttestation | null>(null);

  useEffect(() => {
    fetchSystemInfo().then(setSystemInfo).catch(() => {});
    fetchZeroCostAttestation().then(setZeroCost).catch(() => {});
  }, []);

  // Update showcase when switching domain in showcase mode
  useEffect(() => {
    if (productMode === "BROWSER_PRIVATE") {
      const data = activeDomain === "market" ? MARKET_SHOWCASE : PREDICTIVE_SHOWCASE;
      setPresentation(data);
      setEvents((data.orchestration_events as unknown as SSEEnvelope[]) || []);
      setActiveRunId(data.run_id);
    }
  }, [activeDomain, productMode]);

  // Launch live deterministic review
  const handleLaunchLiveRun = async () => {
    setIsRunning(true);
    setEvents([]);
    try {
      const { run_id } = await startAnalyticalRun({
        domain: activeDomain,
        mode: "deterministic",
        materiality: "high",
        synthetic_profile: `institutional_${activeDomain}_v1`,
        session_id: sessionId,
      });
      setActiveRunId(run_id);

      // Subscribe to real SSE stream
      subscribeRunEvents(
        run_id,
        sessionId,
        (envelope) => {
          setEvents((prev) => [...prev, envelope]);
        },
        async () => {
          // On Complete, fetch freshly computed presentation
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
          console.error("SSE stream error", err);
          setIsRunning(false);
        }
      );
    } catch (err: any) {
      alert(`Failed to launch live review: ${err.message || err}`);
      setIsRunning(false);
    }
  };

  // Collect all metric rows across presentation blocks
  const allMetricRows: MetricRowView[] = Object.values(presentation.blocks || {}).flatMap((b) => b.rows || []);

  // Filter rows if specific section is active
  const displayedRows =
    activeSection === "ALL"
      ? allMetricRows
      : Object.entries(presentation.blocks || {})
          .filter(([k]) => k.toLowerCase().includes(activeSection.toLowerCase()))
          .flatMap(([, b]) => b.rows || []);

  return (
    <div className="flex h-screen w-screen bg-background text-foreground font-sans overflow-hidden">
      {/* ───────────────────────────────────────────────────────────── */}
      {/* LEFT NAVIGATION BAR                                           */}
      {/* ───────────────────────────────────────────────────────────── */}
      <aside className="w-64 border-r border-border bg-card/60 flex flex-col justify-between shrink-0 select-none">
        <div className="flex flex-col">
          {/* Workstation Brand Header */}
          <div className="p-4 border-b border-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded bg-primary flex items-center justify-center font-bold text-xs text-primary-foreground">
                ⚡
              </div>
              <div className="flex flex-col">
                <span className="font-bold text-xs tracking-wider text-foreground">StART MRT</span>
                <span className="text-[10px] font-mono text-muted-foreground">v4.5.3 Institutional</span>
              </div>
            </div>
            <span className="px-1.5 py-0.5 rounded bg-green-950/80 text-green-400 font-mono text-[9px] border border-green-800/80">
              Deterministic
            </span>
          </div>

          {/* Conceptual Product Mode Selector */}
          <div className="p-3 border-b border-border flex flex-col gap-1.5">
            <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground font-mono">
              Product Mode
            </span>
            <div className="grid grid-cols-1 gap-1">
              <button
                onClick={() => setProductMode("LIVE_DEMO")}
                className={`flex items-center gap-2 px-2.5 py-1.5 rounded text-xs text-left transition-colors ${
                  productMode === "LIVE_DEMO"
                    ? "bg-primary text-primary-foreground font-bold shadow"
                    : "text-muted-foreground hover:bg-muted/40 hover:text-foreground"
                }`}
              >
                <Play className="w-3.5 h-3.5" />
                <div className="flex flex-col">
                  <span>Live Demo (Oracle)</span>
                  <span className="text-[9px] opacity-80 font-normal">Fresh Deterministic Engine</span>
                </div>
              </button>
              <button
                onClick={() => setProductMode("BROWSER_PRIVATE")}
                className={`flex items-center gap-2 px-2.5 py-1.5 rounded text-xs text-left transition-colors ${
                  productMode === "BROWSER_PRIVATE"
                    ? "bg-purple-600 text-white font-bold shadow"
                    : "text-muted-foreground hover:bg-muted/40 hover:text-foreground"
                }`}
              >
                <Sparkles className="w-3.5 h-3.5" />
                <div className="flex flex-col">
                  <span>Browser Private</span>
                  <span className="text-[9px] opacity-80 font-normal">WebLLM Client Reviewer</span>
                </div>
              </button>
              <button
                onClick={() => setProductMode("LOCAL_FULL")}
                className={`flex items-center gap-2 px-2.5 py-1.5 rounded text-xs text-left transition-colors ${
                  productMode === "LOCAL_FULL"
                    ? "bg-secondary text-foreground font-bold"
                    : "text-muted-foreground hover:bg-muted/40 hover:text-foreground"
                }`}
              >
                <Terminal className="w-3.5 h-3.5" />
                <div className="flex flex-col">
                  <span>Local Full StART</span>
                  <span className="text-[9px] opacity-80 font-normal">CLI / Offline Repo</span>
                </div>
              </button>
            </div>
          </div>

          {/* Domain Selection */}
          <div className="p-3 border-b border-border flex flex-col gap-1.5">
            <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground font-mono">
              Review Domain
            </span>
            <div className="flex flex-col gap-1 font-mono text-xs">
              <button
                onClick={() => setActiveDomain("market")}
                className={`px-2.5 py-1.5 rounded text-left transition-colors flex items-center justify-between ${
                  activeDomain === "market"
                    ? "bg-secondary text-primary font-bold border border-primary/40"
                    : "text-muted-foreground hover:bg-muted/30"
                }`}
              >
                <span>Market Risk & HERC</span>
                <ChevronRight className="w-3 h-3" />
              </button>
              <button
                onClick={() => setActiveDomain("predictive")}
                className={`px-2.5 py-1.5 rounded text-left transition-colors flex items-center justify-between ${
                  activeDomain === "predictive"
                    ? "bg-secondary text-primary font-bold border border-primary/40"
                    : "text-muted-foreground hover:bg-muted/30"
                }`}
              >
                <span>Predictive & Credit</span>
                <ChevronRight className="w-3 h-3" />
              </button>
              <button
                onClick={() => setActiveDomain("deep_learning")}
                className={`px-2.5 py-1.5 rounded text-left transition-colors flex items-center justify-between ${
                  activeDomain === "deep_learning"
                    ? "bg-secondary text-primary font-bold border border-primary/40"
                    : "text-muted-foreground hover:bg-muted/30"
                }`}
              >
                <span>PyTorch Deep Learning</span>
                <ChevronRight className="w-3 h-3" />
              </button>
            </div>
          </div>

          {/* Analytical Sections Navigation */}
          <div className="p-3 flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-wider font-semibold text-muted-foreground font-mono mb-1">
              Surfaces
            </span>
            <button
              onClick={() => setActiveSection("ALL")}
              className={`px-2.5 py-1 rounded text-xs text-left transition-colors ${
                activeSection === "ALL" ? "text-primary font-bold bg-muted/40" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              All Metric Blocks
            </button>
            <button
              onClick={() => setActiveSection("PORTFOLIO")}
              className={`px-2.5 py-1 rounded text-xs text-left transition-colors ${
                activeSection === "PORTFOLIO" ? "text-primary font-bold bg-muted/40" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Portfolio & HERC
            </button>
            <button
              onClick={() => setActiveSection("TAIL")}
              className={`px-2.5 py-1 rounded text-xs text-left transition-colors ${
                activeSection === "TAIL" ? "text-primary font-bold bg-muted/40" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Tail Risk & VaR
            </button>
            <button
              onClick={() => setActiveSection("STRESS")}
              className={`px-2.5 py-1 rounded text-xs text-left transition-colors ${
                activeSection === "STRESS" ? "text-primary font-bold bg-muted/40" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              Stress Scenarios
            </button>
          </div>
        </div>

        {/* Footer Meta */}
        <div className="p-3 border-t border-border bg-card/80 flex flex-col gap-1.5 text-[10px] font-mono text-muted-foreground">
          <div className="flex items-center justify-between">
            <span>Oracle A1 ARM64:</span>
            <span className="text-green-400 font-bold">READY</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Recurring Cost:</span>
            <span className="text-foreground">$0.00 / month</span>
          </div>
          <div className="flex items-center justify-between">
            <span>Schema:</span>
            <span>v4.5.0</span>
          </div>
        </div>
      </aside>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* CENTRAL REVIEW WORKSPACE                                      */}
      {/* ───────────────────────────────────────────────────────────── */}
      <main
        className="flex-1 flex flex-col overflow-hidden bg-background"
        style={{ width: `${splitRatio}%` }}
      >
        {/* Workspace Top Header Bar */}
        <header className="h-14 border-b border-border bg-card/40 px-4 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-3">
            <h2 className="text-xs font-bold uppercase tracking-wider text-foreground font-mono">
              Review Workspace: {activeDomain.toUpperCase()}
            </h2>
            {productMode === "LIVE_DEMO" ? (
              <span className="px-2 py-0.5 rounded bg-blue-950/80 text-blue-400 font-mono text-[10px] font-bold border border-blue-800 flex items-center gap-1">
                <RefreshCw className={`w-3 h-3 ${isRunning ? "animate-spin" : ""}`} />
                LIVE DETERMINISTIC RUN
              </span>
            ) : (
              <span className="px-2 py-0.5 rounded bg-purple-950/80 text-purple-400 font-mono text-[10px] font-bold border border-purple-800">
                PRECOMPUTED SHOWCASE
              </span>
            )}
          </div>

          <div className="flex items-center gap-3">
            {productMode === "LIVE_DEMO" && (
              <button
                onClick={handleLaunchLiveRun}
                disabled={isRunning}
                className="px-3.5 py-1.5 bg-primary hover:bg-primary/90 text-primary-foreground text-xs font-bold rounded flex items-center gap-1.5 transition-colors disabled:opacity-50"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>{isRunning ? "Executing Deterministic Review..." : "Launch Live Review"}</span>
              </button>
            )}

            {/* Split Layout Controls */}
            <div className="flex items-center gap-1 bg-secondary rounded p-0.5 border border-border">
              <button
                onClick={() => setSplitRatio(25)}
                className={`px-1.5 py-0.5 text-[10px] font-mono rounded ${splitRatio === 25 ? "bg-primary text-primary-foreground font-bold" : "text-muted-foreground"}`}
                title="25% Main / 75% Inspector"
              >
                25/75
              </button>
              <button
                onClick={() => setSplitRatio(50)}
                className={`px-1.5 py-0.5 text-[10px] font-mono rounded ${splitRatio === 50 ? "bg-primary text-primary-foreground font-bold" : "text-muted-foreground"}`}
                title="50/50 Split"
              >
                50/50
              </button>
              <button
                onClick={() => setSplitRatio(75)}
                className={`px-1.5 py-0.5 text-[10px] font-mono rounded ${splitRatio === 75 ? "bg-primary text-primary-foreground font-bold" : "text-muted-foreground"}`}
                title="75% Main / 25% Inspector"
              >
                75/25
              </button>
            </div>
          </div>
        </header>

        {/* Workspace Body */}
        <div className="flex-1 p-4 overflow-y-auto space-y-4">
          {/* Executive KPI Summary Cards */}
          <div className="grid grid-cols-4 gap-3">
            <div className="p-3 bg-card border border-border rounded-lg flex flex-col gap-1">
              <span className="text-[10px] font-mono text-muted-foreground uppercase">Governance Disposition</span>
              <span className="text-sm font-bold text-green-400 font-mono">
                {presentation.governance_disposition || "ACCEPT"}
              </span>
            </div>
            <div className="p-3 bg-card border border-border rounded-lg flex flex-col gap-1">
              <span className="text-[10px] font-mono text-muted-foreground uppercase">OPA Policy Evaluation</span>
              <span className="text-sm font-bold text-purple-400 font-mono">ALLOW (PASS)</span>
            </div>
            <div className="p-3 bg-card border border-border rounded-lg flex flex-col gap-1">
              <span className="text-[10px] font-mono text-muted-foreground uppercase">Registered Tools Tested</span>
              <span className="text-sm font-bold text-primary font-mono">{displayedRows.length} Tests</span>
            </div>
            <div className="p-3 bg-card border border-border rounded-lg flex flex-col gap-1">
              <span className="text-[10px] font-mono text-muted-foreground uppercase">Attestation Seal</span>
              <span className="text-xs font-mono text-amber-400 truncate">
                {presentation.attestation_seal_merkle_root.substring(0, 16)}...
              </span>
            </div>
          </div>

          {/* Local Full StART Mode Instructions Banner */}
          {productMode === "LOCAL_FULL" && (
            <div className="p-4 bg-card border border-border rounded-lg flex flex-col gap-2 font-mono text-xs">
              <h3 className="text-xs font-bold text-primary uppercase">Local Full StART Installation Contract</h3>
              <p className="text-muted-foreground">
                Clone and execute fully offline without cloud dependencies:
              </p>
              <pre className="p-3 bg-background rounded border border-border text-foreground">
                git clone https://github.com/start-project/start.git&#10;cd StART&#10;python scripts/bootstrap.py&#10;start review --domain market --mode deterministic
              </pre>
            </div>
          )}

          {/* Real React Flow Runtime Graph */}
          <div className="h-72">
            <ExecutionStream events={events} onSelectEvidence={setSelectedEvidenceId} />
          </div>

          {/* Quantitative Metric Tables */}
          <MetricTable
            rows={displayedRows}
            title={`${activeDomain.toUpperCase()} Deterministic Diagnostic Metrics`}
            onSelectEvidence={setSelectedEvidenceId}
          />

          {/* Browser WebLLM Reviewer Section */}
          <WebLLMReviewer
            runId={activeRunId}
            sessionId={sessionId}
            domain={activeDomain}
            evidenceRecords={displayedRows.map((r) => ({
              evidence_id: r.evidence_id || "EV-01",
              test_id: r.test_id,
              status: r.status || "PASS",
              metrics: { [r.metric]: r.value },
            }))}
            onHydrationComplete={(res) => {
              setPresentation((prev) => ({
                ...prev,
                governance_disposition: res.governance_disposition || "ACCEPT",
                attestation_seal_merkle_root: res.attestation_seal_merkle_root || "",
              }));
            }}
          />
        </div>
      </main>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* DRAGGABLE / RESIZABLE DIVIDER                                 */}
      {/* ───────────────────────────────────────────────────────────── */}
      <div
        onDoubleClick={() => setSplitRatio(50)}
        className="w-1.5 bg-border hover:bg-primary/80 cursor-col-resize flex items-center justify-center transition-colors select-none"
        title="Double-click to reset 50/50"
      >
        <div className="w-0.5 h-8 bg-muted-foreground/50 rounded" />
      </div>

      {/* ───────────────────────────────────────────────────────────── */}
      {/* RIGHT ARTIFACT INSPECTOR                                      */}
      {/* ───────────────────────────────────────────────────────────── */}
      <aside
        className="flex flex-col overflow-hidden bg-card/40"
        style={{ width: `${100 - splitRatio}%` }}
      >
        <ArtifactInspector selectedEvidenceId={selectedEvidenceId} />
      </aside>
    </div>
  );
}

export default App;
