import React, { useState } from "react";
import ReactECharts from "echarts-for-react";
import {
  FileText,
  BarChart2,
  Image,
  Code,
  Download,
  Maximize2,
  Minimize2,
  ChevronLeft,
  ChevronRight,
  ExternalLink,
} from "lucide-react";

interface ArtifactInspectorProps {
  selectedEvidenceId?: string;
  onClose?: () => void;
}

export const ArtifactInspector: React.FC<ArtifactInspectorProps> = ({ selectedEvidenceId }) => {
  const [activeTab, setActiveTab] = useState<"chart" | "svg" | "pdf" | "html" | "json">("chart");
  const [isFullscreen, setIsFullscreen] = useState(false);

  // ECharts: Efficient Frontier & Factor Risk Contribution
  const getFrontierOption = () => ({
    backgroundColor: "transparent",
    tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
    grid: { left: "10%", right: "8%", top: "12%", bottom: "14%" },
    xAxis: {
      type: "value",
      name: "Volatility (σ)",
      nameTextStyle: { color: "#94a3b8", fontSize: 10 },
      axisLabel: { color: "#94a3b8", fontSize: 10 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    yAxis: {
      type: "value",
      name: "Expected Return (μ)",
      nameTextStyle: { color: "#94a3b8", fontSize: 10 },
      axisLabel: { color: "#94a3b8", fontSize: 10 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: [
      {
        name: "Efficient Frontier (HERC/HRP)",
        type: "line",
        smooth: true,
        data: [
          [0.08, 0.04],
          [0.09, 0.065],
          [0.11, 0.09],
          [0.14, 0.12],
          [0.18, 0.155],
          [0.23, 0.185],
        ],
        lineStyle: { color: "#3b82f6", width: 2.5 },
        itemStyle: { color: "#60a5fa" },
      },
      {
        name: "Active HERC Portfolio",
        type: "scatter",
        symbolSize: 12,
        data: [[0.12, 0.105]],
        itemStyle: { color: "#10b981" },
      },
    ],
  });

  const getRiskContributionOption = () => ({
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    grid: { left: "12%", right: "8%", top: "10%", bottom: "25%" },
    xAxis: {
      type: "category",
      data: ["Equity Beta", "Momentum", "Size", "Value", "Volatility", "Idiosyncratic"],
      axisLabel: { color: "#94a3b8", fontSize: 9, rotate: 25 },
    },
    yAxis: {
      type: "value",
      name: "Risk Share (%)",
      nameTextStyle: { color: "#94a3b8", fontSize: 10 },
      axisLabel: { color: "#94a3b8", fontSize: 10 },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: [
      {
        type: "bar",
        data: [38.5, 14.2, 8.4, 6.1, 18.3, 14.5],
        itemStyle: {
          color: (params: any) => {
            const colors = ["#3b82f6", "#8b5cf6", "#ec4899", "#f59e0b", "#10b981", "#64748b"];
            return colors[params.dataIndex % colors.length];
          },
          borderRadius: [3, 3, 0, 0],
        },
      },
    ],
  });

  return (
    <div
      className={`flex flex-col h-full bg-card border border-border rounded-lg overflow-hidden transition-all ${
        isFullscreen ? "fixed inset-4 z-50 shadow-2xl" : ""
      }`}
    >
      {/* Inspector Header */}
      <div className="p-3 border-b border-border bg-card/80 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-primary" />
          <h3 className="text-xs font-semibold text-foreground uppercase tracking-wider">
            Artifact Inspector
          </h3>
          {selectedEvidenceId && (
            <span className="px-1.5 py-0.5 rounded bg-blue-950/60 text-blue-400 text-[10px] font-mono border border-blue-900/50">
              [{selectedEvidenceId}]
            </span>
          )}
        </div>
        <div className="flex items-center gap-1">
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1 text-muted-foreground hover:text-foreground rounded hover:bg-muted"
            title="Toggle fullscreen"
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-1 border-b border-border px-3 py-1.5 bg-muted/20 font-mono text-xs">
        <button
          onClick={() => setActiveTab("chart")}
          className={`flex items-center gap-1 px-2.5 py-1 rounded transition-colors ${
            activeTab === "chart" ? "bg-secondary text-primary font-bold" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <BarChart2 className="w-3 h-3" />
          <span>Interactive Plots</span>
        </button>
        <button
          onClick={() => setActiveTab("svg")}
          className={`flex items-center gap-1 px-2.5 py-1 rounded transition-colors ${
            activeTab === "svg" ? "bg-secondary text-primary font-bold" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Image className="w-3 h-3" />
          <span>High-Res SVG</span>
        </button>
        <button
          onClick={() => setActiveTab("pdf")}
          className={`flex items-center gap-1 px-2.5 py-1 rounded transition-colors ${
            activeTab === "pdf" ? "bg-secondary text-primary font-bold" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <FileText className="w-3 h-3" />
          <span>PDF Report</span>
        </button>
        <button
          onClick={() => setActiveTab("html")}
          className={`flex items-center gap-1 px-2.5 py-1 rounded transition-colors ${
            activeTab === "html" ? "bg-secondary text-primary font-bold" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <ExternalLink className="w-3 h-3" />
          <span>Sandboxed HTML</span>
        </button>
        <button
          onClick={() => setActiveTab("json")}
          className={`flex items-center gap-1 px-2.5 py-1 rounded transition-colors ${
            activeTab === "json" ? "bg-secondary text-primary font-bold" : "text-muted-foreground hover:text-foreground"
          }`}
        >
          <Code className="w-3 h-3" />
          <span>Raw JSON</span>
        </button>
      </div>

      {/* Content Viewport */}
      <div className="flex-1 p-3 overflow-y-auto bg-background/50 flex flex-col gap-4">
        {activeTab === "chart" && (
          <div className="flex flex-col gap-4 h-full">
            <div className="bg-card border border-border rounded-lg p-2.5 flex flex-col">
              <span className="text-xs font-semibold text-foreground px-2 pt-1 font-mono">
                Efficient Frontier & Optimal Asset Allocation
              </span>
              <ReactECharts option={getFrontierOption()} style={{ height: "180px", width: "100%" }} />
            </div>
            <div className="bg-card border border-border rounded-lg p-2.5 flex flex-col">
              <span className="text-xs font-semibold text-foreground px-2 pt-1 font-mono">
                Factor Risk Contribution Breakdown (%)
              </span>
              <ReactECharts option={getRiskContributionOption()} style={{ height: "180px", width: "100%" }} />
            </div>
          </div>
        )}

        {activeTab === "svg" && (
          <div className="flex flex-col items-center justify-center h-full border border-dashed border-border/80 rounded-lg p-4 bg-card/40">
            <svg viewBox="0 0 400 200" className="w-full max-h-56">
              <rect width="400" height="200" fill="#0f172a" rx="8" />
              <path d="M 50 150 Q 150 50 250 120 T 350 40" fill="none" stroke="#3b82f6" strokeWidth="3" />
              <circle cx="50" cy="150" r="4" fill="#60a5fa" />
              <circle cx="150" cy="80" r="4" fill="#60a5fa" />
              <circle cx="250" cy="120" r="4" fill="#60a5fa" />
              <circle cx="350" cy="40" r="4" fill="#10b981" />
              <text x="20" y="30" fill="#94a3b8" fontSize="11" fontFamily="monospace">
                Hierarchical Risk Parity (HRP) Dendrogram Tree
              </text>
            </svg>
            <span className="text-[11px] font-mono text-muted-foreground mt-3">
              Cryptographically verified SVG artifact generated from deterministic tree clustering
            </span>
          </div>
        )}

        {activeTab === "pdf" && (
          <div className="flex flex-col items-center justify-center h-full border border-border rounded-lg p-6 bg-card/40 text-center gap-3">
            <FileText className="w-12 h-12 text-primary animate-pulse" />
            <div className="flex flex-col gap-1">
              <h4 className="text-xs font-bold text-foreground">Institutional Validation Report (PDF)</h4>
              <p className="text-[11px] text-muted-foreground max-w-sm">
                Deterministic PDF document containing Executive Summary, Technical Validation, Evidence Appendix, and Merkle Attestation Seal.
              </p>
            </div>
            <a
              href="/api/v1/runs/RUN-DEMO-01/pdf"
              download
              className="px-4 py-2 bg-primary hover:bg-primary/90 text-primary-foreground text-xs font-semibold rounded flex items-center gap-1.5 transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download Signed Review PDF</span>
            </a>
          </div>
        )}

        {activeTab === "html" && (
          <div className="h-full border border-border rounded-lg overflow-hidden bg-white">
            <iframe
              title="Sandboxed HTML Report"
              sandbox="allow-scripts"
              srcDoc="<html><body style='font-family:sans-serif;padding:16px;background:#f8fafc;color:#0f172a'><h3>StART Standalone Diagnostic Report</h3><p style='font-size:12px;color:#64748b'>Rendered in strict sandbox iframe without parent-origin privileges.</p></body></html>"
              className="w-full h-full border-none"
            />
          </div>
        )}

        {activeTab === "json" && (
          <div className="bg-background border border-border rounded-lg p-3 font-mono text-xs text-foreground/80 overflow-x-auto">
            <pre>
              {JSON.stringify(
                {
                  artifact_id: "ART-MKT-001",
                  title: "Hierarchical Risk Parity Dendrogram",
                  evidence_id: selectedEvidenceId || "EV-MKT-001",
                  generator: "start.portfolio.hrp",
                  sha256: "8e9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f",
                  status: "VERIFIED_DETERMINISTIC",
                },
                null,
                2
              )}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
};
