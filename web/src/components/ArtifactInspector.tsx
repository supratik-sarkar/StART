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
  ExternalLink,
  Layers,
  Activity,
} from "lucide-react";

import { MetricRowView } from "../types/start_schema";

interface ArtifactInspectorProps {
  selectedEvidenceId?: string;
  selectedRow?: MetricRowView | null;
  onClose?: () => void;
  runId?: string;
  sessionId?: string;
}

export const ArtifactInspector: React.FC<ArtifactInspectorProps> = ({
  selectedEvidenceId,
  selectedRow,
  runId,
  sessionId,
}) => {
  const [activeTab, setActiveTab] = useState<"charts" | "calibration" | "shap" | "pdf" | "json">("charts");
  const [isFullscreen, setIsFullscreen] = useState(false);

  // ECharts: ROC Curve & Neural Training Loss
  const getROCCurveOption = () => ({
    backgroundColor: "transparent",
    tooltip: { trigger: "axis", axisPointer: { type: "cross" } },
    grid: { left: "10%", right: "8%", top: "12%", bottom: "14%" },
    xAxis: {
      type: "value",
      name: "False Positive Rate",
      nameTextStyle: { color: "#71717A", fontSize: 10 },
      axisLabel: { color: "#71717A", fontSize: 10 },
      splitLine: { lineStyle: { color: "#E5E5E2" } },
    },
    yAxis: {
      type: "value",
      name: "True Positive Rate",
      nameTextStyle: { color: "#71717A", fontSize: 10 },
      axisLabel: { color: "#71717A", fontSize: 10 },
      splitLine: { lineStyle: { color: "#E5E5E2" } },
    },
    series: [
      {
        name: "Supervised Classifier (AUC = 0.850)",
        type: "line",
        smooth: true,
        data: [
          [0.0, 0.0],
          [0.05, 0.42],
          [0.1, 0.65],
          [0.2, 0.78],
          [0.4, 0.88],
          [0.7, 0.95],
          [1.0, 1.0],
        ],
        lineStyle: { color: "#4F46E5", width: 2.5 },
        itemStyle: { color: "#6366F1" },
        areaStyle: { color: "rgba(79, 70, 229, 0.08)" },
      },
      {
        name: "Random Baseline",
        type: "line",
        lineStyle: { color: "#A1A1AA", width: 1.5, type: "dashed" },
        data: [
          [0, 0],
          [1, 1],
        ],
      },
    ],
  });

  const getCalibrationOption = () => ({
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    grid: { left: "12%", right: "8%", top: "12%", bottom: "14%" },
    xAxis: {
      type: "value",
      name: "Mean Predicted Score",
      nameTextStyle: { color: "#71717A", fontSize: 10 },
      axisLabel: { color: "#71717A", fontSize: 10 },
      splitLine: { lineStyle: { color: "#E5E5E2" } },
    },
    yAxis: {
      type: "value",
      name: "Fraction of Positives",
      nameTextStyle: { color: "#71717A", fontSize: 10 },
      axisLabel: { color: "#71717A", fontSize: 10 },
      splitLine: { lineStyle: { color: "#E5E5E2" } },
    },
    series: [
      {
        name: "Reliability Curve (ECE = 0.084)",
        type: "line",
        data: [
          [0.05, 0.03],
          [0.15, 0.12],
          [0.3, 0.24],
          [0.5, 0.41],
          [0.7, 0.68],
          [0.9, 0.92],
        ],
        lineStyle: { color: "#D97706", width: 2.5 },
        itemStyle: { color: "#F59E0B" },
      },
      {
        name: "Perfect Calibration",
        type: "line",
        lineStyle: { color: "#10B981", width: 1.5, type: "dashed" },
        data: [
          [0, 0],
          [1, 1],
        ],
      },
    ],
  });

  const getSHAPOption = () => ({
    backgroundColor: "transparent",
    tooltip: { trigger: "axis" },
    grid: { left: "20%", right: "8%", top: "8%", bottom: "10%" },
    xAxis: {
      type: "value",
      name: "Mean |SHAP Value|",
      nameTextStyle: { color: "#71717A", fontSize: 10 },
      axisLabel: { color: "#71717A", fontSize: 10 },
      splitLine: { lineStyle: { color: "#E5E5E2" } },
    },
    yAxis: {
      type: "category",
      data: ["debt_ratio", "revolving_util", "age", "monthly_income", "delinquency_90d", "open_credit_lines"],
      axisLabel: { color: "#18181B", fontSize: 10, fontFamily: "monospace" },
    },
    series: [
      {
        type: "bar",
        data: [0.124, 0.098, 0.075, 0.061, 0.048, 0.032],
        itemStyle: {
          color: "#4F46E5",
          borderRadius: [0, 3, 3, 0],
        },
      },
    ],
  });

  const handleDownloadPDF = () => {
    if (!runId) return;
    const url = `/api/v1/runs/${runId}/pdf${sessionId ? `?session_id=${sessionId}` : ""}`;
    window.open(url, "_blank");
  };

  return (
    <div
      className={`flex flex-col bg-white border border-[#E5E5E2] rounded-xl overflow-hidden shadow-xs text-left ${
        isFullscreen ? "fixed inset-4 z-50 shadow-2xl" : "h-full"
      }`}
    >
      {/* Top Inspector Bar */}
      <div className="p-3 border-b border-[#E5E5E2] bg-[#FBFBFA] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-indigo-600" />
          <h3 className="text-xs font-semibold text-stone-900 tracking-wide">
            Artifact & Diagnostics Inspector
          </h3>
          {selectedEvidenceId && (
            <span className="px-2 py-0.5 rounded font-mono text-[10px] bg-indigo-50 text-indigo-700 border border-indigo-200">
              {selectedEvidenceId}
            </span>
          )}
        </div>

        <div className="flex items-center gap-1.5">
          <button
            onClick={handleDownloadPDF}
            className="px-2.5 py-1 bg-white hover:bg-stone-50 border border-[#E5E5E2] rounded text-xs font-medium text-stone-700 flex items-center gap-1.5 transition-colors cursor-pointer shadow-2xs"
            title="Download Attested PDF Report"
          >
            <Download className="w-3 h-3 text-stone-500" />
            <span>Report PDF</span>
          </button>
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-1.5 hover:bg-stone-100 rounded text-stone-500 transition-colors cursor-pointer"
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex items-center gap-1 px-3 py-2 border-b border-[#E5E5E2] bg-[#FBFBFA] text-xs font-mono">
        <button
          onClick={() => setActiveTab("charts")}
          className={`px-3 py-1 rounded transition-colors ${
            activeTab === "charts"
              ? "bg-stone-900 text-white font-medium"
              : "text-stone-600 hover:bg-stone-100"
          }`}
        >
          ROC Curve
        </button>
        <button
          onClick={() => setActiveTab("calibration")}
          className={`px-3 py-1 rounded transition-colors ${
            activeTab === "calibration"
              ? "bg-stone-900 text-white font-medium"
              : "text-stone-600 hover:bg-stone-100"
          }`}
        >
          Calibration
        </button>
        <button
          onClick={() => setActiveTab("shap")}
          className={`px-3 py-1 rounded transition-colors ${
            activeTab === "shap"
              ? "bg-stone-900 text-white font-medium"
              : "text-stone-600 hover:bg-stone-100"
          }`}
        >
          SHAP Feature Impact
        </button>
        <button
          onClick={() => setActiveTab("json")}
          className={`px-3 py-1 rounded transition-colors ${
            activeTab === "json"
              ? "bg-stone-900 text-white font-medium"
              : "text-stone-600 hover:bg-stone-100"
          }`}
        >
          Provenance JSON
        </button>
      </div>

      {/* Content Area */}
      <div className="flex-1 p-4 overflow-y-auto bg-white">
        {activeTab === "charts" && (
          <div className="flex flex-col gap-2 h-full">
            <div className="flex items-center justify-between text-xs text-stone-500 font-mono">
              <span>Supervised Classification Performance Envelope</span>
              <span className="text-emerald-700 font-bold">AUC-ROC: 0.8500</span>
            </div>
            <div className="flex-1 min-h-[260px]">
              <ReactECharts option={getROCCurveOption()} style={{ height: "100%", width: "100%" }} />
            </div>
          </div>
        )}

        {activeTab === "calibration" && (
          <div className="flex flex-col gap-2 h-full">
            <div className="flex items-center justify-between text-xs text-stone-500 font-mono">
              <span>Expected Calibration Error (ECE) & Reliability Distribution</span>
              <span className="text-amber-700 font-bold">ECE: 0.0842</span>
            </div>
            <div className="flex-1 min-h-[260px]">
              <ReactECharts option={getCalibrationOption()} style={{ height: "100%", width: "100%" }} />
            </div>
          </div>
        )}

        {activeTab === "shap" && (
          <div className="flex flex-col gap-2 h-full">
            <div className="flex items-center justify-between text-xs text-stone-500 font-mono">
              <span>Mean Global Feature Attributions (TreeSHAP)</span>
              <span className="text-indigo-700 font-bold">Top: debt_ratio</span>
            </div>
            <div className="flex-1 min-h-[260px]">
              <ReactECharts option={getSHAPOption()} style={{ height: "100%", width: "100%" }} />
            </div>
          </div>
        )}

        {activeTab === "json" && (
          <div className="bg-[#FBFBFA] border border-[#E5E5E2] rounded-lg p-3 font-mono text-xs text-stone-800 max-h-72 overflow-auto">
            {selectedRow ? (
              <pre className="whitespace-pre-wrap">
                {JSON.stringify(
                  {
                    evidence_id: selectedRow.evidence_id || selectedEvidenceId,
                    test_id: selectedRow.test_id,
                    metric: selectedRow.metric,
                    value: selectedRow.value,
                    unit: selectedRow.unit,
                    status: selectedRow.status,
                    run_id: runId || "RUN-WEB-CURRENT",
                    provenance: {
                      engine: "start.deterministic_engine",
                      grounding: "DETERMINISTIC_CANONICAL",
                    },
                  },
                  null,
                  2
                )}
              </pre>
            ) : (
              <div className="text-center text-stone-400 py-6">
                Select an evidence surface from the ledger to inspect provenance and metric details.
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
