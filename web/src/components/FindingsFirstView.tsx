import React from "react";
import {
  AlertTriangle,
  CheckCircle2,
  HelpCircle,
  ArrowRight,
  Sparkles,
  ExternalLink,
  ShieldCheck,
  RefreshCw,
  Scale,
  Activity,
  Layers,
  FileText,
} from "lucide-react";
import { ReviewPresentationExport } from "../types/start_schema";

interface FindingsFirstViewProps {
  presentation: ReviewPresentationExport;
  onSelectEvidence: (evidenceId: string) => void;
  onTriggerIterateAction: (action: string, context: Record<string, any>) => void;
}

export const FindingsFirstView: React.FC<FindingsFirstViewProps> = ({
  presentation,
  onSelectEvidence,
  onTriggerIterateAction,
}) => {
  const blocks = Object.values(presentation.blocks || {});
  const allRows = blocks.flatMap((b) => b.rows || []);

  const failedOrWarnRows = allRows.filter(
    (r) => r.status === "FAIL" || r.status === "WARN" || r.status === "ERROR"
  );
  const passedRows = allRows.filter((r) => r.status === "PASS");

  // Derive attention items strictly from actual failed/warning deterministic evidence rows
  const attentionItems = failedOrWarnRows.map((row, idx) => {
    const isFail = row.status === "FAIL" || row.status === "ERROR";
    const category = row.test_id ? row.test_id.split("-")[0] : "Diagnostics";
    const title = row.metric || row.test_id || `Evidence Flag ${idx + 1}`;
    const metricText = `Value: ${row.value !== undefined ? String(row.value) : "N/A"} ${row.unit || ""}`.trim();
    const summary = `${row.status || "WARN"} status recorded on deterministic test ${row.test_id || title}.`;
    const recommendation = `Review deterministic surface ${row.test_id || title} output and verify baseline stability.`;

    return {
      title,
      category,
      severity: isFail ? "FAIL" : "WARN",
      summary,
      evidenceId: row.evidence_id || `EV-ATTN-${idx + 1}`,
      recommendation,
      metricText,
    };
  });

  return (
    <div className="flex flex-col gap-6 p-6 bg-white overflow-y-auto">
      {/* Top Banner: Governance Disposition & Merkle Attestation */}
      <div className="flex items-center justify-between p-4 rounded-xl border border-[#E5E5E2] bg-[#FBFBFA]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-700">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div className="flex flex-col text-left">
            <span className="font-semibold text-xs text-stone-900">
              Deterministic Governance Attestation:{" "}
              <span className="text-emerald-700 font-mono">
                {presentation.governance_disposition || "Awaiting evaluation"}
              </span>
            </span>
            <span className="text-[11px] font-mono text-stone-500 truncate max-w-md">
              Merkle Root: {presentation.attestation_seal_merkle_root || "Awaiting attestation seal"}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-stone-600 bg-white px-2.5 py-1 rounded border border-[#E5E5E2]">
            {passedRows.length} Passed / {failedOrWarnRows.length} Flagged
          </span>
        </div>
      </div>

      {/* Primary Section: Key Findings Deserving Attention */}
      <div className="flex flex-col gap-3 text-left">
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold text-stone-900">
            Key Findings Deserving Attention
          </h2>
          <span className="text-xs text-stone-500 font-mono">
            {attentionItems.length} findings identified
          </span>
        </div>

        <div className="flex flex-col gap-3">
          {attentionItems.length === 0 ? (
            <div className="p-6 rounded-xl border border-[#E5E5E2] bg-[#FBFBFA] text-center flex flex-col items-center justify-center gap-2">
              <CheckCircle2 className="w-5 h-5 text-emerald-600" />
              <span className="text-xs font-medium text-stone-700">
                No evidence-derived attention items were identified for this run.
              </span>
            </div>
          ) : (
            attentionItems.map((item, idx) => (
            <div
              key={idx}
              className="p-4 rounded-xl border border-[#E5E5E2] bg-white hover:border-stone-300 transition-all flex flex-col gap-3 shadow-xs"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex items-center gap-2.5">
                  <span className="w-2 h-2 rounded-full bg-amber-500" />
                  <span className="font-semibold text-sm text-stone-900">{item.title}</span>
                  <span className="text-[10px] font-mono font-medium px-1.5 py-0.5 rounded bg-amber-50 text-amber-700 border border-amber-200">
                    {item.category}
                  </span>
                </div>

                <button
                  onClick={() => onSelectEvidence(item.evidenceId)}
                  className="inline-flex items-center gap-1 text-xs font-mono text-indigo-600 hover:text-indigo-800 bg-indigo-50 px-2 py-0.5 rounded border border-indigo-200 cursor-pointer"
                >
                  <span>{item.evidenceId}</span>
                  <ExternalLink className="w-3 h-3" />
                </button>
              </div>

              <p className="text-xs text-stone-600 leading-relaxed">{item.summary}</p>

              <div className="flex items-center justify-between text-xs text-stone-500 font-mono pt-2 border-t border-stone-100">
                <span>{item.metricText}</span>
                <span className="text-stone-400">Rec: {item.recommendation}</span>
              </div>

              {/* Contextual Agentic Actions for this Finding */}
              <div className="flex flex-wrap items-center gap-2 pt-1">
                <button
                  onClick={() =>
                    onTriggerIterateAction("EXPLAIN", {
                      finding: item.title,
                      evidenceId: item.evidenceId,
                    })
                  }
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium bg-stone-100 hover:bg-stone-200 text-stone-800 border border-stone-200 cursor-pointer transition-colors"
                >
                  <Sparkles className="w-3 h-3 text-indigo-600" />
                  <span>Explain with AI</span>
                </button>

                <button
                  onClick={() =>
                    onTriggerIterateAction("CHALLENGE", {
                      finding: item.title,
                      evidenceId: item.evidenceId,
                    })
                  }
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium bg-stone-100 hover:bg-stone-200 text-stone-800 border border-stone-200 cursor-pointer transition-colors"
                >
                  <HelpCircle className="w-3 h-3 text-amber-600" />
                  <span>Challenge Finding</span>
                </button>

                <button
                  onClick={() =>
                    onTriggerIterateAction("DEEPER_TEST", {
                      finding: item.title,
                      evidenceId: item.evidenceId,
                    })
                  }
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium bg-stone-100 hover:bg-stone-200 text-stone-800 border border-stone-200 cursor-pointer transition-colors"
                >
                  <Activity className="w-3 h-3 text-emerald-600" />
                  <span>Run Deeper Test</span>
                </button>

                <button
                  onClick={() =>
                    onTriggerIterateAction("COMPARE", {
                      finding: item.title,
                      evidenceId: item.evidenceId,
                    })
                  }
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium bg-stone-100 hover:bg-stone-200 text-stone-800 border border-stone-200 cursor-pointer transition-colors"
                >
                  <Scale className="w-3 h-3 text-indigo-600" />
                  <span>Compare Candidates</span>
                </button>

                <button
                  onClick={() =>
                    onTriggerIterateAction("RE_RUN", {
                      finding: item.title,
                      evidenceId: item.evidenceId,
                    })
                  }
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium bg-indigo-50 hover:bg-indigo-100 text-indigo-700 border border-indigo-200 cursor-pointer transition-colors ml-auto"
                >
                  <RefreshCw className="w-3 h-3 text-indigo-600" />
                  <span>Re-run from here</span>
                </button>
              </div>
            </div>
          )))}
        </div>
      </div>

      {/* Verified Surfaces List */}
      <div className="flex flex-col gap-3 text-left">
        <h3 className="text-sm font-semibold text-stone-900">
          Verified Deterministic Surfaces
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
          {passedRows.slice(0, 8).map((row, idx) => (
            <div
              key={idx}
              onClick={() => row.evidence_id && onSelectEvidence(row.evidence_id)}
              className="p-2.5 rounded-lg border border-[#E5E5E2] bg-stone-50/50 hover:bg-stone-50 flex items-center justify-between text-xs cursor-pointer transition-colors"
            >
              <div className="flex items-center gap-2 truncate">
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                <span className="text-stone-800 font-medium truncate">{row.metric || row.test_id}</span>
              </div>
              <span className="font-mono text-stone-500 text-[11px] shrink-0 ml-2">
                {String(row.value).substring(0, 10)} {row.unit}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
