import React, { useMemo } from "react";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  Node,
  Edge,
  MarkerType,
} from "reactflow";
import "reactflow/dist/style.css";
import { ReviewPresentationExport } from "../types/start_schema";

interface EvidenceDecisionGraphProps {
  presentation: ReviewPresentationExport;
  onSelectEvidence?: (evidenceId: string) => void;
  activeEvidenceId?: string;
}

export const EvidenceDecisionGraph: React.FC<EvidenceDecisionGraphProps> = ({
  presentation,
  onSelectEvidence,
  activeEvidenceId,
}) => {
  const blocks = Object.values(presentation.blocks || {});
  const allRows = blocks.flatMap((b) => b.rows || []);
  const failedOrWarnRows = allRows.filter(
    (r) => r.status === "FAIL" || r.status === "WARN" || r.status === "ERROR"
  );
  const sampleEvidence = allRows.filter((r) => r.evidence_id).slice(0, 6);

  const { nodes, edges } = useMemo(() => {
    const rawNodes: Node[] = [];
    const rawEdges: Edge[] = [];

    // Node 1: User Goal
    rawNodes.push({
      id: "node-goal",
      position: { x: 50, y: 150 },
      data: {
        label: (
          <div className="p-2 text-left">
            <div className="text-[10px] font-mono font-medium text-stone-400 uppercase">User Goal</div>
            <div className="text-xs font-semibold text-stone-900 truncate">
              {presentation.domains?.join(" + ") || "Deterministic Workflow"}
            </div>
            <div className="text-[10px] text-stone-500 font-mono">Bound Protocol</div>
          </div>
        ),
      },
      style: { border: "1px solid #4F46E5", background: "#EEF2FF", width: 160 },
    });

    // Node 2: Agent Plan
    rawNodes.push({
      id: "node-plan",
      position: { x: 260, y: 150 },
      data: {
        label: (
          <div className="p-2 text-left">
            <div className="text-[10px] font-mono font-medium text-stone-400 uppercase">Agent Plan</div>
            <div className="text-xs font-semibold text-stone-900">Deterministic Suite</div>
            <div className="text-[10px] text-stone-500 font-mono">
              {allRows.length > 0 ? `${allRows.length} Executed Surfaces` : "Awaiting execution"}
            </div>
          </div>
        ),
      },
      style: { border: "1px solid #E5E5E2", background: "#FFFFFF", width: 160 },
    });
    rawEdges.push({
      id: "e-goal-plan",
      source: "node-goal",
      target: "node-plan",
      animated: true,
      markerEnd: { type: MarkerType.ArrowClosed },
    });

    // Node 3+: Tests & Evidence Records
    sampleEvidence.forEach((ev, i) => {
      const yOffset = 40 + i * 75;
      const nodeId = `node-ev-${i}`;
      const isSelected = activeEvidenceId === ev.evidence_id;

      rawNodes.push({
        id: nodeId,
        position: { x: 470, y: yOffset },
        data: {
          label: (
            <div className="p-2 text-left">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-mono text-indigo-600 font-medium truncate max-w-[90px]">
                  {ev.evidence_id}
                </span>
                <span
                  className={`text-[9px] px-1 rounded font-mono shrink-0 ${
                    ev.status === "PASS"
                      ? "bg-emerald-50 text-emerald-700"
                      : "bg-amber-50 text-amber-700"
                  }`}
                >
                  {ev.status}
                </span>
              </div>
              <div className="text-xs font-medium text-stone-800 truncate mt-0.5">
                {ev.metric || ev.test_id}
              </div>
              <div className="text-[10px] text-stone-500 font-mono truncate">
                {ev.value !== undefined ? String(ev.value).substring(0, 14) : "Recorded"} {ev.unit || ""}
              </div>
            </div>
          ),
        },
        style: {
          border: isSelected ? "2px solid #4F46E5" : "1px solid #E5E5E2",
          background: isSelected ? "#EEF2FF" : "#FFFFFF",
          width: 170,
          cursor: "pointer",
        },
      });

      rawEdges.push({
        id: `e-plan-ev-${i}`,
        source: "node-plan",
        target: nodeId,
        markerEnd: { type: MarkerType.ArrowClosed },
      });
    });

    // Node: Findings (rendered ONLY if real failed or warning evidence rows exist)
    const hasFindings = failedOrWarnRows.length > 0;
    if (hasFindings) {
      const topFinding = failedOrWarnRows[0];
      const findingTitle = topFinding.metric || topFinding.test_id || "Attention Item";

      rawNodes.push({
        id: "node-findings",
        position: { x: 700, y: 120 },
        data: {
          label: (
            <div className="p-2 text-left">
              <div className="text-[10px] font-mono font-medium text-amber-600 uppercase">
                Deterministic Finding
              </div>
              <div className="text-xs font-semibold text-stone-900 truncate">{findingTitle}</div>
              <div className="text-[10px] text-stone-500 font-mono">
                {failedOrWarnRows.length} flagged surface{failedOrWarnRows.length > 1 ? "s" : ""}
              </div>
            </div>
          ),
        },
        style: { border: "1px solid #F59E0B", background: "#FFFBEB", width: 170 },
      });

      sampleEvidence.forEach((_, i) => {
        rawEdges.push({
          id: `e-ev-findings-${i}`,
          source: `node-ev-${i}`,
          target: "node-findings",
          markerEnd: { type: MarkerType.ArrowClosed },
        });
      });
    }

    // Node: Attestation / Governance Seal (rendered ONLY if disposition or merkle root exists)
    const hasAttestation =
      Boolean(presentation.governance_disposition) ||
      Boolean(presentation.attestation_seal_merkle_root);

    if (hasAttestation) {
      const merklePreview = presentation.attestation_seal_merkle_root
        ? `${presentation.attestation_seal_merkle_root.substring(0, 12)}...`
        : "Evidence Sealed";

      rawNodes.push({
        id: "node-seal",
        position: { x: hasFindings ? 930 : 700, y: 150 },
        data: {
          label: (
            <div className="p-2 text-left">
              <div className="text-[10px] font-mono font-medium text-emerald-600 uppercase">
                Attestation Seal
              </div>
              <div className="text-xs font-semibold text-emerald-700 truncate">
                {presentation.governance_disposition || "VERIFIED"}
              </div>
              <div className="text-[10px] text-stone-500 font-mono truncate">
                {merklePreview}
              </div>
            </div>
          ),
        },
        style: { border: "1px solid #10B981", background: "#ECFDF5", width: 170 },
      });

      if (hasFindings) {
        rawEdges.push({
          id: "e-findings-seal",
          source: "node-findings",
          target: "node-seal",
          animated: true,
          markerEnd: { type: MarkerType.ArrowClosed },
        });
      } else {
        sampleEvidence.forEach((_, i) => {
          rawEdges.push({
            id: `e-ev-seal-${i}`,
            source: `node-ev-${i}`,
            target: "node-seal",
            animated: true,
            markerEnd: { type: MarkerType.ArrowClosed },
          });
        });
      }
    }

    return { nodes: rawNodes, edges: rawEdges };
  }, [presentation, sampleEvidence, failedOrWarnRows, allRows.length, activeEvidenceId]);

  return (
    <div className="w-full h-full min-h-[350px] bg-[#FBFBFA] relative border border-[#E5E5E2] rounded-xl overflow-hidden">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodeClick={(_, node) => {
          if (node.id.startsWith("node-ev-") && onSelectEvidence) {
            const idx = parseInt(node.id.replace("node-ev-", ""));
            const ev = sampleEvidence[idx];
            if (ev?.evidence_id) onSelectEvidence(ev.evidence_id);
          }
        }}
        fitView
      >
        <Background color="#E5E5E2" gap={16} size={1} />
        <Controls className="bg-white border border-[#E5E5E2] shadow-xs" />
        <MiniMap
          nodeColor={(n) => (n.id === "node-seal" ? "#10B981" : "#4F46E5")}
          className="bg-white border border-[#E5E5E2] rounded-md"
        />
      </ReactFlow>
    </div>
  );
};
