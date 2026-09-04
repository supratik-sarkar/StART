import React, { useMemo } from "react";
import ReactFlow, { Background, Controls, Node, Edge, MarkerType, Position } from "reactflow";
import "reactflow/dist/style.css";
import { SSEEnvelope } from "../types/start_schema";
import { Activity, ShieldCheck, Cpu, Database, CheckCircle, Clock } from "lucide-react";

interface ExecutionStreamProps {
  events: SSEEnvelope[];
  onSelectEvidence?: (evId: string) => void;
}

export const ExecutionStream: React.FC<ExecutionStreamProps> = ({ events, onSelectEvidence }) => {
  const { nodes, edges } = useMemo(() => {
    const calculatedNodes: Node[] = [];
    const calculatedEdges: Edge[] = [];

    // Core Architecture Nodes
    calculatedNodes.push({
      id: "Director",
      position: { x: 50, y: 140 },
      data: { label: "Director Orchestrator", role: "director", status: "SUCCESS" },
      style: { background: "#FFFFFF", color: "#18181B", border: "1px solid #4F46E5", width: 170, fontSize: 11, padding: 8, borderRadius: 8, boxShadow: "0 1px 3px rgba(0,0,0,0.04)" },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    });

    calculatedNodes.push({
      id: "Specialist",
      position: { x: 280, y: 80 },
      data: { label: "Domain Specialist", role: "specialist", status: "SUCCESS" },
      style: { background: "#FFFFFF", color: "#18181B", border: "1px solid #E5E5E2", width: 160, fontSize: 11, padding: 8, borderRadius: 8, boxShadow: "0 1px 3px rgba(0,0,0,0.04)" },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    });

    calculatedNodes.push({
      id: "DeterministicEngine",
      position: { x: 500, y: 80 },
      data: { label: "Deterministic Engine (79 Tools)", role: "engine", status: "SUCCESS" },
      style: { background: "#FFFFFF", color: "#18181B", border: "1px solid #10B981", width: 190, fontSize: 11, padding: 8, borderRadius: 8, boxShadow: "0 1px 3px rgba(0,0,0,0.04)" },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    });

    calculatedNodes.push({
      id: "PolicyEngine",
      position: { x: 280, y: 200 },
      data: { label: "OPA Policy Plane", role: "policy", status: "SUCCESS" },
      style: { background: "#FFFFFF", color: "#18181B", border: "1px solid #6366F1", width: 160, fontSize: 11, padding: 8, borderRadius: 8, boxShadow: "0 1px 3px rgba(0,0,0,0.04)" },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    });

    calculatedNodes.push({
      id: "AttestationLedger",
      position: { x: 740, y: 140 },
      data: { label: "Merkle Attestation Ledger", role: "attestation", status: "SUCCESS" },
      style: { background: "#FFFFFF", color: "#18181B", border: "1px solid #F59E0B", width: 180, fontSize: 11, padding: 8, borderRadius: 8, boxShadow: "0 1px 3px rgba(0,0,0,0.04)" },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    });

    // Build Static Topology Edges
    calculatedEdges.push({
      id: "e-dir-spec",
      source: "Director",
      target: "Specialist",
      animated: true,
      style: { stroke: "#4F46E5" },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#4F46E5" },
    });
    calculatedEdges.push({
      id: "e-spec-eng",
      source: "Specialist",
      target: "DeterministicEngine",
      animated: true,
      style: { stroke: "#10B981" },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#10B981" },
    });
    calculatedEdges.push({
      id: "e-dir-pol",
      source: "Director",
      target: "PolicyEngine",
      animated: true,
      style: { stroke: "#6366F1" },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#6366F1" },
    });
    calculatedEdges.push({
      id: "e-eng-att",
      source: "DeterministicEngine",
      target: "AttestationLedger",
      animated: true,
      style: { stroke: "#F59E0B" },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#F59E0B" },
    });

    return { nodes: calculatedNodes, edges: calculatedEdges };
  }, [events]);

  return (
    <div className="flex flex-col h-full bg-white border border-[#E5E5E2] rounded-xl overflow-hidden text-left shadow-xs">
      {/* Top Banner */}
      <div className="p-3 border-b border-[#E5E5E2] bg-[#FBFBFA] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-indigo-600 animate-pulse" />
          <h3 className="text-xs font-semibold text-stone-900 tracking-wide">
            Live Execution Graph (Canonical Dispatch)
          </h3>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono text-stone-500">
          <span className="flex items-center gap-1">
            <Cpu className="w-3 h-3 text-emerald-600" />
            79 Deterministic Engines
          </span>
          <span className="flex items-center gap-1">
            <ShieldCheck className="w-3 h-3 text-indigo-600" />
            OPA Active
          </span>
          <span className="flex items-center gap-1">
            <Database className="w-3 h-3 text-amber-600" />
            Merkle Sealed
          </span>
        </div>
      </div>

      {/* React Flow Canvas */}
      <div className="flex-1 w-full min-h-[200px] bg-[#FBFBFA] relative">
        <ReactFlow nodes={nodes} edges={edges} fitView>
          <Background color="#E5E5E2" gap={16} size={1} />
          <Controls position="bottom-right" className="bg-white border border-[#E5E5E2] shadow-xs text-stone-700" />
        </ReactFlow>
      </div>

      {/* Recent Event Log Strip */}
      <div className="border-t border-[#E5E5E2] bg-white p-3 max-h-48 overflow-y-auto font-mono text-xs divide-y divide-stone-100">
        {events.length === 0 ? (
          <div className="text-stone-400 text-[11px] text-center py-2">
            Awaiting analytical execution events...
          </div>
        ) : (
          events.slice(-8).reverse().map((evt, idx) => (
            <div key={idx} className="py-1.5 flex items-center justify-between text-[11px]">
              <div className="flex items-center gap-2 truncate">
                <span className="text-indigo-600 font-bold">[{evt.stage || "RUN"}]</span>
                <span className="text-stone-800 truncate">{evt.action || evt.event_type}</span>
                {evt.message && <span className="text-stone-500 truncate text-[10px]">· {evt.message}</span>}
                {evt.evidence_refs && evt.evidence_refs.length > 0 && (
                  <span className="text-indigo-600 shrink-0">
                    {evt.evidence_refs.slice(0, 3).map((ev) => (
                      <button
                        key={ev}
                        onClick={() => onSelectEvidence && onSelectEvidence(ev)}
                        className="hover:underline ml-1 font-medium cursor-pointer"
                      >
                        [{ev}]
                      </button>
                    ))}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 text-stone-400 shrink-0 ml-2">
                {evt.latency_ms ? <span>{evt.latency_ms.toFixed(1)}ms</span> : null}
                <CheckCircle className="w-3 h-3 text-emerald-600" />
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
