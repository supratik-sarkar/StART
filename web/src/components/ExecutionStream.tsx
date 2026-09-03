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
      position: { x: 50, y: 150 },
      data: { label: "Director Orchestrator", role: "director", status: "SUCCESS" },
      style: { background: "#0f172a", color: "#f8fafc", border: "1px solid #3b82f6", width: 170, fontSize: 11, padding: 8 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    });

    calculatedNodes.push({
      id: "Specialist",
      position: { x: 280, y: 80 },
      data: { label: "Domain Specialist", role: "specialist", status: "SUCCESS" },
      style: { background: "#0f172a", color: "#f8fafc", border: "1px solid #64748b", width: 160, fontSize: 11, padding: 8 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    });

    calculatedNodes.push({
      id: "DeterministicEngine",
      position: { x: 500, y: 80 },
      data: { label: "Deterministic Engine (79 Tools)", role: "engine", status: "SUCCESS" },
      style: { background: "#0f172a", color: "#f8fafc", border: "1px solid #10b981", width: 190, fontSize: 11, padding: 8 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    });

    calculatedNodes.push({
      id: "PolicyEngine",
      position: { x: 280, y: 220 },
      data: { label: "OPA Policy Plane", role: "policy", status: "SUCCESS" },
      style: { background: "#0f172a", color: "#f8fafc", border: "1px solid #8b5cf6", width: 160, fontSize: 11, padding: 8 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    });

    calculatedNodes.push({
      id: "AttestationLedger",
      position: { x: 740, y: 150 },
      data: { label: "Merkle Attestation Ledger", role: "attestation", status: "SUCCESS" },
      style: { background: "#0f172a", color: "#f8fafc", border: "1px solid #f59e0b", width: 180, fontSize: 11, padding: 8 },
      sourcePosition: Position.Right,
      targetPosition: Position.Left,
    });

    // Build Static Topology Edges
    calculatedEdges.push({
      id: "e-dir-spec",
      source: "Director",
      target: "Specialist",
      animated: true,
      style: { stroke: "#3b82f6" },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#3b82f6" },
    });
    calculatedEdges.push({
      id: "e-spec-eng",
      source: "Specialist",
      target: "DeterministicEngine",
      animated: true,
      style: { stroke: "#10b981" },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#10b981" },
    });
    calculatedEdges.push({
      id: "e-dir-pol",
      source: "Director",
      target: "PolicyEngine",
      animated: true,
      style: { stroke: "#8b5cf6" },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#8b5cf6" },
    });
    calculatedEdges.push({
      id: "e-eng-att",
      source: "DeterministicEngine",
      target: "AttestationLedger",
      animated: true,
      style: { stroke: "#f59e0b" },
      markerEnd: { type: MarkerType.ArrowClosed, color: "#f59e0b" },
    });

    return { nodes: calculatedNodes, edges: calculatedEdges };
  }, [events]);

  return (
    <div className="flex flex-col h-full bg-card border border-border rounded-lg overflow-hidden">
      {/* Top Banner */}
      <div className="p-3 border-b border-border bg-card/80 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary animate-pulse" />
          <h3 className="text-xs font-semibold text-foreground tracking-wide uppercase">
            Live Agent Execution Graph (React Flow Runtime)
          </h3>
        </div>
        <div className="flex items-center gap-3 text-[11px] font-mono text-muted-foreground">
          <span className="flex items-center gap-1">
            <Cpu className="w-3 h-3 text-green-400" />
            79 Deterministic Engines
          </span>
          <span className="flex items-center gap-1">
            <ShieldCheck className="w-3 h-3 text-purple-400" />
            OPA Active
          </span>
          <span className="flex items-center gap-1">
            <Database className="w-3 h-3 text-amber-400" />
            Merkle Sealed
          </span>
        </div>
      </div>

      {/* React Flow Canvas */}
      <div className="flex-1 w-full min-h-[220px] bg-background relative">
        <ReactFlow nodes={nodes} edges={edges} fitView>
          <Background color="#1e293b" gap={16} size={1} />
          <Controls position="bottom-right" className="bg-card border border-border text-foreground" />
        </ReactFlow>
      </div>

      {/* Recent Event Log Strip */}
      <div className="border-t border-border bg-card/60 p-2.5 max-h-36 overflow-y-auto font-mono text-xs divide-y divide-border/40">
        {events.length === 0 ? (
          <div className="text-muted-foreground text-[11px] text-center py-2">
            Awaiting analytical execution events...
          </div>
        ) : (
          events.slice(-5).reverse().map((evt, idx) => (
            <div key={idx} className="py-1.5 flex items-center justify-between text-[11px]">
              <div className="flex items-center gap-2">
                <span className="text-primary font-bold">[{evt.stage || "RUN"}]</span>
                <span className="text-foreground">{evt.action || evt.event_type}</span>
                {evt.evidence_refs && evt.evidence_refs.length > 0 && (
                  <span className="text-blue-400">
                    {evt.evidence_refs.map((ev) => (
                      <button
                        key={ev}
                        onClick={() => onSelectEvidence && onSelectEvidence(ev)}
                        className="hover:underline ml-1"
                      >
                        [{ev}]
                      </button>
                    ))}
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                {evt.latency_ms ? <span>{evt.latency_ms.toFixed(1)}ms</span> : null}
                <CheckCircle className="w-3 h-3 text-green-400" />
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
