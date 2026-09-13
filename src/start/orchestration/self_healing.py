import asyncio
from typing import Any

from start.agents import CostOptimizationAgent, FairnessAgent, RecoveryAgent, RootCauseAgent
from start.engine.state import StepCheckpointer
from start.telemetry.bus import TelemetryBus, TelemetryEvent


class SelfHealingOrchestrator:
    """v2.4.0 Closed-Loop Controller driving autonomous evaluation repairs and run comparisons."""
    def __init__(self, telemetry_bus: TelemetryBus, checkpointer: StepCheckpointer):
        self.bus = telemetry_bus
        self.checkpointer = checkpointer
        self.root_cause_agent = RootCauseAgent(telemetry_bus=self.bus)
        self.recovery_agent = RecoveryAgent(telemetry_bus=self.bus, checkpointer=self.checkpointer)
        self.cost_agent = CostOptimizationAgent(telemetry_bus=self.bus)
        self.fairness_agent = FairnessAgent(telemetry_bus=self.bus)

    async def run_stage_with_healing(self, workflow_id: str, stage_name: str, execution_context: dict[str, Any]) -> dict[str, Any]:
        self.bus.publish(TelemetryEvent(
            agent_name="System Coordinator",
            stage=stage_name,
            progress_percentage=20.0,
            status_msg=f"Initiating evaluation pass for step: {stage_name}..."
        ))
        await asyncio.sleep(0.5)

        violation_type = execution_context.get("violation_type")
        metrics_snapshot = execution_context.get("metrics", {})
        evidence_id = execution_context.get("evidence_id", "EV-GENERIC")

        if execution_context.get("simulate_anomaly"):
            # Route 1: Core Class Imbalance Data Skew
            if violation_type == "class_imbalance":
                rc_result = self.root_cause_agent.execute({
                    "violation_type": "class_imbalance",
                    "evidence_id": evidence_id,
                    "metrics": metrics_snapshot,
                    "target_column": execution_context.get("target_column")
                })
                await asyncio.sleep(0.5)
                
                self.recovery_agent.execute({
                    "workflow_id": workflow_id,
                    "stage_name": stage_name,
                    "action_directive": rc_result["action_directive"],
                    "evidence_citations": [evidence_id]
                })
                return {"status": "remediated", "active_remedy": "apply_class_weights"}

            # Route 2: Latency and Cost Performance Breaches
            elif violation_type == "cost_sla_breach":
                self.cost_agent.execute({
                    "evidence_id": evidence_id,
                    "metrics": metrics_snapshot,
                    "budget_limit_seconds": execution_context.get("budget_limit_seconds", 1.0)
                })
                await asyncio.sleep(0.5)
                return {"status": "remediated", "active_remedy": "optimize_compute_footprint"}

            # Route 3: Demographic Bias Protection Parity Breaches
            elif violation_type == "fairness_violation":
                self.fairness_agent.execute({
                    "evidence_id": evidence_id,
                    "metrics": metrics_snapshot,
                    "fairness_threshold": execution_context.get("fairness_threshold", 0.80),
                    "protected_attribute": execution_context.get("protected_attribute")
                })
                await asyncio.sleep(0.5)
                return {"status": "remediated", "active_remedy": "calibrate_decision_boundaries"}

        self.bus.publish(TelemetryEvent(
            agent_name="System Coordinator",
            stage=stage_name,
            progress_percentage=100.0,
            status_msg=f"Nominal check clean for stage {stage_name}."
        ))
        return {"status": "clean", "active_remedy": "none"}
