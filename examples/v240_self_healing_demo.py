import asyncio
import sys
import os

# Ensure repo root is explicitly on the path for clean resolution boundaries
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from start.telemetry.bus import TelemetryBus
from start.engine.state import StepCheckpointer
from start.cli.view import ProgressDashboardUI
from start.orchestration.self_healing import SelfHealingOrchestrator

async def run_copilot_evaluation_pipeline():
    """Simulates a model validation workload passing through nominal and anomalous conditions."""
    bus = TelemetryBus()
    checkpointer = StepCheckpointer()
    ui = ProgressDashboardUI(bus)
    orchestrator = SelfHealingOrchestrator(telemetry_bus=bus, checkpointer=checkpointer)

    async def core_execution_sequence():
        # Setup run identification bounds
        workflow_id = "RUN-v240-M4PRO-TRANSITION"
        
        # --- STAGE 1: NOMINAL TRACK ---
        ui.set_global_progress(0.0)
        await orchestrator.run_stage_with_healing(
            workflow_id=workflow_id,
            stage_name="Feature Analysis",
            execution_context={"simulate_anomaly": False}
        )
        
        # --- STAGE 2: CLOSED-LOOP SELF-HEALING TRACK ---
        ui.set_global_progress(40.0)
        remediation_summary = await orchestrator.run_stage_with_healing(
            workflow_id=workflow_id,
            stage_name="Data Audit",
            execution_context={
                "simulate_anomaly": True,
                "target_column": "client_attrition_risk"
            }
        )
        
        # --- STAGE 3: SYNTHESIS & REPORTING ---
        ui.set_global_progress(85.0)
        await asyncio.sleep(1.0)
        
        ui.set_global_progress(100.0)
        # Leave a clean visual frame anchor on completion
        await asyncio.sleep(0.5)
        
    # Bind the execution loop sequence into the Rich Live layout terminal driver
    await ui.monitor_execution(core_execution_sequence())

if __name__ == "__main__":
    try:
        asyncio.run(run_copilot_evaluation_pipeline())
    except KeyboardInterrupt:
        print("\nExecution halted via human supervisor checkpoint override.")
