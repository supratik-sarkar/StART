import time
from collections.abc import Callable
from typing import Any

from start.engine.state import StepCheckpointer
from start.orchestration.self_healing import SelfHealingOrchestrator
from start.telemetry.bus import TelemetryBus, TelemetryEvent


class SelfHealingExecutionInterceptor:
    """Enterprise interceptor middleware executing deterministic tests with automatic remediation routing."""
    def __init__(self, telemetry_bus: TelemetryBus, checkpointer: StepCheckpointer):
        self.bus = telemetry_bus
        self.checkpointer = checkpointer
        self.healing_orchestrator = SelfHealingOrchestrator(
            telemetry_bus=self.bus, 
            checkpointer=self.checkpointer
        )

    async def execute_test_step(
        self, 
        workflow_id: str, 
        stage_name: str, 
        test_callable: Callable[..., Any], 
        test_context: Any, 
        *args: Any, 
        **kwargs: Any
    ) -> dict[str, Any]:
        """Wraps standard test executions, tracking metrics and dynamically spinning up self-healing loops upon failure."""
        start_time = time.time()
        
        # Stream initial test launch event state
        self.bus.publish(TelemetryEvent(
            agent_name="System Interceptor",
            stage=stage_name,
            progress_percentage=10.0,
            status_msg=f"Spawning execution frame for test component in stage: {stage_name}."
        ))

        try:
            # Execute the core deterministic test function from the StART registry
            test_result = test_callable(test_context, *args, **kwargs)
            duration = time.time() - start_time

            # Check if the test result contains explicit threshold failures or warnings
            # This safely ducks under existing StART TestResult property contracts
            is_breached = False
            metrics_payload = {}
            
            if hasattr(test_result, "status") and test_result.status in ["fail", "warn", "error"]:
                is_breached = True
            if hasattr(test_result, "metrics"):
                metrics_payload = test_result.metrics

            if is_breached:
                self.bus.publish(TelemetryEvent(
                    agent_name="System Interceptor",
                    stage=stage_name,
                    progress_percentage=40.0,
                    status_msg=f"Threshold breach encountered during {stage_name}. Activating self-healing loop...",
                    metrics=metrics_payload
                ))
                
                # Divert the execution flow safely to the self-healing layout tracks
                healing_context = {
                    "simulate_anomaly": True,
                    "violation_type": "class_imbalance" if "minority" in str(metrics_payload).lower() else "generic_breach",
                    "evidence_id": getattr(test_result, "test_id", f"EV-{stage_name.upper()}"),
                    "metrics": metrics_payload,
                    "target_column": getattr(test_context, "target_column", "target")
                }
                
                remediation = await self.healing_orchestrator.run_stage_with_healing(
                    workflow_id=workflow_id,
                    stage_name=stage_name,
                    execution_context=healing_context
                )
                
                return {
                    "status": "remediated",
                    "test_id": getattr(test_result, "test_id", "unknown"),
                    "runtime_seconds": duration,
                    "remediation_summary": remediation
                }

            # Nominal clean path execution
            self.bus.publish(TelemetryEvent(
                agent_name="System Interceptor",
                stage=stage_name,
                progress_percentage=100.0,
                status_msg=f"Test pass verified cleanly for stage: {stage_name}.",
                metrics=metrics_payload
            ))
            
            return {
                "status": "passed",
                "test_id": getattr(test_result, "test_id", "unknown"),
                "runtime_seconds": duration,
                "metrics": metrics_payload
            }

        except Exception as exc:
            # Fallback resiliency path for unhandled execution framework exceptions
            duration = time.time() - start_time
            self.bus.publish(TelemetryEvent(
                agent_name="Recovery Agent",
                stage=stage_name,
                progress_percentage=0.0,
                status_msg=f"Fatal structural crash detected in {stage_name}: {str(exc)}"
            ))
            
            # Save a crash fault state snapshot to disk before escalation
            self.checkpointer.save_checkpoint(
                workflow_id=workflow_id,
                stage_name=stage_name,
                payload={"fatal_exception": str(exc), "runtime": duration}
            )
            
            raise exc
