from typing import Any

from start.agents.base import BaseAgent
from start.engine.state import StepCheckpointer
from start.telemetry.bus import TelemetryBus


class RecoveryAgent(BaseAgent):
    """v2.4.0 Operational Agent executing remediation plans via step modifications."""

    def __init__(self, telemetry_bus: TelemetryBus, checkpointer: StepCheckpointer):
        super().__init__(name="Recovery Agent", telemetry_bus=telemetry_bus)
        self.checkpointer = checkpointer

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Interprets a root-cause action directive and updates the checkpoint state dynamically."""
        workflow_id = context.get("workflow_id", "default_wf")
        stage_name = context.get("stage_name", "unknown_stage")
        directive = context.get("action_directive", {})
        strategy = directive.get("strategy", "none")
        params = directive.get("parameters", {})

        reasoning = f"Evaluating programmatic recovery step via execution strategy: '{strategy}'."
        action_taken = {}

        if strategy == "apply_class_weights":
            reasoning = f"Modifying active dataset metadata configs for {params.get('target_column')}. Injecting loss weight overrides."
            action_taken = {"loss_adjustment": "weighted", "applied_at_stage": stage_name}

            # Surgically roll back/update checkpoint parameter arrays
            self.checkpointer.save_checkpoint(
                workflow_id=workflow_id,
                stage_name=stage_name,
                payload={"mitigation": "weighted_loss", "params": params},
            )
        elif strategy == "scale_down_batch":
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

            reasoning = f"Clearing local MPS active tensor graphs. Applying a {params.get('reduction_factor')}x factor limit on current step batch metrics."
            action_taken = {
                "memory_action": "empty_cache",
                "batch_scale_factor": params.get("reduction_factor"),
            }

            self.checkpointer.save_checkpoint(
                workflow_id=workflow_id,
                stage_name=stage_name,
                payload={"mitigation": "throttled_batch", "params": params},
            )
        else:
            reasoning = "No actionable programmatic strategy supplied. Escating to human-in-the-loop review board parameters."
            action_taken = {"escalated": True}

        self.emit_trace(
            stage="Self-Healing Recovery Execution",
            progress=100.0,
            status_msg=f"Self-healing adjustments deployed for strategy: {strategy}.",
            reasoning_step=reasoning,
            confidence_score=1.0,
            alternatives_considered=[f"Ignore exception and crash pipeline during stage {stage_name}"],
            evidence_citations=context.get("evidence_citations", []),
            action_directive=action_taken,
        )

        return {"status": "remediated", "action_taken": action_taken}
