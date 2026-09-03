from typing import Any

from start.agents.base import BaseAgent
from start.telemetry.bus import TelemetryBus


class RootCauseAgent(BaseAgent):
    """v2.4.0 Core Diagnostic Agent identifying anomalies and structuring precise fix directives."""

    def __init__(self, telemetry_bus: TelemetryBus):
        super().__init__(name="Root Cause Agent", telemetry_bus=telemetry_bus)

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyzes a validation failure matrix and generates clear remediation parameters."""
        violation_type = context.get("violation_type", "Unknown")
        evidence_id = context.get("evidence_id", "EV-UNKNOWN")
        metrics = context.get("metrics", {})

        # Default fallback directive structure
        action_directive = {"strategy": "none", "parameters": {}}
        alternatives = []
        reasoning = ""
        confidence = 0.5

        if violation_type == "class_imbalance":
            minority_pct = metrics.get("minority_percentage", 0.0)
            reasoning = f"Minority class proportion is critically low at {minority_pct:.2f}%. Cross-entropy optimization will collapse."
            alternatives = [
                "Strategy A: Inverse Class Weighting",
                "Strategy B: Synthetic Over-sampling (SMOTE)",
            ]
            confidence = 0.95
            action_directive = {
                "strategy": "apply_class_weights",
                "parameters": {"calculate_weights": True, "target_column": context.get("target_column")},
            }
        elif violation_type == "mps_oom_risk":
            reasoning = "Hardware memory allocation exceeds safety bounds on Apple Silicon. Execution loop throttling required."
            alternatives = [
                "Strategy A: Batch Size Reduction",
                "Strategy B: Fallback to local CPU allocation",
            ]
            confidence = 0.88
            action_directive = {
                "strategy": "scale_down_batch",
                "parameters": {"reduction_factor": 0.5, "clear_cache": True},
            }
        else:
            reasoning = f"Generic or unhandled pipeline exception encountered: {violation_type}."
            alternatives = ["Manual pipeline review escalation"]

        # Emit the structured reasoning trace to the bus
        self.emit_trace(
            stage="Anomaly Root Cause Analysis",
            progress=100.0,
            status_msg=f"Root cause processing finished for {violation_type}.",
            reasoning_step=reasoning,
            confidence_score=confidence,
            alternatives_considered=alternatives,
            evidence_citations=[evidence_id] if evidence_id != "EV-UNKNOWN" else [],
            action_directive=action_directive,
            metrics=metrics,
        )

        return {"status": "analyzed", "action_directive": action_directive}
