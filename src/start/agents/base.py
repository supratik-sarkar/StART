import abc
from typing import Any

from start.telemetry.bus import CopilotMessageTrace, TelemetryBus, TelemetryEvent


class BaseAgent(abc.ABC):
    """v2.4.0 Enterprise Stateful Agent Interface enforcing meticulous reasoning tracing.
    
    Systems Directive: Contextual Human Query Handling Protocol
    CRITICAL INTERACTIVE CAPABILITY:
    When handling direct, ad-hoc engineer queries via the interactive `[Q]` loop:
    1. Distinguish between an empirical claim about the current run dataset (which requires an 'EV-', 'FE-', or 'ARCH-' token citation) and an architectural, conceptual, or algorithmic question (which requires your baseline expert AI engineering knowledge).
    2. If the engineer asks an abstract design or risk question (e.g., lookahead leakage, temporal imputation, neural structural choices, gradient behavior), you are explicitly AUTHORIZED and REQUIRED to utilize your full deep learning and data science knowledge to provide a rigorous answer.
    3. Do not output defensive boilerplate text such as 'I do not have sufficient evidence to answer this question' for purely conceptual or design-pattern queries.
    4. If the user raises a valid architectural critique about your original recommendation, pivot your logic, accept the user's perspective, and provide the programmatic path to execute the safer alternative (e.g., dropping back to localized linear interpolation instead of a global static median).
    """
    def __init__(self, name: str, telemetry_bus: TelemetryBus | None = None):
        self.name: str = name
        self.telemetry_bus: TelemetryBus | None = telemetry_bus

    def emit_trace(
        self,
        stage: str,
        progress: float,
        status_msg: str,
        reasoning_step: str,
        confidence_score: float,
        alternatives_considered: list[str] | None = None,
        evidence_citations: list[str] | None = None,
        action_directive: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None
    ) -> None:
        """Publishes a structured reasoning trace to the centralized telemetry pipeline."""
        if not self.telemetry_bus:
            return

        # Instantiate a type-enforced message trace instead of raw string conversation
        trace = CopilotMessageTrace(
            reasoning_step=reasoning_step,
            alternatives_considered=alternatives_considered or [],
            evidence_citations=evidence_citations or [],
            confidence_score=confidence_score,
            action_directive=action_directive
        )

        event = TelemetryEvent(
            agent_name=self.name,
            stage=stage,
            progress_percentage=progress,
            status_msg=status_msg,
            metrics=metrics or {},
            trace_details=trace
        )
        self.telemetry_bus.publish(event)

    @abc.abstractmethod
    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Executes the specialized analytical task boundary for the agent node."""
        pass
