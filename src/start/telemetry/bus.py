import asyncio
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field, field_validator


class CopilotMessageTrace(BaseModel):
    reasoning_step: str = Field(..., description="Mathematical or logical step evaluated.")
    alternatives_considered: list[str] = Field(default_factory=list, description="Alternative hypotheses evaluated.")
    evidence_citations: list[str] = Field(default_factory=list, description="Strict references to EV-xxxx hashes.")
    confidence_score: float = Field(..., description="Calculated metric stability bounds.")
    action_directive: dict[str, Any] | None = Field(None, description="Programmatic instructions for self-healing.")

    @field_validator("evidence_citations")
    @classmethod
    def validate_citations(cls, v: list[str]) -> list[str]:
        for citation in v:
            if not citation.startswith("EV-") and not citation.startswith("FE-") and not citation.startswith("ARCH-") and not citation.startswith("TUNE-"):
                raise ValueError(
                    f"Meticulous Communication Guardrail Violation / Meticulous Citation Guardrail Violation: "
                    f"{citation} must start with standard prefix."
                )
        return v

class TelemetryEvent(BaseModel):
    timestamp: float = Field(default_factory=time.time)
    agent_name: str
    stage: str
    progress_percentage: float
    status_msg: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    trace_details: CopilotMessageTrace | None = None

class TelemetryBus:
    def __init__(self):
        self._subscribers: list[Callable[[TelemetryEvent], None]] = []
        self._latest_state: dict[str, TelemetryEvent] = {}

    def subscribe(self, callback: Callable[[TelemetryEvent], None]) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def publish(self, event: TelemetryEvent) -> None:
        self._latest_state[event.agent_name] = event
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        for subscriber in self._subscribers:
            if loop and loop.is_running():
                loop.call_soon_threadsafe(subscriber, event)
            else:
                subscriber(event)

    def fetch_agent_snapshot(self, agent_name: str) -> TelemetryEvent | None:
        return self._latest_state.get(agent_name)
