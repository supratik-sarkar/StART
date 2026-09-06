"""Authoritative RuntimeEvent and EventSink definitions for StART v5.1.0.

Provides typed event structures emitted at genuine execution boundaries
and shared between CLI, Web transport, and acceptance harnesses.

Strict Invariants:
1. Zero imports of start.web (CORE_RUNTIME_IMPORTS_START_WEB = 0).
2. Events emitted only at genuine operation boundaries.
3. Stable node_id, parent_node_id, test_id, and evidence/artifact bindings.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass
class RuntimeEvent:
    """Canonical event describing an in-process agent, test, tool, or governance transition."""

    event_id: str = field(default_factory=lambda: f"EVT-{uuid.uuid4().hex[:8]}")
    timestamp: float = field(default_factory=time.time)
    run_id: str = ""
    event_type: str = "agent_transition"
    # context_ready, test_started, test_completed, tuning_trial, evidence_committed,
    # artifact_created, governance_decided, attestation_created, workflow_completed, error
    status: str = "SUCCESS"  # PENDING, RUNNING, SUCCESS, COMPLETED, WARN, FAIL, ERROR, SKIPPED
    source_agent: str = "Director"
    target_agent: str = "Specialist"
    stage: str = "PLANNING"  # PLANNING, EXECUTION, TUNING, CHECKPOINTS, GOVERNANCE, COMPLETED
    action: str = ""
    node_id: str | None = None
    parent_node_id: str | None = None
    test_id: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    step: Any | None = None
    phase: str | None = None
    completed: int | None = None
    total: int | None = None
    percent: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        return d

    @property
    def payload(self) -> dict[str, Any]:
        return self.to_dict()


class RuntimeEventSink(Protocol):
    """Protocol for receiving real-time canonical execution events."""

    def emit(self, event: RuntimeEvent) -> None:
        """Emit a canonical runtime event."""
        ...


class NoOpEventSink:
    """Event sink that discards all events (used for headless / silent CLI execution)."""

    def emit(self, event: RuntimeEvent) -> None:
        pass


class ListEventSink:
    """Event sink that records all emitted events in an in-memory list."""

    def __init__(self, target: list[RuntimeEvent] | None = None) -> None:
        self.events: list[RuntimeEvent] = target if target is not None else []

    def emit(self, event: RuntimeEvent) -> None:
        self.events.append(event)


class CallableEventSink:
    """Event sink that forwards events to a callable handler."""

    def __init__(self, callback: Callable[[RuntimeEvent], None]) -> None:
        self.callback = callback

    def emit(self, event: RuntimeEvent) -> None:
        self.callback(event)
