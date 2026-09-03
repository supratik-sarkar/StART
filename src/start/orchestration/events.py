"""Unified Canonical RuntimeEvent Model for StART.

Serves as the single authoritative event schema for:
- Terminal Rich execution trace
- `agent_orchestration.json` export
- LangGraph state and transition history
- OpenTelemetry span collection
- Optional LangSmith, Langfuse, and Phoenix exporters
- Future Web / Ollama / WebLLM interfaces
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RuntimeEvent:
    """Canonical event describing an in-process agent, tool, or governance transition."""

    event_id: str
    timestamp: float = field(default_factory=time.time)
    event_type: str = "agent_transition"  # "agent_transition" | "tool_execution" | "policy_decision" | "evidence_commit" | "artifact_generate" | "governance_seal"
    source_agent: str = "Director"
    target_agent: str = "DeterministicEngine"
    stage: str = "PLANNING"
    action: str = "discover_applicable_tests"
    evidence_refs: list[str] = field(default_factory=list)
    artifact_refs: list[str] = field(default_factory=list)
    policy_decision: str = "ALLOW"
    guardrail_status: str = "PASS"
    latency_ms: float = 0.0
    status: str = "SUCCESS"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        """Cryptographic SHA-256 fingerprint of the event."""
        payload = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source_agent,
            "target": self.target_agent,
            "action": self.action,
            "evidence_refs": sorted(self.evidence_refs),
            "status": self.status,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
