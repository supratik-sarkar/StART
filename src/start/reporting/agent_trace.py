"""Visible agent reasoning traces (v2.1.1 Section K).

Every agent emits an ``AgentTrace`` so a first-time user can see what the agent
reviewed, the evidence it used, a short reasoning summary, the decision it made,
its confidence, the alternative it considered, and the action it took — without
reading source code.

This is additive: the v2.1.0 ``ActionLog`` is unchanged. ``AgentTrace`` carries
the richer "thinking" fields the reviewer assistant needs, and a ``TraceLog`` collects
them for terminal, notebook, dashboard, and report rendering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Canonical agents the review execution layer makes visible (Section K).
TRACEABLE_AGENTS = (
    "DatasetDiscoveryAgent",
    "TargetDiscoveryAgent",
    "TaskInferenceAgent",
    "FeatureEngineeringAgent",
    "ArchitectureReviewAgent",
    "HyperparameterTuningAgent",
    "ReviewPlanner",
    "TestSuggestion",
    "ModelRiskFinding",
    "Challenge",
    "Governance",
    "Signoff",
    "EvidenceCritic",
)


@dataclass
class AgentTrace:
    agent: str
    inputs: str
    decision: str
    reasoning: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    confidence: float | None = None
    alternative_considered: str = ""
    action_taken: str = ""
    user_decision: str | None = None
    artifacts: list[str] = field(default_factory=list)
    backend: str = "deterministic"  # deterministic | openai | anthropic | grok | enterprise
    llm_used: bool = False
    fallback_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "inputs": self.inputs,
            "evidence_ids": self.evidence_ids,
            "reasoning": self.reasoning,
            "decision": self.decision,
            "confidence": self.confidence,
            "alternative_considered": self.alternative_considered,
            "action_taken": self.action_taken,
            "user_decision": self.user_decision,
            "artifacts": self.artifacts,
            "backend": self.backend,
            "llm_used": self.llm_used,
            "fallback_reason": self.fallback_reason,
        }

    def render_terminal(self) -> str:
        conf = f" (confidence {self.confidence:.0%})" if self.confidence is not None else ""
        from start.cli.view import AGENT_COLOR_REGISTRY
        color = AGENT_COLOR_REGISTRY.get(self.agent, "white")
        lines = [
            f"  +-- [{color}]{self.agent}[/{color}]{conf}",
            f"  |   inputs    : {self.inputs}",
        ]
        if self.evidence_ids:
            lines.append(f"  |   evidence  : {', '.join(self.evidence_ids)}")
        if self.reasoning:
            lines.append(f"  |   reasoning : {self.reasoning}")
        if self.alternative_considered:
            lines.append(f"  |   alt       : {self.alternative_considered}")
        lines.append(f"  |   decision  : {self.decision}")
        if self.action_taken:
            lines.append(f"  |   action    : {self.action_taken}")
        if self.user_decision:
            lines.append(f"  |   user      : {self.user_decision}")
        return "\n".join(lines)


@dataclass
class TraceLog:
    traces: list[AgentTrace] = field(default_factory=list)

    def record(self, agent: str, inputs: str, decision: str, **kwargs: Any) -> AgentTrace:
        trace = AgentTrace(agent=agent, inputs=inputs, decision=decision, **kwargs)
        self.traces.append(trace)
        return trace

    def add(self, trace: AgentTrace) -> AgentTrace:
        self.traces.append(trace)
        return trace

    def to_list(self) -> list[dict[str, Any]]:
        return [t.to_dict() for t in self.traces]

    def agents(self) -> list[str]:
        return [t.agent for t in self.traces]

    def render_terminal(self) -> str:
        return "\n".join(t.render_terminal() for t in self.traces)


def render_trace_log_markdown(log: TraceLog) -> str:
    if not log.traces:
        return "### Agent reasoning traces\n\n_No agent traces recorded._\n"
    lines = [
        "### Agent reasoning traces",
        "",
        "| Agent | Inputs | Reasoning | Decision | Confidence | Alternative | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for t in log.traces:
        conf = f"{t.confidence:.0%}" if t.confidence is not None else "—"
        lines.append(
            f"| {t.agent} | {t.inputs} | {t.reasoning or '—'} | {t.decision} "
            f"| {conf} | {t.alternative_considered or '—'} "
            f"| {', '.join(t.evidence_ids) or '—'} |"
        )
    return "\n".join(lines) + "\n"
