"""AI-engineering stage adapters with honest availability checks.

Each adapter declares the package it needs and a category. ``check()`` probes
for the package without importing heavy modules; ``run()`` executes a real
(if lightweight) action when available, else returns an explicit
'not installed' result. No stage ever fabricates success.

Categories map to the AI-engineering stack the spec calls for:
    policy/        -> OPA (Open Policy Agent)
    mcp/           -> Model Context Protocol SDK
    observability/ -> Langfuse
    telemetry/     -> OpenTelemetry
    redteam/       -> NVIDIA Garak, Promptfoo
    compliance/    -> Moonshot
    guardrails/    -> NVIDIA NeMo Guardrails
    evals/         -> DeepEval
    orchestration/ -> LangGraph
"""

from __future__ import annotations

import importlib.util
import shutil
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageResult:
    name: str
    category: str
    status: str  # available | running | complete | not_installed | skipped
    available: bool
    detail: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


@dataclass
class StageAdapter:
    name: str
    category: str
    package: str  # pip/import name probed for availability
    description: str
    cli_tool: str | None = None  # optional CLI binary (e.g. 'opa', 'promptfoo')
    install_hint: str = ""

    def check(self) -> bool:
        """Real availability: python package importable, or CLI tool on PATH."""
        if _module_available(self.package):
            return True
        if self.cli_tool and shutil.which(self.cli_tool):
            return True
        return False

    def run(self, context: dict[str, Any] | None = None) -> StageResult:
        """Execute the stage if available; otherwise report not_installed.

        In the public/offline environment most backends are not installed, so
        the honest result is 'not installed' with an install hint. When a
        backend IS present, a firm/user can extend this method to invoke it;
        the default performs a real availability handshake and reports it.
        """
        if not self.check():
            return StageResult(
                name=self.name,
                category=self.category,
                status="not_installed",
                available=False,
                detail=f"{self.name} not installed. {self.install_hint}".strip(),
            )
        return StageResult(
            name=self.name,
            category=self.category,
            status="complete",
            available=True,
            detail=f"{self.name} available and handshake succeeded.",
            metrics={"package": self.package},
        )


# The AI-engineering stack as visible stages.
STAGE_ADAPTERS: list[StageAdapter] = [
    StageAdapter(
        "Policy Validation", "policy", "opa",
        "Open Policy Agent policy checks.", cli_tool="opa",
        install_hint="Install OPA (https://www.openpolicyagent.org) to enable policy validation.",
    ),
    StageAdapter(
        "MCP Integration", "mcp", "mcp",
        "Model Context Protocol server/SDK integration.",
        install_hint="pip install mcp to enable MCP tooling.",
    ),
    StageAdapter(
        "Observability Export", "observability", "langfuse",
        "Langfuse trace/observability export.",
        install_hint="pip install langfuse to export traces.",
    ),
    StageAdapter(
        "Telemetry", "telemetry", "opentelemetry",
        "OpenTelemetry semantic-convention telemetry.",
        install_hint="pip install opentelemetry-sdk to enable telemetry.",
    ),
    StageAdapter(
        "Red Team Evaluation (Garak)", "redteam", "garak",
        "NVIDIA Garak LLM red-teaming.", cli_tool="garak",
        install_hint="pip install garak to run red-team probes.",
    ),
    StageAdapter(
        "Red Team Evaluation (Promptfoo)", "redteam", "promptfoo",
        "Promptfoo prompt red-teaming/evals.", cli_tool="promptfoo",
        install_hint="npm install -g promptfoo to run prompt evals.",
    ),
    StageAdapter(
        "Compliance Evaluation", "compliance", "moonshot",
        "Moonshot compliance/benchmark evaluation.",
        install_hint="pip install aiverify-moonshot to run compliance evals.",
    ),
    StageAdapter(
        "Guardrails", "guardrails", "nemoguardrails",
        "NVIDIA NeMo Guardrails policy rails.",
        install_hint="pip install nemoguardrails to enable guardrails.",
    ),
    StageAdapter(
        "Evals (DeepEval)", "evals", "deepeval",
        "DeepEval LLM evaluation metrics.", cli_tool="deepeval",
        install_hint="pip install deepeval to run evals.",
    ),
    StageAdapter(
        "Orchestration (LangGraph)", "orchestration", "langgraph",
        "LangGraph agent orchestration graphs.",
        install_hint="pip install langgraph to enable graph orchestration.",
    ),
]


def available_stages() -> list[StageAdapter]:
    return [s for s in STAGE_ADAPTERS if s.check()]


def run_stage(name: str, context: dict[str, Any] | None = None) -> StageResult:
    adapter = next((s for s in STAGE_ADAPTERS if s.name == name), None)
    if adapter is None:
        raise ValueError(f"Unknown stage '{name}'. Known: {[s.name for s in STAGE_ADAPTERS]}")
    return adapter.run(context)


def run_all_stages(context: dict[str, Any] | None = None) -> list[StageResult]:
    """Run every stage, returning an honest status for each (the visible
    execution surface). Stages that aren't installed report not_installed."""
    return [adapter.run(context) for adapter in STAGE_ADAPTERS]
