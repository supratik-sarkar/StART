"""Authoritative Canonical Execution Runtime for StART v5.1.0.

Provides the single non-web shared orchestration and execution layer for:
- Terminal CLI execution
- Web API execution
- Automated acceptance and testing harness

Strict Invariants:
1. CORE_RUNTIME_IMPORTS_START_WEB = 0 (No dependencies on start.web).
2. WEB_PACKAGE_OWNS_EXECUTION_SEMANTICS = NO.
3. Deterministic execution boundary parity between terminal and transport.
4. Real-time canonical events emitted at actual execution boundaries.
"""

from __future__ import annotations

from start.runtime.contexts import (
    ExecutionContextInstance,
    ExecutionContextSpec,
    get_canonical_context_specs,
    instantiate_context,
    resolve_context_spec,
)
from start.runtime.events import (
    CallableEventSink,
    ListEventSink,
    NoOpEventSink,
    RuntimeEvent,
    RuntimeEventSink,
)
from start.runtime.execution import (
    CanonicalExecutionService,
    ExecutionResult,
)
from start.runtime.workflows import (
    WORKFLOW_SPECS,
    EngineKind,
    ResolvedWorkflowExecution,
    WorkflowExecutionSpec,
    get_canonical_workflow_specs,
    get_workflow_catalog,
    resolve_workflow,
)

__all__ = [
    "CallableEventSink",
    "CanonicalExecutionService",
    "EngineKind",
    "ExecutionContextInstance",
    "ExecutionContextSpec",
    "ExecutionResult",
    "ListEventSink",
    "NoOpEventSink",
    "ResolvedWorkflowExecution",
    "RuntimeEvent",
    "RuntimeEventSink",
    "WORKFLOW_SPECS",
    "WorkflowExecutionSpec",
    "get_canonical_context_specs",
    "get_canonical_workflow_specs",
    "get_workflow_catalog",
    "instantiate_context",
    "resolve_context_spec",
    "resolve_workflow",
]
