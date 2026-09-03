"""StART — Standardized Agentic Reusable Tests.

A risk-stripe-agnostic review platform:

    deterministic engines compute  ->  agents reason  ->  evidence proves
                                                       ->  seals attest

**Why this module imports lazily.**

The subpackages that matter most in a constrained environment —
:mod:`start.risk`, :mod:`start.attestation`, :mod:`start.runtime_profile` — are
standard-library only, by design. They have to be, because the first questions
StART answers inside a locked-down environment are *what egress am I under* and
*what does this review owe*, and neither may depend on whether numpy imported
cleanly.

Eagerly importing the modelling stack from this file would destroy that
property: ``import start.risk`` would drag in pydantic, pandas and scikit-learn
and fail in exactly the environment where the risk core is most useful. So
top-level names resolve on first access (PEP 562) instead. ``import start`` is
therefore cheap and dependency-free; ``start.run_review`` pulls in what it
needs at the moment you ask for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "4.5.2"

if TYPE_CHECKING:  # pragma: no cover - static analysers only
    from start.core.config import StartConfig, load_config, load_policy
    from start.core.schemas import EvidenceRecord, RunResult, Status, TestResult
    from start.orchestration.pipeline import build_context, run_review
    from start.registry import TestContext, list_tests, register_test

#: name -> module providing it. Resolved on first attribute access.
_LAZY: dict[str, str] = {
    "StartConfig": "start.core.config",
    "load_config": "start.core.config",
    "load_policy": "start.core.config",
    "EvidenceRecord": "start.core.schemas",
    "RunResult": "start.core.schemas",
    "Status": "start.core.schemas",
    "TestResult": "start.core.schemas",
    "TestContext": "start.registry",
    "register_test": "start.registry",
    "list_tests": "start.registry",
    "build_context": "start.orchestration.pipeline",
    "run_review": "start.orchestration.pipeline",
}

__all__ = [
    "EvidenceRecord",
    "RunResult",
    "StartConfig",
    "Status",
    "TestContext",
    "TestResult",
    "__version__",
    "build_context",
    "list_tests",
    "load_config",
    "load_policy",
    "register_test",
    "run_review",
]


def __getattr__(name: str) -> Any:
    """Resolve a top-level name on first use (PEP 562)."""
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'start' has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path), name)
    globals()[name] = value  # cache so subsequent access is a plain lookup
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
