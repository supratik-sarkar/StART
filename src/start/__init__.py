"""StART: Standardized Agentic Reusable Tests.

Deterministic quantitative validation engines + agent-assisted orchestration,
proof-carrying narratives, tamper-evident evidence, and adaptive compute.
"""

from start.core.config import StartConfig, load_config, load_policy
from start.core.schemas import EvidenceRecord, RunResult, Status, TestResult
from start.orchestration.pipeline import build_context, run_review
from start.registry import TestContext, list_tests, register_test

__version__ = "0.1.0"

__all__ = [
    "StartConfig",
    "load_config",
    "load_policy",
    "EvidenceRecord",
    "RunResult",
    "Status",
    "TestResult",
    "TestContext",
    "register_test",
    "list_tests",
    "build_context",
    "run_review",
    "__version__",
]
