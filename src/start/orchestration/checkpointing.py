"""Project-owned LangGraph checkpoint serialization policy.

The allowlist is passed through LangGraph's supported public constructor.  It
does not mutate LangGraph internals and does not depend on interpreter startup
hooks or ``.pth`` files.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

START_MSGPACK_ALLOWLIST: tuple[tuple[str, str], ...] = (
    ("start.core.schemas", "Status"),
    ("start.core.schemas", "ComputeDevice"),
    ("start.core.schemas", "EvidenceRecord"),
)


def build_checkpoint_serializer() -> JsonPlusSerializer:
    """Return the strict serializer used by StART-owned checkpointers."""
    return JsonPlusSerializer(allowed_msgpack_modules=START_MSGPACK_ALLOWLIST)


def build_memory_checkpointer() -> MemorySaver:
    """Create an in-memory saver with the explicit StART allowlist."""
    return MemorySaver(serde=build_checkpoint_serializer())
