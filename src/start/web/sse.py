"""Typed SSE Event Broadcaster & Reconnection Manager for StART v4.5.

Bridges canonical RuntimeEvents to frontend EventSource listeners with:
- Monotonic sequence numbering
- Reconnection support via Last-Event-ID
- Typed envelope validation against start.web.schemas.SSEEnvelope
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator

from start.web.queue import GLOBAL_QUEUE, ActiveRunContext
from start.web.schemas import SSEEnvelope, START_SCHEMA_VERSION

logger = logging.getLogger("start.web.sse")


async def sse_event_generator(
    run_id: str,
    session_id: str | None = None,
    last_event_id: str | None = None,
    poll_interval: float = 0.05,
    timeout_seconds: float = 300.0,
) -> AsyncGenerator[dict[str, Any], None]:
    """Asynchronously stream typed SSE envelopes for a run until completion or timeout."""
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        yield {
            "event": "error",
            "id": "EVT-ERR-01",
            "data": json.dumps({"error": f"Run '{run_id}' not found or session access denied"}),
        }
        return

    sent_indices = set()
    start_time = asyncio.get_event_loop().time()
    sequence = 1

    # If Last-Event-ID is passed, attempt to seek past previously delivered events
    if last_event_id and last_event_id.startswith("EVT-SEQ-"):
        try:
            last_seq = int(last_event_id.split("EVT-SEQ-")[-1])
            for i in range(min(last_seq, len(ctx.events))):
                sent_indices.add(i)
            sequence = len(sent_indices) + 1
        except Exception:
            pass

    while True:
        # Check elapsed time against hard timeout
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout_seconds:
            yield {
                "event": "timeout",
                "id": f"EVT-SEQ-{sequence}",
                "data": json.dumps({"error": "SSE streaming exceeded maximum timeout"}),
            }
            break

        # Check for new events
        events_len = len(ctx.events)
        for idx in range(events_len):
            if idx not in sent_indices:
                raw_evt = ctx.events[idx]
                sent_indices.add(idx)

                envelope = SSEEnvelope(
                    event_id=raw_evt.get("event_id", f"EVT-SEQ-{sequence}"),
                    sequence=sequence,
                    run_id=run_id,
                    timestamp=raw_evt.get("timestamp", 0.0),
                    event_type=raw_evt.get("event_type", "agent_transition"),
                    schema_version=START_SCHEMA_VERSION,
                    source_agent=raw_evt.get("source_agent", "Director"),
                    target_agent=raw_evt.get("target_agent", "DeterministicEngine"),
                    stage=raw_evt.get("stage", "PLANNING"),
                    action=raw_evt.get("action", ""),
                    status=raw_evt.get("status", "SUCCESS"),
                    latency_ms=raw_evt.get("latency_ms", 0.0),
                    evidence_refs=raw_evt.get("evidence_refs", []),
                    artifact_refs=raw_evt.get("artifact_refs", []),
                    policy_decision=raw_evt.get("policy_decision", "ALLOW"),
                    payload=raw_evt.get("metadata", {}),
                )

                yield {
                    "event": envelope.event_type,
                    "id": f"EVT-SEQ-{sequence}",
                    "data": json.dumps(envelope.model_dump(), default=str),
                }
                sequence += 1

        # Check if the run has completed or failed
        if ctx.status in ("COMPLETED", "FAILED"):
            # Ensure all remaining buffered events are yielded before closing
            if len(sent_indices) >= len(ctx.events):
                final_type = "complete" if ctx.status == "COMPLETED" else "error"
                yield {
                    "event": final_type,
                    "id": f"EVT-SEQ-{sequence}",
                    "data": json.dumps({
                        "run_id": run_id,
                        "status": ctx.status,
                        "error_message": ctx.error_message,
                        "event_count": len(ctx.events),
                        "evidence_count": len(ctx.evidence_records),
                    }),
                }
                break

        await asyncio.sleep(poll_interval)
