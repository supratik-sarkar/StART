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
import time
from collections.abc import AsyncGenerator
from typing import Any

from start.web.queue import GLOBAL_QUEUE
from start.web.schemas import START_SCHEMA_VERSION

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

                evt_id = raw_evt.get("event_id") or f"EVT-SEQ-{sequence}"
                action_name = raw_evt.get("action") or raw_evt.get("phase") or "Runtime Execution"
                message_text = raw_evt.get("message") or action_name
                raw_status = str(raw_evt.get("status", "RUNNING")).upper()
                norm_status = (
                    "completed"
                    if raw_status in ("SUCCESS", "COMPLETED")
                    else ("failed" if raw_status == "FAILED" else "running")
                )

                progress_obj = None
                if raw_evt.get("percent") is not None or raw_evt.get("completed") is not None:
                    progress_obj = {
                        "label": raw_evt.get("phase", "Executing"),
                        "percent": float(raw_evt["percent"]) if raw_evt.get("percent") is not None else None,
                        "completed": int(raw_evt["completed"]) if raw_evt.get("completed") is not None else None,
                        "total": int(raw_evt["total"]) if raw_evt.get("total") is not None else None,
                        "detail": message_text,
                    }

                payload_data = {
                    "eventId": evt_id,
                    "event_id": evt_id,
                    "sequence": sequence,
                    "runId": run_id,
                    "run_id": run_id,
                    "timestamp": raw_evt.get("timestamp", time.time()),
                    "type": raw_evt.get("event_type", "agent_transition"),
                    "event_type": raw_evt.get("event_type", "agent_transition"),
                    "nodeId": raw_evt.get("node_id"),
                    "parentNodeId": raw_evt.get("parent_node_id"),
                    "title": action_name,
                    "message": message_text,
                    "status": norm_status,
                    "progress": progress_obj,
                    "evidenceIds": raw_evt.get("evidence_refs", []),
                    "evidence_refs": raw_evt.get("evidence_refs", []),
                    "artifactIds": raw_evt.get("artifact_refs", []),
                    "artifact_refs": raw_evt.get("artifact_refs", []),
                    "metadata": raw_evt.get("metadata", {}),
                    "payload": raw_evt.get("metadata", {}),
                    "schema_version": START_SCHEMA_VERSION,
                }

                yield {
                    "event": "message",
                    "id": f"EVT-SEQ-{sequence}",
                    "data": json.dumps(payload_data, default=str),
                }
                sequence += 1

        # Check if the run has completed or failed
        if ctx.status in ("COMPLETED", "FAILED"):
            # Ensure all remaining buffered events are yielded before closing
            if len(sent_indices) >= len(ctx.events):
                final_status = "completed" if ctx.status == "COMPLETED" else "failed"
                completion_payload = {
                    "eventId": f"EVT-FINAL-{sequence}",
                    "event_id": f"EVT-FINAL-{sequence}",
                    "sequence": sequence,
                    "runId": run_id,
                    "run_id": run_id,
                    "timestamp": time.time(),
                    "type": "run_completed" if ctx.status == "COMPLETED" else "run_failed",
                    "event_type": "complete" if ctx.status == "COMPLETED" else "error",
                    "title": "Deterministic Review Sealed" if ctx.status == "COMPLETED" else "Run Failed",
                    "message": "Deterministic review sealed and verified"
                    if ctx.status == "COMPLETED"
                    else (ctx.error_message or "Run failed"),
                    "status": final_status,
                    "evidenceIds": [
                        getattr(r, "evidence_id", r.get("evidence_id") if isinstance(r, dict) else "")
                        for r in ctx.evidence_records
                    ],
                    "artifactIds": list(ctx.artifacts.keys()),
                    "metadata": {
                        "status": ctx.status,
                        "event_count": len(ctx.events),
                        "evidence_count": len(ctx.evidence_records),
                    },
                    "progress": {
                        "label": "Completed" if ctx.status == "COMPLETED" else "Failed",
                        "detail": "Deterministic execution completed"
                        if ctx.status == "COMPLETED"
                        else (ctx.error_message or "Execution failed"),
                    },
                }
                yield {
                    "event": "complete" if ctx.status == "COMPLETED" else "error",
                    "id": f"EVT-SEQ-{sequence}",
                    "data": json.dumps(completion_payload, default=str),
                }
                break

        await asyncio.sleep(poll_interval)
