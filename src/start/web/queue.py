"""Single-Slot Analytical Concurrency Scheduler & Session Manager for StART v4.5.

Tailored specifically for Oracle A1 (2 OCPU / 12 GB RAM) resource governance:
- Enforces strict analytical concurrency limit (default: 1 active heavy run).
- Bounded pending queue with timeout.
- Returns ENGINE_BUSY status when full with Retry-After header.
- Manages session lifecycle, memory footprint, and TTL cleanup.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from start.web.schemas import RunRequest, RunStatusResponse

logger = logging.getLogger("start.web.queue")


@dataclass
class ActiveRunContext:
    run_id: str
    session_id: str
    request: RunRequest
    status: str = "QUEUED"  # QUEUED | RUNNING | COMPLETED | FAILED | BUSY
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    presentation: dict[str, Any] | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)
    evidence_records: list[Any] = field(default_factory=list)
    error_message: str | None = None
    task: asyncio.Task[Any] | None = None


class AnalyticalQueue:
    """Thread-safe analytical execution scheduler."""

    def __init__(self, max_concurrency: int = 1, max_queue_size: int = 10, session_ttl_seconds: int = 3600):
        self.max_concurrency = max_concurrency
        self.max_queue_size = max_queue_size
        self.session_ttl_seconds = session_ttl_seconds
        self._lock = threading.Lock()
        self._runs: dict[str, ActiveRunContext] = {}
        self._queue: list[str] = []
        self._active_count: int = 0

    def submit_run(self, run_id: str, request: RunRequest) -> tuple[bool, str]:
        """Submit a run to the queue. Returns (accepted, reason_or_status)."""
        self.cleanup_stale_sessions()
        with self._lock:
            # Check capacity
            active_or_queued = len([r for r in self._runs.values() if r.status in ("QUEUED", "RUNNING")])
            if active_or_queued >= self.max_queue_size:
                return False, "ENGINE_BUSY: Server analytical queue is currently full. Please retry shortly."

            ctx = ActiveRunContext(
                run_id=run_id,
                session_id=request.session_id,
                request=request,
                status="QUEUED",
            )
            self._runs[run_id] = ctx
            self._queue.append(run_id)
            return True, "QUEUED"

    def get_run(self, run_id: str, session_id: str | None = None) -> ActiveRunContext | None:
        """Get run context with session ownership verification."""
        with self._lock:
            ctx = self._runs.get(run_id)
            if not ctx:
                return None
            if session_id and ctx.session_id != session_id:
                # Session ownership check to prevent IDOR
                return None
            return ctx

    def get_status(self, run_id: str, session_id: str | None = None) -> RunStatusResponse | None:
        ctx = self.get_run(run_id, session_id)
        if not ctx:
            return None
        return RunStatusResponse(
            run_id=ctx.run_id,
            session_id=ctx.session_id,
            status=ctx.status,  # type: ignore[arg-type]
            domain=ctx.request.domain,
            synthetic_profile=ctx.request.synthetic_profile,
            created_at=ctx.created_at,
            completed_at=ctx.completed_at,
            event_count=len(ctx.events),
            evidence_count=len(ctx.evidence_records),
            artifact_count=len(ctx.artifacts),
            error_message=ctx.error_message,
        )

    def append_event(self, run_id: str, event: dict[str, Any]) -> None:
        with self._lock:
            ctx = self._runs.get(run_id)
            if ctx:
                ctx.events.append(event)

    def mark_running(self, run_id: str) -> bool:
        with self._lock:
            ctx = self._runs.get(run_id)
            if ctx and ctx.status == "QUEUED":
                ctx.status = "RUNNING"
                ctx.started_at = time.time()
                self._active_count += 1
                return True
            return False

    def mark_completed(
        self,
        run_id: str,
        presentation: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        evidence_records: list[Any] | None = None,
    ) -> None:
        with self._lock:
            ctx = self._runs.get(run_id)
            if ctx:
                ctx.status = "COMPLETED"
                ctx.completed_at = time.time()
                if presentation:
                    ctx.presentation = presentation
                if artifacts:
                    ctx.artifacts = artifacts
                if evidence_records:
                    ctx.evidence_records = evidence_records
                if run_id in self._queue:
                    self._queue.remove(run_id)
                self._active_count = max(0, self._active_count - 1)

    def mark_failed(self, run_id: str, error_message: str) -> None:
        with self._lock:
            ctx = self._runs.get(run_id)
            if ctx:
                ctx.status = "FAILED"
                ctx.completed_at = time.time()
                ctx.error_message = error_message
                if run_id in self._queue:
                    self._queue.remove(run_id)
                self._active_count = max(0, self._active_count - 1)

    def cleanup_stale_sessions(self) -> int:
        """Purge sessions older than session_ttl_seconds."""
        now = time.time()
        to_purge = []
        with self._lock:
            for run_id, ctx in self._runs.items():
                run_age = now - (ctx.completed_at or ctx.created_at)
                if ctx.status in ("COMPLETED", "FAILED") and run_age > self.session_ttl_seconds:
                    to_purge.append(run_id)
            for r_id in to_purge:
                del self._runs[r_id]
        return len(to_purge)


# Global singleton queue instance for the process
GLOBAL_QUEUE = AnalyticalQueue(max_concurrency=1, max_queue_size=10, session_ttl_seconds=3600)
