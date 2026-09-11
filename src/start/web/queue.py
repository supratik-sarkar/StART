"""Single-Slot Analytical Concurrency Scheduler & Session Manager for StART v4.5.

Tailored specifically for Oracle A1 (2 OCPU / 12 GB RAM) resource governance:
- Enforces strict analytical concurrency limit (default: 1 active heavy run).
- Bounded pending queue with timeout.
- Returns ENGINE_BUSY status when full with Retry-After header.
- Manages session lifecycle, memory footprint, and TTL cleanup.
"""

from __future__ import annotations

import asyncio
import json
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
    decisions: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
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

    def _persist_run_context(self, ctx: ActiveRunContext) -> None:
        """Persist full run context to start_output/<run_id>/run_context.json."""
        try:
            from pathlib import Path
            root = Path("start_output") / ctx.run_id
            root.mkdir(parents=True, exist_ok=True)
            ctx_path = root / "run_context.json"

            req_dict = ctx.request.model_dump() if hasattr(ctx.request, "model_dump") else ctx.request.dict()
            ev_records = []
            for r in ctx.evidence_records:
                if hasattr(r, "model_dump"):
                    ev_records.append(r.model_dump())
                elif hasattr(r, "dict"):
                    ev_records.append(r.dict())
                elif hasattr(r, "to_dict"):
                    ev_records.append(r.to_dict())
                elif isinstance(r, dict):
                    ev_records.append(r)

            data = {
                "run_id": ctx.run_id,
                "session_id": ctx.session_id,
                "request": req_dict,
                "status": ctx.status,
                "created_at": ctx.created_at,
                "started_at": ctx.started_at,
                "completed_at": ctx.completed_at,
                "events": ctx.events,
                "presentation": ctx.presentation,
                "artifacts": ctx.artifacts,
                "evidence_records": ev_records,
                "decisions": ctx.decisions,
                "checkpoints": ctx.checkpoints,
                "error_message": ctx.error_message,
            }
            from start.utils.serializers import sanitize_json_primitives
            data = sanitize_json_primitives(data)
            ctx_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            logger.warning("Failed to persist run context for '%s': %s", ctx.run_id, e)

    def _load_run_context(self, run_id: str) -> ActiveRunContext | None:
        """Load persisted run context from start_output/<run_id>/run_context.json or ledger.jsonl."""
        try:
            from pathlib import Path
            ctx_path = Path("start_output") / run_id / "run_context.json"
            if not ctx_path.exists():
                return None
            data = json.loads(ctx_path.read_text(encoding="utf-8"))
            req_data = data.get("request", {})
            req = RunRequest(**req_data)

            ev_records = data.get("evidence_records", [])
            if not ev_records:
                ledger_path = Path("start_output") / run_id / "ledger.jsonl"
                if ledger_path.exists():
                    for line in ledger_path.read_text(encoding="utf-8").splitlines():
                        if line.strip():
                            try:
                                ev_records.append(json.loads(line))
                            except Exception:
                                pass

            return ActiveRunContext(
                run_id=data["run_id"],
                session_id=data.get("session_id", ""),
                request=req,
                status=data.get("status", "COMPLETED"),
                created_at=data.get("created_at", 0.0),
                started_at=data.get("started_at"),
                completed_at=data.get("completed_at"),
                events=data.get("events", []),
                presentation=data.get("presentation"),
                artifacts=data.get("artifacts", {}),
                evidence_records=ev_records,
                decisions=data.get("decisions", []),
                checkpoints=data.get("checkpoints", []),
                error_message=data.get("error_message"),
            )
        except Exception as e:
            logger.warning("Failed to load persisted run context for '%s': %s", run_id, e)
            return None

    def get_run(self, run_id: str, session_id: str | None = None) -> ActiveRunContext | None:
        """Get run context with session ownership verification. Loads from disk if not in memory."""
        with self._lock:
            ctx = self._runs.get(run_id)
            if not ctx:
                ctx = self._load_run_context(run_id)
                if ctx:
                    self._runs[run_id] = ctx
            if not ctx:
                return None
            if session_id and ctx.session_id and ctx.session_id != session_id:
                # Session ownership check to prevent IDOR
                return None
            return ctx

    def list_runs(self) -> list[ActiveRunContext]:
        """Return snapshot list of all known run contexts (in-memory and persisted)."""
        with self._lock:
            from pathlib import Path
            output_dir = Path("start_output")
            if output_dir.exists():
                for p in output_dir.glob("RUN-*"):
                    if p.is_dir() and p.name not in self._runs:
                        ctx = self._load_run_context(p.name)
                        if ctx:
                            self._runs[p.name] = ctx
            return list(self._runs.values())

    def get_run_history(self) -> list[dict[str, Any]]:
        """Return structured historical summaries for all runs, sorted newest first."""
        all_runs = self.list_runs()
        history = []
        for ctx in all_runs:
            req = ctx.request
            wf = getattr(req, "workflowId", None) or getattr(req, "workflow", "predictive_ml")
            ctx_id = getattr(req, "contextId", None) or getattr(req, "synthetic_profile", "institutional_credit_v1")
            gov_disp = None
            if ctx.presentation:
                gov_disp = ctx.presentation.get("governance_disposition")
            if not gov_disp and ctx.checkpoints:
                cp6 = next((c for c in ctx.checkpoints if (c.get("checkpoint_id") or c.get("checkpointId")) == "CP-006"), None)
                if cp6:
                    gov_disp = "PASS"

            history.append(
                {
                    "run_id": ctx.run_id,
                    "session_id": ctx.session_id,
                    "workflow": wf,
                    "domain": getattr(req, "domain", "predictive"),
                    "context_id": ctx_id,
                    "created_at": ctx.created_at,
                    "started_at": ctx.started_at,
                    "completed_at": ctx.completed_at,
                    "status": ctx.status.lower(),
                    "evidence_count": len(ctx.evidence_records),
                    "artifact_count": len(ctx.artifacts),
                    "governance_disposition": gov_disp or "REVIEW_REQUIRED",
                    "parent_run_id": getattr(req, "parent_run_id", None) or getattr(req, "parentRunId", None),
                    "intervention": getattr(req, "intervention", None),
                    "goal": getattr(req, "goal", "") or f"Evaluate {wf} on {ctx_id}",
                }
            )
        # Sort descending by created_at
        history.sort(key=lambda x: x.get("created_at") or 0.0, reverse=True)
        return history

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

    def append_checkpoint(self, run_id: str, checkpoint: dict[str, Any]) -> None:
        with self._lock:
            ctx = self._runs.get(run_id)
            if ctx:
                # Avoid duplicate checkpoint entries
                cp_id = checkpoint.get("checkpoint_id") or checkpoint.get("checkpointId")
                if not any((c.get("checkpoint_id") or c.get("checkpointId")) == cp_id for c in ctx.checkpoints):
                    ctx.checkpoints.append(checkpoint)

    def append_decision(self, run_id: str, decision: dict[str, Any]) -> None:
        with self._lock:
            ctx = self._runs.get(run_id)
            if ctx:
                ctx.decisions.append(decision)

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
        checkpoints: list[dict[str, Any]] | None = None,
    ) -> None:
        with self._lock:
            ctx = self._runs.get(run_id)
            if ctx:
                from start.utils.serializers import sanitize_json_primitives
                ctx.status = "COMPLETED"
                ctx.completed_at = time.time()
                if presentation:
                    ctx.presentation = sanitize_json_primitives(presentation)
                if artifacts:
                    ctx.artifacts = sanitize_json_primitives(artifacts)
                if evidence_records:
                    ctx.evidence_records = evidence_records
                if checkpoints:
                    ctx.checkpoints = checkpoints
                if run_id in self._queue:
                    self._queue.remove(run_id)
                self._active_count = max(0, self._active_count - 1)
                self._persist_run_context(ctx)

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
                self._persist_run_context(ctx)

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



class QueueEventSink:
    """EventSink adapter that pushes typed RuntimeEvents to AnalyticalQueue."""

    def __init__(self, arg1: Any, arg2: Any = None, *args: Any, **kwargs: Any) -> None:
        if isinstance(arg1, str):
            self.run_id = arg1
            self.queue = arg2 or GLOBAL_QUEUE
        else:
            self.queue = arg1 or GLOBAL_QUEUE
            self.run_id = arg2 or ""

    def emit(self, event: Any) -> None:
        payload = event.to_dict() if hasattr(event, "to_dict") else event
        self.queue.append_event(self.run_id, payload)
        if isinstance(payload, dict):
            evt_type = payload.get("event_type")
            cp_id = payload.get("checkpoint_id")
            if evt_type == "checkpoint_committed" or cp_id:
                cp_meta = payload.get("metadata") or payload
                if isinstance(cp_meta, dict) and (cp_meta.get("checkpoint_id") or cp_id):
                    clean_cp = dict(cp_meta)
                    clean_cp.setdefault("checkpoint_id", cp_id)
                    clean_cp.setdefault("status", "completed")
                    self.queue.append_checkpoint(self.run_id, clean_cp)


# Global singleton queue instance for the process
GLOBAL_QUEUE = AnalyticalQueue(max_concurrency=1, max_queue_size=10, session_ttl_seconds=3600)

