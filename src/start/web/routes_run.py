"""Run Lifecycle, SSE Streaming, Presentation & Logical Artifact Routes for StART v4.5."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Response
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse

from start.web.pdf import generate_institutional_pdf
from start.web.queue import GLOBAL_QUEUE
from start.web.schemas import (
    APIResponseEnvelope,
    ReviewPresentationExport,
    RunRequest,
    RunStatusResponse,
    START_SCHEMA_VERSION,
)
from start.web.security import sanitize_artifact_id, verify_turnstile_token
from start.web.sse import sse_event_generator

logger = logging.getLogger("start.web.routes_run")
router = APIRouter(prefix="/api/v1/runs", tags=["runs"])


def _execute_run_in_background(run_id: str, request: RunRequest) -> None:
    """Execute canonical StART deterministic review in an isolated background thread/task."""
    try:
        GLOBAL_QUEUE.mark_running(run_id)

        # Import canonical review executor and architecture models
        from start.review.executor import run_unified_review
        from start.review.architecture import (
            ReviewContextBundle,
            ReviewDomain,
            ReviewLifecycle,
            ReviewMode,
            ReviewGroundingMode,
            LLMReviewConfig,
        )

        # Map domain
        if request.domain == "predictive":
            domains = (ReviewDomain.PREDICTIVE,)
        elif request.domain == "deep_learning":
            domains = (ReviewDomain.PREDICTIVE,)
        else:
            domains = (ReviewDomain.MARKET,)

        bundle = ReviewContextBundle(
            mode=ReviewMode.SINGLE_DOMAIN,
            domains=domains,
            materiality="high",
            lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
            llm_config=LLMReviewConfig(provider="none"),
            grounding_mode=ReviewGroundingMode.STRUCTURED,
        )

        # Emit initial planning event
        GLOBAL_QUEUE.append_event(
            run_id,
            {
                "event_id": f"EVT-{uuid.uuid4().hex[:8]}",
                "timestamp": time.time(),
                "event_type": "agent_transition",
                "source_agent": "Director",
                "target_agent": "Specialist",
                "stage": "DISCOVERY",
                "action": f"initialize_{request.domain}_review",
                "status": "RUNNING",
                "metadata": {"synthetic_profile": request.synthetic_profile, "seed": request.seed},
            },
        )

        # Run canonical StART review
        res = run_unified_review(
            bundle=bundle,
            interactive=False,
        )

        records = res.get("records", [])
        presentation_model = res.get("presentation_model")
        events = res.get("orchestration_events", [])
        artifacts_dict = res.get("artifacts", {})

        # Transfer execution events
        for evt in events:
            if isinstance(evt, dict):
                GLOBAL_QUEUE.append_event(run_id, evt)
            elif hasattr(evt, "to_dict"):
                GLOBAL_QUEUE.append_event(run_id, evt.to_dict())

        # Serialize presentation model
        pres_dict = presentation_model.to_dict() if presentation_model and hasattr(presentation_model, "to_dict") else {}
        if not pres_dict:
            pres_dict = {
                "run_id": run_id,
                "mode": request.mode,
                "domains": [request.domain],
                "materiality": request.materiality,
                "lifecycle": request.lifecycle,
                "governance_disposition": res.get("governance_disposition", "ACCEPT"),
                "attestation_seal_merkle_root": res.get("merkle_root", ""),
                "blocks": {},
                "orchestration_events": [e if isinstance(e, dict) else e.to_dict() for e in events],
            }
        else:
            pres_dict["run_id"] = run_id

        GLOBAL_QUEUE.mark_completed(
            run_id=run_id,
            presentation=pres_dict,
            artifacts=artifacts_dict,
            evidence_records=records,
        )
        logger.info("Run '%s' completed successfully with %d evidence records", run_id, len(records))

    except Exception as exc:
        logger.exception("Run '%s' failed: %s", run_id, exc)
        GLOBAL_QUEUE.mark_failed(run_id, str(exc))


@router.post("/start", response_model=APIResponseEnvelope)
async def start_run(
    request: RunRequest,
    x_forwarded_for: str | None = Header(None),
) -> APIResponseEnvelope:
    """Submit a deterministic analytical review run."""
    # 1. Server-side Turnstile verification
    if not verify_turnstile_token(request.turnstile_token, remote_ip=x_forwarded_for):
        raise HTTPException(status_code=400, detail="Turnstile challenge verification failed")

    # 2. Assign unique run ID
    run_id = f"RUN-WEB-{uuid.uuid4().hex[:10]}"

    # 3. Submit to analytical queue
    accepted, status_msg = GLOBAL_QUEUE.submit_run(run_id, request)
    if not accepted:
        return APIResponseEnvelope(
            success=False,
            run_id=run_id,
            error=status_msg,
            data={"engine_status": "BUSY", "retry_after_seconds": 15},
        )

    # 4. Launch background execution task
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _execute_run_in_background, run_id, request)

    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data={
            "session_id": request.session_id,
            "status": "QUEUED",
            "domain": request.domain,
            "synthetic_profile": request.synthetic_profile,
        },
    )


@router.get("/{run_id}/status", response_model=APIResponseEnvelope)
def get_run_status(
    run_id: str,
    session_id: str | None = Query(None),
) -> APIResponseEnvelope:
    """Query current run status and metrics."""
    status_resp = GLOBAL_QUEUE.get_status(run_id, session_id)
    if not status_resp:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or session access denied")

    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data=status_resp.model_dump(),
    )


@router.get("/{run_id}/stream")
async def stream_run_events(
    run_id: str,
    session_id: str | None = Query(None),
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
) -> EventSourceResponse:
    """Subscribe to Server-Sent Events for a run."""
    gen = sse_event_generator(run_id=run_id, session_id=session_id, last_event_id=last_event_id)
    return EventSourceResponse(gen)


@router.get("/{run_id}/presentation", response_model=APIResponseEnvelope)
def get_run_presentation(
    run_id: str,
    session_id: str | None = Query(None),
) -> APIResponseEnvelope:
    """Retrieve structured ReviewPresentationModel for UI rendering."""
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or session access denied")

    if ctx.status not in ("COMPLETED", "FAILED"):
        return APIResponseEnvelope(
            success=True,
            run_id=run_id,
            data={"status": ctx.status, "presentation": None},
        )

    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data={
            "status": ctx.status,
            "presentation": ctx.presentation or {},
            "schema_version": START_SCHEMA_VERSION,
        },
    )


@router.get("/{run_id}/artifacts/{artifact_id}")
def get_run_artifact(
    run_id: str,
    artifact_id: str,
    session_id: str | None = Query(None),
) -> Response:
    """Retrieve a logical visualization or tabular artifact."""
    clean_artifact_id = sanitize_artifact_id(artifact_id)
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or session access denied")

    # Resolve artifact from context
    art = ctx.artifacts.get(clean_artifact_id)
    if not art:
        # Check if artifact matches by title or key in presentation
        raise HTTPException(status_code=404, detail=f"Artifact '{clean_artifact_id}' not found for run '{run_id}'")

    content = art.get("content", "")
    art_type = art.get("artifact_type", "svg")

    if art_type == "svg":
        return Response(content=content, media_type="image/svg+xml")
    elif art_type == "json":
        return JSONResponse(content=content if isinstance(content, dict) else json.loads(content))
    elif art_type == "html":
        # Sandboxed HTML: prevent same-origin script execution
        headers = {
            "Content-Security-Policy": "sandbox allow-scripts; default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline';",
            "X-Content-Type-Options": "nosniff",
        }
        return Response(content=content, media_type="text/html", headers=headers)

    return Response(content=str(content), media_type="text/plain")


@router.get("/{run_id}/pdf")
def get_run_pdf(
    run_id: str,
    report_type: str = Query("executive"),
    session_id: str | None = Query(None),
) -> Response:
    """Generate and return institutional review PDF."""
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or session access denied")

    pres = ctx.presentation or {"run_id": run_id, "domains": [ctx.request.domain], "blocks": {}}
    pdf_bytes = generate_institutional_pdf(pres, report_type=report_type)

    headers = {
        "Content-Disposition": f'attachment; filename="StART_Report_{run_id}.pdf"',
        "X-Content-Type-Options": "nosniff",
    }
    return Response(content=pdf_bytes, media_type="application/pdf", headers=headers)
