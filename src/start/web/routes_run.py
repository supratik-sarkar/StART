"""Run Lifecycle, SSE Streaming, Presentation & Logical Artifact Routes for StART v5.1.0."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Header, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from start.runtime import (
    CanonicalExecutionService,
    resolve_context_spec,
    resolve_workflow,
)
from start.web.pdf import generate_institutional_pdf
from start.web.queue import GLOBAL_QUEUE, QueueEventSink
from start.web.schemas import (
    START_SCHEMA_VERSION,
    APIResponseEnvelope,
    RunRequest,
)
from start.web.security import sanitize_artifact_id, verify_turnstile_token
from start.web.sse import sse_event_generator

logger = logging.getLogger("start.web.routes_run")
router = APIRouter(prefix="/api/v1/runs", tags=["runs"])


def _execute_run_in_background(run_id: str, request: RunRequest) -> None:
    """Execute canonical StART deterministic review via shared non-web execution service."""
    try:
        GLOBAL_QUEUE.mark_running(run_id)

        workflow_id = getattr(request, "workflowId", None) or request.workflow
        if not workflow_id:
            if request.domain == "deep_learning":
                workflow_id = "deep_learning"
            elif request.domain == "market":
                workflow_id = "quantitative_finance"
            else:
                workflow_id = "predictive_ml"

        context_id = getattr(request, "contextId", None) or request.synthetic_profile
        if not context_id:
            if workflow_id == "deep_learning":
                context_id = "deep_learning_v1"
            elif workflow_id == "quantitative_finance":
                context_id = "institutional_market_v1"
            else:
                context_id = "institutional_credit_v1"

        seed = request.seed if request.seed is not None else 42
        sink = QueueEventSink(run_id, GLOBAL_QUEUE)

        res = CanonicalExecutionService.execute(
            workflow_id=workflow_id,
            context_id=context_id,
            request_params=request.parameters,
            seed=seed,
            event_sink=sink,
            materiality=request.materiality,
            run_id=run_id,
            parent_run_id=request.parent_run_id,
            intervention=request.intervention,
        )

        pres_dict = (
            res.presentation_model.to_dict()
            if res.presentation_model and hasattr(res.presentation_model, "to_dict")
            else (res.presentation_model or {})
        )
        if not isinstance(pres_dict, dict):
            pres_dict = {}

        pres_dict["run_id"] = run_id
        pres_dict["workflow"] = workflow_id
        pres_dict["parent_run_id"] = request.parent_run_id
        pres_dict["intervention"] = request.intervention
        pres_dict["governance_disposition"] = res.governance_disposition
        pres_dict["attestation_seal_merkle_root"] = res.merkle_root

        GLOBAL_QUEUE.mark_completed(
            run_id=run_id,
            presentation=pres_dict,
            artifacts=res.artifacts,
            evidence_records=res.records,
        )
        logger.info("Run '%s' completed successfully with %d evidence records", run_id, len(res.records))

    except Exception as exc:
        logger.exception("Run '%s' failed: %s", run_id, exc)
        GLOBAL_QUEUE.mark_failed(run_id, str(exc))


@router.post("", response_model=APIResponseEnvelope)
@router.post("/start", response_model=APIResponseEnvelope)
async def start_run(
    request: RunRequest,
    x_forwarded_for: str | None = Header(None),
) -> Response:
    """Submit a deterministic analytical review run.

    Validation Order (Amendments 27 & 28):
    1. Turnstile verification
    2. Schema validation (Pydantic RunRequest)
    3. Workflow / Context semantic and compatibility validation
    4. Queue submission (GLOBAL_QUEUE.submit_run)
    """
    # 1. Server-side Turnstile verification
    if not verify_turnstile_token(request.turnstile_token, remote_ip=x_forwarded_for):
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": (
                    "Turnstile challenge verification failed. Please complete the verification challenge."
                ),
                "error_code": "TURNSTILE_FAILED",
                "data": {},
            },
        )

    # 2. Resolve target workflow_id and context_id
    workflow_id = getattr(request, "workflowId", None) or request.workflow
    if not workflow_id:
        if request.domain == "deep_learning":
            workflow_id = "deep_learning"
        elif request.domain == "market":
            workflow_id = "quantitative_finance"
        else:
            workflow_id = "predictive_ml"

    context_id = getattr(request, "contextId", None) or request.synthetic_profile
    if not context_id:
        if workflow_id == "deep_learning":
            context_id = "deep_learning_v1"
        elif workflow_id == "quantitative_finance":
            context_id = "institutional_market_v1"
        else:
            context_id = "institutional_credit_v1"

    # 3. Fail-closed semantic validation BEFORE queue submission (Amendment 27 & 28)
    try:
        resolve_context_spec(context_id)
    except ValueError as exc:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": str(exc),
                "error_code": "UNKNOWN_CONTEXT",
                "data": {"context_id": context_id},
            },
        )

    try:
        resolved = resolve_workflow(workflow_id, context_id)
    except ValueError as exc:
        err_msg = str(exc)
        code = "UNKNOWN_WORKFLOW" if "Unknown workflow" in err_msg else "INCOMPATIBLE_CONTEXT"
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": err_msg,
                "error_code": code,
                "data": {"workflow_id": workflow_id, "context_id": context_id},
            },
        )

    # Normalize fields on request for downstream consumers
    request.workflow = workflow_id
    request.synthetic_profile = context_id

    # 4. Assign unique run ID
    run_id = f"RUN-WEB-{uuid.uuid4().hex[:10]}"

    # 5. Submit to analytical queue
    accepted, status_msg = GLOBAL_QUEUE.submit_run(run_id, request)
    if not accepted:
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "run_id": run_id,
                "error": status_msg,
                "error_code": "ENGINE_BUSY",
                "data": {"engine_status": "BUSY", "retry_after_seconds": 15},
            },
        )

    # 6. Launch background execution task
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _execute_run_in_background, run_id, request)

    ctx = GLOBAL_QUEUE.get_run(run_id)
    from start.web.routes_workbench import serialize_run_snapshot

    snapshot = serialize_run_snapshot(ctx) if ctx else {}

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "schema_version": START_SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": time.time(),
            "data": {
                **snapshot,
                "run": snapshot,
                "run_id": run_id,
                "session_id": request.session_id,
                "status": "QUEUED",
                "domain": request.domain,
                "workflow": request.workflow,
                "synthetic_profile": request.synthetic_profile,
            },
        },
    )


@router.get("/{run_id}", response_model=APIResponseEnvelope)
@router.get("/{run_id}/status", response_model=APIResponseEnvelope)
def get_run_status(
    run_id: str,
    session_id: str | None = Query(None),
) -> APIResponseEnvelope:
    """Query current run status and metrics."""
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or session access denied")

    status_resp = GLOBAL_QUEUE.get_status(run_id, session_id)
    from start.web.routes_workbench import serialize_run_snapshot

    snapshot = serialize_run_snapshot(ctx)
    data = status_resp.model_dump() if status_resp else {}
    data.update(snapshot)
    data["run"] = snapshot

    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data=data,
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


@router.get("/{run_id}/events")
async def get_run_events(
    request: Request,
    run_id: str,
    session_id: str | None = Query(None),
    last_event_id: str | None = Header(None, alias="Last-Event-ID"),
):
    """Retrieve event stream (SSE) or raw event list for a run."""
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        gen = sse_event_generator(run_id=run_id, session_id=session_id, last_event_id=last_event_id)
        return EventSourceResponse(gen)

    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or session access denied")

    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data={"events": ctx.events, "count": len(ctx.events)},
    )


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


@router.get("/{run_id}/evidence", response_model=APIResponseEnvelope)
def get_run_evidence(
    run_id: str,
    session_id: str | None = Query(None),
) -> APIResponseEnvelope:
    """Retrieve raw EvidenceRecord list for run."""
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or session access denied")

    records = [r.to_dict() if hasattr(r, "to_dict") else r for r in ctx.evidence_records]
    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data={"evidence_records": records, "evidence": records, "count": len(records)},
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

    art = ctx.artifacts.get(clean_artifact_id)
    if not art:
        raise HTTPException(
            status_code=404,
            detail=f"Artifact '{clean_artifact_id}' not found for run '{run_id}'",
        )

    content = art.get("content", "")
    art_type = art.get("artifact_type", "svg")

    if art_type == "svg":
        return Response(content=content, media_type="image/svg+xml")
    elif art_type == "json":
        return JSONResponse(content=content if isinstance(content, dict) else json.loads(content))
    elif art_type == "html":
        csp_val = (
            "sandbox allow-scripts; default-src 'none'; "
            "style-src 'unsafe-inline'; script-src 'unsafe-inline';"
        )
        headers = {
            "Content-Security-Policy": csp_val,
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
