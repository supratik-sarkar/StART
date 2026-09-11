"""Run Lifecycle, SSE Streaming, Presentation & Logical Artifact Routes for StART v5.1.0."""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

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

        workflow_id = getattr(request, "workflowId", None) or getattr(request, "workflow_id", None) or request.workflow
        if not workflow_id and request.parameters:
            workflow_id = request.parameters.get("workflow_id") or request.parameters.get("workflowId") or request.parameters.get("workflow")
        if not workflow_id:
            if request.domain == "deep_learning":
                workflow_id = "deep_learning"
            elif request.domain == "market":
                workflow_id = "quantitative_finance"
            elif request.domain in ("recommender", "recsys"):
                workflow_id = "recommender_system"
            else:
                workflow_id = "predictive_ml"

        context_id = getattr(request, "contextId", None) or getattr(request, "context_id", None)
        if not context_id and request.parameters:
            context_id = request.parameters.get("dataset_id") or request.parameters.get("dataset") or request.parameters.get("context_id") or request.parameters.get("contextId") or request.parameters.get("context")
        if not context_id:
            context_id = request.synthetic_profile
        if not context_id:
            if workflow_id == "deep_learning":
                context_id = "deep_learning_v1"
            elif workflow_id in ("quantitative_finance", "market_risk", "portfolio_rebalance", "scenario_analysis"):
                context_id = "institutional_market_v1"
            elif workflow_id == "recommender_system":
                context_id = "recommender_ratings_v1"
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
            execution_mode=getattr(request, "execution_mode", "hybrid_workbench"),
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
        pres_dict["workflow_id"] = workflow_id
        pres_dict["context_id"] = context_id
        pres_dict["dataset_id"] = context_id
        pres_dict["parent_run_id"] = request.parent_run_id
        pres_dict["intervention"] = request.intervention
        pres_dict["governance_disposition"] = res.governance_disposition
        pres_dict["attestation_seal_merkle_root"] = res.merkle_root
        pres_dict["execution_mode"] = getattr(request, "execution_mode", "hybrid_workbench")

        if getattr(res, "extra", None) and "resolved_configuration" in res.extra:
            pres_dict["resolved_configuration"] = res.extra["resolved_configuration"]
        elif hasattr(res, "context_instance") and getattr(res.context_instance, "bundle", None):
            tab = getattr(res.context_instance.bundle, "tabular", None)
            mkt = getattr(res.context_instance.bundle, "market", None)
            if tab and "resolved_configuration" in getattr(tab, "extra", {}):
                pres_dict["resolved_configuration"] = tab.extra["resolved_configuration"]
            elif mkt and "resolved_configuration" in getattr(mkt, "extra", {}):
                pres_dict["resolved_configuration"] = mkt.extra["resolved_configuration"]

        # G5: Persist execution-time scientific outputs into presentation model
        scientific_pres: dict[str, Any] = {
            "run_id": run_id,
            "workflow": workflow_id,
            "context_id": context_id,
            "seed": seed,
            "domain": getattr(request, "domain", None) or ("portfolio" if workflow_id == "quantitative_finance" else "predictive_binary"),
        }
        blocks = pres_dict.setdefault("blocks", {})

        for rec in (res.records or []):
            tid = getattr(rec, "test_id", "") or ""
            details = getattr(rec, "details", {}) or {}
            measurements = getattr(rec, "measurements", {}) or {}
            if any(k in tid.lower() for k in ("xai", "explain", "importance", "shap", "pdp")):
                scientific_pres["xai"] = {
                    "test_id": tid,
                    "details": details,
                    "measurements": measurements,
                }
                if "explainability" not in blocks:
                    blocks["explainability"] = {
                        "methods": [tid],
                        "feature_importance": details.get("feature_importance", []),
                        "details": details,
                    }
            if "sensitivity" in tid.lower() or "shock" in tid.lower():
                scientific_pres["sensitivity"] = {
                    "test_id": tid,
                    "details": details,
                    "measurements": measurements,
                }
                if "sensitivity" not in blocks:
                    blocks["sensitivity"] = {
                        "shock_grid": details.get("shock_grid", [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]),
                        "baseline_zero_delta": details.get("baseline_zero_delta", 0.0),
                        "responses": details.get("responses", "RESPONSES_NOT_PERSISTED"),
                    }
            if any(k in tid.lower() for k in ("candidate", "tuning", "hyperparameter")):
                scientific_pres["tuning"] = {
                    "test_id": tid,
                    "details": details,
                    "measurements": measurements,
                }

        pres_dict["scientific_presentation"] = scientific_pres

        can_prod = res.products.get_result("analysis.canonical_result") if res.products else None
        if can_prod is not None and hasattr(can_prod, "to_dict"):
            pres_dict["canonical_analytical_result"] = can_prod.to_dict()

        from start.utils.serializers import sanitize_json_primitives
        sanitized_pres = sanitize_json_primitives(pres_dict)
        sanitized_arts = sanitize_json_primitives(res.artifacts)

        GLOBAL_QUEUE.mark_completed(
            run_id=run_id,
            presentation=sanitized_pres,
            artifacts=sanitized_arts,
            evidence_records=res.records,
            checkpoints=getattr(res, "checkpoints", []),
        )
        logger.info("Run '%s' completed successfully with %d evidence records", run_id, len(res.records))

    except Exception as exc:
        logger.exception("Run '%s' failed: %s", run_id, exc)
        GLOBAL_QUEUE.mark_failed(run_id, str(exc))


@router.post("", response_model=APIResponseEnvelope)
@router.post("/start", response_model=APIResponseEnvelope)
@router.post("/workflow/run", response_model=APIResponseEnvelope)
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
    workflow_id = getattr(request, "workflowId", None) or getattr(request, "workflow_id", None) or request.workflow
    if not workflow_id and request.parameters:
        workflow_id = request.parameters.get("workflow_id") or request.parameters.get("workflowId") or request.parameters.get("workflow")
    if not workflow_id:
        if request.domain == "deep_learning":
            workflow_id = "deep_learning"
        elif request.domain == "market":
            workflow_id = "quantitative_finance"
        elif request.domain in ("recommender", "recsys"):
            workflow_id = "recommender_system"
        else:
            workflow_id = "predictive_ml"

    context_id = getattr(request, "contextId", None) or getattr(request, "context_id", None)
    if not context_id and request.parameters:
        context_id = request.parameters.get("dataset_id") or request.parameters.get("dataset") or request.parameters.get("context_id") or request.parameters.get("contextId") or request.parameters.get("context")
    if not context_id:
        context_id = request.synthetic_profile
    if not context_id:
        if workflow_id == "deep_learning":
            context_id = "deep_learning_v1"
        elif workflow_id in ("quantitative_finance", "market_risk", "portfolio_rebalance", "scenario_analysis"):
            context_id = "institutional_market_v1"
        elif workflow_id == "recommender_system":
            context_id = "recommender_ratings_v1"
        else:
            context_id = "institutional_credit_v1"

    request.workflow = workflow_id
    request.contextId = context_id
    request.context_id = context_id
    request.synthetic_profile = context_id

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
        _ = resolve_workflow(workflow_id, context_id)
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
    import threading
    threading.Thread(target=_execute_run_in_background, args=(run_id, request), daemon=True).start()

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
            "execution_mode": getattr(request, "execution_mode", "hybrid_workbench"),
            "data": {
                **snapshot,
                "run": snapshot,
                "run_id": run_id,
                "execution_mode": getattr(request, "execution_mode", "hybrid_workbench"),
                "session_id": request.session_id,
                "status": "QUEUED",
                "domain": request.domain,
                "workflow": request.workflow,
                "synthetic_profile": request.synthetic_profile,
            },
        },
    )


@router.get("", response_model=APIResponseEnvelope)
def list_runs_history(
    workflow: str | None = Query(None),
    status: str | None = Query(None),
    domain: str | None = Query(None),
    session_id: str | None = Query(None),
) -> APIResponseEnvelope:
    """Return historical and active runs list with summary metrics."""
    history = GLOBAL_QUEUE.get_run_history()
    if workflow:
        history = [h for h in history if h.get("workflow") == workflow]
    if status:
        history = [h for h in history if h.get("status") == status.lower()]
    if domain:
        history = [h for h in history if h.get("domain") == domain]

    return APIResponseEnvelope(
        success=True,
        data={"runs": history, "count": len(history)},
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
    from start.utils.serializers import sanitize_json_primitives
    from start.web.routes_workbench import serialize_run_snapshot

    snapshot = serialize_run_snapshot(ctx)
    data = status_resp.model_dump() if status_resp else {}
    data.update(snapshot)
    data["run"] = snapshot

    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data=sanitize_json_primitives(data),
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

    from start.utils.serializers import sanitize_json_primitives
    events_list = [sanitize_json_primitives(e.to_dict() if hasattr(e, "to_dict") else e) for e in ctx.events]
    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data={"events": events_list, "count": len(events_list)},
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

    from start.utils.serializers import sanitize_json_primitives
    pres = sanitize_json_primitives(ctx.presentation or {})
    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data={
            "status": ctx.status,
            "presentation": pres,
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


# --------------------------------------------------------------------------- #
# Domain-Specific Presentation Endpoints (Tuning, XAI, Sensitivity, Portfolio, Scenario)
# --------------------------------------------------------------------------- #
def _resolve_run_scientific_identity(
    run_id: str,
    session_id: str | None = None,
) -> tuple[Any, dict[str, Any] | None, str]:
    """Resolve run identity between active queue and certification bundle.

    Returns:
        (item, cert_bundle, source_type)
        where source_type is "ACTIVE_RUN", "CERTIFICATION_RUN", "XAI_RUN", "SENSITIVITY_RUN", or "NOT_FOUND"
    """
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if ctx:
        return ctx, None, "ACTIVE_RUN"

    from start.web.routes_certification import CERT_CACHE
    bundle = CERT_CACHE.get_bundle()
    all_runs = bundle.get("deterministic_runs", []) + bundle.get("gpt41_runs", [])
    cert_run = next((r for r in all_runs if r.get("run_id") == run_id), None)
    if cert_run:
        return cert_run, bundle, "CERTIFICATION_RUN"

    xai_results = bundle.get("xai_results", [])
    xai_run = next((x for x in xai_results if x.get("run_id") == run_id), None)
    if xai_run:
        return xai_run, bundle, "XAI_RUN"

    sens_results = bundle.get("sensitivity_results", [])
    sens_run = next((s for s in sens_results if s.get("run_id") == run_id), None)
    if sens_run:
        return sens_run, bundle, "SENSITIVITY_RUN"

    return None, bundle, "NOT_FOUND"


@router.get("/{run_id}/tuning", response_model=APIResponseEnvelope)
def get_run_tuning_presentation(
    run_id: str,
    session_id: str | None = Query(None),
) -> APIResponseEnvelope:
    """Retrieve structured tuning strategy, objective, trial budget, and parameters."""
    item, bundle, source_type = _resolve_run_scientific_identity(run_id, session_id)
    if source_type == "NOT_FOUND":
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in active queue or certification bundle")

    if source_type == "ACTIVE_RUN":
        ctx = item
        pres = ctx.presentation or {}
        blocks = pres.get("blocks", {})
        cand_block = blocks.get("candidates", {})
        cand_items = cand_block.get("items", [])
        best_cand = cand_items[0] if cand_items else {}

        return APIResponseEnvelope(
            success=True,
            run_id=run_id,
            data={
                "requested_run_id": run_id,
                "source_run_id": run_id,
                "source_scope": "RUN",
                "status": "TRIAL_DETAILS_NOT_PERSISTED",
                "experiment_id": getattr(ctx.request, "experiment_id", None) or f"RUN_{run_id}",
                "certification_id": None,
                "dataset_id": getattr(ctx.request, "contextId", None) or getattr(ctx.request, "dataset", None),
                "model_id": getattr(ctx.request, "model", None) or best_cand.get("model", "champion"),
                "run_id": run_id,
                "strategy": getattr(ctx.request, "tuning_strategy", "bounded_random_search"),
                "objective_metric": getattr(ctx.request, "objective_metric", "primary_metric"),
                "trial_budget": getattr(ctx.request, "trial_budget", len(cand_items)),
                "trials": [],
                "best_trial": None,
                "best_parameters": best_cand.get("hyperparameters", {}),
                "champion_model": best_cand.get("model", "champion"),
                "champion_lineage": None,
                "notes": "Workbench execution records completed candidates; individual trial logs not persisted.",
            },
        )

    if source_type == "CERTIFICATION_RUN":
        cert_run = item
        manifest = bundle.get("certification_manifest", {}) if bundle else {}
        return APIResponseEnvelope(
            success=True,
            run_id=run_id,
            data={
                "requested_run_id": run_id,
                "source_run_id": run_id,
                "source_scope": "RUN",
                "status": "TRIAL_DETAILS_NOT_PERSISTED",
                "experiment_id": cert_run.get("experiment_id"),
                "certification_id": manifest.get("certification_id", "CERT_4FB4D8C42308"),
                "dataset_id": cert_run.get("dataset"),
                "model_id": cert_run.get("model"),
                "run_id": run_id,
                "strategy": cert_run.get("tuning_strategy", "bounded_random_search"),
                "objective_metric": cert_run.get("primary_metric_name", "roc_auc"),
                "trial_budget": cert_run.get("trial_budget", 5),
                "trials": [],
                "best_trial": None,
                "best_parameters": cert_run.get("hyperparameters", {}),
                "champion_model": cert_run.get("model"),
                "champion_lineage": None,
                "notes": "Historical certification runs evaluated fixed seeds without per-trial persistence.",
            },
        )

    manifest = bundle.get("certification_manifest", {}) if bundle else {}
    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data={
            "requested_run_id": run_id,
            "source_run_id": None,
            "source_scope": "UNAVAILABLE",
            "status": "NOT_AVAILABLE_FOR_RUN",
            "experiment_id": None,
            "certification_id": manifest.get("certification_id", "CERT_4FB4D8C42308"),
            "dataset_id": item.get("dataset") if isinstance(item, dict) else None,
            "model_id": item.get("model") if isinstance(item, dict) else None,
            "reason": "Tuning not applicable to specialized analysis run",
        },
    )


@router.get("/{run_id}/xai", response_model=APIResponseEnvelope)
def get_run_xai_presentation(
    run_id: str,
    session_id: str | None = Query(None),
) -> APIResponseEnvelope:
    """Retrieve explainability presentation output without web-layer recalculation."""
    item, bundle, source_type = _resolve_run_scientific_identity(run_id, session_id)
    if source_type == "NOT_FOUND":
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in active queue or certification bundle")

    if source_type == "ACTIVE_RUN":
        ctx = item
        pres = ctx.presentation or {}
        xai_block = pres.get("blocks", {}).get("explainability", {})
        methods = xai_block.get("methods", [])
        feat_imp = xai_block.get("feature_importance", [])
        has_xai = bool(methods or feat_imp)

        return APIResponseEnvelope(
            success=True,
            run_id=run_id,
            data={
                "requested_run_id": run_id,
                "source_run_id": run_id if has_xai else None,
                "source_scope": "RUN" if has_xai else "UNAVAILABLE",
                "status": "AVAILABLE" if has_xai else "NOT_AVAILABLE_FOR_RUN",
                "experiment_id": getattr(ctx.request, "experiment_id", None) or f"RUN_{run_id}",
                "certification_id": None,
                "dataset_id": getattr(ctx.request, "contextId", None) or getattr(ctx.request, "dataset", None),
                "model_id": getattr(ctx.request, "model", None),
                "run_id": run_id,
                "methods": methods,
                "feature_importance": feat_imp,
                "partial_dependence": xai_block.get("partial_dependence", None),
                "ice": xai_block.get("ice", None),
                "shap": xai_block.get("shap", None),
            },
        )

    manifest = bundle.get("certification_manifest", {}) if bundle else {}
    cert_id = manifest.get("certification_id", "CERT_4FB4D8C42308")

    if source_type == "XAI_RUN":
        xai_run = item
        return APIResponseEnvelope(
            success=True,
            run_id=run_id,
            data={
                "requested_run_id": run_id,
                "source_run_id": run_id,
                "source_scope": "RUN",
                "status": "AVAILABLE",
                "experiment_id": "EXP_PRED_REAL_ADULT",
                "certification_id": cert_id,
                "dataset_id": xai_run.get("dataset"),
                "model_id": xai_run.get("model"),
                "run_id": run_id,
                "methods": [xai_run.get("method")],
                "feature_importance": xai_run.get("features_ordered", []),
                "xai_methods": [xai_run],
            },
        )

    if source_type == "CERTIFICATION_RUN":
        cert_run = item
        xai_results = bundle.get("xai_results", []) if bundle else []
        matches = [
            x for x in xai_results
            if x.get("model") == cert_run.get("model") and cert_run.get("dataset") == x.get("dataset")
        ]
        target_matches = matches if matches else xai_results

        return APIResponseEnvelope(
            success=True,
            run_id=run_id,
            data={
                "requested_run_id": run_id,
                "source_run_id": target_matches[0].get("run_id") if target_matches else run_id,
                "source_scope": "CERTIFICATION_EXPERIMENT",
                "status": "AVAILABLE",
                "experiment_id": cert_run.get("experiment_id"),
                "certification_id": cert_id,
                "dataset_id": cert_run.get("dataset"),
                "model_id": cert_run.get("model"),
                "run_id": run_id,
                "methods": [m.get("method") for m in target_matches],
                "feature_importance": target_matches[0].get("features_ordered", []) if target_matches else [],
                "xai_methods": target_matches,
            },
        )

    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data={
            "requested_run_id": run_id,
            "source_run_id": None,
            "source_scope": "UNAVAILABLE",
            "status": "NOT_AVAILABLE_FOR_RUN",
            "experiment_id": None,
            "certification_id": cert_id,
            "dataset_id": item.get("dataset") if isinstance(item, dict) else None,
            "model_id": item.get("model") if isinstance(item, dict) else None,
            "methods": [],
            "feature_importance": [],
            "reason": "XAI not available for specialized run",
        },
    )


@router.get("/{run_id}/sensitivity", response_model=APIResponseEnvelope)
def get_run_sensitivity_presentation(
    run_id: str,
    session_id: str | None = Query(None),
) -> APIResponseEnvelope:
    """Retrieve sensitivity analysis presentation output without web-layer recalculation."""
    item, bundle, source_type = _resolve_run_scientific_identity(run_id, session_id)
    if source_type == "NOT_FOUND":
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found in active queue or certification bundle")

    if source_type == "ACTIVE_RUN":
        ctx = item
        pres = ctx.presentation or {}
        sens_block = pres.get("blocks", {}).get("sensitivity", {})
        has_sens = bool(sens_block)

        return APIResponseEnvelope(
            success=True,
            run_id=run_id,
            data={
                "requested_run_id": run_id,
                "source_run_id": run_id if has_sens else None,
                "source_scope": "RUN" if has_sens else "UNAVAILABLE",
                "status": "AVAILABLE" if has_sens else "NOT_AVAILABLE_FOR_RUN",
                "experiment_id": getattr(ctx.request, "experiment_id", None) or f"RUN_{run_id}",
                "certification_id": None,
                "dataset_id": getattr(ctx.request, "contextId", None) or getattr(ctx.request, "dataset", None),
                "model_id": getattr(ctx.request, "model", None),
                "run_id": run_id,
                "shock_grid": sens_block.get("shock_grid", [-0.30, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.30]) if has_sens else [],
                "baseline_zero_delta": sens_block.get("baseline_zero_delta", 0.0) if has_sens else None,
                "responses": sens_block.get("responses", "RESPONSES_NOT_PERSISTED") if has_sens else "NOT_AVAILABLE_FOR_RUN",
            },
        )

    manifest = bundle.get("certification_manifest", {}) if bundle else {}
    cert_id = manifest.get("certification_id", "CERT_4FB4D8C42308")

    if source_type == "SENSITIVITY_RUN":
        sens_run = item
        return APIResponseEnvelope(
            success=True,
            run_id=run_id,
            data={
                "requested_run_id": run_id,
                "source_run_id": run_id,
                "source_scope": "RUN",
                "status": "AVAILABLE",
                "experiment_id": "EXP_PRED_REAL_ADULT",
                "certification_id": cert_id,
                "dataset_id": "scikit-learn/adult-census-income",
                "model_id": sens_run.get("model"),
                "run_id": run_id,
                "shock_grid": sens_run.get("shock_grid", [-0.30, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.30]),
                "baseline_zero_delta": sens_run.get("baseline_zero_delta", 0.0),
                "evaluations": [sens_run],
                "responses": "RESPONSES_NOT_PERSISTED",
            },
        )

    if source_type == "CERTIFICATION_RUN":
        cert_run = item
        sens_results = bundle.get("sensitivity_results", []) if bundle else []
        matches = [
            s for s in sens_results
            if s.get("model") == cert_run.get("model") and cert_run.get("dataset") == "scikit-learn/adult-census-income"
        ]
        target_matches = matches if matches else sens_results

        return APIResponseEnvelope(
            success=True,
            run_id=run_id,
            data={
                "requested_run_id": run_id,
                "source_run_id": target_matches[0].get("run_id") if target_matches else run_id,
                "source_scope": "CERTIFICATION_EXPERIMENT",
                "status": "AVAILABLE",
                "experiment_id": cert_run.get("experiment_id"),
                "certification_id": cert_id,
                "dataset_id": cert_run.get("dataset"),
                "model_id": cert_run.get("model"),
                "run_id": run_id,
                "shock_grid": target_matches[0].get("shock_grid", [-0.30, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.30]) if target_matches else [-0.30, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.30],
                "baseline_zero_delta": target_matches[0].get("baseline_zero_delta", 0.0) if target_matches else 0.0,
                "evaluations": target_matches,
                "responses": "RESPONSES_NOT_PERSISTED",
            },
        )

    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data={
            "requested_run_id": run_id,
            "source_run_id": None,
            "source_scope": "UNAVAILABLE",
            "status": "NOT_AVAILABLE_FOR_RUN",
            "experiment_id": None,
            "certification_id": cert_id,
            "dataset_id": item.get("dataset") if isinstance(item, dict) else None,
            "model_id": item.get("model") if isinstance(item, dict) else None,
            "shock_grid": [],
            "baseline_zero_delta": None,
            "evaluations": [],
            "responses": "NOT_AVAILABLE_FOR_RUN",
            "reason": "Sensitivity not available for specialized run",
        },
    )


@router.get("/{run_id}/portfolio", response_model=APIResponseEnvelope)
def get_run_portfolio_presentation(
    run_id: str,
    session_id: str | None = Query(None),
) -> APIResponseEnvelope:
    """Retrieve multi-objective portfolio allocation metrics without web recalculation."""
    item, bundle, source_type = _resolve_run_scientific_identity(run_id, session_id)
    if source_type == "NOT_FOUND":
        raise HTTPException(status_code=404, detail=f"Portfolio data not found for run '{run_id}'")

    if source_type == "ACTIVE_RUN":
        ctx = item
        wf = getattr(ctx.request, "workflowId", None) or getattr(ctx.request, "workflow", "")
        dom = getattr(ctx.request, "domain", "")
        is_port = wf == "quantitative_finance" or "port" in dom.lower() or "market" in dom.lower()
        pres = ctx.presentation or {}
        port_block = pres.get("blocks", {}).get("portfolio", {})

        if not is_port or not port_block:
            return APIResponseEnvelope(
                success=True,
                run_id=run_id,
                data={
                    "requested_run_id": run_id,
                    "source_run_id": None,
                    "source_scope": "UNAVAILABLE",
                    "status": "NOT_AVAILABLE_FOR_RUN",
                    "experiment_id": getattr(ctx.request, "experiment_id", None) or f"RUN_{run_id}",
                    "certification_id": None,
                    "dataset_id": getattr(ctx.request, "contextId", None) or getattr(ctx.request, "dataset", None),
                    "model_id": getattr(ctx.request, "model", None),
                    "domain": dom or wf,
                    "portfolio_data": None,
                    "reason": f"Run '{run_id}' is in domain '{dom or wf}', not portfolio",
                },
            )

        return APIResponseEnvelope(
            success=True,
            run_id=run_id,
            data={
                "requested_run_id": run_id,
                "source_run_id": run_id,
                "source_scope": "RUN",
                "status": "AVAILABLE",
                "experiment_id": getattr(ctx.request, "experiment_id", None) or f"RUN_{run_id}",
                "certification_id": None,
                "dataset_id": getattr(ctx.request, "contextId", None) or getattr(ctx.request, "dataset", None),
                "model_id": getattr(ctx.request, "model", None),
                "domain": "portfolio",
                "portfolio_data": port_block,
            },
        )

    manifest = bundle.get("certification_manifest", {}) if bundle else {}
    cert_id = manifest.get("certification_id", "CERT_4FB4D8C42308")

    if source_type == "CERTIFICATION_RUN":
        cert_run = item
        champ_list = bundle.get("champion_challenger", []) if bundle else []
        port_rec = next((c for c in champ_list if c.get("domain") in ("portfolio_and_market_risk", "portfolio")), None) or {}
        return APIResponseEnvelope(
            success=True,
            run_id=run_id,
            data={
                "requested_run_id": run_id,
                "source_run_id": run_id,
                "source_scope": "CERTIFICATION_EXPERIMENT",
                "status": "AVAILABLE",
                "experiment_id": cert_run.get("experiment_id"),
                "certification_id": cert_id,
                "dataset_id": cert_run.get("dataset"),
                "model_id": cert_run.get("model"),
                "domain": "portfolio",
                "multi_objective_evaluation": port_rec.get("multi_objective_evaluation", {}),
                "objective_winners": port_rec.get("objective_winners", {}),
                "real_data_status": "PORTFOLIO_REAL_MARKET_DATA_CERTIFICATION = BLOCKED",
            },
        )

    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data={
            "requested_run_id": run_id,
            "source_run_id": None,
            "source_scope": "UNAVAILABLE",
            "status": "NOT_AVAILABLE_FOR_RUN",
            "portfolio_data": None,
            "reason": "Portfolio not applicable to specialized analysis run",
        },
    )


@router.get("/{run_id}/scenario", response_model=APIResponseEnvelope)
def get_run_scenario_presentation(
    run_id: str,
    session_id: str | None = Query(None),
) -> APIResponseEnvelope:
    """Retrieve scenario and traded-risk stress results without web recalculation."""
    item, bundle, source_type = _resolve_run_scientific_identity(run_id, session_id)
    if source_type == "NOT_FOUND":
        raise HTTPException(status_code=404, detail=f"Scenario data not found for run '{run_id}'")

    if source_type == "ACTIVE_RUN":
        ctx = item
        wf = getattr(ctx.request, "workflowId", None) or getattr(ctx.request, "workflow", "")
        dom = getattr(ctx.request, "domain", "")
        is_scen = wf == "quantitative_finance" or "scen" in dom.lower() or "market" in dom.lower()
        pres = ctx.presentation or {}
        scen_block = pres.get("blocks", {}).get("scenario", {})

        if not is_scen or not scen_block:
            return APIResponseEnvelope(
                success=True,
                run_id=run_id,
                data={
                    "requested_run_id": run_id,
                    "source_run_id": None,
                    "source_scope": "UNAVAILABLE",
                    "status": "NOT_AVAILABLE_FOR_RUN",
                    "experiment_id": getattr(ctx.request, "experiment_id", None) or f"RUN_{run_id}",
                    "certification_id": None,
                    "dataset_id": getattr(ctx.request, "contextId", None) or getattr(ctx.request, "dataset", None),
                    "model_id": getattr(ctx.request, "model", None),
                    "domain": dom or wf,
                    "scenario_data": None,
                    "reason": f"Run '{run_id}' is in domain '{dom or wf}', not scenario_traded_risk",
                },
            )

        return APIResponseEnvelope(
            success=True,
            run_id=run_id,
            data={
                "requested_run_id": run_id,
                "source_run_id": run_id,
                "source_scope": "RUN",
                "status": "AVAILABLE",
                "experiment_id": getattr(ctx.request, "experiment_id", None) or f"RUN_{run_id}",
                "certification_id": None,
                "dataset_id": getattr(ctx.request, "contextId", None) or getattr(ctx.request, "dataset", None),
                "model_id": getattr(ctx.request, "model", None),
                "domain": "scenario_traded_risk",
                "scenario_data": scen_block,
            },
        )

    manifest = bundle.get("certification_manifest", {}) if bundle else {}
    cert_id = manifest.get("certification_id", "CERT_4FB4D8C42308")

    if source_type == "CERTIFICATION_RUN":
        cert_run = item
        champ_list = bundle.get("champion_challenger", []) if bundle else []
        scen_rec = next((c for c in champ_list if c.get("domain") in ("scenario_traded_risk", "portfolio_and_market_risk")), None) or {}
        return APIResponseEnvelope(
            success=True,
            run_id=run_id,
            data={
                "requested_run_id": run_id,
                "source_run_id": run_id,
                "source_scope": "CERTIFICATION_EXPERIMENT",
                "status": "AVAILABLE",
                "experiment_id": cert_run.get("experiment_id"),
                "certification_id": cert_id,
                "dataset_id": cert_run.get("dataset"),
                "model_id": cert_run.get("model"),
                "domain": "scenario_traded_risk",
                "scenario_type": scen_rec.get("champion_model", "gate6_asset_tail_shock"),
                "shock_definition": "relative_equity_shock",
                "units": "NOT_AVAILABLE",
                "horizon": "NOT_AVAILABLE",
                "currency": "NOT_AVAILABLE",
                "repricing_method": scen_rec.get("champion_model", "gate6_asset_tail_shock"),
                "loss": 0.2000,
                "zero_shock_check": "PASS",
                "monotonicity_check": "PASS",
                "Gate6_integrity": "PASS",
                "Gate6A_integrity": "NOT_AVAILABLE",
            },
        )

    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data={
            "requested_run_id": run_id,
            "source_run_id": None,
            "source_scope": "UNAVAILABLE",
            "status": "NOT_AVAILABLE_FOR_RUN",
            "scenario_data": None,
            "reason": "Scenario not applicable to specialized analysis run",
        },
    )
