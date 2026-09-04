"""Run Lifecycle, SSE Streaming, Presentation & Logical Artifact Routes for StART v4.5."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Header, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from start.web.pdf import generate_institutional_pdf
from start.web.queue import GLOBAL_QUEUE
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
    """Execute canonical StART deterministic review in an isolated background thread/task."""
    start_time = time.time()
    try:
        GLOBAL_QUEUE.mark_running(run_id)

        # Import canonical review executor and architecture models
        from start.review.architecture import (
            LLMReviewConfig,
            PredictiveTechnology,
            ReviewContextBundle,
            ReviewDomain,
            ReviewGroundingMode,
            ReviewLifecycle,
            ReviewMode,
        )
        from start.review.executor import run_unified_review

        # 1. Map domain and technology
        is_dl = (
            request.domain == "deep_learning"
            or request.synthetic_profile == "deep_learning_v1"
            or request.workflow == "deep_learning"
        )
        is_market = (
            request.domain == "market"
            or request.synthetic_profile == "institutional_market_v1"
            or request.workflow == "quantitative_finance"
        )

        seed = request.seed or 42

        # Emit Step 1: Initialization & Discovery
        GLOBAL_QUEUE.append_event(
            run_id,
            {
                "event_id": f"EVT-{uuid.uuid4().hex[:8]}",
                "timestamp": time.time(),
                "event_type": "agent_transition",
                "source_agent": "Director",
                "target_agent": "Specialist",
                "stage": "PLANNING",
                "action": f"initialize_{request.domain}_review",
                "status": "RUNNING",
                "metadata": {
                    "workflow": request.workflow,
                    "synthetic_profile": request.synthetic_profile,
                    "seed": seed,
                    "parent_run_id": request.parent_run_id,
                    "intervention": request.intervention,
                },
                "phase": "Planning & Discovery",
                "step": 1,
                "completed": 1,
                "total": 5,
                "percent": 20.0,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "message": f"Initialized {request.domain} engineering review context",
            },
        )

        if is_market:
            domains = (ReviewDomain.MARKET,)
            technology = None
            from start.data.synthetic_market import generate_market_world
            from start.registry.market_contexts import MarketContext, PortfolioSpec

            world = generate_market_world(
                n_assets=50,
                n_periods=1000,
                n_factors=5,
                periods_per_year=252,
                seed=seed,
                include_short_rate=True,
                missing_rate=0.15,
            )
            renamed = {old: f"ASSET_{i + 1:03d}" for i, old in enumerate(world.returns.columns)}
            market_ctx = MarketContext(
                returns=world.returns.rename(columns=renamed),
                prices=world.prices.rename(columns=renamed),
                periods_per_year=world.periods_per_year,
                risk_free_rate=0.02,
                risk_free_frequency="annual",
                factor_returns=world.factor_returns,
                factor_exposures=world.factor_exposures.rename(index=renamed),
                pnl=world.pnl,
                hypothetical_pnl=world.hypothetical_pnl,
                var_series=world.var_series,
                var_confidence=world.var_confidence,
                portfolio=PortfolioSpec(
                    weights=world.weights.rename(renamed),
                    benchmark_weights=world.benchmark_weights.rename(renamed),
                ),
                seed=seed,
            )
            bundle = ReviewContextBundle(
                mode=ReviewMode.SINGLE_DOMAIN,
                domains=domains,
                technology=technology,
                materiality=request.materiality,
                lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
                market=market_ctx,
                short_rate=world.short_rate,
                llm_config=LLMReviewConfig(provider="none"),
                grounding_mode=ReviewGroundingMode.STRUCTURED,
            )
        else:
            domains = (ReviewDomain.PREDICTIVE,)
            technology = (
                PredictiveTechnology.DEEP_LEARNING if is_dl else PredictiveTechnology.TRADITIONAL_ML
            )
            from start.data.synthetic_dl import generate_dl_world
            from start.registry import TestContext

            dl_res = generate_dl_world(n_samples=500, n_features=8, seed=seed)
            train_df = dl_res["train_df"]
            test_df = dl_res["test_df"]
            tab_ctx = TestContext(train=train_df, test=test_df, target_column="target")
            bundle = ReviewContextBundle(
                mode=ReviewMode.SINGLE_DOMAIN,
                domains=domains,
                technology=technology,
                materiality=request.materiality,
                lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
                tabular=tab_ctx,
                llm_config=LLMReviewConfig(provider="none"),
                grounding_mode=ReviewGroundingMode.STRUCTURED,
            )

        # Emit Step 2: Context Building & Pre-flight Diagnostics
        GLOBAL_QUEUE.append_event(
            run_id,
            {
                "event_id": f"EVT-{uuid.uuid4().hex[:8]}",
                "timestamp": time.time(),
                "event_type": "tool_execution",
                "source_agent": "Specialist",
                "target_agent": "DeterministicEngine",
                "stage": "EXECUTION",
                "action": "build_context_and_preflight",
                "status": "RUNNING",
                "metadata": {"technology": str(technology) if technology else "MARKET"},
                "phase": "Context & Pre-flight Diagnostics",
                "step": 2,
                "completed": 2,
                "total": 5,
                "percent": 40.0,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "message": "Building analytical dataset context and verifying integrity",
            },
        )

        # Run canonical StART review
        res = run_unified_review(
            bundle=bundle,
            interactive=False,
        )

        records = res.get("records", [])
        presentation_model = res.get("presentation_model")
        artifacts_dict = res.get("artifacts", {})
        if not artifacts_dict and presentation_model:
            if hasattr(presentation_model, "artifacts") and presentation_model.artifacts:
                artifacts_dict = presentation_model.artifacts
            elif isinstance(presentation_model, dict) and "artifacts" in presentation_model:
                artifacts_dict = presentation_model["artifacts"]

        tracer = res.get("tracer")
        if tracer:
            events = [e.to_dict() if hasattr(e, "to_dict") else e for e in tracer.events]
        else:
            events = res.get("orchestration_events", [])

        # Emit Step 3 & 4: Analytical Surfaces & Governance
        GLOBAL_QUEUE.append_event(
            run_id,
            {
                "event_id": f"EVT-{uuid.uuid4().hex[:8]}",
                "timestamp": time.time(),
                "event_type": "evidence_commit",
                "source_agent": "DeterministicEngine",
                "target_agent": "EvidenceLedger",
                "stage": "CHECKPOINTS",
                "action": "commit_evidence_records",
                "status": "SUCCESS",
                "metadata": {"evidence_count": len(records)},
                "evidence_refs": [r.evidence_id for r in records[:8]],
                "phase": "Deterministic Analytical Execution",
                "step": 4,
                "completed": 4,
                "total": 5,
                "percent": 85.0,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "message": (
                    f"Computed {len(records)} deterministic evidence surfaces and statistical checkpoints"
                ),
            },
        )

        # Transfer execution events
        for evt in events:
            if isinstance(evt, dict):
                GLOBAL_QUEUE.append_event(run_id, evt)
            elif hasattr(evt, "to_dict"):
                GLOBAL_QUEUE.append_event(run_id, evt.to_dict())

        # Serialize presentation model
        if presentation_model and hasattr(presentation_model, "to_dict"):
            pres_dict = presentation_model.to_dict()
        else:
            pres_dict = {}
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

        # Emit Final Step 5: Completed Seal
        GLOBAL_QUEUE.append_event(
            run_id,
            {
                "event_id": f"EVT-{uuid.uuid4().hex[:8]}",
                "timestamp": time.time(),
                "event_type": "governance_seal",
                "source_agent": "ModelGovernanceAgent",
                "target_agent": "AttestationRegistry",
                "stage": "GOVERNANCE",
                "action": "seal_attestation_merkle_root",
                "status": "SUCCESS",
                "metadata": {"merkle_root": pres_dict.get("attestation_seal_merkle_root", "")},
                "phase": "Attestation & Evidence Seal",
                "step": 5,
                "completed": 5,
                "total": 5,
                "percent": 100.0,
                "elapsed_seconds": round(time.time() - start_time, 2),
                "message": "Governance disposition attested and Merkle tree sealed",
            },
        )

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


@router.post("", response_model=APIResponseEnvelope)
@router.post("/start", response_model=APIResponseEnvelope)
async def start_run(
    request: RunRequest,
    x_forwarded_for: str | None = Header(None),
) -> Response:
    """Submit a deterministic analytical review run."""
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

    # 2. Canonical Synthetic Profile Validation
    valid_profiles = {"institutional_credit_v1", "deep_learning_v1", "institutional_market_v1"}
    if request.synthetic_profile not in valid_profiles:
        # Fallback/remap common requests
        if request.domain == "deep_learning":
            request.synthetic_profile = "deep_learning_v1"
        elif request.domain == "market":
            request.synthetic_profile = "institutional_market_v1"
        else:
            request.synthetic_profile = "institutional_credit_v1"

    # 3. Assign unique run ID
    run_id = f"RUN-WEB-{uuid.uuid4().hex[:10]}"

    # 4. Submit to analytical queue
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

    # 5. Launch background execution task
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _execute_run_in_background, run_id, request)

    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "schema_version": START_SCHEMA_VERSION,
            "run_id": run_id,
            "timestamp": time.time(),
            "data": {
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
        data={"evidence_records": records, "count": len(records)},
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
        # Sandboxed HTML: prevent same-origin script execution
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
