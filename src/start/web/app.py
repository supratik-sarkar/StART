"""FastAPI Application Factory & Production Middleware for StART v4.5 Web Transport.

Configures:
- Strict CORS for Cloudflare production and Hugging Face mirror
- Security Headers (CSP, X-Content-Type-Options, Referrer-Policy, Frame-Ancestors)
- Origin HMAC authentication
- Router mounting for health, runs, SSE, artifacts, and reviewer hydration
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Header, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from start.web.routes_certification import router as certification_router
from start.web.routes_data import router as data_router
from start.web.routes_health import get_health
from start.web.routes_health import router as health_router
from start.web.routes_reviewer import router as reviewer_router
from start.web.routes_run import router as run_router
from start.web.routes_workbench import router as workbench_router
from start.web.schemas import START_VERSION, APIResponseEnvelope, RunRequest
from start.web.security import verify_origin_hmac

DEFAULT_DIST = Path(__file__).resolve().parent.parent.parent.parent / "webapp" / "dist"
DIST_DIR = Path(os.environ.get("START_WEBAPP_DIST", str(DEFAULT_DIST)))


def create_app() -> FastAPI:
    """Create and configure the production StART FastAPI application."""
    app = FastAPI(
        title="StART — Browser-Native Agentic Engineering Workbench Web API",
        description=(
            "Deterministic ML/DL & quantitative engineering workbench, "
            "live SSE streaming, and WebLLM reviewer hydration."
        ),
        version=START_VERSION,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    # 1. CORS Configuration
    allowed_origins = [
        "https://start-mrt.org",
        "https://*.pages.dev",
        "https://*.hf.space",
        "https://start-mrt-gateway.sapman.workers.dev",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["Last-Event-ID", "Content-Disposition", "X-StART-Schema-Version"],
    )

    # 2. Origin Authentication & Security Headers Middleware
    @app.middleware("http")
    async def security_and_origin_middleware(request: Request, call_next):
        # Origin HMAC Check for production /api/* routes
        if request.url.path.startswith("/api/v1/runs/start"):
            sig = request.headers.get("X-StART-Origin-Sig")
            ts = request.headers.get("X-StART-Origin-Ts")
            nonce = request.headers.get("X-StART-Origin-Nonce")
            body_bytes = await request.body()
            if not verify_origin_hmac(sig, ts, nonce, request.method, request.url.path, body_bytes):
                return Response(
                    content='{"success": false, "error": "Unauthorized origin gateway"}',
                    status_code=403,
                    media_type="application/json",
                )

        response: Response = await call_next(request)

        # Apply security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-StART-Schema-Version"] = "5.0.0"

        # Explicit fail-closed cache control for dynamic routes
        if request.url.path.startswith("/api/") or request.url.path == "/health":
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

        # Content Security Policy (allows WebAssembly & WebGPU compilation for WebLLM)
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-eval' 'wasm-unsafe-eval' https://challenges.cloudflare.com; "
            "style-src 'self' 'unsafe-inline'; "
            "connect-src 'self' https://* wss://* http://localhost:* http://127.0.0.1:*; "
            "img-src 'self' data: blob:; "
            "font-src 'self' data:; "
            "frame-src 'self' https://challenges.cloudflare.com blob:; "
            "object-src 'none';"
        )
        response.headers["Content-Security-Policy"] = csp
        return response

    # 3. Mount API Routers
    app.include_router(health_router)
    app.include_router(run_router)
    app.include_router(reviewer_router)
    app.include_router(workbench_router)
    app.include_router(data_router)
    app.include_router(certification_router)

    # Root health probe delegating directly to canonical get_health()
    @app.get("/health", response_model=APIResponseEnvelope, tags=["health"])
    def root_health() -> APIResponseEnvelope:
        return get_health()

    # B1 Root route aliases for workbench /plan/generate and /workflow/run
    @app.post("/plan/generate", tags=["workbench"])
    async def root_plan_generate(run_request: RunRequest):
        from start.web.routes_workbench import create_agent_plan
        return create_agent_plan(run_request)

    @app.post("/workflow/run", tags=["runs"])
    @app.post("/api/v1/workflow/run", tags=["runs"])
    async def root_workflow_run(
        run_request: RunRequest,
        x_forwarded_for: str | None = Header(None),
    ):
        from start.web.routes_run import start_run
        return await start_run(run_request, x_forwarded_for=x_forwarded_for)

    # 4. Mount Frontend Static Assets if built
    if DIST_DIR.exists() and (DIST_DIR / "index.html").exists():
        app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")

    return app


# Default ASGI application instance
app = create_app()
