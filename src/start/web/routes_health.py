"""Health and Info Routes for StART Web Transport."""

from __future__ import annotations

import os

from fastapi import APIRouter

from start.web.schemas import (
    START_SCHEMA_VERSION,
    START_VERSION,
    APIResponseEnvelope,
    SystemInfo,
)

router = APIRouter(prefix="/api/v1", tags=["health"])


def get_health_payload() -> dict[str, str]:
    """Authoritative health and version payload."""
    backend_build = os.environ.get("START_BACKEND_BUILD_VERSION", f"{START_VERSION}-local")
    return {
        "status": "HEALTHY",
        "version": START_VERSION,
        "schema_version": START_SCHEMA_VERSION,
        "backend_build_version": backend_build,
    }


@router.get("/health", response_model=APIResponseEnvelope)
def get_health() -> APIResponseEnvelope:
    """Lightweight readiness and liveness probe."""
    return APIResponseEnvelope(
        success=True,
        data=get_health_payload(),
    )


@router.get("/info", response_model=APIResponseEnvelope)
def get_info() -> APIResponseEnvelope:
    """System information, versioning, and compute runtime metadata."""
    runtime = "oracle_a1_arm64" if os.environ.get("START_ORACLE_DEPLOYMENT") else "local"
    info = SystemInfo(
        start_version=START_VERSION,
        start_schema_version=START_SCHEMA_VERSION,
        git_sha=os.environ.get("START_GIT_SHA"),
        compute_runtime=runtime,
        max_concurrency=1,
        engine_status="READY",
    )
    return APIResponseEnvelope(
        success=True,
        data=info.model_dump(),
    )



@router.get("/profiles", response_model=APIResponseEnvelope)
def list_synthetic_profiles() -> APIResponseEnvelope:
    """List available versioned synthetic dataset profiles (ML/DL first)."""
    profiles = [
        {
            "profile_id": "institutional_credit_v1",
            "name": "Predictive ML — Institutional Credit & Classification",
            "domain": "predictive",
            "version": "1.0.0",
            "description": (
                "Tabular classification benchmark with feature diagnostics, "
                "calibration curves, and SHAP explainability."
            ),
            "seed": 42,
        },
        {
            "profile_id": "deep_learning_v1",
            "name": "Deep Learning — Neural Architecture & Gradient Diagnostics",
            "domain": "deep_learning",
            "version": "1.0.0",
            "description": (
                "Deep neural network inspection with layer spectra, activation distributions, "
                "and integrated gradients."
            ),
            "seed": 42,
        },
        {
            "profile_id": "institutional_market_v1",
            "name": "Quantitative Finance — Market Risk & Portfolio Stress",
            "domain": "market",
            "version": "1.0.0",
            "description": (
                "Multi-asset portfolio returns, covariance matrices, VaR/ES backtests, "
                "and historical shock scenarios."
            ),
            "seed": 42,
        },
    ]
    return APIResponseEnvelope(
        success=True,
        data={"profiles": profiles},
    )
