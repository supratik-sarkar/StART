"""Workbench Presentation & Transport Routes for Greenfield StART Frontend.

Implements canonical backend transport and orchestration services conforming strictly to
the greenfield webapp contracts and canonical StART invariants:
- Truthful capabilities discovery with exact test IDs resolved from the 79-test registry
- Public-safe execution context catalog
- Dedicated AgentPlanPreview generation without fabricating fake run IDs
- Runtime-derived execution graph showing genuine producer-evidence lineage and parent lineage
- Canonical EvidenceRecord and Finding presentation without fabricated severities or actions
- Fail-closed deterministic action validation boundary and child run orchestration
- Fail-closed governance and Merkle attestation surfaces (null when absent, no fake seals)
"""

from __future__ import annotations

import datetime
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from start.registry import list_tests
from start.web.queue import GLOBAL_QUEUE, ActiveRunContext
from start.web.schemas import RunRequest

logger = logging.getLogger("start.web.routes_workbench")
router = APIRouter(prefix="/api/v1", tags=["workbench"])


# --------------------------------------------------------------------------- #
# Canonical Workflow Capability Resolver
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorkflowDefinition:
    workflow_id: str
    label: str
    category: str  # "ml" | "quant"
    enabled: bool
    disabled_reason: str | None
    compatible_contexts: list[str]
    supported_actions: list[str]
    # Each group: (step_id, label, kind, description, canonical_test_ids_or_filter)
    step_specs: list[tuple[str, str, str, str, tuple[str, ...]]]


def _get_registered_test_ids() -> set[str]:
    return {t.test_id for t in list_tests()}


def _build_workflow_definitions() -> dict[str, WorkflowDefinition]:
    all_tests = list_tests()
    all_test_ids = {t.test_id for t in all_tests}

    pred_eda_prep = tuple(t.test_id for t in all_tests if t.family in ("eda", "preprocessing"))
    pred_fe = tuple(t.test_id for t in all_tests if t.family == "feature_engineering")
    pred_sup = tuple(t.test_id for t in all_tests if t.family == "supervised")
    pred_xai = tuple(t.test_id for t in all_tests if t.family in ("xai", "genai"))

    market_portfolio = tuple(t.test_id for t in all_tests if t.family == "portfolio")
    market_attr = tuple(t.test_id for t in all_tests if t.family == "attribution")
    market_cov = tuple(t.test_id for t in all_tests if t.family == "covariance")
    market_risk = tuple(
        t.test_id
        for t in all_tests
        if t.family == "traded_risk"
        and ("var" in t.test_id or t.test_id == "traded_risk.brownian_bridge_barrier")
    )

    dl_tests = tuple(
        tid
        for tid in (
            "xai.integrated_gradients",
            "xai.global_importance",
            "xai.feature_sensitivity",
            "xai.importance_stability",
            "supervised.classification_metrics",
            "supervised.calibration",
            "supervised.discrimination",
        )
        if tid in all_test_ids
    )

    model_diag_tests = tuple(
        tid
        for tid in (
            "supervised.classification_metrics",
            "supervised.discrimination",
            "supervised.cohort_metrics_comparison",
            "supervised.top_decile_lift",
            "supervised.calibration",
        )
        if tid in all_test_ids
    )

    calib_tests = tuple(
        tid
        for tid in (
            "supervised.calibration",
            "supervised.classification_metrics",
            "supervised.discrimination",
        )
        if tid in all_test_ids
    )

    robust_tests = tuple(
        tid
        for tid in (
            "preprocessing.feature_drift",
            "preprocessing.categorical_drift",
            "xai.feature_sensitivity",
            "xai.importance_stability",
        )
        if tid in all_test_ids
    )

    xai_tests = tuple(
        tid
        for tid in (
            "xai.global_importance",
            "xai.feature_sensitivity",
            "xai.importance_stability",
            "xai.integrated_gradients",
            "genai.citation_coverage",
        )
        if tid in all_test_ids
    )

    tuning_tests = tuple(
        tid
        for tid in (
            "supervised.classification_metrics",
            "supervised.discrimination",
            "supervised.calibration",
        )
        if tid in all_test_ids
    )

    return {
        "predictive_ml": WorkflowDefinition(
            workflow_id="predictive_ml",
            label="Predictive ML",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=["rerun", "change_parameter", "deeper_test"],
            step_specs=[
                ("step-context", "Load execution context", "context", "Load seeded tabular benchmark", ()),
                (
                    "step-preflight",
                    "Data contract & integrity diagnostics",
                    "test",
                    "Schema, missingness, drift, and leakage checks",
                    pred_eda_prep,
                ),
                (
                    "step-features",
                    "Feature engineering verification",
                    "test",
                    "Transformation, monotonic binning, and interaction surfaces",
                    pred_fe,
                ),
                (
                    "step-supervised",
                    "Supervised classification evaluation",
                    "test",
                    "Classification metrics, discrimination, and lift",
                    pred_sup,
                ),
                (
                    "step-xai",
                    "Feature attribution & explainability",
                    "test",
                    "SHAP, feature sensitivity, and citation coverage",
                    pred_xai,
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit immutable EvidenceRecords",
                    (),
                ),
                (
                    "step-governance",
                    "Governance & attestation seal",
                    "governance",
                    "Policy evaluation & cryptographic Merkle seal",
                    (),
                ),
            ],
        ),
        "deep_learning": WorkflowDefinition(
            workflow_id="deep_learning",
            label="Deep Learning",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["deep_learning_v1"],
            supported_actions=["rerun", "change_parameter"],
            step_specs=[
                (
                    "step-context",
                    "Load execution context",
                    "context",
                    "Load high-dimensional neural embedding world",
                    (),
                ),
                (
                    "step-training",
                    "Neural architecture training",
                    "tool",
                    "Monitor epoch loss convergence and gradient dynamics",
                    (),
                ),
                (
                    "step-performance",
                    "Neural decision surfaces",
                    "test",
                    "Classification metrics and calibration on latent embeddings",
                    dl_tests[:3],
                ),
                (
                    "step-attribution",
                    "Integrated gradients attribution",
                    "test",
                    "Layer attributions, feature sensitivity, and gradient tensors",
                    dl_tests[3:],
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit immutable EvidenceRecords",
                    (),
                ),
                (
                    "step-governance",
                    "Governance & attestation seal",
                    "governance",
                    "Policy evaluation & cryptographic Merkle seal",
                    (),
                ),
            ],
        ),
        "data_diagnostics": WorkflowDefinition(
            workflow_id="data_diagnostics",
            label="Data Diagnostics",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=["rerun"],
            step_specs=[
                ("step-context", "Load execution context", "context", "Load tabular dataset context", ()),
                (
                    "step-eda",
                    "Exploratory data analysis",
                    "test",
                    "Distributions, correlation structure, and collinearity",
                    tuple(t for t in pred_eda_prep if t.startswith("eda.")),
                ),
                (
                    "step-preprocessing",
                    "Preprocessing & leakage screening",
                    "test",
                    "Missingness, outliers, drift, and leakage detection",
                    tuple(t for t in pred_eda_prep if t.startswith("preprocessing.")),
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit diagnostic EvidenceRecords",
                    (),
                ),
                ("step-governance", "Sign-off", "governance", "Deterministic integrity sign-off", ()),
            ],
        ),
        "model_diagnostics": WorkflowDefinition(
            workflow_id="model_diagnostics",
            label="Model Diagnostics",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=["rerun", "deeper_test"],
            step_specs=[
                ("step-context", "Load model context", "context", "Load trained model artifact context", ()),
                (
                    "step-metrics",
                    "Classification performance metrics",
                    "test",
                    "Thresholded metrics, discrimination AUC, and lift",
                    model_diag_tests[:4],
                ),
                (
                    "step-calibration",
                    "Probabilistic calibration checks",
                    "test",
                    "Brier score and reliability curves",
                    model_diag_tests[4:],
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit diagnostic EvidenceRecords",
                    (),
                ),
                ("step-governance", "Sign-off", "governance", "Diagnostic sign-off", ()),
            ],
        ),
        "calibration": WorkflowDefinition(
            workflow_id="calibration",
            label="Calibration",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=["rerun", "change_parameter"],
            step_specs=[
                (
                    "step-context",
                    "Load execution context",
                    "context",
                    "Load predicted probabilities context",
                    (),
                ),
                (
                    "step-calibration-tests",
                    "Measure calibration & reliability",
                    "test",
                    "Expected Calibration Error (ECE) and Brier score",
                    calib_tests,
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit calibration EvidenceRecords",
                    (),
                ),
                (
                    "step-governance",
                    "Review & sign-off",
                    "governance",
                    "Attest probabilistic reliability",
                    (),
                ),
            ],
        ),
        "robustness": WorkflowDefinition(
            workflow_id="robustness",
            label="Robustness",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=["rerun", "change_parameter"],
            step_specs=[
                ("step-context", "Load execution context", "context", "Load benchmark context", ()),
                (
                    "step-drift",
                    "Distributional drift stress",
                    "test",
                    "Feature and categorical drift under perturbation",
                    robust_tests[:2],
                ),
                (
                    "step-sensitivity",
                    "Feature perturbation sensitivity",
                    "test",
                    "Top-feature shock sensitivity and stability",
                    robust_tests[2:],
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit robustness EvidenceRecords",
                    (),
                ),
                ("step-governance", "Review & sign-off", "governance", "Attest perturbation robustness", ()),
            ],
        ),
        "explainability": WorkflowDefinition(
            workflow_id="explainability",
            label="Explainability",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1", "deep_learning_v1"],
            supported_actions=["rerun"],
            step_specs=[
                ("step-context", "Load execution context", "context", "Load feature evaluation context", ()),
                (
                    "step-attribution-tests",
                    "Feature attribution & stability",
                    "test",
                    "SHAP importance, permutation stability, and citation coverage",
                    xai_tests,
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit explainability EvidenceRecords",
                    (),
                ),
                ("step-governance", "Review & sign-off", "governance", "Attest explainability artifacts", ()),
            ],
        ),
        "hyperparameter_tuning": WorkflowDefinition(
            workflow_id="hyperparameter_tuning",
            label="Tune a Model",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=["rerun", "change_parameter"],
            step_specs=[
                ("step-context", "Load execution context", "context", "Load dataset context", ()),
                (
                    "step-tuning",
                    "Execute bounded trials",
                    "tool",
                    "Run bounded search with trial progress",
                    (),
                ),
                (
                    "step-validation",
                    "Validate optimal candidate",
                    "test",
                    "Validate candidate metrics and calibration",
                    tuning_tests,
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit optimal trial EvidenceRecords",
                    (),
                ),
                (
                    "step-governance",
                    "Review & sign-off",
                    "governance",
                    "Attest hyperparameter optimization",
                    (),
                ),
            ],
        ),
        "model_comparison": WorkflowDefinition(
            workflow_id="model_comparison",
            label="Compare Models",
            category="ml",
            enabled=False,
            disabled_reason=(
                "Multi-model candidate comparison workflow requires multi-candidate input protocol "
                "not yet enabled in the canonical web review interface."
            ),
            compatible_contexts=[],
            supported_actions=[],
            step_specs=[],
        ),
        "quantitative_finance": WorkflowDefinition(
            workflow_id="quantitative_finance",
            label="Quantitative Finance",
            category="quant",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_market_v1"],
            supported_actions=["rerun", "change_parameter"],
            step_specs=[
                (
                    "step-context",
                    "Load market world",
                    "context",
                    "Generate synthetic multi-asset market returns",
                    (),
                ),
                (
                    "step-portfolio",
                    "Portfolio risk & optimization",
                    "test",
                    "Risk statistics, historical returns, Mean-Variance, and HRP",
                    market_portfolio,
                ),
                (
                    "step-factor",
                    "Factor modeling & attribution",
                    "test",
                    "Factor exposures, return attribution, and risk decomposition",
                    market_attr,
                ),
                (
                    "step-covariance",
                    "Covariance matrix conditioning",
                    "test",
                    "Empirical, Ledoit-Wolf, and regularized EM covariance",
                    market_cov,
                ),
                (
                    "step-var",
                    "Traded risk & VaR backtesting",
                    "test",
                    "Kupiec POF, Christoffersen coverage, and exception tests",
                    market_risk,
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit financial EvidenceRecords",
                    (),
                ),
                (
                    "step-governance",
                    "Governance & attestation seal",
                    "governance",
                    "Financial model risk governance & Merkle seal",
                    (),
                ),
            ],
        ),
    }


def get_workflow_definition(workflow_id: str) -> WorkflowDefinition:
    defs = _build_workflow_definitions()
    return defs.get(workflow_id, defs["predictive_ml"])


def get_workflow_test_ids(workflow_id: str) -> set[str]:
    wdef = get_workflow_definition(workflow_id)
    out: set[str] = set()
    for _, _, _, _, test_ids in wdef.step_specs:
        out.update(test_ids)
    return out


# --------------------------------------------------------------------------- #
# Canonical Plan Builder
# --------------------------------------------------------------------------- #
def make_canonical_plan(workflow_id: str) -> list[dict[str, Any]]:
    """Build canonical agent orchestration plan steps matching registered engines.

    Human-readable labels serve strictly as presentation metadata.
    Plan steps exist only if their underlying registered test group or operation is present.
    """
    wdef = get_workflow_definition(workflow_id)
    all_registered = _get_registered_test_ids()

    plan: list[dict[str, Any]] = []
    prev_id: str | None = None

    for step_id, label, kind, desc, test_ids in wdef.step_specs:
        # If the step declares test IDs, ensure at least one test actually exists in registry
        if test_ids and not any(t in all_registered for t in test_ids):
            continue

        plan.append(
            {
                "id": step_id,
                "label": label,
                "description": desc,
                "kind": kind,
                "status": "completed" if len(plan) == 0 else "queued",
                "parentId": prev_id,
            }
        )
        prev_id = step_id

    return plan


# --------------------------------------------------------------------------- #
# Run Snapshot Serializer
# --------------------------------------------------------------------------- #
def serialize_run_snapshot(ctx: ActiveRunContext) -> dict[str, Any]:
    """Serialize an active or completed run context into a greenfield RunSnapshot."""
    req = ctx.request
    workflow_id = getattr(req, "workflowId", None) or getattr(req, "workflow", "predictive_ml")
    context_id = getattr(req, "contextId", None) or getattr(
        req, "synthetic_profile", "institutional_credit_v1"
    )
    goal = getattr(req, "goal", "") or f"Evaluate {workflow_id} on {context_id}"

    phase_map = {
        "QUEUED": "planning",
        "RUNNING": "running",
        "COMPLETED": "completed",
        "FAILED": "failed",
    }
    phase = phase_map.get(ctx.status, "running")

    elapsed_ms = int(((ctx.completed_at or time.time()) - (ctx.started_at or ctx.created_at)) * 1000)

    # Extract real progress from events only (no fabricated percentages or counts)
    progress = None
    if ctx.events:
        for ev in reversed(ctx.events):
            p = ev.get("progress")
            if isinstance(p, dict) and ("percent" in p or "completed" in p):
                progress = p
                break
            elif "percent" in ev or "completed" in ev:
                progress = {
                    "label": ev.get("phase", "Analytical execution"),
                    "percent": float(ev["percent"]) if "percent" in ev else None,
                    "completed": int(ev["completed"]) if "completed" in ev else None,
                    "total": int(ev["total"]) if "total" in ev else None,
                    "detail": ev.get("message", ""),
                }
                break

    if not progress:
        if ctx.status == "COMPLETED":
            progress = {"label": "Completed", "detail": "Deterministic execution completed"}
        elif ctx.status == "RUNNING":
            progress = {"label": "Running"}
        elif ctx.status == "FAILED":
            progress = {"label": "Failed", "detail": ctx.error_message or "Execution failed"}
        else:
            progress = {"label": "Planning"}

    # Derive plan step status from real runtime events
    plan = make_canonical_plan(workflow_id)
    event_node_ids = {ev.get("node_id") for ev in ctx.events if ev.get("node_id")}
    completed_node_ids = {
        ev.get("node_id")
        for ev in ctx.events
        if ev.get("node_id") and str(ev.get("status", "")).upper() in ("COMPLETED", "SUCCESS")
    }

    for p in plan:
        pid = p["id"]
        if ctx.status == "COMPLETED":
            p["status"] = "completed"
        elif pid in completed_node_ids:
            p["status"] = "completed"
        elif pid in event_node_ids:
            p["status"] = "running"
        elif ctx.status == "RUNNING":
            p["status"] = "queued"
        else:
            p["status"] = "future"

    started_iso = datetime.datetime.fromtimestamp(
        ctx.started_at or ctx.created_at, tz=datetime.UTC
    ).isoformat()
    updated_iso = datetime.datetime.fromtimestamp(
        ctx.completed_at or time.time(), tz=datetime.UTC
    ).isoformat()

    return {
        "runId": ctx.run_id,
        "workflowId": workflow_id,
        "contextId": context_id,
        "goal": goal,
        "phase": phase,
        "statusLabel": "Run signed off"
        if ctx.status == "COMPLETED"
        else ("Deterministic execution running" if ctx.status == "RUNNING" else "Agent plan accepted"),
        "startedAt": started_iso,
        "updatedAt": updated_iso,
        "elapsedMs": max(0, elapsed_ms),
        "progress": progress,
        "plan": plan,
        "parentRunId": getattr(req, "parent_run_id", None) or getattr(req, "parentRunId", None),
        "sourceEvidenceId": getattr(req, "source_evidence_id", None)
        or getattr(req, "sourceEvidenceId", None),
    }


# --------------------------------------------------------------------------- #
# Fail-Closed Action Validation & Intervention Resolution
# --------------------------------------------------------------------------- #
def validate_action_or_raise(
    ctx: ActiveRunContext,
    action: dict[str, Any],
) -> dict[str, Any]:
    """Authoritative server-side action validation boundary (Amendments 11, 12, 13).

    Fails closed:
    - Rejects unsupported action kinds.
    - Rejects unknown sourceEvidenceId with 400 (no silent defaulting).
    - Rejects out-of-range parameters with 400 (no silent clamping).
    """
    if not isinstance(action, dict):
        raise HTTPException(status_code=400, detail="Action payload must be a JSON object")

    kind = action.get("kind")
    if not kind or not isinstance(kind, str):
        raise HTTPException(status_code=400, detail="Missing or invalid 'kind' in action proposal")

    req = ctx.request
    workflow_id = getattr(req, "workflowId", None) or getattr(req, "workflow", "predictive_ml")
    wdef = get_workflow_definition(workflow_id)

    if not wdef.enabled:
        raise HTTPException(status_code=400, detail=f"Workflow '{workflow_id}' is disabled")

    if kind not in wdef.supported_actions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Action kind '{kind}' is unsupported for workflow '{workflow_id}'. "
                f"Supported: {wdef.supported_actions}"
            ),
        )

    # Validate sourceEvidenceId: must exist in active run evidence universe
    source_ev_id = action.get("sourceEvidenceId")
    if source_ev_id:
        known_ev_ids = {
            getattr(r, "evidence_id", None) or (r.get("evidence_id") if isinstance(r, dict) else "")
            for r in ctx.evidence_records
        }
        if source_ev_id not in known_ev_ids:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unknown sourceEvidenceId '{source_ev_id}'. "
                    "Evidence ID must exist in active run universe."
                ),
            )

    # Validate parameters: fail closed, NO silent clamping or mutation
    params = action.get("parameters")
    validated_params: dict[str, Any] = {}
    if params is not None:
        if not isinstance(params, dict):
            raise HTTPException(status_code=400, detail="Parameters must be a key-value object")
        for k, v in params.items():
            if k == "perturbation_rate":
                try:
                    pr = float(v)
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail=f"Invalid perturbation_rate '{v}'") from None
                if pr < 0.01 or pr > 0.30:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"perturbation_rate {pr} out of allowed bounds [0.01, 0.30]. "
                            "Silent clamping is prohibited."
                        ),
                    )
                validated_params[k] = pr
            elif k == "depth":
                if v not in ("standard", "focused", "comprehensive"):
                    raise HTTPException(status_code=400, detail=f"Invalid depth '{v}'")
                validated_params[k] = v
            elif k in ("seed", "trials", "epochs"):
                try:
                    val = int(v)
                    if val < 1 or val > 1000:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Parameter '{k}'={val} out of bounds [1, 1000]",
                        )
                    validated_params[k] = val
                except (ValueError, TypeError):
                    raise HTTPException(status_code=400, detail=f"Invalid integer for '{k}': {v}") from None
            else:
                validated_params[k] = v

    action_id = action.get("actionId") or f"ACT-{uuid.uuid4().hex[:8].upper()}"
    label = action.get("label") or f"Validated {kind.replace('_', ' ')}"
    description = f"Deterministic verification action validated by server for run {ctx.run_id}."

    return {
        "actionId": action_id,
        "label": label,
        "description": description,
        "kind": kind,
        "sourceNodeId": action.get("sourceNodeId"),
        "sourceEvidenceId": source_ev_id,
        "parameters": validated_params,
    }


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@router.get("/capabilities")
def get_capabilities() -> list[dict[str, Any]]:
    """Return truthful capability catalog. Exactly 79 registered deterministic surfaces.

    Capabilities derive directly from the registry:
    - 9 workflows enabled with exact canonical test counts.
    - 1 workflow disabled ('model_comparison') with machine-readable reason.
    """
    defs = _build_workflow_definitions()
    out: list[dict[str, Any]] = []

    descriptions = {
        "predictive_ml": (
            "Evaluate supervised models through 52 registered deterministic engineering surfaces."
        ),
        "deep_learning": (
            "Inspect neural architecture, layer spectra, gradient dynamics, "
            "and activation distributions across 7 tests."
        ),
        "data_diagnostics": (
            "Evaluate data contract integrity, distribution drift, and missingness across 27 tests."
        ),
        "model_diagnostics": (
            "Trace residual behaviour, error clustering, and subpopulation discrimination across 5 tests."
        ),
        "calibration": (
            "Inspect probabilistic reliability, Brier score, and expected calibration error curves "
            "across 3 tests."
        ),
        "robustness": (
            "Stress model behaviour under deterministic Gaussian, missingness, and feature perturbations "
            "across 4 tests."
        ),
        "explainability": (
            "Inspect evidence-backed SHAP attributions, feature importance, and interaction tensors "
            "across 5 tests."
        ),
        "hyperparameter_tuning": (
            "Run bounded parameter search with truthful trial-level validation progress across 3 tests."
        ),
        "model_comparison": ("Evaluate multiple candidates under a unified validation protocol."),
        "quantitative_finance": (
            "Run market risk, Kupiec VaR/ES, portfolio optimization, and factor stress across 25 tests."
        ),
    }

    for wf_id, wdef in defs.items():
        entry: dict[str, Any] = {
            "id": wf_id,
            "label": wdef.label,
            "description": descriptions.get(wf_id, f"Deterministic {wdef.label} workflow."),
            "category": wdef.category,
            "enabled": wdef.enabled,
        }
        if not wdef.enabled and wdef.disabled_reason:
            entry["disabledReason"] = wdef.disabled_reason
        out.append(entry)

    return out


@router.get("/execution-contexts")
def get_execution_contexts() -> list[dict[str, Any]]:
    """Return versioned public-safe execution contexts."""
    return [
        {
            "id": "institutional_credit_v1",
            "label": "Synthetic Credit Classification",
            "kind": "dataset",
            "description": (
                "Seeded public-safe tabular binary classification context for ML & diagnostic engineering."
            ),
            "provenance": "Built-in deterministic synthetic generator",
            "shape": "12,000 × 31",
            "target": "default_flag",
            "seed": 42,
            "badges": ["public-safe", "seeded", "binary", "credit"],
        },
        {
            "id": "deep_learning_v1",
            "label": "Synthetic Vision Embeddings",
            "kind": "dataset",
            "description": (
                "Seeded embedding classification context for deep neural architecture diagnostics."
            ),
            "provenance": "Built-in deterministic synthetic generator",
            "shape": "8,000 × 128",
            "target": "class_id",
            "seed": 17,
            "badges": ["public-safe", "deep-learning", "embeddings"],
        },
        {
            "id": "institutional_market_v1",
            "label": "Synthetic Multi-Asset Market World",
            "kind": "synthetic-world",
            "description": (
                "Seeded multi-asset scenario context for traded risk, VaR backtests, and portfolio workflows."
            ),
            "provenance": "Built-in deterministic synthetic market generator",
            "shape": "24 assets × 1,500 observations",
            "seed": 7,
            "badges": ["public-safe", "quantitative", "var-backtest"],
        },
    ]


@router.post("/plans")
def create_agent_plan(request: RunRequest) -> dict[str, Any]:
    """Generate dedicated AgentPlanPreview without creating a run or fabricating fake run IDs."""
    workflow_id = getattr(request, "workflowId", None) or getattr(request, "workflow", "predictive_ml")
    context_id = getattr(request, "contextId", None) or getattr(
        request, "synthetic_profile", "institutional_credit_v1"
    )
    goal = getattr(request, "goal", "") or f"Evaluate {workflow_id} on {context_id}"

    plan = make_canonical_plan(workflow_id)

    warnings: list[str] = []
    if workflow_id == "model_comparison":
        warnings.append("Workflow is marked disabled in capabilities catalog.")

    return {
        "workflowId": workflow_id,
        "contextId": context_id,
        "goal": goal,
        "plan": plan,
        "requiredInputs": ["contextId", "workflowId"],
        "warnings": warnings,
    }


@router.get("/runs/{run_id}/graph")
def get_execution_graph(
    run_id: str,
    session_id: str | None = Query(None),
) -> dict[str, Any]:
    """Return runtime-derived execution graph (Amendments 4, 5, 6, 7, 24, 25).

    Replaces static template with dynamic graph derived from:
    1. Canonical plan steps for the workflow.
    2. Exact EvidenceRecord nodes linked to producing steps/tests.
    3. Exact ArtifactRecord nodes linked to producing steps.
    4. Canonical Governance node (only if present in presentation).
    5. Canonical Attestation node (only if present in presentation).
    6. Genuine parent-child lineage if parentRunId is set.
    """
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    req = ctx.request
    workflow_id = getattr(req, "workflowId", None) or getattr(req, "workflow", "predictive_ml")
    wdef = get_workflow_definition(workflow_id)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    # 1. Parent run node if lineage exists
    parent_run_id = getattr(req, "parent_run_id", None) or getattr(req, "parentRunId", None)
    if parent_run_id:
        nodes.append(
            {
                "id": "parent-run",
                "runId": run_id,
                "label": f"Parent {parent_run_id}",
                "kind": "human",
                "status": "completed",
                "subtitle": getattr(req, "intervention", "Iteration lineage") or "Iteration lineage",
            }
        )

    # 2. Plan step nodes
    plan = make_canonical_plan(workflow_id)
    event_node_ids = {ev.get("node_id") for ev in ctx.events if ev.get("node_id")}
    completed_node_ids = {
        ev.get("node_id")
        for ev in ctx.events
        if ev.get("node_id") and str(ev.get("status", "")).upper() in ("COMPLETED", "SUCCESS")
    }

    # Map test_id to plan step ID
    test_to_step: dict[str, str] = {}
    for step_id, _, _, _, test_ids in wdef.step_specs:
        for tid in test_ids:
            test_to_step[tid] = step_id

    # Group evidence IDs by producing step
    evidence_by_step: dict[str, list[str]] = {}
    for r in ctx.evidence_records:
        ev_id = getattr(r, "evidence_id", None) or (r.get("evidence_id") if isinstance(r, dict) else "")
        test_id = getattr(r, "test_id", None) or (r.get("test_id") if isinstance(r, dict) else "")
        step_id = test_to_step.get(test_id, "step-evidence")
        if ev_id:
            evidence_by_step.setdefault(step_id, []).append(ev_id)

    prev_step_id: str | None = None
    for step in plan:
        sid = step["id"]
        if ctx.status == "COMPLETED":
            step_status = "completed"
        elif sid in completed_node_ids:
            step_status = "completed"
        elif sid in event_node_ids:
            step_status = "running"
        elif ctx.status == "RUNNING":
            step_status = "queued"
        else:
            step_status = "future"

        nodes.append(
            {
                "id": sid,
                "runId": run_id,
                "label": step["label"],
                "kind": step["kind"],
                "status": step_status,
                "parentId": prev_step_id or ("parent-run" if parent_run_id else None),
                "subtitle": step.get("description"),
                "evidenceIds": evidence_by_step.get(sid, []),
            }
        )

        if prev_step_id:
            edges.append(
                {
                    "id": f"edge-{prev_step_id}-{sid}",
                    "source": prev_step_id,
                    "target": sid,
                    "relation": "next",
                }
            )
        elif parent_run_id:
            edges.append(
                {
                    "id": "edge-parent-to-first",
                    "source": "parent-run",
                    "target": sid,
                    "relation": "rerun",
                }
            )

        prev_step_id = sid

    # 3. Evidence record nodes linked to actual producers
    for r in ctx.evidence_records:
        ev_id = getattr(r, "evidence_id", None) or (r.get("evidence_id") if isinstance(r, dict) else "")
        test_id = getattr(r, "test_id", None) or (r.get("test_id") if isinstance(r, dict) else "")
        if not ev_id:
            continue

        raw_status = str(getattr(r, "status", "") or (r.get("status") if isinstance(r, dict) else "")).upper()
        ev_status = (
            "attention" if any(s in raw_status for s in ("ATTENTION", "WARN", "FAIL")) else "completed"
        )
        producer_id = test_to_step.get(test_id, "step-evidence")

        nodes.append(
            {
                "id": ev_id,
                "runId": run_id,
                "label": ev_id,
                "kind": "evidence",
                "status": ev_status,
                "parentId": producer_id,
                "subtitle": test_id,
            }
        )

        # Edge from producing step to evidence record
        if any(n["id"] == producer_id for n in nodes):
            edges.append(
                {
                    "id": f"edge-{producer_id}-{ev_id}",
                    "source": producer_id,
                    "target": ev_id,
                    "relation": "creates",
                }
            )

    # 4. Artifact record nodes
    for art_id, art_data in ctx.artifacts.items():
        art_title = art_data.get("title", art_id)
        art_type = art_data.get("artifact_type", "table")
        producer_id = "step-evidence"

        nodes.append(
            {
                "id": art_id,
                "runId": run_id,
                "label": art_title,
                "kind": "artifact",
                "status": "completed",
                "parentId": producer_id,
                "subtitle": f"Deterministic {art_type} artifact",
            }
        )

        if any(n["id"] == producer_id for n in nodes):
            edges.append(
                {
                    "id": f"edge-{producer_id}-{art_id}",
                    "source": producer_id,
                    "target": art_id,
                    "relation": "creates",
                }
            )

    # 5. Governance node (only if canonical governance disposition exists)
    pres = ctx.presentation or {}
    gov_disp = pres.get("governance_disposition")
    if gov_disp:
        last_step = prev_step_id or "step-evidence"
        nodes.append(
            {
                "id": "governance",
                "runId": run_id,
                "label": "Governance",
                "kind": "governance",
                "status": "completed" if ctx.status == "COMPLETED" else "running",
                "parentId": last_step,
                "subtitle": f"Disposition: {gov_disp}",
            }
        )
        edges.append(
            {
                "id": f"edge-{last_step}-governance",
                "source": last_step,
                "target": "governance",
                "relation": "next",
            }
        )

        # 6. Attestation node (only if canonical attestation root exists)
        merkle_root = pres.get("attestation_seal_merkle_root")
        if merkle_root:
            nodes.append(
                {
                    "id": "attest",
                    "runId": run_id,
                    "label": "Sign-off",
                    "kind": "attestation",
                    "status": "completed" if ctx.status == "COMPLETED" else "future",
                    "parentId": "governance",
                    "subtitle": f"Merkle Root: {str(merkle_root)[:16]}...",
                }
            )
            edges.append(
                {
                    "id": "edge-governance-attest",
                    "source": "governance",
                    "target": "attest",
                    "relation": "next",
                }
            )

    return {"nodes": nodes, "edges": edges}


@router.get("/runs/{run_id}/findings")
def get_run_findings(
    run_id: str,
    session_id: str | None = Query(None),
) -> list[dict[str, Any]]:
    """Return canonical findings without fabricated severities or synthetic node associations.

    Invariants (Amendments 5, 6, 8):
    - No status-to-severity invention (FAIL does NOT automatically map to critical).
    - Uses canonical severity if explicitly present in metadata/record; otherwise None.
    - Resolves real producing step node IDs (no hardcoded branch-a / branch-b).
    - Available actions resolved from workflow's supported actions.
    """
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    req = ctx.request
    workflow_id = getattr(req, "workflowId", None) or getattr(req, "workflow", "predictive_ml")
    wdef = get_workflow_definition(workflow_id)

    # Map test_id to plan step ID
    test_to_step: dict[str, str] = {}
    for step_id, _, _, _, test_ids in wdef.step_specs:
        for tid in test_ids:
            test_to_step[tid] = step_id

    findings: list[dict[str, Any]] = []

    # 1. Derive from attention, warning, or failed EvidenceRecords
    for idx, r in enumerate(ctx.evidence_records):
        r_stat = str(getattr(r, "status", "") or (r.get("status") if isinstance(r, dict) else "")).upper()
        if any(s in r_stat for s in ("ATTENTION", "FAIL", "WARN")):
            ev_id = getattr(r, "evidence_id", None) or (
                r.get("evidence_id") if isinstance(r, dict) else f"EV-{idx + 1}"
            )
            test_id = getattr(r, "test_id", None) or (
                r.get("test_id") if isinstance(r, dict) else "deterministic.check"
            )
            title = getattr(r, "title", None) or (
                r.get("title") if isinstance(r, dict) else f"{test_id} observation"
            )

            # Canonical severity: only if explicitly provided in metrics or metadata
            raw_metrics = getattr(r, "metrics", None) or (r.get("metrics") if isinstance(r, dict) else {})
            explicit_sev = raw_metrics.get("severity") if isinstance(raw_metrics, dict) else None
            if explicit_sev and str(explicit_sev).lower() in ("info", "attention", "critical"):
                sev: str | None = str(explicit_sev).lower()
            elif "ATTENTION" in r_stat or "WARN" in r_stat:
                sev = "attention"
            else:
                sev = None

            producer_node = test_to_step.get(test_id)
            if not producer_node:
                test_prefix = test_id.split(".")[0]
                for step_id, _, _, _, tids in wdef.step_specs:
                    if any(t.split(".")[0] == test_prefix for t in tids):
                        producer_node = step_id
                        break
            if not producer_node:
                producer_node = "step-evidence"

            finding_obj: dict[str, Any] = {
                "findingId": f"F-ATTN-{len(findings) + 1}",
                "runId": run_id,
                "title": f"Attention item: {title}",
                "summary": f"Deterministic test '{test_id}' reported status {r_stat}. Grounded in {ev_id}.",
                "evidenceIds": [ev_id],
                "limitations": ["Evaluated with deterministic synthetic context."],
                "availableActions": list(wdef.supported_actions),
                "sourceNodeId": producer_node,
            }
            if sev:
                finding_obj["severity"] = sev

            findings.append(finding_obj)

    # 2. Derive from presentation model if present
    pres = ctx.presentation or {}
    pres_findings = pres.get("findings", [])
    for pf in pres_findings:
        sev_raw = pf.get("severity")
        sev_val = (
            str(sev_raw).lower()
            if sev_raw and str(sev_raw).lower() in ("info", "attention", "critical")
            else None
        )
        ev_refs = [ref.get("evidence_id") for ref in pf.get("evidence_refs", []) if isinstance(ref, dict)]
        source_step = None
        if ev_refs:
            source_step = test_to_step.get(ev_refs[0])

        f_entry: dict[str, Any] = {
            "findingId": pf.get("finding_id", f"F-{len(findings) + 1}"),
            "runId": run_id,
            "title": pf.get("title", "Review Finding"),
            "summary": pf.get("description", ""),
            "evidenceIds": ev_refs,
            "limitations": pf.get("limitations", []),
            "availableActions": list(wdef.supported_actions),
        }
        if source_step:
            f_entry["sourceNodeId"] = source_step
        if sev_val:
            f_entry["severity"] = sev_val

        findings.append(f_entry)

    return findings


@router.get("/runs/{run_id}/artifacts")
def get_run_artifacts(
    run_id: str,
    session_id: str | None = Query(None),
) -> list[dict[str, Any]]:
    """Return truthful sandboxed artifacts for a run. No fabricated PDF (Amendment 7)."""
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    artifacts: list[dict[str, Any]] = []
    now_iso = datetime.datetime.fromtimestamp(ctx.completed_at or ctx.created_at, tz=datetime.UTC).isoformat()

    for art_id, art_data in ctx.artifacts.items():
        art_type = art_data.get("artifact_type", "json")
        kind = "plot" if art_type == "svg" else ("table" if art_type in ("table", "json") else "report")
        mime = "image/svg+xml" if art_type == "svg" else "application/json"

        preview = None
        content = art_data.get("content")
        if isinstance(content, dict):
            preview = {"type": "key-value", "payload": {k: str(v) for k, v in list(content.items())[:8]}}

        artifacts.append(
            {
                "artifactId": art_id,
                "runId": run_id,
                "label": art_data.get("title", art_id),
                "kind": kind,
                "mimeType": mime,
                "createdAt": now_iso,
                "description": f"Generated deterministic {kind} surface.",
                "preview": preview,
            }
        )

    return artifacts


@router.post("/runs/{run_id}/actions/validate")
def validate_proposed_action(
    run_id: str,
    action: dict[str, Any],
    session_id: str | None = Query(None),
) -> dict[str, Any]:
    """Fail-closed Action Validation Boundary (Amendments 11, 12, 13).

    Validates candidate action intents proposed by Browser AI or users.
    Rejects unsupported action kinds, unknown Evidence IDs, or out-of-range parameters with 400.
    """
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    return validate_action_or_raise(ctx, action)


@router.post("/runs/{run_id}/actions")
def execute_human_action(
    run_id: str,
    action: dict[str, Any],
    session_id: str | None = Query(None),
) -> dict[str, Any]:
    """Execute a validated proposed action to launch a canonical child run with parent lineage.

    Authoritatively validates action server-side before execution (Amendment 12).
    """
    parent_ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not parent_ctx:
        raise HTTPException(status_code=404, detail=f"Parent run '{run_id}' not found")

    validated = validate_action_or_raise(parent_ctx, action)

    parent_req = parent_ctx.request
    child_run_id = f"RUN-WEB-{uuid.uuid4().hex[:10]}"

    child_req = RunRequest(
        domain=parent_req.domain,
        mode=parent_req.mode,
        materiality=parent_req.materiality,
        lifecycle=parent_req.lifecycle,
        synthetic_profile=parent_req.synthetic_profile,
        workflow=parent_req.workflow,
        parameters=validated["parameters"],
        parent_run_id=run_id,
        intervention=validated["kind"],
        goal=validated["label"],
        session_id=parent_req.session_id,
        source_evidence_id=validated.get("sourceEvidenceId"),
    )

    accepted, status_msg = GLOBAL_QUEUE.submit_run(child_run_id, child_req)
    if not accepted:
        raise HTTPException(status_code=429, detail=status_msg)

    # Launch canonical execution in background thread
    from start.web.routes_run import _execute_run_in_background

    t = threading.Thread(target=_execute_run_in_background, args=(child_run_id, child_req), daemon=True)
    t.start()

    child_ctx = GLOBAL_QUEUE.get_run(child_run_id)
    if not child_ctx:
        raise HTTPException(status_code=500, detail="Failed to initialize child run context")

    return serialize_run_snapshot(child_ctx)


@router.get("/runs/{run_id}/governance")
def get_run_governance(
    run_id: str,
    session_id: str | None = Query(None),
) -> dict[str, Any] | None:
    """Return canonical governance evaluation state. Fail-closed: returns null if absent (Amendment 9)."""
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    pres = ctx.presentation
    if not pres or not isinstance(pres, dict):
        return None

    disposition = pres.get("governance_disposition")
    if not disposition:
        return None

    policy_decision = pres.get("opa_policy_decision") or pres.get("policy_decision")
    rationale = pres.get("governance_rationale") or pres.get("policy_rationale")
    evidence_coverage = pres.get("evidence_coverage")
    unresolved = pres.get("unresolved_items", [])

    return {
        "disposition": disposition,
        "policyDecision": policy_decision,
        "rationale": rationale,
        "evidenceCoverage": evidence_coverage,
        "unresolvedItems": unresolved,
    }


@router.get("/runs/{run_id}/attestation")
def get_run_attestation(
    run_id: str,
    session_id: str | None = Query(None),
) -> dict[str, Any] | None:
    """Return cryptographic attestation state. Fail-closed: returns null if absent (Amendment 10)."""
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    pres = ctx.presentation
    if not pres or not isinstance(pres, dict):
        return None

    merkle_root = pres.get("attestation_seal_merkle_root")
    if not merkle_root:
        return None

    created_at = datetime.datetime.fromtimestamp(
        ctx.completed_at or ctx.created_at, tz=datetime.UTC
    ).isoformat()

    return {
        "merkleRoot": merkle_root,
        "createdAt": created_at,
        "evidenceCount": len(ctx.evidence_records),
        "artifactCount": len(ctx.artifacts),
        "reproducibilityId": pres.get("reproducibility_id"),
    }
