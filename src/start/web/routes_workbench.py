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
import hashlib
import logging
import threading
import time
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from start.registry import list_tests
from start.runtime import (
    WorkflowExecutionSpec,
    get_canonical_context_specs,
    get_canonical_workflow_specs,
    resolve_workflow,
)
from start.runtime.scenarios import compute_scenario_eda, list_scenarios
from start.web.queue import GLOBAL_QUEUE, ActiveRunContext
from start.web.schemas import RunRequest

logger = logging.getLogger("start.web.routes_workbench")
router = APIRouter(prefix="/api/v1", tags=["workbench"])

# Backward-compatible alias for existing callers
WorkflowDefinition = WorkflowExecutionSpec


def _get_registered_test_ids() -> set[str]:
    return {t.test_id for t in list_tests()}


def _build_workflow_definitions() -> dict[str, WorkflowExecutionSpec]:
    return get_canonical_workflow_specs()


def get_workflow_definition(workflow_id: str) -> WorkflowExecutionSpec:
    defs = get_canonical_workflow_specs()
    alias_map = {
        "calibration_refinement": "calibration",
        "stress_testing": "robustness",
        "explainability_audit": "explainability",
    }
    norm_id = alias_map.get(workflow_id, workflow_id)
    return defs.get(norm_id, defs["predictive_ml"])


def get_workflow_test_ids(workflow_id: str) -> set[str]:
    wdef = get_workflow_definition(workflow_id)
    out: set[str] = set()
    for _, _, _, _, test_ids in wdef.step_specs:
        out.update(test_ids)
    return out


# --------------------------------------------------------------------------- #
# Canonical Plan Builder
# --------------------------------------------------------------------------- #
def make_canonical_plan(workflow_id: str, context_id: str | None = None) -> list[dict[str, Any]]:
    """Build canonical agent orchestration plan steps matching registered engines.

    Human-readable labels serve strictly as presentation metadata.
    Initial plan has zero completed nodes (all future, observed=False).
    """
    cid = context_id or (
        "institutional_market_v1"
        if workflow_id == "quantitative_finance"
        else (
            "deep_learning_v1"
            if workflow_id == "deep_learning"
            else (
                "recommender_ratings_v1"
                if workflow_id == "recommender_system"
                else "institutional_credit_v1"
            )
        )
    )

    try:
        resolved = resolve_workflow(workflow_id, cid)
        step_specs = resolved.step_specs
    except Exception:
        wdef = get_workflow_definition(workflow_id)
        step_specs = wdef.step_specs if wdef else []

    plan: list[dict[str, Any]] = []
    prev_id: str | None = None

    for step_id, label, kind, desc, _test_ids in step_specs:
        plan.append(
            {
                "id": step_id,
                "label": label,
                "description": desc,
                "kind": kind,
                "status": "future",
                "observed": False,
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
                    "percent": float(ev["percent"]) if ev.get("percent") is not None else None,
                    "completed": int(ev["completed"]) if ev.get("completed") is not None else None,
                    "total": int(ev["total"]) if ev.get("total") is not None else None,
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
    plan = make_canonical_plan(workflow_id, context_id)
    event_node_ids = {ev.get("node_id") for ev in ctx.events if ev.get("node_id")}
    completed_node_ids = {
        ev.get("node_id")
        for ev in ctx.events
        if ev.get("node_id") and str(ev.get("status", "")).upper() in ("COMPLETED", "SUCCESS")
    }

    for p in plan:
        pid = p["id"]
        if pid in completed_node_ids:
            p["status"] = "completed"
            p["observed"] = True
        elif pid in event_node_ids:
            p["status"] = "running"
            p["observed"] = True
        elif ctx.status == "RUNNING":
            p["status"] = "queued"
            p["observed"] = False
        else:
            p["status"] = "future"
            p["observed"] = False

    context_event = next(
        (e for e in ctx.events if e.get("event_type") == "context_ready"), None
    )
    context_meta = context_event.get("metadata", {}) if context_event else {}

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
        "runtimeContext": {
            "spec_id": context_meta.get("spec_id", context_id),
            "actual_samples": context_meta.get("actual_samples"),
            "actual_features": context_meta.get("actual_features"),
            "actual_assets": context_meta.get("actual_assets"),
            "actual_periods": context_meta.get("actual_periods"),
            "actual_target": context_meta.get("actual_target"),
            "actual_seed": context_meta.get("actual_seed"),
        },
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

    parent_params = getattr(req, "parameters", {}) or {}
    if kind == "change_parameter":
        delta = {k: v for k, v in validated_params.items() if parent_params.get(k) != v}
        if not delta:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Action kind 'change_parameter' requires a real parameter delta "
                    "differing from parent parameters."
                ),
            )
    elif kind == "challenge":
        raise HTTPException(
            status_code=400,
            detail=(
                "Action kind 'challenge' is conversational only and does not "
                "launch an autonomous child run."
            ),
        )

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

    for wf_id, desc in descriptions.items():
        if wf_id not in defs:
            continue
        wdef = defs[wf_id]
        entry: dict[str, Any] = {
            "id": wf_id,
            "label": wdef.label,
            "description": desc,
            "category": wdef.category,
            "enabled": wdef.enabled,
        }
        if not wdef.enabled and wdef.disabled_reason:
            entry["disabledReason"] = wdef.disabled_reason
        out.append(entry)

    return out


@router.get("/capability-manifest")
def get_capability_manifest() -> dict[str, Any]:
    """Return the authoritative, dynamic capability manifest for the StART Agentic AI Engineering Workbench.

    Exposes real execution modes, supported models, preprocessing options, tuning strategies,
    sensitivity grids, recommender algorithms, deep learning architectures, and explicit deferred items.
    """
    from start.modeling.models import MODEL_CHOICES
    from start.providers.keys import ensure_provider_key

    key_status = ensure_provider_key("openai", interactive=False)

    return {
        "workbench": {
            "name": "StART — Agentic AI Engineering Workbench",
            "tagline": "Build · Tune · Stress · Explain · Compare · Govern",
            "version": "4.0.0",
            "default_execution_mode": "hybrid_workbench",
            "deterministic_science_invariant": True,
        },
        "execution_modes": [
            {
                "id": "hybrid_workbench",
                "label": "Hybrid Workbench",
                "is_default": True,
                "description": "Generative AI plan synthesis and reasoning combined with 100% deterministic science execution engines.",
            },
            {
                "id": "agentic_session",
                "label": "Agentic Session",
                "is_default": False,
                "description": "Deliberative multi-agent committee exploration (12 specialized engineering agents).",
            },
            {
                "id": "deterministic_run",
                "label": "Deterministic Run",
                "is_default": False,
                "description": "Direct, reproducible parameterized pipeline execution with offline determinism.",
            },
        ],
        "domains": {
            "predictive_ml": {
                "name": "Predictive Machine Learning",
                "supported_models": list(MODEL_CHOICES),
                "primary_classification_models": [
                    "xgboost",
                    "lightgbm",
                    "catboost",
                    "random_forest",
                    "gradient_boosting",
                    "logistic_regression",
                    "extra_trees",
                    "mlp",
                ],
                "tuning_strategies": ["optuna_bayesian", "grid_search", "random_search", "none"],
                "data_splitters": ["stratified_kfold", "kfold", "time_series_split", "train_test_split"],
                "preprocessing_options": {
                    "outlier_mitigation": ["iqr", "zscore", "winsorize", "none"],
                    "missing_imputation": ["median", "mean", "most_frequent", "none"],
                    "feature_scaling": ["standard", "minmax", "robust", "none"],
                    "categorical_encoding": ["onehot", "target", "ordinal", "frequency", "none"],
                },
                "threshold_optimization": ["f1", "youden_j", "cost_sensitive", "custom"],
                "sensitivity_analysis": {
                    "shocks_grid": [-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30],
                    "modes": ["one_at_a_time", "parallel_basket", "both"],
                    "max_features": 5,
                    "retraining_support": True,
                },
                "explainability": ["tree_shap", "kernel_shap", "permutation_importance"],
            },
            "deep_learning": {
                "name": "Deep Learning Diagnostics",
                "sequence_models": ["rnn", "lstm", "gru", "bi_lstm"],
                "vision_models": ["simple_cnn_small", "simple_cnn_medium", "simple_cnn_deep"],
                "training_telemetry": ["loss_curves", "lr_schedulers", "early_stopping", "gradient_norms"],
            },
            "fraud_anomaly": {
                "name": "Fraud, Anomaly & AML Monitoring",
                "workflow_id": "fraud_anomaly_aml",
                "executable_techniques": ["supervised_fraud_classification"],
                "cost_optimization": ["financial_loss_matrix", "threshold_scanning"],
                "supported_contexts": ["synthetic_aml_imbalanced"],
                "deferred_techniques": [
                    "isolation_forest",
                    "one_class_svm",
                    "local_outlier_factor",
                    "autoencoder",
                    "graph_aml",
                ],
            },
            "recommender_systems": {
                "name": "Recommender Systems",
                "algorithms": [
                    "matrix_factorization",
                    "neural_collaborative_filtering",
                    "factorization_machine",
                    "field_aware_factorization_machine",
                ],
                "task_modes": ["rating_prediction", "binary_interaction", "top_k_ranking"],
                "ranking_metrics": ["ndcg_at_10", "map_at_10", "hit_rate_at_10", "mrr", "precision_at_10", "recall_at_10"],
            },
            "quantitative_finance": {
                "name": "Quantitative Finance & Portfolio Risk",
                "scenario_shocks": ["historical_replay", "hypothetical_shift", "factor_stress", "reverse_stress"],
                "portfolio_optimizers": ["hierarchical_risk_parity", "minimum_variance", "equal_risk_contribution"],
                "metrics": ["sharpe_ratio", "sortino_ratio", "max_drawdown", "tracking_error"],
            },
            "llm_agents": {
                "name": "LLM & Agent Engineering",
                "committee_roster": [
                    "Model Architect",
                    "Hyperparameter Specialist",
                    "Sensitivity & Robustness Engineer",
                    "Explainability & SHAP Specialist",
                    "Data Quality & Leakage Auditor",
                    "Benchmark & Comparison Engineer",
                    "Calibration & Uncertainty Specialist",
                    "Fairness & Bias Auditor",
                    "Stress Testing Engineer",
                    "Production Readiness & Latency Auditor",
                    "Compliance & Governance Officer",
                    "Lead Synthesis Orchestrator",
                ],
                "active_provider": "openai",
                "active_model": "gpt-5.1",
                "strict_zero_substitution": True,
            },
            "dataset_hub": {
                "name": "Dataset Hub",
                "executable_sources": ["local_synthetic_benchmark", "canonical_context_generator"],
                "contexts": [s.to_dict() for s in get_canonical_context_specs()],
                "deferred_connectors": ["kaggle", "openml", "uci", "huggingface"],
            },
            "governance": {
                "name": "Artifacts & Evidence Governance",
                "evidence_vault": True,
                "cryptographic_seals": True,
                "merkle_attestation": True,
                "sr_11_7_export": True,
            },
        },
        "deferred_capabilities": [
            {
                "capability": "Monte Carlo VaR & Expected Shortfall with Heavy-Tailed Copulas",
                "domain": "quantitative_finance",
                "status": "DEFERRED",
                "rationale": "Standard Gaussian Monte Carlo underestimates non-linear tail risk; heavy-tailed copula calibration requires high-throughput kernels not suited for single-box interactive execution.",
                "activation_requirements": "Calibrated Student-t / Gumbel copula module and GPD extreme-value tail fitting.",
            },
            {
                "capability": "Distributed Ray / Spark Cluster Training",
                "domain": "predictive_ml",
                "status": "DEFERRED",
                "rationale": "Workbench prioritizes deterministic single-node workstation execution without unmanaged infrastructure dependencies.",
                "activation_requirements": "Ray cluster client adapter and remote object store bridge.",
            },
            {
                "capability": "Unsupervised Outlier Detectors & Graph AML Pipelines",
                "domain": "fraud_anomaly",
                "status": "DEFERRED",
                "rationale": "Isolation Forest, One-Class SVM, LOF, deep autoencoders, and Graph AML are architectural specifications deferred for future release; supervised fraud classification with class weighting is currently supported.",
                "activation_requirements": "Dedicated unsupervised anomaly scoring and graph network engine.",
            },
            {
                "capability": "Remote Dataset Hub Connectors (Kaggle, OpenML, UCI, Hugging Face)",
                "domain": "dataset_hub",
                "status": "DEFERRED",
                "rationale": "Remote dataset connectors require external API credentials and network access; currently local and canonical synthetic datasets are executable.",
                "activation_requirements": "Authentication secrets and remote hub client integrations.",
            },
            {
                "capability": "Isolation Forest Outlier Row-Filtering",
                "domain": "predictive_ml",
                "status": "DEFERRED",
                "rationale": "Isolation Forest outlier removal filters/drops rows which alters sample sizes; continuous bound clipping (IQR, Z-Score, Winsorize) is supported.",
                "activation_requirements": "Row-filtering pipeline integration with dataset shape recalculation.",
            },
            {
                "capability": "Online Streaming Drift Detection & Real-Time Feature Store",
                "domain": "monitoring",
                "status": "DEFERRED",
                "rationale": "Focuses on development, stress-testing, and governance rather than continuous real-time streaming telemetry.",
                "activation_requirements": "Continuous Kafka/Flink ingestion harness and streaming Kolmogorov-Smirnov detector.",
            },
        ],
        "ai_provider": {
            "provider": "openai",
            "model": "gpt-5.1",
            "status": "ONLINE" if key_status.ok else "OFFLINE",
            "source": key_status.source,
            "strict_zero_substitution": True,
        },
    }


@router.get("/execution-contexts")
def get_execution_contexts() -> list[dict[str, Any]]:
    """Return versioned public-safe execution contexts."""
    return [s.to_dict() for s in get_canonical_context_specs()]


@router.get("/scenarios")
def get_scenarios() -> list[dict[str, Any]]:
    """Return all mechanical built-in scenario specifications."""
    return [s.to_dict() for s in list_scenarios()]


@router.get("/scenarios/{scenario_id}/eda")
def get_scenario_eda(scenario_id: str) -> dict[str, Any]:
    """Compute bounded deterministic descriptive EDA for a scenario."""
    try:
        return compute_scenario_eda(scenario_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("EDA computation failed for scenario %s: %s", scenario_id, exc)
        raise HTTPException(status_code=500, detail=f"EDA computation failed: {exc}")


@router.get("/execution-contexts/{context_id}/eda")
def get_context_eda(context_id: str) -> dict[str, Any]:
    """Compute bounded deterministic descriptive EDA for a canonical context."""
    try:
        return compute_scenario_eda(context_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("EDA computation failed for context %s: %s", context_id, exc)
        raise HTTPException(status_code=500, detail=f"EDA computation failed: {exc}")


@router.get("/tests")
def get_test_catalog() -> list[dict[str, Any]]:
    """Return authoritative 79-test catalog directly from canonical registry."""
    all_tests = list_tests()
    out = []
    # Canonical domain derivation from TestSpec.context_type
    # (the review applicability contract), NOT from risk_stripes or
    # scenario data-compatibility.
    _CONTEXT_TO_DOMAIN = {
        "tabular": "predictive_ml",
        "market": "quantitative_finance",
        "short_rate": "treasury",
    }

    for t in all_tests:
        ctx_type = getattr(t, "context_type", "tabular")
        domain = _CONTEXT_TO_DOMAIN.get(ctx_type, "predictive_ml")

        out.append({
            "testId": t.test_id,
            "name": t.name,
            "family": t.family,
            "domain": domain,
            "description": getattr(t, "description", ""),
            "contextType": getattr(t, "context_type", "tabular"),
            "riskStripes": list(getattr(t, "risk_stripes", ())),
            "riskDimensions": list(getattr(t, "risk_dimensions", ())),
            "requires": list(getattr(t, "requires", ())),
            "objectKinds": list(getattr(t, "object_kinds", ())),
        })
    return out


@router.post("/plans")
@router.post("/plan/generate")
def create_agent_plan(request: RunRequest) -> dict[str, Any]:
    """Generate dedicated AgentPlanPreview without creating a run or fabricating fake run IDs."""
    workflow_id = getattr(request, "workflowId", None) or getattr(request, "workflow_id", None) or getattr(request, "workflow", None)
    if not workflow_id and request.parameters:
        workflow_id = request.parameters.get("workflow_id") or request.parameters.get("workflowId") or request.parameters.get("workflow")
    if not workflow_id:
        workflow_id = "predictive_ml"

    context_id = getattr(request, "contextId", None) or getattr(request, "context_id", None)
    if not context_id and request.parameters:
        context_id = request.parameters.get("dataset_id") or request.parameters.get("dataset") or request.parameters.get("context_id") or request.parameters.get("contextId") or request.parameters.get("context")
    if not context_id:
        context_id = getattr(request, "synthetic_profile", None)
    if not context_id:
        context_id = "institutional_credit_v1"

    goal = getattr(request, "goal", "") or f"Evaluate {workflow_id} on {context_id}"
    exec_mode = getattr(request, "execution_mode", None) or getattr(request, "executionMode", "hybrid_workbench")

    plan = make_canonical_plan(workflow_id, context_id)

    warnings: list[str] = []
    if workflow_id == "model_comparison":
        warnings.append("Workflow is marked disabled in capabilities catalog.")

    agent_proposal = None
    if exec_mode != "deterministic_run":
        agent_proposal = {
            "mode": exec_mode,
            "recommended_estimator": "lightgbm" if workflow_id == "predictive_ml" else ("mlp" if workflow_id == "deep_learning" else None),
            "rationale": f"Plan configured for {goal} under {exec_mode} mode.",
            "tuning_recommendation": "Bayesian / Optuna 5-trial exploration" if exec_mode == "agentic_session" else None,
        }

    return {
        "workflowId": workflow_id,
        "contextId": context_id,
        "goal": goal,
        "plan": plan,
        "executionMode": exec_mode,
        "agentProposal": agent_proposal,
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

    # Group evidence IDs by producing step using actual runtime event evidence_refs
    evidence_by_step: dict[str, list[str]] = {}
    for ev in ctx.events:
        nid = ev.get("node_id")
        for eid in ev.get("evidence_refs", []):
            if nid and eid:
                evidence_by_step.setdefault(nid, []).append(eid)

    prev_step_id: str | None = None
    observed_step_ids: set[str] = set()

    for step in plan:
        sid = step["id"]
        if sid in completed_node_ids:
            step_status = "completed"
            observed = True
            observed_step_ids.add(sid)
        elif sid in event_node_ids:
            step_status = "running"
            observed = True
            observed_step_ids.add(sid)
        elif ctx.status == "RUNNING":
            step_status = "queued"
            observed = False
        else:
            step_status = "future"
            observed = False

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
                "observed": observed,
            }
        )

        # Planned edges represent intended design flow
        if prev_step_id:
            edges.append(
                {
                    "id": f"edge-plan-{prev_step_id}-{sid}",
                    "source": prev_step_id,
                    "target": sid,
                    "relation": "next",
                    "edgeKind": "planned",
                }
            )

        prev_step_id = sid

    # Parent-child run lineage edge (NOT an observed runtime event edge)
    seen_obs_edges: set[tuple[str, str]] = set()
    if parent_run_id and observed_step_ids:
        first_obs = next((s["id"] for s in plan if s["id"] in observed_step_ids), None)
        if first_obs:
            edges.append(
                {
                    "id": "edge-parent-to-first",
                    "source": "parent-run",
                    "target": first_obs,
                    "relation": "rerun",
                    "edgeKind": "lineage",
                }
            )

    for ev in ctx.events:
        nid = ev.get("node_id")
        pnid = ev.get("parent_node_id")
        if pnid and nid and pnid != nid:
            edge_key = (pnid, nid)
            if (
                edge_key not in seen_obs_edges
                and any(n["id"] == pnid for n in nodes)
                and any(n["id"] == nid for n in nodes)
            ):
                seen_obs_edges.add(edge_key)
                edges.append(
                    {
                        "id": f"edge-obs-{pnid}-{nid}",
                        "source": pnid,
                        "target": nid,
                        "relation": "next",
                        "edgeKind": "observed",
                    }
                )

    # 3. Evidence record nodes: producer provenance from RuntimeEvent.evidence_refs ONLY
    # Build evidence_id → producing node_id map from actual runtime events
    evidence_producer: dict[str, str] = {}
    for ev in ctx.events:
        nid = ev.get("node_id")
        if nid:
            for eid in ev.get("evidence_refs", []):
                if eid and eid not in evidence_producer:
                    evidence_producer[eid] = nid

    for r in ctx.evidence_records:
        ev_id = getattr(r, "evidence_id", None) or (r.get("evidence_id") if isinstance(r, dict) else "")
        if not ev_id:
            continue

        raw_status = str(getattr(r, "status", "") or (r.get("status") if isinstance(r, dict) else "")).upper()
        ev_status = (
            "attention" if any(s in raw_status for s in ("ATTENTION", "WARN", "FAIL")) else "completed"
        )
        producer_id = evidence_producer.get(ev_id)

        nodes.append(
            {
                "id": ev_id,
                "runId": run_id,
                "label": ev_id,
                "kind": "evidence",
                "status": ev_status,
                "parentId": producer_id,
                "subtitle": f"Producer: {producer_id}" if producer_id else "Producer lineage unavailable",
                "observed": True,
            }
        )

        # Edge from producing step to evidence record only if canonical event references it
        if producer_id and any(n["id"] == producer_id for n in nodes):
            edges.append(
                {
                    "id": f"edge-{producer_id}-{ev_id}",
                    "source": producer_id,
                    "target": ev_id,
                    "relation": "creates",
                    "edgeKind": "observed",
                }
            )

    # 4. Artifact record nodes
    artifact_producers: dict[str, str] = {}
    for ev in ctx.events:
        nid = ev.get("node_id")
        if nid:
            for aid in (
                ev.get("artifact_refs", [])
                or ev.get("artifactIds", [])
                or ev.get("emitted_artifact_ids", [])
            ):
                artifact_producers[aid] = nid

    for art_id, art_data in ctx.artifacts.items():
        art_title = art_data.get("title", art_id)
        art_type = art_data.get("artifact_type", "table")
        producer_id = artifact_producers.get(art_id) or art_data.get("producer_node")

        nodes.append(
            {
                "id": art_id,
                "runId": run_id,
                "label": art_title,
                "kind": "artifact",
                "status": "completed",
                "parentId": producer_id,
                "subtitle": (
                    f"Deterministic {art_type} artifact" if producer_id else "Producer lineage unavailable"
                ),
                "observed": True,
            }
        )

        if producer_id and any(n["id"] == producer_id for n in nodes):
            edges.append(
                {
                    "id": f"edge-{producer_id}-{art_id}",
                    "source": producer_id,
                    "target": art_id,
                    "relation": "creates",
                    "edgeKind": "observed",
                }
            )

    # 5. Governance node (only if canonical governance disposition exists)
    pres = ctx.presentation or {}
    gov_disp = pres.get("governance_disposition")
    has_gov_step = any(s["id"] == "step-governance" for s in plan)
    if gov_disp:
        if has_gov_step:
            gov_node_id = "step-governance"
        else:
            gov_node_id = "governance"
            last_step = prev_step_id or "step-evidence"
            nodes.append(
                {
                    "id": "governance",
                    "runId": run_id,
                    "label": "Governance",
                    "kind": "governance",
                    "status": "completed",
                    "parentId": last_step,
                    "subtitle": f"Disposition: {gov_disp}",
                    "observed": True,
                }
            )
            edges.append(
                {
                    "id": f"edge-{last_step}-governance",
                    "source": last_step,
                    "target": "governance",
                    "relation": "next",
                    "edgeKind": "observed",
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
                    "status": "completed",
                    "parentId": gov_node_id,
                    "subtitle": f"Merkle Root: {str(merkle_root)[:16]}...",
                    "observed": True,
                }
            )
            edges.append(
                {
                    "id": f"edge-{gov_node_id}-attest",
                    "source": gov_node_id,
                    "target": "attest",
                    "relation": "next",
                    "edgeKind": "observed",
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
            raw_metadata = getattr(r, "metadata", None) or (r.get("metadata") if isinstance(r, dict) else {})
            explicit_sev = (
                (raw_metrics.get("severity") if isinstance(raw_metrics, dict) else None)
                or (raw_metadata.get("severity") if isinstance(raw_metadata, dict) else None)
            )
            if explicit_sev and str(explicit_sev).lower() in ("info", "attention", "critical"):
                sev: str | None = str(explicit_sev).lower()
            else:
                sev = None

            producer_node = test_to_step.get(test_id)

            finding_obj: dict[str, Any] = {
                "findingId": f"F-ATTN-{len(findings) + 1}",
                "runId": run_id,
                "title": f"Attention item: {title}",
                "summary": f"Deterministic test '{test_id}' reported status {r_stat}. Grounded in {ev_id}.",
                "evidenceIds": [ev_id],
                "testStatus": r_stat,
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
            first_ev_id = ev_refs[0]
            target_ev = next(
                (
                    r
                    for r in ctx.evidence_records
                    if (
                        getattr(r, "evidence_id", None)
                        or (r.get("evidence_id") if isinstance(r, dict) else "")
                    )
                    == first_ev_id
                ),
                None,
            )
            if target_ev:
                tid = getattr(target_ev, "test_id", None) or (
                    target_ev.get("test_id") if isinstance(target_ev, dict) else ""
                )
                source_step = test_to_step.get(tid)

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
@router.get("/workbench/runs/{run_id}/artifacts")
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
        art_title = art_data.get("title") or art_data.get("name") or art_id
        kind = art_data.get("kind") or ("plot" if art_type == "svg" or "plot" in art_type else ("table" if art_type in ("table", "json", "summary_table", "diagnostic_table") else "metric"))
        mime = art_data.get("mimeType") or ("image/svg+xml" if art_type == "svg" else "application/json")
        producer_step = art_data.get("producing_step_id") or art_data.get("producer_node")
        content = art_data.get("content")
        ev_ids = art_data.get("evidence_ids", [])
        data_fp = art_data.get("data_fingerprint", "")

        preview = None
        if isinstance(content, dict):
            preview = {"type": "key-value", "payload": {k: str(v) for k, v in list(content.items())[:8]}}

        artifacts.append(
            {
                "artifactId": art_id,
                "runId": run_id,
                "label": art_title,
                "title": art_title,
                "kind": kind,
                "artifactType": art_type,
                "mimeType": mime,
                "createdAt": now_iso,
                "description": art_data.get("description", f"Generated deterministic {kind} surface."),
                "preview": preview,
                "content": content,
                "evidenceIds": ev_ids,
                "dataFingerprint": data_fp,
                "producerNodeId": producer_step,
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

    wdef = get_workflow_definition(child_req.workflow or "predictive_ml")
    resolved_test_ids = [tid for _, _, _, _, tids in wdef.step_specs for tid in tids]
    param_delta = {
        k: v
        for k, v in validated["parameters"].items()
        if getattr(parent_req, "parameters", {}).get(k) != v
    }

    snapshot = serialize_run_snapshot(child_ctx)
    snapshot.update(
        {
            "child_run_id": child_run_id,
            "parent_run_id": run_id,
            "action_kind": validated["kind"],
            "resolved_test_ids": resolved_test_ids,
            "resolved_parameter_delta": param_delta,
            "source_evidence_id": validated.get("sourceEvidenceId"),
            "source_node_id": validated.get("sourceNodeId"),
        }
    )
    return snapshot


@router.get("/runs/{run_id}/governance")
@router.get("/workbench/runs/{run_id}/governance")
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
@router.get("/workbench/runs/{run_id}/attestation")
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


@router.get("/runs/{run_id}/milestones")
@router.get("/runs/{run_id}/checkpoints")
@router.get("/workbench/runs/{run_id}/milestones")
@router.get("/workbench/runs/{run_id}/checkpoints")
def get_run_checkpoints(
    run_id: str,
    session_id: str | None = Query(None),
) -> list[dict[str, Any]]:
    """Return verified checkpoints committed during the run lifecycle."""
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    if ctx.checkpoints:
        return ctx.checkpoints

    # Fallback to deriving from runtime events
    checkpoints: list[dict[str, Any]] = []
    seen = set()
    for ev in ctx.events:
        if ev.get("event_type") == "checkpoint_committed" or ev.get("checkpoint_id"):
            cid = ev.get("checkpoint_id") or (ev.get("metadata", {}).get("checkpoint_id"))
            if cid and cid not in seen:
                seen.add(cid)
                meta = ev.get("metadata", {})
                checkpoints.append(
                    {
                        "checkpoint_id": cid,
                        "name": meta.get("name", ev.get("action", cid)),
                        "status": meta.get("status", "completed"),
                        "producing_stage": meta.get("producing_stage", ev.get("node_id")),
                        "agent_signature": meta.get("agent_signature", ev.get("source_agent", "DeterministicEngine")),
                        "evidence_ids": meta.get("evidence_ids", ev.get("evidence_refs", [])),
                        "commit_hash": meta.get("commit_hash", ""),
                        "timestamp": meta.get("timestamp", datetime.datetime.fromtimestamp(ev.get("timestamp", time.time()), tz=datetime.UTC).isoformat()),
                        "summary": meta.get("summary", ev.get("message", "")),
                    }
                )
    return checkpoints


@router.post("/runs/{run_id}/decisions")
def record_human_decision(
    run_id: str,
    payload: dict[str, Any],
    session_id: str | None = Query(None),
) -> dict[str, Any]:
    """Record a human decision with cryptographic receipt while preserving evidence immutability."""
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    action = str(payload.get("action", "")).upper()
    valid_actions = {"ACCEPT", "QUESTION", "CHALLENGE", "OVERRIDE", "RERUN", "ESCALATE"}
    if action not in valid_actions:
        raise HTTPException(status_code=400, detail=f"Invalid decision action '{action}'. Valid: {valid_actions}")

    target_stage = payload.get("target_stage") or payload.get("targetStage")
    target_checkpoint = payload.get("target_checkpoint") or payload.get("targetCheckpoint")
    evidence_ids = payload.get("evidence_ids") or payload.get("evidenceIds") or []
    rationale = payload.get("rationale") or payload.get("message") or f"Human action {action} recorded."
    override_value = payload.get("override_value") or payload.get("overrideValue")
    author = payload.get("author") or "Risk Officer"

    receipt_id = f"REC-{uuid.uuid4().hex[:8].upper()}"
    timestamp_str = datetime.datetime.fromtimestamp(time.time(), tz=datetime.UTC).isoformat()

    # Cryptographic hash of receipt preserving immutable audit chain
    hash_material = f"{receipt_id}:{run_id}:{action}:{target_stage}:{target_checkpoint}:{','.join(str(e) for e in evidence_ids)}:{rationale}:{timestamp_str}"
    decision_hash = hashlib.sha256(hash_material.encode("utf-8")).hexdigest()

    receipt = {
        "receipt_id": receipt_id,
        "receiptId": receipt_id,
        "run_id": run_id,
        "runId": run_id,
        "action": action,
        "target_stage": target_stage,
        "targetStage": target_stage,
        "target_checkpoint": target_checkpoint,
        "targetCheckpoint": target_checkpoint,
        "evidence_ids": evidence_ids,
        "evidenceIds": evidence_ids,
        "rationale": rationale,
        "override_value": override_value,
        "overrideValue": override_value,
        "author": author,
        "decision_hash": decision_hash,
        "decisionHash": decision_hash,
        "timestamp": timestamp_str,
        "status": "RECORDED",
        "immutable_evidence_preserved": True,
    }

    ctx.decisions.append(receipt)

    # Update canonical governance disposition in presentation model if applicable
    if isinstance(ctx.presentation, dict):
        if action == "ACCEPT":
            ctx.presentation["governance_disposition"] = "APPROVED"
        elif action == "ESCALATE":
            ctx.presentation["governance_disposition"] = "ESCALATED"
        elif action == "CHALLENGE":
            ctx.presentation["governance_disposition"] = "CHALLENGED_PENDING_REVIEW"

    # Emit human_decision event into run event log
    evt = {
        "event_id": f"EVT-{receipt_id}",
        "run_id": run_id,
        "event_type": "human_decision",
        "status": "COMPLETED",
        "source_agent": "HumanReviewer",
        "target_agent": "ModelGovernance",
        "stage": "HUMAN_CONTROL",
        "action": f"decision_{action.lower()}",
        "node_id": target_stage or "human-control",
        "timestamp": time.time(),
        "elapsed_seconds": round(time.time() - (ctx.started_at or ctx.created_at), 2),
        "message": f"Human {action} recorded: {rationale[:60]}... (Receipt {receipt_id})",
        "evidence_refs": evidence_ids,
        "metadata": receipt,
    }
    ctx.events.append(evt)

    return receipt


@router.post("/runs/{run_id}/question")
def ask_evidence_question(
    run_id: str,
    payload: dict[str, Any],
    session_id: str | None = Query(None),
) -> dict[str, Any]:
    """Answer evidence-grounded questions using genuine EvidenceRecords and the OpenAI provider."""
    import os
    import re

    from start.core.config import LLMConfig, load_config
    from start.providers.keys import ensure_provider_key
    from start.providers.llm import get_llm_provider

    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Missing question text")

    target_stage = payload.get("target_stage") or payload.get("targetStage")
    target_evidence_id = payload.get("evidence_id") or payload.get("evidenceId")

    matched_records: list[Any] = []
    if target_evidence_id:
        for r in ctx.evidence_records:
            eid = getattr(r, "evidence_id", None) or (r.get("evidence_id") if isinstance(r, dict) else "")
            if eid == target_evidence_id:
                matched_records.append(r)
                break

    if not matched_records and target_stage:
        wdef = get_workflow_definition(getattr(ctx.request, "workflowId", None) or getattr(ctx.request, "workflow", "predictive_ml"))
        stage_test_ids = set()
        for sid, _, _, _, tids in wdef.step_specs:
            if sid == target_stage:
                stage_test_ids.update(tids)
        for r in ctx.evidence_records:
            tid = getattr(r, "test_id", None) or (r.get("test_id") if isinstance(r, dict) else "")
            if tid in stage_test_ids:
                matched_records.append(r)

    # 1. Build authoritative evidence context strictly from immutable EvidenceRecords
    evidence_lines: list[str] = []
    evidence_map: dict[str, Any] = {}
    for r in ctx.evidence_records:
        eid = getattr(r, "evidence_id", None) or (r.get("evidence_id") if isinstance(r, dict) else "")
        tid = getattr(r, "test_id", None) or (r.get("test_id") if isinstance(r, dict) else "")
        stat = str(getattr(r, "status", "") or (r.get("status") if isinstance(r, dict) else "UNKNOWN")).upper()
        metrics = getattr(r, "metrics", {}) or (r.get("metrics") if isinstance(r, dict) else {})
        interp = getattr(r, "interpretation", "") or (r.get("interpretation") if isinstance(r, dict) else "")
        stage = getattr(r, "parent_node_id", None) or (r.get("parent_node_id") if isinstance(r, dict) else "")
        if eid:
            evidence_map[eid] = r
            metric_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(metrics.items())[:6])
            evidence_lines.append(
                f"- [{eid}] (Test: {tid}, Stage: {stage or 'N/A'}, Status: {stat})\n"
                f"  Metrics: {metric_str}\n"
                f"  Interpretation: {interp or 'None'}"
            )

    evidence_context = "\n".join(evidence_lines)

    system_prompt = (
        "You are the StART (Scientific Testing, Attestation, and Review Topology) Evidence Reviewer. "
        "Your mandate is to provide rigorous, evidence-grounded review of empirical validation results. "
        "\n\nSTRICT GOVERNANCE INVARIANTS:"
        "\n1. Rely strictly and exclusively on the provided EvidenceRecord universe."
        "\n2. You must NEVER invent, calculate, extrapolate, or estimate any new numerical values, metrics, thresholds, or conclusions."
        "\n3. All numerical values must come verbatim from the supplied EvidenceRecords."
        "\n4. You MUST cite the exact Evidence ID in brackets (e.g. [EV-01-...]) for every finding, assertion, or observation."
        "\n5. Explicitly identify and distinguish negative or conditional findings (status FAIL or WARN) from passing tests (PASS)."
        "\n6. If challenging a finding, evaluate whether conclusions are supported by cited records without inventing criteria."
    )

    user_prompt = (
        f"RUN IDENTIFIER: {run_id}\n"
        f"FOCUSED EVIDENCE: {target_evidence_id or 'None (Whole-Run Context)'}\n"
        f"FOCUSED STAGE: {target_stage or 'None'}\n\n"
        f"AVAILABLE IMMUTABLE EVIDENCERECORDS ({len(ctx.evidence_records)} total):\n"
        f"{evidence_context}\n\n"
        f"REVIEWER INSTRUCTION / QUERY:\n"
        f"{question}"
    )

    # 2. Resolve credentials safely without exposing key
    key_status = ensure_provider_key("openai", interactive=False)

    # 3. Resolve LLM provider with configured model
    cfg = load_config()
    raw_prov = os.environ.get("START_LLM__PROVIDER")
    provider_name = raw_prov if raw_prov is not None else (cfg.llm.provider or "openai")
    if not provider_name:
        provider_name = "openai"
    model_name = os.environ.get("START_LLM__MODEL") or (cfg.llm.model if cfg.llm.model and cfg.llm.model not in ("gpt-5-mini", "none") else "gpt-5.1")

    prov_cfg = LLMConfig(provider=provider_name, model=model_name)
    provider = get_llm_provider(prov_cfg)

    # 4. Fail closed if provider unavailable
    if not provider.available or provider.name == "none":
        logger.error("OpenAI provider unavailable (key status source: %s)", key_status.source)
        raise HTTPException(
            status_code=503,
            detail=f"OpenAI provider unavailable: credential missing or provider not ready (source: {key_status.source})",
        )

    # 5. Execute live OpenAI Responses API call
    try:
        res = provider.complete_result(system=system_prompt, user=user_prompt, output_token_budget=2048)
    except Exception as exc:
        logger.error("OpenAI request failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {exc}")

    if res.status == "error":
        logger.error("OpenAI returned error: %s (%s)", res.error_type, res.error_message)
        raise HTTPException(status_code=502, detail=f"OpenAI error: {res.error_message or res.error_type}")

    if not res.text or not res.text.strip():
        logger.error("OpenAI returned empty text (refusal=%s)", res.refusal)
        raise HTTPException(status_code=502, detail="OpenAI returned empty text response")

    full_answer = res.text.strip()
    logger.info(
        "OpenAI completion success: response_id=%s, model=%s, latency=%.2fs",
        res.response_id,
        res.model,
        res.latency_seconds,
    )

    # 6. Extract cited evidence IDs from response and match to canonical universe
    found_eids = set(re.findall(r"EV-[A-Za-z0-9_-]+", full_answer))
    if target_evidence_id:
        found_eids.add(target_evidence_id)

    citations: list[dict[str, Any]] = []
    for eid in sorted(found_eids):
        if eid in evidence_map:
            r = evidence_map[eid]
            tid = getattr(r, "test_id", None) or (r.get("test_id") if isinstance(r, dict) else "test")
            stat = str(getattr(r, "status", "") or (r.get("status") if isinstance(r, dict) else "UNKNOWN")).upper()
            metrics = getattr(r, "metrics", {}) or (r.get("metrics") if isinstance(r, dict) else {})
            interp = getattr(r, "interpretation", "") or (r.get("interpretation") if isinstance(r, dict) else "")
            citations.append({
                "evidence_id": eid,
                "evidenceId": eid,
                "test_id": tid,
                "testId": tid,
                "status": stat,
                "metrics": metrics,
                "snippet": f"{tid} reported {stat}. {interp}",
            })

    if not citations and matched_records:
        for r in matched_records[:3]:
            eid = getattr(r, "evidence_id", None) or (r.get("evidence_id") if isinstance(r, dict) else "EV")
            tid = getattr(r, "test_id", None) or (r.get("test_id") if isinstance(r, dict) else "test")
            stat = str(getattr(r, "status", "") or (r.get("status") if isinstance(r, dict) else "UNKNOWN")).upper()
            metrics = getattr(r, "metrics", {}) or (r.get("metrics") if isinstance(r, dict) else {})
            interp = getattr(r, "interpretation", "") or (r.get("interpretation") if isinstance(r, dict) else "")
            citations.append({
                "evidence_id": eid,
                "evidenceId": eid,
                "test_id": tid,
                "testId": tid,
                "status": stat,
                "metrics": metrics,
                "snippet": f"{tid} reported {stat}. {interp}",
            })

    # 7. Record decision receipt in audit trail
    receipt_id = f"REC-Q-{uuid.uuid4().hex[:6].upper()}"
    timestamp_str = datetime.datetime.fromtimestamp(time.time(), tz=datetime.UTC).isoformat()
    action_type = "CHALLENGE" if "challenge" in question.lower() else "QUESTION"
    receipt = {
        "receipt_id": receipt_id,
        "receiptId": receipt_id,
        "run_id": run_id,
        "runId": run_id,
        "action": action_type,
        "target_stage": target_stage,
        "targetStage": target_stage,
        "evidence_ids": [c["evidence_id"] for c in citations],
        "evidenceIds": [c["evidence_id"] for c in citations],
        "rationale": question,
        "author": "Risk Officer",
        "timestamp": timestamp_str,
        "status": "ANSWERED",
        "provider": "OpenAI",
        "model": res.model or model_name,
        "response_id": res.response_id,
    }
    ctx.decisions.append(receipt)

    return {
        "question": question,
        "answer": full_answer,
        "citations": citations,
        "targetStage": target_stage,
        "targetEvidenceId": target_evidence_id,
        "receipt": receipt,
        "timestamp": timestamp_str,
        "provider": "OpenAI",
        "model": res.model or model_name,
        "response_id": res.response_id,
        "responseId": res.response_id,
        "latency_seconds": res.latency_seconds,
    }


# --------------------------------------------------------------------------- #
# Phase 3: History, Compare, Lineage, and Search Endpoints
# --------------------------------------------------------------------------- #
@router.get("/workbench/runs")
def list_workbench_runs_history(
    workflow: str | None = Query(None),
    status: str | None = Query(None),
    domain: str | None = Query(None),
) -> list[dict[str, Any]]:
    """Return historical runs summary list for workbench history browser."""
    history = GLOBAL_QUEUE.get_run_history()
    if workflow:
        history = [h for h in history if h.get("workflow") == workflow]
    if status:
        history = [h for h in history if h.get("status") == status.lower()]
    if domain:
        history = [h for h in history if h.get("domain") == domain]
    return history


@router.get("/runs/{run_id}/lineage")
@router.get("/workbench/runs/{run_id}/lineage")
def get_run_lineage(
    run_id: str,
    session_id: str | None = Query(None),
) -> dict[str, Any]:
    """Return parent/child run lineage and parameter override delta."""
    ctx = GLOBAL_QUEUE.get_run(run_id, session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    req = ctx.request
    parent_run_id = getattr(req, "parent_run_id", None) or getattr(req, "parentRunId", None)
    intervention = getattr(req, "intervention", None)
    params = getattr(req, "parameters", {}) or {}

    # Find children
    all_runs = GLOBAL_QUEUE.list_runs()
    children = []
    for r in all_runs:
        r_parent = getattr(r.request, "parent_run_id", None) or getattr(r.request, "parentRunId", None)
        if r_parent == run_id:
            children.append(
                {
                    "runId": r.run_id,
                    "createdAt": r.created_at,
                    "status": r.status.lower(),
                    "intervention": getattr(r.request, "intervention", None),
                    "parameters": getattr(r.request, "parameters", {}) or {},
                }
            )

    param_delta = {}
    if parent_run_id:
        parent_ctx = GLOBAL_QUEUE.get_run(parent_run_id)
        if parent_ctx:
            parent_params = getattr(parent_ctx.request, "parameters", {}) or {}
            for k, v in params.items():
                if parent_params.get(k) != v:
                    param_delta[k] = {"parent": parent_params.get(k), "child": v}

    return {
        "runId": run_id,
        "parentRunId": parent_run_id,
        "intervention": intervention,
        "parameterDelta": param_delta,
        "children": children,
    }


@router.get("/compare")
@router.get("/workbench/compare")
def compare_runs(
    run_a: str | None = Query(None, alias="run_a"),
    run_b: str | None = Query(None, alias="run_b"),
    runA: str | None = Query(None, alias="runA"),
    runB: str | None = Query(None, alias="runB"),
    session_id: str | None = Query(None),
) -> dict[str, Any]:
    """Authoritative deterministic comparison of two execution runs."""
    target_a = run_a or runA
    target_b = run_b or runB
    if not target_a or not target_b:
        raise HTTPException(status_code=422, detail="Both run_a (runA) and run_b (runB) are required")
    run_a = target_a
    run_b = target_b

    ctx_a = GLOBAL_QUEUE.get_run(run_a, session_id)
    if not ctx_a:
        raise HTTPException(status_code=404, detail=f"Run '{run_a}' not found")

    ctx_b = GLOBAL_QUEUE.get_run(run_b, session_id)
    if not ctx_b:
        raise HTTPException(status_code=404, detail=f"Run '{run_b}' not found")

    req_a = ctx_a.request
    req_b = ctx_b.request

    wf_a = getattr(req_a, "workflowId", None) or getattr(req_a, "workflow", "predictive_ml")
    wf_b = getattr(req_b, "workflowId", None) or getattr(req_b, "workflow", "predictive_ml")
    dom_a = getattr(req_a, "domain", "predictive")
    dom_b = getattr(req_b, "domain", "predictive")
    ctx_id_a = getattr(req_a, "contextId", None) or getattr(req_a, "synthetic_profile", "")
    ctx_id_b = getattr(req_b, "contextId", None) or getattr(req_b, "synthetic_profile", "")

    # Scientific Compatibility Enforcement
    if wf_a != wf_b:
        return {
            "compatible": False,
            "incompatibleReason": f"Incompatible workflows: Cannot compare '{wf_a}' ({dom_a}) with '{wf_b}' ({dom_b}).",
            "runA": {"runId": run_a, "workflow": wf_a, "domain": dom_a, "contextId": ctx_id_a},
            "runB": {"runId": run_b, "workflow": wf_b, "domain": dom_b, "contextId": ctx_id_b},
        }

    if dom_a != dom_b:
        return {
            "compatible": False,
            "incompatibleReason": f"Incompatible domains: Cannot compare domain '{dom_a}' with '{dom_b}'.",
            "runA": {"runId": run_a, "workflow": wf_a, "domain": dom_a, "contextId": ctx_id_a},
            "runB": {"runId": run_b, "workflow": wf_b, "domain": dom_b, "contextId": ctx_id_b},
        }

    if ctx_id_a and ctx_id_b and ctx_id_a != ctx_id_b:
        return {
            "compatible": False,
            "incompatibleReason": f"Incompatible execution contexts: Cannot compare context '{ctx_id_a}' with '{ctx_id_b}'.",
            "runA": {"runId": run_a, "workflow": wf_a, "domain": dom_a, "contextId": ctx_id_a},
            "runB": {"runId": run_b, "workflow": wf_b, "domain": dom_b, "contextId": ctx_id_b},
        }

    # Metadata & Parameter comparison
    params_a = getattr(req_a, "parameters", {}) or {}
    params_b = getattr(req_b, "parameters", {}) or {}
    all_param_keys = sorted(set(params_a.keys()) | set(params_b.keys()))
    param_diffs = []
    for k in all_param_keys:
        va = params_a.get(k)
        vb = params_b.get(k)
        param_diffs.append({
            "param": k,
            "valA": va,
            "valB": vb,
            "changed": va != vb,
        })

    is_lineage = (
        getattr(req_b, "parent_run_id", None) == run_a
        or getattr(req_a, "parent_run_id", None) == run_b
    )

    # Data fingerprint comparison
    def get_fp(ctx: ActiveRunContext) -> str:
        for ev in ctx.events:
            if ev.get("event_type") == "context_ready":
                fp = (ev.get("metadata") or {}).get("data_fingerprint")
                if fp:
                    return str(fp)
        for art in ctx.artifacts.values():
            fp = art.get("data_fingerprint")
            if fp:
                return str(fp)
        return ""

    fp_a = get_fp(ctx_a)
    fp_b = get_fp(ctx_b)

    # Test & Evidence Outcomes matrix
    def extract_evidence(ctx: ActiveRunContext) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for r in ctx.evidence_records:
            eid = getattr(r, "evidence_id", None) or (r.get("evidence_id") if isinstance(r, dict) else "")
            tid = getattr(r, "test_id", None) or (r.get("test_id") if isinstance(r, dict) else "")
            stat = getattr(r, "status", None) or (r.get("status") if isinstance(r, dict) else "UNKNOWN")
            m = getattr(r, "metrics", None) or (r.get("metrics") if isinstance(r, dict) else {})
            if hasattr(m, "to_dict"):
                m = m.to_dict()
            elif not isinstance(m, dict):
                m = {}
            if tid:
                out[tid] = {
                    "evidenceId": eid,
                    "testId": tid,
                    "status": str(stat).upper(),
                    "metrics": m,
                }
        return out

    ev_a = extract_evidence(ctx_a)
    ev_b = extract_evidence(ctx_b)

    common_tids = sorted(set(ev_a.keys()) & set(ev_b.keys()))
    only_in_a_tids = sorted(set(ev_a.keys()) - set(ev_b.keys()))
    only_in_b_tids = sorted(set(ev_b.keys()) - set(ev_a.keys()))

    metric_comparisons = []
    changed_count = 0
    unchanged_count = 0

    for tid in common_tids:
        ea = ev_a[tid]
        eb = ev_b[tid]
        stat_a = ea["status"]
        stat_b = eb["status"]
        stat_changed = stat_a != stat_b

        ma = ea["metrics"]
        mb = eb["metrics"]
        all_metrics = sorted(set(ma.keys()) | set(mb.keys()))

        metric_deltas = []
        any_metric_changed = False

        for mk in all_metrics:
            val_a = ma.get(mk)
            val_b = mb.get(mk)
            delta = None
            pct_change = None

            if isinstance(val_a, (int, float)) and isinstance(val_b, (int, float)):
                delta = round(float(val_b) - float(val_a), 6)
                pct_change = round(((float(val_b) - float(val_a)) / abs(float(val_a))) * 100, 2) if float(val_a) != 0 else 0.0
                if delta != 0.0:
                    any_metric_changed = True

            metric_deltas.append({
                "metric": mk,
                "valA": val_a,
                "valB": val_b,
                "delta": delta,
                "pctChange": pct_change,
            })

        is_changed = stat_changed or any_metric_changed
        if is_changed:
            changed_count += 1
        else:
            unchanged_count += 1

        metric_comparisons.append({
            "testId": tid,
            "statusA": stat_a,
            "statusB": stat_b,
            "statusChanged": stat_changed,
            "evidenceIdA": ea["evidenceId"],
            "evidenceIdB": eb["evidenceId"],
            "metrics": metric_deltas,
            "isChanged": is_changed,
        })

    # Findings diff
    def extract_findings(ctx: ActiveRunContext) -> list[dict[str, Any]]:
        try:
            return get_run_findings(ctx.run_id)
        except Exception:
            return []

    findings_a = extract_findings(ctx_a)
    findings_b = extract_findings(ctx_b)

    f_map_a = {f.get("title") or f.get("findingId"): f for f in findings_a}
    f_map_b = {f.get("title") or f.get("findingId"): f for f in findings_b}

    all_f_keys = sorted(set(f_map_a.keys()) | set(f_map_b.keys()))
    findings_diff = []
    for fk in all_f_keys:
        fa = f_map_a.get(fk)
        fb = f_map_b.get(fk)
        findings_diff.append({
            "title": fk,
            "presentInA": fa is not None,
            "presentInB": fb is not None,
            "severityA": fa.get("severity") if fa else None,
            "severityB": fb.get("severity") if fb else None,
            "severityChanged": (fa.get("severity") if fa else None) != (fb.get("severity") if fb else None),
            "evidenceIdsA": fa.get("evidenceIds", []) if fa else [],
            "evidenceIdsB": fb.get("evidenceIds", []) if fb else [],
        })

    # Artifacts diff
    arts_a = ctx_a.artifacts
    arts_b = ctx_b.artifacts
    all_art_keys = sorted(set(arts_a.keys()) | set(arts_b.keys()))
    artifacts_diff = []
    for ak in all_art_keys:
        aa = arts_a.get(ak)
        ab = arts_b.get(ak)
        fp_art_a = aa.get("data_fingerprint") if aa else None
        fp_art_b = ab.get("data_fingerprint") if ab else None
        artifacts_diff.append({
            "artifactId": ak,
            "title": (ab or aa or {}).get("title", ak),
            "presentInA": aa is not None,
            "presentInB": ab is not None,
            "fingerprintA": fp_art_a,
            "fingerprintB": fp_art_b,
            "fingerprintChanged": fp_art_a != fp_art_b,
        })

    gov_a = (ctx_a.presentation or {}).get("governance_disposition") or "REVIEW_REQUIRED"
    gov_b = (ctx_b.presentation or {}).get("governance_disposition") or "REVIEW_REQUIRED"

    return {
        "compatible": True,
        "runA": {
            "runId": run_a,
            "workflow": wf_a,
            "domain": dom_a,
            "contextId": ctx_id_a,
            "status": ctx_a.status.lower(),
            "createdAt": ctx_a.created_at,
            "completedAt": ctx_a.completed_at,
            "evidenceCount": len(ctx_a.evidence_records),
            "artifactCount": len(ctx_a.artifacts),
            "governanceDisposition": gov_a,
        },
        "runB": {
            "runId": run_b,
            "workflow": wf_b,
            "domain": dom_b,
            "contextId": ctx_id_b,
            "status": ctx_b.status.lower(),
            "createdAt": ctx_b.created_at,
            "completedAt": ctx_b.completed_at,
            "evidenceCount": len(ctx_b.evidence_records),
            "artifactCount": len(ctx_b.artifacts),
            "governanceDisposition": gov_b,
        },
        "lineage": {
            "isLineage": is_lineage,
            "relation": "parent_child" if is_lineage else "peers",
            "parentRunId": getattr(req_b, "parent_run_id", None) or getattr(req_a, "parent_run_id", None),
        },
        "parameters": param_diffs,
        "fingerprints": {
            "fingerprintA": fp_a,
            "fingerprintB": fp_b,
            "matched": fp_a == fp_b,
        },
        "metricsSummary": {
            "totalCommonTests": len(common_tids),
            "changedTestsCount": changed_count,
            "unchangedTestsCount": unchanged_count,
            "onlyInACount": len(only_in_a_tids),
            "onlyInBCount": len(only_in_b_tids),
        },
        "metricComparisons": metric_comparisons,
        "onlyInA": [{"testId": tid, "status": ev_a[tid]["status"], "evidenceId": ev_a[tid]["evidenceId"]} for tid in only_in_a_tids],
        "onlyInB": [{"testId": tid, "status": ev_b[tid]["status"], "evidenceId": ev_b[tid]["evidenceId"]} for tid in only_in_b_tids],
        "findings": findings_diff,
        "artifacts": artifacts_diff,
        "governance": {
            "dispositionA": gov_a,
            "dispositionB": gov_b,
            "changed": gov_a != gov_b,
        },
    }


@router.get("/search")
@router.get("/workbench/search")
def search_workbench(
    q: str = Query(..., min_length=1),
    session_id: str | None = Query(None),
) -> dict[str, Any]:
    """Search only genuine indexed entities (runs, tests, scenarios, contexts)."""
    q_lower = q.lower().strip()
    results = []

    # 1. Runs
    for h in GLOBAL_QUEUE.get_run_history():
        if q_lower in h["run_id"].lower() or q_lower in h["workflow"].lower() or q_lower in h["context_id"].lower():
            results.append({
                "category": "run",
                "id": h["run_id"],
                "title": f"Run {h['run_id']}",
                "subtitle": f"{h['workflow']} ({h['context_id']}) • {h['status'].upper()}",
                "data": h,
            })

    # 2. Tests (from 79-test catalog)
    for t in list_tests():
        tid = t.test_id
        name = t.name
        desc = getattr(t, "description", "")
        if q_lower in tid.lower() or q_lower in name.lower() or q_lower in desc.lower():
            results.append({
                "category": "test",
                "id": tid,
                "title": f"Test: {tid}",
                "subtitle": name,
                "data": {"testId": tid, "name": name, "family": t.family},
            })

    # 3. Scenarios
    for sc in list_scenarios():
        sc_dict = sc.to_dict() if hasattr(sc, "to_dict") else sc
        sc_id = sc_dict.get("id", "")
        title = sc_dict.get("title", "")
        if q_lower in sc_id.lower() or q_lower in title.lower():
            results.append({
                "category": "scenario",
                "id": sc_id,
                "title": f"Scenario: {title}",
                "subtitle": sc_dict.get("description", ""),
                "data": sc_dict,
            })

    # 4. Evidence (from persisted runs' evidence records)
    for ctx in GLOBAL_QUEUE.list_runs():
        for ev in (ctx.evidence_records or []):
            eid = ev.get("evidence_id", "") if isinstance(ev, dict) else getattr(ev, "evidence_id", "")
            tid = ev.get("test_id", "") if isinstance(ev, dict) else getattr(ev, "test_id", "")
            if q_lower in eid.lower() or q_lower in tid.lower():
                results.append({
                    "category": "evidence",
                    "id": eid,
                    "title": f"Evidence: {eid}",
                    "subtitle": f"Test {tid} • Run {ctx.run_id}",
                    "data": {"evidenceId": eid, "testId": tid, "runId": ctx.run_id},
                })

    # 5. Execution contexts
    for cspec in get_canonical_context_specs():
        cd = cspec.to_dict()
        cid = cd.get("id", "")
        clabel = cd.get("label", cd.get("title", ""))
        if q_lower in cid.lower() or q_lower in clabel.lower():
            results.append({
                "category": "context",
                "id": cid,
                "title": f"Context: {clabel}",
                "subtitle": cid,
                "data": cd,
            })

    # 6. Artifacts (from persisted runs' artifacts)
    for ctx in GLOBAL_QUEUE.list_runs():
        arts = ctx.artifacts or {}
        if isinstance(arts, dict):
            for art_id, art_data in arts.items():
                title = ""
                if isinstance(art_data, dict):
                    title = art_data.get("title", art_data.get("name", art_id))
                else:
                    title = getattr(art_data, "title", str(art_id))
                if q_lower in art_id.lower() or (title and q_lower in title.lower()):
                    results.append({
                        "category": "artifact",
                        "id": art_id,
                        "title": f"Artifact: {title or art_id}",
                        "subtitle": f"Run {ctx.run_id}",
                        "data": {"artifactId": art_id, "title": title, "runId": ctx.run_id},
                    })

    return {
        "query": q,
        "results": results[:50],
        "count": len(results),
    }
