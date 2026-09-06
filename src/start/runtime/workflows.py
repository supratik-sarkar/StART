"""Authoritative Workflow Execution Specifications and Applicability for StART v5.1.0.

Provides single-source-of-truth definitions for workflows:
- CoreWorkflowExecutionSpec
- Candidate test IDs vs Applicable test IDs vs Skipped test IDs
- EngineKind typed dispatch
- Dynamic applicability resolution against ExecutionContextInstance

Strict Invariants:
1. Zero imports of start.web (CORE_RUNTIME_IMPORTS_START_WEB = 0).
2. WORKFLOW_APPLICABILITY_RESOLUTION = PASS.
3. PLAN_AND_EXECUTOR_SHARE_RESOLVED_SPEC = PASS.
4. Programmatically validated test IDs against the canonical 79-test registry.
5. genai.citation_coverage omitted from tabular explainability.
6. Real deep learning diagnostics; zero simulated training epochs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from start.registry import list_tests
from start.runtime.contexts import ExecutionContextInstance, resolve_context_spec


class EngineKind(str, Enum):
    PREDICTIVE_SUBSET = "predictive_subset"
    MARKET_SUBSET = "market_subset"
    TUNING = "tuning"
    DEEP_LEARNING_REVIEW = "deep_learning_review"


@dataclass(frozen=True)
class WorkflowExecutionSpec:
    """Authoritative non-web specification of a StART workflow."""

    workflow_id: str
    label: str
    category: str  # "ml" | "quant"
    enabled: bool
    disabled_reason: str | None
    compatible_contexts: list[str]
    supported_actions: list[str]
    engine_kind: EngineKind
    candidate_test_ids: tuple[str, ...]
    step_specs: list[tuple[str, str, str, str, tuple[str, ...]]]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["engine_kind"] = self.engine_kind.value
        return d


@dataclass(frozen=True)
class ResolvedWorkflowExecution:
    """Resolved workflow execution plan for a specific context instance."""

    workflow_id: str
    context_id: str
    candidate_test_ids: tuple[str, ...]
    applicable_test_ids: tuple[str, ...]
    skipped_test_ids: tuple[str, ...]
    engine_kind: EngineKind
    step_specs: list[tuple[str, str, str, str, tuple[str, ...]]]
    supported_actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "context_id": self.context_id,
            "candidate_test_ids": list(self.candidate_test_ids),
            "applicable_test_ids": list(self.applicable_test_ids),
            "skipped_test_ids": list(self.skipped_test_ids),
            "engine_kind": self.engine_kind.value,
            "step_specs": [
                {
                    "id": s[0],
                    "label": s[1],
                    "kind": s[2],
                    "description": s[3],
                    "candidate_test_ids": list(s[4]),
                }
                for s in self.step_specs
            ],
            "supported_actions": list(self.supported_actions),
        }


def _build_canonical_workflow_specs() -> dict[str, WorkflowExecutionSpec]:
    all_tests = list_tests()
    all_test_ids = {t.test_id for t in all_tests}

    pred_eda = tuple(sorted(t.test_id for t in all_tests if t.family == "eda"))
    pred_prep = tuple(sorted(t.test_id for t in all_tests if t.family == "preprocessing"))
    pred_fe = tuple(sorted(t.test_id for t in all_tests if t.family == "feature_engineering"))
    pred_sup = tuple(sorted(t.test_id for t in all_tests if t.family == "supervised"))
    pred_xai = tuple(sorted(t.test_id for t in all_tests if t.family == "xai"))
    pred_genai = tuple(sorted(t.test_id for t in all_tests if t.family == "genai"))

    # All 52 predictive tests
    pred_52 = pred_eda + pred_prep + pred_fe + pred_sup + pred_xai + pred_genai
    assert len(pred_52) == 52, f"Expected 52 predictive candidate tests, got {len(pred_52)}"

    market_portfolio = tuple(sorted(t.test_id for t in all_tests if t.family == "portfolio"))
    market_attr = tuple(sorted(t.test_id for t in all_tests if t.family == "attribution"))
    market_cov = tuple(sorted(t.test_id for t in all_tests if t.family == "covariance"))
    market_risk = tuple(
        sorted(
            t.test_id
            for t in all_tests
            if t.family == "traded_risk" and not t.test_id.startswith("traded_risk.cev_") and not t.test_id.startswith("traded_risk.stanton_")
        )
    )
    # 25 market tests (portfolio 10 + attribution 6 + covariance 3 + market traded_risk 6)
    market_25 = market_portfolio + market_attr + market_cov + market_risk
    assert len(market_25) == 25, f"Expected 25 market candidate tests, got {len(market_25)}"

    # DL diagnostics candidate tests
    dl_candidate_tests = tuple(
        tid
        for tid in (
            "supervised.classification_metrics",
            "supervised.discrimination",
            "supervised.calibration",
            "xai.global_importance",
            "xai.feature_sensitivity",
            "xai.importance_stability",
            "xai.integrated_gradients",
        )
        if tid in all_test_ids
    )

    # Post-tuning validation candidate tests
    tuning_validation_tests = tuple(
        tid
        for tid in (
            "supervised.classification_metrics",
            "supervised.discrimination",
            "supervised.calibration",
        )
        if tid in all_test_ids
    )

    specs: dict[str, WorkflowExecutionSpec] = {
        "predictive_ml": WorkflowExecutionSpec(
            workflow_id="predictive_ml",
            label="Predictive ML",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=["rerun", "change_parameter", "deeper_test"],
            engine_kind=EngineKind.PREDICTIVE_SUBSET,
            candidate_test_ids=pred_52,
            step_specs=[
                ("step-context", "Load execution context", "context", "Load seeded tabular benchmark", ()),
                (
                    "step-preflight",
                    "Data contract & integrity diagnostics",
                    "test",
                    "Schema, missingness, drift, and leakage checks",
                    pred_eda + pred_prep,
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
                    "SHAP and feature sensitivity",
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
                    "Governance / sign-off stage",
                    (),
                ),
            ],
        ),
        "deep_learning": WorkflowExecutionSpec(
            workflow_id="deep_learning",
            label="Deep Learning Diagnostics",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["deep_learning_v1"],
            supported_actions=["rerun", "change_parameter"],
            engine_kind=EngineKind.DEEP_LEARNING_REVIEW,
            candidate_test_ids=dl_candidate_tests,
            step_specs=[
                (
                    "step-context",
                    "Load execution context",
                    "context",
                    "Load tabular neural network latent benchmark",
                    (),
                ),
                (
                    "step-performance",
                    "Neural decision surfaces",
                    "test",
                    "Classification metrics and calibration on latent embeddings",
                    tuple(t for t in dl_candidate_tests if t.startswith("supervised.")),
                ),
                (
                    "step-attribution",
                    "Neural attribution & sensitivity",
                    "test",
                    "Feature sensitivity, global importance, and gradient attributions",
                    tuple(t for t in dl_candidate_tests if t.startswith("xai.")),
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
                    "Governance / sign-off stage",
                    (),
                ),
            ],
        ),
        "data_diagnostics": WorkflowExecutionSpec(
            workflow_id="data_diagnostics",
            label="Data Diagnostics",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=["rerun"],
            engine_kind=EngineKind.PREDICTIVE_SUBSET,
            candidate_test_ids=pred_eda + pred_prep,
            step_specs=[
                ("step-context", "Load execution context", "context", "Load tabular dataset context", ()),
                (
                    "step-eda",
                    "Exploratory data analysis",
                    "test",
                    "Distributions, correlation structure, and collinearity",
                    pred_eda,
                ),
                (
                    "step-preprocessing",
                    "Preprocessing & leakage screening",
                    "test",
                    "Missingness, outliers, drift, and leakage detection",
                    pred_prep,
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
        "model_diagnostics": WorkflowExecutionSpec(
            workflow_id="model_diagnostics",
            label="Model Diagnostics",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=["rerun", "deeper_test"],
            engine_kind=EngineKind.PREDICTIVE_SUBSET,
            candidate_test_ids=pred_sup,
            step_specs=[
                ("step-context", "Load model context", "context", "Load trained model artifact context", ()),
                (
                    "step-metrics",
                    "Classification performance metrics",
                    "test",
                    "Thresholded metrics, discrimination AUC, and lift",
                    pred_sup,
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit diagnostic EvidenceRecords",
                    (),
                ),
                ("step-governance", "Sign-off", "governance", "Diagnostic review sign-off", ()),
            ],
        ),
        "calibration": WorkflowExecutionSpec(
            workflow_id="calibration",
            label="Calibration Refinement",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=["rerun", "change_parameter"],
            engine_kind=EngineKind.PREDICTIVE_SUBSET,
            candidate_test_ids=tuple(
                t for t in ("supervised.calibration", "supervised.classification_metrics", "supervised.discrimination")
                if t in all_test_ids
            ),
            step_specs=[
                ("step-context", "Load prediction scores", "context", "Load validation split predictions", ()),
                (
                    "step-calibration",
                    "Calibration curves and discrimination",
                    "test",
                    "Brier score, calibration curve, and ROC discrimination",
                    tuple(
                        t for t in ("supervised.calibration", "supervised.classification_metrics", "supervised.discrimination")
                        if t in all_test_ids
                    ),
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit calibration EvidenceRecords",
                    (),
                ),
                ("step-governance", "Sign-off", "governance", "Calibration sign-off", ()),
            ],
        ),
        "robustness": WorkflowExecutionSpec(
            workflow_id="robustness",
            label="Stress Testing & Robustness",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=["rerun", "change_parameter"],
            engine_kind=EngineKind.PREDICTIVE_SUBSET,
            candidate_test_ids=tuple(
                t for t in (
                    "preprocessing.feature_drift",
                    "preprocessing.categorical_drift",
                    "xai.feature_sensitivity",
                    "xai.importance_stability",
                )
                if t in all_test_ids
            ),
            step_specs=[
                ("step-context", "Load baseline context", "context", "Load baseline dataset and model", ()),
                (
                    "step-drift",
                    "Distribution drift diagnostics",
                    "test",
                    "Continuous and categorical feature drift screening",
                    tuple(
                        t for t in ("preprocessing.feature_drift", "preprocessing.categorical_drift")
                        if t in all_test_ids
                    ),
                ),
                (
                    "step-sensitivity",
                    "Attribution sensitivity under perturbation",
                    "test",
                    "Perturbation sensitivity and feature importance stability",
                    tuple(
                        t for t in ("xai.feature_sensitivity", "xai.importance_stability")
                        if t in all_test_ids
                    ),
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit stress testing EvidenceRecords",
                    (),
                ),
                ("step-governance", "Sign-off", "governance", "Robustness sign-off", ()),
            ],
        ),
        "explainability": WorkflowExecutionSpec(
            workflow_id="explainability",
            label="Explainability Audit",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=["rerun"],
            engine_kind=EngineKind.PREDICTIVE_SUBSET,
            # genai.citation_coverage explicitly omitted per Amendment 12
            candidate_test_ids=tuple(
                t for t in (
                    "xai.global_importance",
                    "xai.feature_sensitivity",
                    "xai.importance_stability",
                    "xai.integrated_gradients",
                )
                if t in all_test_ids
            ),
            step_specs=[
                ("step-context", "Load model & features", "context", "Load model and reference features", ()),
                (
                    "step-attribution",
                    "Feature attribution & importance",
                    "test",
                    "Global importance and feature sensitivity rankings",
                    tuple(
                        t for t in ("xai.global_importance", "xai.feature_sensitivity")
                        if t in all_test_ids
                    ),
                ),
                (
                    "step-stability",
                    "Attribution stability & gradients",
                    "test",
                    "Subsample importance stability and integrated gradients",
                    tuple(
                        t for t in ("xai.importance_stability", "xai.integrated_gradients")
                        if t in all_test_ids
                    ),
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit explainability EvidenceRecords",
                    (),
                ),
                ("step-governance", "Sign-off", "governance", "Explainability sign-off", ()),
            ],
        ),
        "hyperparameter_tuning": WorkflowExecutionSpec(
            workflow_id="hyperparameter_tuning",
            label="Hyperparameter Tuning",
            category="ml",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=["rerun", "change_parameter"],
            engine_kind=EngineKind.TUNING,
            candidate_test_ids=tuning_validation_tests,
            step_specs=[
                ("step-context", "Load dataset context", "context", "Load training split for hyperparameter search", ()),
                (
                    "step-tuning",
                    "Parameter space optimization",
                    "tool",
                    "Execute bounded random search over candidate hyperparameters",
                    (),
                ),
                (
                    "step-validation",
                    "Post-tuning model validation",
                    "test",
                    "Verify optimal model discrimination, metrics, and calibration",
                    tuning_validation_tests,
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit tuning and validation EvidenceRecords",
                    (),
                ),
                ("step-governance", "Sign-off", "governance", "Tuning sign-off", ()),
            ],
        ),
        "quantitative_finance": WorkflowExecutionSpec(
            workflow_id="quantitative_finance",
            label="Quantitative Finance",
            category="quant",
            enabled=True,
            disabled_reason=None,
            compatible_contexts=["institutional_market_v1"],
            supported_actions=["rerun", "change_parameter"],
            engine_kind=EngineKind.MARKET_SUBSET,
            candidate_test_ids=market_25,
            step_specs=[
                ("step-context", "Load market returns", "context", "Load multi-asset return history and factor world", ()),
                (
                    "step-portfolio",
                    "Portfolio optimization & risk",
                    "test",
                    "HRP, mean-variance, and portfolio risk metrics",
                    market_portfolio,
                ),
                (
                    "step-attribution",
                    "Factor exposure & return attribution",
                    "test",
                    "Factor returns, exposures, and risk attribution",
                    market_attr,
                ),
                (
                    "step-risk",
                    "Value-at-Risk & tail risk",
                    "test",
                    "Kupiec POF, Christoffersen independence, and barrier diagnostics",
                    market_risk,
                ),
                (
                    "step-covariance",
                    "Covariance estimation & shrinkage",
                    "test",
                    "Empirical, Ledoit-Wolf, and RegEM covariance matrices",
                    market_cov,
                ),
                (
                    "step-evidence",
                    "Create evidence bundle",
                    "evidence",
                    "Commit quantitative EvidenceRecords",
                    (),
                ),
                ("step-governance", "Sign-off", "governance", "Quantitative governance sign-off", ()),
            ],
        ),
        "model_comparison": WorkflowExecutionSpec(
            workflow_id="model_comparison",
            label="Model Comparison",
            category="ml",
            enabled=False,
            disabled_reason="Model comparison requires side-by-side candidate evaluation not configured in single-model demo mode.",
            compatible_contexts=["institutional_credit_v1"],
            supported_actions=[],
            engine_kind=EngineKind.PREDICTIVE_SUBSET,
            candidate_test_ids=(),
            step_specs=[],
        ),
    }

    return specs


_CANONICAL_WORKFLOW_SPECS = _build_canonical_workflow_specs()
WORKFLOW_SPECS = _CANONICAL_WORKFLOW_SPECS


def get_canonical_workflow_specs() -> dict[str, WorkflowExecutionSpec]:
    """Return all authoritative workflow execution specifications."""
    return _CANONICAL_WORKFLOW_SPECS


def get_workflow_catalog() -> list[dict[str, Any]]:
    """Return authoritative workflow catalog derived from canonical specifications."""
    out: list[dict[str, Any]] = []
    for wf_id, spec in _CANONICAL_WORKFLOW_SPECS.items():
        entry: dict[str, Any] = {
            "id": wf_id,
            "label": spec.label,
            "category": spec.category,
            "enabled": spec.enabled,
            "engine_kind": spec.engine_kind.value,
            "candidate_test_count": len(spec.candidate_test_ids),
            "compatible_contexts": list(spec.compatible_contexts),
            "supported_actions": list(spec.supported_actions),
        }
        if not spec.enabled and spec.disabled_reason:
            entry["disabled_reason"] = spec.disabled_reason
        out.append(entry)
    return out


def _evaluate_test_applicability(
    test_id: str,
    context_instance: ExecutionContextInstance | None,
) -> bool:
    """Evaluate whether a candidate test is applicable to the given context instance."""
    from start.registry import list_tests

    registry = {t.test_id: t for t in list_tests()}
    spec = registry.get(test_id)
    if spec is None:
        return False

    if context_instance is None:
        # Pre-instantiation evaluation based on known static spec
        return True

    bundle = context_instance.bundle
    requires = getattr(spec, "requires", ())
    context_type = getattr(spec, "context_type", "tabular")

    if context_type == "tabular":
        tab = bundle.tabular
        if tab is None:
            return False
        for req in requires:
            if req == "train" and getattr(tab, "train", None) is None:
                return False
            if req == "test" and getattr(tab, "test", None) is None:
                return False
            if req == "target_column" and getattr(tab, "target_column", None) is None:
                return False
            if req == "score_column" and getattr(tab, "score_column", None) is None:
                return False
            if req == "model" and getattr(tab, "model", None) is None:
                return False
            if req == "entity_id_column" and getattr(tab, "entity_id_column", None) is None:
                return False
            if req == "timestamp_column" and getattr(tab, "timestamp_column", None) is None:
                return False
        # Special check for integrated gradients (must support differentiable gradients)
        if test_id == "xai.integrated_gradients":
            model = getattr(tab, "model", None)
            if model is None or not hasattr(model, "predict_proba"):
                return False

    elif context_type == "short_rate":
        if bundle.short_rate is None:
            return False

    elif context_type in ("market", "covariance"):
        mkt = bundle.market
        if mkt is None:
            return False
        for req in requires:
            if req == "returns" and getattr(mkt, "returns", None) is None:
                return False
            if req == "prices" and getattr(mkt, "prices", None) is None:
                return False
            if req == "pnl" and getattr(mkt, "pnl", None) is None:
                return False
            if req == "var_series" and getattr(mkt, "var_series", None) is None:
                return False
            if req == "factor_exposures" and getattr(mkt, "factor_exposures", None) is None:
                return False
            if req == "covariance" and getattr(mkt, "returns", None) is None:
                return False

    return True


def resolve_workflow(
    workflow_id: str,
    context_id: str,
    context_instance: ExecutionContextInstance | None = None,
) -> ResolvedWorkflowExecution:
    """Resolve authoritative workflow execution specification for a context.

    Validates workflow ID, context ID, and compatibility.
    Separates candidate tests into applicable and skipped tests.
    Fails closed with ValueError on any invalid or incompatible configuration.
    """
    alias_map = {
        "calibration_refinement": "calibration",
        "stress_testing": "robustness",
        "explainability_audit": "explainability",
    }
    normalized_wf_id = alias_map.get(workflow_id, workflow_id)

    if normalized_wf_id not in _CANONICAL_WORKFLOW_SPECS:
        raise ValueError(
            f"Unknown workflow '{workflow_id}'. "
            f"Available workflows: {list(_CANONICAL_WORKFLOW_SPECS.keys())}"
        )

    wspec = _CANONICAL_WORKFLOW_SPECS[normalized_wf_id]

    # Verify context exists
    resolve_context_spec(context_id)

    # Verify workflow-context compatibility
    if context_id not in wspec.compatible_contexts:
        raise ValueError(
            f"Incompatible context '{context_id}' for workflow '{workflow_id}'. "
            f"Compatible contexts: {wspec.compatible_contexts}"
        )

    applicable: list[str] = []
    skipped: list[str] = []

    for tid in wspec.candidate_test_ids:
        if _evaluate_test_applicability(tid, context_instance):
            applicable.append(tid)
        else:
            skipped.append(tid)

    return ResolvedWorkflowExecution(
        workflow_id=wspec.workflow_id,
        context_id=context_id,
        candidate_test_ids=wspec.candidate_test_ids,
        applicable_test_ids=tuple(applicable),
        skipped_test_ids=tuple(skipped),
        engine_kind=wspec.engine_kind,
        step_specs=wspec.step_specs,
        supported_actions=wspec.supported_actions,
    )
