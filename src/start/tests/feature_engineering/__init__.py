"""Feature engineering — fifteen registered surfaces.

Each registered test is a thin wrapper that calls the corresponding executor in
``transforms.py`` and emits **audit evidence** from the result. The mathematics lives in
the executor and is not duplicated here, so a transformation cannot pass its audit while
behaving differently in a pipeline.

What crosses into evidence
--------------------------

Scalars and hashes. Never a frame. ``TransformExecutionResult`` holds the transformed
data as a runtime payload; ``evidence_metrics()`` is the narrow projection of it that is
safe to hash into a tamper-evident ledger and still readable in five years.

Registered surfaces
-------------------

``plan`` declares what will be done and freezes it. ``fitting_scope_audit`` verifies the
leakage invariant across every stateful step. The remaining thirteen are transformations
and one selection.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from start.core.schemas import Status, TestResult, ThresholdSpec
from start.registry import TestContext, register_test
from start.tests.feature_engineering.audit import audit_executor
from start.tests.feature_engineering.execution import (
    FittingScope,
    TransformExecutionResult,
    canonical_state_hash,
)
from start.tests.feature_engineering.transforms import (
    run_aggregation_features,
    run_categorical_encoding,
    run_imputation,
    run_interactions,
    run_monotonic_binning,
    run_numeric_transform,
    run_pca_transform,
    run_rare_category_grouping,
    run_scaling,
    run_selection,
    run_temporal_features,
    run_winsorization,
    run_woe_iv,
)

__all__ = [
    "plan",
    "imputation",
    "scaling",
    "numeric_transform",
    "winsorization",
    "categorical_encoding",
    "rare_category_grouping",
    "woe_iv",
    "monotonic_binning",
    "interactions",
    "temporal_features",
    "aggregation_features",
    "pca_transform",
    "selection",
    "fitting_scope_audit",
    "IV_BANDS",
]

_STRIPES = ("model", "credit")
_OBJECTS = ("ml_model", "statistical_model", "scorecard", "data_pipeline")

#: Credit-scoring industry convention, not a result with a single originating citation.
IV_BANDS: tuple[tuple[float, str], ...] = (
    (0.02, "useless"),
    (0.10, "weak"),
    (0.30, "medium"),
    (0.50, "strong"),
    (float("inf"), "suspicious"),
)


def _cohorts(ctx: TestContext) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    return ctx.train, ctx.test, (ctx.extra or {}).get("oos")


def _skip(test_id: str, name: str, reason: str, **params: Any) -> TestResult:
    return TestResult(
        test_id=test_id, test_name=name, status=Status.SKIPPED, params=params, interpretation=reason
    )


def _emit(
    test_id: str,
    name: str,
    result: TransformExecutionResult,
    extra_metrics: dict[str, Any] | None = None,
    extra_limitations: list[str] | None = None,
) -> TestResult:
    """Turn a runtime payload into an audit record. Scalars and hashes only."""
    metrics = result.evidence_metrics()
    if extra_metrics:
        metrics.update(extra_metrics)
    limitations = [
        "Audit evidence only: the transformed frames are a runtime payload and are "
        "deliberately not serialised into the ledger. Reproduction is via the recorded "
        "fitted-state and output hashes.",
        "Determinism: numerical. State hashes round floats to a declared precision, "
        "because two honest fits can differ in the last bit through BLAS "
        "non-determinism.",
        *result.notes,
        *(extra_limitations or []),
    ]
    return TestResult(
        test_id=test_id,
        test_name=name,
        status=Status.RECORDED,
        params=dict(result.params),
        metrics=metrics,
        interpretation=(
            f"{result.step}: {metrics['n_features_before']} feature(s) in, "
            f"{metrics['n_features_after']} out, {metrics['n_features_affected']} "
            f"affected; fitted on {result.fitting_scope}."
        ),
        limitations=limitations,
    )


def _run(test_id: str, name: str, executor: Any, ctx: TestContext, **kwargs: Any) -> TestResult:
    train, test, oos = _cohorts(ctx)
    try:
        result = executor(train, test, oos, **kwargs)
    except ValueError as exc:
        # A domain violation is a SKIP with the reason, not a crash: "Box-Cox needs
        # positive values" is information the reviewer can act on.
        return _skip(test_id, name, str(exc), **kwargs)
    except Exception as exc:
        return TestResult(
            test_id=test_id,
            test_name=name,
            status=Status.ERROR,
            params=kwargs,
            interpretation=f"{type(exc).__name__}: {exc}",
            limitations=["The transformation did not complete."],
        )
    return _emit(test_id, name, result)


# --------------------------------------------------------------------------- #
# 1. plan
# --------------------------------------------------------------------------- #
@register_test(
    "feature_engineering.plan",
    family="feature_engineering",
    name="Feature engineering plan",
    requires=("train",),
    default_params={"steps": (), "fitting_scope": FittingScope.TRAIN_ONLY, "temporal_policy": "causal_only"},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("documentation_completeness", "reproducibility"),
    object_kinds=_OBJECTS,
)
def plan(
    ctx: TestContext,
    steps: tuple[str, ...] = (),
    fitting_scope: str = FittingScope.TRAIN_ONLY,
    temporal_policy: str = "causal_only",
    missing_policy: str = "median",
    categorical_policy: str = "onehot",
    monotonicity_policy: str = "none",
) -> TestResult:
    """Declares the intended pipeline and freezes it with a content hash.

    The hash is the point. A plan recorded before execution and compared after is the
    difference between "we intended to do this" and "this is what happened" — and a
    plan that changed during execution is a FAIL, the same discipline the review plan
    hash already applies elsewhere in the product.
    """
    declared = {
        "steps": list(steps),
        "fitting_scope": fitting_scope,
        "temporal_policy": temporal_policy,
        "missing_policy": missing_policy,
        "categorical_policy": categorical_policy,
        "monotonicity_policy": monotonicity_policy,
        "seed": int(ctx.seed),
    }
    plan_hash = canonical_state_hash(declared)
    recorded = (ctx.extra or {}).get("feature_engineering_plan_hash")
    mutated = bool(recorded) and recorded != plan_hash

    return TestResult(
        test_id="feature_engineering.plan",
        test_name="Feature engineering plan",
        status=Status.FAIL if mutated else Status.RECORDED,
        params=declared,
        metrics={
            "plan_hash": plan_hash,
            "n_steps": len(steps),
            "steps": ", ".join(steps),
            "fitting_scope": fitting_scope,
            "temporal_policy": temporal_policy,
            "plan_mutated_during_execution": mutated,
        },
        interpretation=(
            f"Plan of {len(steps)} step(s) frozen at {plan_hash[:16]}…"
            if not mutated
            else f"PLAN MUTATED during execution: recorded {str(recorded)[:16]}…, computed {plan_hash[:16]}…"
        ),
        limitations=[
            "The hash covers the declared plan, not the data it was applied to.",
            "A plan that changed during execution is a FAIL: the review would otherwise "
            "record an intent that was not carried out.",
        ],
    )


# --------------------------------------------------------------------------- #
# 2-13. transformations
# --------------------------------------------------------------------------- #
@register_test(
    "feature_engineering.imputation",
    family="feature_engineering",
    name="Imputation",
    requires=("train",),
    default_params={"strategy": "median", "add_indicator": False},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("data_quality_lineage",),
    object_kinds=_OBJECTS,
)
def imputation(
    ctx: TestContext, strategy: str = "median", add_indicator: bool = False, fill_value: Any = 0
) -> TestResult:
    """Fill missing values from training statistics only."""
    return _run(
        "feature_engineering.imputation",
        "Imputation",
        run_imputation,
        ctx,
        strategy=strategy,
        add_indicator=add_indicator,
        fill_value=fill_value,
        exclude=(ctx.target_column, ctx.score_column, ctx.prediction_column),
    )


@register_test(
    "feature_engineering.scaling",
    family="feature_engineering",
    name="Scaling",
    requires=("train",),
    default_params={"method": "standard"},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification",),
    object_kinds=_OBJECTS,
)
def scaling(ctx: TestContext, method: str = "standard") -> TestResult:
    """Scale numeric features using training statistics only."""
    return _run(
        "feature_engineering.scaling",
        "Scaling",
        run_scaling,
        ctx,
        method=method,
        exclude=(ctx.target_column, ctx.score_column, ctx.prediction_column),
    )


@register_test(
    "feature_engineering.numeric_transform",
    family="feature_engineering",
    name="Numeric transform",
    requires=("train",),
    default_params={"method": "yeo_johnson"},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("assumption_validity",),
    object_kinds=_OBJECTS,
)
def numeric_transform(ctx: TestContext, method: str = "yeo_johnson") -> TestResult:
    """Reshape numeric distributions. Domain-invalid columns are skipped, not shifted."""
    return _run(
        "feature_engineering.numeric_transform",
        "Numeric transform",
        run_numeric_transform,
        ctx,
        method=method,
        exclude=(ctx.target_column, ctx.score_column, ctx.prediction_column),
    )


@register_test(
    "feature_engineering.winsorization",
    family="feature_engineering",
    name="Winsorization",
    requires=("train",),
    default_params={"method": "iqr", "k": 1.5},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("data_quality_lineage",),
    object_kinds=_OBJECTS,
)
def winsorization(
    ctx: TestContext,
    method: str = "iqr",
    k: float = 1.5,
    lower_pct: float = 1.0,
    upper_pct: float = 99.0,
    z_threshold: float = 3.0,
) -> TestResult:
    """Clip extremes to bounds learned on train. A transformation, unlike
    ``preprocessing.outliers`` which only reports."""
    return _run(
        "feature_engineering.winsorization",
        "Winsorization",
        run_winsorization,
        ctx,
        method=method,
        k=k,
        lower_pct=lower_pct,
        upper_pct=upper_pct,
        z_threshold=z_threshold,
        exclude=(ctx.target_column, ctx.score_column, ctx.prediction_column),
    )


@register_test(
    "feature_engineering.categorical_encoding",
    family="feature_engineering",
    name="Categorical encoding",
    requires=("train",),
    default_params={"method": "onehot", "n_folds": 5, "smoothing": 10.0},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("implementation_verification", "use_boundary"),
    object_kinds=_OBJECTS,
)
def categorical_encoding(
    ctx: TestContext, method: str = "onehot", n_folds: int = 5, smoothing: float = 10.0
) -> TestResult:
    """Encode categoricals. ``target`` encoding is out-of-fold on the train side."""
    return _run(
        "feature_engineering.categorical_encoding",
        "Categorical encoding",
        run_categorical_encoding,
        ctx,
        method=method,
        target_column=ctx.target_column,
        n_folds=n_folds,
        smoothing=smoothing,
        seed=ctx.seed,
        exclude=(ctx.score_column, ctx.prediction_column),
    )


@register_test(
    "feature_engineering.rare_category_grouping",
    family="feature_engineering",
    name="Rare category grouping",
    requires=("train",),
    default_params={"min_pct": 1.0},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("data_quality_lineage",),
    object_kinds=_OBJECTS,
)
def rare_category_grouping(
    ctx: TestContext, min_pct: float = 1.0, other_label: str = "__OTHER__"
) -> TestResult:
    """Collapse infrequent levels using training frequencies only."""
    return _run(
        "feature_engineering.rare_category_grouping",
        "Rare category grouping",
        run_rare_category_grouping,
        ctx,
        min_pct=min_pct,
        other_label=other_label,
        exclude=(ctx.target_column, ctx.score_column, ctx.prediction_column),
    )


@register_test(
    "feature_engineering.woe_iv",
    family="feature_engineering",
    name="Weight of evidence / IV",
    requires=("train", "target_column"),
    default_params={"bins": 10, "min_bin_pct": 0.05, "smoothing": 0.5},
    context_type="tabular",
    risk_stripes=("credit", "model"),
    risk_dimensions=("conceptual_soundness", "use_boundary"),
    object_kinds=("scorecard", "statistical_model"),
)
def woe_iv(
    ctx: TestContext,
    bins: int = 10,
    min_bin_pct: float = 0.05,
    smoothing: float = 0.5,
    n_folds: int = 5,
    iv_suspicious: float = 0.5,
) -> TestResult:
    """WoE transformation with information value. Binary target only, out-of-fold.

    An IV above ``iv_suspicious`` is WARNed rather than celebrated: in credit scoring an
    IV that high usually means the feature encodes the outcome.
    """
    train, test, oos = _cohorts(ctx)
    try:
        result = run_woe_iv(
            train,
            test,
            oos,
            target_column=ctx.target_column,
            bins=bins,
            min_bin_pct=min_bin_pct,
            smoothing=smoothing,
            n_folds=n_folds,
            seed=ctx.seed,
            exclude=(ctx.score_column, ctx.prediction_column),
        )
    except ValueError as exc:
        return _skip("feature_engineering.woe_iv", "Weight of evidence / IV", str(exc))

    ivs: dict[str, float] = result.fitted_state.get("iv", {})
    max_iv = max(ivs.values()) if ivs else 0.0
    top = max(ivs, key=lambda c: ivs[c]) if ivs else ""

    def band(value: float) -> str:
        for bound, label in IV_BANDS:
            if value < bound:
                return label
        return "suspicious"

    extra = {
        "max_iv": round(max_iv, 6),
        "max_iv_feature": top,
        "max_iv_band": band(max_iv),
        "n_features_binned": len(ivs),
    }
    for column, value in sorted(ivs.items()):
        extra[f"iv.{column}"] = round(float(value), 6)

    emitted = _emit(
        "feature_engineering.woe_iv",
        "Weight of evidence / IV",
        result,
        extra,
        [
            "IV bands (0.02 / 0.10 / 0.30 / 0.50) are credit-scoring industry practice "
            "with textbook treatments, not a result attributable to a single paper.",
            "An IV above the suspicious threshold is more often leakage than a strong feature.",
        ],
    )
    emitted.thresholds = [ThresholdSpec(metric="max_iv", warn=iv_suspicious, fail=None)]
    if max_iv >= iv_suspicious:
        emitted.status = Status.WARN
        emitted.interpretation += (
            f" Highest IV is {max_iv:.4f} ('{top}'), in the '{band(max_iv)}' band — "
            "review for leakage before treating this as predictive power."
        )
    return emitted


@register_test(
    "feature_engineering.monotonic_binning",
    family="feature_engineering",
    name="Monotonic binning",
    requires=("train", "target_column"),
    default_params={"max_bins": 10, "min_bin_pct": 0.05, "direction": "auto"},
    context_type="tabular",
    risk_stripes=("credit", "model"),
    risk_dimensions=("conceptual_soundness",),
    object_kinds=("scorecard", "statistical_model"),
)
def monotonic_binning(
    ctx: TestContext, max_bins: int = 10, min_bin_pct: float = 0.05, direction: str = "auto"
) -> TestResult:
    """Bin numeric features so the target rate is monotone. Documented merge rule."""
    train, test, oos = _cohorts(ctx)
    try:
        result = run_monotonic_binning(
            train,
            test,
            oos,
            target_column=ctx.target_column,
            max_bins=max_bins,
            min_bin_pct=min_bin_pct,
            direction=direction,
            exclude=(ctx.score_column, ctx.prediction_column),
        )
    except ValueError as exc:
        return _skip("feature_engineering.monotonic_binning", "Monotonic binning", str(exc))

    edges = result.fitted_state.get("edges", {})
    extra = {
        "n_features_binned": len(edges),
        "mean_bins": round(float(np.mean([len(v) - 1 for v in edges.values()])) if edges else 0.0, 4),
        "directions": ", ".join(
            f"{k}:{v}" for k, v in sorted(result.fitted_state.get("direction", {}).items())
        )[:200],
    }
    return _emit("feature_engineering.monotonic_binning", "Monotonic binning", result, extra)


@register_test(
    "feature_engineering.interactions",
    family="feature_engineering",
    name="Feature interactions",
    requires=("train",),
    default_params={"method": "product", "max_features": 50, "denominator_floor": 1e-6},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("conceptual_soundness",),
    object_kinds=_OBJECTS,
)
def interactions(
    ctx: TestContext, method: str = "product", max_features: int = 50, denominator_floor: float = 1e-6
) -> TestResult:
    """Pairwise interactions, deterministically named and hard-capped in count."""
    return _run(
        "feature_engineering.interactions",
        "Feature interactions",
        run_interactions,
        ctx,
        method=method,
        max_features=max_features,
        denominator_floor=denominator_floor,
        exclude=(ctx.target_column, ctx.score_column, ctx.prediction_column),
    )


@register_test(
    "feature_engineering.temporal_features",
    family="feature_engineering",
    name="Temporal features",
    requires=("train", "timestamp_column"),
    default_params={"features": ("hour", "dow", "month", "days_since")},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("use_boundary", "data_quality_lineage"),
    object_kinds=_OBJECTS,
)
def temporal_features(
    ctx: TestContext,
    features: tuple[str, ...] = ("hour", "dow", "month", "days_since"),
) -> TestResult:
    """Calendar features from each row's own timestamp. No future reference."""
    return _run(
        "feature_engineering.temporal_features",
        "Temporal features",
        run_temporal_features,
        ctx,
        timestamp_column=ctx.timestamp_column,
        features=features,
        exclude=(ctx.target_column, ctx.score_column, ctx.prediction_column),
    )


@register_test(
    "feature_engineering.aggregation_features",
    family="feature_engineering",
    name="Aggregation features",
    requires=("train", "entity_id_column", "timestamp_column"),
    default_params={"windows": (7, 30, 90)},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("use_boundary",),
    object_kinds=_OBJECTS,
)
def aggregation_features(ctx: TestContext, windows: tuple[int, ...] = (7, 30, 90)) -> TestResult:
    """Rolling per-entity statistics, strictly backward-looking."""
    return _run(
        "feature_engineering.aggregation_features",
        "Aggregation features",
        run_aggregation_features,
        ctx,
        entity_id_column=ctx.entity_id_column,
        timestamp_column=ctx.timestamp_column,
        windows=windows,
        exclude=(ctx.target_column, ctx.score_column, ctx.prediction_column),
    )


@register_test(
    "feature_engineering.pca_transform",
    family="feature_engineering",
    name="PCA transform",
    requires=("train",),
    default_params={"n_components": 0.95, "whiten": False},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("conceptual_soundness", "explainability"),
    object_kinds=_OBJECTS,
)
def pca_transform(ctx: TestContext, n_components: float | int = 0.95, whiten: bool = False) -> TestResult:
    """Fit PCA on train, apply unchanged to evaluation cohorts.

    Distinct from ``preprocessing.dimensionality_diagnostic``, which reports a number
    and produces no data.
    """
    train, test, oos = _cohorts(ctx)
    try:
        result = run_pca_transform(
            train,
            test,
            oos,
            n_components=n_components,
            whiten=whiten,
            seed=ctx.seed,
            exclude=(ctx.target_column, ctx.score_column, ctx.prediction_column),
        )
    except ValueError as exc:
        return _skip("feature_engineering.pca_transform", "PCA transform", str(exc))

    ratios = result.fitted_state.get("explained_variance_ratio", [])
    extra = {
        "n_components_actual": result.fitted_state.get("n_components_actual", 0),
        "explained_variance_total": round(float(sum(ratios)), 6),
    }
    for index, value in enumerate(ratios[:10], start=1):
        extra[f"explained_variance_ratio_{index}"] = round(float(value), 6)
    return _emit(
        "feature_engineering.pca_transform",
        "PCA transform",
        result,
        extra,
        [
            "Components are not interpretable as original features; explainability "
            "downstream refers to components, not to the inputs."
        ],
    )


@register_test(
    "feature_engineering.selection",
    family="feature_engineering",
    name="Feature selection",
    requires=("train",),
    default_params={"method": "mutual_info", "top_k": None, "threshold": None},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("conceptual_soundness", "use_boundary"),
    object_kinds=_OBJECTS,
)
def selection(
    ctx: TestContext,
    method: str = "mutual_info",
    top_k: int | None = None,
    threshold: float | None = None,
    vif_threshold: float = 10.0,
) -> TestResult:
    """Choose a feature subset using training labels only.

    Supervised selection legitimately uses every training label — it is a column
    decision, not a per-row value. It must not see evaluation labels.
    """
    train, test, oos = _cohorts(ctx)
    try:
        result = run_selection(
            train,
            test,
            oos,
            method=method,
            target_column=ctx.target_column,
            top_k=top_k,
            threshold=threshold,
            vif_threshold=vif_threshold,
            seed=ctx.seed,
            model=ctx.model,
            exclude=(ctx.score_column, ctx.prediction_column),
        )
    except ValueError as exc:
        return _skip("feature_engineering.selection", "Feature selection", str(exc), method=method)

    extra = {
        "n_kept": len(result.fitted_state.get("kept", [])),
        "n_dropped": len(result.fitted_state.get("dropped", [])),
        "method": method,
        "kept_features": ", ".join(result.fitted_state.get("kept", [])[:30]),
    }
    return _emit("feature_engineering.selection", "Feature selection", result, extra)


# --------------------------------------------------------------------------- #
# 15. fitting scope audit
# --------------------------------------------------------------------------- #
_AUDITABLE: tuple[tuple[str, Any, dict[str, Any]], ...] = (
    ("imputation", run_imputation, {"strategy": "median"}),
    ("scaling", run_scaling, {"method": "standard"}),
    ("winsorization", run_winsorization, {"method": "iqr", "k": 1.5}),
    ("rare_category_grouping", run_rare_category_grouping, {"min_pct": 1.0}),
    ("categorical_encoding", run_categorical_encoding, {"method": "target"}),
    ("woe_iv", run_woe_iv, {}),
    ("monotonic_binning", run_monotonic_binning, {}),
    ("pca_transform", run_pca_transform, {"n_components": 2}),
    ("selection", run_selection, {"method": "mutual_info", "top_k": 3}),
)


@register_test(
    "feature_engineering.fitting_scope_audit",
    family="feature_engineering",
    name="Fitting-scope audit",
    requires=("train",),
    default_params={"steps": ()},
    context_type="tabular",
    risk_stripes=_STRIPES,
    risk_dimensions=("use_boundary", "implementation_verification"),
    object_kinds=_OBJECTS,
)
def fitting_scope_audit(ctx: TestContext, steps: tuple[str, ...] = ()) -> TestResult:
    """Verify that no transformation learned from data it was not permitted to see.

    Four checks per applicable step. The one that does the work is Check 2, which
    perturbs evaluation **values** — shuffling evaluation rows leaves a leaky scaler's
    fitted mean identical, so a shuffle-based check passes a pipeline that is plainly
    leaking.

    A violation names the step and the check, because "something leaked" is not
    actionable.
    """
    train, test, oos = _cohorts(ctx)
    if test is None:
        return _skip(
            "feature_engineering.fitting_scope_audit",
            "Fitting-scope audit",
            "No evaluation cohort available. The principal check perturbs evaluation "
            "values, so without a test cohort the audit cannot establish isolation.",
        )

    selected = set(steps) if steps else {name for name, _, _ in _AUDITABLE}
    exclude = (ctx.score_column, ctx.prediction_column)
    findings: list[Any] = []
    audited: list[str] = []
    skipped: list[str] = []

    for name, executor, kwargs in _AUDITABLE:
        if name not in selected:
            continue
        call = dict(kwargs)
        needs_target = name in {"categorical_encoding", "woe_iv", "monotonic_binning", "selection"}
        if needs_target and not ctx.target_column:
            skipped.append(f"{name} (no target column)")
            continue
        call["exclude"] = (
            exclude if needs_target else (ctx.target_column, ctx.score_column, ctx.prediction_column)
        )
        audit = audit_executor(
            executor,
            train,
            test,
            oos,
            step=name,
            target_column=ctx.target_column if needs_target else None,
            **call,
        )
        # An executor that could not run at all on this data is not a leakage finding.
        if len(audit.findings) == 1 and audit.findings[0].check == "check_0_execution":
            skipped.append(f"{name} (not applicable to this data)")
            continue
        audited.append(name)
        findings.extend(audit.findings)

    violations = [f for f in findings if not f.passed]
    metrics: dict[str, Any] = {
        "n_steps_audited": len(audited),
        "n_steps_skipped": len(skipped),
        "n_checks": len(findings),
        "n_violations": len(violations),
        "steps_audited": ", ".join(sorted(audited)),
        "steps_skipped": ", ".join(sorted(skipped)),
        "violating_steps": ", ".join(sorted({f.step for f in violations})),
    }
    for finding in findings:
        metrics[f"{finding.check}.{finding.step}"] = "pass" if finding.passed else "VIOLATION"

    if not audited:
        return _skip(
            "feature_engineering.fitting_scope_audit",
            "Fitting-scope audit",
            "No transformation was applicable to this data; nothing was audited. "
            + (f"Skipped: {', '.join(skipped)}." if skipped else ""),
        )

    return TestResult(
        test_id="feature_engineering.fitting_scope_audit",
        test_name="Fitting-scope audit",
        status=Status.FAIL if violations else Status.PASS,
        params={"steps": list(steps)},
        metrics=metrics,
        interpretation=(
            f"{len(violations)} fitting-scope violation(s) across {len(audited)} "
            f"transformation(s). First: '{violations[0].step}' failed "
            f"{violations[0].check} — {violations[0].detail}"
            if violations
            else f"All {len(findings)} check(s) passed across {len(audited)} "
            "transformation(s): no transformation learned from data outside its "
            "permitted scope."
        ),
        limitations=[
            "Checks the transformations in this registry. A bespoke step outside it is not covered.",
            "Check 2 perturbs evaluation values rather than row order, because "
            "order-invariant statistics (mean, median, level frequency) are unchanged "
            "by shuffling — a shuffle-based check passes a leaking pipeline.",
            "Floating state is compared at a declared tolerance, so a leak smaller than "
            "that tolerance would not be detected. It is set far below any plausible "
            "leakage signal and far above BLAS last-bit noise.",
            "Passing does not establish that the modelling design is sound; it "
            "establishes that the fitted transformations respected their scope.",
        ],
    )
