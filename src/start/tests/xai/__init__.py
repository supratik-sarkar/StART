"""XAI test family: importance stability via permutation importance.

SHAP-based checks are optional (start[xai] extra); the permutation engine
below depends only on scikit-learn and is fully deterministic given a seed.
"""

from __future__ import annotations

import numpy as np

from start.core.schemas import Status, TestResult, ThresholdSpec
from start.registry import TestContext, register_test


@register_test(
    "xai.importance_stability",
    family="xai",
    name="Permutation importance stability",
    requires=("model", "test", "target_column"),
    default_params={"n_repeats": 5, "top_k": 5, "jaccard_warn": 0.6, "jaccard_fail": 0.3},
)
def importance_stability(
    ctx: TestContext,
    n_repeats: int = 5,
    top_k: int = 5,
    jaccard_warn: float = 0.6,
    jaccard_fail: float = 0.3,
) -> TestResult:
    """Stability of top-k permutation-importance features across two seeds."""
    from sklearn.inspection import permutation_importance

    df = ctx.test
    if ctx.model is None or df is None or ctx.target_column is None:
        return TestResult(
            test_id="xai.importance_stability",
            test_name="Permutation importance stability",
            status=Status.SKIPPED,
            interpretation="Model or holdout data unavailable; XAI stability skipped.",
        )
    feature_cols = [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in {ctx.target_column, ctx.score_column, ctx.prediction_column}
    ]
    frame = df[[*feature_cols, ctx.target_column]].dropna()
    n_dropped = len(df) - len(frame)
    X, y = frame[feature_cols], frame[ctx.target_column]

    def top_features(seed: int) -> list[str]:
        imp = permutation_importance(ctx.model, X, y, n_repeats=n_repeats, random_state=seed)
        order = np.argsort(imp.importances_mean)[::-1][:top_k]
        return [feature_cols[i] for i in order]

    a, b = set(top_features(ctx.seed)), set(top_features(ctx.seed + 1))
    jaccard = len(a & b) / max(len(a | b), 1)
    result = TestResult(
        test_id="xai.importance_stability",
        test_name="Permutation importance stability",
        params={"n_repeats": n_repeats, "top_k": top_k},
        metrics={
            "topk_jaccard": round(float(jaccard), 6),
            "top_features_seed_a": ", ".join(sorted(a)),
            "top_features_seed_b": ", ".join(sorted(b)),
            "n_rows_used": int(len(frame)),
            "n_rows_dropped_nan": int(n_dropped),
        },
        thresholds=[
            ThresholdSpec(metric="topk_jaccard", warn=jaccard_warn, fail=jaccard_fail, direction="lower")
        ],
        interpretation=(
            f"Top-{top_k} permutation-importance Jaccard overlap across seeds is {jaccard:.2f}."
        ),
        limitations=[
            "Rows containing NaN in features or target are dropped before scoring.",
            "Permutation importance can mislead under correlated features.",
            "SHAP-based global/local checks require the start[xai] extra.",
        ],
    )
    return result.apply_thresholds()


@register_test(
    "xai.global_importance",
    family="xai",
    name="Global feature importance (SHAP or permutation)",
    requires=("model", "test", "target_column"),
    default_params={"top_k": 5},
)
def global_importance_test(ctx: TestContext, top_k: int = 5) -> TestResult:
    """Global importance with explicit method attribution; SHAP TreeExplainer
    when available for tree models, otherwise permutation importance. The
    method actually used is recorded — fallbacks are never silent."""
    from start.modeling.explain import global_importance as compute_importance

    df = ctx.test if ctx.test is not None else ctx.train
    if ctx.model is None or df is None or ctx.target_column is None:
        return TestResult(
            test_id="xai.global_importance",
            test_name="Global feature importance (SHAP or permutation)",
            status=Status.SKIPPED,
            interpretation="Model or holdout data unavailable; importance skipped.",
        )
    feature_cols = [
        c
        for c in df.select_dtypes(include=np.number).columns
        if c not in {ctx.target_column, ctx.score_column, ctx.prediction_column}
    ]
    frame = df[[*feature_cols, ctx.target_column]].dropna()
    importance = compute_importance(
        ctx.model, frame[feature_cols], frame[ctx.target_column], seed=ctx.seed
    )
    top = importance.global_importance[:top_k]
    result = TestResult(
        test_id="xai.global_importance",
        test_name="Global feature importance (SHAP or permutation)",
        params={"top_k": top_k},
        metrics={
            "method": importance.method,
            "top_features": ", ".join(name for name, _ in top),
            "top_feature": top[0][0] if top else "",
            "top_feature_importance": top[0][1] if top else 0.0,
            "n_local_examples": len(importance.local_examples),
        },
        interpretation=(
            f"Global importance computed via {importance.method}; "
            f"the most influential feature is '{top[0][0] if top else 'n/a'}'."
            + (f" Note: {importance.note}" if importance.note else "")
        ),
        limitations=(
            ["Local attributions unavailable on the permutation path."]
            if importance.method == "permutation"
            else ["SHAP values computed on a row sample for tractability."]
        ),
    )
    return result.apply_thresholds()


@register_test(
    "xai.feature_sensitivity",
    family="xai",
    name="Top-feature shock sensitivity",
    requires=("model", "test", "target_column"),
    default_params={"cohort": "test", "top_k": 5, "drift_warn": 0.02, "drift_fail": 0.10},
)
def feature_sensitivity(
    ctx: TestContext,
    cohort: str = "test",
    top_k: int = 5,
    drift_warn: float = 0.02,
    drift_fail: float = 0.10,
) -> TestResult:
    """Parallel multiplicative shocks (-30%..+30%) to the top-k most important
    features, measuring AUC-ROC drift from baseline. The 0% shock equals the
    baseline by construction. Cohort: test | oos | development (train+test+oos)."""
    import pandas as pd

    from start.modeling.explain import global_importance as compute_importance
    from start.modeling.sensitivity import DEFAULT_SHOCKS, run_feature_shocks

    if ctx.model is None or ctx.target_column is None:
        return TestResult(
            test_id="xai.feature_sensitivity",
            test_name="Top-feature shock sensitivity",
            status=Status.SKIPPED,
            interpretation="Model or target unavailable; sensitivity skipped.",
        )
    frames = {"train": ctx.train, "test": ctx.test, "oos": ctx.extra.get("oos")}
    if cohort == "development":
        parts = [f for f in frames.values() if f is not None]
        df = pd.concat(parts, ignore_index=True) if parts else None
    else:
        df = frames.get(cohort)
    if df is None:
        return TestResult(
            test_id="xai.feature_sensitivity",
            test_name="Top-feature shock sensitivity",
            status=Status.SKIPPED,
            interpretation=f"Requested cohort '{cohort}' not available; sensitivity skipped.",
            params={"cohort": cohort},
        )
    feature_cols = [
        c
        for c in df.select_dtypes(include=np.number).columns
        if c not in {ctx.target_column, ctx.score_column, ctx.prediction_column}
    ]
    frame = df[[*feature_cols, ctx.target_column]].dropna()
    importance = compute_importance(
        ctx.model, frame[feature_cols], frame[ctx.target_column], seed=ctx.seed
    )
    top_features = importance.top_features(top_k)
    rows = run_feature_shocks(
        ctx.model, frame, top_features, ctx.target_column, feature_cols, DEFAULT_SHOCKS
    )
    metrics: dict = {
        "cohort": cohort,
        "importance_method": importance.method,
        "shocked_features": ", ".join(top_features),
        "baseline_auc": next(r["auc_roc"] for r in rows if r["shock"] == 0.0),
    }
    for row in rows:
        label = f"{int(row['shock'] * 100):+d}pct"
        metrics[f"auc_{label}"] = row["auc_roc"]
        metrics[f"drift_{label}"] = row["auc_drift"]
    metrics["max_abs_auc_drift"] = round(max(abs(r["auc_drift"]) for r in rows), 6)
    result = TestResult(
        test_id="xai.feature_sensitivity",
        test_name="Top-feature shock sensitivity",
        params={"cohort": cohort, "top_k": top_k, "drift_warn": drift_warn, "drift_fail": drift_fail},
        metrics=metrics,
        thresholds=[ThresholdSpec(metric="max_abs_auc_drift", warn=drift_warn, fail=drift_fail)],
        interpretation=(
            f"Parallel shocks to the top {top_k} features ({importance.method} ranking) on the "
            f"{cohort} cohort produced a maximum absolute AUC drift of "
            f"{metrics['max_abs_auc_drift']:.4f} from baseline {metrics['baseline_auc']:.4f}."
        ),
        limitations=[
            "Parallel multiplicative shocks; interaction-specific shocks are not isolated.",
            "Shocked feature distributions may leave the training support at large shocks.",
        ],
    )
    return result.apply_thresholds()
