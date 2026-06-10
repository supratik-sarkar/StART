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
