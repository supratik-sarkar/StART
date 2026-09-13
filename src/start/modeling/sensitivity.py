"""Parallel feature-shock sensitivity testing.

The top-k features (from global explainability) are shocked in parallel by a
multiplicative factor (1 + shock); predicted probabilities and AUC-ROC are
recomputed at each level. By construction the 0% shock reproduces the
baseline AUC exactly.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_SHOCKS = (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30)


def run_feature_shocks(
    model: Any,
    df: pd.DataFrame,
    features: list[str],
    target_column: str,
    feature_cols: list[str],
    shocks: tuple[float, ...] = DEFAULT_SHOCKS,
) -> list[dict[str, float]]:
    """Returns one row per shock: {shock, auc_roc, auc_drift}."""
    frame = df[[*feature_cols, target_column]].dropna()
    X, y = frame[feature_cols], frame[target_column].to_numpy()
    baseline_auc = float(roc_auc_score(y, model.predict_proba(X)[:, 1]))
    rows: list[dict[str, float]] = []
    for shock in shocks:
        X_shocked = X.copy()
        for feature in features:
            if feature in X_shocked.columns:
                X_shocked[feature] = X_shocked[feature] * (1.0 + shock)
        auc = (
            baseline_auc
            if shock == 0.0
            else float(roc_auc_score(y, model.predict_proba(X_shocked)[:, 1]))
        )
        rows.append(
            {
                "shock": shock,
                "auc_roc": round(auc, 6),
                "auc_drift": round(auc - baseline_auc, 6),
            }
        )
    return rows
