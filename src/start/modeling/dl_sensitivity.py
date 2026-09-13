"""Deep-learning sensitivity and robustness checks.

Three families, each returning AUC and drift-vs-baseline tables:
    1. top-feature shocks   -- multiplicative -30%..+30% on the top features
    2. input-noise robustness -- additive Gaussian noise at increasing scales
    3. feature-masking robustness -- zero out the top 1 / 3 / 5 features

All operate through ``predict_proba`` so they apply uniformly to any
StART model. By construction the zero-shock / zero-noise rows reproduce the
baseline AUC exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DEFAULT_SHOCKS = (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30)
DEFAULT_NOISE_LEVELS = (0.0, 0.01, 0.03, 0.05, 0.10)
DEFAULT_MASK_COUNTS = (1, 3, 5)


def _auc(model: Any, X: pd.DataFrame, y: np.ndarray) -> float:
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def feature_shock_sensitivity(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    shocks: tuple[float, ...] = DEFAULT_SHOCKS,
) -> list[dict[str, float]]:
    baseline = _auc(model, X, y)
    rows: list[dict[str, float]] = []
    for shock in shocks:
        if shock == 0.0:
            auc = baseline
        else:
            Xs = X.copy()
            for feat in features:
                if feat in Xs.columns:
                    Xs[feat] = Xs[feat] * (1.0 + shock)
            auc = _auc(model, Xs, y)
        rows.append(
            {"shock": shock, "auc_roc": round(auc, 6), "auc_drift": round(auc - baseline, 6)}
        )
    return rows


def input_noise_robustness(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    features: list[str],
    noise_levels: tuple[float, ...] = DEFAULT_NOISE_LEVELS,
    seed: int = 42,
) -> list[dict[str, float]]:
    baseline = _auc(model, X, y)
    rng = np.random.default_rng(seed)
    stds = {f: float(X[f].std()) or 1.0 for f in features if f in X.columns}
    rows: list[dict[str, float]] = []
    for level in noise_levels:
        if level == 0.0:
            auc = baseline
        else:
            Xn = X.copy()
            for feat, std in stds.items():
                Xn[feat] = Xn[feat] + rng.normal(0.0, level * std, size=len(Xn))
            auc = _auc(model, Xn, y)
        rows.append(
            {"noise": level, "auc_roc": round(auc, 6), "auc_drift": round(auc - baseline, 6)}
        )
    return rows


def feature_masking_robustness(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    ranked_features: list[str],
    mask_counts: tuple[int, ...] = DEFAULT_MASK_COUNTS,
) -> list[dict[str, float]]:
    """Zero out the top-k most important features (in standardized space, a
    zero approximates the feature mean) and measure AUC drift."""
    baseline = _auc(model, X, y)
    means = {f: float(X[f].mean()) for f in ranked_features if f in X.columns}
    rows: list[dict[str, float]] = []
    for k in mask_counts:
        Xm = X.copy()
        masked = [f for f in ranked_features[:k] if f in Xm.columns]
        for feat in masked:
            Xm[feat] = means[feat]
        auc = _auc(model, Xm, y)
        rows.append(
            {
                "masked_top_k": k,
                "auc_roc": round(auc, 6),
                "auc_drift": round(auc - baseline, 6),
                "masked_features": ", ".join(masked),
            }
        )
    return rows
