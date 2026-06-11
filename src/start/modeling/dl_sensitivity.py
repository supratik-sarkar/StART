"""DL robustness and sensitivity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from start.modeling.dl_training import predict_proba


def top_feature_shock_sensitivity(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    top_features: list[str],
    feature_names: list[str],
    *,
    shocks: tuple[float, ...] = (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30),
) -> pd.DataFrame:
    baseline = float(roc_auc_score(y, predict_proba(model, X)))
    idx = [feature_names.index(f) for f in top_features if f in feature_names]
    rows = []
    for shock in shocks:
        Xs = X.copy()
        if idx:
            Xs[:, idx] = Xs[:, idx] * (1.0 + shock)
        auc = float(roc_auc_score(y, predict_proba(model, Xs)))
        rows.append({"shock": shock, "auc_roc": auc, "auc_drift": auc - baseline})
    return pd.DataFrame(rows)


def input_noise_robustness(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    *,
    noise_levels: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05, 0.10),
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline = float(roc_auc_score(y, predict_proba(model, X)))
    rows = []
    for sigma in noise_levels:
        Xn = X + rng.normal(0, sigma, size=X.shape).astype(X.dtype)
        auc = float(roc_auc_score(y, predict_proba(model, Xn)))
        rows.append({"noise_sigma": sigma, "auc_roc": auc, "auc_drift": auc - baseline})
    return pd.DataFrame(rows)


def feature_masking_robustness(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    top_features: list[str],
    feature_names: list[str],
) -> pd.DataFrame:
    baseline = float(roc_auc_score(y, predict_proba(model, X)))
    rows = []
    for k in (1, 3, 5):
        selected = top_features[: min(k, len(top_features))]
        idx = [feature_names.index(f) for f in selected if f in feature_names]
        Xm = X.copy()
        if idx:
            Xm[:, idx] = 0.0
        auc = float(roc_auc_score(y, predict_proba(model, Xm)))
        rows.append({"masked_top_k": len(idx), "auc_roc": auc, "auc_drift": auc - baseline})
    return pd.DataFrame(rows)
