"""Deep-learning explainability helpers.

Captum methods are optional. Permutation importance is always available and is
used as the safe default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from start.modeling.dl_training import predict_proba


@dataclass(frozen=True)
class DLAttributionResult:
    method: str
    table: pd.DataFrame


def permutation_importance(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    seed: int = 42,
    max_features: int = 50,
) -> DLAttributionResult:
    rng = np.random.default_rng(seed)
    baseline = float(roc_auc_score(y, predict_proba(model, X)))
    rows = []
    for j, name in enumerate(feature_names):
        Xp = X.copy()
        rng.shuffle(Xp[:, j])
        auc = float(roc_auc_score(y, predict_proba(model, Xp)))
        rows.append({"feature": name, "importance": baseline - auc, "baseline_auc": baseline, "permuted_auc": auc})
    table = pd.DataFrame(rows).sort_values("importance", ascending=False).head(max_features).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return DLAttributionResult(method="permutation", table=table)


def integrated_gradients_importance(
    model: object,
    X: np.ndarray,
    feature_names: list[str],
    *,
    max_rows: int = 128,
    max_features: int = 50,
) -> DLAttributionResult:
    try:
        import torch
        from captum.attr import IntegratedGradients
    except Exception as exc:
        raise ImportError("Integrated Gradients requires captum and torch") from exc

    device = next(model.parameters()).device  # type: ignore[attr-defined]
    model.eval()  # type: ignore[attr-defined]

    def forward_fn(inputs):
        return torch.sigmoid(model(inputs))  # type: ignore[operator]

    sample = torch.tensor(X[:max_rows], dtype=torch.float32, device=device)
    baseline = torch.zeros_like(sample)
    ig = IntegratedGradients(forward_fn)
    attr = ig.attribute(sample, baselines=baseline).detach().cpu().numpy()
    scores = np.mean(np.abs(attr), axis=0)
    table = pd.DataFrame({"feature": feature_names, "importance": scores})
    table = table.sort_values("importance", ascending=False).head(max_features).reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return DLAttributionResult(method="integrated_gradients", table=table)


def choose_explainability(
    model: object,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    *,
    prefer_captum: bool = True,
) -> DLAttributionResult:
    if prefer_captum:
        try:
            return integrated_gradients_importance(model, X, feature_names)
        except Exception:
            pass
    return permutation_importance(model, X, y, feature_names)
