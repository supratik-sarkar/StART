"""Deep-learning explainability with honest method routing.

Preference order for a fitted torch model:
    1. Integrated Gradients (Captum)        -- requires captum
    2. Gradient SHAP (Captum)               -- optional, requires captum
    3. permutation importance (sklearn)     -- always-available fallback

The method actually executed is always recorded. Captum / Integrated
Gradients / Gradient SHAP is NEVER claimed unless it ran; SHAP-for-trees is
never claimed for a DL model. When Captum is absent the permutation fallback
is used and the note says so explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class DLImportanceResult:
    method: str  # integrated_gradients | gradient_shap | permutation
    global_importance: list[tuple[str, float]]
    note: str = ""
    available_methods: list[str] = field(default_factory=list)

    def top_features(self, k: int = 5) -> list[str]:
        return [name for name, _ in self.global_importance[:k]]


def _captum_attribution(
    model,
    X: pd.DataFrame,
    method: str,
    n_samples: int,
    seed: int,
) -> list[tuple[str, float]]:
    import torch
    from captum.attr import GradientShap, IntegratedGradients

    sample = X.sample(n=min(n_samples, len(X)), random_state=seed)
    X_arr = model._standardize(model._to_numpy(sample), fit=False)
    device = torch.device(model.device_used)
    inputs = torch.tensor(X_arr, dtype=torch.float32, device=device, requires_grad=True)
    model._net.eval()
    forward = lambda t: model._net(t)  # noqa: E731

    if method == "gradient_shap":
        # baseline distribution = standardized zeros + small noise for GradientShap
        rng = torch.Generator(device="cpu").manual_seed(seed)
        baselines = torch.cat(
            [torch.zeros_like(inputs), torch.randn(inputs.shape, generator=rng).to(device) * 0.1]
        )
        explainer = GradientShap(forward)
        attributions = explainer.attribute(inputs, baselines=baselines, n_samples=8, stdevs=0.09)
    else:  # integrated_gradients
        explainer = IntegratedGradients(forward)
        attributions = explainer.attribute(inputs, baselines=torch.zeros_like(inputs), n_steps=32)

    mean_abs = attributions.abs().mean(dim=0).detach().cpu().numpy()
    order = np.argsort(mean_abs)[::-1]
    return [(str(sample.columns[i]), round(float(mean_abs[i]), 6)) for i in order]


def _permutation_importance(model, X: pd.DataFrame, y, seed: int) -> list[tuple[str, float]]:
    from sklearn.inspection import permutation_importance

    imp = permutation_importance(model, X, y, n_repeats=5, random_state=seed, n_jobs=1)
    order = np.argsort(imp.importances_mean)[::-1]
    return [(str(X.columns[i]), round(float(imp.importances_mean[i]), 6)) for i in order]


def dl_global_importance(
    model,
    X: pd.DataFrame,
    y=None,
    *,
    prefer: str = "integrated_gradients",
    n_samples: int = 200,
    seed: int = 42,
) -> DLImportanceResult:
    """Global feature importance for a deep-learning model with honest routing.

    prefer: 'integrated_gradients' (default) or 'gradient_shap'. Falls back to
    permutation importance when Captum is unavailable or the preferred method
    fails, recording exactly what ran.
    """
    from start.modeling.deep_learning import (
        TorchMLPClassifier,
        captum_available,
        torch_available,
    )

    available = ["permutation"]
    if torch_available() and captum_available():
        available = ["integrated_gradients", "gradient_shap", "permutation"]

    fitted = isinstance(model, TorchMLPClassifier) and model._net is not None
    if not fitted:
        if y is None:
            return DLImportanceResult(
                "unavailable", [], "Model is not a fitted DL classifier.", available
            )
        ranked = _permutation_importance(model, X, y, seed)
        return DLImportanceResult("permutation", ranked, "Model is not a torch DL classifier.", available)

    if torch_available() and captum_available() and prefer in {"integrated_gradients", "gradient_shap"}:
        try:
            ranked = _captum_attribution(model, X, prefer, n_samples, seed)
            return DLImportanceResult(prefer, ranked, "", available)
        except Exception as exc:  # honest fallback, never silent
            if y is not None:
                ranked = _permutation_importance(model, X, y, seed)
                return DLImportanceResult(
                    "permutation",
                    ranked,
                    f"{prefer} failed ({type(exc).__name__}); used permutation importance.",
                    available,
                )
            return DLImportanceResult("unavailable", [], f"{prefer} failed: {exc}", available)

    if y is None:
        return DLImportanceResult(
            "unavailable",
            [],
            "Captum not installed (pip install -e \".[torch]\"); permutation importance "
            "needs labels (pass y).",
            available,
        )
    ranked = _permutation_importance(model, X, y, seed)
    return DLImportanceResult(
        "permutation",
        ranked,
        "Captum not installed (pip install -e \".[torch]\"); used permutation importance.",
        available,
    )
