"""Explainability with honest fallbacks.

SHAP TreeExplainer is used when the `shap` package is installed AND the model
is tree-based. Otherwise we fall back to permutation importance and SAY SO:
the method actually used is recorded in the result and propagated into
evidence. Local (per-row) explanations are only produced on the SHAP path;
the permutation fallback records that local attribution is unavailable.

Note: Optuna/Hyperopt are hyperparameter-optimization tools, not
explainability engines — explainability is model-specific (SHAP / permutation
here; Captum-style gradient methods for deep learning on the roadmap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ImportanceResult:
    method: str  # "shap" | "permutation"
    global_importance: list[tuple[str, float]]  # sorted desc by |importance|
    local_examples: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def top_features(self, k: int = 5) -> list[str]:
        return [name for name, _ in self.global_importance[:k]]


def shap_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False


def _is_tree_model(model: Any) -> bool:
    mod = type(model).__module__
    return any(key in mod for key in ("ensemble", "tree", "xgboost", "lightgbm"))


def global_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    seed: int = 42,
    use_shap: bool | None = None,
    n_local_examples: int = 5,
    max_shap_rows: int = 500,
) -> ImportanceResult:
    """Global feature importance with explicit method attribution."""
    wants_shap = shap_available() if use_shap is None else use_shap
    if wants_shap and _is_tree_model(model):
        try:
            return _shap_importance(model, X, seed, n_local_examples, max_shap_rows)
        except Exception as exc:  # honest fallback, never silent
            fallback = _permutation_importance(model, X, y, seed)
            fallback.note = f"SHAP failed ({type(exc).__name__}); used permutation importance."
            return fallback
    result = _permutation_importance(model, X, y, seed)
    if wants_shap and not _is_tree_model(model):
        result.note = "Model is not tree-based; used permutation importance instead of TreeExplainer."
    elif not wants_shap:
        result.note = (
            "shap is not installed (pip install -e \".[xai]\"); used permutation importance. "
            "Local attributions unavailable on this path."
        )
    return result


def _shap_importance(
    model: Any, X: pd.DataFrame, seed: int, n_local: int, max_rows: int
) -> ImportanceResult:
    import shap

    sample = X.sample(n=min(max_rows, len(X)), random_state=seed)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)
    if isinstance(values, list):  # sklearn binary: [class0, class1]
        values = values[1]
    if getattr(values, "ndim", 2) == 3:  # (n, features, classes)
        values = values[:, :, 1]
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    global_imp = [(str(sample.columns[i]), round(float(mean_abs[i]), 6)) for i in order]

    locals_: list[dict[str, Any]] = []
    for row_i in range(min(n_local, len(sample))):
        contrib = values[row_i]
        top_idx = np.argsort(np.abs(contrib))[::-1][:3]
        locals_.append(
            {
                "row_index": int(sample.index[row_i]),
                "top_contributions": {
                    str(sample.columns[i]): round(float(contrib[i]), 6) for i in top_idx
                },
            }
        )
    return ImportanceResult(method="shap", global_importance=global_imp, local_examples=locals_)


def _permutation_importance(
    model: Any, X: pd.DataFrame, y: pd.Series | np.ndarray, seed: int
) -> ImportanceResult:
    from sklearn.inspection import permutation_importance

    imp = permutation_importance(model, X, y, n_repeats=5, random_state=seed, n_jobs=-1)
    order = np.argsort(imp.importances_mean)[::-1]
    global_imp = [(str(X.columns[i]), round(float(imp.importances_mean[i]), 6)) for i in order]
    return ImportanceResult(method="permutation", global_importance=global_imp)


# --------------------------------------------------------------------------- #
# ExplainabilityRouter: explainability is never tied to a single library.
# The router chooses model-appropriate methods and is honest about what is
# implemented today vs roadmap.
# --------------------------------------------------------------------------- #
@dataclass
class ExplainabilityPlan:
    model_family: str
    methods: list[tuple[str, bool]]  # (method, implemented_now)

    def implemented(self) -> list[str]:
        return [m for m, ok in self.methods if ok]

    def roadmap(self) -> list[str]:
        return [m for m, ok in self.methods if not ok]


def detect_model_family(model: Any = None, model_family: str | None = None) -> str:
    if model_family:
        return model_family
    if model is None:
        return "unknown"
    declared = getattr(model, "_start_model_family", None)
    if declared:
        return str(declared)
    mod = type(model).__module__
    name = type(model).__name__.lower()
    if any(k in mod for k in ("xgboost", "lightgbm")) or any(
        k in mod for k in ("ensemble", "tree")
    ):
        return "tree"
    if "linear" in mod or "logistic" in name:
        return "linear"
    if "torch" in mod:
        return "transformer" if "transformer" in name or "attention" in name else "deep_learning"
    return "unknown"


def _captum_ready() -> bool:
    from start.modeling.deep_learning import captum_available, torch_available

    return torch_available() and captum_available()


def route_explainability(model: Any = None, model_family: str | None = None) -> ExplainabilityPlan:
    family = detect_model_family(model, model_family)
    shap_ok = shap_available()
    routes: dict[str, list[tuple[str, bool]]] = {
        "tree": [("shap_tree_explainer", shap_ok), ("permutation_importance", True)],
        "linear": [("coefficients", True), ("permutation_importance", True)],
        "deep_learning": [
            ("integrated_gradients", _captum_ready()),
            ("deeplift", False),
            ("gradient_shap", False),
            ("permutation_sensitivity", True),
            ("occlusion_analysis", False),
        ],
        "transformer": [
            ("attention_attribution", False),
            ("integrated_gradients", False),
            ("permutation_sensitivity", True),
        ],
        "multimodal": [("modality_contribution_attribution", False)],
        "unknown": [("permutation_importance", True)],
    }
    return ExplainabilityPlan(model_family=family, methods=routes.get(family, routes["unknown"]))
