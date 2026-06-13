"""Model factory: Random Forest (always available), XGBoost / LightGBM optional.

XGBoost and LightGBM are NOT core dependencies. When unavailable, resolution
degrades cleanly to Random Forest with an explicit note that is surfaced to
the user and recorded in run metadata.
"""

from __future__ import annotations

from typing import Any

MODEL_CHOICES = ("random_forest", "xgboost", "lightgbm", "mlp")

# Five standard tunable hyperparameters per model (suggested spaces shown to
# the user in interactive mode; "grid" lists feed grid search, low/high feed
# random search and Bayesian optimization).
HYPERPARAM_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "random_forest": {
        "n_estimators": {"type": "int", "grid": [100, 200, 400], "low": 100, "high": 600, "step": 50},
        "max_depth": {"type": "int", "grid": [4, 8, 16], "low": 3, "high": 24, "step": 1},
        "min_samples_split": {"type": "int", "grid": [2, 5, 10], "low": 2, "high": 20, "step": 1},
        "min_samples_leaf": {"type": "int", "grid": [1, 2, 5], "low": 1, "high": 10, "step": 1},
        "max_features": {"type": "cat", "grid": ["sqrt", "log2", None], "choices": ["sqrt", "log2", None]},
    },
    "xgboost": {
        "n_estimators": {"type": "int", "grid": [100, 200, 400], "low": 100, "high": 600, "step": 50},
        "max_depth": {"type": "int", "grid": [3, 5, 8], "low": 2, "high": 12, "step": 1},
        "learning_rate": {"type": "float", "grid": [0.03, 0.1, 0.3], "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "grid": [0.7, 0.85, 1.0], "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "grid": [0.7, 0.85, 1.0], "low": 0.5, "high": 1.0},
    },
    "lightgbm": {
        "n_estimators": {"type": "int", "grid": [100, 200, 400], "low": 100, "high": 600, "step": 50},
        "num_leaves": {"type": "int", "grid": [15, 31, 63], "low": 7, "high": 127, "step": 2},
        "learning_rate": {"type": "float", "grid": [0.03, 0.1, 0.3], "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "grid": [0.7, 0.85, 1.0], "low": 0.5, "high": 1.0},
        "colsample_bytree": {"type": "float", "grid": [0.7, 0.85, 1.0], "low": 0.5, "high": 1.0},
    },
    # Laptop-safe by design: epochs capped at 10, batch size at 128.
    "mlp": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {
            "type": "float", "grid": [3e-4, 1e-3, 3e-3], "low": 1e-4, "high": 1e-2, "log": True
        },
        "dropout": {"type": "float", "grid": [0.0, 0.1, 0.3], "low": 0.0, "high": 0.5},
        "activation": {"type": "cat", "grid": ["relu", "leaky_relu"], "choices": ["relu", "leaky_relu"]},
    },
}


def xgboost_available() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except ImportError:
        return False


def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401

        return True
    except ImportError:
        return False


def _make_random_forest(seed: int) -> Any:
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)


def resolve_model(name: str, seed: int = 42) -> tuple[Any, str, str]:
    """Return (estimator, resolved_name, note). Degrades to Random Forest
    with an explicit note when an optional backend is unavailable."""
    name = name.lower().strip()
    if name not in MODEL_CHOICES:
        return (
            _make_random_forest(seed),
            "random_forest",
            f"Unknown model '{name}'; using Random Forest.",
        )
    if name == "xgboost":
        if not xgboost_available():
            return (
                _make_random_forest(seed),
                "random_forest",
                "xgboost is not installed (pip install -e \".[tree-models]\"); "
                "falling back to Random Forest.",
            )
        from xgboost import XGBClassifier

        return (
            XGBClassifier(
                n_estimators=200,
                random_state=seed,
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
            ),
            "xgboost",
            "",
        )
    if name == "lightgbm":
        if not lightgbm_available():
            return (
                _make_random_forest(seed),
                "random_forest",
                "lightgbm is not installed (pip install -e \".[tree-models]\"); "
                "falling back to Random Forest.",
            )
        from lightgbm import LGBMClassifier

        return (
            LGBMClassifier(n_estimators=200, random_state=seed, verbose=-1, n_jobs=-1),
            "lightgbm",
            "",
        )
    if name == "mlp":
        from start.modeling.deep_learning import TorchMLPClassifier, torch_available

        if not torch_available():
            return (
                _make_random_forest(seed),
                "random_forest",
                "torch is not installed (pip install -e \".[torch]\"); "
                "falling back to Random Forest.",
            )
        return TorchMLPClassifier(random_state=seed), "mlp", ""
    return _make_random_forest(seed), "random_forest", ""
