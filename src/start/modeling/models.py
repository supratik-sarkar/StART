"""Model factory: Random Forest (always available), XGBoost / LightGBM optional.

XGBoost and LightGBM are NOT core dependencies. When unavailable, resolution
degrades cleanly to Random Forest with an explicit note that is surfaced to
the user and recorded in run metadata.
"""

from __future__ import annotations

from typing import Any

MODEL_CHOICES = (
    "random_forest",
    "xgboost",
    "lightgbm",
    "mlp",
    "catboost",
    "distributed_random_forest",
    "extra_trees",
    "random_rotation_forest",
    "rnn",
    "lstm",
    "gru",
    "bi_lstm",
    "cnn",
    "simple_cnn_small",
    "simple_cnn_medium",
    "simple_cnn_deep"
)

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
    "catboost": {
        "iterations": {"type": "int", "grid": [100, 200, 400], "low": 100, "high": 600, "step": 50},
        "depth": {"type": "int", "grid": [4, 6, 8], "low": 3, "high": 10, "step": 1},
        "learning_rate": {"type": "float", "grid": [0.03, 0.1, 0.3], "low": 0.01, "high": 0.3, "log": True},
        "l2_leaf_reg": {"type": "float", "grid": [1.0, 3.0, 5.0], "low": 1.0, "high": 10.0},
        "border_count": {"type": "int", "grid": [32, 128, 254], "low": 32, "high": 255, "step": 1},
    },
    "distributed_random_forest": {
        "n_estimators": {"type": "int", "grid": [100, 200, 400], "low": 100, "high": 600, "step": 50},
        "max_depth": {"type": "int", "grid": [4, 8, 16], "low": 3, "high": 24, "step": 1},
        "min_samples_split": {"type": "int", "grid": [2, 5, 10], "low": 2, "high": 20, "step": 1},
        "min_samples_leaf": {"type": "int", "grid": [1, 2, 5], "low": 1, "high": 10, "step": 1},
        "max_features": {"type": "cat", "grid": ["sqrt", "log2", None], "choices": ["sqrt", "log2", None]},
    },
    "extra_trees": {
        "n_estimators": {"type": "int", "grid": [100, 200, 400], "low": 100, "high": 600, "step": 50},
        "max_depth": {"type": "int", "grid": [4, 8, 16], "low": 3, "high": 24, "step": 1},
        "min_samples_split": {"type": "int", "grid": [2, 5, 10], "low": 2, "high": 20, "step": 1},
        "min_samples_leaf": {"type": "int", "grid": [1, 2, 5], "low": 1, "high": 10, "step": 1},
        "max_features": {"type": "cat", "grid": ["sqrt", "log2", None], "choices": ["sqrt", "log2", None]},
    },
    "random_rotation_forest": {
        "n_estimators": {"type": "int", "grid": [100, 200, 400], "low": 100, "high": 600, "step": 50},
        "max_depth": {"type": "int", "grid": [4, 8, 16], "low": 3, "high": 24, "step": 1},
        "min_samples_split": {"type": "int", "grid": [2, 5, 10], "low": 2, "high": 20, "step": 1},
        "min_samples_leaf": {"type": "int", "grid": [1, 2, 5], "low": 1, "high": 10, "step": 1},
        "max_features": {"type": "cat", "grid": ["sqrt", "log2", None], "choices": ["sqrt", "log2", None]},
    },
    "rnn": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {"type": "float", "grid": [3e-4, 1e-3, 3e-3], "low": 1e-4, "high": 1e-2, "log": True},
        "dropout": {"type": "float", "grid": [0.0, 0.1, 0.3], "low": 0.0, "high": 0.5},
    },
    "lstm": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {"type": "float", "grid": [3e-4, 1e-3, 3e-3], "low": 1e-4, "high": 1e-2, "log": True},
        "dropout": {"type": "float", "grid": [0.0, 0.1, 0.3], "low": 0.0, "high": 0.5},
    },
    "gru": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {"type": "float", "grid": [3e-4, 1e-3, 3e-3], "low": 1e-4, "high": 1e-2, "log": True},
        "dropout": {"type": "float", "grid": [0.0, 0.1, 0.3], "low": 0.0, "high": 0.5},
    },
    "bi_lstm": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {"type": "float", "grid": [3e-4, 1e-3, 3e-3], "low": 1e-4, "high": 1e-2, "log": True},
        "dropout": {"type": "float", "grid": [0.0, 0.1, 0.3], "low": 0.0, "high": 0.5},
    },
    "cnn": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {"type": "float", "grid": [3e-4, 1e-3, 3e-3], "low": 1e-4, "high": 1e-2, "log": True},
    },
    "simple_cnn_small": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {"type": "float", "grid": [3e-4, 1e-3, 3e-3], "low": 1e-4, "high": 1e-2, "log": True},
    },
    "simple_cnn_medium": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {"type": "float", "grid": [3e-4, 1e-3, 3e-3], "low": 1e-4, "high": 1e-2, "log": True},
    },
    "simple_cnn_deep": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {"type": "float", "grid": [3e-4, 1e-3, 3e-3], "low": 1e-4, "high": 1e-2, "log": True},
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


def catboost_available() -> bool:
    try:
        import catboost  # noqa: F401

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
    if name == "catboost":
        if not catboost_available():
            return (
                _make_random_forest(seed),
                "random_forest",
                "catboost is not installed; falling back to Random Forest.",
            )
        from catboost import CatBoostClassifier
        return (
            CatBoostClassifier(iterations=200, random_seed=seed, verbose=0),
            "catboost",
            "",
        )
    if name == "distributed_random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return (
            RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
            "distributed_random_forest",
            "",
        )
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return (
            ExtraTreesClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
            "extra_trees",
            "",
        )
    if name == "random_rotation_forest":
        return (
            _make_random_forest(seed),
            "random_forest",
            "Random Rotation Forest is not standard; falling back to Random Forest.",
        )
    if name in ("rnn", "lstm", "gru", "bi_lstm"):
        from start.modeling.deep_learning import torch_available
        if not torch_available():
            return (
                _make_random_forest(seed),
                "random_forest",
                f"torch is not installed; falling back to Random Forest for {name}.",
            )
        from start.modeling.sequence_dl import SequenceClassifier
        return SequenceClassifier(family=name, random_state=seed), name, ""
    if name in ("cnn", "simple_cnn_small", "simple_cnn_medium", "simple_cnn_deep"):
        from start.modeling.deep_learning import torch_available
        if not torch_available():
            return (
                _make_random_forest(seed),
                "random_forest",
                f"torch is not installed; falling back to Random Forest for {name}.",
            )
        from start.modeling.vision_dl import VisionCNNClassifier
        arch = "simple_cnn_small" if name == "cnn" else name
        return VisionCNNClassifier(architecture=arch, random_state=seed), name, ""
    return _make_random_forest(seed), "random_forest", ""
