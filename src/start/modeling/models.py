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
    "logistic_regression",
    "gradient_boosting",
    "rnn",
    "lstm",
    "gru",
    "bi_lstm",
    "cnn",
    "simple_cnn_small",
    "simple_cnn_medium",
    "simple_cnn_deep",
)

# Five standard tunable hyperparameters per model (suggested spaces shown to
# the user in interactive mode; "grid" lists feed grid search, low/high feed
# random search and Bayesian optimization).
HYPERPARAM_SPACES: dict[str, dict[str, dict[str, Any]]] = {
    "logistic_regression": {
        "C": {"type": "float", "grid": [0.01, 0.1, 1.0, 10.0], "low": 0.001, "high": 100.0, "log": True},
        "penalty": {"type": "cat", "grid": ["l2", None], "choices": ["l2", None]},
        "solver": {"type": "cat", "grid": ["lbfgs"], "choices": ["lbfgs"]},
        "max_iter": {"type": "int", "grid": [100, 200, 500], "low": 100, "high": 1000, "step": 100},
        "tol": {"type": "float", "grid": [1e-4, 1e-3], "low": 1e-5, "high": 1e-2, "log": True},
    },
    "gradient_boosting": {
        "n_estimators": {"type": "int", "grid": [100, 200, 400], "low": 100, "high": 600, "step": 50},
        "max_depth": {"type": "int", "grid": [3, 5, 8], "low": 2, "high": 12, "step": 1},
        "learning_rate": {"type": "float", "grid": [0.03, 0.1, 0.3], "low": 0.01, "high": 0.3, "log": True},
        "subsample": {"type": "float", "grid": [0.7, 0.85, 1.0], "low": 0.5, "high": 1.0},
        "min_samples_split": {"type": "int", "grid": [2, 5, 10], "low": 2, "high": 20, "step": 1},
    },
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
            "type": "float",
            "grid": [3e-4, 1e-3, 3e-3],
            "low": 1e-4,
            "high": 1e-2,
            "log": True,
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
        "learning_rate": {
            "type": "float",
            "grid": [3e-4, 1e-3, 3e-3],
            "low": 1e-4,
            "high": 1e-2,
            "log": True,
        },
        "dropout": {"type": "float", "grid": [0.0, 0.1, 0.3], "low": 0.0, "high": 0.5},
    },
    "lstm": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {
            "type": "float",
            "grid": [3e-4, 1e-3, 3e-3],
            "low": 1e-4,
            "high": 1e-2,
            "log": True,
        },
        "dropout": {"type": "float", "grid": [0.0, 0.1, 0.3], "low": 0.0, "high": 0.5},
    },
    "gru": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {
            "type": "float",
            "grid": [3e-4, 1e-3, 3e-3],
            "low": 1e-4,
            "high": 1e-2,
            "log": True,
        },
        "dropout": {"type": "float", "grid": [0.0, 0.1, 0.3], "low": 0.0, "high": 0.5},
    },
    "bi_lstm": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {
            "type": "float",
            "grid": [3e-4, 1e-3, 3e-3],
            "low": 1e-4,
            "high": 1e-2,
            "log": True,
        },
        "dropout": {"type": "float", "grid": [0.0, 0.1, 0.3], "low": 0.0, "high": 0.5},
    },
    "cnn": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {
            "type": "float",
            "grid": [3e-4, 1e-3, 3e-3],
            "low": 1e-4,
            "high": 1e-2,
            "log": True,
        },
    },
    "simple_cnn_small": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {
            "type": "float",
            "grid": [3e-4, 1e-3, 3e-3],
            "low": 1e-4,
            "high": 1e-2,
            "log": True,
        },
    },
    "simple_cnn_medium": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {
            "type": "float",
            "grid": [3e-4, 1e-3, 3e-3],
            "low": 1e-4,
            "high": 1e-2,
            "log": True,
        },
    },
    "simple_cnn_deep": {
        "epochs": {"type": "int", "grid": [5, 8, 10], "low": 3, "high": 10, "step": 1},
        "batch_size": {"type": "int", "grid": [32, 64, 128], "low": 16, "high": 128, "step": 16},
        "learning_rate": {
            "type": "float",
            "grid": [3e-4, 1e-3, 3e-3],
            "low": 1e-4,
            "high": 1e-2,
            "log": True,
        },
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


def resolve_model(name: str, seed: int = 42, fail_closed: bool = True, **kwargs: Any) -> tuple[Any, str, str]:
    """Return (estimator, resolved_name, note).
    Enforces FAIL CLOSED contract when fail_closed=True: never silently substitutes an uninstalled estimator."""
    name = name.lower().strip()
    if name in ("tabulardlclassifier", "tabulardl", "torch_mlp", "torchmlp", "neural_net", "neural_network"):
        name = "mlp"
    elif name in ("logisticregression", "lr", "logistic"):
        name = "logistic_regression"
    elif name in ("randomforest", "randomforestclassifier", "rf"):
        name = "random_forest"
    elif name in ("gradientboosting", "gradientboostingclassifier", "gb"):
        name = "gradient_boosting"
    elif name in ("lgbm", "lightgbmclassifier"):
        name = "lightgbm"
    elif name in ("xgb", "xgboostclassifier"):
        name = "xgboost"

    if name not in MODEL_CHOICES:
        if fail_closed:
            raise ValueError(f"Unknown model '{name}'. Supported choices: {MODEL_CHOICES}")
        return (
            _make_random_forest(seed),
            "random_forest",
            f"Unknown model '{name}'; using Random Forest.",
        )
    if name == "xgboost":
        if not xgboost_available():
            if fail_closed:
                raise ValueError('xgboost is not installed (pip install -e ".[tree-models]"). FAIL CLOSED: silent substitution prohibited.')
            return (
                _make_random_forest(seed),
                "random_forest",
                'xgboost is not installed (pip install -e ".[tree-models]"); falling back to Random Forest.',
            )
        import sys

        from xgboost import XGBClassifier

        n_jobs = 1 if sys.platform == "darwin" else -1
        params = {
            "n_estimators": 200,
            "random_state": seed,
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": n_jobs,
        }
        params.update(kwargs)
        if sys.platform == "darwin" and params.get("n_jobs") == -1:
            params["n_jobs"] = 1
        return XGBClassifier(**params), "xgboost", ""

    if name == "lightgbm":
        if not lightgbm_available():
            if fail_closed:
                raise ValueError('lightgbm is not installed (pip install -e ".[tree-models]"). FAIL CLOSED: silent substitution prohibited.')
            return (
                _make_random_forest(seed),
                "random_forest",
                'lightgbm is not installed (pip install -e ".[tree-models]"); falling back to Random Forest.',
            )
        import sys

        from lightgbm import LGBMClassifier

        n_jobs = 1 if sys.platform == "darwin" else -1
        params = {"n_estimators": 200, "random_state": seed, "verbose": -1, "n_jobs": n_jobs}
        params.update(kwargs)
        if sys.platform == "darwin" and params.get("n_jobs") == -1:
            params["n_jobs"] = 1
        return LGBMClassifier(**params), "lightgbm", ""

    if name == "mlp":
        from start.modeling.deep_learning import TorchMLPClassifier, torch_available

        if not torch_available():
            if fail_closed:
                raise ValueError('torch is not installed. FAIL CLOSED: silent substitution prohibited.')
            return (
                _make_random_forest(seed),
                "random_forest",
                'torch is not installed (pip install -e ".[torch]"); falling back to Random Forest.',
            )
        params = {"random_state": seed}
        params.update(kwargs)
        return TorchMLPClassifier(**params), "mlp", ""

    if name == "catboost":
        if not catboost_available():
            if fail_closed:
                raise ValueError("catboost is not installed in current environment. FAIL CLOSED: silent substitution prohibited.")
            return (
                _make_random_forest(seed),
                "random_forest",
                "catboost is not installed; falling back to Random Forest.",
            )
        from catboost import CatBoostClassifier

        params = {"iterations": 200, "random_seed": seed, "verbose": 0}
        params.update(kwargs)
        return CatBoostClassifier(**params), "catboost", ""

    if name in ("random_forest", "rf"):
        import sys

        from sklearn.ensemble import RandomForestClassifier

        n_jobs = 1 if sys.platform == "darwin" else -1
        params = {"n_estimators": 200, "random_state": seed, "n_jobs": n_jobs}
        params.update(kwargs)
        if sys.platform == "darwin" and params.get("n_jobs") == -1:
            params["n_jobs"] = 1
        return RandomForestClassifier(**params), "random_forest", ""

    if name == "distributed_random_forest":
        import sys

        from sklearn.ensemble import RandomForestClassifier

        n_jobs = 1 if sys.platform == "darwin" else -1
        params = {"n_estimators": 200, "random_state": seed, "n_jobs": n_jobs}
        params.update(kwargs)
        if sys.platform == "darwin" and params.get("n_jobs") == -1:
            params["n_jobs"] = 1
        return RandomForestClassifier(**params), "distributed_random_forest", ""

    if name == "extra_trees":
        import sys

        from sklearn.ensemble import ExtraTreesClassifier

        n_jobs = 1 if sys.platform == "darwin" else -1
        params = {"n_estimators": 200, "random_state": seed, "n_jobs": n_jobs}
        params.update(kwargs)
        if sys.platform == "darwin" and params.get("n_jobs") == -1:
            params["n_jobs"] = 1
        return ExtraTreesClassifier(**params), "extra_trees", ""

    if name == "random_rotation_forest":
        if fail_closed:
            raise ValueError("random_rotation_forest is not available in current environment. FAIL CLOSED.")
        return (
            _make_random_forest(seed),
            "random_forest",
            "Random Rotation Forest is not standard; falling back to Random Forest.",
        )
    if name in ("rnn", "lstm", "gru", "bi_lstm"):
        from start.modeling.deep_learning import torch_available

        if not torch_available():
            if fail_closed:
                raise ValueError(f"torch is not installed for {name}. FAIL CLOSED.")
            return (
                _make_random_forest(seed),
                "random_forest",
                f"torch is not installed; falling back to Random Forest for {name}.",
            )
        from start.modeling.sequence_dl import SequenceClassifier

        params = {"family": name, "random_state": seed}
        params.update(kwargs)
        return SequenceClassifier(**params), name, ""

    if name in ("cnn", "simple_cnn_small", "simple_cnn_medium", "simple_cnn_deep"):
        from start.modeling.deep_learning import torch_available

        if not torch_available():
            if fail_closed:
                raise ValueError(f"torch is not installed for {name}. FAIL CLOSED.")
            return (
                _make_random_forest(seed),
                "random_forest",
                f"torch is not installed; falling back to Random Forest for {name}.",
            )
        from start.modeling.vision_dl import VisionCNNClassifier

        arch = "simple_cnn_small" if name == "cnn" else name
        params = {"architecture": arch, "random_state": seed}
        params.update(kwargs)
        return VisionCNNClassifier(**params), name, ""

    if name in ("logistic_regression", "logistic"):
        from sklearn.linear_model import LogisticRegression

        params = {"random_state": seed, "max_iter": 500}
        params.update(kwargs)
        return LogisticRegression(**params), "logistic_regression", ""

    if name in ("gradient_boosting", "gb"):
        from sklearn.ensemble import GradientBoostingClassifier

        params = {"n_estimators": 200, "random_state": seed}
        params.update(kwargs)
        return GradientBoostingClassifier(**params), "gradient_boosting", ""

    params = {"n_estimators": 200, "random_state": seed, "n_jobs": -1}
    params.update(kwargs)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(**params), "random_forest", ""
