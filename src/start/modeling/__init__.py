"""Modeling utilities for StART demos: datasets, model factory, tuning,
metrics, explainability, and sensitivity. Core-light: only sklearn required;
xgboost/lightgbm/optuna/shap are optional with explicit fallbacks."""

from start.modeling.data import (
    SCORE_COLUMN,
    TARGET_COLUMN,
    feature_columns,
    load_attrition_dataset,
    three_way_split,
)
from start.modeling.explain import ImportanceResult, global_importance, shap_available
from start.modeling.metrics import (
    METRIC_NAMES,
    cohort_comparison,
    compute_cohort_metrics,
    top_decile_lift,
)
from start.modeling.models import (
    HYPERPARAM_SPACES,
    MODEL_CHOICES,
    lightgbm_available,
    resolve_model,
    xgboost_available,
)
from start.modeling.sensitivity import DEFAULT_SHOCKS, run_feature_shocks

__all__ = [
    "SCORE_COLUMN",
    "TARGET_COLUMN",
    "feature_columns",
    "load_attrition_dataset",
    "three_way_split",
    "ImportanceResult",
    "global_importance",
    "shap_available",
    "METRIC_NAMES",
    "cohort_comparison",
    "compute_cohort_metrics",
    "top_decile_lift",
    "HYPERPARAM_SPACES",
    "MODEL_CHOICES",
    "lightgbm_available",
    "resolve_model",
    "xgboost_available",
    "DEFAULT_SHOCKS",
    "run_feature_shocks",
]
