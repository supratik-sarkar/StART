"""Recursive serializer for converting numpy and custom types into standard JSON primitives."""

from __future__ import annotations

from typing import Any

import numpy as np


def sanitize_json_primitives(obj: Any) -> Any:
    """Recursively convert numpy types, NaN/Inf, and non-serializable objects to JSON primitives."""
    if isinstance(obj, dict):
        return {str(k): sanitize_json_primitives(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [sanitize_json_primitives(v) for v in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_json_primitives(obj.tolist())
    elif hasattr(obj, "to_dict") and not isinstance(obj, type):
        try:
            return sanitize_json_primitives(obj.to_dict())
        except Exception:
            return str(obj)
    elif hasattr(obj, "model_dump") and not isinstance(obj, type):
        try:
            return sanitize_json_primitives(obj.model_dump())
        except Exception:
            return str(obj)
    return obj
