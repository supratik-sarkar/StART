"""Cost-sensitive prediction utilities (v3.1.1).

Provides post-hoc expected-cost minimization and cost-matrix validation
for multiclass classification. The expected-cost formula:

    expected_cost = probabilities @ cost_matrix
    predicted_labels = classes[np.argmin(expected_cost, axis=1)]

This module also provides utilities to derive class weights from a cost
specification and to validate cost matrix structure.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_cost_matrix(
    matrix: dict[str, dict[str, float]],
    classes: list[str],
) -> list[str]:
    """Validate a cost matrix against the dataset's class ordering.

    Returns a list of error messages (empty = valid).

    Checks:
    - rows and columns follow the exact classes_ ordering
    - matrix shape is K × K
    - values are finite and non-negative
    - diagonal is normally zero (warning, not error)
    - all dataset classes are represented
    """
    errors: list[str] = []

    if set(matrix.keys()) != set(classes):
        missing = set(classes) - set(matrix.keys())
        extra = set(matrix.keys()) - set(classes)
        if missing:
            errors.append(f"Missing rows for classes: {sorted(missing)}")
        if extra:
            errors.append(f"Extra rows not in classes: {sorted(extra)}")

    for ci in matrix:
        if set(matrix[ci].keys()) != set(classes):
            missing_cols = set(classes) - set(matrix[ci].keys())
            extra_cols = set(matrix[ci].keys()) - set(classes)
            if missing_cols:
                errors.append(f"Row '{ci}' missing columns: {sorted(missing_cols)}")
            if extra_cols:
                errors.append(f"Row '{ci}' has extra columns: {sorted(extra_cols)}")

        for cj, cost in matrix[ci].items():
            if not np.isfinite(cost):
                errors.append(f"Cost({ci},{cj}) = {cost} is not finite")
            if cost < 0:
                errors.append(f"Cost({ci},{cj}) = {cost} is negative")
            if ci == cj and cost != 0:
                errors.append(
                    f"Warning: diagonal Cost({ci},{cj}) = {cost} is non-zero "
                    "(expected 0 for correct predictions)"
                )

    return errors


def cost_matrix_to_numpy(
    matrix: dict[str, dict[str, float]],
    classes: list[str],
) -> np.ndarray:
    """Convert a dict-of-dict cost matrix into a K×K numpy array.

    Rows = true classes, Columns = predicted classes, following classes order.
    """
    K = len(classes)
    arr = np.zeros((K, K), dtype=np.float64)
    for i, ci in enumerate(classes):
        for j, cj in enumerate(classes):
            arr[i, j] = matrix.get(ci, {}).get(cj, 0.0)
    return arr


def cost_sensitive_predictions(
    probabilities: np.ndarray,
    cost_matrix: np.ndarray,
    classes: np.ndarray,
) -> np.ndarray:
    """Apply expected-cost minimization.

    Args:
        probabilities: (N, K) class probability matrix
        cost_matrix: (K, K) cost matrix where C[i,j] = cost of predicting j
                     when true class is i
        classes: (K,) array of class labels in the same order

    Returns:
        predicted_labels: (N,) array using classes[argmin(expected_cost)]
    """
    expected_cost = probabilities @ cost_matrix
    return classes[np.argmin(expected_cost, axis=1)]


def derive_class_weights_from_spec(
    cost_spec: dict[str, Any],
    classes: list[str],
) -> dict[str, float] | None:
    """Derive per-class weights from a cost specification.

    For "balanced": returns None (let sklearn handle it).
    For "critical_class": returns a weight dict with the critical class upweighted.
    For "matrix": cannot be reduced to class weights — returns None.
    """
    spec_type = cost_spec.get("type", "balanced")

    if spec_type == "balanced":
        return None

    if spec_type == "critical_class":
        crit = cost_spec.get("critical_class", "")
        rel_cost = cost_spec.get("relative_cost", 5.0)
        weights = {}
        for c in classes:
            weights[c] = rel_cost if c == crit else 1.0
        return weights

    # "matrix" type cannot be faithfully reduced to class weights
    return None


def cost_spec_to_matrix(
    cost_spec: dict[str, Any],
    classes: list[str],
) -> np.ndarray | None:
    """Convert a cost specification to a K×K numpy cost matrix.

    Only "matrix" type produces a full matrix.
    "critical_class" produces a simplified matrix with higher off-diagonal
    costs for the critical class row.
    "balanced" returns None.
    """
    spec_type = cost_spec.get("type", "balanced")

    if spec_type == "balanced":
        return None

    if spec_type == "matrix":
        return cost_matrix_to_numpy(cost_spec["matrix"], classes)

    if spec_type == "critical_class":
        crit = cost_spec.get("critical_class", "")
        rel_cost = cost_spec.get("relative_cost", 5.0)
        K = len(classes)
        arr = np.ones((K, K), dtype=np.float64)
        np.fill_diagonal(arr, 0.0)
        if crit in classes:
            crit_idx = classes.index(crit)
            for j in range(K):
                if j != crit_idx:
                    arr[crit_idx, j] = rel_cost
        return arr

    return None
