"""Sensitivity analysis via feature shocks.

For the top-N most important features, shock each individually across a grid of
multiplicative perturbations (-30%..+30%), re-score the model, and measure
metric drift from the unshocked baseline. The 0% row is, by construction, the
original model score.

The metric follows the same priority routing as tuning/validation (e.g.
recall/PR-AUC when false negatives are costlier). Deterministic and
model-agnostic: works with any fitted estimator exposing predict_proba/predict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_SHOCKS = (-0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30)


@dataclass
class ShockRow:
    feature: str
    shock: float
    metric_value: float
    drift: float


@dataclass
class SensitivityResult:
    metric_name: str
    baseline: float
    shock_rows: list[ShockRow] = field(default_factory=list)
    most_sensitive_feature: str | None = None
    max_abs_drift: float = 0.0
    interpretation: str = ""

    def drift_table(self) -> dict[str, dict[str, float]]:
        table: dict[str, dict[str, float]] = {}
        for row in self.shock_rows:
            table.setdefault(row.feature, {})[f"{int(row.shock * 100):+d}%"] = row.drift
        return table

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "baseline": self.baseline,
            "most_sensitive_feature": self.most_sensitive_feature,
            "max_abs_drift": self.max_abs_drift,
            "drift_table": self.drift_table(),
            "rows": [
                {"feature": r.feature, "shock": r.shock, "metric": r.metric_value,
                 "drift": r.drift, "baseline": self.baseline,
                 "risk_impact": _risk_impact(r.drift)}
                for r in self.shock_rows
            ],
        }


def _risk_impact(drift: float) -> str:
    """Categorize a metric drift into a governance risk-impact label (#4)."""
    a = abs(drift)
    if a >= 0.10:
        return "high"
    if a >= 0.03:
        return "moderate"
    if a >= 0.005:
        return "low"
    return "negligible"


def _scorer_for(metric_name: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """Return f(y_true, proba_pos, preds) -> metric value."""
    from sklearn.metrics import average_precision_score, f1_score, recall_score, roc_auc_score

    def auc(y, p, _):
        return float(roc_auc_score(y, p))

    def prauc(y, p, _):
        return float(average_precision_score(y, p))

    def rec(y, _, pred):
        return float(recall_score(y, pred, zero_division=0))

    def f1(y, _, pred):
        return float(f1_score(y, pred, zero_division=0))

    return {"auc_roc": auc, "pr_auc": prauc, "recall": rec, "f1": f1}.get(metric_name, auc)


def run_sensitivity_analysis(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    top_features: list[str],
    metric_name: str = "auc_roc",
    shocks: tuple[float, ...] = DEFAULT_SHOCKS,
    max_features: int = 5,
) -> SensitivityResult:
    """Shock each of the top features across the grid and measure metric drift."""
    features = [f for f in top_features if f in X.columns][:max_features]
    scorer = _scorer_for(metric_name)
    y = np.asarray(y).reshape(-1)

    def score(frame: pd.DataFrame) -> float:
        proba = model.predict_proba(frame)
        p_pos = proba[:, 1] if proba.ndim == 2 and proba.shape[1] >= 2 else np.asarray(proba).reshape(-1)
        preds = model.predict(frame)
        return scorer(y, p_pos, preds)

    baseline = score(X)
    rows: list[ShockRow] = []
    per_feature_max: dict[str, float] = {}

    for feature in features:
        for shock in shocks:
            if shock == 0.0:
                value, drift = baseline, 0.0  # 0% row == baseline by construction
            else:
                shocked = X.copy()
                shocked[feature] = shocked[feature] * (1.0 + shock)
                value = score(shocked)
                drift = round(value - baseline, 6)
            rows.append(ShockRow(feature=feature, shock=shock, metric_value=round(value, 6), drift=drift))
            per_feature_max[feature] = max(per_feature_max.get(feature, 0.0), abs(drift))

    most_sensitive = max(per_feature_max, key=per_feature_max.get) if per_feature_max else None
    max_drift = round(max(per_feature_max.values()), 6) if per_feature_max else 0.0
    interp = (
        f"Most sensitive feature: {most_sensitive} (max |drift| {max_drift:.4f} in {metric_name}). "
        "Large drift indicates the model relies heavily on that feature; review for stability."
        if most_sensitive
        else "No features available for sensitivity analysis."
    )
    return SensitivityResult(
        metric_name=metric_name, baseline=round(baseline, 6), shock_rows=rows,
        most_sensitive_feature=most_sensitive, max_abs_drift=max_drift, interpretation=interp,
    )


def render_sensitivity_markdown(result: SensitivityResult, top_n: int = 20) -> str:
    """Shared sensitivity table for terminal / dashboard / transcript (#4).

    Columns: feature | shock % | baseline | shocked | delta | risk impact.
    The 0% row equals the baseline by construction.
    """
    rows = result.shock_rows[:top_n] if top_n else result.shock_rows
    lines = [
        "### Sensitivity analysis",
        "",
        f"- Metric: {result.metric_name}",
        f"- Baseline (0% shock): {result.baseline:.4f}",
        f"- Most sensitive feature: {result.most_sensitive_feature or 'n/a'}",
        f"- Max |drift|: {result.max_abs_drift:.4f}",
        "",
        "| Feature | Shock % | Baseline | Shocked | Delta | Risk impact |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r.feature} | {int(r.shock * 100):+d}% | {result.baseline:.4f} "
            f"| {r.metric_value:.4f} | {r.drift:+.4f} | {_risk_impact(r.drift)} |"
        )
    if result.interpretation:
        lines += ["", result.interpretation]
    return "\n".join(lines) + "\n"
