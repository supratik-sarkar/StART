"""Feature-engineering diagnostics as evidence.

This agent does not silently mutate data; it *diagnoses* it and emits evidence
the review can cite. Each modality has its own checks:

    tabular     - scaling need, encoding need, leakage detection, missingness
                  indicators, train/test drift
    sequential  - window/lag feasibility, rolling-stat and trend hooks
    vision      - normalization stats, class balance, augmentation diagnostics
    text        - tokenization / length diagnostics, embedding readiness

Transforms that *are* applied (e.g. standardization inside the DL classifier)
are reported, never hidden. The point is governable preprocessing: a reviewer
can see exactly what was done and why.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from start.core.schemas import TestResult, ThresholdSpec


@dataclass
class FeatureDiagnostics:
    modality: str
    findings: dict[str, Any]
    leakage_suspects: list[str] = field(default_factory=list)
    high_missing: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


class FeatureEngineeringAgent:
    """Modality-aware feature diagnostics."""

    def diagnose(
        self,
        train: pd.DataFrame,
        target_column: str,
        *,
        modality: str = "tabular",
        test: pd.DataFrame | None = None,
        time_column: str | None = None,
    ) -> FeatureDiagnostics:
        if modality == "sequential":
            return self._sequential(train, target_column, time_column)
        if modality == "vision":
            return self._vision(train, target_column)
        if modality == "text":
            return self._text(train, target_column)
        return self._tabular(train, target_column, test)

    # -- tabular ----------------------------------------------------------- #
    def _tabular(self, train, target_column, test) -> FeatureDiagnostics:
        features = [c for c in train.columns if c != target_column]
        numeric = [c for c in features if pd.api.types.is_numeric_dtype(train[c])]
        categorical = [c for c in features if c not in numeric]

        # scaling need: features on very different scales
        scales = {c: float(train[c].std()) for c in numeric if train[c].std() == train[c].std()}
        _scale_vals = [v for v in scales.values() if v > 0]
        needs_scaling = bool(_scale_vals) and (max(_scale_vals) / (min(_scale_vals) or 1) > 100)

        # missingness
        missing = {c: float(train[c].isna().mean()) for c in features}
        high_missing = [c for c, m in missing.items() if m > 0.30]

        # leakage detection: a feature almost perfectly correlated with target
        leakage = []
        if pd.api.types.is_numeric_dtype(train[target_column]):
            y = train[target_column]
            for c in numeric:
                if c == target_column:
                    continue
                corr = train[c].corr(y)
                if corr == corr and abs(corr) > 0.98:
                    leakage.append(c)

        # train/test drift: mean shift in standardized space
        drift = {}
        if test is not None:
            for c in numeric:
                s = train[c].std()
                if s and s == s:
                    shift = abs(train[c].mean() - test[c].mean()) / s
                    if shift > 0.25:
                        drift[c] = round(float(shift), 4)

        findings = {
            "n_numeric": len(numeric),
            "n_categorical": len(categorical),
            "needs_scaling": needs_scaling,
            "needs_encoding": len(categorical) > 0,
            "max_missing_pct": round(max(missing.values()) * 100, 2) if missing else 0.0,
            "n_drift_features": len(drift),
            "drift_features": ", ".join(sorted(drift, key=lambda k: -drift[k])[:5]),
        }
        notes = []
        if needs_scaling:
            notes.append("Feature scales differ by >100x; standardization recommended.")
        if categorical:
            notes.append(f"{len(categorical)} categorical feature(s) need encoding.")
        return FeatureDiagnostics("tabular", findings, leakage, high_missing, notes)

    # -- sequential -------------------------------------------------------- #
    def _sequential(self, train, target_column, time_column) -> FeatureDiagnostics:
        n = len(train)
        findings = {
            "n_rows": n,
            "has_time_column": bool(time_column and time_column in train.columns),
            "window_feasible": n >= 50,
            "suggested_window": min(20, max(2, n // 20)),
            "lag_features_recommended": True,
            "rolling_stats_recommended": True,
        }
        notes = ["Sequential modality: sliding windows, lag, rolling stats, and trend features apply."]
        if not findings["has_time_column"]:
            notes.append("No time column supplied; ordering assumed by row index.")
        return FeatureDiagnostics("sequential", findings, [], [], notes)

    # -- vision ------------------------------------------------------------ #
    def _vision(self, train, target_column) -> FeatureDiagnostics:
        balance = (
            train[target_column].value_counts(normalize=True).to_dict()
            if target_column in train.columns
            else {}
        )
        min_share = min(balance.values()) if balance else 0.0
        findings = {
            "n_images": len(train),
            "n_classes": len(balance),
            "min_class_share": round(float(min_share), 4),
            "normalization_recommended": True,
            "augmentation_recommended": min_share < 0.4,
        }
        notes = ["Vision modality: per-channel normalization and augmentation diagnostics apply."]
        if min_share and min_share < 0.4:
            notes.append("Class imbalance detected; augmentation/resampling recommended.")
        return FeatureDiagnostics("vision", findings, [], [], notes)

    # -- text -------------------------------------------------------------- #
    def _text(self, train, target_column) -> FeatureDiagnostics:
        text_cols = [
            c
            for c in train.columns
            if c != target_column
            and (train[c].dtype == object or pd.api.types.is_string_dtype(train[c]))
        ]
        lengths = []
        for c in text_cols:
            lengths.extend(train[c].dropna().astype(str).str.split().str.len().tolist())
        avg_tokens = float(np.mean(lengths)) if lengths else 0.0
        findings = {
            "n_text_columns": len(text_cols),
            "avg_tokens": round(avg_tokens, 2),
            "tokenization_recommended": True,
            "embedding_recommended": avg_tokens > 3,
        }
        return FeatureDiagnostics(
            "text", findings, [], [], ["Text modality: tokenization and embedding diagnostics apply."]
        )

    # -- evidence ---------------------------------------------------------- #
    def to_evidence(self, diag: FeatureDiagnostics) -> TestResult:
        metrics = dict(diag.findings)
        metrics["n_leakage_suspects"] = len(diag.leakage_suspects)
        if diag.leakage_suspects:
            metrics["leakage_suspects"] = ", ".join(diag.leakage_suspects[:5])
        metrics["n_high_missing"] = len(diag.high_missing)

        thresholds = [
            ThresholdSpec(metric="n_leakage_suspects", warn=0, fail=0, direction="upper"),
        ]
        result = TestResult(
            test_id="feature_engineering.diagnostics",
            test_name=f"Feature engineering diagnostics ({diag.modality})",
            metrics=metrics,
            thresholds=thresholds,
            interpretation=(
                f"{diag.modality} diagnostics. " + " ".join(diag.notes)
                + (
                    f" Possible leakage: {', '.join(diag.leakage_suspects)}."
                    if diag.leakage_suspects
                    else ""
                )
            ),
            limitations=[
                "Diagnostics flag preprocessing needs; transforms are applied transparently downstream.",
            ],
        )
        return result.apply_thresholds()
