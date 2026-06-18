"""Evidence store for evidence-constrained agent dialogue (v2.3.0 #1).

The store holds the *actual* diagnostics produced during a review — outlier
counts, correlation pairs, missingness, feature importance, sensitivity drift,
cohort metrics, tuning results, and the review-session decisions. Ask-Agent
retrieves from this store and answers ONLY from retrieved evidence; if the
relevant evidence is absent, the agent must explicitly refuse rather than
fabricate feature names, percentages, metrics, counts, thresholds, or drift
values.

This is the anti-hallucination foundation: no diagnostic value reaches the
reviewer unless it came from a real artifact recorded here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceItem:
    """A single retrieved fact with provenance, for grounding answers."""

    kind: str          # outliers | correlation | missingness | importance | ...
    label: str         # human-readable description
    value: Any         # the actual value(s)
    source: str        # which artifact this came from

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "label": self.label,
                "value": self.value, "source": self.source}


@dataclass
class EvidenceStore:
    """All diagnostics available to answer reviewer questions, by category.

    Every field defaults to empty. A category that was never computed stays
    empty, and retrieval for it returns nothing — which the dialogue layer
    turns into an explicit "insufficient evidence" answer, never a guess.
    """

    # data diagnostics (from DataStatistics)
    outliers: dict[str, int] = field(default_factory=dict)          # column -> count
    missingness: dict[str, float] = field(default_factory=dict)     # column -> pct
    correlations: list[dict[str, Any]] = field(default_factory=list)  # pair rows
    leakage_candidates: list[str] = field(default_factory=list)
    high_cardinality: list[str] = field(default_factory=list)
    low_variance: list[str] = field(default_factory=list)
    n_rows: int | None = None
    n_features: int | None = None
    n_numeric: int | None = None
    n_categorical: int | None = None
    class_distribution: dict[str, float] = field(default_factory=dict)
    target: str | None = None
    candidate_targets: list[str] = field(default_factory=list)

    # model diagnostics
    feature_importance: list[dict[str, Any]] = field(default_factory=list)
    cohort_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    sensitivity_rows: list[dict[str, Any]] = field(default_factory=list)
    most_sensitive_feature: str | None = None
    max_abs_drift: float | None = None
    tuning_trials: list[dict[str, Any]] = field(default_factory=list)
    best_params: dict[str, Any] = field(default_factory=dict)

    # ----- construction from existing artifacts ----------------------------- #
    @classmethod
    def from_artifacts(
        cls,
        data_stats: Any = None,
        copilot_exec: Any = None,
        sensitivity: Any = None,
        tuning_run: Any = None,
        candidate_targets: list[str] | None = None,
    ) -> EvidenceStore:
        store = cls()
        if data_stats is not None:
            store.outliers = dict(getattr(data_stats, "outlier_summary", {}) or {})
            store.missingness = dict(getattr(data_stats, "missing_by_column", {}) or {})
            store.leakage_candidates = list(getattr(data_stats, "leakage_candidates", []) or [])
            store.high_cardinality = list(getattr(data_stats, "high_cardinality_columns", []) or [])
            store.low_variance = list(getattr(data_stats, "low_variance_columns", []) or [])
            store.n_rows = getattr(data_stats, "n_rows", None)
            store.n_features = getattr(data_stats, "n_columns", None)
            store.n_numeric = getattr(data_stats, "n_numeric", None)
            store.n_categorical = getattr(data_stats, "n_categorical", None)
            store.class_distribution = dict(getattr(data_stats, "class_distribution", {}) or {})
            store.target = getattr(data_stats, "target_column", None)
            corr = getattr(data_stats, "correlation_summary", {}) or {}
            # correlation_summary may hold a list of high-correlation pairs
            pairs = corr.get("high_correlation_pairs") or corr.get("pairs") or []
            store.correlations = list(pairs)
        if copilot_exec is not None:
            store.feature_importance = list(getattr(copilot_exec, "global_importance", []) or [])
            store.cohort_metrics = dict(getattr(copilot_exec, "metrics_by_split", {}) or {})
        if sensitivity is not None:
            sd = sensitivity.to_dict() if hasattr(sensitivity, "to_dict") else sensitivity
            store.sensitivity_rows = list(sd.get("rows", []) or [])
            store.most_sensitive_feature = sd.get("most_sensitive_feature")
            store.max_abs_drift = sd.get("max_abs_drift")
        if tuning_run is not None:
            td = tuning_run.to_dict() if hasattr(tuning_run, "to_dict") else tuning_run
            store.tuning_trials = list(td.get("trials", []) or [])
            store.best_params = dict(td.get("best_params", {}) or {})
        if candidate_targets:
            store.candidate_targets = list(candidate_targets)
        return store

    # ----- retrieval -------------------------------------------------------- #
    def top_outliers(self, n: int = 10) -> list[EvidenceItem]:
        if not self.outliers:
            return []
        ranked = sorted(self.outliers.items(), key=lambda kv: kv[1], reverse=True)
        ranked = [(c, n_out) for c, n_out in ranked if n_out > 0][:n]
        return [
            EvidenceItem("outliers", f"{col}: {cnt} outlier rows", cnt, "data_statistics.outlier_summary")
            for col, cnt in ranked
        ]

    def top_missing(self, n: int = 10) -> list[EvidenceItem]:
        if not self.missingness:
            return []
        ranked = sorted(self.missingness.items(), key=lambda kv: kv[1], reverse=True)
        ranked = [(c, p) for c, p in ranked if p > 0][:n]
        return [
            EvidenceItem("missingness", f"{col}: {pct:.1f}% missing", pct,
                         "data_statistics.missing_by_column")
            for col, pct in ranked
        ]

    def top_correlations(self, n: int = 10) -> list[EvidenceItem]:
        if not self.correlations:
            return []
        items = []
        for pair in self.correlations[:n]:
            a = pair.get("a") or pair.get("feature_a") or pair.get("left")
            b = pair.get("b") or pair.get("feature_b") or pair.get("right")
            r = pair.get("r") or pair.get("corr") or pair.get("correlation")
            if a and b and r is not None:
                items.append(EvidenceItem(
                    "correlation", f"{a} ~ {b}: r={float(r):.3f}", {"a": a, "b": b, "r": r},
                    "data_statistics.correlation_summary"))
        return items

    def top_importance(self, n: int = 20) -> list[EvidenceItem]:
        if not self.feature_importance:
            return []
        items = []
        for row in self.feature_importance[:n]:
            feat = row.get("feature")
            imp = row.get("importance")
            if feat is not None and imp is not None:
                items.append(EvidenceItem(
                    "importance", f"{feat}: importance={imp}", row,
                    "copilot_execution.global_importance"))
        return items

    def sensitivity_evidence(self, n: int = 20) -> list[EvidenceItem]:
        if not self.sensitivity_rows:
            return []
        items = []
        for row in self.sensitivity_rows[:n]:
            items.append(EvidenceItem(
                "sensitivity",
                f"{row.get('feature')} @ {int(row.get('shock', 0) * 100):+d}%: "
                f"delta={row.get('drift')}, risk={row.get('risk_impact')}",
                row, "sensitivity_analysis.rows"))
        return items

    def metrics_evidence(self) -> list[EvidenceItem]:
        if not self.cohort_metrics:
            return []
        items = []
        for split, m in self.cohort_metrics.items():
            items.append(EvidenceItem(
                "metrics", f"{split}: " + ", ".join(
                    f"{k}={v:.4f}" for k, v in m.items()
                    if isinstance(v, (int, float))),
                m, "copilot_execution.metrics_by_split"))
        return items

    def has_any(self) -> bool:
        return any([
            self.outliers, self.missingness, self.correlations, self.feature_importance,
            self.cohort_metrics, self.sensitivity_rows, self.tuning_trials,
        ])
