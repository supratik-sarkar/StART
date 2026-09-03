"""Feature-engineering recommendation engine for the review execution layer.

Turns the frozen FeatureEngineeringAgent diagnostics + the data statistics into
a list of explicit, reviewable recommendations. Each carries: the action, the
reason, an evidence ID, the risk if ignored, the default action, and a slot for
the user's override choice. The user can accept the agent's recommendation or
keep their own — this module surfaces the choice, it does not silently mutate
data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from start.modeling.data_statistics import DataStatistics


@dataclass
class FERecommendation:
    step: str  # imputation | encoding | scaling | outliers | imbalance | low_variance | ...
    recommendation: str
    reason: str
    evidence_id: str
    risk_if_ignored: str
    default_action: str
    applies: bool = True
    user_override: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "recommendation": self.recommendation,
            "reason": self.reason,
            "evidence_id": self.evidence_id,
            "risk_if_ignored": self.risk_if_ignored,
            "default_action": self.default_action,
            "user_override": self.user_override,
        }

    @property
    def effective_action(self) -> str:
        return self.user_override or self.default_action


@dataclass
class FERecommendationSet:
    recommendations: list[FERecommendation] = field(default_factory=list)

    def applicable(self) -> list[FERecommendation]:
        return [r for r in self.recommendations if r.applies]

    def to_list(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.applicable()]

    def apply_overrides(self, overrides: dict[str, str]) -> None:
        """overrides maps step -> user choice ('accept' | 'skip' | custom text)."""
        for rec in self.recommendations:
            if rec.step in overrides:
                rec.user_override = overrides[rec.step]


def recommend_feature_engineering(
    stats: DataStatistics,
    modality: str = "tabular",
    evidence_prefix: str = "FE",
    cost_specification: dict[str, Any] | None = None,
) -> FERecommendationSet:
    recs: list[FERecommendation] = []
    i = 0

    def ev() -> str:
        nonlocal i
        i += 1
        return f"{evidence_prefix}-{i:02d}"

    # missing-value imputation
    cols_missing = {k: v for k, v in stats.missing_by_column.items() if v > 0}
    if cols_missing:
        worst = max(cols_missing.values())
        recs.append(
            FERecommendation(
                step="imputation",
                recommendation="Impute missing values (median for numeric, mode for categorical).",
                reason=f"{len(cols_missing)} column(s) contain missing values (max {worst:.1f}%).",
                evidence_id=ev(),
                risk_if_ignored="Rows dropped or model errors; biased estimates if missingness is informative.",
                default_action="impute_median_mode",
            )
        )

    # categorical encoding
    if stats.n_categorical > 0:
        recs.append(
            FERecommendation(
                step="encoding",
                recommendation=(
                    "Encode categorical features (one-hot for low-cardinality, target/ordinal otherwise)."
                ),
                reason=f"{stats.n_categorical} categorical feature(s) present.",
                evidence_id=ev(),
                risk_if_ignored="Most estimators cannot consume raw strings.",
                default_action="onehot_low_cardinality",
            )
        )

    # scaling
    if stats.n_numeric > 1:
        recs.append(
            FERecommendation(
                step="scaling",
                recommendation="Standardize numeric features (fit on train only).",
                reason=(
                    "Multiple numeric features; gradient-based and distance-based "
                    "models benefit from scaling."
                ),
                evidence_id=ev(),
                risk_if_ignored="Slow/unstable convergence; features on large scales dominate.",
                default_action="standardize_train_only",
            )
        )

    # outlier handling
    if stats.outlier_summary:
        worst_col = max(stats.outlier_summary, key=stats.outlier_summary.get)
        recs.append(
            FERecommendation(
                step="outliers",
                recommendation="Winsorize/clip extreme numeric outliers (e.g. 1st/99th percentile).",
                reason=f"Outliers detected in {len(stats.outlier_summary)} column(s) "
                f"(worst: {worst_col}, {stats.outlier_summary[worst_col]} points).",
                evidence_id=ev(),
                risk_if_ignored="Outliers distort scaling and can dominate the loss.",
                default_action="winsorize_1_99",
            )
        )

    # class imbalance
    cost_spec = cost_specification or {"type": "balanced"}
    has_imbalance_warning = "severe" in stats.imbalance_warning or "moderate" in stats.imbalance_warning

    # Check if target is approximately balanced (minority class fraction >= 30%)
    is_approx_balanced = False
    if stats.class_distribution:
        vals = list(stats.class_distribution.values())
        if vals:
            min_frac = min(vals)
            if any(v > 1.0 for v in vals):
                min_frac = min_frac / 100.0
            if min_frac >= 0.30:
                is_approx_balanced = True

    recommend_weighting = False
    if is_approx_balanced:
        if cost_spec.get("type", "balanced") != "balanced":
            recommend_weighting = True
    else:
        if has_imbalance_warning:
            recommend_weighting = True

    if recommend_weighting:
        recs.append(
            FERecommendation(
                step="imbalance",
                recommendation="Handle class imbalance (class weights, or resampling on train only).",
                reason=f"Target imbalance: {stats.imbalance_warning} (cost spec: {cost_spec.get('type')})."
                if is_approx_balanced
                else f"Target imbalance: {stats.imbalance_warning}.",
                evidence_id=ev(),
                risk_if_ignored="Model optimizes majority class; minority recall collapses.",
                default_action="class_weights",
            )
        )

    # low-variance removal
    if stats.low_variance_columns:
        recs.append(
            FERecommendation(
                step="low_variance",
                recommendation=(
                    "Drop low-variance/constant columns: " + ", ".join(stats.low_variance_columns[:5]) + "."
                ),
                reason=f"{len(stats.low_variance_columns)} column(s) carry no usable signal.",
                evidence_id=ev(),
                risk_if_ignored="Wasted capacity; possible numerical issues.",
                default_action="drop_low_variance",
            )
        )

    # high-correlation pruning
    n_hi_corr = stats.correlation_summary.get("n_features_corr_gt_0.7", 0)
    if n_hi_corr:
        recs.append(
            FERecommendation(
                step="correlation_pruning",
                recommendation="Review highly correlated feature pairs; consider pruning redundancy.",
                reason=f"{n_hi_corr} feature(s) correlate > 0.7 with the target or each other.",
                evidence_id=ev(),
                risk_if_ignored="Multicollinearity; unstable coefficients and attributions.",
                default_action="review_only",
            )
        )

    # leakage exclusion (always strongly recommended, not optional)
    if stats.leakage_candidates:
        recs.append(
            FERecommendation(
                step="leakage_exclusion",
                recommendation=f"EXCLUDE leakage candidates: {', '.join(stats.leakage_candidates[:5])}.",
                reason="Feature(s) almost perfectly correlated with the target — likely post-event leakage.",
                evidence_id=ev(),
                risk_if_ignored="Inflated metrics that collapse in production; invalid model.",
                default_action="exclude_leakage",
            )
        )

    # high-cardinality categoricals
    if stats.high_cardinality_columns:
        recs.append(
            FERecommendation(
                step="high_cardinality",
                recommendation="Use hashing/target encoding for high-cardinality categoricals.",
                reason=(
                    f"{len(stats.high_cardinality_columns)} high-cardinality column(s) "
                    "would explode one-hot width."
                ),
                evidence_id=ev(),
                risk_if_ignored="One-hot explosion; sparse, overfit-prone features.",
                default_action="target_or_hash_encoding",
            )
        )

    # modality-specific routing
    if modality == "sequential":
        recs.append(
            FERecommendation(
                step="datetime_expansion",
                recommendation="Expand datetime into lag/rolling/trend features for sequence modeling.",
                reason="Sequential modality detected.",
                evidence_id=ev(),
                risk_if_ignored="Temporal structure unused.",
                default_action="expand_datetime",
            )
        )
    elif modality == "vision":
        recs.append(
            FERecommendation(
                step="image_transforms",
                recommendation="Apply per-channel normalization and augmentation (train only).",
                reason="Vision modality detected.",
                evidence_id=ev(),
                risk_if_ignored="Unnormalized inputs; reduced generalization.",
                default_action="normalize_augment",
            )
        )
    elif modality == "text":
        recs.append(
            FERecommendation(
                step="text_vectorization",
                recommendation="Route text columns through tokenization/embedding.",
                reason="Text modality detected.",
                evidence_id=ev(),
                risk_if_ignored="Raw text unusable by tabular models.",
                default_action="tokenize_embed",
            )
        )

    return FERecommendationSet(recommendations=recs)


def render_fe_recommendations_markdown(rec_set: FERecommendationSet) -> str:
    apps = rec_set.applicable()
    if not apps:
        return "### Feature-engineering recommendations\n\n_No actions recommended._\n"
    lines = [
        "### Feature-engineering recommendations",
        "",
        "| Step | Recommendation | Reason | Evidence | Risk if ignored | Default | Override |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in apps:
        lines.append(
            f"| {r.step} | {r.recommendation} | {r.reason} | {r.evidence_id} "
            f"| {r.risk_if_ignored} | {r.default_action} | {r.user_override or '—'} |"
        )
    return "\n".join(lines) + "\n"
