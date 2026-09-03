"""Architecture review, hyperparameter tuning, and metric-priority agents.

Three review execution agents that make model-engineering decisions visible and
reviewable:

  * ``select_primary_metric`` — routes task type + cost preference to a primary
    metric (e.g. false-negatives-costly -> recall / PR-AUC).
  * ``ArchitectureReviewAgent`` — compares the user's architecture/activation
    choice against a recommendation given dataset shape/modality/imbalance,
    with reason, evidence ID, and risk-if-ignored.
  * ``HyperparameterTuningAgent`` — proposes a bounded, leakage-safe tuning
    plan (strategy, search space, trials, early stopping, metric) and records
    the selected/rejected parameters.

All deterministic; each produces a structured, evidence-bearing recommendation
the user can accept or override.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------- #
# Metric priority routing
# ---------------------------------------------------------------------------- #
def select_primary_metric(task_type: str, *, costlier_errors: str = "balanced") -> dict[str, Any]:
    """Route task type + which errors are costlier to a primary metric.

    costlier_errors: 'false_negatives' | 'false_positives' | 'balanced' | 'recall' | 'precision' | 'f1'.
    """
    costlier = (costlier_errors or "balanced").lower()
    if task_type == "binary_classification":
        if costlier in ("false_negatives", "fn", "recall", "pr_auc"):
            metric = "recall" if costlier == "recall" else "pr_auc"
            reason = "False negatives are costlier; prioritize recall/PR-AUC."
            secondary = ["recall", "auc_roc", "f1"]
        elif costlier in ("false_positives", "fp", "precision"):
            metric = "precision"
            reason = "False positives are costlier; prioritize precision."
            secondary = ["auc_roc", "f1", "specificity"]
        elif costlier in ("f1", "f1_score"):
            metric = "f1"
            reason = "F1 score selected as primary balance metric."
            secondary = ["auc_roc", "pr_auc"]
        elif costlier in ("auc_roc", "roc_auc", "auc"):
            metric = "auc_roc"
            reason = "AUC-ROC selected as primary ranking metric."
            secondary = ["pr_auc", "f1"]
        else:
            metric, reason = "auc_roc", "Balanced error costs; AUC-ROC is the default ranking metric."
            secondary = ["pr_auc", "f1", "brier_score"]
    elif task_type == "multiclass_classification":
        metric, reason = "f1_macro", "Multiclass; macro-F1 weights classes equally."
        secondary = ["accuracy", "f1_weighted"]
    elif task_type == "multilabel_classification":
        metric, reason = "f1_micro", "Multilabel; micro-F1 aggregates across labels."
        secondary = ["subset_accuracy", "mean_auc"]
    elif task_type in ("regression", "forecasting"):
        metric, reason = "rmse", "Regression; RMSE penalizes large errors."
        secondary = ["mae", "r2"]
    else:
        metric, reason, secondary = "accuracy", "Default metric.", ["f1"]
    return {
        "primary_metric": metric,
        "reason": reason,
        "secondary_metrics": secondary,
        "costlier_errors": costlier,
    }


# ---------------------------------------------------------------------------- #
# Architecture review agent
# ---------------------------------------------------------------------------- #
@dataclass
class ArchitectureReview:
    user_choice: dict[str, str]
    recommendation: dict[str, str]
    reason: str
    evidence_id: str
    risk_if_ignored: str
    agrees: bool
    user_override: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_choice": self.user_choice,
            "recommendation": self.recommendation,
            "reason": self.reason,
            "evidence_id": self.evidence_id,
            "risk_if_ignored": self.risk_if_ignored,
            "agrees": self.agrees,
            "user_override": self.user_override,
        }


class ArchitectureReviewAgent:
    """Reviews the user's architecture/activation against a recommendation."""

    def review(
        self,
        *,
        user_family: str,
        user_activation: str,
        modality: str,
        n_samples: int,
        n_features: int,
        task_type: str,
        imbalanced: bool = False,
        evidence_id: str = "ARCH-01",
    ) -> ArchitectureReview:
        rec_family, rec_activation = user_family, user_activation
        reasons: list[str] = []

        if modality == "tabular":
            small = n_samples < 5000
            complex_choice = user_family in ("wide_deep", "residual_mlp")
            if small and complex_choice:
                rec_family = "mlp"
                reasons.append(
                    f"Small tabular dataset ({n_samples} rows); a simpler MLP has lower "
                    "overfitting risk and better interpretability than "
                    f"{user_family}."
                )
            if user_activation in ("selu", "elu") and small:
                rec_activation = "relu"
                reasons.append("ReLU is a robust default for small tabular data.")
        elif modality == "sequence":
            if user_family not in ("rnn", "gru", "lstm", "bi_lstm"):
                rec_family = "lstm"
                reasons.append("Sequence modality; an LSTM is a strong default for temporal data.")
        elif modality == "vision":
            if not user_family.startswith(("simple_cnn", "resnet")):
                rec_family = "simple_cnn_small"
                reasons.append("Vision modality; a compact CNN is the appropriate baseline.")

        agrees = (rec_family == user_family) and (rec_activation == user_activation)
        if agrees:
            reasons.append("User choice is appropriate for the data; no change recommended.")
        risk = (
            "Higher overfitting risk and reduced interpretability."
            if not agrees
            else "None — choice validated."
        )
        return ArchitectureReview(
            user_choice={"family": user_family, "activation": user_activation},
            recommendation={"family": rec_family, "activation": rec_activation},
            reason=" ".join(reasons),
            evidence_id=evidence_id,
            risk_if_ignored=risk,
            agrees=agrees,
        )


# ---------------------------------------------------------------------------- #
# Hyperparameter tuning agent
# ---------------------------------------------------------------------------- #
@dataclass
class TuningPlan:
    strategy: str
    primary_metric: str
    search_space: dict[str, list[Any]]
    n_trials: int
    early_stopping: bool
    validation: str
    best_params: dict[str, Any] = field(default_factory=dict)
    rejected_params: list[dict[str, Any]] = field(default_factory=list)
    evidence_id: str = "TUNE-01"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "primary_metric": self.primary_metric,
            "search_space": self.search_space,
            "n_trials": self.n_trials,
            "early_stopping": self.early_stopping,
            "validation": self.validation,
            "best_params": self.best_params,
            "rejected_params": self.rejected_params,
            "evidence_id": self.evidence_id,
        }


class HyperparameterTuningAgent:
    """Proposes a bounded, leakage-safe tuning plan and records the outcome."""

    def plan(
        self,
        *,
        task_type: str,
        family: str = "mlp",
        n_samples: int = 1000,
        costlier_errors: str = "balanced",
        n_trials: int | None = None,
        evidence_id: str = "TUNE-01",
    ) -> TuningPlan:
        metric = select_primary_metric(task_type, costlier_errors=costlier_errors)["primary_metric"]
        trials = n_trials or (5 if n_samples < 2000 else 10 if n_samples < 20000 else 15)
        if family in ("rnn", "lstm", "gru", "bi_lstm"):
            search_space = {
                "learning_rate": [1e-3, 3e-3, 1e-2],
                "hidden_size": [16, 32, 64],
                "num_layers": [1, 2],
                "dropout": [0.0, 0.1, 0.2],
            }
        else:
            search_space = {
                "learning_rate": [1e-3, 3e-3, 1e-2],
                "hidden_dims": [[32], [64, 32], [128, 64]],
                "dropout": [0.0, 0.1, 0.2],
            }
        return TuningPlan(
            strategy="bounded_randomized_search",
            primary_metric=metric,
            search_space=search_space,
            n_trials=trials,
            early_stopping=True,
            validation="train_internal_holdout",  # never test/OOS
            evidence_id=evidence_id,
        )

    def record_outcome(
        self, plan: TuningPlan, best_params: dict[str, Any], rejected: list[dict[str, Any]]
    ) -> TuningPlan:
        plan.best_params = best_params
        plan.rejected_params = rejected
        return plan


def render_architecture_review_markdown(review: ArchitectureReview) -> str:
    return (
        "### Architecture review\n\n"
        f"- User selected: `{review.user_choice['family']} + {review.user_choice['activation']}`\n"
        f"- Agent recommends: `{review.recommendation['family']} + "
        f"{review.recommendation['activation']}`\n"
        f"- Reason: {review.reason}\n"
        f"- Evidence: {review.evidence_id}\n"
        f"- Risk if ignored: {review.risk_if_ignored}\n"
        f"- Agreement: {'yes' if review.agrees else 'no — user decision required'}\n"
    )


def render_tuning_plan_markdown(plan: TuningPlan) -> str:
    lines = [
        "### Hyperparameter tuning plan",
        "",
        f"- Strategy: {plan.strategy}",
        f"- Primary metric: {plan.primary_metric}",
        f"- Trials: {plan.n_trials}",
        f"- Early stopping: {plan.early_stopping}",
        f"- Validation: {plan.validation} (no test/OOS leakage)",
        f"- Evidence: {plan.evidence_id}",
    ]
    if plan.best_params:
        lines.append(f"- Best params: {plan.best_params}")
    return "\n".join(lines) + "\n"
