"""Trivial-baseline benchmarking.

Why a decision stump
--------------------

In a live review, a twelve-agent committee examined an MLP, produced thirty artefacts,
and returned ``READY WITH CONDITIONS`` at OOS AUC 0.6717. A single-split decision stump
— one threshold on one feature — reached 0.66 on the same data.

Nobody computed the stump, so nobody knew. The review was thorough about everything
except whether the model was worth having.

SR 11-7 asks for comparison "against alternative theories and approaches". In practice
that is usually read as champion-versus-challenger between two serious models, which is
expensive and often deferred. The cheap version is more revealing: if a model cannot
beat one rule on one feature by a meaningful margin, its complexity is unjustified, and
that is a finding regardless of how good the AUC looks in isolation.

This costs milliseconds and runs on every review.

Three baselines
---------------

**Majority class.** Predicts the negative class always. Establishes the accuracy floor,
and exists to make the point that accuracy is meaningless at low prevalence — a
model with 94.5% accuracy at 5.5% prevalence has learned nothing.

**Base rate.** Predicts the prevalence as a constant probability for everyone. AUC is
exactly 0.5 by construction; the useful output is its Brier score, which is the
calibration bar every real model should clear.

**Decision stump.** One threshold on the single most informative feature. The honest
question: what does the model add over the best single rule?

Verdicts are thresholded on the AUC lift over the stump, expressed in the language a
model risk function uses rather than as a bare number.

Requires numpy and scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "BaselineResult",
    "BenchmarkReport",
    "benchmark_against_baselines",
    "LIFT_THRESHOLDS",
]

#: AUC lift over the stump, and what each band means for sign-off.
LIFT_THRESHOLDS: tuple[tuple[float, str, str], ...] = (
    (0.15, "ok", "the model adds substantial discrimination over a single rule"),
    (0.07, "ok", "the model adds meaningful discrimination over a single rule"),
    (
        0.03,
        "concern",
        "the model adds only marginal discrimination over a single rule; "
        "its additional complexity needs justification",
    ),
    (
        0.00,
        "blocker",
        "the model does not meaningfully outperform a one-feature decision "
        "stump; its complexity is not justified by its performance",
    ),
    (-1.0, "blocker", "the model performs WORSE than a one-feature decision stump"),
)


@dataclass(frozen=True)
class BaselineResult:
    name: str
    description: str
    auc: float | None = None
    brier: float | None = None
    accuracy: float | None = None
    recall: float | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "auc": self.auc,
            "brier": self.brier,
            "accuracy": self.accuracy,
            "recall": self.recall,
            "detail": self.detail,
        }


@dataclass
class BenchmarkReport:
    model_auc: float
    baselines: list[BaselineResult] = field(default_factory=list)
    lift_over_stump: float = 0.0
    status: str = "unknown"
    verdict: str = ""

    def stump(self) -> BaselineResult | None:
        for baseline in self.baselines:
            if baseline.name == "decision_stump":
                return baseline
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_auc": self.model_auc,
            "lift_over_stump": self.lift_over_stump,
            "status": self.status,
            "verdict": self.verdict,
            "baselines": [b.as_dict() for b in self.baselines],
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"  {'Baseline':<22}{'AUC':>9}{'Brier':>9}{'Recall':>9}   Description",
        ]
        for baseline in self.baselines:
            auc = f"{baseline.auc:.4f}" if baseline.auc is not None else "—"
            brier = f"{baseline.brier:.4f}" if baseline.brier is not None else "—"
            recall = f"{baseline.recall:.4f}" if baseline.recall is not None else "—"
            lines.append(f"  {baseline.name:<22}{auc:>9}{brier:>9}{recall:>9}   {baseline.description}")
        lines.append(f"  {'model under review':<22}{self.model_auc:>9.4f}")
        lines.append("")
        lines.append(f"  Lift over stump: {self.lift_over_stump:+.4f} — {self.verdict}")
        return lines


def benchmark_against_baselines(
    X_train: Any,
    y_train: Any,
    X_eval: Any,
    y_eval: Any,
    model_scores: Any,
    *,
    seed: int = 42,
) -> BenchmarkReport:
    """Compare the model's evaluation-cohort AUC against three trivial baselines.

    Baselines are fitted on training data and evaluated on the same cohort as the
    model, so the comparison is like-for-like. A stump fitted on the evaluation set
    would flatter itself and understate the model.
    """
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import accuracy_score, brier_score_loss, recall_score, roc_auc_score
    from sklearn.tree import DecisionTreeClassifier

    y_eval_arr = np.asarray(y_eval).reshape(-1).astype(int)
    scores = np.asarray(model_scores, dtype=float).reshape(-1)

    baselines: list[BaselineResult] = []

    if len(np.unique(y_eval_arr)) < 2:
        return BenchmarkReport(
            model_auc=float("nan"),
            baselines=[],
            status="unknown",
            verdict="evaluation cohort contains a single class; benchmarking undefined",
        )

    model_auc = float(roc_auc_score(y_eval_arr, scores))

    # 1. Majority class -----------------------------------------------------
    try:
        majority = DummyClassifier(strategy="most_frequent")
        majority.fit(X_train, y_train)
        predictions = majority.predict(X_eval)
        baselines.append(
            BaselineResult(
                name="majority_class",
                description="predicts the majority class for every case",
                auc=0.5,
                accuracy=float(accuracy_score(y_eval_arr, predictions)),
                recall=float(recall_score(y_eval_arr, predictions, zero_division=0)),
                detail="accuracy here is the floor: any model must beat it to mean anything",
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        baselines.append(BaselineResult("majority_class", "unavailable", detail=str(exc)))

    # 2. Base rate ----------------------------------------------------------
    try:
        prevalence = float(np.asarray(y_train).reshape(-1).astype(int).mean())
        constant = np.full(y_eval_arr.shape, prevalence, dtype=float)
        baselines.append(
            BaselineResult(
                name="base_rate",
                description=f"predicts the training prevalence ({prevalence:.3f}) for everyone",
                auc=0.5,
                brier=float(brier_score_loss(y_eval_arr, constant)),
                detail="its Brier score is the calibration bar the model should clear",
            )
        )
    except Exception as exc:  # pragma: no cover
        baselines.append(BaselineResult("base_rate", "unavailable", detail=str(exc)))

    # 3. Decision stump -----------------------------------------------------
    stump_auc: float | None = None
    try:
        stump = DecisionTreeClassifier(max_depth=1, random_state=seed, class_weight="balanced")
        stump.fit(X_train, y_train)
        stump_scores = stump.predict_proba(X_eval)[:, 1]
        stump_predictions = stump.predict(X_eval)
        stump_auc = float(roc_auc_score(y_eval_arr, stump_scores))

        feature_name = ""
        try:
            index = int(stump.tree_.feature[0])
            names = list(getattr(X_train, "columns", []))
            if 0 <= index < len(names):
                feature_name = f"{names[index]} <= {float(stump.tree_.threshold[0]):.4f}"
        except Exception:
            feature_name = ""

        baselines.append(
            BaselineResult(
                name="decision_stump",
                description="one threshold on one feature",
                auc=stump_auc,
                brier=float(brier_score_loss(y_eval_arr, stump_scores)),
                recall=float(recall_score(y_eval_arr, stump_predictions, zero_division=0)),
                detail=f"split: {feature_name}" if feature_name else "",
            )
        )
    except Exception as exc:  # pragma: no cover
        baselines.append(BaselineResult("decision_stump", "unavailable", detail=str(exc)))

    if stump_auc is None:
        return BenchmarkReport(
            model_auc=model_auc,
            baselines=baselines,
            status="unknown",
            verdict="stump baseline could not be fitted; benchmarking incomplete",
        )

    lift = model_auc - stump_auc
    status, verdict = "blocker", LIFT_THRESHOLDS[-1][2]
    for threshold, band_status, band_verdict in LIFT_THRESHOLDS:
        if lift >= threshold:
            status, verdict = band_status, band_verdict
            break

    return BenchmarkReport(
        model_auc=model_auc,
        baselines=baselines,
        lift_over_stump=round(lift, 4),
        status=status,
        verdict=verdict,
    )
