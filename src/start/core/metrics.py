from typing import Any

import numpy as np


def calculate_safe_metrics_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None
) -> dict[str, Any]:
    """Surgically isolates mathematical processing from Scikit-Learn division-by-zero warnings."""
    epsilon = 1e-15
    metrics_summary = {}

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # Enforce strict denominator value clipping guardrails
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        tn = np.sum((y_true == 0) & (y_pred == 0))

        precision = tp / max(tp + fp, epsilon)
        recall = tp / max(tp + fn, epsilon)
        f1_score = 2 * (precision * recall) / max(precision + recall, epsilon)
        accuracy = (tp + tn) / max(tp + tn + fp + fn, epsilon)

        metrics_summary["accuracy"] = float(accuracy)
        metrics_summary["precision"] = float(precision)
        metrics_summary["recall"] = float(recall)
        metrics_summary["f1"] = float(f1_score)

        return metrics_summary


def calculate_population_stability_index(
    baseline: np.ndarray, target: np.ndarray, num_bins: int = 10
) -> float:
    """Calculates PSI safely by clipping probabilities to eliminate division-by-zero math noise."""
    epsilon = 1e-15
    with np.errstate(divide="ignore", invalid="ignore"):
        baseline_percents, bin_edges = np.histogram(baseline, bins=num_bins, density=False)
        target_percents, _ = np.histogram(target, bins=bin_edges, density=False)

        b_pct = baseline_percents / max(sum(baseline_percents), epsilon)
        t_pct = target_percents / max(sum(target_percents), epsilon)

        # Clip array values elements securely to support extreme stress distributions
        b_pct = np.clip(b_pct, epsilon, 1.0)
        t_pct = np.clip(t_pct, epsilon, 1.0)

        psi_value = np.sum((t_pct - b_pct) * np.log(t_pct / b_pct))
        return float(psi_value)
