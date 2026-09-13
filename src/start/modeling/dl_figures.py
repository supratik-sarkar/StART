"""Deep-learning review figures.

Generates and saves PNGs under start_output/figures/deep_learning/<RUN_ID>/.
matplotlib is optional: if it is unavailable, each function degrades to
returning an empty path with a note, and the figure-generation evidence
records the degradation honestly rather than failing the run.

Uses the non-interactive 'Agg' backend so figures render headless (terminal,
CI, Databricks job clusters) without a display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def matplotlib_available() -> bool:
    try:
        import matplotlib  # noqa: F401

        return True
    except ImportError:
        return False


def _figure_dir(output_root: str, run_id: str) -> Path:
    path = Path(output_root) / "figures" / "deep_learning" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def _new_axes():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def learning_curve_figure(
    history: dict[str, list[float]], output_root: str, run_id: str
) -> str:
    if not matplotlib_available():
        return ""
    plt = _new_axes()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    ax.plot(list(epochs), history.get("train_loss", []), marker="o", label="train loss")
    if history.get("val_loss"):
        ax.plot(list(epochs), history["val_loss"], marker="s", label="validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("BCE loss")
    ax.set_title("Learning curve")
    ax.legend()
    fig.tight_layout()
    path = _figure_dir(output_root, run_id) / "learning_curve.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def calibration_curve_figure(
    y_true: np.ndarray, scores: np.ndarray, output_root: str, run_id: str, n_bins: int = 10
) -> str:
    if not matplotlib_available():
        return ""
    plt = _new_axes()
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    centers, observed = [], []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (scores > lo) & (scores <= hi)
        if mask.sum():
            centers.append((lo + hi) / 2)
            observed.append(float(y_true[mask].mean()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
    ax.plot(centers, observed, marker="o", label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed event rate")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    path = _figure_dir(output_root, run_id) / "calibration_curve.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def attribution_figure(
    ranked: list[tuple[str, float]], method: str, output_root: str, run_id: str, top_k: int = 10
) -> str:
    if not matplotlib_available() or not ranked:
        return ""
    plt = _new_axes()
    top = ranked[:top_k][::-1]
    names = [n for n, _ in top]
    values = [v for _, v in top]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names, values, color="#3b6ea5")
    ax.set_xlabel(f"importance ({method})")
    ax.set_title(f"Top feature attribution — {method}")
    fig.tight_layout()
    path = _figure_dir(output_root, run_id) / "attribution_top_features.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


def shock_sensitivity_figure(
    rows: list[dict[str, Any]], output_root: str, run_id: str
) -> str:
    if not matplotlib_available() or not rows:
        return ""
    plt = _new_axes()
    shocks = [r["shock"] * 100 for r in rows]
    aucs = [r["auc_roc"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(shocks, aucs, marker="o", color="#a5453b")
    ax.axvline(0, linestyle="--", color="gray", alpha=0.6)
    ax.set_xlabel("parallel feature shock (%)")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Top-feature shock sensitivity")
    fig.tight_layout()
    path = _figure_dir(output_root, run_id) / "top_feature_shock_sensitivity.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)
