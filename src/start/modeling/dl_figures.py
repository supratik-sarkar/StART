"""Figure generation for DL model review."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _import_pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_learning_curve(history: dict[str, list[float]], output_dir: str | Path) -> Path:
    plt = _import_pyplot()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "learning_curve.png"
    fig = plt.figure()
    plt.plot(history.get("train_loss", []), label="train_loss")
    plt.plot(history.get("val_loss", []), label="validation_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Deep Learning Training Diagnostics")
    plt.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_attribution_bar(attribution: pd.DataFrame, output_dir: str | Path, *, top_n: int = 20) -> Path:
    plt = _import_pyplot()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "attribution_top_features.png"
    df = attribution.head(top_n).iloc[::-1]
    fig = plt.figure(figsize=(8, max(4, 0.25 * len(df))))
    plt.barh(df["feature"], df["importance"])
    plt.xlabel("Importance")
    plt.title("Top DL Feature Attributions")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_sensitivity_curve(sensitivity: pd.DataFrame, output_dir: str | Path) -> Path:
    plt = _import_pyplot()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "top_feature_shock_sensitivity.png"
    fig = plt.figure()
    x = sensitivity["shock"] * 100
    plt.plot(x, sensitivity["auc_roc"], marker="o")
    plt.xlabel("Parallel shock to top features (%)")
    plt.ylabel("AUC-ROC")
    plt.title("DL Sensitivity to Top Feature Shocks")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_calibration(y_true, y_score, output_dir: str | Path, *, n_bins: int = 10) -> Path:
    import numpy as np

    plt = _import_pyplot()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "calibration_curve.png"
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (y_score >= lo) & (y_score < hi if hi < 1 else y_score <= hi)
        if mask.any():
            xs.append(float(y_score[mask].mean()))
            ys.append(float(y_true[mask].mean()))
    fig = plt.figure()
    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect")
    plt.plot(xs, ys, marker="o", label="model")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event rate")
    plt.title("DL Calibration Curve")
    plt.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
