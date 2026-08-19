"""Inline ASCII and sparkline terminal plots (Workstream C1).

Rendered live in theme colors, <= 100 columns wide, degrading to plain text
under NO_COLOR or non-TTY environments.

Plots:
1. Calibration reliability diagram (binned confidence vs observed rate)
2. Precision-Recall curve (PR curve with baseline rate)
3. ROC curve (True Positive Rate vs False Positive Rate)
4. Score distribution by class (ASCII histogram)
5. Threshold sweep (Precision, Recall, Alert Rate vs Threshold)
6. Cohort metric drift sparkline (bar heights across features)

Deterministic rendering across runs and PYTHONHASHSEED (Amendment 5b).
"""

from __future__ import annotations

import os
import sys

import numpy as np

SPARK_CHARS = (" ", "▂", "▃", "▄", "▅", "▆", "▇", "█")


def _is_no_color() -> bool:
    return bool(os.environ.get("NO_COLOR")) or not sys.stdout.isatty()


def _color(text: str, code: str) -> str:
    if _is_no_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _dim(text: str) -> str:
    return _color(text, "2")


def _cyan(text: str) -> str:
    return _color(text, "36")


def _green(text: str) -> str:
    return _color(text, "32")


def _yellow(text: str) -> str:
    return _color(text, "33")


def _red(text: str) -> str:
    return _color(text, "31")


def _bold(text: str) -> str:
    return _color(text, "1")


def render_calibration_ascii(
    y_true: np.ndarray | list[float],
    scores: np.ndarray | list[float],
    n_bins: int = 10,
    width: int = 40,
) -> str:
    """Render an ASCII reliability/calibration diagram.

    Shows mean predicted probability vs observed event rate per bin.
    """
    y = np.asarray(y_true, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) == 0:
        return "No data for calibration plot."

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    lines = [
        _bold("  Calibration Reliability Diagram (Mean Score vs Observed Event Rate)"),
        _dim(f"  {'Bin Range':<15} | {'Observed Rate':<15} | Histogram"),
        "  " + "─" * min(width + 35, 90),
    ]

    total_n = len(s)
    total_gap = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask = (s >= lo) & (s <= hi) if lo == 0.0 else (s > lo) & (s <= hi)
        count = int(mask.sum())
        bin_label = f"[{lo:.2f} - {hi:.2f}]"
        if count == 0:
            lines.append(f"  {bin_label:<15} | {'—':<15} |")
            continue
        obs_rate = float(y[mask].mean())
        mean_score = float(s[mask].mean())
        gap = abs(mean_score - obs_rate)
        total_gap += (count / total_n) * gap

        bar_len = int(round(obs_rate * (width - 10)))
        bar = "█" * bar_len
        diff_str = f"Δ={obs_rate - mean_score:+.2f}"
        lines.append(
            f"  {bin_label:<15} | {obs_rate:6.2%} (n={count:<3}) | {_cyan(bar)} {_dim(diff_str)}"
        )

    cal_msg = (
        "well-calibrated probabilities"
        if total_gap < 0.08
        else "uncalibrated; probability scaling recommended"
    )
    interpretation = f"ECE={total_gap:.4f} — {cal_msg}"
    lines.append("  " + "─" * min(width + 35, 90))
    lines.append(_dim(f"  Interpretation: {interpretation}"))
    return "\n".join(lines)


def render_roc_curve_ascii(
    y_true,
    scores,
    width: int = 35,
    height: int = 10,
    *,
    cohort: str = "",
) -> str:
    """ASCII ROC curve. ``cohort`` labels which split this is — never omit it."""
    from sklearn.metrics import roc_auc_score, roc_curve

    y = np.asarray(y_true, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if len(np.unique(y)) < 2:
        return _dim("  ROC curve undefined: cohort contains a single class.")

    fpr, tpr, _ = roc_curve(y, s)
    auc_val = float(roc_auc_score(y, s))

    order = np.argsort(fpr, kind="stable")
    fpr, tpr = fpr[order], tpr[order]

    grid = [[" " for _ in range(width)] for _ in range(height)]

    def plot(col: int, row: int, ch: str, *, overwrite: bool = True) -> None:
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        if overwrite or grid[height - 1 - row][col] == " ":
            grid[height - 1 - row][col] = ch

    # 1. The empirical step function, filled between vertices.
    cols = [int(round(f * (width - 1))) for f in fpr]
    rows = [int(round(t * (height - 1))) for t in tpr]
    for i in range(len(cols) - 1):
        c0, r0, c1, r1 = cols[i], rows[i], cols[i + 1], rows[i + 1]
        for r in range(min(r0, r1), max(r0, r1) + 1):   # vertical rise
            plot(c0, r, "■")
        for c in range(min(c0, c1), max(c0, c1) + 1):   # horizontal run
            plot(c, r1, "■")
    if cols:
        plot(cols[0], rows[0], "■")
        plot(cols[-1], rows[-1], "■")

    # 2. Chance line LAST, into cells the curve did not claim
    for r in range(height):
        frac = r / (height - 1) if height > 1 else 0.0
        col = max(0, min(width - 1, int(round(frac * (width - 1)))))
        if grid[height - 1 - r][col] == " ":
            grid[height - 1 - r][col] = "·"

    label = f" — {cohort}" if cohort else ""
    lines = [
        _bold(f"  ROC curve{label}  (AUC {auc_val:.4f})"),
        _dim("  TPR ^"),
    ]
    for r_idx, row in enumerate(grid):
        tick = (
            f"{1.0 - (r_idx / (height - 1)):3.1f}"
            if r_idx in (0, height // 2, height - 1)
            else "   "
        )
        painted = "".join(_green(ch) if ch == "■" else _dim(ch) for ch in row)
        lines.append(f"  {_dim(tick)} │ {painted}")

    lines.append("  " + " " * 5 + "└" + "─" * width + " > FPR")
    lines.append("  " + " " * 6 + f"0.0{' ' * (width - 6)}1.0")

    if auc_val >= 0.80:
        verdict = "strong separation between classes"
    elif auc_val >= 0.65:
        verdict = "moderate separation"
    elif auc_val >= 0.55:
        verdict = "weak separation, close to chance"
    else:
        verdict = "no useful separation — at or below the chance line"
    lines.append(_dim(f"  Interpretation: {verdict}. The dotted line is chance (AUC 0.5)."))
    return "\n".join(lines)


def render_pr_curve_ascii(
    y_true: np.ndarray | list[float],
    scores: np.ndarray | list[float],
    width: int = 35,
    height: int = 10,
) -> str:
    """Render an ASCII Precision-Recall curve."""
    from sklearn.metrics import average_precision_score, precision_recall_curve

    y = np.asarray(y_true, dtype=float).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if len(np.unique(y)) < 2:
        return "Single-class cohort: PR curve undefined."

    precision, recall, _ = precision_recall_curve(y, s)
    pr_auc = average_precision_score(y, s)
    base_rate = float(np.mean(y))

    grid = [[" " for _ in range(width)] for _ in range(height)]

    # Baseline rate horizontal line
    base_row = int(round((1.0 - base_rate) * (height - 1)))
    base_row = max(0, min(height - 1, base_row))
    for c in range(width):
        grid[base_row][c] = "·"

    # Draw PR curve points
    for p, r_val in zip(precision, recall, strict=False):
        c = int(round(r_val * (width - 1)))
        r = int(round(p * (height - 1)))
        c = max(0, min(width - 1, c))
        r = max(0, min(height - 1, r))
        grid[height - 1 - r][c] = "■"

    lines = [
        _bold(f"  Precision-Recall Curve (PR-AUC: {pr_auc:.4f} vs Base Rate: {base_rate:.2%})"),
        _dim("  Prec ^"),
    ]
    for r_idx, row in enumerate(grid):
        val_label = f"{1.0 - (r_idx / (height - 1)):3.1f}" if r_idx in (0, height // 2, height - 1) else "   "
        lines.append(f"  {_dim(val_label)} │ {''.join(_yellow(ch) if ch == '■' else _dim(ch) for ch in row)}")

    lines.append("  " + " " * 5 + "└" + "─" * width + " > Recall")
    lines.append("  " + " " * 6 + f"0.0{' ' * (width - 6)}1.0")
    lift = pr_auc / base_rate if base_rate > 0 else 1.0
    interp = f"PR-AUC represents a {lift:.2f}x lift over random prevalence baseline."
    lines.append(_dim(f"  Interpretation: {interp}"))
    return "\n".join(lines)


def render_score_distribution_ascii(
    y_true: np.ndarray | list[float],
    scores: np.ndarray | list[float],
    bins: int = 15,
    width: int = 35,
) -> str:
    """Render an ASCII score distribution histogram grouped by class."""
    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if len(y) == 0:
        return "No data for score distribution."

    edges = np.linspace(0.0, 1.0, bins + 1)
    lines = [
        _bold("  Score Distribution by Class (Negatives vs Positives)"),
        _dim(f"  {'Score Bin':<15} | {'Class 0 (Neg)':<15} | {'Class 1 (Pos)':<15}"),
        "  " + "─" * min(width + 40, 90),
    ]

    neg_s = s[y == 0]
    pos_s = s[y == 1]
    max_neg = max(1, len(neg_s))
    max_pos = max(1, len(pos_s))

    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        mask_neg = (neg_s >= lo) & (neg_s <= hi) if lo == 0.0 else (neg_s > lo) & (neg_s <= hi)
        mask_pos = (pos_s >= lo) & (pos_s <= hi) if lo == 0.0 else (pos_s > lo) & (pos_s <= hi)
        c0, c1 = int(mask_neg.sum()), int(mask_pos.sum())
        bar0 = "■" * int(round((c0 / max_neg) * (width // 2)))
        bar1 = "■" * int(round((c1 / max_pos) * (width // 2)))
        bin_label = f"[{lo:.2f} - {hi:.2f}]"
        lines.append(
            f"  {bin_label:<15} | {_cyan(f'{c0:4d} ' + bar0):<25} | {_yellow(f'{c1:4d} ' + bar1):<25}"
        )

    lines.append("  " + "─" * min(width + 40, 90))
    lines.append(_dim("  Interpretation: Class separation & probability density across prediction range."))
    return "\n".join(lines)


def render_threshold_sweep_ascii(
    y_true: np.ndarray | list[float],
    scores: np.ndarray | list[float],
    thresholds: list[float] | None = None,
    width: int = 50,
) -> str:
    """Render precision, recall, and alert volume across decision thresholds."""
    from sklearn.metrics import precision_score, recall_score

    y = np.asarray(y_true, dtype=int).reshape(-1)
    s = np.asarray(scores, dtype=float).reshape(-1)
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    lines = [
        _bold("  Threshold Sweep Analysis"),
        _dim(f"  {'Cutoff':<8} | {'Precision':<10} | {'Recall':<10} | {'Alert Rate':<12} | Trade-off"),
        "  " + "─" * min(width + 25, 90),
    ]

    total = len(y)
    for t in thresholds:
        preds = (s >= t).astype(int)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        alert_rate = preds.sum() / total if total else 0.0

        p_bar = "█" * int(round(prec * 10))
        r_bar = "█" * int(round(rec * 10))
        lines.append(
            f"  {t:<8.2f} | {prec:9.2%}  | {rec:9.2%}  | {alert_rate:10.2%}   | "
            f"P:{_cyan(p_bar):<12} R:{_yellow(r_bar)}"
        )

    lines.append("  " + "─" * min(width + 25, 90))
    lines.append(_dim("  Interpretation: Lower cutoff catches more positives at higher alert cost."))
    return "\n".join(lines)


def render_drift_sparkline(
    drifts: dict[str, float] | list[float],
    max_features: int = 8,
) -> str:
    """Render a deterministic sparkline of drift values across features."""
    if isinstance(drifts, dict):
        items = sorted(drifts.items(), key=lambda kv: abs(kv[1]), reverse=True)[:max_features]
        vals = [v for _, v in items]
        labels = [k for k, _ in items]
    else:
        vals = list(drifts)[:max_features]
        labels = [f"f_{i}" for i in range(len(vals))]

    if not vals:
        return "No drift metrics available."

    max_v = max(map(abs, vals)) if any(vals) else 1.0
    sparks = []
    for v in vals:
        ratio = min(1.0, abs(v) / (max_v if max_v > 0 else 1.0))
        idx = int(round(ratio * (len(SPARK_CHARS) - 1)))
        idx = max(0, min(len(SPARK_CHARS) - 1, idx))
        ch = SPARK_CHARS[idx]
        sparks.append(_yellow(ch) if ratio > 0.5 else _green(ch))

    spark_str = "".join(sparks)
    header = _bold("  Feature Sensitivity / Drift Sparkline:")
    detail = " · ".join(f"{labels[i]}: {vals[i]:+.3f}" for i in range(len(vals)))
    return f"{header} {spark_str}\n  {_dim(detail)}"
