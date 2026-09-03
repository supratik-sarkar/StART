"""Method options — the reviewer control surface.

The problem
-----------

The reviewer was offered this::

    FeatureEngineeringAgent recommends: Winsorize/clip extreme numeric outliers
      reason: Outliers detected in 4 column(s) (worst: credit_amount, 72 points)
      Apply outliers? [Y]es / o / [Q] ask

Three things are wrong with it. The rule is fixed — IQR at 1.5 — and never named. The
multiplier is not adjustable. And the only alternative to *this* method is *no* method,
as though winsorizing at Tukey's default and keeping every observation were the only
two defensible positions in statistics.

That is a library with a wizard on top. A reviewer accepting it has exercised no
judgement, and sealing the acceptance seals nothing worth sealing.

What replaces it
----------------

Every methodological choice becomes the same interaction: the agent proposes, the
*consequences of every option* are computed from the actual data, and the reviewer
chooses::

    Method                Affected    Retained    Rationale
    [1] IQR x 1.5         171 (7.2%)  92.8%       Tukey default; standard for skewed financials
    [2] IQR x 3.0          44 (1.8%)  98.2%       far-outliers only; preserves genuine tail risk
    [3] percentile 1/99    20 (2.0%)  98.0%       fixed proportion; distribution-agnostic
    [4] z-score |z| > 3    38 (1.6%)  98.4%       assumes approximate normality
    [5] none                0         100%        retain all observations

Four properties, each of which is the point:

**Consequences are real.** Every count is computed by running the rule over the data
before the prompt renders. Nothing is illustrative. A reviewer choosing blind is not
exercising judgement, and a menu of untested options is worse than no menu because it
looks like control.

**Parameters are adjustable.** The IQR multiplier, the percentile bounds, the z-score
threshold — all reviewer-settable. `custom` is a first-class option, not an escape hatch.

**Every option carries a rationale.** A menu without reasons is a quiz. The reviewer
should be able to defend the choice afterwards from what was on screen.

**A non-recommended choice requires a reviewer rationale** and lands in the seal, exactly
as an override does.

The decision threshold matters most
-----------------------------------

Every metric in this product has been computed at 0.5. At 5.5% prevalence that is
indefensible: it produced a model predicting zero positives on every cohort. Against a
published 5:1 cost matrix it is simply the wrong number. :func:`threshold_options`
computes the F1-optimal, F2-optimal, cost-optimal and alert-budget thresholds and shows
the confusion matrix each would produce, so the reviewer picks the operating point
rather than inheriting a default nobody chose.

Requires numpy and pandas; scikit-learn only for the threshold helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "MethodOption",
    "MethodMenu",
    "outlier_options",
    "imputation_options",
    "encoding_options",
    "scaling_options",
    "imbalance_options",
    "threshold_options",
]


@dataclass(frozen=True)
class MethodOption:
    """One selectable method, with its computed consequence."""

    key: str
    label: str
    rationale: str
    #: Rows / cells / features the method would act on. None when not applicable.
    affected: int | None = None
    #: Denominator for the affected count.
    total: int | None = None
    #: Adjustable parameters, name -> current value.
    parameters: dict[str, Any] = field(default_factory=dict)
    #: Extra computed detail shown beneath the row (e.g. a confusion matrix).
    detail: str = ""
    recommended: bool = False

    @property
    def affected_pct(self) -> float | None:
        if self.affected is None or not self.total:
            return None
        return round(100.0 * self.affected / self.total, 2)

    @property
    def retained_pct(self) -> float | None:
        pct = self.affected_pct
        return None if pct is None else round(100.0 - pct, 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "rationale": self.rationale,
            "affected": self.affected,
            "total": self.total,
            "affected_pct": self.affected_pct,
            "retained_pct": self.retained_pct,
            "parameters": dict(sorted(self.parameters.items())),
            "detail": self.detail,
            "recommended": self.recommended,
        }


@dataclass(frozen=True)
class MethodMenu:
    """A decision the reviewer makes, with every option's consequence computed."""

    decision: str
    prompt: str
    options: tuple[MethodOption, ...]
    #: Why the agent proposes what it proposes.
    agent_reason: str = ""
    #: Column the affected counts refer to, when the decision is per-feature.
    scope: str = ""

    def recommended_option(self) -> MethodOption | None:
        for option in self.options:
            if option.recommended:
                return option
        return self.options[0] if self.options else None

    def option(self, key: str) -> MethodOption | None:
        for candidate in self.options:
            if candidate.key == key:
                return candidate
        return None

    def render_lines(self) -> list[str]:
        """Plain-text table. The caller styles it; this module stays render-agnostic."""
        lines: list[str] = []
        if self.agent_reason:
            lines.append(f"  reason: {self.agent_reason}")
        lines.append("")
        lines.append(f"  {'':4}{'Method':<26}{'Affected':>14}{'Retained':>11}   Rationale")
        for index, option in enumerate(self.options, start=1):
            affected = (
                f"{option.affected:,} ({option.affected_pct:.1f}%)"
                if option.affected is not None and option.affected_pct is not None
                else "—"
            )
            retained = f"{option.retained_pct:.1f}%" if option.retained_pct is not None else "—"
            mark = "*" if option.recommended else " "
            lines.append(
                f"  [{index}]{mark}{option.label:<26}{affected:>14}{retained:>11}   {option.rationale}"
            )
            if option.detail:
                lines.append(f"       {option.detail}")
        lines.append("")
        lines.append("  * = agent recommendation")
        return lines

    def as_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "prompt": self.prompt,
            "agent_reason": self.agent_reason,
            "scope": self.scope,
            "options": [o.as_dict() for o in self.options],
        }


# --------------------------------------------------------------------------- #
# Outliers
# --------------------------------------------------------------------------- #
def _iqr_mask(values: np.ndarray, multiplier: float) -> np.ndarray:
    q1, q3 = np.nanpercentile(values, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return np.zeros(values.shape, dtype=bool)
    return (values < q1 - multiplier * iqr) | (values > q3 + multiplier * iqr)


def _percentile_mask(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    lo, hi = np.nanpercentile(values, [lower, upper])
    return (values < lo) | (values > hi)


def _zscore_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    std = np.nanstd(values)
    if std <= 0:
        return np.zeros(values.shape, dtype=bool)
    return np.abs((values - np.nanmean(values)) / std) > threshold


def outlier_options(
    frame: Any,
    numeric_columns: list[str],
    *,
    iqr_multiplier: float = 1.5,
    percentile_bounds: tuple[float, float] = (1.0, 99.0),
    zscore_threshold: float = 3.0,
) -> MethodMenu:
    """Outlier treatment options with real affected-cell counts.

    Counts are cells across all numeric columns, not rows: a row with three extreme
    values is three treatments, and reporting it as one understates the intervention.
    """
    total_cells = 0
    counts = {"iqr_default": 0, "iqr_wide": 0, "percentile": 0, "zscore": 0}

    for column in numeric_columns:
        series = frame[column]
        values = (
            series.to_numpy(dtype=float, copy=False)
            if hasattr(series, "to_numpy")
            else np.asarray(series, dtype=float)
        )
        values = values[~np.isnan(values)]
        if values.size == 0:
            continue
        total_cells += int(values.size)
        counts["iqr_default"] += int(_iqr_mask(values, iqr_multiplier).sum())
        counts["iqr_wide"] += int(_iqr_mask(values, iqr_multiplier * 2.0).sum())
        counts["percentile"] += int(_percentile_mask(values, *percentile_bounds).sum())
        counts["zscore"] += int(_zscore_mask(values, zscore_threshold).sum())

    worst = ""
    if numeric_columns and total_cells:
        per_column = {}
        for column in numeric_columns:
            series = frame[column]
            values = (
                series.to_numpy(dtype=float, copy=False)
                if hasattr(series, "to_numpy")
                else np.asarray(series, dtype=float)
            )
            values = values[~np.isnan(values)]
            if values.size:
                per_column[column] = int(_iqr_mask(values, iqr_multiplier).sum())
        if per_column:
            worst_col = max(per_column, key=lambda c: per_column[c])
            worst = f"worst: {worst_col} ({per_column[worst_col]} points)"

    options = (
        MethodOption(
            key="iqr",
            label=f"IQR x {iqr_multiplier}",
            rationale="Tukey default; robust to skew, standard for financial magnitudes",
            affected=counts["iqr_default"],
            total=total_cells,
            parameters={"multiplier": iqr_multiplier},
            recommended=True,
        ),
        MethodOption(
            key="iqr_wide",
            label=f"IQR x {iqr_multiplier * 2.0}",
            rationale="far-outliers only; preserves genuine tail risk, which matters "
            "when the tail is the phenomenon being modelled",
            affected=counts["iqr_wide"],
            total=total_cells,
            parameters={"multiplier": iqr_multiplier * 2.0},
        ),
        MethodOption(
            key="percentile",
            label=f"percentile {percentile_bounds[0]:g}/{percentile_bounds[1]:g}",
            rationale="fixed proportion clipped regardless of distribution shape",
            affected=counts["percentile"],
            total=total_cells,
            parameters={"lower": percentile_bounds[0], "upper": percentile_bounds[1]},
        ),
        MethodOption(
            key="zscore",
            label=f"z-score |z| > {zscore_threshold:g}",
            rationale="assumes approximate normality; questionable on skewed data",
            affected=counts["zscore"],
            total=total_cells,
            parameters={"threshold": zscore_threshold},
        ),
        MethodOption(
            key="none",
            label="none",
            rationale="retain every observation; extremes may be real and material",
            affected=0,
            total=total_cells,
        ),
        MethodOption(
            key="custom",
            label="custom",
            rationale="specify rule and parameter",
            parameters={"multiplier": iqr_multiplier},
        ),
    )

    return MethodMenu(
        decision="outlier_treatment",
        prompt="Outlier treatment",
        options=options,
        agent_reason=f"{len(numeric_columns)} numeric column(s) examined" + (f"; {worst}" if worst else ""),
    )


# --------------------------------------------------------------------------- #
# Missing values
# --------------------------------------------------------------------------- #
def imputation_options(frame: Any, columns: list[str] | None = None) -> MethodMenu:
    """Imputation options with real missing-cell counts."""
    columns = columns or list(frame.columns)
    total_cells = int(len(frame) * len(columns)) if len(columns) else 0
    missing = int(sum(int(frame[c].isna().sum()) for c in columns))
    rows_with_missing = int(frame[columns].isna().any(axis=1).sum()) if columns else 0
    cols_with_missing = [c for c in columns if int(frame[c].isna().sum()) > 0]

    options = (
        MethodOption(
            key="median_mode",
            label="median / mode",
            rationale="robust central tendency; does not assume a distribution",
            affected=missing,
            total=total_cells,
            recommended=True,
        ),
        MethodOption(
            key="mean_mode",
            label="mean / mode",
            rationale="efficient when approximately normal; sensitive to outliers",
            affected=missing,
            total=total_cells,
        ),
        MethodOption(
            key="indicator",
            label="indicator + median",
            rationale="preserves missingness as signal — often predictive in credit "
            "and fraud, where absence of data is itself informative",
            affected=missing,
            total=total_cells,
            detail=f"adds {len(cols_with_missing)} indicator column(s)",
        ),
        MethodOption(
            key="drop_rows",
            label="drop rows",
            rationale="no imputation assumption; costs observations",
            affected=rows_with_missing,
            total=int(len(frame)),
            detail=f"retains {len(frame) - rows_with_missing:,} of {len(frame):,} rows",
        ),
        MethodOption(
            key="drop_columns",
            label="drop columns",
            rationale="removes affected features entirely",
            affected=len(cols_with_missing),
            total=len(columns),
        ),
        MethodOption(
            key="none",
            label="none",
            rationale="leave missing values in place; downstream must handle them",
            affected=0,
            total=total_cells,
        ),
    )

    return MethodMenu(
        decision="imputation",
        prompt="Missing value treatment",
        options=options,
        agent_reason=f"{len(cols_with_missing)} column(s) contain missing values "
        f"({missing:,} cells, {rows_with_missing:,} rows affected)",
    )


# --------------------------------------------------------------------------- #
# Categorical encoding
# --------------------------------------------------------------------------- #
def encoding_options(frame: Any, categorical_columns: list[str]) -> MethodMenu:
    """Encoding options with a real cardinality picture.

    One-hot on a 40-level column is a very different proposition from one-hot on a
    binary flag, and the reviewer cannot judge without seeing the widths.
    """
    cardinalities = {c: int(frame[c].nunique(dropna=True)) for c in categorical_columns}
    total_levels = int(sum(cardinalities.values()))
    high_card = [c for c, n in cardinalities.items() if n > 10]
    worst = max(cardinalities, key=lambda c: cardinalities[c]) if cardinalities else ""

    # Encoding and scaling act on every column by definition, so an
    # "affected / retained" ratio is meaningless for them. Counts go in `detail`
    # where they inform, rather than in a percentage column that would read as
    # "0% of your data retained".
    options = (
        MethodOption(
            key="onehot",
            label="one-hot",
            rationale="no ordinal assumption; widens the matrix",
            detail=f"{len(categorical_columns)} features -> {total_levels} columns",
            recommended=True,
        ),
        MethodOption(
            key="ordinal",
            label="ordinal",
            rationale="compact, but imposes an order the data may not have",
            detail=f"{len(categorical_columns)} features -> {len(categorical_columns)} columns",
        ),
        MethodOption(
            key="target",
            label="target / WoE",
            rationale="strong for credit scorecards, but leaks the target unless "
            "fitted inside the training fold only",
            detail="LEAKAGE RISK: must be fitted on train folds only",
        ),
        MethodOption(
            key="frequency",
            label="frequency",
            rationale="compact and leakage-free; discards level identity",
            detail=f"{len(categorical_columns)} features -> {len(categorical_columns)} columns",
        ),
        MethodOption(
            key="mixed",
            label="one-hot low / target high",
            rationale=f"one-hot below 10 levels, target-encode above "
            f"({len(high_card)} high-cardinality feature(s))",
            detail=f"high-cardinality: {', '.join(high_card) if high_card else 'none'}",
        ),
        MethodOption(
            key="none",
            label="none",
            rationale="leave as-is; only viable if the estimator handles categoricals",
        ),
    )

    return MethodMenu(
        decision="encoding",
        prompt="Categorical encoding",
        options=options,
        agent_reason=f"{len(categorical_columns)} categorical feature(s), "
        f"{total_levels} total levels"
        + (f"; highest cardinality: {worst} ({cardinalities.get(worst, 0)})" if worst else ""),
    )


# --------------------------------------------------------------------------- #
# Scaling and imbalance
# --------------------------------------------------------------------------- #
def scaling_options(frame: Any, numeric_columns: list[str]) -> MethodMenu:
    """Scaling options, with the actual scale disparity that motivates the choice."""
    ranges = {}
    for column in numeric_columns:
        series = frame[column]
        values = (
            series.to_numpy(dtype=float, copy=False)
            if hasattr(series, "to_numpy")
            else np.asarray(series, dtype=float)
        )
        values = values[~np.isnan(values)]
        if values.size:
            ranges[column] = float(np.nanmax(values) - np.nanmin(values))
    spread = (max(ranges.values()) / min(ranges.values())) if ranges and min(ranges.values()) > 0 else 0.0

    options = (
        MethodOption(
            key="standard",
            label="standard (z-score)",
            rationale="zero mean, unit variance; the default for gradient methods",
            detail=f"applies to all {len(numeric_columns)} numeric feature(s)",
            recommended=True,
        ),
        MethodOption(
            key="robust",
            label="robust (median/IQR)",
            rationale="resistant to the outliers you may have chosen to keep",
            parameters={"quantile_range": (25.0, 75.0)},
        ),
        MethodOption(
            key="minmax",
            label="min-max",
            rationale="bounded to [0,1]; sensitive to extremes",
        ),
        MethodOption(
            key="none",
            label="none",
            rationale="raw scale; acceptable for tree models, harmful for neural nets",
        ),
    )

    return MethodMenu(
        decision="scaling",
        prompt="Numeric scaling",
        options=options,
        agent_reason=f"{len(numeric_columns)} numeric feature(s); "
        f"largest-to-smallest range ratio {spread:,.0f}x"
        if spread
        else f"{len(numeric_columns)} numeric feature(s)",
    )


def imbalance_options(y: Any, *, already_weighted: bool = False) -> MethodMenu:
    """Imbalance options, with the real class counts and a calibration warning.

    ``already_weighted`` prevents the double-correction that a reviewer cannot see:
    class weights enabled at configuration, then resampling applied on top, producing
    a model that systematically over-predicts the positive class and reports
    calibration error nobody can explain.
    """
    values = np.asarray(y).reshape(-1)
    total = int(values.size)
    positives = int((values == 1).sum())
    negatives = total - positives
    prevalence = (positives / total) if total else 0.0

    options = (
        MethodOption(
            key="class_weight",
            label="class weights",
            rationale="reweights the loss; leaves the data and the base rate intact",
            affected=positives,
            total=total,
            recommended=not already_weighted,
            detail="ALREADY ENABLED at configuration — selecting another method here "
            "applies a second correction"
            if already_weighted
            else "",
        ),
        MethodOption(
            key="oversample",
            label="random oversample",
            rationale="duplicates minority rows; changes the base rate and therefore "
            "the meaning of every predicted probability",
            affected=negatives - positives,
            total=total,
            detail="CALIBRATION: predicted probabilities no longer estimate the true rate",
        ),
        MethodOption(
            key="undersample",
            label="random undersample",
            rationale="discards majority rows; cheap but throws away information",
            affected=negatives - positives,
            total=total,
            detail=f"retains {2 * positives:,} of {total:,} rows",
        ),
        MethodOption(
            key="smote",
            label="SMOTE (if available)",
            rationale="synthesises minority examples; can fabricate implausible records "
            "in categorical or bounded feature spaces",
            affected=negatives - positives,
            total=total,
        ),
        MethodOption(
            key="threshold_only",
            label="threshold adjustment only",
            rationale="train unweighted and move the decision threshold instead — "
            "preserves calibration and is often the honest choice",
            affected=0,
            total=total,
        ),
        MethodOption(
            key="none",
            label="none",
            rationale="accept the imbalance; defensible when the model is used for "
            "ranking rather than classification",
            affected=0,
            total=total,
        ),
    )

    return MethodMenu(
        decision="imbalance",
        prompt="Class imbalance treatment",
        options=options,
        agent_reason=f"{positives:,} positive of {total:,} ({prevalence:.1%} prevalence)"
        + ("; class weighting already enabled at configuration" if already_weighted else ""),
    )


# --------------------------------------------------------------------------- #
# Decision threshold — the highest-value control in this module
# --------------------------------------------------------------------------- #
def threshold_options(
    y_true: Any,
    scores: Any,
    *,
    cost_false_negative: float = 1.0,
    cost_false_positive: float = 1.0,
    alert_budget: float = 0.05,
) -> MethodMenu:
    """Candidate operating points, each with its real confusion matrix.

    Every metric in this product has been reported at 0.5. At low prevalence that is
    not a decision rule, it is an accident of the sigmoid — and it produced a model
    that predicted zero positives on every cohort while reporting a recall of 0.6.

    Where a dataset publishes a cost matrix, the cost-optimal threshold is the
    defensible operating point and it can be cited rather than argued.
    """
    y = np.asarray(y_true).reshape(-1).astype(int)
    s = np.asarray(scores, dtype=float).reshape(-1)
    total = int(y.size)
    positives = int(y.sum())

    def confusion(threshold: float) -> tuple[int, int, int, int]:
        predicted = (s >= threshold).astype(int)
        tp = int(((predicted == 1) & (y == 1)).sum())
        fp = int(((predicted == 1) & (y == 0)).sum())
        fn = int(((predicted == 0) & (y == 1)).sum())
        tn = int(((predicted == 0) & (y == 0)).sum())
        return tn, fp, fn, tp

    def fbeta(threshold: float, beta: float) -> float:
        _, fp, fn, tp = confusion(threshold)
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            return 0.0
        b2 = beta * beta
        return (1 + b2) * precision * recall / (b2 * precision + recall)

    grid = np.unique(np.round(np.linspace(0.01, 0.99, 99), 4))

    best_f1 = max(grid, key=lambda t: fbeta(t, 1.0))
    best_f2 = max(grid, key=lambda t: fbeta(t, 2.0))

    def total_cost(threshold: float) -> float:
        _, fp, fn, _ = confusion(threshold)
        return cost_false_negative * fn + cost_false_positive * fp

    best_cost = min(grid, key=total_cost)

    target_alerts = max(1, int(round(alert_budget * total)))
    budget_threshold = float(np.quantile(s, 1.0 - alert_budget)) if s.size else 0.5

    def describe(threshold: float) -> tuple[int, str]:
        tn, fp, fn, tp = confusion(threshold)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        alerts = tp + fp
        return alerts, (
            f"t={threshold:.2f}  TP={tp} FP={fp} FN={fn} TN={tn}  "
            f"precision={precision:.3f} recall={recall:.3f}"
        )

    specs = [
        ("fixed_050", "fixed 0.50", 0.5, "the default; rarely defensible at low prevalence", False),
        ("f1_optimal", "F1-optimal", float(best_f1), "balances precision and recall equally", False),
        (
            "f2_optimal",
            "F2-optimal",
            float(best_f2),
            "weights recall 2x — appropriate when misses cost more than false alarms",
            True,
        ),
        (
            "cost_optimal",
            f"cost-optimal ({cost_false_negative:g}:{cost_false_positive:g})",
            float(best_cost),
            "minimises expected cost under the stated matrix; citable where the dataset publishes one",
            False,
        ),
        (
            "alert_budget",
            f"alert budget {alert_budget:.0%}",
            budget_threshold,
            f"caps alerts at roughly {target_alerts:,} cases, matching review capacity",
            False,
        ),
    ]

    options = []
    for key, label, threshold, rationale, recommended in specs:
        alerts, detail = describe(threshold)
        options.append(
            MethodOption(
                key=key,
                label=label,
                rationale=rationale,
                affected=alerts,
                total=total,
                parameters={"threshold": round(float(threshold), 4)},
                detail=detail,
                recommended=recommended,
            )
        )
    options.append(
        MethodOption(
            key="custom",
            label="custom",
            rationale="specify the operating threshold directly",
            parameters={"threshold": 0.5},
        )
    )

    return MethodMenu(
        decision="decision_threshold",
        prompt="Decision threshold",
        options=tuple(options),
        agent_reason=f"{positives:,} positive of {total:,} "
        f"({positives / total if total else 0:.1%} prevalence); "
        f"cost ratio FN:FP = {cost_false_negative:g}:{cost_false_positive:g}",
    )
