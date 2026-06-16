"""Split planning: user-selected strategy, user-controlled proportions.

The split is never hardcoded. The user (or an upstream agent) chooses a
strategy and the train/test/OOS proportions; OOS generation is always
explicit. Every plan emits an evidence record describing exactly how the data
was partitioned, so the review can cite the split it validated against.

Strategies:
    random      - shuffle then slice (no stratification)
    stratified  - preserve target class balance across splits (classification)
    time_based  - order by a timestamp column; OOS is the most recent block
                  (the honest setup for temporal generalization / forecasting)
    group       - keep all rows of an entity together (no leakage across split)
    custom      - caller supplies explicit boolean masks or index arrays
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from start.core.schemas import Status, TestResult

SPLIT_STRATEGIES = ("random", "stratified", "time_based", "group", "custom")


@dataclass
class SplitPlan:
    strategy: str
    fractions: tuple[float, float, float]
    train: pd.DataFrame
    test: pd.DataFrame
    oos: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def sizes(self) -> tuple[int, int, int]:
        return len(self.train), len(self.test), len(self.oos)


class SplitPlanner:
    """Produce a train/test/OOS SplitPlan under a chosen strategy."""

    def plan(
        self,
        df: pd.DataFrame,
        *,
        strategy: str = "stratified",
        target_column: str | None = None,
        fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
        time_column: str | None = None,
        group_column: str | None = None,
        custom_masks: dict[str, Any] | None = None,
        seed: int = 42,
    ) -> SplitPlan:
        if strategy not in SPLIT_STRATEGIES:
            raise ValueError(f"Unknown split strategy '{strategy}'. Known: {SPLIT_STRATEGIES}")
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError(f"Split fractions must sum to 1.0, got {fractions}")

        if strategy == "custom":
            return self._custom(df, fractions, custom_masks)
        if strategy == "time_based":
            return self._time_based(df, fractions, time_column)
        if strategy == "group":
            return self._group(df, fractions, group_column, seed)
        if strategy == "stratified":
            return self._stratified(df, fractions, target_column, seed)
        return self._random(df, fractions, seed)

    # -- strategies -------------------------------------------------------- #
    def _random(self, df, fractions, seed) -> SplitPlan:
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df))
        return self._slice_by_index(df, idx, fractions, "random", [])

    def _stratified(self, df, fractions, target_column, seed) -> SplitPlan:
        if not target_column or target_column not in df.columns:
            return self._random(df, fractions, seed)  # fall back explicitly
        from sklearn.model_selection import train_test_split

        train_frac, test_frac, oos_frac = fractions
        strat = df[target_column]
        can_stratify = strat.value_counts().min() >= 3
        try:
            train, rest = train_test_split(
                df,
                test_size=test_frac + oos_frac,
                stratify=strat if can_stratify else None,
                random_state=seed,
            )
            rest_strat = rest[target_column] if can_stratify else None
            test, oos = train_test_split(
                rest,
                test_size=oos_frac / (test_frac + oos_frac),
                stratify=rest_strat,
                random_state=seed,
            )
        except ValueError:
            return self._random(df, fractions, seed)
        notes = [] if can_stratify else ["Too few per-class rows to stratify; used random split."]
        return SplitPlan(
            "stratified",
            fractions,
            train.reset_index(drop=True),
            test.reset_index(drop=True),
            oos.reset_index(drop=True),
            notes,
        )

    def _time_based(self, df, fractions, time_column) -> SplitPlan:
        if not time_column or time_column not in df.columns:
            raise ValueError("time_based split requires a valid time_column.")
        ordered = df.sort_values(time_column).reset_index(drop=True)
        idx = np.arange(len(ordered))
        plan = self._slice_by_index(
            ordered, idx, fractions, "time_based",
            [f"Ordered by '{time_column}'; OOS is the most recent block (no shuffling)."],
        )
        return plan

    def _group(self, df, fractions, group_column, seed) -> SplitPlan:
        if not group_column or group_column not in df.columns:
            raise ValueError("group split requires a valid group_column.")
        rng = np.random.default_rng(seed)
        groups = df[group_column].unique()
        rng.shuffle(groups)
        train_frac, test_frac, _ = fractions
        n = len(groups)
        n_train, n_test = int(n * train_frac), int(n * test_frac)
        g_train = set(groups[:n_train])
        g_test = set(groups[n_train : n_train + n_test])
        train = df[df[group_column].isin(g_train)]
        test = df[df[group_column].isin(g_test)]
        oos = df[~df[group_column].isin(g_train | g_test)]
        return SplitPlan(
            "group",
            fractions,
            train.reset_index(drop=True),
            test.reset_index(drop=True),
            oos.reset_index(drop=True),
            [f"Grouped by '{group_column}'; no entity appears in more than one split."],
        )

    def _custom(self, df, fractions, custom_masks) -> SplitPlan:
        if not custom_masks or not {"train", "test", "oos"} <= set(custom_masks):
            raise ValueError("custom split requires masks for 'train', 'test', and 'oos'.")

        def _take(key):
            mask = custom_masks[key]
            if isinstance(mask, pd.Series):
                return df[mask].reset_index(drop=True)
            return df.iloc[list(mask)].reset_index(drop=True)

        return SplitPlan(
            "custom",
            fractions,
            _take("train"),
            _take("test"),
            _take("oos"),
            ["User-supplied custom split masks."],
        )

    def _slice_by_index(self, df, idx, fractions, strategy, notes) -> SplitPlan:
        train_frac, test_frac, _ = fractions
        n = len(idx)
        n_train, n_test = int(n * train_frac), int(n * test_frac)
        tr, te, oo = idx[:n_train], idx[n_train : n_train + n_test], idx[n_train + n_test :]
        return SplitPlan(
            strategy,
            fractions,
            df.iloc[tr].reset_index(drop=True),
            df.iloc[te].reset_index(drop=True),
            df.iloc[oo].reset_index(drop=True),
            notes,
        )

    # -- evidence ---------------------------------------------------------- #
    def to_evidence(self, plan: SplitPlan, target_column: str | None = None) -> TestResult:
        n_train, n_test, n_oos = plan.sizes
        total = n_train + n_test + n_oos
        metrics: dict[str, Any] = {
            "strategy": plan.strategy,
            "n_train": n_train,
            "n_test": n_test,
            "n_oos": n_oos,
            "train_pct": round(100 * n_train / total, 2) if total else 0,
            "test_pct": round(100 * n_test / total, 2) if total else 0,
            "oos_pct": round(100 * n_oos / total, 2) if total else 0,
        }
        status = Status.PASS
        if n_oos == 0 or n_test == 0:
            status = Status.WARN
        if target_column and target_column in plan.train.columns:
            for name, frame in (("train", plan.train), ("test", plan.test), ("oos", plan.oos)):
                if len(frame):
                    metrics[f"{name}_pos_rate"] = round(float(frame[target_column].mean()), 4)
        return TestResult(
            test_id="split.plan",
            test_name="Data split plan",
            status=status,
            metrics=metrics,
            interpretation=(
                f"{plan.strategy} split: {n_train}/{n_test}/{n_oos} "
                f"(train/test/OOS). " + " ".join(plan.notes)
            ),
            limitations=[
                "OOS is held out explicitly; metrics on it estimate generalization.",
            ],
        )
