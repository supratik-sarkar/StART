"""Fitting-scope audit.

The check that row-shuffling cannot make
----------------------------------------

The obvious way to test leakage isolation is to shuffle the evaluation rows and confirm
the fitted state is unchanged. It is also useless. Consider the classic defect::

    scaler.fit(pd.concat([train, test]))

Shuffle the test rows and refit: the mean and standard deviation are **identical**,
because a mean does not depend on row order. The check passes. The pipeline is leaking.

Row order is not the leakage channel — *values* are. So the principal check here
perturbs evaluation **values**, not their order, and asserts the learned train-side
state does not move. A pipeline that touched the evaluation data cannot survive that.

The four checks
---------------

``Check 1 — train-only reproduction``
    Fit through the normal pipeline; fit again on train alone. The states must agree.
    Discrete state (level sets, kept columns, bin edges as counts) compares exactly;
    floating state compares at a declared tolerance, because two honest fits can differ
    in the last bit through BLAS non-determinism and treating that as a violation would
    get the audit deleted.

``Check 2 — evaluation-influence perturbation``  ← the one that matters
    Refit with a materially perturbed evaluation cohort. Train-side state must be
    invariant. Catches ``fit(concat(train, test))``, which Check 1 alone can miss when
    the pipeline is *consistently* wrong in both runs.

``Check 3 — target-derived training representation``
    For target encoding and WoE, verify the train-side value for a row was not produced
    by a mapping that included that row's own target. Perturb one row's target and
    confirm its own encoded value is unchanged — under out-of-fold encoding it must be,
    because that row's fold never saw its target.

``Check 4 — future influence``
    For temporal and aggregation features, perturb observations that occur *later* and
    confirm earlier rows' features are unchanged. A rolling window that forgot
    ``closed="left"`` fails here.

A violation names the step and the check. "Something leaked" is not actionable;
"``scaling`` failed Check 2: train-side state changed when evaluation values were
perturbed" is.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from start.tests.feature_engineering.execution import (
    FittingScope,
    TransformExecutionResult,
    canonical_state_hash,
)

__all__ = ["AuditFinding", "FittingScopeAudit", "audit_executor", "STATE_ATOL"]

#: Tolerance for comparing floating fitted state between two independent fits.
#: Far tighter than any real leakage signal, far looser than BLAS last-bit noise.
STATE_ATOL = 1e-9


@dataclass
class AuditFinding:
    step: str
    check: str
    passed: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "check": self.check,
            "passed": self.passed,
            "detail": self.detail,
            **self.evidence,
        }


@dataclass
class FittingScopeAudit:
    findings: list[AuditFinding] = field(default_factory=list)

    @property
    def violations(self) -> list[AuditFinding]:
        return [f for f in self.findings if not f.passed]

    @property
    def passed(self) -> bool:
        return not self.violations

    def metrics(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_checks": len(self.findings),
            "n_violations": len(self.violations),
            "checks_run": ", ".join(sorted({f.check for f in self.findings})),
            "violating_steps": ", ".join(sorted({f.step for f in self.violations})),
        }
        for finding in self.findings:
            out[f"{finding.check}.{finding.step}"] = "pass" if finding.passed else "VIOLATION"
        return out

    def summary(self) -> str:
        if self.passed:
            return (
                f"All {len(self.findings)} fitting-scope check(s) passed across "
                f"{len({f.step for f in self.findings})} transformation(s)."
            )
        first = self.violations[0]
        return (
            f"{len(self.violations)} fitting-scope violation(s). "
            f"'{first.step}' failed {first.check}: {first.detail}"
        )


def _states_agree(a: Any, b: Any, atol: float = STATE_ATOL) -> tuple[bool, str]:
    """Compare fitted state: exact for discrete, tolerant for floats."""
    if canonical_state_hash(a) == canonical_state_hash(b):
        return True, ""

    def walk(x: Any, y: Any, path: str) -> str:
        if isinstance(x, dict) and isinstance(y, dict):
            if set(x) != set(y):
                only = sorted(set(x) ^ set(y))
                return f"{path}: key sets differ ({', '.join(map(str, only[:5]))})"
            for key in sorted(x, key=str):
                found = walk(x[key], y[key], f"{path}.{key}")
                if found:
                    return found
            return ""
        if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
            if len(x) != len(y):
                return f"{path}: lengths differ ({len(x)} vs {len(y)})"
            for index, (xi, yi) in enumerate(zip(x, y, strict=False)):
                found = walk(xi, yi, f"{path}[{index}]")
                if found:
                    return found
            return ""
        if isinstance(x, (float, np.floating)) and isinstance(y, (float, np.floating)):
            fx, fy = float(x), float(y)
            if math.isnan(fx) and math.isnan(fy):
                return ""
            scale = max(1.0, abs(fx), abs(fy))
            if abs(fx - fy) / scale <= atol:
                return ""
            return f"{path}: {fx:.10g} vs {fy:.10g}"
        if x != y:
            return f"{path}: {x!r} vs {y!r}"
        return ""

    difference = walk(a, b, "state")
    return (not difference), difference


def _perturb_frame(frame: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Materially change evaluation VALUES while preserving schema and row count.

    Values, not order. Row order is not the leakage channel — a mean, a quantile and a
    level frequency are all order-invariant, so shuffling proves nothing.
    """
    rng = np.random.default_rng(seed)
    out = frame.copy()
    n = len(out)
    # Alternate the sign of the numeric perturbation. Shifting every value in the SAME
    # direction leaves rank-based statistics untouched when the evaluation cohort
    # already sits entirely above or below the training range: a median fitted on
    # train + test would be identical before and after, and a leaky median-imputer
    # would escape this check. Straddling the training range moves the mean, the
    # median, every quantile, the min and the max together.
    sign = np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    for column in out.columns:
        series = out[column]
        if pd.api.types.is_numeric_dtype(series):
            out[column] = series.astype(float) * 1000.0 + sign * 987654.0
        elif pd.api.types.is_datetime64_any_dtype(series):
            offsets = pd.to_timedelta(sign * 3650, unit="D")
            out[column] = series + offsets
        else:
            out[column] = [f"__PERTURBED_{rng.integers(0, 3)}__" for _ in range(n)]
    return out


def audit_executor(
    executor: Callable[..., TransformExecutionResult],
    train: pd.DataFrame,
    test: pd.DataFrame | None,
    oos: pd.DataFrame | None = None,
    *,
    step: str = "",
    target_column: str | None = None,
    timestamp_column: str | None = None,
    **kwargs: Any,
) -> FittingScopeAudit:
    """Run every applicable check against one transformation executor.

    ``target_column`` and ``timestamp_column`` are named here because the audit needs
    them to decide which checks apply, but they are also **forwarded** to the executor:
    swallowing them would call the executor without arguments it requires and produce a
    spurious execution failure rather than a real finding.
    """
    audit = FittingScopeAudit()

    forwarded = dict(kwargs)
    if target_column is not None:
        forwarded.setdefault("target_column", target_column)
    if timestamp_column is not None:
        forwarded.setdefault("timestamp_column", timestamp_column)
    kwargs = forwarded

    try:
        baseline = executor(train, test, oos, **kwargs)
    except Exception as exc:
        audit.findings.append(
            AuditFinding(
                step or "unknown", "check_0_execution", False, f"executor raised {type(exc).__name__}: {exc}"
            )
        )
        return audit

    name = step or baseline.step
    if baseline.fitting_scope == FittingScope.STATELESS and not baseline.fitted_state:
        audit.findings.append(
            AuditFinding(
                name,
                "check_1_train_only_reproduction",
                True,
                "stateless transformation: no fitted state to isolate",
            )
        )
    else:
        # ---- Check 1 -----------------------------------------------------
        try:
            train_only = executor(train, None, None, **kwargs)
            agree, difference = _states_agree(baseline.fitted_state, train_only.fitted_state)
            audit.findings.append(
                AuditFinding(
                    name,
                    "check_1_train_only_reproduction",
                    agree,
                    "fitted state reproduces from train alone"
                    if agree
                    else f"state differs when fitted on train alone — {difference}",
                    {
                        "state_hash_pipeline": baseline.state_hash(),
                        "state_hash_train_only": train_only.state_hash(),
                    },
                )
            )
        except Exception as exc:
            audit.findings.append(
                AuditFinding(
                    name,
                    "check_1_train_only_reproduction",
                    False,
                    f"train-only refit raised {type(exc).__name__}: {exc}",
                )
            )

        # ---- Check 2 — the one row-shuffling cannot make -----------------
        if test is not None:
            try:
                perturbed = executor(train, _perturb_frame(test), oos, **kwargs)
                agree, difference = _states_agree(baseline.fitted_state, perturbed.fitted_state)
                audit.findings.append(
                    AuditFinding(
                        name,
                        "check_2_evaluation_influence",
                        agree,
                        "train-side state is invariant to evaluation values"
                        if agree
                        else (
                            "train-side state CHANGED when evaluation values were "
                            f"perturbed — the fit saw evaluation data ({difference})"
                        ),
                        {
                            "state_hash_baseline": baseline.state_hash(),
                            "state_hash_perturbed_eval": perturbed.state_hash(),
                        },
                    )
                )
            except Exception as exc:
                audit.findings.append(
                    AuditFinding(
                        name,
                        "check_2_evaluation_influence",
                        False,
                        f"perturbed refit raised {type(exc).__name__}: {exc}",
                    )
                )

    # ---- Check 3 — target-derived training representation ---------------
    if baseline.fitting_scope == FittingScope.TRAIN_FOLDS and target_column:
        try:
            flipped = train.copy()
            y = pd.to_numeric(flipped[target_column], errors="coerce")
            row = 0
            flipped.loc[flipped.index[row], target_column] = 0 if float(y.iloc[row]) != 0 else 1
            after = executor(
                flipped,
                test,
                oos,
                target_column=target_column,
                **{k: v for k, v in kwargs.items() if k != "target_column"},
            )

            changed_columns: list[str] = []
            for column in baseline.affected_features:
                if column not in after.transformed_train.columns:
                    continue
                a = baseline.transformed_train[column].iloc[row]
                b = after.transformed_train[column].iloc[row]
                try:
                    if abs(float(a) - float(b)) > 1e-12:
                        changed_columns.append(column)
                except (TypeError, ValueError):
                    if a != b:
                        changed_columns.append(column)

            passed = not changed_columns
            audit.findings.append(
                AuditFinding(
                    name,
                    "check_3_out_of_fold_target_encoding",
                    passed,
                    "a row's own encoded value does not depend on its own target"
                    if passed
                    else "a row's encoded value CHANGED when its own target was flipped — "
                    f"the encoding is not out-of-fold ({', '.join(changed_columns[:5])})",
                    {"row_inspected": row, "columns_changed": ", ".join(changed_columns[:5])},
                )
            )
        except Exception as exc:
            audit.findings.append(
                AuditFinding(
                    name,
                    "check_3_out_of_fold_target_encoding",
                    False,
                    f"target-flip refit raised {type(exc).__name__}: {exc}",
                )
            )

    # ---- Check 4 — future influence -------------------------------------
    if timestamp_column and timestamp_column in train.columns:
        try:
            ts = pd.to_datetime(train[timestamp_column], errors="coerce")
            order = np.argsort(ts.to_numpy())
            cutoff = len(order) // 2
            early, late = order[:cutoff], order[cutoff:]

            future_changed = train.copy()
            numeric = [
                c for c in future_changed.select_dtypes(include=[np.number]).columns if c != target_column
            ]
            for column in numeric:
                future_changed.iloc[late, future_changed.columns.get_loc(column)] = (
                    future_changed.iloc[late][column].astype(float) * 500.0 + 999.0
                )

            after = executor(future_changed, test, oos, **kwargs)
            generated = [c for c in baseline.transformed_train.columns if c not in train.columns]
            changed_columns: list[str] = []
            for column in generated:
                if column not in after.transformed_train.columns:
                    continue
                a = pd.to_numeric(baseline.transformed_train[column].iloc[early], errors="coerce").to_numpy(
                    dtype=float
                )
                b = pd.to_numeric(after.transformed_train[column].iloc[early], errors="coerce").to_numpy(
                    dtype=float
                )
                both_nan = np.isnan(a) & np.isnan(b)
                if not np.allclose(
                    np.where(both_nan, 0.0, a),
                    np.where(both_nan, 0.0, b),
                    rtol=1e-9,
                    atol=1e-9,
                    equal_nan=True,
                ):
                    changed_columns.append(column)

            passed = not changed_columns
            audit.findings.append(
                AuditFinding(
                    name,
                    "check_4_future_influence",
                    passed,
                    "historical feature values are invariant to future observations"
                    if passed
                    else "historical feature values CHANGED when future observations were "
                    f"perturbed — the transformation is not causal "
                    f"({', '.join(changed_columns[:5])})",
                    {"n_early_rows": int(len(early)), "columns_changed": ", ".join(changed_columns[:5])},
                )
            )
        except Exception as exc:
            audit.findings.append(
                AuditFinding(
                    name,
                    "check_4_future_influence",
                    False,
                    f"future-perturbation refit raised {type(exc).__name__}: {exc}",
                )
            )

    return audit
