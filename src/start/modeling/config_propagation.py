"""Configuration propagation audit.

The defect this exists to catch
-------------------------------

A reviewer enabled class-weight balancing in the wizard. The agent recommended
it. The reviewer accepted it. The model trained without it, predicted zero
positives out of sample, and the review sealed anyway.

Nothing failed. No exception, no warning, no diverging test. A setting the
reviewer chose simply did not arrive, and the only way anyone noticed was a
confusion matrix with a suspicious zero in it.

That is the worst defect class this product can have. A governance tool whose
own controls silently do not apply is worse than no tool, because it manufactures
false assurance: the transcript records a decision that had no effect.

Fixing one dropped setting is not the fix. Every setting can drop the same way,
and each one is invisible in exactly the same manner. So this module makes
propagation *checkable*: declare where each setting must arrive, then assert it
arrived.

How it works
------------

Each :class:`PropagationRule` names a setting, the config attribute that carries
it, and one or more **observation points** — a dotted path to something that
should hold the value by the time the model is built. The audit resolves each
observation against a captured run context and reports any that are missing or
that disagree with the configured value.

This is deliberately a *runtime* audit rather than a static one. Static analysis
cannot see a value dropped by a ``**kwargs`` splat, a dataclass default silently
overriding a passed argument, or a branch that rebuilds options from scratch —
which are the three ways settings actually get lost in this codebase.

Standard library only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "PropagationRule",
    "PropagationFinding",
    "PropagationReport",
    "RULES",
    "audit_propagation",
    "resolve_path",
]


def resolve_path(root: Any, path: str) -> tuple[bool, Any]:
    """Resolve a dotted path against attributes, mappings and indices.

    Returns ``(found, value)``. ``found`` is False when any segment is absent —
    which is itself a finding, because a setting cannot have arrived somewhere
    that does not exist.
    """
    current = root
    for segment in path.split("."):
        if not segment:
            continue
        if segment.endswith("]") and "[" in segment:
            name, _, index = segment.partition("[")
            index = index.rstrip("]").strip("'\"")
            if name:
                found, current = resolve_path(current, name)
                if not found:
                    return False, None
            try:
                current = current[int(index)] if index.lstrip("-").isdigit() else current[index]
            except Exception:
                return False, None
            continue

        if isinstance(current, dict):
            if segment not in current:
                return False, None
            current = current[segment]
        else:
            if not hasattr(current, segment):
                return False, None
            current = getattr(current, segment)
    return True, current


@dataclass(frozen=True)
class PropagationRule:
    """One reviewer setting and where it must be observable."""

    setting: str
    #: Attribute on the review config that carries the reviewer's choice.
    config_attr: str
    #: Dotted paths, resolved against the captured context, that must hold it.
    observation_points: tuple[str, ...]
    #: Why it matters — surfaced in the report so a failure explains itself.
    consequence: str
    #: Only audit when the setting is actually set to something.
    skip_when_falsy: bool = True
    #: Compare loosely (str()) rather than by identity — settings are often
    #: normalised in transit ("balanced" -> "balanced", 0.7 -> 0.7000000001).
    tolerant: bool = True
    #: Accepted renamings, keyed by the lower-cased configured value. A boolean
    #: ``stratify=True`` legitimately arrives downstream as the strategy string
    #: ``"stratified"``; without this the audit would report a false mismatch,
    #: and a noisy audit is an audit people switch off.
    equivalences: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass
class PropagationFinding:
    setting: str
    status: str  # propagated | missing | mismatch | skipped
    configured: Any = None
    observed: Any = None
    observation_point: str = ""
    consequence: str = ""

    @property
    def failed(self) -> bool:
        return self.status in {"missing", "mismatch"}


@dataclass
class PropagationReport:
    findings: list[PropagationFinding] = field(default_factory=list)

    @property
    def failures(self) -> list[PropagationFinding]:
        return [f for f in self.findings if f.failed]

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "checked": len(self.findings),
            "failed": len(self.failures),
            "findings": [
                {
                    "setting": f.setting,
                    "status": f.status,
                    "configured": _safe(f.configured),
                    "observed": _safe(f.observed),
                    "observation_point": f.observation_point,
                    "consequence": f.consequence,
                }
                for f in self.findings
            ],
        }

    def summary_lines(self) -> list[str]:
        lines = [
            f"Configuration propagation: {len(self.findings) - len(self.failures)}"
            f"/{len(self.findings)} settings verified"
        ]
        for finding in self.findings:
            if finding.status == "propagated":
                lines.append(f"  ✓ {finding.setting}")
            elif finding.status == "skipped":
                lines.append(f"  · {finding.setting} (not set)")
            elif finding.status == "missing":
                lines.append(
                    f"  ✗ {finding.setting}: configured {finding.configured!r} but "
                    f"{finding.observation_point} does not exist — {finding.consequence}"
                )
            else:
                lines.append(
                    f"  ✗ {finding.setting}: configured {finding.configured!r}, "
                    f"observed {finding.observed!r} at {finding.observation_point} — "
                    f"{finding.consequence}"
                )
        return lines


def _safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


#: The settings a reviewer can choose, and where each must arrive.
#:
#: ``class_weight`` is first because it is the one that was silently dropped.
#: The rest are here because every one of them could drop the same way, and the
#: only reason we know about class_weight is that its failure happened to be
#: visible in a confusion matrix. Most would not be.
RULES: tuple[PropagationRule, ...] = (
    PropagationRule(
        setting="class_weight",
        config_attr="class_weight",
        observation_points=(
            "estimator.class_weight",
            "fit_kwargs.sample_weight",
            "model_params.class_weight",
        ),
        consequence="the model trains unweighted on imbalanced data and can predict "
        "zero positives while the review records that balancing was applied",
    ),
    PropagationRule(
        setting="stratify",
        config_attr="stratify",
        observation_points=(
            "split_params.stratify",
            "split_plan.strategy",
            "split_plan[0].strategy",
            "split_summary.strategy",
        ),
        consequence="cohorts may not preserve the event rate, making OOS metrics incomparable to train",
        equivalences={"true": ("stratified", "stratify", "true"), "false": ("random", "false")},
    ),
    PropagationRule(
        setting="split_proportions",
        config_attr="train_prop",
        observation_points=(
            "split_params.train_prop",
            "split_plan.train_pct",
            "split_summary.train_prop",
        ),
        consequence="the review reports a split it did not perform",
    ),
    PropagationRule(
        setting="tuning_strategy",
        config_attr="tuning_strategy",
        observation_points=("tuning.strategy", "tuning_params.strategy"),
        consequence="tuning silently uses a different search than the one chosen",
    ),
    PropagationRule(
        setting="tuning_trials",
        config_attr="tuning_trials",
        observation_points=("tuning.n_trials", "tuning_params.trials"),
        consequence="a different number of trials runs than the reviewer authorised",
    ),
    PropagationRule(
        setting="validation_scheme",
        config_attr="validation_scheme",
        observation_points=("tuning.validation", "tuning_params.validation"),
        consequence="holdout is used where k-fold was chosen, or vice versa",
    ),
    PropagationRule(
        setting="explain_method",
        config_attr="explain_method",
        observation_points=("explainability.method", "explain_params.method"),
        consequence="attributions are produced by a method other than the one "
        "the reviewer selected and cited",
    ),
    PropagationRule(
        setting="metric_priority",
        config_attr="costlier_errors",
        observation_points=("tuning.primary_metric", "metric_routing.primary"),
        consequence="the model is selected on a metric the reviewer overrode — the "
        "override appears in the transcript but had no effect",
        equivalences={
            "recall": ("recall", "pr_auc", "average_precision", "false_negatives"),
            "false_negatives": ("recall", "pr_auc", "average_precision"),
            "precision": ("precision", "false_positives"),
            "false_positives": ("precision",),
            "balanced": ("auc_roc", "roc_auc", "balanced", "f1"),
        },
    ),
    PropagationRule(
        setting="architecture",
        config_attr="architecture_family",
        observation_points=("estimator.family", "model_params.family", "model_params.architecture"),
        consequence="a different model is trained than the one approved at the architecture checkpoint",
    ),
    PropagationRule(
        setting="activation",
        config_attr="activation",
        observation_points=("estimator.activation", "model_params.activation"),
        consequence="the trained network differs from the documented specification",
    ),
    PropagationRule(
        setting="seed",
        config_attr="seed",
        observation_points=("estimator.random_state", "model_params.random_state", "split_params.seed"),
        consequence="the run is not reproducible from the recorded configuration",
    ),
)


def audit_propagation(
    config: Any,
    context: dict[str, Any],
    rules: tuple[PropagationRule, ...] = RULES,
) -> PropagationReport:
    """Check that each configured setting is observable where it must be.

    ``context`` is a mapping captured during the run — for example::

        {
            "estimator": clf,
            "fit_kwargs": fit_kwargs,
            "model_params": kwargs,
            "split_plan": result.split_table,
            "tuning": tuning_summary,
        }

    A rule passes if **any** of its observation points holds the configured
    value. Points that do not exist are not failures on their own — only a rule
    with no satisfied point at all is a failure, because a codebase legitimately
    has several shapes for the same setting.
    """
    report = PropagationReport()

    for rule in rules:
        found_cfg, configured = resolve_path(config, rule.config_attr)
        if not found_cfg or (rule.skip_when_falsy and not configured):
            report.findings.append(
                PropagationFinding(
                    setting=rule.setting,
                    status="skipped",
                    configured=configured if found_cfg else None,
                    consequence=rule.consequence,
                )
            )
            continue

        matched_point = ""
        observed_any: Any = None
        any_point_existed = False

        for point in rule.observation_points:
            found, observed = resolve_path(context, point)
            if not found:
                continue
            any_point_existed = True
            observed_any = observed
            if _values_agree(configured, observed, tolerant=rule.tolerant, equivalences=rule.equivalences):
                matched_point = point
                break

        if matched_point:
            report.findings.append(
                PropagationFinding(
                    setting=rule.setting,
                    status="propagated",
                    configured=configured,
                    observed=observed_any,
                    observation_point=matched_point,
                    consequence=rule.consequence,
                )
            )
        elif any_point_existed:
            report.findings.append(
                PropagationFinding(
                    setting=rule.setting,
                    status="mismatch",
                    configured=configured,
                    observed=observed_any,
                    observation_point=", ".join(rule.observation_points),
                    consequence=rule.consequence,
                )
            )
        else:
            report.findings.append(
                PropagationFinding(
                    setting=rule.setting,
                    status="missing",
                    configured=configured,
                    observation_point=", ".join(rule.observation_points),
                    consequence=rule.consequence,
                )
            )

    return report


def _values_agree(
    configured: Any,
    observed: Any,
    *,
    tolerant: bool,
    equivalences: dict[str, tuple[str, ...]] | None = None,
) -> bool:
    if configured is observed or configured == observed:
        return True

    accepted = (equivalences or {}).get(str(configured).strip().lower())
    if accepted and isinstance(observed, str):
        if observed.strip().lower() in accepted:
            return True
        if any(token in observed.strip().lower() for token in accepted):
            return True
    # A sample-weight array standing in for class_weight="balanced": presence of
    # a non-degenerate weight vector is the evidence, not equality.
    if hasattr(observed, "__len__") and not isinstance(observed, (str, bytes, dict)):
        try:
            unique = {round(float(v), 9) for v in observed}
            return len(unique) > 1
        except Exception:
            return False
    if not tolerant:
        return False
    try:
        if isinstance(configured, float) or isinstance(observed, float):
            return abs(float(configured) - float(observed)) <= 1e-6
    except (TypeError, ValueError):
        pass
    return str(configured).strip().lower() == str(observed).strip().lower()
