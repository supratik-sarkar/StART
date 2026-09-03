"""MRM-grade signoff evaluation (v2.3.0 #11).

Replaces "all evidence passed" with a model-risk decision that weighs
performance, generalization, calibration, sensitivity / feature dependence,
drift, reviewer overrides, and outstanding reviewer challenges. Produces an
explicit verdict — READY / READY WITH CONDITIONS / NOT READY — with a rationale
that cites the actual evidence behind each consideration.

The evaluator is deterministic and evidence-driven: every factor that moves the
verdict is backed by a value from the EvidenceStore or the ReviewSession, never
a vague assertion. A model with excessive feature dependence (high sensitivity
drift) cannot silently receive READY (#8 signoff integration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

READY = "READY"
CONDITIONAL = "READY WITH CONDITIONS"
NOT_READY = "NOT READY"


@dataclass
class SignoffFactor:
    name: str
    status: str  # ok | concern | blocker | unknown
    detail: str
    evidence: str = ""  # provenance / value behind the judgment

    def to_dict(self) -> dict[str, Any]:
        return {"factor": self.name, "status": self.status, "detail": self.detail, "evidence": self.evidence}


@dataclass
class SignoffDecision:
    verdict: str
    factors: list[SignoffFactor] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "rationale": self.rationale,
            "factors": [f.to_dict() for f in self.factors],
        }


# thresholds (public, model-agnostic defaults)
_MIN_PRIMARY = 0.65  # below this primary metric -> blocker
_WEAK_PRIMARY = 0.75  # below this -> concern
_MAX_GEN_GAP = 0.10  # train-OOS gap above this -> concern
_MAX_ECE = 0.10  # calibration error above this -> concern
_MAX_SENS_DRIFT = 0.15  # sensitivity drift above this -> concern (feature dependence)
_HIGH_SENS_DRIFT = 0.30  # above this -> blocker


def evaluate_signoff(
    store: Any,  # EvidenceStore
    session: Any = None,  # ReviewSession
    *,
    primary_metric: str = "auc_roc",
    max_ece: float | None = None,  # configurable calibration threshold (#9)
) -> SignoffDecision:
    factors: list[SignoffFactor] = []
    concerns = 0
    blockers = 0
    ece_threshold = _MAX_ECE if max_ece is None else max_ece

    # --- performance (OOS primary metric) ---
    oos = (store.cohort_metrics or {}).get("oos") or (store.cohort_metrics or {}).get("test")
    lower_is_better = primary_metric.lower() in ("rmse", "mae", "mse", "mape", "brier_score", "ece")
    if oos and primary_metric in oos:
        val = oos[primary_metric]
        if lower_is_better:
            r2_val = oos.get("r2")
            if r2_val is not None:
                if r2_val < 0.1:
                    factors.append(
                        SignoffFactor(
                            "Performance",
                            "blocker",
                            f"OOS r2={r2_val:.4f} is below minimum 0.1 (poor regression fit)",
                            "cohort_metrics.oos.r2",
                        )
                    )
                    blockers += 1
                elif r2_val < 0.4:
                    factors.append(
                        SignoffFactor(
                            "Performance",
                            "concern",
                            f"OOS r2={r2_val:.4f} is weak (<0.4)",
                            "cohort_metrics.oos.r2",
                        )
                    )
                    concerns += 1
                else:
                    factors.append(
                        SignoffFactor(
                            "Performance",
                            "ok",
                            f"OOS {primary_metric}={val:.4f} (r2={r2_val:.4f})",
                            "cohort_metrics.oos",
                        )
                    )
            else:
                factors.append(
                    SignoffFactor(
                        "Performance", "ok", f"OOS {primary_metric}={val:.4f}", "cohort_metrics.oos"
                    )
                )
        else:
            if val < _MIN_PRIMARY:
                factors.append(
                    SignoffFactor(
                        "Performance",
                        "blocker",
                        f"OOS {primary_metric}={val:.4f} below minimum {_MIN_PRIMARY}",
                        "cohort_metrics.oos",
                    )
                )
                blockers += 1
            elif val < _WEAK_PRIMARY:
                factors.append(
                    SignoffFactor(
                        "Performance",
                        "concern",
                        f"OOS {primary_metric}={val:.4f} is weak (<{_WEAK_PRIMARY})",
                        "cohort_metrics.oos",
                    )
                )
                concerns += 1
            else:
                factors.append(
                    SignoffFactor(
                        "Performance", "ok", f"OOS {primary_metric}={val:.4f}", "cohort_metrics.oos"
                    )
                )
    else:
        factors.append(SignoffFactor("Performance", "blocker", "No OOS metric available.", ""))
        blockers += 1

    # --- generalization gap (train vs OOS) ---
    cm = store.cohort_metrics or {}
    if "train" in cm and ("oos" in cm or "test" in cm):
        hold = cm.get("oos") or cm.get("test")
        use_r2_gap = "r2" in cm["train"] and "r2" in hold
        metric_for_gap = "r2" if use_r2_gap else primary_metric
        if metric_for_gap in cm["train"] and metric_for_gap in hold:
            if metric_for_gap == "r2" or not lower_is_better:
                gap = cm["train"][metric_for_gap] - hold[metric_for_gap]
            else:
                gap = hold[metric_for_gap] - cm["train"][metric_for_gap]

            is_scale_dependent = metric_for_gap != "r2" and lower_is_better
            if not is_scale_dependent and gap > _MAX_GEN_GAP:
                factors.append(
                    SignoffFactor(
                        "Generalization",
                        "concern",
                        f"train-OOS {metric_for_gap} gap {gap:+.4f} exceeds {_MAX_GEN_GAP}",
                        "cohort_metrics",
                    )
                )
                concerns += 1
            else:
                factors.append(
                    SignoffFactor(
                        "Generalization", "ok", f"train-OOS {metric_for_gap} gap {gap:+.4f}", "cohort_metrics"
                    )
                )

    # --- calibration (ECE) — threshold is configurable, not a universal law ---
    if oos and "ece" in oos:
        ece = oos["ece"]
        if ece > ece_threshold:
            factors.append(
                SignoffFactor(
                    "Calibration",
                    "concern",
                    f"OOS ECE={ece:.4f} exceeds the configured threshold "
                    f"{ece_threshold:.3f} (adjustable per model/risk appetite)",
                    "cohort_metrics.oos.ece",
                )
            )
            concerns += 1
        else:
            factors.append(
                SignoffFactor(
                    "Calibration",
                    "ok",
                    f"OOS ECE={ece:.4f} within the configured threshold {ece_threshold:.3f}",
                    "cohort_metrics.oos.ece",
                )
            )

    # --- sensitivity / feature dependence ---
    if store.max_abs_drift is not None:
        drift = store.max_abs_drift
        feat = store.most_sensitive_feature or "a feature"
        if drift > _HIGH_SENS_DRIFT:
            factors.append(
                SignoffFactor(
                    "Feature dependence",
                    "blocker",
                    f"excessive sensitivity: {feat} drives drift {drift:.4f} (>{_HIGH_SENS_DRIFT})",
                    "sensitivity_analysis",
                )
            )
            blockers += 1
        elif drift > _MAX_SENS_DRIFT:
            factors.append(
                SignoffFactor(
                    "Feature dependence",
                    "concern",
                    f"{feat} drives drift {drift:.4f} (>{_MAX_SENS_DRIFT})",
                    "sensitivity_analysis",
                )
            )
            concerns += 1
        else:
            factors.append(
                SignoffFactor(
                    "Feature dependence",
                    "ok",
                    f"max drift {drift:.4f} (most sensitive: {feat})",
                    "sensitivity_analysis",
                )
            )
    else:
        factors.append(
            SignoffFactor("Feature dependence", "unknown", "No sensitivity analysis available.", "")
        )

    # --- benchmark against baselines ---
    if getattr(store, "benchmark", None):
        b = store.benchmark
        b_status = b.get("status", "ok")
        b_verdict = b.get("verdict", "")
        factors.append(
            SignoffFactor("Trivial-baseline benchmark", b_status, b_verdict, "benchmark.decision_stump")
        )
        if b_status == "blocker":
            blockers += 1
        elif b_status == "concern":
            concerns += 1

    # --- reviewer challenges (open/conceded block, resolved ok) ---
    if session is not None:
        from start.governance.challenge_disposition import (
            ChallengeDisposition,
            challenge_factor,
            classify_challenge,
            classify_override,
            override_factor,
        )

        verdicts = [classify_challenge(c) for c in session.challenges]
        status, detail, evidence = challenge_factor(verdicts)
        factors.append(SignoffFactor("Reviewer challenges", status, detail, evidence))
        if status == "blocker":
            blockers += 1
        elif status == "concern":
            concerns += 1

        conceded_keys = {v.agent for v in verdicts if v.disposition is ChallengeDisposition.CONCEDED}
        checkpoint_agent_map = {
            "architecture": "ArchitectureReviewAgent",
            "metric_priority": "HyperparameterTuningAgent",
            "target": "DatasetDiscoveryAgent",
            "outliers": "FeatureEngineeringAgent",
            "imputation": "FeatureEngineeringAgent",
            "encoding": "FeatureEngineeringAgent",
            "scaling": "FeatureEngineeringAgent",
            "imbalance": "FeatureEngineeringAgent",
        }
        conceded_all = set(conceded_keys)
        for ch in session.challenges:
            if getattr(ch, "conceded", False) or getattr(ch, "changes_disposition", False):
                conceded_all.add(ch.agent)
                if getattr(ch, "checkpoint", None):
                    conceded_all.add(ch.checkpoint)
                for cp, ag in checkpoint_agent_map.items():
                    if ag == ch.agent:
                        conceded_all.add(cp)

        ov = [classify_override(d, conceded_keys=conceded_all) for d in session.overrides()]
        status, detail, evidence = override_factor(ov)
        factors.append(SignoffFactor("Reviewer overrides", status, detail, evidence))
        if status == "concern":
            concerns += 1
        # "informational" contributes to NEITHER blockers nor concerns — that is A2.

    # --- verdict ---
    if blockers:
        verdict = NOT_READY
    elif concerns:
        verdict = CONDITIONAL
    else:
        verdict = READY

    ok_n = sum(1 for f in factors if f.status in ("ok", "informational"))
    rationale = (
        f"{verdict}: {blockers} blocker(s), {concerns} concern(s), {ok_n} factor(s) "
        f"clear across performance, generalization, calibration, feature "
        f"dependence, and reviewer activity. "
    )
    if verdict == NOT_READY:
        rationale += "Blocking issues must be resolved before sign-off."
    elif verdict == CONDITIONAL:
        rationale += "Sign-off is conditional on addressing the listed concerns."
    else:
        rationale += "No blocking issues or concerns identified."

    return SignoffDecision(verdict=verdict, factors=factors, rationale=rationale)


def render_signoff_rich(decision: SignoffDecision) -> Any:
    """Render the MRM signoff as a Rich table + verdict panel (#9)."""
    from rich.panel import Panel
    from rich.table import Table

    table = Table(show_header=True, header_style="bold")
    table.add_column("Factor")
    table.add_column("Status")
    table.add_column("Detail")
    table.add_column("Evidence")
    style = {
        "ok": "green",
        "concern": "yellow",
        "blocker": "red",
        "unknown": "dim",
        "informational": "cyan",
    }
    for f in decision.factors:
        table.add_row(f.name, f"[{style.get(f.status, 'white')}]{f.status}[/]", f.detail, f.evidence)

    vstyle = {"READY": "green", "READY WITH CONDITIONS": "yellow", "NOT READY": "red"}.get(
        decision.verdict, "white"
    )
    panel = Panel(
        f"[bold {vstyle}]{decision.verdict}[/]\n{decision.rationale}",
        title="[bold]GovernanceSignoffAgent — MRM decision[/bold]",
        border_style=vstyle,
        title_align="left",
    )
    return table, panel


def render_signoff_markdown(decision: SignoffDecision) -> str:
    lines = [
        "### GovernanceSignoffAgent — MRM decision",
        "",
        f"**Verdict: {decision.verdict}**",
        "",
        decision.rationale,
        "",
        "| Factor | Status | Detail | Evidence |",
        "| --- | --- | --- | --- |",
    ]
    for f in decision.factors:
        lines.append(f"| {f.name} | {f.status} | {f.detail} | {f.evidence} |")
    return "\n".join(lines) + "\n"
