"""Cross-agent collision detection and human adjudication layer.

Governing principle:
    "The machine detects the collision and presents the evidence.
     The human decides.
     The seal remembers who decided and why."

Under no circumstances may an LLM be called to choose between conflicting
agent conclusions. Presenting, summarising and formatting the conflict is
permitted; selecting a winner is exclusively reserved for human reviewers.

Standard library only.
"""

from __future__ import annotations

import datetime
import json
import os
import uuid
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

#: The immutable adjudicator constant. Nothing in this codebase may write any other value.
ADJUDICATOR_CONSTANT: str = "human"


@dataclass(frozen=True)
class CollisionRule:
    """Declarative specification for cross-agent conflict detection."""

    name: str
    agents: tuple[str, ...]
    severity: str  # high | medium | low
    description: str
    detectability: str  # implemented | not_detectable_yet
    gap_reason: str = ""


COLLISION_RULES: tuple[CollisionRule, ...] = (
    CollisionRule(
        name="architecture_contradiction",
        agents=("ArchitectureReviewAgent", "ValidationPlannerAgent"),
        severity="high",
        description=(
            "ArchitectureAgent recommended tabular MLP/CatBoost, but ValidationPlanner "
            "or downstream code flagged non-tabular or contradictory assumptions."
        ),
        detectability="implemented",
    ),
    CollisionRule(
        name="cascading_failure",
        agents=("FeatureEngineeringAgent", "ValidationAgent"),
        severity="high",
        description=(
            "FeatureEngineering flags severe data leakage or drift, yet ModelExecution "
            "and Validation evaluate high AUC without penalizing leakage."
        ),
        detectability="implemented",
    ),
    CollisionRule(
        name="hallucination_injection",
        agents=("EvidenceCriticAgent", "SignoffAgent"),
        severity="high",
        description=(
            "EvidenceCritic or Invariance check detects blocking quantitative "
            "divergences / unbound figures in proposed section narrative."
        ),
        detectability="implemented",
    ),
    CollisionRule(
        name="inconsistent_recommendations",
        agents=("HyperparameterTuningAgent", "OverfittingAgent"),
        severity="medium",
        description=(
            "Tuning agent recommends increasing model complexity / capacity while Overfitting "
            "agent flags overparameterization and recommends capacity reduction."
        ),
        detectability="not_detectable_yet",
        gap_reason=(
            "Requires semantic claim-graph dependency mapping between agent "
            "hyperparameter directives (v4.0.1 Mechanism A)."
        ),
    ),
    CollisionRule(
        name="missing_evidence_chain",
        agents=("ReviewPlannerAgent", "ValidationPlannerAgent"),
        severity="high",
        description=(
            "Review plan specifies mandatory test dimensions, but evidence ledger is "
            "missing corresponding test records or chain is broken."
        ),
        detectability="implemented",
    ),
)


@dataclass(frozen=True)
class Collision:
    """An instance of detected conflict between two or more agents."""

    collision_id: str
    rule_name: str
    severity: str
    agents_involved: tuple[str, ...]
    evidence_citations: tuple[str, ...]
    agent_a_position: str
    agent_b_position: str
    context: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "collision_id": self.collision_id,
            "rule_name": self.rule_name,
            "severity": self.severity,
            "agents_involved": list(self.agents_involved),
            "evidence_citations": list(self.evidence_citations),
            "agent_a_position": self.agent_a_position,
            "agent_b_position": self.agent_b_position,
            "context": dict(self.context),
        }


class AdjudicationDecision:
    UPHOLD_A = "uphold_a"
    UPHOLD_B = "uphold_b"
    RECONCILE_PARTIAL = "reconcile_partial"
    DEFER = "defer"
    REJECT_RUN = "reject_run"


@dataclass(frozen=True)
class AdjudicationRecord:
    """The immutable human verdict on a collision, committed to the seal."""

    collision_id: str
    rule_name: str
    decision: str  # uphold_a | uphold_b | reconcile_partial | defer | reject_run
    reviewer: str
    rationale: str
    adjudicator: str = ADJUDICATOR_CONSTANT
    timestamp_utc: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_evidence_record(self) -> dict[str, Any]:
        is_failing = self.decision in (AdjudicationDecision.DEFER, AdjudicationDecision.REJECT_RUN)
        return {
            "evidence_id": f"EV-ADJ-{self.collision_id[:8]}",
            "test_id": "consensus.human_adjudication",
            "test_name": f"Human Adjudication: {self.rule_name}",
            "status": "fail" if is_failing else "pass",
            "adjudicator": self.adjudicator,
            "decision": self.decision,
            "reviewer": self.reviewer,
            "rationale": self.rationale,
            "collision_id": self.collision_id,
            "timestamp_utc": self.timestamp_utc,
            "metadata": self.metadata,
        }

    def canonical_json(self) -> str:
        return json.dumps(self.as_evidence_record(), sort_keys=True, separators=(",", ":"))

    def adjudication_hash(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()


def detect_collisions(
    *,
    evidence_records: list[dict[str, Any]],
    agent_outputs: dict[str, Any] | None = None,
    attestations: list[Any] | None = None,
    plan: dict[str, Any] | None = None,
) -> list[Collision]:
    """Evaluate implemented collision rules against review evidence and agent outputs."""
    collisions: list[Collision] = []
    outputs = agent_outputs or {}

    # 1. Architecture Contradiction
    arch_out = outputs.get("ArchitectureReviewAgent") or outputs.get("architecture") or {}
    val_out = outputs.get("ValidationPlannerAgent") or outputs.get("validation_plan") or {}
    if arch_out and val_out:
        arch_family = str(arch_out.get("recommended_family", "")).lower()
        val_modal = str(val_out.get("expected_modality", "")).lower()
        if "tabular" in val_modal and arch_family in ("lstm", "simple_cnn_small", "transformer_seq"):
            collisions.append(
                Collision(
                    collision_id=str(uuid.uuid4()),
                    rule_name="architecture_contradiction",
                    severity="high",
                    agents_involved=("ArchitectureReviewAgent", "ValidationPlannerAgent"),
                    evidence_citations=(),
                    agent_a_position=(
                        f"ArchitectureReviewAgent recommended sequence/vision family '{arch_family}'."
                    ),
                    agent_b_position=(
                        f"ValidationPlannerAgent asserts dataset modality is tabular '{val_modal}'."
                    ),
                    context={"arch": arch_out, "validation": val_out},
                )
            )

    # 2. Cascading Failure (Feature leakage vs High AUC pass)
    fe_records = [
        r for r in evidence_records
        if "leakage" in r.get("test_id", "") or "drift" in r.get("test_id", "")
    ]
    metric_records = [
        r for r in evidence_records
        if "cohort_metrics" in r.get("test_id", "") or "performance" in r.get("test_id", "")
    ]

    has_leakage_warn_or_fail = any(r.get("status") in ("warn", "fail") for r in fe_records)
    high_auc_without_penalty = any(
        r.get("status") == "pass" and r.get("metrics", {}).get("test_auc", 0.0) > 0.85
        for r in metric_records
    )
    if has_leakage_warn_or_fail and high_auc_without_penalty:
        fe_ids = tuple(r.get("evidence_id", "") for r in fe_records)
        collisions.append(
            Collision(
                collision_id=str(uuid.uuid4()),
                rule_name="cascading_failure",
                severity="high",
                agents_involved=("FeatureEngineeringAgent", "ValidationAgent"),
                evidence_citations=fe_ids,
                agent_a_position=(
                    "FeatureEngineeringAgent flagged potential data leakage / temporal drift in predictors."
                ),
                agent_b_position=(
                    "ValidationAgent evaluated model metrics as PASS with high test AUC without "
                    "accounting for leakage."
                ),
                context={"fe_records": fe_records, "metric_records": metric_records},
            )
        )

    # 3. Hallucination Injection
    if attestations:
        for att in attestations:
            blocking = getattr(att, "blocking_divergences", lambda: ())()
            if blocking:
                collisions.append(
                    Collision(
                        collision_id=str(uuid.uuid4()),
                        rule_name="hallucination_injection",
                        severity="high",
                        agents_involved=("EvidenceCriticAgent", "SignoffAgent"),
                        evidence_citations=(),
                        agent_a_position=(
                            f"EvidenceCriticAgent detected {len(blocking)} blocking divergence(s) "
                            "(unbound/contradictory figures)."
                        ),
                        agent_b_position="SignoffAgent proposed narrative containing ungrounded claims.",
                        context={
                            "section": getattr(att, "section", ""),
                            "blocking": [b.as_dict() for b in blocking],
                        },
                    )
                )

    # 5. Missing Evidence Chain
    if plan and "dimensions" in plan:
        mandatory = set(plan.get("dimensions", []))
        ledger_tests = {r.get("test_id", "") for r in evidence_records}
        missing = [dim for dim in mandatory if not any(dim in t for t in ledger_tests)]
        if missing:
            collisions.append(
                Collision(
                    collision_id=str(uuid.uuid4()),
                    rule_name="missing_evidence_chain",
                    severity="high",
                    agents_involved=("ReviewPlannerAgent", "ValidationPlannerAgent"),
                    evidence_citations=(),
                    agent_a_position=f"ReviewPlannerAgent mandated dimensions: {', '.join(mandatory)}.",
                    agent_b_position=(
                        "Validation executed without emitting tests for mandatory "
                        f"dimensions: {', '.join(missing)}."
                    ),
                    context={"missing_dimensions": missing},
                )
            )

    return collisions


def adjudicate_collisions_interactive(
    collisions: list[Collision],
    *,
    non_interactive: bool = False,
    reviewer: str | None = None,
    input_func: Any = None,
    output_func: Any = None,
) -> tuple[list[AdjudicationRecord], bool]:
    """Present detected collisions to the human reviewer for explicit adjudication.

    Returns (adjudication_records, can_proceed).
    Under non-interactive mode: surfaces collisions, refuses auto-resolution, and returns ([], False).
    """
    out = output_func or print
    prompt = input_func or input

    if not collisions:
        return [], True

    if non_interactive:
        out("\n[!] Cross-agent collisions detected during review:")
        for col in collisions:
            out(f"    ⛔ {col.rule_name} (Severity: {col.severity})")
            out(f"       Agent A: {col.agent_a_position}")
            out(f"       Agent B: {col.agent_b_position}")
        out("\nNon-interactive mode cannot auto-resolve cross-agent collisions.")
        out("Re-run interactively to adjudicate:\n    start review\n")
        return [], False

    resolved: list[AdjudicationRecord] = []
    rev_name = (
        reviewer
        or os.environ.get("START_REVIEWER")
        or os.environ.get("USER")
        or "human_reviewer"
    ).strip()

    out(f"\n{'='*78}")
    out(" ⚖  ST-ART HUMAN ADJUDICATION COUNCIL")
    out(" The machine detected conflicts between agent recommendations.")
    out(" As the designated Model Risk Officer, you must choose the resolution.")
    out(f"{'='*78}\n")

    for i, col in enumerate(collisions, 1):
        rule_up = col.rule_name.upper()
        sev_up = col.severity.upper()
        header = f"\n--- Collision {i}/{len(collisions)}: [{rule_up}] (Severity: {sev_up}) ---"
        out(header)
        out(f"  • Position A ({col.agents_involved[0]}):\n      {col.agent_a_position}")
        out(f"  • Position B ({col.agents_involved[1]}):\n      {col.agent_b_position}")
        if col.evidence_citations:
            out(f"  • Cited Evidence: {', '.join(col.evidence_citations)}")

        out("\nAdjudication Options:")
        out("  [1] Uphold Agent A position")
        out("  [2] Uphold Agent B position")
        out("  [3] Both partially valid (reconcile with notes)")
        out("  [4] Defer to Senior Model Risk Committee (BLOCKS sign-off)")
        out("  [5] Reject whole run (TERMINATES review)")

        choice_map = {
            "1": AdjudicationDecision.UPHOLD_A,
            "2": AdjudicationDecision.UPHOLD_B,
            "3": AdjudicationDecision.RECONCILE_PARTIAL,
            "4": AdjudicationDecision.DEFER,
            "5": AdjudicationDecision.REJECT_RUN,
        }

        choice = ""
        while choice not in choice_map:
            raw = prompt("Enter decision [1-5]: ").strip()
            if raw in choice_map:
                choice = raw
            else:
                out("Please select a valid option between 1 and 5.")

        decision = choice_map[choice]

        rationale = ""
        while not rationale.strip():
            rationale = prompt("Enter mandatory reviewer rationale: ").strip()
            if not rationale:
                out("Rationale cannot be empty. Please explain your decision for the audit log.")

        rec = AdjudicationRecord(
            collision_id=col.collision_id,
            rule_name=col.rule_name,
            decision=decision,
            reviewer=rev_name,
            rationale=rationale,
            adjudicator=ADJUDICATOR_CONSTANT,
            metadata={"agents": list(col.agents_involved)},
        )
        resolved.append(rec)

        if decision == AdjudicationDecision.REJECT_RUN:
            out("\n[!] Run rejected by reviewer. Halting review.")
            return resolved, False

    # Check blocking conditions
    has_defer = any(r.decision == AdjudicationDecision.DEFER for r in resolved)
    if has_defer:
        out("\n[!] One or more collisions were deferred to Senior Committee. Sign-off is BLOCKED.")
        return resolved, False

    return resolved, True
