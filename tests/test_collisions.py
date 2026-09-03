"""Tests for cross-agent collision detection, human adjudication, and seal commitment."""

from __future__ import annotations

from pathlib import Path

from start.attestation.seal import build_seal, verify_seal
from start.consensus import (
    ADJUDICATOR_CONSTANT,
    COLLISION_RULES,
    AdjudicationDecision,
    AdjudicationRecord,
    adjudicate_collisions_interactive,
    detect_collisions,
)


def test_rule_registry_and_detectability() -> None:
    """Every rule must declare detectability; gaps must give reasons."""
    assert len(COLLISION_RULES) == 5
    implemented = [r for r in COLLISION_RULES if r.detectability == "implemented"]
    not_yet = [r for r in COLLISION_RULES if r.detectability == "not_detectable_yet"]

    assert len(implemented) == 4
    assert len(not_yet) == 1
    assert not_yet[0].name == "inconsistent_recommendations"
    assert "claim-graph" in not_yet[0].gap_reason


def test_adjudicator_is_strictly_human() -> None:
    """The adjudicator must always be the constant 'human'."""
    assert ADJUDICATOR_CONSTANT == "human"

    rec = AdjudicationRecord(
        collision_id="c-1",
        rule_name="architecture_contradiction",
        decision="uphold_a",
        reviewer="risk_officer",
        rationale="Tabular MLP is standard for this baseline.",
    )
    ev = rec.as_evidence_record()
    assert ev["adjudicator"] == "human"


def test_no_non_human_adjudicator_in_source_tree() -> None:
    """Verify that no code in src/ sets adjudicator to anything other than human."""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    for py_file in src_dir.rglob("*.py"):
        content = py_file.read_text(encoding="utf-8")
        if '"adjudicator":' in content or "'adjudicator':" in content:
            assert (
                '"adjudicator": "human"' in content
                or "'adjudicator': 'human'" in content
                or "self.adjudicator" in content
                or "ADJUDICATOR_CONSTANT" in content
            )


def test_architecture_contradiction_detection() -> None:
    agent_outputs = {
        "ArchitectureReviewAgent": {"recommended_family": "lstm"},
        "ValidationPlannerAgent": {"expected_modality": "tabular"},
    }
    cols = detect_collisions(evidence_records=[], agent_outputs=agent_outputs)
    assert len(cols) == 1
    assert cols[0].rule_name == "architecture_contradiction"


def test_cascading_failure_detection() -> None:
    evidence = [
        {"evidence_id": "EV-LEAK-1", "test_id": "preprocessing.leakage", "status": "warn"},
        {
            "evidence_id": "EV-MET-1",
            "test_id": "supervised.cohort_metrics",
            "status": "pass",
            "metrics": {"test_auc": 0.89},
        },
    ]
    cols = detect_collisions(evidence_records=evidence)
    assert len(cols) == 1
    assert cols[0].rule_name == "cascading_failure"
    assert "EV-LEAK-1" in cols[0].evidence_citations


def test_interactive_adjudication_five_paths() -> None:
    agent_outputs = {
        "ArchitectureReviewAgent": {"recommended_family": "lstm"},
        "ValidationPlannerAgent": {"expected_modality": "tabular"},
    }
    cols = detect_collisions(evidence_records=[], agent_outputs=agent_outputs)
    assert len(cols) == 1

    # Path 1: Uphold A
    inputs_1 = iter(["1", "Overriding with Tabular MLP for credit data."])
    records, ok = adjudicate_collisions_interactive(
        cols, reviewer="MRO_1", input_func=lambda _: next(inputs_1), output_func=lambda _: None
    )
    assert ok is True
    assert records[0].decision == AdjudicationDecision.UPHOLD_A

    # Path 2: Uphold B
    inputs_2 = iter(["2", "Strictly enforcing tabular validation requirements."])
    records, ok = adjudicate_collisions_interactive(
        cols, reviewer="MRO_2", input_func=lambda _: next(inputs_2), output_func=lambda _: None
    )
    assert ok is True
    assert records[0].decision == AdjudicationDecision.UPHOLD_B

    # Path 3: Reconcile Partial
    inputs_3 = iter(["3", "Both valid in transition; documenting dual evaluation."])
    records, ok = adjudicate_collisions_interactive(
        cols, reviewer="MRO_3", input_func=lambda _: next(inputs_3), output_func=lambda _: None
    )
    assert ok is True
    assert records[0].decision == AdjudicationDecision.RECONCILE_PARTIAL

    # Path 4: Defer (Blocks signoff)
    inputs_4 = iter(["4", "Escalating to Senior Risk Committee."])
    records, ok = adjudicate_collisions_interactive(
        cols, reviewer="MRO_4", input_func=lambda _: next(inputs_4), output_func=lambda _: None
    )
    assert ok is False
    assert records[0].decision == AdjudicationDecision.DEFER

    # Path 5: Reject (Halts review)
    inputs_5 = iter(["5", "Flawed model premise; rejecting run."])
    records, ok = adjudicate_collisions_interactive(
        cols, reviewer="MRO_5", input_func=lambda _: next(inputs_5), output_func=lambda _: None
    )
    assert ok is False
    assert records[0].decision == AdjudicationDecision.REJECT_RUN


def test_non_interactive_refuses_auto_resolution() -> None:
    agent_outputs = {
        "ArchitectureReviewAgent": {"recommended_family": "lstm"},
        "ValidationPlannerAgent": {"expected_modality": "tabular"},
    }
    cols = detect_collisions(evidence_records=[], agent_outputs=agent_outputs)
    records, ok = adjudicate_collisions_interactive(cols, non_interactive=True, output_func=lambda _: None)
    assert ok is False
    assert len(records) == 0


def test_adjudications_commit_as_eighth_seal_leaf() -> None:
    rec = AdjudicationRecord(
        collision_id="c-99",
        rule_name="architecture_contradiction",
        decision="uphold_a",
        reviewer="MRO_AUDIT",
        rationale="Resolved after consultation.",
    )
    seal = build_seal(
        review_id="R-COLLISION-SEAL",
        plan={"scope": "adjudicated"},
        adjudications=[rec.as_evidence_record()],
    )
    manifest = seal.manifest()
    assert len(manifest["leaves"]) == 9
    assert manifest["leaves"][7]["name"] == "adjudications"
    assert manifest["leaves"][8]["name"] == "execution_path"

    res = verify_seal(manifest, seal.seal_string())
    assert res["verified"] is True
