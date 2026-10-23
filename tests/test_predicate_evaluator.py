"""Tests for RecursivePredicateEvaluator proving strict closure invariants:
1. One false Gate-A predicate => Gate A NOT CLOSED
2. One false Gate-B predicate => Gate B NOT CLOSED
3. Gate B cannot close if Gate A is false (prerequisite invariant)
4. Overall status is derived dynamically and cannot be hardcoded
"""

import pytest

from start.closure.predicate_evaluator import (
    RecursivePredicateEvaluator,
)


@pytest.fixture
def evaluator():
    return RecursivePredicateEvaluator()


def test_flatten_predicates(evaluator):
    nested = {
        "a": {"b": 1, "c": {"d": True}},
        "e": "hello"
    }
    flat = evaluator.flatten_predicates(nested)
    assert flat == {
        "a.b": 1,
        "a.c.d": True,
        "e": "hello"
    }


def test_one_false_gate_a_predicate_fails_gate_a(evaluator):
    """Prove: one false Gate-A predicate => Gate A NOT CLOSED."""
    # Start with ideal Gate A evidence (all matching spec)
    flat_specs = evaluator.flatten_predicates(evaluator.gate_a_spec["hard_predicates"])
    evidence = dict(flat_specs)
    
    # Verify baseline passes
    summary = evaluator.evaluate_gate_a(evidence)
    assert summary.is_closed is True
    assert summary.failed_predicates == 0

    # Mutate exactly one predicate: e.g. target_leakage = 1 (should be 0)
    evidence["integrity.target_leakage"] = 1
    summary = evaluator.evaluate_gate_a(evidence)
    
    assert summary.is_closed is False
    assert summary.failed_predicates == 1
    assert summary.failed_predicate_details[0]["predicate"] == "integrity.target_leakage"
    assert summary.failed_predicate_details[0]["required"] == 0
    assert summary.failed_predicate_details[0]["actual"] == 1


def test_one_false_gate_b_predicate_fails_gate_b(evaluator):
    """Prove: one false Gate-B predicate => Gate B NOT CLOSED."""
    flat_a_specs = evaluator.flatten_predicates(evaluator.gate_a_spec["hard_predicates"])
    summary_a = evaluator.evaluate_gate_a(dict(flat_a_specs))
    assert summary_a.is_closed is True

    flat_b_specs = evaluator.flatten_predicates(evaluator.gate_b_spec["hard_predicates"])
    evidence_b = dict(flat_b_specs)

    # Verify baseline passes
    summary_b = evaluator.evaluate_gate_b(evidence_b, gate_a_summary=summary_a)
    assert summary_b.is_closed is True
    assert summary_b.failed_predicates == 0

    # Mutate exactly one predicate: e.g. secrets_in_spans = 1 (should be 0)
    evidence_b["opentelemetry.secrets_in_spans"] = 1
    summary_b = evaluator.evaluate_gate_b(evidence_b, gate_a_summary=summary_a)

    assert summary_b.is_closed is False
    assert summary_b.failed_predicates == 1
    assert summary_b.failed_predicate_details[0]["predicate"] == "opentelemetry.secrets_in_spans"


def test_gate_b_cannot_close_if_gate_a_is_false(evaluator):
    """Prove: Gate B CANNOT close if Gate A is false, even if all Gate B predicates match."""
    # Gate A fails
    flat_a_specs = evaluator.flatten_predicates(evaluator.gate_a_spec["hard_predicates"])
    evidence_a = dict(flat_a_specs)
    evidence_a["integrity.silent_model_substitution"] = 1  # Fails Gate A
    summary_a = evaluator.evaluate_gate_a(evidence_a)
    assert summary_a.is_closed is False

    # Gate B predicates all match required values
    flat_b_specs = evaluator.flatten_predicates(evaluator.gate_b_spec["hard_predicates"])
    evidence_b = dict(flat_b_specs)

    # Evaluate Gate B with failing Gate A summary
    summary_b = evaluator.evaluate_gate_b(evidence_b, gate_a_summary=summary_a)

    # Gate B MUST NOT close
    assert summary_b.is_closed is False
    assert summary_b.prerequisite_met is False
    assert "Gate A prerequisite failed" in summary_b.prerequisite_failure_reason


def test_joint_closure_dynamic_evaluation(evaluator):
    """Prove joint closure dynamic derivation."""
    flat_a = dict(evaluator.flatten_predicates(evaluator.gate_a_spec["hard_predicates"]))
    flat_b = dict(evaluator.flatten_predicates(evaluator.gate_b_spec["hard_predicates"]))

    # 1. When all pass and git_ops = 0
    res = evaluator.evaluate_closure(flat_a, flat_b, git_operations_performed=0)
    assert res["run_summary"]["final"]["status"] == "PASS"
    assert res["run_summary"]["final"]["gate_a_closed"] is True
    assert res["run_summary"]["final"]["gate_b_closed"] is True
    assert len(res["run_summary"]["final"]["remaining_blockers"]) == 0

    # 2. When git_ops > 0
    res_git = evaluator.evaluate_closure(flat_a, flat_b, git_operations_performed=1)
    assert res_git["run_summary"]["final"]["status"] == "FAIL"
    assert "mutating git operations performed" in res_git["run_summary"]["final"]["remaining_blockers"][0]
