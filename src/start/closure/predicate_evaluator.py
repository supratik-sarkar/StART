"""Dynamic Predicate Evaluator for StART Gate A and Gate B Closure.

Recursively evaluates all hard predicates defined in:
- START_GATE_AB_CLOSURE_PACK/02_GATE_A_ACCEPTANCE.json
- START_GATE_AB_CLOSURE_PACK/05_GATE_B_ACCEPTANCE.json

Enforces:
1. Every predicate is computed, never hardcoded.
2. gate_a_closed = all(gate_a_predicates_pass)
3. gate_b_closed = gate_a_closed and all(gate_b_predicates_pass)
   (Gate B CANNOT close if Gate A is false)
4. final_pass = gate_a_closed and gate_b_closed
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PredicateEvaluationResult:
    predicate_path: str
    required_value: Any
    actual_value: Any
    passed: bool
    source_artifact: str = ""
    source_field_or_run: str = ""
    verification_method: str = ""
    independently_recomputed: bool = True
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate": self.predicate_path,
            "required_value": self.required_value,
            "computed_value": self.actual_value,
            "passed": self.passed,
            "source_artifact": self.source_artifact,
            "source_field_or_run": self.source_field_or_run,
            "verification_method": self.verification_method,
            "independently_recomputed": self.independently_recomputed,
            "status": "PASS" if self.passed else "FAIL",
            "notes": self.notes,
        }


@dataclass
class GateEvaluationSummary:
    gate: str
    is_closed: bool
    total_predicates: int
    passed_predicates: int
    failed_predicates: int
    failed_predicate_details: list[dict[str, Any]] = field(default_factory=list)
    predicate_results: dict[str, PredicateEvaluationResult] = field(default_factory=dict)
    prerequisite_met: bool = True
    prerequisite_failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "status": "PASS" if self.is_closed else "FAIL",
            "closed": self.is_closed,
            "total_predicates": self.total_predicates,
            "passed_predicates": self.passed_predicates,
            "failed_predicates": self.failed_predicates,
            "prerequisite_met": self.prerequisite_met,
            "prerequisite_failure_reason": self.prerequisite_failure_reason,
            "failed_predicates_list": self.failed_predicate_details,
            "predicates": {k: v.to_dict() for k, v in self.predicate_results.items()},
        }


class RecursivePredicateEvaluator:
    """Evaluates hierarchical predicate dictionaries against actual computed evidence."""

    def __init__(
        self,
        gate_a_spec_path: Path | None = None,
        gate_b_spec_path: Path | None = None,
    ):
        specs_dir = Path(__file__).resolve().parent / "specs"
        fallback_path = Path(__file__).resolve().parents[3] / "START_GATE_AB_CLOSURE_PACK"
        base_path = specs_dir if specs_dir.exists() else fallback_path
        self.gate_a_spec_path = gate_a_spec_path or (base_path / "02_GATE_A_ACCEPTANCE.json")
        self.gate_b_spec_path = gate_b_spec_path or (base_path / "05_GATE_B_ACCEPTANCE.json")
        
        self.gate_a_spec = self._load_json(self.gate_a_spec_path)
        self.gate_b_spec = self._load_json(self.gate_b_spec_path)

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Spec file not found: {path}")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def flatten_predicates(spec_dict: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten a nested dictionary into dot-separated paths."""
        flat: dict[str, Any] = {}
        for k, v in spec_dict.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                flat.update(RecursivePredicateEvaluator.flatten_predicates(v, prefix=full_key))
            else:
                flat[full_key] = v
        return flat

    def evaluate_gate_a(
        self,
        actual_evidence: dict[str, Any],
        metadata_map: dict[str, dict[str, Any]] | None = None,
    ) -> GateEvaluationSummary:
        """Evaluate Gate A predicates recursively from actual evidence."""
        metadata_map = metadata_map or {}
        hard_specs = self.flatten_predicates(self.gate_a_spec.get("hard_predicates", {}))
        
        results: dict[str, PredicateEvaluationResult] = {}
        failed_details: list[dict[str, Any]] = []

        for pred_path, req_val in hard_specs.items():
            meta = metadata_map.get(pred_path, {})
            actual_val = self._extract_value(actual_evidence, pred_path)
            
            passed = (actual_val == req_val)
            res = PredicateEvaluationResult(
                predicate_path=pred_path,
                required_value=req_val,
                actual_value=actual_val,
                passed=passed,
                source_artifact=meta.get("source_artifact", ""),
                source_field_or_run=meta.get("source_field_or_run", ""),
                verification_method=meta.get("verification_method", "direct_comparison"),
                independently_recomputed=meta.get("independently_recomputed", True),
                notes=meta.get("notes", ""),
            )
            results[pred_path] = res
            if not passed:
                failed_details.append({
                    "predicate": pred_path,
                    "required": req_val,
                    "actual": actual_val,
                    "notes": meta.get("notes", "Requirement not satisfied"),
                })

        all_passed = (len(failed_details) == 0) and (len(results) > 0)
        return GateEvaluationSummary(
            gate="A",
            is_closed=all_passed,
            total_predicates=len(hard_specs),
            passed_predicates=len(hard_specs) - len(failed_details),
            failed_predicates=len(failed_details),
            failed_predicate_details=failed_details,
            predicate_results=results,
            prerequisite_met=True,
            prerequisite_failure_reason=None,
        )

    def evaluate_gate_b(
        self,
        actual_evidence: dict[str, Any],
        gate_a_summary: GateEvaluationSummary | None = None,
        metadata_map: dict[str, dict[str, Any]] | None = None,
    ) -> GateEvaluationSummary:
        """Evaluate Gate B predicates.
        
        CRITICAL INVARIANT: Gate B cannot close if Gate A is not closed!
        """
        metadata_map = metadata_map or {}
        hard_specs = self.flatten_predicates(self.gate_b_spec.get("hard_predicates", {}))

        results: dict[str, PredicateEvaluationResult] = {}
        failed_details: list[dict[str, Any]] = []

        # Check prerequisite
        prereq_met = True
        prereq_reason = None
        if gate_a_summary is not None and not gate_a_summary.is_closed:
            prereq_met = False
            prereq_reason = (
                f"Gate A prerequisite failed: {gate_a_summary.failed_predicates} "
                f"predicate(s) failing in Gate A."
            )

        for pred_path, req_val in hard_specs.items():
            meta = metadata_map.get(pred_path, {})
            actual_val = self._extract_value(actual_evidence, pred_path)
            passed = (actual_val == req_val)
            res = PredicateEvaluationResult(
                predicate_path=pred_path,
                required_value=req_val,
                actual_value=actual_val,
                passed=passed,
                source_artifact=meta.get("source_artifact", ""),
                source_field_or_run=meta.get("source_field_or_run", ""),
                verification_method=meta.get("verification_method", "direct_comparison"),
                independently_recomputed=meta.get("independently_recomputed", True),
                notes=meta.get("notes", ""),
            )
            results[pred_path] = res
            if not passed:
                failed_details.append({
                    "predicate": pred_path,
                    "required": req_val,
                    "actual": actual_val,
                    "notes": meta.get("notes", "Requirement not satisfied"),
                })

        # Gate B closes iff all its hard predicates pass AND Gate A prerequisite is met
        all_predicates_passed = (len(failed_details) == 0) and (len(results) > 0)
        gate_b_closed = all_predicates_passed and prereq_met

        return GateEvaluationSummary(
            gate="B",
            is_closed=gate_b_closed,
            total_predicates=len(hard_specs),
            passed_predicates=len(hard_specs) - len(failed_details),
            failed_predicates=len(failed_details),
            failed_predicate_details=failed_details,
            predicate_results=results,
            prerequisite_met=prereq_met,
            prerequisite_failure_reason=prereq_reason,
        )

    def evaluate_closure(
        self,
        gate_a_evidence: dict[str, Any],
        gate_b_evidence: dict[str, Any],
        metadata_map_a: dict[str, dict[str, Any]] | None = None,
        metadata_map_b: dict[str, dict[str, Any]] | None = None,
        git_operations_performed: int = 0,
    ) -> dict[str, Any]:
        """Compute the complete joint Gate A + Gate B closure evaluation.
        
        Derives all statuses dynamically without hardcoding.
        """
        summary_a = self.evaluate_gate_a(gate_a_evidence, metadata_map_a)
        summary_b = self.evaluate_gate_b(gate_b_evidence, summary_a, metadata_map_b)

        overall_pass = (
            summary_a.is_closed
            and summary_b.is_closed
            and (git_operations_performed == 0)
        )

        remaining_blockers = []
        if not summary_a.is_closed:
            for item in summary_a.failed_predicate_details:
                remaining_blockers.append(f"Gate A: {item['predicate']} (req: {item['required']}, got: {item['actual']})")
        if not summary_b.is_closed:
            if not summary_b.prerequisite_met:
                remaining_blockers.append(f"Gate B: Blocked by Gate A failure ({summary_b.prerequisite_failure_reason})")
            for item in summary_b.failed_predicate_details:
                remaining_blockers.append(f"Gate B: {item['predicate']} (req: {item['required']}, got: {item['actual']})")
        if git_operations_performed > 0:
            remaining_blockers.append(f"Git: {git_operations_performed} mutating git operations performed (must be 0)")

        return {
            "run_summary": {
                "gate_a": summary_a.to_dict(),
                "gate_b": summary_b.to_dict(),
                "final": {
                    "status": "PASS" if overall_pass else "FAIL",
                    "gate_a_closed": summary_a.is_closed,
                    "gate_b_closed": summary_b.is_closed,
                    "git_operations_performed": git_operations_performed,
                    "remaining_blockers": remaining_blockers,
                },
            }
        }

    @staticmethod
    def _extract_value(d: dict[str, Any], dot_path: str) -> Any:
        """Extract value by dot path, supporting both flat and nested keys."""
        if dot_path in d:
            return d[dot_path]
        parts = dot_path.split(".")
        curr: Any = d
        for part in parts:
            if isinstance(curr, dict) and part in curr:
                curr = curr[part]
            else:
                return None
        return curr
