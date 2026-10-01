"""Focused zero-cost tests for the generalized workflow-coherence architecture."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from rich.console import Console

from start.review.coherence_verifier import (
    _configuration_resolution_proofs,
    _pairwise_cover_v2,
    _predictive_fixture,
    _unified_fixture,
    failure_reconciliation_v2,
    generalized_explainability_invariant,
    run_verification,
    semantic_invariants_v3,
)
from start.review.terminal_observability import PresentationEventKind, TerminalReviewObserver
from start.review.workflow_coherence import (
    AgentDecisionTrace,
    CapabilityState,
    DeterministicExecutionSummary,
    OutputStructure,
    UnsupportedCapabilityError,
    build_canonical_runtime_coherence_envelope,
    build_predictive_coherence_envelope,
    build_run_outcome_capsule,
    resolve_predictive_output_structure,
    synchronize_predictive_explainability,
)


def _predictive_envelope(run_id: str = "RUN-COHERENCE"):
    frame = pd.DataFrame(
        {
            "feature_a": [0.0, 1.0, 2.0, 3.0],
            "feature_b": [1.0, 0.0, 1.0, 0.0],
            "target": [0, 1, 1, 0],
        }
    )
    config = SimpleNamespace(
        target="target",
        sequence_bundle=None,
        architecture_family="wide_deep",
        activation="relu",
        explain_method="integrated_gradients",
        robustness_suite="standard",
        enterprise_mode=True,
        run_dl=True,
        objective="review a classification model",
        costlier_errors="balanced",
        data_path=None,
    )
    context = SimpleNamespace(
        model_sample_count=4,
        model_feature_count=2,
        timesteps=None,
        sample_structure="4 rows × 2 model features",
        model_input_contract="(4,2) rank-2 tabular input",
        split_description="stratified holdout",
        architecture_alternatives=("mlp", "wide_deep", "residual_mlp"),
        modality="tabular",
    )
    return build_predictive_coherence_envelope(
        run_id=run_id,
        config=config,
        frame=frame,
        task_type="binary_classification",
        presentation_context=context,
    )


def test_predictive_contract_separates_intake_problem_and_capability_plan() -> None:
    envelope = _predictive_envelope()

    assert envelope.data_intake.source_kind == "synthetic"
    assert envelope.data_intake.dimensions == {
        "samples": 4,
        "features": 2,
        "timesteps": None,
    }
    assert envelope.problem.target_specification == ("target",)
    assert envelope.problem.output_structure is OutputStructure.SINGLE_TARGET
    assert envelope.problem.model_input_contract == "(4,2) rank-2 tabular input"
    assert envelope.capability_plan.state_for("AGENT_DECISIONS") is CapabilityState.APPLICABLE
    assert envelope.capability_plan.state_for("GROUNDING") is CapabilityState.APPLICABLE


def test_predictive_contract_uses_actual_tabular_and_temporal_explainability_identity() -> None:
    tabular = _predictive_fixture(sequence=False)
    temporal = _predictive_fixture(sequence=True)

    assert tabular.problem.domain_payloads["predictive"]["explainability"] == (
        "PENDING_RUNTIME_RESOLUTION"
    )
    assert tabular.problem.domain_payloads["predictive"]["requested_explainability"] == (
        "integrated_gradients"
    )
    assert synchronize_predictive_explainability(tabular, "permutation") == "permutation"
    assert tabular.problem.domain_payloads["predictive"]["explainability"] == "permutation"

    assert synchronize_predictive_explainability(
        temporal,
        "Temporal Input-Gradient Saliency",
    ) == "Temporal Input-Gradient Saliency / INPUT_GRADIENT"
    assert temporal.problem.domain_payloads["predictive"]["explainability"] == (
        "Temporal Input-Gradient Saliency / INPUT_GRADIENT"
    )


def test_generalized_explainability_invariant_discovers_predictive_cases() -> None:
    def predictive_case(
        case_id: str,
        modality: str,
        contract: str,
        execution: str,
        evidence: str | None,
    ) -> dict[str, object]:
        return {
            "case_id": case_id,
            "problem_contract": {
                "domains": ["predictive"],
                "domain_payloads": {
                    "predictive": {
                        "modality": modality,
                        "explainability": contract,
                    }
                },
            },
            "deterministic_execution_summaries": [
                {"operations": [f"explainability={execution}"]}
            ],
            "evidence_explainability_method_identities": (
                []
                if evidence is None
                else [{"test_id": "registered.explainability", "method": evidence}]
            ),
        }

    result = generalized_explainability_invariant(
        [
            predictive_case("arbitrary-a", "tabular", "permutation", "permutation", "permutation"),
            predictive_case(
                "arbitrary-b",
                "temporal_sequence",
                "Temporal Input-Gradient Saliency / INPUT_GRADIENT",
                "Temporal Input-Gradient Saliency / INPUT_GRADIENT",
                None,
            ),
            {"case_id": "non-predictive", "problem_contract": {"domains": ["market"]}},
        ]
    )

    assert result["status"] == "PASS"
    assert result["applicable_case_count"] == 2
    assert result["discovered_predictive_cases"] == ["arbitrary-a", "arbitrary-b"]

    missing_contract = predictive_case(
        "arbitrary-missing-contract",
        "tabular",
        "",
        "permutation",
        "permutation",
    )
    rejected = generalized_explainability_invariant([missing_contract])
    assert rejected["status"] == "FAIL"
    assert rejected["failures"][0]["case_id"] == "arbitrary-missing-contract"


@pytest.mark.parametrize(
    ("task_type", "targets", "code"),
    [
        ("regression", ("y1", "y2"), "MULTI_TARGET_ROUTE_UNSUPPORTED"),
        ("multilabel_classification", ("y1", "y2"), "MULTI_TARGET_ROUTE_UNSUPPORTED"),
        ("sequence_output", ("target",), "SEQUENCE_OUTPUT_UNSUPPORTED"),
    ],
)
def test_unsupported_output_forms_fail_closed(
    task_type: str,
    targets: tuple[str, ...],
    code: str,
) -> None:
    with pytest.raises(UnsupportedCapabilityError) as exc_info:
        resolve_predictive_output_structure(
            task_type=task_type,
            targets=targets,
            route_supports_multi_output=False,
            sequence_input=task_type == "sequence_output",
        )
    assert exc_info.value.code == code
    assert exc_info.value.to_dict()["status"] == "UNSUPPORTED"


def test_canonical_deterministic_route_marks_human_agent_grounding_not_applicable() -> None:
    envelope = build_canonical_runtime_coherence_envelope(
        run_id="RUN-CONTROL",
        workflow_id="quantitative_finance",
        context_id="institutional_market_v1",
        context_metadata={
            "spec_id": "institutional_market_v1",
            "actual_assets": 50,
            "actual_periods": 1000,
            "actual_target": "N/A",
        },
    )

    plan = envelope.capability_plan
    assert plan.state_for("HUMAN_DECISIONS") is CapabilityState.NOT_APPLICABLE
    assert plan.state_for("AGENT_DECISIONS") is CapabilityState.NOT_APPLICABLE
    assert plan.state_for("GROUNDING") is CapabilityState.NOT_APPLICABLE
    assert plan.state_for("DETERMINISTIC_EXECUTION") is CapabilityState.APPLICABLE
    assert plan.state_for("GOVERNANCE_POLICY") is CapabilityState.APPLICABLE


def test_outcome_does_not_mislabel_missing_applicable_decisions_as_not_applicable() -> None:
    envelope = _predictive_envelope("RUN-NO-DECISIONS")
    state = SimpleNamespace(
        grounding_state=None,
        governance_disposition=None,
        policy_decision=None,
    )

    capsule = build_run_outcome_capsule(
        envelope=envelope,
        events=[],
        presentation_state=state,
    )

    assert capsule.human_decisions == ("APPLICABLE — no human decision event recorded",)
    assert capsule.agent_contribution == (
        "APPLICABLE — no consolidated decision trace emitted",
    )
    assert "human decision stage applicable but no decision event was recorded" in (
        capsule.remaining_conditions
    )
    assert "agent decision stage applicable but no consolidated trace was recorded" in (
        capsule.remaining_conditions
    )


def test_live_observer_consumes_contract_trace_summary_and_outcome(tmp_path: Path) -> None:
    envelope = _predictive_envelope("RUN-LIVE-COHERENCE")
    stream = io.StringIO()
    observer = TerminalReviewObserver(
        envelope.run_id,
        console=Console(file=stream, width=132, color_system=None),
        session_kind="GENERIC",
    )
    observer.bind_coherence_envelope(envelope)
    observer.session_started(source_component="fixture")
    observer.publish_coherence_contract()
    observer.human_action("A", "architecture")
    observer.agent_decision_trace(
        AgentDecisionTrace(
            agent_identity="ArchitectureReviewAgent",
            role="architecture review",
            trigger="checkpoint",
            checkpoint="architecture",
            canonical_input_references=("(4,2)",),
            evidence_record_references=("EV-ARCH",),
            applicable_alternatives=("mlp", "wide_deep"),
            recommendation="wide_deep",
            rationale_summary="Registered tabular family selected for the resolved input contract.",
            limitations=("No numerical authority.",),
            human_action="ACCEPT",
            resulting_action="wide_deep selected",
        )
    )
    observer.engine_completed("ModelExecution", output_count=1, details="registered execution complete")
    observer.evidence_reference_committed(
        evidence_id="EV-ARCH",
        test_id="model.architecture",
        record_status="RECORDED",
    )
    observer.deterministic_execution_summary(
        DeterministicExecutionSummary(
            engine_ids=("ModelExecution",),
            canonical_input_references=("(4,2)",),
            operations=("fit", "evaluate"),
            authoritative_outputs=("model.architecture",),
            statuses=("RECORDED",),
            evidence_record_references=("EV-ARCH",),
            artifact_references=(),
            limitations=(),
            skipped_or_not_applicable=(),
        )
    )
    observer.governance_card(
        disposition="ACCEPT_WITH_CONDITIONS",
        evidence_count=1,
        unresolved_count=0,
        validation_failures=0,
    )
    observer.session_completed(source_component="fixture")

    kinds = [event.kind for event in observer.events]
    assert PresentationEventKind.DATA_SOURCE_RESOLVED in kinds
    assert PresentationEventKind.PROBLEM_CONTRACT_RESOLVED in kinds
    assert PresentationEventKind.CAPABILITY_PLAN_RESOLVED in kinds
    assert PresentationEventKind.AGENT_DECISION_RECORDED in kinds
    assert PresentationEventKind.DETERMINISTIC_EXECUTION_SUMMARIZED in kinds
    assert PresentationEventKind.OUTCOME_READY in kinds
    assert observer.outcome_capsule is not None
    assert observer.outcome_capsule.evidence_record_references == ("EV-ARCH",)
    assert envelope.outcome is observer.outcome_capsule

    export_path = observer.export(tmp_path / "presentation_events.json")
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["workflow_coherence"]["outcome"]["run_id"] == envelope.run_id
    assert "RUN OUTCOME CAPSULE" in stream.getvalue()


def test_identical_consecutive_spine_state_is_coalesced() -> None:
    stream = io.StringIO()
    observer = TerminalReviewObserver(
        "RUN-COALESCE",
        console=Console(file=stream, width=132, color_system=None),
        session_kind="GENERIC",
    )
    observer.artifact_available(
        artifact_id="ART-1",
        file_path="/tmp/one.json",
        evidence_ids=(),
        checkpoint="artifact",
    )
    observer.artifact_available(
        artifact_id="ART-2",
        file_path="/tmp/two.json",
        evidence_ids=(),
        checkpoint="artifact",
    )

    assert stream.getvalue().count("ARCHITECTURE SPINE") == 1
    assert len(observer.events) == 2


def test_repository_owned_verifier_is_deterministic_and_complete() -> None:
    workspace = Path(__file__).parents[1]
    first = run_verification(workspace)
    second = run_verification(workspace)

    assert first == second
    assert first["status"] == "PASS"
    assert first["hosted_calls"] == 0
    assert first["contract_enumeration"]["count"] > 20
    assert first["pairwise_cover"]["count"] > 0
    assert all(case["status"] == "PASS" for case in first["bounded_executable_cases"])
    assert all(
        case["execution_proof"]["executed_surfaces"]
        for case in first["bounded_executable_cases"]
    )
    assert all(case["status"] == "PASS" for case in first["fail_closed_negative_cases"])
    assert all(item["status"] == "PASS" for item in first["production_wiring"])
    assert first["generalization_guards"]["status"] == "PASS"


def test_v2_pairwise_cover_spans_every_feasible_pair() -> None:
    cover = _pairwise_cover_v2()

    assert cover["case_count"] < cover["candidate_count"]
    assert cover["feasible_pairs_left_uncovered"] == 0
    assert cover["covered_pair_count"] > 0
    assert cover["uncovered_pairs"]
    assert all(row["reason"].startswith("CONSTRAINED_OUT") for row in cover["uncovered_pairs"])
    assert {
        "review_mode",
        "domain",
        "technology",
        "model_architecture_family",
        "data_source_class",
        "target_output_mode",
        "lifecycle",
        "review_scope",
        "materiality",
        "reviewer_backend",
    } == set(cover["dimensions"])
    assert all(case["verification_level"] == "CONFIGURATION_RESOLVED" for case in cover["cases"])
    assert not any(case["hosted_call"] for case in cover["cases"])


def test_v2_lifecycle_scope_and_backend_configuration_resolution() -> None:
    proof = _configuration_resolution_proofs()

    assert proof["status"] == "PASS"
    assert len(proof["lifecycles"]) == 5
    assert len(proof["reviewer_backends"]) == 4
    assert proof["review_scope"]["customized_is_registry_subset"] is True
    assert proof["hosted_calls"] == 0


def test_v2_reconciles_exact_cached_failure_manifest() -> None:
    result = failure_reconciliation_v2(Path(__file__).parents[1])

    assert result["previous_failures_accounted"] == "18/18"
    assert result["cache_exact_match"] is True
    assert result["classification_counts"] == {
        "INTERRUPTED_RUN_ARTIFACT": 1,
        "PRE_EXISTING_UNRELATED_DEFECT": 5,
        "STALE_TEST_EXPECTATION": 12,
    }
    assert len(result["records"]) == 18
    required = {
        "node_id",
        "test_file",
        "failure_assertion",
        "expected_behavior",
        "actual_behavior",
        "authoritative_contract",
        "predates_accepted_semantic_change",
        "bounded_rerun_result",
        "classification",
    }
    assert all(required <= set(record) for record in result["records"])


def test_non_demo_temporal_metadata_and_explainability_are_runtime_derived() -> None:
    from start.data.selection import select_temporal_sequence
    from start.interactive_review import ReviewConfig
    from start.review.terminal_observability import build_flight_a_presentation_context

    selection = select_temporal_sequence(n_series=30, timesteps=4, n_features=2, seed=7)
    context = build_flight_a_presentation_context(
        df=selection.frame,
        sequence_bundle=selection.sequence_bundle,
        selected_architecture="gru",
        configured_split="stratified",
    )
    envelope = build_predictive_coherence_envelope(
        run_id="RUN-NON-DEMO-TEMPORAL",
        config=ReviewConfig(
            target="target",
            dataset_selection=selection,
            sequence_bundle=selection.sequence_bundle,
            architecture_family="gru",
            explain_method="integrated_gradients",
            run_dl=True,
            enterprise_mode=True,
        ),
        frame=selection.frame,
        task_type="binary_classification",
        presentation_context=context,
        dataset_selection=selection,
    )

    assert envelope.data_intake.dimensions == {"samples": 30, "features": 2, "timesteps": 4}
    metadata = json.dumps(envelope.data_intake.provenance_metadata, sort_keys=True)
    assert "30 sequences, 4 timesteps, 2 features" in metadata
    assert "18 train / 6 test / 6 OOS" in metadata
    assert "800 sequences" not in metadata
    assert "480 train" not in metadata
    assert envelope.problem.domain_payloads["predictive"]["explainability"] == (
        "Temporal Input-Gradient Saliency / INPUT_GRADIENT"
    )


@pytest.mark.parametrize(
    ("governance", "unresolved", "failures", "expected"),
    [
        ("ACCEPT", 0, 0, "NO_REMAINING_CONDITIONS"),
        ("ACCEPT_WITH_CONDITIONS", 2, 0, "2 unresolved governance item(s)"),
        ("REMEDIATION_REQUIRED", 0, 3, "3 validation failure(s) require disposition"),
    ],
)
def test_outcome_conditions_agree_with_governance(
    governance: str, unresolved: int, failures: int, expected: str
) -> None:
    envelope = _predictive_envelope(f"RUN-{governance}")
    state = SimpleNamespace(
        grounding_state="ACCEPTED",
        governance_disposition=governance,
        governance_unresolved_count=unresolved,
        governance_validation_failures=failures,
        policy_decision="ALLOW",
    )
    events = [
        SimpleNamespace(
            kind=SimpleNamespace(value="HUMAN_ACTION"),
            payload={"action": "A", "checkpoint": "architecture"},
            semantic_checkpoint_id="architecture",
        ),
        SimpleNamespace(
            kind=SimpleNamespace(value="AGENT_DECISION_RECORDED"),
            payload={"agent_identity": "ArchitectureReviewAgent", "recommendation": "mlp"},
            semantic_checkpoint_id="architecture",
        ),
    ]
    capsule = build_run_outcome_capsule(
        envelope=envelope, events=events, presentation_state=state
    )

    assert expected in capsule.remaining_conditions
    if governance != "ACCEPT":
        assert "NO_REMAINING_CONDITIONS" not in capsule.remaining_conditions
        assert "NONE_RECORDED" not in capsule.remaining_conditions


def test_review_run_ids_are_unique_when_wall_clock_is_frozen(monkeypatch) -> None:
    import start.review.executor as executor

    monkeypatch.setattr(executor.time, "time", lambda: 1_700_000_000.0)
    run_ids = [executor.generate_review_run_id() for _ in range(1_000)]

    assert len(run_ids) == len(set(run_ids))
    assert all(run_id.startswith("RUN-REVIEW-1700000000-") for run_id in run_ids)


@pytest.mark.parametrize(
    "domains",
    [
        pytest.param(("market",), id="market"),
        pytest.param(("treasury",), id="treasury"),
        pytest.param(("market", "treasury"), id="cross-domain"),
    ],
)
def test_domain_alternatives_are_registered_surfaces(domains: tuple[str, ...]) -> None:
    from start.review.architecture import ReviewDomain

    enum_domains = tuple(ReviewDomain(domain) for domain in domains)
    envelope = _unified_fixture(enum_domains)
    alternatives = envelope.problem.candidate_methods

    assert alternatives
    assert all("." in alternative for alternative in alternatives)
    assert not set(alternatives) <= {"portfolio", "attribution", "traded_risk", "covariance"}
    if len(domains) > 1:
        assert all(":" in alternative for alternative in alternatives)
    for domain in domains:
        assert envelope.problem.domain_payloads[domain]["analytical_surfaces"]


def test_v3_semantic_invariants_reject_known_bad_shapes_and_conditions() -> None:
    case = {
        "case_id": "bad-temporal",
        "production_route": "interactive_predictive_review",
        "data_intake_contract": {
            "dimensions": {"samples": 30, "timesteps": 4, "features": 2},
            "provenance_metadata": {
                "notes": [
                    "Multivariate temporal sequence dataset: 800 sequences, 24 timesteps, 3 features.",
                    "Order-preserving contiguous temporal block split (480 train / 160 test / 160 OOS).",
                ],
                "parameters": {"split_sizes": {"train": 18, "test": 6, "oos": 6}},
            },
            "limitations": [],
        },
        "problem_contract": {
            "domain_payloads": {
                "predictive": {
                    "modality": "temporal_sequence",
                    "explainability": "integrated_gradients",
                }
            },
            "candidate_methods": ["gru"],
        },
        "deterministic_execution_summaries": [{"operations": ["explainability=integrated_gradients"]}],
        "run_outcome_capsule": {
            "run_id": "DUPLICATE",
            "governance_policy": "governance=REMEDIATION_REQUIRED; policy=ALLOW",
            "remaining_conditions": ["NONE_RECORDED"],
            "deterministic_results": ["['model development', 'evidence commit']"],
        },
    }
    result = semantic_invariants_v3([case, {**case, "case_id": "duplicate-run"}])

    failed = {item["invariant"] for item in result["failures"]}
    assert result["status"] == "FAIL"
    assert "temporal_dimensions_match_provenance" in failed
    assert "temporal_split_matches_provenance" in failed
    assert "explainability_identity_consistent" in failed
    assert "governance_conditions_agree" in failed
    assert "run_ids_unique_within_verification_batch" in failed
    assert "outcome_values_are_semantic_not_collection_repr" in failed
