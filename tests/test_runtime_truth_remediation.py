"""Focused regressions for the v6.0.2 runtime-truth remediation."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
from rich.console import Console

from start.core.schemas import ComputeDevice, EvidenceRecord, ReproducibilityMeta, Status
from start.data.selection import select_temporal_sequence
from start.modeling.data_statistics import compute_data_statistics
from start.modeling.fe_recommendations import recommend_feature_engineering
from start.orchestration.checkpointing import build_checkpoint_serializer
from start.orchestration.state_graph import build_canonical_review_graph
from start.providers.offline_demo_twin import OfflineDemoTwinProvider
from start.reporting.current_run import (
    RunLineage,
    build_artifact_entry,
    current_run_presentation_root,
    write_current_run_manifest,
)
from start.review.applicability import build_plan_preview
from start.review.architecture import (
    PredictiveTechnology,
    ReviewContextBundle,
    ReviewDomain,
)
from start.review.structured_contract import (
    CheckpointDegradationPolicy,
    StructuredReviewerResponse,
    get_checkpoint_degradation_policy,
)
from start.review.terminal_observability import render_temporal_contract
from start.telemetry.engineering_trace import EngineeringTracer, TerminalEngineeringRenderer


class _ReviewCancelled(BaseException):
    """Exercise the non-Exception cancellation path without stopping pytest."""


def test_engineering_span_preserves_base_exception_and_records_error() -> None:
    tracer = EngineeringTracer(run_id="RUN-CANCEL")
    original = _ReviewCancelled("review cancelled")

    with pytest.raises(_ReviewCancelled) as caught:
        with tracer.span("start.review.cancel"):
            raise original

    assert caught.value is original
    records = tracer.get_records()
    assert len(records) == 1
    assert records[0].status == "ERROR: _ReviewCancelled: review cancelled"


@pytest.mark.parametrize("exc", [RuntimeError("ordinary failure"), GeneratorExit()])
def test_engineering_span_preserves_other_exceptional_exits(exc: BaseException) -> None:
    tracer = EngineeringTracer(run_id="RUN-EXCEPTIONAL")

    with pytest.raises(type(exc)) as caught:
        with tracer.span("start.review.exceptional"):
            raise exc

    assert caught.value is exc
    assert tracer.get_records()[0].status.startswith(f"ERROR: {type(exc).__name__}")


def test_engineering_span_records_normal_completion() -> None:
    tracer = EngineeringTracer(run_id="RUN-OK")
    with tracer.span("start.review.ok"):
        pass
    assert tracer.get_records()[0].status == "OK"


def test_checkpoint_serializer_strict_roundtrip_has_no_warnings() -> None:
    serializer = build_checkpoint_serializer()
    record = EvidenceRecord(
        evidence_id="EV-SERIALIZER",
        test_id="serializer.roundtrip",
        test_name="Serializer round trip",
        model_id="MOD-1",
        dataset_id="DS-1",
        run_id="RUN-SERIALIZER",
        status=Status.PASS,
        repro=ReproducibilityMeta(device=ComputeDevice.CPU),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        encoded = serializer.dumps_typed(
            {
                "status": Status.PASS,
                "device": ComputeDevice.CPU,
                "record": record,
            }
        )
        decoded = serializer.loads_typed(encoded)

    assert decoded["status"] is Status.PASS
    assert decoded["device"] is ComputeDevice.CPU
    assert decoded["record"] == record
    assert isinstance(decoded["record"], EvidenceRecord)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        graph = build_canonical_review_graph()
        graph.invoke(
            {
                "run_id": "RUN-SERIALIZER",
                "thread_id": "thread-serializer",
                "current_node": "__start__",
                "evidence_records": [record],
                "evidence_ids": [record.evidence_id],
                "governance_state": {"disposition": "ACCEPT"},
                "step_history": [],
                "errors": [],
                "retry_count": 0,
                "max_retries": 1,
            },
            config={"configurable": {"thread_id": "thread-serializer"}},
        )
        snapshot = graph.get_state(
            {"configurable": {"thread_id": "thread-serializer"}}
        )
    assert snapshot.values["evidence_records"][0] == record


def test_temporal_plan_and_contract_are_sequence_native() -> None:
    selection = select_temporal_sequence(seed=42)
    bundle = ReviewContextBundle(
        domains=(ReviewDomain.PREDICTIVE,),
        technology=PredictiveTechnology.DEEP_LEARNING,
        tabular=selection.frame,
        temporal_sequence=selection.sequence_bundle,
    )
    rendered_plan = build_plan_preview(bundle).render()
    assert "[x] temporal_sequence" in rendered_plan
    assert "[x] tabular" not in rendered_plan
    assert "Temporal Input-Gradient Saliency" in rendered_plan
    assert "SHAP" not in rendered_plan

    console = Console(record=True, width=120)
    sequence = selection.sequence_bundle
    total = len(sequence.X_train) + len(sequence.X_test) + len(sequence.X_oos)
    console.print(
        render_temporal_contract(
            n_sequences=total,
            timesteps=sequence.timesteps,
            n_features=sequence.n_features,
            task="binary sequence classification",
        )
    )
    text = console.export_text()
    assert f"({total}, {sequence.timesteps}, {sequence.n_features})" in text
    assert "independent sequences; not chronological future periods" in text


def test_temporal_recommendations_never_encode_metadata_ids() -> None:
    selection = select_temporal_sequence(seed=42)
    stats = compute_data_statistics(selection.frame, selection.target_column)
    recommendations = recommend_feature_engineering(stats, modality="temporal_sequence")
    content = " ".join(
        f"{item.step} {item.recommendation} {item.default_action}"
        for item in recommendations.applicable()
    ).lower()
    assert "sequence_id" not in content
    assert "one-hot" not in content
    assert "correlation_pruning" not in content
    assert "low_variance" not in content


def test_offline_demo_twin_is_zero_hosted_and_checkpoint_aware() -> None:
    provider = OfflineDemoTwinProvider()
    assert provider.name == "offline_demo_twin"
    assert provider.network_enabled is False
    assert provider.hosted_calls == 0

    system = "Return StructuredReviewerResponse JSON."
    mandatory_prompt = """Checkpoint: Portfolio Risk & Volatility Assumptions
Semantic Checkpoint ID: market.portfolio_allocation
Interaction Action: QUESTION
Question: "Compare the registered portfolio methods"
- [EV-REAL] Test: portfolio.risk_statistics (Status: pass)
  Admissible Canonical Metric Paths for [EV-REAL]:
    * metrics.annualised_volatility (value: 0.2)
"""
    result = provider.complete_result(system, mandatory_prompt)
    response = StructuredReviewerResponse.model_validate_json(result.text)
    ref = response.findings[0].evidence_refs[0]
    assert ref.evidence_id == "EV-REAL"
    assert ref.metric_path == "metrics.annualised_volatility"
    assert result.usage.input_tokens == result.usage.output_tokens == 0

    malformed = StructuredReviewerResponse.model_validate_json(
        provider.complete(
            system,
            mandatory_prompt.replace(
                "Portfolio Risk & Volatility Assumptions",
                "Cross-Analytical Committee Synthesis",
            )
            .replace("market.portfolio_allocation", "market.cross_analytical_synthesis")
            .replace(
                "Compare the registered portfolio methods",
                "Give me a final evidence-oriented summary",
            ),
        )
    )
    assert malformed.findings[0].evidence_refs[0].evidence_id == "EV-NOT-IN-CHECKPOINT"
    signoff = StructuredReviewerResponse.model_validate_json(
        provider.complete(
            system,
            mandatory_prompt.replace(
                "Portfolio Risk & Volatility Assumptions",
                "Model Governance & Attestation Sign-off",
            ).replace("market.portfolio_allocation", "market.governance_signoff"),
        )
    )
    assert signoff.findings[0].evidence_refs[0].evidence_id == "EV-REAL"


def test_degradation_policy_is_semantic_not_blanket_question() -> None:
    assert (
        get_checkpoint_degradation_policy("market.cross_analytical_synthesis", action="Q")
        is CheckpointDegradationPolicy.ALLOW_EVIDENCE_ONLY_DEGRADATION
    )
    for checkpoint in (
        "market.portfolio_allocation",
        "market.governance_signoff",
        "unregistered.question",
    ):
        assert (
            get_checkpoint_degradation_policy(checkpoint, action="Q")
            is CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE
        )
    for action in ("C", "O", "A"):
        assert (
            get_checkpoint_degradation_policy("market.cross_analytical_synthesis", action=action)
            is CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE
        )


def test_current_run_manifest_is_atomic_exact_and_strict(tmp_path: Path) -> None:
    review_id = "RUN-REVIEW-EXACT"
    execution_id = "RUN-ENT-EXACT"
    presentation_root = current_run_presentation_root(tmp_path, review_id)
    scientific_root = tmp_path / "model_execution" / execution_id / "artifacts"
    scientific_root.mkdir(parents=True)
    artifacts = [
        scientific_root / "ART-TEMPORAL-CONTRACT-test.svg",
        scientific_root / "ART-TEMPORAL-INPUT-GRADIENT-test.svg",
        scientific_root / "ART-TEMPORAL-ROBUSTNESS-test.svg",
    ]
    for artifact in artifacts:
        artifact.write_text("<svg/>", encoding="utf-8")
    lineage = RunLineage(review_id, (execution_id,), "RUN-MROS-EXACT")
    entries = [
        build_artifact_entry(
            path=artifact,
            owner_run_id=execution_id,
            lineage=lineage,
            artifact_id=artifact.stem,
            artifact_type="figure (SVG)",
            title=artifact.stem.replace("-", " ").title(),
            checkpoint="Flight A",
        )
        for artifact in artifacts
    ]
    manifest = write_current_run_manifest(
        presentation_root=presentation_root,
        lineage=lineage,
        groups={"flight_a": entries, "flight_b": [], "flight_c": []},
    )
    assert manifest == presentation_root / "artifact_manifest.json"
    assert not list(presentation_root.glob(".artifact_manifest.*.tmp"))
    payload = manifest.read_text(encoding="utf-8")
    assert "ART-TEMPORAL-CONTRACT-test" in payload
    assert "ART-TEMPORAL-INPUT-GRADIENT-test" in payload
    assert "ART-TEMPORAL-ROBUSTNESS-test" in payload

    artifacts[0].unlink()
    with pytest.raises(ValueError, match="inconsistent current-run artifact"):
        write_current_run_manifest(
            presentation_root=presentation_root,
            lineage=lineage,
            groups={"flight_a": entries},
        )


def test_engineering_renderer_uses_domain_shape_and_release_version() -> None:
    output = TerminalEngineeringRenderer.render(
        run_id="RUN-C",
        dataset_contract={
            "dataset_id": "institutional_market_v1",
            "rows": None,
            "features": None,
            "shape_descriptor": "50 assets x 1000 observations; return matrix (1000, 50)",
        },
        orchestration={},
        dispatch={},
        scientific_invariants={},
        evidence_lineage={},
        governance_policy={},
        reproducibility={},
        resource_ledger={},
    )
    assert "None rows x None features" not in output
    assert "50 assets x 1000 observations" in output
    assert "StART version 6.0.2" in output
    assert "2.0-Enterprise" not in output
    assert "Engine:           NOT_EVALUATED" in output
    assert "Engine:           OPA_LOCAL" not in output
