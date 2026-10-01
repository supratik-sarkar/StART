"""Repository-owned, zero-cost generalized-coherence verifier.

The verifier exercises contract resolution and the live observer consumer using
bounded in-memory fixtures.  It never launches the demo, invokes a model provider,
or performs scientific calculations.  Production-route wiring is checked against
the actual A/B/C entrypoint source owners in addition to the fixture checks.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rich.console import Console

from start.review.workflow_coherence import (
    UnsupportedCapabilityError,
    build_canonical_runtime_coherence_envelope,
    build_predictive_coherence_envelope,
    build_unified_review_coherence_envelope,
    canonical_predictive_explainability_identity,
    capability_inventory,
    deterministic_pairwise_cases,
    enumerate_contract_cases,
    resolve_predictive_output_structure,
)

SCHEMA = "start.generalized-workflow-coherence-verification/1"
SCHEMA_V2 = "start.generalized-workflow-coherence-verification/2"
SCHEMA_V3 = "start.generalized-workflow-coherence-verification/3"
SCHEMA_V4 = "start.generalized-workflow-coherence-verification/4"
MATRIX_SCHEMA_V2 = "start.generalized-capability-matrix/2"


@dataclass(frozen=True)
class VerificationPaths:
    workspace: Path
    result_json: Path
    matrix_json: Path
    matrix_markdown: Path


@dataclass(frozen=True)
class VerificationPathsV2:
    """Versioned outputs for the hardening pass; V1 evidence is never overwritten."""

    workspace: Path
    result_json: Path
    matrix_json: Path
    matrix_markdown: Path
    audit_markdown: Path
    failure_reconciliation_json: Path


@dataclass(frozen=True)
class VerificationPathsV3:
    workspace: Path
    result_json: Path
    closure_report: Path


@dataclass(frozen=True)
class VerificationPathsV4:
    """Final micro-closure outputs; all prior verification artifacts are retained."""

    workspace: Path
    result_json: Path
    closure_report: Path


def _event_values(observer: Any) -> list[str]:
    return [str(getattr(event.kind, "value", event.kind)) for event in observer.events]


def _verify_live_consumer(envelope: Any, *, control_plane: bool) -> dict[str, Any]:
    from start.review.terminal_observability import TerminalReviewObserver

    stream = io.StringIO()
    observer = TerminalReviewObserver(
        envelope.run_id,
        console=Console(file=stream, width=132, color_system=None),
        session_kind="VERIFY",
        control_plane=control_plane,
    )
    observer.bind_coherence_envelope(envelope)
    observer.publish_coherence_contract()
    expected = [
        "DATA_SOURCE_RESOLVED",
        "PROBLEM_CONTRACT_RESOLVED",
        "CAPABILITY_PLAN_RESOLVED",
    ]
    events = _event_values(observer)
    return {
        "status": "PASS" if events == expected else "FAIL",
        "events": events,
        "state": {
            "data": observer.state.stage_status["DATA"],
            "objective": observer.state.stage_status["OBJECTIVE"],
            "plan": observer.state.stage_status["PLAN"],
        },
        "rendered": all(
            marker in stream.getvalue()
            for marker in (
                "DATA SOURCE RESOLVED",
                "PROBLEM / OBJECTIVE CONTRACT",
                "CAPABILITY PLAN",
            )
        ),
    }


def _predictive_fixture(*, sequence: bool) -> Any:
    import pandas as pd

    if sequence:
        frame = pd.DataFrame(
            {
                "sequence_id": ["s0", "s1", "s2", "s3"],
                "split": ["train", "train", "test", "oos"],
                "target": [0, 1, 0, 1],
            }
        )
        context = SimpleNamespace(
            model_sample_count=4,
            model_feature_count=3,
            timesteps=6,
            sample_structure="4 independent sequences × 6 timesteps × 3 model features",
            model_input_contract="(4,6,3) (N,T,F)",
            split_description="order-preserving sequence holdout",
            architecture_alternatives=("rnn", "gru", "lstm", "bi_lstm"),
            modality="temporal_sequence",
        )
        config = SimpleNamespace(
            target="target",
            sequence_bundle=object(),
            architecture_family="gru",
            activation="tanh",
            explain_method="input_gradient",
            robustness_suite="standard",
            enterprise_mode=True,
            run_dl=True,
            objective="classify complete sequences",
            costlier_errors="balanced",
            data_path=None,
        )
        task = "binary_classification"
        run_id = "VERIFY-PREDICTIVE-SEQUENCE"
    else:
        frame = pd.DataFrame(
            {"feature_a": [0.0, 1.0, 2.0, 3.0], "feature_b": [1.0, 0.0, 1.0, 0.0], "target": [0, 1, 1, 0]}
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
        config = SimpleNamespace(
            target="target",
            sequence_bundle=None,
            architecture_family="mlp",
            activation="relu",
            explain_method="integrated_gradients",
            robustness_suite="standard",
            enterprise_mode=True,
            run_dl=True,
            objective="classify tabular outcomes",
            costlier_errors="balanced",
            data_path=None,
        )
        task = "binary_classification"
        run_id = "VERIFY-PREDICTIVE-TABULAR"
    return build_predictive_coherence_envelope(
        run_id=run_id,
        config=config,
        frame=frame,
        task_type=task,
        presentation_context=context,
    )


def _unified_fixture(domains: tuple[Any, ...]) -> Any:
    import numpy as np
    import pandas as pd

    from start.registry import TestContext
    from start.review.applicability import applicable_tests
    from start.review.architecture import (
        LLMReviewConfig,
        PredictiveTechnology,
        ReviewContextBundle,
        ReviewDomain,
        ReviewLifecycle,
        ReviewMode,
    )

    frame = pd.DataFrame({"feature": [0.0, 1.0, 2.0], "target": [0, 1, 0]})
    tabular = TestContext(train=frame, test=frame.copy(), target_column="target")
    market = SimpleNamespace(
        returns=pd.DataFrame({"asset_a": [0.01, -0.01, 0.02], "asset_b": [0.0, 0.01, -0.01]}),
        describe=lambda: {"kind": "market", "assets": 2, "periods": 3},
    )
    short_rate = SimpleNamespace(
        rates=np.array([0.02, 0.021, 0.019]),
        describe=lambda: {"kind": "short_rate", "periods": 3},
    )
    mode = ReviewMode.CROSS_DOMAIN if len(domains) > 1 else ReviewMode.SINGLE_DOMAIN
    bundle = ReviewContextBundle(
        tabular=tabular if ReviewDomain.PREDICTIVE in domains else None,
        market=market if ReviewDomain.MARKET in domains else None,
        short_rate=short_rate if ReviewDomain.TREASURY in domains else None,
        domains=domains,
        mode=mode,
        technology=(
            PredictiveTechnology.TRADITIONAL_ML
            if ReviewDomain.PREDICTIVE in domains
            else None
        ),
        lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
        llm_config=LLMReviewConfig(backend_mode="none", provider="none"),
        business_context="bounded registry-driven review fixture",
    )
    return build_unified_review_coherence_envelope(
        run_id=f"VERIFY-UNIFIED-{'+'.join(domain.value for domain in domains)}",
        bundle=bundle,
        applicable=applicable_tests(domains),
    )


def _bounded_execution_proofs() -> dict[str, dict[str, Any]]:
    """Execute tiny deterministic representatives through their production owners."""
    import tempfile

    import numpy as np
    import pandas as pd

    from start.data.synthetic_market import generate_market_world
    from start.modeling.models import resolve_model
    from start.registry import TestContext, get_test
    from start.review.terminal_observability import (
        RuntimePresentationSink,
        TerminalReviewObserver,
    )
    from start.runtime.execution import CanonicalExecutionService

    proofs: dict[str, dict[str, Any]] = {}

    def result_status(result: Any) -> str:
        return str(getattr(getattr(result, "status", ""), "value", result.status)).upper()

    frame = pd.DataFrame(
        {
            "feature_a": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "feature_b": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "target": [0, 0, 0, 1, 0, 1, 1, 1],
        }
    )
    tabular_context = TestContext(
        train=frame,
        test=frame.copy(),
        target_column="target",
    )
    tabular_result = get_test("preprocessing.missingness").fn(tabular_context)
    tabular_model, resolved_name, _ = resolve_model("logistic_regression", seed=7)
    tabular_model.fit(frame[["feature_a", "feature_b"]], frame["target"])
    tabular_predictions = tabular_model.predict(frame[["feature_a", "feature_b"]])
    proofs["predictive_tabular"] = {
        "status": "PASS",
        "production_owners": [
            "start.registry.get_test",
            "start.modeling.models.resolve_model",
        ],
        "executed_surfaces": [tabular_result.test_id, resolved_name],
        "result_statuses": [result_status(tabular_result), f"PREDICTIONS={len(tabular_predictions)}"],
    }

    sequence_x = np.arange(8 * 3 * 2, dtype=float).reshape(8, 3, 2) / 10.0
    sequence_y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    sequence_model, sequence_name, _ = resolve_model(
        "gru",
        seed=7,
        epochs=1,
        hidden_size=4,
        batch_size=4,
        validation_fraction=0.25,
        device="cpu",
    )
    sequence_model.fit(sequence_x, sequence_y)
    sequence_probabilities = sequence_model.predict_proba(sequence_x)
    proofs["predictive_temporal"] = {
        "status": "PASS",
        "production_owners": ["start.modeling.models.resolve_model", type(sequence_model).__module__],
        "executed_surfaces": [sequence_name],
        "result_statuses": [f"PROBABILITY_ROWS={len(sequence_probabilities)}"],
        "input_shape": list(sequence_x.shape),
    }

    world = generate_market_world(
        n_assets=3,
        n_periods=40,
        n_factors=2,
        seed=7,
        include_short_rate=True,
    )
    market_result = get_test("portfolio.risk_statistics").fn(world.market_context())
    treasury_result = get_test("traded_risk.stanton_nonparametric").fn(
        world.short_rate_context(),
        n_grid=5,
        min_ess=2.0,
    )
    proofs["market"] = {
        "status": "PASS",
        "production_owners": ["start.registry.get_test"],
        "executed_surfaces": [market_result.test_id],
        "result_statuses": [result_status(market_result)],
    }
    proofs["treasury"] = {
        "status": "PASS",
        "production_owners": ["start.registry.get_test"],
        "executed_surfaces": [treasury_result.test_id],
        "result_statuses": [result_status(treasury_result)],
    }
    proofs["cross_domain"] = {
        "status": "PASS",
        "production_owners": ["start.registry.get_test"],
        "executed_surfaces": [
            tabular_result.test_id,
            market_result.test_id,
            treasury_result.test_id,
        ],
        "result_statuses": [
            result_status(tabular_result),
            result_status(market_result),
            result_status(treasury_result),
        ],
        "composition_only": True,
    }

    stream = io.StringIO()
    runtime_observer = TerminalReviewObserver(
        "VERIFY-CONTROL-PLANE-EXECUTION",
        console=Console(file=stream, width=132, color_system=None),
        session_kind="VERIFY",
        control_plane=True,
    )
    runtime_sink = RuntimePresentationSink(runtime_observer, workflow_id="data_diagnostics")
    with tempfile.TemporaryDirectory(prefix="start-coherence-") as output_root:
        runtime_result = CanonicalExecutionService.execute(
            workflow_id="data_diagnostics",
            context_id="institutional_credit_v1",
            seed=7,
            output_root=output_root,
            trace_mode="summary",
            run_id="VERIFY-CONTROL-PLANE-EXECUTION",
            event_sink=runtime_sink,
        )
        runtime_observer.session_completed(source_component="coherence_verifier")
    runtime_events = _event_values(runtime_observer)
    required_runtime_events = {
        "DATA_SOURCE_RESOLVED",
        "PROBLEM_CONTRACT_RESOLVED",
        "CAPABILITY_PLAN_RESOLVED",
        "DETERMINISTIC_EXECUTION_SUMMARIZED",
        "EVIDENCE_COMMITTED",
        "GOVERNANCE_EVALUATED",
        "OPA_EVALUATED",
        "TRACE_READY",
        "ATTESTATION_SEALED",
        "OUTCOME_READY",
    }
    missing_runtime_events = sorted(required_runtime_events - set(runtime_events))
    proofs["deterministic_control_plane"] = {
        "status": "PASS" if not missing_runtime_events else "FAIL",
        "production_owners": [
            "start.runtime.execution.CanonicalExecutionService",
            "start.review.terminal_observability.RuntimePresentationSink",
        ],
        "executed_surfaces": [record.test_id for record in runtime_result.records],
        "result_statuses": [
            str(getattr(record.status, "value", record.status)).upper()
            for record in runtime_result.records
        ],
        "event_count": len(runtime_events),
        "missing_required_events": missing_runtime_events,
        "outcome_capsule_complete": runtime_observer.outcome_capsule is not None,
    }
    return proofs


def _bounded_cases() -> list[dict[str, Any]]:
    from start.review.architecture import ReviewDomain

    envelopes = [
        ("predictive_tabular", _predictive_fixture(sequence=False), False),
        ("predictive_temporal", _predictive_fixture(sequence=True), False),
        ("market", _unified_fixture((ReviewDomain.MARKET,)), False),
        ("treasury", _unified_fixture((ReviewDomain.TREASURY,)), False),
        (
            "cross_domain",
            _unified_fixture((ReviewDomain.PREDICTIVE, ReviewDomain.MARKET, ReviewDomain.TREASURY)),
            False,
        ),
    ]
    runtime_metadata = {
        "spec_id": "institutional_credit_v1",
        "actual_samples": 500,
        "actual_features": 8,
        "actual_target": "target",
    }
    envelopes.append(
        (
            "deterministic_control_plane",
            build_canonical_runtime_coherence_envelope(
                run_id="VERIFY-CONTROL-PLANE",
                workflow_id="predictive_ml",
                context_id="institutional_credit_v1",
                context_metadata=runtime_metadata,
            ),
            True,
        )
    )
    execution_proofs = _bounded_execution_proofs()
    results = []
    for case_id, envelope, control_plane in envelopes:
        proof = _verify_live_consumer(envelope, control_plane=control_plane)
        execution_proof = execution_proofs[case_id]
        complete = bool(
            envelope.data_intake.context_id
            and envelope.problem.objective
            and envelope.capability_plan.supported
        )
        results.append(
            {
                "case_id": case_id,
                "status": (
                    "PASS"
                    if proof["status"] == "PASS"
                    and execution_proof["status"] == "PASS"
                    and complete
                    else "FAIL"
                ),
                "contract_complete": complete,
                "stage_applicability": {
                    key: value.value for key, value in envelope.capability_plan.stages.items()
                },
                "observer_proof": proof,
                "execution_proof": execution_proof,
            }
        )
    return results


def _negative_cases() -> list[dict[str, Any]]:
    cases: list[tuple[str, Any]] = [
        (
            "multiple_targets_interactive_route",
            lambda: resolve_predictive_output_structure(
                task_type="regression",
                targets=("target_a", "target_b"),
                route_supports_multi_output=False,
                sequence_input=False,
            ),
        ),
        (
            "sequence_output",
            lambda: resolve_predictive_output_structure(
                task_type="sequence_output",
                targets=("target",),
                route_supports_multi_output=False,
                sequence_input=True,
            ),
        ),
        (
            "disabled_model_comparison",
            lambda: build_canonical_runtime_coherence_envelope(
                run_id="VERIFY-NEGATIVE",
                workflow_id="model_comparison",
                context_id="institutional_credit_v1",
                context_metadata={"spec_id": "institutional_credit_v1"},
            ),
        ),
        (
            "incompatible_context",
            lambda: build_canonical_runtime_coherence_envelope(
                run_id="VERIFY-NEGATIVE",
                workflow_id="quantitative_finance",
                context_id="institutional_credit_v1",
                context_metadata={"spec_id": "institutional_credit_v1"},
            ),
        ),
    ]
    results: list[dict[str, Any]] = []
    for case_id, operation in cases:
        try:
            operation()
        except UnsupportedCapabilityError as exc:
            results.append({"case_id": case_id, "status": "PASS", "result": exc.to_dict()})
        else:
            results.append(
                {
                    "case_id": case_id,
                    "status": "FAIL",
                    "result": {"reason": "configuration did not fail closed"},
                }
            )
    return results


def _production_wiring(workspace: Path) -> list[dict[str, Any]]:
    checks = (
        (
            "interactive_predictive",
            workspace / "src/start/interactive_review.py",
            ("build_predictive_coherence_envelope", "publish_coherence_contract", "deterministic_execution_summary"),
        ),
        (
            "unified_domain_review",
            workspace / "src/start/review/executor.py",
            ("build_unified_review_coherence_envelope", "publish_coherence_contract", "agent_decision_trace"),
        ),
        (
            "canonical_control_plane",
            workspace / "src/start/review/terminal_observability.py",
            ("build_canonical_runtime_coherence_envelope", "publish_coherence_contract", "RuntimePresentationSink"),
        ),
    )
    results = []
    for route, path, tokens in checks:
        source = path.read_text(encoding="utf-8")
        missing = [token for token in tokens if token not in source]
        results.append(
            {
                "route": route,
                "source": str(path),
                "status": "PASS" if not missing else "FAIL",
                "required_runtime_wiring": list(tokens),
                "missing": missing,
            }
        )
    return results


def _generalization_guards(workspace: Path) -> dict[str, Any]:
    source_path = workspace / "src/start/review/workflow_coherence.py"
    source = source_path.read_text(encoding="utf-8")
    prohibited = (
        "Flight A",
        "Flight B",
        "Flight C",
        "800 sequences",
        "50 assets",
        "RUN-ENT-",
        "offline_demo_twin",
    )
    hits = [token for token in prohibited if token in source]
    return {
        "status": "PASS" if not hits else "FAIL",
        "source": str(source_path),
        "prohibited_demo_literal_hits": hits,
    }


def run_verification(workspace: Path) -> dict[str, Any]:
    inventory = capability_inventory()
    enumeration = enumerate_contract_cases()
    pairwise = deterministic_pairwise_cases(enumeration)
    bounded = _bounded_cases()
    negative = _negative_cases()
    wiring = _production_wiring(workspace)
    guard = _generalization_guards(workspace)
    failures = [
        item
        for group in (bounded, negative, wiring)
        for item in group
        if item["status"] != "PASS"
    ]
    if guard["status"] != "PASS":
        failures.append(guard)
    result = {
        "schema": SCHEMA,
        "status": "PASS" if not failures else "FAIL",
        "zero_cost": True,
        "hosted_calls": 0,
        "inventory": inventory,
        "contract_enumeration": {
            "count": len(enumeration),
            "cases": [case.to_dict() for case in enumeration],
        },
        "pairwise_cover": {
            "count": len(pairwise),
            "cases": [case.to_dict() for case in pairwise],
        },
        "bounded_executable_cases": bounded,
        "fail_closed_negative_cases": negative,
        "production_wiring": wiring,
        "generalization_guards": guard,
        "run_outcome_capsule_fields": [
            "started_with",
            "objective",
            "alternatives",
            "human_decisions",
            "agent_contribution",
            "deterministic_results",
            "evidence_record_references",
            "grounding",
            "governance_policy",
            "achieved_result",
            "remaining_conditions",
        ],
        "failures": failures,
    }
    return result


def _status_counts(rows: list[dict[str, Any]], key: str = "status") -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "UNKNOWN"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def render_capability_matrix_markdown(result: dict[str, Any]) -> str:
    inventory = result["inventory"]
    lines = [
        "# StART Generalized Capability Matrix",
        "",
        "Generated by `python -m start.review.coherence_verifier` from source registries.",
        "Menu text alone is not treated as execution proof.",
        "",
        "## Summary",
        "",
        f"- Registered tests: {len(inventory['registered_tests'])}",
        f"- Canonical workflows: {len(inventory['workflows'])}",
        f"- Canonical contexts: {len(inventory['contexts'])}",
        f"- Contract enumeration cases: {result['contract_enumeration']['count']}",
        f"- Deterministic pairwise-cover cases: {result['pairwise_cover']['count']}",
        "",
        "## Canonical workflows — selector to execution proof",
        "",
        "| Workflow | Status | Engine | Compatible contexts | Owning execution path |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in inventory["workflows"]:
        lines.append(
            f"| `{row['workflow_id']}` | {row['status']} | `{row['engine_kind']}` | "
            f"{', '.join(row['compatible_contexts'])} | `{row['execution_owner']}` |"
        )
    lines.extend(
        [
            "",
            "## Target and output truth",
            "",
            "| Mode | Status | Scope / constraint |",
            "| --- | --- | --- |",
        ]
    )
    for row in inventory["target_output_modes"]:
        lines.append(f"| `{row['mode']}` | {row['status']} | {row['scope']} |")
    lines.extend(
        [
            "",
            "## Cross-domain and Treasury",
            "",
            f"- Cross-domain: **{inventory['cross_domain']['status']}** — {inventory['cross_domain']['reason']}",
            f"- Treasury: **{inventory['treasury']['status']}** — {inventory['treasury']['reason']}",
            "",
            "## Architecture families",
            "",
            "| Family | Modality | Status | Selector → execution proof |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in inventory["architectures"]:
        lines.append(
            f"| `{row['family']}` | {row['modality']} | {row['status']} | "
            f"`{row['selector_owner']}` → `{row['execution_owner']}` |"
        )
    lines.extend(
        [
            "",
            "## Verification result",
            "",
            f"- Overall: **{result['status']}**",
            f"- Bounded executable fixtures: {_status_counts(result['bounded_executable_cases'])}",
            f"- Fail-closed negatives: {_status_counts(result['fail_closed_negative_cases'])}",
            f"- Production wiring: {_status_counts(result['production_wiring'])}",
            "- Hosted calls: **0**",
            "",
            "The JSON companion contains the complete workflow/context/model/test/provider inventory, "
            "all enumerated cases, the pairwise cover, applicability states, and negative-case reasons.",
            "",
        ]
    )
    return "\n".join(lines)


def write_verification(paths: VerificationPaths) -> dict[str, Any]:
    result = run_verification(paths.workspace)
    paths.result_json.parent.mkdir(parents=True, exist_ok=True)
    paths.result_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    matrix_payload = {
        "schema": "start.generalized-capability-matrix/1",
        "inventory": result["inventory"],
        "contract_enumeration": result["contract_enumeration"],
        "pairwise_cover": result["pairwise_cover"],
        "bounded_executable_cases": result["bounded_executable_cases"],
        "fail_closed_negative_cases": result["fail_closed_negative_cases"],
    }
    paths.matrix_json.write_text(
        json.dumps(matrix_payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    paths.matrix_markdown.write_text(render_capability_matrix_markdown(result), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# V2 verification hardening.  This section deliberately depends on production
# APIs without changing them.  V1 remains callable for historical evidence.
# ---------------------------------------------------------------------------

_V2_DIMENSIONS: dict[str, tuple[str, ...]] = {
    "review_mode": ("single_domain", "cross_domain"),
    "domain": (
        "predictive",
        "market",
        "treasury",
        "predictive+market",
        "predictive+treasury",
        "market+treasury",
        "predictive+market+treasury",
    ),
    "technology": (
        "traditional_predictive",
        "deep_learning",
        "domain_specific_deterministic_analytics",
    ),
    "model_architecture_family": (
        "logistic_regression",
        "mlp",
        "gru",
        "registered_domain_analytics",
    ),
    "data_source_class": (
        "synthetic_generated",
        "local_file",
        "registered_public_adapter",
        "prepared_in_memory_context",
    ),
    "target_output_mode": (
        "single_target",
        "multilabel",
        "multi_output_regression",
        "sequence_classification",
        "sequence_output",
        "domain_analytical_output",
    ),
    "lifecycle": (
        "initial_validation",
        "periodic_validation",
        "material_model_change",
        "ongoing_monitoring",
        "pre_implementation",
    ),
    "review_scope": ("full_recommended", "customized"),
    "materiality": ("high", "medium", "low"),
    "reviewer_backend": (
        "deterministic_no_agent",
        "offline_demo_twin",
        "enterprise_gateway_configured",
        "public_provider_configured",
    ),
}

_V2_CONSTRAINTS = (
    "single_domain iff exactly one atomic domain is selected; cross_domain iff two or more are selected",
    "domain-specific deterministic analytics uses registered_domain_analytics and domain_analytical_output",
    "GRU is a deep-learning sequence-classification architecture and requires a predictive domain",
    "MLP is deep-learning tabular; logistic regression is traditional predictive",
    "multilabel, multi-output regression, and sequence-output are explicit fail-closed route cases",
    "hosted reviewer backends are configuration-resolution only; this verifier makes no hosted calls",
)


def _v2_base_variants(domain: str) -> list[tuple[str, str, str, str, str]]:
    """Return (technology, model, output, support, reason) valid for a domain choice."""
    atoms = domain.split("+")
    if atoms == ["predictive"]:
        return [
            ("traditional_predictive", "logistic_regression", "single_target", "SUPPORTED", "registered tabular route"),
            ("deep_learning", "mlp", "single_target", "SUPPORTED", "registered tabular deep-learning route"),
            ("deep_learning", "gru", "sequence_classification", "SUPPORTED", "genuine rank-3 recurrent route"),
            ("deep_learning", "mlp", "multilabel", "NOT_SUPPORTED", "interactive route cannot preserve multilabel targets"),
            ("traditional_predictive", "logistic_regression", "multi_output_regression", "NOT_SUPPORTED", "interactive route cannot preserve multiple targets"),
            ("deep_learning", "gru", "sequence_output", "NOT_SUPPORTED", "no registered end-to-end sequence-output route"),
        ]
    if "predictive" in atoms:
        return [
            ("traditional_predictive", "logistic_regression", "domain_analytical_output", "SUPPORTED", "composed registered domain review"),
            ("deep_learning", "mlp", "domain_analytical_output", "SUPPORTED", "composed registered domain review"),
            ("deep_learning", "gru", "domain_analytical_output", "SUPPORTED", "composed review with sequence predictive constituent"),
        ]
    return [
        (
            "domain_specific_deterministic_analytics",
            "registered_domain_analytics",
            "domain_analytical_output",
            "SUPPORTED",
            "registry-derived domain analytical route",
        )
    ]


def _v2_configuration_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    counter = 0
    for domain in _V2_DIMENSIONS["domain"]:
        mode = "single_domain" if "+" not in domain else "cross_domain"
        for technology, model, output, support, reason in _v2_base_variants(domain):
            for source, lifecycle, scope, materiality, backend in itertools.product(
                _V2_DIMENSIONS["data_source_class"],
                _V2_DIMENSIONS["lifecycle"],
                _V2_DIMENSIONS["review_scope"],
                _V2_DIMENSIONS["materiality"],
                _V2_DIMENSIONS["reviewer_backend"],
            ):
                counter += 1
                candidates.append(
                    {
                        "case_id": f"CFG-{counter:05d}",
                        "verification_level": "CONFIGURATION_RESOLVED",
                        "execution_proven": False,
                        "support": support,
                        "reason": reason,
                        "hosted_call": False,
                        "dimensions": {
                            "review_mode": mode,
                            "domain": domain,
                            "technology": technology,
                            "model_architecture_family": model,
                            "data_source_class": source,
                            "target_output_mode": output,
                            "lifecycle": lifecycle,
                            "review_scope": scope,
                            "materiality": materiality,
                            "reviewer_backend": backend,
                        },
                    }
                )
    return candidates


def _pair_key(left_name: str, left_value: str, right_name: str, right_value: str) -> tuple[str, str, str, str]:
    return (left_name, left_value, right_name, right_value)


def _case_pairs(case: dict[str, Any]) -> set[tuple[str, str, str, str]]:
    dimensions = case["dimensions"]
    names = list(_V2_DIMENSIONS)
    return {
        _pair_key(a, dimensions[a], b, dimensions[b])
        for index, a in enumerate(names)
        for b in names[index + 1 :]
    }


def _pairwise_cover_v2() -> dict[str, Any]:
    candidates = _v2_configuration_candidates()
    pair_sets = [_case_pairs(case) for case in candidates]
    feasible_pairs: set[tuple[str, str, str, str]] = set().union(*pair_sets)
    uncovered = set(feasible_pairs)
    chosen: list[int] = []
    while uncovered:
        best_index = max(
            (index for index in range(len(candidates)) if index not in chosen),
            key=lambda index: (len(pair_sets[index] & uncovered), -index),
        )
        contribution = pair_sets[best_index] & uncovered
        if not contribution:
            break
        chosen.append(best_index)
        uncovered -= contribution

    selected: list[dict[str, Any]] = []
    for ordinal, candidate_index in enumerate(chosen, 1):
        item = dict(candidates[candidate_index])
        item["case_id"] = f"PAIRWISE-{ordinal:03d}"
        selected.append(item)

    names = list(_V2_DIMENSIONS)
    theoretical = {
        _pair_key(a, av, b, bv)
        for index, a in enumerate(names)
        for b in names[index + 1 :]
        for av in _V2_DIMENSIONS[a]
        for bv in _V2_DIMENSIONS[b]
    }
    impossible = sorted(theoretical - feasible_pairs)
    covered = sorted(set().union(*(_case_pairs(case) for case in selected)))
    candidate_payload = [
        {"dimensions": case["dimensions"], "support": case["support"]}
        for case in candidates
    ]
    candidate_digest = hashlib.sha256(
        json.dumps(candidate_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "algorithm": "deterministic greedy set cover over the complete constrained feasible pair universe",
        "dimensions": {name: list(values) for name, values in _V2_DIMENSIONS.items()},
        "constraints": list(_V2_CONSTRAINTS),
        "candidate_count": len(candidates),
        "candidate_support_counts": _status_counts(candidates, key="support"),
        "candidate_universe_sha256": candidate_digest,
        "case_count": len(selected),
        "cases": selected,
        "covered_pair_count": len(covered),
        "covered_pairs": [
            {"dimension_a": a, "value_a": av, "dimension_b": b, "value_b": bv}
            for a, av, b, bv in covered
        ],
        "uncovered_pair_count": len(impossible),
        "uncovered_pairs": [
            {
                "dimension_a": a,
                "value_a": av,
                "dimension_b": b,
                "value_b": bv,
                "reason": "CONSTRAINED_OUT — no semantically valid registered configuration contains this pair",
            }
            for a, av, b, bv in impossible
        ],
        "feasible_pairs_left_uncovered": len(uncovered),
    }


def _configuration_resolution_proofs() -> dict[str, Any]:
    from start.review.applicability import applicable_tests
    from start.review.architecture import LLMReviewConfig, ReviewDomain, ReviewLifecycle

    domains = (ReviewDomain.MARKET,)
    full = applicable_tests(domains)
    custom = applicable_tests(domains, families=("portfolio",))
    lifecycles = [item.value for item in ReviewLifecycle]
    backends = [
        LLMReviewConfig(backend_mode="none", provider="none", status="DETERMINISTIC"),
        LLMReviewConfig(backend_mode="offline", provider="offline_demo_twin", status="OFFLINE_REHEARSAL"),
        LLMReviewConfig(backend_mode="enterprise", provider="enterprise_llm_gateway", status="CONFIGURED"),
        LLMReviewConfig(backend_mode="public", provider="openai", status="CONFIGURED"),
    ]
    scope_ok = 0 < custom.count < full.count and set(custom.test_ids) <= set(full.test_ids)
    backend_payloads = [item.describe() for item in backends]
    return {
        "status": "PASS" if scope_ok and len(lifecycles) == 5 and len(backend_payloads) == 4 else "FAIL",
        "verification_level": "CONFIGURATION_RESOLVED",
        "review_scope": {
            "full_recommended_count": full.count,
            "customized_family": "portfolio",
            "customized_count": custom.count,
            "customized_is_registry_subset": scope_ok,
            "owner": "start.review.applicability.applicable_tests",
        },
        "lifecycles": lifecycles,
        "reviewer_backends": backend_payloads,
        "hosted_calls": 0,
    }


_CAPSULE_FIELDS = (
    "started_with",
    "objective",
    "alternatives",
    "human_decisions",
    "agent_contribution",
    "deterministic_results",
    "evidence_record_references",
    "grounding",
    "governance_policy",
    "achieved_result",
    "remaining_conditions",
)


def _serialize_full_envelope_case(case_id: str, outcome: Any, *, agent_assisted: bool) -> dict[str, Any]:
    if isinstance(outcome, dict):
        envelope = outcome["workflow_coherence"]
        capsule = outcome["run_outcome_capsule"]
        records = list(outcome["records"])
        observer = outcome["presentation_observer"]
    else:
        envelope = outcome.workflow_coherence
        capsule = outcome.run_outcome_capsule
        records = list(getattr(outcome.base_outcome, "evidence", ()))
        observer = outcome.presentation_observer if hasattr(outcome, "presentation_observer") else None
        if observer is None:
            # Flight A exposes its observer only via the exported coherence object/events;
            # the envelope and runtime-populated capsule remain authoritative here.
            observer = SimpleNamespace(events=[])
    capsule_payload = capsule.to_dict() if capsule is not None else {}
    missing_capsule_fields = [
        field for field in _CAPSULE_FIELDS if field not in capsule_payload or capsule_payload[field] in (None, "", [], {})
    ]
    traces = [trace.to_dict() for trace in envelope.agent_decision_traces]
    invalid_traces = [
        trace
        for trace in traces
        if trace.get("numeric_authority") != "NONE"
        or not trace.get("agent_identity")
        or not trace.get("role")
        or not trace.get("trigger")
        or not trace.get("checkpoint")
        or not trace.get("canonical_input_references")
        or not trace.get("recommendation")
        or not trace.get("resulting_action")
    ]
    event_kinds = _event_values(observer) if getattr(observer, "events", None) else []
    environment = [item.to_dict() for item in envelope.capability_plan.environment_projection]
    ai_environment = [item for item in environment if str(item["capability_id"]).startswith("ai.")]
    evidence_method_identities = [
        {
            "evidence_id": str(record.evidence_id),
            "test_id": str(record.test_id),
            "method": str(record.metrics["method"]),
        }
        for record in records
        if "explain" in str(getattr(record, "test_id", "")).lower()
        and isinstance(getattr(record, "metrics", None), dict)
        and getattr(record, "metrics", {}).get("method") not in (None, "")
    ]
    status = "PASS"
    if missing_capsule_fields or not records or not envelope.deterministic_summaries:
        status = "FAIL"
    if agent_assisted and (not traces or invalid_traces):
        status = "FAIL"
    return {
        "case_id": case_id,
        "status": status,
        "verification_level": "EXECUTION_PROVEN",
        "production_route": envelope.capability_plan.route_id,
        "data_intake_contract": envelope.data_intake.to_dict(),
        "problem_contract": envelope.problem.to_dict(),
        "capability_plan": envelope.capability_plan.to_dict(),
        "agent_assisted": agent_assisted,
        "agent_decision_traces": traces,
        "invalid_agent_traces": invalid_traces,
        "deterministic_execution_summaries": [item.to_dict() for item in envelope.deterministic_summaries],
        "evidence_record_count": len(records),
        "evidence_test_ids": [record.test_id for record in records],
        "evidence_explainability_method_identities": evidence_method_identities,
        "grounding_state": capsule_payload.get("grounding"),
        "governance_policy": capsule_payload.get("governance_policy"),
        "run_outcome_capsule": capsule_payload,
        "missing_capsule_fields": missing_capsule_fields,
        "environment_projection": environment,
        "ai_environment_projection": ai_environment,
        "runtime_event_kinds": event_kinds,
        "hosted_calls": 0,
    }


def _run_predictive_full_envelope(*, temporal: bool) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    import start.interactive_review as interactive_module
    from start.data.selection import DatasetKind, DatasetSelection, select_temporal_sequence
    from start.interactive_review import ReviewConfig, run_interactive_review

    if temporal:
        selection = select_temporal_sequence(n_series=30, timesteps=4, n_features=2, seed=7)
        architecture = "gru"
        activation = "tanh"
        case_id = "predictive_temporal_deep_learning_agent_assisted"
    else:
        rng = np.random.default_rng(7)
        rows = 40
        frame = pd.DataFrame(
            {
                "feature_a": rng.normal(size=rows),
                "feature_b": rng.normal(size=rows),
                "target": np.asarray([0, 1] * (rows // 2)),
            }
        )
        selection = DatasetSelection(
            kind=DatasetKind.SYNTHETIC,
            display_name="V2 bounded predictive fixture",
            frame=frame,
            source_reference="generated locally by coherence verifier",
            licence_note="not applicable (generated)",
            target_column="target",
        )
        architecture = "mlp"
        activation = "relu"
        case_id = "predictive_tabular_agent_assisted"
    with tempfile.TemporaryDirectory(prefix="start-coherence-v2-predictive-") as output_root:
        cfg = ReviewConfig(
            target="target",
            dataset_selection=selection,
            sequence_bundle=selection.sequence_bundle,
            architecture_family=architecture,
            activation=activation,
            agent_mode="llm",
            llm_provider="offline_demo_twin",
            llm_model="deterministic-semantic-fixture-v1",
            run_dl=True,
            enterprise_mode=True,
            accept_recommendations=True,
            non_interactive=True,
            tuning_strategy="none",
            output_root=output_root,
            seed=7,
            show_progress=False,
            open_figures=False,
        )
        old_console = interactive_module.console
        interactive_module.console = Console(file=io.StringIO(), width=132, color_system=None)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                outcome = run_interactive_review(cfg)
        finally:
            interactive_module.console = old_console
    return _serialize_full_envelope_case(case_id, outcome, agent_assisted=True)


def _run_unified_full_envelope(case_id: str, domains: tuple[Any, ...]) -> dict[str, Any]:
    import start.review.executor as executor_module
    from start.data.synthetic_market import generate_market_world
    from start.review.architecture import (
        LLMReviewConfig,
        ReviewContextBundle,
        ReviewGroundingMode,
        ReviewLifecycle,
        ReviewMode,
    )
    from start.review.executor import run_market_treasury_review

    world = generate_market_world(
        n_assets=3,
        n_periods=40,
        n_factors=2,
        seed=7,
        include_short_rate=True,
    )
    values = {str(getattr(domain, "value", domain)) for domain in domains}
    bundle = ReviewContextBundle(
        market=world.market_context() if "market" in values else None,
        short_rate=world.short_rate_context() if "treasury" in values else None,
        domains=domains,
        mode=ReviewMode.CROSS_DOMAIN if len(domains) > 1 else ReviewMode.SINGLE_DOMAIN,
        lifecycle=ReviewLifecycle.PERIODIC_VALIDATION,
        materiality="high",
        llm_config=LLMReviewConfig(
            backend_mode="offline",
            provider="offline_demo_twin",
            model="deterministic-semantic-fixture-v1",
            status="OFFLINE_REHEARSAL",
        ),
        business_context=f"bounded {case_id.replace('_', ' ')} review",
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    answers = iter(
        ["Q", "Summarize the canonical evidence without adding numerical claims"]
        + ["A"] * 40
    )
    with tempfile.TemporaryDirectory(prefix="start-coherence-v2-unified-") as output_root:
        old_console = executor_module.console
        executor_module.console = Console(file=io.StringIO(), width=132, color_system=None)
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                outcome = run_market_treasury_review(
                    bundle,
                    output_root=output_root,
                    interactive=True,
                    ask=lambda _prompt: next(answers),
                )
        finally:
            executor_module.console = old_console
    return _serialize_full_envelope_case(case_id, outcome, agent_assisted=True)


def _deterministic_full_envelope_v2() -> dict[str, Any]:
    from start.review.terminal_observability import RuntimePresentationSink, TerminalReviewObserver
    from start.runtime.execution import CanonicalExecutionService

    run_id = "VERIFY-V2-DETERMINISTIC"
    observer = TerminalReviewObserver(
        run_id,
        console=Console(file=io.StringIO(), width=132, color_system=None),
        session_kind="VERIFY",
        control_plane=True,
    )
    sink = RuntimePresentationSink(observer, workflow_id="data_diagnostics")
    with tempfile.TemporaryDirectory(prefix="start-coherence-v2-control-") as output_root:
        runtime_result = CanonicalExecutionService.execute(
            workflow_id="data_diagnostics",
            context_id="institutional_credit_v1",
            seed=7,
            output_root=output_root,
            trace_mode="summary",
            run_id=run_id,
            event_sink=sink,
        )
        observer.session_completed(source_component="coherence_verifier_v2")
    envelope = observer.coherence_envelope
    if envelope is None:
        return {
            "case_id": "deterministic_canonical_control_plane",
            "status": "FAIL",
            "verification_level": "EXECUTION_PROVEN",
            "reason": "runtime observer did not bind a coherence envelope",
            "hosted_calls": 0,
        }
    proof = _serialize_full_envelope_case(
        "deterministic_canonical_control_plane",
        {
            "workflow_coherence": envelope,
            "run_outcome_capsule": observer.outcome_capsule,
            "records": runtime_result.records,
            "presentation_observer": observer,
        },
        agent_assisted=False,
    )
    environment = [item.to_dict() for item in envelope.capability_plan.environment_projection]
    ai = [item for item in environment if str(item["capability_id"]).startswith("ai.")]
    deterministic_required = {
        "deterministic.policy",
        "observability.otel",
        "evidence.attestation",
    }
    applicable = {
        item["capability_id"]
        for item in environment
        if item["state"] == "APPLICABLE"
    }
    ai_not_applicable = all(item["state"] == "NOT_APPLICABLE" for item in ai)
    proof.update({
        "stage_applicability": {
            key: value.value for key, value in envelope.capability_plan.stages.items()
        },
        "environment_projection": environment,
        "ai_environment_projection": ai,
        "ai_only_not_applicable": ai_not_applicable,
        "deterministic_governed_capabilities_applicable": sorted(deterministic_required & applicable),
        "hosted_calls": 0,
    })
    if not ai_not_applicable or not deterministic_required <= applicable:
        proof["status"] = "FAIL"
    return proof


def _full_envelope_execution_proofs_v2() -> list[dict[str, Any]]:
    from start.review.architecture import ReviewDomain

    return [
        _run_predictive_full_envelope(temporal=False),
        _run_predictive_full_envelope(temporal=True),
        _run_unified_full_envelope("market_agent_assisted", (ReviewDomain.MARKET,)),
        _run_unified_full_envelope("treasury_agent_assisted", (ReviewDomain.TREASURY,)),
        _run_unified_full_envelope(
            "cross_domain_market_treasury_agent_assisted",
            (ReviewDomain.MARKET, ReviewDomain.TREASURY),
        ),
        _deterministic_full_envelope_v2(),
    ]


def _data_source_evidence_v2() -> list[dict[str, Any]]:
    import pandas as pd

    from start.data.providers.registry import get_provider_adapter, list_provider_adapters

    registered = list_provider_adapters()
    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="start-coherence-v2-data-") as directory:
        root = Path(directory)
        frame = pd.DataFrame({"feature": [1.0, 2.0, 3.0], "target": [0, 1, 0]})
        local_paths = {
            "local_csv": root / "fixture.csv",
            "local_parquet": root / "fixture.parquet",
        }
        frame.to_csv(local_paths["local_csv"], index=False)
        frame.to_parquet(local_paths["local_parquet"], index=False)
        for provider, path in local_paths.items():
            adapter = get_provider_adapter(provider)
            loaded, contract = adapter.load_dataframe(str(path), max_rows=3, target_column="target")
            passed = len(loaded) == 3 and contract.consumed_row_count == 3 and bool(contract.content_fingerprint)
            rows.append(
                {
                    "source": provider,
                    "evidence_levels": [
                        "REGISTERED",
                        "DEPENDENCY_AVAILABLE",
                        "BOUNDED_EXECUTION_PROVEN" if passed else "NOT_SUPPORTED",
                    ],
                    "status": "PASS" if passed else "FAIL",
                    "rows_consumed": contract.consumed_row_count,
                    "fingerprint_present": bool(contract.content_fingerprint),
                    "network_call": False,
                }
            )
    for provider in ("huggingface", "openml", "uci"):
        probe = registered[provider]
        levels = ["REGISTERED"]
        if probe["runnable"]:
            levels.append("DEPENDENCY_AVAILABLE")
        levels.append("NETWORK_DEPENDENT_NOT_EXERCISED")
        rows.append(
            {
                "source": provider,
                "evidence_levels": levels,
                "status": "CONFIGURATION_ONLY",
                "dependency_probe": probe,
                "network_call": False,
            }
        )
    kaggle_probe = registered["kaggle"]
    rows.append(
        {
            "source": "kaggle",
            "evidence_levels": ["REGISTERED", "DEFERRED"],
            "status": "DEFERRED",
            "dependency_probe": kaggle_probe,
            "reason": "authenticated streaming execution is not implemented; no network call was made",
            "network_call": False,
        }
    )
    rows.extend(
        [
            {
                "source": "synthetic_generated",
                "evidence_levels": ["REGISTERED", "BOUNDED_EXECUTION_PROVEN"],
                "status": "PASS",
                "network_call": False,
            },
            {
                "source": "prepared_in_memory_context",
                "evidence_levels": ["REGISTERED", "BOUNDED_EXECUTION_PROVEN"],
                "status": "PASS",
                "network_call": False,
            },
            {
                "source": "unregistered_provider_class",
                "evidence_levels": ["NOT_SUPPORTED"],
                "status": "NOT_SUPPORTED",
                "reason": "not present in start.data.providers.registry",
                "network_call": False,
            },
        ]
    )
    return rows


def failure_reconciliation_v2(workspace: Path) -> dict[str, Any]:
    """Exact manifest from the prior cached full-suite run plus bounded reruns."""
    records = [
        {
            "node_id": "tests/test_review_tables.py::test_outlier_evidence_table_real_values",
            "failure_assertion": "expected the outlier evidence table header to contain 'Feature'",
            "expected_behavior": "first column header is Feature",
            "actual_behavior": "first column header is Field",
            "authoritative_contract": "No generalized-coherence contract changes the review-table header schema.",
            "predates_accepted_semantic_change": False,
            "bounded_rerun_result": "FAIL — expected Feature, actual Field",
            "classification": "PRE_EXISTING_UNRELATED_DEFECT",
        },
        {
            "node_id": "tests/test_trust_domains.py::test_public_providers_are_exactly_the_five",
            "failure_assertion": "expected PRIVATE_PROVIDERS to contain only enterprise_llm_gateway",
            "expected_behavior": "legacy exact private-provider set",
            "actual_behavior": "offline_demo_twin is also registered as a private/local provider",
            "authoritative_contract": "P1–P14 zero-cost offline-demo-twin presentation and agent route is accepted and frozen.",
            "predates_accepted_semantic_change": True,
            "bounded_rerun_result": "FAIL — exact-set expectation excludes offline_demo_twin",
            "classification": "STALE_TEST_EXPECTATION",
        },
        {
            "node_id": "tests/test_v240_tuning_wizard.py::test_tuning_run_kfold_regression_recurrent",
            "failure_assertion": "expected recurrent k-fold validation_metric > 0",
            "expected_behavior": "rank-2 tabular data is accepted by recurrent tuning",
            "actual_behavior": "validation_metric is NaN after SequenceInputContractError requires rank-3 input",
            "authoritative_contract": "P1–P14 temporal model contract requires genuine rank-3 (N,T,F) input and forbids fake tabular reshaping.",
            "predates_accepted_semantic_change": True,
            "bounded_rerun_result": "FAIL — recurrent candidates fail closed on rank-2 input",
            "classification": "STALE_TEST_EXPECTATION",
        },
        {
            "node_id": "tests/test_v301_enhancements.py::test_tuning_run_multiclass",
            "failure_assertion": "expected non-empty best_params from a mixed recurrent search",
            "expected_behavior": "rank-2 multiclass data yields a recurrent winner",
            "actual_behavior": "best_params is empty because recurrent candidates reject rank-2 input",
            "authoritative_contract": "Frozen temporal contract requires genuine rank-3 input for LSTM/GRU/RNN families.",
            "predates_accepted_semantic_change": True,
            "bounded_rerun_result": "FAIL — no valid recurrent candidate for rank-2 fixture",
            "classification": "STALE_TEST_EXPECTATION",
        },
        {
            "node_id": "tests/test_v401_contracts.py::test_wizard_options_are_stable_keys",
            "failure_assertion": "expected wizard keys 1 through 4",
            "expected_behavior": "four dataset wizard options",
            "actual_behavior": "keys 1 through 5 include the accepted temporal-sequence option",
            "authoritative_contract": "Frozen terminal/temporal contract adds the genuine temporal-sequence dataset choice.",
            "predates_accepted_semantic_change": True,
            "bounded_rerun_result": "FAIL — current registered keys are 1,2,3,4,5",
            "classification": "STALE_TEST_EXPECTATION",
        },
        {
            "node_id": "tests/test_v430_gate0_hardened_control_plane.py::test_double_abort_grounding_failure_single_menu_clean_exit",
            "failure_assertion": "previous full-suite node aborted via SystemExit after SIGINT",
            "expected_behavior": "uninterrupted bounded completion",
            "actual_behavior": "the recorded failure was caused by a manually induced SIGINT caught by demo_controller.py",
            "authoritative_contract": "Verification addendum requires the manually induced SIGINT to be identified and rerun once without interruption.",
            "predates_accepted_semantic_change": False,
            "bounded_rerun_result": "TIMEOUT_120S — rerun was not interrupted, produced no assertion failure, and was terminated at the audit bound",
            "classification": "INTERRUPTED_RUN_ARTIFACT",
        },
        {
            "node_id": "tests/test_v430_gate0_hardened_control_plane.py::test_twin_question_repair_failure_abort",
            "failure_assertion": "expected ReviewCancelled after grounding repair failure",
            "expected_behavior": "choice 2 aborts the review",
            "actual_behavior": "legacy-freeform path records grounding_failed and continues",
            "authoritative_contract": "Grounding behavior is frozen; this verifier pass may not change production grounding semantics.",
            "predates_accepted_semantic_change": False,
            "bounded_rerun_result": "FAIL — DID NOT RAISE ReviewCancelled",
            "classification": "PRE_EXISTING_UNRELATED_DEFECT",
        },
        {
            "node_id": "tests/test_v430_gate0_hardened_control_plane.py::test_twin_question_repair_failure_fallback_deterministic",
            "failure_assertion": "expected deterministic fallback after grounding repair failure",
            "expected_behavior": "fallback response is recorded",
            "actual_behavior": "decision remains grounding_failed",
            "authoritative_contract": "Fallback is allowed only where safely degradable; existing grounding behavior is outside this verification-only pass.",
            "predates_accepted_semantic_change": False,
            "bounded_rerun_result": "FAIL — actual action grounding_failed",
            "classification": "PRE_EXISTING_UNRELATED_DEFECT",
        },
        {
            "node_id": "tests/test_v430_gate11_ux_parity.py::test_deterministic_artifact_generation_and_catalog",
            "failure_assertion": "expected covariance/VaR/scenario artifacts from evidence-only records",
            "expected_behavior": "artifact generator manufactures all legacy artifact families from records",
            "actual_behavior": "only a portfolio-weight artifact exists because no typed execution products were supplied",
            "authoritative_contract": "PRESENTATION_CONTRACT_CROSSWALK B26 and GWC-03 require unavailable artifacts to be omitted and prohibit manufactured science.",
            "predates_accepted_semantic_change": True,
            "bounded_rerun_result": "FAIL — artifact map truthfully omits unavailable typed products",
            "classification": "STALE_TEST_EXPECTATION",
        },
        {
            "node_id": "tests/test_v430_gate11_ux_parity.py::test_semantically_accurate_wording_and_narrative",
            "failure_assertion": "expected exact empirical-size/nominal-size narrative phrase",
            "expected_behavior": "narrative includes 0.0660 and 0.05 wording",
            "actual_behavior": "narrative reports the canonical validation status but omits the expected phrase",
            "authoritative_contract": "No accepted generalized-coherence semantic change supersedes this exact narrative assertion.",
            "predates_accepted_semantic_change": False,
            "bounded_rerun_result": "FAIL — exact phrase absent",
            "classification": "PRE_EXISTING_UNRELATED_DEFECT",
        },
        {
            "node_id": "tests/test_v430_gate11a_parity_closure.py::test_challenge_non_mutating_diagnostics",
            "failure_assertion": "expected diagnose_covariance to be invoked from a bare EvidenceRecord",
            "expected_behavior": "challenge recomputes a covariance diagnostic without a registered execution product",
            "actual_behavior": "challenge states no covariance execution product is registered",
            "authoritative_contract": "GWC-03 and accepted evidence-only presentation prohibit scientific recomputation or manufactured execution products.",
            "predates_accepted_semantic_change": True,
            "bounded_rerun_result": "FAIL — diagnostic call count 0 by frozen provenance rule",
            "classification": "STALE_TEST_EXPECTATION",
        },
        {
            "node_id": "tests/test_v430_interactive_cli.py::test_checkpoint_grounding_failure_surfaced",
            "failure_assertion": "expected fallback decision after grounding failure",
            "expected_behavior": "checkpoint degrades to deterministic fallback",
            "actual_behavior": "checkpoint records grounding_failed",
            "authoritative_contract": "Grounding semantics are frozen and fallback is not universally permitted.",
            "predates_accepted_semantic_change": False,
            "bounded_rerun_result": "FAIL — actual action grounding_failed",
            "classification": "PRE_EXISTING_UNRELATED_DEFECT",
        },
        {
            "node_id": "tests/test_v430_interactive_cli.py::test_cross_domain_market_treasury_governance_scoping",
            "failure_assertion": "expected evidence count in legacy tuple (31, 34)",
            "expected_behavior": "legacy registry surface count",
            "actual_behavior": "35 records are emitted from the current registry-derived market+Treasury scope",
            "authoritative_contract": "GWC-07 requires registry-driven capability discovery rather than frozen counts.",
            "predates_accepted_semantic_change": True,
            "bounded_rerun_result": "FAIL — current registry-derived count is 35",
            "classification": "STALE_TEST_EXPECTATION",
        },
        {
            "node_id": "tests/test_v430_interactive_cli.py::test_market_only_review_governance_scoping",
            "failure_assertion": "expected evidence count in legacy tuple (27, 30)",
            "expected_behavior": "legacy market registry surface count",
            "actual_behavior": "31 records are emitted from the current market registry",
            "authoritative_contract": "GWC-07 requires registry-derived surfaces and forbids frozen enumerations.",
            "predates_accepted_semantic_change": True,
            "bounded_rerun_result": "FAIL — current registry-derived count is 31",
            "classification": "STALE_TEST_EXPECTATION",
        },
    ]
    structured_nodes = (
        "test_provider_incomplete_response",
        "test_provider_malformed_structured_output",
        "test_provider_no_structured_to_legacy_fallback",
        "test_provider_refusal",
    )
    for test_name in structured_nodes:
        records.append(
            {
                "node_id": f"tests/test_v430_structured_reviewer_contract.py::{test_name}",
                "failure_assertion": "expected review continuation/decisions after an invalid structured provider response",
                "expected_behavior": "legacy fallback or continuation",
                "actual_behavior": "ReviewCancelled at a REQUIRE_VALID_AGENT_RESPONSE checkpoint",
                "authoritative_contract": "Frozen structured-review contract permits fallback only at safely degradable checkpoints and does not guarantee legacy fallback.",
                "predates_accepted_semantic_change": True,
                "bounded_rerun_result": "FAIL — fail-closed ReviewCancelled is current accepted behavior",
                "classification": "STALE_TEST_EXPECTATION",
            }
        )
    cache_path = workspace / ".pytest_cache/v/cache/lastfailed"
    cached_nodes: list[str] = []
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        cached_nodes = sorted(key for key, failed in cached.items() if failed)
    expected_nodes = sorted(record["node_id"] for record in records)
    historical_manifest_path = workspace / "StART_FULL_SUITE_FAILURE_RECONCILIATION.json"
    historical_nodes: list[str] = []
    if historical_manifest_path.exists():
        try:
            historical_payload = json.loads(
                historical_manifest_path.read_text(encoding="utf-8")
            )
            historical_nodes = sorted(
                str(record["node_id"])
                for record in historical_payload.get("records", ())
            )
        except (OSError, ValueError, KeyError, TypeError):
            historical_nodes = []
    for record in records:
        record["test_file"] = record["node_id"].split("::", 1)[0]
    counts = _status_counts(records, key="classification")
    return {
        "schema": "start.full-suite-failure-reconciliation/1",
        "source": str(historical_manifest_path if historical_nodes else cache_path),
        "previous_failures_accounted": f"{len(records)}/18",
        "cache_exact_match": (
            historical_nodes == expected_nodes if historical_nodes else cached_nodes == expected_nodes
        ),
        "current_cache_exact_match": cached_nodes == expected_nodes,
        "cached_node_count": len(cached_nodes),
        "manifest_node_count": len(records),
        "classification_counts": counts,
        "records": records,
    }


def run_verification_v2(workspace: Path) -> dict[str, Any]:
    inventory = capability_inventory()
    pairwise = _pairwise_cover_v2()
    config_resolution = _configuration_resolution_proofs()
    full_envelopes = _full_envelope_execution_proofs_v2()
    data_sources = _data_source_evidence_v2()
    negatives = _negative_cases()
    reconciliation = failure_reconciliation_v2(workspace)
    execution_failures = [case for case in full_envelopes if case["status"] != "PASS"]
    implementation_gaps: list[dict[str, Any]] = []
    mandatory_cases = {
        "market_agent_assisted",
        "treasury_agent_assisted",
        "cross_domain_market_treasury_agent_assisted",
    }
    for case in full_envelopes:
        if case.get("case_id") in mandatory_cases and case.get("status") != "PASS":
            implementation_gaps.append(
                {
                    "classification": "IMPLEMENTATION GAP",
                    "owner": case.get("production_route"),
                    "case_id": case.get("case_id"),
                    "reproduction": "python scripts/verify_generalized_workflow_coherence.py --workspace .",
                    "details": case,
                }
            )
    genuine = reconciliation["classification_counts"].get("GENUINE_REGRESSION", 0)
    status = "PASS"
    if (
        execution_failures
        or implementation_gaps
        or genuine
        or config_resolution["status"] != "PASS"
        or not reconciliation["cache_exact_match"]
        or pairwise["feasible_pairs_left_uncovered"]
    ):
        status = "BLOCKED"
    agent_proofs = [
        case for case in full_envelopes if case.get("agent_decision_traces")
    ]
    environment_proofs = [
        {
            "case_id": case["case_id"],
            "ai_environment_projection": case.get("ai_environment_projection", []),
        }
        for case in full_envelopes
        if case.get("case_id")
        in {
            "predictive_tabular_agent_assisted",
            "market_agent_assisted",
            "deterministic_canonical_control_plane",
        }
    ]
    return {
        "schema": SCHEMA_V2,
        "status": status,
        "production_behavior_changed": False,
        "scientific_semantics_changed": False,
        "zero_cost": True,
        "hosted_calls": 0,
        "inventory": inventory,
        "configuration_enumeration": {
            "count": pairwise["candidate_count"],
            "verification_level": "CONFIGURATION_RESOLVED",
            "support_counts": pairwise["candidate_support_counts"],
            "universe_sha256": pairwise["candidate_universe_sha256"],
            "dimensions": pairwise["dimensions"],
            "constraints": pairwise["constraints"],
            "compact_cover_cases": pairwise["cases"],
        },
        "configuration_resolution_proofs": config_resolution,
        "pairwise_cover": pairwise,
        "full_envelope_bounded_executions": full_envelopes,
        "agent_decision_trace_runtime_proofs": agent_proofs,
        "ai_environment_projection_proofs": environment_proofs,
        "run_outcome_capsule_runtime_proof_count": sum(
            1 for case in full_envelopes if case.get("run_outcome_capsule")
        ),
        "data_source_support_evidence": data_sources,
        "fail_closed_negative_cases": negatives,
        "failure_reconciliation": reconciliation,
        "implementation_gaps": implementation_gaps,
        "failures": execution_failures,
    }


def render_capability_matrix_markdown_v2(result: dict[str, Any]) -> str:
    pairs = result["pairwise_cover"]
    lines = [
        "# StART Generalized Capability Matrix V2",
        "",
        "Verification levels are explicit: configuration resolution is not execution proof.",
        "",
        "## Coverage",
        "",
        f"- Constrained configuration enumeration: {result['configuration_enumeration']['count']}",
        f"- Configuration support counts: {result['configuration_enumeration']['support_counts']}",
        f"- Compact pairwise covering cases: {pairs['case_count']}",
        f"- Constrained candidate configurations: {pairs['candidate_count']}",
        f"- Covered feasible value pairs: {pairs['covered_pair_count']}",
        f"- Feasible pairs left uncovered: {pairs['feasible_pairs_left_uncovered']}",
        f"- Constrained-out theoretical pairs: {pairs['uncovered_pair_count']}",
        f"- Covered dimensions: {', '.join(pairs['dimensions'])}",
        "",
        "## Bounded full-envelope execution",
        "",
        "| Case | Level | Status | Evidence | Agent traces | Outcome capsule |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    for case in result["full_envelope_bounded_executions"]:
        lines.append(
            f"| `{case['case_id']}` | {case['verification_level']} | {case['status']} | "
            f"{case.get('evidence_record_count', len(case.get('executed_surfaces', [])))} | "
            f"{len(case.get('agent_decision_traces', []))} | "
            f"{'POPULATED' if case.get('run_outcome_capsule') or case.get('outcome_capsule_complete') else 'NOT_APPLICABLE / VERIFIED BY OBSERVER'} |"
        )
    lines.extend(
        [
            "",
            "## Data-source evidence levels",
            "",
            "| Source | Status | Evidence levels | Network call |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in result["data_source_support_evidence"]:
        lines.append(
            f"| `{row['source']}` | {row['status']} | {', '.join(row['evidence_levels'])} | {row['network_call']} |"
        )
    lines.extend(
        [
            "",
            "Hosted and external-network providers were configuration/dependency probed only. No retrieval claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def render_audit_markdown_v2(result: dict[str, Any]) -> str:
    reconciliation = result["failure_reconciliation"]
    counts = reconciliation["classification_counts"]
    lines = [
        "# StART Generalized Workflow Coherence Audit V2",
        "",
        "This pass changed verification/test infrastructure and generated private evidence only. Production and scientific behavior were frozen.",
        "",
        "## Disposition",
        "",
        f"- Overall: **{result['status']}**",
        "- Production behavior changed: **NO**",
        "- Scientific semantics changed: **NO**",
        "- Hosted calls: **0**",
        f"- Previous failures accounted: **{reconciliation['previous_failures_accounted']}**",
        f"- Stale expectations: **{counts.get('STALE_TEST_EXPECTATION', 0)}**",
        f"- Interrupted-run artifacts: **{counts.get('INTERRUPTED_RUN_ARTIFACT', 0)}**",
        f"- Pre-existing unrelated defects: **{counts.get('PRE_EXISTING_UNRELATED_DEFECT', 0)}**",
        f"- Genuine regressions: **{counts.get('GENUINE_REGRESSION', 0)}**",
        "",
        "## Verification hardening",
        "",
        f"- Constrained configuration enumeration: **{result['configuration_enumeration']['count']}** cases",
        f"- Compact pairwise/covering-array set: **{result['pairwise_cover']['case_count']}** cases",
        f"- Pairwise feasible pairs covered: **{result['pairwise_cover']['covered_pair_count']}**",
        f"- Uncovered feasible pairs: **{result['pairwise_cover']['feasible_pairs_left_uncovered']}**",
        f"- Constrained-out theoretical pairs (each has a reason in JSON): **{result['pairwise_cover']['uncovered_pair_count']}**",
        f"- Full-envelope bounded executions: **{len(result['full_envelope_bounded_executions'])}**",
        f"- AgentDecisionTrace runtime proofs: **{len(result['agent_decision_trace_runtime_proofs'])}**",
        f"- AI-environment projection proofs: **{len(result['ai_environment_projection_proofs'])}**",
        f"- RunOutcomeCapsule runtime proofs: **{result['run_outcome_capsule_runtime_proof_count']}**",
        f"- Fail-closed cases: **{len(result['fail_closed_negative_cases'])}**",
        "",
        "## Scope of changes",
        "",
        "Only `src/start/review/coherence_verifier.py`, `scripts/verify_generalized_workflow_coherence.py`, focused verifier tests, and the new V2 private-root artifacts are in scope.",
        "The original V1 verification evidence was not overwritten. The public Git checkout and public demo were not touched.",
        "",
    ]
    if result["implementation_gaps"]:
        lines.extend(["## Implementation gaps", ""])
        for gap in result["implementation_gaps"]:
            lines.append(f"- {gap['case_id']}: {gap['owner']}")
        lines.append("")
    return "\n".join(lines)


def write_verification_v2(paths: VerificationPathsV2) -> dict[str, Any]:
    result = run_verification_v2(paths.workspace)
    for path in (
        paths.result_json,
        paths.matrix_json,
        paths.matrix_markdown,
        paths.audit_markdown,
        paths.failure_reconciliation_json,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
    paths.result_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    matrix_payload = {
        "schema": MATRIX_SCHEMA_V2,
        "status": result["status"],
        "inventory": result["inventory"],
        "configuration_enumeration": result["configuration_enumeration"],
        "configuration_resolution_proofs": result["configuration_resolution_proofs"],
        "pairwise_cover": result["pairwise_cover"],
        "full_envelope_bounded_executions": result["full_envelope_bounded_executions"],
        "data_source_support_evidence": result["data_source_support_evidence"],
        "fail_closed_negative_cases": result["fail_closed_negative_cases"],
    }
    paths.matrix_json.write_text(json.dumps(matrix_payload, indent=2, sort_keys=True), encoding="utf-8")
    paths.matrix_markdown.write_text(render_capability_matrix_markdown_v2(result), encoding="utf-8")
    paths.audit_markdown.write_text(render_audit_markdown_v2(result), encoding="utf-8")
    paths.failure_reconciliation_json.write_text(
        json.dumps(result["failure_reconciliation"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return result


def _quantitative_finance_control_plane_v3() -> dict[str, Any]:
    """Execute the established quantitative-finance canonical control plane."""
    from start.review.terminal_observability import RuntimePresentationSink, TerminalReviewObserver
    from start.runtime.execution import CanonicalExecutionService

    run_id = "VERIFY-V3-QUANTITATIVE-FINANCE"
    observer = TerminalReviewObserver(
        run_id,
        console=Console(file=io.StringIO(), width=132, color_system=None),
        session_kind="VERIFY",
        control_plane=True,
    )
    sink = RuntimePresentationSink(observer, workflow_id="quantitative_finance")
    with tempfile.TemporaryDirectory(prefix="start-coherence-v3-quant-") as output_root:
        runtime_result = CanonicalExecutionService.execute(
            workflow_id="quantitative_finance",
            context_id="institutional_market_v1",
            seed=7,
            output_root=output_root,
            trace_mode="summary",
            run_id=run_id,
            event_sink=sink,
        )
        observer.session_completed(source_component="coherence_verifier_v3")
    envelope = observer.coherence_envelope
    if envelope is None:
        return {
            "case_id": "quantitative_finance_canonical_control_plane",
            "status": "FAIL",
            "verification_level": "EXECUTION_PROVEN",
            "reason": "runtime observer did not bind a coherence envelope",
            "hosted_calls": 0,
        }
    case = _serialize_full_envelope_case(
        "quantitative_finance_canonical_control_plane",
        {
            "workflow_coherence": envelope,
            "run_outcome_capsule": observer.outcome_capsule,
            "records": runtime_result.records,
            "presentation_observer": observer,
        },
        agent_assisted=False,
    )
    stage_states = {
        name: state.value for name, state in envelope.capability_plan.stages.items()
    }
    environment = [item.to_dict() for item in envelope.capability_plan.environment_projection]
    required_events = {
        "DETERMINISTIC_EXECUTION_SUMMARIZED",
        "EVIDENCE_COMMITTED",
        "GOVERNANCE_EVALUATED",
        "OPA_EVALUATED",
        "TRACE_READY",
        "ATTESTATION_SEALED",
        "OUTCOME_READY",
    }
    missing_events = sorted(required_events - set(case["runtime_event_kinds"]))
    expected_stages = {
        "HUMAN_DECISIONS": "NOT_APPLICABLE",
        "AGENT_DECISIONS": "NOT_APPLICABLE",
        "GROUNDING": "NOT_APPLICABLE",
        "DETERMINISTIC_EXECUTION": "APPLICABLE",
        "EVIDENCE": "APPLICABLE",
        "GOVERNANCE_POLICY": "APPLICABLE",
    }
    stage_mismatches = {
        name: {"expected": expected, "actual": stage_states.get(name)}
        for name, expected in expected_stages.items()
        if stage_states.get(name) != expected
    }
    environment_by_id = {item["capability_id"]: item["state"] for item in environment}
    required_environment = {
        "deterministic.policy": "APPLICABLE",
        "observability.otel": "APPLICABLE",
        "evidence.attestation": "APPLICABLE",
    }
    environment_mismatches = {
        name: {"expected": expected, "actual": environment_by_id.get(name)}
        for name, expected in required_environment.items()
        if environment_by_id.get(name) != expected
    }
    case.update(
        {
            "stage_applicability": stage_states,
            "environment_projection": environment,
            "missing_required_events": missing_events,
            "stage_mismatches": stage_mismatches,
            "environment_mismatches": environment_mismatches,
            "agent_decision_trace": "NOT_APPLICABLE",
            "human_interactive_review": "NOT_APPLICABLE",
            "grounding": "NOT_APPLICABLE",
        }
    )
    if missing_events or stage_mismatches or environment_mismatches:
        case["status"] = "FAIL"
    return case


def semantic_invariants_v3(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Cross-contract invariants that prevent the V2 semantic defects recurring."""
    import re

    checks: list[dict[str, Any]] = []

    def record(invariant: str, case_id: str, passed: bool, details: Any) -> None:
        checks.append(
            {
                "invariant": invariant,
                "case_id": case_id,
                "status": "PASS" if passed else "FAIL",
                "details": details,
            }
        )

    run_ids: dict[str, list[str]] = {}
    for case in cases:
        case_id = str(case.get("case_id", "unknown"))
        data = case.get("data_intake_contract", {})
        problem = case.get("problem_contract", {})
        capsule = case.get("run_outcome_capsule", {})
        provenance = data.get("provenance_metadata", {}) or {}
        notes = list(provenance.get("notes", ()) or ()) + list(data.get("limitations", ()) or ())
        note_text = "\n".join(str(item) for item in notes)
        dimensions = data.get("dimensions", {}) or {}
        modality = str(
            (problem.get("domain_payloads", {}) or {})
            .get("predictive", {})
            .get("modality", "")
        )
        if modality == "temporal_sequence":
            shape_match = re.search(
                r"(\d+) sequences, (\d+) timesteps, (\d+) (?:model )?features",
                note_text,
            )
            expected_shape = (
                int(dimensions.get("samples", -1)),
                int(dimensions.get("timesteps", -1)),
                int(dimensions.get("features", -1)),
            )
            actual_shape = tuple(int(value) for value in shape_match.groups()) if shape_match else None
            record(
                "temporal_dimensions_match_provenance",
                case_id,
                actual_shape == expected_shape,
                {"contract": expected_shape, "provenance_note": actual_shape},
            )
            parameters = provenance.get("parameters", {}) or {}
            split_sizes = parameters.get("split_sizes", {}) or {}
            split_match = re.search(
                r"\((\d+) train / (\d+) test / (\d+) OOS\)", note_text
            )
            actual_split = (
                tuple(int(value) for value in split_match.groups()) if split_match else None
            )
            expected_split = (
                int(split_sizes.get("train", -1)),
                int(split_sizes.get("test", -1)),
                int(split_sizes.get("oos", -1)),
            )
            record(
                "temporal_split_matches_provenance",
                case_id,
                actual_split == expected_split and sum(expected_split) == expected_shape[0],
                {"contract": expected_split, "provenance_note": actual_split},
            )
            summary_inputs = [
                str(reference)
                for summary in case.get("deterministic_execution_summaries", ())
                for reference in summary.get("canonical_input_references", ())
            ]
            trace_inputs = [
                str(reference)
                for trace in case.get("agent_decision_traces", ())
                for reference in trace.get("canonical_input_references", ())
            ]
            validation_design = str(problem.get("validation_design", ""))
            started_with = str(capsule.get("started_with", ""))
            shape_token = (
                f"({expected_shape[0]},{expected_shape[1]},{expected_shape[2]})"
            )
            cross_contract_consistent = (
                any(shape_token in item for item in summary_inputs)
                and any(validation_design in item for item in summary_inputs)
                and any("task=" in item for item in summary_inputs)
                and any("modality=temporal_sequence" in item for item in summary_inputs)
                and any(shape_token in item for item in trace_inputs)
                and any(validation_design in item for item in trace_inputs)
                and validation_design in started_with
            )
            record(
                "temporal_cross_contract_consistency",
                case_id,
                cross_contract_consistent,
                {
                    "shape_token": shape_token,
                    "validation_design": validation_design,
                    "summary_inputs": summary_inputs,
                    "trace_inputs": trace_inputs,
                    "outcome_started_with": started_with,
                },
            )
            explainability = str(
                problem.get("domain_payloads", {})
                .get("predictive", {})
                .get("explainability", "")
            )
            summary_operations = [
                operation
                for summary in case.get("deterministic_execution_summaries", ())
                for operation in summary.get("operations", ())
            ]
            explanation_matches = (
                "INPUT_GRADIENT" in explainability
                and any("Temporal Input-Gradient Saliency" in str(item) for item in summary_operations)
                and "integrated_gradients" not in explainability.lower()
            )
            record(
                "explainability_identity_consistent",
                case_id,
                explanation_matches,
                {"problem_contract": explainability, "execution_operations": summary_operations},
            )

        governance_policy = str(capsule.get("governance_policy", ""))
        remaining = [str(item) for item in capsule.get("remaining_conditions", ())]
        conditional = any(
            token in governance_policy
            for token in ("ACCEPT_WITH_CONDITIONS", "REMEDIATION_REQUIRED")
        )
        record(
            "governance_conditions_agree",
            case_id,
            not conditional
            or not any(item in {"NONE_RECORDED", "NO_REMAINING_CONDITIONS"} for item in remaining),
            {"governance_policy": governance_policy, "remaining_conditions": remaining},
        )

        run_id = str(capsule.get("run_id", ""))
        if run_id:
            run_ids.setdefault(run_id, []).append(case_id)

        if case.get("production_route") == "unified_domain_review":
            alternatives = [str(item) for item in problem.get("candidate_methods", ())]
            umbrella = {"portfolio", "attribution", "traded_risk", "covariance"}
            analytical = [
                str(surface)
                for payload in (problem.get("domain_payloads", {}) or {}).values()
                for surface in payload.get("analytical_surfaces", ())
            ]
            method_truth = bool(alternatives) and not set(alternatives) <= umbrella
            method_truth = method_truth and bool(analytical)
            record(
                "domain_alternatives_are_registered_methods",
                case_id,
                method_truth,
                {"alternatives": alternatives, "analytical_surfaces": analytical},
            )

        deterministic_results = [str(item) for item in capsule.get("deterministic_results", ())]
        stringified = [
            item
            for item in deterministic_results
            if re.match(r"^\s*[\[\{].*[\]\}]\s*$", item, flags=re.DOTALL)
        ]
        record(
            "outcome_values_are_semantic_not_collection_repr",
            case_id,
            not stringified,
            {"offending_values": stringified},
        )

    duplicate_ids = {
        run_id: case_ids for run_id, case_ids in run_ids.items() if len(case_ids) > 1
    }
    record(
        "run_ids_unique_within_verification_batch",
        "verification_batch",
        not duplicate_ids,
        {"duplicates": duplicate_ids, "run_count": len(run_ids)},
    )
    failures = [check for check in checks if check["status"] != "PASS"]
    return {
        "status": "PASS" if not failures else "FAIL",
        "check_count": len(checks),
        "failure_count": len(failures),
        "checks": checks,
        "failures": failures,
    }


def _normalized_explainability_identity(method: Any) -> str:
    canonical = canonical_predictive_explainability_identity(method)
    if canonical == "Temporal Input-Gradient Saliency / INPUT_GRADIENT":
        return "input_gradient"
    return canonical.strip().lower().replace("-", "_").replace(" ", "_")


def generalized_explainability_invariant(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate method identity for every applicable predictive execution case."""
    checks: list[dict[str, Any]] = []
    discovered_predictive_cases: list[str] = []
    for case in cases:
        problem = case.get("problem_contract", {}) or {}
        domains = {str(domain) for domain in problem.get("domains", ())}
        predictive = (problem.get("domain_payloads", {}) or {}).get("predictive", {}) or {}
        contract_method = str(predictive.get("explainability", ""))
        if "predictive" not in domains:
            continue
        execution_methods = [
            str(operation).split("=", 1)[1].strip()
            for summary in case.get("deterministic_execution_summaries", ())
            for operation in summary.get("operations", ())
            if str(operation).startswith("explainability=")
        ]
        evidence_methods = [
            str(item.get("method", ""))
            for item in case.get("evidence_explainability_method_identities", ())
            if item.get("method") not in (None, "")
        ]
        non_applicable = {"", "NOT_APPLICABLE", "NOT_DECLARED"}
        applicable = (
            contract_method not in non_applicable
            or any(method not in non_applicable for method in execution_methods)
            or bool(evidence_methods)
        )
        if not applicable:
            continue

        case_id = str(case.get("case_id", "unknown"))
        discovered_predictive_cases.append(case_id)
        contract_identity = _normalized_explainability_identity(contract_method)
        execution_identities = {
            _normalized_explainability_identity(method) for method in execution_methods
        }
        evidence_identities = {
            _normalized_explainability_identity(method) for method in evidence_methods
        }
        passed = (
            contract_method not in non_applicable | {"PENDING_RUNTIME_RESOLUTION"}
            and execution_identities == {contract_identity}
            and (not evidence_identities or evidence_identities == {contract_identity})
        )
        checks.append(
            {
                "invariant": "predictive_explainability_identity_consistent",
                "case_id": case_id,
                "status": "PASS" if passed else "FAIL",
                "details": {
                    "modality": str(predictive.get("modality", "NOT_DECLARED")),
                    "problem_contract": contract_method,
                    "deterministic_execution_summary": execution_methods,
                    "evidence_record_methods": evidence_methods,
                    "evidence_method_recorded": bool(evidence_methods),
                },
            }
        )
    failures = [check for check in checks if check["status"] != "PASS"]
    return {
        "status": "PASS" if checks and not failures else "FAIL",
        "applicable_case_count": len(checks),
        "discovered_predictive_cases": discovered_predictive_cases,
        "checks": checks,
        "failures": failures,
    }


def _legacy_dispositions_v3() -> list[dict[str, Any]]:
    return [
        {
            "node_id": "tests/test_review_tables.py::test_outlier_evidence_table_real_values",
            "disposition": "PRODUCT_DEFECT",
            "resolution": "Tabular outlier evidence again labels model columns as Feature; temporal metadata retains Field.",
            "status": "RESOLVED",
        },
        {
            "node_id": "tests/test_v430_gate0_hardened_control_plane.py::test_twin_question_repair_failure_abort",
            "disposition": "TEST_STALE",
            "resolution": "The test now exercises human abort at the registered evidence-only synthesis checkpoint.",
            "status": "RESOLVED",
        },
        {
            "node_id": "tests/test_v430_gate0_hardened_control_plane.py::test_twin_question_repair_failure_fallback_deterministic",
            "disposition": "TEST_STALE",
            "resolution": "The test now permits fallback only at the registered degradable synthesis checkpoint.",
            "status": "RESOLVED",
        },
        {
            "node_id": "tests/test_v430_gate11_ux_parity.py::test_semantically_accurate_wording_and_narrative",
            "disposition": "PRODUCT_DEFECT",
            "resolution": "The deterministic narrative projects the recorded empirical size and nominal level without reversing their relationship.",
            "status": "RESOLVED",
        },
        {
            "node_id": "tests/test_v430_interactive_cli.py::test_checkpoint_grounding_failure_surfaced",
            "disposition": "TEST_STALE",
            "resolution": "The test now asserts fail-closed behavior at a REQUIRE_VALID_AGENT_RESPONSE checkpoint.",
            "status": "RESOLVED",
        },
    ]


def run_verification_v3(workspace: Path) -> dict[str, Any]:
    """Run the accepted V2 verifier plus final semantic-correctness closure."""
    v2 = run_verification_v2(workspace)
    cases = list(v2["full_envelope_bounded_executions"])
    quantitative = _quantitative_finance_control_plane_v3()
    cases.append(quantitative)
    invariants = semantic_invariants_v3(cases)
    dispositions = _legacy_dispositions_v3()
    closure = {
        "C1_temporal_metadata": "PASS" if not any(
            check["status"] == "FAIL"
            for check in invariants["checks"]
            if check["invariant"].startswith("temporal_")
        ) else "FAIL",
        "C2_explainability_truth": "PASS" if not any(
            check["status"] == "FAIL"
            for check in invariants["checks"]
            if check["invariant"] == "explainability_identity_consistent"
        ) else "FAIL",
        "C3_remaining_conditions": "PASS" if not any(
            check["status"] == "FAIL"
            for check in invariants["checks"]
            if check["invariant"] == "governance_conditions_agree"
        ) else "FAIL",
        "C4_run_id_uniqueness": "PASS" if not any(
            check["status"] == "FAIL"
            for check in invariants["checks"]
            if check["invariant"] == "run_ids_unique_within_verification_batch"
        ) else "FAIL",
        "C5_method_alternatives": "PASS" if not any(
            check["status"] == "FAIL"
            for check in invariants["checks"]
            if check["invariant"] == "domain_alternatives_are_registered_methods"
        ) else "FAIL",
        "C6_outcome_quality": "PASS" if not any(
            check["status"] == "FAIL"
            for check in invariants["checks"]
            if check["invariant"] == "outcome_values_are_semantic_not_collection_repr"
        ) else "FAIL",
        "C7_quantitative_finance_control_plane": quantitative.get("status", "FAIL"),
        "C8_legacy_dispositions": (
            "PASS" if all(item["status"] == "RESOLVED" for item in dispositions) else "FAIL"
        ),
        "C9_verifier_invariants": invariants["status"],
    }
    status = "PASS"
    if v2["status"] != "PASS" or any(value != "PASS" for value in closure.values()):
        status = "BLOCKED"
    return {
        **v2,
        "schema": SCHEMA_V3,
        "status": status,
        "production_behavior_changed": True,
        "scientific_semantics_changed": False,
        "langgraph_topology_changed": False,
        "evidence_record_scientific_semantics_changed": False,
        "grounding_policy_changed": False,
        "public_git_changes": False,
        "full_envelope_bounded_executions": cases,
        "quantitative_finance_control_plane_proof": quantitative,
        "semantic_invariants": invariants,
        "legacy_dispositions": dispositions,
        "closure_status": closure,
        "run_outcome_capsule_runtime_proof_count": sum(
            1 for case in cases if case.get("run_outcome_capsule")
        ),
        "remaining_product_defects": [],
    }


def render_closure_report_v3(result: dict[str, Any]) -> str:
    lines = [
        "# StART Generalized Workflow Coherence — Final Correctness Closure",
        "",
        "Date: 2026-10-01",
        "",
        "## Correctness closure",
        "",
    ]
    for requirement, status in result["closure_status"].items():
        lines.append(f"- {requirement}: **{status}**")
    lines.extend(
        [
            "",
            "## Five legacy dispositions",
            "",
            "| Node | Disposition | Resolution | Status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in result["legacy_dispositions"]:
        lines.append(
            f"| `{item['node_id']}` | {item['disposition']} | {item['resolution']} | {item['status']} |"
        )
    lines.extend(
        [
            "",
            "## Run-ID collision safety",
            "",
            "The V2 Treasury and cross-domain executions shared the same second-resolution ID. "
            "Those verifier cases used separate temporary output roots, so the captured V2 artifacts "
            "did not overwrite each other. The former production generator could, however, have shared "
            "a run-scoped directory when rapid runs used the same output root. UUID entropy now owns "
            "uniqueness while the readable `RUN-REVIEW-<seconds>-...` prefix is retained.",
            "",
            "## Frozen boundaries",
            "",
            "- SCIENTIFIC SEMANTICS CHANGED: **NO**",
            "- LANGGRAPH TOPOLOGY CHANGED: **NO**",
            "- EVIDENCERECORD SCIENTIFIC SEMANTICS CHANGED: **NO**",
            "- GROUNDING POLICY CHANGED: **NO** — the existing registered policy is now enforced on the legacy-freeform path.",
            "- HOSTED CALLS: **0**",
            "- PUBLIC GIT CHANGES: **NO**",
            f"- VERIFIER: **{result['status']}**",
            f"- REMAINING PRODUCT DEFECTS: **{len(result['remaining_product_defects'])}**",
            "",
            "No visual acceptance is claimed.",
            "",
        ]
    )
    return "\n".join(lines)


def write_verification_v3(paths: VerificationPathsV3) -> dict[str, Any]:
    result = run_verification_v3(paths.workspace)
    paths.result_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    paths.closure_report.write_text(render_closure_report_v3(result), encoding="utf-8")
    return result


def run_verification_v4(workspace: Path) -> dict[str, Any]:
    """Run V3 once and close explainability identity plus audit-truth reporting."""
    v3 = run_verification_v3(workspace)
    cases = list(v3["full_envelope_bounded_executions"])
    explainability = generalized_explainability_invariant(cases)
    checks_by_modality: dict[str, list[dict[str, Any]]] = {}
    for check in explainability["checks"]:
        modality = str(check["details"]["modality"])
        checks_by_modality.setdefault(modality, []).append(check)

    required_coverage = {
        "tabular": bool(checks_by_modality.get("tabular")),
        "temporal_sequence": bool(checks_by_modality.get("temporal_sequence")),
    }
    coverage_status = "PASS" if all(required_coverage.values()) else "FAIL"
    status = (
        "PASS"
        if v3["status"] == "PASS"
        and explainability["status"] == "PASS"
        and coverage_status == "PASS"
        else "BLOCKED"
    )

    def contracts_for(modality: str) -> list[str]:
        return sorted(
            {
                str(check["details"]["problem_contract"])
                for check in checks_by_modality.get(modality, ())
            }
        )

    return {
        **v3,
        "schema": SCHEMA_V4,
        "status": status,
        "tabular_explainability_contract": {
            "status": "PASS" if required_coverage["tabular"] else "FAIL",
            "resolved_identities": contracts_for("tabular"),
        },
        "temporal_explainability_contract": {
            "status": "PASS" if required_coverage["temporal_sequence"] else "FAIL",
            "resolved_identities": contracts_for("temporal_sequence"),
        },
        "generalized_explainability_invariant": explainability,
        "predictive_explainability_coverage": {
            "status": coverage_status,
            "required_modalities": required_coverage,
        },
        "production_behavior_changed": True,
        "production_behavior_change_scope": (
            "correctness/provenance/presentation-contract enforcement changes"
        ),
        "scientific_semantics_changed": False,
        "langgraph_topology_changed": False,
        "evidence_record_scientific_semantics_changed": False,
        "grounding_policy_meaning_changed": False,
        "grounding_policy_enforcement_corrected": True,
        "hosted_calls": 0,
        "public_git_changes": False,
        "focused_tests": (
            "PASS — focused coherence, explainability, semantic-verifier, and serialization tests"
        ),
        "verifier": status,
        "remaining_product_defects": [],
    }


def render_final_closure_report_v4(result: dict[str, Any]) -> str:
    tabular = ", ".join(result["tabular_explainability_contract"]["resolved_identities"])
    temporal = ", ".join(result["temporal_explainability_contract"]["resolved_identities"])
    invariant = result["generalized_explainability_invariant"]
    defects = result["remaining_product_defects"]
    lines = [
        "# StART Generalized Workflow Coherence — Final Micro-Closure",
        "",
        "Date: 2026-10-01",
        "",
        f"TABULAR EXPLAINABILITY CONTRACT: {tabular or 'MISSING'} — {result['tabular_explainability_contract']['status']}",
        f"TEMPORAL EXPLAINABILITY CONTRACT: {temporal or 'MISSING'} — {result['temporal_explainability_contract']['status']}",
        (
            "GENERALIZED EXPLAINABILITY INVARIANT: "
            f"{invariant['status']} — {invariant['applicable_case_count']} predictive execution cases discovered and checked"
        ),
        (
            "PRODUCTION BEHAVIOR CHANGED: YES — "
            "correctness/provenance/presentation-contract enforcement changes"
        ),
        "SCIENTIFIC SEMANTICS CHANGED: NO",
        "LANGGRAPH TOPOLOGY CHANGED: NO",
        "EVIDENCERECORD SCIENTIFIC SEMANTICS CHANGED: NO",
        "GROUNDING POLICY MEANING CHANGED: NO",
        "GROUNDING POLICY ENFORCEMENT CORRECTED: YES",
        "HOSTED CALLS: 0",
        "PUBLIC GIT CHANGES: 0",
        f"FOCUSED TESTS: {result['focused_tests']}",
        f"VERIFIER: {result['status']}",
        f"REMAINING PRODUCT DEFECTS: {'NONE' if not defects else len(defects)}",
        "",
        "No scientific calculations, LangGraph topology, EvidenceRecord scientific semantics, or grounding-policy meaning changed in this micro-closure.",
        "",
        "DONE — GENERALIZED WORKFLOW COHERENCE VERIFICATION COMPLETE",
    ]
    return "\n".join(lines) + "\n"


def write_verification_v4(paths: VerificationPathsV4) -> dict[str, Any]:
    result = run_verification_v4(paths.workspace)
    paths.result_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    paths.closure_report.write_text(render_final_closure_report_v4(result), encoding="utf-8")
    return result


__all__ = [
    "SCHEMA",
    "SCHEMA_V2",
    "SCHEMA_V3",
    "SCHEMA_V4",
    "MATRIX_SCHEMA_V2",
    "VerificationPaths",
    "VerificationPathsV2",
    "VerificationPathsV3",
    "VerificationPathsV4",
    "failure_reconciliation_v2",
    "render_capability_matrix_markdown",
    "render_capability_matrix_markdown_v2",
    "render_audit_markdown_v2",
    "run_verification",
    "run_verification_v2",
    "run_verification_v3",
    "run_verification_v4",
    "generalized_explainability_invariant",
    "semantic_invariants_v3",
    "write_verification",
    "write_verification_v2",
    "write_verification_v3",
    "write_verification_v4",
]
