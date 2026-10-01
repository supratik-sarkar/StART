"""Focused presentation-truth and artifact-board regressions for v6.0.2."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from demo_artifact_board import curate_entries
from rich.console import Console

from start.registry.market_contexts import MarketContext, ShortRateContext
from start.review.applicability import applicable_tests, build_plan_preview
from start.review.architecture import (
    ReviewContextBundle,
    ReviewDomain,
    ReviewMode,
)
from start.review.terminal_observability import (
    PresentationEventKind,
    ReviewPresentationState,
    TerminalReviewObserver,
    predictive_review_path,
    predictive_runtime_context_title,
    treasury_review_path,
    workflow_runtime_context_title,
)
from start.review.workflow_coherence import (
    build_canonical_runtime_coherence_envelope,
    build_run_outcome_capsule,
    build_unified_review_coherence_envelope,
)


def test_generic_titles_paths_and_treasury_opening_are_domain_derived() -> None:
    predictive = predictive_runtime_context_title(model_development=True)
    cross_domain = workflow_runtime_context_title((ReviewDomain.MARKET, ReviewDomain.TREASURY))

    assert "FLIGHT" not in predictive
    assert "FLIGHT" not in cross_domain
    assert "PREDICTIVE" in predictive
    assert "CROSS-DOMAIN" in cross_domain
    assert "Market Risk & Portfolio Analytics" in cross_domain
    assert "Treasury / IRRBB & Short-Rate Modeling" in cross_domain

    tabular_path = predictive_review_path("tabular")
    temporal_path = predictive_review_path("temporal_sequence")
    assert "TABULAR INPUT CONTRACT" in tabular_path
    assert "TEMPORAL CONTRACT" not in tabular_path
    assert "TEMPORAL CONTRACT" in temporal_path
    assert treasury_review_path(("traded_risk.cev_elasticity", "traded_risk.stanton_nonparametric")) == (
        "SHORT-RATE CONTEXT → CEV → STANTON → VALIDATION → EVIDENCE → GOVERNANCE"
    )


def test_tabular_offline_leakage_wording_does_not_claim_sequence_cohorts() -> None:
    from start.providers.offline_demo_twin import OfflineDemoTwinProvider

    tabular = OfflineDemoTwinProvider._plain_response(
        "Data Quality, Imbalance & Preprocessing Assumptions",
        "Predictive Input Modality: tabular\nA reviewer asked: \"Review leakage across train/test and OOS.\".",
    )
    temporal = OfflineDemoTwinProvider._plain_response(
        "Data Quality, Imbalance & Preprocessing Assumptions",
        "Predictive Input Modality: temporal_sequence\nA reviewer asked: \"Review leakage across train/test and OOS.\".",
    )

    assert "sequence cohorts" not in tabular.lower()
    assert "tabular contract" in tabular.lower()
    assert "sequence cohorts" in temporal.lower()


def test_predictive_review_plan_uses_supported_explainability_vocabulary() -> None:
    bundle = ReviewContextBundle(
        tabular=pd.DataFrame({"x": [1.0, 2.0], "target": [0, 1]}),
        domains=(ReviewDomain.PREDICTIVE,),
    )
    rendered = build_plan_preview(bundle).render()

    assert "Feature Attribution / Explainability" in rendered
    assert "SHAP" not in rendered
    assert "Integrated Gradients" not in rendered


def test_governance_condition_details_project_into_run_outcome() -> None:
    from start.review.coherence_verifier import _predictive_fixture

    envelope = _predictive_fixture(sequence=False)
    envelope.run_id = "RUN-CONDITIONS-PRESENTATION"
    state = ReviewPresentationState(run_id=envelope.run_id)
    state.governance_disposition = "ACCEPT_WITH_CONDITIONS"
    state.governance_conditions = [
        "Calibration: OOS ECE exceeds the configured threshold (cohort_metrics.oos.ece)"
    ]
    capsule = build_run_outcome_capsule(envelope=envelope, events=[], presentation_state=state)

    assert state.governance_conditions[0] in capsule.remaining_conditions
    assert not any("details unavailable" in item for item in capsule.remaining_conditions)


def test_grounding_pass_names_required_and_soft_claim_denominators() -> None:
    stream = io.StringIO()
    observer = TerminalReviewObserver(
        "RUN-GROUNDING-DENOMINATOR",
        console=Console(file=stream, width=120, color_system=None),
    )
    observer.grounding_result(
        accepted=True,
        quantitative_claims=16,
        grounded_claims=13,
        grounding_required_claims=13,
        other_exempt_claims=3,
    )
    assert observer.events[-1].kind is PresentationEventKind.GROUNDING_ACCEPTED
    assert "Grounding-required claims  13" in stream.getvalue()
    assert "Canonically grounded  13/13" in stream.getvalue()
    assert "Other (soft/exempt) claims  3" in stream.getvalue()
    assert "13/16" not in stream.getvalue()

    observer.grounding_result(
        accepted=True,
        quantitative_claims=16,
        grounded_claims=13,
        grounding_required_claims=16,
    )
    assert observer.events[-1].kind is PresentationEventKind.GROUNDING_REJECTED


def test_cross_domain_intake_preserves_original_source_and_resolved_context() -> None:
    bundle = ReviewContextBundle(
        market=MarketContext(returns=pd.DataFrame({"asset": [0.01, -0.01]})),
        short_rate=ShortRateContext(rates=pd.Series([0.02, 0.021])),
        domains=(ReviewDomain.MARKET, ReviewDomain.TREASURY),
        mode=ReviewMode.CROSS_DOMAIN,
        selected_source="Built-in Synthetic Market World",
    )
    envelope = build_unified_review_coherence_envelope(
        run_id="RUN-ORIGIN-TRUTH",
        bundle=bundle,
        applicable=applicable_tests(bundle.domains),
    )
    stream = io.StringIO()
    observer = TerminalReviewObserver(
        envelope.run_id,
        console=Console(file=stream, width=140, color_system=None),
    )
    observer.bind_coherence_envelope(envelope)
    observer.publish_coherence_contract()

    rendered = stream.getvalue()
    assert "Built-in Synthetic Market World" in rendered
    assert "MarketContext + ShortRateContext" in rendered
    assert "typed in-memory review contexts" in rendered


def test_capability_environment_projection_is_visible_in_plan() -> None:
    envelope = build_canonical_runtime_coherence_envelope(
        run_id="RUN-ENVIRONMENT-TRUTH",
        workflow_id="quantitative_finance",
        context_id="institutional_market_v1",
        context_metadata={"spec_id": "institutional_market_v1"},
    )
    stream = io.StringIO()
    observer = TerminalReviewObserver(
        envelope.run_id,
        console=Console(file=stream, width=140, color_system=None),
    )
    observer.bind_coherence_envelope(envelope)
    observer.publish_coherence_contract()

    rendered = stream.getvalue()
    assert "ai.policy_and_guardrails" in rendered
    assert "NOT_APPLICABLE" in rendered
    assert "deterministic.policy" in rendered


def test_terminal_outcome_groups_alternatives_but_keeps_full_machine_payload() -> None:
    market = MarketContext(returns=pd.DataFrame({"asset": [0.01, -0.01]}))
    treasury = ShortRateContext(rates=pd.Series([0.02, 0.021]))
    bundle = ReviewContextBundle(
        market=market,
        short_rate=treasury,
        domains=(ReviewDomain.MARKET, ReviewDomain.TREASURY),
        mode=ReviewMode.CROSS_DOMAIN,
    )
    envelope = build_unified_review_coherence_envelope(
        run_id="RUN-GROUPED-OUTCOME",
        bundle=bundle,
        applicable=applicable_tests(bundle.domains),
    )
    full_alternatives = tuple(envelope.problem.candidate_methods)
    stream = io.StringIO()
    observer = TerminalReviewObserver(
        envelope.run_id,
        console=Console(file=stream, width=140, color_system=None),
    )
    observer.bind_coherence_envelope(envelope)
    capsule = observer.publish_outcome_capsule()

    rendered = stream.getvalue()
    event_payload = observer.events[-1].payload
    assert "Market:" in rendered and "Treasury:" in rendered
    assert len(capsule.alternatives) == len(full_alternatives)
    assert tuple(event_payload["alternatives"]) == full_alternatives


def test_artifact_board_curation_prioritizes_treasury_diagnostics() -> None:
    entries = [
        {"artifact_id": f"ART-{index}", "title": title}
        for index, title in enumerate(
            (
                "VaR PnL Timeline",
                "Backtest Exceptions",
                "Scenario Contribution",
                "Reverse Stress",
                "Covariance Summary",
                "CEV Elasticity Diagnostic",
                "Stanton Drift Diagnostic",
            )
        )
    ]

    selected = curate_entries("flight_c", entries, limit=6)
    titles = {entry["title"] for entry in selected}
    assert "CEV Elasticity Diagnostic" in titles
    assert "Stanton Drift Diagnostic" in titles


def test_required_artifact_board_wiring_remains_exact_run_and_controlled() -> None:
    root = Path(__file__).parents[1]
    source = (root / "demo_controller.py").read_text()
    board_source = (root / "demo_artifact_board.py").read_text()
    current_run_source = (root / "src/start/reporting/current_run.py").read_text()

    assert "--presentation-mode" in source
    assert 'action == "artifact_board"' in source
    assert "manifest_path=str(payload.get(\"manifest_path\", \"\"))" in source
    assert "run_id=str(payload.get(\"run_id\", \"\"))" in source
    assert "load_current_run_manifest(root, run_id)" in board_source
    assert "current-run artifact manifest" in board_source
    assert "setInterval(" in board_source
    assert "dialog.showModal()" in board_source
    assert "terminal_ratio: float = 0.62" in board_source
    assert "no historical fallback" in board_source
    assert "write_current_run_manifest" in current_run_source
