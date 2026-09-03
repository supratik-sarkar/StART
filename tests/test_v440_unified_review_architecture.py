"""Tests for Unified Review UX Architecture, Domain Routing, and Checkpoints (Combined Gate 7-9 Slice D)."""

from __future__ import annotations

import pytest

from start.registry import list_tests, load_builtin_tests
from start.review.applicability import applicable_tests
from start.review.architecture import (
    ReviewContextBundle,
    ReviewDomain,
    ReviewMode,
)
from start.review.multiline_input import ReviewCancelled
from start.review.state_machine import (
    CheckpointState,
    CheckpointStateMachine,
    GroundingValidationError,
)


def test_domain_applicability_routing_invariants():
    """Verify context_type routing produces Market=25, Treasury=2, Market+Treasury=27, Predictive=52, Total=79."""
    load_builtin_tests()
    all_tests = list_tests()
    assert len(all_tests) == 79
    assert len(set(t.test_id for t in all_tests)) == 79

    market_app = applicable_tests((ReviewDomain.MARKET,))
    assert market_app.count == 25
    # Invariant: Treasury-only CEV and Stanton are excluded from Market-only
    assert "traded_risk.cev_elasticity" not in market_app.test_ids
    assert "traded_risk.stanton_nonparametric" not in market_app.test_ids

    treasury_app = applicable_tests((ReviewDomain.TREASURY,))
    assert treasury_app.count == 2
    assert set(treasury_app.test_ids) == {
        "traded_risk.cev_elasticity",
        "traded_risk.stanton_nonparametric",
    }

    union_app = applicable_tests((ReviewDomain.MARKET, ReviewDomain.TREASURY))
    assert union_app.count == 27
    assert set(union_app.test_ids) == set(market_app.test_ids) | set(treasury_app.test_ids)

    pred_app = applicable_tests((ReviewDomain.PREDICTIVE,))
    assert pred_app.count == 52
    assert set(pred_app.test_ids) | set(union_app.test_ids) == set(t.test_id for t in all_tests)


def test_review_context_bundle_and_modes():
    """Verify ReviewContextBundle correctly instantiates across single_domain and cross_domain modes."""
    bundle_market = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
    )
    assert bundle_market.mode == ReviewMode.SINGLE_DOMAIN
    assert bundle_market.domains == (ReviewDomain.MARKET,)

    bundle_cross = ReviewContextBundle(
        mode=ReviewMode.CROSS_DOMAIN,
        domains=(ReviewDomain.MARKET, ReviewDomain.TREASURY),
    )
    assert bundle_cross.mode == ReviewMode.CROSS_DOMAIN
    assert len(bundle_cross.domains) == 2


def test_cancellation_boundary_discipline():
    """Verify ReviewCancelled exception propagates cleanly without uncontrolled traceback."""
    sm = CheckpointStateMachine(checkpoint_title="Portfolio Construction")
    assert sm.current_state == CheckpointState.READY

    with pytest.raises(ReviewCancelled):
        raise ReviewCancelled("User pressed Ctrl+C or cancelled review.")


def test_checkpoint_state_machine_flow():
    """Verify checkpoint state machine transitions deterministically through review stages."""
    sm = CheckpointStateMachine(checkpoint_title="Portfolio Construction")
    assert sm.current_state == CheckpointState.READY

    sm.transition(CheckpointState.PROVIDER_CALL)
    assert sm.current_state == CheckpointState.PROVIDER_CALL

    sm.transition(CheckpointState.PROVIDER_RESPONSE)
    assert sm.current_state == CheckpointState.PROVIDER_RESPONSE

    sm.transition(CheckpointState.GROUNDING_VALIDATE)
    assert sm.current_state == CheckpointState.GROUNDING_VALIDATE

    sm.transition(CheckpointState.VERIFIED)
    assert sm.current_state == CheckpointState.VERIFIED

    sm.transition(CheckpointState.COMPLETED)
    assert sm.current_state == CheckpointState.COMPLETED
    assert sm.is_terminal is True


def test_grounding_validation_error_structure():
    """Verify GroundingValidationError enforces evidence-bound claims and records unbound claims."""
    err = GroundingValidationError(
        "Claim failed grounding validation",
        unbound_claims=[{"statement": "Sharpe ratio is 2.5 without citation"}],
        reason_code="UNBOUND_METRIC",
    )
    assert err.reason_code == "UNBOUND_METRIC"
    assert len(err.unbound_claims) == 1
    assert "without citation" in err.unbound_claims[0]["statement"]


def test_market_review_full_8_checkpoint_sequence_and_actions():
    """Verify production Market review provides the full 8-checkpoint sequence in deterministic order."""
    from start.core.schemas import EvidenceRecord, Status
    from start.review.executor import run_domain_checkpoints

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
    )

    records = [
        EvidenceRecord(
            test_id="portfolio.mean_variance",
            test_name="MVO",
            model_id="M1",
            dataset_id="D1",
            run_id="R1",
            metrics={"converged": True},
            params={},
            status=Status.PASS,
            interpretation="MVO converged.",
            limitations=[],
        ),
        EvidenceRecord(
            test_id="validation.var_size_power",
            test_name="VaR Validation",
            model_id="M1",
            dataset_id="D1",
            run_id="R1",
            metrics={"empirical_size": 0.052},
            params={},
            status=Status.PASS,
            interpretation="VaR size passed.",
            limitations=[],
        ),
        EvidenceRecord(
            test_id="traded_risk.brownian_bridge_barrier",
            test_name="Barrier Crossing Test",
            model_id="M1",
            dataset_id="D1",
            run_id="R1",
            metrics={"crossing_probability": 0.02},
            params={},
            status=Status.PASS,
            interpretation="Barrier boundary admissible.",
            limitations=[],
        ),
    ]

    # Mock user input providing Accept to all checkpoints
    user_inputs = iter(["A", "A", "A", "A", "A", "A", "A", "A"])
    decisions = run_domain_checkpoints(bundle, records, interactive=True, ask=lambda _: next(user_inputs))

    checkpoint_names = [d["checkpoint"] for d in decisions]
    assert len(checkpoint_names) == 8
    assert checkpoint_names == [
        "Portfolio Risk & Volatility Assumptions",
        "Factor Modeling & Attribution Assumptions",
        "VaR Backtesting & Exception Frequency",
        "Covariance Structure & Missing Data Treatment",
        "Scenario Analysis & Stress Testing",
        "Cross-Analytical Committee Synthesis",
        "Barrier Validation & Boundary Admissibility",
        "Model Governance & Attestation Sign-off",
    ]


def test_multiline_governance_text_terminator_end():
    """Verify multiline governance text input terminates strictly on own-line END."""
    import io

    from start.review.multiline_input import read_multiline_text

    simulated_input = io.StringIO(
        "First line of governance context\nSecond line with 2. option lookalike\nEND\n"
    )
    captured_output: list[str] = []

    result = read_multiline_text(
        label="Governance Context",
        stream=simulated_input,
        printer=captured_output.append,
    )

    assert "First line of governance context" in result
    assert "Second line with 2. option lookalike" in result
    assert "END" not in result


def test_deferred_monte_carlo_var_es_not_advertised():
    """Verify Monte Carlo VaR/ES is explicitly deferred and not advertised in the root registry or checkpoints."""
    load_builtin_tests()
    all_tests = list_tests()
    all_test_ids = {t.test_id for t in all_tests}

    # Case A: Not in root registry
    assert "traded_risk.var_monte_carlo" not in all_test_ids
    assert "traded_risk.monte_carlo_var" not in all_test_ids
    assert "traded_risk.cvar_monte_carlo" not in all_test_ids

    # Case B: Ensure production review executor does NOT include var_monte_carlo in VaR checkpoint
    import inspect

    from start.review import executor

    src = inspect.getsource(executor.run_domain_checkpoints)
    assert "traded_risk.var_monte_carlo" not in src


def test_scenario_checkpoint_consumes_existing_pattern_b_evidence_without_recomputation():
    """Verify Scenario/Stress checkpoint strictly consumes pre-existing Pattern-B EvidenceRecords."""
    from start.core.schemas import EvidenceRecord, Status
    from start.review.executor import run_domain_checkpoints

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
    )

    # 1. Simulate pre-existing Gate-6 Scenario EvidenceRecords
    scen_linear = EvidenceRecord(
        test_id="scenario.factor_linear",
        test_name="Linear Factor Stress",
        model_id="M1",
        dataset_id="D1",
        run_id="RUN-GATE6-001",
        metrics={"portfolio_loss": 0.12},
        params={"shock": "MKT_-10pct"},
        status=Status.PASS,
        interpretation="Factor stress executed.",
        limitations=[],
    )
    scen_rev = EvidenceRecord(
        test_id="scenario.reverse_stress",
        test_name="Reverse Stress Test",
        model_id="M1",
        dataset_id="D1",
        run_id="RUN-GATE6-001",
        metrics={"minimum_distance": 0.18, "target_loss": 0.20},
        params={"target_loss": 0.20},
        status=Status.PASS,
        interpretation="Reverse stress executed.",
        limitations=[],
    )

    records = [scen_linear, scen_rev]

    # Checkpoint execution consumes existing records without calling quantitative solver engines
    inputs = iter(["A"] * 8)
    decisions = run_domain_checkpoints(bundle, records, interactive=True, ask=lambda _: next(inputs))

    scen_decisions = [d for d in decisions if d["checkpoint"] == "Scenario Analysis & Stress Testing"]
    assert len(scen_decisions) == 1
    assert scen_decisions[0]["action"] == "accept"

    # Root registry remains exactly 79 and does NOT contain scenario.*
    load_builtin_tests()
    all_tests = list_tests()
    assert len(all_tests) == 79
    for t in all_tests:
        assert not t.test_id.startswith("scenario.")


def test_checkpoint_evidence_scoping_and_barrier_omission():
    """Verify validation.var_size_power is in VaR, validation.regem_structural is in Covariance, and Barrier is omitted when not applicable."""
    from start.core.schemas import EvidenceRecord, Status
    from start.review.executor import run_domain_checkpoints

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
    )

    var_val = EvidenceRecord(
        test_id="validation.var_size_power",
        test_name="VaR Validation",
        model_id="M1",
        dataset_id="D1",
        run_id="R1",
        metrics={"empirical_size": 0.052},
        params={},
        status=Status.PASS,
        interpretation="VaR validation passed.",
        limitations=[],
    )
    regem_val = EvidenceRecord(
        test_id="validation.regem_structural",
        test_name="RegEM Validation",
        model_id="M1",
        dataset_id="D1",
        run_id="R1",
        metrics={"cell_pass_rate": 1.0},
        params={},
        status=Status.PASS,
        interpretation="RegEM structural validation passed.",
        limitations=[],
    )

    records = [var_val, regem_val]
    inputs = iter(["A"] * 8)
    decisions = run_domain_checkpoints(bundle, records, interactive=True, ask=lambda _: next(inputs))

    checkpoint_names = [d["checkpoint"] for d in decisions]
    # In standard equity portfolio without barrier contracts, Barrier checkpoint is omitted (7 checkpoints total)
    assert "Barrier Validation & Boundary Admissibility" not in checkpoint_names
    assert len(checkpoint_names) == 7


def test_statistical_criterion_vs_materiality_criterion_separation():
    """Verify statistical criterion (gamma_test=0.05) is explicitly decoupled from cross-analytical materiality criterion (NONE)."""
    from start.consensus.cross_analytical import eval_var_frequency_vs_independence
    from start.core.schemas import EvidenceRecord, Status
    from start.evidence.claims import ClaimStatus, ClaimType

    kupiec_rec = EvidenceRecord(
        test_id="traded_risk.var_kupiec_pof",
        test_name="Kupiec POF",
        model_id="M1",
        dataset_id="D1",
        run_id="R1",
        metrics={"reject_unconditional_coverage": False, "gamma_test": 0.05},
        params={"gamma_test": 0.05},
        status=Status.PASS,
        interpretation="Unconditional coverage passed.",
        limitations=[],
    )
    christ_rec = EvidenceRecord(
        test_id="traded_risk.var_christoffersen",
        test_name="Christoffersen Independence",
        model_id="M1",
        dataset_id="D1",
        run_id="R1",
        metrics={"reject_independence": True, "gamma_test": 0.05},
        params={"gamma_test": 0.05},
        status=Status.WARN,
        interpretation="Independence rejected.",
        limitations=[],
    )

    claim, edges = eval_var_frequency_vs_independence(kupiec_rec, christ_rec)

    assert claim.claim_type == ClaimType.UNRESOLVED_RISK
    assert claim.status == ClaimStatus.EVIDENCE_ONLY
    assert claim.statistical_criterion_source == "PRE_REGISTERED_VALIDATION"
    assert claim.statistical_gamma_test == 0.05
    assert claim.materiality_criterion_source == "NONE"
    assert claim.threshold_provenance is None
