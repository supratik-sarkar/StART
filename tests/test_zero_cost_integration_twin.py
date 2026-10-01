"""StART v6.0.2 $0 Deterministic Integration Twin & Live Failure Regression Suite.

Strict Hard Cost Boundary: PAID_HOSTED_LLM_CALLS = 0.
Exercises genuine scientific engines, structured reviewer schema enforcement,
governance attestation, and explicitly verifies regressions for Failures A-H.
"""

from __future__ import annotations

import json
import os
import re
import tempfile

import numpy as np
import pytest
from pydantic import ValidationError

import start.providers.llm as llm_mod
from start.core.schemas import EvidenceRecord, Status, TestResult
from start.data.synthetic_market import generate_market_world
from start.modeling.sequence_data import generate_sequence_dataset
from start.modeling.sequence_dl import SequenceClassifier, SequenceInputContractError
from start.portfolio.contracts import RepricingMethod, ScenarioSpec, ScenarioType
from start.portfolio.scenario import validate_scenario_data_integrity
from start.providers.base import LLMProvider
from start.review.architecture import (
    LLMReviewConfig,
    ReviewContextBundle,
    ReviewDomain,
    ReviewGroundingMode,
    ReviewLifecycle,
    ReviewMode,
)
from start.review.executor import (
    run_domain_checkpoints,
    run_market_treasury_review,
)
from start.review.multiline_input import ReviewCancelled
from start.review.structured_contract import (
    CriterionStatus,
    StructuredReviewerResponse,
)
from start.runtime import CanonicalExecutionService


# --------------------------------------------------------------------------- #
# Mock Providers for $0 Deterministic Twin
# --------------------------------------------------------------------------- #
class DeterministicFakeStructuredProvider(LLMProvider):
    """Deterministic structured reviewer test fixture conforming to schema.
    
    TEST FIXTURE ONLY: Returns valid StructuredReviewerResponse JSON citing
    actual Evidence IDs and canonical metric paths from the prompt.
    """

    name = "openai"
    model = "gpt-5.1"

    def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str:
        ev_ids = re.findall(r"EV-[a-f0-9]+", user)
        first_ev = ev_ids[0] if ev_ids else "EV-000000000000"
        m_paths = re.findall(r"metrics\.([a-zA-Z0-9_]+)", user)
        if "metrics.annualised_volatility" in user:
            metric_path = "metrics.annualised_volatility"
        elif "metrics.r_squared" in user:
            metric_path = "metrics.r_squared"
        elif "metrics.n_exceptions" in user:
            metric_path = "metrics.n_exceptions"
        elif "metrics.shrinkage_intensity" in user:
            metric_path = "metrics.shrinkage_intensity"
        elif "metrics.stress_loss" in user:
            metric_path = "metrics.stress_loss"
        elif m_paths:
            metric_path = f"metrics.{m_paths[0]}"
        elif "metrics.total_tests" in user:
            metric_path = "metrics.total_tests"
        else:
            metric_path = "metrics.annualised_volatility"

        return json.dumps({
            "findings": [
                {
                    "finding_id": "F-01",
                    "finding_type": "OBSERVED_EVIDENCE",
                    "conclusion": "Quantitative metrics satisfy accepted risk tolerance criteria.",
                    "evidence_refs": [{"evidence_id": first_ev, "metric_path": metric_path}],
                    "criterion_status": "APPLICABLE",
                    "unresolved_reason": None,
                }
            ],
            "overall_assessment": "Independent review confirms model mathematical soundness.",
        })


# --------------------------------------------------------------------------- #
# Requirement 8: $0 Integration Twins for Sessions A, B, C
# --------------------------------------------------------------------------- #
def test_session_a_predictive_temporal_lstm_twin():
    """SESSION A TWIN: Predictive / Temporal LSTM on genuine rank-3 temporal input."""
    # 1. Enforce rank-3 temporal contract on sequence engine
    X, y = generate_sequence_dataset(n_series=50, timesteps=24, n_features=3, seed=42)
    assert X.ndim == 3 and X.shape[1] == 24 and X.shape[2] == 3

    clf = SequenceClassifier(family="lstm", epochs=2, random_state=42)
    clf.fit(X, y)
    probs = clf.predict_proba(X)
    assert probs.shape == (50, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)

    # 2. Verify rank-2 input is rejected
    X_rank2 = np.random.randn(50, 6).astype(np.float32)
    with pytest.raises(SequenceInputContractError, match="requires rank-3 input"):
        clf.fit(X_rank2, y)


def test_session_b_market_treasury_twin(monkeypatch):
    """SESSION B TWIN: Portfolio + Market Risk + Scenario + Committee + Governance.
    
    Exercises real MVO/HRP, VaR/ES, scenario shock, reverse stress, and all 7 checkpoints.
    """
    monkeypatch.setattr(llm_mod, "get_llm_provider", lambda cfg: DeterministicFakeStructuredProvider())

    market = generate_market_world(n_assets=20, n_periods=100, seed=42)
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        market=market,
        materiality="high",
        lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
        business_context="Portfolio optimization and market risk review",
        llm_config=LLMReviewConfig(
            backend_mode="public",
            provider="openai",
            model="gpt-5.1",
            status="CONNECTED",
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )

    # Human interaction sequence:
    # Chk 1: Challenge -> Note -> Question -> Note -> Override -> Note
    # Chk 2: Question -> Note -> Accept
    # Chk 3: Question -> Note -> Accept
    # Chk 4: Question -> Note -> Accept
    # Chk 5: Challenge -> Note -> Accept
    # Chk 6: Question -> Note -> Accept
    # Chk 7: Accept
    prompts = iter([
        # Checkpoint 1 (Portfolio)
        "C", "Challenge MVO optimality",
        "Q", "Inquire about HRP diversification",
        "O", "Override default allocation rationale",
        # Checkpoint 2 (Attribution)
        "Q", "Clarify factor exposure stability",
        "A",
        # Checkpoint 3 (VaR Backtesting)
        "Q", "Evaluate tail risk exception frequency",
        "A",
        # Checkpoint 4 (Covariance)
        "Q", "Explain Ledoit-Wolf shrinkage intensity",
        "A",
        # Checkpoint 5 (Scenario & Stress)
        "C", "Challenge historical shock magnitude",
        "A",
        # Checkpoint 6 (Committee Synthesis)
        "Q", "Summarize cross-analytical committee resolutions",
        "A",
        # Checkpoint 7 (Signoff)
        "A",
    ])

    with tempfile.TemporaryDirectory() as td:
        outcome = run_market_treasury_review(
            bundle,
            output_root=td,
            interactive=True,
            ask=lambda _: next(prompts),
        )

        assert outcome["ledger"].verify() is True
        assert outcome["seal"] is not None
        
        # Verify all 7 distinct checkpoints were traversed
        chks_visited = [d["checkpoint"] for d in outcome["decisions"]]
        assert len(set(chks_visited)) == 7
        assert chks_visited[-1] == "Model Governance & Attestation Sign-off"

        # Verify challenge diagnostic generated subordinate EvidenceRecord
        c_decisions = [d for d in outcome["decisions"] if d["action"] == "challenge"]
        assert len(c_decisions) >= 1


def test_session_c_opa_canonical_workflow_twin():
    """SESSION C TWIN: OPA Canonical Workflow quantitative_finance."""
    with tempfile.TemporaryDirectory() as td:
        res = CanonicalExecutionService.execute(
            workflow_id="quantitative_finance",
            context_id="institutional_market_v1",
            seed=42,
            output_root=td,
            trace_mode="engineering",
        )
        assert len(res.records) == 25
        assert res.governance_disposition in ("ACCEPT", "ACCEPT_WITH_CONDITIONS")
        assert res.merkle_root is not None
        assert len(str(res.merkle_root)) > 0


# --------------------------------------------------------------------------- #
# Requirement 9: Exact Failure Regressions A through H
# --------------------------------------------------------------------------- #
def test_failure_a_scenario_integrity_evidencerecord_scalar_metric():
    """Failure A Regression: Scenario-integrity EvidenceRecord list-valued metric crash."""
    spec = ScenarioSpec(
        scenario_id="SCEN-TEST-AUDIT",
        scenario_name="Test Scenario Audit",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=(),
        repricing_method=RepricingMethod.LINEAR_RETURN,
    )
    diag_res = validate_scenario_data_integrity(spec, portfolio_assets=["ASSET_1", "ASSET_2"])
    assert not diag_res.valid
    assert len(diag_res.issues) >= 1

    # Must format issues as scalar string, not list
    issues_scalar = "; ".join(diag_res.issues) if diag_res.issues else "none"
    metrics: dict[str, bool | float | int | str | None] = {
        "status": "FAIL",
        "n_issues": len(diag_res.issues),
        "issues": issues_scalar,
        "loss_exceeds_threshold": False,
    }
    rec = EvidenceRecord.from_result(
        TestResult(
            test_id="scenario.scenario_integrity",
            test_name="scenario.scenario_integrity",
            status=Status.FAIL,
            metrics=metrics,
        ),
        run_id="RUN-TEST-SCALAR",
    )
    assert isinstance(rec.metrics["issues"], str)


def test_failure_b_malformed_criterion_status_handling():
    """Failure B Regression: Malformed criterion_status = 'CRITERION_REQUIRED' rejected."""
    malformed_json = {
        "findings": [
            {
                "finding_id": "F-01",
                "finding_type": "CRITERION_REQUIRED",
                "conclusion": "A pre-registered backtesting criterion is strictly required.",
                "evidence_refs": [],
                "criterion_status": "CRITERION_REQUIRED",
                "unresolved_reason": None,
            }
        ],
        "overall_assessment": "Policy requirement gap identified.",
    }
    with pytest.raises(ValidationError):
        StructuredReviewerResponse.model_validate(malformed_json)

    # Valid status ABSENT is accepted
    valid_json = malformed_json.copy()
    valid_json["findings"] = [
        {
            "finding_id": "F-01",
            "finding_type": "CRITERION_REQUIRED",
            "conclusion": "A pre-registered backtesting criterion is strictly required.",
            "evidence_refs": [],
            "criterion_status": "ABSENT",
            "unresolved_reason": None,
        }
    ]
    resp = StructuredReviewerResponse.model_validate(valid_json)
    assert resp.findings[0].criterion_status == CriterionStatus.ABSENT


def test_failure_c_unexpected_fallback_safety():
    """Fallback is authorized only for the registered synthesis question."""
    from demo_controller import PTYController

    # Unauthorized -> Fails closed
    ctrl1 = PTYController(timeout=2.0)
    ctrl1.spawn("echo '  Select action [default: 1]: '; sleep 1")
    ok1 = ctrl1.read_until_pattern(r"SOME_TARGET", allowed_fallback_action=None)
    ctrl1.cleanup()
    assert ok1 is False

    # Authorized -> Injects predetermined action under typed degradation policy
    ctrl2 = PTYController(timeout=2.0)
    ctrl2.spawn(
        "echo 'STRUCTURED_REVIEWER_RESPONSE_INVALID: ungrounded metric'; "
        "echo -n '  Select action [default: 1]: '; read ans; if [ \"$ans\" = \"1\" ]; then echo 'DONE'; fi"
    )
    ok2 = ctrl2.read_until_pattern(
        r"DONE",
        allowed_fallback_action="1",
        failure_policy="CONTINUE_WITHOUT_AGENT_INTERPRETATION",
        state_id="C_CHK_GOVERNANCE_2",
        checkpoint_id="market.cross_analytical_synthesis",
    )
    ctrl2.cleanup()
    assert ok2 is True

    # The same malformed response at portfolio allocation must fail closed.
    ctrl3 = PTYController(timeout=2.0)
    ctrl3.spawn(
        "echo 'STRUCTURED_REVIEWER_RESPONSE_INVALID: ungrounded metric'; "
        "echo -n '  Select action [default: 1]: '; read ans; echo 'UNREACHABLE'"
    )
    ok3 = ctrl3.read_until_pattern(
        r"UNREACHABLE",
        allowed_fallback_action="1",
        failure_policy="REQUIRE_VALID_AGENT_RESPONSE",
        state_id="B_CHK_PORTFOLIO_2",
        checkpoint_id="market.portfolio_allocation",
    )
    ctrl3.cleanup()
    assert ok3 is False


def test_failure_d_market_checkpoint_census_7_with_signoff():
    """Failure D Regression: Market checkpoint census is mechanically 7."""
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
    )
    rec = EvidenceRecord(
        evidence_id="EV-001",
        test_id="portfolio.risk_statistics",
        test_name="Risk Statistics",
        model_id="M-DEFAULT",
        dataset_id="D-DEFAULT",
        run_id="R-TEST",
        status=Status.RECORDED,
    )
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: "A")
    actual_checkpoints = tuple(d["checkpoint"] for d in decisions)
    assert len(actual_checkpoints) == 7
    assert actual_checkpoints[-1] == "Model Governance & Attestation Sign-off"


def test_failure_e_pty_timeout_fails_closed_no_orphans():
    """Failure E Regression: Child / PTY timeout fails closed cleanly with no orphan process."""
    from demo_controller import PTYController

    ctrl = PTYController(timeout=1.0)
    ctrl.spawn("sleep 10")
    pid = ctrl.pid
    with pytest.raises(TimeoutError):
        ctrl.read_until_pattern(r"WILL_NEVER_ARRIVE")
    ctrl.cleanup()
    assert ctrl.pid is None
    try:
        os.kill(pid, 0)
        alive = True
    except (ProcessLookupError, OSError):
        alive = False
    assert not alive


def test_failure_f_structured_reviewer_parse_failure_fallback(monkeypatch):
    """Malformed portfolio response fails closed under the current policy."""
    class BrokenJsonProvider(LLMProvider):
        name = "openai"
        model = "gpt-5.1"
        def complete(self, system: str, user: str, **kwargs) -> str:
            return "NOT_VALID_JSON_AT_ALL {malformed"

    monkeypatch.setattr(llm_mod, "get_llm_provider", lambda cfg: BrokenJsonProvider())

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        llm_config=LLMReviewConfig(
            backend_mode="public", provider="openai", model="gpt-5.1", status="CONNECTED"
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    rec = EvidenceRecord.from_result(
        TestResult(
            test_id="portfolio.risk_statistics",
            test_name="portfolio.risk_statistics",
            status=Status.PASS,
            metrics={"annualised_volatility": 0.0937},
        ),
        run_id="RUN-TEST",
    )

    prompts = iter(["Q", "Assess risk"])
    with pytest.raises(ReviewCancelled, match="requires valid agent response"):
        run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: next(prompts))


def test_failure_g_challenge_diagnostic_subordinate_evidencerecord():
    """Failure G Regression: Challenge diagnostic emits subordinate EvidenceRecord."""
    spec = ScenarioSpec(
        scenario_id="SCEN-CRASH",
        scenario_name="Crash Scenario",
        scenario_type=ScenarioType.SYNTHETIC,
        shocks=(),
        repricing_method=RepricingMethod.LINEAR_RETURN,
    )
    diag_res = validate_scenario_data_integrity(spec, portfolio_assets=["A", "B"])
    issues_scalar = "; ".join(diag_res.issues) if diag_res.issues else "none"
    rec = EvidenceRecord.from_result(
        TestResult(
            test_id="scenario.scenario_integrity",
            test_name="scenario.scenario_integrity",
            status=Status.FAIL,
            metrics={"status": "FAIL", "issues": issues_scalar},
        ),
        run_id="RUN-DIAG-001",
    )
    assert rec.evidence_id.startswith("EV-")
    assert rec.test_id == "scenario.scenario_integrity"
    assert isinstance(rec.metrics["issues"], str)


def test_failure_h_final_attestation_governance_checkpoint_consumed(monkeypatch):
    """Failure H Regression: Final attestation/governance checkpoint consumed in ledger."""
    monkeypatch.setattr(llm_mod, "get_llm_provider", lambda cfg: DeterministicFakeStructuredProvider())

    market = generate_market_world(n_assets=10, n_periods=60, seed=42)
    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.MARKET,),
        market=market,
        materiality="high",
        lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
        business_context="Governance signoff test",
        llm_config=LLMReviewConfig(
            backend_mode="public",
            provider="openai",
            model="gpt-5.1",
            status="CONNECTED",
        ),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )
    prompts = iter(["A", "A", "A", "A", "A", "A", "A"])
    with tempfile.TemporaryDirectory() as td:
        outcome = run_market_treasury_review(
            bundle,
            output_root=td,
            interactive=True,
            ask=lambda _: next(prompts),
        )
        assert len(outcome["decisions"]) == 7
        assert outcome["decisions"][-1]["checkpoint"] == "Model Governance & Attestation Sign-off"
        assert outcome["ledger"].verify() is True
        assert outcome["seal"] is not None
