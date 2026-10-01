"""Test proving mechanical census of Market domain checkpoints and preventing drift in demo controller."""

from start.core.schemas import EvidenceRecord, Status
from start.review.architecture import ReviewContextBundle, ReviewDomain, ReviewMode
from start.review.executor import run_domain_checkpoints

EXPECTED_MARKET_CHECKPOINTS = (
    "Portfolio Risk & Volatility Assumptions",
    "Factor Modeling & Attribution Assumptions",
    "VaR Backtesting & Exception Frequency",
    "Covariance Structure & Missing Data Treatment",
    "Scenario Analysis & Stress Testing",
    "Cross-Analytical Committee Synthesis",
    "Model Governance & Attestation Sign-off",
)


def test_market_checkpoint_census_matches_executor_source():
    """Mechanically verify that Market domain executes exactly the 7 expected checkpoints in order."""
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
    
    # Run through domain checkpoints with automated acceptance
    decisions = run_domain_checkpoints(bundle, [rec], interactive=True, ask=lambda _: "A")
    actual_checkpoints = tuple(d["checkpoint"] for d in decisions)
    
    assert len(actual_checkpoints) == 7, f"Expected 7 checkpoints, found {len(actual_checkpoints)}: {actual_checkpoints}"
    assert actual_checkpoints == EXPECTED_MARKET_CHECKPOINTS
    assert actual_checkpoints[6] == "Model Governance & Attestation Sign-off"


def test_demo_controller_maps_all_seven_market_checkpoints():
    """Verify that demo_controller.py STATE_TABLE has explicit transitions matching all 7 checkpoints."""
    from demo_controller import STATE_TABLE
    
    transitions = STATE_TABLE["SESSION_B_MARKET"]
    # Check that portfolio, attribution, var, cov, scenario, governance synthesis, and signoff exist
    state_ids = [t.state_id for t in transitions]
    
    assert "B_CHK_PORTFOLIO_1" in state_ids  # Checkpoint 1
    assert "B_CHK_ATTRIBUTION_1" in state_ids  # Checkpoint 2
    assert "C_CHK_VAR_1" in state_ids  # Checkpoint 3
    assert "C_CHK_COV_1" in state_ids  # Checkpoint 4
    assert "C_CHK_SCENARIO_1" in state_ids  # Checkpoint 5
    assert "C_CHK_GOVERNANCE_1" in state_ids  # Checkpoint 6
    assert "B_CHK_SIGNOFF" in state_ids  # Checkpoint 7: Model Governance & Attestation Sign-off
