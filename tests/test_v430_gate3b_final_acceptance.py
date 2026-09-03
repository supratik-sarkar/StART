"""Comprehensive Gate 3B Final Acceptance Audit Test Suite.

Verifies:
1. Adversarial Challenge Diagnostic Evidence Bridge & State Machine
   - Tool success alone != NO_BREACH
   - Generated EvidenceRecord is distinct from triggering source EvidenceRecord
   - Generated evidence resolves in context
   - Same source evidence + different parameters -> distinct deterministic evidence
   - Missing materiality threshold -> RESOLVED_EVIDENCE_ONLY with decision_criterion="NONE"
   - Explicit policy threshold -> deterministic evaluation (NO_BREACH vs RESOLVED_FINDING)
   - Tool failure -> BLOCKED / UNRESOLVED
   - Governance handles RESOLVED_EVIDENCE_ONLY as ACCEPT_WITH_CONDITIONS
2. Financial Unit / Horizon Contracts
   - Periodic mu + periodic cov + ppy=252: VALID
   - Annual mu + annual cov + ppy=1: VALID
   - Periodic mu + annual cov mismatch: REJECTED
   - Annual mu + periodic cov mismatch: REJECTED
   - Already-annual inputs + ppy=252: REJECTED (double annualization)
   - 252 periods/year with non-daily frequency: REJECTED (no false daily inference)
3. Group Coverage Policy Contracts
   - EXHAUSTIVE_PARTITION rejects unmapped assets
   - OPTIONAL_UNMAPPED_ALLOWED permits unmapped assets
   - DISJOINT_GROUPS_REQUIRED rejects overlapping assets
   - Unknown assets fail closed
4. HERC Algorithm Identity & Analytical Reference Fixture
   - 4-asset analytical reference benchmark: w = [1/6, 1/6, 1/3, 1/3]
   - Cluster risk equality: W(C_L) * sqrt(v_L) == W(C_R) * sqrt(v_R)
   - HERC != HRP mathematical distinction proof (HRP = [0.10, 0.10, 0.40, 0.40])
   - Permutation and scale invariance
"""

import math

import numpy as np
import pandas as pd
import pytest

from start.agents.market_review import (
    AdversarialChallengeAgent,
    GovernanceAgent,
)
from start.core.schemas import EvidenceRecord, Status
from start.portfolio import (
    ChallengeState,
    GroupConstraintSpec,
    GroupCoveragePolicy,
    MetricHorizon,
    UncertaintyDerivationPolicy,
    hrp_weights_and_tree,
    solve_herc,
    solve_robust_mvo,
    validate_horizon_alignment,
)


# =========================================================================== #
# FIXTURES
# =========================================================================== #
@pytest.fixture
def market_universe():
    np.random.seed(42)
    assets = ["SPY", "QQQ", "TLT", "GLD"]
    vols = np.array([0.16, 0.22, 0.14, 0.15])
    corr = np.array([
        [1.00, 0.80, -0.20, 0.05],
        [0.80, 1.00, -0.30, 0.10],
        [-0.20, -0.30, 1.00, 0.20],
        [0.05, 0.10, 0.20, 1.00],
    ])
    cov_annual = np.diag(vols) @ corr @ np.diag(vols)
    cov_periodic = cov_annual / 252.0
    mu_annual = np.array([0.09, 0.12, 0.035, 0.06])
    mu_periodic = mu_annual / 252.0

    returns_df = pd.DataFrame(
        np.random.multivariate_normal(mu_periodic, cov_periodic, size=250),
        columns=assets,
    )
    current_weights = {"SPY": 0.40, "QQQ": 0.20, "TLT": 0.25, "GLD": 0.15}
    benchmark_weights = {"SPY": 0.35, "QQQ": 0.25, "TLT": 0.25, "GLD": 0.15}

    return {
        "assets": assets,
        "cov_annual": cov_annual,
        "cov_periodic": cov_periodic,
        "mu_annual": mu_annual,
        "mu_periodic": mu_periodic,
        "returns_df": returns_df,
        "current_weights": current_weights,
        "benchmark_weights": benchmark_weights,
        "P": np.array([[1.0, 0.0, 0.0, 0.0]]),
        "Q": np.array([0.10]),
    }


# =========================================================================== #
# 1. CHALLENGE DIAGNOSTIC EVIDENCE & PROVENANCE TESTS
# =========================================================================== #
def test_tool_success_alone_cannot_produce_no_breach(market_universe):
    """Successful tool execution with no explicit threshold MUST NOT yield RESOLVED_NO_BREACH."""
    agent = AdversarialChallengeAgent()
    u = market_universe

    challenge = {
        "challenge_id": "CHAL-TEST-NO-THRESH",
        "target_area": "View Dominance",
        "challenge_question": "Is BL allocation overly sensitive to tau?",
        "evidence_ids": ("EV-ORIG-001",),
        "required_tool": "solve_black_litterman",
        "parameters": {"tau": 0.05},
        # No decision_criterion or threshold supplied
    }
    context = {
        "covariance": u["cov_annual"],
        "market_weights": u["benchmark_weights"],
        "P": u["P"],
        "Q": u["Q"],
        "assets": u["assets"],
    }

    resolution = agent.resolve_challenge(challenge, context)
    assert resolution.status == ChallengeState.RESOLVED_EVIDENCE_ONLY
    assert resolution.status != ChallengeState.RESOLVED_NO_BREACH
    assert resolution.decision_criterion == "NONE"
    assert "materiality threshold not adjudicated" in resolution.limitations[0]


def test_challenge_creates_new_diagnostic_evidence_differing_from_source(market_universe):
    """Challenge resolution must create a NEW diagnostic EvidenceRecord with a different ID from source."""
    agent = AdversarialChallengeAgent()
    u = market_universe

    source_ev_id = "EV-SOURCE-TRIGGER-1234"
    challenge = {
        "challenge_id": "CHAL-TEST-NEW-EV",
        "target_area": "Robust Concentration",
        "challenge_question": "Does robust MVO concentrate under radius 0.50?",
        "evidence_ids": (source_ev_id,),
        "required_tool": "robust_mvo_sensitivity_grid",
        "parameters": {"radii": (0.0, 0.25, 0.50, 1.0)},
    }
    ev_pool: list[EvidenceRecord] = []
    context = {
        "mu": u["mu_annual"],
        "covariance": u["cov_annual"],
        "assets": u["assets"],
        "uncertainty_policy": UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N,
        "n_observations": 250,
        "evidence_records": ev_pool,
    }

    res = agent.resolve_challenge(challenge, context)
    assert len(res.generated_evidence_ids) == 1
    new_diag_id = res.generated_evidence_ids[0]

    # Must NOT reuse the triggering source ID
    assert new_diag_id != source_ev_id
    assert res.source_evidence_ids == (source_ev_id,)

    # The new evidence must be present in the evidence pool
    assert len(ev_pool) == 1
    diag_record = ev_pool[0]
    assert diag_record.evidence_id == new_diag_id
    assert diag_record.test_id == "portfolio.adversarial.robust_mvo_sensitivity"
    assert "radii_count" in diag_record.metrics


def test_distinct_evidence_for_different_parameters(market_universe):
    """Same source evidence with different parameters must produce distinct diagnostic evidence."""
    agent = AdversarialChallengeAgent()
    u = market_universe
    source_ev_id = "EV-SHARED-SOURCE"

    chal_1 = {
        "challenge_id": "CHAL-TAU-LOW",
        "evidence_ids": (source_ev_id,),
        "required_tool": "solve_black_litterman",
        "parameters": {"tau": 0.01},
    }
    chal_2 = {
        "challenge_id": "CHAL-TAU-HIGH",
        "evidence_ids": (source_ev_id,),
        "required_tool": "solve_black_litterman",
        "parameters": {"tau": 0.10},
    }
    context = {
        "covariance": u["cov_annual"],
        "market_weights": u["benchmark_weights"],
        "P": u["P"],
        "Q": u["Q"],
        "assets": u["assets"],
    }

    res_1 = agent.resolve_challenge(chal_1, context)
    res_2 = agent.resolve_challenge(chal_2, context)

    ev_id_1 = res_1.generated_evidence_ids[0]
    ev_id_2 = res_2.generated_evidence_ids[0]

    assert ev_id_1 != ev_id_2
    assert ev_id_1 != source_ev_id
    assert ev_id_2 != source_ev_id


def test_explicit_policy_threshold_evaluates_breach_or_no_breach(market_universe):
    """Explicit decision policy threshold deterministically classifies NO_BREACH vs RESOLVED_FINDING."""
    agent = AdversarialChallengeAgent()
    u = market_universe

    # 1. Satisfied policy: max acceptable turnover = 0.80 (actual BL turnover is ~0.40)
    chal_pass = {
        "challenge_id": "CHAL-POLICY-PASS",
        "evidence_ids": ("EV-BL-01",),
        "required_tool": "solve_black_litterman",
        "parameters": {"tau": 0.05},
        "decision_criterion": "USER_POLICY",
        "decision_threshold": 0.80,
    }
    context = {
        "covariance": u["cov_annual"],
        "market_weights": u["benchmark_weights"],
        "P": u["P"],
        "Q": u["Q"],
        "assets": u["assets"],
    }
    res_pass = agent.resolve_challenge(chal_pass, context)
    assert res_pass.status == ChallengeState.RESOLVED_NO_BREACH
    assert res_pass.decision_criterion == "USER_POLICY"
    assert res_pass.decision_threshold == 0.80
    assert len(res_pass.finding_ids) == 0

    # 2. Breached policy: max acceptable turnover = 0.05 (breached!)
    chal_fail = {
        "challenge_id": "CHAL-POLICY-FAIL",
        "evidence_ids": ("EV-BL-01",),
        "required_tool": "solve_black_litterman",
        "parameters": {"tau": 0.05},
        "decision_criterion": "MODEL_CONTRACT",
        "decision_threshold": 0.05,
    }
    res_fail = agent.resolve_challenge(chal_fail, context)
    assert res_fail.status == ChallengeState.RESOLVED_FINDING
    assert res_fail.decision_criterion == "MODEL_CONTRACT"
    assert len(res_fail.finding_ids) == 1
    assert "FIND-CHAL-CHAL-POLICY-FAIL" in res_fail.finding_ids


def test_tool_failure_yields_unresolved_or_blocked():
    """Unallowed tool yields BLOCKED; missing parameter/exception yields UNRESOLVED."""
    agent = AdversarialChallengeAgent()

    # Unallowed tool
    res_blocked = agent.resolve_challenge(
        {"challenge_id": "CHAL-BAD-TOOL", "required_tool": "unauthorized_arbitrary_eval"},
        {},
    )
    assert res_blocked.status == ChallengeState.BLOCKED

    # Allowed tool but missing required inputs
    res_unresolved = agent.resolve_challenge(
        {"challenge_id": "CHAL-MISSING-INPUTS", "required_tool": "solve_cvar_portfolio"},
        {},
    )
    assert res_unresolved.status == ChallengeState.UNRESOLVED
    assert "Challenge resolution failed" in res_unresolved.limitations[0]


def test_governance_evidence_only_yields_accept_with_conditions():
    """Governance evaluates challenges with RESOLVED_EVIDENCE_ONLY as ACCEPT_WITH_CONDITIONS."""
    gov = GovernanceAgent()
    records = [
        EvidenceRecord(
            evidence_id="EV-TEST-1",
            test_id="portfolio.cvar",
            test_name="CVaR",
            status=Status.RECORDED,
            run_id="RUN-1",
            model_id="MOD-1",
            dataset_id="DS-1",
            metrics={"converged": True, "is_valid": True},
        )
    ]
    challenges = [{"challenge_id": "CHAL-TAIL", "status": "RESOLVED_EVIDENCE_ONLY"}]
    resolutions = [{"challenge_id": "CHAL-TAIL", "status": "RESOLVED_EVIDENCE_ONLY"}]

    signoff = gov.evaluate_signoff(
        critic_disposition="READY_FOR_GOVERNANCE",
        challenges=challenges,
        findings=[],
        records=records,
        resolutions=resolutions,
    )
    assert signoff["verdict"] == "ACCEPT_WITH_CONDITIONS"
    assert "explicit materiality threshold policy" in signoff["reason"]
    assert len(signoff["conditions"]) == 1


# =========================================================================== #
# 2. FINANCIAL UNIT / HORIZON CONTRACT TESTS
# =========================================================================== #
def test_unit_contract_periodic_periodic_ppy252_valid(market_universe):
    """Periodic mu + periodic cov + ppy=252 is valid."""
    u = market_universe
    validate_horizon_alignment(
        mu_horizon=MetricHorizon.PERIODIC,
        cov_horizon=MetricHorizon.PERIODIC,
        periods_per_year=252.0,
    )
    rob_res = solve_robust_mvo(
        mu=u["mu_periodic"],
        covariance=u["cov_periodic"],
        uncertainty_radius=0.20,
        uncertainty_policy=UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N,
        n_observations=250,
        assets=u["assets"],
        periods_per_year=252.0,
        mu_horizon=MetricHorizon.PERIODIC,
        cov_horizon=MetricHorizon.PERIODIC,
    )
    assert rob_res.usable_solution is True


def test_unit_contract_annual_annual_ppy1_valid(market_universe):
    """Annual mu + annual cov + ppy=1.0 is valid."""
    u = market_universe
    validate_horizon_alignment(
        mu_horizon=MetricHorizon.ANNUAL,
        cov_horizon=MetricHorizon.ANNUAL,
        periods_per_year=1.0,
    )
    rob_res = solve_robust_mvo(
        mu=u["mu_annual"],
        covariance=u["cov_annual"],
        uncertainty_radius=0.50,
        uncertainty_policy=UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N,
        n_observations=250,
        assets=u["assets"],
        periods_per_year=1.0,
        mu_horizon=MetricHorizon.ANNUAL,
        cov_horizon=MetricHorizon.ANNUAL,
    )
    assert rob_res.usable_solution is True


def test_unit_contract_periodic_annual_mismatch_rejected(market_universe):
    """Periodic mu + annual covariance must be rejected fail-closed."""
    u = market_universe
    with pytest.raises(ValueError, match="Financial horizon mismatch"):
        validate_horizon_alignment(
            mu_horizon=MetricHorizon.PERIODIC,
            cov_horizon=MetricHorizon.ANNUAL,
        )

    with pytest.raises(ValueError, match="Financial horizon mismatch"):
        solve_robust_mvo(
            mu=u["mu_periodic"],
            covariance=u["cov_annual"],
            uncertainty_radius=0.50,
            uncertainty_policy=UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N,
            assets=u["assets"],
            mu_horizon=MetricHorizon.PERIODIC,
            cov_horizon=MetricHorizon.ANNUAL,
        )


def test_unit_contract_annual_periodic_mismatch_rejected(market_universe):
    """Annual mu + periodic covariance must be rejected fail-closed."""
    u = market_universe
    with pytest.raises(ValueError, match="Financial horizon mismatch"):
        solve_robust_mvo(
            mu=u["mu_annual"],
            covariance=u["cov_periodic"],
            uncertainty_radius=0.50,
            uncertainty_policy=UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N,
            assets=u["assets"],
            mu_horizon=MetricHorizon.ANNUAL,
            cov_horizon=MetricHorizon.PERIODIC,
        )


def test_unit_contract_double_annualization_rejected(market_universe):
    """Already-annual inputs with periods_per_year > 1.0 must be rejected (double annualization)."""
    u = market_universe
    with pytest.raises(ValueError, match="Double annualization error"):
        validate_horizon_alignment(
            mu_horizon=MetricHorizon.ANNUAL,
            cov_horizon=MetricHorizon.ANNUAL,
            periods_per_year=252.0,  # Error: already annual but ppy=252!
        )

    with pytest.raises(ValueError, match="Double annualization error"):
        solve_robust_mvo(
            mu=u["mu_annual"],
            covariance=u["cov_annual"],
            uncertainty_radius=0.50,
            uncertainty_policy=UncertaintyDerivationPolicy.SAMPLE_COVARIANCE_DIV_N,
            assets=u["assets"],
            periods_per_year=252.0,
            mu_horizon=MetricHorizon.ANNUAL,
            cov_horizon=MetricHorizon.ANNUAL,
        )


def test_unit_contract_ppy252_alone_does_not_infer_daily():
    """252 periods_per_year with non-daily frequency metadata must raise contradiction error."""
    with pytest.raises(ValueError, match="Frequency contradiction"):
        validate_horizon_alignment(
            mu_horizon=MetricHorizon.PERIODIC,
            cov_horizon=MetricHorizon.PERIODIC,
            periods_per_year=252.0,
            frequency="monthly",  # Contradiction: 252 is not monthly!
        )


# =========================================================================== #
# 3. GROUP COVERAGE POLICY TESTS
# =========================================================================== #
def test_group_coverage_exhaustive_rejects_unmapped_asset():
    """EXHAUSTIVE_PARTITION policy fails closed when any asset in universe is unmapped."""
    spec = GroupConstraintSpec(
        group_name="Sector",
        memberships={"Tech": ("AAPL", "MSFT")},  # GOOG is unmapped
        coverage_policy=GroupCoveragePolicy.EXHAUSTIVE_PARTITION,
    )
    with pytest.raises(ValueError, match="Unmapped asset.*not permitted under EXHAUSTIVE_PARTITION"):
        spec.validate_asset_coverage(assets=["AAPL", "MSFT", "GOOG"])


def test_group_coverage_optional_allows_unmapped_asset():
    """OPTIONAL_UNMAPPED_ALLOWED policy permits unmapped assets."""
    spec = GroupConstraintSpec(
        group_name="Sector",
        memberships={"Tech": ("AAPL", "MSFT")},
        coverage_policy=GroupCoveragePolicy.OPTIONAL_UNMAPPED_ALLOWED,
    )
    # Should not raise
    spec.validate_asset_coverage(assets=["AAPL", "MSFT", "GOOG"])


def test_group_coverage_disjoint_rejects_overlapping_assets():
    """DISJOINT_GROUPS_REQUIRED policy fails closed when an asset belongs to multiple groups."""
    spec = GroupConstraintSpec(
        group_name="Sector",
        memberships={
            "Tech": ("AAPL", "MSFT"),
            "Hardware": ("AAPL", "DELL"),  # AAPL overlaps!
        },
        coverage_policy=GroupCoveragePolicy.DISJOINT_GROUPS_REQUIRED,
    )
    with pytest.raises(ValueError, match="Overlapping group membership.*not permitted under DISJOINT_GROUPS_REQUIRED"):
        spec.validate_asset_coverage(assets=["AAPL", "MSFT", "DELL"])


def test_group_coverage_unknown_asset_fails_closed():
    """Group membership containing assets outside universe fails closed."""
    spec = GroupConstraintSpec(
        group_name="Sector",
        memberships={"Tech": ("AAPL", "UNKNOWN_XYZ")},
    )
    with pytest.raises(ValueError, match="unknown assets"):
        spec.validate_asset_coverage(assets=["AAPL", "MSFT"])


# =========================================================================== #
# 4. HERC ALGORITHM IDENTITY & REFERENCE FIXTURE TESTS
# =========================================================================== #
def test_herc_analytical_4asset_reference_fixture():
    """Exact analytical benchmark fixture for HERC (Raffinot, 2018).

    Structure:
    4 uncorrelated assets:
    Cluster Left (L): A0 (var=0.04), A1 (var=0.04) -> v_L = 0.02, std_L = sqrt(0.02)
    Cluster Right (R): A2 (var=0.01), A3 (var=0.01) -> v_R = 0.005, std_R = sqrt(0.005)

    Top Split:
    alpha = sqrt(v_R) / (sqrt(v_L) + sqrt(v_R)) = 1 / 3
    Weight to Cluster Left = 1/3
    Weight to Cluster Right = 2/3

    Sub-cluster splits:
    Left (equal var): A0 = (1/3)*0.5 = 1/6, A1 = 1/6
    Right (equal var): A2 = (2/3)*0.5 = 1/3, A3 = 1/3

    Expected analytical weights: w_HERC = [1/6, 1/6, 1/3, 1/3]
    """
    cov_diag = np.diag([0.04, 0.04, 0.01, 0.01])
    assets = ["A0", "A1", "A2", "A3"]

    herc_res = solve_herc(
        cov_diag,
        assets=assets,
        linkage_method="single",
        risk_measure="volatility",
    )

    w = np.array([herc_res.weights[a] for a in assets])
    expected_w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0])

    np.testing.assert_allclose(w, expected_w, atol=1e-5)

    # Verify cluster risk equality between Left and Right branches
    w_L = herc_res.weights["A0"] + herc_res.weights["A1"]  # 1/3
    w_R = herc_res.weights["A2"] + herc_res.weights["A3"]  # 2/3
    std_L = math.sqrt(0.5**2 * 0.04 + 0.5**2 * 0.04)  # sqrt(0.02)
    std_R = math.sqrt(0.5**2 * 0.01 + 0.5**2 * 0.01)  # sqrt(0.005)

    risk_contrib_L = w_L * std_L
    risk_contrib_R = w_R * std_R

    # Equal risk contribution across dendrogram clusters:
    assert math.isclose(risk_contrib_L, risk_contrib_R, rel_tol=1e-5)


def test_herc_distinct_from_hrp_on_reference_fixture():
    """HERC (cluster risk parity) is mathematically distinct from HRP (inverse-variance bisection).

    On the 4-asset fixture:
    - HERC: [1/6, 1/6, 1/3, 1/3] = [0.1667, 0.1667, 0.3333, 0.3333]
    - HRP:  [0.10, 0.10, 0.40, 0.40]
    """
    cov_diag = np.diag([0.04, 0.04, 0.01, 0.01])
    assets = ["A0", "A1", "A2", "A3"]

    herc_res = solve_herc(cov_diag, assets=assets, linkage_method="single")
    hrp_w, _ = hrp_weights_and_tree(cov_diag, assets=assets, linkage_method="single")

    w_herc = np.array([herc_res.weights[a] for a in assets])
    w_hrp = np.array([hrp_w[a] for a in assets])

    expected_hrp = np.array([0.10, 0.10, 0.40, 0.40])
    np.testing.assert_allclose(w_hrp, expected_hrp, atol=1e-5)

    # Assert mathematical distinction
    assert not np.allclose(w_herc, w_hrp, atol=1e-3)
    assert np.max(np.abs(w_herc - w_hrp)) > 0.05


def test_herc_permutation_invariance(market_universe):
    """Permuting the asset column order must produce identical weights per asset."""
    u = market_universe
    cov = u["cov_annual"]
    assets = u["assets"]

    res_orig = solve_herc(cov, assets=assets, linkage_method="single")

    # Permute assets
    perm_idx = [2, 0, 3, 1]
    perm_assets = [assets[i] for i in perm_idx]
    perm_cov = cov[np.ix_(perm_idx, perm_idx)]

    res_perm = solve_herc(perm_cov, assets=perm_assets, linkage_method="single")

    for a in assets:
        assert math.isclose(res_orig.weights[a], res_perm.weights[a], abs_tol=1e-6)


def test_herc_scale_invariance(market_universe):
    """Multiplying the covariance matrix by constant c > 0 preserves exact HERC weights."""
    u = market_universe
    cov = u["cov_annual"]
    assets = u["assets"]

    res_1 = solve_herc(cov, assets=assets, linkage_method="single")
    res_100 = solve_herc(cov * 100.0, assets=assets, linkage_method="single")

    for a in assets:
        assert math.isclose(res_1.weights[a], res_100.weights[a], abs_tol=1e-6)


def test_herc_unambiguous_correlation_structure_debt():
    """Non-blocking HERC test debt verification (Gate 3 Item #62 / Gate 4 Amendment 22).

    Verify HERC on an unambiguous 4-asset block correlation structure where within-cluster
    correlations (0.80) are strictly higher than cross-cluster correlations (0.05).
    This proves that cluster partitioning is mathematically driven by the correlation geometry
    and is not a scipy single-linkage tie-order artifact.
    """
    stds = np.array([0.20, 0.20, 0.10, 0.10])
    corr = np.array([
        [1.00, 0.80, 0.05, 0.05],
        [0.80, 1.00, 0.05, 0.05],
        [0.05, 0.05, 1.00, 0.80],
        [0.05, 0.05, 0.80, 1.00],
    ])
    cov = np.outer(stds, stds) * corr
    assets = ["A0", "A1", "A2", "A3"]

    # Solve across single, complete, and average linkage
    for linkage in ("single", "complete", "average"):
        res = solve_herc(cov, assets=assets, linkage_method=linkage)
        assert res.converged
        assert res.usable_solution

        # Symmetry within clusters: A0 == A1, A2 == A3
        assert math.isclose(res.weights["A0"], res.weights["A1"], rel_tol=1e-5)
        assert math.isclose(res.weights["A2"], res.weights["A3"], rel_tol=1e-5)

        # Higher allocation to lower-volatility cluster {A2, A3}
        assert res.weights["A2"] > res.weights["A0"]
        assert math.isclose(sum(res.weights.values()), 1.0, abs_tol=1e-6)

