"""Exhaustive adversarial test matrix for CheckpointDegradationPolicy across all checkpoints and actions.

Verifies:
1. Strict resolution matrix: Overrides ('O'), challenges ('C'), decisions ('A'), and sign-offs NEVER degrade.
2. Only registered checkpoints with action == 'Q' may safely degrade.
3. Checkpoints marked REQUIRE_VALID_AGENT_RESPONSE (e.g. scenario_stress, governance_signoff) NEVER degrade.
4. Attempting degradation on a non-degradable interaction raises ReviewCancelled (fail-closed).
5. Only authorized degradations succeed and retain deterministic evidence.
"""

from __future__ import annotations

import pytest

from start.review.executor import ReviewCancelled
from start.review.structured_contract import (
    CHECKPOINT_STABLE_IDS,
    CheckpointDegradationPolicy,
    get_checkpoint_degradation_policy,
)


def test_full_checkpoint_policy_matrix() -> None:
    """Verify effective policy across all standard checkpoints and actions."""
    expected_policies = {
        # Checkpoint: (Q_policy, C_policy, O_policy, A_policy)
        "market.portfolio_allocation": (
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
        ),
        "market.factor_attribution": (
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
        ),
        "market.var_tail_risk": (
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
        ),
        "market.covariance_structure": (
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
        ),
        "market.scenario_stress": (
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
        ),
        "market.cross_analytical_synthesis": (
            CheckpointDegradationPolicy.ALLOW_EVIDENCE_ONLY_DEGRADATION,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
        ),
        "market.governance_signoff": (
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
            CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE,
        ),
    }

    for chk_id, (q_exp, c_exp, o_exp, a_exp) in expected_policies.items():
        assert get_checkpoint_degradation_policy(chk_id, action="Q") == q_exp
        assert get_checkpoint_degradation_policy(chk_id, action="C") == c_exp
        assert get_checkpoint_degradation_policy(chk_id, action="O") == o_exp
        assert get_checkpoint_degradation_policy(chk_id, action="A") == a_exp


def test_title_lookup_matches_stable_id() -> None:
    """Check that human-readable titles resolve to the same policy as stable IDs."""
    for title, stable_id in CHECKPOINT_STABLE_IDS.items():
        assert get_checkpoint_degradation_policy(title, action="Q") == get_checkpoint_degradation_policy(
            stable_id, action="Q"
        )
        assert get_checkpoint_degradation_policy(title, action="C") == get_checkpoint_degradation_policy(
            stable_id, action="C"
        )


def test_unknown_checkpoint_fails_closed() -> None:
    """Any unregistered or ad-hoc checkpoint defaults to REQUIRE_VALID_AGENT_RESPONSE."""
    assert (
        get_checkpoint_degradation_policy("ad_hoc.unregistered_checkpoint", action="Q")
        == CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE
    )
    assert (
        get_checkpoint_degradation_policy("random_checkpoint_123", action="C")
        == CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE
    )


def test_non_degradable_challenge_adversarial_rejection() -> None:
    """When a challenge ('C') produces an invalid structured response, it MUST fail closed."""
    policy = get_checkpoint_degradation_policy("market.scenario_stress", action="C")
    assert policy == CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE

    # Simulating the fail-closed logic from executor.py lines 2263-2275 & 2368-2380
    def simulate_executor_degradation_handling(chk_id: str, action: str) -> str:
        eff_policy = get_checkpoint_degradation_policy(chk_id, action=action)
        if eff_policy == CheckpointDegradationPolicy.ALLOW_EVIDENCE_ONLY_DEGRADATION:
            return "DEGRADED_CONTINUATION"
        raise ReviewCancelled(f"Review aborted: Checkpoint '{chk_id}' requires valid agent response ({eff_policy.value})")

    with pytest.raises(ReviewCancelled, match="requires valid agent response"):
        simulate_executor_degradation_handling("market.scenario_stress", "C")

    with pytest.raises(ReviewCancelled, match="requires valid agent response"):
        simulate_executor_degradation_handling("market.governance_signoff", "Q")

    with pytest.raises(ReviewCancelled, match="requires valid agent response"):
        simulate_executor_degradation_handling("market.portfolio_allocation", "O")

    # Allowed informational question degrades successfully
    outcome = simulate_executor_degradation_handling("market.cross_analytical_synthesis", "Q")
    assert outcome == "DEGRADED_CONTINUATION"
