"""StART consensus, human-adjudicated collision, and cross-analytical reasoning framework."""

from __future__ import annotations

from start.consensus.collisions import (
    ADJUDICATOR_CONSTANT,
    COLLISION_RULES,
    AdjudicationDecision,
    AdjudicationRecord,
    Collision,
    CollisionRule,
    adjudicate_collisions_interactive,
    detect_collisions,
)
from start.consensus.cross_analytical import (
    eval_attribution_vs_factor_risk,
    eval_factor_exposure_vs_scenario_alignment,
    eval_optimization_covariance_sensitivity,
    eval_reconciliation_identity_contradiction,
    eval_solver_convergence_vs_scenario_stress,
    eval_var_frequency_vs_independence,
    eval_var_vs_reverse_stress,
)

__all__ = [
    "ADJUDICATOR_CONSTANT",
    "COLLISION_RULES",
    "CollisionRule",
    "Collision",
    "AdjudicationDecision",
    "AdjudicationRecord",
    "detect_collisions",
    "adjudicate_collisions_interactive",
    "eval_var_frequency_vs_independence",
    "eval_optimization_covariance_sensitivity",
    "eval_factor_exposure_vs_scenario_alignment",
    "eval_reconciliation_identity_contradiction",
    "eval_var_vs_reverse_stress",
    "eval_attribution_vs_factor_risk",
    "eval_solver_convergence_vs_scenario_stress",
]
