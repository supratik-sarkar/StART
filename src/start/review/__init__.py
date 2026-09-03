"""v4.3.0 interactive review architecture.

Routing vocabulary, multiline governance input and registry-driven applicability.
Nothing here computes an analytical result: the interactive layer selects and orders,
and registered deterministic engines compute.
"""

from start.review.applicability import (
    ApplicableTests,
    ReviewPlanPreview,
    applicable_tests,
    build_plan_preview,
)
from start.review.architecture import (
    DOMAIN_CONTEXT,
    DOMAIN_LABELS,
    TRADITIONAL_ML_MODELS,
    LLMReviewConfig,
    PredictiveTechnology,
    ReviewContextBundle,
    ReviewDomain,
    ReviewGroundingMode,
    ReviewLifecycle,
    ReviewMode,
    parse_domain_selection,
    required_context_types,
    requires_predictive_technology,
)
from start.review.evidence_view import (
    CheckpointEvidenceMetric,
    CheckpointEvidenceView,
    CheckpointMetricRef,
    build_checkpoint_evidence_view,
)
from start.review.executor import (
    execute_market_treasury_tests,
    run_domain_checkpoints,
    run_market_treasury_review,
)
from start.review.multiline_input import (
    MULTILINE_TERMINATOR,
    ReviewCancelled,
    read_multiline_text,
)
from start.review.state_machine import (
    CheckpointState,
    CheckpointStateMachine,
    GroundingRepairError,
    GroundingValidationError,
    InvalidStateTransitionError,
    ProviderInvocationError,
)
from start.review.structured_contract import (
    ReviewerAssessment,
    ReviewerObservation,
    format_assessment_markdown,
    hydrate_assessment_values,
)
from start.review.wizard import run_review_wizard

__all__ = [
    "ReviewMode", "ReviewDomain", "PredictiveTechnology", "ReviewLifecycle",
    "ReviewGroundingMode",
    "LLMReviewConfig", "ReviewContextBundle", "DOMAIN_CONTEXT", "DOMAIN_LABELS",
    "TRADITIONAL_ML_MODELS", "parse_domain_selection", "required_context_types",
    "requires_predictive_technology", "read_multiline_text", "MULTILINE_TERMINATOR",
    "ReviewCancelled", "applicable_tests", "ApplicableTests", "build_plan_preview",
    "ReviewPlanPreview", "run_review_wizard", "run_market_treasury_review",
    "execute_market_treasury_tests", "run_domain_checkpoints",
    "CheckpointState", "CheckpointStateMachine", "ProviderInvocationError",
    "GroundingValidationError", "GroundingRepairError", "InvalidStateTransitionError",
    "ReviewerAssessment", "ReviewerObservation", "hydrate_assessment_values",
    "format_assessment_markdown", "CheckpointEvidenceMetric", "CheckpointEvidenceView",
    "CheckpointMetricRef", "build_checkpoint_evidence_view",
]

