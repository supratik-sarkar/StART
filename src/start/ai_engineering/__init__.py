"""AI-engineering layer: executable enterprise integrations as visible stages.

v2.0.0 evolves the original availability-checking stages into real adapters
with the full contract (available/validate/execute/collect_artifacts/
emit_evidence). Each runs its real backend when installed, otherwise reports
unavailability explicitly (with install guidance), still emits evidence, and
remains visible in reports. No fake success, no silent degradation.

The original ``STAGE_ADAPTERS`` / ``run_all_stages`` API is preserved for
backward compatibility.
"""

from start.ai_engineering.adapters import ADAPTER_CLASSES, build_adapters
from start.ai_engineering.base import (
    Artifact,
    BaseAdapter,
    ExecutionResult,
    ValidationResult,
)
from start.ai_engineering.layer import AIEngineeringReport, run_ai_engineering_layer
from start.ai_engineering.stages import (
    STAGE_ADAPTERS,
    StageAdapter,
    StageResult,
    available_stages,
    run_all_stages,
    run_stage,
)
from start.ai_engineering.tracking import (
    ExperimentTracker,
    MLflowExperimentTracker,
    NoOpExperimentTracker,
    TrackingStatus,
)

__all__ = [
    # v2.0.0 adapter framework
    "ADAPTER_CLASSES",
    "AIEngineeringReport",
    "Artifact",
    "BaseAdapter",
    "ExecutionResult",
    "ValidationResult",
    "build_adapters",
    "run_ai_engineering_layer",
    # experiment tracking
    "ExperimentTracker",
    "NoOpExperimentTracker",
    "MLflowExperimentTracker",
    "TrackingStatus",
    # legacy API (preserved)
    "STAGE_ADAPTERS",
    "StageAdapter",
    "StageResult",
    "available_stages",
    "run_all_stages",
    "run_stage",
]
