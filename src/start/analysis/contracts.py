"""Canonical Analytical Result Contracts for StART v5.2.0.

Provides single-source-of-truth analytical result schemas across all supported model families:
- Predictive ML (Classification & Regression)
- Deep Learning (Neural Networks & Latent Diagnostics)
- Recommender Systems (MF, NCF, FM)
- Portfolio Optimization (HRP, Minimum Variance, Equal Risk Contribution)

Core Invariant:
The deterministic layer owns the complete scientific lifecycle: data selection,
validation, preprocessing, split, model fit, evaluation, diagnostics, explainability,
sensitivity, baseline comparison, deterministic findings, artifacts, and evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def compute_deterministic_hash(data: Any) -> str:
    """Compute SHA-256 fingerprint from any JSON-serializable structure."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass
class CanonicalAnalyticalResult:
    """Authoritative analytical result container emitted by StART deterministic engines."""

    run_id: str
    model_family: str  # "predictive_ml" | "deep_learning" | "recommender" | "portfolio"
    technique: str     # e.g. "hrp", "matrix_factorization", "neural_collaborative_filtering", "torch_mlp", etc.
    task_type: str     # e.g. "binary_classification", "regression", "top_k_ranking", "portfolio_optimization"

    # 1. Data Selection & Semantic Contract
    data_selection: dict[str, Any] = field(default_factory=dict)

    # 2. Data Validation & Diagnostics
    data_validation: dict[str, Any] = field(default_factory=dict)

    # 2b. Preprocessing & Transformations
    preprocessing: dict[str, Any] = field(default_factory=dict)

    # 3. Split & Evaluation Protocol
    split_protocol: dict[str, Any] = field(default_factory=dict)

    # 4. Resolved Hyperparameter Configuration
    resolved_configuration: dict[str, Any] = field(default_factory=dict)

    # 5. Execution Summary
    execution_summary: dict[str, Any] = field(default_factory=dict)

    # 6. Task Performance Metrics
    metrics: dict[str, Any] = field(default_factory=dict)

    # 7. Model & Task Diagnostics
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # 8. Structural Analysis & Explainability
    structural_analysis: dict[str, Any] = field(default_factory=dict)

    # 9. Sensitivity & Stability Validation
    sensitivity: dict[str, Any] = field(default_factory=dict)

    # 10. Baseline / Challenger Comparison
    baseline_comparison: dict[str, Any] | None = None

    # 11. Deterministic Findings
    deterministic_findings: list[dict[str, Any]] = field(default_factory=list)

    # 12. Cryptographic Provenance
    provenance: dict[str, Any] = field(default_factory=dict)

    # 13. Generated Artifact Records
    artifacts: list[dict[str, Any]] = field(default_factory=list)

    # 14. Evidence Records Linkage
    evidence_records: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if "content_hash" not in self.provenance:
            core_payload = {
                "run_id": self.run_id,
                "model_family": self.model_family,
                "technique": self.technique,
                "task_type": self.task_type,
                "metrics": self.metrics,
                "resolved_configuration": self.resolved_configuration,
            }
            self.provenance["content_hash"] = compute_deterministic_hash(core_payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)
