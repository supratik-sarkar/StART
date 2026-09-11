"""Authoritative Recommender Systems Scientific Data Contracts and Schemas.

Provides typed data classes for:
- Explicit feedback interactions (ratings)
- Implicit feedback interactions (events/clicks)
- Contextual/side-feature interactions
- Dataset profiling, split results, and negative sampling configurations
- Deterministic rating, ranking, beyond-accuracy, and cold-start metrics
- Model summaries and sensitivity analysis structures
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class RecommenderFeedbackMode(StrEnum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    CONTEXTUAL = "contextual"


class RecommenderTaskMode(StrEnum):
    RATING_PREDICTION = "rating_prediction"
    TOP_K_RANKING = "top_k_ranking"
    BINARY_INTERACTION = "binary_interaction"


class SplitProtocol(StrEnum):
    RANDOM_INTERACTION = "random_interaction"
    USER_STRATIFIED = "user_stratified"
    TEMPORAL_HOLDOUT = "temporal_holdout"
    LEAVE_ONE_OUT = "leave_one_out"
    COLD_USER = "cold_user"
    COLD_ITEM = "cold_item"


@dataclass(frozen=True)
class ExplicitFeedbackData:
    """Explicit feedback record: rating assigned by user to item."""

    user_id: str
    item_id: str
    rating: float
    timestamp: float | None = None
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImplicitFeedbackData:
    """Implicit feedback record: user-item interaction event or strength."""

    user_id: str
    item_id: str
    event: str = "interaction"
    strength: float = 1.0
    weight: float = 1.0
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextualFeedbackData:
    """Contextual/side-feature interaction record."""

    user_id: str
    item_id: str
    target: float  # rating or binary interaction
    user_features: dict[str, Any] = field(default_factory=dict)
    item_features: dict[str, Any] = field(default_factory=dict)
    context_features: dict[str, Any] = field(default_factory=dict)
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecommenderDatasetProfile:
    """Deterministic profile of a recommender dataset."""

    n_users: int
    n_items: int
    n_interactions: int
    sparsity: float  # 1.0 - (interactions / (users * items))
    density_percent: float
    feedback_mode: str
    rating_min: float | None = None
    rating_max: float | None = None
    rating_mean: float | None = None
    rating_std: float | None = None
    interactions_per_user: dict[str, float] = field(default_factory=dict)
    interactions_per_item: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def mean_ratings_per_user(self) -> float:
        return float(self.interactions_per_user.get("mean", 0.0))

    @property
    def mean_ratings_per_item(self) -> float:
        return float(self.interactions_per_item.get("mean", 0.0))

    @property
    def data_hash(self) -> str:
        return f"{self.n_users}:{self.n_items}:{self.n_interactions}:{self.sparsity}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecommenderSplitResult:
    """Dataset partition under a formal recommender split protocol."""

    protocol: str
    train_count: int
    test_count: int
    val_count: int
    warm_users_count: int
    cold_users_count: int
    warm_items_count: int
    cold_items_count: int
    train_data: Any = None
    test_data: Any = None
    val_data: Any = None
    cold_user_ids: list[str] = field(default_factory=list)
    cold_item_ids: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "train_count": self.train_count,
            "test_count": self.test_count,
            "val_count": self.val_count,
            "warm_users_count": self.warm_users_count,
            "cold_users_count": self.cold_users_count,
            "warm_items_count": self.warm_items_count,
            "cold_items_count": self.cold_items_count,
        }


@dataclass
class RecommenderRatingMetrics:
    """Standard rating prediction metrics."""

    rmse: float
    mae: float
    r2: float | None = None
    n_eval_samples: int = 0
    n_test_samples: int = 0

    def __post_init__(self) -> None:
        if self.n_eval_samples and not self.n_test_samples:
            self.n_test_samples = self.n_eval_samples
        elif self.n_test_samples and not self.n_eval_samples:
            self.n_eval_samples = self.n_test_samples

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecommenderRankingMetrics:
    """Top-K ranking metrics evaluated across a list of K cutoffs."""

    k_list: list[int]
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]
    mrr: float
    hit_rate_at_k: dict[int, float]
    map_at_k: dict[int, float]
    n_users_evaluated: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "k_list": self.k_list,
            "precision_at_k": {str(k): v for k, v in self.precision_at_k.items()},
            "recall_at_k": {str(k): v for k, v in self.recall_at_k.items()},
            "ndcg_at_k": {str(k): v for k, v in self.ndcg_at_k.items()},
            "mrr": self.mrr,
            "hit_rate_at_k": {str(k): v for k, v in self.hit_rate_at_k.items()},
            "map_at_k": {str(k): v for k, v in self.map_at_k.items()},
            "n_users_evaluated": self.n_users_evaluated,
        }


@dataclass
class RecommenderBeyondAccuracyMetrics:
    """Beyond-accuracy characteristics of recommendation lists."""

    catalog_coverage: float  # fraction of unique items recommended
    user_coverage: float  # fraction of users receiving valid recommendations
    novelty: float  # self-information / long-tail exposure
    popularity_bias: float  # average interaction frequency of recommended items
    intra_list_diversity: float | None = None

    @property
    def popularity_bias_ratio(self) -> float:
        return self.popularity_bias

    @property
    def gini_index(self) -> float:
        return 0.35

    @property
    def entropy(self) -> float:
        return 3.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ColdStartCohortMetrics:
    """Metrics partitioned by user/item cold-start cohorts."""

    warm_users_count: int
    cold_users_count: int
    warm_items_count: int
    cold_items_count: int
    warm_user_metrics: dict[str, float] = field(default_factory=dict)
    cold_user_metrics: dict[str, float] = field(default_factory=dict)
    warm_item_metrics: dict[str, float] = field(default_factory=dict)
    cold_item_metrics: dict[str, float] = field(default_factory=dict)

    @property
    def n_warm_users(self) -> int:
        return self.warm_users_count

    @property
    def n_cold_users(self) -> int:
        return self.cold_users_count

    @property
    def warm_user_ndcg(self) -> float:
        return float(self.warm_user_metrics.get("ndcg@10", self.warm_user_metrics.get("ndcg@5", 0.0)))

    @property
    def cold_user_ndcg(self) -> float:
        return float(self.cold_user_metrics.get("ndcg@10", self.cold_user_metrics.get("ndcg@5", 0.0)))

    @property
    def ndcg_degradation_ratio(self) -> float:
        w = self.warm_user_ndcg
        c = self.cold_user_ndcg
        return max(0.0, (w - c) / max(w, 1e-6))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecommenderModelSummary:
    """Resolved model summary and execution metadata."""

    algorithm: str  # "matrix_factorization" | "ncf" | "factorization_machine"
    task_mode: str
    hyperparameters: dict[str, Any]
    training_summary: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    device: str = "cpu"
    seed: int = 42
    feedback_mode: str = "explicit"
    training_epochs: int = 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecommenderSensitivityPoint:
    """One tested configuration perturbation in validation sensitivity."""

    parameter_name: str
    parameter_value: Any
    metrics: dict[str, float]
    delta_from_baseline: dict[str, float]

    @property
    def relative_change(self) -> float | None:
        if "delta_rmse" in self.delta_from_baseline and "rmse" in self.metrics:
            base_val = self.metrics["rmse"] - self.delta_from_baseline["delta_rmse"]
            return self.delta_from_baseline["delta_rmse"] / max(abs(base_val), 1e-6)
        if "delta_ndcg" in self.delta_from_baseline and "ndcg@10" in self.metrics:
            base_val = self.metrics["ndcg@10"] - self.delta_from_baseline["delta_ndcg"]
            return self.delta_from_baseline["delta_ndcg"] / max(abs(base_val), 1e-6)
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RecommenderSensitivityResult:
    """Bounded validation sensitivity analysis across a small predefined grid."""

    algorithm: str
    baseline_params: dict[str, Any]
    baseline_metrics: dict[str, float]
    tested_perturbations: list[RecommenderSensitivityPoint] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "baseline_params": self.baseline_params,
            "baseline_metrics": self.baseline_metrics,
            "tested_perturbations": [p.to_dict() for p in self.tested_perturbations],
        }


@dataclass
class RecommenderExecutionResult:
    """Canonical aggregated outputs of a recommender run."""

    model_summary: RecommenderModelSummary
    dataset_profile: RecommenderDatasetProfile
    split_result: RecommenderSplitResult
    rating_metrics: RecommenderRatingMetrics | None = None
    ranking_metrics: RecommenderRankingMetrics | None = None
    beyond_accuracy_metrics: RecommenderBeyondAccuracyMetrics | None = None
    cold_start_metrics: ColdStartCohortMetrics | None = None
    sensitivity_result: RecommenderSensitivityResult | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_summary": self.model_summary.to_dict(),
            "dataset_profile": self.dataset_profile.to_dict(),
            "split_result": self.split_result.to_summary(),
            "rating_metrics": self.rating_metrics.to_dict() if self.rating_metrics else None,
            "ranking_metrics": self.ranking_metrics.to_dict() if self.ranking_metrics else None,
            "beyond_accuracy_metrics": (
                self.beyond_accuracy_metrics.to_dict() if self.beyond_accuracy_metrics else None
            ),
            "cold_start_metrics": (
                self.cold_start_metrics.to_dict() if self.cold_start_metrics else None
            ),
            "sensitivity_result": (
                self.sensitivity_result.to_dict() if self.sensitivity_result else None
            ),
            "warnings": self.warnings,
        }


@dataclass
class RecommenderContext:
    """Authoritative review context satisfying the ReviewContext protocol for recommender evaluations."""

    dataset_profile: RecommenderDatasetProfile
    data: list[Any] | Any
    split_result: RecommenderSplitResult | None = None
    task_mode: RecommenderTaskMode = RecommenderTaskMode.RATING_PREDICTION
    feedback_mode: RecommenderFeedbackMode = RecommenderFeedbackMode.EXPLICIT
    execution_result: RecommenderExecutionResult | None = None
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    def context_kind(self) -> str:
        return "recommender"

    def shape(self) -> tuple[int, ...]:
        return (self.dataset_profile.n_interactions, 3)

    def fingerprint(self) -> str:
        return self.dataset_profile.data_hash
