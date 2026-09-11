"""Authoritative Artifact Generators for Recommender Systems.

Produces typed, cryptographic, inspectable ArtifactRecords:
1. Recommender Metrics Summary Table
2. Ranking Evaluation Breakdown across K cutoffs
3. Interaction Sparsity & Density Analysis
4. Cold-Start Cohort Evaluation
5. Parameter Sensitivity Table
6. Top-K Recommendation Sample Table
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from start.recommender.contracts import (
    ColdStartCohortMetrics,
    RecommenderBeyondAccuracyMetrics,
    RecommenderDatasetProfile,
    RecommenderRankingMetrics,
    RecommenderRatingMetrics,
    RecommenderSensitivityResult,
)


def _compute_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def render_recommender_metrics_artifact(
    run_id: str,
    rating_metrics: RecommenderRatingMetrics | None,
    ranking_metrics: RecommenderRankingMetrics | None,
    beyond_accuracy: RecommenderBeyondAccuracyMetrics | None,
    node_id: str = "step-model",
) -> dict[str, Any]:
    """Produce canonical metrics summary table artifact."""
    metrics_list: list[dict[str, Any]] = []

    if rating_metrics:
        metrics_list.append({"metric": "RMSE", "value": rating_metrics.rmse, "category": "rating_quality", "unit": "error"})
        metrics_list.append({"metric": "MAE", "value": rating_metrics.mae, "category": "rating_quality", "unit": "error"})
        if rating_metrics.r2 is not None:
            metrics_list.append({"metric": "R²", "value": rating_metrics.r2, "category": "rating_quality", "unit": "score"})

    if ranking_metrics:
        for k in ranking_metrics.k_list:
            metrics_list.append({"metric": f"Precision@{k}", "value": ranking_metrics.precision_at_k.get(k, 0.0), "category": "ranking_quality", "unit": "ratio"})
            metrics_list.append({"metric": f"Recall@{k}", "value": ranking_metrics.recall_at_k.get(k, 0.0), "category": "ranking_quality", "unit": "ratio"})
            metrics_list.append({"metric": f"NDCG@{k}", "value": ranking_metrics.ndcg_at_k.get(k, 0.0), "category": "ranking_quality", "unit": "score"})
            metrics_list.append({"metric": f"HitRate@{k}", "value": ranking_metrics.hit_rate_at_k.get(k, 0.0), "category": "ranking_quality", "unit": "ratio"})
        metrics_list.append({"metric": "MRR", "value": ranking_metrics.mrr, "category": "ranking_quality", "unit": "reciprocal_rank"})

    if beyond_accuracy:
        metrics_list.append({"metric": "Catalog Coverage", "value": beyond_accuracy.catalog_coverage, "category": "coverage_diversity", "unit": "ratio"})
        metrics_list.append({"metric": "User Coverage", "value": beyond_accuracy.user_coverage, "category": "coverage_diversity", "unit": "ratio"})
        metrics_list.append({"metric": "Novelty", "value": beyond_accuracy.novelty, "category": "coverage_diversity", "unit": "bits"})
        metrics_list.append({"metric": "Popularity Bias", "value": beyond_accuracy.popularity_bias, "category": "coverage_diversity", "unit": "interactions"})

    payload = {"metrics": metrics_list, "run_id": run_id}
    h = _compute_hash(payload)

    return {
        "artifactId": f"ART-REC-METRICS-{run_id[-6:]}",
        "runId": run_id,
        "label": "Recommender Evaluation Metrics",
        "kind": "table",
        "mimeType": "application/json",
        "createdAt": time.time(),
        "description": "Deterministic rating, ranking, and beyond-accuracy validation metrics.",
        "preview": {"type": "table", "payload": payload},
        "hash": h,
        "producerNodeId": node_id,
    }


def render_recommender_ranking_table_artifact(
    run_id: str,
    ranking_metrics: RecommenderRankingMetrics,
    node_id: str = "step-ranking",
) -> dict[str, Any]:
    """Produce per-cutoff Top-K ranking performance table artifact."""
    rows = []
    for k in ranking_metrics.k_list:
        rows.append({
            "cutoff_k": k,
            "precision": ranking_metrics.precision_at_k.get(k, 0.0),
            "recall": ranking_metrics.recall_at_k.get(k, 0.0),
            "ndcg": ranking_metrics.ndcg_at_k.get(k, 0.0),
            "hit_rate": ranking_metrics.hit_rate_at_k.get(k, 0.0),
            "map": ranking_metrics.map_at_k.get(k, 0.0),
        })

    payload = {"k_evaluations": rows, "mrr": ranking_metrics.mrr}
    h = _compute_hash(payload)

    return {
        "artifactId": f"ART-REC-RANKING-{run_id[-6:]}",
        "runId": run_id,
        "label": "Top-K Ranking Performance Table",
        "kind": "table",
        "mimeType": "application/json",
        "createdAt": time.time(),
        "description": "Cutoff-level Precision@K, Recall@K, NDCG@K, and HitRate@K breakdown.",
        "preview": {"type": "table", "payload": payload},
        "hash": h,
        "producerNodeId": node_id,
    }


def render_recommender_sparsity_artifact(
    run_id: str,
    profile: RecommenderDatasetProfile,
    node_id: str = "step-preflight",
) -> dict[str, Any]:
    """Produce interaction density and matrix sparsity summary artifact."""
    payload = {
        "users": profile.n_users,
        "items": profile.n_items,
        "interactions": profile.n_interactions,
        "sparsity_ratio": profile.sparsity,
        "density_percent": profile.density_percent,
        "interactions_per_user": profile.interactions_per_user,
        "interactions_per_item": profile.interactions_per_item,
        "warnings": profile.warnings,
    }
    h = _compute_hash(payload)

    return {
        "artifactId": f"ART-REC-SPARSITY-{run_id[-6:]}",
        "runId": run_id,
        "label": "Interaction Sparsity & Matrix Profile",
        "kind": "table",
        "mimeType": "application/json",
        "createdAt": time.time(),
        "description": "Interaction matrix dimensions, sparsity ratio, and power-law distribution stats.",
        "preview": {"type": "key-value", "payload": payload},
        "hash": h,
        "producerNodeId": node_id,
    }


def render_recommender_cold_start_artifact(
    run_id: str,
    cold_start: ColdStartCohortMetrics,
    node_id: str = "step-cold-start",
) -> dict[str, Any]:
    """Produce cold-start cohort performance degradation artifact."""
    payload = {
        "cohorts": [
            {
                "cohort": "Warm Users",
                "count": cold_start.warm_users_count,
                "metrics": cold_start.warm_user_metrics,
            },
            {
                "cohort": "Cold Users",
                "count": cold_start.cold_users_count,
                "metrics": cold_start.cold_user_metrics,
            },
        ],
        "degradation_ratio": cold_start.ndcg_degradation_ratio,
        "ndcg_degradation_ratio": cold_start.ndcg_degradation_ratio,
    }
    h = _compute_hash(payload)

    return {
        "artifactId": f"ART-REC-COLDSTART-{run_id[-6:]}",
        "runId": run_id,
        "label": "Cold-Start Cohort Analysis",
        "kind": "table",
        "mimeType": "application/json",
        "createdAt": time.time(),
        "description": "Validation metrics partitioned across warm and cold user interaction histories.",
        "preview": {"type": "table", "payload": payload},
        "hash": h,
        "producerNodeId": node_id,
    }


def render_recommender_sensitivity_artifact(
    run_id: str,
    sensitivity: RecommenderSensitivityResult,
    node_id: str = "step-sensitivity",
) -> dict[str, Any]:
    """Produce parameter sensitivity table artifact."""
    payload = sensitivity.to_dict()
    h = _compute_hash(payload)

    return {
        "artifactId": f"ART-REC-SENSITIVITY-{run_id[-6:]}",
        "runId": run_id,
        "label": "Parameter Sensitivity Evaluation",
        "kind": "table",
        "mimeType": "application/json",
        "createdAt": time.time(),
        "description": "Model performance stability under bounded parameter perturbations.",
        "preview": {"type": "table", "payload": payload},
        "hash": h,
        "producerNodeId": node_id,
    }


def render_recommender_topk_examples_artifact(
    run_id: str,
    recommendations: dict[str, list[str]],
    node_id: str = "step-ranking",
    max_users: int = 5,
) -> dict[str, Any]:
    """Produce concrete Top-K recommendation examples for sample users."""
    sample_users = list(recommendations.keys())[:max_users]
    examples = [{"user_id": u, "recommended_items": recommendations[u]} for u in sample_users]
    payload = {"sample_recommendations": examples}
    h = _compute_hash(payload)

    return {
        "artifactId": f"ART-REC-TOPK-{run_id[-6:]}",
        "runId": run_id,
        "label": "Sample Top-K Recommendation Rankings",
        "kind": "table",
        "mimeType": "application/json",
        "createdAt": time.time(),
        "description": "Concrete item ranking outputs generated for evaluation cohort samples.",
        "preview": {"type": "table", "payload": payload},
        "hash": h,
        "producerNodeId": node_id,
    }
