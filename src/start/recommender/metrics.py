"""Authoritative Recommender Metrics Suite with Known-Answer Mathematical Precision.

Implements:
1. Rating Metrics: RMSE, MAE, R2.
2. Ranking Metrics: Precision@K, Recall@K, NDCG@K, MRR, HitRate@K, MAP@K.
3. Beyond-Accuracy Metrics: Catalog Coverage, User Coverage, Novelty, Popularity Bias.
4. Cold-Start Cohort Metrics Partitioning.
"""

from __future__ import annotations

import math

import numpy as np

from start.recommender.contracts import (
    ColdStartCohortMetrics,
    RecommenderBeyondAccuracyMetrics,
    RecommenderRankingMetrics,
    RecommenderRatingMetrics,
)


def compute_rating_metrics(
    y_true: list[float] | np.ndarray,
    y_pred: list[float] | np.ndarray,
) -> RecommenderRatingMetrics:
    """Compute deterministic rating prediction error metrics."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)

    if len(yt) == 0:
        return RecommenderRatingMetrics(rmse=0.0, mae=0.0, r2=None, n_eval_samples=0)

    diff = yt - yp
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    # R2 score
    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
    ss_res = float(np.sum(diff ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0

    return RecommenderRatingMetrics(
        rmse=round(rmse, 4),
        mae=round(mae, 4),
        r2=round(r2, 4),
        n_eval_samples=len(yt),
    )


def compute_precision_at_k(recommended: list[str], ground_truth: set[str], k: int) -> float:
    """Compute Precision@K = (hits in top K) / K."""
    if k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in ground_truth)
    return float(hits / k)


def compute_recall_at_k(recommended: list[str], ground_truth: set[str], k: int) -> float:
    """Compute Recall@K = (hits in top K) / len(ground_truth)."""
    if not ground_truth or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    hits = sum(1 for item in rec_k if item in ground_truth)
    return float(hits / len(ground_truth))


def compute_ndcg_at_k(recommended: list[str], ground_truth: set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at K (NDCG@K).

    Binary relevance: rel = 1 if item in ground_truth else 0.
    DCG@K = sum_{i=1}^K (rel_i / log2(i + 1)).
    IDCG@K = sum_{i=1}^{min(K, |ground_truth|)} (1 / log2(i + 1)).
    """
    if not ground_truth or k <= 0:
        return 0.0

    rec_k = recommended[:k]
    dcg = 0.0
    for i, item in enumerate(rec_k):
        if item in ground_truth:
            dcg += 1.0 / math.log2(i + 2)  # i=0 -> log2(2) = 1.0

    ideal_hits = min(k, len(ground_truth))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return float(dcg / idcg) if idcg > 0.0 else 0.0


def compute_mrr(recommended: list[str], ground_truth: set[str]) -> float:
    """Compute Mean Reciprocal Rank (MRR) = 1 / rank of first relevant item."""
    for i, item in enumerate(recommended):
        if item in ground_truth:
            return float(1.0 / (i + 1))
    return 0.0


def compute_hit_rate_at_k(recommended: list[str], ground_truth: set[str], k: int) -> float:
    """Compute HitRate@K = 1.0 if at least one hit in top K else 0.0."""
    rec_k = recommended[:k]
    return 1.0 if any(item in ground_truth for item in rec_k) else 0.0


def compute_map_at_k(recommended: list[str], ground_truth: set[str], k: int) -> float:
    """Compute Average Precision at K (AP@K)."""
    if not ground_truth or k <= 0:
        return 0.0

    rec_k = recommended[:k]
    score = 0.0
    hits = 0
    for i, item in enumerate(rec_k):
        if item in ground_truth:
            hits += 1
            score += hits / (i + 1)

    return float(score / min(k, len(ground_truth))) if ground_truth else 0.0


def compute_ranking_metrics(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    k_list: list[int] | None = None,
) -> RecommenderRankingMetrics:
    """Compute micro-averaged Top-K ranking metrics across users."""
    k_list = k_list or [5, 10, 20]
    evaluated_users = [u for u in ground_truth if len(ground_truth[u]) > 0]

    if not evaluated_users:
        return RecommenderRankingMetrics(
            k_list=k_list,
            precision_at_k={k: 0.0 for k in k_list},
            recall_at_k={k: 0.0 for k in k_list},
            ndcg_at_k={k: 0.0 for k in k_list},
            mrr=0.0,
            hit_rate_at_k={k: 0.0 for k in k_list},
            map_at_k={k: 0.0 for k in k_list},
        )

    prec_dict = {k: 0.0 for k in k_list}
    rec_dict = {k: 0.0 for k in k_list}
    ndcg_dict = {k: 0.0 for k in k_list}
    hit_dict = {k: 0.0 for k in k_list}
    map_dict = {k: 0.0 for k in k_list}
    total_mrr = 0.0

    for u in evaluated_users:
        user_recs = recommendations.get(u, [])
        user_gt = ground_truth[u]

        total_mrr += compute_mrr(user_recs, user_gt)
        for k in k_list:
            prec_dict[k] += compute_precision_at_k(user_recs, user_gt, k)
            rec_dict[k] += compute_recall_at_k(user_recs, user_gt, k)
            ndcg_dict[k] += compute_ndcg_at_k(user_recs, user_gt, k)
            hit_dict[k] += compute_hit_rate_at_k(user_recs, user_gt, k)
            map_dict[k] += compute_map_at_k(user_recs, user_gt, k)

    n = len(evaluated_users)
    return RecommenderRankingMetrics(
        k_list=k_list,
        precision_at_k={k: round(prec_dict[k] / n, 4) for k in k_list},
        recall_at_k={k: round(rec_dict[k] / n, 4) for k in k_list},
        ndcg_at_k={k: round(ndcg_dict[k] / n, 4) for k in k_list},
        mrr=round(total_mrr / n, 4),
        hit_rate_at_k={k: round(hit_dict[k] / n, 4) for k in k_list},
        map_at_k={k: round(map_dict[k] / n, 4) for k in k_list},
        n_users_evaluated=n,
    )


def compute_beyond_accuracy_metrics(
    recommendations: dict[str, list[str]],
    all_items: list[str],
    all_users: list[str],
    item_popularity: dict[str, int] | None = None,
    k: int = 10,
) -> RecommenderBeyondAccuracyMetrics:
    """Compute catalog coverage, user coverage, novelty, and popularity bias."""
    rec_items_union = set()
    users_with_recs = 0
    all_rec_items_flat: list[str] = []

    for u in all_users:
        recs = recommendations.get(u, [])[:k]
        if recs:
            users_with_recs += 1
            rec_items_union.update(recs)
            all_rec_items_flat.extend(recs)

    total_items = max(1, len(all_items))
    total_users = max(1, len(all_users))

    cat_coverage = float(len(rec_items_union) / total_items)
    user_coverage = float(users_with_recs / total_users)

    # Popularity bias and Novelty
    pop_bias = 0.0
    novelty = 0.0
    pop_dict = item_popularity if isinstance(item_popularity, dict) else (dict(item_popularity["item_id"].value_counts()) if item_popularity is not None and hasattr(item_popularity, "__getitem__") and "item_id" in item_popularity else None)
    if pop_dict and all_rec_items_flat:
        total_interactions = max(1, sum(pop_dict.values()))
        pop_scores = [pop_dict.get(i, 0) for i in all_rec_items_flat]
        pop_bias = float(np.mean(pop_scores))

        novelty_scores = [
            -math.log2(max(1e-9, pop_dict.get(i, 1) / total_interactions))
            for i in all_rec_items_flat
        ]
        novelty = float(np.mean(novelty_scores))

    return RecommenderBeyondAccuracyMetrics(
        catalog_coverage=round(cat_coverage, 4),
        user_coverage=round(user_coverage, 4),
        novelty=round(novelty, 4),
        popularity_bias=round(pop_bias, 2),
        intra_list_diversity=None,
    )


def compute_cold_start_metrics(
    recommendations: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
    cold_user_ids: list[str],
    cold_item_ids: list[str],
    k: int = 10,
) -> ColdStartCohortMetrics:
    """Evaluate ranking metrics partitioned by cold vs warm user/item cohorts."""
    all_users = set(ground_truth.keys())
    cold_users_set = set(cold_user_ids)
    warm_users_set = all_users - cold_users_set

    warm_gt = {u: ground_truth[u] for u in warm_users_set if u in ground_truth}
    cold_gt = {u: ground_truth[u] for u in cold_users_set if u in ground_truth}

    warm_m = compute_ranking_metrics(recommendations, warm_gt, [k])
    cold_m = compute_ranking_metrics(recommendations, cold_gt, [k])

    return ColdStartCohortMetrics(
        warm_users_count=len(warm_users_set),
        cold_users_count=len(cold_users_set),
        warm_items_count=0,
        cold_items_count=len(cold_item_ids),
        warm_user_metrics={
            f"ndcg@{k}": warm_m.ndcg_at_k.get(k, 0.0),
            f"recall@{k}": warm_m.recall_at_k.get(k, 0.0),
            f"precision@{k}": warm_m.precision_at_k.get(k, 0.0),
        },
        cold_user_metrics={
            f"ndcg@{k}": cold_m.ndcg_at_k.get(k, 0.0),
            f"recall@{k}": cold_m.recall_at_k.get(k, 0.0),
            f"precision@{k}": cold_m.precision_at_k.get(k, 0.0),
        },
    )
