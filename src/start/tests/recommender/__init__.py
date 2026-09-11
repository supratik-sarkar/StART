"""Authoritative Recommender Systems Scientific Test Family.

Provides registered, deterministic validation tests:
- recommender.data_quality: interaction density, sparsity, user/item bounds
- recommender.rating_fidelity: RMSE, MAE, R² on held-out ratings
- recommender.ranking_ndcg: NDCG@10 and MRR ranking metrics
- recommender.ranking_recall: Recall@10, Precision@10, HitRate@10
- recommender.beyond_accuracy_coverage: catalog coverage, user coverage, novelty
- recommender.cold_start_robustness: cold vs warm cohort NDCG degradation
- recommender.sensitivity_stability: latent factor & regularization stability
"""

from __future__ import annotations

from typing import Any

from start.core.schemas import Status, TestResult, ThresholdSpec
from start.recommender.contracts import RecommenderContext
from start.registry import register_test


def _extract_exec_result(ctx: Any) -> Any:
    """Extract RecommenderExecutionResult from context if available."""
    if hasattr(ctx, "execution_result") and ctx.execution_result is not None:
        return ctx.execution_result
    if hasattr(ctx, "extra") and isinstance(ctx.extra, dict):
        return ctx.extra.get("execution_result")
    return None


@register_test(
    "recommender.data_quality",
    family="recommender",
    name="Recommender Data Quality & Sparsity",
    description="Validates interaction matrix sparsity, user/item catalog coverage, and data hygiene.",
    context_type="recommender",
    default_params={"sparsity_warn": 0.999, "min_interactions": 10},
)
def test_data_quality(
    ctx: RecommenderContext,
    sparsity_warn: float = 0.999,
    min_interactions: int = 10,
) -> TestResult:
    profile = getattr(ctx, "dataset_profile", None)
    if profile is None:
        return TestResult(
            test_id="recommender.data_quality",
            test_name="Recommender Data Quality & Sparsity",
            status=Status.SKIPPED,
            interpretation="Dataset profile not present in recommender context.",
        )

    sparsity = profile.sparsity
    n_interactions = profile.n_interactions
    n_users = profile.n_users
    n_items = profile.n_items

    status = Status.PASS
    interpretation = (
        f"Interaction matrix contains {n_interactions} interactions across {n_users} users and {n_items} items. "
        f"Sparsity is {sparsity:.4%} (density {1.0 - sparsity:.4%})."
    )

    if n_interactions < min_interactions:
        status = Status.FAIL
        interpretation = f"Critical: only {n_interactions} interactions, below minimum threshold {min_interactions}."
    elif sparsity > sparsity_warn:
        status = Status.WARN
        interpretation += f" Warning: extreme sparsity ({sparsity:.4%}) exceeds threshold ({sparsity_warn:.4%})."

    return TestResult(
        test_id="recommender.data_quality",
        test_name="Recommender Data Quality & Sparsity",
        status=status,
        params={"sparsity_warn": sparsity_warn, "min_interactions": min_interactions},
        metrics={
            "n_users": n_users,
            "n_items": n_items,
            "n_interactions": n_interactions,
            "sparsity": round(sparsity, 6),
            "density": round(1.0 - sparsity, 6),
            "mean_ratings_per_user": round(profile.mean_ratings_per_user, 2),
            "mean_ratings_per_item": round(profile.mean_ratings_per_item, 2),
        },
        thresholds=[ThresholdSpec(metric="sparsity", warn=sparsity_warn, direction="upper")],
        interpretation=interpretation,
    )


@register_test(
    "recommender.rating_fidelity",
    family="recommender",
    name="Rating Prediction Fidelity (RMSE / MAE)",
    description="Evaluates rating prediction error against predefined error bounds on held-out interactions.",
    context_type="recommender",
    default_params={"rmse_warn": 1.25, "rmse_fail": 2.0},
)
def test_rating_fidelity(
    ctx: RecommenderContext,
    rmse_warn: float = 1.25,
    rmse_fail: float = 2.0,
) -> TestResult:
    exec_res = _extract_exec_result(ctx)
    if exec_res is None or exec_res.rating_metrics is None:
        return TestResult(
            test_id="recommender.rating_fidelity",
            test_name="Rating Prediction Fidelity (RMSE / MAE)",
            status=Status.SKIPPED,
            interpretation="No rating prediction metrics available (task may be implicit or ranking only).",
        )

    rm = exec_res.rating_metrics
    rmse = rm.rmse
    mae = rm.mae
    r2 = rm.r2

    status = Status.PASS
    if rmse >= rmse_fail:
        status = Status.FAIL
    elif rmse >= rmse_warn:
        status = Status.WARN

    interpretation = (
        f"Rating prediction achieved RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f} "
        f"evaluated across {rm.n_test_samples} held-out interactions."
    )

    return TestResult(
        test_id="recommender.rating_fidelity",
        test_name="Rating Prediction Fidelity (RMSE / MAE)",
        status=status,
        params={"rmse_warn": rmse_warn, "rmse_fail": rmse_fail},
        metrics={
            "rmse": round(rmse, 6),
            "mae": round(mae, 6),
            "r2": round(r2, 6),
            "n_test_samples": rm.n_test_samples,
        },
        thresholds=[ThresholdSpec(metric="rmse", warn=rmse_warn, fail=rmse_fail, direction="upper")],
        interpretation=interpretation,
    )


@register_test(
    "recommender.ranking_ndcg",
    family="recommender",
    name="Top-K Ranking Quality (NDCG / MRR)",
    description="Measures position-discounted ranking quality (NDCG) and Mean Reciprocal Rank (MRR) on candidate rankings.",
    context_type="recommender",
    default_params={"ndcg10_warn": 0.25, "ndcg10_fail": 0.05},
)
def test_ranking_ndcg(
    ctx: RecommenderContext,
    ndcg10_warn: float = 0.25,
    ndcg10_fail: float = 0.05,
) -> TestResult:
    exec_res = _extract_exec_result(ctx)
    if exec_res is None or exec_res.ranking_metrics is None:
        return TestResult(
            test_id="recommender.ranking_ndcg",
            test_name="Top-K Ranking Quality (NDCG / MRR)",
            status=Status.SKIPPED,
            interpretation="No ranking metrics available in execution result.",
        )

    rm = exec_res.ranking_metrics
    ndcg10 = rm.ndcg_at_k.get(10, rm.ndcg_at_k.get(5, 0.0))
    mrr = rm.mrr

    status = Status.PASS
    if ndcg10 < ndcg10_fail:
        status = Status.FAIL
    elif ndcg10 < ndcg10_warn:
        status = Status.WARN

    interpretation = (
        f"Ranking performance: NDCG@10={ndcg10:.4f}, MRR={mrr:.4f}, MAP@10={rm.map_at_k.get(10, 0.0):.4f} "
        f"across {rm.n_users_evaluated} evaluated users."
    )

    metrics: dict[str, Any] = {
        "ndcg_10": round(ndcg10, 6),
        "mrr": round(mrr, 6),
        "n_users_evaluated": rm.n_users_evaluated,
    }
    for k, val in rm.ndcg_at_k.items():
        metrics[f"ndcg_{k}"] = round(val, 6)

    return TestResult(
        test_id="recommender.ranking_ndcg",
        test_name="Top-K Ranking Quality (NDCG / MRR)",
        status=status,
        params={"ndcg10_warn": ndcg10_warn, "ndcg10_fail": ndcg10_fail},
        metrics=metrics,
        thresholds=[ThresholdSpec(metric="ndcg_10", warn=ndcg10_warn, fail=ndcg10_fail, direction="lower")],
        interpretation=interpretation,
    )


@register_test(
    "recommender.ranking_recall",
    family="recommender",
    name="Top-K Ranking Coverage (Recall@K / Precision@K)",
    description="Evaluates Recall@K, Precision@K, and Hit Rate@K across recommendation cutoffs.",
    context_type="recommender",
    default_params={"recall10_warn": 0.20, "recall10_fail": 0.02},
)
def test_ranking_recall(
    ctx: RecommenderContext,
    recall10_warn: float = 0.20,
    recall10_fail: float = 0.02,
) -> TestResult:
    exec_res = _extract_exec_result(ctx)
    if exec_res is None or exec_res.ranking_metrics is None:
        return TestResult(
            test_id="recommender.ranking_recall",
            test_name="Top-K Ranking Coverage (Recall@K / Precision@K)",
            status=Status.SKIPPED,
            interpretation="No ranking metrics available in execution result.",
        )

    rm = exec_res.ranking_metrics
    rec10 = rm.recall_at_k.get(10, rm.recall_at_k.get(5, 0.0))
    prec10 = rm.precision_at_k.get(10, rm.precision_at_k.get(5, 0.0))
    hr10 = rm.hit_rate_at_k.get(10, rm.hit_rate_at_k.get(5, 0.0))

    status = Status.PASS
    if rec10 < recall10_fail:
        status = Status.FAIL
    elif rec10 < recall10_warn:
        status = Status.WARN

    interpretation = (
        f"Retrieval effectiveness: Recall@10={rec10:.4f}, Precision@10={prec10:.4f}, HitRate@10={hr10:.4f}."
    )

    metrics: dict[str, Any] = {
        "recall_10": round(rec10, 6),
        "precision_10": round(prec10, 6),
        "hit_rate_10": round(hr10, 6),
        "n_users_evaluated": rm.n_users_evaluated,
    }
    for k, val in rm.recall_at_k.items():
        metrics[f"recall_{k}"] = round(val, 6)
        metrics[f"precision_{k}"] = round(rm.precision_at_k.get(k, 0.0), 6)
        metrics[f"hit_rate_{k}"] = round(rm.hit_rate_at_k.get(k, 0.0), 6)

    return TestResult(
        test_id="recommender.ranking_recall",
        test_name="Top-K Ranking Coverage (Recall@K / Precision@K)",
        status=status,
        params={"recall10_warn": recall10_warn, "recall10_fail": recall10_fail},
        metrics=metrics,
        thresholds=[ThresholdSpec(metric="recall_10", warn=recall10_warn, fail=recall10_fail, direction="lower")],
        interpretation=interpretation,
    )


@register_test(
    "recommender.beyond_accuracy_coverage",
    family="recommender",
    name="Beyond-Accuracy (Catalog & User Coverage / Novelty)",
    description="Monitors recommendation diversity, catalog exploration, user coverage, and popularity concentration.",
    context_type="recommender",
    default_params={"catalog_coverage_warn": 0.20, "catalog_coverage_fail": 0.05},
)
def test_beyond_accuracy(
    ctx: RecommenderContext,
    catalog_coverage_warn: float = 0.20,
    catalog_coverage_fail: float = 0.05,
) -> TestResult:
    exec_res = _extract_exec_result(ctx)
    if exec_res is None or exec_res.beyond_accuracy_metrics is None:
        return TestResult(
            test_id="recommender.beyond_accuracy_coverage",
            test_name="Beyond-Accuracy (Catalog & User Coverage / Novelty)",
            status=Status.SKIPPED,
            interpretation="Beyond-accuracy metrics not computed for this run.",
        )

    ba = exec_res.beyond_accuracy_metrics
    cat_cov = ba.catalog_coverage
    user_cov = ba.user_coverage
    novelty = ba.novelty
    pop_bias = ba.popularity_bias_ratio

    status = Status.PASS
    if cat_cov < catalog_coverage_fail:
        status = Status.FAIL
    elif cat_cov < catalog_coverage_warn:
        status = Status.WARN

    interpretation = (
        f"Beyond-accuracy metrics: Catalog Coverage={cat_cov:.2%}, User Coverage={user_cov:.2%}, "
        f"Novelty Score={novelty:.4f}, Popularity Concentration={pop_bias:.2%}."
    )

    return TestResult(
        test_id="recommender.beyond_accuracy_coverage",
        test_name="Beyond-Accuracy (Catalog & User Coverage / Novelty)",
        status=status,
        params={"catalog_coverage_warn": catalog_coverage_warn, "catalog_coverage_fail": catalog_coverage_fail},
        metrics={
            "catalog_coverage": round(cat_cov, 6),
            "user_coverage": round(user_cov, 6),
            "novelty": round(novelty, 6),
            "popularity_bias_ratio": round(pop_bias, 6),
            "gini_index": round(ba.gini_index, 6),
            "entropy": round(ba.entropy, 6),
        },
        thresholds=[
            ThresholdSpec(
                metric="catalog_coverage",
                warn=catalog_coverage_warn,
                fail=catalog_coverage_fail,
                direction="lower",
            )
        ],
        interpretation=interpretation,
    )


@register_test(
    "recommender.cold_start_robustness",
    family="recommender",
    name="Cold-Start User & Item Robustness",
    description="Tests recommendation degradation on cold-start users against established warm users.",
    context_type="recommender",
    default_params={"max_degradation_warn": 0.80, "max_degradation_fail": 0.95},
)
def test_cold_start(
    ctx: RecommenderContext,
    max_degradation_warn: float = 0.80,
    max_degradation_fail: float = 0.95,
) -> TestResult:
    exec_res = _extract_exec_result(ctx)
    if exec_res is None or exec_res.cold_start_metrics is None:
        return TestResult(
            test_id="recommender.cold_start_robustness",
            test_name="Cold-Start User & Item Robustness",
            status=Status.SKIPPED,
            interpretation="Cold-start cohort evaluation not available.",
        )

    cs = exec_res.cold_start_metrics
    deg = cs.ndcg_degradation_ratio

    status = Status.PASS
    if deg >= max_degradation_fail:
        status = Status.FAIL
    elif deg >= max_degradation_warn:
        status = Status.WARN

    interpretation = (
        f"Cold-start evaluation: {cs.n_cold_users} cold users (NDCG@10={cs.cold_user_ndcg:.4f}) "
        f"vs {cs.n_warm_users} warm users (NDCG@10={cs.warm_user_ndcg:.4f}), "
        f"representing a {deg:.2%} relative degradation."
    )

    return TestResult(
        test_id="recommender.cold_start_robustness",
        test_name="Cold-Start User & Item Robustness",
        status=status,
        params={"max_degradation_warn": max_degradation_warn, "max_degradation_fail": max_degradation_fail},
        metrics={
            "warm_user_ndcg": round(cs.warm_user_ndcg, 6),
            "cold_user_ndcg": round(cs.cold_user_ndcg, 6),
            "ndcg_degradation_ratio": round(deg, 6),
            "n_warm_users": cs.n_warm_users,
            "n_cold_users": cs.n_cold_users,
        },
        thresholds=[
            ThresholdSpec(
                metric="ndcg_degradation_ratio",
                warn=max_degradation_warn,
                fail=max_degradation_fail,
                direction="upper",
            )
        ],
        interpretation=interpretation,
    )


@register_test(
    "recommender.sensitivity_stability",
    family="recommender",
    name="Hyperparameter Sensitivity & Latent Dimension Stability",
    description="Validates ranking stability and metric variance across bounded parameter perturbations.",
    context_type="recommender",
    default_params={"max_delta_warn": 0.25, "max_delta_fail": 0.50},
)
def test_sensitivity_stability(
    ctx: RecommenderContext,
    max_delta_warn: float = 0.25,
    max_delta_fail: float = 0.50,
) -> TestResult:
    exec_res = _extract_exec_result(ctx)
    if exec_res is None or exec_res.sensitivity_result is None:
        return TestResult(
            test_id="recommender.sensitivity_stability",
            test_name="Hyperparameter Sensitivity & Latent Dimension Stability",
            status=Status.SKIPPED,
            interpretation="Sensitivity analysis not executed.",
        )

    sr = exec_res.sensitivity_result
    perturbations = sr.tested_perturbations
    max_rel_change = 0.0
    for p in perturbations:
        if p.relative_change is not None and abs(p.relative_change) > max_rel_change:
            max_rel_change = abs(p.relative_change)

    status = Status.PASS
    if max_rel_change >= max_delta_fail:
        status = Status.FAIL
    elif max_rel_change >= max_delta_warn:
        status = Status.WARN

    interpretation = (
        f"Sensitivity sweep on algorithm '{sr.algorithm}' tested {len(perturbations)} perturbations. "
        f"Maximum metric relative shift observed was {max_rel_change:.2%}."
    )

    return TestResult(
        test_id="recommender.sensitivity_stability",
        test_name="Hyperparameter Sensitivity & Latent Dimension Stability",
        status=status,
        params={"max_delta_warn": max_delta_warn, "max_delta_fail": max_delta_fail},
        metrics={
            "n_perturbations_tested": len(perturbations),
            "max_relative_change": round(max_rel_change, 6),
            "algorithm": sr.algorithm,
        },
        thresholds=[
            ThresholdSpec(
                metric="max_relative_change",
                warn=max_delta_warn,
                fail=max_delta_fail,
                direction="upper",
            )
        ],
        interpretation=interpretation,
    )
