"""Bounded Recommender Parameter Sensitivity Analysis.

Evaluates deterministic perturbations over a small predefined grid to assess
model stability under parameter variation without unconstrained search.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from start.recommender.contracts import (
    RecommenderSensitivityPoint,
    RecommenderSensitivityResult,
)
from start.recommender.metrics import compute_ranking_metrics, compute_rating_metrics
from start.recommender.models import MatrixFactorizationModel, NeuralCollaborativeFilteringModel


def run_mf_sensitivity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline_latent: int = 16,
    baseline_reg: float = 0.02,
    seed: int = 42,
) -> RecommenderSensitivityResult:
    """Run bounded sensitivity on Matrix Factorization across latent_dim and regularization."""
    # 1. Baseline
    base_model = MatrixFactorizationModel(latent_dim=baseline_latent, regularization=baseline_reg, epochs=10, seed=seed)
    base_model.fit(train_df)

    rating_col = "rating" if "rating" in test_df.columns else ("target" if "target" in test_df.columns else ("weight" if "weight" in test_df.columns else "label"))
    y_true = test_df[rating_col].to_list() if rating_col in test_df.columns else [1.0] * len(test_df)

    preds = [base_model.predict_one(row["user_id"], row["item_id"]) for _, row in test_df.iterrows()]
    base_m = compute_rating_metrics(y_true, preds)
    baseline_metrics = {"rmse": base_m.rmse, "mae": base_m.mae}

    points: list[RecommenderSensitivityPoint] = []

    # Perturbation grid: latent_dim in [8, 32], reg in [0.005, 0.08]
    grid = [
        ("latent_dim", 8),
        ("latent_dim", 32),
        ("regularization", 0.005),
        ("regularization", 0.08),
    ]

    for param_name, param_val in grid:
        l_dim = param_val if param_name == "latent_dim" else baseline_latent
        r_val = param_val if param_name == "regularization" else baseline_reg

        m = MatrixFactorizationModel(latent_dim=l_dim, regularization=r_val, epochs=10, seed=seed)
        m.fit(train_df)
        p_eval = [m.predict_one(row["user_id"], row["item_id"]) for _, row in test_df.iterrows()]
        met = compute_rating_metrics(y_true, p_eval)
        m_dict = {"rmse": met.rmse, "mae": met.mae}

        delta = {
            "delta_rmse": round(met.rmse - base_m.rmse, 4),
            "delta_mae": round(met.mae - base_m.mae, 4),
        }
        points.append(
            RecommenderSensitivityPoint(
                parameter_name=param_name,
                parameter_value=param_val,
                metrics=m_dict,
                delta_from_baseline=delta,
            )
        )

    return RecommenderSensitivityResult(
        algorithm="matrix_factorization",
        baseline_params={"latent_dim": baseline_latent, "regularization": baseline_reg, "seed": seed},
        baseline_metrics=baseline_metrics,
        tested_perturbations=points,
    )


def run_ncf_sensitivity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline_embed: int = 16,
    baseline_dropout: float = 0.1,
    seed: int = 42,
) -> RecommenderSensitivityResult:
    """Run bounded sensitivity on NCF across embedding_dim and dropout."""
    base_model = NeuralCollaborativeFilteringModel(
        embedding_dim=baseline_embed, dropout=baseline_dropout, epochs=10, seed=seed
    )
    base_model.fit(train_df)

    test_gt = {u: set(group["item_id"]) for u, group in test_df.groupby("user_id")}
    recs = {u: base_model.recommend(u, k=10) for u in test_gt}
    base_rank = compute_ranking_metrics(recs, test_gt, [10])
    baseline_metrics = {"ndcg@10": base_rank.ndcg_at_k.get(10, 0.0), "recall@10": base_rank.recall_at_k.get(10, 0.0)}

    points: list[RecommenderSensitivityPoint] = []
    grid = [
        ("embedding_dim", 8),
        ("embedding_dim", 24),
        ("dropout", 0.0),
        ("dropout", 0.3),
    ]

    for param_name, param_val in grid:
        e_dim = param_val if param_name == "embedding_dim" else baseline_embed
        d_val = param_val if param_name == "dropout" else baseline_dropout

        m = NeuralCollaborativeFilteringModel(embedding_dim=e_dim, dropout=d_val, epochs=10, seed=seed)
        m.fit(train_df)
        p_recs = {u: m.recommend(u, k=10) for u in test_gt}
        r_met = compute_ranking_metrics(p_recs, test_gt, [10])
        m_dict = {"ndcg@10": r_met.ndcg_at_k.get(10, 0.0), "recall@10": r_met.recall_at_k.get(10, 0.0)}

        delta = {
            "delta_ndcg": round(m_dict["ndcg@10"] - baseline_metrics["ndcg@10"], 4),
            "delta_recall": round(m_dict["recall@10"] - baseline_metrics["recall@10"], 4),
        }
        points.append(
            RecommenderSensitivityPoint(
                parameter_name=param_name,
                parameter_value=param_val,
                metrics=m_dict,
                delta_from_baseline=delta,
            )
        )

    return RecommenderSensitivityResult(
        algorithm="ncf",
        baseline_params={"embedding_dim": baseline_embed, "dropout": baseline_dropout, "seed": seed},
        baseline_metrics=baseline_metrics,
        tested_perturbations=points,
    )


def run_ffm_sensitivity(
    raw_data: list[Any] | None,
    split_res: Any,
    baseline_latent: int = 4,
    seed: int = 42,
) -> RecommenderSensitivityResult:
    """Run bounded sensitivity on Field-Aware Factorization Machine across latent_dim."""
    from start.recommender.fixtures import DATASET_C
    from start.recommender.models import FieldAwareFactorizationMachineModel

    data = raw_data if raw_data is not None else list(DATASET_C)
    base_model = FieldAwareFactorizationMachineModel(latent_dim=baseline_latent, epochs=10, seed=seed)
    base_model.fit(data)

    test_df = split_res.test_data
    all_items = sorted(test_df["item_id"].unique())
    ground_truth = {u: set(grp["item_id"].values) for u, grp in test_df.groupby("user_id")}
    recs = {u: base_model.recommend(u, k=10, candidate_items=all_items) for u in ground_truth}
    base_rank = compute_ranking_metrics(recs, ground_truth, [10])
    baseline_metrics = {"ndcg@10": base_rank.ndcg_at_k.get(10, 0.0), "recall@10": base_rank.recall_at_k.get(10, 0.0)}

    points: list[RecommenderSensitivityPoint] = []
    grid = [
        ("latent_dim", 2),
        ("latent_dim", 8),
    ]

    for param_name, param_val in grid:
        l_dim = param_val if param_name == "latent_dim" else baseline_latent
        m = FieldAwareFactorizationMachineModel(latent_dim=l_dim, epochs=10, seed=seed)
        m.fit(data)
        p_recs = {u: m.recommend(u, k=10, candidate_items=all_items) for u in ground_truth}
        r_met = compute_ranking_metrics(p_recs, ground_truth, [10])
        m_dict = {"ndcg@10": r_met.ndcg_at_k.get(10, 0.0), "recall@10": r_met.recall_at_k.get(10, 0.0)}
        delta = {
            "delta_ndcg": round(m_dict["ndcg@10"] - baseline_metrics["ndcg@10"], 4),
            "delta_recall": round(m_dict["recall@10"] - baseline_metrics["recall@10"], 4),
        }
        points.append(
            RecommenderSensitivityPoint(
                parameter_name=param_name,
                parameter_value=param_val,
                metrics=m_dict,
                delta_from_baseline=delta,
            )
        )

    return RecommenderSensitivityResult(
        algorithm="field_aware_factorization_machine",
        baseline_params={"latent_dim": baseline_latent, "seed": seed},
        baseline_metrics=baseline_metrics,
        tested_perturbations=points,
    )


def evaluate_recommender_sensitivity(
    model: Any,
    split_res: Any,
    algorithm_type: str = "mf",
    raw_data: list[Any] | None = None,
) -> RecommenderSensitivityResult:
    """Canonical dispatcher for recommender sensitivity analysis."""
    train_df = split_res.train_data
    test_df = split_res.test_data
    seed = getattr(model, "seed", 42)

    if algorithm_type in ("ffm", "field_aware_factorization_machine"):
        return run_ffm_sensitivity(raw_data, split_res, baseline_latent=getattr(model, "latent_dim", 4), seed=seed)
    elif algorithm_type == "ncf":
        return run_ncf_sensitivity(train_df, test_df, seed=seed)
    else:
        return run_mf_sensitivity(train_df, test_df, seed=seed)

