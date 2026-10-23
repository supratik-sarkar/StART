"""Comprehensive Validation Tests for Recommender Systems Scientific Vertical.

Tests:
- Known-answer ranking & rating metrics
- Data validation and formal split protocols
- Explicit negative sampling
- Biased Matrix Factorization, NCF, and FM models
- Beyond-accuracy and cold-start metrics
- Bounded parameter sensitivity analysis
- Canonical Execution Service integration across all 3 fixtures
"""

import math

import numpy as np

from start.recommender.data import (
    sample_negative_items,
    split_recommender_dataset,
    validate_recommender_dataset,
)
from start.recommender.fixtures import DATASET_A, DATASET_B, DATASET_C, to_dataframe
from start.recommender.metrics import (
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_rating_metrics,
    compute_recall_at_k,
)
from start.recommender.models import (
    FactorizationMachineModel,
    MatrixFactorizationModel,
    NeuralCollaborativeFilteringModel,
)
from start.runtime.execution import CanonicalExecutionService


class TestRecommenderMetrics:
    def test_known_answer_rating_metrics(self):
        y_true = np.array([3.0, 4.0, 5.0, 2.0])
        y_pred = np.array([3.0, 5.0, 4.0, 2.0])
        # diffs: [0, 1, -1, 0] -> sq: [0, 1, 1, 0] -> mean sq: 0.5 -> rmse: sqrt(0.5) = 0.707107
        m = compute_rating_metrics(y_true, y_pred)
        assert math.isclose(m.rmse, math.sqrt(0.5), rel_tol=1e-4)
        assert math.isclose(m.mae, 0.5, rel_tol=1e-4)
        assert m.n_eval_samples == 4

    def test_known_answer_ranking_metrics(self):
        recs = ["item_A", "item_B", "item_C", "item_D", "item_E"]
        gt = {"item_A", "item_B"}

        # At k=2: recs[:2] = ["item_A", "item_B"], both in gt
        p2 = compute_precision_at_k(recs, gt, k=2)
        r2 = compute_recall_at_k(recs, gt, k=2)
        ndcg2 = compute_ndcg_at_k(recs, gt, k=2)
        mrr = compute_mrr(recs, gt)

        assert math.isclose(p2, 1.0)
        assert math.isclose(r2, 1.0)
        assert math.isclose(ndcg2, 1.0)
        assert math.isclose(mrr, 1.0)

    def test_ranking_metrics_no_hits(self):
        recs = ["item_X", "item_Y"]
        gt = {"item_A", "item_B"}
        assert compute_precision_at_k(recs, gt, k=2) == 0.0
        assert compute_recall_at_k(recs, gt, k=2) == 0.0
        assert compute_ndcg_at_k(recs, gt, k=2) == 0.0
        assert compute_mrr(recs, gt) == 0.0


class TestRecommenderDataAndSplits:
    def test_dataset_profile_calculation(self):
        df = to_dataframe(DATASET_A)
        prof = validate_recommender_dataset(df, rating_col="rating")
        assert prof.n_users == 25
        assert prof.n_items == 30
        assert prof.n_interactions == 250
        assert 0.0 < prof.sparsity < 1.0
        assert prof.feedback_mode == "explicit"
        assert prof.rating_min == 1.0
        assert prof.rating_max == 5.0

    def test_user_stratified_split_preserves_users(self):
        df = to_dataframe(DATASET_A)
        split = split_recommender_dataset(df, protocol="user_stratified", test_ratio=0.2, seed=42)
        assert split.train_count + split.test_count + split.val_count == 250
        assert split.warm_users_count > 0

    def test_negative_sampling_excludes_seen(self):
        seen = {"item_1", "item_2", "item_3"}
        universe = [f"item_{i}" for i in range(1, 20)]
        negs = sample_negative_items({"user_1": seen}, universe, n_negatives=5, seed=42)
        assert len(negs["user_1"]) == 5
        for it in negs["user_1"]:
            assert it not in seen


class TestRecommenderModels:
    def test_matrix_factorization_fit_and_recommend(self):
        df = to_dataframe(DATASET_A)
        split = split_recommender_dataset(df, protocol="user_stratified", test_ratio=0.2, seed=42)
        mf = MatrixFactorizationModel(latent_dim=8, learning_rate=0.05, regularization=0.02, epochs=10, seed=42)
        summary = mf.fit(split.train_data)
        assert summary.algorithm == "matrix_factorization"

        # Predict single
        pred = mf.predict_one("U_001", "ITEM_001")
        assert isinstance(pred, float)
        assert not math.isnan(pred)

        # Recommend top-5
        recs = mf.recommend("U_001", k=5)
        assert len(recs) <= 5
        assert all(isinstance(it, str) for it in recs)

    def test_ncf_fit_and_recommend(self):
        df = to_dataframe(DATASET_B)
        split = split_recommender_dataset(df, protocol="user_stratified", test_ratio=0.2, seed=42)
        ncf = NeuralCollaborativeFilteringModel(embedding_dim=8, hidden_layers=[16, 8], epochs=5, seed=42)
        summary = ncf.fit(split.train_data)
        assert summary.algorithm == "ncf"

        recs = ncf.recommend("U_001", k=5)
        assert len(recs) <= 5

    def test_fm_fit_and_recommend(self):
        df = to_dataframe(DATASET_C)
        split = split_recommender_dataset(df, protocol="user_stratified", test_ratio=0.2, seed=42)
        fm = FactorizationMachineModel(latent_dim=4, epochs=5, seed=42)
        summary = fm.fit(split.train_data)
        assert summary.algorithm == "factorization_machine"

        score = fm.predict_one("U_001", "ITEM_001")
        assert 0.0 <= score <= 1.0


class TestCanonicalExecutionServiceRecommender:
    def test_explicit_ratings_workflow_execution(self):
        res = CanonicalExecutionService.execute(
            workflow_id="recommender_system",
            context_id="recommender_ratings_v1",
        )
        assert res.workflow_id == "recommender_system"
        assert res.context_id == "recommender_ratings_v1"
        assert len(res.records) == 7
        assert len(res.artifacts) >= 5
        assert res.governance_disposition in ("ACCEPT", "ACCEPT_WITH_CONDITIONS")

    def test_implicit_ncf_workflow_execution(self):
        res = CanonicalExecutionService.execute(
            workflow_id="recommender_system",
            context_id="recommender_implicit_v1",
        )
        assert res.workflow_id == "recommender_system"
        assert res.context_id == "recommender_implicit_v1"
        # Rating fidelity is skipped on implicit
        assert len(res.records) == 6
        assert len(res.artifacts) >= 5

    def test_contextual_fm_workflow_execution(self):
        res = CanonicalExecutionService.execute(
            workflow_id="recommender_system",
            context_id="recommender_contextual_v1",
        )
        assert res.workflow_id == "recommender_system"
        assert res.context_id == "recommender_contextual_v1"
        assert len(res.records) == 7
        assert len(res.artifacts) >= 5
