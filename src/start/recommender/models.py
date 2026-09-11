"""Authoritative Recommender Models: Matrix Factorization, NCF, and Factorization Machine.

All models:
- Are deterministic under explicit seed.
- Support bounded training budgets without infinite loops or hanging threads.
- Support both rating prediction and Top-K ranking recommendation generation.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from start.recommender.contracts import (
    ContextualFeedbackData,
    RecommenderModelSummary,
    RecommenderTaskMode,
)


class MatrixFactorizationModel:
    """Regularized Biased Matrix Factorization (Koren et al., 2009).

    Formula: r_hat_{u,i} = mu + b_u + b_i + P_u . Q_i
    """

    def __init__(
        self,
        latent_dim: int = 16,
        learning_rate: float = 0.05,
        regularization: float = 0.02,
        epochs: int = 20,
        seed: int = 42,
        task_mode: RecommenderTaskMode = RecommenderTaskMode.RATING_PREDICTION,
    ):
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.seed = seed
        self.task_mode = task_mode

        self.mu: float = 0.0
        self.user_to_idx: dict[str, int] = {}
        self.item_to_idx: dict[str, int] = {}
        self.idx_to_user: list[str] = []
        self.idx_to_item: list[str] = []

        self.b_u: np.ndarray = np.array([])
        self.b_i: np.ndarray = np.array([])
        self.P: np.ndarray = np.array([])  # n_users x latent_dim
        self.Q: np.ndarray = np.array([])  # n_items x latent_dim

        self.user_seen_items: dict[str, set[str]] = {}
        self.training_history: list[dict[str, float]] = []

    def fit(self, df: pd.DataFrame, user_col: str = "user_id", item_col: str = "item_id", rating_col: str = "rating") -> RecommenderModelSummary:
        start_time = time.time()
        rng = np.random.default_rng(self.seed)

        users = sorted(df[user_col].unique())
        items = sorted(df[item_col].unique())

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {it: i for i, it in enumerate(items)}
        self.idx_to_user = users
        self.idx_to_item = items

        n_users = len(users)
        n_items = len(items)

        # Track seen items
        self.user_seen_items = {u: set() for u in users}
        for _, row in df.iterrows():
            self.user_seen_items[str(row[user_col])].add(str(row[item_col]))

        # Initialize parameters
        ratings = df[rating_col].to_numpy(dtype=float) if rating_col in df.columns else np.ones(len(df))
        self.mu = float(np.mean(ratings)) if len(ratings) > 0 else 0.0

        self.b_u = np.zeros(n_users, dtype=float)
        self.b_i = np.zeros(n_items, dtype=float)

        scale = 1.0 / math.sqrt(self.latent_dim)
        self.P = rng.normal(0, scale, size=(n_users, self.latent_dim))
        self.Q = rng.normal(0, scale, size=(n_items, self.latent_dim))

        u_indices = df[user_col].map(self.user_to_idx).to_numpy()
        i_indices = df[item_col].map(self.item_to_idx).to_numpy()

        # SGD optimization
        self.training_history = []
        for ep in range(self.epochs):
            perm = rng.permutation(len(df))
            total_loss = 0.0

            for idx in perm:
                u = u_indices[idx]
                i = i_indices[idx]
                r = ratings[idx]

                pred = self.mu + self.b_u[u] + self.b_i[i] + float(np.dot(self.P[u], self.Q[i]))
                err = r - pred
                total_loss += err ** 2

                # Gradient updates with L2 regularization
                self.b_u[u] += self.learning_rate * (err - self.regularization * self.b_u[u])
                self.b_i[i] += self.learning_rate * (err - self.regularization * self.b_i[i])

                p_u_old = self.P[u].copy()
                self.P[u] += self.learning_rate * (err * self.Q[i] - self.regularization * self.P[u])
                self.Q[i] += self.learning_rate * (err * p_u_old - self.regularization * self.Q[i])

            rmse_ep = math.sqrt(total_loss / max(1, len(df)))
            self.training_history.append({"epoch": ep + 1, "loss": round(total_loss, 4), "rmse": round(rmse_ep, 4)})

        latency = round((time.time() - start_time) * 1000, 2)
        return RecommenderModelSummary(
            algorithm="matrix_factorization",
            task_mode=str(self.task_mode),
            hyperparameters={
                "latent_dim": self.latent_dim,
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "epochs": self.epochs,
                "seed": self.seed,
            },
            training_summary={
                "final_loss": self.training_history[-1]["loss"] if self.training_history else 0.0,
                "final_rmse": self.training_history[-1]["rmse"] if self.training_history else 0.0,
                "epochs_trained": len(self.training_history),
                "n_users": n_users,
                "n_items": n_items,
            },
            latency_ms=latency,
            device="cpu",
            seed=self.seed,
        )

    def predict_one(self, user_id: str, item_id: str) -> float:
        """Predict rating for a single (user, item) pair."""
        u = self.user_to_idx.get(user_id)
        i = self.item_to_idx.get(item_id)

        # Baseline fallback if user or item unseen (cold-start)
        if u is None and i is None:
            return self.mu
        if u is None:
            return float(self.mu + self.b_i[i])
        if i is None:
            return float(self.mu + self.b_u[u])

        pred = self.mu + self.b_u[u] + self.b_i[i] + float(np.dot(self.P[u], self.Q[i]))
        return float(pred)

    def predict_batch(self, user_ids: list[str], item_ids: list[str]) -> np.ndarray:
        return np.array([self.predict_one(u, i) for u, i in zip(user_ids, item_ids)])

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidate_items: list[str] | None = None,
        exclude_seen: bool = True,
    ) -> list[str]:
        """Generate Top-K item recommendations for a user."""
        candidates = candidate_items if candidate_items is not None else self.idx_to_item
        seen = self.user_seen_items.get(user_id, set()) if exclude_seen else set()

        scored: list[tuple[str, float]] = []
        for it in candidates:
            if it not in seen:
                score = self.predict_one(user_id, it)
                scored.append((it, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored[:k]]


# --------------------------------------------------------------------------- #
# Neural Collaborative Filtering (NCF) Engine
# --------------------------------------------------------------------------- #
class _NCFNetwork(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 16, hidden_layers: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        layers = hidden_layers or [32, 16]
        self.user_embed = nn.Embedding(n_users, embedding_dim)
        self.item_embed = nn.Embedding(n_items, embedding_dim)

        mlp_input_dim = embedding_dim * 2
        net_layers: list[nn.Module] = []
        in_dim = mlp_input_dim
        for h_dim in layers:
            net_layers.append(nn.Linear(in_dim, h_dim))
            net_layers.append(nn.ReLU())
            if dropout > 0.0:
                net_layers.append(nn.Dropout(p=dropout))
            in_dim = h_dim

        net_layers.append(nn.Linear(in_dim, 1))
        net_layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*net_layers)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        u_emb = self.user_embed(user_idx)
        i_emb = self.item_embed(item_idx)
        x = torch.cat([u_emb, i_emb], dim=-1)
        return self.mlp(x).squeeze(-1)


class NeuralCollaborativeFilteringModel:
    """Neural Collaborative Filtering (He et al., WWW 2017).

    Features user & item embeddings mapped through an MLP interaction network.
    """

    def __init__(
        self,
        embedding_dim: int = 16,
        hidden_layers: list[int] | None = None,
        dropout: float = 0.1,
        learning_rate: float = 0.01,
        epochs: int = 15,
        batch_size: int = 32,
        negative_ratio: int = 3,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers or [32, 16]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.seed = seed

        self.net: _NCFNetwork | None = None
        self.user_to_idx: dict[str, int] = {}
        self.item_to_idx: dict[str, int] = {}
        self.idx_to_user: list[str] = []
        self.idx_to_item: list[str] = []
        self.user_seen_items: dict[str, set[str]] = {}
        self.training_history: list[dict[str, float]] = []

    def fit(self, df: pd.DataFrame, user_col: str = "user_id", item_col: str = "item_id") -> RecommenderModelSummary:
        start_time = time.time()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        users = sorted(df[user_col].unique())
        items = sorted(df[item_col].unique())

        self.user_to_idx = {u: i for i, u in enumerate(users)}
        self.item_to_idx = {it: i for i, it in enumerate(items)}
        self.idx_to_user = users
        self.idx_to_item = items

        self.user_seen_items = {u: set() for u in users}
        positives: list[tuple[int, int]] = []
        for _, row in df.iterrows():
            u = str(row[user_col])
            it = str(row[item_col])
            self.user_seen_items[u].add(it)
            positives.append((self.user_to_idx[u], self.item_to_idx[it]))

        # Synthesize negative pairs for ranking loss
        rng = np.random.default_rng(self.seed)
        n_items = len(items)
        negatives: list[tuple[int, int]] = []

        for u_idx, i_idx in positives:
            u_str = self.idx_to_user[u_idx]
            seen_indices = {self.item_to_idx[it] for it in self.user_seen_items[u_str]}
            for _ in range(self.negative_ratio):
                neg_i = rng.integers(0, n_items)
                if neg_i not in seen_indices:
                    negatives.append((u_idx, neg_i))

        # Dataset tensors
        all_u = [p[0] for p in positives] + [n[0] for n in negatives]
        all_i = [p[1] for p in positives] + [n[1] for n in negatives]
        all_y = [1.0] * len(positives) + [0.0] * len(negatives)

        u_tensor = torch.tensor(all_u, dtype=torch.long)
        i_tensor = torch.tensor(all_i, dtype=torch.long)
        y_tensor = torch.tensor(all_y, dtype=torch.float32)

        self.net = _NCFNetwork(
            n_users=len(users),
            n_items=len(items),
            embedding_dim=self.embedding_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
        )

        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        dataset_size = len(all_u)
        self.training_history = []

        self.net.train()
        for ep in range(self.epochs):
            perm = torch.randperm(dataset_size)
            total_loss = 0.0

            for batch_start in range(0, dataset_size, self.batch_size):
                batch_idx = perm[batch_start : batch_start + self.batch_size]
                b_u = u_tensor[batch_idx]
                b_i = i_tensor[batch_idx]
                b_y = y_tensor[batch_idx]

                optimizer.zero_grad()
                preds = self.net(b_u, b_i)
                loss = criterion(preds, b_y)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.item()) * len(batch_idx)

            avg_loss = total_loss / dataset_size
            self.training_history.append({"epoch": ep + 1, "bce_loss": round(avg_loss, 4)})

        self.net.eval()
        param_count = sum(p.numel() for p in self.net.parameters())
        latency = round((time.time() - start_time) * 1000, 2)

        return RecommenderModelSummary(
            algorithm="ncf",
            task_mode="top_k_ranking",
            hyperparameters={
                "embedding_dim": self.embedding_dim,
                "hidden_layers": self.hidden_layers,
                "dropout": self.dropout,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "negative_ratio": self.negative_ratio,
                "seed": self.seed,
            },
            training_summary={
                "final_bce_loss": self.training_history[-1]["bce_loss"] if self.training_history else 0.0,
                "epochs_trained": len(self.training_history),
                "parameter_count": param_count,
                "n_users": len(users),
                "n_items": len(items),
            },
            latency_ms=latency,
            device="cpu",
            seed=self.seed,
        )

    def predict_score(self, user_id: str, item_id: str) -> float:
        """Predict recommendation interaction probability score."""
        if self.net is None:
            return 0.5
        u = self.user_to_idx.get(user_id)
        i = self.item_to_idx.get(item_id)
        if u is None or i is None:
            return 0.5

        with torch.no_grad():
            u_t = torch.tensor([u], dtype=torch.long)
            i_t = torch.tensor([i], dtype=torch.long)
            score = float(self.net(u_t, i_t).item())
        return score

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidate_items: list[str] | None = None,
        exclude_seen: bool = True,
    ) -> list[str]:
        candidates = candidate_items if candidate_items is not None else self.idx_to_item
        seen = self.user_seen_items.get(user_id, set()) if exclude_seen else set()

        u = self.user_to_idx.get(user_id)
        if u is None:
            # Cold user fallback
            return [it for it in candidates if it not in seen][:k]

        valid_cands = [it for it in candidates if it not in seen]
        if not valid_cands:
            return []

        cand_indices = [self.item_to_idx[it] for it in valid_cands if it in self.item_to_idx]
        if not cand_indices:
            return valid_cands[:k]

        with torch.no_grad():
            u_batch = torch.tensor([u] * len(cand_indices), dtype=torch.long)
            i_batch = torch.tensor(cand_indices, dtype=torch.long)
            scores = self.net(u_batch, i_batch).numpy()

        ranked_indices = np.argsort(-scores)
        return [self.idx_to_item[cand_indices[idx]] for idx in ranked_indices[:k]]


# --------------------------------------------------------------------------- #
# Factorization Machine (FM) Engine
# --------------------------------------------------------------------------- #
class FactorizationMachineModel:
    """2-Way Factorization Machine (Rendle, 2010).

    Supports sparse interaction data with user, item, and side/context features.
    Computes pairwise interactions in O(k * p) linear time.
    """

    def __init__(
        self,
        latent_dim: int = 8,
        learning_rate: float = 0.02,
        regularization: float = 0.01,
        epochs: int = 15,
        seed: int = 42,
    ):
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.seed = seed

        self.feature_map: dict[str, int] = {}
        self.w0: float = 0.0
        self.w: np.ndarray = np.array([])  # 1-way weights
        self.V: np.ndarray = np.array([])  # 2-way factor matrix: n_features x latent_dim
        self.user_seen_items: dict[str, set[str]] = {}
        self.all_items: list[str] = []

    def _extract_sparse_features(self, rec: ContextualFeedbackData) -> list[int]:
        """Map user, item, and contextual key-values to feature index vector."""
        tokens: list[str] = [f"user={rec.user_id}", f"item={rec.item_id}"]
        for k, v in rec.user_features.items():
            tokens.append(f"uf_{k}={v}")
        for k, v in rec.item_features.items():
            tokens.append(f"if_{k}={v}")
        for k, v in rec.context_features.items():
            tokens.append(f"ctx_{k}={v}")

        return [self.feature_map[t] for t in tokens if t in self.feature_map]

    def fit(self, records: list[ContextualFeedbackData] | pd.DataFrame) -> RecommenderModelSummary:
        start_time = time.time()
        rng = np.random.default_rng(self.seed)

        if isinstance(records, pd.DataFrame):
            rec_list = []
            for _, row in records.iterrows():
                u_feat = row.get("user_features") if isinstance(row.get("user_features"), dict) else {}
                i_feat = row.get("item_features") if isinstance(row.get("item_features"), dict) else {}
                c_feat = row.get("context_features") if isinstance(row.get("context_features"), dict) else {}
                tgt = float(row.get("target", row.get("label", row.get("rating", row.get("weight", 1.0)))))
                rec_list.append(
                    ContextualFeedbackData(
                        user_id=str(row["user_id"]),
                        item_id=str(row["item_id"]),
                        target=tgt,
                        user_features=u_feat,
                        item_features=i_feat,
                        context_features=c_feat,
                    )
                )
            records = rec_list

        # Build vocabulary
        vocab: set[str] = set()
        self.all_items = sorted(list({r.item_id for r in records}))
        self.user_seen_items = {}

        for r in records:
            vocab.add(f"user={r.user_id}")
            vocab.add(f"item={r.item_id}")
            self.user_seen_items.setdefault(r.user_id, set()).add(r.item_id)
            for k, v in r.user_features.items():
                vocab.add(f"uf_{k}={v}")
            for k, v in r.item_features.items():
                vocab.add(f"if_{k}={v}")
            for k, v in r.context_features.items():
                vocab.add(f"ctx_{k}={v}")

        self.feature_map = {t: i for i, t in enumerate(sorted(vocab))}
        p = len(self.feature_map)

        self.w0 = 0.0
        self.w = np.zeros(p, dtype=float)
        scale = 1.0 / math.sqrt(self.latent_dim)
        self.V = rng.normal(0, scale, size=(p, self.latent_dim))

        # Prepare sparse indices & targets
        indices_list = [self._extract_sparse_features(r) for r in records]
        y_list = np.array([r.target for r in records], dtype=float)

        # SGD training
        for _ in range(self.epochs):
            perm = rng.permutation(len(records))
            for idx in perm:
                feats = indices_list[idx]
                y = y_list[idx]
                if not feats:
                    continue

                # 1-way sum
                linear_sum = float(np.sum(self.w[feats]))

                # 2-way sum
                v_subset = self.V[feats]
                sum_v = np.sum(v_subset, axis=0)
                sum_sq_v = np.sum(v_subset ** 2, axis=0)
                inter_sum = 0.5 * float(np.sum(sum_v ** 2 - sum_sq_v))

                raw_pred = self.w0 + linear_sum + inter_sum
                p_hat = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, raw_pred))))
                err = p_hat - y

                # Updates
                self.w0 -= self.learning_rate * err
                self.w[feats] -= self.learning_rate * (err + self.regularization * self.w[feats])

                # Factor gradient
                grad_v = err * (sum_v - v_subset) + self.regularization * v_subset
                self.V[feats] -= self.learning_rate * grad_v

        latency = round((time.time() - start_time) * 1000, 2)
        return RecommenderModelSummary(
            algorithm="factorization_machine",
            task_mode="binary_interaction",
            hyperparameters={
                "latent_dim": self.latent_dim,
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "epochs": self.epochs,
                "seed": self.seed,
            },
            training_summary={
                "n_features": p,
                "n_records": len(records),
            },
            latency_ms=latency,
            device="cpu",
            seed=self.seed,
        )

    def predict_score(self, user_id: str, item_id: str, user_features: dict | None = None, item_features: dict | None = None, context_features: dict | None = None) -> float:
        dummy_rec = ContextualFeedbackData(
            user_id=user_id,
            item_id=item_id,
            target=0.0,
            user_features=user_features or {},
            item_features=item_features or {},
            context_features=context_features or {},
        )
        feats = self._extract_sparse_features(dummy_rec)
        if not feats:
            return 0.5

        linear_sum = float(np.sum(self.w[feats]))
        v_subset = self.V[feats]
        sum_v = np.sum(v_subset, axis=0)
        sum_sq_v = np.sum(v_subset ** 2, axis=0)
        inter_sum = 0.5 * float(np.sum(sum_v ** 2 - sum_sq_v))

        pred = self.w0 + linear_sum + inter_sum
        return 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, pred))))

    def predict_one(self, user_id: str, item_id: str, **kwargs: Any) -> float:
        """Alias for predict_score for common interface compatibility."""
        return self.predict_score(user_id, item_id, **kwargs)

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidate_items: list[str] | None = None,
        user_features: dict | None = None,
        context_features: dict | None = None,
        exclude_seen: bool = True,
    ) -> list[str]:
        candidates = candidate_items if candidate_items is not None else self.all_items
        seen = self.user_seen_items.get(user_id, set()) if exclude_seen else set()

        scored: list[tuple[str, float]] = []
        for it in candidates:
            if it not in seen:
                s = self.predict_score(user_id, it, user_features=user_features, context_features=context_features)
                scored.append((it, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored[:k]]


class FieldAwareFactorizationMachineModel:
    """Field-Aware Factorization Machine (FFM; Juan et al., 2016).

    Formula:
        y_hat(x) = w0 + sum_{i} w_i * x_i + sum_{i} sum_{j > i} <V_{i, f(j)}, V_{j, f(i)}> * x_i * x_j

    Each feature belongs to a discrete field (user, item, context feature fields).
    For every feature pair (i, j) with fields (f_i, f_j), the interaction uses latent vectors
    V[i, f_j] and V[j, f_i].
    Distinct from standard Factorization Machine (FM) which uses a single latent vector per feature.
    """

    def __init__(
        self,
        latent_dim: int = 4,
        learning_rate: float = 0.02,
        regularization: float = 0.01,
        epochs: int = 10,
        seed: int = 42,
    ):
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.epochs = epochs
        self.seed = seed

        self.feature_map: dict[str, int] = {}
        self.field_map: dict[str, int] = {}
        self.feature_to_field: np.ndarray = np.array([], dtype=int)
        self.w0: float = 0.0
        self.w: np.ndarray = np.array([])
        self.V: np.ndarray = np.array([])  # shape: (n_features, n_fields, latent_dim)
        self.user_seen_items: dict[str, set[str]] = {}
        self.all_items: list[str] = []

    def _extract_sparse_features_with_fields(self, rec: ContextualFeedbackData) -> list[tuple[int, int]]:
        """Map record to list of (feature_index, field_index) tuples."""
        pairs: list[tuple[str, str]] = [
            (f"user={rec.user_id}", "user"),
            (f"item={rec.item_id}", "item"),
        ]
        for k, v in rec.user_features.items():
            pairs.append((f"uf_{k}={v}", f"uf_{k}"))
        for k, v in rec.item_features.items():
            pairs.append((f"if_{k}={v}", f"if_{k}"))
        for k, v in rec.context_features.items():
            pairs.append((f"ctx_{k}={v}", f"ctx_{k}"))

        result: list[tuple[int, int]] = []
        for feat_token, field_token in pairs:
            if feat_token in self.feature_map and field_token in self.field_map:
                result.append((self.feature_map[feat_token], self.field_map[field_token]))
        return result

    def fit(self, records: list[ContextualFeedbackData] | pd.DataFrame) -> RecommenderModelSummary:
        start_time = time.time()
        rng = np.random.default_rng(self.seed)

        if isinstance(records, pd.DataFrame):
            rec_list = []
            for _, row in records.iterrows():
                u_feat = row.get("user_features") if isinstance(row.get("user_features"), dict) else {}
                i_feat = row.get("item_features") if isinstance(row.get("item_features"), dict) else {}
                c_feat = row.get("context_features") if isinstance(row.get("context_features"), dict) else {}
                tgt = float(row.get("target", row.get("label", row.get("rating", row.get("weight", 1.0)))))
                rec_list.append(
                    ContextualFeedbackData(
                        user_id=str(row["user_id"]),
                        item_id=str(row["item_id"]),
                        target=tgt,
                        user_features=u_feat,
                        item_features=i_feat,
                        context_features=c_feat,
                    )
                )
            records = rec_list

        vocab: set[str] = set()
        fields: set[str] = {"user", "item"}
        self.all_items = sorted(list({r.item_id for r in records}))
        self.user_seen_items = {}

        for r in records:
            vocab.add(f"user={r.user_id}")
            vocab.add(f"item={r.item_id}")
            self.user_seen_items.setdefault(r.user_id, set()).add(r.item_id)
            for k, v in r.user_features.items():
                vocab.add(f"uf_{k}={v}")
                fields.add(f"uf_{k}")
            for k, v in r.item_features.items():
                vocab.add(f"if_{k}={v}")
                fields.add(f"if_{k}")
            for k, v in r.context_features.items():
                vocab.add(f"ctx_{k}={v}")
                fields.add(f"ctx_{k}")

        sorted_vocab = sorted(vocab)
        sorted_fields = sorted(fields)
        self.feature_map = {t: i for i, t in enumerate(sorted_vocab)}
        self.field_map = {f: i for i, f in enumerate(sorted_fields)}

        p = len(self.feature_map)
        num_fields = len(self.field_map)

        self.w0 = 0.0
        self.w = np.zeros(p, dtype=float)
        scale = 1.0 / math.sqrt(max(1, self.latent_dim))
        self.V = rng.normal(0, scale, size=(p, num_fields, self.latent_dim))

        indices_list = [self._extract_sparse_features_with_fields(r) for r in records]
        y_list = np.array([r.target for r in records], dtype=float)

        # SGD training loop
        for _ in range(self.epochs):
            perm = rng.permutation(len(records))
            for idx in perm:
                feat_pairs = indices_list[idx]
                y = y_list[idx]
                if not feat_pairs:
                    continue

                feat_indices = [f_idx for f_idx, _ in feat_pairs]
                linear_sum = float(np.sum(self.w[feat_indices]))

                # 2-way Field-Aware interaction sum
                inter_sum = 0.0
                m = len(feat_pairs)
                for a in range(m):
                    i_a, f_a = feat_pairs[a]
                    for b in range(a + 1, m):
                        i_b, f_b = feat_pairs[b]
                        inter_sum += float(np.dot(self.V[i_a, f_b], self.V[i_b, f_a]))

                raw_pred = self.w0 + linear_sum + inter_sum
                p_hat = 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, raw_pred))))
                err = p_hat - y

                # Update linear terms
                self.w0 -= self.learning_rate * err
                self.w[feat_indices] -= self.learning_rate * (err + self.regularization * self.w[feat_indices])

                # Update field-aware factor vectors
                for a in range(m):
                    i_a, f_a = feat_pairs[a]
                    for b in range(a + 1, m):
                        i_b, f_b = feat_pairs[b]
                        v_a_old = self.V[i_a, f_b].copy()
                        v_b_old = self.V[i_b, f_a].copy()

                        self.V[i_a, f_b] -= self.learning_rate * (err * v_b_old + self.regularization * v_a_old)
                        self.V[i_b, f_a] -= self.learning_rate * (err * v_a_old + self.regularization * v_b_old)

        latency = round((time.time() - start_time) * 1000, 2)
        return RecommenderModelSummary(
            algorithm="field_aware_factorization_machine",
            task_mode="field_aware_interaction",
            hyperparameters={
                "latent_dim": self.latent_dim,
                "learning_rate": self.learning_rate,
                "regularization": self.regularization,
                "epochs": self.epochs,
                "seed": self.seed,
            },
            training_summary={
                "n_features": p,
                "n_fields": num_fields,
                "n_records": len(records),
            },
            latency_ms=latency,
            device="cpu",
            seed=self.seed,
        )

    def predict_score(
        self,
        user_id: str,
        item_id: str,
        user_features: dict | None = None,
        item_features: dict | None = None,
        context_features: dict | None = None,
    ) -> float:
        dummy_rec = ContextualFeedbackData(
            user_id=user_id,
            item_id=item_id,
            target=0.0,
            user_features=user_features or {},
            item_features=item_features or {},
            context_features=context_features or {},
        )
        feat_pairs = self._extract_sparse_features_with_fields(dummy_rec)
        if not feat_pairs:
            return 0.5

        feat_indices = [f_idx for f_idx, _ in feat_pairs]
        linear_sum = float(np.sum(self.w[feat_indices]))

        inter_sum = 0.0
        m = len(feat_pairs)
        for a in range(m):
            i_a, f_a = feat_pairs[a]
            for b in range(a + 1, m):
                i_b, f_b = feat_pairs[b]
                inter_sum += float(np.dot(self.V[i_a, f_b], self.V[i_b, f_a]))

        pred = self.w0 + linear_sum + inter_sum
        return 1.0 / (1.0 + math.exp(-max(-15.0, min(15.0, pred))))

    def predict_one(self, user_id: str, item_id: str, **kwargs: Any) -> float:
        """Alias for predict_score for common interface compatibility."""
        return self.predict_score(user_id, item_id, **kwargs)

    def recommend(
        self,
        user_id: str,
        k: int = 10,
        candidate_items: list[str] | None = None,
        user_features: dict | None = None,
        context_features: dict | None = None,
        exclude_seen: bool = True,
    ) -> list[str]:
        candidates = candidate_items if candidate_items is not None else self.all_items
        seen = self.user_seen_items.get(user_id, set()) if exclude_seen else set()

        scored: list[tuple[str, float]] = []
        for it in candidates:
            if it not in seen:
                s = self.predict_score(user_id, it, user_features=user_features, context_features=context_features)
                scored.append((it, s))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored[:k]]

