"""Authoritative Recommender Data Validation, Split Protocols, and Negative Sampling.

Strict Invariants:
1. Deterministic behavior under seed.
2. Sparsity reported as a characteristic, not rejected as an error.
3. Explicit negative sampling with candidate universe and seen-item exclusion.
4. Cold-start cohort partitioning without silent cohort mixing.
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np
import pandas as pd

from start.recommender.contracts import (
    RecommenderDatasetProfile,
    RecommenderSplitResult,
    SplitProtocol,
)


def validate_recommender_dataset(
    df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "item_id",
    rating_col: str | None = "rating",
    min_rating: float = 0.5,
    max_rating: float = 5.0,
    min_user_interactions: int = 2,
    min_item_interactions: int = 2,
) -> RecommenderDatasetProfile:
    """Perform deterministic validation checks on a recommender interaction DataFrame."""
    warnings: list[str] = []

    # Check required columns
    if user_col not in df.columns:
        raise ValueError(f"Missing user column: {user_col}")
    if item_col not in df.columns:
        raise ValueError(f"Missing item column: {item_col}")

    # Null checks
    null_users = int(df[user_col].isnull().sum())
    if null_users > 0:
        raise ValueError(f"Dataset contains {null_users} null user IDs.")

    null_items = int(df[item_col].isnull().sum())
    if null_items > 0:
        raise ValueError(f"Dataset contains {null_items} null item IDs.")

    # Duplicate interactions
    dup_count = int(df.duplicated(subset=[user_col, item_col]).sum())
    if dup_count > 0:
        warnings.append(f"Detected {dup_count} duplicate user-item interaction pairs.")

    # Ratings validity (if explicit)
    r_min, r_max, r_mean, r_std = None, None, None, None
    feedback_mode = "implicit"
    if rating_col and rating_col in df.columns:
        feedback_mode = "explicit"
        ratings = df[rating_col].dropna()
        if len(ratings) > 0:
            r_min = float(ratings.min())
            r_max = float(ratings.max())
            r_mean = float(ratings.mean())
            r_std = float(ratings.std()) if len(ratings) > 1 else 0.0

            if r_min < min_rating or r_max > max_rating:
                warnings.append(
                    f"Rating range [{r_min}, {r_max}] exceeds expected bounds [{min_rating}, {max_rating}]."
                )

    # Cardinality and Sparsity
    n_users = int(df[user_col].nunique())
    n_items = int(df[item_col].nunique())
    n_interactions = int(len(df))

    possible_interactions = max(1, n_users * n_items)
    density = n_interactions / possible_interactions
    sparsity = max(0.0, min(1.0, 1.0 - density))
    density_percent = round(density * 100.0, 4)

    # History distributions
    user_counts = df[user_col].value_counts()
    item_counts = df[item_col].value_counts()

    low_history_users = int((user_counts < min_user_interactions).sum())
    if low_history_users > 0:
        warnings.append(
            f"{low_history_users} users have fewer than {min_user_interactions} interactions."
        )

    low_history_items = int((item_counts < min_item_interactions).sum())
    if low_history_items > 0:
        warnings.append(
            f"{low_history_items} items have fewer than {min_item_interactions} interactions."
        )

    return RecommenderDatasetProfile(
        n_users=n_users,
        n_items=n_items,
        n_interactions=n_interactions,
        sparsity=round(sparsity, 4),
        density_percent=density_percent,
        feedback_mode=feedback_mode,
        rating_min=r_min,
        rating_max=r_max,
        rating_mean=round(r_mean, 4) if r_mean is not None else None,
        rating_std=round(r_std, 4) if r_std is not None else None,
        interactions_per_user={
            "mean": round(float(user_counts.mean()), 2),
            "median": round(float(user_counts.median()), 2),
            "min": int(user_counts.min()),
            "max": int(user_counts.max()),
        },
        interactions_per_item={
            "mean": round(float(item_counts.mean()), 2),
            "median": round(float(item_counts.median()), 2),
            "min": int(item_counts.min()),
            "max": int(item_counts.max()),
        },
        warnings=warnings,
    )


def profile_recommender_dataset(data: Any) -> RecommenderDatasetProfile:
    """Convenience helper converting arbitrary raw recommender data (list of records or DataFrame) to profile."""
    if isinstance(data, pd.DataFrame):
        df = data
    elif hasattr(data, "to_dataframe"):
        df = data.to_dataframe()
    elif isinstance(data, list):
        if len(data) > 0 and hasattr(data[0], "to_dict"):
            df = pd.DataFrame([d.to_dict() for d in data])
        elif len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
        elif len(data) > 0 and hasattr(data[0], "__dict__"):
            df = pd.DataFrame([vars(d) for d in data])
        else:
            df = pd.DataFrame(data)
    else:
        df = pd.DataFrame(data)

    rating_col = "rating" if "rating" in df.columns else ("weight" if "weight" in df.columns else ("label" if "label" in df.columns else None))
    return validate_recommender_dataset(df, rating_col=rating_col)


def split_recommender_dataset(
    df: pd.DataFrame,
    protocol: SplitProtocol | str = SplitProtocol.USER_STRATIFIED,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    user_col: str = "user_id",
    item_col: str = "item_id",
    time_col: str = "timestamp",
    seed: int = 42,
) -> RecommenderSplitResult:
    """Partition interactions under the specified split protocol."""
    rng = np.random.default_rng(seed)
    protocol_str = str(protocol)

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []

    cold_user_ids: list[str] = []
    cold_item_ids: list[str] = []

    if protocol_str == SplitProtocol.RANDOM_INTERACTION:
        indices = np.arange(len(df))
        rng.shuffle(indices)
        n_test = int(len(df) * test_ratio)
        n_val = int(len(df) * val_ratio)
        test_indices = list(indices[:n_test])
        val_indices = list(indices[n_test : n_test + n_val])
        train_indices = list(indices[n_test + n_val :])

    elif protocol_str == SplitProtocol.USER_STRATIFIED:
        # For each user, hold out test_ratio interactions
        for _, group in df.groupby(user_col):
            g_idx = group.index.to_numpy().copy()
            if len(g_idx) < 3:
                # If too small, keep in train
                train_indices.extend(g_idx)
                continue
            rng.shuffle(g_idx)
            n_test = max(1, int(len(g_idx) * test_ratio))
            n_val = max(1, int(len(g_idx) * val_ratio)) if val_ratio > 0 else 0
            test_indices.extend(g_idx[:n_test])
            val_indices.extend(g_idx[n_test : n_test + n_val])
            train_indices.extend(g_idx[n_test + n_val :])

    elif protocol_str == SplitProtocol.TEMPORAL_HOLDOUT:
        # Sort by timestamp globally
        if time_col in df.columns and df[time_col].notnull().any():
            sorted_df = df.sort_values(time_col)
            n_test = int(len(df) * test_ratio)
            n_val = int(len(df) * val_ratio)
            train_indices = list(sorted_df.index[: len(df) - n_test - n_val])
            val_indices = list(sorted_df.index[len(df) - n_test - n_val : len(df) - n_test])
            test_indices = list(sorted_df.index[len(df) - n_test :])
        else:
            # Fallback to random interaction if timestamp unavailable
            return split_recommender_dataset(df, SplitProtocol.RANDOM_INTERACTION, test_ratio, val_ratio, user_col, item_col, time_col, seed)

    elif protocol_str == SplitProtocol.LEAVE_ONE_OUT:
        # Leave latest or random 1 interaction per user for test, 1 for val
        for _, group in df.groupby(user_col):
            if time_col in group.columns and group[time_col].notnull().any():
                g_sorted = group.sort_values(time_col).index.to_list()
            else:
                g_sorted = group.index.to_list()
                rng.shuffle(g_sorted)

            if len(g_sorted) >= 3:
                test_indices.append(g_sorted[-1])
                val_indices.append(g_sorted[-2])
                train_indices.extend(g_sorted[:-2])
            elif len(g_sorted) == 2:
                test_indices.append(g_sorted[-1])
                train_indices.append(g_sorted[0])
            else:
                train_indices.extend(g_sorted)

    elif protocol_str == SplitProtocol.COLD_USER:
        unique_users = df[user_col].unique()
        rng.shuffle(unique_users)
        n_cold = max(1, int(len(unique_users) * test_ratio))
        cold_user_ids = list(unique_users[:n_cold])
        warm_user_ids = set(unique_users[n_cold:])

        test_indices = list(df[df[user_col].isin(cold_user_ids)].index)
        train_indices = list(df[df[user_col].isin(warm_user_ids)].index)

    elif protocol_str == SplitProtocol.COLD_ITEM:
        unique_items = df[item_col].unique()
        rng.shuffle(unique_items)
        n_cold = max(1, int(len(unique_items) * test_ratio))
        cold_item_ids = list(unique_items[:n_cold])
        warm_item_ids = set(unique_items[n_cold:])

        test_indices = list(df[df[item_col].isin(cold_item_ids)].index)
        train_indices = list(df[df[item_col].isin(warm_item_ids)].index)

    else:
        raise ValueError(f"Unsupported split protocol: {protocol}")

    train_df = df.loc[train_indices].copy()
    test_df = df.loc[test_indices].copy()
    val_df = df.loc[val_indices].copy() if val_indices else pd.DataFrame(columns=df.columns)

    train_users = set(train_df[user_col].unique())
    test_users = set(test_df[user_col].unique())
    train_items = set(train_df[item_col].unique())
    test_items = set(test_df[item_col].unique())

    warm_users = test_users.intersection(train_users)
    cold_users = test_users - train_users
    warm_items = test_items.intersection(train_items)
    cold_items = test_items - train_items

    return RecommenderSplitResult(
        protocol=protocol_str,
        train_count=len(train_df),
        test_count=len(test_df),
        val_count=len(val_df),
        warm_users_count=len(warm_users),
        cold_users_count=len(cold_users),
        warm_items_count=len(warm_items),
        cold_items_count=len(cold_items),
        train_data=train_df,
        test_data=test_df,
        val_data=val_df,
        cold_user_ids=list(cold_users),
        cold_item_ids=list(cold_items),
    )


def sample_negative_items(
    positive_user_items: dict[str, set[str]],
    all_items: list[str],
    n_negatives: int = 4,
    seed: int = 42,
) -> dict[str, list[str]]:
    """Sample non-interacted negative items per user without replacement."""
    rng = random.Random(seed)
    item_set = set(all_items)
    negatives_per_user: dict[str, list[str]] = {}

    for u, pos_items in positive_user_items.items():
        candidates = list(item_set - pos_items)
        if not candidates:
            negatives_per_user[u] = []
            continue
        k = min(n_negatives, len(candidates))
        negatives_per_user[u] = rng.sample(candidates, k=k)

    return negatives_per_user
