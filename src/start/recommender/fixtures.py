"""Deterministic Recommender Test Datasets and Fixtures.

Provides three structurally distinct fixtures:
- DATASET_A: Explicit-feedback ratings (1-5 scale)
- DATASET_B: Implicit-feedback interaction events
- DATASET_C: Contextual / side-feature interactions
"""

from __future__ import annotations

import random
from typing import Any

import pandas as pd

from start.recommender.contracts import (
    ContextualFeedbackData,
    ExplicitFeedbackData,
    ImplicitFeedbackData,
)


def get_dataset_a_explicit(seed: int = 42) -> list[ExplicitFeedbackData]:
    """DATASET_A: Explicit ratings fixture (MovieLens-style).

    25 users × 30 items with 250 explicit ratings on a 1.0 - 5.0 scale.
    """
    rng = random.Random(seed)
    users = [f"U_{i:03d}" for i in range(1, 26)]
    items = [f"ITEM_{j:03d}" for j in range(1, 31)]

    records: list[ExplicitFeedbackData] = []
    seen = set()

    # Give every user at least 4 ratings
    for u in users:
        sampled_items = rng.sample(items, k=rng.randint(6, 12))
        for item in sampled_items:
            rating = round(rng.uniform(1.0, 5.0) * 2) / 2.0  # 1.0, 1.5, ... 5.0
            ts = 1700000000.0 + float(rng.randint(0, 1000000))
            records.append(ExplicitFeedbackData(user_id=u, item_id=item, rating=rating, timestamp=ts))
            seen.add((u, item))

    # Add extra ratings up to ~250
    while len(records) < 250:
        u = rng.choice(users)
        item = rng.choice(items)
        if (u, item) not in seen:
            rating = round(rng.uniform(1.0, 5.0) * 2) / 2.0
            ts = 1700000000.0 + float(rng.randint(0, 1000000))
            records.append(ExplicitFeedbackData(user_id=u, item_id=item, rating=rating, timestamp=ts))
            seen.add((u, item))

    records.sort(key=lambda r: (r.user_id, r.item_id))
    return records


def get_dataset_b_implicit(seed: int = 42) -> list[ImplicitFeedbackData]:
    """DATASET_B: Implicit interactions fixture (e-commerce clicks / events).

    30 users × 40 items with 350 implicit interactions and interaction strengths.
    """
    rng = random.Random(seed)
    users = [f"U_{i:03d}" for i in range(1, 31)]
    items = [f"ITEM_{j:03d}" for j in range(1, 41)]
    events = ["click", "view", "add_to_cart", "purchase"]
    event_weights = {"click": 1.0, "view": 0.5, "add_to_cart": 3.0, "purchase": 5.0}

    records: list[ImplicitFeedbackData] = []
    seen = set()

    for u in users:
        sampled_items = rng.sample(items, k=rng.randint(8, 14))
        for item in sampled_items:
            ev = rng.choice(events)
            strength = float(rng.randint(1, 5))
            w = event_weights.get(ev, 1.0)
            ts = 1700000000.0 + float(rng.randint(0, 1000000))
            records.append(
                ImplicitFeedbackData(
                    user_id=u,
                    item_id=item,
                    event=ev,
                    strength=strength,
                    weight=w,
                    timestamp=ts,
                )
            )
            seen.add((u, item))

    while len(records) < 350:
        u = rng.choice(users)
        item = rng.choice(items)
        if (u, item) not in seen:
            ev = rng.choice(events)
            strength = float(rng.randint(1, 5))
            w = event_weights.get(ev, 1.0)
            ts = 1700000000.0 + float(rng.randint(0, 1000000))
            records.append(
                ImplicitFeedbackData(
                    user_id=u,
                    item_id=item,
                    event=ev,
                    strength=strength,
                    weight=w,
                    timestamp=ts,
                )
            )
            seen.add((u, item))

    records.sort(key=lambda r: (r.user_id, r.item_id))
    return records


def get_dataset_c_contextual(seed: int = 42) -> list[ContextualFeedbackData]:
    """DATASET_C: Contextual / side-feature interactions fixture.

    20 users × 25 items with user features, item features, and context features.
    """
    rng = random.Random(seed)
    users = [f"U_{i:03d}" for i in range(1, 21)]
    items = [f"ITEM_{j:03d}" for j in range(1, 26)]

    user_tiers = ["STANDARD", "PREMIUM", "ENTERPRISE"]
    item_categories = ["ELECTRONICS", "FINANCE", "BOOKS", "SOFTWARE"]
    devices = ["mobile", "desktop", "tablet"]

    user_meta = {
        u: {
            "age": rng.randint(20, 65),
            "tier": rng.choice(user_tiers),
            "engagement_score": round(rng.uniform(0.1, 1.0), 3),
        }
        for u in users
    }

    item_meta = {
        item: {
            "category": rng.choice(item_categories),
            "price": round(rng.uniform(10.0, 500.0), 2),
            "rating_count": rng.randint(5, 100),
        }
        for item in items
    }

    records: list[ContextualFeedbackData] = []
    seen = set()

    for u in users:
        sampled_items = rng.sample(items, k=rng.randint(6, 12))
        for item in sampled_items:
            ctx = {
                "device": rng.choice(devices),
                "day_of_week": rng.randint(1, 7),
                "hour": rng.randint(0, 23),
            }
            # Target is binary interaction (0 or 1) with higher probability for higher engagement
            p = (user_meta[u]["engagement_score"] + (1.0 if item_meta[item]["category"] == "FINANCE" else 0.5)) / 2.0
            target = 1.0 if rng.random() < min(0.9, p) else 0.0
            ts = 1700000000.0 + float(rng.randint(0, 1000000))
            records.append(
                ContextualFeedbackData(
                    user_id=u,
                    item_id=item,
                    target=target,
                    user_features=user_meta[u],
                    item_features=item_meta[item],
                    context_features=ctx,
                    timestamp=ts,
                )
            )
            seen.add((u, item))

    while len(records) < 200:
        u = rng.choice(users)
        item = rng.choice(items)
        if (u, item) not in seen:
            ctx = {
                "device": rng.choice(devices),
                "day_of_week": rng.randint(1, 7),
                "hour": rng.randint(0, 23),
            }
            target = 1.0 if rng.random() < 0.6 else 0.0
            ts = 1700000000.0 + float(rng.randint(0, 1000000))
            records.append(
                ContextualFeedbackData(
                    user_id=u,
                    item_id=item,
                    target=target,
                    user_features=user_meta[u],
                    item_features=item_meta[item],
                    context_features=ctx,
                    timestamp=ts,
                )
            )
            seen.add((u, item))

    records.sort(key=lambda r: (r.user_id, r.item_id))
    return records


def to_dataframe(records: list[Any]) -> pd.DataFrame:
    """Convert any feedback record list into a pandas DataFrame."""
    data = [r.to_dict() for r in records]
    return pd.DataFrame(data)


# Canonical pre-instantiated deterministic fixtures (seed 42)
DATASET_A: list[ExplicitFeedbackData] = get_dataset_a_explicit(42)
DATASET_B: list[ImplicitFeedbackData] = get_dataset_b_implicit(42)
DATASET_C: list[ContextualFeedbackData] = get_dataset_c_contextual(42)
