"""Synthetic transaction generator (Workstream B1).

Generates a realistic, parameterized tabular transaction dataset with AML/fraud
characteristics for reproducible local demonstrations without external network
downloads or credentials.

Pre-flight panel text: "synthetic, generated locally, no external source."
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def generate_synthetic_transactions(
    n_rows: int = 1000,
    prevalence: float = 0.055,
    n_features: int = 25,
    snr: float | None = None,
    signal_to_noise: float | None = None,
    inject_leakage: bool = False,
    inject_proxy: bool = False,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic financial transaction dataset for fraud detection.

    Parameters
    ----------
    n_rows : int
        Number of transaction rows (default 1,000).
    prevalence : float
        Minority class prevalence for `is_fraud` (default 5.5%).
    n_features : int
        Total number of feature columns (default 25).
    snr : float | None
        Alias for `signal_to_noise`.
    signal_to_noise : float | None
        Signal-to-noise multiplier controlling separability (default 0.85).
    inject_leakage : bool
        If True, injects `post_txn_chargeback_status` (post-event target leakage).
    inject_proxy : bool
        If True, injects `zip_code_risk_cluster` (proxy demographic characteristic).
    seed : int
        Random seed for deterministic generation.
    """
    rng = np.random.default_rng(seed)
    s = signal_to_noise if signal_to_noise is not None else (snr if snr is not None else 0.85)

    # 1. Target vector with exact prevalence
    n_pos = max(1, int(round(n_rows * prevalence)))
    y = np.zeros(n_rows, dtype=int)
    pos_indices = rng.choice(n_rows, size=n_pos, replace=False)
    y[pos_indices] = 1

    # 2. Core realistic domain features with overlapping continuous/discrete distributions
    # External risk score (loc=30, scale=15, shift with overlap)
    risk_score = np.clip(rng.normal(loc=30.0, scale=15.0, size=n_rows) + (y * 12.0 * s), 0.0, 100.0)

    # Velocity in 24h
    velocity = np.clip(
        rng.poisson(lam=2.5, size=n_rows) + (y * rng.choice([1, 2], p=[0.5, 0.5], size=n_rows) * s),
        0,
        30,
    )

    # Transaction amount z-score
    txn_amount = rng.standard_normal(n_rows) + (y * 0.60 * s)

    # Shipping and billing ZIP match
    p_zip = np.clip(0.90 - (y * 0.18 * s), 0.0, 1.0)
    zip_match = (rng.uniform(0, 1, size=n_rows) < p_zip).astype(int)

    # Device fingerprint match
    p_dev = np.clip(0.88 - (y * 0.20 * s), 0.0, 1.0)
    dev_match = (rng.uniform(0, 1, size=n_rows) < p_dev).astype(int)

    # Card present flag
    p_cp = np.clip(0.70 - (y * 0.22 * s), 0.0, 1.0)
    card_present = (rng.uniform(0, 1, size=n_rows) < p_cp).astype(int)

    # Merchant risk band (1 to 5)
    merchant_risk = rng.integers(1, 6, size=n_rows)
    merchant_risk[y == 1] = np.clip(
        merchant_risk[y == 1] + rng.choice([0, 1], p=[0.6, 0.4], size=n_pos), 1, 5
    )

    # Hours since previous transaction
    hours_since_prev = np.clip(rng.exponential(scale=12.0, size=n_rows) - (y * 1.8 * s), 0.1, 72.0)

    # Foreign transaction flag
    p_foreign = np.clip(0.08 + (y * 0.14 * s), 0.0, 1.0)
    is_foreign = (rng.uniform(0, 1, size=n_rows) < p_foreign).astype(int)

    # Online order flag
    p_online = np.clip(0.38 + (y * 0.18 * s), 0.0, 1.0)
    online_order = (rng.uniform(0, 1, size=n_rows) < p_online).astype(int)

    # Account age in days
    acct_age = np.clip(
        rng.integers(30, 3650, size=n_rows) - (y * rng.integers(50, 200, size=n_rows) * s), 5, 3650
    )

    data: dict[str, Any] = {
        "txn_amount_zscore": np.round(txn_amount, 4),
        "merchant_risk_band": merchant_risk,
        "hours_since_prev_txn": np.round(hours_since_prev, 2),
        "is_foreign_txn": is_foreign,
        "card_present": card_present,
        "velocity_24h": velocity,
        "acct_age_days": np.round(acct_age, 0).astype(int),
        "online_order_flag": online_order,
        "device_fingerprint_match": dev_match,
        "shipping_billing_zip_match": zip_match,
        "risk_score_external": np.round(risk_score, 2),
    }

    # Optional Injected Leakage Feature (post-event signal)
    if inject_leakage:
        leak_pos = rng.choice([1, 0], p=[0.98, 0.02], size=n_rows)
        leak_neg = rng.choice([0, 1], p=[0.99, 0.01], size=n_rows)
        data["post_txn_chargeback_status"] = np.where(y == 1, leak_pos, leak_neg)

    # Optional Injected Proxy Feature (demographic cluster proxy)
    if inject_proxy:
        proxy = np.clip(rng.integers(1, 10, size=n_rows) + (y * 2), 1, 10)
        data["zip_code_risk_cluster"] = proxy

    # Additional standard numeric features to reach requested count
    core_count = len(data)
    for i in range(core_count, n_features):
        feat_name = f"feature_{i:02d}"
        if i % 3 == 0:
            data[feat_name] = np.round(rng.standard_normal(n_rows) + (y * 0.20 * s), 4)
        else:
            data[feat_name] = np.round(rng.standard_normal(n_rows), 4)

    # Target
    data["is_fraud"] = y

    df = pd.DataFrame(data)
    return df
