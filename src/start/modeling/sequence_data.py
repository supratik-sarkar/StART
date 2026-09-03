"""Sequence data layer: generation, windowing, and splitting.

Provides a genuinely sequential synthetic dataset (so the recurrent models are
exercised on data with real temporal structure, never tabular rows reshaped),
a sliding-window builder that turns a long multivariate series into
(n_windows, timesteps, features) tensors with a label per window, and an
explicit train/test/OOS split that respects temporal order (OOS is the most
recent block).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SequenceBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_oos: np.ndarray
    y_oos: np.ndarray
    n_features: int
    timesteps: int
    source: str = "synthetic"

    @property
    def shapes(self) -> dict[str, tuple]:
        return {
            "train": self.X_train.shape,
            "test": self.X_test.shape,
            "oos": self.X_oos.shape,
        }


def generate_sequence_dataset(
    n_series: int = 800,
    timesteps: int = 24,
    n_features: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sequential binary-classification data with real temporal
    structure. Positive class has a rising trend + a sinusoidal component in
    feature 0; negatives are trendless noise. Returns (X, y) with
    X shaped (n_series, timesteps, n_features)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n_series, timesteps, n_features)).astype(np.float32)
    y = rng.integers(0, 2, size=n_series).astype(np.int64)
    t = np.linspace(0, 3.14, timesteps)
    for i in range(n_series):
        if y[i] == 1:
            X[i, :, 0] += np.linspace(0, 2.5, timesteps)  # trend
            X[i, :, 0] += 0.8 * np.sin(2 * t)  # seasonality
            if n_features > 1:
                X[i, :, 1] += np.cumsum(rng.normal(0.05, 0.1, timesteps))  # drift
    return X, y


def sliding_windows(
    series: np.ndarray, labels: np.ndarray, window: int, stride: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a long (T, F) series + per-step labels into windowed
    (n_windows, window, F) sequences; each window's label is its last step."""
    if series.ndim != 2:
        raise ValueError("series must be 2-D (timesteps, features).")
    T = len(series)
    if window > T:
        raise ValueError(f"window ({window}) larger than series length ({T}).")
    xs, ys = [], []
    for start in range(0, T - window + 1, stride):
        xs.append(series[start : start + window])
        ys.append(labels[start + window - 1])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys)


def split_sequences(
    X: np.ndarray, y: np.ndarray, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
) -> SequenceBundle:
    """Order-preserving train/test/OOS split (OOS = most recent block)."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")
    n = len(X)
    n_train = int(n * fractions[0])
    n_test = int(n * fractions[1])
    return SequenceBundle(
        X_train=X[:n_train],
        y_train=y[:n_train],
        X_test=X[n_train : n_train + n_test],
        y_test=y[n_train : n_train + n_test],
        X_oos=X[n_train + n_test :],
        y_oos=y[n_train + n_test :],
        n_features=X.shape[2],
        timesteps=X.shape[1],
    )


def load_sequence_demo(timesteps: int = 24, n_features: int = 3, seed: int = 42) -> SequenceBundle:
    """Convenience: generate + split a demo sequence dataset."""
    X, y = generate_sequence_dataset(timesteps=timesteps, n_features=n_features, seed=seed)
    return split_sequences(X, y)
