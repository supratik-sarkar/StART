"""Deep-learning demo data utilities for StART.

These helpers are intentionally laptop-safe and public-data only. They keep the
same train/test/OOS split convention used by the propensity workflow while
supporting user-supplied pandas DataFrames in notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DLDatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    oos: pd.DataFrame
    target_column: str
    feature_columns: list[str]
    dataset_name: str = "demo:dl_binary"

    @property
    def all_data(self) -> pd.DataFrame:
        return pd.concat([self.train, self.test, self.oos], axis=0, ignore_index=True)


def make_demo_dl_binary_dataset(
    *,
    n_samples: int = 1500,
    n_features: int = 24,
    n_informative: int = 12,
    seed: int = 42,
    target_column: str = "target",
) -> pd.DataFrame:
    """Create a public synthetic binary dataset for laptop-safe DL review."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=4,
        n_repeated=0,
        n_classes=2,
        weights=[0.58, 0.42],
        class_sep=1.25,
        flip_y=0.025,
        random_state=seed,
    )
    cols = [f"feature_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df[target_column] = y.astype(int)
    return df


def split_train_test_oos(
    df: pd.DataFrame,
    *,
    target_column: str,
    train_size: float = 0.60,
    test_size: float = 0.20,
    seed: int = 42,
    dataset_name: str = "user:dl_binary",
) -> DLDatasetBundle:
    """Split a single dataframe into 60/20/20 train/test/OOS by default."""
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in dataframe")
    if not 0 < train_size < 1 or not 0 < test_size < 1 or train_size + test_size >= 1:
        raise ValueError("train_size and test_size must be positive and leave room for OOS")

    feature_columns = [c for c in df.columns if c != target_column]
    y = df[target_column]
    stratify = y if y.nunique(dropna=True) == 2 else None

    train, tmp = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=stratify,
    )
    tmp_y = tmp[target_column]
    tmp_stratify = tmp_y if tmp_y.nunique(dropna=True) == 2 else None
    rel_test_size = test_size / (1.0 - train_size)
    test, oos = train_test_split(
        tmp,
        train_size=rel_test_size,
        random_state=seed,
        stratify=tmp_stratify,
    )
    return DLDatasetBundle(
        train=train.reset_index(drop=True),
        test=test.reset_index(drop=True),
        oos=oos.reset_index(drop=True),
        target_column=target_column,
        feature_columns=feature_columns,
        dataset_name=dataset_name,
    )


def prepare_feature_matrices(
    bundle: DLDatasetBundle,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    StandardScaler,
]:
    """Standardize features using train statistics only."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(bundle.train[bundle.feature_columns]).astype("float32")
    X_test = scaler.transform(bundle.test[bundle.feature_columns]).astype("float32")
    X_oos = scaler.transform(bundle.oos[bundle.feature_columns]).astype("float32")
    y_train = bundle.train[bundle.target_column].to_numpy(dtype="float32")
    y_test = bundle.test[bundle.target_column].to_numpy(dtype="float32")
    y_oos = bundle.oos[bundle.target_column].to_numpy(dtype="float32")
    return X_train, y_train, X_test, y_test, X_oos, y_oos, scaler


def load_default_bundle(seed: int = 42) -> DLDatasetBundle:
    df = make_demo_dl_binary_dataset(seed=seed)
    return split_train_test_oos(df, target_column="target", seed=seed, dataset_name="demo:dl_binary")
