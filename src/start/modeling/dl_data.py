"""Deep-learning data loading.

Reuses StART's universal connector layer (demo / local files / pandas / Spark
/ Snowflake) so DL reviews accept user data with no code changes, and adds a
DL-friendly synthetic dataset with non-linear structure for the offline
default. Returns the same cohort contract as the classical workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from start.modeling.data import SCORE_COLUMN, TARGET_COLUMN, feature_columns, three_way_split

__all__ = [
    "SCORE_COLUMN",
    "TARGET_COLUMN",
    "feature_columns",
    "three_way_split",
    "load_dl_demo_dataset",
    "load_dl_bundle",
]


def load_dl_demo_dataset(seed: int = 42, n: int = 3000, n_features: int = 24) -> pd.DataFrame:
    """Synthetic binary-classification data with non-linear interactions that
    reward a neural model — fully offline, no downloads. Falls back to the
    shared attrition demo set if sklearn's generator is unavailable."""
    try:
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=n,
            n_features=n_features,
            n_informative=12,
            n_redundant=4,
            n_clusters_per_class=3,
            class_sep=1.1,
            flip_y=0.02,
            weights=[0.62, 0.38],
            random_state=seed,
        )
        # inject a couple of explicit non-linear interaction features
        inter = (X[:, 0] * X[:, 1]).reshape(-1, 1)
        squared = (X[:, 2] ** 2).reshape(-1, 1)
        X = np.hstack([X, inter, squared])
        cols = [f"feature_{i:02d}" for i in range(n_features)] + ["interaction_01", "squared_02"]
        df = pd.DataFrame(X, columns=cols)
        df[TARGET_COLUMN] = y.astype(int)
        return df
    except Exception:
        from start.modeling.data import load_attrition_dataset

        return load_attrition_dataset(seed=seed)


def load_dl_bundle(
    source: str = "demo",
    *,
    train_path: str | None = None,
    test_path: str | None = None,
    oos_path: str | None = None,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    oos_df: pd.DataFrame | None = None,
    spark_train: object = None,
    spark: object = None,
    target_column: str = TARGET_COLUMN,
    seed: int = 42,
):
    """Resolve a DatasetBundle for the DL review across all connector modes.

    source: demo | files | pandas | spark. The bundle auto-splits 60/20/20
    (stratified) when only a single cohort is supplied."""
    from start.connectors import (
        LocalFileConnector,
        PandasConnector,
        SparkConnector,
    )

    if source == "pandas" or train_df is not None:
        if train_df is None:
            raise ValueError("source 'pandas' requires train_df.")
        connector = PandasConnector(
            train_df, test_df, oos_df, seed=seed, target_column=target_column
        )
    elif source == "files":
        if not train_path:
            raise ValueError("source 'files' requires train_path.")
        connector = LocalFileConnector(
            train_path, test_path, oos_path, seed=seed, target_column=target_column
        )
    elif source == "spark":
        if spark_train is None:
            raise ValueError("source 'spark' requires spark_train (DataFrame/table/SQL).")
        connector = SparkConnector(
            spark_train, spark=spark, seed=seed, target_column=target_column
        )
    else:  # demo
        connector = _DLDemoConnector(seed=seed, target_column=target_column)
    return connector.load_bundle()


class _DLDemoConnector:
    """Demo connector variant that serves the DL-friendly synthetic dataset."""

    name = "dl_demo"

    def __init__(self, seed: int = 42, target_column: str = TARGET_COLUMN) -> None:
        self.seed = seed
        self.target_column = target_column

    def load_bundle(self):
        from start.connectors import DatasetBundle

        df = load_dl_demo_dataset(seed=self.seed)
        bundle = DatasetBundle(
            train=df,
            source="demo:dl-synthetic",
            target_column=self.target_column,
            notes=["DL-friendly synthetic dataset; substitute your own via any connector."],
        )
        return bundle.ensure_split(self.seed)
