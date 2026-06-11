"""Datasets and cohort splitting for the propensity-style demo.

The default dataset is sklearn's public breast-cancer dataset, reframed as a
generic "client attrition / propensity" binary classification case: the
positive class represents the event of interest. No client data is involved.
A synthetic make_classification fallback is provided.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "attrition"
SCORE_COLUMN = "score"


def load_attrition_dataset(seed: int = 42) -> pd.DataFrame:
    """Public binary-classification dataset framed as a propensity case."""
    try:
        from sklearn.datasets import load_breast_cancer

        bundle = load_breast_cancer(as_frame=True)
        df = bundle.data.copy()
        df.columns = [c.replace(" ", "_") for c in df.columns]
        # sklearn encodes 0 = malignant; treat that as the positive event so
        # the demo has a realistic ~37% event rate.
        df[TARGET_COLUMN] = (bundle.target == 0).astype(int)
        return df
    except Exception:
        return _synthetic_fallback(seed)


def _synthetic_fallback(seed: int, n: int = 2000, n_features: int = 20) -> pd.DataFrame:
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=8,
        n_redundant=4,
        weights=[0.7, 0.3],
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i:02d}" for i in range(n_features)])
    df[TARGET_COLUMN] = y
    return df


def three_way_split(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    fracs: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / test / out-of-sample split (default 60/20/20)."""
    from sklearn.model_selection import train_test_split

    if not np.isclose(sum(fracs), 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {fracs}")
    train_frac, test_frac, oos_frac = fracs
    train, rest = train_test_split(
        df, test_size=test_frac + oos_frac, stratify=df[target_column], random_state=seed
    )
    test, oos = train_test_split(
        rest,
        test_size=oos_frac / (test_frac + oos_frac),
        stratify=rest[target_column],
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        oos.reset_index(drop=True),
    )


def feature_columns(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> list[str]:
    return [
        c
        for c in df.select_dtypes(include=[np.number]).columns
        if c not in {target_column, SCORE_COLUMN}
    ]
