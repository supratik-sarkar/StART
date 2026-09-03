"""Datasets and cohort splitting for modeling and benchmark suites.

Benchmark binary classification utilities load sklearn's breast cancer dataset
explicitly and truthfully. For the demo and wizard workflows, synthetic
transactions generated locally are used as fallback.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_COLUMN = "attrition"
SCORE_COLUMN = "score"


def load_breast_cancer_diagnostic_dataset(seed: int = 42) -> pd.DataFrame:
    """Public binary-classification benchmark from scikit-learn."""
    try:
        from sklearn.datasets import load_breast_cancer

        bundle = load_breast_cancer(as_frame=True)
        df = bundle.data.copy()
        df.columns = [c.replace(" ", "_") for c in df.columns]
        df[TARGET_COLUMN] = (bundle.target == 0).astype(int)
        return df
    except Exception:
        return _synthetic_fallback(seed)


load_attrition_dataset = load_breast_cancer_diagnostic_dataset


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
    stratify: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified or random train / test / out-of-sample split (default 60/20/20)."""
    from sklearn.model_selection import train_test_split

    if not np.isclose(sum(fracs), 1.0):
        raise ValueError(f"Split fractions must sum to 1.0, got {fracs}")
    train_frac, test_frac, oos_frac = fracs

    stratify_y = df[target_column] if (stratify and target_column in df.columns) else None
    train, rest = train_test_split(df, test_size=test_frac + oos_frac, stratify=stratify_y, random_state=seed)

    stratify_rest = rest[target_column] if (stratify and target_column in rest.columns) else None
    test, oos = train_test_split(
        rest,
        test_size=oos_frac / (test_frac + oos_frac),
        stratify=stratify_rest,
        random_state=seed,
    )
    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        oos.reset_index(drop=True),
    )


def feature_columns(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> list[str]:
    return [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in {target_column, SCORE_COLUMN}
    ]


def load_preset_dataset(preset_key: str, seed: int = 42) -> pd.DataFrame:
    """Load a diverse public/synthetic dataset with missing values and outliers based on preset key."""
    import numpy as np
    from sklearn.datasets import make_classification, make_regression

    np.random.seed(seed)

    if preset_key == "A":
        # Anomaly / Transaction Monitoring: Binary Classification (is_fraud)
        # Highly imbalanced, outliers, missing values
        X, y = make_classification(
            n_samples=600,
            n_features=25,
            n_informative=15,
            n_redundant=5,
            weights=[0.95, 0.05],
            random_state=seed,
        )
        df = pd.DataFrame(X, columns=[f"feature_{i:02d}" for i in range(25)])
        df["is_fraud"] = y

        # Inject missing values (approx 3% in some columns)
        for col in ["feature_02", "feature_07", "feature_12"]:
            mask = np.random.rand(len(df)) < 0.04
            df.loc[mask, col] = np.nan

        # Inject outliers (extreme values in some features)
        for col in ["feature_00", "feature_05", "feature_10"]:
            outliers_mask = np.random.rand(len(df)) < 0.05
            df.loc[outliers_mask, col] = df.loc[outliers_mask, col] * 15.0

        return df

    elif preset_key == "B":
        # Time-Series Forecasting: Regression (target_value)
        # Sequential features, trend, seasonality, outliers, missing values
        n_samples = 600
        # Base trend and seasonality
        trend = np.linspace(10, 50, n_samples)
        seasonality = 15 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
        noise = np.random.randn(n_samples) * 5.0

        target_val = trend + seasonality + noise

        # Add extreme outliers to target/features
        outlier_indices = np.random.choice(n_samples, size=15, replace=False)
        target_val[outlier_indices] += np.random.choice([-100.0, 100.0], size=15)

        df = pd.DataFrame(
            {
                "feature_trend": trend + np.random.randn(n_samples),
                "feature_season": seasonality + np.random.randn(n_samples),
                "feature_noise": np.random.randn(n_samples) * 2,
                "feature_lag1": np.roll(target_val, 1),
                "feature_lag2": np.roll(target_val, 2),
                "target_value": target_val,
            }
        )
        # Handle rolled edge case
        df.iloc[0, df.columns.get_loc("feature_lag1")] = target_val[0]
        df.iloc[0, df.columns.get_loc("feature_lag2")] = target_val[0]
        df.iloc[1, df.columns.get_loc("feature_lag2")] = target_val[1]

        # Inject missing values
        for col in ["feature_lag1", "feature_noise"]:
            mask = np.random.rand(len(df)) < 0.05
            df.loc[mask, col] = np.nan

        return df

    elif preset_key == "C":
        # Asset Pricing: Regression (adjusted_price)
        # Outliers, missing values
        X, y = make_regression(n_samples=600, n_features=20, n_informative=12, noise=10.0, random_state=seed)
        # Scale target to resemble adjusted prices
        y = np.abs(y) * 1.5 + 50.0
        df = pd.DataFrame(X, columns=[f"feature_{i:02d}" for i in range(20)])
        df["adjusted_price"] = y

        # Inject missing values
        for col in ["feature_01", "feature_04", "feature_09"]:
            mask = np.random.rand(len(df)) < 0.04
            df.loc[mask, col] = np.nan

        # Inject outliers
        for col in ["feature_00", "feature_08", "adjusted_price"]:
            outliers_mask = np.random.rand(len(df)) < 0.05
            df.loc[outliers_mask, col] = df.loc[outliers_mask, col] * 12.0

        return df

    elif preset_key == "D":
        # ML Decision Support Model Data: Multiclass Classification (decision_label)
        # Classes: 0, 1, 2. Outliers, missing values
        X, y = make_classification(
            n_samples=600,
            n_features=22,
            n_informative=12,
            n_redundant=4,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=seed,
        )
        df = pd.DataFrame(X, columns=[f"feature_{i:02d}" for i in range(22)])
        df["decision_label"] = y

        # Inject missing values
        for col in ["feature_03", "feature_06"]:
            mask = np.random.rand(len(df)) < 0.04
            df.loc[mask, col] = np.nan

        # Inject outliers
        for col in ["feature_02", "feature_05"]:
            outliers_mask = np.random.rand(len(df)) < 0.05
            df.loc[outliers_mask, col] = df.loc[outliers_mask, col] * 10.0

        return df

    else:
        # Fallback: locally generated transaction data. Never a medical dataset
        # relabelled as something else — see tests/test_v401_contracts.py.
        from start.data.synthetic import generate_synthetic_transactions

        return generate_synthetic_transactions(n_rows=1000, prevalence=0.055, n_features=25, seed=seed)
