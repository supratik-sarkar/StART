"""Deterministic Tabular Preprocessing Engine for StART.

Invariants:
1. Strict zero test-data leakage: all imputation parameters, outlier bounds,
   scalers, and target encoding statistics are fitted exclusively on training data.
2. The exact selected transformer pipeline is executed and returned.
3. No silent ignoring of user preprocessing configurations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def apply_preprocessing_pipeline(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray | None = None,
    preprocessing: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Apply deterministic preprocessing pipeline fitted strictly on training data.

    Returns:
        tuple of (X_train_transformed, X_test_transformed, resolved_summary)
    """
    prep = preprocessing or {}
    X_tr = X_train.copy()
    X_te = X_test.copy()

    # Identify numeric and categorical columns
    cat_cols = [c for c in X_tr.columns if X_tr[c].dtype == object or str(X_tr[c].dtype) == "category"]
    num_cols = [c for c in X_tr.columns if c not in cat_cols]

    # 1. Categorical Encoding (if categorical columns exist)
    raw_enc = prep.get("encoding") or prep.get("categorical_encoding")
    if raw_enc and str(raw_enc).lower() not in ("none", ""):
        enc_strat = str(raw_enc).lower()
    else:
        enc_strat = "ordinal" if cat_cols else "none"
    applied_encoding = "none"

    if cat_cols and enc_strat not in ("none", ""):
        if enc_strat in ("onehot", "one_hot"):
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            tr_enc = encoder.fit_transform(X_tr[cat_cols])
            te_enc = encoder.transform(X_te[cat_cols])
            enc_feature_names = [f"{col}_{cat}" for col, cats in zip(cat_cols, encoder.categories_) for cat in cats]

            X_tr_enc = pd.DataFrame(tr_enc, columns=enc_feature_names, index=X_tr.index)
            X_te_enc = pd.DataFrame(te_enc, columns=enc_feature_names, index=X_te.index)

            X_tr = pd.concat([X_tr[num_cols], X_tr_enc], axis=1)
            X_te = pd.concat([X_te[num_cols], X_te_enc], axis=1)
            num_cols = list(X_tr.columns)
            cat_cols = []
            applied_encoding = "onehot"

        elif enc_strat == "ordinal":
            from sklearn.preprocessing import OrdinalEncoder
            encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_tr[cat_cols] = encoder.fit_transform(X_tr[cat_cols])
            X_te[cat_cols] = encoder.transform(X_te[cat_cols])
            num_cols = list(X_tr.columns)
            cat_cols = []
            applied_encoding = "ordinal"

        elif enc_strat == "target" and y_train is not None:
            # Target encoding with smoothing (fit exclusively on train)
            global_mean = float(np.mean(y_train))
            smoothing_weight = 10.0
            for col in cat_cols:
                series_tr = X_tr[col].astype(str)
                df_temp = pd.DataFrame({"cat": series_tr, "target": y_train})
                stats = df_temp.groupby("cat")["target"].agg(["count", "mean"])
                # Smoothed target estimate
                smoothed = (stats["count"] * stats["mean"] + smoothing_weight * global_mean) / (stats["count"] + smoothing_weight)
                mapping = smoothed.to_dict()

                X_tr[col] = series_tr.map(mapping).fillna(global_mean).astype(float)
                X_te[col] = X_te[col].astype(str).map(mapping).fillna(global_mean).astype(float)
            num_cols = list(X_tr.columns)
            cat_cols = []
            applied_encoding = "target"

        elif enc_strat == "frequency":
            for col in cat_cols:
                freq = X_tr[col].value_counts(normalize=True).to_dict()
                X_tr[col] = X_tr[col].map(freq).fillna(0.0).astype(float)
                X_te[col] = X_te[col].map(freq).fillna(0.0).astype(float)
            num_cols = list(X_tr.columns)
            cat_cols = []
            applied_encoding = "frequency"

    # 2. Imputation
    impute_strat = str(prep.get("imputation") or "median").lower()
    applied_imputation = "none"
    if impute_strat in ("mean", "median", "most_frequent") and num_cols:
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy=impute_strat)
        X_tr[num_cols] = imputer.fit_transform(X_tr[num_cols])
        X_te[num_cols] = imputer.transform(X_te[num_cols])
        applied_imputation = impute_strat

    # 3. Outlier Mitigation (Compute bounds strictly on X_tr, clip both X_tr and X_te)
    outlier_strat = str(prep.get("outlier") or prep.get("outliers") or prep.get("outlier_mitigation") or "none").lower()
    applied_outlier = "none"
    outlier_bounds: dict[str, tuple[float, float]] = {}

    if outlier_strat not in ("none", "") and num_cols:
        if outlier_strat in ("clip_iqr", "iqr"):
            for col in num_cols:
                q1 = float(X_tr[col].quantile(0.25))
                q3 = float(X_tr[col].quantile(0.75))
                iqr = q3 - q1
                lb = q1 - 1.5 * iqr
                ub = q3 + 1.5 * iqr
                outlier_bounds[col] = (lb, ub)
                X_tr[col] = X_tr[col].clip(lower=lb, upper=ub)
                X_te[col] = X_te[col].clip(lower=lb, upper=ub)
            applied_outlier = "iqr"

        elif outlier_strat in ("zscore", "z_score"):
            for col in num_cols:
                mean = float(X_tr[col].mean())
                std = float(X_tr[col].std())
                if std > 1e-8:
                    lb = mean - 3.0 * std
                    ub = mean + 3.0 * std
                    outlier_bounds[col] = (lb, ub)
                    X_tr[col] = X_tr[col].clip(lower=lb, upper=ub)
                    X_te[col] = X_te[col].clip(lower=lb, upper=ub)
            applied_outlier = "zscore"

        elif outlier_strat == "winsorize":
            for col in num_cols:
                lb = float(X_tr[col].quantile(0.01))
                ub = float(X_tr[col].quantile(0.99))
                outlier_bounds[col] = (lb, ub)
                X_tr[col] = X_tr[col].clip(lower=lb, upper=ub)
                X_te[col] = X_te[col].clip(lower=lb, upper=ub)
            applied_outlier = "winsorize"

    # 4. Feature Scaling (Fit strictly on X_tr, transform X_tr and X_te)
    scaler_strat = str(prep.get("scaler") or prep.get("feature_scaling") or "none").lower()
    applied_scaler = "none"

    if scaler_strat == "standard" and num_cols:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
        X_te[num_cols] = scaler.transform(X_te[num_cols])
        applied_scaler = "standard"
    elif scaler_strat == "minmax" and num_cols:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
        X_te[num_cols] = scaler.transform(X_te[num_cols])
        applied_scaler = "minmax"
    elif scaler_strat == "robust" and num_cols:
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        X_tr[num_cols] = scaler.fit_transform(X_tr[num_cols])
        X_te[num_cols] = scaler.transform(X_te[num_cols])
        applied_scaler = "robust"

    summary = {
        "imputation": applied_imputation,
        "scaler": applied_scaler,
        "outlier": applied_outlier,
        "outlier_mitigation": {
            "method": applied_outlier,
            "bounds": {col: {"lower": b[0], "upper": b[1]} for col, b in outlier_bounds.items()},
        },
        "encoding": applied_encoding,
        "categorical_encoding": {
            "method": applied_encoding,
        },
        "transformed_features": list(X_tr.columns),
        "n_features_transformed": len(X_tr.columns),
    }

    return X_tr, X_te, summary
