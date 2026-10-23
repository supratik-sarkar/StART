"""Authoritative Scenario Catalog and Descriptive Profiling for StART v5.1.3.

Preserves strict architectural boundaries:
1. SCENARIO != EXECUTION CONTEXT.
   A Scenario is a selectable built-in dataset or world generator.
   An Execution Context is a canonical execution specification accepted by the runtime engine.
2. Every scenario is classified as:
   - CANONICAL_EXECUTION_CONTEXT: Maps 1:1 to an authoritative ExecutionContextSpec.
   - SUPPORTED_DATA_SCENARIO: Backed by a truthful generator in start.data that maps to a compatible context.
   - GENERATOR_ONLY: Data generator only; explicitly flagged so it is never presented as an executable review engine.
   - UNSUPPORTED: Prohibited from appearing in the catalog (count = 0).
3. Hard EDA Boundary:
   EDA computes bounded deterministic descriptive statistics from raw data only.
   EDA never creates runs, never executes registered tests, never writes evidence, and never trains models.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from start.runtime.contexts import instantiate_context


class ScenarioClassification(StrEnum):
    CANONICAL_EXECUTION_CONTEXT = "CANONICAL_EXECUTION_CONTEXT"
    SUPPORTED_DATA_SCENARIO = "SUPPORTED_DATA_SCENARIO"
    GENERATOR_ONLY = "GENERATOR_ONLY"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class ScenarioSpec:
    id: str
    label: str
    domain: str  # "predictive_ml" | "deep_learning" | "quantitative_finance" | "treasury"
    classification: ScenarioClassification
    compatible_context_id: str
    compatible_workflows: list[str]
    shape: str
    rows: int
    features: int
    target: str
    categories: list[str]
    registered_tests_count: int
    applicable_tests_count: int
    generator_identity: str
    seed: int
    description: str
    provenance_note: str

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["classification"] = self.classification.value
        return d


# --------------------------------------------------------------------------- #
# Canonical Scenario Catalog (Mechanically Inventory-Verified)
# --------------------------------------------------------------------------- #
_SCENARIOS: dict[str, ScenarioSpec] = {
    "institutional_credit_v1": ScenarioSpec(
        id="institutional_credit_v1",
        label="Synthetic Binary Classification Benchmark",
        domain="predictive_ml",
        classification=ScenarioClassification.CANONICAL_EXECUTION_CONTEXT,
        compatible_context_id="institutional_credit_v1",
        compatible_workflows=[
            "predictive_ml",
            "data_diagnostics",
            "model_diagnostics",
            "calibration",
            "robustness",
            "explainability",
            "hyperparameter_tuning",
        ],
        shape="500 × 8",
        rows=500,
        features=8,
        target="target",
        categories=["Calibration", "Robustness", "Explainability"],
        registered_tests_count=52,
        applicable_tests_count=52,
        generator_identity="start.runtime.contexts.instantiate_context('institutional_credit_v1')",
        seed=42,
        description="Seeded tabular binary classification benchmark for predictive risk model validation.",
        provenance_note="Canonical execution context. Generated deterministically via seed 42.",
    ),
    "synthetic_aml_imbalanced": ScenarioSpec(
        id="synthetic_aml_imbalanced",
        label="Synthetic AML / Fraud Transaction Monitoring",
        domain="predictive_ml",
        classification=ScenarioClassification.SUPPORTED_DATA_SCENARIO,
        compatible_context_id="institutional_credit_v1",
        compatible_workflows=["predictive_ml", "data_diagnostics", "robustness"],
        shape="1,000 × 25",
        rows=1000,
        features=25,
        target="is_fraud",
        categories=["Imbalanced Classification", "Prevalence (5.5%)", "Leakage Screening"],
        registered_tests_count=52,
        applicable_tests_count=27,
        generator_identity="start.data.synthetic.generate_synthetic_transactions",
        seed=42,
        description="Seeded financial transaction dataset with realistic AML patterns and 5.5% fraud prevalence.",
        provenance_note="Data scenario generated locally via generate_synthetic_transactions. Maps to institutional_credit_v1.",
    ),
    "synthetic_leakage_proxy": ScenarioSpec(
        id="synthetic_leakage_proxy",
        label="Target Leakage & Demographic Proxy Stress",
        domain="predictive_ml",
        classification=ScenarioClassification.SUPPORTED_DATA_SCENARIO,
        compatible_context_id="institutional_credit_v1",
        compatible_workflows=["predictive_ml", "data_diagnostics", "robustness"],
        shape="1,000 × 26",
        rows=1000,
        features=26,
        target="is_fraud",
        categories=["Target Leakage", "Proxy Clustering", "Stress Testing"],
        registered_tests_count=52,
        applicable_tests_count=27,
        generator_identity="start.data.synthetic.generate_synthetic_transactions(inject_leakage=True, inject_proxy=True)",
        seed=42,
        description="Injects post-event chargeback target leakage and demographic proxy clustering for stress testing.",
        provenance_note="Data scenario with intentional structural stress. Maps to institutional_credit_v1.",
    ),
    "deep_learning_v1": ScenarioSpec(
        id="deep_learning_v1",
        label="Synthetic Tabular Neural Latent Benchmark",
        domain="deep_learning",
        classification=ScenarioClassification.CANONICAL_EXECUTION_CONTEXT,
        compatible_context_id="deep_learning_v1",
        compatible_workflows=["deep_learning"],
        shape="500 × 8",
        rows=500,
        features=8,
        target="target",
        categories=["PyTorch MLP", "Neural Attribution", "Calibration"],
        registered_tests_count=7,
        applicable_tests_count=7,
        generator_identity="start.runtime.contexts.instantiate_context('deep_learning_v1')",
        seed=17,
        description="Seeded tabular neural network benchmark for deep learning performance, sensitivity, and calibration diagnostics.",
        provenance_note="Canonical execution context with real PyTorch MLP fixture. Seed 17.",
    ),
    "institutional_market_v1": ScenarioSpec(
        id="institutional_market_v1",
        label="Synthetic Multi-Asset Market World",
        domain="quantitative_finance",
        classification=ScenarioClassification.CANONICAL_EXECUTION_CONTEXT,
        compatible_context_id="institutional_market_v1",
        compatible_workflows=["quantitative_finance"],
        shape="50 assets × 1,000 observations",
        rows=1000,
        features=50,
        target="N/A",
        categories=["Traded Risk", "Factor Attribution", "VaR Backtest"],
        registered_tests_count=25,
        applicable_tests_count=25,
        generator_identity="start.runtime.contexts.instantiate_context('institutional_market_v1')",
        seed=7,
        description="Seeded multi-asset scenario context for traded risk, VaR backtests, and portfolio optimization workflows.",
        provenance_note="Canonical execution context with 50 assets and 5 factor return series. Seed 7.",
    ),
    "market_regime_shift": ScenarioSpec(
        id="market_regime_shift",
        label="Market Regime Shift & Volatility Shock",
        domain="quantitative_finance",
        classification=ScenarioClassification.SUPPORTED_DATA_SCENARIO,
        compatible_context_id="institutional_market_v1",
        compatible_workflows=["quantitative_finance"],
        shape="50 assets × 1,000 observations",
        rows=1000,
        features=50,
        target="N/A",
        categories=["Regime Shift", "Correlation Breakdown", "Volatility Clustering"],
        registered_tests_count=25,
        applicable_tests_count=25,
        generator_identity="start.data.synthetic_market.generate_market_world(adversarial='regime_shift')",
        seed=7,
        description="Deterministic structural break and correlation breakdown simulation for market stress testing.",
        provenance_note="Adversarial market generator mode. Maps to institutional_market_v1.",
    ),
    "treasury_short_rate_cir": ScenarioSpec(
        id="treasury_short_rate_cir",
        label="Treasury Short-Rate Diffusion (CIR Path)",
        domain="treasury",
        classification=ScenarioClassification.GENERATOR_ONLY,
        compatible_context_id="institutional_market_v1",
        compatible_workflows=["quantitative_finance"],
        shape="1,000 daily observations",
        rows=1000,
        features=1,
        target="short_rate",
        categories=["Short-Rate Series", "CIR SDE", "IRRBB Data"],
        registered_tests_count=2,
        applicable_tests_count=0,
        generator_identity="start.data.synthetic_market.generate_short_rate_path",
        seed=7,
        description="Seeded daily short-rate path with CIR mean-reversion and non-linear diffusion. Data generator only; maps to institutional_market_v1.",
        provenance_note="Generator-only data scenario. Carried in institutional_market_v1 bundle; no independent treasury workflow.",
    ),
}

_RECOMMENDER_SCENARIOS: dict[str, ScenarioSpec] = {
    "recommender_ratings_v1": ScenarioSpec(
        id="recommender_ratings_v1",
        label="Synthetic Explicit Ratings Benchmark",
        domain="recommender",
        classification=ScenarioClassification.CANONICAL_EXECUTION_CONTEXT,
        compatible_context_id="recommender_ratings_v1",
        compatible_workflows=["recommender_system"],
        shape="25 users × 30 items",
        rows=250,
        features=3,
        target="rating",
        categories=["Explicit Ratings", "Matrix Factorization", "Cold-Start"],
        registered_tests_count=7,
        applicable_tests_count=7,
        generator_identity="start.runtime.contexts.instantiate_context('recommender_ratings_v1')",
        seed=42,
        description="Seeded explicit rating matrix (25 users × 30 items) with timestamps for Matrix Factorization validation.",
        provenance_note="Canonical execution context. Generated deterministically from DATASET_A.",
    ),
    "recommender_implicit_v1": ScenarioSpec(
        id="recommender_implicit_v1",
        label="Synthetic Implicit Interactions Benchmark",
        domain="recommender",
        classification=ScenarioClassification.CANONICAL_EXECUTION_CONTEXT,
        compatible_context_id="recommender_implicit_v1",
        compatible_workflows=["recommender_system"],
        shape="30 users × 40 items",
        rows=350,
        features=3,
        target="weight",
        categories=["Implicit Feedback", "Neural CF", "Top-K Ranking"],
        registered_tests_count=7,
        applicable_tests_count=7,
        generator_identity="start.runtime.contexts.instantiate_context('recommender_implicit_v1')",
        seed=42,
        description="Seeded implicit interaction events (30 users × 40 items) for Neural Collaborative Filtering validation.",
        provenance_note="Canonical execution context. Generated deterministically from DATASET_B.",
    ),
    "recommender_contextual_v1": ScenarioSpec(
        id="recommender_contextual_v1",
        label="Synthetic Contextual Recommendations Benchmark",
        domain="recommender",
        classification=ScenarioClassification.CANONICAL_EXECUTION_CONTEXT,
        compatible_context_id="recommender_contextual_v1",
        compatible_workflows=["recommender_system"],
        shape="20 users × 25 items",
        rows=200,
        features=8,
        target="label",
        categories=["Contextual Side Features", "Factorization Machines", "Cold-Start Cohorts"],
        registered_tests_count=7,
        applicable_tests_count=7,
        generator_identity="start.runtime.contexts.instantiate_context('recommender_contextual_v1')",
        seed=42,
        description="Seeded contextual interaction dataset (20 users × 25 items with user/item/context features) for Factorization Machines.",
        provenance_note="Canonical execution context. Generated deterministically from DATASET_C.",
    ),
}


def list_scenarios(include_recommender: bool = False) -> list[ScenarioSpec]:
    """Return all truthful scenario specifications."""
    if include_recommender:
        return list(_SCENARIOS.values()) + list(_RECOMMENDER_SCENARIOS.values())
    return list(_SCENARIOS.values())


def get_scenario(scenario_id: str) -> ScenarioSpec:
    """Resolve scenario specification by ID."""
    all_scenarios = {**_SCENARIOS, **_RECOMMENDER_SCENARIOS}
    if scenario_id not in all_scenarios:
        raise KeyError(f"Unknown scenario '{scenario_id}'. Available: {list(all_scenarios.keys())}")
    return all_scenarios[scenario_id]


# --------------------------------------------------------------------------- #
# Hard EDA Boundary: Descriptive Profiler Only
# --------------------------------------------------------------------------- #
def compute_scenario_eda(scenario_id: str) -> dict[str, Any]:
    """Compute bounded deterministic descriptive statistics for a scenario.

    Strict Invariants:
    1. Zero StART runs created (EDA_CREATES_RUN = 0).
    2. Zero registered validation tests executed (EDA_EXECUTES_REGISTERED_TESTS = 0).
    3. Zero EvidenceRecords created or written (EDA_WRITES_EVIDENCE = 0).
    4. Zero models trained (EDA_TRAINS_MODEL = 0).
    """
    spec = get_scenario(scenario_id)

    if scenario_id.startswith("recommender_"):
        inst = instantiate_context(scenario_id)
        return _profile_recommender(spec, inst.raw_data)

    # 1. Dispatch to raw data generator for descriptive profiling
    if scenario_id == "synthetic_aml_imbalanced":
        from start.data.synthetic import generate_synthetic_transactions
        df = generate_synthetic_transactions(n_rows=1000, prevalence=0.055, seed=42)
        return _profile_tabular(spec, df, target_col="is_fraud")

    elif scenario_id == "synthetic_leakage_proxy":
        from start.data.synthetic import generate_synthetic_transactions
        df = generate_synthetic_transactions(n_rows=1000, prevalence=0.055, seed=42, inject_leakage=True, inject_proxy=True)
        return _profile_tabular(spec, df, target_col="is_fraud")

    elif scenario_id == "market_regime_shift":
        from start.data.synthetic_market import generate_market_world
        world = generate_market_world(n_assets=50, n_periods=1000, seed=7, regime_shift_at=500)
        return _profile_market_world(spec, world)

    elif scenario_id == "treasury_short_rate_cir":
        from start.data.synthetic_market import generate_short_rate_path
        rates_series, _ = generate_short_rate_path(n_periods=1000, seed=7, gamma=0.5, r0=0.03, kappa=0.15, theta=0.035, sigma=0.06)
        return _profile_short_rate(spec, rates_series.to_numpy())

    elif scenario_id == "institutional_market_v1":
        inst = instantiate_context("institutional_market_v1")
        world = inst.raw_data
        return _profile_market_world(spec, world)

    elif scenario_id == "deep_learning_v1":
        inst = instantiate_context("deep_learning_v1")
        raw = inst.raw_data
        return _profile_deep_learning_fixture(spec, raw)

    else:  # institutional_credit_v1 default
        inst = instantiate_context("institutional_credit_v1")
        train_df = inst.raw_data["train_df"]
        test_df = inst.raw_data["test_df"]
        combined = pd.concat([train_df, test_df], ignore_index=True)
        return _profile_tabular(spec, combined, target_col="target")


def _provenance_block(spec: ScenarioSpec, raw_summary: str) -> dict[str, Any]:
    fingerprint = hashlib.sha256(raw_summary.encode("utf-8")).hexdigest()[:16]
    return {
        "scenario_id": spec.id,
        "context_id": spec.compatible_context_id,
        "classification": spec.classification.value,
        "generator_identity": spec.generator_identity,
        "seed": spec.seed,
        "input_shape": spec.shape,
        "profiling_operation": "deterministic_descriptive_profiling",
        "sha256_fingerprint": f"SHA256:{fingerprint}",
        "provenance_note": spec.provenance_note,
    }


def _profile_tabular(spec: ScenarioSpec, df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """Compute strictly descriptive statistics for a tabular dataset."""
    n_rows, n_cols = df.shape
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    
    # 1. Overview & Schema
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    memory_kb = round(df.memory_usage(deep=True).sum() / 1024, 1)

    schema = []
    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        schema.append({
            "name": col,
            "dtype": str(s.dtype),
            "null_count": null_count,
            "null_pct": round(null_count / max(n_rows, 1) * 100, 2),
            "distinct_count": int(s.nunique(dropna=True)),
            "is_target": col == target_col,
        })

    # 2. Target Distribution
    target_dist: dict[str, Any] = {}
    if target_col in df.columns:
        counts = df[target_col].value_counts(dropna=False).to_dict()
        str_counts = {str(k): int(v) for k, v in counts.items()}
        pos_count = int(counts.get(1, counts.get("1", 0)))
        pos_ratio = round(pos_count / max(n_rows, 1), 4)
        target_dist = {
            "target_column": target_col,
            "counts": str_counts,
            "positive_ratio": pos_ratio,
            "prevalence_pct": round(pos_ratio * 100, 2),
            "total_samples": n_rows,
            "is_binary": len(str_counts) == 2,
        }

    # 3. Numeric Feature Moments
    feature_moments = []
    for col in numeric_cols[:16]:  # Bounded to primary features
        vals = df[col].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        q25, median_val, q75 = [float(v) for v in np.percentile(vals, [25, 50, 75])]
        skew_val = float(stats.skew(vals)) if len(vals) > 2 else 0.0
        
        # Simple IQR outlier count
        iqr = q75 - q25
        lower = q25 - 1.5 * iqr
        upper = q75 + 1.5 * iqr
        outliers = int(np.sum((vals < lower) | (vals > upper)))

        feature_moments.append({
            "feature": col,
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "min": round(float(np.min(vals)), 4),
            "q25": round(q25, 4),
            "median": round(median_val, 4),
            "q75": round(q75, 4),
            "max": round(float(np.max(vals)), 4),
            "skewness": round(skew_val, 4),
            "outlier_count": outliers,
            "outlier_pct": round(outliers / len(vals) * 100, 2),
        })

    # 4. Correlation Matrix (bounded to top 8 features)
    corr_cols = numeric_cols[:8]
    corr_matrix: list[list[float]] = []
    if len(corr_cols) > 1:
        c_df = df[corr_cols].corr().fillna(0.0)
        corr_matrix = [[round(float(c_df.iloc[i, j]), 3) for j in range(len(corr_cols))] for i in range(len(corr_cols))]

    # 5. Feature-Target Relationships
    feature_target_corrs = []
    if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        for col in numeric_cols[:12]:
            clean = df[[col, target_col]].dropna()
            if len(clean) > 2 and clean[col].std() > 0:
                r = float(clean[col].corr(clean[target_col]))
                feature_target_corrs.append({
                    "feature": col,
                    "pearson_r": round(r, 4),
                    "abs_r": round(abs(r), 4),
                })
        feature_target_corrs.sort(key=lambda x: x["abs_r"], reverse=True)

    summary_str = f"{spec.id}:{n_rows}:{n_cols}:{missing_cells}:{duplicate_rows}"
    return {
        "domain": "tabular",
        "provenance": _provenance_block(spec, summary_str),
        "overview": {
            "rows": n_rows,
            "columns": n_cols,
            "feature_count": len(numeric_cols),
            "target": target_col,
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
            "memory_kb": memory_kb,
        },
        "schema": schema,
        "target_distribution": target_dist,
        "feature_moments": feature_moments,
        "correlation": {
            "columns": corr_cols,
            "matrix": corr_matrix,
        },
        "feature_target_correlations": feature_target_corrs,
    }


def _profile_deep_learning_fixture(spec: ScenarioSpec, raw_fixture: dict[str, Any]) -> dict[str, Any]:
    """Profile deep learning fixture with strict separation of Model Context vs Data Profile."""
    train_df = raw_fixture["train_df"]
    test_df = raw_fixture["test_df"]
    val_df = raw_fixture.get("val_df", pd.DataFrame())
    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # 1. Data Profile (pure descriptive statistics of input tensors)
    tab_profile = _profile_tabular(spec, combined, target_col="target")

    # 2. Model / Fixture Context (explicitly segregated per Correction 4)
    model = raw_fixture.get("model")
    model_context = {
        "architecture": "PyTorch Multi-Layer Perceptron (TabularDLClassifier)",
        "task": getattr(model, "task", "binary_classification"),
        "family": getattr(model, "family", "mlp"),
        "hidden_dims": list(getattr(model, "hidden_dims", [64, 32])),
        "epochs": getattr(model, "epochs", 8),
        "batch_size": getattr(model, "batch_size", 64),
        "learning_rate": getattr(model, "learning_rate", 0.005),
        "device": str(getattr(model, "device", "cpu")),
        "splits": {
            "train_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
            "total_samples": len(combined),
        },
        "tensor_shapes": {
            "train_features": [len(train_df), tab_profile["overview"]["feature_count"]],
            "train_labels": [len(train_df)],
            "val_features": [len(val_df), tab_profile["overview"]["feature_count"]],
            "test_features": [len(test_df), tab_profile["overview"]["feature_count"]],
        },
        "normalization": "Standard Gaussian normal (mean=0, variance=1) with controlled 5% missingness in feat_04",
    }

    summary_str = f"dl:{len(combined)}:{model_context['hidden_dims']}:{spec.seed}"
    return {
        "domain": "deep_learning",
        "provenance": _provenance_block(spec, summary_str),
        "data_profile": tab_profile,
        "model_fixture_context": model_context,
    }


def _profile_market_world(spec: ScenarioSpec, world: Any) -> dict[str, Any]:
    """Profile multi-asset market world descriptive statistics."""
    returns: pd.DataFrame = world.returns
    prices: pd.DataFrame = world.prices
    factor_returns: pd.DataFrame = world.factor_returns
    factor_exposures: pd.DataFrame = world.factor_exposures
    weights: pd.Series = world.weights
    benchmark_weights: pd.Series = getattr(world, "benchmark_weights", pd.Series())

    n_periods, n_assets = returns.shape
    n_factors = factor_returns.shape[1] if factor_returns is not None else 0

    # Asset weights summary
    top_weights = []
    sorted_weights = weights.sort_values(ascending=False)
    for asset, w in sorted_weights.head(10).items():
        top_weights.append({
            "asset": str(asset),
            "portfolio_weight": round(float(w), 4),
            "benchmark_weight": round(float(benchmark_weights.get(asset, 0.0)), 4) if not benchmark_weights.empty else 0.0,
        })

    # Portfolio descriptive return statistics
    port_returns = returns.dot(weights)
    ann_factor = np.sqrt(252)
    daily_vol = float(port_returns.std())
    ann_vol = float(daily_vol * ann_factor)

    # Factor overview
    factors_summary = []
    if factor_returns is not None:
        for fcol in factor_returns.columns:
            fvals = factor_returns[fcol].dropna().to_numpy(dtype=float)
            factors_summary.append({
                "factor": str(fcol),
                "mean_daily": round(float(np.mean(fvals)), 6),
                "annualized_vol": round(float(np.std(fvals) * ann_factor), 4),
                "min": round(float(np.min(fvals)), 4),
                "max": round(float(np.max(fvals)), 4),
            })

    # Covariance correlation summary
    corr = returns.corr().to_numpy()
    upper_tri = corr[np.triu_indices_from(corr, k=1)]
    mean_corr = round(float(np.nanmean(upper_tri)), 3)

    summary_str = f"market:{n_assets}:{n_periods}:{mean_corr}:{spec.seed}"
    return {
        "domain": "market_risk",
        "provenance": _provenance_block(spec, summary_str),
        "overview": {
            "assets": n_assets,
            "periods": n_periods,
            "frequency": "daily (252 periods/year)",
            "factor_count": n_factors,
            "portfolio_weight_sum": round(float(weights.sum()), 4),
            "max_single_holding_pct": round(float(weights.max() * 100), 2),
            "annualized_volatility": round(ann_vol, 4),
            "mean_asset_correlation": mean_corr,
        },
        "portfolio_composition": top_weights,
        "factor_coverage": factors_summary,
        "risk_summary": {
            "daily_mean_return": round(float(port_returns.mean()), 6),
            "daily_volatility": round(daily_vol, 4),
            "annualized_volatility": round(ann_vol, 4),
            "min_single_day_return": round(float(port_returns.min()), 4),
            "max_single_day_return": round(float(port_returns.max()), 4),
            "var_confidence_level": getattr(world, "var_confidence", 0.99),
        },
    }


def _profile_short_rate(spec: ScenarioSpec, rates: np.ndarray) -> dict[str, Any]:
    """Profile treasury short-rate path descriptive statistics."""
    n_obs = len(rates)
    mean_rate = float(np.mean(rates))
    std_rate = float(np.std(rates))
    min_rate = float(np.min(rates))
    max_rate = float(np.max(rates))

    # Downsample path to 50 points for lightweight interactive SVG chart preview
    step = max(1, n_obs // 50)
    sampled_indices = list(range(0, n_obs, step))
    sampled_path = [{"step": i, "rate": round(float(rates[i]), 5)} for i in sampled_indices]

    summary_str = f"treasury:{n_obs}:{mean_rate}:{spec.seed}"
    return {
        "domain": "treasury",
        "provenance": _provenance_block(spec, summary_str),
        "overview": {
            "observations": n_obs,
            "model": "Cox-Ingersoll-Ross (CIR) Stochastic Differential Equation",
            "initial_rate_r0": round(float(rates[0]), 4),
            "terminal_rate": round(float(rates[-1]), 4),
            "mean_rate": round(mean_rate, 4),
            "volatility": round(std_rate, 4),
            "min_rate": round(min_rate, 4),
            "max_rate": round(max_rate, 4),
            "non_negative": bool(np.all(rates >= 0.0)),
        },
        "path_sample": sampled_path,
        "classification_notice": (
            "GENERATOR_ONLY: This short-rate series is provided for exploratory data inspection only. "
            "It maps to canonical context institutional_market_v1 which carries short_rate; "
            "it is not an executable independent review workflow."
        ),
    }


def _profile_recommender(spec: ScenarioSpec, data: list[Any]) -> dict[str, Any]:
    """Compute strictly descriptive statistics for a recommender interaction dataset."""
    from collections import Counter

    from start.recommender.data import profile_recommender_dataset

    profile = profile_recommender_dataset(data)
    user_counts = Counter(getattr(d, "user_id", "") for d in data)
    item_counts = Counter(getattr(d, "item_id", "") for d in data)

    user_lens = list(user_counts.values()) or [0]
    item_lens = list(item_counts.values()) or [0]

    values = []
    for d in data:
        if hasattr(d, "rating"):
            values.append(float(d.rating))
        elif hasattr(d, "weight"):
            values.append(float(d.weight))
        elif hasattr(d, "label"):
            values.append(float(d.label))

    val_arr = np.array(values, dtype=float) if values else np.array([0.0])

    top_items = [{"item_id": it, "interaction_count": cnt} for it, cnt in item_counts.most_common(5)]
    top_users = [{"user_id": u, "interaction_count": cnt} for u, cnt in user_counts.most_common(5)]

    summary_str = f"recommender:{profile.n_users}:{profile.n_items}:{profile.n_interactions}:{profile.sparsity}:{spec.seed}"

    return {
        "domain": "recommender",
        "provenance": _provenance_block(spec, summary_str),
        "overview": {
            "users": profile.n_users,
            "items": profile.n_items,
            "interactions": profile.n_interactions,
            "sparsity": round(profile.sparsity, 6),
            "density": round(1.0 - profile.sparsity, 6),
            "mean_interactions_per_user": round(float(np.mean(user_lens)), 2),
            "median_interactions_per_user": round(float(np.median(user_lens)), 2),
            "max_interactions_per_user": int(np.max(user_lens)),
            "min_interactions_per_user": int(np.min(user_lens)),
            "mean_interactions_per_item": round(float(np.mean(item_lens)), 2),
            "median_interactions_per_item": round(float(np.median(item_lens)), 2),
            "target_mean": round(float(np.mean(val_arr)), 4),
            "target_std": round(float(np.std(val_arr)), 4),
            "target_min": round(float(np.min(val_arr)), 4),
            "target_max": round(float(np.max(val_arr)), 4),
        },
        "top_items": top_items,
        "top_users": top_users,
        "schema": [
            {"name": "user_id", "dtype": "string", "distinct_count": profile.n_users, "is_target": False},
            {"name": "item_id", "dtype": "string", "distinct_count": profile.n_items, "is_target": False},
            {"name": spec.target, "dtype": "float", "distinct_count": len(np.unique(val_arr)), "is_target": True},
        ],
    }
