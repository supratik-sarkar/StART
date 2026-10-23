"""Authoritative Execution Context Specifications and Generators for StART v5.1.0.

Provides single-source-of-truth definitions for public demonstration contexts:
- institutional_credit_v1: 500 samples × 8 features, target="target", seed=42
- deep_learning_v1: 500 samples × 8 features, target="target", seed=17
- institutional_market_v1: 50 assets × 1,000 periods, target="N/A", seed=7

Strict Invariants:
1. Zero imports of start.web (CORE_RUNTIME_IMPORTS_START_WEB = 0).
2. CONTEXT_METADATA_EQUALS_RUNTIME_CONTEXT = PASS.
3. CONTEXT_TARGET_METADATA_EQUALS_RUNTIME = PASS.
4. Truthful context descriptions reflecting actual generated data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from start.review.architecture import (
    LLMReviewConfig,
    PredictiveTechnology,
    ReviewContextBundle,
    ReviewDomain,
    ReviewGroundingMode,
    ReviewLifecycle,
    ReviewMode,
)


@dataclass(frozen=True)
class ExecutionContextSpec:
    """Static metadata specification for an execution context catalog item."""

    id: str
    label: str
    kind: str  # "dataset" | "synthetic-world"
    description: str
    provenance: str
    shape: str
    target: str
    seed: int
    badges: list[str]
    configured_samples: int | None = None
    configured_features: int | None = None
    configured_assets: int | None = None
    configured_periods: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def context_id(self) -> str:
        return self.id

    @property
    def target_column(self) -> str:
        return self.target


@dataclass
class ExecutionContextInstance:
    """Runtime instance of a generated execution context."""

    spec_id: str
    actual_samples: int | None
    actual_features: int | None
    actual_assets: int | None
    actual_periods: int | None
    actual_target: str
    actual_seed: int
    bundle: ReviewContextBundle
    data_summary: dict[str, Any] = field(default_factory=dict)
    raw_data: Any = None

    @property
    def context(self) -> Any:
        return self.bundle.tabular or self.bundle.market or self.bundle.short_rate or self.bundle.recommender

    @property
    def actual_rows(self) -> int:
        return self.actual_samples if self.actual_samples is not None else (self.actual_periods or 0)

    def describe(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "actual_samples": self.actual_samples,
            "actual_features": self.actual_features,
            "actual_assets": self.actual_assets,
            "actual_periods": self.actual_periods,
            "actual_target": self.actual_target,
            "actual_seed": self.actual_seed,
            "data_summary": self.data_summary,
        }


# Canonical specifications
_CANONICAL_CONTEXT_SPECS: dict[str, ExecutionContextSpec] = {
    "institutional_credit_v1": ExecutionContextSpec(
        id="institutional_credit_v1",
        label="Synthetic Binary Classification Benchmark",
        kind="dataset",
        description=(
            "Seeded tabular binary classification benchmark for "
            "predictive risk model validation."
        ),
        provenance="Built-in deterministic synthetic generator",
        shape="500 × 8",
        target="target",
        seed=42,
        badges=["public-safe", "seeded", "binary", "benchmark"],
        configured_samples=500,
        configured_features=8,
    ),
    "deep_learning_v1": ExecutionContextSpec(
        id="deep_learning_v1",
        label="Synthetic Tabular Neural Latent Benchmark",
        kind="dataset",
        description=(
            "Seeded tabular neural network benchmark for deep learning performance, "
            "sensitivity, and calibration diagnostics."
        ),
        provenance="Built-in deterministic synthetic generator",
        shape="500 × 8",
        target="target",
        seed=17,
        badges=["public-safe", "deep-learning", "tabular", "neural"],
        configured_samples=500,
        configured_features=8,
    ),
    "institutional_market_v1": ExecutionContextSpec(
        id="institutional_market_v1",
        label="Synthetic Multi-Asset Market World",
        kind="synthetic-world",
        description=(
            "Seeded multi-asset scenario context for traded risk, VaR backtests, "
            "and portfolio optimization workflows."
        ),
        provenance="Built-in deterministic synthetic market generator",
        shape="50 assets × 1,000 observations",
        target="N/A",
        seed=7,
        badges=["public-safe", "quantitative", "var-backtest", "portfolio"],
        configured_assets=50,
        configured_periods=1000,
    ),
    "recommender_ratings_v1": ExecutionContextSpec(
        id="recommender_ratings_v1",
        label="Synthetic Explicit Ratings Benchmark (250 ratings)",
        kind="dataset",
        description=(
            "Seeded explicit rating matrix (25 users × 30 items) with timestamps for "
            "Matrix Factorization validation and rating fidelity."
        ),
        provenance="start.recommender.fixtures.DATASET_A",
        shape="25 users × 30 items (250 ratings)",
        target="rating",
        seed=42,
        badges=["recommender", "explicit-feedback", "matrix-factorization", "benchmark"],
        configured_samples=250,
        configured_features=3,
    ),
    "recommender_implicit_v1": ExecutionContextSpec(
        id="recommender_implicit_v1",
        label="Synthetic Implicit Interactions Benchmark (350 events)",
        kind="dataset",
        description=(
            "Seeded implicit interaction events (30 users × 40 items) for "
            "Neural Collaborative Filtering validation and Top-K ranking."
        ),
        provenance="start.recommender.fixtures.DATASET_B",
        shape="30 users × 40 items (350 interactions)",
        target="weight",
        seed=42,
        badges=["recommender", "implicit-feedback", "neural-cf", "benchmark"],
        configured_samples=350,
        configured_features=3,
    ),
    "recommender_contextual_v1": ExecutionContextSpec(
        id="recommender_contextual_v1",
        label="Synthetic Contextual Recommendations Benchmark (200 interactions)",
        kind="dataset",
        description=(
            "Seeded contextual interaction dataset (20 users × 25 items with user/item/context side features) "
            "for Factorization Machines and cold-start diagnostics."
        ),
        provenance="start.recommender.fixtures.DATASET_C",
        shape="20 users × 25 items (200 interactions)",
        target="label",
        seed=42,
        badges=["recommender", "contextual", "factorization-machine", "benchmark"],
        configured_samples=200,
        configured_features=8,
    ),
    "recommender_ffm_v1": ExecutionContextSpec(
        id="recommender_ffm_v1",
        label="Synthetic Field-Aware Contextual Recommendation Benchmark",
        kind="dataset",
        description=(
            "Seeded field-aware contextual interaction dataset for "
            "Field-Aware Factorization Machines (FFM) evaluation."
        ),
        provenance="start.recommender.fixtures.DATASET_C",
        shape="20 users × 25 items (200 interactions)",
        target="label",
        seed=42,
        badges=["recommender", "field-aware", "ffm", "benchmark"],
        configured_samples=200,
        configured_features=8,
    ),
    "synthetic_aml_imbalanced": ExecutionContextSpec(
        id="synthetic_aml_imbalanced",
        label="Synthetic AML & Transaction Monitoring Imbalance Benchmark",
        kind="dataset",
        description=(
            "Seeded transaction monitoring benchmark with 5.5% minority fraud prevalence "
            "for class imbalance and AML anomaly modeling."
        ),
        provenance="start.data.synthetic.generate_synthetic_transactions",
        shape="1,000 × 12",
        target="is_fraud",
        seed=42,
        badges=["fraud", "aml", "imbalanced", "transaction-monitoring"],
        configured_samples=1000,
        configured_features=12,
    ),
}


def get_canonical_context_specs() -> list[ExecutionContextSpec]:
    """Return all authoritative execution context specifications."""
    return list(_CANONICAL_CONTEXT_SPECS.values())


def resolve_context_spec(context_id: str) -> ExecutionContextSpec:
    """Resolve context specification by ID or raise ValueError."""
    if context_id in _CANONICAL_CONTEXT_SPECS:
        return _CANONICAL_CONTEXT_SPECS[context_id]

    # Support live provider datasets (e.g. scikit-learn/adult-census-income, huggingface:..., uci:...)
    try:
        from start.data.providers.registry import get_provider_adapter

        provider = "huggingface"
        dataset_id = context_id
        if ":" in context_id:
            provider, dataset_id = context_id.split(":", 1)
        elif context_id in ("statlog_german_credit", "german_credit", "credit_approval", "default_of_credit_card_clients"):
            provider, dataset_id = "uci", context_id
        elif "/" in context_id:
            provider, dataset_id = "huggingface", context_id

        adapter = get_provider_adapter(provider)
        contract = adapter.get_contract(dataset_id)
        return ExecutionContextSpec(
            id=context_id,
            label=f"{contract.provider.upper()}: {contract.dataset_id}",
            kind="dataset",
            description=f"Live dataset streamed from {contract.provider} ({contract.dataset_id})",
            provenance=contract.source_uri,
            shape=f"{contract.row_count or 'streaming'} × {len(contract.schema)}",
            target=contract.target_column or "target",
            seed=42,
            badges=[contract.provider, "live", "tabular"],
            configured_samples=contract.row_count or 1000,
            configured_features=len(contract.schema) - 1 if contract.schema else 14,
        )
    except Exception as exc:
        raise ValueError(
            f"Unknown execution context '{context_id}'. "
            f"Available contexts: {list(_CANONICAL_CONTEXT_SPECS.keys())}"
        ) from exc


def instantiate_context(
    context_id: str,
    seed: int | None = None,
    materiality: str = "TIER_1",
) -> ExecutionContextInstance:
    """Instantiate a real execution context instance exactly once."""
    spec = resolve_context_spec(context_id)
    actual_seed = seed if seed is not None else spec.seed

    if context_id not in _CANONICAL_CONTEXT_SPECS:
        import numpy as np
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        from start.data.providers.registry import get_provider_adapter
        from start.registry import TestContext

        provider = "huggingface"
        did = context_id
        if ":" in context_id:
            provider, did = context_id.split(":", 1)
        elif context_id in ("statlog_german_credit", "german_credit", "credit_approval", "default_of_credit_card_clients"):
            provider, did = "uci", context_id
        elif "/" in context_id:
            provider, did = "huggingface", context_id

        adapter = get_provider_adapter(provider)
        chunks = []
        total_rows = 0
        for chunk in adapter.stream(did, max_rows=1000):
            chunks.append(chunk)
            total_rows += len(chunk)
            if total_rows >= 1000:
                break
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

        target_col = spec.target if spec.target in df.columns else (df.columns[-1] if not df.empty else "target")
        feature_cols = [c for c in df.columns if c != target_col]

        # Clean string / categorical target into binary integer
        if target_col in df.columns:
            if df[target_col].dtype == object or str(df[target_col].dtype) == "category" or not np.issubdtype(df[target_col].dtype, np.number):
                uniques = sorted(list(df[target_col].dropna().unique()))
                label_map = {val: idx for idx, val in enumerate(uniques)}
                df[target_col] = df[target_col].map(label_map).fillna(0).astype(int)

        train_df, test_df = train_test_split(df, test_size=0.25, random_state=actual_seed)
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        y_tr = train_df[target_col].to_numpy(dtype=int) if target_col in train_df.columns else np.zeros(len(train_df))
        num_cols = [c for c in feature_cols if train_df[c].dtype in ("int64", "float64", "int32", "float32")]
        clf = RandomForestClassifier(n_estimators=50, random_state=actual_seed)
        if num_cols:
            clf.fit(train_df[num_cols].fillna(0), y_tr)
            test_p = clf.predict_proba(test_df[num_cols].fillna(0))[:, 1]
            test_df["score"] = test_p
            test_df["prediction"] = (test_p >= 0.5).astype(int)

        tab_ctx = TestContext(
            train=train_df,
            test=test_df,
            target_column=target_col,
            model=clf,
            score_column="score" if "score" in test_df.columns else None,
            prediction_column="prediction" if "prediction" in test_df.columns else None,
            seed=actual_seed,
        )
        bundle = ReviewContextBundle(
            mode=ReviewMode.SINGLE_DOMAIN,
            domains=(ReviewDomain.PREDICTIVE,),
            technology=PredictiveTechnology.TRADITIONAL_ML,
            materiality=materiality,
            lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
            tabular=tab_ctx,
            llm_config=LLMReviewConfig(provider="none"),
            grounding_mode=ReviewGroundingMode.STRUCTURED,
        )
        return ExecutionContextInstance(
            spec_id=spec.id,
            actual_samples=len(df),
            actual_features=len(feature_cols),
            actual_assets=None,
            actual_periods=None,
            actual_target=target_col,
            actual_seed=actual_seed,
            bundle=bundle,
            data_summary={
                "n_samples": len(df),
                "n_features": len(feature_cols),
                "target": target_col,
                "train_samples": len(train_df),
                "test_samples": len(test_df),
                "provider": provider,
                "dataset_id": did,
            },
            raw_data=df,
        )

    if spec.id.startswith("recommender_"):
        from start.recommender.contracts import (
            RecommenderContext,
            RecommenderFeedbackMode,
            RecommenderTaskMode,
        )
        from start.recommender.data import profile_recommender_dataset
        from start.recommender.fixtures import DATASET_A, DATASET_B, DATASET_C

        if spec.id == "recommender_ratings_v1":
            data = list(DATASET_A)
            fb_mode = RecommenderFeedbackMode.EXPLICIT
            task_mode = RecommenderTaskMode.RATING_PREDICTION
        elif spec.id == "recommender_implicit_v1":
            data = list(DATASET_B)
            fb_mode = RecommenderFeedbackMode.IMPLICIT
            task_mode = RecommenderTaskMode.TOP_K_RANKING
        else:
            data = list(DATASET_C)
            fb_mode = RecommenderFeedbackMode.CONTEXTUAL
            task_mode = RecommenderTaskMode.BINARY_INTERACTION

        profile = profile_recommender_dataset(data)
        rec_ctx = RecommenderContext(
            dataset_profile=profile,
            data=data,
            task_mode=task_mode,
            feedback_mode=fb_mode,
            seed=actual_seed,
        )

        bundle = ReviewContextBundle(
            mode=ReviewMode.SINGLE_DOMAIN,
            domains=(ReviewDomain.PREDICTIVE,),
            technology=None,
            materiality=materiality,
            lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
            recommender=rec_ctx,
            llm_config=LLMReviewConfig(provider="none"),
            grounding_mode=ReviewGroundingMode.STRUCTURED,
        )

        return ExecutionContextInstance(
            spec_id=spec.id,
            actual_samples=profile.n_interactions,
            actual_features=spec.configured_features,
            actual_assets=None,
            actual_periods=None,
            actual_target=spec.target,
            actual_seed=actual_seed,
            bundle=bundle,
            data_summary={
                "n_users": profile.n_users,
                "n_items": profile.n_items,
                "n_interactions": profile.n_interactions,
                "sparsity": profile.sparsity,
                "density": round(1.0 - profile.sparsity, 6),
                "feedback_mode": fb_mode.value,
                "mean_ratings_per_user": round(profile.mean_ratings_per_user, 2),
                "mean_ratings_per_item": round(profile.mean_ratings_per_item, 2),
            },
            raw_data=data,
        )

    if spec.id == "institutional_market_v1":
        from start.data.synthetic_market import generate_market_world
        from start.registry.market_contexts import MarketContext, PortfolioSpec

        world = generate_market_world(
            n_assets=spec.configured_assets or 50,
            n_periods=spec.configured_periods or 1000,
            n_factors=5,
            periods_per_year=252,
            seed=actual_seed,
            include_short_rate=True,
            missing_rate=0.15,
        )
        renamed = {old: f"ASSET_{i + 1:03d}" for i, old in enumerate(world.returns.columns)}
        market_ctx = MarketContext(
            returns=world.returns.rename(columns=renamed),
            prices=world.prices.rename(columns=renamed),
            periods_per_year=world.periods_per_year,
            risk_free_rate=0.02,
            risk_free_frequency="annual",
            factor_returns=world.factor_returns,
            factor_exposures=world.factor_exposures.rename(index=renamed),
            pnl=world.pnl,
            hypothetical_pnl=world.hypothetical_pnl,
            var_series=world.var_series,
            var_confidence=world.var_confidence,
            portfolio=PortfolioSpec(
                weights=world.weights.rename(renamed),
                benchmark_weights=world.benchmark_weights.rename(renamed),
            ),
            seed=actual_seed,
        )
        bundle = ReviewContextBundle(
            mode=ReviewMode.SINGLE_DOMAIN,
            domains=(ReviewDomain.MARKET,),
            technology=None,
            materiality=materiality,
            lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
            market=market_ctx,
            short_rate=world.short_rate,
            llm_config=LLMReviewConfig(provider="none"),
            grounding_mode=ReviewGroundingMode.STRUCTURED,
        )
        return ExecutionContextInstance(
            spec_id=spec.id,
            actual_samples=None,
            actual_features=None,
            actual_assets=int(world.returns.shape[1]),
            actual_periods=int(world.returns.shape[0]),
            actual_target=spec.target,
            actual_seed=actual_seed,
            bundle=bundle,
            data_summary={
                "n_assets": int(world.returns.shape[1]),
                "n_periods": int(world.returns.shape[0]),
                "periods_per_year": 252,
            },
            raw_data=world,
        )

    if spec.id == "synthetic_aml_imbalanced":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split

        from start.data.synthetic import generate_synthetic_transactions
        from start.registry import TestContext

        df = generate_synthetic_transactions(n_rows=spec.configured_samples or 1000, prevalence=0.055, seed=actual_seed)
        feature_cols = [c for c in df.columns if c != "is_fraud"]
        train_df, test_df = train_test_split(df, test_size=0.25, random_state=actual_seed, stratify=df["is_fraud"])
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        clf = RandomForestClassifier(n_estimators=100, random_state=actual_seed, class_weight="balanced")
        clf.fit(train_df[feature_cols], train_df["is_fraud"].to_numpy())
        test_df["score"] = clf.predict_proba(test_df[feature_cols])[:, 1]
        test_df["prediction"] = (test_df["score"] >= 0.5).astype(int)

        tab_ctx = TestContext(
            train=train_df,
            test=test_df,
            target_column="is_fraud",
            model=clf,
            score_column="score",
            prediction_column="prediction",
            seed=actual_seed,
        )
        bundle = ReviewContextBundle(
            mode=ReviewMode.SINGLE_DOMAIN,
            domains=(ReviewDomain.PREDICTIVE,),
            technology=PredictiveTechnology.TRADITIONAL_ML,
            materiality=materiality,
            lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
            tabular=tab_ctx,
            llm_config=LLMReviewConfig(provider="none"),
            grounding_mode=ReviewGroundingMode.STRUCTURED,
        )
        return ExecutionContextInstance(
            spec_id=spec.id,
            actual_samples=len(df),
            actual_features=len(feature_cols),
            actual_assets=None,
            actual_periods=None,
            actual_target="is_fraud",
            actual_seed=actual_seed,
            bundle=bundle,
            data_summary={
                "n_samples": len(df),
                "n_features": len(feature_cols),
                "target": "is_fraud",
                "train_samples": len(train_df),
                "test_samples": len(test_df),
            },
            raw_data=df,
        )

    # Tabular contexts (institutional_credit_v1 and deep_learning_v1)
    from start.data.synthetic_dl import generate_dl_world
    from start.registry import TestContext

    n_samples = spec.configured_samples or 500
    n_features = spec.configured_features or 8

    dl_res = generate_dl_world(n_samples=n_samples, n_features=n_features, seed=actual_seed)
    train_df = dl_res["train_df"]
    test_df = dl_res["test_df"]
    model_obj = dl_res.get("model")

    tab_ctx = TestContext(
        train=train_df,
        test=test_df,
        target_column=spec.target,
        model=model_obj,
        score_column="score" if "score" in test_df.columns else None,
        prediction_column="prediction" if "prediction" in test_df.columns else None,
        seed=actual_seed,
    )

    is_dl = spec.id == "deep_learning_v1"
    tech = PredictiveTechnology.DEEP_LEARNING if is_dl else PredictiveTechnology.TRADITIONAL_ML

    bundle = ReviewContextBundle(
        mode=ReviewMode.SINGLE_DOMAIN,
        domains=(ReviewDomain.PREDICTIVE,),
        technology=tech,
        materiality=materiality,
        lifecycle=ReviewLifecycle.INITIAL_VALIDATION,
        tabular=tab_ctx,
        llm_config=LLMReviewConfig(provider="none"),
        grounding_mode=ReviewGroundingMode.STRUCTURED,
    )

    total_samples = len(train_df) + len(test_df) + len(dl_res.get("val_df", []))
    feature_cols = [c for c in train_df.columns if c not in (spec.target, "score", "prediction")]

    return ExecutionContextInstance(
        spec_id=spec.id,
        actual_samples=total_samples,
        actual_features=len(feature_cols),
        actual_assets=None,
        actual_periods=None,
        actual_target=spec.target,
        actual_seed=actual_seed,
        bundle=bundle,
        data_summary={
            "n_samples": total_samples,
            "n_features": len(feature_cols),
            "target": spec.target,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        },
        raw_data=dl_res,
    )
