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
        return self.bundle.tabular or self.bundle.market or self.bundle.short_rate

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
}


def get_canonical_context_specs() -> list[ExecutionContextSpec]:
    """Return all authoritative execution context specifications."""
    return list(_CANONICAL_CONTEXT_SPECS.values())


def resolve_context_spec(context_id: str) -> ExecutionContextSpec:
    """Resolve context specification by ID or raise ValueError."""
    if context_id not in _CANONICAL_CONTEXT_SPECS:
        raise ValueError(
            f"Unknown execution context '{context_id}'. "
            f"Available contexts: {list(_CANONICAL_CONTEXT_SPECS.keys())}"
        )
    return _CANONICAL_CONTEXT_SPECS[context_id]


def instantiate_context(
    context_id: str,
    seed: int | None = None,
    materiality: str = "TIER_1",
) -> ExecutionContextInstance:
    """Instantiate a real execution context instance exactly once."""
    spec = resolve_context_spec(context_id)
    actual_seed = seed if seed is not None else spec.seed

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
