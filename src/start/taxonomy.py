"""Dataset taxonomy: deterministic profiling and type-aware mappings.

Profiles a dataset's structure (rows, feature types, time structure, entity
structure, target type) and classifies it into a dataset type. Domain types
that cannot be reliably inferred from columns alone — limit order books, tick
event streams, volatility surfaces — are honored when DECLARED via
`data.dataset_type` in the config; inference never pretends to detect them.

The mappings below drive ModelRecommendationAgent and ValidationPlannerAgent:
recommendations depend on dataset type, and every entry is honestly labeled
as available-now or roadmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATASET_TYPES = (
    "tabular",
    "time_series",
    "panel_time_series",
    "limit_order_book",
    "tick_events",
    "volatility_surface",
    "text_alternative",
)

DECLARED_ONLY_TYPES = {"limit_order_book", "tick_events", "volatility_surface"}


@dataclass
class DatasetProfile:
    n_rows: int = 0
    n_features: int = 0
    n_numeric: int = 0
    n_text: int = 0
    has_timestamp: bool = False
    has_entity: bool = False
    n_entities: int | None = None
    target_type: str = "unknown"  # binary | multiclass | continuous | unknown
    dataset_type: str = "tabular"
    declared: bool = False
    notes: list[str] = field(default_factory=list)

    def describe(self) -> str:
        bits = [
            f"{self.n_rows} rows",
            f"{self.n_features} features ({self.n_numeric} numeric, {self.n_text} text)",
            f"target: {self.target_type}",
        ]
        if self.has_timestamp:
            bits.append("time-indexed")
        if self.has_entity:
            bits.append(f"{self.n_entities} entities" if self.n_entities else "entity-keyed")
        origin = "declared" if self.declared else "inferred"
        return f"{self.dataset_type} ({origin}): " + ", ".join(bits)


def _infer_target_type(series: pd.Series) -> str:
    clean = series.dropna()
    if clean.empty:
        return "unknown"
    n_unique = clean.nunique()
    if n_unique == 2:
        return "binary"
    if n_unique <= 20 and (clean.dtype == object or np.issubdtype(clean.dtype, np.integer)):
        return "multiclass"
    if np.issubdtype(clean.dtype, np.number):
        return "continuous"
    return "multiclass"


def profile_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
    timestamp_column: str | None = None,
    entity_id_column: str | None = None,
    declared_type: str = "auto",
) -> DatasetProfile:
    feature_cols = [c for c in df.columns if c != target_column]
    numeric = df[feature_cols].select_dtypes(include=[np.number]).columns
    text = [
        c
        for c in df[feature_cols].select_dtypes(include=["object", "string", "category"]).columns
        if df[c].dropna().astype(str).str.len().mean() > 30
    ]
    has_ts = bool(timestamp_column and timestamp_column in df.columns) or any(
        np.issubdtype(df[c].dtype, np.datetime64) for c in feature_cols
    )
    has_entity = bool(entity_id_column and entity_id_column in df.columns)
    profile = DatasetProfile(
        n_rows=len(df),
        n_features=len(feature_cols),
        n_numeric=len(numeric),
        n_text=len(text),
        has_timestamp=has_ts,
        has_entity=has_entity,
        n_entities=int(df[entity_id_column].nunique()) if has_entity else None,
        target_type=_infer_target_type(df[target_column]) if target_column in df.columns else "unknown"
        if target_column
        else "unknown",
    )

    if declared_type not in ("auto", None, ""):
        if declared_type not in DATASET_TYPES:
            raise ValueError(f"Unknown dataset_type '{declared_type}'. Known: {DATASET_TYPES}")
        profile.dataset_type = declared_type
        profile.declared = True
        return profile

    # Inference (never claims the declared-only domain types)
    if profile.n_text > 0 and profile.n_text >= profile.n_numeric:
        profile.dataset_type = "text_alternative"
    elif has_ts and has_entity:
        profile.dataset_type = "panel_time_series"
    elif has_ts:
        profile.dataset_type = "time_series"
    else:
        profile.dataset_type = "tabular"
    if declared_type == "auto":
        profile.notes.append(
            "Domain types (limit_order_book, tick_events, volatility_surface) are never "
            "auto-inferred; declare data.dataset_type in the config to use them."
        )
    return profile


# --------------------------------------------------------------------------- #
# Type-aware model recommendations (model, rationale, implemented-now?)
# --------------------------------------------------------------------------- #
MODEL_RECOMMENDATIONS: dict[str, list[tuple[str, str, bool]]] = {
    "tabular": [
        ("random_forest", "strong tabular baseline; works out of the box", True),
        ("xgboost", "gradient boosting often wins on tabular data ([tree-models] extra)", True),
        ("lightgbm", "fast gradient boosting for larger tabular data ([tree-models] extra)", True),
        ("mlp", "tabular neural baseline (roadmap, [torch] extra)", False),
    ],
    "time_series": [
        ("temporal_cnn", "convolutional sequence model; robust default (roadmap)", False),
        ("lstm", "recurrent baseline for sequential dependence (roadmap)", False),
        ("gru", "lighter recurrent alternative to LSTM (roadmap)", False),
        ("transformer", "attention over long horizons (roadmap)", False),
    ],
    "panel_time_series": [
        ("tft", "Temporal Fusion Transformer for multi-entity panels (roadmap)", False),
        ("temporal_cnn", "per-entity temporal convolutions (roadmap)", False),
        ("lstm", "recurrent baseline across entities (roadmap)", False),
        ("gru", "lighter recurrent alternative (roadmap)", False),
    ],
    "limit_order_book": [
        ("deeplob", "CNN architecture designed for order-book tensors (roadmap)", False),
        ("cnn", "convolutional baseline over price-level grids (roadmap)", False),
        ("transformer", "attention over book states (roadmap)", False),
        ("temporal_transformer", "time-aware attention for book dynamics (roadmap)", False),
    ],
    "tick_events": [
        ("signature_network", "path-signature features for irregular event streams (roadmap)", False),
        ("neural_point_process", "models event timing and intensity directly (roadmap)", False),
        ("temporal_attention", "attention over irregular timestamps (roadmap)", False),
        ("transformer", "general sequence baseline (roadmap)", False),
    ],
    "volatility_surface": [
        ("cnn", "treats the vol grid as an image (roadmap)", False),
        ("neural_pde", "embeds no-arbitrage dynamics (roadmap)", False),
        ("gnn", "graph structure across strikes/tenors (roadmap)", False),
        ("transformer", "attention over surface patches (roadmap)", False),
    ],
    "text_alternative": [
        ("finbert_variant", "domain-tuned encoder for financial text (roadmap)", False),
        ("multimodal_transformer", "joint text + tabular signals (roadmap)", False),
        ("llm_rag", "retrieval-augmented LLM with citation discipline (roadmap)", False),
    ],
}

# --------------------------------------------------------------------------- #
# Type/model-aware validation plans: (check, test_id or roadmap, available?)
# --------------------------------------------------------------------------- #
_TABULAR_CORE: list[tuple[str, str, bool]] = [
    ("data quality screens", "preprocessing.*", True),
    ("train/test/OOS metrics + overfit gap", "supervised.cohort_metrics_comparison", True),
    ("top-decile lift", "supervised.top_decile_lift", True),
    ("calibration", "supervised.calibration", True),
    ("global importance (SHAP/permutation)", "xai.global_importance", True),
    ("top-feature shock sensitivity", "xai.feature_sensitivity", True),
]

VALIDATION_PLANS: dict[str, list[tuple[str, str, bool]]] = {
    "tabular": _TABULAR_CORE,
    "time_series": [
        *_TABULAR_CORE[:1],
        ("temporal leakage screen", "roadmap:temporal_leakage", False),
        ("temporal drift across windows", "roadmap:temporal_drift", False),
        ("regime sensitivity", "roadmap:regime_sensitivity", False),
    ],
    "panel_time_series": [
        *_TABULAR_CORE[:1],
        ("cross-sectional stability", "roadmap:cross_sectional_stability", False),
        ("regime sensitivity", "roadmap:regime_sensitivity", False),
        ("temporal drift", "roadmap:temporal_drift", False),
    ],
    "limit_order_book": [
        ("latency analysis", "roadmap:latency_analysis", False),
        ("microstructure stability", "roadmap:microstructure_stability", False),
        ("regime robustness", "roadmap:regime_robustness", False),
        ("temporal leakage screen", "roadmap:temporal_leakage", False),
    ],
    "tick_events": [
        ("temporal leakage screen", "roadmap:temporal_leakage", False),
        ("timestamp integrity", "roadmap:timestamp_integrity", False),
        ("event ordering validation", "roadmap:event_ordering", False),
    ],
    "volatility_surface": [
        ("arbitrage constraints", "roadmap:arbitrage_constraints", False),
        ("surface smoothness", "roadmap:surface_smoothness", False),
        ("stress sensitivity", "roadmap:stress_sensitivity", False),
    ],
    "text_alternative": [
        ("citation coverage", "genai.citation_coverage", True),
        ("hallucination risk", "roadmap:hallucination_risk", False),
        ("retrieval quality", "roadmap:retrieval_quality", False),
        ("temporal relevance", "roadmap:temporal_relevance", False),
    ],
}

# Model-family additions layered on top of the dataset-type plan.
MODEL_FAMILY_PLANS: dict[str, list[tuple[str, str, bool]]] = {
    "tree": [
        ("SHAP global/local attribution", "xai.global_importance", True),
        ("permutation importance stability", "xai.importance_stability", True),
    ],
    "deep_learning": [
        ("integrated gradients attribution", "roadmap:integrated_gradients", False),
        ("attention stability", "roadmap:attention_stability", False),
        ("counterfactual testing", "roadmap:counterfactual_testing", False),
        ("input drift monitoring", "preprocessing.feature_drift", True),
    ],
}
