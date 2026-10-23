"""Authoritative Schema Definitions for StART Scientific Certification."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class DatasetManifestItem:
    layer: Literal["golden_known_answer", "real_external"]
    domain: str
    dataset_id: str
    provider: str
    revision: str
    fingerprint: str
    rows: int
    features: int
    target: str
    license: str
    citation: str
    precertified: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentMatrixItem:
    experiment_id: str
    domain: str
    task_type: str
    layer: str
    dataset_id: str
    models: list[str]
    seeds: list[int]
    tuning_budget: int
    primary_metric: str
    secondary_metrics: list[str]
    xai_methods: list[str]
    sensitivity_modes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentRunRecord:
    run_id: str
    experiment_id: str
    domain: str
    task: str
    layer: str
    dataset: str
    provider: str
    dataset_revision: str
    dataset_fingerprint: str
    seed: int
    policy: Literal["deterministic", "gpt41"]
    model: str
    hyperparameters: dict[str, Any]
    preprocessing: dict[str, Any]
    split: dict[str, Any]
    tuning_strategy: str
    trial_budget: int
    primary_metric_name: str
    primary_metric_value: float
    secondary_metrics: dict[str, float]
    xai_results: dict[str, Any]
    sensitivity_results: dict[str, Any]
    runtime_seconds: float
    device: str
    gpt41_response_id: str | None = None
    gpt41_prompt_tokens: int | None = None
    gpt41_completion_tokens: int | None = None
    evidence_ids: list[str] = field(default_factory=list)
    artifact_hashes: dict[str, str] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChampionChallengerRecord:
    domain: str
    experiment_id: str
    layer: str
    dataset: str
    primary_metric: str
    champion_model: str
    champion_policy: str
    champion_mean: float
    champion_std: float
    champion_ci95: tuple[float, float]
    challenger_summaries: list[dict[str, Any]]
    gpt41_vs_deterministic: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class InvariantResult:
    invariant_id: str
    description: str
    domain: str
    status: Literal["PASS", "FAIL"]
    expected: Any
    observed: Any
    margin: float
    details: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class XAIResult:
    run_id: str
    model: str
    method: str
    feature_count: int
    features_ordered: list[str]
    finite_attributions: bool
    lineage_verified: bool
    status: Literal["PASS", "FAIL"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SensitivityResult:
    run_id: str
    model: str
    mode: Literal["one_at_a_time", "parallel_basket"]
    top_features: list[str]
    shock_grid: list[float]
    baseline_zero_delta: float
    finite_metrics: bool
    status: Literal["PASS", "FAIL"]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProviderTraceRecord:
    call_id: str
    timestamp: float
    provider: str
    model: str
    experiment_id: str
    prompt: str
    response_raw: str
    response_parsed: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    schema_valid: bool
    correction_call_required: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
