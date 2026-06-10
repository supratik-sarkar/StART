"""Configuration layer.

Configs are YAML files validated into typed Pydantic models. Environment
variables (prefix ``START_``) override file values for secrets and runtime
toggles. Policy files (thresholds, allowed paths, allowed model categories)
are separate, versioned YAML whose content hash is stamped into every
evidence record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from start.core.hashing import hash_obj


class ComputeConfig(BaseModel):
    mode: Literal["auto", "cpu", "gpu", "databricks"] = "auto"
    device: Literal["auto", "cuda", "mps", "cpu"] = "auto"
    distributed_backend: Literal["none", "databricks_spark", "ray", "dask"] = "none"
    batch_size: int = 1024
    max_memory_gb: float | None = None


class LLMConfig(BaseModel):
    provider: Literal[
        "none",
        "openai",
        "anthropic",
        "grok",
        "huggingface",
        "hf_local",
        "enterprise_llm_gateway",
    ] = "none"
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024


class DataConfig(BaseModel):
    provider: Literal["csv_parquet", "snowflake_placeholder"] = "csv_parquet"
    path: str = "examples/data"
    dataset_id: str = "dataset-local"


class ExperimentConfig(BaseModel):
    provider: Literal["local", "mlflow"] = "local"
    tracking_uri: str | None = None
    experiment_name: str = "start-runs"


class OutputConfig(BaseModel):
    root: str = "start_output"
    ledger_file: str = "ledger.jsonl"
    evidence_store: str = "evidence_store"
    reports_dir: str = "reports"


class ModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str = "model-unnamed"
    task_type: str = "binary_classification"
    materiality: str = "medium"
    target_column: str | None = None
    prediction_column: str | None = None
    score_column: str | None = None


class TestFamiliesConfig(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: ["preprocessing", "supervised", "xai"]
    )
    disabled: list[str] = Field(default_factory=list)
    overrides: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Per-test parameter overrides keyed by test_id."
    )


class PolicyConfig(BaseModel):
    """Loaded from a separate, versioned policy YAML."""

    name: str = "default"
    version: str = "0.1.0"
    allowed_task_types: list[str] = Field(default_factory=list)
    allowed_data_roots: list[str] = Field(default_factory=list)
    max_materiality_without_review: str = "high"
    require_citations: bool = True
    thresholds: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def content_hash(self) -> str:
        return hash_obj(self.model_dump())


class StartConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="START_", env_nested_delimiter="__", extra="ignore"
    )

    project_name: str = "start-project"
    seed: int = 42
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    test_families: TestFamiliesConfig = Field(default_factory=TestFamiliesConfig)
    policy_file: str = "configs/policy/default_policy.yaml"


def load_config(path: str | Path | None = None) -> StartConfig:
    """Load YAML config (if given) merged with environment overrides."""
    if path is None:
        return StartConfig()
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return StartConfig(**raw)


def load_policy(path: str | Path) -> PolicyConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return PolicyConfig(**raw)
