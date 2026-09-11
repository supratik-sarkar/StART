"""Authoritative Data Contracts and Telemetry for StART Live Data Providers.

Strict Invariants:
1. STREAM BY DEFAULT: Large and remote datasets stream in batches.
2. CACHE OPTIONALLY: Datasets can be cached locally in Arrow format.
3. FINGERPRINT ALWAYS: Every consumed dataset calculates a cryptographic SHA-256 fingerprint.
4. CREDENTIALS BACKEND ONLY: No API tokens or secrets are ever exposed in telemetry or client envelopes.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import pandas as pd


@dataclass
class DatasetContract:
    """Canonical specification and metadata for an ingested dataset."""

    provider: str                      # "huggingface" | "openml" | "uci" | "kaggle" | "local_csv" | "local_parquet"
    dataset_id: str                    # Unique provider identifier (e.g. "scikit-learn/adult-census-income")
    revision: str                      # Git commit, version tag, or file timestamp
    source_uri: str                    # Official endpoint or file path
    license: str                       # e.g. "MIT", "CC-BY-4.0", "Public Domain"
    citation: str                      # Recommended academic / data citation
    schema: dict[str, str]             # Mapping of column name -> data type
    target_column: str | None          # Default or specified target label
    feature_roles: dict[str, str]      # e.g. {"age": "numeric", "country": "categorical"}
    row_count: int | None              # Discovered or estimated row count
    consumed_row_count: int            # Exactly consumed row count
    content_fingerprint: str           # SHA-256 hex digest of data content
    partition_manifest: dict[str, Any] # Train/val/test slice indices or chunk bounds
    cache_policy: str                  # "memory" | "arrow_disk" | "none"
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetPartitionPlan:
    """Deterministic partition plan for reproducible model evaluation."""

    strategy: Literal["stratified", "random", "time_series", "group"] = "stratified"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42
    stratify_col: str | None = None
    group_col: str | None = None
    time_col: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetTelemetry:
    """Real-time backend telemetry for streaming data ingestion."""

    provider: str
    dataset_id: str
    revision: str
    streaming_status: Literal["idle", "discovering", "streaming", "caching", "completed", "error"] = "idle"
    rows_discovered: int | None = None
    rows_consumed: int = 0
    bytes_read: int = 0
    schema: dict[str, str] = field(default_factory=dict)
    partitions: int = 1
    worker_count: int = 1
    rows_per_sec: float = 0.0
    cache_state: str = "uncached"
    partition_progress: float = 0.0
    current_transformation: str = "raw_stream"
    train_shard_size: int = 0
    val_shard_size: int = 0
    test_shard_size: int = 0
    device: str = "cpu"
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DataPreCertificationReport:
    """Automated scientific quality check executed before model training."""

    source_identity: str
    license_status: str
    schema_valid: bool
    target_valid: bool
    feature_roles_resolved: bool
    duplicate_count: int
    missing_value_summary: dict[str, int]
    class_distribution: dict[str, int] | None
    temporal_order_verified: bool | None
    leakage_checks_passed: bool
    split_valid: bool
    fingerprint: str
    is_certified: bool
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DatasetStream:
    """Streaming iterator over Arrow RecordBatches or Pandas DataFrames."""

    def __init__(
        self,
        batch_generator: Iterator[pd.DataFrame],
        contract: DatasetContract,
        telemetry: DatasetTelemetry | None = None,
    ) -> None:
        self._generator = batch_generator
        self.contract = contract
        self.telemetry = telemetry or DatasetTelemetry(
            provider=contract.provider,
            dataset_id=contract.dataset_id,
            revision=contract.revision,
        )
        self._hasher = hashlib.sha256()
        self._consumed_rows = 0
        self._start_time = time.perf_counter()

    def __iter__(self) -> Iterator[pd.DataFrame]:
        self.telemetry.streaming_status = "streaming"
        for chunk in self._generator:
            n_rows = len(chunk)
            self._consumed_rows += n_rows
            chunk_bytes = chunk.memory_usage(deep=True).sum()
            self.telemetry.rows_consumed = self._consumed_rows
            self.telemetry.bytes_read += int(chunk_bytes)
            elapsed = max(1e-4, time.perf_counter() - self._start_time)
            self.telemetry.rows_per_sec = round(self._consumed_rows / elapsed, 2)

            # Update content hash incrementally
            self._hasher.update(chunk.to_csv(index=False).encode("utf-8"))
            yield chunk

        self.telemetry.streaming_status = "completed"
        self.contract.consumed_row_count = self._consumed_rows
        self.contract.content_fingerprint = self._hasher.hexdigest()

    def to_dataframe(self, max_rows: int | None = None) -> pd.DataFrame:
        """Exhaust the stream into a single pandas DataFrame up to max_rows."""
        chunks: list[pd.DataFrame] = []
        collected = 0
        for chunk in self:
            if max_rows is not None and collected + len(chunk) > max_rows:
                chunks.append(chunk.iloc[: max_rows - collected])
                break
            chunks.append(chunk)
            collected += len(chunk)
        if not chunks:
            return pd.DataFrame()
        return pd.concat(chunks, ignore_index=True)
