"""Generic, public-safe data source adapters and deterministic schema contracts.

Invariants:
- All datasets, examples, and test fixtures are strictly synthetic or public-safe.
- Zero confidential enterprise data, internal SQL, proprietary schemas, or firm endpoints.
- Deterministic schema validation: fails closed on column mismatch, non-finite values, or duplicates.
- Explicit chunking classification: CHUNKABLE_EXACT, CHUNKABLE_WITH_DETERMINISTIC_REDUCTION, NOT_CHUNKABLE.
- Chunked execution preserves row ordering and data fingerprints.
"""

from __future__ import annotations

import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from start.core.hashing import canonical_json, sha256_hex

T = TypeVar("T")


class ChunkClassification(StrEnum):
    """Scientific classification of an algorithm's chunkability."""

    CHUNKABLE_EXACT = "CHUNKABLE_EXACT"
    CHUNKABLE_WITH_DETERMINISTIC_REDUCTION = "CHUNKABLE_WITH_DETERMINISTIC_REDUCTION"
    NOT_CHUNKABLE = "NOT_CHUNKABLE"


@dataclass(frozen=True)
class DataSchema:
    """Deterministic schema specification for public data validation."""

    columns: dict[str, str] = field(default_factory=dict)
    required_columns: tuple[str, ...] = ()
    optional_columns: tuple[str, ...] = ()
    nullable_columns: tuple[str, ...] = ()
    unique_columns: tuple[str, ...] = ()
    require_finite: bool = True

    def validate(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Validate DataFrame against deterministic schema contract."""
        errors: list[str] = []

        # 1. Required column presence
        for col in self.required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'.")

        # 2. Check types where specified
        for col, expected_type in self.columns.items():
            if col in df.columns:
                col_type = str(df[col].dtype)
                if expected_type.lower() in ("float", "float64", "numeric"):
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        errors.append(f"Column '{col}' expected numeric type, got '{col_type}'.")
                elif expected_type.lower() in ("int", "int64", "integer"):
                    if not pd.api.types.is_integer_dtype(df[col]):
                        errors.append(f"Column '{col}' expected integer type, got '{col_type}'.")
                elif expected_type.lower() in ("str", "string", "object"):
                    if not (pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                        errors.append(f"Column '{col}' expected string/object type, got '{col_type}'.")

        # 3. Nullability checks
        for col in df.columns:
            if col not in self.nullable_columns and df[col].isnull().any():
                null_count = int(df[col].isnull().sum())
                errors.append(f"Column '{col}' contains {null_count} null value(s) but is non-nullable.")

        # 4. Finite number requirement
        if self.require_finite:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    num_series = df[col].dropna()
                    if np.isinf(num_series.to_numpy(dtype=float)).any():
                        errors.append(f"Column '{col}' contains non-finite (Inf / -Inf) values.")

        # 5. Uniqueness
        for col in self.unique_columns:
            if col in df.columns and df[col].duplicated().any():
                dup_count = int(df[col].duplicated().sum())
                errors.append(
                    f"Column '{col}' contains {dup_count} duplicate value(s) but requires uniqueness."
                )

        return (len(errors) == 0, errors)

    def canonical_fingerprint(self) -> str:
        d = {
            "columns": self.columns,
            "required_columns": list(self.required_columns),
            "optional_columns": list(self.optional_columns),
            "nullable_columns": list(self.nullable_columns),
            "unique_columns": list(self.unique_columns),
            "require_finite": self.require_finite,
        }
        return sha256_hex(canonical_json(d))


def _compute_df_fingerprint(df: pd.DataFrame) -> str:
    """Compute deterministic SHA-256 fingerprint of a DataFrame."""
    buf = []
    buf.append(f"cols:{list(df.columns)}")
    buf.append(f"shape:{df.shape}")
    for col in df.columns:
        s = df[col]
        buf.append(f"{col}:{s.tolist()}")
    return sha256_hex(canonical_json(buf))


class DataSourceAdapter(ABC):
    """Abstract base class for deterministic, public-safe data source adapters."""

    def __init__(
        self,
        schema: DataSchema | None = None,
        chunk_classification: ChunkClassification = ChunkClassification.CHUNKABLE_EXACT,
        as_of_time: str = "",
    ) -> None:
        self.schema = schema
        self.chunk_classification = chunk_classification
        self.as_of_time = as_of_time

    @property
    @abstractmethod
    def row_count(self) -> int:
        """Total row count in dataset."""
        ...

    @property
    @abstractmethod
    def column_count(self) -> int:
        """Total column count in dataset."""
        ...

    @property
    @abstractmethod
    def data_fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint of payload."""
        ...

    @abstractmethod
    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        """Read full dataset into DataFrame."""
        ...

    @abstractmethod
    def read_chunks(self, chunk_size: int, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
        """Yield deterministic chunks of dataset."""
        ...

    def validate_schema(self, df: pd.DataFrame | None = None) -> tuple[bool, list[str]]:
        """Validate dataset against schema contract."""
        if self.schema is None:
            return True, []
        target_df = df if df is not None else self.read()
        return self.schema.validate(target_df)

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_type": self.__class__.__name__,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "data_fingerprint": self.data_fingerprint,
            "chunk_classification": self.chunk_classification.value,
            "as_of_time": self.as_of_time,
            "schema_fingerprint": self.schema.canonical_fingerprint() if self.schema else "",
        }


class DataFrameAdapter(DataSourceAdapter):
    """In-memory pandas DataFrame adapter with deterministic schema contracts."""

    def __init__(
        self,
        df: pd.DataFrame,
        schema: DataSchema | None = None,
        chunk_classification: ChunkClassification = ChunkClassification.CHUNKABLE_EXACT,
        as_of_time: str = "",
    ) -> None:
        super().__init__(schema=schema, chunk_classification=chunk_classification, as_of_time=as_of_time)
        self._df = df.copy()
        self._fingerprint = _compute_df_fingerprint(self._df)
        if self.schema is not None:
            valid, errors = self.schema.validate(self._df)
            if not valid:
                raise ValueError(f"DataFrame schema validation failed: {errors}")

    @property
    def row_count(self) -> int:
        return len(self._df)

    @property
    def column_count(self) -> int:
        return len(self._df.columns)

    @property
    def data_fingerprint(self) -> str:
        return self._fingerprint

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        if columns is not None:
            return self._df[columns].copy()
        return self._df.copy()

    def read_chunks(self, chunk_size: int, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive integer, got {chunk_size}")
        target = self.read(columns=columns)
        for i in range(0, len(target), chunk_size):
            yield target.iloc[i : i + chunk_size].copy()


def is_parquet_available() -> bool:
    """Detect if an optional Parquet engine (pyarrow or fastparquet) is installed."""
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        try:
            import fastparquet  # noqa: F401

            return True
        except ImportError:
            return False


class LocalFileAdapter(DataSourceAdapter):
    """Public-safe local file adapter for CSV (and optional Parquet) datasets."""

    def __init__(
        self,
        file_path: str | Path,
        schema: DataSchema | None = None,
        chunk_classification: ChunkClassification = ChunkClassification.CHUNKABLE_EXACT,
        as_of_time: str = "",
    ) -> None:
        super().__init__(schema=schema, chunk_classification=chunk_classification, as_of_time=as_of_time)
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.file_path}")

        # Compute deterministic fingerprint from file content
        self._fingerprint = sha256_hex(self.file_path.read_bytes())
        # Initial probe to determine dimensions
        probe = self.read()
        self._row_count = len(probe)
        self._column_count = len(probe.columns)
        if self.schema is not None:
            valid, errors = self.schema.validate(probe)
            if not valid:
                raise ValueError(f"File schema validation failed: {errors}")

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def column_count(self) -> int:
        return self._column_count

    @property
    def data_fingerprint(self) -> str:
        return self._fingerprint

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        if self.file_path.suffix.lower() == ".csv":
            return pd.read_csv(self.file_path, usecols=columns)
        elif self.file_path.suffix.lower() in (".parquet", ".pq"):
            if not is_parquet_available():
                raise ImportError(
                    "Parquet format requires optional dependency 'pyarrow' or 'fastparquet'. "
                    "Use CSV format or install pyarrow."
                )
            return pd.read_parquet(self.file_path, columns=columns)
        else:
            raise ValueError(
                f"Unsupported file format: {self.file_path.suffix}. Supported mandatory format: .csv"
            )

    def read_chunks(self, chunk_size: int, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive integer, got {chunk_size}")
        if self.file_path.suffix.lower() == ".csv":
            yield from pd.read_csv(self.file_path, chunksize=chunk_size, usecols=columns)
        else:
            full_df = self.read(columns=columns)
            for i in range(0, len(full_df), chunk_size):
                yield full_df.iloc[i : i + chunk_size].copy()


class SQLDataSourceAdapter(DataSourceAdapter):
    """Generic public SQL adapter. Tested against in-memory/local SQLite fixtures."""

    def __init__(
        self,
        connection_or_path: sqlite3.Connection | str,
        table_or_query: str,
        schema: DataSchema | None = None,
        chunk_classification: ChunkClassification = ChunkClassification.CHUNKABLE_EXACT,
        as_of_time: str = "",
    ) -> None:
        super().__init__(schema=schema, chunk_classification=chunk_classification, as_of_time=as_of_time)
        if isinstance(connection_or_path, str):
            self.conn = sqlite3.connect(connection_or_path)
        else:
            self.conn = connection_or_path
        self.table_or_query = table_or_query
        df = self.read()
        self._row_count = len(df)
        self._column_count = len(df.columns)
        self._fingerprint = _compute_df_fingerprint(df)
        if self.schema is not None:
            valid, errors = self.schema.validate(df)
            if not valid:
                raise ValueError(f"SQL data schema validation failed: {errors}")

    @property
    def row_count(self) -> int:
        return self._row_count

    @property
    def column_count(self) -> int:
        return self._column_count

    @property
    def data_fingerprint(self) -> str:
        return self._fingerprint

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        query = self.table_or_query
        if " " not in query.strip() and not query.strip().upper().startswith("SELECT"):
            cols_str = ", ".join(columns) if columns else "*"
            query = f"SELECT {cols_str} FROM {query}"
        df = pd.read_sql_query(query, self.conn)
        if columns is not None and set(columns).issubset(df.columns):
            return df[columns].copy()
        return df

    def read_chunks(self, chunk_size: int, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
        full_df = self.read(columns=columns)
        for i in range(0, len(full_df), chunk_size):
            yield full_df.iloc[i : i + chunk_size].copy()


class WarehouseAdapter(DataSourceAdapter):
    """Generic public data warehouse adapter interface (INTERFACE_ONLY, zero proprietary firm configs)."""

    def __init__(
        self,
        endpoint_uri: str,
        table_name: str,
        schema: DataSchema | None = None,
        as_of_time: str = "",
    ) -> None:
        super().__init__(
            schema=schema,
            chunk_classification=ChunkClassification.NOT_CHUNKABLE,
            as_of_time=as_of_time,
        )
        self.endpoint_uri = endpoint_uri
        self.table_name = table_name

    @property
    def row_count(self) -> int:
        return 0

    @property
    def column_count(self) -> int:
        return 0

    @property
    def data_fingerprint(self) -> str:
        return sha256_hex(canonical_json({"endpoint": self.endpoint_uri, "table": self.table_name}))

    def read(self, columns: list[str] | None = None) -> pd.DataFrame:
        # Public abstraction interface placeholder
        return pd.DataFrame()

    def read_chunks(self, chunk_size: int, columns: list[str] | None = None) -> Iterator[pd.DataFrame]:
        raise NotImplementedError("WarehouseAdapter is an interface abstraction; not directly executable.")


def execute_chunked[T](
    adapter: DataSourceAdapter,
    chunk_size: int,
    map_fn: Callable[[pd.DataFrame], T],
    reduce_fn: Callable[[list[T]], T],
    columns: list[str] | None = None,
) -> T:
    """Deterministically execute a map-reduce pipeline over data chunks.

    Raises:
        ValueError: if adapter.chunk_classification is NOT_CHUNKABLE.
    """
    if adapter.chunk_classification == ChunkClassification.NOT_CHUNKABLE:
        raise ValueError(
            f"Adapter '{adapter.__class__.__name__}' is classified as NOT_CHUNKABLE. "
            f"Algorithm requires simultaneous access to all observations."
        )

    chunk_results: list[T] = []
    for chunk in adapter.read_chunks(chunk_size=chunk_size, columns=columns):
        res = map_fn(chunk)
        chunk_results.append(res)

    return reduce_fn(chunk_results)
