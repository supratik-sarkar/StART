"""Data providers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from start.core.hashing import hash_dataframe
from start.providers.base import DataProvider


class CSVParquetDataProvider(DataProvider):
    """Loads CSV/Parquet files from a root directory."""

    name = "csv_parquet"

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root)

    def load(self, ref: str) -> Any:
        import pandas as pd

        path = (self.root / ref) if not Path(ref).is_absolute() else Path(ref)
        if path.suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
        raise ValueError(f"Unsupported data file type: {path.suffix}")

    def dataset_id(self, ref: str) -> str:
        return f"file:{ref}"

    def content_hash(self, ref: str) -> str:
        return hash_dataframe(self.load(ref))


class SnowflakePlaceholderProvider(DataProvider):
    """Placeholder for warehouse-backed data access.

    Contains no schemas, connection logic, or credentials. Implement a
    private provider with the same interface for real warehouse access.
    """

    name = "snowflake_placeholder"

    def load(self, ref: str) -> Any:
        raise NotImplementedError(
            "SnowflakePlaceholderProvider is a public-safe stub. Provide a "
            "private implementation outside this repository."
        )

    def dataset_id(self, ref: str) -> str:
        return f"snowflake:{ref}"
