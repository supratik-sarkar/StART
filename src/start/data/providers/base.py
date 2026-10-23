"""Abstract Base Class for StART Live Data Provider Adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from start.data.providers.contract import DatasetContract, DatasetStream


class DatasetProviderAdapter(ABC):
    """Abstract interface for streaming and caching datasets across providers."""

    def __init__(self) -> None:
        self._credentials: dict[str, Any] | None = None

    def set_credentials(self, credentials: dict[str, Any] | None) -> None:
        """Attach server-side ephemeral credentials to the adapter instance."""
        self._credentials = credentials

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the provider family (e.g. 'huggingface', 'openml')."""
        ...

    @abstractmethod
    def is_runnable(self) -> tuple[bool, str]:
        """Check whether provider dependencies and credentials are functional.

        Returns:
            (is_functional, reason_or_error_message)
        """
        ...

    @abstractmethod
    def get_contract(
        self,
        dataset_id: str,
        target_column: str | None = None,
        revision: str | None = None,
    ) -> DatasetContract:
        """Fetch metadata, license, and schema contract for the specified dataset."""
        ...

    @abstractmethod
    def stream(
        self,
        dataset_id: str,
        batch_size: int = 1000,
        max_rows: int | None = None,
        target_column: str | None = None,
        revision: str | None = None,
    ) -> DatasetStream:
        """Return a DatasetStream yielding chunks while updating telemetry and computing hash."""
        ...

    def load_dataframe(
        self,
        dataset_id: str,
        max_rows: int | None = None,
        target_column: str | None = None,
        revision: str | None = None,
    ) -> tuple[pd.DataFrame, DatasetContract]:
        """Helper to stream and load dataset into a DataFrame alongside its validated contract."""
        stream_obj = self.stream(
            dataset_id=dataset_id,
            batch_size=max_rows or 2000,
            max_rows=max_rows,
            target_column=target_column,
            revision=revision,
        )
        df = stream_obj.to_dataframe(max_rows=max_rows)
        return df, stream_obj.contract
