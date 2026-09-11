"""Kaggle Data Provider Adapter for StART.

Strict Invariants:
1. Legitimate official Kaggle API authentication only. Never scrape.
2. Credentials remain strictly backend-only (~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY).
3. If credentials or official package are missing, fail-closed and truthfully report deferred/unavailable.
"""

from __future__ import annotations

import os
from pathlib import Path

from start.data.providers.base import DatasetProviderAdapter
from start.data.providers.contract import DatasetContract, DatasetStream


class KaggleProviderAdapter(DatasetProviderAdapter):
    """Adapter for official Kaggle datasets."""

    @property
    def name(self) -> str:
        return "kaggle"

    def is_runnable(self) -> tuple[bool, str]:
        # 1. Check official library
        try:
            import kaggle  # noqa: F401
        except ImportError:
            return False, "Kaggle package is not installed (pip install kaggle)."

        # 2. Check credentials (file, env, or ephemeral session credentials)
        cfg_path = Path.home() / ".kaggle" / "kaggle.json"
        has_file = cfg_path.exists()
        has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
        has_ephemeral = bool(
            self._credentials
            and (self._credentials.get("username") or self._credentials.get("key"))
        )

        if not (has_file or has_env or has_ephemeral):
            return False, "Kaggle credentials not configured in environment, session, or ~/.kaggle/kaggle.json."

        return True, "Kaggle authenticated API is active."

    def get_contract(
        self,
        dataset_id: str,
        target_column: str | None = None,
        revision: str | None = None,
    ) -> DatasetContract:
        runnable, msg = self.is_runnable()
        if not runnable:
            raise PermissionError(f"Kaggle provider unavailable: {msg}")

        import kaggle

        # Retrieve dataset metadata via official API
        owner, dataset_name = dataset_id.split("/")
        meta = kaggle.api.dataset_metadata(owner, dataset_name)

        return DatasetContract(
            provider=self.name,
            dataset_id=dataset_id,
            revision=revision or "latest",
            source_uri=f"https://www.kaggle.com/datasets/{dataset_id}",
            license=getattr(meta, "license_name", "Kaggle Dataset Terms"),
            citation=f"Kaggle dataset: {dataset_id}",
            schema={"feature": "unknown"},
            target_column=target_column,
            feature_roles={"target": "target"},
            row_count=None,
            consumed_row_count=0,
            content_fingerprint="",
            partition_manifest={"kaggle_slug": dataset_id},
            cache_policy="arrow_cache",
        )

    def stream(
        self,
        dataset_id: str,
        batch_size: int = 1000,
        max_rows: int | None = None,
        target_column: str | None = None,
        revision: str | None = None,
    ) -> DatasetStream:
        runnable, msg = self.is_runnable()
        if not runnable:
            raise PermissionError(f"Kaggle provider unavailable: {msg}")

        raise NotImplementedError("Kaggle streaming download requires authenticated local cache directory.")
