"""Hugging Face Live Streaming Data Provider Adapter for StART.

Uses official Hugging Face `datasets` library with `streaming=True` by default.
"""

from __future__ import annotations

from collections.abc import Iterator

import pandas as pd

from start.data.providers.base import DatasetProviderAdapter
from start.data.providers.contract import DatasetContract, DatasetStream, DatasetTelemetry


class HuggingFaceProviderAdapter(DatasetProviderAdapter):
    """Streaming adapter for Hugging Face Hub tabular datasets."""

    @property
    def name(self) -> str:
        return "huggingface"

    def is_runnable(self) -> tuple[bool, str]:
        try:
            import datasets  # noqa: F401
            return True, "Hugging Face datasets library is installed and active."
        except ImportError:
            return False, "datasets package is not installed."

    def get_contract(
        self,
        dataset_id: str,
        target_column: str | None = None,
        revision: str | None = None,
    ) -> DatasetContract:
        rev = revision or "main"
        source_uri = f"https://huggingface.co/datasets/{dataset_id}"

        # Probe initial schema via lightweight streaming load
        from datasets import load_dataset

        ds = load_dataset(dataset_id, split="train", streaming=True, revision=rev)
        first_row = next(iter(ds))
        schema = {k: str(type(v).__name__) for k, v in first_row.items()}

        target = target_column
        if not target:
            # Common target column heuristics
            candidates = ["income", "class", "target", "label", "default", "y"]
            for c in candidates:
                if c in schema:
                    target = c
                    break
            if not target:
                target = list(schema.keys())[-1]

        roles = {}
        for col, t in schema.items():
            if col == target:
                roles[col] = "target"
            elif t in ("int", "float", "int64", "float64"):
                roles[col] = "numeric"
            else:
                roles[col] = "categorical"

        return DatasetContract(
            provider=self.name,
            dataset_id=dataset_id,
            revision=rev,
            source_uri=source_uri,
            license="Hugging Face Open Dataset (Apache 2.0 / CC-BY / Open)",
            citation=f"Hugging Face Hub repository: {dataset_id}",
            schema=schema,
            target_column=target,
            feature_roles=roles,
            row_count=None,  # streaming discovery
            consumed_row_count=0,
            content_fingerprint="",
            partition_manifest={"stream_chunk_size": 1000},
            cache_policy="arrow_stream",
        )

    def stream(
        self,
        dataset_id: str,
        batch_size: int = 1000,
        max_rows: int | None = None,
        target_column: str | None = None,
        revision: str | None = None,
    ) -> DatasetStream:
        from datasets import load_dataset

        contract = self.get_contract(dataset_id, target_column=target_column, revision=revision)
        telemetry = DatasetTelemetry(
            provider=self.name,
            dataset_id=dataset_id,
            revision=contract.revision,
            schema=contract.schema,
        )

        def _generator() -> Iterator[pd.DataFrame]:
            ds = load_dataset(dataset_id, split="train", streaming=True, revision=contract.revision)
            buffer = []
            yielded_total = 0

            for item in ds:
                buffer.append(item)
                if len(buffer) >= batch_size:
                    df_chunk = pd.DataFrame(buffer)
                    buffer.clear()
                    yielded_total += len(df_chunk)
                    yield df_chunk
                    if max_rows and yielded_total >= max_rows:
                        return

            if buffer:
                df_chunk = pd.DataFrame(buffer)
                yield df_chunk

        return DatasetStream(
            batch_generator=_generator(),
            contract=contract,
            telemetry=telemetry,
        )
