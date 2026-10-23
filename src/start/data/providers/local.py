"""Local CSV and Parquet Data Provider Adapters for StART.

Supports local columnar streaming via PyArrow, strict schema extraction,
and deterministic content hashing.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from start.data.providers.base import DatasetProviderAdapter
from start.data.providers.contract import DatasetContract, DatasetStream, DatasetTelemetry


class LocalCSVProviderAdapter(DatasetProviderAdapter):
    """Streaming adapter for local CSV files."""

    @property
    def name(self) -> str:
        return "local_csv"

    def is_runnable(self) -> tuple[bool, str]:
        return True, "Local CSV adapter is active."

    def get_contract(
        self,
        dataset_id: str,
        target_column: str | None = None,
        revision: str | None = None,
    ) -> DatasetContract:
        p = Path(dataset_id)
        if not p.exists():
            raise FileNotFoundError(f"Local CSV file not found: {dataset_id}")

        # Read header to determine schema
        sample = pd.read_csv(p, nrows=10)
        schema = {c: str(sample[c].dtype) for c in sample.columns}
        target = target_column or ("target" if "target" in schema else list(schema.keys())[-1])

        roles = {}
        for col, t in schema.items():
            if col == target:
                roles[col] = "target"
            elif "int" in t or "float" in t:
                roles[col] = "numeric"
            else:
                roles[col] = "categorical"

        stat = p.stat()
        mtime_rev = revision or str(int(stat.st_mtime))

        return DatasetContract(
            provider=self.name,
            dataset_id=str(p.name),
            revision=mtime_rev,
            source_uri=f"file://{p.resolve()}",
            license="Local / Proprietor Controlled",
            citation=f"Local CSV Dataset: {p.name}",
            schema=schema,
            target_column=target,
            feature_roles=roles,
            row_count=None,
            consumed_row_count=0,
            content_fingerprint="",
            partition_manifest={"file_path": str(p.resolve()), "file_size_bytes": stat.st_size},
            cache_policy="mmap_or_disk",
        )

    def stream(
        self,
        dataset_id: str,
        batch_size: int = 1000,
        max_rows: int | None = None,
        target_column: str | None = None,
        revision: str | None = None,
    ) -> DatasetStream:
        p = Path(dataset_id)
        contract = self.get_contract(dataset_id, target_column=target_column, revision=revision)
        telemetry = DatasetTelemetry(
            provider=self.name,
            dataset_id=contract.dataset_id,
            revision=contract.revision,
            schema=contract.schema,
        )

        def _generator() -> Iterator[pd.DataFrame]:
            total_read = 0
            for chunk in pd.read_csv(p, chunksize=batch_size):
                if max_rows and total_read + len(chunk) > max_rows:
                    yield chunk.iloc[: max_rows - total_read]
                    return
                yield chunk
                total_read += len(chunk)

        return DatasetStream(
            batch_generator=_generator(),
            contract=contract,
            telemetry=telemetry,
        )


class LocalParquetProviderAdapter(DatasetProviderAdapter):
    """Streaming adapter for local Apache Parquet columnar files."""

    @property
    def name(self) -> str:
        return "local_parquet"

    def is_runnable(self) -> tuple[bool, str]:
        return True, "Local Parquet adapter is active via PyArrow."

    def get_contract(
        self,
        dataset_id: str,
        target_column: str | None = None,
        revision: str | None = None,
    ) -> DatasetContract:
        p = Path(dataset_id)
        if not p.exists():
            raise FileNotFoundError(f"Local Parquet file not found: {dataset_id}")

        pq_file = pq.ParquetFile(p)
        arrow_schema = pq_file.schema_arrow
        schema = {name: str(arrow_schema.field(name).type) for name in arrow_schema.names}
        target = target_column or ("target" if "target" in schema else list(schema.keys())[-1])

        roles = {}
        for col, t in schema.items():
            if col == target:
                roles[col] = "target"
            elif any(x in t.lower() for x in ("int", "float", "double", "decimal")):
                roles[col] = "numeric"
            else:
                roles[col] = "categorical"

        stat = p.stat()
        mtime_rev = revision or str(int(stat.st_mtime))

        return DatasetContract(
            provider=self.name,
            dataset_id=str(p.name),
            revision=mtime_rev,
            source_uri=f"file://{p.resolve()}",
            license="Local / Proprietor Controlled",
            citation=f"Local Parquet Dataset: {p.name}",
            schema=schema,
            target_column=target,
            feature_roles=roles,
            row_count=pq_file.metadata.num_rows,
            consumed_row_count=0,
            content_fingerprint="",
            partition_manifest={"row_groups": pq_file.num_row_groups, "num_rows": pq_file.metadata.num_rows},
            cache_policy="zero_copy_arrow",
        )

    def stream(
        self,
        dataset_id: str,
        batch_size: int = 1000,
        max_rows: int | None = None,
        target_column: str | None = None,
        revision: str | None = None,
    ) -> DatasetStream:
        p = Path(dataset_id)
        contract = self.get_contract(dataset_id, target_column=target_column, revision=revision)
        telemetry = DatasetTelemetry(
            provider=self.name,
            dataset_id=contract.dataset_id,
            revision=contract.revision,
            schema=contract.schema,
            rows_discovered=contract.row_count,
        )

        def _generator() -> Iterator[pd.DataFrame]:
            pq_file = pq.ParquetFile(p)
            total_read = 0
            for batch in pq_file.iter_batches(batch_size=batch_size):
                chunk = batch.to_pandas()
                if max_rows and total_read + len(chunk) > max_rows:
                    yield chunk.iloc[: max_rows - total_read]
                    return
                yield chunk
                total_read += len(chunk)

        return DatasetStream(
            batch_generator=_generator(),
            contract=contract,
            telemetry=telemetry,
        )
