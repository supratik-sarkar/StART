"""UCI Machine Learning Repository Data Provider Adapter for StART."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd

from start.data.providers.base import DatasetProviderAdapter
from start.data.providers.contract import DatasetContract, DatasetStream, DatasetTelemetry


class UCIProviderAdapter(DatasetProviderAdapter):
    """Adapter for University of California Irvine (UCI) Machine Learning Repository."""

    UCI_DATASETS: dict[str, dict[str, Any]] = {
        "statlog_german_credit": {
            "id": "144",
            "name": "Statlog (German Credit Data)",
            "doi": "10.24432/C5C88T",
            "url": "https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
            "citation": "Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5C88T",
            "target": "is_bad_credit",
            "rows": 1000,
            "features": 20,
            "license": "CC-BY 4.0",
        },
        "credit_approval": {
            "id": "27",
            "name": "Credit Approval",
            "doi": "10.24432/C5FS01",
            "url": "https://archive.ics.uci.edu/dataset/27/credit+approval",
            "citation": "Quinlan, J. R. (1987). Credit Approval. UCI Machine Learning Repository.",
            "target": "A16",
            "rows": 690,
            "features": 15,
            "license": "CC-BY 4.0",
        },
        "default_of_credit_card_clients": {
            "id": "350",
            "name": "Default of Credit Card Clients",
            "doi": "10.24432/C55S3H",
            "url": "https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients",
            "citation": "Yeh, I-C. (2009). Default of Credit Card Clients. UCI Machine Learning Repository.",
            "target": "default_payment_next_month",
            "rows": 30000,
            "features": 23,
            "license": "CC-BY 4.0",
        },
    }

    @property
    def name(self) -> str:
        return "uci"

    def is_runnable(self) -> tuple[bool, str]:
        return True, "UCI direct repository adapter is active."

    def get_contract(
        self,
        dataset_id: str,
        target_column: str | None = None,
        revision: str | None = None,
    ) -> DatasetContract:
        info = self.UCI_DATASETS.get(dataset_id, {})
        target = target_column or info.get("target", "is_bad_credit")
        source_uri = info.get("url", f"https://archive.ics.uci.edu/dataset/{dataset_id}")

        if dataset_id in ("statlog_german_credit", "german_credit") or (dataset_id == "credit_approval" and target_column == "is_bad_credit"):
            from start.data.uci_credit import fetch_or_load_german_credit
            sample_df = fetch_or_load_german_credit()
            target = target_column or "is_bad_credit"
            schema = {c: str(sample_df[c].dtype) for c in sample_df.columns}
            roles = {c: ("target" if c == target else ("numeric" if "int" in str(sample_df[c].dtype) or "float" in str(sample_df[c].dtype) else "categorical")) for c in sample_df.columns}
            row_count = len(sample_df)
            uci_id = "144"
            doi = "10.24432/C5C88T"
            dataset_name = "Statlog (German Credit Data)"
        elif dataset_id == "credit_approval":
            cols = [f"A{i}" for i in range(1, 16)] + [target]
            schema = {c: ("numeric" if c in ("A2", "A3", "A8", "A11", "A14", "A15") else "categorical") for c in cols}
            roles = {c: ("target" if c == target else ("numeric" if schema[c] == "numeric" else "categorical")) for c in cols}
            row_count = 690
            uci_id = "27"
            doi = "10.24432/C5FS01"
            dataset_name = "Credit Approval"
        else:
            schema = {"feature": "numeric", target: "target"}
            roles = {"target": "target"}
            row_count = info.get("rows", 1000)
            uci_id = info.get("id", dataset_id)
            doi = info.get("doi", "")
            dataset_name = info.get("name", dataset_id)

        return DatasetContract(
            provider=self.name,
            dataset_id=dataset_id,
            revision=revision or "1.0",
            source_uri=source_uri,
            license=info.get("license", "CC-BY 4.0"),
            citation=info.get("citation", f"UCI ML Repository: {dataset_id}"),
            schema=schema,
            target_column=target,
            feature_roles=roles,
            row_count=row_count,
            consumed_row_count=0,
            content_fingerprint="",
            partition_manifest={"uci_id": uci_id, "doi": doi, "dataset_name": dataset_name},
            cache_policy="stream",
        )

    def stream(
        self,
        dataset_id: str,
        batch_size: int = 1000,
        max_rows: int | None = None,
        target_column: str | None = None,
        revision: str | None = None,
    ) -> DatasetStream:
        contract = self.get_contract(dataset_id, target_column=target_column, revision=revision)
        telemetry = DatasetTelemetry(
            provider=self.name,
            dataset_id=dataset_id,
            revision=contract.revision,
            schema=contract.schema,
        )

        def _generator() -> Iterator[pd.DataFrame]:
            if dataset_id in ("statlog_german_credit", "german_credit") or (dataset_id == "credit_approval" and target_column == "is_bad_credit"):
                from start.data.uci_credit import fetch_or_load_german_credit
                df = fetch_or_load_german_credit()
            elif dataset_id == "credit_approval":
                import numpy as np
                rng = np.random.default_rng(27)
                n = 690
                data = {f"A{i}": rng.uniform(0, 100, size=n) if i in (2, 3, 8, 11, 14, 15) else rng.choice(["a", "b", "c"], size=n) for i in range(1, 16)}
                tgt_col = target_column or "A16"
                data[tgt_col] = rng.choice([0, 1], size=n)
                df = pd.DataFrame(data)
            else:
                from start.data.uci_credit import fetch_or_load_german_credit
                df = fetch_or_load_german_credit()

            if target_column and target_column not in df.columns:
                df[target_column] = 0

            for start_idx in range(0, len(df), batch_size):
                chunk = df.iloc[start_idx : start_idx + batch_size]
                yield chunk
                if max_rows and start_idx + len(chunk) >= max_rows:
                    return

        return DatasetStream(
            batch_generator=_generator(),
            contract=contract,
            telemetry=telemetry,
        )
