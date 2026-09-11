"""StART Live Data Provider Runtime.

Provides streaming, caching, fingerprinting, and pre-certification across:
- Hugging Face (`huggingface`)
- OpenML (`openml`)
- UCI Machine Learning Repository (`uci`)
- Kaggle (`kaggle`)
- Local CSV (`local_csv`)
- Local Parquet (`local_parquet`)
"""

from __future__ import annotations

from start.data.providers.base import DatasetProviderAdapter
from start.data.providers.contract import (
    DataPreCertificationReport,
    DatasetContract,
    DatasetPartitionPlan,
    DatasetStream,
    DatasetTelemetry,
)
from start.data.providers.parallel import ArrowColumnarBatchPipeline, evaluate_ray_backend
from start.data.providers.precertification import precertify_dataset
from start.data.providers.registry import get_provider_adapter, list_provider_adapters

__all__ = [
    "DatasetContract",
    "DatasetPartitionPlan",
    "DatasetStream",
    "DatasetTelemetry",
    "DataPreCertificationReport",
    "DatasetProviderAdapter",
    "get_provider_adapter",
    "list_provider_adapters",
    "precertify_dataset",
    "evaluate_ray_backend",
    "ArrowColumnarBatchPipeline",
]
