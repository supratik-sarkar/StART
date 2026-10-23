"""Registry of StART Live Data Provider Adapters."""

from __future__ import annotations

from typing import Any

from start.data.providers.base import DatasetProviderAdapter
from start.data.providers.huggingface import HuggingFaceProviderAdapter
from start.data.providers.kaggle import KaggleProviderAdapter
from start.data.providers.local import LocalCSVProviderAdapter, LocalParquetProviderAdapter
from start.data.providers.openml import OpenMLProviderAdapter
from start.data.providers.uci import UCIProviderAdapter

_REGISTRY: dict[str, type[DatasetProviderAdapter]] = {
    "huggingface": HuggingFaceProviderAdapter,
    "openml": OpenMLProviderAdapter,
    "uci": UCIProviderAdapter,
    "kaggle": KaggleProviderAdapter,
    "local_csv": LocalCSVProviderAdapter,
    "local_parquet": LocalParquetProviderAdapter,
}


def get_provider_adapter(name: str, credentials: dict[str, Any] | None = None) -> DatasetProviderAdapter:
    """Retrieve an instantiated adapter for the requested provider family with optional credentials."""
    key = name.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown data provider '{name}'. Supported: {sorted(_REGISTRY.keys())}")
    adapter = _REGISTRY[key]()
    if credentials is not None and hasattr(adapter, "set_credentials"):
        adapter.set_credentials(credentials)
    return adapter


def list_provider_adapters() -> dict[str, dict[str, Any]]:
    """Probe all registered data providers and return their truthful operational status."""
    report = {}
    for name, adapter_cls in _REGISTRY.items():
        adapter = adapter_cls()
        runnable, details = adapter.is_runnable()
        report[name] = {
            "name": name,
            "runnable": runnable,
            "status": "RUNNABLE" if runnable else "DEFERRED",
            "details": details,
        }
    return report
