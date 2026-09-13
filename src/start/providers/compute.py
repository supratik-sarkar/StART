"""Adaptive compute routing.

Device detection order: CUDA -> MPS (Apple Silicon) -> CPU. Databricks
runtime detection is independent of device detection. Distributed backends
(Databricks Spark, Ray, Dask) are declared in the router but only local and
Databricks-stub providers are implemented in v0.1; everything degrades safely
to local CPU.
"""

from __future__ import annotations

import os
from typing import Any

from start.core.config import ComputeConfig
from start.core.schemas import ComputeDevice
from start.providers.base import ComputeProvider


def detect_device() -> ComputeDevice:
    """CUDA -> MPS -> CPU. Never raises; torch is optional."""
    try:
        import torch  # type: ignore
    except ImportError:
        return ComputeDevice.CPU
    try:
        if torch.cuda.is_available():
            return ComputeDevice.CUDA
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return ComputeDevice.MPS
    except Exception:
        pass
    return ComputeDevice.CPU


def is_databricks_runtime() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def mlflow_available() -> bool:
    try:
        import mlflow  # noqa: F401

        return True
    except ImportError:
        return False


class LocalCPUProvider(ComputeProvider):
    name = "local_cpu"

    def device(self) -> ComputeDevice:
        return ComputeDevice.CPU

    def run(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)


class LocalGPUProvider(ComputeProvider):
    """Local accelerated provider: CUDA when present, else Apple MPS.

    Falls back to CPU transparently if no accelerator is detected, so
    requesting GPU on a CPU-only machine degrades instead of failing.
    """

    name = "local_gpu"

    def __init__(self) -> None:
        self._device = detect_device()

    def device(self) -> ComputeDevice:
        return self._device

    def run(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)


class DatabricksCPUProvider(LocalCPUProvider):
    """Stub: on Databricks, execution is still in-process; Spark-aware
    dispatch belongs to a future distributed backend."""

    name = "databricks_cpu"


class DatabricksGPUProvider(LocalGPUProvider):
    name = "databricks_gpu"


class DistributedBackendNotImplemented(RuntimeError):
    pass


def get_compute_provider(config: ComputeConfig) -> ComputeProvider:
    """Route to a compute provider from config + environment.

    Routing order: explicit distributed backend -> Databricks runtime ->
    requested device -> auto-detected device -> CPU.
    """
    if config.distributed_backend in {"ray", "dask"}:
        raise DistributedBackendNotImplemented(
            f"Distributed backend '{config.distributed_backend}' is declared in the "
            "architecture but not implemented in v0.1. Use 'none' or 'databricks_spark'."
        )

    on_databricks = is_databricks_runtime() or config.mode == "databricks"
    wants_gpu = config.mode == "gpu" or (
        config.mode == "auto" and detect_device() != ComputeDevice.CPU
    )
    if config.device in {"cuda", "mps"}:
        wants_gpu = True
    if config.device == "cpu" or config.mode == "cpu":
        wants_gpu = False

    if on_databricks:
        return DatabricksGPUProvider() if wants_gpu else DatabricksCPUProvider()
    return LocalGPUProvider() if wants_gpu else LocalCPUProvider()
