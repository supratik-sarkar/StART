from start.core.config import ComputeConfig
from start.core.schemas import ComputeDevice
from start.providers.compute import (
    LocalCPUProvider,
    LocalGPUProvider,
    detect_device,
    get_compute_provider,
)


def test_detect_device_never_raises():
    assert detect_device() in set(ComputeDevice)


def test_cpu_mode_forces_cpu():
    provider = get_compute_provider(ComputeConfig(mode="cpu"))
    assert isinstance(provider, LocalCPUProvider)
    assert provider.device() == ComputeDevice.CPU


def test_gpu_request_degrades_safely():
    provider = get_compute_provider(ComputeConfig(mode="gpu"))
    assert isinstance(provider, LocalGPUProvider)
    assert provider.device() in set(ComputeDevice)  # CPU fallback allowed


def test_provider_runs_callable():
    assert LocalCPUProvider().run(lambda x: x + 1, 41) == 42
