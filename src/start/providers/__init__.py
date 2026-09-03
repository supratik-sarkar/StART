"""Provider interfaces and implementations.

Imports lazily for the same reason :mod:`start` does: gateway *discovery* and
runtime-profile detection must work in an environment where nothing has been
installed. Eagerly importing :mod:`start.providers.base` here would pull in
pydantic via ``start.core.schemas``, and profile detection would then fall back
to its permissive default in exactly the partially-installed environment where
getting the profile right matters most.

``start.providers.gateway_discovery`` is standard-library only and can always be
imported directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from start.providers.base import (
        ComputeProvider,
        DataProvider,
        EvidenceProvider,
        ExperimentProvider,
        GenerationRequest,
        LLMProvider,
        StorageProvider,
    )

_LAZY = {
    "ComputeProvider": "start.providers.base",
    "DataProvider": "start.providers.base",
    "EvidenceProvider": "start.providers.base",
    "ExperimentProvider": "start.providers.base",
    "GenerationRequest": "start.providers.base",
    "LLMProvider": "start.providers.base",
    "StorageProvider": "start.providers.base",
}

__all__ = [
    "ComputeProvider",
    "DataProvider",
    "EvidenceProvider",
    "ExperimentProvider",
    "GenerationRequest",
    "LLMProvider",
    "StorageProvider",
]


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'start.providers' has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
