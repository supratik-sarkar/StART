"""Provider abstraction interfaces.

StART has no hard dependency on any enterprise system. Every external
capability (compute, data, experiment tracking, LLMs, storage, evidence
persistence) sits behind one of these interfaces, and every interface has a
local, dependency-light implementation so the framework degrades safely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from start.core.schemas import ComputeDevice, EvidenceRecord


class ComputeProvider(ABC):
    """Executes deterministic test callables on a target device/runtime."""

    name: str = "compute"

    @abstractmethod
    def device(self) -> ComputeDevice: ...

    @abstractmethod
    def run(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        """Run a callable. Local providers call directly; distributed
        providers may serialize and dispatch."""


class DataProvider(ABC):
    name: str = "data"

    @abstractmethod
    def load(self, ref: str) -> Any:
        """Load a dataset reference into a pandas DataFrame."""

    @abstractmethod
    def dataset_id(self, ref: str) -> str: ...


class ExperimentProvider(ABC):
    name: str = "experiment"

    @abstractmethod
    def start_run(self, run_name: str) -> str: ...

    @abstractmethod
    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None: ...

    @abstractmethod
    def log_artifact(self, run_id: str, path: str) -> None: ...

    @abstractmethod
    def end_run(self, run_id: str) -> None: ...


class LLMProvider(ABC):
    """Backend-agnostic chat interface. May be a no-op (NoLLMProvider)."""

    name: str = "llm"

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    def complete(self, system: str, user: str, *, max_tokens: int = 1024) -> str: ...

    def generate(
        self, prompt: str, *, system: str | None = None, metadata: dict | None = None
    ) -> str:
        """Common cross-provider interface (spec: prompt + optional system +
        optional metadata). Delegates to ``complete``; metadata is advisory
        (e.g. max_tokens) and never contains raw confidential data."""
        max_tokens = int((metadata or {}).get("max_tokens", 1024))
        return self.complete(system or "", prompt, max_tokens=max_tokens)


class StorageProvider(ABC):
    name: str = "storage"

    @abstractmethod
    def write_text(self, relpath: str, content: str) -> str: ...

    @abstractmethod
    def read_text(self, relpath: str) -> str: ...


class EvidenceProvider(ABC):
    """Persists evidence records with tamper-evidence guarantees."""

    name: str = "evidence"

    @abstractmethod
    def append(self, record: EvidenceRecord) -> str:
        """Persist a record; return its content hash."""

    @abstractmethod
    def verify(self) -> bool:
        """Verify ledger integrity (hash chain)."""
