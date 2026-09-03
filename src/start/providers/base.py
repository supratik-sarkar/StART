"""Provider abstraction interfaces.

StART has no hard dependency on any enterprise system. Every external
capability (compute, data, experiment tracking, LLMs, storage, evidence
persistence) sits behind one of these interfaces, and every interface has a
local, dependency-light implementation so the framework degrades safely.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from start.core.schemas import ComputeDevice


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


@dataclass
class GenerationRequest:
    """Provider-neutral LLM generation request specification."""

    prompt: str
    system: str = ""
    output_token_budget: int = 1024
    temperature: float | None = None
    reasoning_effort: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProviderUsage:
    """Safe token usage statistics."""

    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass
class ProviderResult:
    """Typed normalized provider execution result preserving response lifecycle metadata."""

    text: str = ""
    provider: str = ""
    model: str = ""
    response_id: str = ""
    status: str = "completed"  # "completed", "incomplete", "error", "refusal", "empty"
    incomplete_reason: str | None = None  # e.g., "max_output_tokens", "content_filter"
    refusal: str | None = None
    error_type: str | None = None
    error_message: str | None = None
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    latency_seconds: float = 0.0
    api_surface: str = "responses"  # "responses" or "chat_completions" or "messages"
    max_output_tokens: int = 0
    output_item_types: list[str] = field(default_factory=list)
    content_part_types: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "completed" and bool(self.text.strip())


class LLMProvider(ABC):
    """Backend-agnostic chat interface. May be a no-op (NoLLMProvider)."""

    name: str = "llm"
    # response telemetry — populated after each complete() / complete_result() call
    last_response_id: str = ""
    last_latency_seconds: float = 0.0
    last_input_tokens: int = 0
    last_output_tokens: int = 0
    last_reasoning_tokens: int = 0

    @property
    def available(self) -> bool:
        return True

    @abstractmethod
    def complete(self, system: str, user: str, *, output_token_budget: int = 1024) -> str: ...

    def complete_result(
        self, system: str, user: str, *, output_token_budget: int = 1024
    ) -> ProviderResult:
        """Complete request returning a typed, normalized ProviderResult."""
        import time

        t0 = time.perf_counter()
        try:
            text = self.complete(system, user, output_token_budget=output_token_budget)
            latency = getattr(self, "last_latency_seconds", None) or (time.perf_counter() - t0)
            in_tok = getattr(self, "last_input_tokens", 0)
            out_tok = getattr(self, "last_output_tokens", 0)
            reas_tok = getattr(self, "last_reasoning_tokens", 0)
            resp_id = getattr(self, "last_response_id", "")
            return ProviderResult(
                text=text,
                provider=self.name,
                model=getattr(self, "model", ""),
                response_id=resp_id,
                status="completed" if text.strip() else "empty",
                usage=ProviderUsage(input_tokens=in_tok, output_tokens=out_tok, reasoning_tokens=reas_tok),
                latency_seconds=latency,
                max_output_tokens=output_token_budget,
            )
        except Exception as exc:
            latency = time.perf_counter() - t0
            return ProviderResult(
                text="",
                provider=self.name,
                model=getattr(self, "model", ""),
                response_id=getattr(self, "last_response_id", ""),
                status="error",
                error_type=type(exc).__name__,
                error_message=str(exc),
                latency_seconds=latency,
                max_output_tokens=output_token_budget,
            )

    def generate(
        self, prompt: str, *, system: str | None = None, metadata: dict | None = None
    ) -> str:
        """Common cross-provider interface (spec: prompt + optional system +
        optional metadata). Delegates to ``complete`` with semantic token budget;
        metadata is advisory and never contains raw confidential data."""
        meta = metadata or {}
        budget = int(meta.get("output_token_budget", meta.get("max_tokens", 1024)))
        try:
            return self.complete(system or "", prompt, output_token_budget=budget)
        except TypeError as te:
            if "output_token_budget" in str(te):
                return self.complete(system or "", prompt, max_tokens=budget)  # type: ignore[call-arg]
            raise

    def generate_request(self, req: GenerationRequest) -> str:
        """Execute a provider-neutral GenerationRequest."""
        try:
            return self.complete(
                req.system,
                req.prompt,
                output_token_budget=req.output_token_budget,
            )
        except TypeError as te:
            if "output_token_budget" in str(te):
                return self.complete(  # type: ignore[call-arg]
                    req.system,
                    req.prompt,
                    max_tokens=req.output_token_budget,
                )
            raise


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
    def append(self, record: Any, *args: Any, **kwargs: Any) -> Any:
        """Persist a record; return its content hash or record."""

    @abstractmethod
    def verify(self) -> bool:
        """Verify ledger integrity (hash chain)."""
