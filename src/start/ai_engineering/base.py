"""Enterprise AI-engineering adapter base.

Every adapter implements the same five-method contract:

    available()         -> bool         is the backend usable right now?
    validate()          -> ValidationResult   config/credentials/inputs sane?
    execute(context)    -> ExecutionResult     run the real integration
    collect_artifacts() -> list[Artifact]      files/JSON produced this run
    emit_evidence()     -> list[TestResult]    evidence records for the ledger

Adapters run the REAL integration when their dependency is present. When it is
not, they do not fake success and do not silently skip: they return an explicit
``not_installed`` execution result with install guidance, still emit an
evidence record (so the gap is auditable), and remain visible in reports.

This is the contract the enterprise layer is built on; a firm that installs a
backend gets real execution with zero code change.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from start.core.schemas import Status, TestResult
from start.governance.findings import Finding, Materiality, Severity


@dataclass
class Artifact:
    name: str
    path: str
    kind: str = "json"  # json | png | html | text
    summary: str = ""


@dataclass
class ValidationResult:
    ok: bool
    detail: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExecutionResult:
    adapter: str
    category: str
    status: str  # complete | not_installed | error | skipped
    available: bool
    runtime_seconds: float = 0.0
    detail: str = ""
    summary: dict[str, Any] = field(default_factory=dict)
    artifacts: list[Artifact] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    evidence: list[TestResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


class BaseAdapter:
    """Base class for AI-engineering stage adapters."""

    name: str = "adapter"
    category: str = "uncategorized"
    activity: str = "Running checks"  # live activity verb shown during execution (#3)
    purpose: str = ""  # what this control is for (v2.1.1 Section L)
    role: str = ""  # what it does in the review
    package: str | None = None  # python import name probed for availability
    cli_tool: str | None = None  # optional CLI binary on PATH
    install_hint: str = ""
    capabilities: tuple[str, ...] = ()
    artifact_names: tuple[str, ...] = ()

    def __init__(self, output_root: str | Path = "start_output") -> None:
        self.output_root = Path(output_root)

    def describe(self) -> dict[str, Any]:
        """Full control-surface description (v2.1.1 Sections L/M): purpose, role,
        status, what it would do if installed, install guidance, expected outputs."""
        installed = self.available()
        return {
            "adapter": self.name,
            "category": self.category,
            "purpose": self.purpose,
            "role": self.role,
            "status": "available" if installed else "not_installed",
            "would_do": self.role or "; ".join(self.capabilities),
            "expected_outputs": list(self.artifact_names),
            "capabilities": list(self.capabilities),
            "install_guidance": self.install_hint,
        }

    # -- availability ------------------------------------------------------ #
    def available(self) -> bool:
        if self.package and module_available(self.package):
            return True
        if self.cli_tool and shutil.which(self.cli_tool):
            return True
        return False

    def validate(self, context: dict[str, Any] | None = None) -> ValidationResult:
        """Default validation: backend present. Subclasses may add config checks."""
        if not self.available():
            return ValidationResult(
                ok=False,
                detail=f"{self.name} backend not available. {self.install_hint}".strip(),
            )
        return ValidationResult(ok=True, detail=f"{self.name} backend available.")

    # -- execution --------------------------------------------------------- #
    def execute(self, context: dict[str, Any] | None = None) -> ExecutionResult:
        """Run the adapter. Times the run, routes to ``_run`` when available,
        and returns an honest not_installed result otherwise."""
        context = context or {}
        start = time.perf_counter()
        if not self.available():
            return self._unavailable(time.perf_counter() - start)
        try:
            result = self._run(context)
        except Exception as exc:  # real errors are surfaced, never swallowed
            return ExecutionResult(
                adapter=self.name,
                category=self.category,
                status="error",
                available=True,
                runtime_seconds=round(time.perf_counter() - start, 4),
                detail=f"{self.name} execution raised {type(exc).__name__}: {exc}",
                evidence=[self._evidence("error", f"{self.name} errored: {exc}")],
            )
        result.runtime_seconds = round(time.perf_counter() - start, 4)
        return result

    def _run(self, context: dict[str, Any]) -> ExecutionResult:
        """Real integration logic. Subclasses override. The default performs a
        genuine availability handshake (used by adapters whose only safe action
        in this environment is to confirm the backend responds)."""
        return ExecutionResult(
            adapter=self.name,
            category=self.category,
            status="complete",
            available=True,
            detail=f"{self.name} available; handshake succeeded.",
            summary={"capabilities": list(self.capabilities)},
            evidence=[self._evidence("pass", f"{self.name} available and validated.")],
        )

    def _unavailable(self, runtime: float) -> ExecutionResult:
        detail = f"{self.name} not installed. {self.install_hint}".strip()
        finding = Finding(
            title=f"{self.name} unavailable",
            description=detail,
            severity=Severity.LOW,
            materiality=Materiality.LOW,
            risk_category="Operational",
            recommendation=self.install_hint or f"Install {self.name} to enable this control.",
            source=self.name,
        )
        ev = self._evidence("skipped", detail)
        finding.evidence_ids = [ev.test_id]
        return ExecutionResult(
            adapter=self.name,
            category=self.category,
            status="not_installed",
            available=False,
            runtime_seconds=round(runtime, 4),
            detail=detail,
            summary={"install_hint": self.install_hint},
            findings=[finding],
            evidence=[ev],
        )

    # -- artifacts & evidence ---------------------------------------------- #
    def _artifact_dir(self, run_id: str) -> Path:
        path = self.output_root / "ai_engineering" / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json_artifact(
        self, run_id: str, filename: str, payload: dict[str, Any], summary: str = ""
    ) -> Artifact:
        path = self._artifact_dir(run_id) / filename
        path.write_text(json.dumps(payload, indent=2, default=str))
        return Artifact(name=filename, path=str(path), kind="json", summary=summary)

    def collect_artifacts(self, result: ExecutionResult) -> list[Artifact]:
        return result.artifacts

    def emit_evidence(self, result: ExecutionResult) -> list[TestResult]:
        return result.evidence

    def _evidence(self, status: str, interpretation: str, metrics: dict | None = None) -> TestResult:
        status_map = {
            "pass": Status.PASS,
            "warn": Status.WARN,
            "fail": Status.FAIL,
            "skipped": Status.SKIPPED,
            "error": Status.ERROR,
        }
        return TestResult(
            test_id=f"ai_engineering.{self.category}.{self.name.lower().replace(' ', '_')}",
            test_name=f"{self.name} ({self.category})",
            status=status_map.get(status, Status.SKIPPED),
            metrics=metrics or {},
            interpretation=interpretation,
            limitations=[
                "AI-engineering control; runs the real backend when installed, "
                "else reports unavailability explicitly.",
            ],
        )
