"""AI-engineering layer: run all adapters and aggregate the results.

Executes each adapter's full contract (validate -> execute -> collect_artifacts
-> emit_evidence), streaming a visible status per adapter, and aggregates
artifacts, findings, and evidence for the dashboard and reports. Adapters that
are unavailable report ``not_installed`` with guidance and remain visible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from start.ai_engineering.adapters import build_adapters
from start.ai_engineering.base import Artifact, ExecutionResult
from start.core.schemas import TestResult
from start.governance.findings import Finding


@dataclass
class AIEngineeringReport:
    results: list[ExecutionResult]
    artifacts: list[Artifact] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    evidence: list[TestResult] = field(default_factory=list)
    descriptions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def available_count(self) -> int:
        return sum(1 for r in self.results if r.available)

    @property
    def total(self) -> int:
        return len(self.results)

    def control_surface(self) -> list[dict[str, Any]]:
        """Full control-surface rows (v2.1.1 Sections L/M): purpose, role,
        status, runtime, what it would do, expected outputs, install guidance."""
        by_name = {d["adapter"]: d for d in self.descriptions}
        rows = []
        for r in self.results:
            desc = by_name.get(r.adapter, {})
            rows.append({
                "adapter": r.adapter,
                "category": r.category,
                "purpose": desc.get("purpose", ""),
                "role": desc.get("role", ""),
                "status": r.status,
                "runtime_s": r.runtime_seconds,
                "would_do": desc.get("would_do", ""),
                "expected_outputs": desc.get("expected_outputs", []),
                "install_guidance": desc.get("install_guidance", ""),
                "artifacts": len(r.artifacts),
                "evidence": len(r.evidence),
            })
        return rows

    def summary_rows(self) -> list[dict[str, Any]]:
        return [
            {
                "adapter": r.adapter,
                "category": r.category,
                "status": r.status,
                "runtime_s": r.runtime_seconds,
                "artifacts": len(r.artifacts),
                "findings": len(r.findings),
                "evidence": len(r.evidence),
            }
            for r in self.results
        ]


def run_ai_engineering_layer(
    context: dict[str, Any] | None = None,
    *,
    output_root: str = "start_output",
    on_adapter: Callable[[ExecutionResult], None] | None = None,
    on_adapter_start: Callable[[str, str], None] | None = None,
) -> AIEngineeringReport:
    """Run every adapter and aggregate. ``on_adapter`` is called after each for
    visible streaming progress; ``on_adapter_start`` is called *before* each
    adapter runs with (adapter_name, activity) so the user sees live activity
    announcements like "[OPA] Validating policy controls…" (#3)."""
    context = context or {}
    adapters = build_adapters(output_root=output_root)
    results: list[ExecutionResult] = []
    artifacts: list[Artifact] = []
    findings: list[Finding] = []
    evidence: list[TestResult] = []
    descriptions: list[dict[str, Any]] = []

    for adapter in adapters:
        descriptions.append(adapter.describe())
        if on_adapter_start:
            on_adapter_start(adapter.name, getattr(adapter, "activity", "Running checks"))
        result = adapter.execute(context)
        artifacts.extend(adapter.collect_artifacts(result))
        findings.extend(result.findings)
        evidence.extend(adapter.emit_evidence(result))
        results.append(result)
        if on_adapter:
            on_adapter(result)

    return AIEngineeringReport(
        results=results, artifacts=artifacts, findings=findings, evidence=evidence,
        descriptions=descriptions,
    )
