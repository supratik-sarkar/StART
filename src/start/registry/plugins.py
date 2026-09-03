"""Plugin and capability management layer for StART test packs.

Invariants:
- Canonical built-in registry census (79/79/0) is strictly invariant.
- Plugins are registered in an isolated overlay (PluginRegistry / ComposedRegistryView).
- Plugins cannot shadow built-in test IDs (fails closed).
- Airgapped profile strictly rejects plugins requesting NETWORK or PROVIDER_ACCESS capabilities.
- Plugin loading failures are isolated and never corrupt the core built-in registry.
- Resource budgets distinguish enforced timeouts/row-limits from MEMORY_ESTIMATE_ONLY.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from start.core.hashing import canonical_json, sha256_hex
from start.registry import TestSpec, list_tests, load_builtin_tests
from start.runtime_profile import RuntimeProfile


class PluginCapability(StrEnum):
    """Declared capabilities required by an external plugin or test pack."""

    LOCAL_FILESYSTEM = "LOCAL_FILESYSTEM"
    NETWORK = "NETWORK"
    GPU = "GPU"
    DATABASE = "DATABASE"
    PROVIDER_ACCESS = "PROVIDER_ACCESS"
    ARTIFACT_OUTPUT = "ARTIFACT_OUTPUT"


@dataclass(frozen=True)
class PluginManifest:
    """Metadata specification and declared capabilities for a StART plugin."""

    name: str
    version: str
    description: str
    test_specs: tuple[TestSpec, ...]
    declared_capabilities: tuple[PluginCapability, ...]
    required_context_type: str = "TestContext"
    optional_dependencies: tuple[str, ...] = ()
    plugin_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "test_ids": [s.test_id for s in self.test_specs],
            "declared_capabilities": [c.value for c in self.declared_capabilities],
            "required_context_type": self.required_context_type,
            "optional_dependencies": list(self.optional_dependencies),
            "plugin_fingerprint": self.plugin_fingerprint,
        }

    def canonical_fingerprint(self) -> str:
        d = self.to_dict()
        d.pop("plugin_fingerprint", None)
        return sha256_hex(canonical_json(d))


@dataclass(frozen=True)
class ResourceBudget:
    """Resource constraints and execution bounds for test execution."""

    max_wall_time_sec: float | None = None
    max_rows: int | None = None
    estimated_memory_mb: float | None = None
    memory_enforcement_type: str = "MEMORY_ESTIMATE_ONLY"
    concurrency_limit: int = 1
    retry_limit: int = 0

    def enforce_row_budget(self, row_count: int) -> bool:
        """Validate row count against maximum row budget."""
        if self.max_rows is not None and row_count > self.max_rows:
            raise ValueError(
                f"Row budget exceeded: {row_count} rows exceeds maximum allowed {self.max_rows}."
            )
        return True

    def enforce_timeout(self, duration_sec: float) -> bool:
        """Validate execution duration against wall-time budget."""
        if self.max_wall_time_sec is not None and duration_sec > self.max_wall_time_sec:
            raise TimeoutError(
                f"Wall-time budget exceeded: execution duration {duration_sec:.2f}s "
                f"exceeds limit {self.max_wall_time_sec:.2f}s."
            )
        return True


class PluginRegistry:
    """Isolated overlay registry for external plugin test packs."""

    def __init__(self) -> None:
        self._plugins: dict[str, PluginManifest] = {}
        self._plugin_tests: dict[str, TestSpec] = {}

    def register_plugin(
        self,
        manifest: PluginManifest,
        runtime_profile: RuntimeProfile | str = RuntimeProfile.PUBLIC_DEMO,
    ) -> None:
        """Register a plugin manifest after verifying capabilities and test ID uniqueness."""
        load_builtin_tests()
        from start.registry import _REGISTRY as BUILTIN_REGISTRY

        # 1. Verify capability permissions against active runtime profile
        valid, reasons = self.verify_capabilities(manifest, runtime_profile)
        if not valid:
            from start.runtime_profile import ProfileViolation

            raise ProfileViolation(
                f"Plugin '{manifest.name}' requires capabilities {manifest.declared_capabilities} "
                f"forbidden in {runtime_profile}: {reasons}"
            )

        # 2. Check test ID collisions (fail-closed against built-ins and existing plugins)
        for spec in manifest.test_specs:
            if spec.test_id in BUILTIN_REGISTRY:
                raise ValueError(
                    f"Plugin test_id collision: '{spec.test_id}' shadows a built-in StART test. "
                    f"Plugins cannot overwrite built-in tests."
                )
            if spec.test_id in self._plugin_tests:
                raise ValueError(
                    f"Plugin test_id collision: '{spec.test_id}' is already registered by another plugin."
                )

        # 3. Store plugin
        self._plugins[manifest.name] = manifest
        for spec in manifest.test_specs:
            self._plugin_tests[spec.test_id] = spec

    def verify_capabilities(
        self,
        manifest: PluginManifest,
        runtime_profile: RuntimeProfile | str,
    ) -> tuple[bool, list[str]]:
        """Verify plugin declared capabilities against runtime profile policy."""
        prof_str = (
            runtime_profile.value
            if isinstance(runtime_profile, RuntimeProfile)
            else str(runtime_profile).lower()
        )
        violations: list[str] = []

        if prof_str == RuntimeProfile.AIRGAPPED.value:
            forbidden = {PluginCapability.NETWORK, PluginCapability.PROVIDER_ACCESS}
            for cap in manifest.declared_capabilities:
                if cap in forbidden:
                    violations.append(
                        f"Capability '{cap.value}' is prohibited under airgapped profile."
                    )

        if prof_str == RuntimeProfile.ENTERPRISE.value:
            if PluginCapability.PROVIDER_ACCESS in manifest.declared_capabilities:
                violations.append(
                    "Direct provider access capability requires enterprise gateway routing."
                )

        return (len(violations) == 0, violations)

    def get_plugin(self, name: str) -> PluginManifest | None:
        return self._plugins.get(name)

    def get_test(self, test_id: str) -> TestSpec | None:
        return self._plugin_tests.get(test_id)

    def list_plugin_tests(self) -> list[TestSpec]:
        return sorted(self._plugin_tests.values(), key=lambda s: s.test_id)

    def list_plugins(self) -> list[PluginManifest]:
        return list(self._plugins.values())


class ComposedRegistryView:
    """Composed read-only view uniting canonical built-ins with an active plugin registry overlay."""

    def __init__(self, plugin_registry: PluginRegistry | None = None) -> None:
        self._plugin_registry = plugin_registry or PluginRegistry()

    def get_test(self, test_id: str) -> TestSpec:
        load_builtin_tests()
        from start.registry import _REGISTRY as BUILTIN_REGISTRY

        if test_id in BUILTIN_REGISTRY:
            return BUILTIN_REGISTRY[test_id]

        plugin_spec = self._plugin_registry.get_test(test_id)
        if plugin_spec is not None:
            return plugin_spec

        raise KeyError(f"Unknown test_id in composed registry: {test_id}")

    def list_tests(self, family: str | None = None) -> list[TestSpec]:
        load_builtin_tests()
        builtins = list_tests(family=family)
        plugin_tests = [
            s
            for s in self._plugin_registry.list_plugin_tests()
            if family is None or s.family == family
        ]
        return sorted(builtins + plugin_tests, key=lambda s: s.test_id)

    def list_families(self) -> list[str]:
        builtins = list_tests()
        plugin_tests = self._plugin_registry.list_plugin_tests()
        return sorted({s.family for s in builtins + plugin_tests})
