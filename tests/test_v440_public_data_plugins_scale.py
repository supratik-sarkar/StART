"""Tests for Public-Safe Data, Plugin Registry, and Scale Architecture (Combined Gate 7-9 Slice B)."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from start.core.schemas import Status, TestResult
from start.data.adapters import (
    ChunkClassification,
    DataFrameAdapter,
    DataSchema,
    LocalFileAdapter,
    SQLDataSourceAdapter,
    WarehouseAdapter,
    execute_chunked,
)
from start.registry import TestSpec, list_tests, load_builtin_tests
from start.registry.plugins import (
    ComposedRegistryView,
    PluginCapability,
    PluginManifest,
    PluginRegistry,
    ResourceBudget,
)
from start.runtime_profile import ProfileViolation, RuntimeProfile


def test_dataschema_validation_and_fail_closed():
    """Verify deterministic DataSchema enforces required columns, types, finiteness, and uniqueness."""
    schema = DataSchema(
        columns={"returns": "float", "ticker": "str"},
        required_columns=("returns", "ticker"),
        nullable_columns=(),
        unique_columns=("ticker",),
        require_finite=True,
    )

    # Valid dataframe
    df_valid = pd.DataFrame({"returns": [0.01, -0.02, 0.05], "ticker": ["AAPL", "MSFT", "GOOG"]})
    valid, errors = schema.validate(df_valid)
    assert valid is True
    assert len(errors) == 0

    # Invalid: missing required column
    df_missing = pd.DataFrame({"returns": [0.01, -0.02]})
    valid, errors = schema.validate(df_missing)
    assert valid is False
    assert any("Missing required column: 'ticker'" in e for e in errors)

    # Invalid: non-finite (Inf) value
    df_inf = pd.DataFrame({"returns": [0.01, np.inf], "ticker": ["AAPL", "MSFT"]})
    valid, errors = schema.validate(df_inf)
    assert valid is False
    assert any("contains non-finite" in e for e in errors)

    # Invalid: duplicate unique column
    df_dup = pd.DataFrame({"returns": [0.01, 0.02], "ticker": ["AAPL", "AAPL"]})
    valid, errors = schema.validate(df_dup)
    assert valid is False
    assert any("requires uniqueness" in e for e in errors)


def test_dataframe_adapter_and_deterministic_fingerprint():
    """Verify DataFrameAdapter computes deterministic SHA-256 fingerprint and supports exact chunking."""
    df = pd.DataFrame({"asset": ["A", "B", "C", "D"], "weight": [0.4, 0.3, 0.2, 0.1]})
    adapter = DataFrameAdapter(df)

    assert adapter.row_count == 4
    assert adapter.column_count == 2
    assert len(adapter.data_fingerprint) == 64

    # Identical content yields identical fingerprint
    adapter2 = DataFrameAdapter(df.copy())
    assert adapter2.data_fingerprint == adapter.data_fingerprint

    # Chunked iteration
    chunks = list(adapter.read_chunks(chunk_size=2))
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 2


def test_local_file_adapter_and_chunked_reduction():
    """Verify LocalFileAdapter and execute_chunked reduction yields identical semantic result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "synthetic_returns.csv"
        df_full = pd.DataFrame({
            "asset_ret": [0.01 * (i + 1) for i in range(100)],
            "market_ret": [0.005 * (i + 1) for i in range(100)],
        })
        df_full.to_csv(csv_path, index=False)

        adapter = LocalFileAdapter(csv_path)
        assert adapter.row_count == 100
        assert adapter.column_count == 2

        # Non-chunked sum
        total_sum_non_chunked = float(adapter.read()["asset_ret"].sum())

        # Chunked sum reduction
        total_sum_chunked = execute_chunked(
            adapter=adapter,
            chunk_size=25,
            map_fn=lambda c: float(c["asset_ret"].sum()),
            reduce_fn=lambda parts: float(sum(parts)),
        )

        assert pytest.approx(total_sum_chunked, abs=1e-8) == total_sum_non_chunked


def test_sql_datasource_adapter_in_memory_sqlite():
    """Verify SQLDataSourceAdapter works with in-memory SQLite fixture without external DB credentials."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE portfolio_holdings (asset TEXT, qty REAL);")
    conn.execute("INSERT INTO portfolio_holdings VALUES ('AAPL', 100.0), ('MSFT', 50.0);")
    conn.commit()

    adapter = SQLDataSourceAdapter(conn, "portfolio_holdings")
    assert adapter.row_count == 2
    assert adapter.column_count == 2
    df = adapter.read()
    assert list(df["asset"]) == ["AAPL", "MSFT"]


def test_chunkable_classification_enforcement():
    """Verify execute_chunked raises ValueError if adapter is NOT_CHUNKABLE."""
    warehouse = WarehouseAdapter("https://public-warehouse.example.com", "public_table")
    assert warehouse.chunk_classification == ChunkClassification.NOT_CHUNKABLE

    with pytest.raises(ValueError, match="classified as NOT_CHUNKABLE"):
        execute_chunked(
            adapter=warehouse,
            chunk_size=10,
            map_fn=lambda c: len(c),
            reduce_fn=lambda parts: sum(parts),
        )


def test_plugin_registration_overlay_and_builtins_invariance():
    """Verify PluginRegistry isolates plugins and built-in census remains strictly 79/79/0."""
    load_builtin_tests()
    builtins_before = list_tests()
    assert len(builtins_before) == 79
    assert len(set(s.test_id for s in builtins_before)) == 79

    # Create synthetic plugin test spec
    dummy_spec = TestSpec(
        test_id="plugin.synthetic_custom_risk",
        family="custom_risk",
        name="Custom Synthetic Risk Metric",
        fn=lambda ctx: TestResult(test_id="plugin.synthetic_custom_risk", test_name="Custom", status=Status.PASS),
    )
    manifest = PluginManifest(
        name="synthetic_risk_plugin",
        version="1.0.0",
        description="Public synthetic risk test pack.",
        test_specs=(dummy_spec,),
        declared_capabilities=(PluginCapability.LOCAL_FILESYSTEM, PluginCapability.ARTIFACT_OUTPUT),
    )

    registry = PluginRegistry()
    registry.register_plugin(manifest, runtime_profile=RuntimeProfile.PUBLIC_DEMO)

    composed = ComposedRegistryView(registry)
    all_tests = composed.list_tests()
    assert len(all_tests) == 80
    assert composed.get_test("plugin.synthetic_custom_risk").test_id == "plugin.synthetic_custom_risk"

    # Invariant: built-in list_tests() remains 79
    builtins_after = list_tests()
    assert len(builtins_after) == 79
    assert len(set(s.test_id for s in builtins_after)) == 79


def test_plugin_collision_rejection():
    """Verify plugin registration fails closed if attempting to shadow a built-in test ID."""
    load_builtin_tests()

    # Shadowing existing built-in "portfolio.mean_variance"
    shadow_spec = TestSpec(
        test_id="portfolio.mean_variance",
        family="portfolio",
        name="Shadow Mean Variance",
        fn=lambda ctx: TestResult(test_id="portfolio.mean_variance", test_name="Shadow", status=Status.PASS),
    )
    manifest = PluginManifest(
        name="shadow_plugin",
        version="1.0.0",
        description="Attempt to shadow core test.",
        test_specs=(shadow_spec,),
        declared_capabilities=(),
    )

    registry = PluginRegistry()
    with pytest.raises(ValueError, match="shadows a built-in StART test"):
        registry.register_plugin(manifest)


def test_airgapped_profile_rejects_network_plugins():
    """Verify airgapped runtime profile strictly rejects plugins declaring NETWORK capability."""
    net_spec = TestSpec(
        test_id="plugin.network_fetcher",
        family="external",
        name="Network Fetcher",
        fn=lambda ctx: TestResult(test_id="plugin.network_fetcher", test_name="Net", status=Status.PASS),
    )
    manifest = PluginManifest(
        name="net_plugin",
        version="1.0.0",
        description="Plugin requiring external network.",
        test_specs=(net_spec,),
        declared_capabilities=(PluginCapability.NETWORK,),
    )

    registry = PluginRegistry()
    with pytest.raises(ProfileViolation, match="prohibited under airgapped profile"):
        registry.register_plugin(manifest, runtime_profile=RuntimeProfile.AIRGAPPED)


def test_resource_budget_enforcement():
    """Verify ResourceBudget enforces row limits and timeouts, and accurately reports MEMORY_ESTIMATE_ONLY."""
    budget = ResourceBudget(
        max_wall_time_sec=2.0,
        max_rows=500,
        estimated_memory_mb=128.0,
    )
    assert budget.memory_enforcement_type == "MEMORY_ESTIMATE_ONLY"

    # Row budget pass & fail
    assert budget.enforce_row_budget(450) is True
    with pytest.raises(ValueError, match="Row budget exceeded"):
        budget.enforce_row_budget(600)

    # Timeout pass & fail
    assert budget.enforce_timeout(1.2) is True
    with pytest.raises(TimeoutError, match="Wall-time budget exceeded"):
        budget.enforce_timeout(2.5)
