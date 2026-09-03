"""Tests for Runtime Profiles, Compute Router, Tracking, and Artifact SHA-256 (Combined Gate 7-9 Slice C)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from start.ai_engineering.tracking import MLflowExperimentTracker, NoOpExperimentTracker
from start.core.schemas import ComputeDevice
from start.portfolio.artifacts import (
    _hash_payload,
    render_scenario_pnl_waterfall_artifact,
)
from start.portfolio.contracts import PartitionContract, RepricingMethod, ScenarioResult
from start.providers.compute import detect_device
from start.runtime_profile import (
    ProfileViolation,
    RuntimeProfile,
    assert_provider_allowed,
)


def test_compute_router_priority():
    """Verify compute router evaluates CUDA -> MPS -> CPU priority deterministically with mocks."""
    # 1. CUDA available
    with patch("torch.cuda.is_available", return_value=True):
        dev = detect_device()
        assert dev == ComputeDevice.CUDA

    # 2. CUDA unavailable, MPS available
    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.backends.mps.is_available", return_value=True
    ):
        dev = detect_device()
        assert dev == ComputeDevice.MPS

    # 3. Neither available
    with patch("torch.cuda.is_available", return_value=False), patch(
        "torch.backends.mps.is_available", return_value=False
    ):
        dev = detect_device()
        assert dev == ComputeDevice.CPU


def test_runtime_profile_policy_enforcement():
    """Verify enterprise and airgapped runtime profiles reject public SaaS providers."""
    # public_demo allows public providers
    assert_provider_allowed("openai", profile=RuntimeProfile.PUBLIC_DEMO)
    assert_provider_allowed("anthropic", profile=RuntimeProfile.PUBLIC_DEMO)

    # enterprise allows internal gateway but strictly rejects public SaaS
    assert_provider_allowed("enterprise_llm_gateway", profile=RuntimeProfile.ENTERPRISE)
    assert_provider_allowed("gateway", profile=RuntimeProfile.ENTERPRISE)

    for prov in ("openai", "anthropic", "gemini", "deepseek", "grok", "huggingface"):
        with pytest.raises(ProfileViolation, match="does not permit|not permitted"):
            assert_provider_allowed(prov, profile=RuntimeProfile.ENTERPRISE)

    # airgapped rejects all SaaS providers and gateways
    for prov in ("openai", "anthropic", "gemini", "gateway", "enterprise_llm_gateway"):
        with pytest.raises(ProfileViolation, match="does not permit|not permitted"):
            assert_provider_allowed(prov, profile=RuntimeProfile.AIRGAPPED)

    # airgapped allows local/replay providers
    assert_provider_allowed("none", profile=RuntimeProfile.AIRGAPPED)
    assert_provider_allowed("replay", profile=RuntimeProfile.AIRGAPPED)


def test_experiment_tracker_abstraction():
    """Verify ExperimentTracker records metrics/params and handles fallback safely."""
    tracker = NoOpExperimentTracker()
    tracker.log_param("learning_rate", 0.001)
    tracker.log_metric("loss", 0.045, step=1)
    tracker.set_tag("version", "v4.4.0")

    assert tracker.get_logged_params()["learning_rate"] == 0.001
    assert tracker.get_logged_metrics()["loss"] == 0.045
    assert tracker.get_logged_tags()["version"] == "v4.4.0"

    # MLflow resilient fallback
    mlflow_tracker = MLflowExperimentTracker()
    mlflow_tracker.log_param("batch_size", 32)
    mlflow_tracker.log_metric("accuracy", 0.98)
    assert mlflow_tracker.get_logged_params()["batch_size"] == 32
    assert mlflow_tracker.get_logged_metrics()["accuracy"] == 0.98


def test_artifact_sha256_hardening_and_canonicalization():
    """Verify artifact semantic payload hash produces 64-hex SHA-256 and is invariant to key order."""
    payload_a = {
        "scenario_id": "SCEN-001",
        "scenario_return": -0.05,
        "contributions": {"AAPL": -0.03, "MSFT": -0.02},
    }
    # Different key insertion order
    payload_b = {
        "contributions": {"MSFT": -0.02, "AAPL": -0.03},
        "scenario_return": -0.05,
        "scenario_id": "SCEN-001",
    }
    payload_changed = {
        "scenario_id": "SCEN-001",
        "scenario_return": -0.06,  # modified
        "contributions": {"AAPL": -0.04, "MSFT": -0.02},
    }

    hash_a = _hash_payload(payload_a)
    hash_b = _hash_payload(payload_b)
    hash_c = _hash_payload(payload_changed)

    # 1. 64-hex lowercase SHA-256
    assert isinstance(hash_a, str)
    assert len(hash_a) == 64
    assert all(c in "0123456789abcdef" for c in hash_a)

    # 2. Key order invariance
    assert hash_a == hash_b

    # 3. Content modification sensitivity
    assert hash_a != hash_c


def test_artifact_record_renderer_emits_sha256():
    """Verify artifact renderer populates 64-hex semantic_payload_hash and semantic_hash_algorithm."""
    scen_res = ScenarioResult(
        scenario_id="SCEN-TEST",
        scenario_type="SYNTHETIC",
        repricing_method=RepricingMethod.LINEAR_RETURN.value,
        scenario_return=-0.08,
        scenario_loss=0.08,
        portfolio_value=1_000_000.0,
        scenario_pnl=-80_000.0,
        scenario_monetary_loss=80_000.0,
        asset_contributions={"AAPL": -0.05, "MSFT": -0.03},
        factor_contributions={},
        specific_contribution=None,
        group_contributions={},
        partition_contract=PartitionContract.EXHAUSTIVE_PARTITION.value,
        reconciliation_error=0.0,
        converged=True,
        limitations=("Test scenario waterfall.",),
        data_fingerprint="FP-001",
    )

    art = render_scenario_pnl_waterfall_artifact(scen_res, evidence_ids=("EV-SCEN-001",))
    assert len(art.semantic_payload_hash) == 64
    assert art.semantic_hash_algorithm == "sha256"
    assert art.to_dict()["semantic_payload_hash"] == art.semantic_payload_hash
    assert art.to_dict()["semantic_hash_algorithm"] == "sha256"


def test_mlflow_tracker_failure_status_exposure():
    """Verify configured MLflow tracking failure is surfaced explicitly via TrackingStatus and does not corrupt evidence."""
    from start.ai_engineering.tracking import TrackingStatus

    tracker = MLflowExperimentTracker()
    # Force mock failure on log_metric
    tracker._mlflow_available = True
    tracker._mlflow = object()  # object has no log_metric, raises AttributeError

    tracker.log_metric("loss", 0.05)
    # 1. Failure state is explicitly exposed
    assert tracker.tracking_status == TrackingStatus.FAILED_NON_BLOCKING
    assert tracker.last_error is not None
    assert "AttributeError" in tracker.last_error

    # 2. In-memory data is safely preserved
    assert tracker.get_logged_metrics()["loss"] == 0.05


def test_legacy_artifact_hash_recognition():
    """Verify 32-character legacy MD5 artifact hashes are recognized as md5_legacy and not mislabeled sha256."""
    from start.portfolio.artifacts import ArtifactRecord, parse_artifact_hash_algorithm

    assert parse_artifact_hash_algorithm("a" * 64) == "sha256"
    assert parse_artifact_hash_algorithm("b" * 32) == "md5_legacy"
    assert parse_artifact_hash_algorithm("c" * 16) == "unknown"

    legacy_dict = {
        "artifact_id": "ART-LEGACY-001",
        "artifact_type": "summary",
        "title": "Legacy Artifact",
        "test_id": "legacy.test",
        "evidence_ids": ["EV-LEGACY-001"],
        "data_fingerprint": "FP-LEGACY",
        "semantic_payload_hash": "d41d8cd98f00b204e9800998ecf8427e",  # 32-char MD5
    }
    loaded = ArtifactRecord.from_dict(legacy_dict)
    assert loaded.semantic_hash_algorithm == "md5_legacy"
    assert loaded.semantic_payload_hash == "d41d8cd98f00b204e9800998ecf8427e"


def test_airgapped_profile_rejects_network_plugins():
    """Verify AIRGAPPED profile rejects plugins requesting NETWORK or PROVIDER_ACCESS capabilities."""
    from start.registry.plugins import (
        PluginCapability,
        PluginManifest,
        PluginRegistry,
    )

    registry = PluginRegistry()
    network_plugin = PluginManifest(
        name="ExternalNetworkPack",
        version="1.0.0",
        description="Pulls remote market factors.",
        test_specs=(),
        declared_capabilities=(PluginCapability.NETWORK, PluginCapability.PROVIDER_ACCESS),
    )

    with pytest.raises(ProfileViolation, match="Plugin .* requires capabilities .* forbidden in airgapped"):
        registry.register_plugin(network_plugin, runtime_profile=RuntimeProfile.AIRGAPPED)
