import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from start.registry import list_tests
from start.runtime.scenarios import (
    get_scenario,
    list_scenarios,
)
from start.web.app import app

client = TestClient(app)


def test_scenario_catalog_inventory():
    """Verify mechanical scenario inventory requirements."""
    scenarios = list_scenarios()
    assert len(scenarios) == 7

    classifications = {s.id: s.classification for s in scenarios}
    # Check strict classifications
    assert classifications["institutional_credit_v1"] == "CANONICAL_EXECUTION_CONTEXT"
    assert classifications["deep_learning_v1"] == "CANONICAL_EXECUTION_CONTEXT"
    assert classifications["institutional_market_v1"] == "CANONICAL_EXECUTION_CONTEXT"
    assert classifications["synthetic_aml_imbalanced"] == "SUPPORTED_DATA_SCENARIO"
    assert classifications["synthetic_leakage_proxy"] == "SUPPORTED_DATA_SCENARIO"
    assert classifications["market_regime_shift"] == "SUPPORTED_DATA_SCENARIO"
    assert classifications["treasury_short_rate_cir"] == "GENERATOR_ONLY"

    # Zero unsupported scenarios
    unsupported = [s for s in scenarios if s.classification == "UNSUPPORTED"]
    assert len(unsupported) == 0

    # Generator-only semantics preserved: Treasury CIR maps to institutional_market_v1
    cir = get_scenario("treasury_short_rate_cir")
    assert cir is not None
    assert cir.classification == "GENERATOR_ONLY"
    assert cir.compatible_context_id == "institutional_market_v1"


def test_test_catalog_schema_discovered():
    """Verify authoritative 79-test catalog schema from list_tests()."""
    tests = list_tests()
    assert len(tests) == 79

    # No duplicates
    test_ids = [t.test_id for t in tests]
    assert len(test_ids) == len(set(test_ids))

    # All tests contain non-empty canonical fields
    for t in tests:
        assert t.test_id
        assert t.name
        assert t.family
        assert t.description and len(t.description) > 0
        assert t.context_type


def test_api_scenarios_endpoint():
    """GET /api/v1/scenarios returns all 7 scenarios."""
    resp = client.get("/api/v1/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 7
    ids = [s["id"] for s in data]
    assert "institutional_credit_v1" in ids
    assert "treasury_short_rate_cir" in ids


def test_api_tests_endpoint():
    """GET /api/v1/tests returns exactly 79 canonical tests."""
    resp = client.get("/api/v1/tests")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 79
    # Check domain tagging: Predictive ML (54) + Market Quant (25) = 79
    # Treasury is generator-only and has 0 registered test methods
    domains = set(t["domain"] for t in data)
    assert "predictive_ml" in domains
    assert "quantitative_finance" in domains
    assert len(data) == 79


def test_api_scenario_eda_descriptive_only():
    """Verify EDA is descriptive-only and does NOT create a run or write evidence."""
    # Test tabular credit scenario
    resp = client.get("/api/v1/scenarios/institutional_credit_v1/eda")
    assert resp.status_code == 200
    eda = resp.json()
    assert eda["domain"] in ("tabular", "predictive_ml")
    assert "provenance" in eda
    assert eda["provenance"]["profiling_operation"] == "deterministic_descriptive_profiling"
    assert eda["provenance"]["seed"] == 42
    assert "target_distribution" in eda
    assert "feature_moments" in eda
    assert "correlation" in eda

    # Verify Deep Learning segregation: model_fixture_context vs data_profile
    resp_dl = client.get("/api/v1/scenarios/deep_learning_v1/eda")
    assert resp_dl.status_code == 200
    eda_dl = resp_dl.json()
    assert eda_dl["domain"] == "deep_learning"
    assert "model_fixture_context" in eda_dl
    assert "data_profile" in eda_dl
    assert "TabularDLClassifier" in eda_dl["model_fixture_context"]["architecture"]
    assert eda_dl["model_fixture_context"]["hidden_dims"] == [64, 32]

    # Verify Market Quant scenario
    resp_mkt = client.get("/api/v1/scenarios/institutional_market_v1/eda")
    assert resp_mkt.status_code == 200
    eda_mkt = resp_mkt.json()
    assert eda_mkt["domain"] in ("market_risk", "market_quant", "quantitative_finance")
    assert "portfolio_composition" in eda_mkt
    assert "factor_coverage" in eda_mkt

    # Verify Treasury generator-only scenario
    resp_cir = client.get("/api/v1/scenarios/treasury_short_rate_cir/eda")
    assert resp_cir.status_code == 200
    eda_cir = resp_cir.json()
    assert eda_cir["domain"] in ("treasury", "treasury_short_rate")
    assert eda_cir["provenance"]["classification"] == "GENERATOR_ONLY"
    assert "path_sample" in eda_cir
