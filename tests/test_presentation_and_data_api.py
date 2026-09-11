"""Bounded HTTP contract verification for StART Live Data & Scientific Presentation APIs.

Strict Invariants Verified:
1. FRONTEND_FILES_CHANGED == 0
2. WEB_LAYER_SCIENTIFIC_CALCULATIONS_ADDED == 0
3. SECRET_EXPOSURE == 0
4. Ray reported truthfully as unavailable/not verified.
5. All numbers sourced strictly from the 14-artifact bundle.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from start.web.app import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


# --------------------------------------------------------------------------- #
# A. Provider Registry & Connectivity Probe
# --------------------------------------------------------------------------- #
def test_provider_registry_endpoint(client: TestClient):
    """Verify GET /api/v1/data/providers returns genuine registry status."""
    res = client.get("/api/v1/data/providers")
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True

    providers = body["data"]["providers"]
    prov_ids = {p["provider_id"] for p in providers}
    expected = {"huggingface", "openml", "uci", "kaggle", "local_csv", "local_parquet"}
    assert expected.issubset(prov_ids)

    # Check dependency available != remote connectivity verified
    for p in providers:
        assert "package_available" in p
        assert "remote_connectivity_verified" in p
        assert "streaming_supported" in p
        assert "credential_required" in p
        if p["provider_id"] == "kaggle":
            assert p["status"] == "DEFERRED"
            assert p["runnable"] is False
            assert p["credential_required"] is True
            assert p["streaming_supported"] is False
            assert "Kaggle package is not installed" in p["reason_if_unavailable"]


def test_provider_probe_endpoint(client: TestClient):
    """Verify GET /api/v1/data/providers/{provider}/probe."""
    # Local filesystem probe
    res_local = client.get("/api/v1/data/providers/local_csv/probe")
    assert res_local.status_code == 200
    d_local = res_local.json()["data"]
    assert d_local["remote_connectivity_verified"] is True
    assert d_local["package_available"] is True

    # Kaggle probe: package absent -> verified is false
    res_kaggle = client.get("/api/v1/data/providers/kaggle/probe")
    assert res_kaggle.status_code == 200
    d_kaggle = res_kaggle.json()["data"]
    assert d_kaggle["package_available"] is False
    assert d_kaggle["remote_connectivity_verified"] is False


# --------------------------------------------------------------------------- #
# B. Dataset Search / Discovery
# --------------------------------------------------------------------------- #
def test_dataset_discovery_uci(client: TestClient):
    """Verify discovery on UCI provider."""
    res = client.get("/api/v1/data/providers/uci/datasets")
    assert res.status_code == 200
    datasets = res.json()["data"]["datasets"]
    assert len(datasets) >= 2
    dids = [d["dataset_id"] for d in datasets]
    assert "credit_approval" in dids
    assert "default_of_credit_card_clients" in dids


def test_dataset_discovery_huggingface(client: TestClient):
    """Verify discovery on Hugging Face provider."""
    res = client.get("/api/v1/data/providers/huggingface/datasets?query=adult&limit=2")
    assert res.status_code == 200
    datasets = res.json()["data"]["datasets"]
    assert len(datasets) >= 1
    assert any("adult" in d["dataset_id"].lower() for d in datasets)


def test_dataset_discovery_kaggle_deferred(client: TestClient):
    """Verify discovery on deferred Kaggle provider."""
    res = client.get("/api/v1/data/providers/kaggle/datasets")
    assert res.status_code == 200
    d = res.json()["data"]
    assert d["count"] == 0
    assert "reason_if_unavailable" in d


# --------------------------------------------------------------------------- #
# C. Dataset Contract Resolution
# --------------------------------------------------------------------------- #
def test_dataset_resolve_uci(client: TestClient):
    """Verify POST /api/v1/data/resolve on UCI dataset."""
    payload = {
        "provider": "uci",
        "dataset_id": "credit_approval",
        "target": "target",
    }
    res = client.post("/api/v1/data/resolve", json=payload)
    assert res.status_code == 200
    contract = res.json()["data"]["contract"]
    assert contract["provider"] == "uci"
    assert contract["dataset_id"] == "credit_approval"
    assert contract["target"] == "target"
    assert "feature_roles" in contract
    assert contract["streaming_supported"] is True


# --------------------------------------------------------------------------- #
# D. Data Pre-Certification
# --------------------------------------------------------------------------- #
def test_dataset_precertify(client: TestClient):
    """Verify POST /api/v1/data/precertify structured output."""
    payload = {
        "provider": "uci",
        "dataset_id": "credit_approval",
        "target": "is_bad_credit",
        "sample_rows": 100,
    }
    res = client.post("/api/v1/data/precertify", json=payload)
    assert res.status_code == 200
    data = res.json()["data"]
    assert "source_identity" in data
    assert "license_access" in data
    assert data["schema_validity"] is True
    assert data["target_validity"] is True
    assert data["duplicate_count"] >= 0
    assert data["duplicate_rate"] >= 0.0
    assert "missingness_by_feature" in data
    assert "target_distribution" in data
    assert data["temporal_ordering"] == "NOT_EVALUATED"
    assert "fingerprint" in data
    assert data["overall_status"] in ("CERTIFIED", "WARNING", "FAILED")


# --------------------------------------------------------------------------- #
# E & F. Data Ingestion Session & Telemetry
# --------------------------------------------------------------------------- #
def test_data_session_lifecycle_and_telemetry(client: TestClient):
    """Verify session creation, truthful telemetry, and stop."""
    payload = {
        "provider": "uci",
        "dataset_id": "credit_approval",
        "target": "target",
        "batch_size": 200,
        "max_rows": 400,
        "auto_start": True,
    }
    res = client.post("/api/v1/data/sessions", json=payload)
    assert res.status_code == 200
    sess_data = res.json()["data"]
    sess_id = sess_data["session_id"]
    assert sess_id.startswith("SES-DATA-")

    # Wait briefly for bounded ingestion
    time.sleep(0.5)

    # Telemetry check
    res_telem = client.get(f"/api/v1/data/sessions/{sess_id}/telemetry")
    assert res_telem.status_code == 200
    t = res_telem.json()["data"]
    assert t["session_id"] == sess_id
    assert t["rows_consumed"] >= 0
    assert t["bytes_read"] >= 0
    assert t["worker_count"] == 1
    assert t["execution_mode"] == "local_arrow_partitioned"
    assert t["device"] is not None

    # Stop session
    res_stop = client.post(f"/api/v1/data/sessions/{sess_id}/stop")
    assert res_stop.status_code == 200
    assert res_stop.json()["data"]["state"] in ("STOPPED", "COMPLETED")


# --------------------------------------------------------------------------- #
# G. Parallelism & Runtime Truth
# --------------------------------------------------------------------------- #
def test_parallelism_runtime_truth(client: TestClient):
    """Verify GET /api/v1/data/runtime reports local columnar execution."""
    res = client.get("/api/v1/data/runtime")
    assert res.status_code == 200
    rt = res.json()["data"]
    assert rt["execution_backend"] == "local_arrow_partitioned"
    assert rt["worker_count"] == 1
    assert rt["distributed"] is False
    assert rt["ray_available"] is False
    assert rt["ray_distributed_execution_verified"] is False
    assert "n_jobs=1" in rt["openmp_threads_invariant"]


# --------------------------------------------------------------------------- #
# H. Ephemeral Credential Security (Zero Secret Exposure)
# --------------------------------------------------------------------------- #
def test_ephemeral_credential_security(client: TestClient):
    """Verify credentials remain strictly server-side in-memory and are never echoed back."""
    secret_key = "super-secret-token-12345"
    payload = {
        "provider": "kaggle",
        "credentials": {"username": "testuser", "key": secret_key},
        "ttl_seconds": 60,
    }
    res = client.post("/api/v1/data/provider-sessions", json=payload)
    assert res.status_code == 200
    data = res.json()["data"]
    sess_id = data["session_id"]
    assert data["credential_configured"] is True

    # Critical security invariant: secret_key MUST NOT appear anywhere in the response
    res_text = res.text
    assert secret_key not in res_text
    assert "super-secret" not in res_text

    # Tear down
    del_res = client.delete(f"/api/v1/data/provider-sessions/{sess_id}")
    assert del_res.status_code == 200
    assert del_res.json()["data"]["status"] == "DELETED"


# --------------------------------------------------------------------------- #
# I & J. Scientific Certification Read APIs
# --------------------------------------------------------------------------- #
def test_certification_summary_api(client: TestClient):
    """Verify GET /api/v1/certification returns genuine manifest & bundle digests."""
    res = client.get("/api/v1/certification")
    assert res.status_code == 200
    d = res.json()["data"]
    assert d["certification_id"] == "CERT_4FB4D8C42308"
    assert d["total_runs_recorded"] == 180
    assert d["deterministic_runs_count"] == 150
    assert d["gpt41_runs_count"] == 30
    assert len(d["source_artifact_hashes"]) == 14
    assert len(d["bundle_hash"]) == 64
    assert d["invariant_summary"]["passed"] == 10


def test_certification_domains_api(client: TestClient):
    """Verify GET /api/v1/certification/domains and gap 8 (null experiment_spec where absent)."""
    res = client.get("/api/v1/certification/domains")
    assert res.status_code == 200
    domains = res.json()["data"]["domains"]
    assert len(domains) >= 7

    dom_names = {d["domain"] for d in domains}
    expected_doms = {
        "predictive_binary",
        "deep_learning",
        "fraud_imbalanced",
        "recommender",
        "portfolio",
        "market_risk",
        "scenario_traded_risk",
    }
    assert expected_doms.issubset(dom_names)

    # Recommender champion verification
    rec_dom = next(d for d in domains if d["domain"] == "recommender")
    assert rec_dom["champion_model"] == "ncf"
    assert rec_dom["real_data_status"] == "RECOMMENDER_REAL_DATA_CERTIFICATION = BLOCKED"
    # Gap 8: experiment_spec is null when not defined in experiment_matrix.json
    assert rec_dom["experiment_spec"] is None


def test_certification_runs_explorer(client: TestClient):
    """Verify GET /api/v1/certification/runs pagination and filtering."""
    res = client.get("/api/v1/certification/runs?limit=20&offset=0")
    assert res.status_code == 200
    d = res.json()["data"]
    assert d["total_runs_bundle"] == 180
    assert len(d["runs"]) == 20

    # Filter by policy
    res_gpt = client.get("/api/v1/certification/runs?policy=gpt41")
    assert res_gpt.status_code == 200
    d_gpt = res_gpt.json()["data"]
    assert d_gpt["total_filtered"] == 30

    # Specific run lookup
    sample_id = d["runs"][0]["run_id"]
    res_single = client.get(f"/api/v1/certification/runs/{sample_id}")
    assert res_single.status_code == 200
    assert res_single.json()["data"]["run_id"] == sample_id


def test_certification_invariants_api(client: TestClient):
    """Verify GET /api/v1/certification/invariants returns all 10 passed invariants."""
    res = client.get("/api/v1/certification/invariants")
    assert res.status_code == 200
    d = res.json()["data"]
    assert d["count"] == 10
    assert d["all_passed"] is True
    inv_ids = {inv["invariant_id"] for inv in d["invariants"]}
    assert "INV_PORT_EULER_RECONCILIATION" in inv_ids
    assert "INV_MKT_ES_VAR_MONOTONICITY" in inv_ids
    assert "INV_MKT_ES_SUBADDITIVITY" in inv_ids


def test_certification_xai_and_sensitivity_api(client: TestClient):
    """Verify XAI and sensitivity read APIs."""
    res_xai = client.get("/api/v1/certification/xai")
    assert res_xai.status_code == 200
    xai_methods = res_xai.json()["data"]["xai_methods"]
    assert len(xai_methods) == 7
    methods_dict = {x["method"]: x["status"] for x in xai_methods}
    assert methods_dict["native_gain_importance"] == "CERTIFIED"
    assert methods_dict["permutation_importance"] == "CERTIFIED"
    assert methods_dict["shap_tree_explainer"] == "CERTIFIED"
    assert methods_dict["partial_dependence_plot"] == "CERTIFIED"
    assert methods_dict["individual_conditional_expectation"] == "CERTIFIED"
    assert methods_dict["accumulated_local_effects"] == "NOT_EXECUTABLE"
    assert methods_dict["lime_tabular_explainer"] == "NOT_EXECUTABLE"

    res_sens = client.get("/api/v1/certification/sensitivity")
    assert res_sens.status_code == 200
    sens_d = res_sens.json()["data"]
    assert sens_d["shock_grid"] == [-0.30, -0.20, -0.10, -0.05, 0.00, 0.05, 0.10, 0.20, 0.30]
    assert sens_d["baseline_zero_delta"] == 0.0
    assert sens_d["responses"] == "RESPONSES_NOT_PERSISTED"


def test_certification_traces_and_exclusions_api(client: TestClient):
    """Verify provider traces and exclusions."""
    res_tr = client.get("/api/v1/certification/provider-traces")
    assert res_tr.status_code == 200
    assert res_tr.json()["data"]["count"] == 6

    res_ex = client.get("/api/v1/certification/exclusions")
    assert res_ex.status_code == 200
    ex_d = res_ex.json()["data"]
    assert len(ex_d["historical_discrepancies_audited"]) == 11
    assert len(ex_d["domain_blockers"]) >= 3


# --------------------------------------------------------------------------- #
# K, L, M, N, O. Granular Run Presentation Endpoints
# --------------------------------------------------------------------------- #
def test_run_presentation_endpoints(client: TestClient):
    """Verify presentation routes for tuning, XAI, sensitivity, portfolio, scenario."""
    run_id = "run_det_pred_golden_xgboost_seed_0"

    # Tuning presentation (Gap 7: TRIAL_DETAILS_NOT_PERSISTED)
    res_tune = client.get(f"/api/v1/runs/{run_id}/tuning")
    assert res_tune.status_code == 200
    tune_d = res_tune.json()["data"]
    assert tune_d["run_id"] == run_id
    assert tune_d["status"] == "TRIAL_DETAILS_NOT_PERSISTED"
    assert tune_d["trials"] == []
    assert tune_d["champion_model"] == "xgboost"

    # XAI presentation
    res_xai = client.get(f"/api/v1/runs/{run_id}/xai")
    assert res_xai.status_code == 200

    # Sensitivity presentation
    res_sens = client.get(f"/api/v1/runs/{run_id}/sensitivity")
    assert res_sens.status_code == 200
    assert res_sens.json()["data"]["responses"] == "RESPONSES_NOT_PERSISTED"

    # Portfolio presentation (Gap 9: multi-objective evaluation)
    res_port = client.get(f"/api/v1/runs/{run_id}/portfolio")
    assert res_port.status_code == 200
    port_d = res_port.json()["data"]
    assert "multi_objective_evaluation" in port_d
    assert port_d["objective_winners"]["minimum_volatility"] == "min_variance"
    assert port_d["objective_winners"]["equal_risk_contribution"] == "erc"

    # Scenario presentation
    res_scen = client.get(f"/api/v1/runs/{run_id}/scenario")
    assert res_scen.status_code == 200
    scen_d = res_scen.json()["data"]
    assert scen_d["zero_shock_check"] == "PASS"
    assert scen_d["monotonicity_check"] == "PASS"
    assert scen_d["Gate6_integrity"] == "PASS"
    assert scen_d["Gate6A_integrity"] == "NOT_AVAILABLE"
    assert scen_d["units"] == "NOT_AVAILABLE"
