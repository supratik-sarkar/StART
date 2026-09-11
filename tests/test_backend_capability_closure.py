"""Targeted test suite verifying backend capability closure B1 through B8.

Covers:
- B1: Execution Mode Dispatch & Route Aliases (/plan/generate, /workflow/run, CP-AGENTIC)
- B2: Requested Config == Resolved Config == Actual Executed Config (LightGBM, XGBoost, fail-closed for CatBoost)
- B3: Dataset Hub Truthfulness (Local/Synthetic executable, remote connectors deferred)
- B4: Supervised Fraud & AML Imbalance (synthetic_aml_imbalanced runnable, unsupervised deferred)
- B5: FFM Recommender Dispatch & Mathematical Interaction Form
- B6: Market Optimizer & Scenario Parameter Propagation
- B7: 9-point Sensitivity Grid (-30% to +30%) in OAT and Parallel Basket Modes
- B8: Tuning Engine Strategy Propagation, Objective Metric in Trial Events, and Champion Promotion
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["START_TORCH_DEVICE"] = "cpu"
os.environ["START_DISABLE_MPS"] = "1"
os.environ["START_DEVICE"] = "cpu"

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("starlette")
from starlette.testclient import TestClient

from start.modeling.models import resolve_model
from start.modeling.sensitivity_analysis import DEFAULT_SHOCKS, run_sensitivity_analysis
from start.recommender.fixtures import DATASET_C
from start.recommender.models import FieldAwareFactorizationMachineModel
from start.runtime.execution import CanonicalExecutionService
from start.web.app import app


@pytest.fixture
def client():
    return TestClient(app)


# =========================================================================== #
# B1: Execution Mode Dispatch & Route Aliases
# =========================================================================== #

def test_b1_route_aliases_and_execution_modes(client):
    """B1: Verify /plan/generate and /workflow/run route aliases exist and respect executionMode."""
    # Test /plan/generate alias
    resp_plan = client.post(
        "/plan/generate",
        json={
            "workflowId": "predictive_ml",
            "contextId": "institutional_credit_v1",
            "executionMode": "agentic_session",
            "goal": "Test plan generation under agentic session",
        },
    )
    assert resp_plan.status_code == 200, resp_plan.text
    data_plan = resp_plan.json()
    assert "plan" in data_plan
    assert data_plan.get("executionMode") == "agentic_session"
    assert "agentProposal" in data_plan

    # Test /workflow/run alias
    resp_run = client.post(
        "/workflow/run",
        json={
            "workflowId": "predictive_ml",
            "contextId": "institutional_credit_v1",
            "executionMode": "agentic_session",
            "parameters": {"model": "logistic_regression"},
        },
    )
    assert resp_run.status_code == 200, resp_run.text
    data_run = resp_run.json()
    assert "run_id" in data_run
    assert data_run.get("execution_mode") == "agentic_session"


def test_b1_agentic_session_emits_cp_agentic():
    """B1: Verify agentic_session execution mode commits CP-AGENTIC checkpoint."""
    service = CanonicalExecutionService()
    result = service.execute(
        workflow_id="predictive_ml",
        context_id="institutional_credit_v1",
        request_params={"model": "logistic_regression"},
        execution_mode="agentic_session",
    )
    assert result.run_id is not None
    assert len(result.records) > 0
    cp_ids = [cp["checkpoint_id"] for cp in result.checkpoints]
    assert "CP-AGENTIC" in cp_ids
    agentic_cp = next(cp for cp in result.checkpoints if cp["checkpoint_id"] == "CP-AGENTIC")
    assert agentic_cp["producing_stage"] == "step-synthesis"
    assert agentic_cp["agent_signature"] == "ExecutiveDirector"


# =========================================================================== #
# B2: Parameter Propagation & Fail-Closed Behavior
# =========================================================================== #

def test_b2_parameter_propagation_lightgbm():
    """B2: Verify requested LightGBM model, split, and preprocessing propagate to actual executed config."""
    service = CanonicalExecutionService()
    req_params = {
        "model": "lightgbm",
        "hyperparameters": {"n_estimators": 25, "learning_rate": 0.05},
        "preprocessing": {"imputation": "median", "scaler": "standard"},
        "split": {"strategy": "stratified", "test_size": 0.2},
    }
    result = service.execute(
        workflow_id="predictive_ml",
        context_id="institutional_credit_v1",
        request_params=req_params,
        execution_mode="deterministic_run",
    )
    assert result.run_id is not None
    assert len(result.records) > 0

    # Find the tabular context from bundle
    tab = result.context_instance.bundle.tabular
    resolved_cfg = tab.extra.get("resolved_configuration", {})
    assert resolved_cfg.get("model") == "lightgbm"
    assert resolved_cfg.get("hyperparameters") == {"n_estimators": 25, "learning_rate": 0.05}
    assert resolved_cfg.get("preprocessing") == {"imputation": "median", "scaler": "standard"}
    assert resolved_cfg.get("split") == {"strategy": "stratified", "test_size": 0.2}

    # Verify fitted model class
    assert "LGBMClassifier" in tab.model.__class__.__name__


def test_b2_fail_closed_on_uninstalled_model():
    """B2: Verify resolve_model fails closed (raises ValueError) if an uninstalled model is requested."""
    with pytest.raises(ValueError) as excinfo:
        resolve_model("catboost", fail_closed=True)
    assert "catboost is not installed" in str(excinfo.value).lower()


# =========================================================================== #
# B3: Dataset Hub Truthfulness
# =========================================================================== #

def test_b3_dataset_hub_manifest(client):
    """B3: Verify capability manifest truthfully reports local/synthetic datasets as executable and remote connectors as deferred."""
    resp = client.get("/api/v1/capability-manifest")
    assert resp.status_code == 200
    manifest = resp.json()

    dataset_hub = manifest["domains"]["dataset_hub"]
    assert "local_synthetic_benchmark" in dataset_hub["executable_sources"]
    assert "canonical_context_generator" in dataset_hub["executable_sources"]
    assert set(dataset_hub["deferred_connectors"]) == {"kaggle", "openml", "uci", "huggingface"}

    # Ensure all 8 canonical contexts are declared
    context_ids = {c["id"] for c in dataset_hub["contexts"]}
    assert "institutional_credit_v1" in context_ids
    assert "deep_learning_v1" in context_ids
    assert "institutional_market_v1" in context_ids
    assert "recommender_ratings_v1" in context_ids
    assert "recommender_implicit_v1" in context_ids
    assert "recommender_contextual_v1" in context_ids
    assert "recommender_ffm_v1" in context_ids
    assert "synthetic_aml_imbalanced" in context_ids


# =========================================================================== #
# B4: Fraud & AML Imbalance Modeling
# =========================================================================== #

def test_b4_supervised_fraud_workflow_runnable():
    """B4: Verify synthetic_aml_imbalanced runs successfully under fraud_anomaly_aml workflow."""
    service = CanonicalExecutionService()
    result = service.execute(
        workflow_id="fraud_anomaly_aml",
        context_id="synthetic_aml_imbalanced",
        request_params={"model": "random_forest"},
        execution_mode="hybrid_workbench",
    )
    assert result.run_id is not None
    assert len(result.records) > 0
    assert result.context_instance.actual_target == "is_fraud"


# =========================================================================== #
# B5: FFM Recommender Dispatch & Mathematical Interaction Form
# =========================================================================== #

def test_b5_ffm_recommender_execution():
    """B5: Verify Field-Aware Factorization Machine (FFM) dispatches and executes on contextual data."""
    service = CanonicalExecutionService()
    result = service.execute(
        workflow_id="recommender_system",
        context_id="recommender_ffm_v1",
        request_params={"algorithm": "field_aware_factorization_machine", "latent_dim": 4, "epochs": 5},
        execution_mode="deterministic_run",
    )
    assert result.run_id is not None
    assert len(result.records) > 0
    rec_ctx = result.context_instance.bundle.recommender
    assert rec_ctx.execution_result is not None
    assert "Field-Aware" in rec_ctx.execution_result.model_summary.algorithm

    # Check FFM model structure and interaction tensor
    raw_data = list(DATASET_C)
    ffm = FieldAwareFactorizationMachineModel(latent_dim=4, epochs=5, seed=42)
    ffm.fit(raw_data)
    assert ffm.V.ndim == 3  # (n_features, n_fields, latent_dim)
    assert ffm.V.shape[2] == 4  # latent_dim


# =========================================================================== #
# B6: Market Optimizer & Scenario Parameters
# =========================================================================== #

def test_b6_market_parameter_propagation():
    """B6: Verify market workflow propagates requested optimizer and scenario into resolved configuration."""
    service = CanonicalExecutionService()
    req_params = {
        "optimizer": "hierarchical_risk_parity",
        "scenario": "hypothetical_shift",
        "risk_free_rate": 0.035,
    }
    result = service.execute(
        workflow_id="quantitative_finance",
        context_id="institutional_market_v1",
        request_params=req_params,
        execution_mode="deterministic_run",
    )
    assert result.run_id is not None
    assert len(result.records) > 0
    market = result.context_instance.bundle.market
    resolved_cfg = market.extra.get("resolved_configuration", {})
    assert resolved_cfg.get("optimizer") == "hierarchical_risk_parity"
    assert resolved_cfg.get("scenario") == "hypothetical_shift"


# =========================================================================== #
# B7: 9-point Sensitivity Grid (-30% to +30%) in OAT and Parallel Basket Modes
# =========================================================================== #

def test_b7_9_point_sensitivity_oat_and_parallel_basket():
    """B7: Verify 9-point sensitivity grid (-30% to +30%) executes in both OAT and Parallel Basket modes."""
    from sklearn.linear_model import LogisticRegression
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 4), columns=["f1", "f2", "f3", "f4"])
    y = (X["f1"] + X["f2"] > 0).astype(int).to_numpy()
    clf = LogisticRegression()
    clf.fit(X, y)

    # 1. One-at-a-time mode
    res_oat = run_sensitivity_analysis(
        clf, X, y, top_features=["f1", "f2"], metric_name="auc_roc", shocks=DEFAULT_SHOCKS, mode="one_at_a_time"
    )
    assert len(DEFAULT_SHOCKS) == 9
    assert DEFAULT_SHOCKS == (-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30)
    # 2 features * 9 shocks = 18 rows
    assert len(res_oat.shock_rows) == 18
    # 0% shock has drift 0.0 exactly by construction
    zero_shocks = [r for r in res_oat.shock_rows if r.shock == 0.0]
    for zs in zero_shocks:
        assert zs.drift == 0.0
        assert zs.metric_value == res_oat.baseline

    # 2. Parallel basket mode
    res_basket = run_sensitivity_analysis(
        clf, X, y, top_features=["f1", "f2"], metric_name="auc_roc", shocks=DEFAULT_SHOCKS, mode="parallel_basket"
    )
    # 1 basket * 9 shocks = 9 rows
    assert len(res_basket.shock_rows) == 9
    assert res_basket.shock_rows[0].feature.startswith("[PARALLEL BASKET")


# =========================================================================== #
# B8: Tuning Engine Strategy Propagation & Champion Promotion
# =========================================================================== #

def test_b8_tuning_optuna_strategy_and_champion_promotion():
    """B8: Verify Optuna tuning runs requested 5 trials, emits objective_metric, and promotes champion model."""
    service = CanonicalExecutionService()
    req_params = {
        "strategy": "optuna",
        "model": "lightgbm",
        "trials": 5,
        "primary_metric": "auc_roc",
    }
    result = service.execute(
        workflow_id="hyperparameter_tuning",
        context_id="institutional_credit_v1",
        request_params=req_params,
        execution_mode="deterministic_run",
    )
    assert result.run_id is not None
    assert len(result.records) > 0

    # Verify tuning trial events contained objective_metric
    trial_events = [ev for ev in result.events if ev.event_type == "tuning_trial"]
    assert len(trial_events) >= 5
    for ev in trial_events:
        assert "objective_metric" in ev.metadata
        assert ev.metadata["objective_metric"] == "auc_roc"
        assert ev.metadata["strategy"] == "optuna"

    # Verify champion model promotion in bundle tabular
    tab = result.context_instance.bundle.tabular
    assert "champion_lineage" in tab.extra
    champ = tab.extra["champion_lineage"]
    assert champ["champion_model"] == "lightgbm"
    assert champ["tuning_strategy"] == "optuna"
    assert "best_hyperparameters" in champ
    assert "LGBMClassifier" in tab.model.__class__.__name__
