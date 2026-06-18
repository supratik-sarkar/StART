from __future__ import annotations

from start.evidence_store import EvidenceStore
from start.modeling.data import load_attrition_dataset
from start.modeling.data_statistics import compute_data_statistics
from start.review_tables import (
    adapter_inventory_table,
    challenge_log_table,
    correlation_evidence_table,
    dataset_discovery_table,
    importance_table,
    metrics_table,
    outlier_evidence_table,
    sensitivity_ranking_table,
    shock_table,
    tuning_table,
)


def _store():
    df = load_attrition_dataset(seed=0)
    return EvidenceStore.from_artifacts(data_stats=compute_data_statistics(df, "attrition"))


def test_dataset_discovery_table_builds():
    t = dataset_discovery_table(_store(), ["attrition"], (0.6, 0.2, 0.2))
    assert t.row_count >= 8  # several transparency fields


def test_outlier_evidence_table_real_values():
    t, has = outlier_evidence_table(_store(), 5)
    assert has is True
    assert t.row_count == 5
    assert t.columns[0].header == "Feature"


def test_outlier_evidence_empty_when_none():
    t, has = outlier_evidence_table(EvidenceStore(), 5)
    assert has is False


def test_correlation_evidence_table():
    store = EvidenceStore(correlations=[{"a": "x", "b": "y", "r": 0.97}])
    t, has = correlation_evidence_table(store, 5)
    assert has is True and t.row_count == 1


def test_metrics_table_builds():
    t = metrics_table({"train": {"auc_roc": 0.9}, "oos": {"auc_roc": 0.85}})
    assert t.row_count == 2


def test_tuning_table_builds():
    trials = [{"trial": 1, "params": {"learning_rate": 0.01, "hidden_dims": [32],
               "dropout": 0.2}, "validation_metric": 0.9, "status": "ok"}]
    assert tuning_table(trials).row_count == 1


def test_importance_table_builds():
    rows = [{"rank": 1, "feature": "f", "importance": 0.1, "direction": "positive"}]
    assert importance_table(rows).row_count == 1


def test_sensitivity_ranking_collapses_to_per_feature_max():
    rows = [
        {"feature": "a", "shock": -0.3, "drift": -0.05, "risk_impact": "moderate"},
        {"feature": "a", "shock": 0.3, "drift": 0.02, "risk_impact": "low"},
        {"feature": "b", "shock": 0.3, "drift": 0.10, "risk_impact": "high"},
    ]
    t = sensitivity_ranking_table(rows)
    assert t.row_count == 2  # a and b, one row each


def test_shock_table_builds():
    rows = [{"feature": "a", "shock": s, "drift": 0.01 * i}
            for i, s in enumerate((-0.3, -0.2, -0.1, 0.1, 0.2, 0.3))]
    t = shock_table(rows)
    assert t.row_count == 1


def test_adapter_inventory_table():
    cs = [{"adapter": "OPA", "status": "complete", "purpose": "policy",
           "runtime_s": 0.01, "artifacts": 1, "evidence": 2}]
    assert adapter_inventory_table(cs).row_count == 1


def test_challenge_log_table():
    ch = [{"status": "closed", "agent": "A", "text": "why not X",
           "evidence_used": ["src"]}]
    assert challenge_log_table(ch).row_count == 1
