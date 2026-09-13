from __future__ import annotations

from start.evidence_dialogue import (
    INSUFFICIENT,
    answer_from_evidence,
    classify_intent,
    evidence_critic,
    retrieve,
)
from start.evidence_store import EvidenceStore
from start.modeling.data import load_attrition_dataset
from start.modeling.data_statistics import compute_data_statistics


def _store_with_data():
    df = load_attrition_dataset(seed=0)
    return EvidenceStore.from_artifacts(data_stats=compute_data_statistics(df, "attrition"))


# -- the core anti-hallucination requirement --------------------------------- #
def test_outlier_question_returns_real_values():
    ea = answer_from_evidence("Show top 10 variables with highest outlier burden",
                              _store_with_data())
    assert ea is not None and ea.grounded is True
    assert "area_error" in ea.answer  # a real column, retrieved from evidence


def test_missing_evidence_triggers_explicit_refusal():
    # no missingness exists -> must refuse, not fabricate
    ea = answer_from_evidence("Which 5 features have the most missing values?",
                              _store_with_data())
    assert ea is not None and ea.refused is True
    assert INSUFFICIENT in ea.answer


def test_importance_without_model_refuses():
    ea = answer_from_evidence("List the top 10 most important features",
                              _store_with_data())
    assert ea.refused is True
    assert INSUFFICIENT in ea.answer


def test_sensitivity_without_run_refuses():
    ea = answer_from_evidence("What is the maximum drift in sensitivity?",
                              _store_with_data())
    assert ea.refused is True


def test_empty_store_refuses_everything_diagnostic():
    empty = EvidenceStore()
    for q in ("top outliers", "feature importance", "correlation pairs",
              "missing values", "auc by split"):
        ea = answer_from_evidence(q, empty)
        assert ea is not None and ea.refused is True


def test_non_diagnostic_question_returns_none():
    # "why MLP" is not a diagnostic-value question; the evidence layer defers
    assert answer_from_evidence("Why do you recommend MLP?", _store_with_data()) is None


def test_no_fabricated_feature_names_in_any_answer():
    # an adversarial store with known columns; answer must only contain those
    store = EvidenceStore(outliers={"real_col_a": 10, "real_col_b": 5})
    ea = answer_from_evidence("show top 10 outlier variables", store)
    assert "real_col_a" in ea.answer and "real_col_b" in ea.answer
    # generic hallucination tokens must never appear
    for fake in ("variable_2", "variable_3", "feature_x", "X%", "Y%"):
        assert fake not in ea.answer


def test_intent_classification():
    assert classify_intent("top outlier variables") == "outliers"
    assert classify_intent("missing values") == "missingness"
    assert classify_intent("correlated pairs") == "correlation"
    assert classify_intent("feature importance") == "importance"
    assert classify_intent("sensitivity drift") == "sensitivity"
    assert classify_intent("auc by split") == "metrics"
    assert classify_intent("why mlp") is None


def test_evidence_critic_rejects_ungrounded():
    from start.evidence_store import EvidenceItem

    items = [EvidenceItem("outliers", "col_a: 10 outlier rows", 10, "src")]
    assert evidence_critic("col_a had 10 outlier rows", items) is True
    assert evidence_critic("some unrelated fabricated claim", items) is False
    assert evidence_critic(INSUFFICIENT, []) is True  # refusal is always valid


def test_top_n_respected():
    store = EvidenceStore(outliers={f"c{i}": 100 - i for i in range(30)})
    items = retrieve("outliers", store, "show top 5 outliers")
    assert len(items) == 5
