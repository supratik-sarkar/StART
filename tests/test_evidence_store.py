from __future__ import annotations

from start.evidence_store import EvidenceStore
from start.modeling.data import load_attrition_dataset
from start.modeling.data_statistics import compute_data_statistics


def _store():
    df = load_attrition_dataset(seed=0)
    return EvidenceStore.from_artifacts(data_stats=compute_data_statistics(df, "attrition"))


def test_outliers_retrieved_from_real_data():
    items = _store().top_outliers(5)
    assert items
    assert all(i.kind == "outliers" and i.value > 0 for i in items)
    assert all(i.source == "data_statistics.outlier_summary" for i in items)


def test_outliers_ranked_descending():
    items = _store().top_outliers(10)
    counts = [i.value for i in items]
    assert counts == sorted(counts, reverse=True)


def test_missingness_empty_when_none_present():
    # breast-cancer cohort has no missing values
    assert _store().top_missing(10) == []


def test_importance_empty_without_model_run():
    assert _store().top_importance(10) == []


def test_has_any_reflects_content():
    assert _store().has_any() is True
    assert EvidenceStore().has_any() is False


def test_from_artifacts_populates_shape():
    s = _store()
    assert s.n_rows == 569
    assert s.n_numeric is not None
