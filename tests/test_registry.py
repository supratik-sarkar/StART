import pytest

from start.registry import get_test, list_families, list_tests


def test_builtin_tests_registered():
    ids = {s.test_id for s in list_tests()}
    assert "preprocessing.missingness" in ids
    assert "supervised.discrimination" in ids
    assert "xai.importance_stability" in ids
    assert "genai.citation_coverage" in ids


def test_families_present():
    families = list_families()
    assert {"preprocessing", "supervised", "xai", "genai"} <= set(families)


def test_unknown_test_raises():
    with pytest.raises(KeyError):
        get_test("does.not.exist")
