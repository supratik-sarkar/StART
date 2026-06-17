from __future__ import annotations

from start.validation_review import business_interpretation


def test_business_interpretation_ranks_top_features():
    rows = [
        {"feature": "worst_perimeter", "shock": 0.3, "drift": 0.25, "risk_impact": "high"},
        {"feature": "mean_radius", "shock": 0.3, "drift": 0.02, "risk_impact": "low"},
    ]
    lines = business_interpretation(rows, top=2)
    assert len(lines) == 2
    assert "worst_perimeter" in lines[0]  # highest drift first


def test_business_interpretation_empty():
    assert business_interpretation([], top=3) == []


def test_business_interpretation_describes_dependence():
    rows = [{"feature": "f", "shock": 0.3, "drift": 0.35, "risk_impact": "high"}]
    line = business_interpretation(rows)[0]
    assert "f" in line and "drift" in line.lower()
