from __future__ import annotations

import tempfile

from start.ai_engineering.layer import run_ai_engineering_layer


def test_on_adapter_start_fires_with_activity():
    seen = []
    run_ai_engineering_layer(
        {}, output_root=tempfile.mkdtemp(),
        on_adapter_start=lambda name, activity: seen.append((name, activity)),
    )
    assert seen, "on_adapter_start should fire for each adapter"
    names = {n for n, _ in seen}
    assert "OPA" in names
    # each announcement carries a non-empty activity verb
    assert all(activity for _, activity in seen)


def test_adapters_have_distinct_activities():
    from start.ai_engineering.adapters import build_adapters

    adapters = build_adapters(output_root=tempfile.mkdtemp())
    activities = {a.name: a.activity for a in adapters}
    assert activities["OPA"] == "Validating policy controls"
    assert activities["DeepEval"] == "Running quality checks"
    assert activities["Phoenix"] == "Recording observability artifact"
