from __future__ import annotations

from start.reporting.progress import (
    PROGRESS_PHASES,
    ActionLog,
    ProgressReporter,
    render_action_log_markdown,
    render_bar,
)


def test_render_bar_bounds():
    assert "0.0%" in render_bar(0)
    assert "100.0%" in render_bar(100)
    assert "50.0%" in render_bar(50)
    # clamps out-of-range
    assert "100.0%" in render_bar(150)
    assert "0.0%" in render_bar(-10)


def test_render_bar_is_cross_platform_ascii():
    bar = render_bar(50)
    # only ascii bracket/hash/dash/space/digits/percent/dot
    assert all(ord(c) < 128 for c in bar)
    assert "[" in bar and "]" in bar


def test_progress_advance_tracks_phases():
    rep = ProgressReporter(quiet=True)
    rep.advance("Data loading")
    first = rep.history[-1][1]
    rep.advance("Reporting")  # last phase -> 100%
    last = rep.history[-1][1]
    assert first < last
    assert abs(last - 100.0) < 0.01


def test_progress_update_and_table():
    rep = ProgressReporter(quiet=True)
    rep.update("Custom", 42.0)
    rows = rep.table_rows()
    assert rows[-1] == {"phase": "Custom", "percent": 42.0}


def test_phases_complete():
    assert "Model fitting" in PROGRESS_PHASES
    assert "Sensitivity" in PROGRESS_PHASES
    assert len(PROGRESS_PHASES) >= 10


def test_action_log_records_all_fields():
    log = ActionLog()
    a = log.record(
        "DatasetDiscoveryAgent",
        "raw dataframe",
        "profiled schema",
        recommendation="target=attrition",
        evidence_ids=["EV-1"],
        user_decision="confirmed",
        output_artifact="profile.json",
    )
    d = a.to_dict()
    for key in (
        "agent",
        "input_reviewed",
        "action",
        "recommendation",
        "evidence_ids",
        "user_decision",
        "output_artifact",
    ):
        assert key in d
    assert d["evidence_ids"] == ["EV-1"]


def test_action_log_ordering_and_agents():
    log = ActionLog()
    log.record("A", "in", "did x")
    log.record("B", "in", "did y")
    log.record("C", "in", "did z")
    assert log.agents() == ["A", "B", "C"]
    assert len(log.to_list()) == 3


def test_action_log_markdown():
    log = ActionLog()
    log.record(
        "ArchitectureReviewAgent",
        "mlp choice",
        "validated",
        recommendation="keep mlp",
        evidence_ids=["ARCH-01"],
        user_decision="accepted",
    )
    md = render_action_log_markdown(log)
    assert "### Agentic action log" in md
    assert "ArchitectureReviewAgent" in md and "ARCH-01" in md


def test_empty_action_log_markdown():
    assert "No agent actions" in render_action_log_markdown(ActionLog())
