from __future__ import annotations

from start.agent_roster import (
    AGENT_ROSTER,
    announce_adapter_activity,
    render_adapter_panel,
    render_agent_roster,
    roster_as_list,
)


def test_roster_has_all_committee_agents():
    names = {r.name for r in AGENT_ROSTER}
    for expected in ("DatasetDiscoveryAgent", "FeatureEngineeringAgent",
                     "ArchitectureReviewAgent", "HyperparameterTuningAgent",
                     "ModelExecutionAgent", "ValidationAgent",
                     "GovernanceSignoffAgent", "EvidenceCriticAgent"):
        assert expected in names


def test_render_roster_includes_purposes():
    out = render_agent_roster()
    assert "DatasetDiscoveryAgent" in out
    assert "Dataset understanding" in out


def test_roster_as_list_shape():
    rows = roster_as_list()
    assert len(rows) == len(AGENT_ROSTER)
    assert all("agent" in r and "purpose" in r for r in rows)


def test_adapter_panel_marks_available_and_absent():
    cs = [{"adapter": "OPA", "status": "complete", "purpose": "gov"},
          {"adapter": "Moonshot", "status": "not_installed", "purpose": "excluded"}]
    out = render_adapter_panel(cs)
    assert "[+] OPA" in out
    assert "[-] Moonshot" in out


def test_announce_adapter_activity_format():
    assert announce_adapter_activity("DeepEval", "Running hallucination evaluation") == \
        "[DeepEval] Running hallucination evaluation…"
