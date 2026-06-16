from __future__ import annotations

from start.reporting.agent_trace import (
    TRACEABLE_AGENTS,
    AgentTrace,
    TraceLog,
    render_trace_log_markdown,
)
from start.reporting.artifacts import (
    ArtifactRegistry,
    render_artifact_catalog_markdown,
)


# -- AgentTrace -------------------------------------------------------------- #
def test_agent_trace_has_thinking_fields():
    t = AgentTrace(
        agent="ArchitectureReviewAgent", inputs="569 rows", decision="recommend MLP",
        reasoning="small tabular", evidence_ids=["ARCH-01"], confidence=0.8,
        alternative_considered="Wide&Deep", action_taken="surfaced choice",
    )
    d = t.to_dict()
    for key in ("agent", "inputs", "evidence_ids", "reasoning", "decision",
                "confidence", "alternative_considered", "action_taken"):
        assert key in d


def test_agent_trace_terminal_render():
    t = AgentTrace(agent="TaskInferenceAgent", inputs="2 classes", decision="binary",
                   reasoning="target has 2 unique values", confidence=0.95)
    out = t.render_terminal()
    assert "TaskInferenceAgent" in out
    assert "95%" in out
    assert "decision" in out and "binary" in out


def test_trace_log_records_and_lists():
    log = TraceLog()
    log.record("A", inputs="x", decision="d1")
    log.record("B", inputs="y", decision="d2", confidence=0.5)
    assert log.agents() == ["A", "B"]
    assert len(log.to_list()) == 2
    assert log.to_list()[1]["confidence"] == 0.5


def test_traceable_agents_cover_core_set():
    for agent in ("DatasetDiscoveryAgent", "TaskInferenceAgent", "ArchitectureReviewAgent",
                  "HyperparameterTuningAgent", "Governance", "Signoff", "EvidenceCritic"):
        assert agent in TRACEABLE_AGENTS


def test_trace_log_markdown():
    log = TraceLog()
    log.record("FeatureEngineeringAgent", inputs="stats", decision="scale + encode",
               reasoning="numeric + categorical present", evidence_ids=["FE-01"],
               confidence=0.7, alternative_considered="leave raw")
    md = render_trace_log_markdown(log)
    assert "### Agent reasoning traces" in md
    assert "FeatureEngineeringAgent" in md and "FE-01" in md


def test_empty_trace_log_markdown():
    assert "No agent traces" in render_trace_log_markdown(TraceLog())


# -- ArtifactRegistry -------------------------------------------------------- #
def test_artifact_registration_infers_type():
    reg = ArtifactRegistry()
    a = reg.register("/out/sensitivity.csv", category="sensitivity")
    assert a.name == "sensitivity.csv"
    assert "CSV" in a.artifact_type
    assert a.category == "sensitivity"


def test_artifact_announcement_fires():
    announced = []
    reg = ArtifactRegistry(announce=announced.append)
    reg.register("/out/telemetry.json")
    assert announced
    assert "telemetry.json" in announced[0]
    assert "Generated:" in announced[0]


def test_artifact_type_mapping():
    reg = ArtifactRegistry()
    assert "HTML" in reg.register("/out/dashboard.html").artifact_type
    assert "PNG" in reg.register("/out/plot.png").artifact_type
    assert "JSON" in reg.register("/out/data.json").artifact_type


def test_artifact_by_category():
    reg = ArtifactRegistry()
    reg.register("/out/a.csv", category="sensitivity")
    reg.register("/out/b.png", category="explainability")
    reg.register("/out/c.csv", category="sensitivity")
    cats = reg.by_category()
    assert len(cats["sensitivity"]) == 2
    assert len(cats["explainability"]) == 1


def test_register_many():
    reg = ArtifactRegistry()
    arts = reg.register_many(["/out/x.json", "/out/y.png"], category="report")
    assert len(arts) == 2
    assert reg.names() == ["x.json", "y.png"]


def test_artifact_catalog_markdown():
    reg = ArtifactRegistry()
    reg.register("/out/dashboard.html", category="report")
    md = render_artifact_catalog_markdown(reg)
    assert "### Artifact catalog" in md
    assert "dashboard.html" in md


def test_empty_catalog_markdown():
    assert "No artifacts" in render_artifact_catalog_markdown(ArtifactRegistry())


# -- v2.1.1 remediation Section N: backend / llm_used in traces --------------- #
def test_agent_trace_has_backend_and_llm_used():
    t = AgentTrace(agent="X", inputs="i", decision="d", backend="openai", llm_used=True)
    d = t.to_dict()
    assert d["backend"] == "openai"
    assert d["llm_used"] is True
    assert "fallback_reason" in d


def test_agent_trace_defaults_deterministic():
    t = AgentTrace(agent="X", inputs="i", decision="d")
    assert t.backend == "deterministic"
    assert t.llm_used is False
