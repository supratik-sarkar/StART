from __future__ import annotations

import json

from start.reporting.review_transcript import (
    render_transcript_html,
    render_transcript_markdown,
    write_transcript,
)
from start.review_session import Decision, Exchange, ReviewSession


def _populated_session():
    s = ReviewSession(run_id="RUN-T")
    s.record_decision(Decision(
        key="architecture", prompt="Family?", recommended="mlp",
        user_value="wide_deep", effective="wide_deep", choice="keep",
        rationale="user prefers wide_deep",
    ))
    s.record_exchange(Exchange(
        agent="ArchitectureReviewAgent", question="Why not wide_deep?",
        answer="wide_deep is viable but mlp lowers overfitting risk.",
        checkpoint="architecture", backend="deterministic",
    ))
    s.add_clarification("false negatives are costlier")
    return s


def test_markdown_has_all_sections():
    md = render_transcript_markdown(_populated_session())
    assert "Review committee transcript" in md
    assert "## Decisions" in md
    assert "## User overrides" in md
    assert "## Agent conversations" in md
    assert "Why not wide_deep?" in md


def test_markdown_shows_override():
    md = render_transcript_markdown(_populated_session())
    assert "wide_deep" in md and "mlp" in md


def test_html_renders_and_escapes():
    s = _populated_session()
    s.record_exchange(Exchange(agent="A", question="<script>", answer="ok"))
    htm = render_transcript_html(s)
    assert "<h1>" in htm
    assert "&lt;script&gt;" in htm  # escaped, not raw


def test_empty_session_renders_gracefully():
    md = render_transcript_markdown(ReviewSession(run_id="EMPTY"))
    assert "No interactive decisions" in md
    assert "No overrides" in md


def test_write_transcript_creates_files(tmp_path):
    paths = write_transcript(_populated_session(), str(tmp_path), "RUN-T")
    for key in ("md", "html", "json"):
        assert key in paths
    d = json.loads(open(paths["json"]).read())
    assert len(d["decisions"]) == 1
    assert len(d["conversations"]) == 1
    assert len(d["overrides"]) == 1
