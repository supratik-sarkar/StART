"""Committee review transcript (v2.2.0 item 11).

Renders the review *journey* — agent conversations, user questions, decisions,
overrides, and clarifications — from a ``ReviewSession`` into markdown and HTML
that read like a model-risk committee transcript. Written alongside the
dashboard so the review is reproducible as a narrative, not just a set of
output tables.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from start.review_session import ReviewSession


def render_transcript_markdown(session: ReviewSession) -> str:
    d = session.to_dict()
    lines = [f"# Review committee transcript — {session.run_id}", ""]

    lines += ["## Decisions", ""]
    if d["decisions"]:
        lines += ["| Checkpoint | Recommended | User chose | Outcome | Rationale |",
                  "| --- | --- | --- | --- | --- |"]
        for dec in d["decisions"]:
            lines.append(
                f"| {dec['key']} | {dec['recommended']} | {dec['effective']} "
                f"| {dec['choice']} | {dec['rationale'][:80]} |"
            )
    else:
        lines.append("_No interactive decisions recorded (non-interactive run)._")

    lines += ["", "## User overrides", ""]
    if d["overrides"]:
        for dec in d["overrides"]:
            lines.append(
                f"- **{dec['key']}**: user chose `{dec['effective']}` over "
                f"recommended `{dec['recommended']}` ({dec['choice']})"
            )
    else:
        lines.append("_No overrides — user accepted all recommendations._")

    lines += ["", "## Agent conversations", ""]
    if d["conversations"]:
        for ex in d["conversations"]:
            lines.append(f"**{ex['agent']}** _(at {ex['checkpoint'] or 'review'}, "
                         f"via {ex['backend']})_")
            lines.append(f"> Q: {ex['question']}")
            lines.append(f"> A: {ex['answer']}")
            lines.append("")
    else:
        lines.append("_No questions asked of the agents._")

    if d["clarifications"]:
        lines += ["", "## User clarifications", ""]
        for c in d["clarifications"]:
            lines.append(f"- {c}")

    return "\n".join(lines) + "\n"


def render_transcript_html(session: ReviewSession) -> str:
    d = session.to_dict()

    def esc(x: Any) -> str:
        return html.escape(str(x))

    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>Review transcript — {esc(session.run_id)}</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;"
        "padding:0 1rem;color:#1a1a1a;line-height:1.5}"
        "h1{border-bottom:2px solid #333}h2{margin-top:2rem;color:#333}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;"
        "padding:.4rem .6rem;text-align:left;font-size:.9rem}"
        ".qa{background:#f6f8fa;border-left:3px solid #0969da;padding:.5rem .8rem;"
        "margin:.6rem 0;border-radius:4px}.q{font-weight:600}"
        ".override{color:#a40e26}</style></head><body>",
        f"<h1>Review committee transcript — {esc(session.run_id)}</h1>",
    ]

    parts.append("<h2>Decisions</h2>")
    if d["decisions"]:
        parts.append("<table><tr><th>Checkpoint</th><th>Recommended</th>"
                     "<th>User chose</th><th>Outcome</th><th>Rationale</th></tr>")
        for dec in d["decisions"]:
            parts.append(
                f"<tr><td>{esc(dec['key'])}</td><td>{esc(dec['recommended'])}</td>"
                f"<td>{esc(dec['effective'])}</td><td>{esc(dec['choice'])}</td>"
                f"<td>{esc(dec['rationale'])}</td></tr>"
            )
        parts.append("</table>")
    else:
        parts.append("<p><em>No interactive decisions (non-interactive run).</em></p>")

    parts.append("<h2>User overrides</h2>")
    if d["overrides"]:
        parts.append("<ul>")
        for dec in d["overrides"]:
            parts.append(
                f"<li class='override'><b>{esc(dec['key'])}</b>: chose "
                f"<code>{esc(dec['effective'])}</code> over "
                f"<code>{esc(dec['recommended'])}</code> ({esc(dec['choice'])})</li>"
            )
        parts.append("</ul>")
    else:
        parts.append("<p><em>No overrides — all recommendations accepted.</em></p>")

    parts.append("<h2>Agent conversations</h2>")
    if d["conversations"]:
        for ex in d["conversations"]:
            parts.append(
                f"<div class='qa'><div>{esc(ex['agent'])} "
                f"<small>(at {esc(ex['checkpoint'] or 'review')}, via "
                f"{esc(ex['backend'])})</small></div>"
                f"<div class='q'>Q: {esc(ex['question'])}</div>"
                f"<div>A: {esc(ex['answer'])}</div></div>"
            )
    else:
        parts.append("<p><em>No questions asked of the agents.</em></p>")

    if d["clarifications"]:
        parts.append("<h2>User clarifications</h2><ul>")
        for c in d["clarifications"]:
            parts.append(f"<li>{esc(c)}</li>")
        parts.append("</ul>")

    parts.append("</body></html>")
    return "".join(parts)


def write_transcript(session: ReviewSession, output_root: str, run_id: str,
                     sensitivity: Any = None) -> dict[str, str]:
    """Write transcript.md/.html/.json next to the dashboards; return paths.

    If ``sensitivity`` (a SensitivityResult) is provided, its table is appended
    to the markdown transcript so the review journey includes the sensitivity
    findings (#4)."""
    out_dir = Path(output_root) / "transcripts" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    md = render_transcript_markdown(session)
    if sensitivity is not None:
        try:
            from start.modeling.sensitivity_analysis import render_sensitivity_markdown
            md += "\n## Sensitivity\n\n" + render_sensitivity_markdown(sensitivity)
        except Exception:
            pass
    paths = {
        "md": out_dir / "transcript.md",
        "html": out_dir / "transcript.html",
        "json": out_dir / "transcript.json",
    }
    paths["md"].write_text(md)
    paths["html"].write_text(render_transcript_html(session))
    paths["json"].write_text(json.dumps(session.to_dict(), indent=2))
    return {k: str(v) for k, v in paths.items()}
