#!/usr/bin/env python3
"""Private current-run artifact board for the terminal-native StART review.

Callers must provide both an exact run directory and its run ID. The board
never scans for a newest directory, reads a showcase/archive location, or
opens a browser or plot window itself.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PaneGeometry:
    x: int
    y: int
    width: int
    height: int

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


def landscape_geometry(
    screen_width: int = 1920,
    screen_height: int = 1080,
    terminal_ratio: float = 0.62,
) -> dict[str, PaneGeometry]:
    """Return deterministic, gap-free bounds constrained to one screen."""
    if screen_width < 800 or screen_height < 600:
        raise ValueError("landscape board requires a screen of at least 800x600")
    if not 0.5 <= terminal_ratio <= 0.8:
        raise ValueError("terminal_ratio must be between 0.5 and 0.8")
    terminal_width = min(max(1, round(screen_width * terminal_ratio)), screen_width - 1)
    return {
        "terminal": PaneGeometry(0, 0, terminal_width, screen_height),
        "artifact_board": PaneGeometry(
            terminal_width, 0, screen_width - terminal_width, screen_height
        ),
    }


def load_current_run_manifest(run_path: str | Path, run_id: str) -> dict[str, Any]:
    """Load and validate one exact current-run artifact manifest."""
    root = Path(run_path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"run path is not a directory: {root}")
    if not run_id or run_id.lower() in {"latest", "newest", "current"}:
        raise ValueError("an exact non-alias run ID is required")
    manifest_path = root / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"current-run artifact manifest not found: {manifest_path}; "
            "the board will not fall back to historical output"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema = manifest.get("schema")
    if schema not in {"start.current-run-artifacts/1", "start.current-run-artifacts/2"}:
        raise ValueError("unsupported or missing current-run artifact manifest schema")
    declared_run_id = (
        manifest.get("review_run_id")
        if schema == "start.current-run-artifacts/2"
        else manifest.get("run_id")
    )
    if declared_run_id != run_id:
        raise ValueError(
            f"run ID mismatch: requested {run_id!r}, manifest contains {declared_run_id!r}"
        )
    declared_root = Path(
        str(
            manifest.get("presentation_root")
            if schema == "start.current-run-artifacts/2"
            else manifest.get("run_path", "")
        )
    ).expanduser().resolve()
    if declared_root != root:
        raise ValueError(f"run path mismatch: requested {root}, manifest declares {declared_root}")
    allowed_roots = [
        Path(str(value)).expanduser().resolve(strict=True)
        for value in manifest.get(
            "scientific_artifact_roots"
            if schema == "start.current-run-artifacts/2"
            else "artifact_roots",
            [str(root)],
        )
    ]
    if root not in allowed_roots:
        allowed_roots.append(root)

    groups = manifest.get("groups")
    if not isinstance(groups, dict):
        raise ValueError("manifest groups must be an object")
    allowed_groups = {"flight_a", "flight_b", "flight_c"}
    if set(groups) - allowed_groups:
        raise ValueError(f"unsupported artifact groups: {sorted(set(groups) - allowed_groups)}")
    for group_name in allowed_groups:
        entries = groups.setdefault(group_name, [])
        if not isinstance(entries, list):
            raise ValueError(f"manifest group {group_name!r} must be a list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"manifest entry in {group_name!r} must be an object")
            owner_run_id = str(entry.get("owner_run_id") or entry.get("run_id") or "")
            parent_run_id = str(entry.get("parent_review_run_id") or owner_run_id)
            if schema == "start.current-run-artifacts/2":
                if parent_run_id != run_id:
                    raise ValueError("artifact entry is not bound to the requested review run ID")
            elif owner_run_id != run_id:
                raise ValueError("artifact entry is not bound to the requested run ID")
            raw_path = entry.get("file_path")
            if raw_path:
                artifact_path = Path(str(raw_path)).expanduser().resolve(strict=True)
                if not any(artifact_path.is_relative_to(base) for base in allowed_roots):
                    raise ValueError(f"artifact escapes declared current-run roots: {artifact_path}")
                entry["file_path"] = str(artifact_path)
            companion = entry.get("semantic_companion")
            if companion:
                companion_path = Path(str(companion)).expanduser().resolve(strict=True)
                if not any(companion_path.is_relative_to(base) for base in allowed_roots):
                    raise ValueError(
                        f"semantic companion escapes declared current-run roots: {companion_path}"
                    )
                entry["semantic_companion"] = str(companion_path)
            entry["run_id"] = run_id
            entry["owner_run_id"] = owner_run_id or run_id
    manifest["run_id"] = run_id
    manifest["run_path"] = str(root)
    return manifest


def _file_data_uri(path: Path) -> str:
    mime = {
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(path.suffix.lower())
    if not mime:
        return ""
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def _semantic_preview(entry: dict[str, Any]) -> str:
    companion = entry.get("semantic_companion")
    candidate = Path(companion) if companion else None
    if candidate is None or not candidate.is_file():
        file_path = entry.get("file_path")
        if file_path and Path(file_path).suffix.lower() == ".json":
            candidate = Path(file_path)
    if candidate is None or not candidate.is_file():
        return "No inline visual. Canonical provenance remains available in the run manifest."
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        text = json.dumps(payload, indent=2, ensure_ascii=False)
    except Exception:
        text = candidate.read_text(encoding="utf-8", errors="replace")
    return text[:5000] + ("\n…" if len(text) > 5000 else "")


def _artifact_card(entry: dict[str, Any]) -> str:
    file_path = Path(entry["file_path"]) if entry.get("file_path") else None
    title = html.escape(str(entry.get("title") or entry.get("artifact_id") or "Artifact"))
    artifact_id = html.escape(str(entry.get("artifact_id", "UNSPECIFIED")))
    evidence = ", ".join(str(v) for v in entry.get("evidence_ids", [])) or "NONE"
    test_id = html.escape(str(entry.get("test_id", "UNSPECIFIED")))
    run_id = html.escape(str(entry.get("run_id", "UNSPECIFIED")))
    provenance = html.escape(
        f"Run {entry.get('run_id', 'UNSPECIFIED')} | Evidence {evidence} | "
        f"Surface {entry.get('test_id', 'UNSPECIFIED')}",
        quote=True,
    )
    data_uri = _file_data_uri(file_path) if file_path and file_path.is_file() else ""
    if data_uri:
        content = (
            f'<button class="visual-button" type="button" data-title="{title}" '
            f'data-image="{data_uri}" data-provenance="{provenance}" '
            f'aria-label="Enlarge {title}">'
            f'<img src="{data_uri}" alt="{title}" loading="lazy" /></button>'
        )
    else:
        preview = html.escape(_semantic_preview(entry))
        content = (
            f'<button class="semantic-button" type="button" data-title="{title}" '
            f'data-semantic="{preview}" data-provenance="{provenance}" '
            f'aria-label="Inspect {title}"><pre>{preview}</pre></button>'
        )
    return f"""
      <article class="artifact-card">
        <div class="card-header"><h3>{title}</h3><code>{artifact_id}</code></div>
        <div class="card-content">{content}</div>
        <dl class="provenance">
          <div><dt>Run</dt><dd>{run_id}</dd></div>
          <div><dt>Evidence</dt><dd>{html.escape(evidence)}</dd></div>
          <div><dt>Surface</dt><dd>{test_id}</dd></div>
        </dl>
      </article>
    """


_CURATION_TOKENS: dict[str, tuple[str, ...]] = {
    "flight_a": ("temporal-contract", "saliency", "robustness", "performance", "metrics"),
    "flight_b": ("dendrogram", "seriated", "correlation", "covariance", "asset_weights", "risk_contribution"),
    "flight_c": (
        "cev",
        "stanton",
        "var_pnl_timeline",
        "backtest",
        "exception_transition",
        "scenario_pnl",
        "reverse_stress",
        "short_rate",
        "diffusion",
    ),
}


def curate_entries(group: str, entries: list[dict[str, Any]], limit: int = 6) -> list[dict[str, Any]]:
    """Select high-value artifacts from canonical metadata, never filenames or run IDs."""
    tokens = _CURATION_TOKENS[group]

    def score(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        index, entry = item
        searchable = " ".join(
            str(entry.get(key, "")).lower()
            for key in ("artifact_type", "title", "test_id", "artifact_id")
        ).replace(" ", "_")
        return next((rank for rank, token in enumerate(tokens) if token in searchable), len(tokens)), index

    return [entry for _, entry in sorted(enumerate(entries), key=score)[:limit]]


def _selector(entry: dict[str, Any], *, selected: bool) -> str:
    file_path = Path(entry["file_path"]) if entry.get("file_path") else None
    title = str(entry.get("title") or entry.get("artifact_id") or "Artifact")
    evidence_ids = [str(value) for value in entry.get("evidence_ids", [])]
    semantic_hash = str(entry.get("semantic_payload_hash", ""))
    evidence = ", ".join(evidence_ids)
    proof = (
        f"Evidence {evidence}"
        if evidence
        else f"Semantic sha256:{semantic_hash[:16]}…"
        if semantic_hash
        else "Evidence NOT_APPLICABLE"
    )
    provenance = (
        f"Run {entry.get('run_id', 'UNSPECIFIED')} | Owner {entry.get('owner_run_id', 'UNSPECIFIED')} | "
        f"{proof} | Surface {entry.get('test_id', 'UNSPECIFIED')}"
    )
    image_uri = _file_data_uri(file_path) if file_path and file_path.is_file() else ""
    semantic = "" if image_uri else _semantic_preview(entry)
    preview = (
        f'<img src="{image_uri}" alt="" />'
        if image_uri
        else '<span class="semantic-mark">DATA</span>'
    )
    return (
        f'<button class="selector{" selected" if selected else ""}" type="button" '
        f'data-title="{html.escape(title, quote=True)}" '
        f'data-provenance="{html.escape(provenance, quote=True)}" '
        f'data-image="{html.escape(image_uri, quote=True)}" '
        f'data-semantic="{html.escape(semantic, quote=True)}">'
        f'{preview}<span>{html.escape(title)}</span></button>'
    )


def build_artifact_board_html(manifest: dict[str, Any]) -> str:
    """Build a self-contained, no-scroll 38% landscape artifact surface."""
    run_id = str(manifest["run_id"])
    meta = (
        ("flight_a", "Flight A", "Temporal model evidence"),
        ("flight_b", "Flight B", "Portfolio construction"),
        ("flight_c", "Flight C", "Market risk and stress"),
    )
    tabs: list[str] = []
    sections: list[str] = []
    active_assigned = False
    for group, label, description in meta:
        entries = curate_entries(group, list(manifest["groups"].get(group, [])))
        if not entries:
            continue
        active = not active_assigned
        active_assigned = True
        tabs.append(
            f'<button class="tab{" active" if active else ""}" data-group="{group}" type="button">'
            f'{label}<small>{len(entries)}</small></button>'
        )
        selectors = "".join(
            _selector(entry, selected=index == 0) for index, entry in enumerate(entries)
        )
        sections.append(
            f'<section class="flight{" active" if active else ""}" data-flight="{group}">'
            f'<header><strong>{label}</strong><span>{html.escape(description)}</span></header>'
            '<button class="hero" type="button"><div class="hero-media"></div>'
            '<div class="hero-copy"><strong></strong><span></span></div></button>'
            f'<div class="selectors">{selectors}</div></section>'
        )
    if not sections:
        sections.append(
            '<section class="flight active empty"><strong>NO CURRENT-RUN VISUALS</strong>'
            '<span>The manifest is valid, but no applicable scientific visual was produced.</span></section>'
        )

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>StART Current-Run Artifacts — {html.escape(run_id)}</title>
<style>
:root{{color-scheme:dark;--bg:#080d13;--panel:#101720;--line:#293647;--text:#edf4fa;--muted:#8fa0b3;--cyan:#67d4ff;--green:#62d59b}}
*{{box-sizing:border-box}}html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:var(--bg);color:var(--text);font:14px/1.35 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
body{{display:grid;grid-template-rows:auto auto 1fr auto}}.top{{padding:14px 16px 10px;border-bottom:1px solid var(--line);background:#0d141d}}h1{{margin:0;font-size:17px;letter-spacing:.03em}}.top p{{margin:4px 0 0;color:var(--muted);font:10px ui-monospace,SFMono-Regular,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
nav{{display:flex;gap:6px;padding:8px;border-bottom:1px solid var(--line)}}button{{font:inherit;color:inherit}}.tab{{flex:1;border:1px solid var(--line);border-radius:6px;padding:7px;background:#111a24;cursor:pointer}}.tab.active{{border-color:var(--cyan);color:var(--cyan);background:#102231}}.tab small{{display:block;color:var(--muted);font-size:9px}}
main{{min-height:0;padding:8px}}.flight{{display:none;height:100%;min-height:0;grid-template-rows:auto 1fr 116px;gap:7px}}.flight.active{{display:grid}}.flight>header{{display:flex;justify-content:space-between;gap:8px;color:var(--muted);font-size:11px}}.flight>header strong{{color:var(--cyan);font-size:13px}}
.hero{{min-height:0;display:grid;grid-template-rows:1fr auto;width:100%;padding:0;border:1px solid var(--line);border-radius:8px;background:#fff;overflow:hidden;cursor:zoom-in}}.hero-media{{min-height:0;display:grid;place-items:center;overflow:hidden}}.hero-media img{{display:block;width:100%;height:100%;object-fit:contain}}.hero-media pre{{width:100%;height:100%;margin:0;padding:12px;overflow:hidden;text-align:left;background:#0b1118;color:#b8c7d5;font:10px/1.3 ui-monospace,SFMono-Regular,monospace;white-space:pre-wrap}}.hero-copy{{display:grid;gap:2px;padding:7px 9px;text-align:left;background:#0c131b;border-top:1px solid var(--line)}}.hero-copy strong{{font-size:12px}}.hero-copy span{{color:var(--muted);font:9px ui-monospace,SFMono-Regular,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.selectors{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));grid-template-rows:repeat(2,minmax(0,1fr));gap:6px;overflow:hidden}}.selector{{min-width:0;display:grid;grid-template-columns:45px 1fr;align-items:center;gap:6px;border:1px solid var(--line);border-radius:6px;padding:5px;background:#101821;text-align:left;cursor:pointer;overflow:hidden}}.selector.selected{{border-color:var(--green);background:#10231d}}.selector img{{width:45px;height:42px;object-fit:cover;background:#fff}}.selector span{{font-size:10px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}}.semantic-mark{{display:grid;place-items:center;width:45px;height:42px;color:var(--cyan);font:9px ui-monospace,SFMono-Regular,monospace;background:#091018}}
.empty{{place-content:center;text-align:center;color:var(--muted)}}footer{{padding:6px 10px;border-top:1px solid var(--line);text-align:center;color:var(--muted);font-size:9px}}dialog{{width:min(94vw,700px);height:min(92vh,980px);padding:0;border:1px solid #51667f;border-radius:10px;background:#080d13;color:var(--text)}}dialog::backdrop{{background:rgba(0,0,0,.82)}}.lightbox-head{{height:52px;display:grid;grid-template-columns:1fr auto;gap:8px;align-items:center;padding:0 12px;border-bottom:1px solid var(--line)}}.lightbox-head div{{min-width:0}}.lightbox-head span{{display:block;color:var(--muted);font:9px ui-monospace,SFMono-Regular,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}.lightbox-head button{{padding:7px 10px;border:1px solid var(--line);border-radius:5px;background:#121b25;cursor:pointer}}.lightbox-body{{height:calc(100% - 52px);display:grid;place-items:center;overflow:auto;background:#fff}}.lightbox-body img{{max-width:100%;max-height:100%;object-fit:contain}}.lightbox-body pre{{width:100%;height:100%;margin:0;padding:18px;overflow:auto;background:#0b1118;color:#c4d1dc;font:12px/1.4 ui-monospace,SFMono-Regular,monospace;white-space:pre-wrap}}
</style></head><body>
<header class="top"><h1>StART — CURRENT-RUN SCIENTIFIC ARTIFACTS</h1><p>{html.escape(run_id)} · exact manifest · no historical fallback</p></header>
<nav>{''.join(tabs)}</nav><main>{''.join(sections)}</main>
<footer>Secondary evidence surface · terminal remains primary · click the current visual to enlarge</footer>
<dialog id="lightbox"><div class="lightbox-head"><div><strong id="lightbox-title"></strong><span id="lightbox-provenance"></span></div><button id="lightbox-close" type="button">Return to review</button></div><div id="lightbox-body" class="lightbox-body"></div></dialog>
<script>
const dialog=document.getElementById('lightbox'),boxBody=document.getElementById('lightbox-body'),boxTitle=document.getElementById('lightbox-title'),boxProv=document.getElementById('lightbox-provenance');let interacted=false;
function activate(button){{const flight=button.closest('.flight'),hero=flight.querySelector('.hero'),media=hero.querySelector('.hero-media');flight.querySelectorAll('.selector').forEach(x=>x.classList.toggle('selected',x===button));hero.dataset.title=button.dataset.title;hero.dataset.provenance=button.dataset.provenance;hero.dataset.image=button.dataset.image;hero.dataset.semantic=button.dataset.semantic;hero.querySelector('.hero-copy strong').textContent=button.dataset.title;hero.querySelector('.hero-copy span').textContent=button.dataset.provenance;media.replaceChildren();if(button.dataset.image){{const img=document.createElement('img');img.src=button.dataset.image;img.alt=button.dataset.title;media.appendChild(img)}}else{{const pre=document.createElement('pre');pre.textContent=button.dataset.semantic;media.appendChild(pre)}}}}
document.querySelectorAll('.flight').forEach(f=>{{const first=f.querySelector('.selector');if(first)activate(first)}});document.querySelectorAll('.selector').forEach(b=>b.addEventListener('click',()=>{{interacted=true;activate(b)}}));document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>{{interacted=true;document.querySelectorAll('.tab').forEach(x=>x.classList.toggle('active',x===tab));document.querySelectorAll('.flight').forEach(x=>x.classList.toggle('active',x.dataset.flight===tab.dataset.group))}}));
document.querySelectorAll('.hero').forEach(hero=>hero.addEventListener('click',()=>{{interacted=true;boxTitle.textContent=hero.dataset.title;boxProv.textContent=hero.dataset.provenance;boxBody.replaceChildren();if(hero.dataset.image){{const img=document.createElement('img');img.src=hero.dataset.image;img.alt=hero.dataset.title;boxBody.appendChild(img)}}else{{const pre=document.createElement('pre');pre.textContent=hero.dataset.semantic;boxBody.appendChild(pre)}}dialog.showModal()}}));document.getElementById('lightbox-close').addEventListener('click',()=>dialog.close());dialog.addEventListener('click',e=>{{if(e.target===dialog)dialog.close()}});setInterval(()=>{{if(interacted||dialog.open)return;const flight=document.querySelector('.flight.active'),items=[...flight.querySelectorAll('.selector')];if(items.length<2)return;const current=items.findIndex(x=>x.classList.contains('selected'));activate(items[(current+1)%items.length])}},9000);
</script></body></html>"""


def render_board(
    *, run_path: str | Path, run_id: str, dest_file: str | Path | None = None
) -> Path:
    """Render one exact run; destination defaults inside that run."""
    root = Path(run_path).expanduser().resolve(strict=True)
    manifest = load_current_run_manifest(root, run_id)
    destination = (
        Path(dest_file).expanduser().resolve()
        if dest_file is not None
        else root / "artifact_board.html"
    )
    if not destination.is_relative_to(root):
        raise ValueError("artifact board destination must remain inside the exact run directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_artifact_board_html(manifest), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an exact StART current-run artifact board")
    parser.add_argument("--run-path", required=True, help="Exact current run directory")
    parser.add_argument("--run-id", required=True, help="Exact run ID expected in the manifest")
    parser.add_argument("--dest", help="Destination HTML path inside the run directory")
    parser.add_argument("--geometry", action="store_true", help="Print 1920x1080 pane geometry")
    args = parser.parse_args()
    if args.geometry:
        print(json.dumps({k: v.to_dict() for k, v in landscape_geometry().items()}, indent=2))
    out_path = render_board(run_path=args.run_path, run_id=args.run_id, dest_file=args.dest)
    print(f"Artifact board rendered: {out_path}")


if __name__ == "__main__":
    main()
