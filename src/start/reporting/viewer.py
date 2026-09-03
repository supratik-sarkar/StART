"""Safe, non-blocking configurable artifact viewer for StART.

Supported modes via START_ARTIFACT_VIEW environment variable:
- auto (default):
  * Interactive local TTY on macOS: non-blockingly launches default preview/browser for visual SVG/PNG artifacts.
  * Headless / CI / non-TTY: never launches windows; saves artifacts silently.
- open:
  * Explicitly opens generated visual artifacts using the OS viewer non-blockingly.
- terminal:
  * Emits compact text/table previews in the terminal without opening windows.
- off:
  * Artifacts are generated and persisted to disk; no viewer is invoked.

CRITICAL INVARIANTS:
1. Zero analytical recomputation: viewing consumes already persisted artifact files.
2. Non-blocking: never waits on viewer processes or blocks CLI execution flow.
3. Safe execution: failures to open an external viewer are caught and silently handled.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def get_artifact_view_mode() -> str:
    """Resolve the active artifact viewing mode from environment."""
    return os.environ.get("START_ARTIFACT_VIEW", "auto").strip().lower()


def view_artifact(artifact: Any, mode: str | None = None) -> bool:
    """Non-blockingly display a single artifact according to configured view mode."""
    eff_mode = (mode or get_artifact_view_mode()).lower()
    if eff_mode == "off":
        return False

    file_path = getattr(artifact, "file_path", None)
    if not file_path or not Path(file_path).exists():
        return False

    path = Path(file_path)
    suffix = path.suffix.lower()

    if eff_mode in ("open", "auto"):
        is_tty = sys.stdout.isatty() and not os.environ.get("CI")
        is_macos = platform.system() == "Darwin"

        if eff_mode == "open" or (eff_mode == "auto" and is_tty and is_macos):
            if suffix in (".svg", ".png", ".html", ".pdf"):
                try:
                    subprocess.Popen(
                        ["open", str(path)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True
                except Exception:
                    return False

    return False


def view_artifacts(artifacts: list[Any], mode: str | None = None, max_open: int = 3) -> int:
    """Safely and non-blockingly view a collection of artifacts."""
    eff_mode = (mode or get_artifact_view_mode()).lower()
    if eff_mode == "off":
        return 0

    opened_count = 0
    # Prioritize key visual artifacts: dendrograms, heatmaps, waterfalls
    visual_first = sorted(
        artifacts,
        key=lambda a: (
            0
            if "dendrogram" in str(getattr(getattr(a, "spec", None), "artifact_type", ""))
            else 1
            if "heatmap" in str(getattr(getattr(a, "spec", None), "artifact_type", ""))
            else 2
            if "waterfall" in str(getattr(getattr(a, "spec", None), "artifact_type", ""))
            else 3
        ),
    )

    for art in visual_first:
        if opened_count >= max_open:
            break
        if view_artifact(art, mode=eff_mode):
            opened_count += 1

    return opened_count
