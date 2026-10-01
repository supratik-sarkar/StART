#!/usr/bin/env python3
"""Collect explicitly named demo session results without any ``latest`` lookup."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def collect_results(paths: list[str | Path], output: str | Path) -> Path:
    if not paths:
        raise ValueError("at least one explicit session_result.json path is required")
    sessions: list[dict[str, Any]] = []
    seen: set[str] = set()
    sources: list[str] = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.name != "session_result.json" or not path.is_file():
            raise FileNotFoundError(f"explicit session result not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        session_id = str(payload.get("session_id", ""))
        if not session_id or session_id in seen:
            raise ValueError(f"missing or duplicate session ID in {path}")
        if str(payload.get("result_path", "")) != str(path):
            raise ValueError(f"session result provenance does not point to itself: {path}")
        seen.add(session_id)
        sources.append(str(path))
        sessions.append(payload)

    summary = {
        "schema": "start.demo-diagnostic-summary/1",
        "sources": sources,
        "sessions": sessions,
        "all_complete": all(s.get("controller_status") == "COMPLETE" for s in sessions),
    }
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect exact StART demo session results")
    parser.add_argument("session_results", nargs="+", help="Explicit session_result.json paths")
    parser.add_argument("--output", required=True, help="Exact diagnostic_summary.json output path")
    args = parser.parse_args()
    print(collect_results(args.session_results, args.output))


if __name__ == "__main__":
    main()
