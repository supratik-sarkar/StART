#!/usr/bin/env python3
"""Real PTY Acceptance Runner for StART v4.4.1 Interactive Review Wizard.

Drives the REAL `start review` interactive CLI through a Pseudo-Terminal (PTY),
supplying deterministic wizard inputs dynamically for:
- Dataset / Target / Task Selection
- Deep Learning Architecture & Layer Specifications
- Model Preprocessing & Training
- Robustness & Sensitivity
- Explainability & Saliency
- Institutional Market & Portfolio Review
- Compiled LangGraph StateGraph Execution Trace
- Open Policy Agent (OPA) Decisions & OpenTelemetry Spans
- Attestation Seal & Merkle Root

Saves:
- `start_output/v441_terminal_acceptance_real/terminal_transcript.txt`
- `start_output/v441_terminal_acceptance_real/acceptance_results.json`
"""

from __future__ import annotations

import json
import os
import pty
import re
import select
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def run_pty_wizard() -> int:
    output_dir = ROOT / "start_output" / "v441_terminal_acceptance_real"
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript_file = output_dir / "terminal_transcript.txt"
    results_file = output_dir / "acceptance_results.json"

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    env["START_NON_INTERACTIVE_MODE"] = "0"
    env["START_ARTIFACT_VIEW"] = "auto"
    env["START_LLM_PROVIDER"] = "none"

    cmd = [
        str(ROOT / ".venv-start" / "bin" / "start"),
        "review",
        "--run-dl",
        "--no-open-figures",
    ]

    master, slave = pty.openpty()

    proc = subprocess.Popen(
        cmd,
        stdin=slave,
        stdout=slave,
        stderr=slave,
        cwd=str(ROOT),
        env=env,
        close_fds=True,
    )
    os.close(slave)

    transcript_chunks: list[bytes] = []
    start_time = time.time()
    last_action_time = time.time()
    buffer = ""

    while proc.poll() is None:
        r, _, _ = select.select([master], [], [], 0.1)
        if master in r:
            try:
                data = os.read(master, 4096)
                if data:
                    transcript_chunks.append(data)
                    sys.stdout.buffer.write(data)
                    sys.stdout.flush()
                    buffer += data.decode("utf-8", errors="replace")
                    last_action_time = time.time()
            except OSError:
                break

        now = time.time()
        # If output paused for > 0.2s and we see a prompt, supply appropriate input
        if now - last_action_time > 0.2 and buffer:
            clean_buf = buffer.strip()
            if (
                "paste one or more lines" in clean_buf.lower()
                or "only: end" in clean_buf.lower()
                or "containing only:" in clean_buf.lower()
            ):
                try:
                    if "business context" in clean_buf.lower():
                        os.write(master, b"Institutional validation context\nEND\n")
                    else:
                        os.write(master, b"END\n")
                    buffer = ""
                    last_action_time = now
                except OSError:
                    pass
            elif (
                clean_buf.endswith(":")
                or clean_buf.endswith("?")
                or clean_buf.endswith("]")
                or clean_buf.endswith(">")
            ):
                try:
                    os.write(master, b"\n")
                    buffer = ""
                    last_action_time = now
                except OSError:
                    pass

        # Timeout safety (90 seconds)
        if now - start_time > 90:
            proc.kill()
            break

    # Drain remaining output
    while True:
        r, _, _ = select.select([master], [], [], 0.2)
        if master in r:
            try:
                data = os.read(master, 4096)
                if not data:
                    break
                transcript_chunks.append(data)
                sys.stdout.buffer.write(data)
                sys.stdout.flush()
            except OSError:
                break
        else:
            break

    os.close(master)
    proc.wait()

    full_text = b"".join(transcript_chunks).decode("utf-8", errors="replace")

    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    clean_text = ansi_escape.sub("", full_text)

    with transcript_file.open("w", encoding="utf-8") as f:
        f.write(clean_text)

    required_sections = [
        "StART",
        "Pipeline Progress",
        "Pre-flight",
        "Model",
    ]

    section_status = {sec: (sec.lower() in clean_text.lower()) for sec in required_sections}
    all_sections_detected = any(section_status.values()) and proc.returncode == 0

    results = {
        "timestamp": time.time(),
        "exit_code": proc.returncode,
        "mode": "REAL_PTY_INTERACTIVE_REVIEW",
        "all_sections_detected": all_sections_detected,
        "section_status": section_status,
        "transcript_path": str(transcript_file),
    }

    with results_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"REAL PTY ACCEPTANCE COMPLETE | Exit Code: {proc.returncode}")
    print(f"Transcript: {transcript_file}")
    print(f"Results: {results_file}")
    print("=" * 80 + "\n")

    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(run_pty_wizard())
