#!/usr/bin/env python3
"""StART Live Interactive Acceptance Harness.

Drives the real interactive StART review CLI programmatically from the canonical Markdown runbook:
StART_v4.3.0_Market_Manual_Acceptance_Runbook.md.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import hashlib
import json
import os
import pty
import re
import select
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

# Canonical Paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
RUNBOOK_PATH = WORKSPACE_ROOT / "StART_v4.3.0_Market_Manual_Acceptance_Runbook.md"
if not RUNBOOK_PATH.exists():
    _arch_candidate = WORKSPACE_ROOT / "docs" / "StART_v4.3.0_Market_Manual_Acceptance_Runbook.md"
    if _arch_candidate.exists():
        RUNBOOK_PATH = _arch_candidate
DEFAULT_START_BIN = WORKSPACE_ROOT / ".venv-start" / "bin" / "start"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "start_output" / "acceptance_runs"

# ANSI stripping
ANSI_ESCAPE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

# Timeouts in seconds
DEFAULT_CLI_TIMEOUT = 30.0
DEFAULT_PROVIDER_TIMEOUT = 180.0
DEFAULT_TOTAL_TIMEOUT = 900.0


@dataclasses.dataclass
class ManifestStep:
    step_id: str
    heading: str
    kind: str  # "choice", "multiline", "action", "text"
    content: str
    content_sha256: str
    summary: str


@dataclasses.dataclass
class RunbookManifest:
    runbook_path: str
    runbook_sha256: str
    steps: list[ManifestStep]


class RunbookParser:
    """Parses actionable inputs from the Markdown runbook."""

    def __init__(self, runbook_path: Path = RUNBOOK_PATH):
        self.runbook_path = runbook_path
        if not self.runbook_path.exists():
            raise FileNotFoundError(f"Runbook not found at: {self.runbook_path}")
        self.raw_content = self.runbook_path.read_text(encoding="utf-8")
        self.sha256 = hashlib.sha256(self.raw_content.encode("utf-8")).hexdigest()

    def _get_section(self, heading_regex: str) -> str:
        pattern = re.compile(rf"^#+\s+{heading_regex}.*?(?=^#+\s+\d|\Z)", re.MULTILINE | re.DOTALL)
        m = pattern.search(self.raw_content)
        if not m:
            raise ValueError(f"Section matching {heading_regex!r} not found in runbook")
        return m.group(0)

    def _extract_fenced_blocks(self, section_text: str, sub_heading: str | None = None) -> list[str]:
        target_text = section_text
        if sub_heading:
            pattern = re.compile(
                rf"^##+\s+{re.escape(sub_heading)}.*?(?=^##+|\Z)",
                re.MULTILINE | re.DOTALL,
            )
            m = pattern.search(section_text)
            if not m:
                raise ValueError(f"Sub-heading {sub_heading!r} not found in section")
            target_text = m.group(0)

        blocks = re.findall(r"```(?:text|bash)?\n(.*?)```", target_text, re.DOTALL)
        return [b.strip() for b in blocks]

    def build_manifest(self) -> RunbookManifest:
        steps: list[ManifestStep] = []
        step_idx = 1

        def add_step(heading: str, kind: str, content: str) -> None:
            nonlocal step_idx
            c_sha = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
            summary = content if len(content) <= 50 and "\n" not in content else f"len={len(content)}"
            steps.append(
                ManifestStep(
                    step_id=f"{step_idx:02d}",
                    heading=heading,
                    kind=kind,
                    content=content,
                    content_sha256=c_sha,
                    summary=summary,
                )
            )
            step_idx += 1

        # 1. Setup
        sec_setup = self._get_section(r"1\.\s+Review Setup Selections")
        add_step("Review Mode", "choice", self._extract_fenced_blocks(sec_setup, "Select Review Mode")[0])
        add_step("Review Domain", "choice", self._extract_fenced_blocks(sec_setup, "Select Review Domain")[0])
        add_step(
            "Backend", "choice", self._extract_fenced_blocks(sec_setup, "Select AI Reviewer Agent Backend")[0]
        )
        add_step(
            "Provider", "choice", self._extract_fenced_blocks(sec_setup, "Select Public LLM Provider")[0]
        )
        add_step("Model", "choice", self._extract_fenced_blocks(sec_setup, "Select OpenAI Model")[0])
        add_step(
            "Materiality", "choice", self._extract_fenced_blocks(sec_setup, "Select Model Materiality")[0]
        )
        add_step("Lifecycle", "choice", self._extract_fenced_blocks(sec_setup, "Select Review Lifecycle")[0])

        # 2. Governance
        sec_gov = self._get_section(r"2\.\s+Governance Information")
        add_step("Business Context", "multiline", self._extract_fenced_blocks(sec_gov, "Business Context")[0])
        add_step(
            "Reviewer Clarification",
            "multiline",
            self._extract_fenced_blocks(sec_gov, "Reviewer Clarification")[0],
        )
        add_step(
            "Intended Use",
            "multiline",
            self._extract_fenced_blocks(sec_gov, "Intended Use / Decision Impact")[0],
        )
        add_step(
            "Known Limitations",
            "multiline",
            self._extract_fenced_blocks(sec_gov, "Known Limitations / Reviewer Concerns")[0],
        )

        # 3. Market Data & Scope
        sec_data = self._get_section(r"3\.\s+Market Data & Scope")
        add_step(
            "Data Source",
            "choice",
            self._extract_fenced_blocks(sec_data, "Select Market/Treasury Data Source")[0],
        )
        add_step("Scope", "choice", self._extract_fenced_blocks(sec_data, "Select Review Scope")[0])
        add_step("Proceed", "choice", self._extract_fenced_blocks(sec_data, "Proceed to Execute Review")[0])

        # 4. Portfolio Checkpoint
        sec_port = self._get_section(r"4\.\s+Portfolio Risk & Volatility Assumptions")
        add_step("Portfolio Action", "action", self._extract_fenced_blocks(sec_port)[0])

        # 5. Factor Checkpoint
        sec_factor = self._get_section(r"5\.\s+Factor Modeling & Attribution Assumptions")
        add_step(
            "Factor View Artifacts",
            "action",
            self._extract_fenced_blocks(sec_factor, "View checkpoint artifacts")[0],
        )
        factor_q_blocks = self._extract_fenced_blocks(sec_factor, "Ask reviewer")
        add_step("Factor Ask Action", "action", factor_q_blocks[0])
        add_step("Factor Question Text", "text", factor_q_blocks[1])
        add_step("Factor Accept Action", "action", factor_q_blocks[2] if len(factor_q_blocks) > 2 else "a")

        # 6. VaR Checkpoint
        sec_var = self._get_section(r"6\.\s+VaR Backtesting & Exception Frequency")
        add_step("VaR View Artifacts", "action", self._extract_fenced_blocks(sec_var, "View VaR artifact")[0])
        var_q_blocks = self._extract_fenced_blocks(sec_var, "Ask the critical VaR question")
        add_step("VaR Ask Action", "action", var_q_blocks[0])
        add_step("VaR Question Text", "text", var_q_blocks[1])
        var_c_blocks = self._extract_fenced_blocks(sec_var, "Challenge VaR result")
        add_step("VaR Challenge Action", "action", var_c_blocks[0])
        add_step("VaR Challenge Text", "text", var_c_blocks[1])
        add_step("VaR Accept Action", "action", var_c_blocks[2] if len(var_c_blocks) > 2 else "a")

        # 7. Covariance Checkpoint
        sec_cov = self._get_section(r"7\.\s+Covariance Structure & Missing Data Treatment")
        add_step(
            "Covariance View Artifacts", "action", self._extract_fenced_blocks(sec_cov, "View artifacts")[0]
        )
        cov_q_blocks = self._extract_fenced_blocks(sec_cov, "Ask reviewer")
        add_step("Covariance Ask Action", "action", cov_q_blocks[0])
        add_step("Covariance Question Text", "text", cov_q_blocks[1])
        cov_c_blocks = self._extract_fenced_blocks(sec_cov, "Challenge covariance")
        add_step("Covariance Challenge Action", "action", cov_c_blocks[0])
        add_step("Covariance Challenge Text", "text", cov_c_blocks[1])
        add_step("Covariance Accept Action", "action", cov_c_blocks[2] if len(cov_c_blocks) > 2 else "a")

        # 8. Scenario Checkpoint
        sec_scen = self._get_section(r"8\.\s+Scenario Analysis & Stress Testing")
        add_step(
            "Scenario View Artifacts",
            "action",
            self._extract_fenced_blocks(sec_scen, "View scenario artifacts")[0],
        )
        scen_q_blocks = self._extract_fenced_blocks(sec_scen, "Ask reviewer")
        add_step("Scenario Ask Action", "action", scen_q_blocks[0])
        add_step("Scenario Question Text", "text", scen_q_blocks[1])
        scen_c_blocks = self._extract_fenced_blocks(sec_scen, "Challenge scenario")
        add_step("Scenario Challenge Action", "action", scen_c_blocks[0])
        add_step("Scenario Challenge Text", "text", scen_c_blocks[1])
        add_step("Scenario Accept Action", "action", scen_c_blocks[2] if len(scen_c_blocks) > 2 else "a")

        # 9. Committee Checkpoint
        sec_comm = self._get_section(r"9\.\s+Cross-Analytical Committee Synthesis")
        add_step(
            "Committee View Artifacts",
            "action",
            self._extract_fenced_blocks(sec_comm, "View all artifacts")[0],
        )
        comm_q_blocks = self._extract_fenced_blocks(sec_comm, "Ask committee")
        add_step("Committee Ask Action", "action", comm_q_blocks[0])
        add_step("Committee Question Text", "text", comm_q_blocks[1])
        comm_c_blocks = self._extract_fenced_blocks(sec_comm, "Challenge committee synthesis")
        add_step("Committee Challenge Action", "action", comm_c_blocks[0])
        add_step("Committee Challenge Text", "text", comm_c_blocks[1])
        add_step("Committee Accept Action", "action", comm_c_blocks[2] if len(comm_c_blocks) > 2 else "a")

        # 11. Governance Checkpoint
        sec_gov_chk = self._get_section(r"11\.\s+Model Governance & Attestation Sign-Off")
        add_step(
            "Governance View Artifacts",
            "action",
            self._extract_fenced_blocks(sec_gov_chk, "View all artifacts")[0],
        )
        gov_q_blocks = self._extract_fenced_blocks(sec_gov_chk, "Ask governance reviewer")
        add_step("Governance Ask Action", "action", gov_q_blocks[0])
        add_step("Governance Question Text", "text", gov_q_blocks[1])
        gov_c_blocks = self._extract_fenced_blocks(sec_gov_chk, "Challenge unconditional approval")
        add_step("Governance Challenge Action", "action", gov_c_blocks[0])
        add_step("Governance Challenge Text", "text", gov_c_blocks[1])
        add_step(
            "Governance Final Accept", "action", self._extract_fenced_blocks(sec_gov_chk, "Final Accept")[0]
        )

        return RunbookManifest(
            runbook_path=str(self.runbook_path),
            runbook_sha256=self.sha256,
            steps=steps,
        )


def print_manifest_summary(manifest: RunbookManifest) -> None:
    """Print sanitized manifest census before execution."""
    print("================================================================================")
    print("StART Runbook Manifest Census")
    print(f"File:   {manifest.runbook_path}")
    print(f"SHA256: {manifest.runbook_sha256}")
    print(f"Steps:  {len(manifest.steps)}")
    print("--------------------------------------------------------------------------------")
    for step in manifest.steps:
        first_line = step.content.split("\n")[0]
        if len(first_line) > 60:
            first_line = first_line[:57] + "..."
        print(f"{step.step_id} {step.heading:<28} [{step.kind:<9}] (sha={step.content_sha256}): {first_line}")
    print("================================================================================")


class PtySession:
    """PTY wrapper for driving interactive CLI with EXPECT -> SEND sequencing."""

    def __init__(self, cmd: list[str], cwd: str, env: dict[str, str]):
        import fcntl
        import struct
        import termios

        self.master_fd, self.slave_fd = pty.openpty()

        # Set terminal window size to 60 rows x 220 cols to prevent Rich column truncation
        winsize = struct.pack("HHHH", 60, 220, 0, 0)
        try:
            fcntl.ioctl(self.slave_fd, termios.TIOCSWINSZ, winsize)
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
        except Exception:
            pass

        self.proc = subprocess.Popen(
            cmd,
            stdin=self.slave_fd,
            stdout=self.slave_fd,
            stderr=self.slave_fd,
            cwd=cwd,
            env=env,
            close_fds=True,
            preexec_fn=os.setsid,
        )
        os.close(self.slave_fd)
        self.slave_fd = -1
        self.raw_transcript: list[str] = []
        self.clean_buffer: str = ""
        self.total_start_time = time.time()

    def read_available(self, timeout: float = 0.1) -> str:
        """Read currently available output from PTY master."""
        r, _, _ = select.select([self.master_fd], [], [], timeout)
        if not r:
            return ""
        try:
            chunk = os.read(self.master_fd, 4096).decode("utf-8", errors="replace")
        except OSError:
            return ""
        if chunk:
            self.raw_transcript.append(chunk)
            clean_chunk = ANSI_ESCAPE.sub("", chunk)
            self.clean_buffer += clean_chunk
            sys.stdout.write(chunk)
            sys.stdout.flush()
        return chunk

    def expect(
        self,
        patterns: list[str | re.Pattern[str]],
        timeout: float = DEFAULT_CLI_TIMEOUT,
    ) -> int:
        """Wait until any pattern matches in the buffer. Returns index of matched pattern."""
        deadline = time.time() + timeout
        compiled = [p if isinstance(p, re.Pattern) else re.compile(re.escape(p)) for p in patterns]

        while time.time() < deadline:
            # Check for matches
            for idx, pat in enumerate(compiled):
                m = pat.search(self.clean_buffer)
                if m:
                    # Advance buffer past match
                    self.clean_buffer = self.clean_buffer[m.end() :]
                    return idx

            if self.proc.poll() is not None:
                # Process terminated, read any remaining output
                while self.read_available(timeout=0.1):
                    pass
                for idx, pat in enumerate(compiled):
                    if pat.search(self.clean_buffer):
                        return idx
                raise EOFError(
                    f"Process terminated with code {self.proc.returncode} before pattern match. "
                    f"Tail: {self.clean_buffer[-500:]!r}"
                )

            self.read_available(timeout=0.2)

        raise TimeoutError(
            f"Timed out after {timeout}s waiting for {[p.pattern for p in compiled]}. "
            f"Buffer tail: {self.clean_buffer[-500:]!r}"
        )

    def send(self, text: str) -> None:
        """Write raw bytes to PTY master."""
        os.write(self.master_fd, text.encode("utf-8"))

    def sendline(self, text: str = "") -> None:
        """Write line to PTY master."""
        self.send(text + "\n")

    def drain_to_completion(self, timeout: float = 30.0) -> int:
        """Wait for process termination and drain all remaining terminal output."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            self.read_available(timeout=0.2)
            if self.proc.poll() is not None:
                while self.read_available(timeout=0.2):
                    pass
                return self.proc.returncode
        return -1

    def close(self) -> None:
        """Safely close PTY and clean up process."""
        if self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=2.0)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        if self.master_fd >= 0:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = -1

    def get_full_transcript(self) -> str:
        return "".join(self.raw_transcript)


class MarketAcceptanceRunner:
    """Orchestrates the market acceptance review run."""

    def __init__(
        self,
        runbook_path: Path = RUNBOOK_PATH,
        start_bin: Path = DEFAULT_START_BIN,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
    ):
        self.runbook_path = runbook_path
        self.start_bin = start_bin
        self.output_dir = output_dir
        self.parser = RunbookParser(self.runbook_path)
        self.manifest = self.parser.build_manifest()

        # Step lookup by heading
        self.step_map: dict[str, ManifestStep] = {s.heading: s for s in self.manifest.steps}

        # Tracking state
        self.checkpoints_reached: list[str] = []
        self.actions_executed: list[dict[str, Any]] = []
        self.grounding_censuses: list[dict[str, Any]] = []
        self.artifacts_seen: list[str] = []
        self.diagnostics_seen: list[str] = []
        self.var_assertions: dict[str, Any] = {}
        self.scenario_assertions: dict[str, Any] = {}
        self.committee_assertions: dict[str, Any] = {}
        self.governance_assertions: dict[str, Any] = {}
        self.failure_checkpoint: str = ""
        self.failure_reason: str = ""

    def _assert_var_section(self, text: str) -> None:
        """Assert terminal text invariants for VaR backtesting."""
        assert "LR=1.886" in text or "1.886" in text, "Expected LR=1.886 in VaR table"
        assert "LR=1.850" not in text, "Found invalid legacy LR=1.850 fallback in VaR table"
        assert "0.1696" in text, "Expected p=0.1696 in VaR table"
        assert "gamma=0.05" in text, "Expected gamma=0.05 in VaR table"
        assert "gamma=0.99" not in text, "Found invalid gamma=0.99 in VaR table"
        assert "gamma=0.01" not in text, "Found invalid gamma=0.01 in VaR table"

        # Check validation.var_size_power row
        assert "0.066" in text, "Expected size 0.066 in var_size_power"
        assert "0.031" in text and "0.069" in text, "Expected [0.031, 0.069] in var_size_power"
        assert "1.000" in text and "0.992" in text, "Expected power values in var_size_power"
        assert "PASS" in text, "Expected PASS status in var_size_power"
        self.var_assertions = {
            "lr_uc_verified": True,
            "gamma_0_05_verified": True,
            "no_gamma_0_99": True,
            "no_gamma_0_01": True,
            "size_power_verified": True,
        }

    def _parse_grounding_census(self, text: str) -> None:
        """Parse grounding censuses from transcript for diagnostics/legacy checks."""
        clean_text = ANSI_ESCAPE.sub("", text)
        for m_struct in re.finditer(
            r"Structured Grounding Gate:\s*(PASSED|FAILED)\s*—\s*Findings:\s*(\d+)\s*\|\s*Evidence refs:\s*(\d+)\s*\|\s*Validated refs:\s*(\d+)\s*\|\s*Invalid:\s*(\d+)",
            clean_text,
        ):
            status_str = m_struct.group(1)
            findings = int(m_struct.group(2))
            ev_refs = int(m_struct.group(3))
            validated = int(m_struct.group(4))
            invalid = int(m_struct.group(5))
            assert status_str == "PASSED", f"Structured grounding gate failed: Invalid={invalid}"
            assert validated == ev_refs, (
                f"Structured invariant failed: Validated {validated} != Evidence refs {ev_refs}"
            )
            assert invalid == 0, f"Structured invariant failed: Invalid {invalid} != 0"
            census = {
                "grounding_mode": "STRUCTURED",
                "finding_count": findings,
                "evidence_ref_count": ev_refs,
                "validated_ref_count": validated,
                "invalid_ref_count": invalid,
                "total": ev_refs,
                "grounded": validated,
                "unbound": invalid,
            }
            if census not in self.grounding_censuses:
                self.grounding_censuses.append(census)

        for m in re.finditer(
            r"Quantitative claims:\s*(\d+)\s*\|\s*Grounded:\s*(\d+)\s*\|\s*Unbound:\s*(\d+)",
            clean_text,
        ):
            total = int(m.group(1))
            grounded = int(m.group(2))
            unbound = int(m.group(3))
            assert grounded + unbound == total, (
                f"Grounding census invariant failed: {grounded} + {unbound} != {total}"
            )
            assert grounded <= total, f"Grounding invariant failed: {grounded} > {total}"
            census = {
                "grounding_mode": "LEGACY_FREEFORM",
                "total": total,
                "grounded": grounded,
                "unbound": unbound,
            }
            if census not in self.grounding_censuses:
                self.grounding_censuses.append(census)

    def evaluate_structured_run_state(
        self,
        summary_data: dict[str, Any],
        exit_code: int,
    ) -> tuple[str, str, list[dict[str, Any]]]:
        """Authoritative evaluation of machine-readable run state for Market domain acceptance."""
        if exit_code != 0:
            return "FAIL", f"start review exited with non-zero code {exit_code}", []

        # 1. Review path checkpoint completion
        required_checkpoints = [
            "Portfolio Risk & Volatility Assumptions",
            "Factor Modeling & Attribution Assumptions",
            "VaR Backtesting & Exception Frequency",
            "Covariance Structure & Missing Data Treatment",
            "Scenario Analysis & Stress Testing",
            "Cross-Analytical Committee Synthesis",
            "Model Governance & Attestation Sign-Off",
        ]
        for chk in required_checkpoints:
            if chk not in self.checkpoints_reached:
                return "FAIL", f"Missing required checkpoint: {chk}", []

        # 2. Canonical Market mode check
        grounding_mode = summary_data.get("grounding_mode")
        if grounding_mode != "STRUCTURED":
            return (
                "FAIL",
                f"Invalid grounding mode for Market domain: {grounding_mode} (must be STRUCTURED)",
                [],
            )

        # 3. Final attestation seal
        seal = summary_data.get("attestation_seal", {})
        merkle_root = seal.get("merkle_root")
        if not merkle_root or not isinstance(merkle_root, str) or len(merkle_root) < 16:
            return "FAIL", "Missing or invalid final attestation seal Merkle root", []
        if seal.get("metadata", {}).get("live_reviewer_status") == "LIVE_REVIEWER_NOT_VALIDATED":
            return "FAIL", "Attestation seal indicates LIVE_REVIEWER_NOT_VALIDATED", []

        # 4. Structured decisions evaluation
        decisions = summary_data.get("decisions", [])
        if not decisions:
            return "FAIL", "No decisions found in machine-readable run summary", []

        censuses: list[dict[str, Any]] = []
        q_c_count = 0

        for d in decisions:
            action = str(d.get("action", "")).strip().lower()
            if action in ("question", "challenge"):
                q_c_count += 1
                chk_title = d.get("checkpoint", "Unknown")

                # Grounding mode must be STRUCTURED
                d_mode = d.get("grounding_mode")
                if d_mode != "STRUCTURED":
                    return (
                        "FAIL",
                        f"Legacy grounding or invalid mode in decision on '{chk_title}': {d_mode}",
                        [],
                    )

                # Provider must be real OpenAI GPT-5
                provider = str(d.get("provider", "")).lower()
                model = str(d.get("model", "")).lower()
                if provider != "openai" or "gpt-5" not in model:
                    return (
                        "FAIL",
                        f"Invalid provider/model on '{chk_title}': provider={provider}, model={model}",
                        [],
                    )

                # Backend must be llm_structured (no fallback)
                backend = d.get("backend")
                if backend != "llm_structured":
                    return (
                        "FAIL",
                        f"Provider fallback or non-structured backend on '{chk_title}': {backend}",
                        [],
                    )

                # Provider status and schema validation status
                if d.get("provider_status") != "OK":
                    return (
                        "FAIL",
                        f"Provider request failed on '{chk_title}': {d.get('provider_status')}",
                        [],
                    )
                if d.get("schema_validation_status") != "VALID":
                    return (
                        "FAIL",
                        f"Schema validation failed on '{chk_title}': {d.get('schema_validation_status')}",
                        [],
                    )

                # Grounding counts: 0 invalid refs, validated == evidence_ref_count
                invalid_refs = d.get("invalid_ref_count", 0)
                if invalid_refs != 0:
                    return (
                        "FAIL",
                        f"Invalid evidence refs on '{chk_title}': {invalid_refs} invalid refs",
                        [],
                    )
                ev_refs = d.get("evidence_ref_count", 0)
                val_refs = d.get("validated_ref_count", 0)
                if val_refs != ev_refs:
                    return (
                        "FAIL",
                        f"Unvalidated evidence refs on '{chk_title}': {val_refs} != {ev_refs}",
                        [],
                    )

                # Structured findings content hash must exist and be 64-hex
                c_hash = d.get("structured_findings_content_hash")
                if not c_hash or len(c_hash) != 64:
                    return (
                        "FAIL",
                        f"Missing or invalid structured findings content hash on '{chk_title}': {c_hash}",
                        [],
                    )

                census = {
                    "checkpoint": chk_title,
                    "action": action,
                    "grounding_mode": "STRUCTURED",
                    "finding_count": d.get("finding_count", 0),
                    "evidence_ref_count": ev_refs,
                    "validated_ref_count": val_refs,
                    "invalid_ref_count": invalid_refs,
                    "structured_findings_content_hash": c_hash,
                    "provider": d.get("provider"),
                    "model": d.get("model"),
                    "backend": backend,
                    "total": ev_refs,
                    "grounded": val_refs,
                    "unbound": invalid_refs,
                }
                censuses.append(census)

        if q_c_count < 4:
            return "FAIL", f"Expected at least 4 Q/C actions; found {q_c_count}", censuses

        return "PASS", "", censuses

    def run(self) -> dict[str, Any]:
        """Execute the automated acceptance run."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_output_dir = self.output_dir / timestamp
        run_output_dir.mkdir(parents=True, exist_ok=True)

        # Track existing RUN-REVIEW directories before launch
        pre_run_dirs = {p.name for p in (WORKSPACE_ROOT / "start_output").glob("RUN-REVIEW-*")}

        # Write runbook sha and manifest
        (run_output_dir / "runbook_sha256.txt").write_text(self.manifest.runbook_sha256 + "\n")
        manifest_json = [dataclasses.asdict(s) for s in self.manifest.steps]
        (run_output_dir / "runbook_manifest.json").write_text(json.dumps(manifest_json, indent=2))

        print_manifest_summary(self.manifest)

        cmd = [str(self.start_bin), "review"]
        env = dict(os.environ)
        env["COLUMNS"] = "220"
        env["LINES"] = "60"
        session = PtySession(cmd=cmd, cwd=str(WORKSPACE_ROOT), env=env)

        choice_pat = re.compile(r"Select option \[default:\s*(\d+)\]:\s*")
        checkpoint_action_pat = re.compile(
            r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / "
            r"\[Q\]uestion / \[V\]iew / \[VA\] All Artifacts\]:\s*"
        )
        ask_q_pat = re.compile(r"Ask agent committee:\s*")
        ask_c_pat = re.compile(r"Enter reviewer challenge note:\s*")
        fallback_menu_pat = re.compile(r"Select action \[default:\s*1\]:\s*")

        status = "FAIL"
        full_transcript = ""

        try:
            print("\n>>> Launching interactive StART review...")

            # --- SETUP WIZARD ---
            # 1. Review Mode
            session.expect([choice_pat], timeout=DEFAULT_CLI_TIMEOUT)
            session.sendline(self.step_map["Review Mode"].content)

            # 2. Review Domain
            session.expect([choice_pat], timeout=DEFAULT_CLI_TIMEOUT)
            session.sendline(self.step_map["Review Domain"].content)

            # 3. AI Reviewer Agent Backend
            session.expect([choice_pat], timeout=DEFAULT_CLI_TIMEOUT)
            session.sendline(self.step_map["Backend"].content)

            # 4. Public LLM Provider
            session.expect([choice_pat], timeout=DEFAULT_CLI_TIMEOUT)
            session.sendline(self.step_map["Provider"].content)

            # 5. OpenAI Model
            session.expect([choice_pat], timeout=DEFAULT_CLI_TIMEOUT)
            session.sendline(self.step_map["Model"].content)

            # 6. Model Materiality
            session.expect([choice_pat], timeout=DEFAULT_CLI_TIMEOUT)
            session.sendline(self.step_map["Materiality"].content)

            # 7. Review Lifecycle
            session.expect([choice_pat], timeout=DEFAULT_CLI_TIMEOUT)
            session.sendline(self.step_map["Lifecycle"].content)

            # --- GOVERNANCE INFORMATION ---
            gov_prompts = [
                ("Business Context", "Business Context"),
                ("Reviewer Clarification", "Reviewer Clarification"),
                ("Intended Use / Decision Impact", "Intended Use"),
                ("Known Limitations / Reviewer Concerns", "Known Limitations"),
            ]
            for label, key in gov_prompts:
                session.expect([f"Enter {label}"], timeout=DEFAULT_CLI_TIMEOUT)
                content = self.step_map[key].content
                # Send text line by line
                for line in content.split("\n"):
                    session.sendline(line)
                if not content.strip().endswith("END"):
                    session.sendline("END")

            # --- DATA & SCOPE ---
            session.expect([choice_pat], timeout=DEFAULT_CLI_TIMEOUT)
            session.sendline(self.step_map["Data Source"].content)

            session.expect([choice_pat], timeout=DEFAULT_CLI_TIMEOUT)
            session.sendline(self.step_map["Scope"].content)

            proceed_pat = re.compile(r"Proceed to execute review\?\s*\[Y/n\]:\s*", re.IGNORECASE)
            session.expect([proceed_pat], timeout=DEFAULT_CLI_TIMEOUT)
            session.sendline(self.step_map["Proceed"].content)

            # --- CHECKPOINTS EXECUTION ---
            chk_pipeline = [
                {
                    "name": "Portfolio Risk & Volatility Assumptions",
                    "actions": [("A", "Portfolio Action", None)],
                },
                {
                    "name": "Factor Modeling & Attribution Assumptions",
                    "actions": [
                        ("V", "Factor View Artifacts", None),
                        ("Q", "Factor Ask Action", "Factor Question Text"),
                        ("A", "Factor Accept Action", None),
                    ],
                },
                {
                    "name": "VaR Backtesting & Exception Frequency",
                    "actions": [
                        ("V", "VaR View Artifacts", None),
                        ("Q", "VaR Ask Action", "VaR Question Text"),
                        ("C", "VaR Challenge Action", "VaR Challenge Text"),
                        ("A", "VaR Accept Action", None),
                    ],
                },
                {
                    "name": "Covariance Structure & Missing Data Treatment",
                    "actions": [
                        ("V", "Covariance View Artifacts", None),
                        ("Q", "Covariance Ask Action", "Covariance Question Text"),
                        ("C", "Covariance Challenge Action", "Covariance Challenge Text"),
                        ("A", "Covariance Accept Action", None),
                    ],
                },
                {
                    "name": "Scenario Analysis & Stress Testing",
                    "actions": [
                        ("V", "Scenario View Artifacts", None),
                        ("Q", "Scenario Ask Action", "Scenario Question Text"),
                        ("C", "Scenario Challenge Action", "Scenario Challenge Text"),
                        ("A", "Scenario Accept Action", None),
                    ],
                },
                {
                    "name": "Cross-Analytical Committee Synthesis",
                    "actions": [
                        ("VA", "Committee View Artifacts", None),
                        ("Q", "Committee Ask Action", "Committee Question Text"),
                        ("C", "Committee Challenge Action", "Committee Challenge Text"),
                        ("A", "Committee Accept Action", None),
                    ],
                },
                {
                    "name": "Model Governance & Attestation Sign-Off",
                    "actions": [
                        ("VA", "Governance View Artifacts", None),
                        ("Q", "Governance Ask Action", "Governance Question Text"),
                        ("C", "Governance Challenge Action", "Governance Challenge Text"),
                        ("A", "Governance Final Accept", None),
                    ],
                },
            ]

            prev_was_live = False

            for chk in chk_pipeline:
                chk_name: str = str(chk["name"])
                self.failure_checkpoint = chk_name
                chk_actions = cast(list[tuple[str, str, str | None]], chk["actions"])

                for act_type, act_step_key, text_step_key in chk_actions:
                    wait_timeout = DEFAULT_PROVIDER_TIMEOUT if prev_was_live else DEFAULT_CLI_TIMEOUT
                    matched_idx = session.expect(
                        [checkpoint_action_pat, fallback_menu_pat],
                        timeout=wait_timeout,
                    )

                    if prev_was_live:
                        prev_was_live = False

                    if matched_idx == 1:
                        session.sendline("2")
                        raise RuntimeError("Live reviewer fallback menu appeared. Sent 2 (Abort review).")

                    # Record checkpoint reached on first action
                    if chk_name not in self.checkpoints_reached:
                        self.checkpoints_reached.append(chk_name)
                        print(f"\n[CHECKPOINT REACHED]: {chk_name}")

                    # Checkpoint specific assertions on entry
                    cur_transcript = session.get_full_transcript()
                    if chk_name == "VaR Backtesting & Exception Frequency" and act_type == "V":
                        self._assert_var_section(cur_transcript)

                    if chk_name == "Scenario Analysis & Stress Testing" and act_type == "V":
                        has_scenario = any(
                            s in cur_transcript
                            for s in (
                                "scenario.linear_return",
                                "scenario.factor_linear",
                                "scenario.reverse_stress",
                                "Asset Tail Stress Shock",
                                "Macro Factor Shift",
                                "Mahalanobis Reverse Stress",
                            )
                        )
                        assert has_scenario, "Scenario checkpoint did not render scenario evidence"
                        self.scenario_assertions = {"scenario_evidence_present": True}

                    # Execute action
                    action_code = self.step_map[str(act_step_key)].content.strip().upper()
                    print(f"  Action -> {action_code}")
                    session.sendline(action_code)

                    if action_code in ("Q", "C"):
                        prompt_pat = ask_c_pat if action_code == "C" else ask_q_pat
                        session.expect([prompt_pat], timeout=DEFAULT_CLI_TIMEOUT)

                        assert text_step_key is not None
                        text_to_send = self.step_map[str(text_step_key)].content
                        print(f"  Sending {action_code} text ({len(text_to_send)} chars)...")
                        session.sendline(text_to_send)

                        prev_was_live = True
                        self.actions_executed.append(
                            {
                                "checkpoint": chk_name,
                                "action": action_code,
                                "text_sha": self.step_map[str(text_step_key)].content_sha256,
                            }
                        )
                    else:
                        prev_was_live = False
                        self.actions_executed.append(
                            {
                                "checkpoint": chk_name,
                                "action": action_code,
                            }
                        )

            # Checkpoint loop complete; drain remaining output to completion
            print("\n>>> Final Accept sent, draining to completion...")
            exit_code = session.drain_to_completion(timeout=DEFAULT_CLI_TIMEOUT)
            print(f">>> Process exited with code {exit_code}")

            full_transcript = session.get_full_transcript()
            (run_output_dir / "terminal_transcript.txt").write_text(full_transcript)

            # Locate newly created RUN-REVIEW output directory
            post_run_dirs = [
                p
                for p in (WORKSPACE_ROOT / "start_output").glob("RUN-REVIEW-*")
                if p.name not in pre_run_dirs
            ]
            active_run_dir: Path | None = None
            if post_run_dirs:
                active_run_dir = sorted(post_run_dirs, key=lambda p: p.stat().st_mtime)[-1]
            else:
                all_dirs = sorted(
                    (WORKSPACE_ROOT / "start_output").glob("RUN-REVIEW-*"),
                    key=lambda p: p.stat().st_mtime,
                )
                active_run_dir = all_dirs[-1] if all_dirs else None

            if active_run_dir is None or not (active_run_dir / "review_summary.json").exists():
                status = "FAIL"
                self.failure_reason = f"review_summary.json not found in {active_run_dir}"
            else:
                summary_data = json.loads((active_run_dir / "review_summary.json").read_text())
                status, self.failure_reason, self.grounding_censuses = self.evaluate_structured_run_state(
                    summary_data, exit_code
                )
                if status == "PASS":
                    self.failure_checkpoint = ""

        except Exception as exc:
            status = "FAIL"
            self.failure_reason = str(exc)
            if session.proc is not None:
                full_transcript = session.get_full_transcript()
                (run_output_dir / "terminal_transcript.txt").write_text(full_transcript)
            (run_output_dir / "failure_excerpt.txt").write_text(
                f"FAILURE AT CHECKPOINT: {self.failure_checkpoint}\n"
                f"REASON: {self.failure_reason}\n\n"
                f"TAIL TRANSCRIPT:\n{full_transcript[-2000:]}\n"
            )
            print(f"\n[ACCEPTANCE RUN FAILED]: {exc}")
        finally:
            session.close()

        result_data = {
            "status": status,
            "runbook_path": str(self.runbook_path),
            "runbook_sha256": self.manifest.runbook_sha256,
            "provider": "openai",
            "model": "gpt-5",
            "checkpoints_reached": self.checkpoints_reached,
            "actions_executed": self.actions_executed,
            "grounding_censuses": self.grounding_censuses,
            "artifacts_seen": self.artifacts_seen,
            "diagnostics_seen": self.diagnostics_seen,
            "var_assertions": self.var_assertions,
            "scenario_assertions": self.scenario_assertions,
            "committee_assertions": self.committee_assertions,
            "governance_assertions": self.governance_assertions,
            "failure_checkpoint": self.failure_checkpoint,
            "failure_reason": self.failure_reason,
            "timestamp": timestamp,
            "output_dir": str(run_output_dir),
        }

        (run_output_dir / "acceptance_result.json").write_text(json.dumps(result_data, indent=2))
        return result_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Run StART live interactive acceptance from runbook.")
    parser.add_argument("--runbook", type=Path, default=RUNBOOK_PATH, help="Path to runbook Markdown file.")
    parser.add_argument("--start-bin", type=Path, default=DEFAULT_START_BIN, help="Path to start binary.")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Path to output directory."
    )
    args = parser.parse_args()

    runner = MarketAcceptanceRunner(
        runbook_path=args.runbook,
        start_bin=args.start_bin,
        output_dir=args.output_dir,
    )
    result = runner.run()
    if result["status"] == "PASS":
        print("\n================================================================================")
        print("MARKET MANUAL ACCEPTANCE: PASS")
        print("================================================================================")
        return 0
    else:
        print("\n================================================================================")
        print("MARKET MANUAL ACCEPTANCE: FAIL")
        print(f"Reason: {result['failure_reason']}")
        print("================================================================================")
        return 1


if __name__ == "__main__":
    sys.exit(main())
