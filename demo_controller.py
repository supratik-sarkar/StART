#!/usr/bin/env python3
"""
StART v6.0.2 Demo Controller (Private / Non-Git)

Deterministic PTY/state-machine architecture to automate predetermined human input.
Strictly adheres to:
- Scripting the HUMAN operator, NEVER simulating or faking model responses.
- Fail-closed on prompt mismatch, missing prompts, or unexpected process exit.
- Bounded timeouts per prompt.
- Clean process cleanup with no orphan child processes.
- Never prints prompt bodies or private credentials.
- Zero public Git tree leakage.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import pty
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from start.review.structured_contract import (
    DEFAULT_CHECKPOINT_POLICIES,
    CheckpointDegradationPolicy,
)

REPOSITORY_ROOT = Path(__file__).resolve().parent
CHEATSHEET_PATH = REPOSITORY_ROOT / "StART_2.0_DEMO_CHEATSHEET_PRIVATE.md"

# Exact verified CLI commands supported by current StART implementation
SESSION_COMMANDS = {
    "SESSION_A_TEMPORAL": "start review",
    "SESSION_B_MARKET": "start review",
    "SESSION_C_POLICY": "start workflow quantitative_finance --trace engineering",
}

# ANSI escape sequence regex for terminal output cleaning
ANSI_ESCAPE = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")

_PRESENTATION_MODE = False
_CONTROLLER_AUDIT_TEXT = Path("/tmp/demo_controller.log")
_CONTROLLER_AUDIT_JSONL = Path("/tmp/demo_controller.jsonl")
_PRESENTATION_SIGNAL_PREFIX = b"\x1b]777;StART="
_PRESENTATION_SIGNAL_TERMINATOR = b"\x07"


def log(msg: str) -> None:
    """Log to stderr to avoid polluting the PTY passthrough on stdout."""
    if not _PRESENTATION_MODE:
        print(msg, file=sys.stderr)
    try:
        with _CONTROLLER_AUDIT_TEXT.open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
        with _CONTROLLER_AUDIT_JSONL.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"timestamp": time.time(), "component": "demo_controller", "message": msg}
                )
                + "\n"
            )
    except Exception:
        pass


def configure_presentation_mode(enabled: bool) -> None:
    """Keep controller diagnostics private while leaving child PTY output intact."""
    global _PRESENTATION_MODE
    _PRESENTATION_MODE = bool(enabled)


def strip_ansi(text: str) -> str:
    """Remove ANSI color codes from terminal output."""
    return ANSI_ESCAPE.sub("", text)


@dataclass(frozen=True)
class StateTransition:
    state_id: str
    expected_prompt_regex: str
    prompt_source: str
    input_type: str
    intent_label: str
    next_state: str
    allowed_fallback_action: str | None = None
    failure_policy: str = "FAIL_CLOSED"
    checkpoint_id: str = ""


@dataclass(frozen=True)
class TypingPolicy:
    """Viewer-facing input cadence; the transmitted text is never rewritten."""

    action_key_pre_delay: float = 0.4
    ordinary_chars_per_second: float = 50.0
    long_chars_per_second: float = 75.0
    long_text_threshold: int = 120
    post_message_dwell: float = 0.65

    def chars_per_second(self, text: str) -> float:
        return (
            self.long_chars_per_second
            if len(text) >= self.long_text_threshold
            else self.ordinary_chars_per_second
        )


class ArtifactBoardPresenter:
    """Open only the exact signaled run in the 38% landscape evidence pane."""

    _MACOS_BROWSER_CANDIDATES = (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
    )

    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self._profile: tempfile.TemporaryDirectory[str] | None = None
        self.board_path: Path | None = None

    @classmethod
    def _browser_executable(cls) -> str:
        for candidate in cls._MACOS_BROWSER_CANDIDATES:
            if Path(candidate).is_file():
                return candidate
        for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
            executable = shutil.which(name)
            if executable:
                return executable
        raise RuntimeError("presentation mode requires Chrome, Chromium, or Microsoft Edge")

    def show(self, *, manifest_path: str | Path, run_id: str) -> Path:
        from demo_artifact_board import landscape_geometry, render_board

        manifest = Path(manifest_path).expanduser().resolve(strict=True)
        if manifest.name != "artifact_manifest.json":
            raise ValueError("artifact-board signal must name artifact_manifest.json")
        board = render_board(run_path=manifest.parent, run_id=run_id)
        geometry = landscape_geometry()["artifact_board"]
        self.close()
        self._profile = tempfile.TemporaryDirectory(prefix="start-artifact-board-")
        command = [
            self._browser_executable(),
            f"--app={board.as_uri()}",
            f"--user-data-dir={self._profile.name}",
            f"--window-position={geometry.x},{geometry.y}",
            f"--window-size={geometry.width},{geometry.height}",
            "--no-first-run",
            "--disable-default-apps",
        ]
        self.process = subprocess.Popen(  # noqa: S603 - fixed executable and validated arguments
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.board_path = board
        return board

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2.0)
        self.process = None
        if self._profile is not None:
            self._profile.cleanup()
            self._profile = None


class SessionStatus(StrEnum):
    COMPLETE = "COMPLETE"
    FAIL_CLOSED = "FAIL_CLOSED"
    CRASHED = "CRASHED"
    CONTROLLER_ERROR = "CONTROLLER_ERROR"
    NOT_REACHED = "NOT_REACHED"


@dataclass
class SessionResult:
    session_id: str
    controller_status: SessionStatus = SessionStatus.NOT_REACHED
    child_exit_code: int | None = None
    review_run_id: str | None = None
    execution_run_ids: list[str] | None = None
    last_successful_semantic_checkpoint: str | None = None
    failure_class: str | None = None
    failure_message: str | None = None
    evidence_record_count: int | None = None
    artifact_manifest_path: str | None = None
    opa_decision: str | None = None
    attestation_merkle_state: str | None = None
    final_hold_reached: bool = False
    result_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["controller_status"] = self.controller_status.value
        return payload


class DemoProgress:
    """Controller-owned, result-derived multi-session presentation state."""

    def __init__(self, sessions: list[str]) -> None:
        self.sessions = tuple(sessions)
        self.states = {session: "NOT_STARTED" for session in self.sessions}
        self.opa = "NOT_REACHED"
        self.attestation = "NOT_REACHED"

    def start(self, session: str) -> None:
        if session not in self.states:
            raise KeyError(session)
        self.states[session] = "ACTIVE"

    def finish(self, result: SessionResult) -> None:
        self.states[result.session_id] = result.controller_status.value
        if result.opa_decision:
            decision = result.opa_decision.split(":")[-1].upper()
            self.opa = decision if decision in {"ALLOW", "DENY"} else "NOT_REACHED"
        if result.attestation_merkle_state:
            value = result.attestation_merkle_state.strip().upper()
            self.attestation = (
                "WITHHELD"
                if value in {"NOT_SEALED", "NONE", "WITHHELD", "UNAVAILABLE", "NOT_REACHED"}
                else "SEALED"
            )

    def render(self) -> Any:
        from rich.columns import Columns
        from rich.panel import Panel
        from rich.text import Text

        cells: list[Text] = []
        labels = {
            "SESSION_A_TEMPORAL": "FLIGHT A",
            "SESSION_B_MARKET": "FLIGHT B",
            "SESSION_C_POLICY": "FLIGHT C",
        }
        for session in self.sessions:
            status = self.states[session]
            marker, style = {
                "COMPLETE": ("✓", "green"),
                "ACTIVE": ("◉", "bold cyan"),
                "NOT_STARTED": ("○", "dim"),
                "FAIL_CLOSED": ("✗", "red"),
                "CRASHED": ("✗", "red"),
                "CONTROLLER_ERROR": ("✗", "red"),
            }.get(status, ("○", "dim"))
            cells.append(Text(f"{labels.get(session, session)} {marker}", style=style))
        if len(self.sessions) > 1:
            cells.append(
                Text(
                    f"OPA {'✓' if self.opa == 'ALLOW' else '○'}",
                    style="green" if self.opa == "ALLOW" else "dim",
                )
            )
            cells.append(
                Text(
                    "ATTESTATION "
                    + ("✓" if self.attestation == "SEALED" else "✗" if self.attestation == "WITHHELD" else "○"),
                    style=(
                        "green"
                        if self.attestation == "SEALED"
                        else "red" if self.attestation == "WITHHELD" else "dim"
                    ),
                )
            )
        return Panel(
            Columns(cells, expand=True),
            title="DEMO PROGRESS — CONTROLLER VERIFIED",
            border_style="bright_black",
            width=132,
        )


# Static prompt/state mapping proving valid state transitions for live public-LLM demo
STATE_TABLE: dict[str, list[StateTransition]] = {
    "SESSION_A_TEMPORAL": [
        StateTransition("A_MODE", r"Select Review Mode:[\s\S]*?Select option.*:\s*$", "wizard.py:53", "Menu Option [1]", "SETUP_MODE", "A_DOMAIN"),
        StateTransition("A_DOMAIN", r"Select Review Domain:[\s\S]*?Select option.*:\s*$", "wizard.py:90", "Menu Option [1]", "SETUP_DOMAIN", "A_TECH"),
        StateTransition("A_TECH", r"Select Predictive Modeling Technology:[\s\S]*?Select option.*:\s*$", "wizard.py:113", "Menu Option [2]", "SETUP_TECH", "A_BACKEND"),
        StateTransition("A_BACKEND", r"Select AI Reviewer Agent Backend.*:\s*$", "wizard.py", "Menu Option [4]", "SETUP_OFFLINE_TWIN", "A_MATERIALITY"),
        StateTransition("A_MATERIALITY", r"Select Model Materiality:[\s\S]*?Select option.*:\s*$", "wizard.py:363", "Menu Option [1]", "SETUP_MATERIALITY", "A_LIFECYCLE"),
        StateTransition("A_LIFECYCLE", r"Select Review Lifecycle:[\s\S]*?Select option.*:\s*$", "wizard.py:375", "Menu Option [1]", "SETUP_LIFECYCLE", "A_GOV_BIZ"),
        StateTransition("A_GOV_BIZ", r"Enter Business Context[\s\S]*?(?:only:\s*END|to skip\.)", "multiline_input.py:95", "Multiline Text + END", "A_OPENING", "A_GOV_CLAR"),
        StateTransition("A_GOV_CLAR", r"Enter Reviewer Clarification[\s\S]*?(?:only:\s*END|to skip\.)", "multiline_input.py:95", "Sentinel [END]", "SETUP_GOV_SKIP", "A_GOV_USE"),
        StateTransition("A_GOV_USE", r"Enter Intended Use / Decision Impact[\s\S]*?(?:only:\s*END|to skip\.)", "multiline_input.py:95", "Sentinel [END]", "SETUP_GOV_SKIP", "A_GOV_LIM"),
        StateTransition("A_GOV_LIM", r"Enter Known Limitations / Reviewer Concerns[\s\S]*?(?:only:\s*END|to skip\.)", "multiline_input.py:95", "Sentinel [END]", "SETUP_GOV_SKIP", "A_ARCH_SELECT"),
        StateTransition("A_ARCH_SELECT", r"Select Neural Network Architecture:[\s\S]*?Select option.*:\s*$", "wizard.py:484", "Menu Option [3]", "SETUP_ARCH_LSTM", "A_ACT_SELECT"),
        StateTransition("A_ACT_SELECT", r"Select Activation Function:[\s\S]*?Select option.*:\s*$", "wizard.py:510", "Menu Option [1]", "SETUP_ACT_RELU", "A_DATA_SELECT"),
        StateTransition("A_DATA_SELECT", r"Select Predictive Dataset Source:[\s\S]*?Select option.*:\s*$", "wizard.py:520", "Menu Option [5]", "SETUP_DATA_TEMPORAL", "A_SCOPE_SELECT"),
        StateTransition("A_SCOPE_SELECT", r"Select Review Scope:[\s\S]*?Select option.*:\s*$", "wizard.py:659", "Menu Option [1]", "SETUP_SCOPE_FULL", "A_PROCEED_WIZ"),
        StateTransition("A_PROCEED_WIZ", r"Proceed to execute review\? \[Y/n\]:\s*$", "wizard.py:722", "Confirmation [Y]", "SETUP_PROCEED", "A_ARCH_CHECKPOINT_1"),
        # Interactive Checkpoint Intents
        StateTransition("A_ARCH_CHECKPOINT_1", r"(?:\[architecture\]\s*)?Accept \(A\) / Override \(O\) / Challenge \(C\).*?\?\s*$", "checkpoints.py:204", "Action [C]", "A_ARCH_CHALLENGE", "A_ARCH_CHAL_NOTE"),
        StateTransition("A_ARCH_CHAL_NOTE", r"Enter challenge to ArchitectureReviewAgent:\s*$", "checkpoints.py:251", "Prompt Text", "A_ARCH_CHALLENGE", "A_CONCEDE_1"),
        StateTransition("A_CONCEDE_1", r"Does this response change the disposition\?.*$", "checkpoints.py:276", "Concession [N]", "A_CONCESSION_REFUSED", "A_ARCH_CHECKPOINT_2"),
        StateTransition("A_ARCH_CHECKPOINT_2", r"(?:\[architecture\]\s*)?Accept \(A\) / Override \(O\) / Challenge \(C\).*?\?\s*$", "checkpoints.py:204", "Action [C]", "A_DATA_CHALLENGE", "A_DATA_CHAL_NOTE"),
        StateTransition("A_DATA_CHAL_NOTE", r"Enter challenge to ArchitectureReviewAgent:\s*$", "checkpoints.py:251", "Prompt Text", "A_DATA_CHALLENGE", "A_CONCEDE_2"),
        StateTransition("A_CONCEDE_2", r"Does this response change the disposition\?.*$", "checkpoints.py:276", "Concession [N]", "A_CONCESSION_REFUSED", "A_ARCH_CHECKPOINT_3"),
        StateTransition("A_ARCH_CHECKPOINT_3", r"(?:\[architecture\]\s*)?Accept \(A\) / Override \(O\) / Challenge \(C\).*?\?\s*$", "checkpoints.py:204", "Action [Q]", "A_VALIDATION_CHALLENGE", "A_VAL_QUEST_NOTE"),
        StateTransition("A_VAL_QUEST_NOTE", r"Ask ArchitectureReviewAgent:\s*$", "checkpoints.py:232", "Prompt Text", "A_VALIDATION_CHALLENGE", "A_ARCH_DECISION"),
        StateTransition("A_ARCH_DECISION", r"(?:\[architecture\]\s*)?Accept \(A\) / Override \(O\) / Challenge \(C\).*?\?\s*$", "checkpoints.py:204", "Action [A]", "A_DECISION", "A_EXECUTE"),
        StateTransition("A_EXECUTE", r"(?:Review complete|AI review committee complete|seal: start-seal)[\s\S]*?$", "orchestrator.py:1007", "Deterministic Execution", "A_EXECUTE", "A_EVIDENCE_SEPARATION"),
        StateTransition("A_EVIDENCE_SEPARATION", r"(?:evidence records:|sign-off:|seal:|dashboard:)[\s\S]*?$", "interactive_review.py:1370", "Evidence Verification", "A_EVIDENCE_SEPARATION", "A_COMPLETE"),
    ],
    "SESSION_B_MARKET": [
        StateTransition("B_MODE", r"Select Review Mode:[\s\S]*?Select option.*:\s*$", "wizard.py:53", "Menu Option [1]", "SETUP_MODE", "B_DOMAIN"),
        StateTransition("B_DOMAIN", r"Select Review Domain:[\s\S]*?Select option.*:\s*$", "wizard.py:90", "Menu Option [2]", "SETUP_DOMAIN", "B_BACKEND"),
        StateTransition("B_BACKEND", r"Select AI Reviewer Agent Backend.*:\s*$", "wizard.py", "Menu Option [4]", "SETUP_OFFLINE_TWIN", "B_MATERIALITY"),
        StateTransition("B_MATERIALITY", r"Select Model Materiality:[\s\S]*?Select option.*:\s*$", "wizard.py:363", "Menu Option [1]", "SETUP_MATERIALITY", "B_LIFECYCLE"),
        StateTransition("B_LIFECYCLE", r"Select Review Lifecycle:[\s\S]*?Select option.*:\s*$", "wizard.py:375", "Menu Option [1]", "SETUP_LIFECYCLE", "B_GOV_BIZ"),
        StateTransition("B_GOV_BIZ", r"Enter Business Context[\s\S]*?(?:only:\s*END|to skip\.)", "multiline_input.py:95", "Multiline Text + END", "B_OPENING", "B_GOV_CLAR"),
        StateTransition("B_GOV_CLAR", r"Enter Reviewer Clarification[\s\S]*?(?:only:\s*END|to skip\.)", "multiline_input.py:95", "Sentinel [END]", "SETUP_GOV_SKIP", "B_GOV_USE"),
        StateTransition("B_GOV_USE", r"Enter Intended Use / Decision Impact[\s\S]*?(?:only:\s*END|to skip\.)", "multiline_input.py:95", "Sentinel [END]", "SETUP_GOV_SKIP", "B_GOV_LIM"),
        StateTransition("B_GOV_LIM", r"Enter Known Limitations / Reviewer Concerns[\s\S]*?(?:only:\s*END|to skip\.)", "multiline_input.py:95", "Sentinel [END]", "SETUP_GOV_SKIP", "B_DATA_SELECT"),
        StateTransition("B_DATA_SELECT", r"Select Market(?:/| & )Treasury Data Source:[\s\S]*?Select option.*:\s*$", "wizard.py:570", "Menu Option [1]", "SETUP_DATA_MARKET", "B_SCOPE_SELECT"),
        StateTransition("B_SCOPE_SELECT", r"Select Review Scope:[\s\S]*?Select option.*:\s*$", "wizard.py:659", "Menu Option [1]", "SETUP_SCOPE_FULL", "B_PROCEED_WIZ"),
        StateTransition("B_PROCEED_WIZ", r"Proceed to execute review\? \[Y/n\]:\s*$", "wizard.py:722", "Confirmation [Y]", "B_EXECUTE", "B_CHK_PORTFOLIO_1"),
        # Flight B Checkpoints (Portfolio)
        StateTransition("B_CHK_PORTFOLIO_1", r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$", "executor.py:1603", "Action [C]", "B_OBJECTIVE_CHALLENGE", "B_OBJ_CHAL_NOTE"),
        StateTransition("B_OBJ_CHAL_NOTE", r"Enter reviewer challenge note:\s*$", "executor.py:1690", "Prompt Text", "B_OBJECTIVE_CHALLENGE", "B_CHK_PORTFOLIO_2"),
        StateTransition("B_CHK_PORTFOLIO_2", r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$", "executor.py:1603", "Action [Q]", "B_ALGORITHM_DISCUSSION", "B_ALGO_QUEST_NOTE"),
        StateTransition("B_ALGO_QUEST_NOTE", r"Ask agent committee:\s*$", "executor.py:1690", "Prompt Text", "B_ALGORITHM_DISCUSSION", "B_CHK_PORTFOLIO_3"),
        StateTransition(
            "B_CHK_PORTFOLIO_3",
            r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$",
            "executor.py:1603",
            "Action [O]",
            "B_GOVERNANCE_OVERRIDE",
            "B_OVERRIDE_NOTE",
            allowed_fallback_action="1",
            failure_policy="REQUIRE_VALID_AGENT_RESPONSE",
            checkpoint_id="market.portfolio_allocation",
        ),
        StateTransition("B_OVERRIDE_NOTE", r"Enter reviewer override justification:\s*$", "executor.py:1632", "Prompt Text", "B_GOVERNANCE_OVERRIDE", "B_CHK_ATTRIBUTION_1"),
        StateTransition("B_CHK_ATTRIBUTION_1", r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$", "executor.py:1603", "Action [Q]", "B_EVIDENCE_QUESTION", "B_ATTRIB_QUEST_NOTE"),
        StateTransition("B_ATTRIB_QUEST_NOTE", r"Ask agent committee:\s*$", "executor.py:1690", "Prompt Text", "B_EVIDENCE_QUESTION", "B_CHK_ATTRIBUTION_2"),
        StateTransition(
            "B_CHK_ATTRIBUTION_2",
            r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$",
            "executor.py:1603",
            "Action [A]",
            "INTERVENING_ACCEPT",
            "C_CHK_VAR_1",
            allowed_fallback_action="1",
            failure_policy="REQUIRE_VALID_AGENT_RESPONSE",
            checkpoint_id="market.factor_attribution",
        ),
        # Flight C Checkpoints (Risk / Scenario / Governance)
        StateTransition("C_CHK_VAR_1", r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$", "executor.py:1603", "Action [Q]", "C_RISK_FORMULATION", "C_VAR_QUEST_NOTE"),
        StateTransition("C_VAR_QUEST_NOTE", r"Ask agent committee:\s*$", "executor.py:1690", "Prompt Text", "C_RISK_FORMULATION", "C_CHK_VAR_2"),
        StateTransition(
            "C_CHK_VAR_2",
            r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$",
            "executor.py:1603",
            "Action [A]",
            "INTERVENING_ACCEPT",
            "C_CHK_COV_1",
            allowed_fallback_action="1",
            failure_policy="REQUIRE_VALID_AGENT_RESPONSE",
            checkpoint_id="market.var_tail_risk",
        ),
        StateTransition("C_CHK_COV_1", r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$", "executor.py:1603", "Action [Q]", "C_TAIL_RISK_INQUIRY", "C_COV_QUEST_NOTE"),
        StateTransition("C_COV_QUEST_NOTE", r"Ask agent committee:\s*$", "executor.py:1690", "Prompt Text", "C_TAIL_RISK_INQUIRY", "C_CHK_COV_2"),
        StateTransition(
            "C_CHK_COV_2",
            r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$",
            "executor.py:1603",
            "Action [A]",
            "INTERVENING_ACCEPT",
            "C_CHK_SCENARIO_1",
            allowed_fallback_action="1",
            failure_policy="REQUIRE_VALID_AGENT_RESPONSE",
            checkpoint_id="market.covariance_structure",
        ),
        StateTransition("C_CHK_SCENARIO_1", r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$", "executor.py:1603", "Action [C]", "C_SCENARIO_CHALLENGE", "C_SCEN_CHAL_NOTE"),
        StateTransition("C_SCEN_CHAL_NOTE", r"Enter reviewer challenge note:\s*$", "executor.py:1690", "Prompt Text", "C_SCENARIO_CHALLENGE", "C_CHK_SCENARIO_2"),
        StateTransition(
            "C_CHK_SCENARIO_2",
            r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$",
            "executor.py:1603",
            "Action [A]",
            "C_SCENARIO_DECISION",
            "C_CHK_GOVERNANCE_1",
            allowed_fallback_action=None,
            failure_policy="REQUIRE_VALID_AGENT_RESPONSE",
            checkpoint_id="market.scenario_stress",
        ),
        StateTransition("C_CHK_GOVERNANCE_1", r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$", "executor.py:1603", "Action [Q]", "C_FINAL_EVIDENCE_SUMMARY", "C_GOV_QUEST_NOTE"),
        StateTransition("C_GOV_QUEST_NOTE", r"Ask agent committee:\s*$", "executor.py:1690", "Prompt Text", "C_FINAL_EVIDENCE_SUMMARY", "C_CHK_GOVERNANCE_2"),
        StateTransition(
            "C_CHK_GOVERNANCE_2",
            r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$",
            "executor.py:1603",
            "Action [A]",
            "C_GOV_ACCEPT",
            "B_CHK_SIGNOFF",
            allowed_fallback_action="1",
            failure_policy="ALLOW_EVIDENCE_ONLY_DEGRADATION",
            checkpoint_id="market.cross_analytical_synthesis",
        ),
        StateTransition(
            "B_CHK_SIGNOFF",
            r"Action \[\[A\]ccept \(default\) / \[O\]verride / \[C\]hallenge / \[Q\]uestion.*\]:\s*$",
            "executor.py:1603",
            "Action [A]",
            "B_SIGNOFF_ACCEPT",
            "B_EXECUTE_FINALIZE",
            allowed_fallback_action=None,
            failure_policy="REQUIRE_VALID_AGENT_RESPONSE",
            checkpoint_id="market.governance_signoff",
        ),
        StateTransition("B_EXECUTE_FINALIZE", r"(?:Attestation Seal|Merklized Ledger Replay|Review complete)[\s\S]*?$", "executor.py:2750", "Deterministic Execution", "B_FINALIZE", "B_COMPLETE"),
    ],
    "SESSION_C_POLICY": [
        StateTransition("C_EXECUTE", r"Executing Canonical Workflow 'quantitative_finance'.*$", "main.py:898", "Autonomous Workflow Invocation", "C_WORKFLOW_EXECUTE", "C_COMPLETE"),
    ],
}


# Stable semantic route contract for the private rehearsal harness. Cosmetic
# Rich formatting is intentionally excluded from this pre-runtime check.
SESSION_ROUTE_CONTRACTS: dict[str, tuple[str, ...]] = {
    "SESSION_A_TEMPORAL": (
        "A_MODE", "A_DOMAIN", "A_TECH", "A_BACKEND", "A_MATERIALITY", "A_LIFECYCLE",
        "A_GOV_BIZ", "A_GOV_CLAR", "A_GOV_USE", "A_GOV_LIM", "A_ARCH_SELECT", "A_ACT_SELECT",
        "A_DATA_SELECT", "A_SCOPE_SELECT", "A_PROCEED_WIZ", "A_ARCH_CHECKPOINT_1",
        "A_ARCH_CHAL_NOTE", "A_CONCEDE_1", "A_ARCH_CHECKPOINT_2", "A_DATA_CHAL_NOTE",
        "A_CONCEDE_2", "A_ARCH_CHECKPOINT_3", "A_VAL_QUEST_NOTE", "A_ARCH_DECISION",
        "A_EXECUTE", "A_EVIDENCE_SEPARATION",
    ),
    "SESSION_B_MARKET": (
        "B_MODE", "B_DOMAIN", "B_BACKEND", "B_MATERIALITY", "B_LIFECYCLE", "B_GOV_BIZ",
        "B_GOV_CLAR", "B_GOV_USE", "B_GOV_LIM", "B_DATA_SELECT", "B_SCOPE_SELECT",
        "B_PROCEED_WIZ", "B_CHK_PORTFOLIO_1", "B_OBJ_CHAL_NOTE", "B_CHK_PORTFOLIO_2",
        "B_ALGO_QUEST_NOTE", "B_CHK_PORTFOLIO_3", "B_OVERRIDE_NOTE", "B_CHK_ATTRIBUTION_1",
        "B_ATTRIB_QUEST_NOTE", "B_CHK_ATTRIBUTION_2", "C_CHK_VAR_1", "C_VAR_QUEST_NOTE",
        "C_CHK_VAR_2", "C_CHK_COV_1", "C_COV_QUEST_NOTE", "C_CHK_COV_2",
        "C_CHK_SCENARIO_1", "C_SCEN_CHAL_NOTE", "C_CHK_SCENARIO_2",
        "C_CHK_GOVERNANCE_1", "C_GOV_QUEST_NOTE", "C_CHK_GOVERNANCE_2",
        "B_CHK_SIGNOFF", "B_EXECUTE_FINALIZE",
    ),
    "SESSION_C_POLICY": ("C_EXECUTE",),
}


def validate_route_contracts() -> tuple[bool, str]:
    """Validate semantic state order and edges before any child is launched."""
    if set(SESSION_ROUTE_CONTRACTS) != set(STATE_TABLE) or set(STATE_TABLE) != set(SESSION_COMMANDS):
        return False, "predefined session keys differ across commands, states, and route contracts"
    for session, expected_states in SESSION_ROUTE_CONTRACTS.items():
        transitions = STATE_TABLE.get(session, [])
        actual_states = tuple(transition.state_id for transition in transitions)
        if actual_states != expected_states:
            return False, f"{session}: expected route {expected_states}, got {actual_states}"
        if len(actual_states) != len(set(actual_states)):
            return False, f"{session}: duplicate semantic state IDs"
        for current, following in zip(transitions, transitions[1:], strict=False):
            if current.next_state != following.state_id:
                return (
                    False,
                    f"{session}: edge {current.state_id}->{current.next_state} does not reach {following.state_id}",
                )
    temporal_states = SESSION_ROUTE_CONTRACTS["SESSION_A_TEMPORAL"]
    if "A_METRIC" in temporal_states:
        return False, "SESSION_A_TEMPORAL: obsolete A_METRIC is not part of the canonical route"
    temporal = {transition.state_id: transition for transition in STATE_TABLE["SESSION_A_TEMPORAL"]}
    if temporal["A_ARCH_DECISION"].next_state != "A_EXECUTE":
        return False, "SESSION_A_TEMPORAL: architecture decision must enter deterministic execution"
    return True, "semantic route contracts are consistent"


def parse_cheatsheet(path: str | Path = CHEATSHEET_PATH) -> dict[str, list[str]]:
    """Parse the private markdown cheatsheet to extract session prompts."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cheatsheet not found at {path}")

    with open(path, encoding="utf-8") as f:
        content = f.read()

    sessions: dict[str, list[str]] = {}
    current_session = None

    for line in content.split("\n"):
        if line.startswith("# Flight A"):
            current_session = "SESSION_A_TEMPORAL"
            sessions[current_session] = []
        elif line.startswith("# Flight B"):
            current_session = "SESSION_B_MARKET"
            sessions[current_session] = []
        elif line.startswith("# Flight C"):
            current_session = "SESSION_C_POLICY"
            sessions[current_session] = []
        elif line.startswith("> ") and current_session:
            prompt = line[2:].strip()
            sessions[current_session].append(prompt)

    return sessions


class PTYController:
    """
    Deterministic PTY controller.
    Spawns child processes in a pseudo-terminal and waits for expected prompt patterns.
    """

    def __init__(
        self,
        timeout: float = 120.0,
        *,
        presentation_mode: bool = False,
        typing_policy: TypingPolicy | None = None,
    ):
        self.timeout = timeout
        self.presentation_mode = presentation_mode
        self.typing_policy = typing_policy or TypingPolicy()
        self.pid: int | None = None
        self.fd: int | None = None
        self.buffer = ""
        self.child_exited = False
        self.child_exit_code: int | None = None
        self.transcript = ""
        self._viewer_stream: Any = getattr(sys.stdout, "buffer", sys.stdout)
        self._viewer_hold_until = 0.0
        self._pending_viewer_output = bytearray()
        self._signal_tail = b""
        self._viewer_suppressed = False
        self.artifact_presenter = ArtifactBoardPresenter() if presentation_mode else None

    def _write_viewer(self, data: bytes) -> None:
        if not data:
            return
        self._viewer_stream.write(data)
        self._viewer_stream.flush()

    def _flush_viewer_if_due(self, *, force: bool = False) -> None:
        if not self._pending_viewer_output:
            return
        if force or time.monotonic() >= self._viewer_hold_until:
            self._write_viewer(bytes(self._pending_viewer_output))
            self._pending_viewer_output.clear()

    def release_viewer_hold(self) -> None:
        """Reveal buffered output immediately when a genuine prompt needs an answer."""
        self._viewer_hold_until = 0.0
        self._flush_viewer_if_due(force=True)

    def finish_viewer_hold(self) -> None:
        """Honor final viewer dwell after runtime execution has already ended."""
        remaining = self._viewer_hold_until - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self._flush_viewer_if_due(force=True)

    def _handle_presentation_signal(self, encoded: bytes) -> None:
        if not self.presentation_mode:
            return
        try:
            padding = b"=" * (-len(encoded) % 4)
            payload = json.loads(base64.urlsafe_b64decode(encoded + padding))
        except (ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("invalid presentation-controller signal") from exc
        action = payload.get("action")
        if action == "dwell":
            seconds = max(0.0, min(float(payload.get("seconds", 0.0)), 15.0))
            self._viewer_hold_until = max(
                self._viewer_hold_until,
                time.monotonic() + seconds,
            )
        elif action == "artifact_board":
            if self.artifact_presenter is None:
                raise RuntimeError("artifact board received outside presentation mode")
            self.artifact_presenter.show(
                manifest_path=str(payload.get("manifest_path", "")),
                run_id=str(payload.get("run_id", "")),
            )
        elif action == "viewer_suppression":
            self._viewer_suppressed = bool(payload.get("enabled", False))
        else:
            raise RuntimeError(f"unsupported presentation-controller action: {action!r}")

    @staticmethod
    def _partial_prefix_length(data: bytes) -> int:
        maximum = min(len(data), len(_PRESENTATION_SIGNAL_PREFIX) - 1)
        for size in range(maximum, 0, -1):
            if data[-size:] == _PRESENTATION_SIGNAL_PREFIX[:size]:
                return size
        return 0

    def _consume_visible(self, data: bytes) -> None:
        if not data:
            return
        text = data.decode("utf-8", errors="replace")
        self.buffer += text
        self.transcript += text
        if self.presentation_mode and self._viewer_suppressed:
            return
        if self.presentation_mode and time.monotonic() < self._viewer_hold_until:
            self._pending_viewer_output.extend(data)
        else:
            self._flush_viewer_if_due(force=True)
            self._write_viewer(data)

    def consume_data(self, data: bytes) -> None:
        """Record child output immediately while applying only viewer-side holds."""
        pending = self._signal_tail + data
        self._signal_tail = b""
        while pending:
            start = pending.find(_PRESENTATION_SIGNAL_PREFIX)
            if start < 0:
                partial = self._partial_prefix_length(pending)
                visible = pending[:-partial] if partial else pending
                self._consume_visible(visible)
                if partial:
                    self._signal_tail = pending[-partial:]
                return
            self._consume_visible(pending[:start])
            signal_start = start + len(_PRESENTATION_SIGNAL_PREFIX)
            end = pending.find(_PRESENTATION_SIGNAL_TERMINATOR, signal_start)
            if end < 0:
                self._signal_tail = pending[start:]
                return
            self._handle_presentation_signal(pending[signal_start:end])
            pending = pending[end + len(_PRESENTATION_SIGNAL_TERMINATOR) :]

    def _record_wait_status(self, status: int) -> None:
        self.child_exited = True
        if os.WIFEXITED(status):
            self.child_exit_code = os.WEXITSTATUS(status)
        elif os.WIFSIGNALED(status):
            self.child_exit_code = 128 + os.WTERMSIG(status)
        else:
            self.child_exit_code = 1

    def poll_exit(self) -> bool:
        if self.child_exited or not self.pid:
            return self.child_exited
        try:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
        except ChildProcessError:
            self.child_exited = True
            return True
        if pid == self.pid:
            self._record_wait_status(status)
        return self.child_exited

    def reap_exit(self) -> None:
        if self.child_exited or not self.pid:
            return
        try:
            _, status = os.waitpid(self.pid, 0)
            self._record_wait_status(status)
        except ChildProcessError:
            self.child_exited = True

    def spawn(self, cmd: str) -> None:
        if self.presentation_mode:
            self._write_viewer(b"\x1b[2J\x1b[H")
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            env = os.environ.copy()
            venv_bin = os.path.dirname(sys.executable)
            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"
            env.setdefault("OMP_NUM_THREADS", "1")
            env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
            env.setdefault("OPENBLAS_NUM_THREADS", "1")
            env.setdefault("MKL_NUM_THREADS", "1")
            env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
            env.setdefault("NUMEXPR_NUM_THREADS", "1")
            env.setdefault("FORCE_COLOR", "1")
            env.setdefault("START_FORCE_COLOR", "1")
            env.setdefault("TERM", "xterm-256color")
            if self.presentation_mode:
                env["START_PRESENTATION_MODE"] = "1"
            os.execvpe("/bin/bash", ["bash", "-c", cmd], env)
        else:
            self.child_exited = False

    def handle_adjudication_council(self, clean: str) -> bool:
        """Handle optional Human Adjudication Council prompt with strict fail-closed validation per Sections 1 & 2."""
        is_arch_contradiction = "[ARCHITECTURE_CONTRADICTION]" in clean.upper()
        has_lstm = "lstm" in clean.lower()
        has_tabular = "tabular" in clean.lower()
        has_expected_agents = (
            "ArchitectureReviewAgent" in clean and "ValidationPlannerAgent" in clean
        )
        if is_arch_contradiction and has_lstm and has_tabular and has_expected_agents:
            log("[CONTROLLER] CRITICAL: CANONICAL GROUNDING REGRESSION REAPPEARED IN LIVE PATH")
            log("[CONTROLLER] The exact known [ARCHITECTURE_CONTRADICTION] reached Human Adjudication Council.")
            log("[CONTROLLER] Grounding fix failed to dominate agent claims. Failing closed immediately.")
            return False
        else:
            log("[CONTROLLER] CRITICAL: Unexpected Human Adjudication Council conflict detected without authorization:")
            log(clean)
            log("[CONTROLLER] Failing closed per deterministic policy. No unscripted council decisions authorized.")
            return False

    def read_until_pattern(
        self,
        pattern: str,
        allowed_fallback_action: str | None = None,
        failure_policy: str = "FAIL_CLOSED",
        state_id: str = "",
        checkpoint_id: str = "",
    ) -> bool:
        """Read until the regex pattern matches the terminal buffer or timeout expires."""
        regex = re.compile(pattern, re.MULTILINE)
        start_time = time.time()

        while True:
            self._flush_viewer_if_due()
            if time.time() - start_time > self.timeout:
                raise TimeoutError(f"Timeout waiting for pattern: {pattern}")

            if self.fd is None:
                return False

            r, _, _ = select.select([self.fd], [], [], 0.05)
            if r:
                try:
                    data = os.read(self.fd, 1024)
                except OSError:
                    self.reap_exit()
                    return False

                if not data:
                    self.reap_exit()
                    return False

                self.consume_data(data)

                clean = strip_ansi(self.buffer)
                if regex.search(clean):
                    self.buffer = ""
                    return True
                if "Enter decision [1-5]:" in clean:
                    ok_adj = self.handle_adjudication_council(clean)
                    if not ok_adj:
                        return False
                if "Select action [default: 1]:" in clean:
                    is_invalid_resp = (
                        "STRUCTURED_REVIEWER_RESPONSE_INVALID" in clean
                        or "LIVE_REVIEWER_NOT_VALIDATED" in clean
                    )
                    eff_chk_id = checkpoint_id
                    if not eff_chk_id and state_id:
                        for tr_list in STATE_TABLE.values():
                            for tr in tr_list:
                                if tr.state_id == state_id:
                                    eff_chk_id = getattr(tr, "checkpoint_id", "")
                                    break
                    effective_policy = DEFAULT_CHECKPOINT_POLICIES.get(
                        eff_chk_id, CheckpointDegradationPolicy.REQUIRE_VALID_AGENT_RESPONSE
                    )
                    is_safe_degradation = (
                        effective_policy == CheckpointDegradationPolicy.ALLOW_EVIDENCE_ONLY_DEGRADATION
                        and failure_policy in (
                            "ALLOW_EVIDENCE_ONLY_DEGRADATION",
                            "CONTINUE_WITHOUT_AGENT_INTERPRETATION",
                        )
                        and allowed_fallback_action == "1"
                        and state_id not in (
                            "B_CHK_SIGNOFF",
                            "B_OVERRIDE_NOTE",
                            "A_ARCH_DECISION",
                        )
                    )
                    if is_safe_degradation and is_invalid_resp:
                        log(
                            f"[CONTROLLER] Detected fallback prompt 'Select action [default: 1]:' with policy "
                            f"'{effective_policy.value}' on checkpoint '{checkpoint_id}' (state '{state_id}'). "
                            f"Sending authorized fallback '{allowed_fallback_action}'."
                        )
                        assert allowed_fallback_action is not None
                        self.release_viewer_hold()
                        self.send_line_presented(allowed_fallback_action)
                        self.buffer = ""
                    else:
                        log(
                            f"[CONTROLLER] CRITICAL: Fallback prompt 'Select action [default: 1]:' encountered without valid authorization "
                            f"(state_id={state_id}, checkpoint_id={checkpoint_id}, effective_policy={effective_policy}, "
                            f"failure_policy={failure_policy}, allowed_action={allowed_fallback_action}, invalid_resp={is_invalid_resp}). "
                            f"Failing closed."
                        )
                        return False
            else:
                if self.pid:
                    try:
                        pid, status = os.waitpid(self.pid, os.WNOHANG)
                        if pid == self.pid:
                            self._record_wait_status(status)
                            return False
                    except ChildProcessError:
                        self.child_exited = True
                        return False

    def send_line(self, line: str) -> None:
        if self.fd is not None:
            os.write(self.fd, (line + "\n").encode("utf-8"))

    def send_line_presented(self, line: str) -> None:
        """Type exact operator text in presentation mode; remain instant otherwise."""
        if not self.presentation_mode:
            self.send_line(line)
            return
        if self.fd is None:
            return
        self.release_viewer_hold()
        if len(line) <= 3 and "\n" not in line:
            time.sleep(self.typing_policy.action_key_pre_delay)
        delay = 1.0 / self.typing_policy.chars_per_second(line)
        for character in line:
            os.write(self.fd, character.encode("utf-8"))
            time.sleep(delay)
        os.write(self.fd, b"\n")
        time.sleep(self.typing_policy.post_message_dwell)

    def cleanup(self) -> None:
        """Ensure child process is thoroughly terminated with no orphans."""
        if self.pid and not self.child_exited:
            try:
                os.kill(self.pid, signal.SIGTERM)
                time.sleep(0.1)
                os.kill(self.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            try:
                _, status = os.waitpid(self.pid, 0)
                self._record_wait_status(status)
            except (ChildProcessError, OSError):
                pass
            self.pid = None
        if self.fd:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = None
        self.finish_viewer_hold()
        if self.artifact_presenter is not None:
            self.artifact_presenter.close()


def _persist_session_result(result: SessionResult, result_root: str | Path | None) -> Path:
    root = (
        Path(result_root)
        if result_root is not None
        else REPOSITORY_ROOT
        / "start_output"
        / "demo_results"
        / f"{result.session_id}-{uuid.uuid4().hex[:8]}"
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "session_result.json"
    result.result_path = str(target)
    fd, temp_name = tempfile.mkstemp(prefix=".session_result.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            import json

            json.dump(result.to_dict(), stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return target


def _hydrate_runtime_fields(result: SessionResult, transcript: str) -> None:
    clean = strip_ansi(transcript)
    run_ids = list(
        dict.fromkeys(
            re.findall(r"\bRUN-(?:(?:REVIEW|ENT|MROS)-)?[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*\b", clean)
        )
    )
    explicit_review_ids = re.findall(r"(?i)Review run:\s*(RUN-[A-Za-z0-9-]+)", clean)
    explicit_execution_ids = re.findall(r"(?i)Execution run:\s*(RUN-[A-Za-z0-9-]+)", clean)
    result.review_run_id = (
        explicit_review_ids[-1]
        if explicit_review_ids
        else next((run_id for run_id in run_ids if run_id.startswith("RUN-REVIEW-")), None)
    )
    if result.review_run_id is None and run_ids:
        result.review_run_id = run_ids[0]
    result.execution_run_ids = list(
        dict.fromkeys(
            explicit_execution_ids
            or [run_id for run_id in run_ids if run_id != result.review_run_id]
        )
    )
    evidence_matches = re.findall(
        r"(?i)(?:(\d+)\s+(?:EvidenceRecords?|evidence records?)|Records:\s*(\d+))", clean
    )
    if evidence_matches:
        left, right = evidence_matches[-1]
        result.evidence_record_count = int(left or right)
    manifest_matches = re.findall(r"(?:/[^\s]+)?artifact_manifest\.json", clean)
    if manifest_matches:
        candidate = Path(manifest_matches[-1]).expanduser().resolve()
        if candidate.is_file():
            result.artifact_manifest_path = str(candidate)
    policy_matches = re.findall(r"\b(OPA_LOCAL|NATIVE_ADAPTER)\b[^\n]*(ALLOW|DENY)", clean)
    if policy_matches:
        engine, decision = policy_matches[-1]
        result.opa_decision = f"{engine}:{decision}"
    else:
        engines = re.findall(r"(?i)Engine\s*[│|:]?\s*(OPA_LOCAL|NATIVE_ADAPTER)", clean)
        decisions = re.findall(r"(?i)Decision\s*[│|:]?\s*(ALLOW|DENY)", clean)
        if engines and decisions:
            result.opa_decision = f"{engines[-1].upper()}:{decisions[-1].upper()}"
    merkle_matches = re.findall(
        r"(?i)(?:merkle(?: root)?|seal)\s*[:=]\s*([A-Za-z0-9:_-]+)", clean
    )
    if merkle_matches:
        result.attestation_merkle_state = merkle_matches[-1]


def run_session_result(
    session_name: str,
    cmd: str,
    timeout: float = 180.0,
    *,
    result_root: str | Path | None = None,
    presentation_mode: bool = False,
) -> SessionResult:
    """Run a session and return its authoritative process-backed result."""
    if presentation_mode:
        configure_presentation_mode(True)
    log(f"=== Starting {session_name} ===")
    ctrl = PTYController(timeout=timeout, presentation_mode=presentation_mode)
    result = SessionResult(session_id=session_name, execution_run_ids=[])
    controller_failure: tuple[str, str] | None = None
    route_ok, route_detail = validate_route_contracts()
    if not route_ok:
        result.controller_status = SessionStatus.CONTROLLER_ERROR
        result.failure_class = "RouteContractError"
        result.failure_message = route_detail
        result_path = _persist_session_result(result, result_root)
        log(f"[{session_name}] Route contract invalid: {route_detail} -> {result_path}")
        return result

    def signal_handler(sig, frame):
        log("\nInterrupt caught, cleaning up...")
        ctrl.cleanup()
        sys.exit(1)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    cheatsheet = parse_cheatsheet()
    prompts_a = cheatsheet.get("SESSION_A_TEMPORAL", [])
    prompts_b = cheatsheet.get("SESSION_B_MARKET", [])
    prompts_c = cheatsheet.get("SESSION_C_POLICY", [])

    # Map state transitions to concrete inputs
    input_mapping: dict[str, str] = {}
    if session_name == "SESSION_A_TEMPORAL":
        input_mapping = {
            "A_MODE": "1",
            "A_DOMAIN": "1",
            "A_TECH": "2",
            "A_BACKEND": "4",  # Offline demo twin
            "A_MATERIALITY": "1",
            "A_LIFECYCLE": "1",
            "A_GOV_BIZ": (prompts_a[0] if len(prompts_a) > 0 else "Temporal sequence prediction") + "\nEND",
            "A_GOV_CLAR": "END",
            "A_GOV_USE": "END",
            "A_GOV_LIM": "END",
            "A_ARCH_SELECT": "3",  # LSTM
            "A_ACT_SELECT": "1",  # ReLU
            "A_DATA_SELECT": "5",  # Synthetic Temporal Sequence
            "A_SCOPE_SELECT": "1",  # Full Recommended
            "A_PROCEED_WIZ": "Y",
            "A_ARCH_CHECKPOINT_1": "C",
            "A_ARCH_CHAL_NOTE": prompts_a[1] if len(prompts_a) > 1 else "Challenge LSTM choice",
            "A_CONCEDE_1": "N",
            "A_ARCH_CHECKPOINT_2": "C",
            "A_DATA_CHAL_NOTE": prompts_a[2] if len(prompts_a) > 2 else "Verify temporal rank-3 input",
            "A_CONCEDE_2": "N",
            "A_ARCH_CHECKPOINT_3": "Q",
            "A_VAL_QUEST_NOTE": prompts_a[3] if len(prompts_a) > 3 else "Challenge validation leakage",
            "A_ARCH_DECISION": "A",
            "A_EXECUTE": "",
            "A_EVIDENCE_SEPARATION": "",
        }
    elif session_name == "SESSION_B_MARKET":
        input_mapping = {
            "B_MODE": "1",
            "B_DOMAIN": "2",  # Market Risk & Portfolio
            "B_BACKEND": "4",  # Offline demo twin
            "B_MATERIALITY": "1",
            "B_LIFECYCLE": "1",
            "B_GOV_BIZ": (prompts_b[0] if len(prompts_b) > 0 else "Portfolio optimization review") + "\nEND",
            "B_GOV_CLAR": "END",
            "B_GOV_USE": "END",
            "B_GOV_LIM": "END",
            "B_DATA_SELECT": "1",  # Synthetic 50-asset world
            "B_SCOPE_SELECT": "1",  # Full Recommended
            "B_PROCEED_WIZ": "Y",
            # Flight B (Portfolio)
            "B_CHK_PORTFOLIO_1": "C",
            "B_OBJ_CHAL_NOTE": prompts_b[1] if len(prompts_b) > 1 else "Challenge portfolio objective",
            "B_CHK_PORTFOLIO_2": "Q",
            "B_ALGO_QUEST_NOTE": prompts_b[2] if len(prompts_b) > 2 else "Compare portfolio algorithms",
            "B_CHK_PORTFOLIO_3": "O",
            "B_OVERRIDE_NOTE": prompts_b[3] if len(prompts_b) > 3 else "Override default allocation rationale",
            "B_CHK_ATTRIBUTION_1": "Q",
            "B_ATTRIB_QUEST_NOTE": prompts_b[5] if len(prompts_b) > 5 else "Separate portfolio weights from narrative",
            "B_CHK_ATTRIBUTION_2": "A",
            # Flight C (Risk / Scenario / Governance)
            "C_CHK_VAR_1": "Q",
            "C_VAR_QUEST_NOTE": prompts_c[0] if len(prompts_c) > 0 else "Investigate market risk scenario",
            "C_CHK_VAR_2": "A",
            "C_CHK_COV_1": "Q",
            "C_COV_QUEST_NOTE": prompts_c[2] if len(prompts_c) > 2 else "Explain tail risk behavior",
            "C_CHK_COV_2": "A",
            "C_CHK_SCENARIO_1": "C",
            "C_SCEN_CHAL_NOTE": prompts_c[1] if len(prompts_c) > 1 else "Challenge scenario design",
            "C_CHK_SCENARIO_2": "A",
            "C_CHK_GOVERNANCE_1": "Q",
            "C_GOV_QUEST_NOTE": prompts_c[4] if len(prompts_c) > 4 else "Provide final evidence summary",
            "C_CHK_GOVERNANCE_2": "A",
            "B_CHK_SIGNOFF": "A",
            "B_EXECUTE_FINALIZE": "",
        }
    elif session_name == "SESSION_C_POLICY":
        input_mapping = {
            "C_EXECUTE": "",
        }

    try:
        ctrl.spawn(cmd)
        transitions = STATE_TABLE.get(session_name, [])

        for tr in transitions:
            if tr.input_type in ("Deterministic Execution", "Evidence Verification", "Autonomous Workflow Invocation"):
                log(f"[{session_name}] State {tr.state_id}: awaiting execution completion...")
                start_w = time.time()
                while not ctrl.child_exited and (time.time() - start_w < timeout):
                    ctrl._flush_viewer_if_due()
                    if ctrl.fd is None:
                        break
                    r, _, _ = select.select([ctrl.fd], [], [], 0.5)
                    if r:
                        try:
                            d = os.read(ctrl.fd, 1024)
                            if not d:
                                ctrl.reap_exit()
                                break
                            ctrl.consume_data(d)
                            clean = strip_ansi(ctrl.buffer)
                            if "Enter decision [1-5]:" in clean:
                                ok_adj = ctrl.handle_adjudication_council(clean)
                                if not ok_adj:
                                    controller_failure = (
                                        "AdjudicationPolicyError",
                                        "unauthorized adjudication prompt",
                                    )
                                    break
                        except OSError:
                            ctrl.reap_exit()
                            break
                    else:
                        ctrl.poll_exit()
                if not ctrl.child_exited and controller_failure is None:
                    controller_failure = (
                        "ExecutionTimeout",
                        f"child did not terminate within {timeout} seconds",
                    )
                if controller_failure is None and not re.search(
                    tr.expected_prompt_regex,
                    strip_ansi(ctrl.transcript),
                    re.MULTILINE,
                ):
                    controller_failure = (
                        "PromptPatternMismatch",
                        f"execution boundary not observed at {tr.state_id}",
                    )
                if controller_failure is None:
                    result.last_successful_semantic_checkpoint = tr.state_id
                continue

            log(f"[{session_name}] State {tr.state_id} ({tr.intent_label}): awaiting prompt boundary...")
            matched = ctrl.read_until_pattern(
                tr.expected_prompt_regex,
                allowed_fallback_action=tr.allowed_fallback_action,
                failure_policy=tr.failure_policy,
                state_id=tr.state_id,
                checkpoint_id=getattr(tr, "checkpoint_id", ""),
            )
            if not matched:
                log(f"[{session_name}] ERROR: Prompt pattern '{tr.expected_prompt_regex}' not matched. Failing closed.")
                controller_failure = (
                    "PromptPatternMismatch",
                    f"expected prompt not reached at {tr.state_id}",
                )
                break

            send_val = input_mapping.get(tr.state_id, "")
            log(f"[{session_name}] State {tr.state_id}: injecting input [{tr.input_type}]")
            ctrl.release_viewer_hold()
            ctrl.send_line_presented(send_val)
            result.last_successful_semantic_checkpoint = tr.state_id

        if not ctrl.child_exited and controller_failure is None:
            start_wait = time.time()
            while not ctrl.poll_exit() and time.time() - start_wait < timeout:
                time.sleep(0.05)
            if not ctrl.child_exited:
                controller_failure = (
                    "ChildTerminationTimeout",
                    "state traversal ended before the child process terminated",
                )
    except TimeoutError as e:
        log(f"[{session_name}] TIMEOUT FAILURE: {e}. Failing closed.")
        controller_failure = ("PromptTimeout", str(e))
    except BaseException as exc:
        controller_failure = (type(exc).__name__, str(exc))
    finally:
        ctrl.cleanup()
        result.child_exit_code = ctrl.child_exit_code
        _hydrate_runtime_fields(result, ctrl.transcript)

        clean = strip_ansi(ctrl.transcript)
        traceback_seen = "Traceback (most recent call last):" in clean
        cancelled_seen = "ReviewCancelled" in clean or "Review cancelled" in clean
        if traceback_seen:
            result.controller_status = SessionStatus.CRASHED
            result.failure_class = "PythonTraceback"
            result.failure_message = "child emitted an uncaught Python traceback"
        elif cancelled_seen and result.child_exit_code != 0:
            result.controller_status = SessionStatus.FAIL_CLOSED
            result.failure_class = "ReviewCancelled"
            result.failure_message = "review terminated through the fail-closed cancellation path"
        elif controller_failure is not None:
            result.controller_status = SessionStatus.CONTROLLER_ERROR
            result.failure_class, result.failure_message = controller_failure
        elif result.child_exit_code == 0:
            result.controller_status = SessionStatus.COMPLETE
            result.final_hold_reached = True
            log(f"[{session_name}] Reached final hold after verified child exit 0.")
        else:
            result.controller_status = SessionStatus.CRASHED
            result.failure_class = "ChildProcessError"
            result.failure_message = f"child exited with status {result.child_exit_code}"
        result_path = _persist_session_result(result, result_root)
        log(f"[{session_name}] Result: {result.controller_status.value} -> {result_path}")
        log(f"=== Finished {session_name} ===\n")
    return result


def run_session(session_name: str, cmd: str, timeout: float = 180.0) -> bool:
    """Compatibility wrapper returning success only for verified COMPLETE."""
    return run_session_result(session_name, cmd, timeout).controller_status is SessionStatus.COMPLETE


def validate_only() -> bool:
    """Validate internal mappings, path safety, and state graph integrity."""
    log("=======================================================")
    log("     StART v6.0.2 Demo Controller — Validation Report     ")
    log("=======================================================")

    if not os.path.exists(CHEATSHEET_PATH):
        log(f"ERROR: Cheatsheet not found at {CHEATSHEET_PATH}")
        return False

    cheatsheet = parse_cheatsheet(CHEATSHEET_PATH)

    route_ok, route_detail = validate_route_contracts()
    log("\n[0] SEMANTIC ROUTE CONTRACTS:")
    if not route_ok:
        log(f"  ✗ {route_detail}")
        return False
    log(f"  ✓ {route_detail}; Session A has no A_METRIC state.")

    # 1. Offline rehearsal provider representation
    log("\n[1] OFFLINE DEMO-TWIN SELECTION:")
    for s_name in ("SESSION_A_TEMPORAL", "SESSION_B_MARKET"):
        trans = {t.state_id: t for t in STATE_TABLE.get(s_name, [])}
        has_backend = "A_BACKEND" in trans or "B_BACKEND" in trans
        has_hosted_selection = any(
            state in trans for state in ("A_PROVIDER", "A_MODEL", "B_PROVIDER", "B_MODEL")
        )
        if not has_backend or has_hosted_selection:
            log(f"  ✗ {s_name} does not isolate the offline rehearsal provider!")
            return False
        log(f"  ✓ {s_name}: offline_demo_twin; no hosted provider/model selection.")

    # 2. Flight A Human Interaction Intent Coverage
    log("\n[2] FLIGHT A HUMAN INTENT COVERAGE (7/7):")
    req_a = [
        "A_OPENING",
        "A_ARCH_CHALLENGE",
        "A_DATA_CHALLENGE",
        "A_VALIDATION_CHALLENGE",
        "A_DECISION",
        "A_EXECUTE",
        "A_EVIDENCE_SEPARATION",
    ]
    intents_a = {t.intent_label for t in STATE_TABLE["SESSION_A_TEMPORAL"]}
    for req in req_a:
        if req in intents_a:
            log(f"  ✓ Intent {req}: REPRESENTED")
        else:
            log(f"  ✗ Intent {req}: MISSING")
            return False
    log("  Result: 7/7 Flight-A human intents mapped to genuine checkpoints.")

    # 3. Flight B Human Interaction Intent Coverage
    log("\n[3] FLIGHT B HUMAN INTENT COVERAGE (6/6):")
    req_b = [
        "B_OPENING",
        "B_OBJECTIVE_CHALLENGE",
        "B_ALGORITHM_DISCUSSION",
        "B_GOVERNANCE_OVERRIDE",
        "B_EXECUTE",
        "B_EVIDENCE_QUESTION",
    ]
    intents_b = {t.intent_label for t in STATE_TABLE["SESSION_B_MARKET"]}
    for req in req_b:
        if req in intents_b:
            log(f"  ✓ Intent {req}: REPRESENTED")
        else:
            log(f"  ✗ Intent {req}: MISSING")
            return False
    log("  Result: 6/6 Flight-B human intents mapped to genuine checkpoints.")

    # 4. Flight C Human Interaction Intent Coverage
    log("\n[4] FLIGHT C HUMAN INTENT COVERAGE (5/5):")
    req_c = [
        "C_RISK_FORMULATION",
        "C_SCENARIO_CHALLENGE",
        "C_TAIL_RISK_INQUIRY",
        "C_SCENARIO_DECISION",
        "C_FINAL_EVIDENCE_SUMMARY",
    ]
    for req in req_c:
        if req in intents_b:
            log(f"  ✓ Intent {req}: REPRESENTED")
        else:
            log(f"  ✗ Intent {req}: MISSING")
            return False
    log("  Result: 5/5 Flight-C human intents mapped to genuine checkpoints.")

    # 5. Session C Policy Command
    log("\n[5] SESSION C OPA CONTROL PLANE COMMAND:")
    cmd_c = SESSION_COMMANDS.get("SESSION_C_POLICY", "")
    if cmd_c == "start workflow quantitative_finance --trace engineering":
        log(f"  ✓ SESSION_C_POLICY: '{cmd_c}' validated.")
    else:
        log(f"  ✗ Invalid Session C command: '{cmd_c}'")
        return False

    log("\nValidation successful. Controller mappings and offline-isolation invariants verified.\n")
    return True


def run_negative_self_tests() -> bool:
    """Comprehensive negative self-test suite proving all 11 required controller behaviors."""
    log("=== Running Comprehensive Negative Self-Test Suite ===")
    all_passed = True

    # 1. Expected prompt succeeds
    log("Test 1: Expected prompt succeeds...")
    ctrl1 = PTYController(timeout=3.0)
    ctrl1.spawn("echo -n 'StART> '; read ans; echo 'OK'")
    ok1 = ctrl1.read_until_pattern(r"StART>\s*$")
    ctrl1.cleanup()
    if ok1:
        log("  ✓ Test 1 PASS")
    else:
        log("  ✗ Test 1 FAIL")
        all_passed = False

    # 2. Wrong prompt fails closed
    log("Test 2: Wrong prompt fails closed...")
    ctrl2 = PTYController(timeout=1.0)
    ctrl2.spawn("echo 'UNEXPECTED_OUTPUT_HERE'")
    try:
        ok2 = ctrl2.read_until_pattern(r"EXPECTED_PROMPT_NEVER_ARRIVES")
        if not ok2:
            log("  ✓ Test 2 PASS (correctly failed closed on wrong prompt)")
        else:
            log("  ✗ Test 2 FAIL (unexpectedly matched wrong prompt)")
            all_passed = False
    except TimeoutError:
        log("  ✓ Test 2 PASS (correctly timed out on wrong prompt)")
    ctrl2.cleanup()

    # 3. Missing prompt times out
    log("Test 3: Missing prompt times out...")
    ctrl3 = PTYController(timeout=1.0)
    ctrl3.spawn("sleep 10")
    try:
        ctrl3.read_until_pattern(r"ANY_PROMPT")
        log("  ✗ Test 3 FAIL")
        all_passed = False
    except TimeoutError:
        log("  ✓ Test 3 PASS (correctly raised TimeoutError)")
    ctrl3.cleanup()

    # 4. Out-of-order state fails closed
    log("Test 4: Out-of-order state fails closed...")
    ctrl4 = PTYController(timeout=1.0)
    ctrl4.spawn("echo -n 'STATE_B_PROMPT> '")
    try:
        ok4 = ctrl4.read_until_pattern(r"STATE_A_PROMPT>\s*$")
        if not ok4:
            log("  ✓ Test 4 PASS (correctly refused out-of-order state)")
        else:
            log("  ✗ Test 4 FAIL (unexpectedly matched out-of-order state)")
            all_passed = False
    except TimeoutError:
        log("  ✓ Test 4 PASS (correctly timed out on out-of-order state)")
    ctrl4.cleanup()

    # 5. Child crash fails closed
    log("Test 5: Child crash fails closed...")
    ctrl5 = PTYController(timeout=3.0)
    ctrl5.spawn("exit 1")
    ok5 = ctrl5.read_until_pattern(r"ANY_PROMPT")
    ctrl5.cleanup()
    if not ok5:
        log("  ✓ Test 5 PASS (child exit returned False)")
    else:
        log("  ✗ Test 5 FAIL")
        all_passed = False

    # 6. EOF fails closed
    log("Test 6: EOF fails closed...")
    ctrl6 = PTYController(timeout=3.0)
    ctrl6.spawn("true")
    ok6 = ctrl6.read_until_pattern(r"ANY_PROMPT")
    ctrl6.cleanup()
    if not ok6:
        log("  ✓ Test 6 PASS (EOF cleanly returned False)")
    else:
        log("  ✗ Test 6 FAIL")
        all_passed = False

    # 7. No later input injected after failure
    log("Test 7: No later input injected after failure...")
    injected_calls = []

    def mock_inject(step: str):
        injected_calls.append(step)

    ctrl7 = PTYController(timeout=1.0)
    ctrl7.spawn("echo 'CRASH'; exit 1")
    try:
        matched = ctrl7.read_until_pattern(r"PROMPT")
        if not matched:
            raise RuntimeError("Failed closed")
        mock_inject("INPUT_2")
    except Exception:
        pass
    ctrl7.cleanup()
    if len(injected_calls) == 0:
        log("  ✓ Test 7 PASS (subsequent injection stopped upon failure)")
    else:
        log("  ✗ Test 7 FAIL")
        all_passed = False

    # 8. Clean process cleanup
    log("Test 8: Clean process cleanup...")
    ctrl8 = PTYController(timeout=5.0)
    ctrl8.spawn("sleep 30")
    pid8 = ctrl8.pid
    ctrl8.cleanup()
    if ctrl8.pid is None and ctrl8.fd is None:
        log("  ✓ Test 8 PASS (controller references cleanly cleared)")
    else:
        log("  ✗ Test 8 FAIL")
        all_passed = False

    # 9. No orphan child
    log("Test 9: No orphan child process...")
    assert pid8 is not None
    try:
        os.kill(pid8, 0)
        log("  ✗ Test 9 FAIL (orphan process still alive)")
        all_passed = False
    except (ProcessLookupError, OSError):
        log("  ✓ Test 9 PASS (child process confirmed dead)")

    # 10. Unexpected fallback prompt fails closed
    log("Test 10: Unexpected fallback prompt fails closed...")
    ctrl10 = PTYController(timeout=2.0)
    ctrl10.spawn("echo '  Select action [default: 1]: '; sleep 1")
    ok10 = ctrl10.read_until_pattern(r"SOME_EXPECTED_PROMPT", allowed_fallback_action=None)
    ctrl10.cleanup()
    if not ok10:
        log("  ✓ Test 10 PASS (correctly failed closed on unauthorized fallback prompt)")
    else:
        log("  ✗ Test 10 FAIL (unexpectedly accepted unauthorized fallback prompt)")
        all_passed = False

    # 11. Authorized fallback prompt succeeds with typed degradation policy
    log("Test 11: Authorized fallback prompt succeeds with typed degradation policy...")
    ctrl11 = PTYController(timeout=3.0)
    ctrl11.spawn(
        "echo 'STRUCTURED_REVIEWER_RESPONSE_INVALID: ungrounded metric'; "
        "echo -n '  Select action [default: 1]: '; read ans; "
        "if [ \"$ans\" = \"1\" ]; then echo -n 'SUBSEQUENT_PROMPT> '; read ans2; fi"
    )
    ok11 = ctrl11.read_until_pattern(
        r"SUBSEQUENT_PROMPT>\s*$",
        allowed_fallback_action="1",
        failure_policy="ALLOW_EVIDENCE_ONLY_DEGRADATION",
        state_id="C_CHK_GOVERNANCE_2",
        checkpoint_id="market.cross_analytical_synthesis",
    )
    ctrl11.cleanup()
    if ok11:
        log("  ✓ Test 11 PASS (successfully injected authorized fallback action under typed degradation policy)")
    else:
        log("  ✗ Test 11 FAIL (failed to handle authorized fallback prompt under typed degradation policy)")
        all_passed = False

    # 11b. Fallback prompt on non-degradable checkpoint fails closed
    log("Test 11b: Fallback prompt on non-degradable checkpoint fails closed...")
    ctrl11b = PTYController(timeout=2.0)
    ctrl11b.spawn(
        "echo 'STRUCTURED_REVIEWER_RESPONSE_INVALID: ungrounded metric'; "
        "echo -n '  Select action [default: 1]: '; sleep 1"
    )
    ok11b = ctrl11b.read_until_pattern(
        r"SUBSEQUENT_PROMPT>\s*$",
        allowed_fallback_action="1",
        failure_policy="REQUIRE_VALID_AGENT_RESPONSE",
        state_id="C_CHK_SCENARIO_2",
        checkpoint_id="market.scenario_stress",
    )
    ctrl11b.cleanup()
    if not ok11b:
        log("  ✓ Test 11b PASS (correctly failed closed on non-degradable checkpoint)")
    else:
        log("  ✗ Test 11b FAIL (unexpectedly permitted fallback on non-degradable checkpoint)")
        all_passed = False

    # 12. Known architecture contradiction at Adjudication Council fails closed per Section 1
    log("Test 12: Known architecture contradiction at Adjudication Council fails closed...")
    ctrl12 = PTYController(timeout=4.0)
    mock_council_cmd = (
        'echo "--- Collision 1/1: [ARCHITECTURE_CONTRADICTION] (Severity: HIGH) ---"; '
        'echo "  • Position A (ArchitectureReviewAgent): recommended sequence/vision family \'lstm\'."; '
        'echo "  • Position B (ValidationPlannerAgent): asserts dataset modality is tabular \'tabular\'."; '
        'echo -n "Enter decision [1-5]: "; read ans'
    )
    ctrl12.spawn(mock_council_cmd)
    ok12 = ctrl12.read_until_pattern(r"FINAL_REVIEW_PROMPT>\s*$")
    ctrl12.cleanup()
    if not ok12:
        log("  ✓ Test 12 PASS (correctly failed closed on grounding regression contradiction)")
    else:
        log("  ✗ Test 12 FAIL (unexpectedly permitted known architecture contradiction)")
        all_passed = False

    # 13. Unknown or unauthorized Adjudication Council conflict fails closed
    log("Test 13: Unknown or unauthorized Adjudication Council conflict fails closed...")
    ctrl13 = PTYController(timeout=3.0)
    mock_unknown_council = (
        'echo "--- Collision 1/1: [UNKNOWN_POLICY_COLLISION] (Severity: HIGH) ---"; '
        'echo "  • Position A (ArbitraryAgentA): proposal alpha."; '
        'echo "  • Position B (ArbitraryAgentB): proposal beta."; '
        'echo -n "Enter decision [1-5]: "; read ans'
    )
    ctrl13.spawn(mock_unknown_council)
    ok13 = ctrl13.read_until_pattern(r"FINAL_REVIEW_PROMPT>\s*$")
    ctrl13.cleanup()
    if not ok13:
        log("  ✓ Test 13 PASS (correctly failed closed on unknown adjudication conflict)")
    else:
        log("  ✗ Test 13 FAIL (unexpectedly accepted unknown adjudication conflict)")
        all_passed = False

    log("=== Negative Self-Test Suite Finished ===")
    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(description="StART v6.0.2 Demo Controller")
    parser.add_argument("--session", choices=SESSION_COMMANDS.keys(), help="Run a specific session")
    parser.add_argument("--validate-only", action="store_true", help="Validate mapping and paths only")
    parser.add_argument("--self-test", action="store_true", help="Run comprehensive negative self-test suite")
    parser.add_argument("--dry-run", action="store_true", help="Print session sequence without execution")
    parser.add_argument(
        "--presentation-mode",
        action="store_true",
        help="Suppress controller diagnostics from the viewer stream and enable display-only pacing",
    )
    parser.add_argument("--timeout", type=float, default=180.0, help="Timeout in seconds per prompt (default 180)")

    args = parser.parse_args()
    configure_presentation_mode(args.presentation_mode)

    if args.validate_only:
        ok = validate_only()
        sys.exit(0 if ok else 1)

    if args.self_test:
        ok = run_negative_self_tests()
        sys.exit(0 if ok else 1)

    if args.dry_run:
        sessions = parse_cheatsheet()
        for s_name, cmd in SESSION_COMMANDS.items():
            log(f"Session: {s_name}")
            log(f"Command: {cmd}")
            steps = len(sessions.get(s_name, []))
            transitions = len(STATE_TABLE.get(s_name, []))
            log(f"Cheatsheet Prompts: {steps} | State Transitions: {transitions}")
            log("---")
        sys.exit(0)

    target_sessions = [args.session] if args.session else list(SESSION_COMMANDS.keys())
    demo_progress = DemoProgress(target_sessions)
    progress_console = None
    if args.presentation_mode:
        from rich.console import Console

        progress_console = Console()

    for s_name in target_sessions:
        demo_progress.start(s_name)
        if progress_console is not None:
            progress_console.print(demo_progress.render())
        cmd = SESSION_COMMANDS[s_name]
        session_result = run_session_result(
            s_name,
            cmd,
            timeout=args.timeout,
            presentation_mode=args.presentation_mode,
        )
        demo_progress.finish(session_result)
        if progress_console is not None:
            progress_console.print(demo_progress.render())
        if session_result.controller_status is not SessionStatus.COMPLETE:
            log(
                f"Session {s_name}: {session_result.controller_status.value}. "
                "Aborting demo execution."
            )
            sys.exit(1)

    if args.session:
        log(f"Session {args.session}: COMPLETE.")
    else:
        log("ALL DEMO SESSIONS COMPLETED SUCCESSFULLY.")
    try:
        with open("/tmp/demo_controller.done", "w", encoding="utf-8") as f:
            f.write("DONE\n")
    except Exception:
        pass


if __name__ == "__main__":
    main()
