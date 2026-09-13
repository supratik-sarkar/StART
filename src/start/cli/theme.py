"""Rich-based terminal theme, stable 256-color palette, agent badges, and glyph semantics.

Provides a unified visual language across the StART CLI:
  - Stable 256-color agent palette indexed deterministically by sorted agent name.
  - Consistent agent badges ([DSC], [FTR], [ARC], etc.).
  - Status glyphs (✓, !, ✗, ⛔, ⇄, ·, ◆, ◇, ⚖).
  - TTY / NO_COLOR / START_FORCE_COLOR aware Console factory (standard 100 columns).
"""

from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.text import Text
from rich.theme import Theme

# --------------------------------------------------------------------------- #
# Standard 12 Agent Badges & Codes
# --------------------------------------------------------------------------- #
AGENT_BADGES: dict[str, str] = {
    "DatasetDiscoveryAgent": "DSC",
    "TaskInferenceAgent": "TSK",
    "FeatureEngineeringAgent": "FTR",
    "ArchitectureReviewAgent": "ARC",
    "HyperparameterTuningAgent": "HPO",
    "ModelExecutionAgent": "EXE",
    "ExplainabilityAgent": "XAI",
    "SensitivityAgent": "SNS",
    "OverfittingAgent": "OVF",
    "ValidationAgent": "VAL",
    "GovernanceAgent": "GOV",
    "EvidenceCriticAgent": "CRT",
}

# Short aliases mapped to badge codes
AGENT_SHORT_CODES: dict[str, str] = {
    "discovery": "DSC",
    "task": "TSK",
    "features": "FTR",
    "feature_engineering": "FTR",
    "architecture": "ARC",
    "tuning": "HPO",
    "execution": "EXE",
    "explainability": "XAI",
    "sensitivity": "SNS",
    "overfitting": "OVF",
    "validation": "VAL",
    "governance": "GOV",
    "critic": "CRT",
}

# --------------------------------------------------------------------------- #
# Deterministic Palette (256-color safe)
# --------------------------------------------------------------------------- #
AGENT_PALETTE: dict[str, str] = {
    "DSC": "color(33)",    # Blue
    "TSK": "color(39)",    # Cyan-Blue
    "FTR": "color(35)",    # Green
    "ARC": "color(75)",    # Sky Blue
    "HPO": "color(141)",   # Purple
    "EXE": "color(178)",   # Gold / Orange
    "XAI": "color(208)",   # Orange
    "SNS": "color(172)",   # Dark Orange
    "OVF": "color(197)",   # Coral / Pink
    "VAL": "color(43)",    # Mint Green
    "GOV": "color(214)",   # Amber
    "CRT": "color(135)",   # Medium Purple
}

# --------------------------------------------------------------------------- #
# Status Glyphs & Styles
# --------------------------------------------------------------------------- #
GLYPHS: dict[str, str] = {
    "pass": "✓",
    "complete": "✓",
    "success": "✓",
    "warn": "!",
    "warning": "!",
    "fail": "✗",
    "error": "✗",
    "block": "⛔",
    "blocked": "⛔",
    "reconciliation": "⇄",
    "invariant": "⇄",
    "info": "·",
    "dim": "·",
    "model_narrated": "◆",
    "deterministic_fallback": "◇",
    "human_adjudication": "⚖",
}

START_THEME = Theme(
    {
        "info": "dim cyan",
        "warning": "yellow",
        "danger": "bold red",
        "success": "green",
        "muted": "dim",
        "code": "bold cyan",
        "key": "bold white",
        "value": "bright_cyan",
        "badge": "bold white on color(238)",
        "model_badge": "magenta",
        "det_badge": "blue",
        "human_badge": "bold yellow",
    }
)


def get_agent_code(agent_name_or_stage: str) -> str:
    """Return the 3-letter badge code for any agent or stage name."""
    if agent_name_or_stage in AGENT_BADGES:
        return AGENT_BADGES[agent_name_or_stage]
    clean = agent_name_or_stage.lower().replace("agent", "").strip()
    return AGENT_SHORT_CODES.get(clean, clean[:3].upper())


def format_agent_badge(agent_name: str) -> Text:
    """Render a colored [BADGE] Text renderable."""
    code = get_agent_code(agent_name)
    color = AGENT_PALETTE.get(code, "color(245)")
    return Text(f"[{code}]", style=f"bold {color}")


def format_status_glyph(status: str) -> Text:
    """Return a stylized single-character status glyph."""
    key = status.lower().strip()
    glyph = GLYPHS.get(key, "·")
    if key in ("pass", "complete", "success"):
        return Text(glyph, style="bold green")
    elif key in ("warn", "warning"):
        return Text(glyph, style="bold yellow")
    elif key in ("fail", "error"):
        return Text(glyph, style="bold red")
    elif key in ("block", "blocked"):
        return Text(glyph, style="bold white on red")
    elif key in ("invariant", "reconciliation"):
        return Text(glyph, style="bold cyan")
    elif key == "model_narrated":
        return Text(glyph, style="bold magenta")
    elif key == "deterministic_fallback":
        return Text(glyph, style="bold blue")
    elif key == "human_adjudication":
        return Text(glyph, style="bold yellow")
    return Text(glyph, style="dim")


def create_console(*, width: int = 100, force_color: bool | None = None) -> Console:
    """Create a standardized 100-column Rich console with color / NO_COLOR awareness."""
    if force_color is None:
        if os.environ.get("START_FORCE_COLOR", "").strip().lower() in {"1", "true", "yes"}:
            force_color = True
        elif os.environ.get("NO_COLOR", "").strip():
            force_color = False

    is_terminal = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False
    return Console(
        theme=START_THEME,
        width=width,
        force_terminal=force_color if force_color is not None else is_terminal,
        no_color=True if force_color is False else False,
        highlight=False,
    )
