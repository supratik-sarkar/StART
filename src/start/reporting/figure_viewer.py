"""Figure presentation — open generated plots instead of drawing them in the terminal.

Why this exists
---------------

StART generated diagnostic figures for several releases and never showed them. The
terminal drew ASCII approximations instead: a ROC curve rendered in block characters
at 35x10 resolution, a reliability diagram as a bar table. They were a workaround for
having no figures, and they outlived their reason.

An ASCII ROC cannot show a curve crossing the chance line at low FPR. A PNG can. And a
reviewer deciding whether to sign off on a model with ECE 0.20 needs to *see* the
reliability diagram bend, not read a delta column.

So the terminal keeps what is genuinely textual — the threshold sweep, the drift
sparkline, the metric tables — and every plot opens in the system viewer at the moment
it is produced, labelled, in sequence.

Design constraints, each of which comes from a way this could go wrong
---------------------------------------------------------------------

**Never break a review.** A missing viewer, a sandboxed environment, an X-less server:
all are ordinary. Every failure path here is swallowed and reported, never raised. A
figure that will not open is a cosmetic loss; a review that dies because of one is not.

**Never open anything in CI or under pytest.** A test suite that spawns image viewers is
a test suite people stop running. Detection is belt-and-braces: not a TTY, ``CI`` set,
pytest in ``sys.modules``, or an explicit opt-out — any one of them suppresses opening.

**Pace for recording.** Four figures firing in 200ms is a flicker on camera. A
configurable delay between figures lets each one land. Default 1.5s, tunable, zero to
disable.

**Say what is being shown before showing it.** The viewer takes focus; the reviewer needs
to know which figure arrived and what it means before their eyes leave the terminal.

Standard library only.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "FigurePresentation",
    "FigureSpec",
    "open_figure",
    "should_open_figures",
    "DEFAULT_FIGURE_DELAY",
]

DEFAULT_FIGURE_DELAY = 1.5

#: Human-readable titles and one-line readings, keyed by the figure key that
#: ``generate_all_report_figures`` returns. Anything not listed still opens; it
#: just gets a generic label rather than an interpretation.
FIGURE_CATALOGUE: dict[str, tuple[str, str]] = {
    "roc_curve": (
        "ROC curve",
        "Discrimination across all thresholds. The diagonal is chance.",
    ),
    "pr_curve": (
        "Precision-recall curve",
        "The honest performance picture when the positive class is rare — compare "
        "the curve against the base rate, not against 0.5.",
    ),
    "calibration_curve": (
        "Calibration reliability diagram",
        "Whether predicted probabilities mean what they say. Points below the "
        "diagonal are over-confident.",
    ),
    "confusion_matrix": (
        "Confusion matrix",
        "Counts at the chosen decision threshold — not at 0.5 unless that is what "
        "was chosen.",
    ),
    "feature_drift": (
        "Feature sensitivity / drift",
        "How far the metric moves when each input is shocked.",
    ),
    "score_distribution": (
        "Score distribution by class",
        "Two overlapping humps mean the model is not separating the classes.",
    ),
    "global_importance": (
        "Global feature attribution",
        "Which features drive the model overall. Not a causal claim.",
    ),
    "local_explanation": (
        "Local explanation",
        "Why this specific case scored as it did — the reason-code artefact.",
    ),
    "distribution_with_bounds": (
        "Feature distribution with outlier bounds",
        "The candidate cut-points for each outlier rule, drawn on the actual data.",
    ),
}


@dataclass(frozen=True)
class FigureSpec:
    """One figure to present."""

    key: str
    path: str
    #: Short quantitative note shown beside the title, e.g. "AUC 0.8913".
    headline: str = ""
    #: Overrides the catalogue interpretation when the context needs something
    #: more specific than the generic reading.
    interpretation: str = ""
    cohort: str = ""

    def title(self) -> str:
        base = FIGURE_CATALOGUE.get(self.key, (self.key.replace("_", " ").title(), ""))[0]
        return f"{base} — {self.cohort}" if self.cohort else base

    def reading(self) -> str:
        if self.interpretation:
            return self.interpretation
        return FIGURE_CATALOGUE.get(self.key, ("", ""))[1]


def should_open_figures(
    explicit: bool | None = None,
    env: dict[str, str] | None = None,
    stream: Any = None,
) -> tuple[bool, str]:
    """Decide whether figures may be opened, and say why not when they may not.

    ``explicit`` is the CLI flag: ``False`` always wins, ``True`` still yields to the
    hard suppressors below, because a flag set in a config file should not cause a CI
    job to spawn image viewers.

    Returns ``(allowed, reason)``. The reason is surfaced to the user, so an operator
    who expected figures and got none finds out immediately rather than assuming the
    feature is broken.
    """
    env = os.environ if env is None else env  # type: ignore[assignment]
    stream = sys.stdout if stream is None else stream

    if explicit is False:
        return False, "--no-open-figures was set"
    if env.get("START_NO_OPEN_FIGURES", "").strip().lower() in {"1", "true", "yes"}:
        return False, "START_NO_OPEN_FIGURES is set"
    if "pytest" in sys.modules:
        return False, "running under pytest"
    if env.get("CI", "").strip():
        return False, "CI environment detected"
    try:
        if not stream.isatty():
            return False, "output is not a terminal"
    except Exception:
        return False, "output stream does not support isatty()"
    if platform.system() == "Linux" and not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        return False, "no display available (headless Linux)"

    return True, ""


def _open_command(path: str) -> list[str] | None:
    system = platform.system()
    if system == "Darwin":
        return ["open", path]
    if system == "Linux":
        return ["xdg-open", path]
    return None  # Windows handled via os.startfile


def open_figure(path: str) -> tuple[bool, str]:
    """Open one file in the system viewer. Never raises.

    Returns ``(opened, error)``. Callers report the error rather than propagating it:
    the inability to display a picture must not end a model review.
    """
    target = Path(path)
    if not target.exists():
        return False, f"file not found: {path}"

    try:
        if platform.system() == "Windows":
            os.startfile(str(target))  # type: ignore[attr-defined]  # noqa: S606
            return True, ""
        command = _open_command(str(target))
        if command is None:
            return False, f"no viewer command known for {platform.system()}"
        subprocess.run(
            command,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        return True, ""
    except Exception as exc:  # deliberately broad: display is never load-bearing
        return False, f"{type(exc).__name__}: {exc}"


@dataclass
class FigurePresentation:
    """Presents a sequence of figures, labelled and paced.

    ``echo`` is injected so the caller supplies its own console (Rich, plain print, or a
    capture list in tests) without this module importing a rendering library.
    """

    delay_seconds: float = DEFAULT_FIGURE_DELAY
    enabled: bool = True
    suppressed_reason: str = ""
    echo: Callable[[str], None] = print
    sleep: Callable[[float], None] = time.sleep
    opener: Callable[[str], tuple[bool, str]] = open_figure
    presented: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def configure(
        cls,
        *,
        explicit: bool | None = None,
        delay_seconds: float = DEFAULT_FIGURE_DELAY,
        echo: Callable[[str], None] = print,
        env: dict[str, str] | None = None,
        stream: Any = None,
    ) -> FigurePresentation:
        allowed, reason = should_open_figures(explicit=explicit, env=env, stream=stream)
        return cls(
            delay_seconds=max(0.0, delay_seconds),
            enabled=allowed,
            suppressed_reason=reason,
            echo=echo,
        )

    def present(self, figures: list[FigureSpec]) -> list[dict[str, Any]]:
        """Announce and open each figure in order.

        Every figure is announced whether or not it can be opened — the path is useful
        on its own, and a reviewer working over SSH still learns what was produced and
        where to find it.
        """
        if not figures:
            return []

        total = len(figures)
        self.echo("")
        if not self.enabled and self.suppressed_reason:
            self.echo(
                f"  {total} figure(s) written to disk; not opening ({self.suppressed_reason})."
            )

        for index, figure in enumerate(figures, start=1):
            headline = f"  {figure.headline}" if figure.headline else ""
            self.echo(f"  [FIGURE {index}/{total}]  {figure.title()}{headline}")
            self.echo(f"                {figure.path}")
            reading = figure.reading()
            if reading:
                self.echo(f"                {reading}")

            record: dict[str, Any] = {
                "key": figure.key,
                "path": figure.path,
                "title": figure.title(),
                "opened": False,
                "error": "",
            }

            if self.enabled:
                opened, error = self.opener(figure.path)
                record["opened"] = opened
                record["error"] = error
                if not opened:
                    self.echo(f"                (could not open: {error})")
                elif index < total and self.delay_seconds > 0:
                    self.sleep(self.delay_seconds)

            self.presented.append(record)
            self.echo("")

        return self.presented

    def as_evidence(self) -> dict[str, Any]:
        """Summary for the evidence chain — what was produced and whether it was shown."""
        return {
            "figures_produced": len(self.presented),
            "figures_opened": sum(1 for r in self.presented if r["opened"]),
            "opening_enabled": self.enabled,
            "suppressed_reason": self.suppressed_reason,
            "figures": [
                {"key": r["key"], "path": r["path"], "opened": r["opened"]}
                for r in self.presented
            ],
        }
