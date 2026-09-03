"""Progress display helpers for heavy review steps (v2.3.1 #6).

Two honest modes, never faked:

- ``progress_bar(total, description)`` — a horizontal Rich progress bar with
  percentage, for work with a known iteration count (e.g. tuning trials,
  adapter sweep, K-fold folds). Advance it once per completed unit.
- ``spinner(description)`` — an indeterminate Rich spinner with elapsed time,
  for heavy work whose exact progress is not observable (e.g. a single model
  fit). No percentage is shown because none is known.

Both are context managers and degrade to no-ops when Rich is unavailable or the
caller passes ``enabled=False`` (e.g. non-interactive/batch runs), so they never
interfere with logging or tests.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any


@contextlib.contextmanager
def spinner(description: str, *, enabled: bool = True, console: Any = None) -> Iterator[None]:
    """Indeterminate spinner + elapsed time for work without a known count."""
    if not enabled:
        yield
        return

    status = None
    try:
        from rich.console import Console

        c = console or Console()
        status = c.status(f"[bold]{description}[/bold]", spinner="dots")
    except Exception:
        yield
        return

    status.start()
    try:
        yield
    finally:
        status.stop()


@contextlib.contextmanager
def progress_bar(
    total: int, description: str, *, enabled: bool = True, console: Any = None
) -> Iterator[Callable[[int], None]]:
    """Percentage progress bar for counted loops.

    Yields an ``advance(n=1)`` callable. Only use this when ``total`` is the
    real number of units of work; the percentage shown is therefore real.
    """
    if not enabled or total <= 0:
        yield lambda n=1: None
        return

    progress = None
    try:
        from rich.progress import (
            BarColumn,
            Progress,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )
    except Exception:
        yield lambda n=1: None
        return

    progress.start()
    try:
        task = progress.add_task(description, total=total)

        def advance(n: int = 1) -> None:
            progress.advance(task, n)

        yield advance
    finally:
        progress.stop()
