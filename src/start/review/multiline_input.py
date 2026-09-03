"""Multiline governance text input.

The bug this fixes
------------------

Governance free-text fields were read with a single ``input()`` call. A reviewer pasting

    The model is a high-materiality market and treasury risk framework...
    The framework supports independent risk oversight...
    The review should assess portfolio mathematics, covariance, VaR...

got line 1 as their business context, and lines 2 and 3 were consumed by whatever menus
came next. A line beginning "2." would silently select menu option 2.

That is worse than losing the text. The review continued, appeared to succeed, and
recorded settings nobody chose — and the reviewer had no way to tell, because the menus
had already scrolled past.

Why a terminator rather than a blank line or a timer
-----------------------------------------------------

A blank line cannot end the input, because governance text has paragraphs and reviewers
paste them. Paste-boundary or timing detection is worse still: it makes correctness
depend on typing speed and terminal buffering, so the same paste behaves differently on
a slow SSH link. An explicit sentinel line is the only rule that is unambiguous for both
a human typing and a terminal pasting.

``END`` on a line by itself. Nothing else terminates. Everything before it is content,
including blank lines, numeric-only lines, and lines that look exactly like menu options
— those are precisely the cases that used to leak.

Standard library only.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import TextIO

__all__ = [
    "read_multiline_text", "MULTILINE_TERMINATOR", "ReviewCancelled",
    "format_multiline_prompt",
]

#: The sentinel. Compared after stripping surrounding whitespace, case-sensitively:
#: a reviewer writing "the end of the sample" must not terminate their own paragraph.
MULTILINE_TERMINATOR = "END"


class ReviewCancelled(RuntimeError):
    """The reviewer cancelled. Not an error condition — an ordinary outcome."""


def format_multiline_prompt(label: str, *, terminator: str = MULTILINE_TERMINATOR,
                            required: bool = False) -> str:
    lines = [
        f"Enter {label}",
        f"  Paste one or more lines below. When finished, enter a line containing "
        f"only: {terminator}",
    ]
    if not required:
        lines.append(f"  Optional — enter {terminator} on its own to skip.")
    return "\n".join(lines)


def read_multiline_text(
    label: str,
    *,
    terminator: str = MULTILINE_TERMINATOR,
    required: bool = False,
    stream: TextIO | None = None,
    printer: Callable[[str], None] | None = None,
) -> str:
    """Read free text until a line equal to ``terminator``.

    Everything before the terminator is content. Blank lines and paragraph structure are
    preserved; only leading and trailing whitespace on the whole block is stripped, so a
    reviewer's paragraphing survives into the evidence record.

    ``required=True`` re-prompts on empty input rather than accepting a blank governance
    field, because an empty business context recorded as if it were supplied is worse
    than being asked twice.

    EOF (Ctrl-D) and interrupt (Ctrl-C) raise :class:`ReviewCancelled` so the caller can
    exit cleanly. Neither should produce a stack trace: cancelling a prompt is a normal
    thing to do.
    """
    source = stream if stream is not None else sys.stdin
    emit = printer if printer is not None else print

    while True:
        emit(format_multiline_prompt(label, terminator=terminator, required=required))
        collected: list[str] = []
        terminated = False

        while True:
            try:
                line = source.readline()
            except KeyboardInterrupt as exc:
                raise ReviewCancelled(f"{label}: cancelled by reviewer") from exc

            if line == "":
                # EOF without a terminator. Text already typed is not silently
                # discarded — losing a reviewer's paragraphs to a stray Ctrl-D would be
                # its own small disaster — but the cancellation is still surfaced.
                if collected:
                    raise ReviewCancelled(
                        f"{label}: input ended before the {terminator} line "
                        f"({len(collected)} line(s) were entered)"
                    )
                raise ReviewCancelled(f"{label}: input ended (EOF)")

            stripped = line.rstrip("\n").rstrip("\r")
            if stripped.strip() == terminator:
                terminated = True
                break
            collected.append(stripped)

        text = "\n".join(collected).strip()
        if text or not required:
            return text
        if terminated:
            emit(f"  {label} is required. Please enter at least one line.")
