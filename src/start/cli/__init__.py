"""CLI entry points.

Lazy for the same reason the rest of the package is: importing ``start.cli``
must not require Typer and Rich to be present. The console-script entry point
``start = "start.cli:app"`` resolves ``app`` through ``__getattr__``, which
imports Typer at that moment — by which time the user has clearly asked for the
CLI and an ImportError is the right, legible failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from start.cli.main import app
    from start.cli.view import ProgressDashboardUI

_LAZY = {"app": "start.cli.main", "ProgressDashboardUI": "start.cli.view"}

__all__ = ["ProgressDashboardUI", "app"]


def __getattr__(name: str) -> Any:
    module_path = _LAZY.get(name)
    if module_path is None:
        raise AttributeError(f"module 'start.cli' has no attribute {name!r}")
    from importlib import import_module

    value = getattr(import_module(module_path), name)
    globals()[name] = value
    return value
