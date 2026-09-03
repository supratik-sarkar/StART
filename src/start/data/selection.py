"""Dataset selection — a typed contract between the wizard and the review.

The bug this module exists to remove
------------------------------------

The wizard used to load a DataFrame, throw it away, and hand the caller a
*display label* — ``"Synthetic: AML / Transaction Monitoring"``. The caller then
tried to recover a file path from it::

    data = None if "Built-In" in dataset_vector else dataset_vector.split(": ", 1)[1]

which produced ``data_path="AML / Transaction Monitoring"`` and aborted the run
with ``ValueError: Unsupported tabular format ''``. Every dataset option failed,
because none of the new labels happened to contain the substring ``"Built-In"``.

The defect is not the substring. It is that a string written for a human to read
was carrying semantic weight. Change the wording of a panel and the data loading
breaks — and it breaks *silently* until something downstream cannot parse a path.

So selection is a value, not a label. :class:`DatasetSelection` carries the
resolved frame, the resolution strategy, and the provenance needed to describe it
honestly. Display strings are derived *from* it and never parsed back into it.

Provenance is part of the contract
----------------------------------

Where data came from is a reviewable fact, not decoration. A pre-flight panel
that names one source while citing another is a defect, and the previous build
shipped exactly that: a panel reading "Synthetic AML Profile Generator" beside a
UCI breast-cancer URL, over breast-cancer data relabelled as fraud.

:meth:`DatasetSelection.provenance_dict` is what goes in the evidence chain and
the report. :meth:`consistency_errors` refuses to let the name and the source
disagree.

Standard library plus pandas only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "DatasetKind",
    "DatasetSelection",
    "select_synthetic",
    "select_german_credit",
    "select_fannie_mae",
    "select_local_file",
    "WIZARD_OPTIONS",
    "resolve_wizard_choice",
]


class DatasetKind(StrEnum):
    """How the data was obtained. Drives provenance, never display."""

    SYNTHETIC = "synthetic"
    PUBLIC_BENCHMARK = "public_benchmark"
    USER_SUPPLIED = "user_supplied"


@dataclass
class DatasetSelection:
    """A resolved dataset plus everything needed to describe it truthfully."""

    kind: DatasetKind
    #: Short name for panels. Derived output — never parsed back.
    display_name: str
    #: The resolved data. Held in memory: the wizard has already loaded it, and
    #: round-tripping through a temp file to satisfy a path-shaped API is how the
    #: original bug was introduced.
    frame: Any = None
    #: Set only when the data genuinely came from a file the user named.
    source_path: str | None = None
    #: URL for public benchmarks, or a plain statement for generated data.
    #: Never a URL for anything that did not come from that URL.
    source_reference: str = ""
    licence_note: str = ""
    #: Default target column, and how it was derived if it was not present as-is.
    target_column: str = ""
    target_derivation: str = "column present in source"
    #: Free-form generator or loader parameters, recorded for reproducibility.
    parameters: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    # -- shape -------------------------------------------------------------
    @property
    def n_rows(self) -> int:
        return int(self.frame.shape[0]) if self.frame is not None else 0

    @property
    def n_columns(self) -> int:
        return int(self.frame.shape[1]) if self.frame is not None else 0

    def volumetrics(self) -> str:
        return f"{self.n_rows:,} Rows x {self.n_columns} Dimensions"

    # -- provenance --------------------------------------------------------
    def provenance_dict(self) -> dict[str, Any]:
        """The block that goes in the evidence chain and the report."""
        return {
            "kind": self.kind.value,
            "display_name": self.display_name,
            "source_reference": self.source_reference,
            "source_path": self.source_path,
            "licence_note": self.licence_note,
            "target_column": self.target_column,
            "target_derivation": self.target_derivation,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "parameters": dict(sorted(self.parameters.items())),
            "notes": list(self.notes),
        }

    def consistency_errors(self) -> list[str]:
        """Refuse to let the display name and the stated source disagree.

        Checked by a test, and worth checking at run time too: the previous
        build shipped a panel naming a synthetic generator beside a URL for a
        cancer dataset, over data that was neither.
        """
        errors: list[str] = []
        ref = self.source_reference.lower()

        if self.kind is DatasetKind.SYNTHETIC:
            if "http" in ref:
                errors.append(
                    f"{self.display_name!r} is synthetic but cites a URL "
                    f"({self.source_reference!r}). Generated data has no external source."
                )
            if self.source_path:
                errors.append("synthetic data must not claim a source_path")
        elif self.kind is DatasetKind.PUBLIC_BENCHMARK:
            if "http" not in ref:
                errors.append(f"{self.display_name!r} is a public benchmark and must cite its source URL")
            if "synthetic" in self.display_name.lower():
                errors.append("a public benchmark must not be described as synthetic")
        elif self.kind is DatasetKind.USER_SUPPLIED:
            if not self.source_path:
                errors.append("user-supplied data must record the path it was read from")

        # The specific historical failure: a cancer dataset presented as fraud.
        if "breast" in ref or "cancer" in ref:
            if not any(w in self.display_name.lower() for w in ("cancer", "diagnostic", "breast")):
                errors.append(
                    "the source reference names a medical diagnostic dataset while the display "
                    "name describes something else. A dataset must be presented as what it is."
                )

        if self.frame is not None and self.target_column:
            if self.target_column not in getattr(self.frame, "columns", []):
                errors.append(f"declared target {self.target_column!r} is not a column in the frame")
        return errors


# --------------------------------------------------------------------------- #
# Constructors — one per option. Each owns its own provenance.
# --------------------------------------------------------------------------- #
def select_synthetic(
    *,
    n_rows: int = 1000,
    prevalence: float = 0.055,
    n_features: int = 25,
    seed: int = 42,
    inject_leakage: bool = False,
    signal_to_noise: float | None = None,
) -> DatasetSelection:
    """Locally generated transaction-monitoring data. No download, no source URL."""
    from start.data.synthetic import generate_synthetic_transactions

    kwargs: dict[str, Any] = {
        "n_rows": n_rows,
        "prevalence": prevalence,
        "n_features": n_features,
        "seed": seed,
    }
    # Optional knobs, tolerated if the generator does not expose them yet.
    for name, value in (("inject_leakage", inject_leakage), ("signal_to_noise", signal_to_noise)):
        if value not in (None, False):
            kwargs[name] = value

    try:
        frame = generate_synthetic_transactions(**kwargs)
    except TypeError:
        for optional in ("inject_leakage", "signal_to_noise"):
            kwargs.pop(optional, None)
        frame = generate_synthetic_transactions(**kwargs)

    return DatasetSelection(
        kind=DatasetKind.SYNTHETIC,
        display_name="Synthetic — AML / transaction monitoring",
        frame=frame,
        source_reference="generated locally; no external source",
        licence_note="not applicable (generated)",
        target_column="is_fraud",
        target_derivation="generated label",
        parameters=kwargs,
    )


def select_german_credit() -> DatasetSelection:
    """UCI Statlog German Credit. Real credit-risk data with a published cost matrix."""
    from start.data.uci_credit import fetch_or_load_german_credit

    frame = fetch_or_load_german_credit()
    return DatasetSelection(
        kind=DatasetKind.PUBLIC_BENCHMARK,
        display_name="UCI Statlog German Credit",
        frame=frame,
        source_reference="https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
        licence_note="UCI Machine Learning Repository — CC BY 4.0",
        target_column="is_bad_credit",
        target_derivation="derived from Statlog class label (1=good, 2=bad) as bad=1",
        parameters={"cost_matrix": "misclassifying bad credit as good costs 5x the reverse"},
        notes=[
            "The dataset ships a documented asymmetric cost matrix (5:1), so the "
            "metric-priority decision can cite the source rather than argue from first "
            "principles.",
            "Contains age and personal-status attributes, so fairness and proxy analysis have real material.",
        ],
    )


def select_fannie_mae(path: str, *, row_limit: int = 100_000) -> DatasetSelection:
    """Fannie Mae loan performance — user-supplied file.

    Not downloadable: it requires accepting Fannie Mae's terms, and a quarter is
    multiple gigabytes. Writing a downloader would produce a demo that stalls on
    a login wall.
    """
    from pathlib import Path

    from start.data.fannie_mae import load_fannie_mae_dataset

    frame = load_fannie_mae_dataset(path, row_limit=row_limit)
    return DatasetSelection(
        kind=DatasetKind.USER_SUPPLIED,
        display_name=f"Fannie Mae single-family — {Path(path).name}",
        frame=frame,
        source_path=path,
        source_reference="supplied by the operator from Fannie Mae's data portal",
        licence_note="subject to Fannie Mae Single-Family Data Terms & Conditions",
        target_column="is_delinquent",
        target_derivation="derived from the loan delinquency status column",
        parameters={"row_limit": row_limit},
        notes=["Row-limited for demonstration; not the full quarterly file."],
    )


def select_local_file(path: str) -> DatasetSelection:
    """Any tabular file the operator names."""
    from pathlib import Path

    from start.data.loaders import load_any_tabular

    frame = load_any_tabular(path)
    return DatasetSelection(
        kind=DatasetKind.USER_SUPPLIED,
        display_name=f"Local file — {Path(path).name}",
        frame=frame,
        source_path=path,
        source_reference="operator-supplied local file",
        licence_note="unknown; supplied by the operator",
        target_column="",
        target_derivation="inferred by discovery",
    )


# --------------------------------------------------------------------------- #
# Wizard wiring
# --------------------------------------------------------------------------- #
#: (key, menu text). The key is what the code branches on; the text is display.
WIZARD_OPTIONS: tuple[tuple[str, str], ...] = (
    ("1", "Synthetic AML / fraud transactions (default; generated locally, no download)"),
    ("2", "UCI Statlog German Credit (public benchmark; documented 5:1 cost matrix)"),
    ("3", "Fannie Mae single-family loan performance (you supply the file)"),
    ("4", "Custom local dataset (CSV, Parquet, TSV, JSON)"),
)


def resolve_wizard_choice(
    choice: str,
    *,
    prompt: Any = input,
    echo: Any = print,
    seed: int = 42,
) -> DatasetSelection:
    """Turn a menu key into a resolved selection.

    Falls back to synthetic on any failure, and **says so**. A silent
    substitution is how a demo ends up running on data nobody chose.
    """
    choice = (choice or "1").strip()

    if choice == "2":
        try:
            echo("\nLoading UCI Statlog German Credit benchmark...")
            return select_german_credit()
        except Exception as exc:
            echo(f"\n[Warning] Could not load German Credit: {exc}")
            echo("Falling back to the synthetic generator.")
            return select_synthetic(seed=seed)

    if choice == "3":
        try:
            from start.data.fannie_mae import FANNIE_MAE_TERMS_NOTE

            echo(f"\n[Note] {FANNIE_MAE_TERMS_NOTE}")
        except Exception:
            echo(
                "\n[Note] Fannie Mae data must be obtained directly from Fannie Mae, "
                "subject to their Single-Family Data Terms & Conditions."
            )
        path = str(prompt("Enter path to the Fannie Mae data file: ")).strip()
        raw_limit = str(prompt("Enter row limit [default: 100000]: ")).strip()
        row_limit = int(raw_limit) if raw_limit.isdigit() and int(raw_limit) > 0 else 100_000
        try:
            return select_fannie_mae(path, row_limit=row_limit)
        except Exception as exc:
            echo(f"\n[Warning] Could not load the Fannie Mae file: {exc}")
            echo("Falling back to the synthetic generator.")
            return select_synthetic(seed=seed)

    if choice == "4":
        path = str(prompt("\nEnter the dataset file path: ")).strip()
        if path:
            try:
                return select_local_file(path)
            except Exception as exc:
                echo(f"\n[Warning] Could not load {path}: {exc}")
                echo("Falling back to the synthetic generator.")
        return select_synthetic(seed=seed)

    return select_synthetic(seed=seed)
