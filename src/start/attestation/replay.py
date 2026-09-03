"""Deterministic replay and divergence localisation.

The evidence ledger is append-only and hash-chained, so tampering is
detectable. Detectable is not the same as *diagnosable*: a chain that fails
verification tells you something was altered, and nothing about what.

This module replays a ledger and localises the break. It answers three
questions a reviewer actually asks:

* Is the chain intact, and if not, at which entry does it first diverge?
* Do the recorded record hashes still match the records they claim to cover
  (content tampering), or does only the chain linkage differ (re-ordering or
  splicing)?
* Given two ledgers — an original and a re-run — do they agree on the
  quantitative content, ignoring the fields that are legitimately allowed to
  differ between runs (timestamps, run identifiers, wall-clock durations)?

The third question is the one that makes reproducibility a fact rather than an
aspiration. "Reproducible" normally means somebody ran it twice and eyeballed
the output. Here it means a machine compared every metric across two
independent executions and reported which ones moved.

Standard library only, so replay works on an archived ledger file with no
StART installation beyond this package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

__all__ = [
    "GENESIS",
    "ChainVerdict",
    "ReplayComparison",
    "canonical_json",
    "replay_ledger",
    "compare_ledgers",
    "VOLATILE_FIELDS",
]

GENESIS = "0" * 64

#: Fields expected to differ between two honest runs of the same review.
#: Anything not on this list that differs is a reproducibility finding.
VOLATILE_FIELDS: frozenset[str] = frozenset(
    {
        "run_id",
        "evidence_id",
        "created_at",
        "timestamp",
        "started_at",
        "finished_at",
        "duration_seconds",
        "wall_clock_seconds",
        "last_latency_seconds",
        "hostname",
        "pid",
    }
)


def canonical_json(obj: Any) -> str:
    """Stable JSON rendering used for every hash in this module."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)


def _sha(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ChainVerdict:
    """The outcome of replaying one ledger."""

    path: str
    entries: int
    intact: bool
    first_divergence_index: int | None
    divergence_kind: str
    detail: str
    head: str
    record_hashes: tuple[str, ...] = field(default=())

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "entries": self.entries,
            "intact": self.intact,
            "first_divergence_index": self.first_divergence_index,
            "divergence_kind": self.divergence_kind,
            "detail": self.detail,
            "head": self.head,
        }

    def summary_line(self) -> str:
        if self.intact:
            return f"ledger intact: {self.entries} entries, head {self.head[:16]}…"
        return (
            f"ledger DIVERGENT at entry {self.first_divergence_index} "
            f"({self.divergence_kind}): {self.detail}"
        )


def replay_ledger(path: str | Path) -> ChainVerdict:
    """Recompute a ledger's hash chain from its contents.

    Distinguishes four break kinds, in the order they are detected:

    ``malformed``   a line is not valid JSON or lacks required fields
    ``index``       entry indices are not contiguous from zero
    ``content``     a record no longer hashes to its recorded ``record_hash``
    ``linkage``     records are intact but ``prev_hash``/``entry_hash`` do not chain
    """
    ledger_path = Path(path)
    if not ledger_path.exists():
        return ChainVerdict(
            path=str(ledger_path),
            entries=0,
            intact=True,
            first_divergence_index=None,
            divergence_kind="none",
            detail="ledger file does not exist; an empty chain is trivially intact",
            head=GENESIS,
        )

    prev_hash = GENESIS
    record_hashes: list[str] = []
    index = -1

    with ledger_path.open(encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle):
            if not raw.strip():
                continue
            index += 1
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError as exc:
                return ChainVerdict(
                    path=str(ledger_path),
                    entries=index,
                    intact=False,
                    first_divergence_index=index,
                    divergence_kind="malformed",
                    detail=f"line {line_number + 1} is not valid JSON: {exc}",
                    head=prev_hash,
                    record_hashes=tuple(record_hashes),
                )

            missing = [k for k in ("index", "prev_hash", "record_hash", "entry_hash", "record")
                       if k not in entry]
            if missing:
                return ChainVerdict(
                    path=str(ledger_path),
                    entries=index,
                    intact=False,
                    first_divergence_index=index,
                    divergence_kind="malformed",
                    detail=f"entry is missing required field(s): {', '.join(missing)}",
                    head=prev_hash,
                    record_hashes=tuple(record_hashes),
                )

            if entry["index"] != index:
                return ChainVerdict(
                    path=str(ledger_path),
                    entries=index,
                    intact=False,
                    first_divergence_index=index,
                    divergence_kind="index",
                    detail=f"expected index {index}, found {entry['index']}; entries were "
                    "removed, reordered or inserted",
                    head=prev_hash,
                    record_hashes=tuple(record_hashes),
                )

            recomputed_record = _sha(canonical_json(entry["record"]))
            if recomputed_record != entry["record_hash"]:
                return ChainVerdict(
                    path=str(ledger_path),
                    entries=index,
                    intact=False,
                    first_divergence_index=index,
                    divergence_kind="content",
                    detail=(
                        "the record's content no longer hashes to its recorded record_hash — "
                        "this entry was edited after it was written"
                    ),
                    head=prev_hash,
                    record_hashes=tuple(record_hashes),
                )

            if entry["prev_hash"] != prev_hash:
                return ChainVerdict(
                    path=str(ledger_path),
                    entries=index,
                    intact=False,
                    first_divergence_index=index,
                    divergence_kind="linkage",
                    detail=(
                        "record content is intact but prev_hash does not match the previous "
                        "entry_hash — the chain was spliced"
                    ),
                    head=prev_hash,
                    record_hashes=tuple(record_hashes),
                )

            expected_entry = _sha(prev_hash + entry["record_hash"])
            if expected_entry != entry["entry_hash"]:
                return ChainVerdict(
                    path=str(ledger_path),
                    entries=index,
                    intact=False,
                    first_divergence_index=index,
                    divergence_kind="linkage",
                    detail="entry_hash does not equal sha256(prev_hash + record_hash)",
                    head=prev_hash,
                    record_hashes=tuple(record_hashes),
                )

            record_hashes.append(entry["record_hash"])
            prev_hash = entry["entry_hash"]

    return ChainVerdict(
        path=str(ledger_path),
        entries=index + 1,
        intact=True,
        first_divergence_index=None,
        divergence_kind="none",
        detail="every record hash and chain link recomputed identically",
        head=prev_hash,
        record_hashes=tuple(record_hashes),
    )


# --------------------------------------------------------------------------- #
# Cross-run comparison
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReplayComparison:
    """Whether two runs of the same review agree on their numbers."""

    reproducible: bool
    compared_records: int
    matched_metrics: int
    drifted: tuple[dict[str, Any], ...]
    only_in_original: tuple[str, ...]
    only_in_rerun: tuple[str, ...]
    tolerance: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "reproducible": self.reproducible,
            "compared_records": self.compared_records,
            "matched_metrics": self.matched_metrics,
            "drifted_metrics": [dict(d) for d in self.drifted],
            "only_in_original": list(self.only_in_original),
            "only_in_rerun": list(self.only_in_rerun),
            "tolerance": self.tolerance,
        }

    def summary_line(self) -> str:
        if self.reproducible:
            return (
                f"reproducible: {self.matched_metrics} metrics across "
                f"{self.compared_records} records agreed within {self.tolerance}"
            )
        return (
            f"NOT reproducible: {len(self.drifted)} metric(s) drifted, "
            f"{len(self.only_in_original)} missing from the re-run, "
            f"{len(self.only_in_rerun)} new in the re-run"
        )


def _records_by_test(path: str | Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    ledger_path = Path(path)
    if not ledger_path.exists():
        return out
    with ledger_path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)["record"]
            except (json.JSONDecodeError, KeyError):
                continue
            key = str(record.get("test_id") or record.get("test_name") or len(out))
            # A test may run more than once; disambiguate by occurrence.
            suffix = 0
            candidate = key
            while candidate in out:
                suffix += 1
                candidate = f"{key}#{suffix}"
            out[candidate] = record
    return out


def compare_ledgers(
    original: str | Path,
    rerun: str | Path,
    *,
    tolerance: float = 1e-9,
    volatile_fields: frozenset[str] = VOLATILE_FIELDS,
) -> ReplayComparison:
    """Compare two ledgers on quantitative content only.

    Fields in ``volatile_fields`` are expected to differ and are ignored.
    Everything else that differs is reported, with the original and re-run
    values side by side, so a reproducibility finding names the metric rather
    than the file.
    """
    from start.attestation.claims import flatten_evidence_values

    left = _records_by_test(original)
    right = _records_by_test(rerun)

    only_left = tuple(sorted(set(left) - set(right)))
    only_right = tuple(sorted(set(right) - set(left)))

    drifted: list[dict[str, Any]] = []
    matched = 0

    for key in sorted(set(left) & set(right)):
        lv = flatten_evidence_values(left[key])
        rv = flatten_evidence_values(right[key])
        for path in sorted(set(lv) | set(rv)):
            tail = path.rsplit(".", 1)[-1]
            if tail in volatile_fields:
                continue
            a, b = lv.get(path), rv.get(path)
            if a is None or b is None:
                drifted.append(
                    {
                        "record": key,
                        "field": path,
                        "original": a,
                        "rerun": b,
                        "kind": "present_in_one_run_only",
                    }
                )
                continue
            scale = max(abs(a), abs(b))
            delta = abs(a - b)
            if (delta <= tolerance) if scale < 1.0 else (delta / scale <= tolerance):
                matched += 1
            else:
                drifted.append(
                    {
                        "record": key,
                        "field": path,
                        "original": a,
                        "rerun": b,
                        "absolute_delta": delta,
                        "kind": "value_drift",
                    }
                )

    return ReplayComparison(
        reproducible=not drifted and not only_left and not only_right,
        compared_records=len(set(left) & set(right)),
        matched_metrics=matched,
        drifted=tuple(drifted),
        only_in_original=only_left,
        only_in_rerun=only_right,
        tolerance=tolerance,
    )
