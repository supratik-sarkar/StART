"""Feature-engineering execution contract.

The problem this solves
-----------------------

Live inspection of the codebase established that there is no mechanism for a registered
test to return transformed data:

* ``TestResult`` carries ``metrics`` (scalars), ``params``, ``thresholds``,
  ``interpretation``, ``limitations`` and ``artifacts`` (file paths). Nothing for a frame.
* ``ArtifactRegistry`` handles file-backed artifacts, not in-memory DataFrames.
* ``ctx.extra["oos"]`` is an informal convention for passing a third cohort *in*, not a
  contract for passing transformed data *out*.
* ``EvidenceRecord`` is hashed and appended to a tamper-evident ledger.

So a feature-engineering step needs somewhere to put a transformed frame, and none of
the existing surfaces is that place. The temptations, all rejected:

======================================  ===========================================
Tempting                                Why it is wrong
======================================  ===========================================
DataFrame in ``TestResult.metrics``     metrics are scalars; a frame would be
                                        hashed into the ledger and bloat it beyond
                                        use, and ``apply_thresholds`` would choke
DataFrame in ``EvidenceRecord``         the ledger is an audit chain, not a data
                                        store; a review's evidence must stay
                                        readable years later
mutate ``TestContext`` in place         invisible action at a distance — the next
                                        test would silently see different data with
                                        nothing recording that it changed
stash in ``ctx.extra``                  untyped, unhashed, and indistinguishable
                                        from the caller's own scratch space
widen ``EvidenceRecord``                changes a hashed schema to transport
                                        matrices; every historical record's identity
                                        would shift
======================================  ===========================================

The separation
--------------

Two objects with different lifetimes and different audiences::

    TransformExecutionResult    runtime payload   frames, fitted state, in memory
              |                                   never serialised to the ledger
              | hashes and counts only
              v
    TestResult -> EvidenceRecord    audit record   scalars and hashes, sealed

The executor performs the transformation. The registered test wrapper calls the *same*
executor and emits audit evidence from it. The mathematics exists once.

Hashing
-------

``state_hash`` and the frame hashes must be stable across processes and platforms for
the same semantic content, because they are what the fitting-scope audit compares. Two
specific hazards are handled explicitly:

* **Float formatting.** ``repr`` varies across versions and loses the last bit;
  ``float.hex()`` is exact and stable.
* **``PYTHONHASHSEED``.** Python's ``hash()`` on strings and tuples is randomised per
  process. Nothing here uses it — everything goes through SHA-256 over a canonical byte
  stream.

Floating state is hashed at a **declared rounding precision** rather than raw. Two
independent fits of the same estimator on the same data can differ in the last bit
through BLAS non-determinism, and a raw hash would report that as a leakage violation.
The rounding is part of the contract, not a fudge: it is set at ``1e-12`` relative,
which is far tighter than any real leakage signal and far looser than BLAS noise.

Standard library plus numpy/pandas.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "TransformExecutionResult",
    "FittingScope",
    "canonical_frame_hash",
    "canonical_state_hash",
    "STATE_HASH_DECIMALS",
]

#: Decimal places used when hashing floating state. Tighter than any real leakage
#: signal, looser than BLAS last-bit noise between two independent fits.
STATE_HASH_DECIMALS = 12


class FittingScope:
    """Where a transformation is permitted to learn from."""

    TRAIN_ONLY = "train_only"
    TRAIN_FOLDS = "train_folds"   # out-of-fold on the train side (target encoding, WoE)
    STATELESS = "stateless"       # learns nothing (log, rank within row, hour-of-day)


def _canonical_scalar(value: Any) -> str:
    """Stable rendering of one value.

    ``float.hex()`` rather than ``repr``: exact, round-trippable, and identical across
    Python versions. ``repr`` of a float has changed between versions and drops
    precision in ways that would silently alter a hash.
    """
    if value is None:
        return "\x00None"
    if isinstance(value, (bool, np.bool_)):
        return f"\x00b{bool(value)}"
    if isinstance(value, (int, np.integer)):
        return f"\x00i{int(value)}"
    if isinstance(value, (float, np.floating)):
        f = float(value)
        if math.isnan(f):
            return "\x00NaN"
        if math.isinf(f):
            return "\x00Inf" if f > 0 else "\x00-Inf"
        if f == 0.0:
            f = 0.0        # normalise signed zero: IEEE distinguishes, semantics do not
        return "\x00f" + f.hex()
    return "\x00s" + str(value)


def canonical_frame_hash(frame: pd.DataFrame | None, decimals: int | None = None) -> str:
    """SHA-256 over a canonical rendering of a frame.

    Columns are sorted so column order is not semantic. The row index is *not* sorted:
    row order is meaningful for temporal and aggregation features, and sorting it would
    make a future-leaking rolling window hash identically to a correct one.
    """
    if frame is None:
        return "\x00absent"
    digest = hashlib.sha256()
    digest.update(f"shape:{frame.shape[0]}x{frame.shape[1]}\x1f".encode())
    for column in sorted(map(str, frame.columns)):
        digest.update(f"col:{column}\x1f".encode())
        series = frame[column]
        if decimals is not None and pd.api.types.is_float_dtype(series):
            series = series.round(decimals)
        for value in series.tolist():
            digest.update(_canonical_scalar(value).encode())
        digest.update(b"\x1e")
    return digest.hexdigest()


def canonical_state_hash(state: Any, decimals: int = STATE_HASH_DECIMALS) -> str:
    """SHA-256 over fitted state, with floats rounded to a declared precision.

    Rounding is deliberate. Two independent fits of the same estimator on identical data
    can differ in the last bit through BLAS non-determinism, and the fitting-scope audit
    compares exactly these hashes. Without rounding it would report platform noise as a
    leakage violation, and the usual response to that is to delete the check.
    """

    def canonicalise(node: Any) -> Any:
        if isinstance(node, dict):
            return {str(k): canonicalise(node[k]) for k in sorted(node, key=str)}
        if isinstance(node, (list, tuple)):
            return [canonicalise(x) for x in node]
        if isinstance(node, np.ndarray):
            return [canonicalise(x) for x in node.tolist()]
        if isinstance(node, pd.Series):
            return {str(k): canonicalise(v) for k, v in sorted(node.items(), key=lambda kv: str(kv[0]))}
        if isinstance(node, pd.DataFrame):
            return canonical_frame_hash(node, decimals)
        if isinstance(node, (bool, np.bool_)):
            return bool(node)
        if isinstance(node, (int, np.integer)):
            return int(node)
        if isinstance(node, (float, np.floating)):
            f = float(node)
            if math.isnan(f):
                return "NaN"
            if math.isinf(f):
                return "Inf" if f > 0 else "-Inf"
            return round(f, decimals) + 0.0    # +0.0 normalises -0.0
        return str(node)

    payload = json.dumps(canonicalise(state), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class TransformExecutionResult:
    """Runtime payload of one transformation. **Never serialised to the ledger.**

    A stdlib dataclass rather than a pydantic model: it holds DataFrames and arbitrary
    fitted state, and forcing that through pydantic's arbitrary-type machinery would buy
    validation of things that do not need validating at the cost of real friction.
    """

    step: str
    transformed_train: pd.DataFrame
    transformed_test: pd.DataFrame | None = None
    transformed_oos: pd.DataFrame | None = None
    #: Learned parameters. Empty for a stateless transformation.
    fitted_state: dict[str, Any] = field(default_factory=dict)
    fitting_scope: str = FittingScope.TRAIN_ONLY
    output_feature_names: tuple[str, ...] = field(default=())
    input_feature_names: tuple[str, ...] = field(default=())
    params: dict[str, Any] = field(default_factory=dict)
    affected_features: tuple[str, ...] = field(default=())
    notes: list[str] = field(default_factory=list)

    # -- hashes ------------------------------------------------------------
    def input_hashes(self) -> dict[str, str]:
        return {"train": self._input_train_hash, "test": self._input_test_hash,
                "oos": self._input_oos_hash}

    _input_train_hash: str = ""
    _input_test_hash: str = ""
    _input_oos_hash: str = ""

    def output_hashes(self) -> dict[str, str]:
        return {
            "train": canonical_frame_hash(self.transformed_train, STATE_HASH_DECIMALS),
            "test": canonical_frame_hash(self.transformed_test, STATE_HASH_DECIMALS),
            "oos": canonical_frame_hash(self.transformed_oos, STATE_HASH_DECIMALS),
        }

    def state_hash(self) -> str:
        return canonical_state_hash(self.fitted_state)

    # -- audit evidence ----------------------------------------------------
    def evidence_metrics(self) -> dict[str, Any]:
        """Scalars and hashes only. **No frames.**

        This is what crosses from runtime payload into the sealed audit record, and the
        boundary is the point of the whole module.
        """
        outputs = self.output_hashes()
        return {
            "step": self.step,
            "fitting_scope": self.fitting_scope,
            "n_features_before": len(self.input_feature_names),
            "n_features_after": len(self.output_feature_names),
            "n_features_affected": len(self.affected_features),
            "n_features_added": max(
                0, len(self.output_feature_names) - len(self.input_feature_names)
            ),
            "affected_features": ", ".join(sorted(self.affected_features)[:30]),
            "state_hash": self.state_hash(),
            "output_hash_train": outputs["train"],
            "output_hash_test": outputs["test"],
            "output_hash_oos": outputs["oos"],
            "input_hash_train": self._input_train_hash,
            "n_train_rows": int(len(self.transformed_train)),
            "n_test_rows": int(len(self.transformed_test)) if self.transformed_test is not None else 0,
            "oos_present": self.transformed_oos is not None,
            "is_stateful": self.fitting_scope != FittingScope.STATELESS,
        }

    def with_input_hashes(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame | None = None,
        oos: pd.DataFrame | None = None,
    ) -> TransformExecutionResult:
        self._input_train_hash = canonical_frame_hash(train, STATE_HASH_DECIMALS)
        self._input_test_hash = canonical_frame_hash(test, STATE_HASH_DECIMALS)
        self._input_oos_hash = canonical_frame_hash(oos, STATE_HASH_DECIMALS)
        return self
