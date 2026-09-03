"""Narrative invariance attestation.

The question a model risk function will ask about any LLM-assisted review is
short: *if the language model had said something different, would the
conclusion have changed?*

If the answer is yes, the language model is part of the measurement chain and
must itself be validated — an obligation most firms are not willing to take on
for a third-party frontier model whose weights change without notice. If the
answer is no, the language model is a rendering layer, and the review's
quantitative content stands on the deterministic engines underneath it.

Almost every LLM-assisted review tool asserts the second answer. This module
*tests* it.

The procedure:

1. Produce the same review section twice — once through the deterministic
   template path, once through the model path. Both are handed identical
   evidence.
2. Extract the quantitative claims from each (:mod:`start.attestation.claims`).
3. Bind both claim sets to the evidence.
4. Compare. The narratives are expected to differ in wording. They are required
   not to differ in *number*.

Four failure modes are distinguished, because they have different remedies:

``unbound``
    The model asserted a figure that appears nowhere in the evidence. The
    figure was invented. This is the failure everybody fears and nobody
    measures.
``contradiction``
    Both narratives make a claim in the same context, and the values differ.
    The model misread evidence it was given.
``omission``
    The deterministic path reported a figure the model path dropped. Less
    alarming than invention, but a report that quietly loses the breached
    threshold is not a safe report.
``addition``
    The model path reported a figure the deterministic path did not, and it
    *is* bound to evidence. Usually benign — the model surfaced something real
    from a different corner of the evidence — but it is surfaced rather than
    ignored, because a reviewer should know the two paths disagreed on scope.

A review whose narrative fails invariance is not blocked from existing. It is
blocked from being *sealed* as evidence-bound, which is the honest outcome:
the prose still has value, it simply cannot claim the property it failed.

Standard library only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from start.attestation.claims import bind_claims, extract_claims

__all__ = [
    "Divergence",
    "InvarianceAttestation",
    "attest_narrative_invariance",
    "ATTESTATION_VERSION",
]

ATTESTATION_VERSION = "narrative-invariance/1.0"

#: An unbound model figure this close to a deterministic figure is far more
#: likely to be a corrupted transcription of it than an unrelated invention.
#: Reporting it as a contradiction (with both values side by side) tells a
#: reviewer what to look at; reporting it as a bare invention does not.
_CONTRADICTION_PROXIMITY = 0.05


@dataclass(frozen=True)
class Divergence:
    """One way in which the two narratives failed to agree."""

    kind: str  # unbound | contradiction | omission | addition
    detail: str
    deterministic_value: float | None = None
    model_value: float | None = None
    context: str = ""
    severity: str = "high"  # high | medium | low

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "detail": self.detail,
            "deterministic_value": self.deterministic_value,
            "model_value": self.model_value,
            "context": self.context,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class InvarianceAttestation:
    """The result of testing one section for narrative invariance."""

    section: str
    invariant: bool
    divergences: tuple[Divergence, ...]
    deterministic_binding: dict[str, Any]
    model_binding: dict[str, Any]
    tolerance: float
    version: str = ATTESTATION_VERSION
    provider_name: str = ""
    narration_path: str = "deterministic_only"  # deterministic_only | fallback | model
    requested_provider: str | None = None
    requested_model: str | None = None
    fallback_reason: str | None = None
    fallback_detail: str | None = None
    fallback_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- summary ------------------------------------------------------------
    @property
    def unbound_claim_rate(self) -> float:
        """Fraction of the model narrative's figures with no evidential basis."""
        total = self.model_binding.get("total_claims", 0)
        if not total:
            return 0.0
        return round(self.model_binding.get("unbound_claims", 0) / total, 6)

    def blocking_divergences(self) -> tuple[Divergence, ...]:
        return tuple(d for d in self.divergences if d.severity == "high")

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "section": self.section,
            "provider": self.provider_name,
            "narration_path": self.narration_path,
            "requested_provider": self.requested_provider,
            "requested_model": self.requested_model,
            "fallback_reason": self.fallback_reason,
            "fallback_detail": self.fallback_detail,
            "fallback_at": self.fallback_at,
            "invariant": self.invariant,
            "tolerance": self.tolerance,
            "unbound_claim_rate": self.unbound_claim_rate,
            "deterministic_binding": self.deterministic_binding,
            "model_binding": self.model_binding,
            "divergences": [d.as_dict() for d in self.divergences],
            "metadata": dict(sorted(self.metadata.items())),
        }

    def canonical_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))

    def attestation_hash(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def summary_line(self) -> str:
        verdict = "INVARIANT" if self.invariant else "DIVERGENT"
        return (
            f"[{verdict}] {self.section}: "
            f"{self.model_binding.get('bound_claims', 0)}/"
            f"{self.model_binding.get('total_claims', 0)} claims bound, "
            f"{len(self.blocking_divergences())} blocking divergence(s)"
        )


def _values_agree(a: float, b: float, tolerance: float) -> bool:
    scale = max(abs(a), abs(b))
    delta = abs(a - b)
    return (delta <= tolerance) if scale < 1.0 else (delta / scale <= tolerance)


def _relative_gap(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale else 0.0


def attest_narrative_invariance(
    *,
    section: str,
    deterministic_narrative: str,
    model_narrative: str,
    evidence: Any,
    tolerance: float = 5e-4,
    provider_name: str = "",
    narration_path: str = "deterministic_only",
    requested_provider: str | None = None,
    requested_model: str | None = None,
    fallback_reason: str | None = None,
    fallback_detail: str | None = None,
    fallback_at: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> InvarianceAttestation:
    """Test whether two renderings of one section make the same numeric claims.

    ``evidence`` may be raw ledger JSON, a list of evidence records, pydantic
    models, or dataclasses — it is flattened to ``field path -> number`` before
    binding.
    """
    det_claims = extract_claims(deterministic_narrative)
    mod_claims = extract_claims(model_narrative)

    det_binding = bind_claims(det_claims, evidence, tolerance=tolerance)
    mod_binding = bind_claims(mod_claims, evidence, tolerance=tolerance)

    divergences: list[Divergence] = []

    def agrees(a: float, b: float) -> bool:
        return _values_agree(a, b, tolerance)

    det_values = [c.normalised_value() for c in det_claims]
    mod_values = [c.normalised_value() for c in mod_claims]

    # 1. Invention, and its more diagnosable cousin, contradiction.
    #
    # Soft claims (e.g. word numbers like "three tests") can NEVER produce
    # unbound or contradiction divergences.
    unbound_values = {u["value"] for u in mod_binding.unbound if not u.get("soft")}
    for record in mod_binding.unbound:
        if record.get("soft", False):
            continue

        model_value = record["value"]
        normalised = record["normalised_value"]
        near = [d for d in det_values if not agrees(d, normalised)]
        nearest = min(near, key=lambda d: abs(d - normalised), default=None)

        if nearest is not None and _relative_gap(nearest, normalised) <= _CONTRADICTION_PROXIMITY:
            divergences.append(
                Divergence(
                    kind="contradiction",
                    detail=(
                        f"the model narrative asserts {record['surface']!r}, which binds to no "
                        f"evidence value; the deterministic narrative reports {nearest:g} in its "
                        "place, so this reads as a corrupted transcription rather than an "
                        "unrelated invention"
                    ),
                    deterministic_value=nearest,
                    model_value=model_value,
                    context=record["context"],
                    severity="high",
                )
            )
        else:
            divergences.append(
                Divergence(
                    kind="unbound",
                    detail=(
                        f"the model narrative asserts {record['surface']!r}, which matches no "
                        "value in the evidence supplied to it and no figure in the "
                        "deterministic narrative"
                    ),
                    model_value=model_value,
                    context=record["context"],
                    severity="high",
                )
            )

    # 2. Omission — the deterministic path reported a figure the model dropped.
    for claim in det_claims:
        if any(agrees(claim.normalised_value(), m) for m in mod_values):
            continue
        divergences.append(
            Divergence(
                kind="omission",
                detail=(
                    f"the deterministic narrative reports {claim.surface!r}; no corresponding "
                    "figure appears anywhere in the model narrative"
                ),
                deterministic_value=claim.value,
                context=claim.context,
                severity="medium",
            )
        )

    # 3. Addition — the model surfaced a real figure the deterministic path did
    #    not. Benign, but a reviewer should know the two paths differed in scope.
    for claim in mod_claims:
        if claim.soft:
            continue
        if claim.value in unbound_values:
            continue  # already reported above
        if any(agrees(claim.normalised_value(), d) for d in det_values):
            continue
        divergences.append(
            Divergence(
                kind="addition",
                detail=(
                    f"the model narrative introduces {claim.surface!r}, which is bound to "
                    "evidence but absent from the deterministic narrative; a scope difference, "
                    "not an invention"
                ),
                model_value=claim.value,
                context=claim.context,
                severity="low",
            )
        )

    invariant = not any(d.severity == "high" for d in divergences)

    return InvarianceAttestation(
        section=section,
        invariant=invariant,
        divergences=tuple(divergences),
        deterministic_binding=det_binding.as_dict(),
        model_binding=mod_binding.as_dict(),
        tolerance=tolerance,
        provider_name=provider_name,
        narration_path=narration_path,
        requested_provider=requested_provider,
        requested_model=requested_model,
        fallback_reason=fallback_reason,
        fallback_detail=fallback_detail,
        fallback_at=fallback_at,
        metadata=dict(metadata or {}),
    )
