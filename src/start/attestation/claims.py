"""Quantitative claim extraction and evidence binding.

The premise of every "AI writes your validation report" product is that a
language model can be trusted to describe numbers it was shown. The premise is
wrong often enough to matter, and — more importantly — it is *unfalsifiable* as
usually implemented. A reviewer reading generated prose has no mechanical way
to tell which figures came from a computation and which the model produced
because they fit the sentence.

This module makes it falsifiable. Every number in a narrative is extracted,
then matched against the evidence the narrative was supposedly derived from. A
number that matches an evidence value within tolerance is **bound**. A number
that matches nothing is **unbound**, and an unbound number in a review
narrative is a defect, not a stylistic preference.

The extractor is deliberately conservative about what counts as a claim.
Ordinal and structural numbers ("phase 2", "the third cohort", "SR 11-7",
"2024") are not quantitative claims about the artefact under review, and
treating them as such would flood the report with false positives until
everybody turned the check off — which is the usual fate of noisy controls.

Standard library only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Claim",
    "BindingResult",
    "extract_claims",
    "flatten_evidence_values",
    "bind_claims",
    "EVIDENCE_ID_PATTERN",
]

#: Evidence identifiers as emitted by ``start.core.schemas.new_evidence_id``.
EVIDENCE_ID_PATTERN = re.compile(r"\[(EV-[A-Za-z0-9]{4,})\]")

#: A number, optionally signed, with optional thousands separators, decimal
#: part, exponent, and a trailing unit marker.
_NUMBER = re.compile(
    r"""
    (?<![\w.])                       # not mid-identifier
    (?P<sign>[-+]?)
    (?P<int>\d{1,3}(?:,\d{3})+|\d+)  # 1,234,567 or 1234567
    (?P<frac>\.\d+)?
    (?P<exp>[eE][-+]?\d+)?
    (?P<unit>\s?%|\s?bps|\s?pp)?     # percent, basis points, percentage points
    (?!\.?\d)                        # not mid-way through a dotted numeric (1.2.3)
    (?![A-Za-z_])                     # not the head of an identifier
    """,
    re.VERBOSE,
)

#: Contexts in which a number is structural rather than a measurement.
_STRUCTURAL_PREFIX = re.compile(
    r"(?:phase|step|stage|section|table|figure|chapter|tier|version|v|part|"
    r"appendix|footnote|item|question|day|week|quarter|q)\s*$",
    re.IGNORECASE,
)

#: Regulatory and standard citations that contain digits but assert nothing.
_CITATION_TOKENS = re.compile(
    r"\b(?:SR\s?11-7|OCC\s?2011-12|BCBS\s?\d+|IFRS\s?\d+|ISO(?:/IEC)?\s?\d+|"
    r"NIST\s?[A-Z.\d-]+|SS\d+/\d+|Basel\s?(?:I{1,3}|IV))\b",
    re.IGNORECASE,
)

#: Evidence identifier tokens whose internal digits are not measurements.
_EVIDENCE_TOKENS = re.compile(r"\[?EV-[A-Za-z0-9_-]+\]?", re.IGNORECASE)

#: Footnote-style citation markers: "[1]", "[12]". Deliberately NOT "[0.99]" or
#: "[1234]". Masking any bracketed content would open an evasion channel — a model
#: could conceal an invented figure inside brackets and the checker would never see
#: it. Two digits maximum, integers only, so the mask covers real footnote markers
#: and nothing that could plausibly be a measurement.
_FOOTNOTE_MARKER = re.compile(r"\[\d{1,2}\]")

#: Spelled cardinal words extracted as soft claims.
#: Note: Deliberately excluding "no" because phrases like "no evidence of drift"
#: are not quantitative assertions.
_WORD_NUMBERS: dict[str, float] = {
    "zero": 0.0, "none": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0,
    "five": 5.0, "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0, "thirteen": 13.0, "fourteen": 14.0, "fifteen": 15.0,
    "sixteen": 16.0, "seventeen": 17.0, "eighteen": 18.0, "nineteen": 19.0, "twenty": 20.0,
}

_WORD_NUMBERS_PATTERN = re.compile(
    r"\b(" + "|".join(_WORD_NUMBERS.keys()) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Claim:
    """One quantitative assertion lifted out of a narrative."""

    value: float
    #: Text exactly as it appeared, e.g. "12.4%" — kept for reporting.
    surface: str
    unit: str
    #: Character offset in the source narrative.
    position: int
    #: Surrounding words, used to compare like with like across narratives.
    context: str
    #: Evidence ids cited in the same sentence, if any.
    cited_evidence: tuple[str, ...] = field(default=())
    #: Soft claims (e.g. word-form cardinals like "three" or "zero") count towards
    #: reconciling omission but never trigger unbound or contradiction divergences.
    soft: bool = False

    def normalised_value(self) -> float:
        """Percent and basis-point surfaces converted to a common scale.

        A narrative may say "12.4%" where the evidence holds 0.124. Comparing
        the surfaces directly would call that a mismatch; comparing normalised
        values recognises it as the same claim rendered differently.
        """
        if self.unit == "%":
            return self.value / 100.0
        if self.unit == "bps":
            return self.value / 10_000.0
        return self.value

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "normalised_value": self.normalised_value(),
            "surface": self.surface,
            "unit": self.unit,
            "position": self.position,
            "context": self.context,
            "cited_evidence": list(self.cited_evidence),
            "soft": self.soft,
        }


def _context_window(text: str, start: int, end: int, width: int = 48) -> str:
    left = text[max(0, start - width) : start]
    right = text[end : end + width]
    return re.sub(r"\s+", " ", f"{left}⟦⟧{right}").strip()


def _sentence_bounds(text: str, position: int) -> tuple[int, int]:
    left = max(text.rfind(". ", 0, position), text.rfind("\n", 0, position))
    right_candidates = [i for i in (text.find(". ", position), text.find("\n", position)) if i != -1]
    right = min(right_candidates) if right_candidates else len(text)
    return (left + 1 if left != -1 else 0, right)


def extract_claims(narrative: str, *, min_abs_value: float = 0.0) -> list[Claim]:
    """Lift every quantitative claim out of a narrative.

    ``min_abs_value`` can suppress trivially small integers if a caller wants
    that; the default keeps everything, because "0 exceptions" is a claim.
    """
    if not narrative:
        return []

    # Mask citation, evidence, and footnote tokens so their digits are never read as measurements.
    masked = _CITATION_TOKENS.sub(lambda m: "#" * len(m.group(0)), narrative)
    masked = _EVIDENCE_TOKENS.sub(lambda m: "#" * len(m.group(0)), masked)
    masked = _FOOTNOTE_MARKER.sub(lambda m: "#" * len(m.group(0)), masked)

    claims: list[Claim] = []
    for match in _NUMBER.finditer(masked):
        start, end = match.span()

        preceding = masked[max(0, start - 24) : start]
        if _STRUCTURAL_PREFIX.search(preceding):
            continue

        raw_int = match.group("int").replace(",", "")
        raw = f"{match.group('sign')}{raw_int}{match.group('frac') or ''}{match.group('exp') or ''}"
        try:
            value = float(raw)
        except ValueError:  # pragma: no cover - regex guarantees parseability
            continue

        unit = (match.group("unit") or "").strip()
        # A bare four-digit integer with no unit, in the year range, reads as a
        # date far more often than as a measurement.
        if not unit and match.group("frac") is None and 1900 <= value <= 2100 and len(raw_int) == 4:
            continue
        if abs(value) < min_abs_value:
            continue

        s_start, s_end = _sentence_bounds(narrative, start)
        cited = tuple(EVIDENCE_ID_PATTERN.findall(narrative[s_start:s_end]))

        claims.append(
            Claim(
                value=value,
                surface=narrative[start:end],
                unit=unit,
                position=start,
                context=_context_window(narrative, start, end),
                cited_evidence=cited,
                soft=False,
            )
        )

    # Extract word-form cardinal numbers as soft claims
    for match in _WORD_NUMBERS_PATTERN.finditer(masked):
        start, end = match.span()
        preceding = masked[max(0, start - 24) : start]
        if _STRUCTURAL_PREFIX.search(preceding):
            continue

        word = match.group(0).lower()
        val = _WORD_NUMBERS[word]

        s_start, s_end = _sentence_bounds(narrative, start)
        cited = tuple(EVIDENCE_ID_PATTERN.findall(narrative[s_start:s_end]))

        claims.append(
            Claim(
                value=val,
                surface=narrative[start:end],
                unit="",
                position=start,
                context=_context_window(narrative, start, end),
                cited_evidence=cited,
                soft=True,
            )
        )

    # Sort claims by appearance position
    claims.sort(key=lambda c: c.position)
    return claims


# --------------------------------------------------------------------------- #
# Binding claims to evidence
# --------------------------------------------------------------------------- #
def flatten_evidence_values(evidence: Any, prefix: str = "") -> dict[str, float]:
    """Collapse an evidence structure to ``field path -> numeric value``.

    Accepts nested dicts and lists, pydantic models (via ``model_dump``) and
    dataclasses, so the same function works whether the caller passes raw
    ledger JSON or live objects.
    """
    out: dict[str, float] = {}

    if hasattr(evidence, "model_dump"):
        evidence = evidence.model_dump(mode="json")
    elif hasattr(evidence, "__dataclass_fields__"):
        from dataclasses import asdict

        evidence = asdict(evidence)

    if isinstance(evidence, dict):
        for key, value in evidence.items():
            out.update(flatten_evidence_values(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(evidence, (list, tuple)):
        for index, value in enumerate(evidence):
            out.update(flatten_evidence_values(value, f"{prefix}[{index}]"))
    elif isinstance(evidence, bool):
        pass  # booleans are not measurements
    elif isinstance(evidence, (int, float)):
        out[prefix] = float(evidence)
    elif isinstance(evidence, str):
        stripped = evidence.strip().rstrip("%")
        try:
            out[prefix] = float(stripped)
        except ValueError:
            pass
    return out


@dataclass(frozen=True)
class BindingResult:
    """The outcome of checking a narrative against its evidence."""

    total_claims: int
    bound: tuple[dict[str, Any], ...]
    unbound: tuple[dict[str, Any], ...]
    tolerance: float

    @property
    def unbound_count(self) -> int:
        return len([u for u in self.unbound if not u.get("soft")])

    @property
    def grounding_rate(self) -> float:
        """Fraction of quantitative claims traceable to evidence."""
        if self.total_claims == 0:
            return 1.0
        hard_bound_count = len([b for b in self.bound if not b.get("soft")])
        return round(hard_bound_count / self.total_claims, 6)

    def as_dict(self) -> dict[str, Any]:
        hard_bound_count = len([b for b in self.bound if not b.get("soft")])
        return {
            "total_claims": self.total_claims,
            "bound_claims": hard_bound_count,
            "unbound_claims": self.unbound_count,
            "grounding_rate": self.grounding_rate,
            "tolerance": self.tolerance,
            "bound": [dict(b) for b in self.bound],
            "unbound": [dict(u) for u in self.unbound],
        }


def bind_claims(
    claims: list[Claim],
    evidence_values: dict[str, float],
    *,
    tolerance: float = 5e-4,
) -> BindingResult:
    """Match each claim to an evidence field, within relative tolerance.

    A claim binds if some evidence field holds the same number, judged on both
    the literal and the normalised reading (so "12.4%" binds to ``0.124``).
    Tolerance is relative for non-trivial magnitudes and absolute near zero,
    because a relative test around zero is meaningless.
    """
    bound: list[dict[str, Any]] = []
    unbound: list[dict[str, Any]] = []

    for claim in claims:
        candidates = {claim.value, claim.normalised_value()}
        # A narrative may also render a proportion as a percentage.
        candidates.add(claim.value * 100.0)
        matched_path = None
        matched_value = None

        for path, ev_value in evidence_values.items():
            for candidate in candidates:
                scale = max(abs(candidate), abs(ev_value))
                delta = abs(candidate - ev_value)
                if (delta <= tolerance) if scale < 1.0 else (delta / scale <= tolerance):
                    matched_path, matched_value = path, ev_value
                    break
            if matched_path:
                break

        record = claim.as_dict()
        if matched_path is not None:
            record["bound_to"] = matched_path
            record["evidence_value"] = matched_value
            bound.append(record)
        else:
            record["bound_to"] = None
            unbound.append(record)

    hard_claims_count = len([c for c in claims if not c.soft])
    return BindingResult(
        total_claims=hard_claims_count,
        bound=tuple(bound),
        unbound=tuple(unbound),
        tolerance=tolerance,
    )
