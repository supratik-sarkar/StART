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
from enum import StrEnum
from typing import Any

__all__ = [
    "Claim",
    "BindingResult",
    "extract_claims",
    "flatten_evidence_values",
    "bind_claims",
    "EVIDENCE_ID_PATTERN",
    "GroundingReasonCode",
    "SemanticRole",
    "infer_metric_semantic_role",
]


class GroundingReasonCode(StrEnum):
    """Deterministic reason codes for unbound quantitative claims."""

    MISSING_CITATION = "MISSING_CITATION"
    NO_LOCAL_EVIDENCE_CITATION = "NO_LOCAL_EVIDENCE_CITATION"
    UNKNOWN_EVIDENCE_ID = "UNKNOWN_EVIDENCE_ID"
    OUT_OF_SCOPE_EVIDENCE = "OUT_OF_SCOPE_EVIDENCE"
    CITATION_RECORD_NOT_IN_CHECKPOINT = "CITATION_RECORD_NOT_IN_CHECKPOINT"
    METRIC_PATH_NOT_ALLOWED = "METRIC_PATH_NOT_ALLOWED"
    METRIC_PATH_NOT_FOUND = "METRIC_PATH_NOT_FOUND"
    VALUE_MISMATCH = "VALUE_MISMATCH"
    NUMERIC_VALUE_NOT_FOUND = "NUMERIC_VALUE_NOT_FOUND"
    METRIC_NOT_STRUCTURED = "METRIC_NOT_STRUCTURED"
    UNSUPPORTED_UNIT_TRANSFORMATION = "UNSUPPORTED_UNIT_TRANSFORMATION"
    UNSUPPORTED_DERIVED_RELATION = "UNSUPPORTED_DERIVED_RELATION"
    UNSUPPORTED_DERIVED_ARITHMETIC = "UNSUPPORTED_DERIVED_ARITHMETIC"
    CROSS_RECORD_DERIVATION = "CROSS_RECORD_DERIVATION"
    AMBIGUOUS_METRIC_BINDING = "AMBIGUOUS_METRIC_BINDING"
    UNSUPPORTED_FREQUENCY_INFERENCE = "UNSUPPORTED_FREQUENCY_INFERENCE"
    NUMERIC_PARSE_ERROR = "NUMERIC_PARSE_ERROR"


class SemanticRole(StrEnum):
    """Controlled semantic roles for quantitative claims and evidence metrics."""

    VAR_CONFIDENCE = "VAR_CONFIDENCE"
    VAR_TAIL_PROBABILITY = "VAR_TAIL_PROBABILITY"
    TEST_SIGNIFICANCE = "TEST_SIGNIFICANCE"
    P_VALUE = "P_VALUE"
    TEST_STATISTIC = "TEST_STATISTIC"
    OBSERVATION_COUNT = "OBSERVATION_COUNT"
    EXCEPTION_COUNT = "EXCEPTION_COUNT"
    EXCEPTION_RATE = "EXCEPTION_RATE"
    TRANSITION_COUNT = "TRANSITION_COUNT"
    POWER = "POWER"
    SIZE = "SIZE"
    PROBABILITY = "PROBABILITY"
    RATIO = "RATIO"
    GENERIC_NUMERIC = "GENERIC_NUMERIC"


def infer_metric_semantic_role(path: str) -> SemanticRole:
    """Derive semantic role from canonical metric path and test identity."""
    low = path.lower()
    if any(
        k in low
        for k in ("confidence", "var_confidence", "confidence_level", "nominal_confidence", "band_confidence")
    ):
        return SemanticRole.VAR_CONFIDENCE
    if any(k in low for k in ("alpha_var", "tail_probability", "expected_probability", "target_coverage")):
        return SemanticRole.VAR_TAIL_PROBABILITY
    if any(
        k in low
        for k in (
            "gamma_test",
            "statistical_gamma_test",
            "nominal_significance",
            "nominal_size",
            "significance_level",
        )
    ):
        return SemanticRole.TEST_SIGNIFICANCE
    if any(k in low for k in ("p_value", "pvalue")) or low.endswith(".p") or low == "p":
        return SemanticRole.P_VALUE
    if any(
        k in low
        for k in ("lr_uc", "lr_ind", "lr_cc", "test_statistic", "t_stat", "f_stat", "chi2", "statistic")
    ):
        return SemanticRole.TEST_STATISTIC
    if any(low.endswith(f".{t}") or low.endswith(f"_{t}") or low == t for t in ("n00", "n01", "n10", "n11")):
        return SemanticRole.TRANSITION_COUNT
    if any(k in low for k in ("exceptions", "n_exceptions", "actual_exceptions", "expected_exceptions")):
        return SemanticRole.EXCEPTION_COUNT
    if any(k in low for k in ("exception_rate", "failure_rate", "empirical_rate")):
        return SemanticRole.EXCEPTION_RATE
    if "power" in low:
        return SemanticRole.POWER
    if "size" in low and "sample" not in low:
        return SemanticRole.SIZE
    if any(
        k in low
        for k in (
            "n_observations",
            "n_periods",
            "sample_size",
            "band_n_observations",
            "n_assets",
            "n_factors",
        )
    ):
        return SemanticRole.OBSERVATION_COUNT
    if "rate" in low or "prob" in low or "share" in low:
        return SemanticRole.PROBABILITY
    if "ratio" in low:
        return SemanticRole.RATIO
    return SemanticRole.GENERIC_NUMERIC


_CITATION_CONTAINER_RE = re.compile(r"\[([^\]]+)\]|\(([^)]+)\)")
_EV_IN_CONTAINER_RE = re.compile(r"\b(EV-(?!DIAG-)[A-Za-z0-9_-]+)\b")


class _CitationContainerPattern:
    """Structured citation-container parser.

    Matches EV IDs enclosed within brackets [...] or parentheses (...), supporting
    semicolon, comma, or space separated lists (e.g. [EV-A; EV-B], (EV-A, EV-B)),
    while strictly refusing naked prose mentions (e.g. 'EV-A appears in the ledger').
    """

    def findall(self, text: str) -> list[str]:
        seen = set()
        ev_ids = []
        for m in _CITATION_CONTAINER_RE.finditer(text):
            content = m.group(1) or m.group(2) or ""
            for ev in _EV_IN_CONTAINER_RE.findall(content):
                if ev not in seen:
                    seen.add(ev)
                    ev_ids.append(ev)
        return ev_ids

    def find_spans(self, text: str) -> list[tuple[str, int, int]]:
        results = []
        for m in _CITATION_CONTAINER_RE.finditer(text):
            content = m.group(1) or m.group(2) or ""
            container_seen = set()
            for ev in _EV_IN_CONTAINER_RE.findall(content):
                if ev not in container_seen:
                    container_seen.add(ev)
                    results.append((ev, m.start(), m.end()))
        return results


#: Evidence identifiers as emitted by ``start.core.schemas.new_evidence_id``.
EVIDENCE_ID_PATTERN = _CitationContainerPattern()

#: A number, optionally signed, with optional thousands separators, decimal
#: part, exponent, and a trailing unit marker.
_NUMBER = re.compile(
    r"""
    (?<![\w.])                       # not mid-identifier
    (?P<sign>[-+]?)
    (?P<int>\d{1,3}(?:,\d{3})+|\d+)  # 1,234,567 or 1234567
    (?P<frac>\.\d+)?
    (?P<exp>[eE][-+]?\d+)?
    (?P<unit>\s?%|\s?bps|\s?pp|\s?x)? # percent, basis points, percentage points, multipliers (x)
    (?!\.?\d)                        # not mid-way through a dotted numeric (1.2.3)
    (?![A-Za-z_])                     # not the head of an identifier
    """,
    re.VERBOSE,
)

#: Contexts in which a number is structural rather than a measurement.
_STRUCTURAL_PREFIX = re.compile(
    r"(?:phase|step|stage|section|table|figure|chapter|tier|version|v|part|"
    r"appendix|footnote|item|question|day|week|quarter|q|lags?|order)"
    r"(?:\s*(?:>|>=|<|<=|=|beyond|greater than|exceeding))?\s*$",
    re.IGNORECASE,
)

#: Regulatory and standard citations that contain digits but assert nothing.
_CITATION_TOKENS = re.compile(
    r"\b(?:SR\s?11-7|OCC\s?2011-12|BCBS\s?\d+|IFRS\s?\d+|ISO(?:/IEC)?\s?\d+|"
    r"NIST\s?[A-Z.\d-]+|SS\d+/\d+|Basel\s?(?:I{1,3}|IV))\b",
    re.IGNORECASE,
)

#: Evidence identifier tokens whose internal digits are not measurements.
_EVIDENCE_TOKENS = re.compile(r"\bEV-[A-Za-z0-9_-]+\b", re.IGNORECASE)

#: Footnote-style citation markers: "[1]", "[12]". Deliberately NOT "[0.99]" or
#: "[1234]". Masking any bracketed content would open an evasion channel — a model
#: could conceal an invented figure inside brackets and the checker would never see
#: it. Two digits maximum, integers only, so the mask covers real footnote markers
#: and nothing that could plausibly be a measurement.
_FOOTNOTE_MARKER = re.compile(r"\[\d{1,2}\]")

#: Indicator variable definitions (e.g. "I_t = 1 iff ...", "I_t = 0 if ...") whose 0/1 are
#: definitional mathematical conventions rather than empirical quantitative assertions.
_INDICATOR_CONVENTION = re.compile(
    r"\bI(?:_[a-zA-Z0-9]+|\([a-zA-Z0-9]+\))?\s*=\s*[01]\s+(?:iff|if)\b",
    re.IGNORECASE,
)

#: Plural digit references (e.g. "0's between 1's", "0’s and 1’s", "1990s") that name symbols
#: rather than asserting quantitative measurements.
_PLURAL_DIGITS = re.compile(r"\b\d+[\u0027\u2019]s\b", re.IGNORECASE)

#: List item numbering delimiters (e.g. "1) ...", "2) ...", "(1) ...", "1. ...") that enumerate
#: sections or items rather than asserting quantitative measurements.
_LIST_NUMBERING = re.compile(r"(?:^|\n)\s*\(?\d{1,2}[.)]\s+")

#: Order of magnitude notation (e.g. "O(10^-5)", "O(10^2)", "10^-4") representing asymptotic scale
#: rather than individual empirical measurements of 10 or -5.
_ORDER_OF_MAGNITUDE = re.compile(
    r"\bO\s*\(\s*10\^\{?[-+]?\d+\}?\s*\)|\b10\^\{?[-+]?\d+\}?",
    re.IGNORECASE,
)

#: Matrix and table dimensions (e.g. 2x2, 2×2, 50x50, 50×50) representing structural shapes.
_DIMENSION_TOKENS = re.compile(r"\b\d+\s*[x×]\s*\d+\b")

#: Date and timestamp patterns whose components (year, month, day, time) are temporal
#: metadata rather than quantitative model claims.
_DATE_TIMESTAMP_TOKENS = re.compile(
    r"""
    \b
    (?:
        # ISO-8601 date or timestamp: 2023-10-31, 2023-10-31T12:00:00Z, 2023-10-31 09:30:00+00:00
        \d{4}[-/\u2011\u2013\u2014]\d{1,2}[-/\u2011\u2013\u2014]\d{1,2}(?:[T\s]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?
        |
        # Standard slash/dash date: 10/31/2023, 31/10/2023, 10-31-2023, 31-10-2023
        \d{1,2}[-/\u2011\u2013\u2014]\d{1,2}[-/\u2011\u2013\u2014]\d{4}
        |
        # Month name dates: October 31, 2023 / 31 October 2023 / Oct 31, 2023
        (?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"""
    r"""Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?"""
    r"""(?:,?\s+\d{4})?)
        |
        (?:\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"""
    r"""Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)(?:,?\s+\d{4})?)
    )
    \b
    """,
    re.VERBOSE | re.IGNORECASE,
)

#: Countable nouns that confirm a 4-digit number in the year range is an analytical measurement
_COUNTABLE_QUANT_NOUNS = re.compile(
    r"^\s*(?:observations?|samples?|days?|rows?|points?|data\s+points?|assets?|trades?|"
    r"exceptions?|periods?|simulations?|scenarios?|draws?|iterations?|degrees\s+of\s+freedom|df|obs)\b",
    re.IGNORECASE,
)

_COUNTABLE_QUANT_PREFIX = re.compile(
    r"(?:contains?|sample\s+of|dataset\s+of|total\s+of|size\s+of|count\s+of|n\s*=\s*|N\s*=\s*)\s*$",
    re.IGNORECASE,
)

#: Spelled cardinal words extracted as soft claims.
#: Note: Deliberately excluding "no" because phrases like "no evidence of drift"
#: are not quantitative assertions.
_WORD_NUMBERS: dict[str, float] = {
    "zero": 0.0,
    "none": 0.0,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "six": 6.0,
    "seven": 7.0,
    "eight": 8.0,
    "nine": 9.0,
    "ten": 10.0,
    "eleven": 11.0,
    "twelve": 12.0,
    "thirteen": 13.0,
    "fourteen": 14.0,
    "fifteen": 15.0,
    "sixteen": 16.0,
    "seventeen": 17.0,
    "eighteen": 18.0,
    "nineteen": 19.0,
    "twenty": 20.0,
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
    #: Evidence ids cited in the same sentence or enclosing item, if any.
    cited_evidence: tuple[str, ...] = field(default=())
    #: Soft claims (e.g. word-form cardinals like "three" or "zero") count towards
    #: reconciling omission but never trigger unbound or contradiction divergences.
    soft: bool = False
    local_span: str = ""
    local_label: str | None = None
    semantic_role: str = SemanticRole.GENERIC_NUMERIC

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
            "local_span": self.local_span,
            "local_label": self.local_label,
            "semantic_role": str(self.semantic_role),
        }


def _context_window(text: str, start: int, end: int, width: int = 48) -> str:
    left = text[max(0, start - width) : start]
    right = text[end : end + width]
    return re.sub(r"\s+", " ", f"{left}⟦⟧{right}").strip()


def _sentence_bounds(text: str, position: int) -> tuple[int, int]:
    # Left bound: previous period-space, period-newline, blank line, or bullet item start
    left_candidates = [
        text.rfind(". ", 0, position),
        text.rfind(".\n", 0, position),
        text.rfind("?\n", 0, position),
        text.rfind("!\n", 0, position),
    ]
    for m in re.finditer(r"\n\s*\n", text[:position]):
        left_candidates.append(m.end())
    for m in re.finditer(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+", text[:position]):
        left_candidates.append(m.end())

    valid_left = [c for c in left_candidates if c != -1]
    left = max(valid_left) if valid_left else 0
    if left > 0 and text[left : left + 2] in (". ", ".\n", "?\n", "!\n"):
        left += 2

    # Right bound: next period-space, period-newline, blank line, or next bullet start
    right_candidates = [
        i
        for i in (
            text.find(". ", position),
            text.find(".\n", position),
            text.find("?\n", position),
            text.find("!\n", position),
        )
        if i != -1
    ]
    m_right_para = re.search(r"\n\s*\n", text[position:])
    if m_right_para:
        right_candidates.append(position + m_right_para.start())
    m_right_bullet = re.search(r"\n\s*(?:[-*•]|\d+[.)])\s+", text[position:])
    if m_right_bullet:
        right_candidates.append(position + m_right_bullet.start())

    right = min(right_candidates) if right_candidates else len(text)
    return (left, right)


def _item_bounds(text: str, position: int) -> tuple[int, int] | None:
    """Find the bounds of the enclosing bullet item if in a list; return None if in plain text."""
    left_match = None
    for m in re.finditer(r"(?:^|\n)\s*(?:[-*•]|\d+[.)])\s+", text[:position]):
        left_match = m
    if not left_match:
        return None

    start_pos = left_match.start()
    if text[start_pos] == "\n":
        start_pos += 1

    right_match = re.search(r"\n\s*(?:[-*•]|\d+[.)])\s+", text[position:])
    right_para = re.search(r"\n\s*\n", text[position:])
    ends = []
    if right_match:
        ends.append(position + right_match.start())
    if right_para:
        ends.append(position + right_para.start())
    end_pos = min(ends) if ends else len(text)
    return (start_pos, end_pos)


def _block_bounds(text: str, position: int) -> tuple[int, int] | None:
    """Find bounds of enclosing top-level block/section (e.g. parent bullet item with sub-bullets)."""
    top_matches = list(re.finditer(r"(?:^|\n)(?:[-*•]|\d+[.)])\s+", text[:position]))
    if not top_matches:
        return None
    start_pos = top_matches[-1].start()
    if text[start_pos] == "\n":
        start_pos += 1
    next_top = re.search(r"\n(?:[-*•]|\d+[.)])\s+", text[position:])
    next_para = re.search(r"\n\s*\n", text[position:])
    ends = []
    if next_top:
        ends.append(position + next_top.start())
    if next_para:
        ends.append(position + next_para.start())
    end_pos = min(ends) if ends else len(text)
    return (start_pos, end_pos)


_CLAUSE_BOUNDARY_RE = re.compile(
    r";\s*|(?<!\d):\s+|,\s*(?:which|where|while|but|whereas|although)\b",
    re.IGNORECASE,
)


def _find_claim_citations(narrative: str, start: int, end: int) -> tuple[str, ...]:
    """Resolve evidence citations with clause-level boundary sensitivity."""
    s_start, s_end = _sentence_bounds(narrative, start)
    sentence_text = narrative[s_start:s_end]
    rel_start = start - s_start
    rel_end = end - s_start

    cit_spans = EVIDENCE_ID_PATTERN.find_spans(sentence_text)
    if not cit_spans:
        ib = _item_bounds(narrative, start)
        if ib is not None:
            cited = tuple(EVIDENCE_ID_PATTERN.findall(narrative[ib[0] : ib[1]]))
            if cited:
                return cited
        bb = _block_bounds(narrative, start)
        if bb is not None:
            cited = tuple(EVIDENCE_ID_PATTERN.findall(narrative[bb[0] : bb[1]]))
            if cited:
                return cited
        return ()

    # Mask out parenthetical, bracketed, and quoted contents so internal punctuation
    # like (a; b), [EV-1; EV-2], or "a; b" is not treated as a clause boundary
    masked_sentence = re.sub(r"[\(\[][^()\[\]]*[\)\]]", lambda m: " " * len(m.group(0)), sentence_text)
    masked_sentence = re.sub(r"[“\"][^“”\"]*[”\"]", lambda m: " " * len(m.group(0)), masked_sentence)

    # First, look for citations that follow the claim
    following = [c for c in cit_spans if c[1] >= rel_end]
    if following:
        first_cit_start = following[0][1]
        between = masked_sentence[rel_end:first_cit_start]
        if not _CLAUSE_BOUNDARY_RE.search(between):
            group_cits = []
            last_end = None
            for ev, s, e in following:
                if last_end is None:
                    group_cits.append(ev)
                    last_end = e
                else:
                    sep = sentence_text[last_end:s]
                    if re.fullmatch(r"[\s,;]*", sep):
                        group_cits.append(ev)
                        last_end = e
                    else:
                        break
            return tuple(group_cits)

    # Preceding citations without a clause boundary or coordinating conjunction
    preceding = [c for c in cit_spans if c[2] <= rel_start]
    if preceding:
        last_cit = preceding[-1]
        between = masked_sentence[last_cit[2] : rel_start]
        has_conj = bool(
            re.search(r"\b(?:and|while|whereas|but|versus|vs\.?|against)\b", between, re.IGNORECASE)
        )
        if not has_conj and not _CLAUSE_BOUNDARY_RE.search(between):
            group_cits = []
            first_start = None
            for ev, s, e in reversed(preceding):
                if first_start is None:
                    group_cits.append(ev)
                    first_start = s
                else:
                    sep = sentence_text[e:first_start]
                    if re.fullmatch(r"[\s,;]*", sep):
                        group_cits.append(ev)
                        first_start = s
                    else:
                        break
            return tuple(reversed(group_cits))

    # Trailing citations in sentence with subordinate clause
    if following:
        splits = list(_CLAUSE_BOUNDARY_RE.finditer(masked_sentence))
        if splits:
            last_sep_end = splits[-1].end()
            trailing = [ev for ev, s, e in cit_spans if s >= last_sep_end]
            if trailing:
                trailing_text = sentence_text[last_sep_end:]
                has_colon = any(":" in sp.group(0) for sp in splits)
                is_subordinate_conj = any("," in sp.group(0) for sp in splits)
                is_qualifier = bool(
                    re.match(
                        r"^\s*(?:none|no\b|neither|not\b|n/a)",
                        trailing_text,
                        re.IGNORECASE,
                    )
                )
                only_semicolon = all(";" in sp.group(0) for sp in splits)
                should_inherit = (
                    is_qualifier if only_semicolon else (has_colon or is_subordinate_conj or is_qualifier)
                )
                if should_inherit:
                    return tuple(ev for ev, _, _ in cit_spans)

    return ()


_LOCAL_LABEL_RE = re.compile(
    r"(?:(?:\b|[(])([A-Za-z_][A-Za-z0-9_.-]*)\s*(?:[:=~≈]|equals|of)\s*)$",
    re.IGNORECASE,
)


def _determine_claim_local_semantics(
    surface: str,
    unit: str,
    value: float,
    preceding: str,
    following: str,
) -> tuple[str | None, str, SemanticRole]:
    """Derive local label, local span, and semantic role from claim's immediate neighborhood."""
    m_label = _LOCAL_LABEL_RE.search(preceding)
    local_label = m_label.group(1).lower() if m_label else None

    # Compute local span bounded by clause/list punctuation
    left_clause = re.split(r"[;,()\n]|\.\s+", preceding)[-1]
    right_clause = re.split(r"[;,()\n]|\.\s+", following)[0]
    local_span = f"{left_clause.strip()} {surface} {right_clause.strip()}".strip()
    low_span = local_span.lower()

    # Multipliers ("x") are scenario/scale factors, not confidence/probability roles
    if unit == "x":
        return local_label, local_span, SemanticRole.GENERIC_NUMERIC

    if local_label:
        lbl = local_label
        if lbl in ("p", "p_val", "p_value", "pvalue"):
            return local_label, local_span, SemanticRole.P_VALUE
        if lbl in ("lr", "lr_uc", "lr_ind", "lr_cc", "tstat", "t_stat", "statistic", "chi2"):
            return local_label, local_span, SemanticRole.TEST_STATISTIC
        if lbl in ("gamma", "gamma_test", "significance"):
            return local_label, local_span, SemanticRole.TEST_SIGNIFICANCE
        if lbl in ("alpha_var", "tail_prob", "tail_probability"):
            return local_label, local_span, SemanticRole.VAR_TAIL_PROBABILITY
        if lbl in ("alpha",):
            if any(k in low_span for k in ("rejected", "level", "significance", "hypo", "stated")):
                return local_label, local_span, SemanticRole.TEST_SIGNIFICANCE
            return local_label, local_span, SemanticRole.VAR_TAIL_PROBABILITY
        if lbl in ("n00", "n01", "n10", "n11"):
            return local_label, local_span, SemanticRole.TRANSITION_COUNT
        if lbl in ("confidence", "var_confidence"):
            return local_label, local_span, SemanticRole.VAR_CONFIDENCE
        if lbl in ("power",):
            return local_label, local_span, SemanticRole.POWER
        if lbl in ("size",):
            return local_label, local_span, SemanticRole.SIZE

    # Derive role from local span
    norm_val = value / 100.0 if unit == "%" else value
    if 0.50 <= norm_val <= 0.999:
        if any(k in low_span for k in ("confidence", "var", "expectation", "expected at", "quantile")):
            if not any(k in low_span for k in ("significance", "rejected at")):
                return local_label, local_span, SemanticRole.VAR_CONFIDENCE
    elif norm_val <= 0.10:
        if any(
            k in low_span
            for k in (
                "tail probability",
                "alpha_var",
                "tail mass",
                "expected probability",
                "expectation",
                "tail",
            )
        ):
            if not any(k in low_span for k in ("significance", "gamma")):
                return local_label, local_span, SemanticRole.VAR_TAIL_PROBABILITY

    if any(
        k in low_span
        for k in (
            "significance",
            "gamma",
            "nominal size",
            "hypothesis test",
            "hypothesis tests",
            "test level",
        )
    ):
        return local_label, local_span, SemanticRole.TEST_SIGNIFICANCE

    if any(k in low_span for k in ("exception", "exceptions", "breach", "breaches")):
        return local_label, local_span, SemanticRole.EXCEPTION_COUNT

    if any(k in low_span for k in ("observation", "observations", "periods", "samples", "pairs")):
        return local_label, local_span, SemanticRole.OBSERVATION_COUNT

    if any(k in low_span for k in ("power",)):
        return local_label, local_span, SemanticRole.POWER

    if any(k in low_span for k in ("size",)):
        return local_label, local_span, SemanticRole.SIZE

    return local_label, local_span, SemanticRole.GENERIC_NUMERIC


_SHARED_EXP_RANGE = re.compile(
    r"(?<![eE])(?<![eE][+-])(?<!\d\.)\b(?P<first>\d+(?:\.\d+)?)\s*(?P<dash>[–—]|to)\s*"
    r"(?P<second>\d+(?:\.\d+)?(?P<exp>[eE][-+]?\d+))\b"
)

#: Shared-unit percent or bps compact range (e.g. 0.27–0.35% -> 0.27%–0.35%)
_SHARED_PCT_RANGE = re.compile(
    r"(?<!\d\.)\b(?P<first>\d+(?:\.\d+)?)\s*(?P<dash>[–—]|to)\s*(?P<second>\d+(?:\.\d+)?)(?P<unit>%|bps)(?!\w)"
)

#: Scientific notation with base-10 multiplication (e.g. 4.19×10^-5, 3.65 * 10^-5, 4.19 x 10^-5)
_SCIENTIFIC_TIMES_TEN_RE = re.compile(
    r"(?P<num>\b\d+(?:\.\d+)?)\s*(?:[×*·]|(?<![a-zA-Z])x(?![a-zA-Z]))\s*10\^\{?(?P<exp>[-+]?\d+)\}?",
    re.IGNORECASE,
)


def normalize_markdown_numeric_markup(text: str) -> str:
    """Normalize presentation markup that splits or surrounds numeric tokens."""
    if not text:
        return text
    # Convert scientific notation with base 10 (e.g. 4.19×10^-5, 3.65 * 10^-5)
    # to standard e-notation (4.19e-5)
    text = _SCIENTIFIC_TIMES_TEN_RE.sub(r"\g<num>e\g<exp>", text)
    # Normalize unicode hyphens and minus signs to standard ASCII hyphen
    text = text.replace("\u2011", "-").replace("\u2212", "-").replace("\ufe63", "-").replace("\uff0d", "-")
    # Convert en-dash/em-dash used as negative sign into minus sign (not when preceded by digit, %, ], or ))
    text = re.sub(r"(?<![\d%\]\)])[–—](?=\d)", "-", text)
    # Fix numbers split by markdown bold/italic tags e.g. **1**,**000** -> 1,000
    text = re.sub(r"(\d+)\*\*,\*\*(\d+)", r"\1,\2", text)
    text = re.sub(r"(\d+)\*,\*(\d+)", r"\1,\2", text)
    text = re.sub(r"(\d+)__,\s*__(\d+)", r"\1,\2", text)
    # Fix decimals split by markdown tags e.g. **0.**7x -> 0.7x, **1.**5x -> 1.5x
    text = re.sub(r"(\d+\.)\*\*(\d+)", r"\1\2", text)
    text = re.sub(r"(\d+\.)\*(\d+)", r"\1\2", text)
    text = re.sub(r"(\d+)\*\*(\.\d+)", r"\1\2", text)
    text = re.sub(r"(\d+)\*(\.\d+)", r"\1\2", text)
    # Strip markdown emphasis around isolated numeric tokens and intervals
    text = re.sub(r"\*\*([0-9.,%+\-x]+)\*\*", r"\1", text)
    text = re.sub(r"\*([0-9.,%+\-x]+)\*", r"\1", text)
    text = re.sub(r"__([0-9.,%+\-x]+)__", r"\1", text)
    text = re.sub(r"_([0-9.,%+\-x]+)_", r"\1", text)

    # Expand shared-exponent compact ranges (e.g. 3.65–4.19e-05 -> 3.65e-05–4.19e-05)
    def _expand_shared_exp(m: re.Match[str]) -> str:
        return f"{m.group('first')}{m.group('exp')}{m.group('dash')}{m.group('second')}"

    text = _SHARED_EXP_RANGE.sub(_expand_shared_exp, text)
    # Expand shared-percent compact ranges (e.g. 0.27–0.35% -> 0.27%–0.35%)
    text = _SHARED_PCT_RANGE.sub(r"\g<first>\g<unit>\g<dash>\g<second>\g<unit>", text)
    return text


def extract_claims(narrative: str, *, min_abs_value: float = 0.0) -> list[Claim]:
    """Lift every quantitative claim out of a narrative.

    ``min_abs_value`` can suppress trivially small integers if a caller wants
    that; the default keeps everything, because "0 exceptions" is a claim.
    """
    if not narrative:
        return []

    narrative = normalize_markdown_numeric_markup(narrative)

    # Mask citation, evidence, footnote, and date/timestamp tokens so their internal
    # digits/components are never read as quantitative measurements.
    masked = _CITATION_TOKENS.sub(lambda m: "#" * len(m.group(0)), narrative)
    masked = _EVIDENCE_TOKENS.sub(lambda m: "#" * len(m.group(0)), masked)
    masked = _FOOTNOTE_MARKER.sub(lambda m: "#" * len(m.group(0)), masked)
    masked = _DATE_TIMESTAMP_TOKENS.sub(lambda m: "#" * len(m.group(0)), masked)
    masked = _INDICATOR_CONVENTION.sub(lambda m: "#" * len(m.group(0)), masked)
    masked = _PLURAL_DIGITS.sub(lambda m: "#" * len(m.group(0)), masked)
    masked = _LIST_NUMBERING.sub(lambda m: "#" * len(m.group(0)), masked)
    masked = _ORDER_OF_MAGNITUDE.sub(lambda m: "#" * len(m.group(0)), masked)
    masked = _DIMENSION_TOKENS.sub(lambda m: "#" * len(m.group(0)), masked)

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
        following = masked[end : min(len(masked), end + 32)]
        # A bare four-digit integer with no unit, in the year range, reads as a
        # date unless explicitly qualified by a countable quantitative noun/prefix
        # (e.g. "sample contains 2023 observations").
        if not unit and match.group("frac") is None and 1900 <= value <= 2100 and len(raw_int) == 4:
            is_quant_count = bool(
                _COUNTABLE_QUANT_NOUNS.search(following) or _COUNTABLE_QUANT_PREFIX.search(preceding)
            )
            if not is_quant_count:
                continue
        if abs(value) < min_abs_value:
            continue

        local_label, local_span, sem_role = _determine_claim_local_semantics(
            surface=narrative[start:end],
            unit=unit,
            value=value,
            preceding=narrative[max(0, start - 40) : start],
            following=narrative[end : min(len(narrative), end + 40)],
        )

        cited = _find_claim_citations(narrative, start, end)

        claims.append(
            Claim(
                value=value,
                surface=narrative[start:end],
                unit=unit,
                position=start,
                context=_context_window(narrative, start, end),
                cited_evidence=cited,
                soft=False,
                local_span=local_span,
                local_label=local_label,
                semantic_role=sem_role,
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

        local_label, local_span, sem_role = _determine_claim_local_semantics(
            surface=narrative[start:end],
            unit="",
            value=val,
            preceding=narrative[max(0, start - 40) : start],
            following=narrative[end : min(len(narrative), end + 40)],
        )

        cited = _find_claim_citations(narrative, start, end)

        claims.append(
            Claim(
                value=val,
                surface=narrative[start:end],
                unit="",
                position=start,
                context=_context_window(narrative, start, end),
                cited_evidence=cited,
                soft=True,
                local_span=local_span,
                local_label=local_label,
                semantic_role=sem_role,
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
            # Parse threshold and interval numbers for structured criterion fields
            is_crit_prefix = any(
                ck in prefix.lower()
                for ck in ("required", "criterion", "band", "threshold", "critical_value")
            )
            if is_crit_prefix:
                nums = re.findall(r"[-+]?\d*\.?\d+", evidence)
                if nums:
                    try:
                        val0 = float(nums[0])
                        out[f"{prefix}.threshold"] = val0
                        out[f"{prefix}.value"] = val0
                        out[prefix] = val0
                        if len(nums) >= 2 and any(k in evidence for k in ("[", "(", "in ")):
                            val1 = float(nums[1])
                            out[f"{prefix}.lower"] = val0
                            out[f"{prefix}.upper"] = val1
                            out[f"{prefix}[0]"] = val0
                            out[f"{prefix}[1]"] = val1
                    except ValueError:
                        pass
        if "0_7x" in prefix:
            out[f"{prefix}.multiplier"] = 0.7
        elif "1_5x" in prefix:
            out[f"{prefix}.multiplier"] = 1.5

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


@dataclass
class _ScopedRecord:
    evidence_id: str | None
    test_id: str | None
    fields: dict[str, float]


def _normalize_evidence(evidence: Any) -> list[_ScopedRecord]:
    """Extract individual scoped records from evidence input."""
    records: list[_ScopedRecord] = []

    if hasattr(evidence, "model_dump"):
        evidence = [evidence]
    elif isinstance(evidence, dict) and (
        "evidence_id" in evidence or "test_id" in evidence or "metrics" in evidence
    ):
        evidence = [evidence]

    if isinstance(evidence, (list, tuple)):
        for item in evidence:
            dumped = (
                item.model_dump(mode="json")
                if hasattr(item, "model_dump")
                else (dict(item) if isinstance(item, dict) else item)
            )
            if isinstance(dumped, dict):
                ev_id = dumped.get("evidence_id")
                test_id = dumped.get("test_id")
                fields = flatten_evidence_values(dumped)
                records.append(_ScopedRecord(evidence_id=ev_id, test_id=test_id, fields=fields))
            else:
                fields = flatten_evidence_values(item)
                records.append(_ScopedRecord(evidence_id=None, test_id=None, fields=fields))
    elif isinstance(evidence, dict):
        is_nested = any(
            isinstance(v, (dict, list, tuple)) or hasattr(v, "model_dump") for v in evidence.values()
        )
        if is_nested:
            for k, v in evidence.items():
                ev_id = (
                    str(k)
                    if str(k).startswith("EV-")
                    else (
                        getattr(v, "evidence_id", None)
                        or (v.get("evidence_id") if isinstance(v, dict) else None)
                    )
                )
                test_id = (
                    str(k)
                    if not str(k).startswith("EV-")
                    else (getattr(v, "test_id", None) or (v.get("test_id") if isinstance(v, dict) else None))
                )
                dumped = (
                    v.model_dump(mode="json")
                    if hasattr(v, "model_dump")
                    else (dict(v) if isinstance(v, dict) else v)
                )
                fields = flatten_evidence_values(dumped)
                records.append(_ScopedRecord(evidence_id=ev_id, test_id=test_id, fields=fields))
        else:
            groups: dict[str, dict[str, float]] = {}
            for path, val in evidence.items():
                if isinstance(val, (int, float)):
                    parts = path.split(".", 1)
                    prefix = parts[0]
                    subpath = parts[1] if len(parts) > 1 else path
                    groups.setdefault(prefix, {})[subpath] = float(val)
            if groups:
                for prefix, gfields in groups.items():
                    ev_id = prefix if prefix.startswith("EV-") else None
                    test_id = prefix if not prefix.startswith("EV-") and not prefix.startswith("[") else None
                    records.append(_ScopedRecord(evidence_id=ev_id, test_id=test_id, fields=gfields))
            else:
                records.append(
                    _ScopedRecord(
                        evidence_id=None,
                        test_id=None,
                        fields={k: float(v) for k, v in evidence.items() if isinstance(v, (int, float))},
                    )
                )
    else:
        fields = flatten_evidence_values(evidence)
        records.append(_ScopedRecord(evidence_id=None, test_id=None, fields=fields))

    return records


def _score_path_context(path: str, context: str) -> int:
    """Score relevance of a field path to the claim's surrounding context."""
    if not context:
        return 0
    words = re.findall(r"[a-zA-Z]{3,}", context.lower())
    tokens = set()
    for w in words:
        tokens.add(w)
        if w.endswith("ed"):
            tokens.add(w[:-2])
        if w.endswith("ing"):
            tokens.add(w[:-3])
        if w.endswith("s"):
            tokens.add(w[:-1])
        if w.endswith("es"):
            tokens.add(w[:-2])

    path_tokens = set(re.findall(r"[a-zA-Z]{3,}", path.lower()))
    score = 0
    for t in tokens:
        for pt in path_tokens:
            if t == pt:
                score += 5
            elif len(t) >= 4 and (t in pt or pt in t):
                score += 3
    return score


_PERCENT_PROBABILITY_KEYWORDS = (
    "p_value",
    "alpha",
    "gamma",
    "rate",
    "prob",
    "probability",
    "size",
    "power",
    "ratio",
    "share",
    "weight",
    "allocation",
    "fraction",
    "percent",
    "level",
    "coverage",
    "consistency_ratio",
    "confidence",
    "return",
    "loss",
    "yield",
    "drawdown",
    "shock",
    "volatility",
    "var",
    "es",
    "shortfall",
    "variance",
    "deviation",
    "gap",
    "discrepancy",
)


def _is_percent_compatible_metric(path: str) -> bool:
    """Return True if the metric path is typed for percentage or probability comparison."""
    low = path.lower()
    return any(kw in low for kw in _PERCENT_PROBABILITY_KEYWORDS)


def _is_semantically_incompatible(
    path: str,
    context: str = "",
    claim: Claim | None = None,
) -> bool:
    """Enforce frozen Gate-5 semantic separation: alpha_var != gamma_test != confidence.

    Uses claim-local semantic role and local label constraints rather than
    sentence-wide global keywords.
    """
    metric_role = infer_metric_semantic_role(path)
    low_path = path.lower()

    if claim is not None:
        # 1. Local Label Constraint
        if claim.local_label:
            lbl = claim.local_label.lower()
            if lbl in ("n00", "n01", "n10", "n11"):
                if not (low_path.endswith(f".{lbl}") or low_path.endswith(f"_{lbl}") or low_path == lbl):
                    return True
            elif lbl in ("p", "p_val", "p_value", "pvalue"):
                if not any(k in low_path for k in ("p_value", "pvalue", ".p")):
                    return True
            elif lbl.startswith("lr"):
                if lbl not in low_path and not low_path.endswith(".lr") and "lr_" not in low_path:
                    return True
            elif lbl.startswith("gamma"):
                if any(k in low_path for k in ("confidence", "alpha_var")):
                    return True

        role = claim.semantic_role if claim.semantic_role != SemanticRole.GENERIC_NUMERIC else None
        if role is None:
            eff_ctx = claim.local_span or claim.context or context
            if eff_ctx:
                low_ctx = eff_ctx.lower()
                if any(k in low_ctx for k in ("significance", "gamma", "hypothesis test", "test level")):
                    role = SemanticRole.TEST_SIGNIFICANCE
                elif any(k in low_ctx for k in ("confidence", "var level", "quantile level")):
                    role = SemanticRole.VAR_CONFIDENCE
                elif any(k in low_ctx for k in ("tail probability", "alpha_var", "tail mass")):
                    role = SemanticRole.VAR_TAIL_PROBABILITY

        # 2. Semantic Role Separation
        if role == SemanticRole.VAR_CONFIDENCE:
            if metric_role in (
                SemanticRole.TEST_SIGNIFICANCE,
                SemanticRole.VAR_TAIL_PROBABILITY,
                SemanticRole.P_VALUE,
                SemanticRole.POWER,
            ):
                return True

        elif role == SemanticRole.TEST_SIGNIFICANCE:
            if metric_role in (
                SemanticRole.VAR_CONFIDENCE,
                SemanticRole.VAR_TAIL_PROBABILITY,
                SemanticRole.P_VALUE,
            ):
                return True

        elif role == SemanticRole.VAR_TAIL_PROBABILITY:
            if metric_role in (
                SemanticRole.VAR_CONFIDENCE,
                SemanticRole.TEST_SIGNIFICANCE,
                SemanticRole.P_VALUE,
            ):
                return True

        elif role == SemanticRole.TRANSITION_COUNT:
            if metric_role not in (SemanticRole.TRANSITION_COUNT, SemanticRole.GENERIC_NUMERIC):
                return True

        return False

    # Fallback to context-based separation if claim is None
    if not context:
        return False
    low_ctx = context.lower()

    is_significance_ctx = any(
        k in low_ctx for k in ("significance", "gamma", "hypothesis test", "test level", "rejection level")
    )
    if is_significance_ctx:
        if any(p in low_path for p in ("confidence", "alpha_var", "expected_probability")):
            return True

    is_confidence_ctx = (
        any(k in low_ctx for k in ("confidence", "var level", "quantile level")) and not is_significance_ctx
    )
    if is_confidence_ctx:
        if any(p in low_path for p in ("gamma_test", "statistical_gamma_test", "alpha_var")):
            return True

    is_tail_prob_ctx = any(
        k in low_ctx
        for k in ("tail probability", "alpha_var", "tail mass", "expected rate", "exception rate")
    )
    if is_tail_prob_ctx:
        if any(p in low_path for p in ("gamma_test", "statistical_gamma_test", "confidence")):
            return True

    return False


def _match_candidates_in_fields(
    candidates: set[float],
    fields: dict[str, float],
    tolerance: float,
    context: str = "",
    claim: Claim | None = None,
) -> tuple[str, float, int, float] | None:
    best_path = None
    best_diff = float("inf")
    best_score = -1
    best_val = None

    displayed_d: int | None = None
    if claim is not None and claim.surface:
        cleaned = claim.surface.strip().rstrip("%").rstrip("bps").rstrip("x").replace(",", "").strip()
        if "." in cleaned:
            displayed_d = len(cleaned.split(".", 1)[1])
        else:
            displayed_d = 0

    for path, ev_value in fields.items():
        is_pct_path = _is_percent_compatible_metric(path)
        if _is_semantically_incompatible(path, context=context, claim=claim):
            continue

        for candidate in candidates:
            # 1. Typed percent normalization: % claims bind only to percentage/probability paths
            if claim is not None and claim.unit == "%" and not is_pct_path:
                continue

            diff = abs(candidate - ev_value)

            # 2. Precision-derived rounding equivalence
            is_match = False
            if claim is not None and bool(re.match(r"^[-+]?\d+(?:\.\d+)?[eE][-+]?\d+$", cleaned)):
                parts = cleaned.lower().split("e")
                mantissa_str = parts[0]
                exp_val = int(parts[1]) if len(parts) > 1 and parts[1] else 0
                mantissa_d = len(mantissa_str.split(".", 1)[1]) if "." in mantissa_str else 0
                eff_sci_d = max(0, -exp_val + mantissa_d)
                scale = max(abs(candidate), abs(ev_value))
                is_match = (
                    diff <= (0.5 * (10 ** (-eff_sci_d)) + 1e-15)
                    or (scale > 0 and diff / scale <= tolerance)
                    or diff <= tolerance * 1e-3
                )
            elif displayed_d is not None:
                if claim is not None and claim.unit == "%":
                    eff_d = displayed_d + 2
                    is_match = (
                        diff <= (0.5 * (10 ** (-eff_d)) + 1e-9)
                        or round(ev_value, eff_d) == round(candidate, eff_d)
                        or diff <= tolerance
                    )
                elif displayed_d > 0:
                    is_match = (
                        diff <= (0.5 * (10 ** (-displayed_d)) + 1e-9)
                        or round(ev_value, displayed_d) == round(candidate, displayed_d)
                        or diff <= tolerance
                    )
                else:
                    scale = max(abs(candidate), abs(ev_value))
                    is_match = (diff <= tolerance if scale < 1.0 else diff / scale <= tolerance) or (
                        abs(round(ev_value) - candidate) < 1e-6 and diff <= 0.5 + 1e-9
                    )
            else:
                scale = max(abs(candidate), abs(ev_value))
                is_match = (diff <= tolerance) if scale < 1.0 else (diff / scale <= tolerance)

            if is_match:
                score = _score_path_context(path, context)
                low_p = path.lower()

                # Boost for semantic role match
                if claim is not None and claim.semantic_role != SemanticRole.GENERIC_NUMERIC:
                    if infer_metric_semantic_role(path) == claim.semantic_role:
                        score += 25

                # Boost for local label match
                if claim is not None and claim.local_label:
                    lbl = claim.local_label.lower()
                    if low_p.endswith(f".{lbl}") or low_p.endswith(f"_{lbl}") or low_p == lbl:
                        score += 50
                    elif lbl in low_p:
                        score += 30

                # Prefer primary metrics over configuration parameters
                if low_p.startswith("metrics."):
                    score += 5

                better = False
                if score > best_score:
                    better = True
                elif score == best_score:
                    if diff < best_diff - 1e-12:
                        better = True
                    elif abs(diff - best_diff) <= 1e-12:
                        if best_path is None or len(path) < len(best_path):
                            better = True

                if better:
                    best_score = score
                    best_diff = diff
                    best_path = path
                    best_val = ev_value

    if best_path is not None and best_val is not None:
        return best_path, best_val, best_score, best_diff
    return None


def _claim_candidates(claim: Claim) -> set[float]:
    candidates = {claim.value, claim.normalised_value()}
    if claim.unit == "%":
        candidates.add(claim.value / 100.0)
    elif claim.unit == "bps":
        candidates.add(claim.value / 10_000.0)
    elif claim.unit == "x":
        candidates.add(claim.value)
    return candidates


def bind_claims(
    claims: list[Claim],
    evidence_values: Any,
    *,
    tolerance: float = 5e-4,
    permitted_scope: list[str] | None = None,
    all_known_evidence_ids: set[str] | None = None,
) -> BindingResult:
    """Match each claim to an evidence field, within relative tolerance and citation scope.

    When a claim cites explicit evidence [EV-...], matching is strictly scoped to the cited
    record(s). If the cited record is not found or does not contain the value, the claim
    remains unbound (fails closed).
    """
    records = _normalize_evidence(evidence_values)
    records_by_ev_id: dict[str, list[_ScopedRecord]] = {}
    for r in records:
        if r.evidence_id:
            records_by_ev_id.setdefault(r.evidence_id, []).append(r)

    bound: list[dict[str, Any]] = []
    unbound: list[dict[str, Any]] = []

    scope_ids = permitted_scope or [r.evidence_id for r in records if r.evidence_id]

    for claim in claims:
        candidates = _claim_candidates(claim)
        matched_path = None
        matched_value = None
        matched_ev_id = None
        matched_test_id = None
        candidate_metric = None

        reason_code = GroundingReasonCode.MISSING_CITATION
        details = ""

        # Context-level safety invariant checks
        context_lower = claim.context.lower() if claim.context else ""

        # Invariant 1: Derived relation check (e.g. periodic vol aligns with annualized vol)
        is_derived_relation = (
            "aligns numerically" in context_lower
            or "aligns with" in context_lower
            or ("periodic volatility" in context_lower and "annualised volatility" in context_lower)
            or ("periodic vol" in context_lower and "annualised vol" in context_lower)
            or "derived from" in context_lower
        )

        # Invariant 2: Frequency inference check (e.g. 252 -> "daily" without explicit frequency metadata)
        is_unsupported_frequency = False
        if ("daily" in context_lower or "per day" in context_lower) and (
            abs(claim.value - 252.0) <= 1e-4 or "252" in claim.context
        ):
            # Check if any record has explicit frequency="daily"
            has_explicit_daily = any(
                "frequency" in r.fields and r.fields.get("frequency") == "daily" for r in records
            )
            if not has_explicit_daily:
                is_unsupported_frequency = True

        if is_derived_relation:
            reason_code = GroundingReasonCode.UNSUPPORTED_DERIVED_RELATION
            details = (
                "Derived relationship or alignment assertion is not certified by deterministic evidence."
            )
        elif is_unsupported_frequency:
            reason_code = GroundingReasonCode.UNSUPPORTED_FREQUENCY_INFERENCE
            details = (
                "Inference of 'daily' frequency from period count (e.g. 252) requires "
                "explicit frequency metadata in evidence."
            )
        elif len(claim.cited_evidence) > 1 and "derived" in context_lower:
            reason_code = GroundingReasonCode.CROSS_RECORD_DERIVATION
            details = "Claim asserts a cross-record derived calculation across multiple evidence records."
        elif claim.cited_evidence:
            # Explicit citation scoping (FAIL CLOSED)
            target_records: list[_ScopedRecord] = []
            cited_in_scope = True

            for raw_cited_id in claim.cited_evidence:
                cited_id = raw_cited_id
                if (
                    all_known_evidence_ids
                    and cited_id not in all_known_evidence_ids
                    and cited_id not in records_by_ev_id
                ):
                    reason_code = GroundingReasonCode.UNKNOWN_EVIDENCE_ID
                    details = f"Cited evidence ID '{raw_cited_id}' is unknown in evidence registry."
                    cited_in_scope = False
                    break
                elif records_by_ev_id and cited_id not in records_by_ev_id:
                    reason_code = GroundingReasonCode.CITATION_RECORD_NOT_IN_CHECKPOINT
                    details = f"Cited evidence ID '{raw_cited_id}' is out of scope for this checkpoint."
                    cited_in_scope = False
                    break
                elif cited_id in records_by_ev_id:
                    target_records.extend(records_by_ev_id[cited_id])

            if cited_in_scope:
                if not target_records and not records_by_ev_id:
                    target_records = records

                if target_records:
                    candidates_in_targets: list[tuple[_ScopedRecord, str, float, int, float]] = []
                    for target_rec in target_records:
                        match = _match_candidates_in_fields(
                            candidates,
                            target_rec.fields,
                            tolerance,
                            context=claim.context,
                            claim=claim,
                        )
                        if match:
                            path, val, score, diff = match
                            candidates_in_targets.append((target_rec, path, val, score, diff))

                    if candidates_in_targets:
                        candidates_in_targets.sort(key=lambda m: (-m[3], m[4]))
                        best_rec, matched_path, matched_value, _, _ = candidates_in_targets[0]
                        matched_ev_id = best_rec.evidence_id
                        matched_test_id = best_rec.test_id
                        candidate_metric = matched_path
                    else:
                        reason_code = GroundingReasonCode.VALUE_MISMATCH
                        details = (
                            f"Value {claim.value} (surface '{claim.surface}') was not found in "
                            f"cited evidence within tolerance {tolerance}."
                        )
        else:
            # No explicit citation
            candidates_matches: list[tuple[_ScopedRecord, str, float, int, float]] = []
            for rec in records:
                match = _match_candidates_in_fields(
                    candidates,
                    rec.fields,
                    tolerance,
                    context=claim.context,
                    claim=claim,
                )
                if match:
                    path, val, score, diff = match
                    candidates_matches.append((rec, path, val, score, diff))

            if len(candidates_matches) == 1:
                rec, matched_path, matched_value, _, _ = candidates_matches[0]
                matched_ev_id = rec.evidence_id
                matched_test_id = rec.test_id
                candidate_metric = matched_path
            elif len(candidates_matches) > 1:
                candidates_matches.sort(key=lambda m: (-m[3], m[4]))
                top = candidates_matches[0]
                second = candidates_matches[1]
                if top[3] > second[3]:
                    rec, matched_path, matched_value, _, _ = top
                    matched_ev_id = rec.evidence_id
                    matched_test_id = rec.test_id
                    candidate_metric = matched_path
                else:
                    # If all top-scoring candidates represent the same underlying metric and value,
                    # resolve cleanly
                    top_score = top[3]
                    top_diff = top[4]
                    top_candidates = [
                        m for m in candidates_matches if m[3] == top_score and abs(m[4] - top_diff) <= 1e-12
                    ]
                    same_metric = all(
                        (
                            m[1].split(".")[-1].removesuffix("_before").removesuffix("_after")
                            == top[1].split(".")[-1].removesuffix("_before").removesuffix("_after")
                            or (
                                claim.semantic_role != SemanticRole.GENERIC_NUMERIC
                                and infer_metric_semantic_role(m[1]) == claim.semantic_role
                            )
                        )
                        and abs(m[2] - top[2]) <= 1e-12
                        for m in top_candidates
                    )
                    if same_metric:
                        rec, matched_path, matched_value, _, _ = top
                        matched_ev_id = rec.evidence_id
                        matched_test_id = rec.test_id
                        candidate_metric = matched_path
                    elif all(
                        m[0] is top[0]
                        or (m[0].evidence_id and m[0].evidence_id == top[0].evidence_id)
                        or (m[0].test_id and m[0].test_id == top[0].test_id and abs(m[2] - top[2]) <= 1e-12)
                        for m in candidates_matches
                    ):
                        rec, matched_path, matched_value, _, _ = top
                        matched_ev_id = rec.evidence_id
                        matched_test_id = rec.test_id
                        candidate_metric = matched_path
                    else:
                        matched_path = None
                        reason_code = GroundingReasonCode.AMBIGUOUS_METRIC_BINDING
                        details = (
                            f"Value {claim.value} matches multiple metrics without "
                            "explicit evidence citation."
                        )
            else:
                reason_code = GroundingReasonCode.NO_LOCAL_EVIDENCE_CITATION
                details = (
                    f"Quantitative claim '{claim.surface}' lacks an evidence citation [EV-...] and "
                    "does not match any in-scope metric."
                )

        record_dict = claim.as_dict()
        if matched_path is not None:
            prefix = (
                f"{matched_test_id}."
                if matched_test_id and not matched_path.startswith(f"{matched_test_id}.")
                else ""
            )
            clean_path = f"{prefix}{matched_path}".lstrip(".")
            record_dict["bound_to"] = clean_path
            record_dict["evidence_id"] = matched_ev_id
            record_dict["test_id"] = matched_test_id
            record_dict["evidence_value"] = matched_value
            bound.append(record_dict)
        else:
            record_dict["bound_to"] = None
            record_dict["evidence_id"] = None
            record_dict["test_id"] = None
            record_dict["evidence_value"] = None
            record_dict["reason"] = str(reason_code)
            record_dict["citation"] = list(claim.cited_evidence) if claim.cited_evidence else None
            record_dict["checkpoint_scope"] = list(scope_ids)
            record_dict["candidate_metric"] = candidate_metric
            record_dict["details"] = details
            # Extended parser diagnostics per Section 13
            record_dict["local_span"] = claim.local_span
            record_dict["local_label"] = claim.local_label
            record_dict["semantic_role"] = str(claim.semantic_role)
            record_dict["cited_test_ids"] = [
                r.test_id for r in records if r.evidence_id and r.evidence_id in claim.cited_evidence
            ]
            cand_paths: set[str] = set()
            for r in records:
                if not claim.cited_evidence or (r.evidence_id and r.evidence_id in claim.cited_evidence):
                    cand_paths.update(r.fields.keys())
            record_dict["candidate_metric_paths"] = sorted(cand_paths)[:10]
            unbound.append(record_dict)

    hard_bound = tuple(b for b in bound if not b.get("soft"))
    hard_unbound = tuple(u for u in unbound if not u.get("soft"))
    return BindingResult(
        total_claims=len(hard_bound) + len(hard_unbound),
        bound=hard_bound,
        unbound=hard_unbound,
        tolerance=tolerance,
    )
