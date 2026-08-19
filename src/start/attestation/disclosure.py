"""Evidence disclosure envelopes.

Before any text is handed to an inference provider, StART builds an *envelope*:
a policy-derived projection of the evidence, hashed, from which the prompt is
assembled. The provider layer will not send a prompt that the envelope does not
cover.

The reason for the ceremony is that "we only send aggregate metrics" is, in
practice, a claim about a code path that nobody re-checks after the twentieth
refactor. Prompt assembly tends to start disciplined and drift: a debugging
change interpolates a row, a helpful improvement includes a column name, a
convenience adds a free-text field, and eighteen months later the review agent
is shipping customer data to a third party because a template grew.

An envelope converts that from a review problem into a check:

* The projection is computed from an allow-list of field paths, so a field that
  was never allowed cannot be included by an accident of templating.
* The prompt is verified against the envelope before egress: every numeric
  token in the outbound text must be present in the envelope, and any token
  that is not aborts the call.
* The envelope hash goes into the evidence chain, so what was disclosed is
  itself auditable after the fact — not merely what was concluded.

Policies tighten by trust domain. The public-demo policy is permissive because
the evidence is synthetic. An enterprise policy can be narrowed to metrics and
statuses with no identifiers at all, and — importantly — narrowing it requires
no change to any agent, prompt, or template.

Standard library only.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from start.attestation.claims import flatten_evidence_values

__all__ = [
    "DisclosurePolicy",
    "DisclosureEnvelope",
    "DisclosureViolation",
    "POLICIES",
    "policy_for",
    "build_envelope",
    "verify_prompt_covered",
]


class DisclosureViolation(RuntimeError):
    """Outbound text contains material the envelope does not cover."""


@dataclass(frozen=True)
class DisclosurePolicy:
    """What may be projected out of an evidence record, and for whom."""

    id: str
    label: str
    description: str
    #: Glob-ish field paths that may be included. ``*`` matches one segment,
    #: ``**`` matches any number of segments.
    allow: tuple[str, ...]
    #: Paths whose mere *presence* in the input is an upstream defect: raw rows,
    #: identifiers, free-text notes. Deny beats allow, and under ``strict`` a
    #: denied path aborts envelope construction rather than being dropped.
    #:
    #: Note the asymmetry with the allow-list. A field that is simply absent
    #: from ``allow`` is withheld quietly — that is the normal case, and most
    #: of a record is normally withheld. A field on ``deny`` is different: it
    #: should never have reached the disclosure layer at all, so surfacing it
    #: is more useful than filtering it.
    deny: tuple[str, ...] = field(default=())
    #: Maximum characters of free text per field. Free text is where identifiers
    #: leak, so it is truncated rather than trusted.
    max_text_chars: int = 400
    #: Refuse to build an envelope at all if a denied path is present in input.
    strict: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "description": self.description,
            "allow": list(self.allow),
            "deny": list(self.deny),
            "max_text_chars": self.max_text_chars,
            "strict": self.strict,
        }


def _compile(pattern: str) -> re.Pattern[str]:
    """Translate a field-path glob into a regex.

    ``**`` matches across dots; ``*`` matches within a single segment. Array
    indices are treated as ordinary segment content.
    """
    out: list[str] = ["^"]
    i = 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out.append(r".*")
            i += 2
        elif pattern[i] == "*":
            out.append(r"[^.]*")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    out.append("$")
    return re.compile("".join(out))


def _normalise_path(path: str) -> str:
    """Render list indices as ordinary path segments.

    ``"[0].thresholds[0].warn"`` becomes ``"0.thresholds.0.warn"``. Without
    this, a pattern like ``**.thresholds.**`` silently fails to match anything
    inside a list, because the character after ``thresholds`` is ``[`` rather
    than ``.``. Silent non-matching in an allow-list is the dangerous direction
    of failure: the field is withheld, the narrative loses a threshold it needed,
    and nothing announces why.
    """
    return path.replace("[", ".").replace("]", "").strip(".")


def _matches_any(path: str, patterns: tuple[str, ...]) -> bool:
    normalised = _normalise_path(path)
    return any(_compile(p).match(normalised) for p in patterns)


def _matches_or_descends_from(path: str, patterns: tuple[str, ...]) -> bool:
    """Match the path itself or any ancestor of it.

    Deny rules are written against the field that must not be disclosed —
    ``**.raw_rows`` — but the leaf that actually appears in a flattened
    structure is ``0.raw_rows.3.7``. Matching only the exact path would let
    every value *inside* a denied structure through while blocking only the
    empty container, which is precisely backwards.
    """
    normalised = _normalise_path(path)
    segments = normalised.split(".")
    for depth in range(1, len(segments) + 1):
        ancestor = ".".join(segments[:depth])
        if any(_compile(pattern).match(ancestor) for pattern in patterns):
            return True
    return False


# --------------------------------------------------------------------------- #
# Built-in policies
# --------------------------------------------------------------------------- #
_COMMON_DENY = (
    "**.raw_rows",
    "**.raw_data",
    "**.sample",
    "**.samples",
    "**.records",
    "**.customer**",
    "**.account**",
    "**.name",
    "**.email",
    "**.address",
    "**.identifier",
    "**.pii**",
    "**.free_text",
    "**.notes",
)

POLICIES: dict[str, DisclosurePolicy] = {
    "public_demo": DisclosurePolicy(
        id="public_demo",
        label="Public demonstration",
        description=(
            "For synthetic or published demonstration data on a personal machine. "
            "Permissive by design: the point of a demo is to show the reasoning, and the "
            "underlying data is not confidential. Still denies raw-row paths so the demo "
            "exercises the same machinery the restricted policies use."
        ),
        allow=(
            "test_id",
            "test_name",
            "status",
            "interpretation",
            "limitations**",
            "metrics.**",
            "thresholds.**",
            "params.**",
            "evidence_id",
            "model_id",
            "dataset_id",
            "run_id",
            "**.metrics.**",
            "**.status",
            "**.test_id",
            "**.test_name",
            "**.evidence_id",
            "**.interpretation",
            "**.thresholds.**",
        ),
        deny=_COMMON_DENY,
        max_text_chars=1200,
    ),
    "restricted": DisclosurePolicy(
        id="restricted",
        label="Restricted — metrics and verdicts only",
        description=(
            "For internal evidence routed through an operator-supplied gateway. Only "
            "computed metrics, thresholds and statuses are projected. Identifiers, free "
            "text and parameters are withheld, so the narrative can describe what was "
            "measured without reproducing what was measured on."
        ),
        allow=(
            "test_id",
            "test_name",
            "status",
            "metrics.**",
            "thresholds.**",
            "**.metrics.**",
            "**.thresholds.**",
            "**.status",
            "**.test_id",
            "**.test_name",
        ),
        deny=_COMMON_DENY,
        max_text_chars=200,
    ),
    "minimal": DisclosurePolicy(
        id="minimal",
        label="Minimal — verdicts only",
        description=(
            "For the most sensitive contexts. Only test identity and pass/warn/fail status "
            "leave the process. The narrative can say what failed, never by how much. "
            "Quantitative content in the report comes entirely from the deterministic path."
        ),
        allow=("test_id", "test_name", "status", "**.test_id", "**.test_name", "**.status"),
        deny=_COMMON_DENY,
        max_text_chars=0,
    ),
}


def policy_for(profile: str | None = None, *, override: str | None = None) -> DisclosurePolicy:
    """Choose a policy from the runtime profile, unless explicitly overridden.

    The default mapping is conservative in the direction that matters: an
    enterprise profile gets ``restricted`` unless someone deliberately asks for
    something else, and asking is recorded in the envelope.
    """
    if override:
        if override not in POLICIES:
            raise KeyError(
                f"Unknown disclosure policy {override!r}. Known: {', '.join(sorted(POLICIES))}"
            )
        return POLICIES[override]

    if profile is None:
        try:
            from start.runtime_profile import active_profile

            profile = active_profile().value
        except Exception:  # pragma: no cover - defensive
            profile = "enterprise"

    return {
        "public_demo": POLICIES["public_demo"],
        "enterprise": POLICIES["restricted"],
        "airgapped": POLICIES["minimal"],
    }.get(profile, POLICIES["restricted"])


# --------------------------------------------------------------------------- #
# Envelope
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DisclosureEnvelope:
    """The exact material a prompt may be built from."""

    policy_id: str
    projected: dict[str, Any]
    withheld_paths: tuple[str, ...]
    numeric_surface: frozenset[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def canonical_json(self) -> str:
        return json.dumps(
            {
                "policy_id": self.policy_id,
                "projected": self.projected,
                "withheld_paths": sorted(self.withheld_paths),
                "metadata": dict(sorted(self.metadata.items())),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )

    def envelope_hash(self) -> str:
        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def as_evidence(self) -> dict[str, Any]:
        """A loggable summary. Deliberately omits the projected content itself."""
        return {
            "policy_id": self.policy_id,
            "envelope_hash": self.envelope_hash(),
            "projected_field_count": len(self.projected),
            "withheld_field_count": len(self.withheld_paths),
            "withheld_paths": sorted(self.withheld_paths),
            "metadata": dict(sorted(self.metadata.items())),
        }

    def render(self) -> str:
        """The envelope as prompt-ready text. This is the only permitted input."""
        lines = ["EVIDENCE (projected under disclosure policy "
                 f"'{self.policy_id}'; envelope {self.envelope_hash()[:12]}):"]
        for path in sorted(self.projected):
            aliased = _alias_path_indices(path)
            lines.append(f"  {aliased} = {self.projected[path]}")
        if self.withheld_paths:
            lines.append("  [field(s) withheld by disclosure policy]")
        return "\n".join(lines)


def _int_to_letters(n: int) -> str:
    """Convert a 0-based index to spreadsheet-style letters: 0 -> A, 1 -> B, 12 -> M, 26 -> AA."""
    res: list[str] = []
    n += 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        res.append(chr(ord("A") + r))
    return "".join(reversed(res))


def _alias_path_indices(path: str) -> str:
    """Render structural record/list indices as letters so no structural digit enters the prompt.

    '[0].metrics.gap' -> 'rA.metrics.gap'
    '[0].thresholds[0].fail' -> 'rA.thresholds[A].fail'
    """
    def repl_leading(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        suffix = "." if m.group(0).endswith(".") else ""
        return f"r{_int_to_letters(idx)}{suffix}"

    # Leading record index: [0]. -> rA.
    out = re.sub(r"^\[(\d+)\]\.?", repl_leading, path)
    # Any remaining bracketed index: [0] -> [A]
    out = re.sub(r"\[(\d+)\]", lambda m: f"[{_int_to_letters(int(m.group(1)))}]", out)
    return out


def _flatten_any(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten to ``path -> scalar`` keeping strings, unlike the numeric flattener."""
    out: dict[str, Any] = {}
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump(mode="json")
    elif hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict

        obj = asdict(obj)

    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(_flatten_any(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            out.update(_flatten_any(value, f"{prefix}[{index}]"))
    elif obj is not None:
        out[prefix] = obj
    return out


def build_envelope(
    evidence: Any,
    *,
    policy: DisclosurePolicy | None = None,
    profile: str | None = None,
    policy_override: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> DisclosureEnvelope:
    """Project evidence through a disclosure policy.

    Raises :class:`DisclosureViolation` under a strict policy when the input
    contains a denied path. Failing loudly is the right behaviour: a denied path
    in the input means an upstream caller assembled something it should not
    have, and silently dropping it would leave that bug in place.
    """
    policy = policy or policy_for(profile, override=policy_override)
    flat = _flatten_any(evidence)

    projected: dict[str, Any] = {}
    withheld: list[str] = []

    for path, value in flat.items():
        if _matches_or_descends_from(path, policy.deny):
            withheld.append(path)
            if policy.strict:
                raise DisclosureViolation(
                    f"Field {path!r} is denied by disclosure policy {policy.id!r} but was "
                    "present in the evidence handed to the disclosure layer. StART refuses "
                    "to build an envelope rather than silently dropping it — the caller "
                    "assembled material it should not have."
                )
            continue
        if not _matches_any(path, policy.allow):
            withheld.append(path)
            continue
        if isinstance(value, str):
            if policy.max_text_chars <= 0:
                withheld.append(path)
                continue
            value = value[: policy.max_text_chars]
        projected[path] = value

    numeric_surface = {
        _format_number(v) for v in flatten_evidence_values(projected).values()
    }

    return DisclosureEnvelope(
        policy_id=policy.id,
        projected=projected,
        withheld_paths=tuple(sorted(withheld)),
        numeric_surface=frozenset(numeric_surface),
        metadata=dict(metadata or {}),
    )


def _format_number(value: float) -> str:
    """Canonical numeric rendering used for surface comparison."""
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return repr(round(value, 10))


_PROMPT_NUMBER = re.compile(r"(?<![\w.])(?:[-+]?\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![\w.])")


def verify_prompt_covered(
    prompt: str,
    envelope: DisclosureEnvelope,
    *,
    allow_extra: frozenset[str] | None = None,
) -> None:
    """Refuse to send a prompt containing numbers the envelope does not hold.

    This is the egress check. It catches the realistic failure — a template that
    interpolates something outside the projection — at the last possible moment,
    which is the only moment at which the actual outbound bytes are known.

    ``allow_extra`` covers legitimate structural numbers a prompt may carry
    (section numbers, a token budget). Keep it small and explicit.
    """
    allowed = set(envelope.numeric_surface) | set(allow_extra or frozenset())
    offending: list[str] = []

    for match in _PROMPT_NUMBER.finditer(prompt):
        token = match.group(0).replace(",", "")
        try:
            numeric = float(token)
        except ValueError:  # pragma: no cover
            continue
        canonical = _format_number(numeric)
        if canonical in allowed or token in allowed:
            continue
        # Percentage rendering of a projected proportion, and vice versa.
        if _format_number(numeric / 100.0) in allowed or _format_number(numeric * 100.0) in allowed:
            continue
        offending.append(token)

    if offending:
        raise DisclosureViolation(
            "Outbound prompt contains "
            f"{len(offending)} numeric value(s) absent from the disclosure envelope "
            f"({envelope.policy_id}, {envelope.envelope_hash()[:12]}): "
            f"{', '.join(sorted(set(offending))[:10])}. The prompt was assembled from "
            "material outside the projection; the call is refused."
        )
