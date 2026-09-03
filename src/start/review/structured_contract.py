"""Structured Reviewer Output Contract and Deterministic Value Hydration.

Machine-first Pydantic contract for MRM reviewer checkpoints. The LLM asserts
qualitative findings referencing exact EvidenceRecord IDs and canonical metric paths.
All numeric values, units, and display representations are deterministically
hydrated by StART from immutable EvidenceRecords.

The LLM is forbidden from authoring authoritative numerical measurements.
Zero arithmetic in hydration, zero arithmetic in renderer, zero LLM arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from start.core.schemas import EvidenceRecord
from start.review.architecture import ReviewGroundingMode

__all__ = [
    "ReviewGroundingMode",
    "EvidenceMetricRef",
    "FindingType",
    "CriterionStatus",
    "ReviewerFinding",
    "StructuredReviewerResponse",
    "HydratedEvidenceRef",
    "HydratedReviewerFinding",
    "HydratedReviewerResponse",
    "StructuredReviewContext",
    "StructuredValidationResult",
    "validate_and_hydrate_structured_response",
    "render_structured_response_markdown",
    "render_structured_grounding_table",
    "validate_qualitative_text_cleanliness",
    "ReviewerObservation",
    "ReviewerAssessment",
    "hydrate_assessment_values",
    "format_assessment_markdown",
]


class FindingType(StrEnum):
    """Categorization of a reviewer's qualitative finding."""

    OBSERVED_EVIDENCE = "OBSERVED_EVIDENCE"
    METHOD_DISAGREEMENT = "METHOD_DISAGREEMENT"
    DIAGNOSTIC_FINDING = "DIAGNOSTIC_FINDING"
    STATISTICAL_NON_REJECTION = "STATISTICAL_NON_REJECTION"
    STATISTICAL_REJECTION = "STATISTICAL_REJECTION"
    UNRESOLVED_MODEL_RISK = "UNRESOLVED_MODEL_RISK"
    EVIDENCE_GAP = "EVIDENCE_GAP"
    CRITERION_REQUIRED = "CRITERION_REQUIRED"
    CROSS_ANALYTICAL_DEPENDENCY = "CROSS_ANALYTICAL_DEPENDENCY"
    CONDITIONAL_CONCLUSION = "CONDITIONAL_CONCLUSION"


class CriterionStatus(StrEnum):
    """Status of quantitative acceptance criterion for the finding."""

    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    ABSENT = "ABSENT"
    EVIDENCE_ONLY = "EVIDENCE_ONLY"


class EvidenceMetricRef(BaseModel):
    """Canonical pointer to a specific evidence record and metric path.

    The LLM supplies only evidence_id and metric_path. StART deterministically
    supplies the authoritative numeric value.
    """

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(
        ...,
        description="Exact EvidenceRecord ID, e.g. 'EV-16bbbafd361d' or 'EV-DIAG-86ce1cf3'",
    )
    metric_path: str = Field(
        ...,
        description="Canonical metric path, e.g. 'metrics.n11' or 'metrics.condition_number'",
    )


class ReviewerFinding(BaseModel):
    """A single structured reviewer finding with typed evidence references."""

    model_config = ConfigDict(frozen=True)

    finding_id: str = Field(
        ...,
        description="Unique identifier for finding within response, e.g. 'F-01'",
    )
    finding_type: FindingType = Field(
        ...,
        description="Standardized classification of finding",
    )
    conclusion: str = Field(
        ...,
        description="Technical qualitative conclusion (must not contain unreferenced raw measurements)",
    )
    evidence_refs: tuple[EvidenceMetricRef, ...] = Field(
        default_factory=tuple,
        description="Explicit evidence metric references supporting this conclusion",
    )
    criterion_status: CriterionStatus = Field(
        default=CriterionStatus.EVIDENCE_ONLY,
        description="Applicability status of acceptance criteria",
    )
    unresolved_reason: str | None = Field(
        default=None,
        description="Reason why risk remains unresolved if applicable",
    )

    @model_validator(mode="after")
    def validate_ref_cardinality(self) -> ReviewerFinding:
        requires_refs = {
            FindingType.OBSERVED_EVIDENCE,
            FindingType.METHOD_DISAGREEMENT,
            FindingType.DIAGNOSTIC_FINDING,
            FindingType.STATISTICAL_NON_REJECTION,
            FindingType.STATISTICAL_REJECTION,
            FindingType.CROSS_ANALYTICAL_DEPENDENCY,
            FindingType.CONDITIONAL_CONCLUSION,
        }
        if self.finding_type in requires_refs and len(self.evidence_refs) == 0:
            raise ValueError(
                f"Finding type '{self.finding_type.value}' requires at least one EvidenceMetricRef."
            )
        return self


class StructuredReviewerResponse(BaseModel):
    """Top-level structured response returned by LLM reviewer."""

    model_config = ConfigDict(frozen=True)

    findings: tuple[ReviewerFinding, ...] = Field(
        ...,
        description="Tuple of structured findings",
    )
    overall_assessment: str = Field(
        ...,
        description="Qualitative cross-finding synthesis and overall assessment",
    )

    @field_validator("findings")
    @classmethod
    def validate_unique_finding_ids(
        cls, v: tuple[ReviewerFinding, ...]
    ) -> tuple[ReviewerFinding, ...]:
        ids = [f.finding_id for f in v]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate finding_id values detected in findings: {ids}")
        return v


class HydratedEvidenceRef(BaseModel):
    """Immutable evidence reference with deterministically populated value and display."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    test_id: str
    metric_path: str
    value: Any
    unit: str = ""
    display: str = ""


class HydratedReviewerFinding(BaseModel):
    """Reviewer finding with deterministically hydrated evidence references."""

    model_config = ConfigDict(frozen=True)

    finding_id: str
    finding_type: FindingType
    conclusion: str
    hydrated_refs: tuple[HydratedEvidenceRef, ...]
    criterion_status: CriterionStatus
    unresolved_reason: str | None = None


@dataclass(frozen=True)
class StructuredReviewContext:
    """Immutable provenance ensuring every accepted record belongs to current view/run."""

    run_id: str
    checkpoint_id: str
    evidence_view_hash: str
    allowed_evidence_ids: tuple[str, ...]
    records_by_id: dict[str, EvidenceRecord]


class HydratedReviewerResponse(BaseModel):
    """Final attested review object containing hydrated findings and content hash."""

    model_config = ConfigDict(frozen=True)

    findings: tuple[HydratedReviewerFinding, ...]
    overall_assessment: str
    context_run_id: str
    context_checkpoint_id: str
    evidence_view_hash: str
    content_hash: str


@dataclass(frozen=True)
class StructuredValidationResult:
    """Outcome of validating a structured reviewer response against current evidence view."""

    valid: bool
    response: StructuredReviewerResponse | None
    hydrated_response: HydratedReviewerResponse | None
    findings_count: int
    evidence_refs_count: int
    validated_refs_count: int
    invalid_refs_count: int
    invalid_refs_details: tuple[dict[str, Any], ...]
    error_message: str | None = None


_FORBIDDEN_RAW_FLOAT = re.compile(r"(?<![a-zA-Z0-9_\-\.])(?:\d+\.\d{3,}|\d+(?:\.\d+)?e[+-]?\d+)\b")


def validate_qualitative_text_cleanliness(text: str) -> tuple[bool, str | None]:
    """Conservative guard against raw multi-digit floating point measurements in qualitative prose.

    Identifiers like n11, F1, 2020-09-25, 99%, or simple counts are allowed.
    Precise floating point measurements (e.g. 0.001679, 1.91e-8) must be referenced via refs.
    """
    matches = _FORBIDDEN_RAW_FLOAT.findall(text)
    if matches:
        return False, f"Raw measurement values {matches} embedded in qualitative text; use EvidenceMetricRef."
    return True, None


def _format_deterministic_display(val: Any, metric_path: str) -> tuple[str, str]:
    """Deterministically format display representation and unit from raw metric value."""
    if val is None:
        return "", ""
    if isinstance(val, bool):
        return str(val), "boolean"
    if isinstance(val, int):
        return f"{val:,}", "count"
    if isinstance(val, float):
        low = metric_path.lower()
        pct_keys = ("rate", "share", "volatility", "variance_shortfall",
                    "fraction", "ratio_pct")
        if any(k in low for k in pct_keys):
            if abs(val) <= 1.0 and not any(k in low for k in ("count", "df", "degree", "dof")):
                return f"{val * 100.0:.4f}%", "percentage"
            return f"{val:.4f}%", "percentage"
        if any(k in low for k in ("p_value", "alpha", "gamma", "probability")):
            return f"{val:.4f}", "probability"
        if abs(val) < 0.0001 and val != 0.0:
            return f"{val:.6e}", ""
        if abs(val) >= 1000:
            return f"{val:,.4f}".rstrip("0").rstrip("."), ""
        return f"{val:.6g}", ""
    return str(val), ""


def validate_and_hydrate_structured_response(
    raw_response: StructuredReviewerResponse,
    context: StructuredReviewContext,
) -> StructuredValidationResult:
    """Strictly validate and deterministically hydrate a StructuredReviewerResponse.

    Enforces:
    1. Exact Evidence ID exists in current view (no truncation, no fuzzy match).
    2. Exact canonical metric path exists (e.g. 'metrics.n11', 'params.alpha').
    3. EvidenceRecord belongs to the current checkpoint/view and run.
    4. Qualitative text contains no raw unreferenced floating point measurements.
    5. No arithmetic in hydration or renderer.
    """
    invalid_details: list[dict[str, Any]] = []
    hydrated_findings: list[HydratedReviewerFinding] = []
    total_refs = 0
    validated_refs = 0

    if not context.evidence_view_hash:
        return StructuredValidationResult(
            valid=False,
            response=raw_response,
            hydrated_response=None,
            findings_count=len(raw_response.findings),
            evidence_refs_count=0,
            validated_refs_count=0,
            invalid_refs_count=1,
            invalid_refs_details=(
                {
                    "field": "evidence_view_hash",
                    "error": "Missing or empty evidence_view_hash in context",
                },
            ),
            error_message="Missing or empty evidence_view_hash in StructuredReviewContext.",
        )

    # 1. Validate overall assessment qualitative cleanliness
    clean, reason = validate_qualitative_text_cleanliness(raw_response.overall_assessment)
    if not clean:
        return StructuredValidationResult(
            valid=False,
            response=raw_response,
            hydrated_response=None,
            findings_count=len(raw_response.findings),
            evidence_refs_count=0,
            validated_refs_count=0,
            invalid_refs_count=1,
            invalid_refs_details=({"field": "overall_assessment", "error": reason},),
            error_message=f"overall_assessment contains raw numeric measurements: {reason}",
        )

    # 2. Process each finding and its refs
    for finding in raw_response.findings:
        f_clean, f_reason = validate_qualitative_text_cleanliness(finding.conclusion)
        if not f_clean:
            invalid_details.append(
                {
                    "finding_id": finding.finding_id,
                    "field": "conclusion",
                    "error": f_reason,
                }
            )

        hydrated_refs: list[HydratedEvidenceRef] = []
        for ref in finding.evidence_refs:
            total_refs += 1
            ev_id = ref.evidence_id.strip()
            path = ref.metric_path.strip()

            # Rule 1 & 3: Exact Evidence ID membership in current view/run
            if ev_id not in context.allowed_evidence_ids or ev_id not in context.records_by_id:
                invalid_details.append(
                    {
                        "finding_id": finding.finding_id,
                        "evidence_id": ev_id,
                        "metric_path": path,
                        "error": f"Evidence ID '{ev_id}' not found in current CheckpointEvidenceView.",
                    }
                )
                continue

            rec = context.records_by_id[ev_id]
            if rec.run_id != context.run_id and rec.run_id != "deterministic":
                invalid_details.append(
                    {
                        "finding_id": finding.finding_id,
                        "evidence_id": ev_id,
                        "metric_path": path,
                        "error": (
                            f"Cross-run reference violation: record run_id "
                            f"'{rec.run_id}' != current '{context.run_id}'."
                        ),
                    }
                )
                continue

            # Rule 2: Canonical metric path resolution
            valid_prefix = (
                path.startswith("metrics.")
                or path.startswith("params.")
                or path.startswith("thresholds.")
            )
            if not valid_prefix:
                invalid_details.append(
                    {
                        "finding_id": finding.finding_id,
                        "evidence_id": ev_id,
                        "metric_path": path,
                        "error": (
                            f"Invalid non-canonical path format: '{path}'. "
                            "Canonical structured path must start with 'metrics.' or 'params.'."
                        ),
                    }
                )
                continue

            val: Any = None
            found = False

            if path.startswith("metrics."):
                k = path[len("metrics."):]
                if rec.metrics and k in rec.metrics:
                    val = rec.metrics[k]
                    found = True
            elif path.startswith("params."):
                k = path[len("params."):]
                if rec.params and k in rec.params:
                    val = rec.params[k]
                    found = True
            elif path.startswith("thresholds."):
                k = path[len("thresholds."):]
                for t in rec.thresholds:
                    if getattr(t, "metric", None) == k:
                        val = t.warn if t.warn is not None else t.fail
                        found = True
                        break

            if not found:
                invalid_details.append(
                    {
                        "finding_id": finding.finding_id,
                        "evidence_id": ev_id,
                        "metric_path": path,
                        "error": f"Metric path '{path}' does not exist on EvidenceRecord '{ev_id}'.",
                    }
                )
                continue

            display_str, unit_str = _format_deterministic_display(val, path)
            hydrated_ref = HydratedEvidenceRef(
                evidence_id=ev_id,
                test_id=rec.test_id,
                metric_path=path,
                value=val,
                unit=unit_str,
                display=display_str,
            )
            hydrated_refs.append(hydrated_ref)
            validated_refs += 1

        hydrated_findings.append(
            HydratedReviewerFinding(
                finding_id=finding.finding_id,
                finding_type=finding.finding_type,
                conclusion=finding.conclusion,
                hydrated_refs=tuple(hydrated_refs),
                criterion_status=finding.criterion_status,
                unresolved_reason=finding.unresolved_reason,
            )
        )

    if invalid_details:
        return StructuredValidationResult(
            valid=False,
            response=raw_response,
            hydrated_response=None,
            findings_count=len(raw_response.findings),
            evidence_refs_count=total_refs,
            validated_refs_count=validated_refs,
            invalid_refs_count=len(invalid_details),
            invalid_refs_details=tuple(invalid_details),
            error_message=f"Structured validation failed with {len(invalid_details)} invalid refs/fields.",
        )

    # Compute canonical content-addressed hash
    canonical_dict = {
        "context_run_id": context.run_id,
        "context_checkpoint_id": context.checkpoint_id,
        "evidence_view_hash": context.evidence_view_hash,
        "overall_assessment": raw_response.overall_assessment,
        "findings": [
            {
                "finding_id": f.finding_id,
                "finding_type": f.finding_type.value,
                "conclusion": f.conclusion,
                "criterion_status": f.criterion_status.value,
                "unresolved_reason": f.unresolved_reason,
                "refs": [
                    {
                        "evidence_id": r.evidence_id,
                        "test_id": r.test_id,
                        "metric_path": r.metric_path,
                        "value": r.value,
                    }
                    for r in f.hydrated_refs
                ],
            }
            for f in hydrated_findings
        ],
    }
    canonical_json = json.dumps(canonical_dict, sort_keys=True, separators=(",", ":"))
    content_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

    hydrated_resp = HydratedReviewerResponse(
        findings=tuple(hydrated_findings),
        overall_assessment=raw_response.overall_assessment,
        context_run_id=context.run_id,
        context_checkpoint_id=context.checkpoint_id,
        evidence_view_hash=context.evidence_view_hash,
        content_hash=content_hash,
    )

    return StructuredValidationResult(
        valid=True,
        response=raw_response,
        hydrated_response=hydrated_resp,
        findings_count=len(raw_response.findings),
        evidence_refs_count=total_refs,
        validated_refs_count=validated_refs,
        invalid_refs_count=0,
        invalid_refs_details=(),
        error_message=None,
    )


def render_structured_response_markdown(hydrated: HydratedReviewerResponse) -> str:
    """Render human-readable audit-grade markdown reproducibly from hydrated finding graph."""
    lines: list[str] = [
        f"### Structured Review Assessment: {hydrated.context_checkpoint_id}",
        "",
        "#### Findings",
    ]
    for f in hydrated.findings:
        lines.append(f"- **[{f.finding_id}] {f.finding_type.value}**: {f.conclusion}")
        for r in f.hydrated_refs:
            val_display = f"`{r.display}`" if r.display else f"`{r.value}`"
            lines.append(f"  - `{r.metric_path}` = **{val_display}** [{r.evidence_id}] ({r.test_id})")
        if f.criterion_status != CriterionStatus.EVIDENCE_ONLY:
            lines.append(f"  - *Criterion Status*: `{f.criterion_status.value}`")
        if f.unresolved_reason:
            lines.append(f"  - *Unresolved Model Risk*: {f.unresolved_reason}")

    lines.append("")
    lines.append("#### Overall Assessment")
    lines.append(hydrated.overall_assessment)
    lines.append("")
    lines.append(f"*Content Hash (SHA-256)*: `{hydrated.content_hash}`")

    return "\n".join(lines).strip()


def render_structured_grounding_table(
    res: StructuredValidationResult,
    checkpoint_title: str,
) -> str:
    """Format structured grounding gate status table."""
    status_str = "PASSED" if res.valid else "FAILED"
    lines = [
        f"Structured Grounding Gate: {status_str} — "
        f"Findings: {res.findings_count} | "
        f"Evidence refs: {res.evidence_refs_count} | "
        f"Validated refs: {res.validated_refs_count} | "
        f"Invalid: {res.invalid_refs_count}"
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Compatibility Layer: Legacy ReviewerAssessment & ReviewerObservation
# --------------------------------------------------------------------------- #
class ReviewerObservation(BaseModel):
    """A single proof-carrying quantitative observation referenced to evidence."""

    model_config = ConfigDict(protected_namespaces=())

    evidence_id: str = Field(
        ..., description="Exact bracketed EvidenceRecord ID, e.g. EV-123456789abc"
    )
    metric_path: str = Field(
        ..., description="Canonical metric field path, e.g. 'annualised_volatility' or 'n_exceptions'"
    )
    interpretation: str = Field(
        default="", description="Objective technical finding based on this evidence"
    )
    value: float | None = Field(default=None, description="Deterministically hydrated numeric value")
    unit: str = Field(
        default="", description="Deterministically hydrated unit (% | bps | ratio | count | ...)"
    )
    display: str = Field(default="", description="Deterministically hydrated display representation")


class ReviewerAssessment(BaseModel):
    """Structured reviewer assessment contract."""

    model_config = ConfigDict(protected_namespaces=())

    checkpoint: str = Field(..., description="Checkpoint title")
    action: str = Field(default="accept", description="accept | override | challenge | question")
    observations: list[ReviewerObservation] = Field(
        default_factory=list, description="Groundable evidence observations"
    )
    concerns: list[str] = Field(
        default_factory=list, description="Identified model risk concerns"
    )
    missing_criteria: list[str] = Field(
        default_factory=list, description="Unmet criteria or missing evidence"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Actionable governance recommendations"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Explicit model limitations / failure modes"
    )


def _lookup_legacy_evidence_metric(record: EvidenceRecord, metric_path: str) -> tuple[float | None, str, str]:
    """Look up a metric path in an EvidenceRecord and return (value, unit, display)."""
    metrics = record.metrics or {}
    val: Any = None
    if metric_path in metrics:
        val = metrics[metric_path]
    else:
        stripped = metric_path.split(".")[-1]
        if stripped in metrics:
            val = metrics[stripped]

    if val is None:
        params = getattr(record, "params", {}) or {}
        if metric_path in params:
            val = params[metric_path]
        elif metric_path.split(".")[-1] in params:
            val = params[metric_path.split(".")[-1]]

    if val is None:
        return None, "", ""

    try:
        f_val = float(val)
    except (ValueError, TypeError):
        return None, "", str(val)

    low_path = metric_path.lower()
    pct_keys = ("volatility", "variance", "return", "error_rate", "rate", "prop", "ratio_pct", "percentage")
    if any(k in low_path for k in pct_keys) or "%" in str(val):
        if abs(f_val) <= 1.0 and not any(k in low_path for k in ("count", "n_", "df", "degree")):
            return f_val, "%", f"{f_val * 100.0:.4f}%"
        return f_val, "%", f"{f_val:.4f}%"
    elif "bps" in low_path:
        return f_val, "bps", f"{f_val:.1f} bps"
    elif any(k in low_path for k in ("count", "n_", "exceptions", "periods", "assets", "trials", "folds")):
        return f_val, "count", f"{int(round(f_val))}"
    else:
        if abs(f_val) >= 1000:
            return f_val, "", f"{f_val:,.4f}".rstrip("0").rstrip(".")
        elif abs(f_val) < 0.001 and f_val != 0.0:
            return f_val, "", f"{f_val:.6f}"
        else:
            return f_val, "", f"{f_val:.4f}".rstrip("0").rstrip(".")


def hydrate_assessment_values(
    assessment: ReviewerAssessment,
    records: list[EvidenceRecord],
) -> ReviewerAssessment:
    """Deterministically populate value, unit, and display for all observations from evidence."""
    records_by_id: dict[str, EvidenceRecord] = {}
    for r in records:
        if r.evidence_id:
            records_by_id[r.evidence_id] = r
            records_by_id[r.evidence_id.strip("[]()")] = r

    hydrated_observations: list[ReviewerObservation] = []
    for obs in assessment.observations:
        clean_ev_id = obs.evidence_id.strip("[]()")
        rec = records_by_id.get(clean_ev_id) or records_by_id.get(obs.evidence_id)
        if rec is not None:
            val, unit, display = _lookup_legacy_evidence_metric(rec, obs.metric_path)
            hydrated_obs = ReviewerObservation(
                evidence_id=rec.evidence_id,
                metric_path=obs.metric_path,
                interpretation=obs.interpretation,
                value=val,
                unit=unit,
                display=display,
            )
            hydrated_observations.append(hydrated_obs)
        else:
            hydrated_observations.append(obs)

    return ReviewerAssessment(
        checkpoint=assessment.checkpoint,
        action=assessment.action,
        observations=hydrated_observations,
        concerns=list(assessment.concerns),
        missing_criteria=list(assessment.missing_criteria),
        recommendations=list(assessment.recommendations),
        limitations=list(assessment.limitations),
    )


def format_assessment_markdown(assessment: ReviewerAssessment) -> str:
    """Format a ReviewerAssessment into clean, audit-grade markdown."""
    lines: list[str] = [
        f"### Review Assessment: {assessment.checkpoint}",
        f"**Action:** {assessment.action.upper()}",
        "",
    ]
    if assessment.observations:
        lines.append("#### Grounded Observations")
        for obs in assessment.observations:
            val_str = f" `{obs.display}`" if obs.display else (
                f" `{obs.value}`" if obs.value is not None else ""
            )
            clean_id = obs.evidence_id.strip("[]()")
            lines.append(f"- **[{clean_id}]** (`{obs.metric_path}`{val_str}): {obs.interpretation}")
        lines.append("")

    if assessment.concerns:
        lines.append("#### Model Risk Concerns")
        for c in assessment.concerns:
            lines.append(f"- {c}")
        lines.append("")

    if assessment.missing_criteria:
        lines.append("#### Missing Criteria / Scope Gaps")
        for m in assessment.missing_criteria:
            lines.append(f"- {m}")
        lines.append("")

    if assessment.recommendations:
        lines.append("#### Recommendations")
        for r in assessment.recommendations:
            lines.append(f"- {r}")
        lines.append("")

    if assessment.limitations:
        lines.append("#### Known Limitations & Failure Modes")
        for lim in assessment.limitations:
            lines.append(f"- {lim}")
        lines.append("")

    return "\n".join(lines).strip()

