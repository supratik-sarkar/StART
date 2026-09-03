"""Canonical Checkpoint Evidence View for StART.

Unifies structured evidence projection across:
1. Rich terminal table rendering
2. LLM reviewer prompt payloads
3. Quantitative claim grounding
4. Specialist challenges and governance synthesis

Eliminates parallel, out-of-sync evidence dictionaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from start.core.schemas import EvidenceRecord, Status
from start.review.architecture import ReviewDomain


@dataclass(frozen=True)
class CheckpointMetricRef:
    """Immutable collision-safe identity for an evidence metric."""

    evidence_id: str
    test_id: str
    metric_path: str

    @property
    def canonical_key(self) -> tuple[str, str]:
        """Canonical tuple key (evidence_id, metric_path)."""
        return (self.evidence_id, self.metric_path)

    def __str__(self) -> str:
        return f"{self.evidence_id}:{self.test_id}.{self.metric_path}"


@dataclass(frozen=True)
class CheckpointEvidenceMetric:
    """A single canonical quantitative or qualitative evidence metric."""

    ref: CheckpointMetricRef
    path: str
    name: str
    value: Any
    numeric_value: float | None
    evidence_id: str
    test_id: str
    status: Status = Status.RECORDED
    unit: str = ""
    description: str = ""
    is_criterion: bool = False
    criterion_target: str = ""
    provenance: str = ""


@dataclass(frozen=True)
class CheckpointEvidenceView:
    """Immutable typed structure representing the canonical evidence contract for a review checkpoint."""

    checkpoint_title: str
    checkpoint_description: str
    domains: tuple[ReviewDomain, ...]
    evidence_records: tuple[EvidenceRecord, ...]
    metrics: tuple[CheckpointEvidenceMetric, ...]
    allowed_metric_paths: tuple[str, ...]
    evidence_by_id: dict[str, EvidenceRecord]
    metrics_by_ref: dict[CheckpointMetricRef, CheckpointEvidenceMetric]
    metrics_by_evidence_and_path: dict[tuple[str, str], CheckpointEvidenceMetric]
    metrics_by_test_and_path: dict[tuple[str, str], tuple[CheckpointEvidenceMetric, ...]]
    metrics_by_path: dict[str, tuple[CheckpointEvidenceMetric, ...]]
    numeric_grounding_map: dict[str, dict[str, float]]
    limitations: tuple[str, ...] = field(default_factory=tuple)
    diagnostic_evidence: dict[str, Any] = field(default_factory=dict)
    pattern_b_evidence: dict[str, Any] = field(default_factory=dict)
    artifacts: tuple[Any, ...] = field(default_factory=tuple)

    def get_metric(
        self,
        path_or_name: str,
        *,
        evidence_id: str | None = None,
        test_id: str | None = None,
        default: Any = None,
    ) -> Any:
        """Resolve metric raw value using collision-safe identity or path.

        Fails closed (returns default) when path_or_name is ambiguous across records.
        """
        if evidence_id:
            key = (evidence_id, path_or_name)
            if key in self.metrics_by_evidence_and_path:
                return self.metrics_by_evidence_and_path[key].value

        if test_id:
            test_key = (test_id, path_or_name)
            matches = self.metrics_by_test_and_path.get(test_key, ())
            if len(matches) == 1:
                return matches[0].value
            elif len(matches) > 1:
                return default

        matches = self.metrics_by_path.get(path_or_name, ())
        if not matches:
            matches = tuple(m for m in self.metrics if m.name == path_or_name)

        if len(matches) == 1:
            return matches[0].value
        elif len(matches) > 1:
            first_val = matches[0].value
            if all(m.value == first_val for m in matches):
                return first_val
            return default  # Ambiguous across records with different values: fail closed

        return default

    def get_metrics(self, path_or_name: str) -> tuple[CheckpointEvidenceMetric, ...]:
        """Return all metrics matching path or short name without collision overwrite."""
        matches = self.metrics_by_path.get(path_or_name, ())
        if matches:
            return matches
        return tuple(m for m in self.metrics if m.name == path_or_name)

    def get_numeric(
        self,
        path_or_name: str,
        *,
        evidence_id: str | None = None,
        test_id: str | None = None,
        default: float | None = None,
    ) -> float | None:
        """Resolve metric float value using collision-safe identity or path."""
        val = self.get_metric(path_or_name, evidence_id=evidence_id, test_id=test_id, default=None)
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        return default

    def compute_evidence_view_hash(self) -> str:
        """Compute deterministic SHA-256 fingerprint for this exact checkpoint view."""
        import hashlib
        ev_ids = [r.evidence_id for r in self.evidence_records if r.evidence_id]
        payload = f"{self.checkpoint_title}|{','.join(ev_ids)}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get_canonical_admissible_paths(self, evidence_id: str | None = None) -> list[str]:
        """Return list of canonical metric paths (metrics.*, params.*) for records in this view."""
        paths: list[str] = []
        if evidence_id and evidence_id in self.evidence_by_id:
            recs = [self.evidence_by_id[evidence_id]]
        else:
            recs = list(self.evidence_records)
        for r in recs:
            if r.metrics:
                for k in r.metrics:
                    if not k.startswith("extra_"):
                        paths.append(f"metrics.{k}")
            if r.params:
                for pk in r.params:
                    paths.append(f"params.{pk}")
        return sorted(set(paths))

    def format_llm_payload(self) -> str:
        """Format the complete canonical evidence payload for LLM reviewer prompts."""
        if not self.evidence_records:
            return "No specific test metrics for this checkpoint."

        blocks: list[str] = []
        for r in self.evidence_records:
            status_val = r.status.value if hasattr(r.status, "value") else str(r.status)
            lines = [f"- [{r.evidence_id}] Test: {r.test_id} (Status: {status_val})"]
            rec_paths = self.get_canonical_admissible_paths(r.evidence_id)
            if rec_paths:
                lines.append(f"  Admissible Canonical Metric Paths for [{r.evidence_id}]:")
                for p in rec_paths:
                    val = self.get_metric(p, evidence_id=r.evidence_id)
                    lines.append(f"    * {p} (value: {val})")
            else:
                lines.append(
                    f"  Admissible Metric Paths for [{r.evidence_id}]: None "
                    "(if citing, use finding_type EVIDENCE_GAP with evidence_refs: [])"
                )
            if r.interpretation:
                lines.append(f"  Interpretation: {r.interpretation}")
            if r.limitations:
                lines.append("  Limitations:")
                for lim in r.limitations[:3]:
                    lines.append(f"    - {lim}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def allowed_paths_for_repair(self, max_count: int = 50) -> str:
        """Return formatted string of allowed metric paths for grounding repair prompts."""
        paths = list(self.allowed_metric_paths)
        return ", ".join(paths[:max_count])

    def as_grounding_records(self) -> list[dict[str, Any]]:
        """Return structured records for claim binding with verified paths and values."""
        scoped_list: list[dict[str, Any]] = []
        for r in self.evidence_records:
            record_fields: dict[str, float] = {}
            if r.evidence_id in self.numeric_grounding_map:
                record_fields.update(self.numeric_grounding_map[r.evidence_id])
            scoped_list.append({
                "evidence_id": r.evidence_id,
                "test_id": r.test_id,
                "fields": record_fields,
            })
        return scoped_list


def build_checkpoint_evidence_view(
    checkpoint_title: str,
    checkpoint_description: str,
    domains: tuple[ReviewDomain, ...],
    records: list[EvidenceRecord],
    artifacts: list[Any] | None = None,
    diagnostic_evidence: dict[str, Any] | None = None,
    pattern_b_evidence: dict[str, Any] | None = None,
) -> CheckpointEvidenceView:
    """Construct a canonical CheckpointEvidenceView from matched EvidenceRecords."""
    metrics_list: list[CheckpointEvidenceMetric] = []
    allowed_paths: list[str] = []
    evidence_by_id: dict[str, EvidenceRecord] = {}
    metrics_by_ref: dict[CheckpointMetricRef, CheckpointEvidenceMetric] = {}
    metrics_by_evidence_and_path: dict[tuple[str, str], CheckpointEvidenceMetric] = {}
    metrics_by_test_and_path: dict[tuple[str, str], list[CheckpointEvidenceMetric]] = {}
    metrics_by_path: dict[str, list[CheckpointEvidenceMetric]] = {}
    numeric_grounding_map: dict[str, dict[str, float]] = {}
    all_limitations: list[str] = []

    for r in records:
        if r.evidence_id:
            evidence_by_id[r.evidence_id] = r
            numeric_grounding_map.setdefault(r.evidence_id, {})

        if r.limitations:
            all_limitations.extend(r.limitations)

        # 1. Process structured metrics
        for k, v in r.metrics.items():
            if k.startswith("extra_"):
                continue

            full_path = f"{r.test_id}.{k}"
            allowed_paths.append(full_path)

            num_val: float | None = None
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                num_val = float(v)
            elif isinstance(v, str):
                try:
                    num_val = float(v.strip().rstrip("%"))
                except ValueError:
                    num_val = None

            unit = ""
            k_low = k.lower()
            if (
                "probability" in k_low
                or "p_value" in k_low
                or "alpha" in k_low
                or "gamma" in k_low
                or "size" in k_low
                or "power" in k_low
                or "confidence" in k_low
            ):
                unit = "probability"
            elif "rate" in k_low or "share" in k_low or "weight" in k_low or "fraction" in k_low:
                unit = "percentage"
            elif "multiplier" in k_low:
                unit = "multiplier"
            elif "statistic" in k_low or "lr_" in k_low or "stat" in k_low:
                unit = "statistic"
            elif "n_" in k_low or "degrees_of_freedom" in k_low or "count" in k_low:
                unit = "count"

            is_crit = False
            crit_target = ""
            if (
                k.startswith("required.")
                or "critical_value" in k_low
                or "threshold" in k_low
                or "criterion" in k_low
                or "band" in k_low
                or k in ("nominal_size", "nominal_significance_level", "alpha", "gamma_test")
            ):
                is_crit = True
                crit_target = str(v)

            ref = CheckpointMetricRef(
                evidence_id=r.evidence_id or "",
                test_id=r.test_id,
                metric_path=k,
            )

            metric_item = CheckpointEvidenceMetric(
                ref=ref,
                path=full_path,
                name=k,
                value=v,
                numeric_value=num_val,
                evidence_id=r.evidence_id or "",
                test_id=r.test_id,
                status=r.status,
                unit=unit,
                is_criterion=is_crit,
                criterion_target=crit_target,
                provenance=r.repro.runtime if r.repro else "deterministic",
            )
            metrics_list.append(metric_item)
            metrics_by_ref[ref] = metric_item

            if r.evidence_id:
                metrics_by_evidence_and_path[(r.evidence_id, k)] = metric_item
                metrics_by_evidence_and_path[(r.evidence_id, full_path)] = metric_item
                metrics_by_evidence_and_path[(r.evidence_id, f"metrics.{k}")] = metric_item
                metrics_by_evidence_and_path[(r.evidence_id, f"params.{k}")] = metric_item

            metrics_by_test_and_path.setdefault((r.test_id, k), []).append(metric_item)
            metrics_by_test_and_path.setdefault((r.test_id, full_path), []).append(metric_item)
            metrics_by_test_and_path.setdefault((r.test_id, f"metrics.{k}"), []).append(metric_item)
            metrics_by_test_and_path.setdefault((r.test_id, f"params.{k}"), []).append(metric_item)

            metrics_by_path.setdefault(k, []).append(metric_item)
            metrics_by_path.setdefault(full_path, []).append(metric_item)
            metrics_by_path.setdefault(f"metrics.{k}", []).append(metric_item)
            metrics_by_path.setdefault(f"params.{k}", []).append(metric_item)

            if r.evidence_id and num_val is not None:
                numeric_grounding_map[r.evidence_id][full_path] = num_val
                numeric_grounding_map[r.evidence_id][k] = num_val

            # Parse compound string intervals ONLY for structured criterion fields
            if is_crit and isinstance(v, str) and ("[" in v or "(" in v):
                interval_nums = re.findall(r"[-+]?\d*\.?\d+", v)
                if len(interval_nums) >= 2:
                    try:
                        low_val = float(interval_nums[0])
                        high_val = float(interval_nums[1])
                        if r.evidence_id:
                            numeric_grounding_map[r.evidence_id][f"{full_path}.lower"] = low_val
                            numeric_grounding_map[r.evidence_id][f"{k}.lower"] = low_val
                            numeric_grounding_map[r.evidence_id][f"{full_path}.upper"] = high_val
                            numeric_grounding_map[r.evidence_id][f"{k}.upper"] = high_val
                            numeric_grounding_map[r.evidence_id][f"band_lower_{k}"] = low_val
                            numeric_grounding_map[r.evidence_id][f"band_upper_{k}"] = high_val
                    except ValueError:
                        pass

            # Parse multiplier keys (e.g. "power_understated_0_7x" -> 0.7)
            if "0_7x" in k:
                if r.evidence_id:
                    numeric_grounding_map[r.evidence_id]["power_multiplier_0_7x"] = 0.7
                    numeric_grounding_map[r.evidence_id][f"{r.test_id}.power_multiplier_0_7x"] = 0.7
            elif "1_5x" in k:
                if r.evidence_id:
                    numeric_grounding_map[r.evidence_id]["power_multiplier_1_5x"] = 1.5
                    numeric_grounding_map[r.evidence_id][f"{r.test_id}.power_multiplier_1_5x"] = 1.5

        # 2. Process params as potential grounding values
        if r.params:
            for pk, pv in r.params.items():
                if isinstance(pv, (int, float)) and not isinstance(pv, bool):
                    p_num = float(pv)
                    if r.evidence_id:
                        numeric_grounding_map[r.evidence_id][f"{r.test_id}.param.{pk}"] = p_num
                        numeric_grounding_map[r.evidence_id][f"param.{pk}"] = p_num

    # Freeze tuple maps
    frozen_test_and_path = {k: tuple(v) for k, v in metrics_by_test_and_path.items()}
    frozen_by_path = {k: tuple(v) for k, v in metrics_by_path.items()}

    return CheckpointEvidenceView(
        checkpoint_title=checkpoint_title,
        checkpoint_description=checkpoint_description,
        domains=domains,
        evidence_records=tuple(records),
        metrics=tuple(metrics_list),
        allowed_metric_paths=tuple(allowed_paths),
        evidence_by_id=evidence_by_id,
        metrics_by_ref=metrics_by_ref,
        metrics_by_evidence_and_path=metrics_by_evidence_and_path,
        metrics_by_test_and_path=frozen_test_and_path,
        metrics_by_path=frozen_by_path,
        numeric_grounding_map=numeric_grounding_map,
        limitations=tuple(dict.fromkeys(all_limitations)),
        diagnostic_evidence=dict(diagnostic_evidence or {}),
        pattern_b_evidence=dict(pattern_b_evidence or {}),
        artifacts=tuple(artifacts or []),
    )
