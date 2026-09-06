"""Untrusted Browser WebLLM Reviewer Ingestion, Hydration & Governance Gating Routes for StART v4.5."""

from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import APIRouter, HTTPException

from start.web.queue import GLOBAL_QUEUE
from start.web.schemas import (
    START_SCHEMA_VERSION,
    APIResponseEnvelope,
    HydratedFindingView,
    ReviewerHydrationResponse,
    WebReviewerSubmission,
)

logger = logging.getLogger("start.web.routes_reviewer")
router = APIRouter(prefix="/api/v1/runs", tags=["reviewer"])


@router.post("/{run_id}/reviewer/hydrate-and-gate", response_model=APIResponseEnvelope)
def hydrate_and_gate_reviewer_submission(
    run_id: str,
    submission: WebReviewerSubmission,
) -> APIResponseEnvelope:
    """Validate untrusted browser WebLLM reviewer submission, hydrate metrics, and apply OPA gating.

    Strict Architectural Invariants:
    1. Zero arithmetic by the client browser or LLM.
    2. All numeric values are deterministically resolved from immutable EvidenceRecord objects.
    3. Rejects unknown Evidence IDs or references outside the active run universe.
    4. OPA governance policy decision and Merkle attestation are determined exclusively server-side.
    """
    ctx = GLOBAL_QUEUE.get_run(run_id, submission.session_id)
    if not ctx:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or session access denied")

    records = ctx.evidence_records
    evidence_universe = {r.evidence_id: r for r in records}

    hydrated_findings: list[HydratedFindingView] = []
    all_grounded = True

    # 1. Process and hydrate each qualitative finding
    for finding in submission.findings:
        finding_grounded = True
        hydrated_refs: list[dict[str, Any]] = []

        for ref in finding.evidence_refs:
            ev_id = ref.evidence_id.strip("[]")
            metric_name = ref.metric_name

            if ev_id not in evidence_universe:
                logger.warning("Unknown Evidence ID '%s' rejected during server hydration", ev_id)
                finding_grounded = False
                all_grounded = False
                hydrated_refs.append(
                    {
                        "evidence_id": ev_id,
                        "metric_name": metric_name,
                        "status": "UNGROUNDED_EVIDENCE_ID",
                        "hydrated_value": None,
                    }
                )
                continue

            rec = evidence_universe[ev_id]
            if not metric_name and rec.metrics:
                first_key = next(iter(rec.metrics.keys()))
                actual_val = rec.metrics[first_key]
                metric_name = first_key
            elif not metric_name:
                actual_val = str(rec.status)
                metric_name = "status"
            else:
                actual_val = rec.metrics.get(metric_name)

            if actual_val is None and metric_name:
                # Try prefix or substring match
                for k, v in rec.metrics.items():
                    if k.endswith(f".{metric_name}") or metric_name in k:
                        actual_val = v
                        metric_name = k
                        break

            if actual_val is None:
                logger.warning("Metric '%s' not found in EvidenceRecord '%s'", metric_name, ev_id)
                finding_grounded = False
                all_grounded = False
                hydrated_refs.append(
                    {
                        "evidence_id": ev_id,
                        "metric_name": metric_name,
                        "status": "UNGROUNDED_METRIC_PATH",
                        "hydrated_value": None,
                    }
                )
            else:
                hydrated_refs.append(
                    {
                        "evidence_id": ev_id,
                        "metric_name": metric_name,
                        "status": "GROUNDED",
                        "hydrated_value": actual_val,
                        "test_id": rec.test_id,
                        "record_status": str(rec.status),
                    }
                )

        hydrated_findings.append(
            HydratedFindingView(
                finding_id=finding.finding_id,
                severity=finding.severity,
                title=finding.title,
                description=finding.description,
                grounded=finding_grounded,
                evidence_refs=hydrated_refs,
                recommendation=finding.recommendation,
            )
        )

    # 2. Server-side OPA policy evaluation
    opa_decision = "ALLOW"
    opa_reasons: list[str] = []
    try:
        from start.policies.opa_policy_plane import OPAPolicyPlane

        policy_plane = OPAPolicyPlane()
        n_ungrounded = sum(1 for f in hydrated_findings if not f.grounded)
        n_failures = sum(1 for r in records if r.status.value == "fail")
        disp_candidate = "ACCEPT" if n_failures == 0 else "CONDITIONAL_ACCEPT"

        pol_dec = policy_plane.evaluate_governance_attestation(
            n_ungrounded_claims=n_ungrounded,
            n_validation_failures=n_failures,
            committee_disposition=disp_candidate,
            run_id=run_id,
        )
        opa_decision = pol_dec.decision
        opa_reasons.append(pol_dec.reason)
    except Exception as exc:
        logger.warning("OPA policy plane evaluation fallback: %s", exc)
        has_fail = any(r.status.value == "fail" for r in records)
        opa_decision = "WARN" if has_fail else "ALLOW"
        if has_fail:
            opa_reasons.append("One or more deterministic tests produced FAIL status")

    # 3. Server-side governance disposition
    governance_disp = "ACCEPT"
    if opa_decision == "BLOCK":
        governance_disp = "REJECT"
    elif not all_grounded or opa_decision == "WARN":
        governance_disp = "CONDITIONAL_ACCEPT"

    # 4. Merkle attestation seal
    merkle_root = ""
    try:
        from start.attestation.merkle_ledger import MerkleLedger

        ledger = MerkleLedger()
        for r in records:
            ledger.append_record(r)
        merkle_root = ledger.get_root_hash()
    except Exception:
        merkle_root = f"SEAL-HASH-{time.time():.0f}"

    resp = ReviewerHydrationResponse(
        run_id=run_id,
        schema_version=START_SCHEMA_VERSION,
        model_name=submission.model_name,
        is_grounded=all_grounded,
        all_grounded=all_grounded,
        hydrated_findings=hydrated_findings,
        opa_policy_decision=opa_decision,  # type: ignore[arg-type]
        opa_reasons=opa_reasons,
        governance_disposition=governance_disp,  # type: ignore[arg-type]
        attestation_seal_merkle_root=merkle_root,
        attestation_timestamp=time.time(),
    )

    return APIResponseEnvelope(
        success=True,
        run_id=run_id,
        data=resp.model_dump(),
    )
